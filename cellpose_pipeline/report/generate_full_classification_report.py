#!/usr/bin/env python3
"""Build the self-contained SUM159 full-cohort classification report."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


def _load_common() -> Any:
    path = Path(__file__).with_name("classification_report_common.py")
    spec = importlib.util.spec_from_file_location("_full_classification_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLATE_MAP = (
    REPO_ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "analysisi"
    / "resources"
    / "SUM159_AC_Experiment1_PlateMap.csv"
)
DEFAULT_WORKFLOW_PDF = REPO_ROOT / "docs" / "death_classification_workflow.pdf"
BRANCH_DIRS = {
    "original": "classification_fusion",
    "nucleated_only": "classification_fusion_nucleated_only",
}
BRANCH_ANALYSIS_DIRS = {
    "consensus": "fusion-consensus",
    "original": "fusion",
    "nucleated_only": "fusion-nucleated-only",
}
WELL_COUNT_PLOT_DIRECTORY = Path(
    "analysis/well_count_timecourses/fusion-consensus"
)
WELL_COUNT_PLOT_STEM = "well_live_dead_counts_over_time"
DOSE_CONDITION_PANELS = (
    (
        "doxorubicin_alone",
        "Doxorubicin alone",
        "doxorubicin_alone_2N_vs_4N_{filename}.png",
    ),
    (
        "doxorubicin_plus_cyclophosphamide",
        "Doxorubicin + cyclophosphamide",
        "doxorubicin_plus_cyclophosphamide_2N_vs_4N_{filename}.png",
    ),
)
DOSE_FIGURE_SPECS = (
    {
        "id": "auc",
        "subdirectory": "auc",
        "filename": "hill",
        "title": "AUC-normalized Hill responses",
        "text": (
            "Each replicate is normalized to its matched vehicle within the same "
            "treatment background."
        ),
        "caption": (
            "AUC-normalized 2N-versus-4N dose response for the indicated "
            "classification branch and treatment background."
        ),
        "table_block_id": "hill_table_block",
        "table_id": "hill_table",
    },
    {
        "id": "gr",
        "subdirectory": "gr",
        "filename": "gr",
        "title": "Day-4 and Day-5 growth-rate inhibition",
        "text": (
            "The paired Delta GR result compares 4N with 2N while preserving "
            "replicate pairing."
        ),
        "caption": (
            "GR curves and paired ploidy differences for the indicated "
            "classification branch and treatment background."
        ),
        "table_block_id": "gr_table_block",
        "table_id": "gr_table",
    },
    {
        "id": "death",
        "subdirectory": "death",
        "filename": "excess_lethal_fraction",
        "title": "Day-4 and Day-5 excess lethal fraction",
        "text": (
            "Excess lethal fraction adjusts each treated well to matched control "
            "viability in the same replicate and treatment background."
        ),
        "caption": (
            "Excess-lethal-fraction curves and paired ploidy differences for the "
            "indicated classification branch and treatment background."
        ),
        "table_block_id": "death_table_block",
        "table_id": "death_table",
    },
)
STATE_COLORS = {
    "live": np.array([35, 205, 95], dtype=np.float32),
    "dead": np.array([176, 74, 214], dtype=np.float32),
    "artifact": np.array([145, 150, 160], dtype=np.float32),
    "uncertain": np.array([255, 214, 10], dtype=np.float32),
    "transitional": np.array([255, 214, 10], dtype=np.float32),
}
CLASSIFICATION_BOUNDARY_WIDTH = 2
QC_COMPOSITE_FONT_SCALE = 2
QC_COMPOSITE_TITLE_FONT_SIZE = 72 * QC_COMPOSITE_FONT_SCALE
QC_COMPOSITE_LABEL_FONT_SIZE = 68 * QC_COMPOSITE_FONT_SCALE
QC_PLOIDY_ORDER = ("2N", "4N")
QC_TIMEPOINTS = (
    {"id": "day_0", "label": "Day 0", "day": 0, "hours": 0.0},
    {"id": "day_3", "label": "Day 3", "day": 3, "hours": 72.0},
    {"id": "day_5", "label": "Day 5", "day": 5, "hours": 120.0},
)
QC_CONDITION_GROUPS = (
    {
        "id": "no_drug",
        "label": "No drug",
        "short_label": "No drug",
        "cyclophosphamide": False,
        "dose_band": "zero",
        "target_dose_nm": 0.0,
    },
    {
        "id": "cyclophosphamide_only",
        "label": "Cyclophosphamide only",
        "short_label": "Cyclophosphamide only",
        "cyclophosphamide": True,
        "dose_band": "zero",
        "target_dose_nm": 0.0,
    },
    {
        "id": "low_doxorubicin",
        "label": "Low-dose doxorubicin",
        "short_label": "Low Dox",
        "cyclophosphamide": False,
        "dose_band": "low",
        "target_dose_nm": 12.5,
    },
    {
        "id": "low_doxorubicin_cyclophosphamide",
        "label": "Low-dose doxorubicin + cyclophosphamide",
        "short_label": "Low Dox + cyclophosphamide",
        "cyclophosphamide": True,
        "dose_band": "low",
        "target_dose_nm": 12.5,
    },
    {
        "id": "high_doxorubicin",
        "label": "High-dose doxorubicin",
        "short_label": "High Dox",
        "cyclophosphamide": False,
        "dose_band": "high",
        "target_dose_nm": 400.0,
    },
    {
        "id": "high_doxorubicin_cyclophosphamide",
        "label": "High-dose doxorubicin + cyclophosphamide",
        "short_label": "High Dox + cyclophosphamide",
        "cyclophosphamide": True,
        "dose_band": "high",
        "target_dose_nm": 400.0,
    },
)
LOW_DOSE_MAX_NM = 25.0
HIGH_DOSE_MIN_NM = 200.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument(
        "--previous-classification-root",
        type=Path,
        required=True,
        help=(
            "Completed earlier classification result used only to render the "
            "Previous classification QC panel."
        ),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        help="Approved d0+d5 calibration root used only for configuration provenance validation.",
    )
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument(
        "--workflow-pdf",
        type=Path,
        default=DEFAULT_WORKFLOW_PDF,
        help="Single-page workflow PDF embedded near the beginning of the report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <classification-root>/analysis/reports).",
    )
    parser.add_argument(
        "--plugin-root",
        type=Path,
        help="Data Analytics report environment; auto-discovered when omitted.",
    )
    parser.add_argument(
        "--debug-figure-dir",
        type=Path,
        help="Optional directory for review copies of embedded figures.",
    )
    parser.add_argument("--expected-fields-per-branch", type=int, default=27200)
    parser.add_argument("--expected-timepoints", type=int, default=85)
    parser.add_argument("--expected-dose-response-files", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with COMMON.require_file(path).open(newline="") as handle:
        return next(csv.reader(handle), [])


def render_workflow_pdf(path: Path, dpi: int = 300) -> Image.Image:
    """Render the first page of the workflow PDF for the HTML image viewer."""
    pdf_path = COMMON.require_file(path)
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError(
            "pdftoppm is required to render the workflow PDF but was not found"
        )
    with tempfile.TemporaryDirectory(prefix="cpsam_workflow_pdf_") as directory:
        output_stem = Path(directory) / "workflow"
        command = (
            pdftoppm,
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(pdf_path),
            str(output_stem),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Unable to render workflow PDF {pdf_path}: {detail}"
            )
        rendered = output_stem.with_suffix(".png")
        if not rendered.is_file() or rendered.stat().st_size == 0:
            raise RuntimeError(
                f"Workflow PDF renderer did not create a usable image: {rendered}"
            )
        with Image.open(rendered) as image:
            return image.convert("RGB").copy()


def validate_inputs(
    root: Path,
    previous_root: Path,
    calibration_root: Path | None,
    expected_fields: int,
    expected_timepoints: int,
    expected_dose_files: int,
) -> dict[str, Any]:
    COMMON.require_file(root / "workflow_status" / "late_death_trajectory" / "_SUCCESS")
    failures = COMMON.read_csv(root / "late_death_refinement" / "failures.csv")
    if failures:
        raise RuntimeError(f"Late-death refinement contains {len(failures)} failures")
    refinement = COMMON.read_csv(
        root / "late_death_refinement" / "refinement_summary.csv"
    )
    if len(refinement) != 2 * expected_fields:
        raise RuntimeError(
            f"Expected {2 * expected_fields} refinement rows, found {len(refinement)}"
        )
    branch_counts = Counter(row["branch"] for row in refinement)
    if branch_counts != Counter({"original": expected_fields, "nucleated_only": expected_fields}):
        raise RuntimeError(f"Unexpected refinement branch counts: {dict(branch_counts)}")

    summaries: dict[str, list[dict[str, str]]] = {}
    for branch, directory in BRANCH_DIRS.items():
        rows = COMMON.read_csv(root / directory / "summaries" / "cell_count_summary.csv")
        if len(rows) != expected_fields:
            raise RuntimeError(
                f"Expected {expected_fields} {branch} summaries, found {len(rows)}"
            )
        if len({row["key"] for row in rows}) != expected_fields:
            raise RuntimeError(f"Duplicate {branch} summary keys detected")
        timepoints = {COMMON.as_float(row["elapsed_hours"]) for row in rows}
        if len(timepoints) != expected_timepoints:
            raise RuntimeError(
                f"Expected {expected_timepoints} {branch} time points, found {len(timepoints)}"
            )
        summaries[branch] = rows
    consensus_rows = COMMON.read_csv(
        root
        / "classification_consensus"
        / "summaries"
        / "cell_count_summary.csv"
    )
    if len(consensus_rows) != expected_fields:
        raise RuntimeError(
            f"Expected {expected_fields} consensus summaries, "
            f"found {len(consensus_rows)}"
        )
    if len({row["key"] for row in consensus_rows}) != expected_fields:
        raise RuntimeError("Duplicate consensus summary keys detected")
    summaries = {"consensus": consensus_rows, **summaries}
    previous_summary = COMMON.read_csv(
        previous_root
        / BRANCH_DIRS["original"]
        / "summaries"
        / "cell_count_summary.csv"
    )
    if len(previous_summary) != expected_fields:
        raise RuntimeError(
            f"Expected {expected_fields} previous classification summaries, "
            f"found {len(previous_summary)}"
        )
    if len({row["key"] for row in previous_summary}) != expected_fields:
        raise RuntimeError("Duplicate previous classification summary keys detected")

    field_states = COMMON.read_csv(
        root
        / "workflow_status"
        / "late_death_trajectory"
        / "prepared_refinement"
        / "field_states.csv"
    )
    field_state_counts = Counter(row["branch"] for row in field_states)
    if field_state_counts != Counter(
        {"original": expected_fields, "nucleated_only": expected_fields}
    ):
        raise RuntimeError(
            "Unexpected prepared field-state branch counts: "
            f"{dict(field_state_counts)}"
        )
    required_field_state_columns = {
        "branch",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "cyclophosphamide",
        "doxorubicin_nm",
    }
    missing_field_state_columns = required_field_state_columns - set(field_states[0])
    if missing_field_state_columns:
        raise RuntimeError(
            "Prepared field-state table is missing QC-selection columns: "
            f"{sorted(missing_field_state_columns)}"
        )

    dose_root = root / "analysis" / "dose_response"
    dose_files = sorted(path for path in dose_root.rglob("*") if path.is_file())
    if len(dose_files) != expected_dose_files:
        raise RuntimeError(
            f"Expected {expected_dose_files} dose-response files, found {len(dose_files)}"
        )
    required_dose_files = (
        "auc/hill_fit_parameters.csv",
        "day4/hill_fit_parameters.csv",
        "day5/hill_fit_parameters.csv",
        "gr/gr_delta_summary.csv",
        "death/death_delta_summary.csv",
    )
    for analysis_branch in BRANCH_ANALYSIS_DIRS.values():
        for relative in required_dose_files:
            COMMON.require_file(dose_root / analysis_branch / relative)

    well_count_plot_pdf = COMMON.require_file(
        root / WELL_COUNT_PLOT_DIRECTORY / f"{WELL_COUNT_PLOT_STEM}.pdf"
    )
    well_count_plot_png = COMMON.require_file(
        root / WELL_COUNT_PLOT_DIRECTORY / f"{WELL_COUNT_PLOT_STEM}.png"
    )

    production = COMMON.read_json(
        root / "late_death_refinement" / "production_configuration.json"
    )
    production_go_no_go = COMMON.read_json(
        root
        / "late_death_refinement"
        / "FULL_CLASSIFICATION_GO_NO_GO.json"
    )
    approved_source = Path(production["approved_configuration_source"])
    if calibration_root is not None:
        approved_source = (
            calibration_root / "optimization" / "best_configuration.json"
        )
    approved = COMMON.read_json(approved_source)
    for field in ("field_configuration", "object_configuration", "late_min_hours"):
        if production[field] != approved[field]:
            raise RuntimeError(
                f"Production {field} does not match the approved calibration"
            )
    return {
        "refinement": refinement,
        "summaries": summaries,
        "production": production,
        "production_go_no_go": production_go_no_go,
        "approved_source": approved_source,
        "dose_files": dose_files,
        "field_states": field_states,
        "well_count_plot_pdf": well_count_plot_pdf,
        "well_count_plot_png": well_count_plot_png,
        "previous_classification_root": previous_root,
    }


def load_plate_map(path: Path) -> dict[str, dict[str, str]]:
    rows = COMMON.read_csv(path)
    mapping = {row["well"]: row for row in rows}
    if len(mapping) != 80:
        raise RuntimeError(f"Expected 80 plate-map wells, found {len(mapping)}")
    return mapping


def qc_condition_group(
    doxorubicin_nm: float,
    cyclophosphamide: bool,
) -> dict[str, Any] | None:
    if doxorubicin_nm == 0:
        dose_band = "zero"
    elif 0 < doxorubicin_nm <= LOW_DOSE_MAX_NM:
        dose_band = "low"
    elif doxorubicin_nm >= HIGH_DOSE_MIN_NM:
        dose_band = "high"
    else:
        return None
    for group in QC_CONDITION_GROUPS:
        if (
            group["dose_band"] == dose_band
            and group["cyclophosphamide"] == cyclophosphamide
        ):
            return group
    return None


def qc_candidate_score(
    *,
    doxorubicin_nm: float,
    target_dose_nm: float,
) -> float:
    if doxorubicin_nm > 0 and target_dose_nm > 0:
        return abs(math.log2(doxorubicin_nm / target_dose_nm))
    return 0.0


def qc_timepoint(elapsed_hours: float) -> dict[str, Any] | None:
    return next(
        (
            timepoint
            for timepoint in QC_TIMEPOINTS
            if math.isclose(
                elapsed_hours,
                float(timepoint["hours"]),
                rel_tol=0,
                abs_tol=1e-6,
            )
        ),
        None,
    )


def select_qc_samples(
    validated: dict[str, Any],
    plate_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    summary_by_key = {
        row["key"]: row for row in validated["summaries"]["consensus"]
    }
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated["field_states"]:
        if row["branch"] != "original":
            continue
        elapsed_hours = COMMON.as_float(row["elapsed_hours"])
        timepoint = qc_timepoint(elapsed_hours)
        if timepoint is None:
            continue
        well = row["well"]
        if well not in plate_map:
            raise RuntimeError(f"Prepared field state references unknown well: {well}")
        plate = plate_map[well]
        ploidy = plate["ploidy"].strip().upper()
        if ploidy not in QC_PLOIDY_ORDER:
            continue
        doxorubicin_nm = COMMON.as_float(plate["doxorubicin_nm"])
        cyclophosphamide = COMMON.truthy(plate["cyclophosphamide"])
        if (
            not math.isclose(
                doxorubicin_nm,
                COMMON.as_float(row["doxorubicin_nm"]),
                rel_tol=0,
                abs_tol=1e-9,
            )
            or cyclophosphamide != COMMON.truthy(row["cyclophosphamide"])
        ):
            raise RuntimeError(f"Plate-map and field-state treatment mismatch: {row['key']}")
        condition = qc_condition_group(doxorubicin_nm, cyclophosphamide)
        if condition is None:
            continue
        key = row["key"]
        summary = summary_by_key.get(key)
        if summary is None:
            raise RuntimeError(f"Missing consensus summary for QC candidate: {key}")
        score = qc_candidate_score(
            doxorubicin_nm=doxorubicin_nm,
            target_dose_nm=float(condition["target_dose_nm"]),
        )
        candidates[(ploidy, condition["id"], str(timepoint["id"]))].append(
            {
                "ploidy": ploidy,
                "ploidy_order": QC_PLOIDY_ORDER.index(ploidy) + 1,
                "condition_id": condition["id"],
                "condition": condition["label"],
                "condition_short": condition["short_label"],
                "condition_order": next(
                    index
                    for index, item in enumerate(QC_CONDITION_GROUPS, 1)
                    if item["id"] == condition["id"]
                ),
                "time_id": timepoint["id"],
                "time_label": timepoint["label"],
                "time_order": next(
                    index
                    for index, item in enumerate(QC_TIMEPOINTS, 1)
                    if item["id"] == timepoint["id"]
                ),
                "key": key,
                "well": well,
                "site": COMMON.as_int(row["site"]),
                "elapsed_hours": elapsed_hours,
                "day": int(timepoint["day"]),
                "doxorubicin_nm": doxorubicin_nm,
                "cyclophosphamide": cyclophosphamide,
                "replicate": COMMON.as_int(plate["replicate"]),
                "total_cells": COMMON.as_int(summary["total_cell_count"]),
                "dead_fraction": COMMON.as_float(summary["dead_fraction"]),
                "rescued": COMMON.as_int(summary["late_death_rescue_count"]),
                "uncertain": COMMON.as_int(summary["late_death_uncertain_count"]),
                "_score": score,
            }
        )

    selected: list[dict[str, Any]] = []
    for ploidy in QC_PLOIDY_ORDER:
        for condition in QC_CONDITION_GROUPS:
            available_by_time = {
                str(timepoint["id"]): candidates[
                    (ploidy, condition["id"], str(timepoint["id"]))
                ]
                for timepoint in QC_TIMEPOINTS
            }
            missing = [
                str(timepoint["label"])
                for timepoint in QC_TIMEPOINTS
                if not available_by_time[str(timepoint["id"])]
            ]
            if missing:
                raise RuntimeError(
                    "No QC candidate for "
                    f"ploidy={ploidy}, condition={condition['label']}, "
                    f"time={', '.join(missing)}"
                )

            complete_series = set.intersection(
                *(
                    {
                        (candidate["well"], candidate["site"])
                        for candidate in available_by_time[str(timepoint["id"])]
                    }
                    for timepoint in QC_TIMEPOINTS
                )
            )
            selected_series = (
                min(
                    complete_series,
                    key=lambda series: min(
                        (
                            candidate["_score"],
                            candidate["replicate"],
                            candidate["well"],
                            candidate["site"],
                        )
                        for candidate in available_by_time[
                            str(QC_TIMEPOINTS[0]["id"])
                        ]
                        if (candidate["well"], candidate["site"]) == series
                    ),
                )
                if complete_series
                else None
            )

            for timepoint in QC_TIMEPOINTS:
                available = available_by_time[str(timepoint["id"])]
                if selected_series is not None:
                    available = [
                        candidate
                        for candidate in available
                        if (candidate["well"], candidate["site"]) == selected_series
                    ]
                choice = min(
                    available,
                    key=lambda row: (
                        row["_score"],
                        row["replicate"],
                        row["well"],
                        row["site"],
                        row["key"],
                    ),
                )
                choice = {
                    key: value for key, value in choice.items() if key != "_score"
                }
                choice["selection_index"] = len(selected) + 1
                selected.append(choice)

    expected = (
        len(QC_PLOIDY_ORDER)
        * len(QC_CONDITION_GROUPS)
        * len(QC_TIMEPOINTS)
    )
    if len(selected) != expected:
        raise RuntimeError(f"Unexpected QC sample count: {len(selected)}")
    keys = [row["key"] for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("QC selection contains duplicate field keys")
    return selected


def d0_uncertainty_cases(validated: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in validated["refinement"]:
        if (
            row["branch"] != "original"
            or "_00d00h00m" not in row["key"]
            or COMMON.as_int(row["uncertain"]) <= 0
        ):
            continue
        annotations = COMMON.read_csv(Path(row["annotation_path"]))
        for annotation in annotations:
            if not COMMON.truthy(annotation.get("late_death_uncertain", "")):
                continue
            cases.append(
                {
                    "key": row["key"],
                    "branch": row["branch"],
                    "combined_mask_id": COMMON.as_int(
                        annotation["combined_mask_id"]
                    ),
                    "reason": (
                        "Dual-view final-call disagreement"
                        if COMMON.truthy(
                            annotation.get(
                                "late_death_branch_final_call_discordant",
                                "",
                            )
                        )
                        else "Operational evidence conflict"
                    ),
                }
            )
    return sorted(
        cases,
        key=lambda row: (row["key"], row["combined_mask_id"]),
    )


def write_qc_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("QC selection must not be empty")
    fields = (
        "selection_index",
        "ploidy",
        "ploidy_order",
        "condition_id",
        "condition",
        "condition_short",
        "condition_order",
        "time_id",
        "time_label",
        "time_order",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "day",
        "doxorubicin_nm",
        "cyclophosphamide",
        "replicate",
        "total_cells",
        "dead_fraction",
        "rescued",
        "uncertain",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_datasets(
    root: Path,
    validated: dict[str, Any],
    plate_map: dict[str, dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    summaries = validated["summaries"]
    production = validated["production"]
    production_go_no_go = validated["production_go_no_go"]
    overview = [
        {
            "fields_per_branch": production["fields_per_branch"],
            "total_branch_fields": production["total_fields"],
            "total_rescued": production["total_rescued"],
            "total_uncertain": production["total_uncertain"],
            "dose_response_files": len(validated["dose_files"]),
            "analysis_branches": 3,
            "production_decision": production_go_no_go["decision"],
            "passed_validation_gates": sum(
                bool(values["pass"])
                for values in production_go_no_go["gates"].values()
            ),
            "total_validation_gates": len(production_go_no_go["gates"]),
        }
    ]
    convergence_gates = [
        {
            "gate": name.replace("_", " ").title(),
            "status": "PASS" if values["pass"] else "FAIL",
            "pass": bool(values["pass"]),
        }
        for name, values in production_go_no_go["gates"].items()
    ]
    branch_totals: list[dict[str, Any]] = []
    stage_composition: list[dict[str, Any]] = []
    rescue_by_day: list[dict[str, Any]] = []
    rescue_by_condition: list[dict[str, Any]] = []
    global_fields_by_day: list[dict[str, Any]] = []
    relation_composition: list[dict[str, Any]] = []
    top_rescued_fields: list[dict[str, Any]] = []

    for branch, rows in summaries.items():
        totals = {
            field: sum(COMMON.as_int(row.get(field, 0)) for row in rows)
            for field in (
                "total_masks",
                "total_cell_count",
                "live_cell_count",
                "dead_cell_count",
                "artifact_count",
                "supplemental_dead_object_count",
                "object_aware_dead_count",
                "pre_late_death_live_cell_count",
                "pre_late_death_dead_cell_count",
                "late_death_rescue_count",
                "late_death_uncertain_count",
            )
        }
        branch_totals.append(
            {
                "branch": COMMON.display_branch(branch),
                **totals,
                "final_dead_fraction": totals["dead_cell_count"]
                / max(1, totals["total_cell_count"]),
                "pre_dead_fraction": totals["pre_late_death_dead_cell_count"]
                / max(1, totals["total_cell_count"]),
            }
        )
        for stage, live_field, dead_field in (
            (
                "Before late-death rescue",
                "pre_late_death_live_cell_count",
                "pre_late_death_dead_cell_count",
            ),
            ("Final production result", "live_cell_count", "dead_cell_count"),
        ):
            live = totals[live_field]
            dead = totals[dead_field]
            denominator = max(1, live + dead)
            branch_label = COMMON.display_branch(branch)
            stage_branch = f"{stage} · {branch_label}"
            stage_composition.extend(
                (
                    {
                        "branch": branch_label,
                        "stage": stage,
                        "stage_branch": stage_branch,
                        "state": "Live",
                        "count": live,
                        "share": live / denominator,
                    },
                    {
                        "branch": branch_label,
                        "stage": stage,
                        "stage_branch": stage_branch,
                        "state": "Dead",
                        "count": dead,
                        "share": dead / denominator,
                    },
                )
            )

        day_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
        condition_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            day = COMMON.as_int(row["day"])
            day_groups[day].append(row)
            plate = plate_map[row["well"]]
            condition_groups[
                (
                    plate["ploidy"],
                    "Cyclophosphamide"
                    if COMMON.truthy(plate["cyclophosphamide"])
                    else "No cyclophosphamide",
                )
            ].append(row)
        for day, selected in sorted(day_groups.items()):
            rescued = sum(COMMON.as_int(row["late_death_rescue_count"]) for row in selected)
            cells = sum(COMMON.as_int(row["total_cell_count"]) for row in selected)
            global_fields = sum(
                1 for row in selected if COMMON.truthy(row["late_death_field_global"])
            )
            rescue_by_day.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "day": day,
                    "rescued": rescued,
                    "total_cells": cells,
                    "rescued_share": rescued / max(1, cells),
                }
            )
            global_fields_by_day.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "day": day,
                    "global_late_fields": global_fields,
                    "fields": len(selected),
                    "field_share": global_fields / max(1, len(selected)),
                }
            )
        for (ploidy, treatment), selected in sorted(condition_groups.items()):
            rescued = sum(COMMON.as_int(row["late_death_rescue_count"]) for row in selected)
            cells = sum(COMMON.as_int(row["total_cell_count"]) for row in selected)
            rescue_by_condition.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "ploidy": ploidy,
                    "treatment": treatment,
                    "group": f"{ploidy} · {treatment}",
                    "rescued": rescued,
                    "total_cells": cells,
                    "rescued_share": rescued / max(1, cells),
                }
            )

        relation_fields = (
            ("Dead-only", "dead_only_object_count"),
            ("Adjacent dead", "adjacent_dead_object_count"),
            (
                "Multi-nucleus live/dead overlap",
                "overlapping_live_dead_multi_nucleus_count",
            ),
            ("Live with death signal", "live_with_death_signal_count"),
            (
                "Nucleus-unresolved overlap",
                "nucleus_unresolved_dead_overlap_count",
            ),
            ("Merged multiple objects", "merged_multiple_dead_object_count"),
        )
        relation_counts = {
            label: sum(COMMON.as_int(row[field]) for row in rows)
            for label, field in relation_fields
        }
        relation_denominator = max(1, sum(relation_counts.values()))
        for relation, count in relation_counts.items():
            relation_composition.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "relation": relation,
                    "count": count,
                    "share": count / relation_denominator,
                }
            )

        ranked = sorted(
            rows,
            key=lambda row: COMMON.as_int(row["late_death_rescue_count"]),
            reverse=True,
        )[:8]
        top_rescued_fields.extend(
            {
                "branch": COMMON.display_branch(branch),
                "key": row["key"],
                "well": row["well"],
                "elapsed_hours": COMMON.as_float(row["elapsed_hours"]),
                "rescued": COMMON.as_int(row["late_death_rescue_count"]),
                "pre_dead": COMMON.as_int(row["pre_late_death_dead_cell_count"]),
                "final_dead": COMMON.as_int(row["dead_cell_count"]),
                "global_late_field": "Yes"
                if COMMON.truthy(row["late_death_field_global"])
                else "No",
            }
            for row in ranked
        )

    dose_hill: list[dict[str, Any]] = []
    gr_delta: list[dict[str, Any]] = []
    death_delta: list[dict[str, Any]] = []
    for branch, analysis_dir in BRANCH_ANALYSIS_DIRS.items():
        base = root / "analysis" / "dose_response" / analysis_dir
        for row in COMMON.read_csv(base / "auc" / "hill_fit_parameters.csv"):
            dose_hill.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "condition": row["condition"],
                    "ploidy": row["ploidy"],
                    "ec50_nm": COMMON.as_float(row["ec50_nm"]),
                    "bottom": COMMON.as_float(row["bottom"]),
                    "top": COMMON.as_float(row["top"]),
                    "hill_slope": COMMON.as_float(row["hill_slope"]),
                    "r_squared": COMMON.as_float(row["r_squared"]),
                }
            )
        for row in COMMON.read_csv(base / "gr" / "gr_delta_summary.csv"):
            gr_delta.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "endpoint": row["analysis_key"],
                    "condition": row["condition"],
                    "mean_delta_gr": COMMON.as_float(
                        row["mean_delta_gr_log_dose"]
                    ),
                    "ci_low": COMMON.as_float(row["mean_delta_gr_ci_low"]),
                    "ci_high": COMMON.as_float(row["mean_delta_gr_ci_high"]),
                    "bootstrap_successes": COMMON.as_int(
                        row["bootstrap_successes"]
                    ),
                }
            )
        for row in COMMON.read_csv(base / "death" / "death_delta_summary.csv"):
            death_delta.append(
                {
                    "branch": COMMON.display_branch(branch),
                    "endpoint": row["analysis_key"],
                    "condition": row["condition"],
                    "mean_delta_excess_lf": COMMON.as_float(
                        row["mean_delta_excess_lf_log_dose"]
                    ),
                    "ci_low": COMMON.as_float(
                        row["mean_delta_excess_lf_ci_low"]
                    ),
                    "ci_high": COMMON.as_float(
                        row["mean_delta_excess_lf_ci_high"]
                    ),
                    "bootstrap_successes": COMMON.as_int(
                        row["bootstrap_successes"]
                    ),
                }
            )

    return {
        "overview": overview,
        "branch_totals": branch_totals,
        "stage_composition": stage_composition,
        "rescue_by_day": rescue_by_day,
        "global_fields_by_day": global_fields_by_day,
        "rescue_by_condition": rescue_by_condition,
        "relation_composition": relation_composition,
        "top_rescued_fields": top_rescued_fields,
        "dose_hill": dose_hill,
        "gr_delta": gr_delta,
        "death_delta": death_delta,
        "convergence_gates": convergence_gates,
    }


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array[:3], 0, -1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported Combined image shape: {array.shape}")
    result = np.zeros(array.shape, dtype=np.float32)
    for channel in range(3):
        plane = array[..., channel].astype(np.float32)
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        result[..., channel] = np.clip((plane - low) / (high - low), 0, 1)
    return (result * 255).astype(np.uint8)


def normalize_grayscale(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(array))
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., :3].max(axis=-1)
    elif array.ndim == 3 and array.shape[0] in (3, 4):
        array = array[:3].max(axis=0)
    if array.ndim != 2:
        raise ValueError(f"Unsupported single-channel image shape: {array.shape}")
    plane = array.astype(np.float32)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        scaled = np.zeros(plane.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        scaled = (
            np.clip((plane - low) / (high - low), 0, 1) * 255
        ).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=2)


def label_boundary(labels: np.ndarray) -> np.ndarray:
    labels = np.squeeze(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels > 0)


def thicken_boundary(
    boundary: np.ndarray,
    width: int = CLASSIFICATION_BOUNDARY_WIDTH,
) -> np.ndarray:
    if width < 1:
        raise ValueError("Boundary width must be positive")
    thick = np.asarray(boundary, dtype=bool)
    for _ in range(width - 1):
        padded = np.pad(thick, 1, mode="constant", constant_values=False)
        thick = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return thick


def segmentation_overlay(
    raw_rgb: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
) -> Image.Image:
    output = raw_rgb.astype(np.float32)
    for labels, color in masks:
        boundary = label_boundary(labels)
        output[boundary] = output[boundary] * 0.12 + color * 0.88
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def state_overlay(
    raw_rgb: np.ndarray,
    labels: np.ndarray,
    predictions: list[dict[str, str]],
    state_field: str,
) -> Image.Image:
    labels = np.squeeze(labels).astype(np.int64, copy=False)
    output = raw_rgb.astype(np.float32)
    maximum = int(labels.max(initial=0))
    colors = np.zeros((maximum + 1, 3), dtype=np.float32)
    has_color = np.zeros(maximum + 1, dtype=bool)
    for row in predictions:
        mask_id = COMMON.as_int(row.get("mask_id", 0))
        if mask_id <= 0 or mask_id > maximum:
            continue
        state = row.get(state_field, "") or row.get("state", "")
        colors[mask_id] = STATE_COLORS.get(
            state,
            np.array([245, 247, 250], dtype=np.float32),
        )
        has_color[mask_id] = True
    boundary = thicken_boundary(label_boundary(labels))
    boundary &= has_color[labels]
    output[boundary] = output[boundary] * 0.15 + colors[labels[boundary]] * 0.85
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def focus_mask_boundary(image: Image.Image, labels: np.ndarray, mask_id: int) -> Image.Image:
    labels = np.squeeze(labels).astype(np.int64, copy=False)
    focus = labels == mask_id
    if not focus.any():
        raise RuntimeError(f"Focus mask {mask_id} is absent from the supplied labels")
    boundary = label_boundary(focus.astype(np.int8))
    output = np.asarray(image).copy()
    output[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(output, mode="RGB")


def crop_around_mask(
    image: Image.Image,
    labels: np.ndarray,
    mask_id: int,
    padding: int = 96,
) -> Image.Image:
    labels = np.squeeze(labels)
    y, x = np.where(labels == mask_id)
    if len(x) == 0:
        raise RuntimeError(f"Cannot crop absent mask {mask_id}")
    left = max(0, int(x.min()) - padding)
    right = min(labels.shape[1], int(x.max()) + padding + 1)
    top = max(0, int(y.min()) - padding)
    bottom = min(labels.shape[0], int(y.max()) + padding + 1)
    return image.crop((left, top, right, bottom))


def find_record(root: Path, key: str) -> Path:
    well = key.split("_", 1)[0]
    direct = (
        root
        / "workflow_status"
        / "postsegmentation_manifest"
        / "records"
        / well
        / f"{key}.json"
    )
    if direct.is_file():
        return direct
    matches = sorted(
        (
            root
            / "workflow_status"
            / "postsegmentation_manifest"
            / "records"
        ).rglob(f"*{key}*.json")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one field record for {key}, found {len(matches)}")
    return matches[0]


def find_prediction(root: Path, branch: str, key: str) -> Path:
    directory = root / BRANCH_DIRS[branch] / "predictions"
    matches = sorted(directory.glob(f"*{key}*_per_cell_predictions.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {branch} prediction table for {key}, found {len(matches)}"
        )
    return matches[0]


def classification_comparison_panels(
    record: dict[str, Any],
    previous_prediction_path: Path,
    final_prediction_path: Path,
) -> list[tuple[Image.Image, str]]:
    profiles = record["profiles"]
    combined = profiles["Combined"]
    dead = profiles["Dead"]
    combined_raw = normalize_rgb(tifffile.imread(combined["raw"]))
    dead_raw = normalize_grayscale(tifffile.imread(dead["raw"]))
    combined_original = tifffile.imread(combined["original_mask"])
    previous_predictions = COMMON.read_csv(previous_prediction_path)
    final_predictions = COMMON.read_csv(final_prediction_path)
    return [
        (Image.fromarray(combined_raw, mode="RGB"), "Combined RGB · raw"),
        (Image.fromarray(dead_raw, mode="RGB"), "Dead · raw"),
        (
            state_overlay(
                combined_raw,
                combined_original,
                previous_predictions,
                "state",
            ),
            "Previous classification",
        ),
        (
            state_overlay(
                combined_raw,
                combined_original,
                final_predictions,
                "state",
            ),
            "Final classification",
        ),
    ]


def authoritative_qc_panels(
    record: dict[str, Any],
    final_prediction_path: Path,
    previous_prediction_path: Path,
) -> list[tuple[Image.Image, str]]:
    profiles = record["profiles"]
    combined = profiles["Combined"]
    brightfield = profiles["Brightfield"]
    nuclei = profiles["Nuclei"]
    dead = profiles["Dead"]

    combined_raw = normalize_rgb(tifffile.imread(combined["raw"]))
    brightfield_raw = normalize_grayscale(tifffile.imread(brightfield["raw"]))
    nuclei_raw = normalize_grayscale(tifffile.imread(nuclei["raw"]))
    dead_raw = normalize_grayscale(tifffile.imread(dead["raw"]))

    combined_original = tifffile.imread(combined["original_mask"])
    brightfield_original = tifffile.imread(brightfield["original_mask"])
    nuclei_extent = tifffile.imread(nuclei["extent_mask"])
    nuclei_core = tifffile.imread(nuclei["core_mask"])
    dead_labels = tifffile.imread(dead["mask"])
    final_predictions = COMMON.read_csv(final_prediction_path)
    previous_predictions = COMMON.read_csv(previous_prediction_path)

    orange = np.array([255, 159, 67], dtype=np.float32)
    cyan = np.array([59, 201, 219], dtype=np.float32)
    magenta = np.array([221, 87, 190], dtype=np.float32)
    blue = np.array([73, 114, 255], dtype=np.float32)
    return [
        (Image.fromarray(combined_raw, mode="RGB"), "Combined RGB · raw"),
        (Image.fromarray(brightfield_raw, mode="RGB"), "Brightfield · raw"),
        (Image.fromarray(nuclei_raw, mode="RGB"), "Nuclei · raw"),
        (Image.fromarray(dead_raw, mode="RGB"), "Dead · raw"),
        (
            segmentation_overlay(
                combined_raw,
                [(combined_original, orange)],
            ),
            "Combined segmentation",
        ),
        (
            segmentation_overlay(
                brightfield_raw,
                [(brightfield_original, orange)],
            ),
            "Brightfield segmentation",
        ),
        (
            segmentation_overlay(
                nuclei_raw,
                [(nuclei_extent, cyan), (nuclei_core, magenta)],
            ),
            "Nuclei extent (cyan) + core (magenta)",
        ),
        (
            segmentation_overlay(dead_raw, [(dead_labels, blue)]),
            "Dead segmentation",
        ),
        (
            state_overlay(
                combined_raw,
                combined_original,
                final_predictions,
                "state",
            ),
            "Final classification · authoritative",
        ),
        (
            state_overlay(
                combined_raw,
                combined_original,
                previous_predictions,
                "state",
            ),
            "Previous classification · 20260721",
        ),
    ]


def sample_qc_figure(
    root: Path,
    previous_root: Path,
    selection: dict[str, Any],
) -> Image.Image:
    key = selection["key"]
    record = COMMON.read_json(find_record(root, key))
    panels = authoritative_qc_panels(
        record,
        find_prediction(root, "original", key),
        find_prediction(previous_root, "original", key),
    )
    if len(panels) != 10:
        raise RuntimeError(f"Expected ten authoritative QC panels, found {len(panels)}")
    title = (
        f"{key} · {selection['ploidy']} · {selection['condition']} · "
        f"{selection['time_label']} ({selection['elapsed_hours']:g} h) · "
        f"{selection['doxorubicin_nm']:g} nM Dox"
    )
    return COMMON.labeled_grid(
        panels,
        title=title,
        columns=4,
        panel_width=1400,
        title_font_size=QC_COMPOSITE_TITLE_FONT_SIZE,
        label_font_size=QC_COMPOSITE_LABEL_FONT_SIZE,
    )


def d0_uncertainty_figure(root: Path, case: dict[str, Any]) -> Image.Image:
    key = case["key"]
    mask_id = case["combined_mask_id"]
    record = COMMON.read_json(find_record(root, key))
    profiles = record["profiles"]
    combined = profiles["Combined"]
    combined_raw = normalize_rgb(tifffile.imread(combined["raw"]))
    dead_raw = normalize_grayscale(tifffile.imread(profiles["Dead"]["raw"]))
    nuclei_raw = normalize_grayscale(tifffile.imread(profiles["Nuclei"]["raw"]))
    original_labels = tifffile.imread(combined["original_mask"])
    nucleated_labels = tifffile.imread(combined["nucleated_mask"])
    original_predictions = COMMON.read_csv(find_prediction(root, "original", key))
    nucleated_predictions = COMMON.read_csv(
        find_prediction(root, "nucleated_only", key)
    )
    original_final = focus_mask_boundary(
        state_overlay(
            combined_raw,
            original_labels,
            original_predictions,
            "state",
        ),
        original_labels,
        mask_id,
    )
    nucleated_final = state_overlay(
        combined_raw,
        nucleated_labels,
        nucleated_predictions,
        "state",
    )
    panels = [
        (
            crop_around_mask(
                Image.fromarray(combined_raw, mode="RGB"),
                original_labels,
                mask_id,
            ),
            "Combined RGB · focused object",
        ),
        (
            crop_around_mask(
                Image.fromarray(dead_raw, mode="RGB"),
                original_labels,
                mask_id,
            ),
            "Dead raw · same spatial crop",
        ),
        (
            crop_around_mask(
                Image.fromarray(nuclei_raw, mode="RGB"),
                original_labels,
                mask_id,
            ),
            "Nuclei raw · same spatial crop",
        ),
        (
            crop_around_mask(original_final, original_labels, mask_id),
            f"Original final · uncertain mask {mask_id}",
        ),
        (
            crop_around_mask(nucleated_final, original_labels, mask_id),
            "Nucleated-only final · matched region",
        ),
    ]
    return COMMON.labeled_grid(
        panels,
        title=f"{key}: d0 dual-view uncertainty focus",
        columns=3,
        panel_width=720,
        title_font_size=QC_COMPOSITE_TITLE_FONT_SIZE,
        label_font_size=QC_COMPOSITE_LABEL_FONT_SIZE,
    )


def dose_panel_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for metric in DOSE_FIGURE_SPECS:
        for branch, analysis_dir in BRANCH_ANALYSIS_DIRS.items():
            for condition_id, condition, filename_template in DOSE_CONDITION_PANELS:
                records.append(
                    {
                        "metric_id": str(metric["id"]),
                        "metric_title": str(metric["title"]),
                        "subdirectory": str(metric["subdirectory"]),
                        "filename": filename_template.format(
                            filename=metric["filename"]
                        ),
                        "branch": branch,
                        "analysis_dir": analysis_dir,
                        "condition_id": condition_id,
                        "condition": condition,
                        "key": (
                            f"dose_{metric['id']}_{branch}_{condition_id}"
                        ),
                    }
                )
    return records


def report_figures(
    root: Path,
    previous_root: Path,
    workflow_pdf: Path,
    qc_samples: list[dict[str, Any]],
    uncertainty_cases: list[dict[str, Any]],
) -> dict[str, Image.Image]:
    figures = {"classification_workflow": render_workflow_pdf(workflow_pdf)}
    figures.update(
        {
            f"qc_sample_{sample['selection_index']:02d}": sample_qc_figure(
                root,
                previous_root,
                sample,
            )
            for sample in qc_samples
        }
    )
    figures.update(
        {
            f"d0_uncertainty_{index:02d}": d0_uncertainty_figure(root, case)
            for index, case in enumerate(uncertainty_cases, 1)
        }
    )
    figures.update(
        {
            "well_live_dead_counts_over_time": COMMON.read_image(
                root
                / WELL_COUNT_PLOT_DIRECTORY
                / f"{WELL_COUNT_PLOT_STEM}.png"
            ),
        }
    )
    for panel in dose_panel_records():
        figures[panel["key"]] = COMMON.read_image(
            root
            / "analysis"
            / "dose_response"
            / panel["analysis_dir"]
            / panel["subdirectory"]
            / panel["filename"]
        )
    return figures


def sources(
    root: Path,
    previous_root: Path,
    calibration_root: Path | None,
) -> list[dict[str, Any]]:
    label = root.name
    result = [
        COMMON.logical_source(
            "workflow",
            "Death-classification workflow",
            REPO_ROOT.name,
            "docs/death_classification_workflow.pdf",
            "Render the single-page workflow diagram embedded near the beginning of the report.",
        ),
        COMMON.logical_source(
            "submission",
            "Classification-only submission record",
            label,
            "SUBMISSION_SUMMARY.txt",
            "Read the production paths, code SHA, resources, and job identifiers.",
        ),
        COMMON.logical_source(
            "summaries",
            "Authoritative consensus cell-count summary",
            label,
            "classification_consensus/summaries/cell_count_summary.csv",
            "Aggregate authoritative full-cohort cell states with dual-view diagnostics.",
        ),
        COMMON.logical_source(
            "refinement",
            "Late-death refinement summary",
            label,
            "late_death_refinement/refinement_summary.csv",
            "Read branch-field rescue counts and uncertainty annotations.",
        ),
        COMMON.logical_source(
            "configuration",
            "Frozen production configuration",
            label,
            "late_death_refinement/production_configuration.json",
            "Read the applied model version, parameters, and configuration hashes.",
        ),
        COMMON.logical_source(
            "well_counts",
            "Authoritative well time-course cell-state counts",
            label,
            "analysis/well_count_timecourses/fusion-consensus/well_time_cell_state_counts.csv",
            "Read strict-completeness well-by-time live and dead counts.",
        ),
        COMMON.logical_source(
            "well_count_plot",
            "Authoritative well live/dead time-course plot",
            label,
            (
                "analysis/well_count_timecourses/fusion-consensus/"
                "well_live_dead_counts_over_time.pdf"
            ),
            "Render the saved full-cohort well-level live and dead trajectories.",
        ),
        COMMON.logical_source(
            "dose",
            "Authoritative full-cohort dose-response analyses",
            label,
            "analysis/dose_response/fusion-consensus/auc/hill_fit_parameters.csv",
            "Read AUC, endpoint, GR, and excess-lethal-fraction analyses.",
        ),
        COMMON.logical_source(
            "qc",
            "Raw channels, frozen masks, and final classification tables",
            label,
            "workflow_status/postsegmentation_manifest/records/*/*.json",
            "Render deterministic multi-channel QC directly from saved raw images, frozen masks, and final predictions.",
        ),
        COMMON.logical_source(
            "previous_qc",
            "Previous full-cohort classification tables",
            previous_root.name,
            "classification_fusion/predictions/*_per_cell_predictions.csv",
            "Render the Previous classification QC panel from classification_20260721_102004.",
        ),
        COMMON.logical_source(
            "field_states",
            "Prepared field time course and late-death states",
            label,
            "workflow_status/late_death_trajectory/prepared_refinement/field_states.csv",
            "Select exact Day 0, Day 3, and Day 5 fields from the authoritative original branch.",
        ),
        COMMON.logical_source(
            "qc_selection",
            "Deterministic ploidy-by-treatment-by-time QC sample selection",
            label,
            "analysis/reports/qc_sample_selection.csv",
            "Read the selected fields, ploidy and treatment strata, exact time points, and final cell-state summaries.",
        ),
    ]
    if calibration_root is not None:
        result.append(
            COMMON.logical_source(
                "calibration",
                "Approved no-ground-truth d0 and late-time calibration",
                calibration_root.name,
                "optimization/best_configuration.json",
                "Verify that the production configuration matches the approved calibration.",
            )
        )
    result.append(
        COMMON.logical_source(
            "convergence",
            "Operational GO or NO-GO receipt",
            label,
            "late_death_refinement/FULL_CLASSIFICATION_GO_NO_GO.json",
            "Read completeness, d0 invariance, dual-view, and calibration gates.",
        )
    )
    return result


def image_block(
    blocks: list[dict[str, Any]],
    image_data: dict[str, tuple[str, int, int]],
    key: str,
    heading: str,
    text: str,
    caption: str,
    source_id: str = "qc",
    carousel_group: str | None = None,
    carousel_index: int | None = None,
    carousel_label: str | None = None,
) -> None:
    if (carousel_group is None) != (carousel_index is None):
        raise ValueError("Carousel group and index must be provided together")
    heading_level = "####" if carousel_group is not None else "###"
    blocks.extend(
        (
            {
                "id": f"{key}_text",
                "type": "markdown",
                "body": f"{heading_level} {heading}\n\n{text}",
                "sourceId": source_id,
            },
            {
                "id": f"{key}_image",
                "type": "html",
                "body": COMMON.image_body(image_data[key], heading, caption),
                "layout": "full",
                **(
                    {
                        "carouselGroup": carousel_group,
                        "carouselIndex": carousel_index,
                        "carouselLabel": carousel_label or heading,
                        "carouselTextId": f"{key}_text",
                    }
                    if carousel_group is not None
                    else {}
                ),
            },
        )
    )


def report_manifest(
    image_data: dict[str, tuple[str, int, int]],
    qc_samples: list[dict[str, Any]],
    uncertainty_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    charts = [
        {
            "id": "stage_chart",
            "title": "Live and dead composition before and after late-death rescue",
            "subtitle": "Artifacts are excluded from the live-plus-dead denominator.",
            "type": "bar",
            "dataset": "stage_composition",
            "sourceId": "summaries",
            "valueFormat": "percent",
            "options": {"grouping": "stacked"},
            "encodings": {
                "x": {
                    "field": "stage_branch",
                    "type": "nominal",
                    "label": "Processing stage and analysis branch",
                },
                "y": {
                    "field": "share",
                    "type": "quantitative",
                    "label": "Share",
                    "format": "percent",
                },
                "color": {"field": "state", "type": "nominal", "label": "Cell state"},
                "tooltip": [
                    {
                        "field": "count",
                        "type": "quantitative",
                        "label": "Cells",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "rescue_day_chart",
            "title": "Late-death rescue rate over experimental time",
            "subtitle": "Rescued cells divided by total countable cells at each day.",
            "type": "line",
            "dataset": "rescue_by_day",
            "sourceId": "summaries",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "day", "type": "quantitative", "label": "Day"},
                "y": {
                    "field": "rescued_share",
                    "type": "quantitative",
                    "label": "Rescued share",
                    "format": "percent",
                },
                "color": {
                    "field": "branch",
                    "type": "nominal",
                    "label": "Analysis branch",
                },
                "tooltip": [
                    {
                        "field": "rescued",
                        "type": "quantitative",
                        "label": "Rescued cells",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "global_day_chart",
            "title": "Field-level late-collapse calls over time",
            "subtitle": "Share of branch-fields satisfying the persistent multi-site collapse gate.",
            "type": "line",
            "dataset": "global_fields_by_day",
            "sourceId": "refinement",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "day", "type": "quantitative", "label": "Day"},
                "y": {
                    "field": "field_share",
                    "type": "quantitative",
                    "label": "Field share",
                    "format": "percent",
                },
                "color": {
                    "field": "branch",
                    "type": "nominal",
                    "label": "Analysis branch",
                },
                "tooltip": [
                    {
                        "field": "global_late_fields",
                        "type": "quantitative",
                        "label": "Global late fields",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "condition_chart",
            "title": "Late-death rescue rate by ploidy and treatment background",
            "subtitle": "Full time course; denominator is total countable cells in each group.",
            "type": "bar",
            "dataset": "rescue_by_condition",
            "sourceId": "summaries",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "group", "type": "nominal", "label": "Plate group"},
                "y": {
                    "field": "rescued_share",
                    "type": "quantitative",
                    "label": "Rescued share",
                    "format": "percent",
                },
                "color": {
                    "field": "branch",
                    "type": "nominal",
                    "label": "Analysis branch",
                },
                "tooltip": [
                    {
                        "field": "rescued",
                        "type": "quantitative",
                        "label": "Rescued cells",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "relation_chart",
            "title": "Confirmed Death-object relation composition",
            "subtitle": "Aggregated final object relations in each full-cohort branch.",
            "type": "bar",
            "dataset": "relation_composition",
            "sourceId": "summaries",
            "valueFormat": "percent",
            "options": {"grouping": "stacked"},
            "encodings": {
                "x": {"field": "branch", "type": "nominal", "label": "Analysis branch"},
                "y": {
                    "field": "share",
                    "type": "quantitative",
                    "label": "Share",
                    "format": "percent",
                },
                "color": {
                    "field": "relation",
                    "type": "nominal",
                    "label": "Final object relation",
                },
                "tooltip": [
                    {
                        "field": "count",
                        "type": "quantitative",
                        "label": "Objects",
                        "format": "number",
                    }
                ],
            },
        },
    ]
    tables = [
        {
            "id": "qc_selection_table",
            "title": "Ploidy-by-treatment-by-time QC sample selection",
            "subtitle": (
                "Deterministic Day 0, Day 3, and Day 5 longitudinal representatives; "
                "intermediate 50–100 nM Dox fields remain in the cohort analysis "
                "but are not part of the low/high gallery."
            ),
            "dataset": "qc_selection",
            "sourceId": "qc_selection",
            "defaultSort": {"field": "selection_index", "direction": "asc"},
            "columns": [
                {
                    "field": "selection_index",
                    "label": "Order",
                    "format": "number",
                },
                {"field": "ploidy", "label": "Ploidy", "type": "text"},
                {"field": "condition", "label": "Treatment", "type": "text"},
                {"field": "time_label", "label": "Time stage", "type": "text"},
                {"field": "key", "label": "Field key", "type": "text"},
                {"field": "well", "label": "Well", "type": "text"},
                {"field": "site", "label": "Site", "format": "number"},
                {"field": "replicate", "label": "Replicate", "format": "number"},
                {
                    "field": "elapsed_hours",
                    "label": "Elapsed hours",
                    "format": "number",
                },
                {
                    "field": "doxorubicin_nm",
                    "label": "Dox (nM)",
                    "format": "number",
                },
                {
                    "field": "dead_fraction",
                    "label": "Final dead fraction",
                    "format": "percent",
                },
                {"field": "rescued", "label": "Rescued", "format": "number"},
                {
                    "field": "uncertain",
                    "label": "Uncertain",
                    "format": "number",
                },
            ],
        },
        {
            "id": "d0_uncertainty_table",
            "title": "d0 objects retained as operational uncertainty",
            "subtitle": (
                "Object-level dual-view conflicts only; no d0 object was rescued "
                "by the late-death rule."
            ),
            "dataset": "d0_uncertainty_cases",
            "sourceId": "refinement",
            "defaultSort": {"field": "key", "direction": "asc"},
            "columns": [
                {"field": "key", "label": "Field key", "type": "text"},
                {
                    "field": "combined_mask_id",
                    "label": "Combined mask ID",
                    "format": "number",
                },
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "reason", "label": "Reason", "type": "text"},
            ],
        },
        {
            "id": "convergence_table",
            "title": "Operational full-cohort validation gates",
            "subtitle": (
                "Proxy validation only; PASS does not estimate biological "
                "sensitivity or specificity."
            ),
            "dataset": "convergence_gates",
            "sourceId": "convergence",
            "defaultSort": {"field": "gate", "direction": "asc"},
            "columns": [
                {"field": "gate", "label": "Gate", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
            ],
        },
        {
            "id": "branch_table",
            "title": "Full-cohort classification totals",
            "subtitle": "Exact pre-refinement, final, supplemental, and object-aware totals.",
            "dataset": "branch_totals",
            "sourceId": "summaries",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {
                    "field": "total_cell_count",
                    "label": "Countable cells",
                    "format": "number",
                },
                {
                    "field": "pre_late_death_dead_cell_count",
                    "label": "Dead before rescue",
                    "format": "number",
                },
                {
                    "field": "late_death_rescue_count",
                    "label": "Rescued deaths",
                    "format": "number",
                },
                {
                    "field": "dead_cell_count",
                    "label": "Final dead cells",
                    "format": "number",
                },
                {
                    "field": "supplemental_dead_object_count",
                    "label": "Supplemental Death objects",
                    "format": "number",
                },
                {
                    "field": "object_aware_dead_count",
                    "label": "Object-aware death total",
                    "format": "number",
                },
                {
                    "field": "late_death_uncertain_count",
                    "label": "Late-stage uncertainty",
                    "format": "number",
                },
            ],
        },
        {
            "id": "top_rescue_table",
            "title": "Fields with the largest late-death rescue counts",
            "subtitle": "Eight highest-rescue fields per analysis branch.",
            "dataset": "top_rescued_fields",
            "sourceId": "refinement",
            "defaultSort": {"field": "rescued", "direction": "desc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "key", "label": "Field key", "type": "text"},
                {
                    "field": "elapsed_hours",
                    "label": "Elapsed hours",
                    "format": "number",
                },
                {"field": "rescued", "label": "Rescued", "format": "number"},
                {"field": "pre_dead", "label": "Pre dead", "format": "number"},
                {"field": "final_dead", "label": "Final dead", "format": "number"},
                {
                    "field": "global_late_field",
                    "label": "Global field collapse",
                    "type": "text",
                },
            ],
        },
        {
            "id": "hill_table",
            "title": "AUC Hill-fit parameters",
            "subtitle": "Full-cohort 2N and 4N fits for both treatment backgrounds.",
            "dataset": "dose_hill",
            "sourceId": "dose",
            "defaultSort": {"field": "condition", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "condition", "label": "Condition", "type": "text"},
                {"field": "ploidy", "label": "Ploidy", "type": "text"},
                {"field": "ec50_nm", "label": "EC50 (nM)", "format": "number"},
                {"field": "bottom", "label": "Bottom", "format": "number"},
                {"field": "top", "label": "Top", "format": "number"},
                {"field": "hill_slope", "label": "Hill slope", "format": "number"},
                {"field": "r_squared", "label": "R²", "format": "number"},
            ],
        },
        {
            "id": "gr_table",
            "title": "Integrated ploidy difference in GR",
            "subtitle": "Positive Delta GR means the 4N response is higher than the 2N response.",
            "dataset": "gr_delta",
            "sourceId": "dose",
            "defaultSort": {"field": "endpoint", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "endpoint", "label": "Endpoint", "type": "text"},
                {"field": "condition", "label": "Condition", "type": "text"},
                {
                    "field": "mean_delta_gr",
                    "label": "Mean Delta GR",
                    "format": "number",
                },
                {"field": "ci_low", "label": "CI low", "format": "number"},
                {"field": "ci_high", "label": "CI high", "format": "number"},
            ],
        },
        {
            "id": "death_table",
            "title": "Integrated ploidy difference in excess lethal fraction",
            "subtitle": "Positive values mean greater adjusted death in 4N than 2N.",
            "dataset": "death_delta",
            "sourceId": "dose",
            "defaultSort": {"field": "endpoint", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "endpoint", "label": "Endpoint", "type": "text"},
                {"field": "condition", "label": "Condition", "type": "text"},
                {
                    "field": "mean_delta_excess_lf",
                    "label": "Mean Delta excess LF",
                    "format": "number",
                },
                {"field": "ci_low", "label": "CI low", "format": "number"},
                {"field": "ci_high", "label": "CI high", "format": "number"},
            ],
        },
    ]
    cards = [
        {
            "id": "scope_card",
            "description": "Strictly complete full-cohort classification.",
            "dataset": "overview",
            "sourceId": "summaries",
            "metrics": [
                {
                    "label": "Branch-fields",
                    "field": "total_branch_fields",
                    "format": "number",
                },
                {
                    "label": "Analysis branches",
                    "field": "analysis_branches",
                    "format": "number",
                },
            ],
        },
        {
            "id": "rescue_card",
            "description": "Objects reclassified by the late-stage production gate.",
            "dataset": "overview",
            "sourceId": "refinement",
            "metrics": [
                {
                    "label": "Rescued deaths",
                    "field": "total_rescued",
                    "format": "number",
                },
                {
                    "label": "Uncertainty annotations",
                    "field": "total_uncertain",
                    "format": "number",
                },
            ],
        },
        {
            "id": "analysis_card",
            "description": (
                "Operational validation decision and rebuilt downstream "
                "dose-response inventory."
            ),
            "dataset": "overview",
            "sourceId": "convergence",
            "metrics": [
                {
                    "label": "Validation gates passed",
                    "field": "passed_validation_gates",
                    "format": "number",
                },
                {
                    "label": "Dose-response files",
                    "field": "dose_response_files",
                    "format": "number",
                }
            ],
        },
    ]
    blocks: list[dict[str, Any]] = [
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "# SUM159 Full-Cohort Dead-Classification Report\n\n"
                "This QC-first report shows the saved raw channels, frozen segmentation "
                "boundaries, and final dual-view classification before cohort statistics. "
                "Thirty-six deterministic fields span 2N and 4N cells, no-drug, "
                "cyclophosphamide-only, low-dose Dox, and high-dose Dox conditions at "
                "Day 0, Day 3, and Day 5. The "
                "classification run is computationally complete, but its strict production "
                "receipt is NO-GO because two d0 original-branch objects remain uncertain after "
                "the original and nucleated-only final calls disagree. No segmentation was "
                "changed or rerun."
            ),
        },
    ]
    blocks.append(
        {
            "id": "workflow_intro",
            "type": "markdown",
            "body": (
                "## Workflow\n\n"
                "The diagram traces immutable segmentation inputs through base "
                "multichannel classification, late-death trajectory calibration, "
                "three rescue evidence paths, final cell-state decisions, and the "
                "authoritative field-level outputs used by this report."
            ),
            "sourceId": "workflow",
        }
    )
    image_block(
        blocks,
        image_data,
        "classification_workflow",
        "Figure 1. Death-classification workflow",
        (
            "Read the workflow from left to right. Segmentation remains frozen; "
            "classification and late-death refinement operate on saved masks, raw "
            "channels, trajectory context, and cross-view evidence."
        ),
        (
            "End-to-end production death-classification workflow. Click the image "
            "to inspect the high-resolution diagram; zoom or drag inside the viewer "
            "and click outside the image to return to the report."
        ),
        source_id="workflow",
    )
    blocks.extend(
        [
        {
            "id": "qc_selection_intro",
            "type": "markdown",
            "body": (
                "## QC is organized by ploidy, treatment, and time\n\n"
                "The gallery first separates 2N and 4N cells, retains the existing treatment "
                "groups, and then orders each group as Day 0, Day 3, and Day 5. Low "
                "Dox is 3.125–25 nM and high Dox is 200–800 nM; 50–100 nM fields remain in all "
                "cohort-level analyses but are intentionally outside this low/high visual "
                "contrast. Whenever possible, the three time points use the same well and site "
                "to provide a direct longitudinal comparison. Selection does not use density, "
                "dead fraction, uncertainty, or any other classification outcome. Every figure uses "
                "the same ten-panel, three-row-by-four-column layout and the same "
                "display normalization. The final classification is followed by the "
                "matched previous classification from classification_20260721_102004."
            ),
            "sourceId": "field_states",
        },
        {
            "id": "qc_selection_table_block",
            "type": "table",
            "tableId": "qc_selection_table",
        },
        ]
    )
    figure_number = 2
    for ploidy in QC_PLOIDY_ORDER:
        blocks.append(
            {
                "id": f"qc_ploidy_{ploidy.lower()}",
                "type": "markdown",
                "body": (
                    f"## {ploidy}: authoritative QC across treatment and time\n\n"
                    "Each treatment block below is ordered Day 0, Day 3, then Day 5."
                ),
                "sourceId": "qc_selection",
            }
        )
        for condition in QC_CONDITION_GROUPS:
            condition_samples = [
                sample
                for sample in qc_samples
                if sample["ploidy"] == ploidy
                and sample["condition_id"] == condition["id"]
            ]
            blocks.append(
                {
                    "id": f"qc_group_{ploidy.lower()}_{condition['id']}",
                    "type": "markdown",
                    "body": (
                    f"### {ploidy} · {condition['label']}: Day 0 to Day 5\n\n"
                    "Read each three-row-by-four-column composite from raw signals to frozen masks, "
                    "the final authoritative classification, and the matched previous "
                    "classification. Green boundaries are live, purple "
                    "boundaries are dead, yellow boundaries are uncertain or transitional, and "
                    "gray boundaries are artifacts. Classification boundaries are rendered at "
                    "double width. Orange, cyan/magenta, and blue are reserved for the separate "
                    "Combined/Brightfield, Nuclei, and Dead segmentation panels."
                    ),
                    "sourceId": "qc_selection",
                }
            )
            for sample in condition_samples:
                key = f"qc_sample_{sample['selection_index']:02d}"
                treatment_detail = (
                    f"{sample['ploidy']}; {sample['time_label']} "
                    f"({sample['elapsed_hours']:g} h); "
                    f"{sample['doxorubicin_nm']:g} nM Dox; "
                    f"{'with' if sample['cyclophosphamide'] else 'without'} "
                    f"cyclophosphamide; well {sample['well']}, site {sample['site']}, "
                    f"replicate {sample['replicate']}. Final authoritative result: "
                    f"{sample['total_cells']:,} countable cells, "
                    f"{sample['dead_fraction']:.1%} dead, {sample['rescued']:,} rescued, "
                    f"and {sample['uncertain']:,} uncertain."
                )
                image_block(
                    blocks,
                    image_data,
                    key,
                    (
                        f"Figure {figure_number}. {sample['ploidy']} · "
                        f"{sample['condition']} · {sample['time_label']} · "
                        f"{sample['key']}"
                    ),
                    treatment_detail,
                    (
                        f"{sample['key']}: Combined RGB, Brightfield, Nuclei, and Dead "
                        "raw images; frozen channel-specific segmentation overlays; and the final "
                        "authoritative classification."
                    ),
                    source_id="qc_selection",
                    carousel_group=f"full_{ploidy.lower()}_{condition['id']}",
                    carousel_index=int(sample["time_order"]) - 1,
                    carousel_label=sample["time_label"],
                )
                figure_number += 1

    blocks.extend(
        (
            {
                "id": "d0_uncertainty_intro",
                "type": "markdown",
                "body": (
                    "## Two d0 objects remain unresolved between the frozen segmentation views\n\n"
                    "The late-death rule rescued zero d0 objects. The strict d0 gate nevertheless "
                    "failed because two original-branch objects were assigned uncertainty when "
                    "their final call disagreed with the matched nucleated-only view. These are "
                    "object-level operational conflicts, not evidence that an entire d0 image is "
                    "uninterpretable. White outlines identify the original mask under review."
                ),
                "sourceId": "convergence",
            },
            {
                "id": "d0_uncertainty_table_block",
                "type": "table",
                "tableId": "d0_uncertainty_table",
            },
        )
    )
    for index, case in enumerate(uncertainty_cases, 1):
        key = f"d0_uncertainty_{index:02d}"
        image_block(
            blocks,
            image_data,
            key,
            (
                f"Figure {figure_number}. d0 uncertainty · {case['key']} · "
                f"mask {case['combined_mask_id']}"
            ),
            (
                f"{case['reason']}. The same spatial crop is shown in Combined RGB, "
                "Dead, Nuclei, original final classification, and nucleated-only final "
                "classification."
            ),
            (
                f"{case['key']} mask {case['combined_mask_id']}: focused d0 "
                "dual-view uncertainty audit."
            ),
            source_id="refinement",
            carousel_group="full_d0_uncertainty",
            carousel_index=index - 1,
            carousel_label=f"Case {index}",
        )
        figure_number += 1
        figure_number += 1

    blocks.extend(
        (
            {
                "id": "analysis_intro",
                "type": "markdown",
                "body": (
                    "## Full-cohort analysis follows the image-level QC evidence\n\n"
                    "The remaining sections quantify the complete 27,200-field-per-branch run. "
                    "They summarize operational validation, pre-versus-final composition, "
                    "late-death rescue over time and treatment background, final Death-object "
                    "relations, and rebuilt dose-response outputs."
                ),
            },
            {
                "id": "cards",
                "type": "metric-strip",
                "cardIds": ["scope_card", "rescue_card", "analysis_card"],
            },
            {
                "id": "convergence_result",
                "type": "markdown",
                "body": (
                    "## Full-cohort execution is complete, but the d0 invariance gate is NO-GO\n\n"
                    "The receipt checks the frozen-segmentation contract, calibration "
                    "convergence, full-cohort completeness, d0 invariance, and dual-view "
                    "stability. All computational stages completed; the only failed scientific "
                    "gate is strict d0 invariance because of the two uncertainty calls shown "
                    "above. These operational gates do not estimate biological sensitivity or "
                    "specificity."
                ),
                "sourceId": "convergence",
            },
            {
                "id": "convergence_table_block",
                "type": "table",
                "tableId": "convergence_table",
            },
            {
                "id": "stage_result",
                "type": "markdown",
                "body": (
                    "## Late-death refinement changes later classifications while rescuing no d0 object\n\n"
                    "The chart compares saved pre-refinement counts with final production counts. "
                    "Later fields can accumulate probable-death rescues or explicit uncertainty; "
                    "at d0, rescue remains zero while two cross-view conflicts are retained as "
                    "uncertain rather than forced into live or dead."
                ),
                "sourceId": "summaries",
            },
            {"id": "stage_block", "type": "chart", "chartId": "stage_chart"},
            {"id": "branch_table_block", "type": "table", "tableId": "branch_table"},
            {
                "id": "time_result",
                "type": "markdown",
                "body": (
                    "## Rescue and field-collapse calls emerge after the late-time gate\n\n"
                    "Rescue rates and global field-collapse calls are summarized by experimental "
                    "day. The current field state can deactivate after three sustained normalized "
                    "frames, so a transient collapse does not remain active forever."
                ),
                "sourceId": "summaries",
            },
            {"id": "rescue_day_block", "type": "chart", "chartId": "rescue_day_chart"},
            {"id": "global_day_block", "type": "chart", "chartId": "global_day_chart"},
            {
                "id": "condition_result",
                "type": "markdown",
                "body": (
                    "## Ploidy and cyclophosphamide backgrounds retain separate summaries\n\n"
                    "The condition view reports how often the late stage contributes within each "
                    "plate group. It is descriptive and should be interpreted with the image QC, "
                    "well trajectories, and dose-response analyses."
                ),
                "sourceId": "summaries",
            },
            {"id": "condition_block", "type": "chart", "chartId": "condition_chart"},
            {
                "id": "relation_result",
                "type": "markdown",
                "body": (
                    "## Final object relations preserve overlapping and supplemental death evidence\n\n"
                    "The relation ledger distinguishes same-cell death from multi-nucleus overlap, "
                    "live-with-death-signal, dead-only regions, unresolved overlap, and merged "
                    "objects instead of forcing one biological region into incompatible labels."
                ),
                "sourceId": "summaries",
            },
            {"id": "relation_block", "type": "chart", "chartId": "relation_chart"},
            {
                "id": "top_rescue_text",
                "type": "markdown",
                "body": (
                    "## The largest late-death changes remain directly auditable\n\n"
                    "The table identifies the highest-rescue fields in each branch and records "
                    "whether the persistent global field-collapse gate was active."
                ),
                "sourceId": "refinement",
            },
            {"id": "top_rescue_block", "type": "table", "tableId": "top_rescue_table"},
        )
    )
    dose_start = figure_number
    image_block(
        blocks,
        image_data,
        "well_live_dead_counts_over_time",
        f"Figure {dose_start - 1}. Well-level live and dead counts over time",
        (
            "The authoritative fusion-consensus trajectories retain every well and "
            "time point, allowing treatment, ploidy, and replicate behavior to be "
            "reviewed before dose-response reduction."
        ),
        (
            "Saved fusion-consensus well-level live and dead trajectories from "
            "well_live_dead_counts_over_time.pdf."
        ),
        source_id="well_count_plot",
    )
    blocks.extend(
        (
            {
                "id": "dose_intro",
                "type": "markdown",
                "body": (
                    "## Full-cohort dose response is rebuilt from the refined classifications\n\n"
                    "The dose-response stage consumes the strict-completeness well-by-time tables "
                    "generated after late-death refinement. It recreates AUC, exact Day-4, exact "
                    "Day-5, growth-rate inhibition, and excess-lethal-fraction analyses for the "
                    "authoritative consensus and both diagnostic classification branches."
                ),
                "sourceId": "dose",
            },
        )
    )
    dose_panels = dose_panel_records()
    for metric_index, metric in enumerate(DOSE_FIGURE_SPECS):
        figure_id = dose_start + metric_index
        metric_panels = [
            panel
            for panel in dose_panels
            if panel["metric_id"] == metric["id"]
        ]
        blocks.append(
            {
                "id": f"dose_{metric['id']}_group",
                "type": "markdown",
                "body": (
                    f"### Figure {figure_id}. {metric['title']}\n\n"
                    "Use the arrows, dots, keyboard, or horizontal swipe to review "
                    "each classification branch and treatment background as an "
                    "individual plot."
                ),
                "sourceId": "dose",
            }
        )
        for panel_index, panel in enumerate(metric_panels):
            branch_label = COMMON.display_branch(panel["branch"])
            panel_label = f"{branch_label} · {panel['condition']}"
            panel_letter = chr(ord("A") + panel_index)
            image_block(
                blocks,
                image_data,
                panel["key"],
                (
                    f"Figure {figure_id}{panel_letter}. {metric['title']} · "
                    f"{panel_label}"
                ),
                f"{metric['text']} This slide shows {panel_label}.",
                f"{metric['caption']} {panel_label}.",
                source_id="dose",
                carousel_group=f"full_dose_{metric['id']}",
                carousel_index=panel_index,
                carousel_label=panel_label,
            )
        blocks.append(
            {
                "id": metric["table_block_id"],
                "type": "table",
                "tableId": metric["table_id"],
            }
        )
    blocks.extend(
        (
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Scope, outputs, and denominators\n\n"
                    "Both frozen mask branches contain 27,200 fields covering 80 wells, four "
                    "sites, and 85 time points. The consensus output uses the original branch as "
                    "the authoritative cell denominator and carries nucleated-only measurements "
                    "as diagnostics. Cell-state percentages use live plus dead cells unless "
                    "stated otherwise; artifacts are excluded."
                ),
            },
            {
                "id": "process",
                "type": "markdown",
                "body": (
                    "## The final method combines object attribution, field collapse, and dual-view trajectories\n\n"
                    "Initial classification combines Combined RGB state, Dead-channel evidence, "
                    "Brightfield support, nucleus support, object overlap, and nucleus "
                    "multiplicity. The late stage calibrates cell-conditioned Dead signal, "
                    "nuclear-to-cytoplasmic ratio, red-mass loss, cell and cytoplasm depletion, "
                    "shape, mask coverage, site concordance, and multi-frame persistence against "
                    "continuous d0 density and untreated time references. Strong live evidence "
                    "vetoes an initiating rescue; supported calls require compatible dual-view "
                    "evidence, while conflicts remain uncertain."
                ),
                "sourceId": "configuration",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and uncertainty\n\n"
                    "No manual object-level ground truth is available. Late-death uncertainty "
                    "annotations identify objects near the operational boundary, but neither the "
                    "calibration anchors nor the full-cohort rescue fractions are biological "
                    "sensitivity or specificity estimates. Dose-response confidence bands use "
                    "paired bootstrap resampling with only two replicate series per ploidy and "
                    "treatment condition and are therefore descriptive."
                ),
            },
            {
                "id": "recommended_use",
                "type": "markdown",
                "body": (
                    "## Recommended downstream use\n\n"
                    "Use the fusion-consensus outputs for primary well-level and dose-response "
                    "analysis. Use final dead-cell counts when the analytical unit is a Combined cell. Use "
                    "object-aware death totals when supplemental confirmed Death objects are part "
                    "of the biological question. Preserve the analysis branch, uncertainty layer, "
                    "configuration hash, and pre-refinement counts in every downstream comparison."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "A blinded manual audit of selected late treated fields would be required to "
                    "measure biological false-positive and false-negative rates. Until then, "
                    "branch agreement, trajectory persistence, uncertainty, and deterministic QC "
                    "should remain the primary regression checks."
                ),
            },
        )
    )
    return {"cards": cards, "charts": charts, "tables": tables, "blocks": blocks}


def main() -> int:
    args = parse_args()
    root = COMMON.require_dir(args.classification_root)
    previous_root = COMMON.require_dir(args.previous_classification_root)
    calibration_root = (
        COMMON.require_dir(args.calibration_root)
        if args.calibration_root is not None
        else None
    )
    plate_map_path = COMMON.require_file(args.plate_map)
    workflow_pdf = COMMON.require_file(args.workflow_pdf)
    validated = validate_inputs(
        root,
        previous_root,
        calibration_root,
        args.expected_fields_per_branch,
        args.expected_timepoints,
        args.expected_dose_response_files,
    )
    plate_map = load_plate_map(plate_map_path)
    qc_samples = select_qc_samples(validated, plate_map)
    uncertainty_cases = d0_uncertainty_cases(validated)
    datasets = aggregate_datasets(root, validated, plate_map)
    datasets["qc_selection"] = qc_samples
    datasets["d0_uncertainty_cases"] = uncertainty_cases
    figures = report_figures(
        root,
        previous_root,
        workflow_pdf,
        qc_samples,
        uncertainty_cases,
    )
    if args.debug_figure_dir is not None:
        debug = args.debug_figure_dir.expanduser().resolve()
        debug.mkdir(parents=True, exist_ok=True)
        for key, figure in figures.items():
            figure.save(debug / f"{key}.png")
    image_data, high_resolution = COMMON.encode_figures(figures)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "analysis" / "reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_selection_path = output_dir / "qc_sample_selection.csv"
    if qc_selection_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite without --force: {qc_selection_path}"
        )
    write_qc_selection_csv(qc_selection_path, qc_samples)
    report_sources = sources(root, previous_root, calibration_root)
    timestamp = COMMON.generated_at()
    manifest = report_manifest(image_data, qc_samples, uncertainty_cases)
    title = "SUM159 Full-Cohort Dead-Classification Report"
    artifact = COMMON.artifact_payload(
        title=title,
        description=(
            "QC-first technical full-cohort report for raw-channel review, frozen "
            "segmentation overlays, final object-aware classification, late-death "
            "trajectory refinement, and downstream cohort analyses."
        ),
        manifest=manifest,
        datasets=datasets,
        sources=report_sources,
        origin="artifact://sum159-dead-classification/full-cohort",
        timestamp=timestamp,
    )
    submission = COMMON.parse_key_value_file(root / "SUBMISSION_SUMMARY.txt")
    receipt = {
        "report_mode": "full_cohort_classification",
        "classification_root": str(root),
        "previous_classification_root": str(previous_root),
        "calibration_root": str(calibration_root) if calibration_root else None,
        "approved_configuration_source": str(validated["approved_source"]),
        "generated_at": timestamp,
        "project_git_sha": submission.get("project_git_sha", ""),
        "dataset_rows": {key: len(value) for key, value in datasets.items()},
        "embedded_figures": sorted(figures),
        "qc_keys": [row["key"] for row in qc_samples],
        "qc_sample_count": len(qc_samples),
        "qc_condition_count": len(QC_CONDITION_GROUPS),
        "qc_ploidy_levels": list(QC_PLOIDY_ORDER),
        "qc_time_levels_hours": [
            float(timepoint["hours"]) for timepoint in QC_TIMEPOINTS
        ],
        "qc_panels_per_sample": 10,
        "qc_grid_columns": 4,
        "qc_grid_rows": 3,
        "qc_composite_font_scale": QC_COMPOSITE_FONT_SCALE,
        "workflow_pdf": str(workflow_pdf),
        "workflow_figure_embedded": True,
        "classification_boundary_width": CLASSIFICATION_BOUNDARY_WIDTH,
        "classification_colors": {
            state: [int(value) for value in color]
            for state, color in STATE_COLORS.items()
        },
        "qc_selection_csv": str(qc_selection_path),
        "d0_uncertainty_cases": uncertainty_cases,
        "fields_per_branch": args.expected_fields_per_branch,
        "refinement_rows": len(validated["refinement"]),
        "refinement_failures": 0,
        "dose_response_files": len(validated["dose_files"]),
        "dose_response_embedded_panel_count": len(dose_panel_records()),
        "well_count_plot_pdf": str(validated["well_count_plot_pdf"]),
        "well_count_plot_embedded": True,
        "configuration_sha256": validated["production"].get(
            "configuration_sha256", ""
        ),
        "frozen_model_sha256": validated["production"].get(
            "frozen_model_sha256", ""
        ),
    }
    result = COMMON.package_report(
        artifact=artifact,
        artifact_path=output_dir
        / "DEAD_CLASSIFICATION_FULL_COHORT_REPORT.artifact.json",
        output_html=output_dir / "DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html",
        build_receipt=output_dir
        / "DEAD_CLASSIFICATION_FULL_COHORT_REPORT.build.json",
        plugin_root=(
            args.plugin_root.expanduser().resolve()
            if args.plugin_root is not None
            else COMMON.discover_plugin_root()
        ),
        force=args.force,
        high_resolution_images=high_resolution,
        receipt=receipt,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
