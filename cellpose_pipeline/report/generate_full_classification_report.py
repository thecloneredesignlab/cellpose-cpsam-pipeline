#!/usr/bin/env python3
"""Build the self-contained SUM159 full-cohort classification report."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
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
BRANCH_DIRS = {
    "original": "classification_fusion",
    "nucleated_only": "classification_fusion_nucleated_only",
}
BRANCH_ANALYSIS_DIRS = {
    "consensus": "fusion-consensus",
    "original": "fusion",
    "nucleated_only": "fusion-nucleated-only",
}
STATE_COLORS = {
    "live": np.array([45, 205, 110], dtype=np.float32),
    "dead": np.array([235, 70, 72], dtype=np.float32),
    "artifact": np.array([145, 150, 160], dtype=np.float32),
    "uncertain": np.array([255, 220, 55], dtype=np.float32),
    "transitional": np.array([232, 139, 57], dtype=np.float32),
}
FIXED_QC_KEYS = (
    "E2_1_00d00h00m",
    "E9_1_05d00h00m",
    "E9_2_05d00h00m",
    "F9_1_05d00h00m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        help="Approved d0+d5 calibration root used only for configuration provenance validation.",
    )
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
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


def validate_inputs(
    root: Path,
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
    }


def load_plate_map(path: Path) -> dict[str, dict[str, str]]:
    rows = COMMON.read_csv(path)
    mapping = {row["well"]: row for row in rows}
    if len(mapping) != 80:
        raise RuntimeError(f"Expected 80 plate-map wells, found {len(mapping)}")
    return mapping


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


def label_boundary(labels: np.ndarray) -> np.ndarray:
    labels = np.squeeze(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels > 0)


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
    rescued = np.zeros(maximum + 1, dtype=bool)
    for row in predictions:
        mask_id = COMMON.as_int(row.get("mask_id", 0))
        if mask_id <= 0 or mask_id > maximum:
            continue
        state = row.get(state_field, "") or row.get("state", "")
        colors[mask_id] = STATE_COLORS.get(
            state,
            np.array([245, 247, 250], dtype=np.float32),
        )
        rescued[mask_id] = COMMON.truthy(row.get("late_death_rescue_call", ""))
    rescue_pixels = rescued[labels]
    if rescue_pixels.any():
        output[rescue_pixels] = (
            output[rescue_pixels] * 0.72
            + np.array([235, 70, 72], dtype=np.float32) * 0.28
        )
    boundary = label_boundary(labels)
    output[boundary] = output[boundary] * 0.15 + colors[labels[boundary]] * 0.85
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


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


def comparison_figure(root: Path, key: str) -> Image.Image:
    record = COMMON.read_json(find_record(root, key))
    combined = record["profiles"]["Combined"]
    raw_rgb = normalize_rgb(tifffile.imread(combined["raw"]))
    panels: list[tuple[Image.Image, str]] = []
    for stage_field, stage_label in (
        ("pre_late_death_state", "Before late-death rescue"),
        ("state", "Final production result"),
    ):
        for branch in ("original", "nucleated_only"):
            mask_field = "original_mask" if branch == "original" else "nucleated_mask"
            labels = tifffile.imread(combined[mask_field])
            predictions = COMMON.read_csv(find_prediction(root, branch, key))
            panels.append(
                (
                    state_overlay(raw_rgb, labels, predictions, stage_field),
                    f"{COMMON.display_branch(branch)} · {stage_label}",
                )
            )
    # Row 1 is before and row 2 is final; each branch stays in one vertical column.
    return COMMON.labeled_grid(
        panels,
        title=f"{key}: pre-refinement and final cell states",
        columns=2,
        panel_width=1408,
    )


def selected_qc_keys(
    validated: dict[str, Any],
) -> list[str]:
    rescued_by_key: Counter[str] = Counter()
    for row in validated["refinement"]:
        rescued_by_key[row["key"]] += COMMON.as_int(row["rescued"])
    extras = [
        key
        for key, rescued in rescued_by_key.most_common()
        if rescued > 0 and key not in FIXED_QC_KEYS
    ][:2]
    keys = list(FIXED_QC_KEYS) + extras
    if len(keys) < 6:
        raise RuntimeError("Unable to select six deterministic full-cohort QC fields")
    return keys


def dose_figure(
    root: Path,
    subdirectory: str,
    filename: str,
    title: str,
) -> Image.Image:
    panels: list[tuple[Image.Image, str]] = []
    for branch, analysis_dir in BRANCH_ANALYSIS_DIRS.items():
        for condition, condition_file in (
            ("Doxorubicin alone", f"doxorubicin_alone_2N_vs_4N_{filename}.png"),
            (
                "Doxorubicin + cyclophosphamide",
                f"doxorubicin_plus_cyclophosphamide_2N_vs_4N_{filename}.png",
            ),
        ):
            panels.append(
                (
                    COMMON.read_image(
                        root
                        / "analysis"
                        / "dose_response"
                        / analysis_dir
                        / subdirectory
                        / condition_file
                    ),
                    f"{COMMON.display_branch(branch)} · {condition}",
                )
            )
    return COMMON.labeled_grid(panels, title=title, columns=2, panel_width=1250)


def report_figures(
    root: Path,
    validated: dict[str, Any],
) -> tuple[dict[str, Image.Image], list[str]]:
    qc_keys = selected_qc_keys(validated)
    figures = {
        f"qc_{index}": comparison_figure(root, key)
        for index, key in enumerate(qc_keys, 1)
    }
    figures.update(
        {
            "dose_auc": dose_figure(
                root,
                "auc",
                "hill",
                "AUC-normalized dose responses",
            ),
            "dose_gr": dose_figure(
                root,
                "gr",
                "gr",
                "Day-4 and Day-5 growth-rate inhibition",
            ),
            "dose_death": dose_figure(
                root,
                "death",
                "excess_lethal_fraction",
                "Day-4 and Day-5 excess lethal fraction",
            ),
        }
    )
    return figures, qc_keys


def sources(root: Path, calibration_root: Path | None) -> list[dict[str, Any]]:
    label = root.name
    result = [
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
            "dose",
            "Authoritative full-cohort dose-response analyses",
            label,
            "analysis/dose_response/fusion-consensus/auc/hill_fit_parameters.csv",
            "Read AUC, endpoint, GR, and excess-lethal-fraction analyses.",
        ),
        COMMON.logical_source(
            "qc",
            "Final full-cohort classification QC",
            label,
            "classification_fusion/qc/label_overlays/*.png",
            "Read deterministic pre-refinement and final QC comparisons.",
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
) -> None:
    blocks.extend(
        (
            {
                "id": f"{key}_text",
                "type": "markdown",
                "body": f"### {heading}\n\n{text}",
            },
            {
                "id": f"{key}_image",
                "type": "html",
                "body": COMMON.image_body(image_data[key], heading, caption),
                "layout": "full",
            },
        )
    )


def report_manifest(
    image_data: dict[str, tuple[str, int, int]],
    qc_keys: list[str],
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
                "This classification-only run reuses the frozen v3 segmentation masks. "
                "The original and nucleated-only mask views are treated as independent "
                "diagnostic evidence, while the original-cell-mask summary remains the "
                "authoritative counting unit. A post-classification consensus stage combines "
                "continuous density- and time-matched live references, cell-conditioned Dead "
                "signal, nuclear-to-cytoplasmic ratio, red-mass loss, object shape, recoverable "
                "field-collapse states, and multi-frame spatial continuity. Strong live evidence "
                "vetoes an initiating rescue. A supported call is propagated only across a "
                "mutual-nearest dual-view pair, and any remaining final-call or tracking conflict "
                "is retained as uncertainty rather than forced into dead."
            ),
        },
        {
            "id": "cards",
            "type": "metric-strip",
            "cardIds": ["scope_card", "rescue_card", "analysis_card"],
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## Scope, outputs, and denominators\n\n"
                "Both the original-cell-mask and nucleated-only branches contain 27,200 fields "
                "covering 80 wells, four sites, and 85 time points. The consensus output uses "
                "the original branch as the authoritative cell denominator and carries the "
                "nucleated-only measurements as diagnostics. Cell-state percentages use "
                "live plus dead cells unless stated otherwise; artifacts are excluded. Confirmed "
                "Death objects may be associated with a dead cell or retained as supplemental "
                "objects, so the object-aware death total is not identical to dead-cell count."
            ),
        },
        {
            "id": "process",
            "type": "markdown",
            "body": (
                "## Classification proceeds from per-field object attribution to late trajectory rescue\n\n"
                "The first stage combines RGB state, Dead-channel evidence, Brightfield support, "
                "nucleus support, object overlap, and nucleus multiplicity. The late stage begins "
                "only after both per-field classification branches are merged. It calibrates "
                "object features against continuous d0 density percentiles and untreated "
                "time-matched live references; then it checks count, area, cytoplasm, red mass, "
                "cell-conditioned Dead signal, nuclear-to-cytoplasmic ratio, mask coverage, "
                "multi-site concordance, and multi-frame persistence. A field-collapse state can "
                "recover after sustained normalization, preventing a transient collapse from "
                "remaining active forever. Automatic rescue requires compatible dual-view or "
                "strong unmatched evidence and is blocked by strong live evidence. Dual-view "
                "stability is evaluated on mutual-matched objects rather than raw branch-level "
                "fractions, because the two frozen segmentation branches intentionally contain "
                "different cell populations."
            ),
        },
        {
            "id": "convergence_result",
            "type": "markdown",
            "body": (
                "## GO or NO-GO is based on operational proxies, not manual ground truth\n\n"
                "The receipt checks the frozen-segmentation contract, calibration convergence, "
                "full-cohort completeness, d0 invariance, and dual-view stability. Because no "
                "manual object-level annotations exist, these gates do not measure biological "
                "sensitivity or specificity and cannot substantiate a numerical accuracy claim."
            ),
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
                "## Late-death rescue changes the full-cohort composition without rewriting d0\n\n"
                "The chart compares saved pre-refinement counts with the final production result. "
                "The d0 no-change invariant is verified in the production receipt, while later "
                "treated fields can accumulate probable-death rescues or explicit uncertainty."
            ),
        },
        {"id": "stage_block", "type": "chart", "chartId": "stage_chart"},
        {"id": "branch_table_block", "type": "table", "tableId": "branch_table"},
        {
            "id": "time_result",
            "type": "markdown",
            "body": (
                "## Rescue and field-collapse calls emerge after the late-time gate\n\n"
                "Rescue rates and global field-collapse calls are summarized by experimental day. "
                "They remain absent at d0 and require persistent trajectory evidence. Unlike the "
                "previous absorbing state, the current field gate can deactivate after three "
                "sustained normalized frames."
            ),
        },
        {"id": "rescue_day_block", "type": "chart", "chartId": "rescue_day_chart"},
        {"id": "global_day_block", "type": "chart", "chartId": "global_day_chart"},
        {
            "id": "condition_result",
            "type": "markdown",
            "body": (
                "## Ploidy and treatment strata retain separate production summaries\n\n"
                "The condition view reports how frequently the late-stage rule contributes within "
                "each plate group. It is descriptive and should be interpreted alongside the "
                "well-level trajectories and dose-response analyses."
            ),
        },
        {"id": "condition_block", "type": "chart", "chartId": "condition_chart"},
        {
            "id": "relation_result",
            "type": "markdown",
            "body": (
                "## Final object relations preserve overlapping and supplemental death evidence\n\n"
                "The object relation ledger distinguishes same-cell death from multi-nucleus "
                "overlap, live-with-death-signal, dead-only regions, unresolved overlap, and "
                "merged objects. This prevents one biological region from being forced into a "
                "single mutually incompatible cell/object label."
            ),
        },
        {"id": "relation_block", "type": "chart", "chartId": "relation_chart"},
        {
            "id": "top_rescue_text",
            "type": "markdown",
            "body": (
                "## The highest-rescue fields remain directly auditable\n\n"
                "The table identifies the largest field-level changes in each branch and records "
                "whether the persistent global collapse gate was active."
            ),
        },
        {"id": "top_rescue_block", "type": "table", "tableId": "top_rescue_table"},
        {
            "id": "qc_intro",
            "type": "markdown",
            "body": (
                "## Representative QC places the pre-refinement and final states in the same columns\n\n"
                "For each field, the original branch occupies the left column and the nucleated-only "
                "branch occupies the right column. The top row is the classification before the "
                "late stage and the bottom row is the final result, enabling direct vertical comparison."
            ),
        },
    ]
    for index, key in enumerate(qc_keys, 1):
        description = (
            "E2 verifies the d0 no-change guardrail."
            if key.startswith("E2_")
            else "The final row shows the objects added by the frozen late-death rule."
        )
        image_block(
            blocks,
            image_data,
            f"qc_{index}",
            f"Figure {index}. {key} pre-refinement and final classification",
            description,
            f"{key}: original and nucleated-only branches before and after late-death refinement.",
        )
    dose_start = len(qc_keys) + 1
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
            },
        )
    )
    image_block(
        blocks,
        image_data,
        "dose_auc",
        f"Figure {dose_start}. AUC-normalized Hill responses",
        "Each replicate is normalized to its matched vehicle within the same treatment background.",
        "AUC-normalized 2N-versus-4N dose responses for both classification branches.",
    )
    blocks.append({"id": "hill_table_block", "type": "table", "tableId": "hill_table"})
    image_block(
        blocks,
        image_data,
        "dose_gr",
        f"Figure {dose_start + 1}. Day-4 and Day-5 growth-rate inhibition",
        "The paired Delta GR panels compare 4N with 2N while preserving replicate pairing.",
        "GR curves and paired ploidy differences for both treatment backgrounds and branches.",
    )
    blocks.append({"id": "gr_table_block", "type": "table", "tableId": "gr_table"})
    image_block(
        blocks,
        image_data,
        "dose_death",
        f"Figure {dose_start + 2}. Day-4 and Day-5 excess lethal fraction",
        "Excess lethal fraction adjusts each treated well to the matched control viability in the same replicate and treatment background.",
        "Excess-lethal-fraction curves and paired ploidy differences for both branches.",
    )
    blocks.extend(
        (
            {"id": "death_table_block", "type": "table", "tableId": "death_table"},
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
    calibration_root = (
        COMMON.require_dir(args.calibration_root)
        if args.calibration_root is not None
        else None
    )
    plate_map_path = COMMON.require_file(args.plate_map)
    validated = validate_inputs(
        root,
        calibration_root,
        args.expected_fields_per_branch,
        args.expected_timepoints,
        args.expected_dose_response_files,
    )
    plate_map = load_plate_map(plate_map_path)
    datasets = aggregate_datasets(root, validated, plate_map)
    figures, qc_keys = report_figures(root, validated)
    if args.debug_figure_dir is not None:
        debug = args.debug_figure_dir.expanduser().resolve()
        debug.mkdir(parents=True, exist_ok=True)
        for key, figure in figures.items():
            figure.save(debug / f"{key}.png")
    image_data, high_resolution = COMMON.encode_figures(figures)
    report_sources = sources(root, calibration_root)
    timestamp = COMMON.generated_at()
    manifest = report_manifest(image_data, qc_keys)
    title = "SUM159 Full-Cohort Dead-Classification Report"
    artifact = COMMON.artifact_payload(
        title=title,
        description=(
            "Technical full-cohort report for object-aware classification, "
            "late-death trajectory refinement, QC, and dose-response analyses."
        ),
        manifest=manifest,
        datasets=datasets,
        sources=report_sources,
        origin="artifact://sum159-dead-classification/full-cohort",
        timestamp=timestamp,
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "analysis" / "reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = COMMON.parse_key_value_file(root / "SUBMISSION_SUMMARY.txt")
    receipt = {
        "report_mode": "full_cohort_classification",
        "classification_root": str(root),
        "calibration_root": str(calibration_root) if calibration_root else None,
        "approved_configuration_source": str(validated["approved_source"]),
        "generated_at": timestamp,
        "project_git_sha": submission.get("project_git_sha", ""),
        "dataset_rows": {key: len(value) for key, value in datasets.items()},
        "embedded_figures": sorted(figures),
        "qc_keys": qc_keys,
        "fields_per_branch": args.expected_fields_per_branch,
        "refinement_rows": len(validated["refinement"]),
        "refinement_failures": 0,
        "dose_response_files": len(validated["dose_files"]),
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
