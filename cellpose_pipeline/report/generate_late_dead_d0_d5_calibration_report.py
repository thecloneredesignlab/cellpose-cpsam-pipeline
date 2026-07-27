#!/usr/bin/env python3
"""Build the self-contained SUM159 d0+d5 late-death calibration report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def _load_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_path(
    "_late_dead_calibration_common",
    Path(__file__).with_name("classification_report_common.py"),
)
FULL_REPORT = _load_path(
    "_late_dead_calibration_full_report",
    Path(__file__).with_name("generate_full_classification_report.py"),
)
CALIBRATION_QC_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "Parameter_calibration"
    / "29_render_late_dead_rescue_qc.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <calibration-root>/report_final).",
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def calibration_layout(root: Path) -> dict[str, Path]:
    optimization = (
        root / "optimization"
        if (root / "optimization").is_dir()
        else root / "optimization_v3"
    )
    report = root / "report" if (root / "report").is_dir() else root / "report_v3"
    run_summary = (
        root / "RUN_SUMMARY.txt"
        if (root / "RUN_SUMMARY.txt").is_file()
        else root / "RUN_SUMMARY_V3.txt"
    )
    success = root / "_SUCCESS" if (root / "_SUCCESS").is_file() else root / "_SUCCESS_V3"
    return {
        "optimization": optimization,
        "report": report,
        "run_summary": run_summary,
        "success": success,
    }


def d0_datasets(d0_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = COMMON.read_csv(d0_root / "annotations" / "final_annotation_summary.csv")
    states: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for row in summary:
        level = row.get("metric_level", "")
        if level == "cell_state":
            states.append(
                {
                    "branch": COMMON.display_branch(row["analysis_branch"]),
                    "state": row["classification"],
                    "count": COMMON.as_int(row["count"]),
                    "denominator": COMMON.as_int(row["denominator"]),
                    "share": COMMON.as_float(row["percentage"]),
                }
            )
        elif level == "confirmed_death_object_relation":
            relations.append(
                {
                    "branch": COMMON.display_branch(row["analysis_branch"]),
                    "relation": row["subclassification"].replace("_", " "),
                    "count": COMMON.as_int(row["count"]),
                    "denominator": COMMON.as_int(row["denominator"]),
                    "share": COMMON.as_float(row["percentage"]),
                }
            )
    if not states or not relations:
        raise RuntimeError("The frozen d0 annotation summary is incomplete")
    return states, relations


def calibration_datasets(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    layout = calibration_layout(root)
    optimization = layout["optimization"]
    report = layout["report"]
    dataset_summary = COMMON.read_json(root / "feature_cache" / "dataset_summary.json")
    anchor = COMMON.read_json(optimization / "anchor_metrics.json")
    best = COMMON.read_json(optimization / "best_configuration.json")
    go_no_go = COMMON.read_json(
        optimization / "FULL_CLASSIFICATION_GO_NO_GO.json"
    )
    run_summary = COMMON.parse_key_value_file(layout["run_summary"])
    field_trials_raw = COMMON.read_csv(
        optimization / "field_parameter_trials.csv"
    )
    object_trials_raw = COMMON.read_csv(
        optimization / "object_parameter_trials.csv"
    )
    field_summary = COMMON.read_csv(
        optimization / "late_death_field_summary.csv"
    )
    qc_inventory = COMMON.read_csv(report / "qc_inventory.csv")

    d0_root = COMMON.require_dir(Path(dataset_summary["d0_audit_root"]))
    d0_states, d0_relations = d0_datasets(d0_root)

    overview = [
        {
            "branches": len(dataset_summary["branches"]),
            "d0_fields": COMMON.as_int(dataset_summary["d0_fields"]),
            "trajectory_fields": COMMON.as_int(dataset_summary["trajectory_fields"]),
            "trajectory_wells": len(dataset_summary["trajectory_wells"]),
            "completed_shards": COMMON.as_int(dataset_summary["completed_shards"]),
            "failed_shards": COMMON.as_int(dataset_summary["failed_shards"]),
            "selection_seed": COMMON.as_int(dataset_summary["selection_seed"]),
            "changed_d0_objects": COMMON.as_int(anchor["changed_d0_object_count"]),
            "passed_validation_gates": sum(
                bool(values["pass"]) for values in go_no_go["gates"].values()
            ),
        }
    ]
    convergence_gates = [
        {
            "gate": name.replace("_", " ").title(),
            "status": "PASS" if values["pass"] else "FAIL",
        }
        for name, values in go_no_go["gates"].items()
    ]

    anchor_rates = [
        {
            "metric": "Blue-supported dead preservation",
            "rate": COMMON.as_float(anchor["blue_supported_anchor_preservation"]),
            "denominator": 0,
        },
        {
            "metric": "Temporal-remnant recall",
            "rate": COMMON.as_float(anchor["temporal_remnant_anchor_recall"]),
            "denominator": COMMON.as_int(anchor["temporal_remnant_anchor_count"]),
        },
        {
            "metric": "Untreated-live false-positive proxy",
            "rate": COMMON.as_float(anchor["untreated_live_anchor_counterfactual_fpr"]),
            "denominator": COMMON.as_int(
                dataset_summary["proxy_counts"]["untreated_live_anchor"]
            ),
        },
    ]

    field_trials = [
        {
            "trial": row["trial"],
            "untreated_activation": COMMON.as_float(
                row["untreated_late_field_activation_rate"]
            ),
            "treated_activation": COMMON.as_float(
                row["treated_non_e9_field_activation_rate"]
            ),
            "development_pass": "Passed"
            if COMMON.truthy(row["passes_development"])
            else "Rejected",
        }
        for row in field_trials_raw
    ]
    object_trials = [
        {
            "trial": row["trial"],
            "development_e9_dead_fraction": COMMON.as_float(
                row["development_e9_dead_fraction"]
            ),
            "untreated_live_fpr": COMMON.as_float(
                row["untreated_live_anchor_counterfactual_fpr"]
            ),
            "new_calls_outside_e9_f9": COMMON.as_int(
                row["new_calls_outside_e9_f9"]
            ),
            "perturbation_stability": COMMON.as_float(
                row.get("object_evidence_stability_rate", 0.0)
            ),
            "development_pass": "Passed"
            if COMMON.truthy(row["passes_development"])
            else "Rejected",
            "production_candidate": "Passed"
            if COMMON.truthy(
                row.get("passes_production_candidate", False)
            )
            else "Rejected",
            "configuration": row["configuration"],
        }
        for row in object_trials_raw
    ]

    d5_groups = (
        ("E9 development", "development_anchor"),
        ("E9 holdout", "holdout_anchor"),
        ("F9 replicate", "replicate_diagnostic"),
    )
    d5_comparison: list[dict[str, Any]] = []
    d5_details: list[dict[str, Any]] = []
    for label, flag in d5_groups:
        selected = [
            row
            for row in field_summary
            if COMMON.truthy(row[flag])
            and abs(COMMON.as_float(row["elapsed_hours"]) - 120.0) < 1e-6
        ]
        if not selected:
            raise RuntimeError(f"No Day-5 rows found for {label}")
        baseline = [COMMON.as_float(row["baseline_dead_fraction"]) for row in selected]
        final = [COMMON.as_float(row["final_dead_fraction"]) for row in selected]
        d5_comparison.extend(
            (
                {
                    "cohort": label,
                    "stage": "Before late-death rescue",
                    "dead_fraction": median(baseline),
                    "fields": len(selected),
                },
                {
                    "cohort": label,
                    "stage": "Final calibrated method",
                    "dead_fraction": median(final),
                    "fields": len(selected),
                },
            )
        )
        d5_details.append(
            {
                "cohort": label,
                "fields": len(selected),
                "baseline_median": median(baseline),
                "final_minimum": min(final),
                "final_median": median(final),
                "final_maximum": max(final),
            }
        )

    parameters: list[dict[str, Any]] = []
    for layer, values in (
        ("Field collapse", best["field_configuration"]),
        ("Object evidence", best["object_configuration"]),
    ):
        for parameter, value in values.items():
            parameters.append(
                {
                    "layer": layer,
                    "parameter": parameter.replace("_", " "),
                    "value": value,
                }
            )
    parameters.append(
        {
            "layer": "Time gate",
            "parameter": "minimum elapsed hours",
            "value": best["late_min_hours"],
        }
    )

    qc_counts = Counter(
        (COMMON.display_branch(row["branch"]), row["cohort"]) for row in qc_inventory
    )
    qc_summary = [
        {"branch": branch, "cohort": cohort, "qc_fields": count}
        for (branch, cohort), count in sorted(qc_counts.items())
    ]

    datasets = {
        "overview": overview,
        "d0_states": d0_states,
        "d0_relations": d0_relations,
        "anchor_rates": anchor_rates,
        "field_trials": field_trials,
        "object_trials": object_trials,
        "d5_comparison": d5_comparison,
        "d5_details": d5_details,
        "selected_parameters": parameters,
        "qc_summary": qc_summary,
        "convergence_gates": convergence_gates,
    }
    metadata = {
        "dataset_summary": dataset_summary,
        "anchor_metrics": anchor,
        "best_configuration": best,
        "run_summary": run_summary,
        "d0_root": d0_root,
        "go_no_go": go_no_go,
        "layout": layout,
    }
    return datasets, metadata


def report_figures(root: Path, d0_root: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Calibration QC rendering requires pandas in the CellPose environment"
        ) from error
    calibration_qc = _load_path(
        "_late_dead_calibration_qc_renderer",
        CALIBRATION_QC_PATH,
    )

    report = calibration_layout(root)["report"]
    figures: dict[str, Any] = {
        "trajectory_overview": COMMON.read_image(
            report / "charts" / "e9_f9_timecourses.png"
        ),
        "feature_map": COMMON.read_image(
            report / "charts" / "e9_day5_feature_map.png"
        ),
    }
    d0_cases = (
        ("d0_e2", "E2_1_00d00h00m"),
        ("d0_f5", "F5_1_00d00h00m"),
        ("d0_h9", "H9_4_00d00h00m"),
    )
    for figure_key, key in d0_cases:
        well = key.split("_", 1)[0]
        record_path = d0_root / "field_manifest" / "records" / well / f"{key}.json"
        final_prediction_matches = sorted(
            (d0_root / "classification_original" / "predictions").glob(
                f"*{key}*_per_cell_predictions.csv"
            )
        )
        previous_prediction_matches = sorted(
            (
                d0_root
                / "historical_reference"
                / "strong_direct_all_d0"
                / "predictions"
            ).glob(f"*{key}*_per_cell_predictions.csv")
        )
        if len(final_prediction_matches) != 1:
            raise RuntimeError(
                f"Expected one authoritative d0 prediction table for {key}, "
                f"found {len(final_prediction_matches)}"
            )
        if len(previous_prediction_matches) != 1:
            raise RuntimeError(
                f"Expected one frozen previous d0 prediction table for {key}, "
                f"found {len(previous_prediction_matches)}"
            )
        panels = FULL_REPORT.classification_comparison_panels(
            COMMON.read_json(record_path),
            previous_prediction_matches[0],
            final_prediction_matches[0],
        )
        if len(panels) != 4:
            raise RuntimeError(f"Expected four d0 panels for {key}, found {len(panels)}")
        figures[figure_key] = COMMON.labeled_grid(
            panels,
            title=f"{key} · frozen d0 authoritative QC",
            columns=2,
            panel_width=1400,
            title_font_size=FULL_REPORT.QC_COMPOSITE_TITLE_FONT_SIZE,
            label_font_size=FULL_REPORT.QC_COMPOSITE_LABEL_FONT_SIZE,
        )

    qc_cases = (
        (
            "d5_e9_development",
            "E9 development: site 1 at Day 5",
            "E9_1_05d00h00m",
        ),
        (
            "d5_e9_holdout",
            "E9 holdout: site 2 at Day 5",
            "E9_2_05d00h00m",
        ),
        (
            "d5_f9_replicate",
            "F9 replicate diagnostic: site 1 at Day 5",
            "F9_1_05d00h00m",
        ),
    )
    target_keys = {key for _figure_key, _label, key in qc_cases}
    usecols = [
        "branch",
        "cohort",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "combined_raw_path",
        "dead_raw_path",
        "cell_mask_path",
        "combined_mask_id",
        "final_state",
        "countable",
        "border_touching",
        "current_dead_call",
        "late_death_rescue_call",
        "final_dead_call",
        "late_death_uncertain",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "field_global_late_death",
    ]
    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        calibration_layout(root)["optimization"] / "late_death_predictions.csv.gz",
        usecols=usecols,
        chunksize=200_000,
    ):
        selected = chunk.loc[
            chunk["branch"].astype(str).eq("original")
            & chunk["key"].astype(str).isin(target_keys)
        ].copy()
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        raise RuntimeError("No Day-5 calibration rows matched the report QC keys")
    selected_predictions = pd.concat(selected_chunks, ignore_index=True)
    for column in (
        "countable",
        "border_touching",
        "current_dead_call",
        "late_death_rescue_call",
        "final_dead_call",
        "late_death_uncertain",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "field_global_late_death",
    ):
        selected_predictions[column] = calibration_qc.bool_series(
            selected_predictions[column]
        )
    found_keys = set(selected_predictions["key"].astype(str))
    if found_keys != target_keys:
        raise RuntimeError(
            f"Day-5 report QC key mismatch: expected={sorted(target_keys)} "
            f"found={sorted(found_keys)}"
        )
    for figure_key, label, key in qc_cases:
        rows = selected_predictions.loc[
            selected_predictions["key"].astype(str).eq(key)
        ].copy()
        raw_panels, _statistics = calibration_qc.render_field_panel_images(
            rows,
            include_nuclei=False,
            include_evidence=False,
        )
        panels = [
            (Image.fromarray(image, mode="RGB"), title)
            for image, title in raw_panels
        ]
        if len(panels) != 4:
            raise RuntimeError(
                f"Expected four Day-5 panels for {key}, found {len(panels)}"
            )
        figures[figure_key] = COMMON.labeled_grid(
            panels,
            title=label,
            columns=2,
            panel_width=1400,
            title_font_size=FULL_REPORT.QC_COMPOSITE_TITLE_FONT_SIZE,
            label_font_size=FULL_REPORT.QC_COMPOSITE_LABEL_FONT_SIZE,
        )
    return figures


def report_sources(root: Path, d0_root: Path) -> list[dict[str, Any]]:
    calibration = root.name
    d0 = d0_root.name
    return [
        COMMON.logical_source(
            "dataset",
            "Calibration dataset definition",
            calibration,
            "feature_cache/dataset_summary.json",
            "Read the frozen d0 and trajectory cohort contract.",
        ),
        COMMON.logical_source(
            "field_trials",
            "Field-collapse parameter trials",
            calibration,
            "optimization/field_parameter_trials.csv",
            "Compare field-collapse candidates and untreated activation.",
        ),
        COMMON.logical_source(
            "object_trials",
            "Object-evidence parameter trials",
            calibration,
            "optimization/object_parameter_trials.csv",
            "Compare object-level rescue candidates and control false-positive proxies.",
        ),
        COMMON.logical_source(
            "configuration",
            "Approved late-death configuration",
            calibration,
            "optimization/best_configuration.json",
            "Read the approved decision layers and frozen parameters.",
        ),
        COMMON.logical_source(
            "field_summary",
            "Late-death field summary",
            calibration,
            "optimization/late_death_field_summary.csv",
            "Compare baseline and final field-level dead fractions.",
        ),
        COMMON.logical_source(
            "qc",
            "Calibration QC inventory",
            calibration,
            "report/qc_inventory.csv",
            "Validate saved d0, E9, and F9 QC coverage.",
        ),
        COMMON.logical_source(
            "convergence",
            "Operational calibration GO or NO-GO receipt",
            calibration,
            "optimization/FULL_CLASSIFICATION_GO_NO_GO.json",
            "Read d0 safety, positive-anchor, branch, temporal, and perturbation gates.",
        ),
        COMMON.logical_source(
            "d0_annotations",
            "Frozen d0 multilevel annotations",
            d0,
            "annotations/final_annotation_summary.csv",
            "Read frozen d0 cell states and Death-object relations.",
        ),
        COMMON.logical_source(
            "d0_previous",
            "Frozen previous d0 strong-direct classification",
            d0,
            (
                "historical_reference/strong_direct_all_d0/predictions/"
                "*_per_cell_predictions.csv"
            ),
            (
                "Render the previous d0 classification before the final "
                "nucleus-aware dual-layer method."
            ),
        ),
    ]


def image_block(
    blocks: list[dict[str, Any]],
    image_data: dict[str, tuple[str, int, int]],
    key: str,
    heading: str,
    text: str,
    caption: str,
    carousel_group: str | None = None,
    carousel_index: int | None = None,
    carousel_label: str | None = None,
) -> None:
    if (carousel_group is None) != (carousel_index is None):
        raise ValueError("Carousel group and index must be provided together")
    heading_level = "####" if carousel_group is not None else "###"
    blocks.append(
        {
            "id": f"{key}_text",
            "type": "markdown",
            "body": f"{heading_level} {heading}\n\n{text}",
        }
    )
    blocks.append(
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
        }
    )


def report_manifest(
    datasets: dict[str, list[dict[str, Any]]],
    image_data: dict[str, tuple[str, int, int]],
) -> dict[str, Any]:
    charts = [
        {
            "id": "d0_state_chart",
            "title": "Frozen d0 cell-state composition",
            "subtitle": "All 320 zero-time-point fields in both analysis branches.",
            "type": "bar",
            "dataset": "d0_states",
            "sourceId": "d0_annotations",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "state", "type": "nominal", "label": "Cell state"},
                "y": {
                    "field": "share",
                    "type": "quantitative",
                    "label": "Share",
                    "format": "percent",
                },
                "color": {
                    "field": "branch",
                    "type": "nominal",
                    "label": "Analysis branch",
                },
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
            "id": "d0_relation_chart",
            "title": "Frozen d0 confirmed Death-object relations",
            "subtitle": "Relations are mutually exclusive within each analysis branch.",
            "type": "bar",
            "dataset": "d0_relations",
            "sourceId": "d0_annotations",
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
                    "label": "Object relation",
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
        {
            "id": "d5_comparison_chart",
            "title": "Day-5 dead fractions before and after late-death rescue",
            "subtitle": "Median across the saved development, holdout, and replicate fields.",
            "type": "bar",
            "dataset": "d5_comparison",
            "sourceId": "field_summary",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "cohort", "type": "nominal", "label": "Calibration cohort"},
                "y": {
                    "field": "dead_fraction",
                    "type": "quantitative",
                    "label": "Dead fraction",
                    "format": "percent",
                },
                "color": {"field": "stage", "type": "nominal", "label": "Classification stage"},
                "tooltip": [
                    {
                        "field": "fields",
                        "type": "quantitative",
                        "label": "Branch-fields",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "anchor_chart",
            "title": "Operational anchor rates",
            "subtitle": "Proxy checks without manual biological ground truth.",
            "type": "bar",
            "dataset": "anchor_rates",
            "sourceId": "configuration",
            "valueFormat": "percent",
            "options": {"orientation": "horizontal", "grouping": "grouped"},
            "encodings": {
                "x": {
                    "field": "metric",
                    "type": "nominal",
                    "label": "Operational anchor",
                },
                "y": {
                    "field": "rate",
                    "type": "quantitative",
                    "label": "Rate",
                    "format": "percent",
                },
                "tooltip": [
                    {
                        "field": "denominator",
                        "type": "quantitative",
                        "label": "Reference objects",
                        "format": "number",
                    }
                ],
            },
        },
        {
            "id": "field_trial_chart",
            "title": "Field-collapse parameter trial outcomes",
            "subtitle": "Each point is one tested field configuration.",
            "type": "scatter",
            "dataset": "field_trials",
            "sourceId": "field_trials",
            "encodings": {
                "x": {
                    "field": "untreated_activation",
                    "type": "quantitative",
                    "label": "Untreated late-field activation",
                    "format": "percent",
                },
                "y": {
                    "field": "treated_activation",
                    "type": "quantitative",
                    "label": "Treated non-E9 activation",
                    "format": "percent",
                },
                "color": {
                    "field": "development_pass",
                    "type": "nominal",
                    "label": "Development gate",
                },
                "tooltip": [
                    {"field": "trial", "type": "nominal", "label": "Trial"}
                ],
            },
        },
    ]
    tables = [
        {
            "id": "convergence_table",
            "title": "Operational calibration convergence gates",
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
            "id": "d5_detail_table",
            "title": "Day-5 calibration cohort statistics",
            "subtitle": "Exact minimum and median final dead fractions by saved cohort.",
            "dataset": "d5_details",
            "sourceId": "field_summary",
            "defaultSort": {"field": "cohort", "direction": "asc"},
            "columns": [
                {"field": "cohort", "label": "Cohort", "type": "text"},
                {"field": "fields", "label": "Branch-fields", "format": "number"},
                {
                    "field": "baseline_median",
                    "label": "Baseline median",
                    "format": "percent",
                },
                {
                    "field": "final_minimum",
                    "label": "Final minimum",
                    "format": "percent",
                },
                {
                    "field": "final_median",
                    "label": "Final median",
                    "format": "percent",
                },
                {
                    "field": "final_maximum",
                    "label": "Final maximum",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "parameter_table",
            "title": "Approved production parameters",
            "subtitle": "Frozen field, object, and time-gate configuration.",
            "dataset": "selected_parameters",
            "sourceId": "configuration",
            "defaultSort": {"field": "layer", "direction": "asc"},
            "columns": [
                {"field": "layer", "label": "Decision layer", "type": "text"},
                {"field": "parameter", "label": "Parameter", "type": "text"},
                {"field": "value", "label": "Value", "type": "text"},
            ],
        },
        {
            "id": "object_trial_table",
            "title": "Object-level candidate configurations",
            "subtitle": "Development response, untreated-live proxy, perturbation stability, and off-sentinel calls.",
            "dataset": "object_trials",
            "sourceId": "object_trials",
            "defaultSort": {
                "field": "new_calls_outside_e9_f9",
                "direction": "asc",
            },
            "columns": [
                {"field": "trial", "label": "Trial", "type": "text"},
                {
                    "field": "development_e9_dead_fraction",
                    "label": "E9 Day-5 dead fraction",
                    "format": "percent",
                },
                {
                    "field": "untreated_live_fpr",
                    "label": "Untreated-live FP proxy",
                    "format": "percent",
                },
                {
                    "field": "new_calls_outside_e9_f9",
                    "label": "Calls outside E9/F9",
                    "format": "number",
                },
                {
                    "field": "perturbation_stability",
                    "label": "±0.05 stability",
                    "format": "percent",
                },
                {
                    "field": "development_pass",
                    "label": "Development gate",
                    "type": "text",
                },
                {
                    "field": "production_candidate",
                    "label": "All ranking gates",
                    "type": "text",
                },
            ],
        },
        {
            "id": "qc_table",
            "title": "Saved calibration QC coverage",
            "subtitle": "Complete QC inventory by branch and calibration cohort.",
            "dataset": "qc_summary",
            "sourceId": "qc",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "cohort", "label": "Cohort", "type": "text"},
                {"field": "qc_fields", "label": "QC fields", "format": "number"},
            ],
        },
    ]
    cards = [
        {
            "id": "scope_card",
            "description": "Frozen d0 guardrail and selected trajectories.",
            "dataset": "overview",
            "sourceId": "dataset",
            "metrics": [
                {"label": "d0 fields", "field": "d0_fields", "format": "number"},
                {
                    "label": "Trajectory fields",
                    "field": "trajectory_fields",
                    "format": "number",
                },
            ],
        },
        {
            "id": "shard_card",
            "description": "Feature-cache completeness before optimization.",
            "dataset": "overview",
            "sourceId": "dataset",
            "metrics": [
                {
                    "label": "Completed shards",
                    "field": "completed_shards",
                    "format": "number",
                },
                {
                    "label": "Failed shards",
                    "field": "failed_shards",
                    "format": "number",
                },
            ],
        },
        {
            "id": "guardrail_card",
            "description": "The late-stage method is prohibited from changing d0.",
            "dataset": "overview",
            "sourceId": "configuration",
            "metrics": [
                {
                    "label": "Changed d0 objects",
                    "field": "changed_d0_objects",
                    "format": "number",
                },
                {
                    "label": "Validation gates passed",
                    "field": "passed_validation_gates",
                    "format": "number",
                },
            ],
        },
    ]

    blocks: list[dict[str, Any]] = [
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "# SUM159 d0 + Day-5 Dead-Classification Calibration Report\n\n"
                "The calibration combines a frozen d0 classifier with a late-time consensus rescue. "
                "The first calibration stage separated Combined-cell state from independently "
                "segmented Death objects, preventing nearby weak blue signal from automatically "
                "rewriting a visually live cell. The second stage addressed a different failure: "
                "after drug exposure, dead cells can remain in place while blue and red signal "
                "decay and the cell shrinks. The revised stage uses continuous density- and "
                "time-matched references, cell-conditioned Dead signal, two frozen segmentation "
                "views, multi-frame spatial continuity, and a recoverable field-collapse state. "
                "Strong live evidence vetoes an initiating rescue; a high-confidence rescue is "
                "then propagated only across a mutual-nearest dual-view pair. Final pair "
                "disagreements become uncertainty, and the late-stage model is explicitly "
                "forbidden from changing the frozen d0 result."
            ),
        },
        {
            "id": "cards",
            "type": "metric-strip",
            "cardIds": ["scope_card", "shard_card", "guardrail_card"],
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## The test design separates development, holdout, replication, and safety\n\n"
                "The saved calibration dataset contains 320 frozen d0 fields and complete E9/F9 "
                "trajectories in both analysis branches. E9 site 1 is the development anchor, "
                "E9 sites 2–4 are holdouts, and F9 sites 1–4 are an independent replicate "
                "diagnostic. Ten wells provide broader treated and untreated trajectory context. "
                "These are operational anchors rather than manually annotated biological labels."
            ),
        },
        {
            "id": "convergence",
            "type": "markdown",
            "body": (
                "## Convergence is defined by predeclared operational proxy gates\n\n"
                "The optimizer must preserve d0, retain supported-death anchors, maintain "
                "high-confidence multi-frame temporal support, limit matched-pair dual-view "
                "disagreement, and remain stable "
                "under small feature-percentile perturbations. No gate is interpreted as a "
                "biological sensitivity or specificity estimate."
            ),
        },
        {
            "id": "convergence_table_block",
            "type": "table",
            "tableId": "convergence_table",
        },
        {
            "id": "d0_method",
            "type": "markdown",
            "body": (
                "## d0 classification keeps cell state and Death-object evidence separate\n\n"
                "At d0, the cell label belongs to the Combined mask while a Death-object label "
                "belongs to an independently segmented Dead-channel mask. Nucleus multiplicity, "
                "cell-scale overlap, RGB context, and object confidence distinguish same-cell "
                "death from live-with-death-signal, multi-nucleus overlap, dead-only regions, "
                "and supplemental objects. This prevents unconditional spatial association from "
                "creating live-cell false positives."
            ),
        },
        {
            "id": "d0_state_text",
            "type": "markdown",
            "body": (
                "### Frozen d0 state and object composition\n\n"
                "The following charts report the complete frozen d0 cohort used as the safety "
                "constraint for the later optimization."
            ),
        },
        {"id": "d0_state_block", "type": "chart", "chartId": "d0_state_chart"},
        {"id": "d0_relation_block", "type": "chart", "chartId": "d0_relation_chart"},
        {
            "id": "late_failure",
            "type": "markdown",
            "body": (
                "## Blue-signal decay caused a separate late-death false-negative mode\n\n"
                "The d0 classifier was designed around early untreated conditions. In later "
                "drug-treated images, a dead cell can remain attached and shrunken after the "
                "blue death signal fades. Red mass and cytoplasm also decline, so blue intensity "
                "alone is no longer a sufficient death criterion. The calibration therefore "
                "uses trajectory changes and cell morphology without rerunning segmentation."
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## Field collapse gates object-level multi-signal rescue\n\n"
                "A field is first screened for persistent, multi-site collapse in object count, "
                "mask area, cytoplasm, red mass, and mask coverage relative to continuously "
                "density- and time-matched references. The field state can recover after sustained "
                "normalization. Only inside a dual-view accepted late treated field can an object "
                "be rescued. The object decision combines nucleus-to-cytoplasm ratios, area and "
                "cytoplasm depletion, red-mass loss, shape, cell-conditioned Dead signal, "
                "multi-frame temporal evidence, and explicit healthy signals. Existing confirmed "
                "deaths are preserved and d0 is never modified."
            ),
        },
        {
            "id": "field_trial_text",
            "type": "markdown",
            "body": (
                "## Parameter trials balance treated-field recovery against untreated activation\n\n"
                "Each point represents a tested field-collapse configuration. The selected model "
                "must activate the E9 development field while limiting activation in untreated "
                "trajectories and retaining broad treated-field applicability."
            ),
        },
        {"id": "field_trial_block", "type": "chart", "chartId": "field_trial_chart"},
        {
            "id": "object_trial_text",
            "type": "markdown",
            "body": (
                "## Object-level tuning rejects unnecessary calls outside the sentinel wells\n\n"
                "Among configurations that recover E9, the selected object thresholds must also "
                "pass the ±0.05 feature-percentile stability gate and the untreated-live "
                "counterfactual false-positive guardrail before unnecessary calls outside "
                "E9/F9 are minimized."
            ),
        },
        {"id": "object_trial_block", "type": "table", "tableId": "object_trial_table"},
        {
            "id": "parameter_text",
            "type": "markdown",
            "body": (
                "## The approved configuration is frozen for production\n\n"
                "The parameter table records the exact field, object, and elapsed-time gates "
                "promoted into the production classification path."
            ),
        },
        {"id": "parameter_block", "type": "table", "tableId": "parameter_table"},
        {
            "id": "d5_result",
            "type": "markdown",
            "body": (
                "## Day-5 recovery generalizes from E9 development to holdout and F9 replication\n\n"
                "Dead fractions rise sharply in the development field and remain high in the "
                "held-out E9 sites and the F9 replicate diagnostic. These fractions are "
                "calibration anchors, not manual sensitivity estimates."
            ),
        },
        {"id": "d5_comparison_block", "type": "chart", "chartId": "d5_comparison_chart"},
        {"id": "d5_detail_block", "type": "table", "tableId": "d5_detail_table"},
        {
            "id": "anchor_result",
            "type": "markdown",
            "body": (
                "## Safety anchors preserve supported death while protecting untreated live cells\n\n"
                "Blue-supported deaths and high-confidence, multi-frame tracked temporal "
                "remnants are retained, while the "
                "counterfactual untreated-live proxy remains unchanged. A displayed value of "
                "100% or 0% is an operational test outcome, not biological sensitivity or "
                "specificity."
            ),
        },
        {"id": "anchor_block", "type": "chart", "chartId": "anchor_chart"},
    ]

    blocks.extend(
        (
            {
                "id": "visual_qc",
                "type": "markdown",
                "body": (
                    "## Visual QC and calibration examples\n\n"
                    "Use the arrows, progress dots, keyboard, or horizontal swipe to compare "
                    "related figures without separating each figure title from its image."
                ),
            },
            {
                "id": "calibration_overview_carousel",
                "type": "markdown",
                "body": "### Calibration overview\n\nThe two overview figures summarize the trajectory problem and the multi-signal solution.",
            },
        )
    )
    image_block(
        blocks,
        image_data,
        "trajectory_overview",
        "Figure 1. E9 and F9 trajectories define the late-death problem",
        "The saved time courses show where the original blue-centered classifier diverges from the late-stage phenotype.",
        "E9 and F9 time-course evidence used during the calibration.",
        carousel_group="calibration_overview",
        carousel_index=0,
        carousel_label="Trajectory overview",
    )
    image_block(
        blocks,
        image_data,
        "feature_map",
        "Figure 2. Day-5 objects require multiple independent signals",
        "The feature map shows why red loss, cytoplasm depletion, nuclear ratios, shape, and temporal evidence are combined.",
        "Day-5 feature map for the late-death object decision.",
        carousel_group="calibration_overview",
        carousel_index=1,
        carousel_label="Multi-signal feature map",
    )
    blocks.append(
        {
            "id": "d0_qc_carousel",
            "type": "markdown",
            "body": (
                "### Frozen d0 QC cases\n\n"
                "Each case uses the same four-panel sequence as the Day-5 QC: "
                "Combined raw, Dead raw, previous classification, and final classification."
            ),
        }
    )
    image_block(
        blocks,
        image_data,
        "d0_e2",
        "Figure 3. E2 demonstrates the frozen d0 live-cell protection",
        "The previous and final overlays show how the d0 object-aware result protects a live Combined cell from nearby Death-object evidence.",
        "E2 d0 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d0_cases",
        carousel_index=0,
        carousel_label="E2 live-cell protection",
    )
    image_block(
        blocks,
        image_data,
        "d0_f5",
        "Figure 4. F5 demonstrates multi-nucleus live/death overlap",
        "The previous and final overlays show how the nucleus-aware method avoids forcing overlapping biological units into one cell-level label.",
        "F5 d0 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d0_cases",
        carousel_index=1,
        carousel_label="F5 multi-nucleus overlap",
    )
    image_block(
        blocks,
        image_data,
        "d0_h9",
        "Figure 5. H9 retains a one-nucleus live cell with death signal",
        "The previous and final overlays show that the one-nucleus Combined cell remains live rather than inheriting the overlapping Death signal.",
        "H9 d0 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d0_cases",
        carousel_index=2,
        carousel_label="H9 live with Death signal",
    )
    blocks.append(
        {
            "id": "d5_qc_carousel",
            "type": "markdown",
            "body": (
                "### Day-5 QC cases\n\n"
                "Development, holdout, and replicate examples use the same four-panel "
                "sequence: Combined raw, Dead raw, previous classification, and final classification."
            ),
        }
    )
    image_block(
        blocks,
        image_data,
        "d5_e9_development",
        "Figure 6. E9 site 1 is the Day-5 development anchor",
        "The final rescue recovers the visually collapsed field relative to the previous classification.",
        "E9 site 1 Day-5 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d5_cases",
        carousel_index=0,
        carousel_label="E9 development",
    )
    image_block(
        blocks,
        image_data,
        "d5_e9_holdout",
        "Figure 7. E9 site 2 confirms the method on held-out data",
        "The same frozen configuration is applied to a site that was not used as the development anchor.",
        "E9 site 2 Day-5 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d5_cases",
        carousel_index=1,
        carousel_label="E9 holdout",
    )
    image_block(
        blocks,
        image_data,
        "d5_f9_replicate",
        "Figure 8. F9 provides an independent Day-5 replicate diagnostic",
        "Recovery in F9 tests whether the selected configuration extends beyond the E9 sentinel.",
        "F9 site 1 Day-5 raw channels and previous-versus-final classification.",
        carousel_group="calibration_d5_cases",
        carousel_index=2,
        carousel_label="F9 replicate",
    )
    blocks.extend(
        (
            {
                "id": "qc_inventory_text",
                "type": "markdown",
                "body": (
                    "## Complete QC remains available outside the embedded examples\n\n"
                    "The report embeds representative, deterministic examples so the HTML remains "
                    "portable. The full saved QC inventory covers the frozen d0 cohort and every "
                    "selected E9/F9 trajectory field used by the calibration."
                ),
            },
            {"id": "qc_inventory_block", "type": "table", "tableId": "qc_table"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## The calibration establishes operational consistency, not biological accuracy\n\n"
                    "No manual object-level ground-truth table exists. Development fractions, "
                    "holdout fractions, anchor preservation, temporal-remnant recall, and the "
                    "untreated-live false-positive proxy are all defined by existing segmentation, "
                    "trajectory, and classification evidence. They justify the frozen production "
                    "rule operationally but cannot be interpreted as true biological sensitivity "
                    "or specificity."
                ),
            },
            {
                "id": "recommended_use",
                "type": "markdown",
                "body": (
                    "## Recommended production use\n\n"
                    "Use the approved configuration only after the d0 object-aware classification "
                    "and only for fields at or beyond the frozen late-time gate. Preserve the d0 "
                    "no-change invariant, the untreated-live proxy, the saved configuration hash, "
                    "and complete QC inventories as regression checks for every future production run."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Manual annotation of a small blinded late-time set would be required to "
                    "translate these operational anchors into biological sensitivity and "
                    "specificity. Until then, downstream interpretation should retain the "
                    "uncertainty annotations and compare both analysis branches."
                ),
            },
        )
    )
    return {"cards": cards, "charts": charts, "tables": tables, "blocks": blocks}


def main() -> int:
    args = parse_args()
    root = COMMON.require_dir(args.calibration_root)
    layout = calibration_layout(root)
    for required in (
        layout["success"],
        layout["run_summary"],
        root / "feature_cache" / "dataset_summary.json",
        layout["optimization"] / "best_configuration.json",
        layout["optimization"] / "anchor_metrics.json",
        layout["optimization"] / "FULL_CLASSIFICATION_GO_NO_GO.json",
        layout["optimization"] / "field_parameter_trials.csv",
        layout["optimization"] / "object_parameter_trials.csv",
        layout["optimization"] / "late_death_field_summary.csv",
        layout["report"] / "qc_inventory.csv",
    ):
        COMMON.require_file(required)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "report_final"
    )
    plugin_root = (
        args.plugin_root.expanduser().resolve()
        if args.plugin_root is not None
        else COMMON.discover_plugin_root()
    )
    datasets, metadata = calibration_datasets(root)
    figures = report_figures(root, metadata["d0_root"])
    if args.debug_figure_dir is not None:
        debug = args.debug_figure_dir.expanduser().resolve()
        debug.mkdir(parents=True, exist_ok=True)
        for key, figure in figures.items():
            figure.save(debug / f"{key}.png")
    image_data, high_resolution = COMMON.encode_figures(figures)
    sources = report_sources(root, metadata["d0_root"])
    timestamp = COMMON.generated_at()
    manifest = report_manifest(datasets, image_data)
    title = "SUM159 d0 + Day-5 Dead-Classification Calibration Report"
    artifact = COMMON.artifact_payload(
        title=title,
        description=(
            "Technical report for the frozen d0 object-aware classifier and the "
            "Day-5 late-death trajectory refinement calibrated before full-cohort production."
        ),
        manifest=manifest,
        datasets=datasets,
        sources=sources,
        origin="artifact://sum159-dead-classification/d0-d5-calibration",
        timestamp=timestamp,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "report_mode": "d0_d5_calibration",
        "calibration_root": str(root),
        "d0_audit_root": str(metadata["d0_root"]),
        "generated_at": timestamp,
        "dataset_rows": {key: len(value) for key, value in datasets.items()},
        "embedded_figures": sorted(figures),
        "qc_grid_columns": 2,
        "qc_grid_rows": 2,
        "d0_qc_panels_per_case": 4,
        "d5_qc_panels_per_case": 4,
        "qc_composite_font_scale": FULL_REPORT.QC_COMPOSITE_FONT_SCALE,
        "qc_composite_title_font_scale": (
            FULL_REPORT.QC_COMPOSITE_TITLE_FONT_SCALE
        ),
        "qc_composite_label_font_scale": (
            FULL_REPORT.QC_COMPOSITE_LABEL_FONT_SCALE
        ),
        "late_death_evidence_panel_embedded": False,
        "classification_boundary_width": (
            FULL_REPORT.CLASSIFICATION_BOUNDARY_WIDTH
        ),
        "classification_colors": {
            state: [int(value) for value in color]
            for state, color in FULL_REPORT.STATE_COLORS.items()
        },
        "selection_seed": metadata["dataset_summary"]["selection_seed"],
        "completed_shards": metadata["dataset_summary"]["completed_shards"],
        "failed_shards": metadata["dataset_summary"]["failed_shards"],
        "changed_d0_object_count": metadata["anchor_metrics"][
            "changed_d0_object_count"
        ],
        "operational_decision": metadata["go_no_go"]["decision"],
        "biological_accuracy_claimed": False,
        "production_integration": metadata["best_configuration"][
            "production_integration"
        ],
    }
    result = COMMON.package_report(
        artifact=artifact,
        artifact_path=output_dir
        / "DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.artifact.json",
        output_html=output_dir / "DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.html",
        build_receipt=output_dir
        / "DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.build.json",
        plugin_root=plugin_root,
        force=args.force,
        high_resolution_images=high_resolution,
        receipt=receipt,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
