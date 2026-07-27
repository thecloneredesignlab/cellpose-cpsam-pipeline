#!/usr/bin/env python3
"""Contract tests and validator fixtures for classification report manifests."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
IMAGE_DATA = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    1,
    1,
)


def load_module(filename: str, name: str) -> Any:
    path = REPORT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_manifest(mode: str) -> tuple[str, dict[str, Any]]:
    if mode == "calibration":
        module = load_module(
            "generate_late_dead_d0_d5_calibration_report.py",
            "_calibration_report_fixture",
        )
        image_keys = (
            "trajectory_overview",
            "feature_map",
            "d0_e2",
            "d0_f5",
            "d0_h9",
            "d5_e9_development",
            "d5_e9_holdout",
            "d5_f9_replicate",
        )
        title = "SUM159 d0 + Day-5 Dead-Classification Calibration Report"
        return title, module.report_manifest(
            {},
            {key: IMAGE_DATA for key in image_keys},
        )
    if mode == "full":
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_fixture",
        )
        title = "SUM159 Full-Cohort Dead-Classification Report"
        qc_samples = []
        for selection_index, (day, hours) in enumerate(
            ((0, 0.0), (3, 72.0), (5, 120.0)),
            1,
        ):
            qc_samples.append(
                {
                    "selection_index": selection_index,
                    "ploidy": "2N",
                    "ploidy_order": 1,
                    "condition_id": "no_drug",
                    "condition": "No drug",
                    "condition_short": "No drug",
                    "condition_order": 1,
                    "time_id": f"day_{day}",
                    "time_label": f"Day {day}",
                    "time_order": selection_index,
                    "key": f"C2_1_{day:02d}d00h00m",
                    "well": "C2",
                    "site": 1,
                    "elapsed_hours": hours,
                    "day": day,
                    "doxorubicin_nm": 0.0,
                    "cyclophosphamide": False,
                    "replicate": 1,
                    "total_cells": 100,
                    "dead_fraction": 0.02,
                    "rescued": 0,
                    "uncertain": 0,
                }
            )
        uncertainty_cases = [
            {
                "key": "D10_3_00d00h00m",
                "branch": "original",
                "combined_mask_id": 319,
                "reason": "Dual-view final-call disagreement",
            },
            {
                "key": "G6_2_00d00h00m",
                "branch": "original",
                "combined_mask_id": 212,
                "reason": "Operational evidence conflict",
            },
        ]
        full_images = {
            "classification_workflow": IMAGE_DATA,
            "qc_sample_01": IMAGE_DATA,
            "qc_sample_02": IMAGE_DATA,
            "qc_sample_03": IMAGE_DATA,
            "d0_uncertainty_01": IMAGE_DATA,
            "d0_uncertainty_02": IMAGE_DATA,
            "well_live_dead_counts_over_time": IMAGE_DATA,
        }
        full_images.update(
            {
                panel["key"]: IMAGE_DATA
                for panel in module.dose_panel_records()
            }
        )
        return title, module.report_manifest(
            full_images,
            qc_samples,
            uncertainty_cases,
        )
    raise ValueError(f"Unknown fixture mode: {mode}")


def fixture_datasets(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    fields_by_dataset: dict[str, dict[str, Any]] = {}

    def remember(dataset: str, field: str, spec: dict[str, Any]) -> None:
        fields_by_dataset.setdefault(dataset, {}).setdefault(field, spec)

    for card in manifest.get("cards", []):
        for metric in card.get("metrics", []):
            remember(card["dataset"], metric["field"], metric)
    for chart in manifest.get("charts", []):
        dataset = chart["dataset"]
        encodings = chart["encodings"]
        for name in ("x", "y", "color"):
            encoding = encodings.get(name)
            if not encoding:
                continue
            if "field" in encoding:
                remember(dataset, encoding["field"], encoding)
            for field in encoding.get("fields", []):
                remember(dataset, field, encoding)
        for tooltip in encodings.get("tooltip", []):
            remember(dataset, tooltip["field"], tooltip)
    for table in manifest.get("tables", []):
        for column in table.get("columns", []):
            remember(table["dataset"], column["field"], column)

    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset, fields in fields_by_dataset.items():
        row: dict[str, Any] = {}
        for field, spec in fields.items():
            numeric = (
                spec.get("type") == "quantitative"
                or spec.get("format") in {"number", "percent", "currency"}
            )
            row[field] = 1 if numeric else "Example"
        datasets[dataset] = [row]
    return datasets


def fixture_artifact(mode: str) -> dict[str, Any]:
    title, manifest = fixture_manifest(mode)
    source_ids = sorted(
        {
            item["sourceId"]
            for collection in ("cards", "charts", "tables")
            for item in manifest.get(collection, [])
        }
    )
    timestamp = "2026-07-23T00:00:00Z"
    manifest.update(
        {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Schema-validation fixture.",
            "generatedAt": timestamp,
            "sources": [
                {"id": source_id, "label": source_id, "path": f"fixture/{source_id}.csv"}
                for source_id in source_ids
            ],
        }
    )
    sources = [
        {
            "id": source_id,
            "label": source_id,
            "path": f"fixture/{source_id}.csv",
            "query": {
                "engine": "duckdb",
                "description": f"Read the {source_id} fixture.",
                "sql": f"SELECT * FROM read_csv_auto('fixture/{source_id}.csv', header=true)",
                "tables_used": [f"{source_id}.csv"],
            },
        }
        for source_id in source_ids
    ]
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "fixture",
            "datasets": fixture_datasets(manifest),
        },
        "sources": sources,
        "package_info": {
            "originUrl": f"artifact://classification-report/{mode}-fixture",
            "controls": {"edit": False, "refresh": False, "share": False},
        },
    }


def assert_manifest_contract(mode: str) -> None:
    artifact = fixture_artifact(mode)
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    first_markdown = next(
        block for block in manifest["blocks"] if block["type"] == "markdown"
    )
    assert first_markdown["body"].splitlines()[0] == f"# {manifest['title']}"
    assert any(block["type"] == "chart" for block in manifest["blocks"])
    for chart in manifest["charts"]:
        assert "facet" not in chart["encodings"]
        row = datasets[chart["dataset"]][0]
        assert chart["encodings"]["x"]["field"] in row
        y = chart["encodings"]["y"]
        assert y.get("field") in row or all(field in row for field in y["fields"])
    for table in manifest["tables"]:
        declared = {column["field"] for column in table["columns"]}
        assert table["defaultSort"]["field"] in declared
        assert table["defaultSort"]["direction"] in {"asc", "desc"}


class ClassificationReportManifestTests(unittest.TestCase):
    def test_calibration_report_manifest_contract(self) -> None:
        assert_manifest_contract("calibration")

    def test_full_report_manifest_contract(self) -> None:
        assert_manifest_contract("full")

    def test_full_report_qc_selection_covers_ploidy_condition_time_grid(self) -> None:
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_selection_fixture",
        )
        plate_map: dict[str, dict[str, str]] = {}
        field_states: list[dict[str, str]] = []
        summaries: list[dict[str, str]] = []
        for ploidy_index, ploidy in enumerate(module.QC_PLOIDY_ORDER, 1):
            for condition_index, condition in enumerate(
                module.QC_CONDITION_GROUPS,
                1,
            ):
                dose = float(condition["target_dose_nm"])
                well = f"X{ploidy_index}{condition_index}"
                cyclophosphamide = bool(condition["cyclophosphamide"])
                plate_map[well] = {
                    "well": well,
                    "doxorubicin_nm": str(dose),
                    "ploidy": ploidy,
                    "cyclophosphamide": str(cyclophosphamide).lower(),
                    "replicate": "1",
                }
                for timepoint in module.QC_TIMEPOINTS:
                    day = int(timepoint["day"])
                    key = f"{well}_1_{day:02d}d00h00m"
                    field_states.append(
                        {
                            "branch": "original",
                            "key": key,
                            "well": well,
                            "site": "1",
                            "elapsed_hours": str(timepoint["hours"]),
                            "cyclophosphamide": str(cyclophosphamide).lower(),
                            "doxorubicin_nm": str(dose),
                        }
                    )
                    summaries.append(
                        {
                            "key": key,
                            "total_cell_count": "100",
                            "dead_fraction": "0.25",
                            "late_death_rescue_count": "5",
                            "late_death_uncertain_count": "1",
                        }
                    )
        selected = module.select_qc_samples(
            {
                "field_states": field_states,
                "summaries": {"consensus": summaries},
            },
            plate_map,
        )
        self.assertEqual(
            len(selected),
            len(module.QC_PLOIDY_ORDER)
            * len(module.QC_CONDITION_GROUPS)
            * len(module.QC_TIMEPOINTS),
        )
        self.assertEqual(
            [
                (row["ploidy"], row["condition_id"], row["time_id"])
                for row in selected
            ],
            [
                (ploidy, condition["id"], timepoint["id"])
                for ploidy in module.QC_PLOIDY_ORDER
                for condition in module.QC_CONDITION_GROUPS
                for timepoint in module.QC_TIMEPOINTS
            ],
        )
        for ploidy in module.QC_PLOIDY_ORDER:
            for condition in module.QC_CONDITION_GROUPS:
                series = [
                    row
                    for row in selected
                    if row["ploidy"] == ploidy
                    and row["condition_id"] == condition["id"]
                ]
                self.assertEqual(
                    len({(row["well"], row["site"]) for row in series}),
                    1,
                )

    def test_full_report_qc_treatment_bands_exclude_intermediate_dose(self) -> None:
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_dose_fixture",
        )
        self.assertEqual(
            module.qc_condition_group(12.5, False)["id"],
            "low_doxorubicin",
        )
        self.assertEqual(
            module.qc_condition_group(400.0, True)["id"],
            "high_doxorubicin_cyclophosphamide",
        )
        self.assertIsNone(module.qc_condition_group(50.0, False))

    def test_full_report_qc_visual_contract(self) -> None:
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_visual_fixture",
        )
        self.assertEqual(module.CLASSIFICATION_BOUNDARY_WIDTH, 2)
        self.assertEqual(module.STATE_COLORS["live"].astype(int).tolist(), [35, 205, 95])
        self.assertEqual(module.STATE_COLORS["dead"].astype(int).tolist(), [176, 74, 214])
        self.assertEqual(module.STATE_COLORS["uncertain"].astype(int).tolist(), [255, 214, 10])
        self.assertEqual(module.QC_COMPOSITE_FONT_SCALE, 2)
        self.assertEqual(module.QC_COMPOSITE_TITLE_FONT_SCALE, 1)
        self.assertEqual(module.QC_COMPOSITE_LABEL_FONT_SCALE, 2)
        self.assertEqual(module.QC_COMPOSITE_TITLE_FONT_SIZE, 72)
        self.assertEqual(module.QC_COMPOSITE_LABEL_FONT_SIZE, 136)
        _title, manifest = fixture_manifest("full")
        workflow_intro = next(
            block
            for block in manifest["blocks"]
            if block["id"] == "workflow_intro"
        )
        workflow_text = next(
            block
            for block in manifest["blocks"]
            if block["id"] == "classification_workflow_text"
        )
        workflow_image = next(
            block
            for block in manifest["blocks"]
            if block["id"] == "classification_workflow_image"
        )
        self.assertEqual(workflow_intro["body"].splitlines()[0], "## Workflow")
        self.assertEqual(
            workflow_text["body"].splitlines()[0],
            "### Figure 1. Death-classification workflow",
        )
        self.assertIn("click the image", workflow_image["body"].lower())
        self.assertLess(
            next(
                index
                for index, block in enumerate(manifest["blocks"])
                if block["id"] == "classification_workflow_image"
            ),
            next(
                index
                for index, block in enumerate(manifest["blocks"])
                if block["id"] == "qc_selection_intro"
            ),
        )
        group = next(
            block
            for block in manifest["blocks"]
            if block["id"] == "qc_group_2n_no_drug"
        )
        sample_image = next(
            block
            for block in manifest["blocks"]
            if block["id"] == "qc_sample_01_image"
        )
        self.assertIn("three-row-by-four-column", group["body"])
        self.assertIn("previous classification", group["body"])
        self.assertIn("purple", group["body"])
        self.assertNotIn("nucleated-only classification", group["body"])
        self.assertNotIn("nucleated-only classification", sample_image["body"])

    def test_qc_panel_order_keeps_previous_after_final(self) -> None:
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_panel_order_fixture",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined_raw = np.zeros((8, 8, 3), dtype=np.uint8)
            scalar_raw = np.zeros((8, 8), dtype=np.uint16)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[2:6, 2:6] = 1
            paths: dict[str, Path] = {}
            for name, array in (
                ("combined_raw", combined_raw),
                ("brightfield_raw", scalar_raw),
                ("nuclei_raw", scalar_raw),
                ("dead_raw", scalar_raw),
                ("combined_mask", labels),
                ("brightfield_mask", labels),
                ("nuclei_extent", labels),
                ("nuclei_core", labels),
                ("dead_mask", labels),
            ):
                path = root / f"{name}.tif"
                tifffile.imwrite(path, array)
                paths[name] = path
            prediction_paths: dict[str, Path] = {}
            for stage, state in (("previous", "dead"), ("final", "live")):
                path = root / f"{stage}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=("mask_id", "state"),
                    )
                    writer.writeheader()
                    writer.writerow({"mask_id": 1, "state": state})
                prediction_paths[stage] = path
            record = {
                "profiles": {
                    "Combined": {
                        "raw": str(paths["combined_raw"]),
                        "original_mask": str(paths["combined_mask"]),
                    },
                    "Brightfield": {
                        "raw": str(paths["brightfield_raw"]),
                        "original_mask": str(paths["brightfield_mask"]),
                    },
                    "Nuclei": {
                        "raw": str(paths["nuclei_raw"]),
                        "extent_mask": str(paths["nuclei_extent"]),
                        "core_mask": str(paths["nuclei_core"]),
                    },
                    "Dead": {
                        "raw": str(paths["dead_raw"]),
                        "mask": str(paths["dead_mask"]),
                    },
                }
            }
            comparison = module.classification_comparison_panels(
                record,
                prediction_paths["previous"],
                prediction_paths["final"],
            )
            self.assertEqual(
                [label for _image, label in comparison],
                [
                    "Combined RGB · raw",
                    "Dead · raw",
                    "Previous classification",
                    "Final classification",
                ],
            )
            full = module.authoritative_qc_panels(
                record,
                prediction_paths["final"],
                prediction_paths["previous"],
            )
            self.assertEqual(len(full), 10)
            self.assertEqual(
                [label for _image, label in full[-2:]],
                [
                    "Final classification · authoritative",
                    "Previous classification · 20260721",
                ],
            )

    def test_classification_report_carousel_contracts(self) -> None:
        main_report = load_module(
            "generate_results_analysis_report.py",
            "_classification_report_carousel_fixture",
        )
        full_groups = main_report.report_carousel_groups(fixture_artifact("full"))
        calibration_groups = main_report.report_carousel_groups(
            fixture_artifact("calibration")
        )
        self.assertEqual(
            {group["id"]: len(group["slides"]) for group in full_groups},
            {
                "full_2n_no_drug": 3,
                "full_d0_uncertainty": 2,
                "full_dose_auc": 6,
                "full_dose_gr": 6,
                "full_dose_death": 6,
            },
        )
        self.assertEqual(
            {group["id"]: len(group["slides"]) for group in calibration_groups},
            {
                "calibration_overview": 2,
                "calibration_d0_cases": 3,
                "calibration_d5_cases": 3,
            },
        )
        for group in full_groups + calibration_groups:
            self.assertEqual(
                [slide["index"] for slide in group["slides"]],
                list(range(len(group["slides"]))),
            )
            self.assertTrue(all(slide["textId"] for slide in group["slides"]))
            self.assertTrue(all(slide["imageId"] for slide in group["slides"]))
        full_block_ids = {
            block["id"] for block in fixture_artifact("full")["manifest"]["blocks"]
        }
        self.assertIn("well_live_dead_counts_over_time_image", full_block_ids)
        self.assertNotIn("dose_auc_image", full_block_ids)
        self.assertNotIn("dose_gr_image", full_block_ids)
        self.assertNotIn("dose_death_image", full_block_ids)

    def test_classification_report_carousel_fits_current_viewport(self) -> None:
        main_report = load_module(
            "generate_results_analysis_report.py",
            "_classification_report_viewport_fixture",
        )
        style = main_report.navigation_style([])
        script = main_report.navigation_script([], [], [])
        self.assertEqual(main_report.CAROUSEL_IMAGE_HEIGHT_CAP, 1800)
        self.assertIn("--report-carousel-image-height-cap:1800px", style)
        self.assertIn("max-height:calc(100dvh - 64px)", style)
        self.assertIn("CAROUSEL_IMAGE_HEIGHT_CAP=1800", script)
        self.assertIn("function fitCarouselToViewport", script)
        self.assertIn("function fitStandaloneImageFrame", script)
        self.assertIn("window.visualViewport?.height", script)
        self.assertIn("window.visualViewport?.addEventListener", script)
        self.assertIn("keepCarouselInViewport", script)
        self.assertIn('surface=document.createElement("button")', script)
        self.assertIn('frame.hasAttribute("sandbox")', script)
        self.assertIn("openLightbox(id,label)", script)

    def test_full_report_qc_timepoints_are_exact_and_missing_grid_fails(self) -> None:
        module = load_module(
            "generate_full_classification_report.py",
            "_full_report_time_fixture",
        )
        self.assertEqual(module.qc_timepoint(0.0)["id"], "day_0")
        self.assertEqual(module.qc_timepoint(72.0)["id"], "day_3")
        self.assertEqual(module.qc_timepoint(120.0)["id"], "day_5")
        self.assertIsNone(module.qc_timepoint(71.0))
        with self.assertRaisesRegex(
            RuntimeError,
            "ploidy=2N, condition=No drug, time=Day 3, Day 5",
        ):
            module.select_qc_samples(
                {
                    "field_states": [
                        {
                            "branch": "original",
                            "key": "A1_1_00d00h00m",
                            "well": "A1",
                            "site": "1",
                            "elapsed_hours": "0",
                            "cyclophosphamide": "false",
                            "doxorubicin_nm": "0",
                        }
                    ],
                    "summaries": {
                        "consensus": [
                            {
                                "key": "A1_1_00d00h00m",
                                "total_cell_count": "100",
                                "dead_fraction": "0.1",
                                "late_death_rescue_count": "0",
                                "late_death_uncertain_count": "0",
                            }
                        ]
                    },
                },
                {
                    "A1": {
                        "doxorubicin_nm": "0",
                        "ploidy": "2N",
                        "cyclophosphamide": "false",
                        "replicate": "1",
                    }
                },
            )

    def test_classification_report_high_resolution_is_true_six_x(self) -> None:
        common = load_module(
            "classification_report_common.py",
            "_classification_report_common_fixture",
        )
        canonical = {
            "sample": (
                "data:image/png;base64,AA==",
                100,
                50,
            )
        }
        payload = common.build_scaled_high_resolution_payload(
            {"sample": Image.new("RGB", (120, 60), "white")},
            canonical,
        )
        self.assertEqual(payload["render_scale"], 6)
        self.assertEqual(payload["images"]["sample_image"]["width"], 600)
        self.assertEqual(payload["images"]["sample_image"]["height"], 300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibration", "full"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(fixture_artifact(args.mode), separators=(",", ":"))
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
