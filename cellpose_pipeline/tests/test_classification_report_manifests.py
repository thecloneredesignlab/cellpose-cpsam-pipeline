#!/usr/bin/env python3
"""Contract tests and validator fixtures for classification report manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any


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
        return title, module.report_manifest(
            {
                "qc_1": IMAGE_DATA,
                "dose_auc": IMAGE_DATA,
                "dose_gr": IMAGE_DATA,
                "dose_death": IMAGE_DATA,
            },
            ["E2_1_00d00h00m"],
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
