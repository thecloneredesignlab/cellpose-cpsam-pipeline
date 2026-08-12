#!/usr/bin/env python3
"""Validate and publish broad-phenotype predictions as an isolated shadow axis.

This stage deliberately does not edit the existing viability or trajectory
outputs.  It verifies the immutable classifier generation, requires exact
cell/probability coverage, records any legacy viability NO_GO receipt as
informational provenance, and writes a shadow-only promotion decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCHEMA_VERSION = "broad_phenotype_shadow_finalize_v1"
PREDICTION_COLUMNS = (
    "model_id",
    "cell_id",
    "predicted_class_id",
    "prediction_status",
)
PROBABILITY_COLUMNS = (
    "model_id",
    "cell_id",
    "class_id",
    "probability",
    "prediction_status",
)
SAFE_YAML_SCALAR = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<value>.*?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--legacy-no-go", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_absolute_existing(path: Path, label: str, kind: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve(strict=True)
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def yaml_top_level_scalars(path: Path) -> dict[str, str]:
    """Read the simple top-level path scalars emitted by the CPA adapter.

    The authoritative R validator still parses the complete project.  This
    deliberately small reader avoids adding a Python YAML runtime dependency.
    It rejects YAML aliases and structured values for the fields it consumes.
    """

    text = path.read_text()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        if not isinstance(decoded, dict):
            raise ValueError(f"Project must contain a mapping: {path}")
        values = {
            key: str(decoded.get(key, "")).strip()
            for key in ("classes_file", "cells_file", "project_id")
        }
        missing = sorted(key for key, value in values.items() if not value)
        if missing:
            raise ValueError(f"Project is missing path/identity fields: {missing}")
        return values

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        match = SAFE_YAML_SCALAR.match(raw_line)
        if match is None:
            continue
        key = match.group("key")
        value = match.group("value").strip()
        if key not in {"classes_file", "cells_file", "project_id"}:
            continue
        if not value or value.startswith(("&", "*", "!", "[", "{")):
            raise ValueError(f"Unsupported {key} YAML scalar at {path}:{line_number}")
        if value[0:1] in {'"', "'"}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ValueError(f"Malformed quoted {key} at {path}:{line_number}")
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    missing = sorted({"classes_file", "cells_file", "project_id"} - set(values))
    if missing:
        raise ValueError(f"Project is missing simple top-level path fields: {missing}")
    return values


def resolve_project_asset(project: Path, value: str, label: str, shadow_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project.parent / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    if not is_within(resolved, shadow_root):
        raise ValueError(f"{label} escapes shadow root: {resolved}")
    return resolved


def open_tsv(path: Path, label: str) -> tuple[Any, csv.DictReader]:
    stream = path.open("r", newline="")
    reader = csv.DictReader(stream, delimiter="\t")
    if reader.fieldnames is None:
        stream.close()
        raise ValueError(f"{label} is empty: {path}")
    return stream, reader


def require_columns(reader: csv.DictReader, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def iter_cell_ids(cells_path: Path) -> Iterator[str]:
    stream, reader = open_tsv(cells_path, "cells table")
    try:
        require_columns(reader, ("cell_id",), "cells table")
        for row_number, row in enumerate(reader, 2):
            cell_id = (row.get("cell_id") or "").strip()
            if not cell_id:
                raise ValueError(f"Blank cell_id in cells table at row {row_number}")
            yield cell_id
    finally:
        stream.close()


def read_trainable_classes(classes_path: Path) -> list[str]:
    stream, reader = open_tsv(classes_path, "classes table")
    classes: list[tuple[int, str]] = []
    try:
        require_columns(reader, ("class_id", "trainable", "order"), "classes table")
        for row_number, row in enumerate(reader, 2):
            trainable = (row.get("trainable") or "").strip().lower()
            if trainable not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"Invalid classes.trainable at row {row_number}")
            if trainable in {"false", "0", "no"}:
                continue
            class_id = (row.get("class_id") or "").strip()
            try:
                order = int((row.get("order") or "").strip())
            except ValueError as error:
                raise ValueError(f"Invalid classes.order at row {row_number}") from error
            classes.append((order, class_id))
    finally:
        stream.close()
    classes.sort()
    class_ids = [item[1] for item in classes]
    if len(class_ids) < 2 or len(set(class_ids)) != len(class_ids) or any(not item for item in class_ids):
        raise ValueError("Classes table must define at least two unique trainable class IDs")
    return class_ids


def validate_manifest_hashes(
    manifest: dict[str, Any], artifacts: dict[str, Path], label: str
) -> dict[str, str]:
    recorded = manifest.get("artifact_file_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(f"{label} lacks artifact_file_sha256")
    observed: dict[str, str] = {}
    for role, path in artifacts.items():
        expected = recorded.get(role)
        if not isinstance(expected, str):
            raise ValueError(f"{label} lacks hash for {role}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch for {role}: expected={expected}, observed={actual}"
            )
        observed[role] = actual
    return observed


def count_cells(cells_path: Path) -> int:
    count = sum(1 for _ in iter_cell_ids(cells_path))
    if count < 1:
        raise ValueError("cells table contains no cells")
    return count


def validate_predictions(
    cells_path: Path,
    predictions_path: Path,
    output_temporary: Path,
    expected_class_ids: set[str],
    expected_model_id: str,
    status_path: Path,
    cell_count: int,
) -> tuple[int, int, set[str]]:
    status = np.memmap(status_path, dtype=np.uint8, mode="w+", shape=(cell_count,))
    stream, reader = open_tsv(predictions_path, "classifier predictions")
    output_temporary.parent.mkdir(parents=True, exist_ok=True)
    predicted_classes: set[str] = set()
    ok_count = 0
    unavailable_count = 0
    try:
        require_columns(reader, PREDICTION_COLUMNS, "classifier predictions")
        with output_temporary.open("w", newline="") as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ("model_id", "cell_id", "broad_phenotype_class_id", "prediction_status")
            )
            rows = iter(reader)
            for index, cell_id in enumerate(iter_cell_ids(cells_path)):
                try:
                    row = next(rows)
                except StopIteration as error:
                    raise ValueError(
                        f"Predictions ended early: expected {cell_count}, observed {index}"
                    ) from error
                observed_cell_id = (row.get("cell_id") or "").strip()
                if observed_cell_id != cell_id:
                    raise ValueError(
                        "Prediction cell order/identity mismatch at zero-based index "
                        f"{index}: expected={cell_id}, observed={observed_cell_id}"
                    )
                model_id = (row.get("model_id") or "").strip()
                if model_id != expected_model_id:
                    raise ValueError(f"Prediction model_id mismatch at cell {cell_id}")
                prediction_status = (row.get("prediction_status") or "").strip()
                predicted_class = (row.get("predicted_class_id") or "").strip()
                if prediction_status == "ok":
                    if predicted_class not in expected_class_ids:
                        raise ValueError(
                            f"Prediction has unknown trainable class at cell {cell_id}: {predicted_class}"
                        )
                    status[index] = 1
                    ok_count += 1
                    predicted_classes.add(predicted_class)
                elif prediction_status == "unavailable_missing_features":
                    if predicted_class:
                        raise ValueError(
                            f"Unavailable prediction has a class assignment at cell {cell_id}"
                        )
                    status[index] = 0
                    unavailable_count += 1
                else:
                    raise ValueError(
                        f"Unsupported prediction_status at cell {cell_id}: {prediction_status}"
                    )
                writer.writerow((model_id, cell_id, predicted_class, prediction_status))
            try:
                extra = next(rows)
            except StopIteration:
                extra = None
            if extra is not None:
                raise ValueError(f"Predictions contain extra cell_id: {extra.get('cell_id', '')}")
    finally:
        stream.close()
        status.flush()
        del status
    return ok_count, unavailable_count, predicted_classes


def parse_probability(raw: str, available: bool, cell_id: str, class_id: str) -> float:
    text = raw.strip()
    if not available:
        if text and text.lower() not in {"na", "nan"}:
            raise ValueError(
                f"Unavailable cell has a probability for {class_id}: {cell_id}"
            )
        return 0.0
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"Invalid probability for {class_id}: {cell_id}") from error
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"Probability outside [0,1] for {class_id}: {cell_id}")
    return value


def validate_probabilities(
    cells_path: Path,
    probabilities_path: Path,
    expected_classes: list[str],
    expected_model_id: str,
    status_path: Path,
    cell_count: int,
    sums_path: Path,
) -> tuple[int, float]:
    status = np.memmap(status_path, dtype=np.uint8, mode="r", shape=(cell_count,))
    sums = np.memmap(sums_path, dtype=np.float64, mode="w+", shape=(cell_count,))
    sums[:] = 0.0
    stream, reader = open_tsv(probabilities_path, "classifier probabilities")
    seen_classes: list[str] = []
    total_rows = 0
    try:
        require_columns(reader, PROBABILITY_COLUMNS, "classifier probabilities")
        rows = iter(reader)
        pending: dict[str, str] | None = None
        while True:
            if pending is None:
                try:
                    first = next(rows)
                except StopIteration:
                    break
            else:
                first = pending
                pending = None
            class_id = (first.get("class_id") or "").strip()
            if class_id not in expected_classes:
                raise ValueError(f"Probability table contains unknown class: {class_id}")
            if class_id in seen_classes:
                raise ValueError(f"Probability class is not one contiguous block: {class_id}")
            seen_classes.append(class_id)
            cell_iterator = iter_cell_ids(cells_path)
            row = first
            for index, expected_cell_id in enumerate(cell_iterator):
                if index > 0:
                    try:
                        row = next(rows)
                    except StopIteration as error:
                        raise ValueError(
                            f"Probability block ended early for class {class_id}: observed={index}"
                        ) from error
                observed_class = (row.get("class_id") or "").strip()
                if observed_class != class_id:
                    raise ValueError(
                        f"Probability block for {class_id} ended early at row {index}"
                    )
                cell_id = (row.get("cell_id") or "").strip()
                if cell_id != expected_cell_id:
                    raise ValueError(
                        f"Probability cell mismatch for {class_id} at index {index}: "
                        f"expected={expected_cell_id}, observed={cell_id}"
                    )
                if (row.get("model_id") or "").strip() != expected_model_id:
                    raise ValueError(f"Probability model_id mismatch at cell {cell_id}")
                available = bool(status[index])
                expected_status = "ok" if available else "unavailable_missing_features"
                if (row.get("prediction_status") or "").strip() != expected_status:
                    raise ValueError(f"Probability status mismatch at cell {cell_id}")
                sums[index] += parse_probability(
                    row.get("probability") or "", available, cell_id, class_id
                )
                total_rows += 1
            try:
                pending = next(rows)
            except StopIteration:
                pending = None
                break
        if set(seen_classes) != set(expected_classes) or len(seen_classes) != len(expected_classes):
            raise ValueError(
                "Probability class coverage mismatch: "
                f"expected={expected_classes}, observed={seen_classes}"
            )
        available = status.astype(bool)
        maximum_error = (
            float(np.max(np.abs(sums[available] - 1.0))) if np.any(available) else 0.0
        )
        if maximum_error > 1e-8:
            raise ValueError(f"Probability rows do not sum to one; max_error={maximum_error}")
        if np.any(sums[~available] != 0.0):
            raise ValueError("Unavailable cells contain numeric probabilities")
        return total_rows, maximum_error
    finally:
        stream.close()
        sums.flush()
        del sums
        del status


def legacy_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "ignored": True, "enforcement": "informational_only"}
    resolved = require_absolute_existing(path, "--legacy-no-go", "file")
    payload = json.loads(resolved.read_text())
    decision = None
    for key in ("decision", "go_no_go", "status"):
        if key in payload:
            decision = payload[key]
            break
    return {
        "provided": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "recorded_decision": decision,
        "ignored": True,
        "enforcement": "informational_only",
        "ignore_reason": "orthogonal_broad_phenotype_shadow_no_viability_override",
    }


def main() -> int:
    args = parse_args()
    if not args.shadow_root.is_absolute():
        raise ValueError(f"--shadow-root must be absolute: {args.shadow_root}")
    shadow_root = args.shadow_root.resolve(strict=True)
    if not shadow_root.is_dir():
        raise ValueError(f"--shadow-root is not a directory: {shadow_root}")
    project = require_absolute_existing(args.project, "--project", "file")
    prediction_dir = require_absolute_existing(
        args.prediction_dir, "--prediction-dir", "directory"
    )
    model_dir = (
        require_absolute_existing(args.model_dir, "--model-dir", "directory")
        if args.model_dir is not None
        else prediction_dir.parent.parent.resolve(strict=True)
    )
    for path, label in (
        (project, "project"),
        (prediction_dir, "prediction directory"),
        (model_dir, "model directory"),
    ):
        if not is_within(path, shadow_root):
            raise ValueError(f"{label} escapes shadow root: {path}")

    project_values = yaml_top_level_scalars(project)
    cells_path = resolve_project_asset(
        project, project_values["cells_file"], "cells_file", shadow_root
    )
    classes_path = resolve_project_asset(
        project, project_values["classes_file"], "classes_file", shadow_root
    )
    expected_classes = read_trainable_classes(classes_path)
    cell_count = count_cells(cells_path)

    predictions_path = prediction_dir / "all_cell_predictions.tsv"
    probabilities_path = prediction_dir / "all_cell_probabilities.tsv"
    prediction_manifest_path = prediction_dir / "prediction_manifest.json"
    model_path = model_dir / "model.rds"
    model_manifest_path = model_dir / "model_manifest.json"
    for path in (
        predictions_path,
        probabilities_path,
        prediction_manifest_path,
        model_path,
        model_manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"Required immutable classifier artifact is missing: {path}")

    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    model_manifest = json.loads(model_manifest_path.read_text())
    expected_model_id = str(model_manifest.get("model_id", ""))
    if not expected_model_id or str(prediction_manifest.get("model_id", "")) != expected_model_id:
        raise ValueError("Prediction and model manifests have different model_id values")
    expected_project_id = project_values["project_id"]
    if str(model_manifest.get("project_id", "")) != expected_project_id or str(
        prediction_manifest.get("project_id", "")
    ) != expected_project_id:
        raise ValueError("Project and classifier manifests have different project_id values")
    manifest_classes = [str(value) for value in model_manifest.get("class_ids", [])]
    if set(manifest_classes) != set(expected_classes) or len(manifest_classes) != len(expected_classes):
        raise ValueError(
            f"Model class universe mismatch: expected={expected_classes}, observed={manifest_classes}"
        )
    model_sha256 = sha256_file(model_path)
    model_manifest_sha256 = sha256_file(model_manifest_path)
    model_artifact_hashes = model_manifest.get("artifact_file_sha256")
    if (
        not isinstance(model_artifact_hashes, dict)
        or model_artifact_hashes.get("model") != model_sha256
    ):
        raise ValueError("Model manifest hash for model.rds does not match")
    if prediction_manifest.get("model_sha256") != model_sha256:
        raise ValueError("Prediction manifest model_sha256 does not match model.rds")
    if prediction_manifest.get("model_manifest_sha256") != model_manifest_sha256:
        raise ValueError("Prediction manifest model_manifest_sha256 does not match")
    artifact_hashes = validate_manifest_hashes(
        prediction_manifest,
        {
            "all_cell_predictions": predictions_path,
            "all_cell_probabilities": probabilities_path,
        },
        "prediction manifest",
    )

    output_dir = shadow_root / "predictions"
    output_path = output_dir / "broad_phenotype_predictions.tsv"
    receipt_path = shadow_root / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json"
    probability_receipt_path = output_dir / "broad_phenotype_probabilities.json"
    existing = [path for path in (output_path, receipt_path, probability_receipt_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Shadow finalization outputs already exist; use --force to verify and replace: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy = legacy_provenance(args.legacy_no_go)
    with tempfile.TemporaryDirectory(prefix=".broad-finalize-", dir=output_dir) as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_predictions = temporary_root / output_path.name
        status_path = temporary_root / "prediction_status.u8"
        sums_path = temporary_root / "probability_sums.f64"
        ok_count, unavailable_count, observed_predictions = validate_predictions(
            cells_path,
            predictions_path,
            temporary_predictions,
            set(expected_classes),
            expected_model_id,
            status_path,
            cell_count,
        )
        probability_row_count, maximum_probability_sum_error = validate_probabilities(
            cells_path,
            probabilities_path,
            expected_classes,
            expected_model_id,
            status_path,
            cell_count,
            sums_path,
        )
        standardized_sha256 = sha256_file(temporary_predictions)
        os.replace(temporary_predictions, output_path)

    probability_receipt = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(probabilities_path),
        "source_sha256": artifact_hashes["all_cell_probabilities"],
        "row_count": probability_row_count,
        "class_ids": expected_classes,
        "cell_count": cell_count,
        "maximum_probability_sum_error": maximum_probability_sum_error,
        "note": "Canonical long-form probabilities remain in the immutable CPA prediction generation.",
    }
    write_json_atomic(probability_receipt_path, probability_receipt)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shadow_root": str(shadow_root),
        "project": {"path": str(project), "sha256": sha256_file(project)},
        "cells": {
            "path": str(cells_path),
            "sha256": sha256_file(cells_path),
            "row_count": cell_count,
        },
        "classes": {
            "path": str(classes_path),
            "sha256": sha256_file(classes_path),
            "trainable_class_ids": expected_classes,
        },
        "model": {
            "model_id": expected_model_id,
            "model_dir": str(model_dir),
            "model_sha256": model_sha256,
            "model_manifest_sha256": model_manifest_sha256,
        },
        "prediction_generation": {
            "prediction_dir": str(prediction_dir),
            "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
            "all_cell_predictions_sha256": artifact_hashes["all_cell_predictions"],
            "all_cell_probabilities_sha256": artifact_hashes["all_cell_probabilities"],
            "cell_count": cell_count,
            "ok_count": ok_count,
            "unavailable_count": unavailable_count,
            "probability_row_count": probability_row_count,
            "maximum_probability_sum_error": maximum_probability_sum_error,
            "observed_predicted_classes": sorted(observed_predictions),
        },
        "published_shadow_axis": {
            "path": str(output_path),
            "sha256": standardized_sha256,
            "semantic_axis": "broad_phenotype",
            "overwrites_viability_state": False,
            "overwrites_trajectory_state": False,
        },
        "legacy_viability_gate": legacy,
        "technical_decision": "GO",
        "promotion_decision": "NO_GO",
        "overall_decision": "SHADOW_ONLY",
        "promotion_reason": (
            "Independent biological validation and an explicit promotion receipt are required; "
            "this technical gate never replaces RGB viability thresholds."
        ),
    }
    write_json_atomic(receipt_path, receipt)
    print("broad_phenotype_shadow_technical_decision=GO")
    print("broad_phenotype_shadow_promotion_decision=NO_GO")
    print(f"cell_count={cell_count}")
    print(f"prediction_output={output_path}")
    print(f"receipt={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
