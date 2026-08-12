#!/usr/bin/env python3
"""Merge audited shard predictions into the isolated broad-phenotype axis.

The merge is deliberately streaming: it never materializes the full cell or
probability table.  Exact feature and prediction receipts are checked before
each shard is accepted, and the frozen ``cpa/cells.tsv`` order is used as the
global row-universe contract.  A legacy viability NO_GO is provenance only and
cannot block or promote this orthogonal shadow axis.
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
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO


SCHEMA_VERSION = "broad_phenotype_shard_merge_v1"
FEATURE_RECEIPT_SCHEMA = "broad_phenotype_feature_receipt_v1"
PREDICTION_RECEIPT_SCHEMA = "broad_phenotype_shard_prediction_v1"
FEATURE_MANIFEST_COLUMNS = ("key", "feature_path", "receipt_path")
STANDARD_COLUMNS = (
    "model_id",
    "cell_id",
    "broad_phenotype_class_id",
    "prediction_status",
)
CELL_ID_RE = re.compile(r"^(?P<branch>[^|]+)\|(?P<key>[^|]+)\|(?P<label>[1-9][0-9]*)$")
CLASS_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--model-acceptance-receipt", type=Path, required=True)
    parser.add_argument("--model-acceptance-sha256", required=True)
    parser.add_argument(
        "--cells",
        type=Path,
        help="Frozen CPA cells table (default: <shadow-root>/cpa/cells.tsv).",
    )
    parser.add_argument("--legacy-no-go", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def require_absolute_dir(path: Path, label: str, *, create: bool = False) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def require_inside(path: Path, root: Path, label: str) -> Path:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside shadow root {root}: {path}") from error
    return path.resolve()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def open_tsv(path: Path, label: str) -> tuple[TextIO, csv.DictReader]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"{label} is empty: {path}")
    if any(not name for name in reader.fieldnames) or len(reader.fieldnames) != len(
        set(reader.fieldnames)
    ):
        handle.close()
        raise ValueError(f"{label} has blank or duplicate columns: {path}")
    return handle, reader


def exact_columns(reader: csv.DictReader, expected: Sequence[str], label: str) -> None:
    observed = tuple(reader.fieldnames or ())
    if observed != tuple(expected):
        raise ValueError(
            f"{label} columns changed: expected={list(expected)} observed={list(observed)}"
        )


def require_columns(reader: csv.DictReader, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in (reader.fieldnames or ())]
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def cell_id_hasher() -> tuple[hashlib._Hash, list[bool]]:  # type: ignore[name-defined]
    return hashlib.sha256(), [True]


def update_cell_id_hash(state: tuple[Any, list[bool]], cell_id: str) -> None:
    digest, first = state
    if not first[0]:
        digest.update(b"\n")
    digest.update(cell_id.encode("utf-8"))
    first[0] = False


def load_feature_manifest(path: Path) -> list[dict[str, str]]:
    handle, reader = open_tsv(path, "feature manifest")
    rows: list[dict[str, str]] = []
    try:
        exact_columns(reader, FEATURE_MANIFEST_COLUMNS, "feature manifest")
        for row_number, row in enumerate(reader, 2):
            cleaned = {name: (row.get(name) or "").strip() for name in FEATURE_MANIFEST_COLUMNS}
            if any(not value for value in cleaned.values()):
                raise ValueError(f"Blank feature manifest value at row {row_number}")
            rows.append(cleaned)
    finally:
        handle.close()
    if not rows:
        raise ValueError("Feature manifest contains no shards")
    keys = [row["key"] for row in rows]
    feature_paths = [row["feature_path"] for row in rows]
    receipt_paths = [row["receipt_path"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Feature manifest contains duplicate keys")
    if len(set(feature_paths)) != len(feature_paths) or len(set(receipt_paths)) != len(
        receipt_paths
    ):
        raise ValueError("Feature manifest reuses a feature or receipt path")
    return rows


def resolve_manifest_file(value: str, label: str) -> Path:
    path = Path(value)
    return require_absolute_file(path, label)


def prediction_paths(root: Path, feature_receipt: dict[str, Any]) -> tuple[Path, Path]:
    for field in ("key", "well", "branch"):
        if not isinstance(feature_receipt.get(field), str) or not feature_receipt[field]:
            raise ValueError(f"Feature receipt lacks {field}")
    stem = f"{feature_receipt['key']}__{feature_receipt['branch']}"
    prediction = root / "shards" / feature_receipt["well"] / (
        stem + "_broad_phenotype_predictions.tsv"
    )
    receipt = root / "receipts" / feature_receipt["well"] / (stem + ".json")
    return prediction, receipt


def validate_feature_receipt(
    key: str, feature_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], str, str]:
    receipt = read_json(receipt_path, "feature receipt")
    if receipt.get("schema_version") != FEATURE_RECEIPT_SCHEMA or receipt.get("status") != "COMPLETE":
        raise ValueError(f"Feature receipt is not COMPLETE for key={key}")
    if receipt.get("key") != key:
        raise ValueError(f"Feature manifest/receipt key mismatch for key={key}")
    recorded_path = Path(str(receipt.get("feature_tsv", "")))
    if not recorded_path.is_absolute() or recorded_path.resolve(strict=True) != feature_path:
        raise ValueError(f"Feature receipt points to another shard for key={key}")
    feature_sha = sha256_file(feature_path)
    receipt_sha = sha256_file(receipt_path)
    if receipt.get("feature_tsv_sha256") != feature_sha:
        raise ValueError(f"Feature shard hash mismatch for key={key}")
    if not isinstance(receipt.get("feature_columns"), list):
        raise ValueError(f"Feature receipt lacks feature_columns for key={key}")
    return receipt, feature_sha, receipt_sha


def validate_prediction_receipt(
    prediction_path: Path,
    receipt_path: Path,
    feature_receipt: dict[str, Any],
    feature_sha: str,
    feature_receipt_sha: str,
    model_acceptance_receipt: Path,
    model_acceptance_sha256: str,
) -> tuple[dict[str, Any], str]:
    prediction_path = require_absolute_file(prediction_path, "prediction shard")
    receipt_path = require_absolute_file(receipt_path, "prediction receipt")
    receipt = read_json(receipt_path, "prediction receipt")
    if (
        receipt.get("schema_version") != PREDICTION_RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE"
    ):
        raise ValueError(f"Prediction receipt is not COMPLETE: {receipt_path}")
    for field in ("key", "well", "branch"):
        if receipt.get(field) != feature_receipt.get(field):
            raise ValueError(f"Prediction/feature receipt {field} mismatch: {receipt_path}")
    if receipt.get("feature_tsv_sha256") != feature_sha:
        raise ValueError(f"Prediction receipt feature hash mismatch: {receipt_path}")
    if receipt.get("feature_receipt_sha256") != feature_receipt_sha:
        raise ValueError(f"Prediction receipt feature-receipt hash mismatch: {receipt_path}")
    if receipt.get("model_acceptance_receipt") != str(model_acceptance_receipt):
        raise ValueError(f"Prediction receipt model-acceptance path mismatch: {receipt_path}")
    if receipt.get("model_acceptance_sha256") != model_acceptance_sha256:
        raise ValueError(f"Prediction receipt model-acceptance SHA mismatch: {receipt_path}")
    recorded_path = Path(str(receipt.get("prediction_tsv", "")))
    if not recorded_path.is_absolute() or recorded_path.resolve(strict=True) != prediction_path:
        raise ValueError(f"Prediction receipt points to another prediction shard: {receipt_path}")
    prediction_sha = sha256_file(prediction_path)
    if receipt.get("prediction_tsv_sha256") != prediction_sha:
        raise ValueError(f"Prediction shard hash mismatch: {prediction_path}")
    class_ids = receipt.get("class_ids")
    output_columns = receipt.get("output_columns")
    if not isinstance(class_ids, list) or len(class_ids) < 2 or len(class_ids) != len(
        set(class_ids)
    ) or any(not isinstance(value, str) or not CLASS_ID_RE.fullmatch(value) for value in class_ids):
        raise ValueError(f"Prediction receipt has invalid class_ids: {receipt_path}")
    if not isinstance(receipt.get("model_id"), str) or not receipt["model_id"]:
        raise ValueError(f"Prediction receipt has invalid model_id: {receipt_path}")
    for field in (
        "model_sha256",
        "model_manifest_sha256",
        "model_acceptance_sha256",
        "dependency_lock_sha256",
        "implementation_sha256",
    ):
        if not isinstance(receipt.get(field), str) or not SHA256_RE.fullmatch(receipt[field]):
            raise ValueError(f"Prediction receipt has invalid {field}: {receipt_path}")
    for field in ("source_identity", "runtime_identity"):
        if not isinstance(receipt.get(field), dict) or not receipt[field]:
            raise ValueError(f"Prediction receipt has invalid {field}: {receipt_path}")
    expected_columns = [
        "model_id",
        "cell_id",
        "predicted_class_id",
        "prediction_status",
        *[f"probability__{class_id}" for class_id in class_ids],
    ]
    if output_columns != expected_columns:
        raise ValueError(f"Prediction receipt wide schema changed: {receipt_path}")
    return receipt, prediction_sha


def parse_probability(value: str, *, allow_blank: bool, context: str) -> float | None:
    if value == "":
        if allow_blank:
            return None
        raise ValueError(f"Blank probability for available prediction: {context}")
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"Nonnumeric probability: {context}") from error
    if not math.isfinite(numeric) or numeric < -1e-12 or numeric > 1 + 1e-12:
        raise ValueError(f"Invalid probability {numeric}: {context}")
    return numeric


def next_cell_id(cells_reader: csv.DictReader, row_index: int) -> str:
    row = next(cells_reader, None)
    if row is None:
        raise ValueError(f"Shard predictions exceed frozen cells table at merged row {row_index}")
    cell_id = (row.get("cell_id") or "").strip()
    if not cell_id:
        raise ValueError(f"Blank cell_id in frozen cells table at merged row {row_index}")
    return cell_id


def legacy_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "role": "informational_only", "ignored": True}
    resolved = require_absolute_file(path, "--legacy-no-go")
    payload = read_json(resolved, "legacy viability gate")
    return {
        "provided": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "recorded_decision": payload.get("overall_decision", payload.get("status")),
        "role": "informational_only",
        "ignored": True,
        "ignore_reason": "orthogonal_broad_phenotype_shadow_no_viability_override",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    feature_manifest = require_absolute_file(args.feature_manifest, "--feature-manifest")
    prediction_root = require_absolute_dir(args.prediction_root, "--prediction-root")
    shadow_root = require_absolute_dir(args.shadow_root, "--shadow-root", create=True)
    model_acceptance_receipt = require_absolute_file(
        args.model_acceptance_receipt, "--model-acceptance-receipt"
    )
    require_inside(feature_manifest, shadow_root, "--feature-manifest")
    require_inside(prediction_root, shadow_root, "--prediction-root")
    require_inside(model_acceptance_receipt, shadow_root, "--model-acceptance-receipt")
    if not SHA256_RE.fullmatch(args.model_acceptance_sha256):
        raise ValueError("--model-acceptance-sha256 must be one lowercase SHA-256 digest")
    observed_acceptance_sha256 = sha256_file(model_acceptance_receipt)
    if observed_acceptance_sha256 != args.model_acceptance_sha256:
        raise ValueError(
            "Model-acceptance receipt SHA mismatch: "
            f"expected={args.model_acceptance_sha256} observed={observed_acceptance_sha256}"
        )
    cells_path = require_absolute_file(
        args.cells if args.cells is not None else shadow_root / "cpa" / "cells.tsv",
        "--cells",
    )
    feature_manifest_sha = sha256_file(feature_manifest)
    cells_sha = sha256_file(cells_path)
    rows = load_feature_manifest(feature_manifest)

    output_dir = shadow_root / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "broad_phenotype_predictions.tsv"
    receipt_path = shadow_root / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json"
    output_exists = output_path.exists()
    receipt_exists = receipt_path.exists()
    if output_exists != receipt_exists:
        raise RuntimeError("Incomplete existing shard-merge generation was preserved")
    reuse_requested = output_exists and not args.force

    legacy = legacy_provenance(args.legacy_no_go)
    aggregate = hashlib.sha256()
    total_rows = 0
    ok_count = 0
    unavailable_count = 0
    maximum_probability_sum_error = 0.0
    class_counts: Counter[str] = Counter()
    common_identity: dict[str, Any] | None = None
    observed_branches: set[str] = set()

    with tempfile.TemporaryDirectory(prefix=".broad-shard-merge-", dir=output_dir) as temporary:
        temporary_output = Path(temporary) / output_path.name
        cells_handle, cells_reader = open_tsv(cells_path, "frozen cells table")
        require_columns(cells_reader, ("cell_id",), "frozen cells table")
        cells_id_state = cell_id_hasher()
        try:
            with temporary_output.open("w", encoding="utf-8", newline="") as output_handle:
                writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
                writer.writerow(STANDARD_COLUMNS)
                for manifest_row in rows:
                    key = manifest_row["key"]
                    feature_path = resolve_manifest_file(
                        manifest_row["feature_path"], f"feature shard for {key}"
                    )
                    feature_receipt_path = resolve_manifest_file(
                        manifest_row["receipt_path"], f"feature receipt for {key}"
                    )
                    feature_receipt, feature_sha, feature_receipt_sha = validate_feature_receipt(
                        key, feature_path, feature_receipt_path
                    )
                    prediction_path, prediction_receipt_path = prediction_paths(
                        prediction_root, feature_receipt
                    )
                    prediction_receipt, prediction_sha = validate_prediction_receipt(
                        prediction_path,
                        prediction_receipt_path,
                        feature_receipt,
                        feature_sha,
                        feature_receipt_sha,
                        model_acceptance_receipt,
                        observed_acceptance_sha256,
                    )
                    identity = {
                        field: prediction_receipt.get(field)
                        for field in (
                            "model_id",
                            "class_ids",
                            "model_sha256",
                            "model_manifest_sha256",
                            "model_acceptance_receipt",
                            "model_acceptance_sha256",
                            "dependency_lock_sha256",
                            "implementation_sha256",
                            "source_identity",
                            "runtime_identity",
                        )
                    }
                    if common_identity is None:
                        common_identity = identity
                    elif identity != common_identity:
                        raise ValueError(f"Prediction generation identity changed at key={key}")
                    branch = str(feature_receipt["branch"])
                    observed_branches.add(branch)
                    aggregate.update(
                        (
                            json.dumps(
                                {
                                    "key": key,
                                    "feature_sha256": feature_sha,
                                    "feature_receipt_sha256": feature_receipt_sha,
                                    "prediction_sha256": prediction_sha,
                                    "prediction_receipt_sha256": sha256_file(
                                        prediction_receipt_path
                                    ),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                    )

                    feature_handle, feature_reader = open_tsv(
                        feature_path, f"feature shard {key}"
                    )
                    prediction_handle, prediction_reader = open_tsv(
                        prediction_path, f"prediction shard {key}"
                    )
                    try:
                        exact_columns(
                            feature_reader,
                            [str(value) for value in feature_receipt["feature_columns"]],
                            f"feature shard {key}",
                        )
                        exact_columns(
                            prediction_reader,
                            [str(value) for value in prediction_receipt["output_columns"]],
                            f"prediction shard {key}",
                        )
                        class_ids = [str(value) for value in prediction_receipt["class_ids"]]
                        feature_id_state = cell_id_hasher()
                        prediction_id_state = cell_id_hasher()
                        shard_rows = 0
                        shard_ok_count = 0
                        shard_unavailable_count = 0
                        shard_max_probability_sum_error = 0.0
                        previous_label = 0
                        while True:
                            feature_row = next(feature_reader, None)
                            prediction_row = next(prediction_reader, None)
                            if feature_row is None and prediction_row is None:
                                break
                            if feature_row is None or prediction_row is None:
                                raise ValueError(f"Feature/prediction row count mismatch for key={key}")
                            shard_rows += 1
                            total_rows += 1
                            feature_cell = (feature_row.get("cell_id") or "").strip()
                            prediction_cell = (prediction_row.get("cell_id") or "").strip()
                            if not feature_cell or feature_cell != prediction_cell:
                                raise ValueError(f"Feature/prediction cell identity mismatch for key={key}")
                            expected_global_cell = next_cell_id(cells_reader, total_rows)
                            if feature_cell != expected_global_cell:
                                raise ValueError(
                                    f"Merged order/cell universe differs from frozen cells table at row {total_rows}"
                                )
                            match = CELL_ID_RE.fullmatch(feature_cell)
                            if match is None or match.group("key") != key or match.group(
                                "branch"
                            ) != branch:
                                raise ValueError(f"Malformed stable cell_id for key={key}: {feature_cell}")
                            label = int(match.group("label"))
                            try:
                                feature_label = int(feature_row.get("mask_label") or "")
                            except ValueError as error:
                                raise ValueError(
                                    f"Invalid feature mask_label for key={key}: {feature_cell}"
                                ) from error
                            if feature_label != label or label <= previous_label:
                                raise ValueError(
                                    f"Feature labels are not strictly ordered for key={key}"
                                )
                            previous_label = label
                            update_cell_id_hash(feature_id_state, feature_cell)
                            update_cell_id_hash(prediction_id_state, prediction_cell)
                            update_cell_id_hash(cells_id_state, expected_global_cell)

                            model_id = (prediction_row.get("model_id") or "").strip()
                            if model_id != common_identity["model_id"]:
                                raise ValueError(f"Prediction model_id mismatch for key={key}")
                            status = (prediction_row.get("prediction_status") or "").strip()
                            predicted = (prediction_row.get("predicted_class_id") or "").strip()
                            if status == "ok":
                                values = [
                                    parse_probability(
                                        prediction_row.get(f"probability__{class_id}") or "",
                                        allow_blank=False,
                                        context=f"key={key} cell_id={feature_cell} class={class_id}",
                                    )
                                    for class_id in class_ids
                                ]
                                numeric_values = [float(value) for value in values if value is not None]
                                error = abs(sum(numeric_values) - 1.0)
                                maximum_probability_sum_error = max(
                                    maximum_probability_sum_error, error
                                )
                                shard_max_probability_sum_error = max(
                                    shard_max_probability_sum_error, error
                                )
                                if error > 1e-6:
                                    raise ValueError(
                                        f"Probability sum differs from one for cell_id={feature_cell}: {error}"
                                    )
                                expected_predicted = class_ids[
                                    max(range(len(numeric_values)), key=numeric_values.__getitem__)
                                ]
                                if predicted != expected_predicted:
                                    raise ValueError(
                                        f"Prediction is inconsistent with first-maximum tie rule: {feature_cell}"
                                    )
                                ok_count += 1
                                shard_ok_count += 1
                                class_counts[predicted] += 1
                            elif status == "unavailable_missing_features":
                                if predicted:
                                    raise ValueError(
                                        f"Unavailable prediction has a class assignment: {feature_cell}"
                                    )
                                for class_id in class_ids:
                                    parse_probability(
                                        prediction_row.get(f"probability__{class_id}") or "",
                                        allow_blank=True,
                                        context=f"key={key} cell_id={feature_cell} class={class_id}",
                                    )
                                    if prediction_row.get(f"probability__{class_id}") not in (
                                        "",
                                        None,
                                    ):
                                        raise ValueError(
                                            f"Unavailable prediction has nonblank probability: {feature_cell}"
                                        )
                                unavailable_count += 1
                                shard_unavailable_count += 1
                            else:
                                raise ValueError(
                                    f"Unsupported prediction status {status!r}: {feature_cell}"
                                )
                            writer.writerow((model_id, feature_cell, predicted, status))

                        if shard_rows != int(feature_receipt.get("row_count", -1)) or shard_rows != int(
                            prediction_receipt.get("row_count", -1)
                        ):
                            raise ValueError(f"Receipt row_count mismatch for key={key}")
                        feature_cell_hash = feature_id_state[0].hexdigest()
                        prediction_cell_hash = prediction_id_state[0].hexdigest()
                        if feature_cell_hash != feature_receipt.get("cell_id_sha256"):
                            raise ValueError(f"Feature receipt cell hash mismatch for key={key}")
                        if prediction_cell_hash != prediction_receipt.get("cell_id_sha256"):
                            raise ValueError(f"Prediction receipt cell hash mismatch for key={key}")
                        if shard_ok_count != int(prediction_receipt.get("ok_count", -1)):
                            raise ValueError(f"Prediction receipt ok_count mismatch for key={key}")
                        if shard_unavailable_count != int(
                            prediction_receipt.get("unavailable_count", -1)
                        ):
                            raise ValueError(
                                f"Prediction receipt unavailable_count mismatch for key={key}"
                            )
                        try:
                            receipt_sum_error = float(
                                prediction_receipt["max_probability_sum_error"]
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                f"Prediction receipt lacks max_probability_sum_error for key={key}"
                            ) from error
                        if not math.isfinite(receipt_sum_error) or abs(
                            receipt_sum_error - shard_max_probability_sum_error
                        ) > 1e-12:
                            raise ValueError(
                                f"Prediction receipt probability audit mismatch for key={key}"
                            )
                    finally:
                        feature_handle.close()
                        prediction_handle.close()

                if next(cells_reader, None) is not None:
                    raise ValueError("Shard predictions do not cover the complete frozen cells table")
        finally:
            cells_handle.close()
        candidate_output_sha256 = sha256_file(temporary_output)
        if reuse_requested:
            if not output_path.is_file() or sha256_file(output_path) != candidate_output_sha256:
                raise RuntimeError(
                    "Existing merged shadow axis differs after complete input-generation revalidation; "
                    "the existing generation was preserved"
                )
        else:
            os.replace(temporary_output, output_path)

    if common_identity is None:
        raise RuntimeError("No prediction generation identity was observed")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "feature_manifest": str(feature_manifest),
        "feature_manifest_sha256": feature_manifest_sha,
        "prediction_root": str(prediction_root),
        "model_acceptance": {
            "receipt": str(model_acceptance_receipt),
            "sha256": observed_acceptance_sha256,
        },
        "shadow_root": str(shadow_root),
        "cells": str(cells_path),
        "cells_sha256": cells_sha,
        "cells_cell_id_sha256": cells_id_state[0].hexdigest(),
        "shard_count": len(rows),
        "cell_count": total_rows,
        "ok_count": ok_count,
        "unavailable_count": unavailable_count,
        "branches": sorted(observed_branches),
        "model": common_identity,
        "predicted_class_counts": dict(sorted(class_counts.items())),
        "max_probability_sum_error": maximum_probability_sum_error,
        "input_generation_aggregate_sha256": aggregate.hexdigest(),
        "global_uniqueness_contract": (
            "feature manifest key is unique; each shard cell_id is branch|key|strictly-increasing-mask_label; "
            "the merged stream exactly equals frozen cpa/cells.tsv"
        ),
        "published_shadow_axis": {
            "path": str(output_path),
            "sha256": candidate_output_sha256,
            "columns": list(STANDARD_COLUMNS),
            "semantic_axis": "broad_phenotype",
            "overwrites_viability_state": False,
            "overwrites_trajectory_state": False,
        },
        "probability_storage": {
            "mode": "audited_wide_shards",
            "root": str(prediction_root / "shards"),
            "merged_probability_table_written": False,
        },
        "legacy_viability_gate": legacy,
        "technical_decision": "GO",
        "promotion_decision": "NO_GO",
        "overall_decision": "SHADOW_ONLY",
        "promotion_reason": (
            "Independent biological validation and an explicit promotion receipt are required; "
            "this technical result never replaces RGB viability thresholds."
        ),
    }
    if reuse_requested:
        existing_receipt = read_json(receipt_path, "existing shard merge receipt")
        if existing_receipt != receipt:
            raise RuntimeError(
                "Existing shard-merge receipt differs after complete feature/prediction receipt, "
                "hash, schema, cell, probability, and aggregate revalidation; the existing "
                "generation was preserved"
            )
        print(f"broad_phenotype_shard_merge_already_complete=1 output={output_path}")
        return 0

    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp.{os.getpid()}")
    try:
        temporary_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_receipt, receipt_path)
    finally:
        if temporary_receipt.exists():
            temporary_receipt.unlink()

    print("broad_phenotype_shard_merge_technical_decision=GO")
    print("broad_phenotype_shard_merge_promotion_decision=NO_GO")
    print(f"shards={len(rows)} cells={total_rows} output={output_path}")
    print(f"receipt={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
