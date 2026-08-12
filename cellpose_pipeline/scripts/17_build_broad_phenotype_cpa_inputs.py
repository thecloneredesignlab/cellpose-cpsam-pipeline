#!/usr/bin/env python3
"""Build audited Cell Phenotype Annotator inputs from frozen broad-feature shards.

This adapter deliberately keeps the generic annotator checkout out of the data
path.  It reads a resolved post-segmentation field manifest and one feature
shard per field, then writes only into a caller-supplied shadow root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import tifffile
from PIL import Image


SCHEMA_VERSION = "broad_phenotype_cpa_adapter_v1"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
SPLIT_POLICY_VERSION = "broad_phenotype_condition_split_v1"
KEY_RE = re.compile(r"^[A-H]\d+_\d+_\d+d\d+h\d+m$")
KEY_SEARCH_RE = re.compile(r"([A-H]\d+_\d+_\d+d\d+h\d+m)")
CLASS_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}

FIELD_PATH_COLUMNS = (
    "combined_raw",
    "combined_mask",
    "brightfield_raw",
    "brightfield_mask",
    "dead_raw",
    "dead_mask",
    "nuclei_raw",
    "nuclei_extent_mask",
    "nuclei_core_mask",
    "nucleated_combined_mask",
    "nucleated_brightfield_mask",
)
# Broad-morphology review is blinded to the current RGB/viability evidence.
# Brightfield and Nuclei are the only displayed raw channels; the selected
# Combined mask remains the localization anchor.
REVIEW_RAW_COLUMNS = ("brightfield_raw", "nuclei_raw")
BRANCH_MASK_COLUMN = {
    "original": "combined_mask",
    "nucleated": "nucleated_combined_mask",
}
BRANCH_SHARD_NAME = {
    "original": "original",
    "nucleated": "nucleated",
}
PLATE_MAP_REQUIRED_COLUMNS = (
    "well",
    "plate_row",
    "plate_column",
    "doxorubicin_nm",
    "ploidy",
    "cyclophosphamide",
    "replicate",
)
# V1 preregistration: control, low/intermediate, high/intermediate, and maximum
# dose within every ploidy x cyclophosphamide stratum.  Ranks are one-based.
HELDOUT_DOSE_RANKS = (1, 4, 7, 10)
EXPECTED_SUM159_DOSES = (
    "0",
    "3.125",
    "6.25",
    "12.5",
    "25",
    "50",
    "100",
    "200",
    "400",
    "800",
)
EXPECTED_SUM159_STRATA = {
    ("2N", "false"),
    ("2N", "true"),
    ("4N", "false"),
    ("4N", "true"),
}

IDENTITY_COLUMNS = {
    "cell_id",
    "image_id",
    "key",
    "well",
    "site",
    "day",
    "hour",
    "minute",
    "elapsed_hours",
    "mask_id",
    "branch",
    "split",
}
LOCALIZATION_COLUMNS = {
    "centroid_x",
    "centroid_y",
    "bbox_x0",
    "bbox_x1",
    "bbox_y0",
    "bbox_y1",
    "crop_x_min",
    "crop_x_max",
    "crop_y_min",
    "crop_y_max",
}
LOCALIZATION_RE = re.compile(
    r"(^|_)(centroid_[xy](?:_px)?|bbox_[xy](?:_[a-z]+|[01])?)(_|$)",
    re.IGNORECASE,
)
LEGACY_RESPONSE_COLUMNS = {
    "rgb_state",
    "rgb_reason",
    "state",
    "final_state",
    "final_reason",
    "classification_confidence",
    "countable",
}
LEAKAGE_TOKEN_RE = re.compile(
    r"(^|_)(class|label|state|reason|confidence|prediction|probability|"
    r"review|annotation|outcome|response)(_|$)",
    re.IGNORECASE,
)
DECISION_DERIVATIVE_RE = re.compile(
    r"(^|_)(supported|confirmed|keep_signal|strong_direct|dual_channel_confirmed|"
    r"supplemental_dead_object|match_mode|association_relation)(_|$)",
    re.IGNORECASE,
)
PROVENANCE_RE = re.compile(r"(^|_)(path|sha256|source|record)(_|$)", re.IGNORECASE)
ID_RE = re.compile(r"(^|_)(?:[A-Za-z0-9]+_)?id$", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-manifest", type=Path, required=True)
    shards = parser.add_mutually_exclusive_group(required=True)
    shards.add_argument(
        "--feature-dir",
        type=Path,
        help="Directory recursively containing one *__<branch>_broad_phenotype_features.tsv shard per field.",
    )
    shards.add_argument(
        "--feature-manifest",
        type=Path,
        help=(
            "TSV/CSV with unique key, feature_path or shard_path, and receipt_path columns."
        ),
    )
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument(
        "--feature-config",
        type=Path,
        required=True,
        help=(
            "Versioned broad-feature JSON containing numeric_feature_columns, "
            "primary_model_feature_columns, and nuclei_comparator_feature_columns."
        ),
    )
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--branch", choices=sorted(BRANCH_MASK_COLUMN), required=True)
    parser.add_argument("--project-id", default="broad_phenotype")
    parser.add_argument(
        "--feature-columns",
        help=(
            "Deprecated assertion only: when supplied it must exactly match "
            "feature-config.primary_model_feature_columns."
        ),
    )
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument(
        "--heldout-wells",
        help=(
            "Explicit comma-separated heldout wells. Intended for synthetic tests or a "
            "separately preregistered split; formal SUM159 runs use --plate-map."
        ),
    )
    split.add_argument(
        "--plate-map",
        type=Path,
        help=(
            "Plate map used for the frozen condition-balanced V1 split. Every treatment "
            "condition must have exactly two replicate wells."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace a non-identical frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, payload: str) -> None:
    atomic_bytes(path, payload.encode("utf-8"))


def install_temporary_frozen(temporary: Path, path: Path) -> None:
    """Atomically install a large streamed file without reading it into memory."""
    if path.exists():
        if path.is_file() and sha256_file(path) == sha256_file(temporary):
            temporary.unlink()
            return
        raise FileExistsError(f"Refusing to replace a non-identical frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, path)


def tsv_bytes(fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def write_tsv_frozen(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    atomic_bytes(path, tsv_bytes(fieldnames, rows))


def read_delimited(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = path.open("r", newline="", encoding="utf-8-sig")
    sample = handle.read(8192)
    handle.seek(0)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if "\t" in sample.splitlines()[0] if sample.splitlines() else False:
        delimiter = "\t"
    reader = csv.DictReader(handle, delimiter=delimiter)
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"Delimited table has no header: {path}")
    fields = [str(value).strip() for value in reader.fieldnames]
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        handle.close()
        raise ValueError(f"Delimited table has blank or duplicate columns: {path}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {field: str(row.get(field, "") or "").strip() for field in fields}
        finally:
            handle.close()

    return fields, rows()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, iterator = read_delimited(path)
    return fields, list(iterator)


def read_delimited_header(path: Path) -> list[str]:
    """Read and validate only a delimited-table header without opening a row iterator."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        if sample.splitlines() and "\t" in sample.splitlines()[0]:
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Delimited table has no header: {path}")
        fields = [str(value).strip() for value in reader.fieldnames]
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        raise ValueError(f"Delimited table has blank or duplicate columns: {path}")
    return fields


def require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be inside shadow root {root}: {resolved}") from error
    return resolved


def parse_key_metadata(key: str) -> dict[str, Any]:
    if KEY_RE.fullmatch(key) is None:
        raise ValueError(f"Invalid field key: {key}")
    well, site, stamp = key.split("_", 2)
    match = re.fullmatch(r"(\d+)d(\d+)h(\d+)m", stamp)
    assert match is not None
    day, hour, minute = map(int, match.groups())
    return {
        "well": well,
        "site": int(site),
        "day": day,
        "hour": hour,
        "minute": minute,
        "elapsed_hours": day * 24 + hour + minute / 60.0,
    }


def load_field_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, rows = read_table(path)
    required = {
        "key",
        *REVIEW_RAW_COLUMNS,
        "combined_mask",
        "nucleated_combined_mask",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Field manifest is missing columns: {missing}")
    if not rows:
        raise ValueError(f"Field manifest has no rows: {path}")
    counts = Counter(row["key"] for row in rows)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"Field manifest contains duplicate keys: {duplicates[:10]}")
    for row in rows:
        parse_key_metadata(row["key"])
    return fields, sorted(rows, key=lambda row: row["key"])


def feature_path_key(path: Path) -> str:
    match = KEY_SEARCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract a field key from feature shard: {path}")
    return match.group(1)


def index_feature_shards(
    feature_dir: Path | None,
    feature_manifest: Path | None,
    expected_branch: str,
) -> dict[str, tuple[Path, Path]]:
    result: dict[str, tuple[Path, Path]] = {}
    if feature_manifest is not None:
        fields, rows = read_table(feature_manifest)
        path_column = "feature_path" if "feature_path" in fields else "shard_path" if "shard_path" in fields else ""
        if "key" not in fields or not path_column or "receipt_path" not in fields:
            raise ValueError(
                "Feature manifest requires key, feature_path or shard_path, and receipt_path"
            )
        for row in rows:
            key = row["key"]
            path = Path(row[path_column]).expanduser()
            receipt = Path(row["receipt_path"]).expanduser()
            if not path.is_absolute():
                path = feature_manifest.parent / path
            if not receipt.is_absolute():
                receipt = feature_manifest.parent / receipt
            path = path.resolve()
            receipt = receipt.resolve()
            expected_suffix = f"__{expected_branch}_broad_phenotype_features.tsv"
            if not path.name.endswith(expected_suffix) or feature_path_key(path) != key:
                raise ValueError(
                    f"Feature manifest shard does not match key/branch contract: "
                    f"key={key} branch={expected_branch} path={path}"
                )
            if key in result:
                raise ValueError(f"Duplicate feature shard key in manifest: {key}")
            result[key] = (path, receipt)
    else:
        assert feature_dir is not None
        for path in sorted(feature_dir.resolve().rglob("*")):
            name = path.name.lower()
            if not path.is_file() or not name.endswith("_broad_phenotype_features.tsv"):
                continue
            branch_match = re.search(r"__(original|nucleated)_broad_phenotype_features\.tsv$", path.name)
            if branch_match is None:
                raise ValueError(f"Cannot infer branch from feature shard: {path}")
            if branch_match.group(1) != expected_branch:
                continue
            key = feature_path_key(path)
            if key in result:
                raise ValueError(f"Duplicate feature shards for key={key}: {result[key][0]} and {path}")
            receipt = feature_dir.resolve() / "receipts" / key.split("_", 1)[0] / f"{key}__{branch_match.group(1)}.json"
            result[key] = (path.resolve(), receipt.resolve())
    return result


def image_shape(path: Path) -> tuple[int, int, int, int]:
    """Return height, width, component count, and page count without full decode."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError(f"TIFF has no image series: {path}")
            series = tif.series[0]
            shape = tuple(int(value) for value in series.shape)
            axes = str(series.axes)
            if "Y" not in axes or "X" not in axes:
                raise ValueError(f"Cannot resolve TIFF Y/X axes for {path}: axes={axes} shape={shape}")
            height = shape[axes.index("Y")]
            width = shape[axes.index("X")]
            components = shape[axes.index("S")] if "S" in axes else shape[axes.index("C")] if "C" in axes else 1
            pages = len(tif.pages)
            return height, width, components, pages
    with Image.open(path) as image:
        width, height = image.size
        components = len(image.getbands())
        return int(height), int(width), int(components), 1


def exclusion_reason(column: str) -> str | None:
    lowered = column.lower()
    if lowered in IDENTITY_COLUMNS:
        return "identity_or_group"
    if lowered in LOCALIZATION_COLUMNS or LOCALIZATION_RE.search(lowered):
        return "localization_coordinate"
    if lowered in LEGACY_RESPONSE_COLUMNS or LEAKAGE_TOKEN_RE.search(lowered):
        return "legacy_response_or_label"
    if DECISION_DERIVATIVE_RE.search(lowered):
        return "legacy_decision_derivative"
    if PROVENANCE_RE.search(lowered):
        return "provenance"
    if ID_RE.search(lowered):
        return "identifier"
    return None


def numeric_value(value: str) -> tuple[str, float | None]:
    stripped = value.strip()
    if stripped.lower() in MISSING_TOKENS:
        return "missing", None
    try:
        parsed = float(stripped)
    except ValueError:
        return "nonnumeric", None
    if not math.isfinite(parsed):
        return "nonfinite", None
    return "finite", parsed


def format_number(value: float) -> str:
    return format(value, ".17g")


def validate_classes(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, rows = read_table(path)
    required = ["class_id", "display_name", "color", "trainable", "role", "order"]
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(f"Classes file is missing columns: {missing}")
    output_fields = [
        "class_id",
        "display_name",
        "color",
        "description",
        "shortcut",
        "trainable",
        "role",
        "order",
    ]
    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    seen_shortcuts: set[str] = set()
    trainable = 0
    for row in rows:
        class_id = row["class_id"]
        if CLASS_ID_RE.fullmatch(class_id) is None or class_id in seen_ids:
            raise ValueError(f"Invalid or duplicate class_id: {class_id}")
        try:
            order = int(row["order"])
        except ValueError as error:
            raise ValueError(f"Class order must be a positive integer: {row['order']}") from error
        if order < 1 or order in seen_orders:
            raise ValueError(f"Invalid or duplicate class order: {order}")
        role = row["role"].lower()
        if role not in {"phenotype", "uncertainty"}:
            raise ValueError(f"Invalid class role for {class_id}: {role}")
        train = row["trainable"].lower()
        if train not in {"true", "false"}:
            raise ValueError(f"Invalid trainable flag for {class_id}: {train}")
        if role == "uncertainty" and train == "true":
            raise ValueError(f"Uncertainty class cannot be trainable: {class_id}")
        shortcut = row.get("shortcut", "")
        if shortcut:
            if len(shortcut) != 1 or shortcut.isspace() or shortcut in seen_shortcuts:
                raise ValueError(f"Invalid or duplicate class shortcut: {shortcut}")
            seen_shortcuts.add(shortcut)
        seen_ids.add(class_id)
        seen_orders.add(order)
        trainable += int(train == "true")
        normalized.append(
            {
                "class_id": class_id,
                "display_name": row["display_name"],
                "color": row["color"],
                "description": row.get("description", ""),
                "shortcut": shortcut,
                "trainable": train,
                "role": role,
                "order": str(order),
            }
        )
    if trainable < 2:
        raise ValueError("At least two trainable phenotype classes are required")
    normalized.sort(key=lambda row: int(row["order"]))
    return output_fields, normalized


def load_feature_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feature config must contain one JSON object: {path}")
    if payload.get("schema_version") != "broad_phenotype_feature_config_v1":
        raise ValueError(f"Unsupported feature config schema: {payload.get('schema_version')}")
    required_keys = {
        "schema_version",
        "method_version",
        "feature_schema_version",
        "numeric_feature_columns",
        "primary_model_feature_columns",
        "nuclei_comparator_feature_columns",
        "parameters",
    }
    if set(payload) != required_keys:
        raise ValueError(
            f"Feature config keys mismatch: missing={sorted(required_keys-set(payload))} "
            f"unknown={sorted(set(payload)-required_keys)}"
        )
    if not isinstance(payload["method_version"], str) or not payload["method_version"]:
        raise ValueError("Feature config method_version must be nonblank")
    if not isinstance(payload["feature_schema_version"], str) or not payload["feature_schema_version"]:
        raise ValueError("Feature config feature_schema_version must be nonblank")
    if not isinstance(payload["parameters"], dict) or not payload["parameters"]:
        raise ValueError("Feature config parameters must be a nonempty object")
    keys = (
        "numeric_feature_columns",
        "primary_model_feature_columns",
        "nuclei_comparator_feature_columns",
    )
    result: dict[str, list[str]] = {}
    for key in keys:
        value = payload.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(column, str) or not column for column in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(f"Feature config {key} must be a nonempty unique string array")
        result[key] = list(value)
    numeric = set(result["numeric_feature_columns"])
    for key in ("primary_model_feature_columns", "nuclei_comparator_feature_columns"):
        outside = sorted(set(result[key]) - numeric)
        if outside:
            raise ValueError(f"Feature config {key} contains nonnumeric columns: {outside}")
    forbidden_primary = {
        column: exclusion_reason(column)
        for column in result["primary_model_feature_columns"]
        if exclusion_reason(column) is not None
    }
    if forbidden_primary:
        raise ValueError(
            "Primary model feature config contains identity, localization, provenance, or leakage columns: "
            + json.dumps(forbidden_primary, sort_keys=True)
        )
    semantic_payload = {
        "schema_version": payload["schema_version"],
        "method_version": payload["method_version"],
        "feature_schema_version": payload["feature_schema_version"],
        "numeric_feature_columns": result["numeric_feature_columns"],
        "primary_model_feature_columns": result["primary_model_feature_columns"],
        "nuclei_comparator_feature_columns": result["nuclei_comparator_feature_columns"],
        "parameters": payload["parameters"],
    }
    schema_payload = {
        "feature_schema_version": payload["feature_schema_version"],
        "feature_code_version": payload["method_version"],
        "numeric_feature_columns": result["numeric_feature_columns"],
        "parameters": payload["parameters"],
    }
    result.update(
        {
            "method_version": payload["method_version"],
            "feature_schema_version": payload["feature_schema_version"],
            "parameters": payload["parameters"],
            "raw_sha256": sha256_file(path),
            "semantic_sha256": hashlib.sha256(
                json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "feature_schema_sha256": hashlib.sha256(
                json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    )
    return result


def validate_feature_receipt(
    receipt_path: Path,
    shard: Path,
    key: str,
    branch: str,
    feature_config: dict[str, Any],
    observed_shard_sha256: str,
) -> dict[str, Any]:
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing feature receipt for key={key}: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError(f"Feature receipt must contain one object: {receipt_path}")
    expected = {
        "schema_version": "broad_phenotype_feature_receipt_v1",
        "status": "COMPLETE",
        "key": key,
        "branch": branch,
        "feature_schema_version": feature_config["feature_schema_version"],
        "feature_code_version": feature_config["method_version"],
        "feature_schema_sha256": feature_config["feature_schema_sha256"],
        "feature_config_sha256": feature_config["raw_sha256"],
        "feature_config_semantic_sha256": feature_config["semantic_sha256"],
        "feature_tsv_sha256": observed_shard_sha256,
        "numeric_feature_columns": feature_config["numeric_feature_columns"],
        "primary_model_feature_columns": feature_config["primary_model_feature_columns"],
        "nuclei_comparator_feature_columns": feature_config["nuclei_comparator_feature_columns"],
        "feature_parameters": feature_config["parameters"],
        "forbidden_inputs_read": [],
    }
    mismatches = {
        field: {"expected": value, "observed": receipt.get(field)}
        for field, value in expected.items()
        if receipt.get(field) != value
    }
    observed_path = Path(str(receipt.get("feature_tsv", ""))).expanduser()
    if not observed_path.is_absolute():
        observed_path = receipt_path.parent / observed_path
    if observed_path.resolve() != shard.resolve():
        mismatches["feature_tsv"] = {
            "expected": str(shard.resolve()),
            "observed": str(observed_path.resolve()),
        }
    row_count = receipt.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        mismatches["row_count"] = {"expected": "nonnegative integer", "observed": row_count}
    input_contract = receipt.get("input_contract")
    if not isinstance(input_contract, dict) or input_contract.get("dead") != "not_read" or input_contract.get("current_classification") != "not_read":
        mismatches["input_contract"] = {
            "expected": {"dead": "not_read", "current_classification": "not_read"},
            "observed": input_contract,
        }
    implementation = receipt.get("implementation")
    extractor_path = Path(__file__).resolve().with_name(
        "16_extract_broad_phenotype_features.py"
    )
    feature_module_path = Path(__file__).resolve().parent / "_shared" / "broad_phenotype_features.py"
    expected_implementation = {
        "extractor_sha256": sha256_file(extractor_path),
        "feature_module_sha256": sha256_file(feature_module_path),
    }
    if not isinstance(implementation, dict):
        mismatches["implementation"] = {
            "expected": expected_implementation,
            "observed": implementation,
        }
    else:
        for name, expected_value in expected_implementation.items():
            if implementation.get(name) != expected_value:
                mismatches[f"implementation.{name}"] = {
                    "expected": expected_value,
                    "observed": implementation.get(name),
                }
    missing_counts = receipt.get("missing_value_counts")
    if (
        not isinstance(missing_counts, dict)
        or set(missing_counts) != set(feature_config["numeric_feature_columns"])
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (isinstance(row_count, int) and value > row_count)
            for value in missing_counts.values()
        )
    ):
        mismatches["missing_value_counts"] = {
            "expected": "one nonnegative integer <= row_count for every numeric feature",
            "observed": missing_counts,
        }
    diagnostics = receipt.get("diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("object_count") != row_count
        or not isinstance(diagnostics.get("positive_labels_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", diagnostics.get("positive_labels_sha256", ""))
        is None
    ):
        mismatches["diagnostics"] = {
            "expected": "object_count=row_count and a positive-label SHA-256",
            "observed": diagnostics,
        }
    if mismatches:
        raise RuntimeError(
            f"Feature receipt verification failed for key={key}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return receipt


def canonical_dose(value: str) -> tuple[Decimal, str]:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid doxorubicin_nm value in plate map: {value!r}") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"doxorubicin_nm must be finite and nonnegative: {value!r}")
    canonical = format(parsed.normalize(), "f")
    if canonical == "-0":
        canonical = "0"
    return parsed, canonical


def condition_balanced_splits(
    wells: Sequence[str], plate_map_path: Path, seed: int
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    fields, rows = read_table(plate_map_path)
    missing_columns = sorted(set(PLATE_MAP_REQUIRED_COLUMNS) - set(fields))
    if missing_columns:
        raise ValueError(f"Plate map is missing required columns: {missing_columns}")

    expected_wells = set(wells)
    by_well: dict[str, dict[str, str]] = {}
    conditions: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        well = row["well"]
        if not well or well in by_well:
            raise ValueError(f"Plate map contains a blank or duplicate well: {well!r}")
        _, dose = canonical_dose(row["doxorubicin_nm"])
        ploidy = row["ploidy"].strip()
        cyclophosphamide = row["cyclophosphamide"].strip().lower()
        if not ploidy or cyclophosphamide not in {"true", "false"}:
            raise ValueError(
                f"Invalid plate-map condition for well={well}: "
                f"ploidy={ploidy!r} cyclophosphamide={cyclophosphamide!r}"
            )
        try:
            replicate = int(row["replicate"])
        except ValueError as error:
            raise ValueError(f"Invalid replicate for well={well}: {row['replicate']!r}") from error
        if replicate not in {1, 2}:
            raise ValueError(f"Replicate must be 1 or 2 for well={well}: {replicate}")
        plate_row = row["plate_row"].strip()
        try:
            plate_column = int(row["plate_column"])
        except ValueError as error:
            raise ValueError(
                f"Invalid plate_column for well={well}: {row['plate_column']!r}"
            ) from error
        well_match = re.fullmatch(r"([A-H])(\d+)", well)
        if (
            well_match is None
            or plate_row != well_match.group(1)
            or plate_column != int(well_match.group(2))
        ):
            raise ValueError(
                f"Plate-map well coordinate mismatch: well={well} "
                f"plate_row={plate_row!r} plate_column={plate_column}"
            )
        normalized = {
            "well": well,
            "plate_row": plate_row,
            "plate_column": str(plate_column),
            "doxorubicin_nm": dose,
            "ploidy": ploidy,
            "cyclophosphamide": cyclophosphamide,
            "replicate": str(replicate),
        }
        by_well[well] = normalized
        conditions.setdefault((dose, ploidy, cyclophosphamide), []).append(normalized)

    missing_wells = sorted(expected_wells - set(by_well))
    extra_wells = sorted(set(by_well) - expected_wells)
    if missing_wells or extra_wells:
        raise ValueError(
            f"Plate-map well universe differs from field manifest: "
            f"missing={missing_wells} extra={extra_wells}"
        )

    strata: dict[tuple[str, str], set[str]] = {}
    for (dose, ploidy, cyclophosphamide), members in conditions.items():
        replicates = {int(member["replicate"]) for member in members}
        if len(members) != 2 or replicates != {1, 2}:
            raise ValueError(
                "Every plate-map condition must contain exactly replicate 1 and replicate 2: "
                f"condition={(dose, ploidy, cyclophosphamide)} members={members}"
            )
        strata.setdefault((ploidy, cyclophosphamide), set()).add(dose)

    ranked_doses: list[str] | None = None
    stratum_doses: dict[tuple[str, str], list[str]] = {}
    for stratum, doses in sorted(strata.items()):
        ordered = sorted(doses, key=lambda value: canonical_dose(value)[0])
        if max(HELDOUT_DOSE_RANKS) > len(ordered):
            raise ValueError(
                f"Plate-map stratum {stratum} has only {len(ordered)} dose levels; "
                f"cannot select preregistered ranks {HELDOUT_DOSE_RANKS}"
            )
        if ranked_doses is None:
            ranked_doses = ordered
        elif ordered != ranked_doses:
            raise ValueError(
                f"Dose levels differ across plate-map strata: expected={ranked_doses} "
                f"observed={ordered} stratum={stratum}"
            )
        stratum_doses[stratum] = ordered

    if set(strata) != EXPECTED_SUM159_STRATA:
        raise ValueError(
            f"SUM159 plate-map strata drift: expected={sorted(EXPECTED_SUM159_STRATA)} "
            f"observed={sorted(strata)}"
        )
    if ranked_doses != list(EXPECTED_SUM159_DOSES):
        raise ValueError(
            f"SUM159 plate-map dose grid drift: expected={list(EXPECTED_SUM159_DOSES)} "
            f"observed={ranked_doses}"
        )
    if len(by_well) != 80 or len(conditions) != 40:
        raise ValueError(
            f"SUM159 plate-map size drift: wells={len(by_well)} conditions={len(conditions)}"
        )

    splits = {well: "development" for well in expected_wells}
    condition_audit: list[dict[str, Any]] = []
    heldout_replicate_counts = Counter()
    selected_condition_count = 0
    for stratum, ordered_doses in sorted(stratum_doses.items()):
        ploidy, cyclophosphamide = stratum
        first_replicate = 1 + (
            int(stable_hash(SCHEMA_VERSION, seed, "heldout_replicate_start", *stratum), 16) % 2
        )
        selected_rank_set = set(HELDOUT_DOSE_RANKS)
        selected_index = 0
        for dose_rank, dose in enumerate(ordered_doses, start=1):
            members = sorted(
                conditions[(dose, ploidy, cyclophosphamide)],
                key=lambda member: int(member["replicate"]),
            )
            by_replicate = {int(member["replicate"]): member["well"] for member in members}
            selected = dose_rank in selected_rank_set
            heldout_replicate: int | None = None
            heldout_well = ""
            if selected:
                heldout_replicate = first_replicate if selected_index % 2 == 0 else 3 - first_replicate
                selected_index += 1
                heldout_well = by_replicate[heldout_replicate]
                splits[heldout_well] = "heldout"
                heldout_replicate_counts[heldout_replicate] += 1
                selected_condition_count += 1
            development_wells = [
                well for replicate, well in sorted(by_replicate.items()) if well != heldout_well
            ]
            condition_audit.append(
                {
                    "condition_id": (
                        f"doxorubicin_nm={dose}|ploidy={ploidy}|"
                        f"cyclophosphamide={cyclophosphamide}"
                    ),
                    "doxorubicin_nm": dose,
                    "dose_rank_within_stratum": dose_rank,
                    "ploidy": ploidy,
                    "cyclophosphamide": cyclophosphamide,
                    "replicate_1_well": by_replicate[1],
                    "replicate_2_well": by_replicate[2],
                    "selected_for_heldout": str(selected).lower(),
                    "heldout_replicate": heldout_replicate or "",
                    "heldout_well": heldout_well,
                    "development_wells": ",".join(development_wells),
                    "development_support_count": len(development_wells),
                }
            )

    if selected_condition_count != len(strata) * len(HELDOUT_DOSE_RANKS):
        raise RuntimeError("Internal condition-balanced split count mismatch")
    if heldout_replicate_counts[1] != heldout_replicate_counts[2]:
        raise RuntimeError(
            f"Heldout replicate assignments are not balanced: {dict(heldout_replicate_counts)}"
        )
    if any(int(row["development_support_count"]) < 1 for row in condition_audit):
        raise RuntimeError("A treatment condition has no development support")

    normalized_plate_rows = [by_well[well] for well in sorted(by_well)]
    plate_map_semantic_sha256 = hashlib.sha256(
        json.dumps(
            normalized_plate_rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    split_identity_payload = {
        "schema_version": SPLIT_POLICY_VERSION,
        "seed": seed,
        "plate_map_semantic_sha256": plate_map_semantic_sha256,
        "condition_columns": ["doxorubicin_nm", "ploidy", "cyclophosphamide"],
        "stratum_columns": ["ploidy", "cyclophosphamide"],
        "heldout_dose_ranks": list(HELDOUT_DOSE_RANKS),
        "well_splits": [
            {"well": well, "split": split} for well, split in sorted(splits.items())
        ],
        "condition_audit": condition_audit,
    }
    metadata = {
        "strategy": "condition_balanced_paired_replicate_v1",
        "policy_version": SPLIT_POLICY_VERSION,
        "seed": seed,
        "plate_map": str(plate_map_path),
        "plate_map_sha256": sha256_file(plate_map_path),
        "plate_map_semantic_sha256": plate_map_semantic_sha256,
        "plate_map_columns": list(PLATE_MAP_REQUIRED_COLUMNS),
        "condition_columns": ["doxorubicin_nm", "ploidy", "cyclophosphamide"],
        "stratum_columns": ["ploidy", "cyclophosphamide"],
        "ranked_doses": ranked_doses,
        "heldout_dose_ranks": list(HELDOUT_DOSE_RANKS),
        "condition_count": len(condition_audit),
        "heldout_condition_count": selected_condition_count,
        "development_well_count": sum(value == "development" for value in splits.values()),
        "heldout_well_count": sum(value == "heldout" for value in splits.values()),
        "heldout_replicate_1_count": heldout_replicate_counts[1],
        "heldout_replicate_2_count": heldout_replicate_counts[2],
        "split_identity_sha256": hashlib.sha256(
            json.dumps(
                split_identity_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    return splits, condition_audit, by_well, metadata


def choose_well_splits(
    wells: Sequence[str], heldout_wells: str | None, plate_map: Path | None, seed: int
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    unique = sorted(set(wells))
    if len(unique) < 2:
        raise ValueError("At least two wells are required for a frozen grouped split")
    if heldout_wells:
        heldout = {value.strip() for value in heldout_wells.split(",") if value.strip()}
        unknown = sorted(heldout - set(unique))
        if unknown:
            raise ValueError(f"Heldout wells are absent from the field manifest: {unknown}")
        if not heldout or heldout == set(unique):
            raise ValueError("Explicit heldout wells must leave nonempty development and heldout groups")
        splits = {well: "heldout" if well in heldout else "development" for well in unique}
        return splits, [], {}, {
            "strategy": "explicit_heldout_wells",
            "policy_version": "explicit_heldout_wells_v1",
            "seed": seed,
            "plate_map": "",
            "plate_map_sha256": "",
            "plate_map_semantic_sha256": "",
            "plate_map_columns": [],
            "condition_columns": [],
            "stratum_columns": [],
            "ranked_doses": [],
            "heldout_dose_ranks": [],
            "condition_count": "",
            "heldout_condition_count": "",
            "development_well_count": sum(value == "development" for value in splits.values()),
            "heldout_well_count": sum(value == "heldout" for value in splits.values()),
            "heldout_replicate_1_count": "",
            "heldout_replicate_2_count": "",
            "split_identity_sha256": stable_hash(
                "explicit_heldout_wells_v1",
                seed,
                *(f"{well}={splits[well]}" for well in sorted(splits)),
            ),
        }
    if plate_map is None:
        raise ValueError("Either --plate-map or --heldout-wells is required")
    return condition_balanced_splits(unique, plate_map, seed)


def audit_feature_shards(
    fields: Sequence[dict[str, str]],
    shard_index: dict[str, tuple[Path, Path]],
    selected_mask_column: str,
    feature_config: dict[str, Any],
    expected_branch: str,
) -> tuple[
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
]:
    expected_keys = {row["key"] for row in fields}
    missing = sorted(expected_keys - set(shard_index))
    extra = sorted(set(shard_index) - expected_keys)
    if missing or extra:
        raise ValueError(
            f"Feature shard key mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    canonical_header: list[str] | None = None
    stats: dict[str, dict[str, int]] = {}
    field_audit: list[dict[str, Any]] = []
    path_audit: list[dict[str, Any]] = []
    required_path_columns = {*REVIEW_RAW_COLUMNS, selected_mask_column}
    numeric_features = feature_config["numeric_feature_columns"]
    primary_features = feature_config["primary_model_feature_columns"]
    comparator_features = feature_config["nuclei_comparator_feature_columns"]

    for position, field in enumerate(fields, start=1):
        key = field["key"]
        shard, receipt_path = shard_index[key]
        if not shard.is_file():
            raise FileNotFoundError(f"Missing feature shard for {key}: {shard}")
        shard_sha256 = sha256_file(shard)
        receipt = validate_feature_receipt(
            receipt_path,
            shard,
            key,
            expected_branch,
            feature_config,
            shard_sha256,
        )
        receipt_sha256 = sha256_file(receipt_path)

        dimensions: dict[str, tuple[int, int, int, int]] = {}
        # Only BF, Nuclei, and the selected Combined mask belong to this
        # morphology-review contract. Combined RGB and Dead are intentionally
        # not dereferenced so the labels remain blinded to viability evidence.
        for column in (*REVIEW_RAW_COLUMNS, selected_mask_column):
            raw_path = field.get(column, "")
            if not raw_path:
                if column in required_path_columns:
                    raise ValueError(f"Required path is blank for key={key}, column={column}")
                continue
            path = Path(raw_path).expanduser().resolve()
            required = column in required_path_columns
            if not path.is_file():
                path_audit.append(
                    {
                        "key": key,
                        "asset_role": column,
                        "path": str(path),
                        "required": str(required).lower(),
                        "status": "missing_required" if required else "missing_optional",
                        "height": "",
                        "width": "",
                        "components": "",
                        "pages": "",
                        "matches_combined": "",
                    }
                )
                if required:
                    raise FileNotFoundError(path)
                continue
            shape = image_shape(path)
            dimensions[column] = shape
            reference = dimensions.get("brightfield_raw")
            matches = "" if reference is None else str(shape[:2] == reference[:2]).lower()
            path_audit.append(
                {
                    "key": key,
                    "asset_role": column,
                    "path": str(path),
                    "required": str(required).lower(),
                    "status": "ok",
                    "height": shape[0],
                    "width": shape[1],
                    "components": shape[2],
                    "pages": shape[3],
                    "matches_combined": matches,
                }
            )
        reference = dimensions["brightfield_raw"]
        for column in required_path_columns:
            if dimensions[column][:2] != reference[:2]:
                raise ValueError(
                    f"Required asset dimensions differ for key={key}: "
                    f"brightfield_raw={reference[:2]} {column}={dimensions[column][:2]}"
                )
        for column in REVIEW_RAW_COLUMNS:
            shape = dimensions[column]
            if shape[2] != 1 or shape[3] != 1:
                raise ValueError(
                    f"Scalar review channel must be a single 2D plane for key={key}, {column}={shape}"
                )

        mask_path = Path(field[selected_mask_column]).expanduser().resolve()
        header = read_delimited_header(shard)
        if canonical_header is None:
            canonical_header = header
            required_columns = {"cell_id", "key", "well", "branch", "image_id", "mask_label"}
            missing_columns = sorted(required_columns - set(header))
            if missing_columns:
                raise ValueError(f"Feature shards are missing identity columns: {missing_columns}")
            missing_numeric = sorted(set(numeric_features) - set(header))
            if missing_numeric:
                raise ValueError(f"Feature shards are missing configured numeric columns: {missing_numeric}")
            for column in header:
                stats[column] = {
                    "n_rows": 0,
                    "n_missing": 0,
                    "n_finite": 0,
                    "n_nonnumeric": 0,
                    "n_nonfinite": 0,
                }
        elif header != canonical_header:
            raise ValueError(
                f"Feature shard header drift for key={key}: expected={canonical_header} observed={header}"
            )

        row_count = int(receipt["row_count"])
        missing_counts = receipt["missing_value_counts"]
        for column in canonical_header:
            column_stats = stats[column]
            column_stats["n_rows"] += row_count
            if column in numeric_features:
                missing = int(missing_counts[column])
                column_stats["n_missing"] += missing
                column_stats["n_finite"] += row_count - missing
            else:
                # Identity/provenance strings are never model candidates.  The
                # assembly pass validates their exact value contract.
                column_stats["n_nonnumeric"] += row_count
        field_audit.append(
            {
                "key": key,
                "feature_shard": str(shard),
                "mask_path": str(mask_path),
                "feature_rows": row_count,
                "mask_labels": row_count,
                "matched_labels": row_count,
                "feature_only_labels": 0,
                "mask_only_labels": 0,
                "height": reference[0],
                "width": reference[1],
                "join_status": "receipt_hash_and_label_digest_verified",
                "feature_sha256": shard_sha256,
                "receipt_sha256": receipt_sha256,
            }
        )
        if row_count != int(receipt["row_count"]):
            raise RuntimeError(
                f"Feature receipt row_count mismatch for key={key}: "
                f"receipt={receipt['row_count']} observed={row_count}"
            )
        if position % 250 == 0 or position == len(fields):
            print(f"adapter_audit_fields={position}/{len(fields)}", flush=True)

    assert canonical_header is not None
    numeric_set = set(numeric_features)
    primary_set = set(primary_features)
    comparator_set = set(comparator_features)
    feature_policy: list[dict[str, Any]] = []
    selected: list[str] = []
    retained_allowlist = primary_set | comparator_set
    for column in canonical_header:
        column_stats = stats[column]
        safety_reason = exclusion_reason(column)
        reason = safety_reason
        if column in numeric_set:
            reason = None
        if reason is None and column_stats["n_nonfinite"]:
            reason = "nonfinite_values"
        if reason is None and column_stats["n_nonnumeric"]:
            reason = "nonnumeric_values"
        if reason is None and not column_stats["n_finite"]:
            reason = "no_finite_values"
        include = column in primary_set and reason is None
        retain = column in retained_allowlist and reason is None
        if column in primary_set and not include:
            raise ValueError(f"Configured primary feature is invalid: {column} ({reason})")
        if retain:
            selected.append(column)
        usage = (
            "primary_projection_and_classifier"
            if column in primary_set
            else "nuclei_comparator_only"
            if column in comparator_set
            else "retained_numeric_not_modeled"
            if column in numeric_set
            else "excluded"
        )
        feature_policy.append(
            {
                "column": column,
                "included": str(include).lower(),
                "retained_in_features_table": str(retain).lower(),
                "usage": usage,
                "reason": "selected_primary_predictor" if include else reason or usage,
                **column_stats,
            }
        )
    expected_retained = [
        column for column in numeric_features if column in retained_allowlist
    ]
    if selected != expected_retained:
        raise ValueError(
            "Feature config/order and shard columns did not produce the exact retained allowlist: "
            f"expected={expected_retained} observed={selected}"
        )
    return selected, feature_policy, field_audit, path_audit, stats


def yaml_json(payload: dict[str, Any]) -> str:
    # JSON is valid YAML and avoids an additional Python serialization dependency.
    return json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    field_manifest = args.field_manifest.expanduser().resolve()
    classes_file = args.classes_file.expanduser().resolve()
    feature_config_path = args.feature_config.expanduser().resolve()
    shadow_root = args.shadow_root.expanduser().resolve()
    if not field_manifest.is_file():
        raise FileNotFoundError(field_manifest)
    if not classes_file.is_file():
        raise FileNotFoundError(classes_file)
    if not feature_config_path.is_file():
        raise FileNotFoundError(feature_config_path)
    if CLASS_ID_RE.fullmatch(args.project_id) is None:
        raise ValueError(f"Invalid --project-id: {args.project_id}")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Nested cross-validation requires at least two outer and inner folds")
    shadow_root.mkdir(parents=True, exist_ok=True)
    cpa_root = require_within(shadow_root / "cpa", shadow_root, "CPA output")
    audit_root = require_within(
        shadow_root / "workflow_status" / "adapter", shadow_root, "Adapter audit output"
    )

    manifest_fields, fields = load_field_manifest(field_manifest)
    source_branch = BRANCH_SHARD_NAME[args.branch]
    shard_index = index_feature_shards(
        args.feature_dir.expanduser().resolve() if args.feature_dir else None,
        args.feature_manifest.expanduser().resolve() if args.feature_manifest else None,
        source_branch,
    )
    configured = load_feature_config(feature_config_path)
    asserted_features = (
        [value.strip() for value in args.feature_columns.split(",") if value.strip()]
        if args.feature_columns
        else None
    )
    primary_features = configured["primary_model_feature_columns"]
    if asserted_features is not None and asserted_features != primary_features:
        raise ValueError(
            "--feature-columns is an assertion and must exactly match "
            "feature-config.primary_model_feature_columns"
        )
    selected_mask_column = BRANCH_MASK_COLUMN[args.branch]
    retained_features, feature_policy, field_audit, path_audit, _ = audit_feature_shards(
        fields,
        shard_index,
        selected_mask_column,
        configured,
        source_branch,
    )
    class_fields, classes = validate_classes(classes_file)

    wells = [parse_key_metadata(field["key"])["well"] for field in fields]
    plate_map_path = args.plate_map.expanduser().resolve() if args.plate_map else None
    if plate_map_path is not None and not plate_map_path.is_file():
        raise FileNotFoundError(plate_map_path)
    splits, condition_split_rows, well_conditions, split_metadata = choose_well_splits(
        wells,
        args.heldout_wells,
        plate_map_path,
        args.split_seed,
    )
    split_rows = [
        {
            "well": well,
            "plate_row": well_conditions.get(well, {}).get("plate_row", ""),
            "plate_column": well_conditions.get(well, {}).get("plate_column", ""),
            "doxorubicin_nm": well_conditions.get(well, {}).get("doxorubicin_nm", ""),
            "ploidy": well_conditions.get(well, {}).get("ploidy", ""),
            "cyclophosphamide": well_conditions.get(well, {}).get(
                "cyclophosphamide", ""
            ),
            "replicate": well_conditions.get(well, {}).get("replicate", ""),
            "split": split,
            "split_strategy": split_metadata["strategy"],
            "split_seed": args.split_seed,
            "assignment_sha256": stable_hash(
                split_metadata["policy_version"],
                split_metadata["strategy"],
                args.split_seed,
                split_metadata["plate_map_semantic_sha256"],
                well,
                well_conditions.get(well, {}).get("doxorubicin_nm", ""),
                well_conditions.get(well, {}).get("ploidy", ""),
                well_conditions.get(well, {}).get("cyclophosphamide", ""),
                well_conditions.get(well, {}).get("replicate", ""),
                split,
            ),
        }
        for well, split in sorted(splits.items())
    ]

    classes_out = cpa_root / "classes.tsv"
    cells_out = cpa_root / "cells.tsv"
    features_out = cpa_root / "features.tsv"
    images_out = cpa_root / "images.tsv"
    project_out = cpa_root / "project.yml"
    split_out = audit_root / "well_split_freeze.tsv"
    condition_split_out = audit_root / "condition_split_freeze.tsv"
    cpa_root.mkdir(parents=True, exist_ok=True)
    cell_fields = [
        "cell_id",
        "image_id",
        "mask_label",
        "branch",
        "key",
        "well",
        "site",
        "day",
        "hour",
        "minute",
        "elapsed_hours",
        "split",
    ]
    feature_fields = ["cell_id", *retained_features]
    image_fields = [
        "image_id",
        "channel_id",
        "image_path",
        "channel_index",
        "page_index",
        "display_name",
        "display_color",
        "display_percentile_low",
        "display_percentile_high",
        "mask_path",
        "mask_channel_index",
        "width",
        "height",
        "alpha_policy",
    ]
    temporary_cells = cells_out.with_name(f".{cells_out.name}.tmp.{os.getpid()}")
    temporary_features = features_out.with_name(f".{features_out.name}.tmp.{os.getpid()}")
    temporary_images = images_out.with_name(f".{images_out.name}.tmp.{os.getpid()}")
    total_cells = 0
    try:
        with (
            temporary_cells.open("w", newline="", encoding="utf-8") as cells_handle,
            temporary_features.open("w", newline="", encoding="utf-8") as features_handle,
            temporary_images.open("w", newline="", encoding="utf-8") as images_handle,
        ):
            cells_writer = csv.DictWriter(
                cells_handle, fieldnames=cell_fields, delimiter="\t", lineterminator="\n"
            )
            features_writer = csv.DictWriter(
                features_handle, fieldnames=feature_fields, delimiter="\t", lineterminator="\n"
            )
            images_writer = csv.DictWriter(
                images_handle, fieldnames=image_fields, delimiter="\t", lineterminator="\n"
            )
            cells_writer.writeheader()
            features_writer.writeheader()
            images_writer.writeheader()
            for field_position, field in enumerate(fields, start=1):
                key = field["key"]
                metadata = parse_key_metadata(key)
                well = metadata["well"]
                mask_path = Path(field[selected_mask_column]).expanduser().resolve()
                height, width, _, _ = image_shape(
                    Path(field["brightfield_raw"]).expanduser().resolve()
                )
                _, iterator = read_delimited(shard_index[key][0])
                previous_label = 0
                field_rows = 0
                for row in iterator:
                    label = int(row["mask_label"])
                    if label <= previous_label:
                        raise ValueError(
                            f"Feature shard rows must be strictly ordered by mask_label for "
                            f"deterministic streaming: key={key} previous={previous_label} current={label}"
                        )
                    previous_label = label
                    cell_id = row["cell_id"]
                    expected_cell_id = f"{source_branch}|{key}|{label}"
                    if cell_id != expected_cell_id:
                        raise ValueError(f"Stable cell_id changed between audit and assembly: {cell_id}")
                    cells_writer.writerow(
                        {
                            "cell_id": cell_id,
                            "image_id": key,
                            "mask_label": label,
                            "branch": source_branch,
                            "key": key,
                            "well": well,
                            "site": metadata["site"],
                            "day": metadata["day"],
                            "hour": metadata["hour"],
                            "minute": metadata["minute"],
                            "elapsed_hours": format_number(float(metadata["elapsed_hours"])),
                            "split": splits[well],
                        }
                    )
                    output = {"cell_id": cell_id}
                    for column in retained_features:
                        status, parsed = numeric_value(row[column])
                        if status == "finite":
                            assert parsed is not None
                            output[column] = format_number(parsed)
                        elif status == "missing":
                            output[column] = ""
                        else:
                            raise AssertionError(
                                f"Audited feature changed status: {column}={row[column]}"
                            )
                    features_writer.writerow(output)
                    total_cells += 1
                    field_rows += 1
                expected_rows = int(field_audit[field_position - 1]["feature_rows"])
                if field_rows != expected_rows:
                    raise RuntimeError(
                        f"Feature shard changed after audit: key={key} "
                        f"audited={expected_rows} observed={field_rows}"
                    )
                shard_after_sha256 = sha256_file(shard_index[key][0])
                if shard_after_sha256 != field_audit[field_position - 1]["feature_sha256"]:
                    raise RuntimeError(
                        f"Feature shard bytes changed between audit and assembly: key={key}"
                    )
                receipt_after_sha256 = sha256_file(shard_index[key][1])
                if receipt_after_sha256 != field_audit[field_position - 1]["receipt_sha256"]:
                    raise RuntimeError(
                        f"Feature receipt changed between audit and assembly: key={key}"
                    )

                common_image = {
                    "image_id": key,
                    "display_percentile_low": "1",
                    "display_percentile_high": "99",
                    "mask_path": str(mask_path),
                    "mask_channel_index": "1",
                    "width": width,
                    "height": height,
                    "alpha_policy": "reject",
                }
                for channel_id, source, name, color in (
                    ("brightfield", "brightfield_raw", "Brightfield", "#ffffff"),
                    ("nuclei", "nuclei_raw", "Nuclei", "#00ffff"),
                ):
                    images_writer.writerow(
                        {
                            **common_image,
                            "channel_id": channel_id,
                            "image_path": str(Path(field[source]).expanduser().resolve()),
                            "channel_index": "",
                            "page_index": "",
                            "display_name": name,
                            "display_color": color,
                        }
                    )
                if field_position % 250 == 0 or field_position == len(fields):
                    print(f"adapter_assemble_fields={field_position}/{len(fields)}", flush=True)
        if total_cells < 3:
            raise ValueError("At least three cells are required for UMAP preparation")
        install_temporary_frozen(temporary_cells, cells_out)
        install_temporary_frozen(temporary_features, features_out)
        install_temporary_frozen(temporary_images, images_out)
    finally:
        for temporary in (temporary_cells, temporary_features, temporary_images):
            if temporary.exists():
                temporary.unlink()
    write_tsv_frozen(classes_out, class_fields, classes)
    write_tsv_frozen(
        split_out,
        [
            "well",
            "plate_row",
            "plate_column",
            "doxorubicin_nm",
            "ploidy",
            "cyclophosphamide",
            "replicate",
            "split",
            "split_strategy",
            "split_seed",
            "assignment_sha256",
        ],
        split_rows,
    )
    write_tsv_frozen(
        condition_split_out,
        [
            "condition_id",
            "doxorubicin_nm",
            "dose_rank_within_stratum",
            "ploidy",
            "cyclophosphamide",
            "replicate_1_well",
            "replicate_2_well",
            "selected_for_heldout",
            "heldout_replicate",
            "heldout_well",
            "development_wells",
            "development_support_count",
        ],
        condition_split_rows,
    )
    write_tsv_frozen(
        audit_root / "feature_policy.tsv",
        [
            "column",
            "included",
            "retained_in_features_table",
            "usage",
            "reason",
            "n_rows",
            "n_missing",
            "n_finite",
            "n_nonnumeric",
            "n_nonfinite",
        ],
        feature_policy,
    )
    write_tsv_frozen(
        audit_root / "join_audit.tsv",
        [
            "key",
            "feature_shard",
            "mask_path",
            "feature_rows",
            "mask_labels",
            "matched_labels",
            "feature_only_labels",
            "mask_only_labels",
            "height",
            "width",
            "join_status",
        ],
        field_audit,
    )
    write_tsv_frozen(
        audit_root / "path_dimension_audit.tsv",
        [
            "key",
            "asset_role",
            "path",
            "required",
            "status",
            "height",
            "width",
            "components",
            "pages",
            "matches_combined",
        ],
        path_audit,
    )

    input_manifest_rows = [
        {
            "input_role": "field_manifest",
            "key": "",
            "path": str(field_manifest),
            "size_bytes": field_manifest.stat().st_size,
            "sha256": sha256_file(field_manifest),
        },
        {
            "input_role": "classes",
            "key": "",
            "path": str(classes_file),
            "size_bytes": classes_file.stat().st_size,
            "sha256": sha256_file(classes_file),
        },
        {
            "input_role": "feature_config",
            "key": "",
            "path": str(feature_config_path),
            "size_bytes": feature_config_path.stat().st_size,
            "sha256": sha256_file(feature_config_path),
        },
    ]
    if plate_map_path is not None:
        input_manifest_rows.append(
            {
                "input_role": "plate_map",
                "key": "",
                "path": str(plate_map_path),
                "size_bytes": plate_map_path.stat().st_size,
                "sha256": sha256_file(plate_map_path),
            }
        )
    for field_position, field in enumerate(fields):
        shard, receipt = shard_index[field["key"]]
        input_manifest_rows.append(
            {
                "input_role": "feature_shard",
                "key": field["key"],
                "path": str(shard),
                "size_bytes": shard.stat().st_size,
                "sha256": field_audit[field_position]["feature_sha256"],
            }
        )
        input_manifest_rows.append(
            {
                "input_role": "feature_receipt",
                "key": field["key"],
                "path": str(receipt),
                "size_bytes": receipt.stat().st_size,
                "sha256": field_audit[field_position]["receipt_sha256"],
            }
        )
    write_tsv_frozen(
        audit_root / "input_manifest.tsv",
        ["input_role", "key", "path", "size_bytes", "sha256"],
        input_manifest_rows,
    )

    n_neighbors = min(30, total_cells - 1)
    review_quotas: dict[str, int] = {
        "default_per_class": 50,
        "unassigned": 100,
        # Explicit-unassigned regions are optional annotation aids. Requiring a
        # nonzero quota would make review-build fail unless the annotator drew
        # an arbitrary exclusion region even when the six class regions and
        # naturally unassigned cells are sufficient.
        "explicit_unassigned": 0,
    }
    review_quotas.update(
        {
            row["class_id"]: 0
            for row in classes
            if row["role"] == "uncertainty" and row["trainable"] == "false"
        }
    )
    project = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_id": args.project_id,
        "classes_file": "classes.tsv",
        "cells_file": "cells.tsv",
        "features_file": "features.tsv",
        "images_file": "images.tsv",
        "runs_dir": "runs",
        "projection": {
            "mode": "compute_umap",
            "feature_columns": primary_features,
            "transform": "robust",
            "missing_policy": "median_impute",
            "constant_policy": "drop_with_manifest",
            "seed": args.split_seed,
            "n_neighbors": n_neighbors,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_threads": 1,
            "n_sgd_threads": 1,
        },
        "annotation": {
            "title": "Broad cell phenotype annotation",
            "direct_class_limit": 8,
            "point_radius": 1.5,
            "boundary_tolerance": 1e-10,
        },
        "review": {
            "strata": ["well"],
            # Global class quotas plus a shared per-well cap guarantee broad
            # group support without multiplying the requested review count by
            # every well.  Fifty rows per trainable class and max eight rows
            # total per well require each completed class quota to span at
            # least seven independent wells.
            "group_column": "well",
            "quota_scope": "global",
            "quotas": review_quotas,
            "max_per_group": 8,
            "mask_padding": 6,
            "shortage_policy": "fail",
            "seed": args.split_seed,
            "unavailable_image_policy": "fail",
            "displays": [
                {
                    "display_id": "brightfield",
                    "mode": "single",
                    "channel_ids": ["brightfield"],
                    "normalization_scope": "image",
                    "gamma": 1,
                },
                {
                    "display_id": "nuclei",
                    "mode": "single",
                    "channel_ids": ["nuclei"],
                    "normalization_scope": "image",
                    "gamma": 1,
                },
            ],
        },
        "classifier": {
            "feature_columns": primary_features,
            "missing_policy": "median_impute",
            "constant_policy": "drop_with_manifest",
            "group_column": "well",
            "allow_ungrouped": False,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": args.split_seed,
            "engine": "glmnet_multinomial",
            # A strict mixed L1/L2 penalty (rather than the lasso endpoint)
            # implements the requested grouped nested-CV elastic net.
            "alpha": 0.5,
            "lambda_rule": "lambda.1se",
            "eligible_review_status": ["confirmed", "corrected"],
            "minimum_confidence": 0,
        },
    }
    atomic_text(project_out, yaml_json(project))

    outputs = [
        classes_out,
        cells_out,
        features_out,
        images_out,
        project_out,
        split_out,
        condition_split_out,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project_id": args.project_id,
        "branch": args.branch,
        "field_manifest": str(field_manifest),
        "shadow_root": str(shadow_root),
        "selected_mask_column": selected_mask_column,
        "field_count": len(fields),
        "cell_count": total_cells,
        "well_count": len(splits),
        "development_well_count": sum(value == "development" for value in splits.values()),
        "heldout_well_count": sum(value == "heldout" for value in splits.values()),
        "split": split_metadata,
        "condition_split_row_count": len(condition_split_rows),
        "retained_numeric_feature_count": len(retained_features),
        "retained_numeric_features": retained_features,
        "primary_model_feature_count": len(primary_features),
        "primary_model_features": primary_features,
        "nuclei_comparator_features": configured["nuclei_comparator_feature_columns"],
        "feature_config": str(feature_config_path),
        "feature_config_sha256": sha256_file(feature_config_path),
        "group_split_invariant": (
            "one well belongs to exactly one of development or heldout; formal plate-map "
            "splits retain at least one development replicate for every treatment condition"
        ),
        "classifier_group_column": "well",
        "output_sha256": {str(path.relative_to(shadow_root)): sha256_file(path) for path in outputs},
    }
    atomic_text(
        audit_root / "adapter_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(f"cpa_project={project_out}")
    print(f"adapter_manifest={audit_root / 'adapter_manifest.json'}")
    print(
        f"adapter_complete=1 fields={len(fields)} cells={total_cells} "
        f"features={len(retained_features)} primary_features={len(primary_features)} "
        f"development_wells={manifest['development_well_count']} "
        f"heldout_wells={manifest['heldout_well_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
