#!/usr/bin/env python3
"""Extract one Brightfield broad-phenotype feature shard from a field record.

Only ``profiles.Brightfield.raw`` and the requested Combined mask branch are
primary inputs.  ``profiles.Nuclei`` is read only when the explicitly optional
nucleus comparator is enabled.  Dead profiles and current classification
outputs are outside this command's input contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile
from PIL import Image


HELPER_PATH = Path(__file__).resolve().parent / "_shared" / "broad_phenotype_features.py"
HELPER_SPEC = importlib.util.spec_from_file_location(
    "broad_phenotype_features_local",
    HELPER_PATH,
)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Cannot load broad-phenotype feature helpers from {HELPER_PATH}")
FEATURES = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = FEATURES
HELPER_SPEC.loader.exec_module(FEATURES)


KEY_RE = re.compile(r"^(?P<well>[A-H]\d+)_(?P<site>\d+)_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m$")
RECEIPT_SCHEMA_VERSION = "broad_phenotype_feature_receipt_v1"
BRANCH_TO_MASK_FIELD = {
    "original": "original_mask",
    "nucleated": "nucleated_mask",
}
IDENTITY_COLUMNS = (
    "cell_id",
    "key",
    "well",
    "site",
    "elapsed_hours",
    "branch",
    "image_id",
    "mask_label",
    "feature_schema_version",
    "feature_code_version",
    "bf_raw_sha256",
    "combined_mask_sha256",
    "nuclei_mask_sha256",
)
OUTPUT_COLUMNS = IDENTITY_COLUMNS + FEATURES.NUMERIC_FEATURE_COLUMNS
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "broad_phenotype_features_v1.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-record", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Shared extractor/model feature contract JSON.",
    )
    parser.add_argument(
        "--branch",
        choices=tuple(BRANCH_TO_MASK_FIELD),
        default="original",
        help="Combined mask object universe. Defaults to original.",
    )
    parser.add_argument(
        "--include-nuclei-comparator",
        action="store_true",
        help="Add overlap/count comparators from the Nuclei core mask (extent fallback).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace this field's feature shard and completion receipt.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def read_mask(path: Path, *, label: str) -> np.ndarray:
    array = read_image(path)
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(f"{label} must resolve to one 2-D plane, got shape {array.shape}")
    return FEATURES.validate_instance_mask(squeezed, label=label)


def resolve_record_path(value: Any, record_path: Path, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Field record is missing {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = record_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Field record {field} does not exist: {path}")
    return path


def parse_key(key: str) -> dict[str, Any]:
    match = KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid field-record key: {key!r}")
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    site = int(match.group("site"))
    if site < 1:
        raise ValueError(f"Invalid nonpositive site in field-record key: {key!r}")
    if hour >= 24 or minute >= 60:
        raise ValueError(f"Invalid clock component in field-record key: {key!r}")
    return {
        "well": match.group("well"),
        "site": site,
        "elapsed_hours": day * 24.0 + hour + minute / 60.0,
    }


def input_contract(
    field_record: Path,
    branch: str,
    include_nuclei_comparator: bool,
) -> dict[str, Any]:
    record_path = field_record.resolve()
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Field record must contain one JSON object: {record_path}")
    key = str(payload.get("key", ""))
    details = parse_key(key)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Field record is missing profiles object: {record_path}")
    brightfield = profiles.get("Brightfield")
    combined = profiles.get("Combined")
    if not isinstance(brightfield, dict):
        raise ValueError("Field record is missing profiles.Brightfield")
    if not isinstance(combined, dict):
        raise ValueError("Field record is missing profiles.Combined")
    mask_field = BRANCH_TO_MASK_FIELD[branch]
    bf_raw_path = resolve_record_path(
        brightfield.get("raw"),
        record_path,
        "profiles.Brightfield.raw",
    )
    combined_mask_path = resolve_record_path(
        combined.get(mask_field),
        record_path,
        f"profiles.Combined.{mask_field}",
    )
    nuclei_mask_path: Path | None = None
    nuclei_mask_field = ""
    if include_nuclei_comparator:
        nuclei = profiles.get("Nuclei")
        if not isinstance(nuclei, dict):
            raise ValueError(
                "--include-nuclei-comparator requires profiles.Nuclei in the field record"
            )
        for candidate in ("core_mask", "extent_mask"):
            if str(nuclei.get(candidate, "")).strip():
                nuclei_mask_field = candidate
                nuclei_mask_path = resolve_record_path(
                    nuclei[candidate],
                    record_path,
                    f"profiles.Nuclei.{candidate}",
                )
                break
        if nuclei_mask_path is None:
            raise ValueError(
                "--include-nuclei-comparator requires profiles.Nuclei.core_mask "
                "or profiles.Nuclei.extent_mask"
            )
    return {
        "field_record": record_path,
        "field_record_sha256": sha256_file(record_path),
        "key": key,
        **details,
        "branch": branch,
        "combined_mask_field": mask_field,
        "bf_raw_path": bf_raw_path,
        "bf_raw_sha256": sha256_file(bf_raw_path),
        "combined_mask_path": combined_mask_path,
        "combined_mask_sha256": sha256_file(combined_mask_path),
        "nuclei_mask_field": nuclei_mask_field,
        "nuclei_mask_path": nuclei_mask_path,
        "nuclei_mask_sha256": sha256_file(nuclei_mask_path)
        if nuclei_mask_path is not None
        else "",
    }


def output_paths(out_dir: Path, key: str, well: str, branch: str) -> tuple[Path, Path]:
    stem = f"{key}__{branch}"
    feature_path = (
        out_dir.resolve()
        / "shards"
        / well
        / f"{stem}_broad_phenotype_features.tsv"
    )
    receipt_path = out_dir.resolve() / "receipts" / well / f"{stem}.json"
    return feature_path, receipt_path


def _format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return format(numeric, ".17g") if math.isfinite(numeric) else ""
    if isinstance(value, (np.bool_, bool)):
        return "1" if bool(value) else "0"
    return str(value)


def write_tsv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _format_tsv_value(row[name]) for name in OUTPUT_COLUMNS})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def existing_generation_matches(
    feature_path: Path,
    receipt_path: Path,
    contract: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    if not feature_path.exists() and not receipt_path.exists():
        return False
    if not feature_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(
            "Incomplete broad-phenotype feature generation was preserved: "
            f"features={feature_path.exists()}, receipt={receipt_path.exists()}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "key": contract["key"],
        "branch": contract["branch"],
        "field_record_sha256": contract["field_record_sha256"],
        "bf_raw_sha256": contract["bf_raw_sha256"],
        "combined_mask_sha256": contract["combined_mask_sha256"],
        "nuclei_mask_sha256": contract["nuclei_mask_sha256"],
        "feature_config_sha256": config["sha256"],
        "feature_config_semantic_sha256": config["semantic_sha256"],
        "feature_schema_sha256": FEATURES.feature_schema_sha256(),
    }
    mismatches = {
        name: {"expected": value, "observed": receipt.get(name)}
        for name, value in expected.items()
        if receipt.get(name) != value
    }
    observed_output_hash = sha256_file(feature_path)
    if receipt.get("feature_tsv_sha256") != observed_output_hash:
        mismatches["feature_tsv_sha256"] = {
            "expected": receipt.get("feature_tsv_sha256"),
            "observed": observed_output_hash,
        }
    if Path(str(receipt.get("feature_tsv", ""))) != feature_path:
        mismatches["feature_tsv"] = {
            "expected": str(feature_path),
            "observed": receipt.get("feature_tsv"),
        }
    implementation = receipt.get("implementation")
    current_implementation = {
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "feature_module_sha256": sha256_file(HELPER_PATH.resolve()),
    }
    if not isinstance(implementation, dict):
        mismatches["implementation"] = {
            "expected": current_implementation,
            "observed": implementation,
        }
    else:
        for name, value in current_implementation.items():
            if implementation.get(name) != value:
                mismatches[f"implementation.{name}"] = {
                    "expected": value,
                    "observed": implementation.get(name),
                }
    if mismatches:
        raise RuntimeError(
            "Existing broad-phenotype generation does not match requested inputs: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return True


def enrich_rows(
    raw_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        mask_label = int(raw["mask_label"])
        identity = {
            "cell_id": f"{contract['branch']}|{contract['key']}|{mask_label}",
            "key": contract["key"],
            "well": contract["well"],
            "site": contract["site"],
            "elapsed_hours": contract["elapsed_hours"],
            "branch": contract["branch"],
            "image_id": contract["key"],
            "mask_label": mask_label,
            "feature_schema_version": FEATURES.FEATURE_SCHEMA_VERSION,
            "feature_code_version": FEATURES.FEATURE_CODE_VERSION,
            "bf_raw_sha256": contract["bf_raw_sha256"],
            "combined_mask_sha256": contract["combined_mask_sha256"],
            "nuclei_mask_sha256": contract["nuclei_mask_sha256"],
        }
        row = {**identity, **{name: raw[name] for name in FEATURES.NUMERIC_FEATURE_COLUMNS}}
        if tuple(row) != OUTPUT_COLUMNS:
            raise RuntimeError("Internal output column order does not match OUTPUT_COLUMNS")
        rows.append(row)
    cell_ids = [str(row["cell_id"]) for row in rows]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("Broad-phenotype feature rows contain duplicate cell_id values")
    return rows


def build_receipt(
    contract: dict[str, Any],
    config: dict[str, Any],
    feature_path: Path,
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    missing_by_feature = {
        name: sum(
            1
            for row in rows
            if isinstance(row[name], (float, np.floating))
            and not math.isfinite(float(row[name]))
        )
        for name in FEATURES.NUMERIC_FEATURE_COLUMNS
    }
    cell_ids = [str(row["cell_id"]) for row in rows]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "key": contract["key"],
        "well": contract["well"],
        "branch": contract["branch"],
        "include_nuclei_comparator": bool(contract["nuclei_mask_path"] is not None),
        "field_record": str(contract["field_record"]),
        "field_record_sha256": contract["field_record_sha256"],
        "brightfield_raw": str(contract["bf_raw_path"]),
        "bf_raw_sha256": contract["bf_raw_sha256"],
        "combined_mask": str(contract["combined_mask_path"]),
        "combined_mask_field": contract["combined_mask_field"],
        "combined_mask_sha256": contract["combined_mask_sha256"],
        "nuclei_mask": str(contract["nuclei_mask_path"])
        if contract["nuclei_mask_path"] is not None
        else "",
        "nuclei_mask_field": contract["nuclei_mask_field"],
        "nuclei_mask_sha256": contract["nuclei_mask_sha256"],
        "feature_schema_version": FEATURES.FEATURE_SCHEMA_VERSION,
        "feature_code_version": FEATURES.FEATURE_CODE_VERSION,
        "feature_schema_sha256": FEATURES.feature_schema_sha256(),
        "feature_config": str(config["path"]),
        "feature_config_sha256": config["sha256"],
        "feature_config_semantic_sha256": config["semantic_sha256"],
        "feature_columns": list(OUTPUT_COLUMNS),
        "numeric_feature_columns": list(FEATURES.NUMERIC_FEATURE_COLUMNS),
        "primary_model_feature_columns": list(config["primary_model_feature_columns"]),
        "nuclei_comparator_feature_columns": list(
            config["nuclei_comparator_feature_columns"]
        ),
        "feature_parameters": FEATURES.parameters_dict(config["parameters"]),
        "feature_tsv": str(feature_path),
        "feature_tsv_sha256": sha256_file(feature_path),
        "row_count": len(rows),
        "cell_id_sha256": hashlib.sha256("\n".join(cell_ids).encode("utf-8")).hexdigest(),
        "missing_value_counts": missing_by_feature,
        "diagnostics": diagnostics,
        "implementation": {
            "extractor": str(Path(__file__).resolve()),
            "extractor_sha256": sha256_file(Path(__file__).resolve()),
            "feature_module": str(HELPER_PATH.resolve()),
            "feature_module_sha256": sha256_file(HELPER_PATH.resolve()),
        },
        "forbidden_inputs_read": [],
        "input_contract": {
            "brightfield": "profiles.Brightfield.raw",
            "combined": f"profiles.Combined.{contract['combined_mask_field']}",
            "nuclei": (
                f"profiles.Nuclei.{contract['nuclei_mask_field']}"
                if contract["nuclei_mask_field"]
                else "disabled"
            ),
            "dead": "not_read",
            "current_classification": "not_read",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = FEATURES.load_feature_config(args.config)
    contract = input_contract(
        args.field_record,
        args.branch,
        args.include_nuclei_comparator,
    )
    feature_path, receipt_path = output_paths(
        args.out_dir,
        contract["key"],
        contract["well"],
        contract["branch"],
    )
    if not args.force and existing_generation_matches(
        feature_path,
        receipt_path,
        contract,
        config,
    ):
        print(
            "broad_phenotype_features_already_complete=1 "
            f"key={contract['key']} branch={contract['branch']} features={feature_path}"
        )
        return 0
    brightfield_raw = read_image(contract["bf_raw_path"])
    combined_mask = read_mask(contract["combined_mask_path"], label="Combined mask")
    nuclei_mask = (
        read_mask(contract["nuclei_mask_path"], label="Nuclei comparator mask")
        if contract["nuclei_mask_path"] is not None
        else None
    )
    raw_rows, diagnostics = FEATURES.extract_broad_phenotype_features(
        brightfield_raw,
        combined_mask,
        nuclei_mask=nuclei_mask,
        parameters=config["parameters"],
    )
    rows = enrich_rows(raw_rows, contract)
    write_tsv_atomic(feature_path, rows)
    receipt = build_receipt(contract, config, feature_path, rows, diagnostics)
    write_json_atomic(receipt_path, receipt)
    print(
        "broad_phenotype_features_complete=1 "
        f"key={contract['key']} branch={contract['branch']} rows={len(rows)} "
        f"features={feature_path} receipt={receipt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
