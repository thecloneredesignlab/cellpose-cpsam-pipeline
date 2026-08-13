#!/usr/bin/env python3
"""Merge immutable per-field V4 features onto one frozen development universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "multimodal_cell_state_v4_feature_merge_v1"
FIELD_SCHEMA_VERSION = "multimodal_cell_state_v4_feature_generation_v1"
CONFIG_SCHEMA_VERSION = "multimodal_cell_state_v4_config_v1"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--field-extractor", type=Path, required=True)
    parser.add_argument("--feature-helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def canonical_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(f"{label} is unavailable or a symlink: {value}")
    return value


def canonical_dir(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_dir() or value.is_symlink():
        raise FileNotFoundError(f"{label} is unavailable or a symlink: {value}")
    return value


def project_asset(project_path: Path, value: Any, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Project lacks {label}")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = project_path.parent / candidate
    return canonical_file(candidate, label)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(directory: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    output: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 feature merge contains a symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(directory))
            if relative not in excluded:
                output[relative] = sha256_file(path)
    return output


def verify_existing(output: Path, expected_identity: dict[str, Any]) -> None:
    manifest_path = output / "feature_merge_manifest.json"
    identity_path = output / "feature_merge_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing V4 feature merge is partial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("Existing V4 feature merge manifest is not COMPLETE")
    if manifest.get("identity") != expected_identity:
        raise ValueError("Existing V4 feature merge identity differs")
    observed = artifact_hashes(
        output,
        {"feature_merge_manifest.json", "feature_merge_generation_identity.json"},
    )
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 feature merge artifact set/hash differs")
    expected_generation = {
        "schema_version": "multimodal_cell_state_v4_feature_merge_generation_identity_v1",
        "status": "COMPLETE",
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "output_file_sha256": observed,
    }
    if json.loads(identity_path.read_text(encoding="utf-8")) != expected_generation:
        raise ValueError("Existing V4 feature merge generation identity differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_cell_count < 1:
        raise ValueError("--expected-cell-count must be positive")
    project_path = canonical_file(args.project, "project")
    feature_root = canonical_dir(args.feature_root, "feature root")
    config_path = canonical_file(args.feature_config, "V4 feature config")
    extractor = canonical_file(args.field_extractor, "V4 field extractor")
    helper = canonical_file(args.feature_helper, "V4 feature helper")
    implementation = canonical_file(Path(__file__), "V4 merge implementation")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported V4 feature config schema")
    cells_path = project_asset(project_path, project.get("cells_file"), "cells_file")
    cell_fields, cells = read_tsv(cells_path)
    required_cells = {"cell_id", "key", "well", "mask_label", "split"}
    if not required_cells.issubset(cell_fields):
        raise ValueError(
            f"V4 project cells lack fields: {sorted(required_cells-set(cell_fields))}"
        )
    if len(cells) != args.expected_cell_count:
        raise ValueError(f"V4 development cell count differs: {len(cells)}")
    if any(row["split"] != "development" for row in cells):
        raise ValueError("Heldout cells entered V4 feature merge")
    ordered_ids = [row["cell_id"] for row in cells]
    if any(not value for value in ordered_ids) or len(set(ordered_ids)) != len(
        ordered_ids
    ):
        raise ValueError("V4 project cell IDs are blank or duplicated")

    generation_dirs = sorted(
        {path.parent for path in feature_root.rglob("feature_receipt.json")}
    )
    if not generation_dirs:
        raise ValueError(f"No V4 per-field generations found under {feature_root}")
    by_id: dict[str, dict[str, str]] = {}
    field_receipts: list[dict[str, Any]] = []
    feature_fields: list[str] | None = None
    expected_config_sha = sha256_file(config_path)
    expected_extractor_sha = sha256_file(extractor)
    expected_helper_sha = sha256_file(helper)
    for generation in generation_dirs:
        if generation.is_symlink():
            raise ValueError(f"V4 field generation is a symlink: {generation}")
        receipt_path = generation / "feature_receipt.json"
        features_path = generation / "features.tsv"
        if not features_path.is_file() or features_path.is_symlink():
            raise ValueError(f"V4 field generation lacks features.tsv: {generation}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != FIELD_SCHEMA_VERSION
            or receipt.get("status") != "COMPLETE"
        ):
            raise ValueError(f"V4 field receipt is incomplete: {receipt_path}")
        inputs = receipt.get("inputs", {})
        expected_hashes = {
            "config": expected_config_sha,
            "implementation": expected_extractor_sha,
            "feature_helper": expected_helper_sha,
        }
        for name, expected_hash in expected_hashes.items():
            if inputs.get(name, {}).get("sha256") != expected_hash:
                raise ValueError(
                    f"V4 field receipt {name} identity differs: {receipt_path}"
                )
        if receipt.get("output_file_sha256") != {
            "features.tsv": sha256_file(features_path)
        }:
            raise ValueError(f"V4 field artifact hash differs: {features_path}")
        fields, rows = read_tsv(features_path)
        if feature_fields is None:
            feature_fields = fields
        elif fields != feature_fields:
            raise ValueError("V4 per-field feature headers differ")
        if len(rows) != int(receipt.get("row_count", -1)):
            raise ValueError(f"V4 field row count differs: {features_path}")
        for row in rows:
            cell_id = row.get("cell_id", "")
            if not cell_id or cell_id in by_id:
                raise ValueError(
                    f"V4 field features have blank/duplicate cell_id: {cell_id!r}"
                )
            by_id[cell_id] = row
        field_receipts.append(
            {
                "key": receipt.get("key"),
                "well": receipt.get("well"),
                "generation": str(generation),
                "receipt_sha256": sha256_file(receipt_path),
                "features_sha256": sha256_file(features_path),
                "row_count": len(rows),
            }
        )
    if set(by_id) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(by_id))[:10]
        unknown = sorted(set(by_id) - set(ordered_ids))[:10]
        raise ValueError(
            f"V4 field/project cell universe differs: missing={missing} unknown={unknown}"
        )
    assert feature_fields is not None
    merged = [by_id[cell_id] for cell_id in ordered_ids]
    for cell, row in zip(cells, merged, strict=True):
        if (
            row["key"] != cell["key"]
            or row["well"] != cell["well"]
            or row["mask_label"] != cell["mask_label"]
        ):
            raise ValueError(f"V4 merged identity differs for {cell['cell_id']}")
    inputs = {
        "project": {"path": str(project_path), "sha256": sha256_file(project_path)},
        "cells": {"path": str(cells_path), "sha256": sha256_file(cells_path)},
        "feature_root": str(feature_root),
        "feature_config": {"path": str(config_path), "sha256": expected_config_sha},
        "field_extractor": {"path": str(extractor), "sha256": expected_extractor_sha},
        "feature_helper": {"path": str(helper), "sha256": expected_helper_sha},
        "implementation": {
            "path": str(implementation),
            "sha256": sha256_file(implementation),
        },
        "ordered_field_receipts": field_receipts,
    }
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(
                f"V4 feature merge output is not a real directory: {output}"
            )
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs)
        print(f"v4_feature_merge={output}")
        print("generation_status=verified_reuse")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        write_tsv(staging / "features.tsv", feature_fields, merged)
        write_tsv(
            staging / "field_generation_manifest.tsv",
            [
                "key",
                "well",
                "generation",
                "receipt_sha256",
                "features_sha256",
                "row_count",
            ],
            [
                {
                    name: str(row[name])
                    for name in (
                        "key",
                        "well",
                        "generation",
                        "receipt_sha256",
                        "features_sha256",
                        "row_count",
                    )
                }
                for row in field_receipts
            ],
        )
        outputs = artifact_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v4",
            "identity": inputs,
            "row_count": len(merged),
            "field_count": len(field_receipts),
            "ordered_cell_id_sha256": hashlib.sha256(
                "\n".join(ordered_ids).encode()
            ).hexdigest(),
            "feature_columns": feature_fields,
            "heldout_read": False,
            "forbidden_inputs_read": [],
            "output_file_sha256": outputs,
        }
        manifest_path = staging / "feature_merge_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        identity = {
            "schema_version": "multimodal_cell_state_v4_feature_merge_generation_identity_v1",
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(manifest_path),
            "implementation_sha256": sha256_file(implementation),
            "output_file_sha256": outputs,
        }
        (staging / "feature_merge_generation_identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(
                f"V4 feature merge output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"v4_feature_merge={output}")
    print(f"row_count={len(merged)}")
    print(f"field_count={len(field_receipts)}")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
