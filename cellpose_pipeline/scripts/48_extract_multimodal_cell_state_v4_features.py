#!/usr/bin/env python3
"""Build one immutable multimodal-cell-state V4 feature generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "_shared" / "multimodal_cell_state_v4_features.py"
SPEC = importlib.util.spec_from_file_location(
    "multimodal_cell_state_v4_features_local", HELPER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load V4 feature helper: {HELPER_PATH}")
FEATURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FEATURES
SPEC.loader.exec_module(FEATURES)

SCHEMA_VERSION = "multimodal_cell_state_v4_feature_generation_v1"
CONFIG_SCHEMA_VERSION = "multimodal_cell_state_v4_config_v1"
OUTPUT_IDENTITY_COLUMNS = (
    "cell_id",
    "key",
    "well",
    "source_id",
    "mask_label",
    "branch",
)
BROAD_IDENTITY_COLUMNS = ("cell_id", "key", "well", "mask_label", "branch")
OUTPUT_COLUMNS = (
    OUTPUT_IDENTITY_COLUMNS
    + FEATURES.OUTPUT_FEATURE_COLUMNS
    + (
        "nuclei_count",
        "nuclei_overlap_area_px2",
        "nuclei_measurement_status",
        "dead_measurement_status",
    )
)
BBOX_PADDING = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-record", type=Path, required=True)
    parser.add_argument("--broad-feature-tsv", type=Path, required=True)
    parser.add_argument("--selected-cells", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("original",), default="original")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Verify and reuse an identical complete generation",
    )
    return parser.parse_args(argv)


def resolve_file(value: Any, record: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Field record lacks {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = record.parent / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"Field record {label} is unavailable or a symlink: {path}"
        )
    return path


def load_first_tiff_page(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as image:
        if not image.pages:
            raise ValueError(f"TIFF contains no pages: {path}")
        return np.asarray(image.pages[0].asarray())


def load_contract(
    record_path: Path, broad_path: Path, selected_cells_path: Path, config_path: Path
) -> dict[str, Any]:
    record = record_path.resolve()
    broad = broad_path.resolve()
    selected_cells = selected_cells_path.resolve()
    config = config_path.resolve()
    for path in (
        record,
        broad,
        selected_cells,
        config,
        Path(__file__).resolve(),
        HELPER_PATH.resolve(),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"Required V4 input is unavailable or a symlink: {path}"
            )
    payload = json.loads(record.read_text(encoding="utf-8"))
    configuration = json.loads(config.read_text(encoding="utf-8"))
    if configuration.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported multimodal-cell-state V4 config")
    if configuration.get("method_version") != "sum159_multimodal_cell_state_v4":
        raise ValueError("V4 config method_version differs")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Field record lacks profiles")
    nuclei = profiles.get("Nuclei")
    dead = profiles.get("Dead")
    combined = profiles.get("Combined")
    if (
        not isinstance(nuclei, dict)
        or not isinstance(dead, dict)
        or not isinstance(combined, dict)
    ):
        raise ValueError("Field record lacks Nuclei, Dead, or Combined profile")
    mask_field = next(
        (
            name
            for name in ("extent_mask", "core_mask")
            if str(nuclei.get(name, "")).strip()
        ),
        None,
    )
    if mask_field is None:
        raise ValueError("Field record lacks Nuclei extent_mask and core_mask")
    key = str(payload.get("key", ""))
    if not key:
        raise ValueError("Field record lacks key")
    well = key.split("_", 1)[0]
    if not well:
        raise ValueError("Could not derive well from field key")
    paths = {
        "nuclei_raw": resolve_file(nuclei.get("raw"), record, "profiles.Nuclei.raw"),
        "dead_raw": resolve_file(dead.get("raw"), record, "profiles.Dead.raw"),
        "nuclei_mask": resolve_file(
            nuclei.get(mask_field), record, f"profiles.Nuclei.{mask_field}"
        ),
        "combined_mask": resolve_file(
            combined.get("original_mask"), record, "profiles.Combined.original_mask"
        ),
    }
    miss_gate = configuration["nuclei_zero_aware"]["possible_mask_miss_gate"]
    dead_policy = configuration["dead_zero_aware"]
    dead_gate = dead_policy["positive_signal_gate"]
    return {
        "record": record,
        "broad": broad,
        "selected_cells": selected_cells,
        "config": config,
        "key": key,
        "well": well,
        "mask_field": mask_field,
        "paths": paths,
        "nuclei_minimum_contrast": float(
            miss_gate["minimum_cell_to_ring_contrast_mad"]
        ),
        "nuclei_minimum_positive_fraction": float(
            miss_gate["minimum_positive_fraction"]
        ),
        "dead_minimum_contrast": float(
            dead_gate["minimum_cell_to_ring_contrast_mad"]
        ),
        "dead_minimum_positive_fraction": float(
            dead_gate["minimum_positive_fraction"]
        ),
        "dead_minimum_component_pixels": int(
            dead_gate["connected_component_minimum_pixels"]
        ),
        "dead_saturation_fraction_threshold": float(
            dead_policy["saturation_fraction_threshold"]
        ),
    }


def read_broad(path: Path, key: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        required = (
            set(BROAD_IDENTITY_COLUMNS)
            | set(FEATURES.SHAPE_COLUMNS)
            | {
                "bf_object_p10",
                "bf_object_p90",
                *(
                    name
                    for name in FEATURES.BRIGHTFIELD_COLUMNS
                    if name != "bf_object_p90_minus_p10"
                ),
            }
        )
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"Broad feature shard lacks V4 inputs: {missing}")
        rows = list(reader)
    if not rows or any(
        row["key"] != key or row["branch"] != "original" for row in rows
    ):
        raise ValueError(
            "Broad feature shard key/branch differs from V4 field contract"
        )
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("Broad feature shard has duplicate cell_id")
    return rows


def read_selected_cells(path: Path, key: str, well: str) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_id", "key", "well", "mask_label", "split"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Selected-cell table lacks V4 identity columns: {sorted(missing)}"
            )
        rows = [row for row in reader if row["key"] == key]
    if not rows:
        raise ValueError(
            f"Selected-cell table has no development cells for field {key}"
        )
    if any(row["well"] != well or row["split"] != "development" for row in rows):
        raise ValueError(
            f"Selected-cell table includes a non-development or wrong-well row for {key}"
        )
    selected: dict[str, int] = {}
    for row in rows:
        cell_id = row["cell_id"]
        label = int(row["mask_label"])
        if not cell_id or cell_id in selected or label < 1:
            raise ValueError(
                f"Selected-cell identity is blank, duplicate, or invalid for {key}: {cell_id!r}"
            )
        selected[cell_id] = label
    if len(set(selected.values())) != len(selected):
        raise ValueError(f"Selected-cell mask labels are duplicated for field {key}")
    return selected


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g") if math.isfinite(float(value)) else ""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def padded_bounds(
    bounds: tuple[slice, ...], shape: tuple[int, int]
) -> tuple[slice, slice]:
    if len(bounds) != 2:
        raise ValueError(f"Combined object bounds are not two-dimensional: {bounds}")
    return tuple(
        slice(max(0, part.start - BBOX_PADDING), min(limit, part.stop + BBOX_PADDING))
        for part, limit in zip(bounds, shape, strict=True)
    )  # type: ignore[return-value]


def artifact_hashes(directory: Path) -> dict[str, str]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("V4 feature generation contains a symlink")
    return {str(path.relative_to(directory)): sha256_file(path) for path in files}


def expected_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_record": {
            "path": str(contract["record"]),
            "sha256": sha256_file(contract["record"]),
        },
        "broad_feature_tsv": {
            "path": str(contract["broad"]),
            "sha256": sha256_file(contract["broad"]),
        },
        "selected_cells": {
            "path": str(contract["selected_cells"]),
            "sha256": sha256_file(contract["selected_cells"]),
        },
        "config": {
            "path": str(contract["config"]),
            "sha256": sha256_file(contract["config"]),
        },
        **{
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in contract["paths"].items()
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "feature_helper": {
            "path": str(HELPER_PATH.resolve()),
            "sha256": sha256_file(HELPER_PATH.resolve()),
        },
    }


def verify_existing(output: Path, inputs: dict[str, Any]) -> None:
    receipt_path = output / "feature_receipt.json"
    feature_path = output / "features.tsv"
    if (
        not receipt_path.is_file()
        or not feature_path.is_file()
        or receipt_path.is_symlink()
        or feature_path.is_symlink()
    ):
        raise ValueError("Existing V4 feature generation is partial")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "COMPLETE"
    ):
        raise ValueError("Existing V4 feature receipt is not COMPLETE")
    if receipt.get("inputs") != inputs:
        raise ValueError("Existing V4 feature input identity differs")
    declared = receipt.get("output_file_sha256")
    observed = {"features.tsv": sha256_file(feature_path)}
    if declared != observed:
        raise ValueError("Existing V4 feature artifact hash differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract(
        args.field_record, args.broad_feature_tsv, args.selected_cells, args.config
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"V4 feature output is not a real directory: {output}")
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, expected_inputs(contract))
        print(f"v4_feature_generation={output}")
        print("generation_status=verified_reuse")
        return 0

    broad_rows = read_broad(contract["broad"], contract["key"])
    selected = read_selected_cells(
        contract["selected_cells"], contract["key"], contract["well"]
    )
    broad_by_id = {row["cell_id"]: row for row in broad_rows}
    missing_selected = set(selected) - set(broad_by_id)
    if missing_selected:
        raise ValueError(
            f"Selected cells are absent from broad feature shard: {sorted(missing_selected)[:10]}"
        )
    broad_rows = [broad_by_id[cell_id] for cell_id in selected]
    for broad in broad_rows:
        if int(broad["mask_label"]) != selected[broad["cell_id"]]:
            raise ValueError(
                f"Selected-cell mask label differs from broad shard: {broad['cell_id']}"
            )
    nuclei_raw_source = load_first_tiff_page(contract["paths"]["nuclei_raw"])
    nuclei_raw = FEATURES.scalar_first_plane(nuclei_raw_source, label="Nuclei raw")
    dead_raw_source = load_first_tiff_page(contract["paths"]["dead_raw"])
    dead_raw = FEATURES.scalar_first_plane(dead_raw_source, label="Dead raw")
    dead_saturation_value = (
        float(np.iinfo(dead_raw_source.dtype).max)
        if np.issubdtype(dead_raw_source.dtype, np.integer)
        else None
    )
    nuclei_mask = FEATURES.validate_labels(
        tifffile.imread(contract["paths"]["nuclei_mask"]), label="Nuclei mask"
    )
    combined_mask = FEATURES.validate_labels(
        tifffile.imread(contract["paths"]["combined_mask"]), label="Combined mask"
    )
    if (
        nuclei_raw.shape != nuclei_mask.shape
        or dead_raw.shape != nuclei_mask.shape
        or nuclei_raw.shape != combined_mask.shape
    ):
        raise ValueError(
            "V4 field shapes differ: "
            f"nuclei_raw={nuclei_raw.shape} dead_raw={dead_raw.shape} "
            f"nuclei_mask={nuclei_mask.shape} combined={combined_mask.shape}"
        )

    object_slices = ndimage.find_objects(
        combined_mask, max_label=int(max(selected.values()))
    )
    rows: list[dict[str, Any]] = []
    for broad in broad_rows:
        label = int(broad["mask_label"])
        bounds = object_slices[label - 1] if label <= len(object_slices) else None
        if bounds is None:
            raise ValueError(f"Combined mask lacks broad-feature label {label}")
        bounds = padded_bounds(bounds, combined_mask.shape)
        combined_crop = combined_mask[bounds]
        obj = combined_crop == label
        if not np.any(obj):
            raise ValueError(f"Combined mask lacks broad-feature label {label}")
        row: dict[str, Any] = {name: broad[name] for name in BROAD_IDENTITY_COLUMNS}
        row["source_id"] = broad["well"]
        for name in FEATURES.SHAPE_COLUMNS:
            value = float(broad[name])
            if not math.isfinite(value):
                raise ValueError(f"Shape feature is nonfinite: {name}")
            row[name] = value
        row.update(FEATURES.derive_brightfield_features(broad))
        row.update(
            FEATURES.nucleus_features_for_object(
                obj,
                nuclei_mask[bounds],
                nuclei_raw[bounds],
                minimum_contrast=contract["nuclei_minimum_contrast"],
                minimum_positive_fraction=contract[
                    "nuclei_minimum_positive_fraction"
                ],
            )
        )
        row.update(
            FEATURES.dead_features_for_object(
                obj,
                dead_raw[bounds],
                minimum_contrast=contract["dead_minimum_contrast"],
                minimum_positive_fraction=contract[
                    "dead_minimum_positive_fraction"
                ],
                minimum_component_pixels=contract[
                    "dead_minimum_component_pixels"
                ],
                saturation_fraction_threshold=contract[
                    "dead_saturation_fraction_threshold"
                ],
                saturation_value=dead_saturation_value,
            )
        )
        if set(row) != set(OUTPUT_COLUMNS):
            raise RuntimeError(
                f"Internal V4 output schema differs: missing={sorted(set(OUTPUT_COLUMNS)-set(row))}"
            )
        rows.append(row)

    inputs = expected_inputs(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        feature_path = staging / "features.tsv"
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                {name: format_value(row[name]) for name in OUTPUT_COLUMNS}
                for row in rows
            )
        nuclei_statuses: dict[str, int] = {
            name: 0 for name in FEATURES.NUCLEI_MEASUREMENT_STATUSES
        }
        dead_statuses: dict[str, int] = {
            name: 0 for name in FEATURES.DEAD_MEASUREMENT_STATUSES
        }
        for row in rows:
            nuclei_statuses[str(row["nuclei_measurement_status"])] += 1
            dead_statuses[str(row["dead_measurement_status"])] += 1
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v4",
            "feature_schema_version": FEATURES.FEATURE_SCHEMA_VERSION,
            "key": contract["key"],
            "well": contract["well"],
            "branch": "original",
            "nuclei_mask_field": contract["mask_field"],
            "row_count": len(rows),
            "ordered_cell_id_sha256": hashlib.sha256(
                "\n".join(row["cell_id"] for row in rows).encode()
            ).hexdigest(),
            "feature_columns": list(OUTPUT_COLUMNS),
            "nuclei_measurement_status_counts": nuclei_statuses,
            "dead_measurement_status_counts": dead_statuses,
            "inputs": inputs,
            "forbidden_inputs_read": [],
            "input_contract": {
                "brightfield_features": "frozen_broad_feature_shard",
                "nuclei_raw": "profiles.Nuclei.raw_first_tiff_page",
                "nuclei_mask": f"profiles.Nuclei.{contract['mask_field']}",
                "dead_raw": "profiles.Dead.raw_first_tiff_page",
                "combined_mask": "profiles.Combined.original_mask",
                "existing_dead_segmentation": "not_read",
                "current_classification": "not_read",
                "trajectory": "not_read",
            },
            "output_file_sha256": {"features.tsv": sha256_file(feature_path)},
        }
        (staging / "feature_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(
                f"V4 feature output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
    print(f"v4_feature_generation={output}")
    print(f"row_count={len(rows)}")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
