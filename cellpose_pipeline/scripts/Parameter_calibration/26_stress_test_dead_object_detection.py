#!/usr/bin/env python3
"""Stress-test Dead-channel object detection without manual ground truth.

The test is deliberately independent of final cell-state decisions.  Strong
objects from the complete object ledger are checked against a robust raw-channel
seed detector.  Deterministic spatial shifts provide two controls:

* synthetic relocation checks whether the same strong signal remains detectable;
* channel shift measures how often unrelated seeds would land on RGB-live cells.

These controls are calibration evidence only; they never modify production masks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


OBJECT_SUFFIX = "_dead_object_features.csv"
FEATURE_SUFFIX = "_per_cell_fusion_features.csv"


TIMEPOINT_HELPER_PATH = Path(__file__).resolve().parents[1] / "_shared" / "timepoint_selection.py"
TIMEPOINT_HELPER_SPEC = importlib.util.spec_from_file_location(
    "timepoint_selection_local",
    TIMEPOINT_HELPER_PATH,
)
if TIMEPOINT_HELPER_SPEC is None or TIMEPOINT_HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load timepoint selection helpers: {TIMEPOINT_HELPER_PATH}")
TIMEPOINT_HELPER = importlib.util.module_from_spec(TIMEPOINT_HELPER_SPEC)
sys.modules[TIMEPOINT_HELPER_SPEC.name] = TIMEPOINT_HELPER
TIMEPOINT_HELPER_SPEC.loader.exec_module(TIMEPOINT_HELPER)
extract_key_and_timepoint = TIMEPOINT_HELPER.extract_key_and_timepoint
normalize_timepoint = TIMEPOINT_HELPER.normalize_timepoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-record-root", type=Path, required=True)
    parser.add_argument(
        "--branch",
        action="append",
        required=True,
        metavar="NAME=CLASSIFICATION_DIR",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-min-p90-delta", type=float, default=40.0)
    parser.add_argument("--reference-min-snr", type=float, default=100.0)
    parser.add_argument("--seed-min-delta", type=float, default=35.0)
    parser.add_argument("--seed-min-snr", type=float, default=90.0)
    parser.add_argument("--shift-y-fraction", type=float, default=0.37)
    parser.add_argument("--shift-x-fraction", type=float, default=0.29)
    parser.add_argument(
        "--timepoint",
        default="d0",
        help="Exact timepoint to select from complete field-record and classification roots (default: d0).",
    )
    return parser.parse_args()


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CLASSIFICATION_DIR, received: {value}")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not name or not path.is_dir():
        raise FileNotFoundError(path)
    return name, path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in {"", None} else 0


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def record_index(root: Path, timepoint: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*.json"):
        with path.open() as handle:
            record = json.load(handle)
        key = str(record.get("key", ""))
        if key and extract_key_and_timepoint(key)[1] == timepoint:
            output[key] = record
    return output


def robust_seed_map(raw: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, float, float]:
    values = raw.astype(np.float32, copy=False)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1.4826 * mad, 1e-6)
    threshold = median + max(args.seed_min_delta, args.seed_min_snr * sigma)
    return values >= threshold, median, sigma, threshold


def shifted_mask(mask: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.roll(mask, (shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        shifted[:shift_y, :] = False
    elif shift_y < 0:
        shifted[shift_y:, :] = False
    if shift_x > 0:
        shifted[:, :shift_x] = False
    elif shift_x < 0:
        shifted[:, shift_x:] = False
    return shifted


def branch_rows(
    branch_name: str,
    branch_dir: Path,
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    object_paths = [
        path
        for path in sorted((branch_dir / "dead_objects").glob(f"*{OBJECT_SUFFIX}"))
        if extract_key_and_timepoint(path)[1] == args.timepoint
    ]
    for object_path in object_paths:
        key, _observed_timepoint = extract_key_and_timepoint(object_path)
        object_rows = read_rows(object_path)
        record = records.get(key)
        if record is None:
            raise KeyError(f"Missing field record for {key}")
        profiles = record["profiles"]
        dead_raw = tifffile.imread(profiles["Dead"]["raw"]).astype(np.float32)
        if dead_raw.ndim == 3:
            dead_raw = np.mean(dead_raw[:, :, :3], axis=2)
        dead_mask = tifffile.imread(profiles["Dead"]["mask"])
        combined_key = "nucleated_mask" if "nucleated" in branch_name else "original_mask"
        combined_mask = tifffile.imread(profiles["Combined"][combined_key])
        feature_path = next((branch_dir / "features").glob(f"*{key}*{FEATURE_SUFFIX}"))
        feature_rows = read_rows(feature_path)
        live_labels = {
            as_int(row, "combined_mask_id")
            for row in feature_rows
            if row.get("rgb_state") == "live"
        }

        seeds, bg_median, bg_sigma, threshold = robust_seed_map(dead_raw, args)
        shift_y = max(1, int(round(dead_raw.shape[0] * args.shift_y_fraction)))
        shift_x = max(1, int(round(dead_raw.shape[1] * args.shift_x_fraction)))
        shifted_seeds = shifted_mask(seeds, shift_y, shift_x)
        shifted_live_hits = set(np.unique(combined_mask[shifted_seeds]).tolist()) & live_labels

        references = [
            row
            for row in object_rows
            if as_bool(row, "keep_signal")
            and as_float(row, "p90_delta") >= args.reference_min_p90_delta
            and as_float(row, "snr") >= args.reference_min_snr
        ]
        recovered = 0
        relocated_recovered = 0
        for row in references:
            label = as_int(row, "dead_mask_id")
            object_pixels = dead_mask == label
            recovered += bool(np.any(seeds & object_pixels))
            relocated = shifted_mask(object_pixels, shift_y, shift_x)
            relocated_recovered += bool(
                np.any(relocated)
                and as_float(row, "raw_max") >= threshold
            )
        output.append(
            {
                "branch": branch_name,
                "image_id": object_rows[0]["image_id"] if object_rows else key,
                "key": key,
                "raw_bg_median": bg_median,
                "raw_bg_sigma": bg_sigma,
                "seed_threshold": threshold,
                "seed_pixel_count": int(np.count_nonzero(seeds)),
                "strong_reference_count": len(references),
                "strong_reference_recovered": recovered,
                "strong_reference_missed": len(references) - recovered,
                "synthetic_relocation_recovered": relocated_recovered,
                "synthetic_relocation_missed": len(references) - relocated_recovered,
                "rgb_live_cell_count": len(live_labels),
                "shifted_seed_live_cell_hits": len(shifted_live_hits),
                "shifted_seed_live_cell_hit_rate": (
                    len(shifted_live_hits) / len(live_labels) if live_labels else 0.0
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    args.timepoint = normalize_timepoint(args.timepoint)
    records = record_index(args.field_record_root, args.timepoint)
    if not records:
        raise RuntimeError(
            f"No field records matched timepoint={args.timepoint} under {args.field_record_root}"
        )
    all_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for branch_value in args.branch:
        name, path = parse_branch(branch_value)
        rows = branch_rows(name, path, records, args)
        if not rows:
            raise RuntimeError(
                f"No classification rows matched timepoint={args.timepoint} for branch={name}"
            )
        all_rows.extend(rows)
        references = sum(int(row["strong_reference_count"]) for row in rows)
        recovered = sum(int(row["strong_reference_recovered"]) for row in rows)
        relocated = sum(int(row["synthetic_relocation_recovered"]) for row in rows)
        live_cells = sum(int(row["rgb_live_cell_count"]) for row in rows)
        shifted_hits = sum(int(row["shifted_seed_live_cell_hits"]) for row in rows)
        metrics.append(
            {
                "branch": name,
                "field_count": len(rows),
                "strong_reference_count": references,
                "strong_reference_recovered": recovered,
                "strong_reference_recall": recovered / references if references else 0.0,
                "synthetic_relocation_recovered": relocated,
                "synthetic_relocation_recall": relocated / references if references else 0.0,
                "rgb_live_cell_count": live_cells,
                "shifted_seed_live_cell_hits": shifted_hits,
                "shifted_seed_live_cell_hit_rate": shifted_hits / live_cells if live_cells else 0.0,
            }
        )
    write_rows(args.out_dir / "detection_stress_per_field.csv", all_rows, list(all_rows[0]))
    write_rows(args.out_dir / "detection_stress_metrics.csv", metrics, list(metrics[0]))
    for row in metrics:
        print(
            f"branch={row['branch']} raw_seed_recall={row['strong_reference_recall']:.6f} "
            f"synthetic_relocation_recall={row['synthetic_relocation_recall']:.6f} "
            f"shifted_seed_live_hit_rate={row['shifted_seed_live_cell_hit_rate']:.6f} "
            f"selected_timepoint={args.timepoint}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
