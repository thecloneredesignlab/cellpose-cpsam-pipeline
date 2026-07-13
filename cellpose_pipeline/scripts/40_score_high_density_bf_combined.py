#!/usr/bin/env python3
"""Score high-density BF/Combined candidates without manual ground truth.

The score deliberately avoids enforcing one nucleus per cell.  It combines
intensity-supported nucleus-core containment, cell-shape artifact rates,
BF/Combined foreground and boundary agreement, nucleus-multiplicity agreement,
and calibration/validation stability.  The production pair is retained unless
a candidate clears conservative baseline-relative gates and improves the joint
loss by a configurable margin.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import tifffile
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")

TUNER_PATH = Path(__file__).with_name("39_tune_high_density_bf_combined.py")
TUNER_SPEC = importlib.util.spec_from_file_location("high_density_tuner_local", TUNER_PATH)
if TUNER_SPEC is None or TUNER_SPEC.loader is None:
    raise ImportError(f"Cannot load inference utilities from {TUNER_PATH}")
TUNER_MODULE = importlib.util.module_from_spec(TUNER_SPEC)
sys.modules[TUNER_SPEC.name] = TUNER_MODULE
TUNER_SPEC.loader.exec_module(TUNER_MODULE)
CandidateConfig = TUNER_MODULE.CandidateConfig
preprocess_image = TUNER_MODULE.preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly score high-density Brightfield and Combined candidates."
    )
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--core-support-threshold", type=float, default=0.80)
    parser.add_argument("--core-unassigned-threshold", type=float, default=0.50)
    parser.add_argument("--core-secondary-conflict-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    return parser.parse_args()


def extract_key(path_or_name: Path | str) -> str:
    match = KEY_RE.search(Path(path_or_name).name)
    if match is None:
        raise ValueError(f"Cannot extract key from {path_or_name}")
    return match.group(1)


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected 2D integer mask at {path}, got {labels.shape} {labels.dtype}")
    return labels.astype(np.int32, copy=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def label_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def safe_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float("nan")


def cell_shape_metrics(labels: np.ndarray, boundary: np.ndarray) -> dict[str, float | int]:
    maximum = int(labels.max()) if labels.size else 0
    if maximum <= 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
            "small_fraction_lt60": 0.0,
            "tiny_fraction_lt25": 0.0,
            "large_fraction_gt2500": 0.0,
            "huge_fraction_gt5000": 0.0,
            "circularity_median": 0.0,
            "irregular_fraction_circularity_lt0p15": 0.0,
            "axis_ratio_median": 0.0,
            "elongated_fraction_axis_ratio_gt4": 0.0,
            "bbox_fill_median": 0.0,
        }
    flat = labels.ravel().astype(np.int64, copy=False)
    areas = np.bincount(flat, minlength=maximum + 1).astype(np.float64)
    valid = np.flatnonzero(areas > 0)
    valid = valid[valid > 0]
    if not valid.size:
        raise ValueError("Positive maximum label but no foreground labels")

    boundary_labels = labels[boundary & (labels > 0)].astype(np.int64, copy=False)
    perimeters = np.bincount(boundary_labels, minlength=maximum + 1).astype(np.float64)
    circularity = 4.0 * math.pi * areas[valid] / np.maximum(perimeters[valid] ** 2, 1.0)
    circularity = np.clip(circularity, 0.0, 1.0)

    yy, xx = np.nonzero(labels)
    object_ids = labels[yy, xx].astype(np.int64, copy=False)
    sum_y = np.bincount(object_ids, weights=yy, minlength=maximum + 1)
    sum_x = np.bincount(object_ids, weights=xx, minlength=maximum + 1)
    sum_yy = np.bincount(object_ids, weights=yy.astype(np.float64) ** 2, minlength=maximum + 1)
    sum_xx = np.bincount(object_ids, weights=xx.astype(np.float64) ** 2, minlength=maximum + 1)
    sum_xy = np.bincount(
        object_ids,
        weights=yy.astype(np.float64) * xx.astype(np.float64),
        minlength=maximum + 1,
    )
    mean_y = sum_y[valid] / areas[valid]
    mean_x = sum_x[valid] / areas[valid]
    var_y = np.maximum(sum_yy[valid] / areas[valid] - mean_y**2, 0.0)
    var_x = np.maximum(sum_xx[valid] / areas[valid] - mean_x**2, 0.0)
    cov_xy = sum_xy[valid] / areas[valid] - mean_y * mean_x
    root = np.sqrt(np.maximum((var_y - var_x) ** 2 + 4.0 * cov_xy**2, 0.0))
    major = np.maximum((var_y + var_x + root) / 2.0, 1e-3)
    minor = np.maximum((var_y + var_x - root) / 2.0, 1e-3)
    axis_ratio = np.sqrt(major / minor)

    bbox_fill: list[float] = []
    objects = ndimage.find_objects(labels)
    for object_id in valid:
        obj_slice = objects[int(object_id) - 1]
        if obj_slice is None:
            continue
        bbox_area = int(obj_slice[0].stop - obj_slice[0].start) * int(
            obj_slice[1].stop - obj_slice[1].start
        )
        bbox_fill.append(float(areas[object_id] / max(bbox_area, 1)))

    object_areas = areas[valid]
    return {
        "n_objects": int(valid.size),
        "mask_fraction": float(np.mean(labels > 0)),
        "area_median": float(np.median(object_areas)),
        "area_p10": float(np.percentile(object_areas, 10)),
        "area_p90": float(np.percentile(object_areas, 90)),
        "small_fraction_lt60": float(np.mean(object_areas < 60)),
        "tiny_fraction_lt25": float(np.mean(object_areas < 25)),
        "large_fraction_gt2500": float(np.mean(object_areas > 2500)),
        "huge_fraction_gt5000": float(np.mean(object_areas > 5000)),
        "circularity_median": float(np.median(circularity)),
        "irregular_fraction_circularity_lt0p15": float(np.mean(circularity < 0.15)),
        "axis_ratio_median": float(np.median(axis_ratio)),
        "elongated_fraction_axis_ratio_gt4": float(np.mean(axis_ratio > 4.0)),
        "bbox_fill_median": safe_median(bbox_fill),
    }


def nucleus_core_assignment(
    core_labels: np.ndarray,
    cell_labels: np.ndarray,
    support_threshold: float,
    unassigned_threshold: float,
    secondary_conflict_threshold: float,
) -> dict[str, Any]:
    if core_labels.shape != cell_labels.shape:
        raise ValueError(f"Core/cell shape mismatch: {core_labels.shape} versus {cell_labels.shape}")
    core_max = int(core_labels.max()) if core_labels.size else 0
    cell_max = int(cell_labels.max()) if cell_labels.size else 0
    core_areas = np.bincount(core_labels.ravel().astype(np.int64), minlength=core_max + 1)
    valid_cores = np.flatnonzero(core_areas > 0)
    valid_cores = valid_cores[valid_cores > 0]
    if not valid_cores.size:
        raise ValueError("Nucleus core mask contains no objects")

    primary = np.zeros(core_max + 1, dtype=np.int32)
    primary_fraction = np.zeros(core_max + 1, dtype=np.float64)
    secondary_fraction = np.zeros(core_max + 1, dtype=np.float64)
    foreground_fraction = np.zeros(core_max + 1, dtype=np.float64)

    pixels = core_labels > 0
    base = cell_max + 1
    codes = (
        core_labels[pixels].astype(np.int64, copy=False) * base
        + cell_labels[pixels].astype(np.int64, copy=False)
    )
    unique_codes, counts = np.unique(codes, return_counts=True)
    core_ids = unique_codes // base
    cell_ids = unique_codes % base
    current_core = -1
    foreground_counts: list[tuple[int, int]] = []

    def finalize(core_id: int, pairs: list[tuple[int, int]]) -> None:
        if core_id <= 0:
            return
        total = float(core_areas[core_id])
        positive = sorted(
            ((cell_id, count) for cell_id, count in pairs if cell_id > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        positive_total = sum(count for _, count in positive)
        foreground_fraction[core_id] = positive_total / max(total, 1.0)
        if positive:
            primary[core_id] = int(positive[0][0])
            primary_fraction[core_id] = float(positive[0][1] / max(total, 1.0))
        if len(positive) > 1:
            secondary_fraction[core_id] = float(positive[1][1] / max(total, 1.0))

    for core_id_value, cell_id_value, count_value in zip(core_ids, cell_ids, counts):
        core_id = int(core_id_value)
        if core_id != current_core:
            finalize(current_core, foreground_counts)
            current_core = core_id
            foreground_counts = []
        foreground_counts.append((int(cell_id_value), int(count_value)))
    finalize(current_core, foreground_counts)

    yy, xx = np.nonzero(core_labels)
    ids = core_labels[yy, xx].astype(np.int64, copy=False)
    sum_y = np.bincount(ids, weights=yy, minlength=core_max + 1)
    sum_x = np.bincount(ids, weights=xx, minlength=core_max + 1)
    cy = np.rint(sum_y[valid_cores] / core_areas[valid_cores]).astype(np.int64)
    cx = np.rint(sum_x[valid_cores] / core_areas[valid_cores]).astype(np.int64)
    cy = np.clip(cy, 0, cell_labels.shape[0] - 1)
    cx = np.clip(cx, 0, cell_labels.shape[1] - 1)
    centroid_cell = np.zeros(core_max + 1, dtype=np.int32)
    centroid_cell[valid_cores] = cell_labels[cy, cx]

    supported = np.zeros(core_max + 1, dtype=bool)
    supported[valid_cores] = (
        (primary[valid_cores] > 0)
        & (primary_fraction[valid_cores] >= support_threshold)
        & (centroid_cell[valid_cores] == primary[valid_cores])
    )
    unassigned = np.zeros(core_max + 1, dtype=bool)
    unassigned[valid_cores] = (
        (primary[valid_cores] == 0)
        | (foreground_fraction[valid_cores] < unassigned_threshold)
    )
    conflict = np.zeros(core_max + 1, dtype=bool)
    conflict[valid_cores] = (
        (secondary_fraction[valid_cores] >= secondary_conflict_threshold)
        | (
            (foreground_fraction[valid_cores] >= unassigned_threshold)
            & (primary_fraction[valid_cores] < support_threshold)
        )
        | ((primary[valid_cores] > 0) & (centroid_cell[valid_cores] != primary[valid_cores]))
    )

    counts_per_cell = np.bincount(
        primary[valid_cores][supported[valid_cores]].astype(np.int64, copy=False),
        minlength=cell_max + 1,
    )
    cell_areas = np.bincount(cell_labels.ravel().astype(np.int64), minlength=cell_max + 1)
    valid_cells = np.flatnonzero(cell_areas > 0)
    valid_cells = valid_cells[valid_cells > 0]
    nucleated = counts_per_cell[valid_cells] > 0 if valid_cells.size else np.zeros(0, dtype=bool)
    multiplicity = np.zeros(core_max + 1, dtype=np.int32)
    selected_primary = primary[valid_cores]
    positive_primary = selected_primary > 0
    multiplicity[valid_cores[positive_primary]] = counts_per_cell[selected_primary[positive_primary]]
    nucleated_counts = counts_per_cell[valid_cells][nucleated] if valid_cells.size else np.zeros(0)

    return {
        "valid_core_ids": valid_cores,
        "primary_cell": primary,
        "primary_fraction": primary_fraction,
        "foreground_fraction": foreground_fraction,
        "secondary_fraction": secondary_fraction,
        "centroid_cell": centroid_cell,
        "supported": supported,
        "unassigned": unassigned,
        "conflict": conflict,
        "multiplicity": multiplicity,
        "metrics": {
            "nucleus_core_count": int(valid_cores.size),
            "core_supported_fraction": float(np.mean(supported[valid_cores])),
            "core_unassigned_fraction": float(np.mean(unassigned[valid_cores])),
            "core_boundary_conflict_fraction": float(np.mean(conflict[valid_cores])),
            "core_primary_fraction_mean": float(np.mean(primary_fraction[valid_cores])),
            "core_foreground_fraction_mean": float(np.mean(foreground_fraction[valid_cores])),
            "empty_cell_fraction": float(np.mean(~nucleated)) if valid_cells.size else 0.0,
            "multi_nucleus_cell_fraction": (
                float(np.mean(nucleated_counts > 1)) if nucleated_counts.size else 0.0
            ),
            "mean_nuclei_per_nucleated_cell": (
                float(np.mean(nucleated_counts)) if nucleated_counts.size else 0.0
            ),
            "cell_count_to_nucleus_count": float(valid_cells.size / valid_cores.size),
        },
    }


def gradient_enrichment(prepared: np.ndarray, boundary: np.ndarray) -> float:
    image = prepared.astype(np.float32, copy=False)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    if not np.any(boundary):
        return 0.0
    global_mean = float(np.mean(gradient))
    return float(np.mean(gradient[boundary]) / max(global_mean, 1e-6))


def boundary_agreement(
    first: np.ndarray,
    second: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    tolerance: float,
) -> tuple[float, float, float]:
    first_count = int(np.count_nonzero(first))
    second_count = int(np.count_nonzero(second))
    if not first_count or not second_count:
        return 0.0, 0.0, 0.0
    precision = float(np.mean(second_distance[first] <= tolerance))
    recall = float(np.mean(first_distance[second] <= tolerance))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def pair_field_metrics(
    bf: dict[str, Any],
    combined: dict[str, Any],
    tolerance: float,
) -> dict[str, float]:
    bf_mask = bf["mask"]
    combined_mask = combined["mask"]
    foreground_intersection = np.count_nonzero((bf_mask > 0) & (combined_mask > 0))
    foreground_total = np.count_nonzero(bf_mask > 0) + np.count_nonzero(combined_mask > 0)
    foreground_dice = float(2.0 * foreground_intersection / max(foreground_total, 1))
    boundary_precision, boundary_recall, boundary_f1 = boundary_agreement(
        bf["boundary"],
        combined["boundary"],
        bf["boundary_distance"],
        combined["boundary_distance"],
        tolerance,
    )

    bf_assignment = bf["assignment"]
    combined_assignment = combined["assignment"]
    valid = bf_assignment["valid_core_ids"]
    if not np.array_equal(valid, combined_assignment["valid_core_ids"]):
        raise ValueError("Brightfield and Combined core ID sets differ")
    both_supported = (
        bf_assignment["supported"][valid]
        & combined_assignment["supported"][valid]
    )
    multiplicity_a = bf_assignment["multiplicity"][valid]
    multiplicity_b = combined_assignment["multiplicity"][valid]
    if np.any(both_supported):
        multiplicity_agreement = float(
            np.mean(multiplicity_a[both_supported] == multiplicity_b[both_supported])
        )
        multiplicity_log_gap = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(multiplicity_a[both_supported], 1)
                        / np.maximum(multiplicity_b[both_supported], 1)
                    )
                )
            )
        )
    else:
        multiplicity_agreement = 0.0
        multiplicity_log_gap = float("inf")

    bf_metrics = bf["metrics"]
    combined_metrics = combined["metrics"]
    return {
        "foreground_dice": foreground_dice,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "both_core_supported_fraction": float(np.mean(both_supported)),
        "core_multiplicity_agreement_fraction": multiplicity_agreement,
        "core_multiplicity_log_gap": multiplicity_log_gap,
        "mask_fraction_gap": abs(
            float(bf_metrics["mask_fraction"]) - float(combined_metrics["mask_fraction"])
        ),
        "cell_count_log_ratio": abs(
            math.log(
                max(float(bf_metrics["n_objects"]), 1.0)
                / max(float(combined_metrics["n_objects"]), 1.0)
            )
        ),
        "mean_boundary_gradient_enrichment": (
            float(bf_metrics["boundary_gradient_enrichment"])
            + float(combined_metrics["boundary_gradient_enrichment"])
        )
        / 2.0,
        "bf_core_boundary_conflict_fraction": float(
            bf_metrics["core_boundary_conflict_fraction"]
        ),
        "combined_core_boundary_conflict_fraction": float(
            combined_metrics["core_boundary_conflict_fraction"]
        ),
        "bf_small_fraction_lt60": float(bf_metrics["small_fraction_lt60"]),
        "combined_small_fraction_lt60": float(combined_metrics["small_fraction_lt60"]),
        "bf_large_fraction_gt2500": float(bf_metrics["large_fraction_gt2500"]),
        "combined_large_fraction_gt2500": float(
            combined_metrics["large_fraction_gt2500"]
        ),
        "bf_empty_cell_fraction": float(bf_metrics["empty_cell_fraction"]),
        "combined_empty_cell_fraction": float(combined_metrics["empty_cell_fraction"]),
    }


PAIR_METRICS = (
    "foreground_dice",
    "boundary_f1",
    "both_core_supported_fraction",
    "core_multiplicity_agreement_fraction",
    "core_multiplicity_log_gap",
    "mask_fraction_gap",
    "cell_count_log_ratio",
    "mean_boundary_gradient_enrichment",
    "bf_core_boundary_conflict_fraction",
    "combined_core_boundary_conflict_fraction",
    "bf_small_fraction_lt60",
    "combined_small_fraction_lt60",
    "bf_large_fraction_gt2500",
    "combined_large_fraction_gt2500",
    "bf_empty_cell_fraction",
    "combined_empty_cell_fraction",
)


def aggregate_pair_rows(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    if not selected:
        return {f"{split}_{metric}": float("nan") for metric in PAIR_METRICS}
    result: dict[str, float] = {}
    for metric in PAIR_METRICS:
        result[f"{split}_{metric}"] = safe_mean(float(row[metric]) for row in selected)
    result[f"{split}_worst_core_uncovered_fraction"] = quantile(
        (1.0 - float(row["both_core_supported_fraction"]) for row in selected), 0.90
    )
    return result


def base_loss(row: dict[str, Any], prefix: str) -> float:
    both = float(row[f"{prefix}_both_core_supported_fraction"])
    boundary = float(row[f"{prefix}_boundary_f1"])
    foreground = float(row[f"{prefix}_foreground_dice"])
    multiplicity = float(row[f"{prefix}_core_multiplicity_agreement_fraction"])
    bf_conflict = float(row[f"{prefix}_bf_core_boundary_conflict_fraction"])
    combined_conflict = float(row[f"{prefix}_combined_core_boundary_conflict_fraction"])
    small = float(row[f"{prefix}_bf_small_fraction_lt60"]) + float(
        row[f"{prefix}_combined_small_fraction_lt60"]
    )
    large = float(row[f"{prefix}_bf_large_fraction_gt2500"]) + float(
        row[f"{prefix}_combined_large_fraction_gt2500"]
    )
    count_gap = float(row[f"{prefix}_cell_count_log_ratio"])
    return (
        2.5 * (1.0 - both)
        + 1.2 * (1.0 - boundary)
        + 0.4 * (1.0 - foreground)
        + 0.8 * (1.0 - multiplicity)
        + 0.35 * (bf_conflict + combined_conflict)
        + 0.20 * small
        + 0.20 * large
        + 0.15 * count_gap
    )


def pareto_front(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    objectives = (
        ("all_both_core_supported_fraction", -1.0),
        ("all_boundary_f1", -1.0),
        ("all_core_multiplicity_agreement_fraction", -1.0),
        ("all_bf_core_boundary_conflict_fraction", 1.0),
        ("all_combined_core_boundary_conflict_fraction", 1.0),
        ("artifact_rate", 1.0),
        ("generalization_gap", 1.0),
    )
    eligible = [row for row in rows if row["hard_gate_pass"]]
    front: set[tuple[str, str]] = set()
    for row in eligible:
        values = [float(row[name]) * direction for name, direction in objectives]
        dominated = False
        for other in eligible:
            if other is row:
                continue
            other_values = [float(other[name]) * direction for name, direction in objectives]
            if all(a <= b + 1e-12 for a, b in zip(other_values, values)) and any(
                a < b - 1e-12 for a, b in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            front.add((str(row["bf_config"]), str(row["combined_config"])))
    return front


def main() -> int:
    args = parse_args()
    args.screen_root = args.screen_root.resolve()
    out_dir = (args.out_dir or args.screen_root / "scoring").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.screen_root / "screen_manifest.json"
    inference_path = args.screen_root / "inference_summary.csv"
    if not manifest_path.is_file() or not inference_path.is_file():
        raise SystemExit(f"Incomplete screen root: {args.screen_root}")
    manifest = json.loads(manifest_path.read_text())
    configs = [CandidateConfig(**row) for row in manifest["selected_configs"]]
    config_by_tag = {config.tag: config for config in configs}
    profile_tags = {
        profile: [config.tag for config in configs if config.profile == profile]
        for profile in ("Brightfield", "Combined")
    }
    for profile, tags in profile_tags.items():
        if not tags:
            raise SystemExit(f"No {profile} candidates in screen manifest")
    baseline_tags = {
        profile: next(
            (config.tag for config in configs if config.profile == profile and config.baseline),
            None,
        )
        for profile in ("Brightfield", "Combined")
    }
    if any(value is None for value in baseline_tags.values()):
        raise SystemExit(f"Both profiles require a baseline: {baseline_tags}")

    inference_rows = csv_rows(inference_path)
    row_index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in inference_rows:
        index_key = (row["profile"], row["config"], row["key"])
        if index_key in row_index:
            raise ValueError(f"Duplicate inference row: {index_key}")
        row_index[index_key] = row

    keys = list(manifest["selected_keys"])
    calibration_keys = set(manifest["calibration_keys"])
    nucleus_run_root = Path(manifest["nuclei_run_root"])
    core_dir_candidates = (
        nucleus_run_root / "Nuclei" / "nucleus_core_seeds",
        nucleus_run_root / "nucleus_core_seeds" / "Nuclei",
        nucleus_run_root / "nucleus_core_seeds",
    )
    core_dir = next((path for path in core_dir_candidates if path.is_dir()), None)
    if core_dir is None:
        raise FileNotFoundError(f"Cannot find nucleus cores under {nucleus_run_root}")
    core_paths = {extract_key(path): path for path in core_dir.glob("*_core_masks.tif")}

    single_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key_index, key in enumerate(keys, start=1):
        print(f"[{key_index}/{len(keys)}] key={key}", flush=True)
        core = read_mask(core_paths[key])
        states: dict[str, dict[str, dict[str, Any]]] = {
            "Brightfield": {},
            "Combined": {},
        }
        raw_cache: dict[str, np.ndarray] = {}
        for profile in ("Brightfield", "Combined"):
            first_row = row_index[(profile, profile_tags[profile][0], key)]
            raw_cache[profile] = tifffile.imread(Path(first_row["image_path"]))
            for tag in profile_tags[profile]:
                config = config_by_tag[tag]
                inference_row = row_index[(profile, tag, key)]
                mask = read_mask(Path(inference_row["mask_path"]))
                if mask.shape != core.shape:
                    raise ValueError(f"Shape mismatch for {profile}/{tag}/{key}")
                boundary = label_boundary(mask)
                assignment = nucleus_core_assignment(
                    core,
                    mask,
                    args.core_support_threshold,
                    args.core_unassigned_threshold,
                    args.core_secondary_conflict_threshold,
                )
                metrics = {
                    **cell_shape_metrics(mask, boundary),
                    **assignment["metrics"],
                }
                prepared = preprocess_image(raw_cache[profile], config)
                metrics["boundary_gradient_enrichment"] = gradient_enrichment(prepared, boundary)
                state = {
                    "mask": mask,
                    "boundary": boundary,
                    "boundary_distance": ndimage.distance_transform_edt(~boundary).astype(
                        np.float32, copy=False
                    ),
                    "assignment": assignment,
                    "metrics": metrics,
                }
                states[profile][tag] = state
                single_rows.append(
                    {
                        "profile": profile,
                        "config": tag,
                        "baseline": config.baseline,
                        "family": config.family,
                        "key": key,
                        "split": "calibration" if key in calibration_keys else "validation",
                        **metrics,
                        "mask_path": inference_row["mask_path"],
                    }
                )

        for bf_tag in profile_tags["Brightfield"]:
            for combined_tag in profile_tags["Combined"]:
                pair_rows.append(
                    {
                        "bf_config": bf_tag,
                        "combined_config": combined_tag,
                        "baseline_pair": (
                            bf_tag == baseline_tags["Brightfield"]
                            and combined_tag == baseline_tags["Combined"]
                        ),
                        "key": key,
                        "split": "calibration" if key in calibration_keys else "validation",
                        **pair_field_metrics(
                            states["Brightfield"][bf_tag],
                            states["Combined"][combined_tag],
                            args.boundary_tolerance,
                        ),
                    }
                )
    write_rows(out_dir / "single_field_metrics.csv", single_rows)
    write_rows(out_dir / "pair_field_metrics.csv", pair_rows)

    single_summary: list[dict[str, Any]] = []
    for profile in ("Brightfield", "Combined"):
        for tag in profile_tags[profile]:
            selected = [
                row for row in single_rows if row["profile"] == profile and row["config"] == tag
            ]
            summary: dict[str, Any] = {
                "profile": profile,
                "config": tag,
                "baseline": config_by_tag[tag].baseline,
                "family": config_by_tag[tag].family,
                "n_fields": len(selected),
            }
            for split in ("all", "calibration", "validation"):
                subset = selected if split == "all" else [row for row in selected if row["split"] == split]
                for metric in (
                    "n_objects",
                    "mask_fraction",
                    "area_median",
                    "small_fraction_lt60",
                    "large_fraction_gt2500",
                    "circularity_median",
                    "axis_ratio_median",
                    "core_supported_fraction",
                    "core_unassigned_fraction",
                    "core_boundary_conflict_fraction",
                    "empty_cell_fraction",
                    "multi_nucleus_cell_fraction",
                    "cell_count_to_nucleus_count",
                    "boundary_gradient_enrichment",
                ):
                    summary[f"{split}_{metric}"] = safe_mean(
                        float(row[metric]) for row in subset
                    )
            single_summary.append(summary)
    write_rows(out_dir / "single_candidate_summary.csv", single_summary)

    grouped_pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        grouped_pairs.setdefault((row["bf_config"], row["combined_config"]), []).append(row)
    ranking: list[dict[str, Any]] = []
    for (bf_tag, combined_tag), rows in grouped_pairs.items():
        aggregate: dict[str, Any] = {
            "bf_config": bf_tag,
            "combined_config": combined_tag,
            "baseline_pair": (
                bf_tag == baseline_tags["Brightfield"]
                and combined_tag == baseline_tags["Combined"]
            ),
            "n_fields": len(rows),
        }
        for split in ("all", "calibration", "validation"):
            aggregate.update(aggregate_pair_rows(rows, split))
        aggregate["calibration_loss"] = base_loss(aggregate, "calibration")
        aggregate["validation_loss"] = base_loss(aggregate, "validation")
        aggregate["generalization_gap"] = abs(
            float(aggregate["calibration_loss"]) - float(aggregate["validation_loss"])
        )
        aggregate["base_loss"] = base_loss(aggregate, "all")
        aggregate["total_loss"] = float(aggregate["base_loss"]) + 0.25 * float(
            aggregate["generalization_gap"]
        )
        aggregate["artifact_rate"] = (
            float(aggregate["all_bf_small_fraction_lt60"])
            + float(aggregate["all_combined_small_fraction_lt60"])
            + float(aggregate["all_bf_large_fraction_gt2500"])
            + float(aggregate["all_combined_large_fraction_gt2500"])
        )
        ranking.append(aggregate)

    baseline = next(row for row in ranking if row["baseline_pair"])
    for row in ranking:
        failures: list[str] = []
        if float(row["all_both_core_supported_fraction"]) < float(
            baseline["all_both_core_supported_fraction"]
        ) - 0.005:
            failures.append("core_support")
        if float(row["all_bf_core_boundary_conflict_fraction"]) > float(
            baseline["all_bf_core_boundary_conflict_fraction"]
        ) + 0.005:
            failures.append("bf_core_conflict")
        if float(row["all_combined_core_boundary_conflict_fraction"]) > float(
            baseline["all_combined_core_boundary_conflict_fraction"]
        ) + 0.005:
            failures.append("combined_core_conflict")
        if float(row["all_boundary_f1"]) < float(baseline["all_boundary_f1"]) - 0.015:
            failures.append("boundary_agreement")
        for profile_name in ("bf", "combined"):
            if float(row[f"all_{profile_name}_small_fraction_lt60"]) > float(
                baseline[f"all_{profile_name}_small_fraction_lt60"]
            ) + 0.03:
                failures.append(f"{profile_name}_small_objects")
            if float(row[f"all_{profile_name}_large_fraction_gt2500"]) > float(
                baseline[f"all_{profile_name}_large_fraction_gt2500"]
            ) + 0.02:
                failures.append(f"{profile_name}_large_objects")
        if float(row["generalization_gap"]) > float(baseline["generalization_gap"]) + 0.05:
            failures.append("generalization")
        row["hard_gate_pass"] = not failures
        row["hard_gate_failures"] = ";".join(failures)
        row["delta_total_loss_vs_baseline"] = float(row["total_loss"]) - float(
            baseline["total_loss"]
        )

    front = pareto_front(ranking)
    for row in ranking:
        row["pareto_front"] = (row["bf_config"], row["combined_config"]) in front
    ranking.sort(
        key=lambda row: (
            not bool(row["hard_gate_pass"]),
            float(row["total_loss"]),
            str(row["bf_config"]),
            str(row["combined_config"]),
        )
    )

    eligible = [row for row in ranking if row["hard_gate_pass"]]
    best = eligible[0] if eligible else baseline
    improvement = float(baseline["total_loss"]) - float(best["total_loss"])
    if best is not baseline and improvement >= args.minimum_improvement:
        selected = best
        selection_reason = "candidate_passed_all_gates_and_improved_joint_loss"
    else:
        selected = baseline
        selection_reason = "production_baseline_retained_without_clear_improvement"
    for row in ranking:
        row["selected"] = row is selected

    write_rows(out_dir / "pair_candidate_ranking.csv", ranking)
    write_rows(
        out_dir / "pareto_pairs.csv",
        [row for row in ranking if row["pareto_front"]],
    )

    selected_payload = {
        "selection_reason": selection_reason,
        "minimum_improvement": args.minimum_improvement,
        "baseline_total_loss": float(baseline["total_loss"]),
        "selected_total_loss": float(selected["total_loss"]),
        "improvement_vs_baseline": float(baseline["total_loss"])
        - float(selected["total_loss"]),
        "brightfield": asdict(config_by_tag[str(selected["bf_config"])]),
        "combined": asdict(config_by_tag[str(selected["combined_config"])]),
        "selected_metrics": selected,
        "baseline_metrics": baseline,
        "calibration_keys": manifest["calibration_keys"],
        "validation_keys": manifest["validation_keys"],
    }
    (out_dir / "selected_pair.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n"
    )

    report_lines = [
        "# High-density Brightfield/Combined joint calibration",
        "",
        f"- fields: {len(keys)}",
        f"- calibration: {', '.join(manifest['calibration_keys'])}",
        f"- validation: {', '.join(manifest['validation_keys'])}",
        f"- baseline pair: {baseline_tags['Brightfield']} + {baseline_tags['Combined']}",
        f"- selected pair: {selected['bf_config']} + {selected['combined_config']}",
        f"- selection reason: {selection_reason}",
        f"- baseline total loss: {float(baseline['total_loss']):.6f}",
        f"- selected total loss: {float(selected['total_loss']):.6f}",
        f"- improvement: {float(baseline['total_loss']) - float(selected['total_loss']):.6f}",
        "",
        "The score does not penalize multinucleated cells by itself. It penalizes only",
        "unsupported nucleus cores, boundary conflicts, BF/Combined disagreement,",
        "artifact tails, and calibration/validation instability.",
        "",
        "## Top eligible pairs",
        "",
        "| BF | Combined | loss | core support | boundary F1 | multiplicity agreement | gap | Pareto |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in eligible[:12]:
        report_lines.append(
            f"| {row['bf_config']} | {row['combined_config']} | {float(row['total_loss']):.4f} | "
            f"{float(row['all_both_core_supported_fraction']):.4f} | "
            f"{float(row['all_boundary_f1']):.4f} | "
            f"{float(row['all_core_multiplicity_agreement_fraction']):.4f} | "
            f"{float(row['generalization_gap']):.4f} | {row['pareto_front']} |"
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- single-field metrics: `{out_dir / 'single_field_metrics.csv'}`",
            f"- pair-field metrics: `{out_dir / 'pair_field_metrics.csv'}`",
            f"- pair ranking: `{out_dir / 'pair_candidate_ranking.csv'}`",
            f"- selected configuration: `{out_dir / 'selected_pair.json'}`",
        ]
    )
    (out_dir / "scoring_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"selected_bf={selected['bf_config']}", flush=True)
    print(f"selected_combined={selected['combined_config']}", flush=True)
    print(f"selection_reason={selection_reason}", flush=True)
    print(f"selected_pair={out_dir / 'selected_pair.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
