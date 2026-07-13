#!/usr/bin/env python3
"""Quantify nuclear morphology and Nuclei/BF/Combined mask alignment.

This analysis is intentionally independent of the segmentation model runtime.  It
uses the instance masks already written by the production workflow and produces
object-level, cell-level, field-level, and visual QC outputs.

The script only depends on packages already present in the local CellPose pyenv:
NumPy, SciPy, OpenCV, tifffile, and Pillow.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
KEY_DETAIL_RE = re.compile(
    r"(?P<well>[A-H]\d+)_(?P<site>\d+)_(?P<day>\d{2})d"
    r"(?P<hour>\d{2})h(?P<minute>\d{2})m"
)

PROFILE_STEMS = {
    "Combined": "SUM159_AC_{key}",
    "Brightfield": "SUM159_AC_Exp1_BF_{key}",
    "Nuclei": "SUM159_AC_Exp1_{key}",
}

PALETTE = {
    "ink": "#20252b",
    "muted": "#66717d",
    "grid": "#d9dee3",
    "blue": "#2c6eba",
    "gold": "#d7a928",
    "orange": "#d97a30",
    "olive": "#7a8f36",
    "pink": "#c85b8e",
    "red": "#b8423f",
    "light_blue": "#a9c7e8",
}

OVERLAY_COLORS = {
    "matched_consensus": (58, 166, 94),
    "matched_one_way": (44, 110, 186),
    "crosses_both": (204, 65, 150),
    "crosses_combined": (204, 65, 150),
    "crosses_brightfield": (204, 65, 150),
    "unmatched_both": (184, 66, 63),
    "combined_unmatched": (217, 122, 48),
    "brightfield_unmatched": (217, 122, 48),
    "bf_combined_assignment_conflict": (217, 122, 48),
    "low_containment": (215, 169, 40),
}


@dataclass
class LabelStats:
    labels: np.ndarray
    areas: np.ndarray
    centroids: np.ndarray
    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray
    major_axis: np.ndarray
    minor_axis: np.ndarray
    eccentricity: np.ndarray
    orientation_deg: np.ndarray
    border_touch: np.ndarray

    @property
    def count(self) -> int:
        return int(self.labels.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure nuclear morphology, match nuclei to Brightfield and Combined "
            "cell masks, detect multinucleated cells, calculate nuclear/cell ratios, "
            "and classify alignment failures."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--min-nucleus-area", type=float, default=35.0)
    parser.add_argument("--min-cell-area", type=float, default=35.0)
    parser.add_argument("--high-containment", type=float, default=0.80)
    parser.add_argument("--moderate-containment", type=float, default=0.50)
    parser.add_argument("--unmatched-overlap", type=float, default=0.20)
    parser.add_argument("--significant-intersection-fraction", type=float, default=0.05)
    parser.add_argument("--significant-secondary-fraction", type=float, default=0.10)
    parser.add_argument("--min-significant-overlap-px", type=int, default=3)
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument("--max-fields", type=int, default=None)
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.run_root / "nuclear_cell_alignment_analysis"
    for name in (
        "high_containment",
        "moderate_containment",
        "unmatched_overlap",
        "significant_intersection_fraction",
        "significant_secondary_fraction",
        "overlay_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.high_containment < args.moderate_containment:
        raise SystemExit("--high-containment must be >= --moderate-containment")
    if args.moderate_containment < args.unmatched_overlap:
        raise SystemExit("--moderate-containment must be >= --unmatched-overlap")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        raise SystemExit("--pixel-size-um must be positive")
    return args


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def key_details(key: str) -> dict[str, Any]:
    match = KEY_DETAIL_RE.fullmatch(key)
    if match is None:
        return {"well": "", "site": "", "day": "", "hour": "", "minute": "", "elapsed_hours": ""}
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    return {
        "well": match.group("well"),
        "site": int(match.group("site")),
        "day": day,
        "hour": hour,
        "minute": minute,
        "elapsed_hours": day * 24 + hour + minute / 60.0,
    }


def index_mask_files(run_root: Path, profile: str) -> dict[str, Path]:
    mask_dir = run_root / profile / "segmentations"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing {profile} mask directory: {mask_dir}")
    result: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*_cp_masks.tif")):
        key = key_from_path(path)
        if key in result:
            raise ValueError(f"Duplicate {profile} mask key {key}: {result[key]} and {path}")
        result[key] = path
    return result


def index_nucleus_core_files(run_root: Path) -> dict[str, Path]:
    candidate_dirs = (
        run_root / "Nuclei" / "nucleus_core_seeds",
        run_root / "Nuclei" / "nucleus_core_seeds" / "Nuclei",
        run_root / "nucleus_core_seeds",
        run_root / "nucleus_core_seeds" / "Nuclei",
    )
    result: dict[str, Path] = {}
    for mask_dir in candidate_dirs:
        if not mask_dir.is_dir():
            continue
        for path in sorted(mask_dir.glob("*_core_masks.tif")):
            key = key_from_path(path)
            if key in result and result[key].resolve() != path.resolve():
                raise ValueError(f"Duplicate nucleus-core mask key {key}: {result[key]} and {path}")
            result[key] = path
    return result


def index_raw_files(input_root: Path, profile: str) -> dict[str, Path]:
    raw_dir = input_root / profile
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing {profile} raw-image directory: {raw_dir}")
    suffix_rank = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}
    result: dict[str, Path] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffix_rank:
            continue
        try:
            key = key_from_path(path)
        except ValueError:
            continue
        current = result.get(key)
        if current is None or suffix_rank[path.suffix.lower()] < suffix_rank[current.suffix.lower()]:
            result[key] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D instance mask at {path}, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Expected integer instance mask at {path}, got {mask.dtype}")
    if np.any(mask < 0):
        raise ValueError(f"Negative instance labels in {path}")
    return mask.astype(np.int32, copy=False)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] >= 3:
        arr = image[..., :3].astype(np.float64)
        return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    raise ValueError(f"Cannot convert image with shape {image.shape} to grayscale")


def robust_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Cannot convert image with shape {image.shape} to RGB")
    arr = image.astype(np.float64, copy=False)
    lo, hi = np.percentile(arr, (1.0, 99.0))
    scaled = np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def compute_label_stats(mask: np.ndarray) -> LabelStats:
    max_label = int(mask.max()) if mask.size else 0
    size = max_label + 1
    empty_float = np.zeros(size, dtype=np.float64)
    empty_int = np.zeros(size, dtype=np.int32)
    if max_label == 0:
        return LabelStats(
            labels=np.zeros(0, dtype=np.int32),
            areas=empty_float,
            centroids=np.zeros((size, 2), dtype=np.float64),
            y0=empty_int.copy(),
            x0=empty_int.copy(),
            y1=empty_int.copy(),
            x1=empty_int.copy(),
            major_axis=empty_float.copy(),
            minor_axis=empty_float.copy(),
            eccentricity=empty_float.copy(),
            orientation_deg=empty_float.copy(),
            border_touch=np.zeros(size, dtype=bool),
        )

    ys, xs = np.nonzero(mask)
    labels_at_pixels = mask[ys, xs].astype(np.int64, copy=False)
    areas = np.bincount(labels_at_pixels, minlength=size).astype(np.float64)
    labels = np.flatnonzero(areas > 0).astype(np.int32)
    labels = labels[labels > 0]

    sum_y = np.bincount(labels_at_pixels, weights=ys, minlength=size)
    sum_x = np.bincount(labels_at_pixels, weights=xs, minlength=size)
    sum_y2 = np.bincount(labels_at_pixels, weights=ys.astype(np.float64) ** 2, minlength=size)
    sum_x2 = np.bincount(labels_at_pixels, weights=xs.astype(np.float64) ** 2, minlength=size)
    sum_xy = np.bincount(
        labels_at_pixels,
        weights=ys.astype(np.float64) * xs.astype(np.float64),
        minlength=size,
    )
    centroids = np.zeros((size, 2), dtype=np.float64)
    nonzero = areas > 0
    centroids[nonzero, 0] = sum_y[nonzero] / areas[nonzero]
    centroids[nonzero, 1] = sum_x[nonzero] / areas[nonzero]

    cov_yy = np.zeros(size, dtype=np.float64)
    cov_xx = np.zeros(size, dtype=np.float64)
    cov_yx = np.zeros(size, dtype=np.float64)
    cov_yy[nonzero] = sum_y2[nonzero] / areas[nonzero] - centroids[nonzero, 0] ** 2
    cov_xx[nonzero] = sum_x2[nonzero] / areas[nonzero] - centroids[nonzero, 1] ** 2
    cov_yx[nonzero] = sum_xy[nonzero] / areas[nonzero] - centroids[nonzero, 0] * centroids[nonzero, 1]
    trace = cov_xx + cov_yy
    delta = np.sqrt(np.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_yx**2, 0.0))
    eig_major = np.maximum((trace + delta) / 2.0, 0.0)
    eig_minor = np.maximum((trace - delta) / 2.0, 0.0)
    major_axis = 4.0 * np.sqrt(eig_major)
    minor_axis = 4.0 * np.sqrt(eig_minor)
    eccentricity = np.zeros(size, dtype=np.float64)
    valid_major = eig_major > 0
    eccentricity[valid_major] = np.sqrt(
        np.clip(1.0 - eig_minor[valid_major] / eig_major[valid_major], 0.0, 1.0)
    )
    orientation_deg = np.degrees(0.5 * np.arctan2(2.0 * cov_yx, cov_xx - cov_yy))

    height, width = mask.shape
    y0 = np.full(size, height, dtype=np.int32)
    x0 = np.full(size, width, dtype=np.int32)
    y1 = np.zeros(size, dtype=np.int32)
    x1 = np.zeros(size, dtype=np.int32)
    np.minimum.at(y0, labels_at_pixels, ys)
    np.minimum.at(x0, labels_at_pixels, xs)
    np.maximum.at(y1, labels_at_pixels, ys + 1)
    np.maximum.at(x1, labels_at_pixels, xs + 1)
    border_touch = (y0 == 0) | (x0 == 0) | (y1 == height) | (x1 == width)
    border_touch[0] = False

    return LabelStats(
        labels=labels,
        areas=areas,
        centroids=centroids,
        y0=y0,
        x0=x0,
        y1=y1,
        x1=x1,
        major_axis=major_axis,
        minor_axis=minor_axis,
        eccentricity=eccentricity,
        orientation_deg=orientation_deg,
        border_touch=border_touch,
    )


def object_shape_metrics(mask: np.ndarray, stats: LabelStats, label: int) -> dict[str, float]:
    y0, x0, y1, x1 = (int(stats.y0[label]), int(stats.x0[label]), int(stats.y1[label]), int(stats.x1[label]))
    obj = (mask[y0:y1, x0:x1] == label).astype(np.uint8)
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    points = np.concatenate(contours, axis=0) if contours else np.zeros((0, 1, 2), dtype=np.int32)
    if points.size:
        hull = cv2.convexHull(points)
        hull_mask = np.zeros_like(obj)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        convex_area = float(np.count_nonzero(hull_mask))
    else:
        convex_area = float(stats.areas[label])
    area = float(stats.areas[label])
    bbox_area = float(max((y1 - y0) * (x1 - x0), 1))
    return {
        "perimeter_px": perimeter,
        "convex_area_px2": convex_area,
        "solidity": area / max(convex_area, area, 1.0),
        "extent": area / bbox_area,
        "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0,
    }


def intensity_arrays(raw_gray: np.ndarray, mask: np.ndarray, stats: LabelStats) -> tuple[np.ndarray, np.ndarray]:
    max_label = len(stats.areas) - 1
    labels = mask.ravel().astype(np.int64, copy=False)
    values = raw_gray.ravel().astype(np.float64, copy=False)
    sums = np.bincount(labels, weights=values, minlength=max_label + 1)
    sums2 = np.bincount(labels, weights=values * values, minlength=max_label + 1)
    means = np.zeros(max_label + 1, dtype=np.float64)
    stds = np.zeros(max_label + 1, dtype=np.float64)
    nonzero = stats.areas > 0
    means[nonzero] = sums[nonzero] / stats.areas[nonzero]
    variance = np.zeros_like(means)
    variance[nonzero] = sums2[nonzero] / stats.areas[nonzero] - means[nonzero] ** 2
    stds[nonzero] = np.sqrt(np.maximum(variance[nonzero], 0.0))
    return means, stds


def object_intensity_metrics(
    raw_gray: np.ndarray,
    nuclear_mask: np.ndarray,
    stats: LabelStats,
    label: int,
    mean_value: float,
    std_value: float,
) -> dict[str, float]:
    margin = 4
    y0 = max(int(stats.y0[label]) - margin, 0)
    x0 = max(int(stats.x0[label]) - margin, 0)
    y1 = min(int(stats.y1[label]) + margin, nuclear_mask.shape[0])
    x1 = min(int(stats.x1[label]) + margin, nuclear_mask.shape[1])
    mask_crop = nuclear_mask[y0:y1, x0:x1]
    obj = mask_crop == label
    values = raw_gray[y0:y1, x0:x1][obj].astype(np.float64)
    if values.size:
        p10, median, p90 = np.percentile(values, (10, 50, 90))
    else:
        p10 = median = p90 = float("nan")
    dilated = cv2.dilate(obj.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    ring = dilated & (mask_crop == 0)
    background = raw_gray[y0:y1, x0:x1][ring].astype(np.float64)
    if background.size >= 10:
        bg_median = float(np.median(background))
        bg_mad = float(np.median(np.abs(background - bg_median)))
        local_snr = (float(median) - bg_median) / max(1.4826 * bg_mad, 1e-6)
    else:
        bg_median = float("nan")
        bg_mad = float("nan")
        local_snr = float("nan")
    return {
        "nuclear_intensity_mean": float(mean_value),
        "nuclear_intensity_std": float(std_value),
        "nuclear_intensity_p10": float(p10),
        "nuclear_intensity_median": float(median),
        "nuclear_intensity_p90": float(p90),
        "local_background_median": bg_median,
        "local_background_mad": bg_mad,
        "local_intensity_snr": float(local_snr),
    }


def label_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary &= mask > 0
    return boundary


def overlap_lists(source_mask: np.ndarray, target_mask: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    selected = source_mask > 0
    if not np.any(selected):
        return {}
    base = int(target_mask.max()) + 1
    pair_codes = source_mask[selected].astype(np.int64) * base + target_mask[selected].astype(np.int64)
    unique_codes, counts = np.unique(pair_codes, return_counts=True)
    result: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for code, count in zip(unique_codes, counts):
        source_label = int(code // base)
        target_label = int(code % base)
        if target_label > 0:
            result[source_label].append((target_label, int(count)))
    for values in result.values():
        values.sort(key=lambda item: (-item[1], item[0]))
    return result


def compute_nucleus_target_matches(
    nuclear_mask: np.ndarray,
    target_mask: np.ndarray,
    nuclear_stats: LabelStats,
    target_stats: LabelStats,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    overlaps = overlap_lists(nuclear_mask, target_mask)
    boundary_distance = ndimage.distance_transform_edt(~label_boundary(target_mask))
    if np.any(target_mask > 0):
        background_distance, nearest_indices = ndimage.distance_transform_edt(
            target_mask == 0,
            return_indices=True,
        )
    else:
        background_distance = np.full(target_mask.shape, np.inf)
        nearest_indices = np.indices(target_mask.shape)

    output: dict[int, dict[str, Any]] = {}
    height, width = nuclear_mask.shape
    for label_value in nuclear_stats.labels:
        label = int(label_value)
        area = float(nuclear_stats.areas[label])
        values = overlaps.get(label, [])
        total_inside = int(sum(count for _, count in values))
        primary_id, primary_overlap = values[0] if values else (0, 0)
        secondary_id, secondary_overlap = values[1] if len(values) > 1 else (0, 0)
        primary_fraction = primary_overlap / area
        secondary_fraction = secondary_overlap / area
        total_fraction = total_inside / area
        outside_fraction = max(0.0, 1.0 - total_fraction)
        significant_cells = sum(
            1
            for _, count in values
            if count >= args.min_significant_overlap_px
            and count / area >= args.significant_intersection_fraction
        )

        cy, cx = nuclear_stats.centroids[label]
        yy = min(max(int(round(float(cy))), 0), height - 1)
        xx = min(max(int(round(float(cx))), 0), width - 1)
        centroid_target_id = int(target_mask[yy, xx])
        centroid_in_primary = primary_id > 0 and centroid_target_id == primary_id
        if primary_id > 0:
            target_area = float(target_stats.areas[primary_id])
            iou = primary_overlap / max(area + target_area - primary_overlap, 1.0)
            nucleus_fraction_of_cell = primary_overlap / max(target_area, 1.0)
            nearest_id = primary_id
            nearest_distance = 0.0
        else:
            target_area = 0.0
            iou = 0.0
            nucleus_fraction_of_cell = 0.0
            ny = int(nearest_indices[0, yy, xx])
            nx = int(nearest_indices[1, yy, xx])
            nearest_id = int(target_mask[ny, nx])
            nearest_distance = float(background_distance[yy, xx])

        if primary_id == 0 or total_fraction < args.unmatched_overlap:
            category = "unmatched"
        elif significant_cells >= 2 or secondary_fraction >= args.significant_secondary_fraction:
            category = "crosses_cells"
        elif primary_fraction >= args.high_containment and centroid_in_primary:
            category = "contained"
        elif outside_fraction >= 1.0 - args.high_containment:
            category = "partly_outside"
        else:
            category = "low_containment"

        assignment_score = 0.0
        if primary_id > 0:
            assignment_score = (
                0.75 * primary_fraction
                + 0.15 * float(centroid_in_primary)
                + 0.10 * max(0.0, 1.0 - secondary_fraction)
            )
        output[label] = {
            "primary_cell_id": primary_id,
            "primary_overlap_px": primary_overlap,
            "primary_nucleus_fraction": primary_fraction,
            "primary_cell_fraction": nucleus_fraction_of_cell,
            "primary_iou": iou,
            "secondary_cell_id": secondary_id,
            "secondary_overlap_px": secondary_overlap,
            "secondary_nucleus_fraction": secondary_fraction,
            "total_cell_overlap_px": total_inside,
            "total_cell_coverage_fraction": total_fraction,
            "outside_cell_fraction": outside_fraction,
            "n_intersected_cells": len(values),
            "n_significant_cells": significant_cells,
            "centroid_cell_id": centroid_target_id,
            "centroid_in_primary": centroid_in_primary,
            "centroid_boundary_distance_px": float(boundary_distance[yy, xx]),
            "nearest_cell_id": nearest_id,
            "nearest_cell_distance_px": nearest_distance,
            "match_category": category,
            "assignment_score": min(max(float(assignment_score), 0.0), 1.0),
        }
    return output


def compute_cell_correspondence(
    combined_mask: np.ndarray,
    bf_mask: np.ndarray,
    combined_stats: LabelStats,
    bf_stats: LabelStats,
    key: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]], dict[int, int], dict[int, int]]:
    both = (combined_mask > 0) & (bf_mask > 0)
    if not np.any(both):
        return [], {}, {}, {}
    base = int(bf_mask.max()) + 1
    codes = combined_mask[both].astype(np.int64) * base + bf_mask[both].astype(np.int64)
    unique_codes, counts = np.unique(codes, return_counts=True)
    raw_pairs: list[tuple[int, int, int]] = []
    best_combined: dict[int, tuple[int, int]] = {}
    best_bf: dict[int, tuple[int, int]] = {}
    for code, count_value in zip(unique_codes, counts):
        combined_id = int(code // base)
        bf_id = int(code % base)
        count = int(count_value)
        raw_pairs.append((combined_id, bf_id, count))
        if count > best_combined.get(combined_id, (0, 0))[1]:
            best_combined[combined_id] = (bf_id, count)
        if count > best_bf.get(bf_id, (0, 0))[1]:
            best_bf[bf_id] = (combined_id, count)
    combined_to_bf = {label: value[0] for label, value in best_combined.items()}
    bf_to_combined = {label: value[0] for label, value in best_bf.items()}

    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for combined_id, bf_id, count in raw_pairs:
        combined_area = float(combined_stats.areas[combined_id])
        bf_area = float(bf_stats.areas[bf_id])
        row = {
            "key": key,
            "combined_cell_id": combined_id,
            "bf_cell_id": bf_id,
            "overlap_px": count,
            "combined_covered_by_bf_fraction": count / max(combined_area, 1.0),
            "bf_covered_by_combined_fraction": count / max(bf_area, 1.0),
            "overlap_of_smaller_cell_fraction": count / max(min(combined_area, bf_area), 1.0),
            "cell_iou": count / max(combined_area + bf_area - count, 1.0),
            "combined_best_bf": combined_to_bf.get(combined_id) == bf_id,
            "bf_best_combined": bf_to_combined.get(bf_id) == combined_id,
            "reciprocal_best": (
                combined_to_bf.get(combined_id) == bf_id
                and bf_to_combined.get(bf_id) == combined_id
            ),
        }
        rows.append(row)
        lookup[(combined_id, bf_id)] = row
    return rows, lookup, combined_to_bf, bf_to_combined


def morphology_score(row: dict[str, Any], min_area: float) -> float:
    area_score = min(max(float(row["nuclear_area_px2"]) / max(min_area, 1.0), 0.0), 1.0)
    solidity_score = min(max((float(row["solidity"]) - 0.55) / 0.35, 0.0), 1.0)
    snr = float(row["local_intensity_snr"])
    intensity_score = 0.5 if not math.isfinite(snr) else min(max((snr - 0.5) / 4.0, 0.0), 1.0)
    border_score = 0.0 if row["border_touch"] else 1.0
    return 0.45 * area_score + 0.25 * solidity_score + 0.20 * intensity_score + 0.10 * border_score


def classify_consensus(
    combined_id: int,
    bf_id: int,
    pair_lookup: dict[tuple[int, int], dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if combined_id <= 0 and bf_id <= 0:
        return "unmatched_both", None
    if combined_id <= 0:
        return "brightfield_only", None
    if bf_id <= 0:
        return "combined_only", None
    pair = pair_lookup.get((combined_id, bf_id))
    if pair is None:
        return "conflict_no_overlap", None
    if pair["reciprocal_best"]:
        return "reciprocal_best", pair
    if pair["combined_best_bf"] or pair["bf_best_combined"]:
        return "one_way_best", pair
    return "overlapping_nonbest", pair


def consensus_score(category: str) -> float:
    return {
        "reciprocal_best": 1.0,
        "one_way_best": 0.75,
        "overlapping_nonbest": 0.40,
        "brightfield_only": 0.25,
        "combined_only": 0.25,
        "conflict_no_overlap": 0.0,
        "unmatched_both": 0.0,
    }[category]


def overall_alignment_status(
    combined_match: dict[str, Any],
    bf_match: dict[str, Any],
    consensus: str,
) -> str:
    combined_category = str(combined_match["match_category"])
    bf_category = str(bf_match["match_category"])
    if combined_category == "crosses_cells" and bf_category == "crosses_cells":
        return "crosses_both"
    if combined_category == "crosses_cells":
        return "crosses_combined"
    if bf_category == "crosses_cells":
        return "crosses_brightfield"
    if combined_category == "unmatched" and bf_category == "unmatched":
        return "unmatched_both"
    if combined_category == "unmatched":
        return "combined_unmatched"
    if bf_category == "unmatched":
        return "brightfield_unmatched"
    if consensus in {"conflict_no_overlap", "overlapping_nonbest"}:
        return "bf_combined_assignment_conflict"
    if combined_category == "contained" and bf_category == "contained":
        if consensus == "reciprocal_best":
            return "matched_consensus"
        if consensus == "one_way_best":
            return "matched_one_way"
    return "low_containment"


def load_combined_classification(run_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    feature_dir = run_root / "classification_fusion" / "features"
    if not feature_dir.is_dir():
        return result
    for path in sorted(feature_dir.glob("*_per_cell_fusion_features.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = str(row["key"])
                    cell_id = int(row["combined_mask_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                result[(key, cell_id)] = {
                    "final_state": row.get("final_state", row.get("state", "")),
                    "classification_confidence": row.get("classification_confidence", ""),
                    "countable": str(row.get("countable", "")).lower() == "true",
                }
    return result


def load_density_calls(run_root: Path) -> dict[str, bool]:
    path = run_root / "qc" / "density_calls.csv"
    if not path.is_file():
        return {}
    result: dict[str, bool] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("key", ""))
            if key:
                result[key] = str(row.get("high_density", "")).strip().lower() == "true"
    return result


def cross_method_nuclear_status_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = {
        (str(row["key"]), int(row["cell_id"])): row
        for row in cell_rows
        if row["cell_mask_profile"] == "Combined"
    }
    brightfield = {
        (str(row["key"]), int(row["cell_id"])): row
        for row in cell_rows
        if row["cell_mask_profile"] == "Brightfield"
    }
    output: list[dict[str, Any]] = []
    for (key, combined_id), combined_row in combined.items():
        if not combined_row["cell_countable_by_area"] or not combined_row["paired_cell_reciprocal_best"]:
            continue
        bf_id = int(combined_row["paired_other_profile_cell_id"])
        bf_row = brightfield.get((key, bf_id))
        if bf_row is None or not bf_row["cell_countable_by_area"]:
            continue
        combined_status = str(combined_row["nuclear_status"])
        bf_status = str(bf_row["nuclear_status"])
        combined_confirmed = combined_status == "confirmed_multinucleated"
        bf_confirmed = bf_status == "confirmed_multinucleated"
        output.append(
            {
                "key": key,
                **key_details(key),
                "high_density_field": bool(combined_row.get("high_density_field", False)),
                "combined_cell_id": combined_id,
                "bf_cell_id": bf_id,
                "cell_pair_iou": float(combined_row["paired_cell_iou"]),
                "combined_nuclear_status": combined_status,
                "bf_nuclear_status": bf_status,
                "exact_status_concordance": combined_status == bf_status,
                "combined_confirmed_multinucleated": combined_confirmed,
                "bf_confirmed_multinucleated": bf_confirmed,
                "confirmed_multinucleation_concordance": combined_confirmed == bf_confirmed,
                "combined_n_nuclei_high_confidence": int(combined_row["n_nuclei_high_confidence"]),
                "bf_n_nuclei_high_confidence": int(bf_row["n_nuclei_high_confidence"]),
                "combined_nucleus_to_cell_area_ratio": float(combined_row["nucleus_to_cell_area_ratio"]),
                "bf_nucleus_to_cell_area_ratio": float(bf_row["nucleus_to_cell_area_ratio"]),
                "absolute_nucleus_to_cell_ratio_difference": abs(
                    float(combined_row["nucleus_to_cell_area_ratio"])
                    - float(bf_row["nucleus_to_cell_area_ratio"])
                ),
            }
        )
    return output


def cell_summary_rows(
    key: str,
    profile: str,
    cell_mask: np.ndarray,
    cell_stats: LabelStats,
    nuclear_mask: np.ndarray,
    nucleus_core_mask: np.ndarray | None,
    nucleus_rows: list[dict[str, Any]],
    match_prefix: str,
    args: argparse.Namespace,
    classification: dict[tuple[str, int], dict[str, Any]],
    correspondence_lookup: dict[tuple[int, int], dict[str, Any]],
    combined_to_bf: dict[int, int],
    bf_to_combined: dict[int, int],
) -> list[dict[str, Any]]:
    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nucleus_rows:
        cell_id = int(row[f"{match_prefix}_cell_id"])
        if cell_id > 0:
            assigned[cell_id].append(row)
    nuclear_area_inside = np.bincount(
        cell_mask[nuclear_mask > 0].astype(np.int64),
        minlength=len(cell_stats.areas),
    ).astype(np.float64)
    nucleus_core_area_inside = (
        np.bincount(
            cell_mask[nucleus_core_mask > 0].astype(np.int64),
            minlength=len(cell_stats.areas),
        ).astype(np.float64)
        if nucleus_core_mask is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    details = key_details(key)
    for cell_label_value in cell_stats.labels:
        cell_id = int(cell_label_value)
        cell_area = float(cell_stats.areas[cell_id])
        related = assigned.get(cell_id, [])
        moderate = [
            row
            for row in related
            if float(row[f"{match_prefix}_primary_nucleus_fraction"]) >= args.moderate_containment
            and not row["fragment_suspect"]
        ]
        high = [
            row
            for row in related
            if row[f"{match_prefix}_match_category"] == "contained"
            and row["quality_nucleus"]
        ]
        if len(high) >= 2:
            nuclear_status = "confirmed_multinucleated"
        elif len(moderate) >= 2:
            nuclear_status = "candidate_multinucleated"
        elif len(high) == 1 or len(moderate) == 1:
            nuclear_status = "single_nucleus"
        else:
            nuclear_status = "no_confident_nucleus"

        inside = float(nuclear_area_inside[cell_id])
        nucleus_to_cell = inside / max(cell_area, 1.0)
        cytoplasm = max(cell_area - inside, 0.0)
        nucleus_to_cytoplasm = inside / cytoplasm if cytoplasm > 0 else float("inf")
        core_inside = float(nucleus_core_area_inside[cell_id]) if nucleus_core_area_inside is not None else float("nan")
        core_nucleus_to_cell = core_inside / max(cell_area, 1.0) if math.isfinite(core_inside) else float("nan")
        core_cytoplasm = max(cell_area - core_inside, 0.0) if math.isfinite(core_inside) else float("nan")
        core_nucleus_to_cytoplasm = (
            core_inside / core_cytoplasm
            if math.isfinite(core_cytoplasm) and core_cytoplasm > 0
            else float("inf") if math.isfinite(core_cytoplasm) else float("nan")
        )
        row: dict[str, Any] = {
            "key": key,
            **details,
            "cell_mask_profile": profile,
            "cell_id": cell_id,
            "cell_area_px2": cell_area,
            "cell_centroid_y": float(cell_stats.centroids[cell_id, 0]),
            "cell_centroid_x": float(cell_stats.centroids[cell_id, 1]),
            "cell_border_touch": bool(cell_stats.border_touch[cell_id]),
            "cell_countable_by_area": cell_area >= args.min_cell_area,
            "n_nuclei_primary_assigned": len(related),
            "n_nuclei_moderate": len(moderate),
            "n_nuclei_high_confidence": len(high),
            "nuclear_status": nuclear_status,
            "nuclear_area_inside_cell_px2": inside,
            "nucleus_to_cell_area_ratio": nucleus_to_cell,
            "cytoplasm_area_px2": cytoplasm,
            "nucleus_to_cytoplasm_area_ratio": nucleus_to_cytoplasm,
            "core_nuclear_area_inside_cell_px2": core_inside if math.isfinite(core_inside) else "",
            "core_nucleus_to_cell_area_ratio": (
                core_nucleus_to_cell if math.isfinite(core_nucleus_to_cell) else ""
            ),
            "core_based_cytoplasm_area_px2": core_cytoplasm if math.isfinite(core_cytoplasm) else "",
            "core_nucleus_to_cytoplasm_area_ratio": (
                core_nucleus_to_cytoplasm if math.isfinite(core_nucleus_to_cytoplasm) else ""
            ),
            "sum_full_area_high_conf_nuclei_px2": float(sum(r["nuclear_area_px2"] for r in high)),
        }
        if args.pixel_size_um is not None:
            scale2 = args.pixel_size_um**2
            row["cell_area_um2"] = cell_area * scale2
            row["nuclear_area_inside_cell_um2"] = inside * scale2
            row["core_nuclear_area_inside_cell_um2"] = (
                core_inside * scale2 if math.isfinite(core_inside) else ""
            )
            row["cytoplasm_area_um2"] = cytoplasm * scale2
        else:
            row["cell_area_um2"] = ""
            row["nuclear_area_inside_cell_um2"] = ""
            row["core_nuclear_area_inside_cell_um2"] = ""
            row["cytoplasm_area_um2"] = ""

        if profile == "Combined":
            paired_id = combined_to_bf.get(cell_id, 0)
            pair = correspondence_lookup.get((cell_id, paired_id)) if paired_id else None
            class_row = classification.get((key, cell_id), {})
            row.update(
                {
                    "paired_other_profile_cell_id": paired_id,
                    "paired_cell_reciprocal_best": bool(pair and pair["reciprocal_best"]),
                    "paired_cell_iou": float(pair["cell_iou"]) if pair else 0.0,
                    "classification_state": class_row.get("final_state", ""),
                    "classification_confidence": class_row.get("classification_confidence", ""),
                    "classification_countable": class_row.get("countable", ""),
                }
            )
        else:
            paired_id = bf_to_combined.get(cell_id, 0)
            pair = correspondence_lookup.get((paired_id, cell_id)) if paired_id else None
            row.update(
                {
                    "paired_other_profile_cell_id": paired_id,
                    "paired_cell_reciprocal_best": bool(pair and pair["reciprocal_best"]),
                    "paired_cell_iou": float(pair["cell_iou"]) if pair else 0.0,
                    "classification_state": "",
                    "classification_confidence": "",
                    "classification_countable": "",
                }
            )
        rows.append(row)
    return rows


def fit_gmm_1d(values: np.ndarray, max_components: int = 3) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Cannot fit a size model without observations")
    global_var = max(float(np.var(x)), 1e-4)
    candidates: list[dict[str, Any]] = []
    for n_components in range(1, max_components + 1):
        if n_components == 1:
            means = np.array([float(np.mean(x))])
        else:
            quantiles = np.linspace(0, 1, n_components + 2)[1:-1]
            means = np.quantile(x, quantiles).astype(np.float64)
        variances = np.full(n_components, global_var, dtype=np.float64)
        weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        previous_ll = -np.inf
        for _ in range(300):
            log_probability = np.empty((x.size, n_components), dtype=np.float64)
            for component in range(n_components):
                variance = max(float(variances[component]), 1e-4)
                log_probability[:, component] = (
                    math.log(max(float(weights[component]), 1e-12))
                    - 0.5 * math.log(2.0 * math.pi * variance)
                    - 0.5 * (x - means[component]) ** 2 / variance
                )
            row_max = np.max(log_probability, axis=1, keepdims=True)
            probability = np.exp(log_probability - row_max)
            denominator = np.sum(probability, axis=1, keepdims=True)
            responsibilities = probability / denominator
            log_likelihood = float(np.sum(row_max[:, 0] + np.log(denominator[:, 0])))
            component_mass = np.sum(responsibilities, axis=0) + 1e-9
            weights = component_mass / x.size
            means = np.sum(responsibilities * x[:, None], axis=0) / component_mass
            variances = (
                np.sum(responsibilities * (x[:, None] - means[None, :]) ** 2, axis=0)
                / component_mass
            )
            variances = np.maximum(variances, 1e-4)
            if abs(log_likelihood - previous_ll) < 1e-7 * (1.0 + abs(previous_ll)):
                break
            previous_ll = log_likelihood
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        weights = weights[order]
        n_parameters = 3 * n_components - 1
        bic = -2.0 * log_likelihood + n_parameters * math.log(x.size)
        candidates.append(
            {
                "n_components": n_components,
                "weights": weights,
                "means": means,
                "variances": variances,
                "log_likelihood": log_likelihood,
                "bic": bic,
            }
        )
    selected = min(candidates, key=lambda item: item["bic"])
    return {"selected": selected, "candidates": candidates}


def gmm_posterior_components(values: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    means = np.asarray(model["means"])
    variances = np.asarray(model["variances"])
    weights = np.asarray(model["weights"])
    log_probability = np.stack(
        [
            np.log(max(float(weight), 1e-12))
            - 0.5 * np.log(2.0 * math.pi * max(float(variance), 1e-4))
            - 0.5 * (x - mean) ** 2 / max(float(variance), 1e-4)
            for weight, mean, variance in zip(weights, means, variances)
        ],
        axis=1,
    )
    return np.argmax(log_probability, axis=1)


def gmm_summary(model_result: dict[str, Any]) -> dict[str, Any]:
    selected = model_result["selected"]
    means = np.asarray(selected["means"])
    variances = np.asarray(selected["variances"])
    weights = np.asarray(selected["weights"])
    separations = []
    for index in range(len(means) - 1):
        distance = math.sqrt(2.0) * abs(float(means[index + 1] - means[index]))
        denominator = math.sqrt(float(variances[index] + variances[index + 1]))
        separations.append(distance / max(denominator, 1e-12))
    cleanly_separable = (
        len(means) > 1
        and bool(separations)
        and min(separations) >= 2.0
        and float(np.min(weights)) >= 0.05
    )
    return {
        "selected_components": int(selected["n_components"]),
        "bic_by_components": {
            str(candidate["n_components"]): float(candidate["bic"])
            for candidate in model_result["candidates"]
        },
        "components": [
            {
                "component": index + 1,
                "weight": float(weights[index]),
                "median_area_px2": float(math.exp(means[index])),
                "log_area_sd": float(math.sqrt(variances[index])),
            }
            for index in range(len(means))
        ],
        "adjacent_ashman_d": [float(value) for value in separations],
        "cleanly_separable_modes": cleanly_separable,
    }


def clean_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for field in row:
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
        fieldnames = ordered
    fieldnames = list(fieldnames)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_csv_value(row.get(field, "")) for field in fieldnames})
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def summarize_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[field])] += 1
    return dict(sorted(counts.items()))


def blossom_svg(cx: int, cy: int) -> str:
    petals = []
    colors = [PALETTE["blue"], PALETTE["gold"], PALETTE["orange"], PALETTE["olive"], PALETTE["pink"]]
    for index, color in enumerate(colors):
        angle = 2.0 * math.pi * index / len(colors) - math.pi / 2.0
        petals.append(
            f'<circle cx="{cx + 8 * math.cos(angle):.1f}" cy="{cy + 8 * math.sin(angle):.1f}" '
            f'r="4.2" fill="{color}" fill-opacity="0.72"/>'
        )
    petals.append(f'<circle cx="{cx}" cy="{cy}" r="3.2" fill="{PALETTE["ink"]}"/>')
    return "".join(petals)


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{body}{blossom_svg(width - 28, 26)}</svg>\n"
    )


def save_horizontal_bar_svg(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str] | None = None,
    value_suffix: str = "",
    value_decimals: int = 0,
    width: int = 960,
) -> None:
    row_height = 34
    height = 115 + row_height * len(labels) + 45
    left = 260
    right = 105
    top = 100
    chart_width = width - left - right
    maximum = max(values) if values else 1.0
    maximum = max(maximum * 1.10, 1e-9)
    body = [
        f'<text x="24" y="34" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="{PALETTE["ink"]}">{html.escape(title)}</text>',
        f'<text x="24" y="60" font-family="Arial,sans-serif" font-size="13" fill="{PALETTE["muted"]}">{html.escape(subtitle)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_height
        bar_width = chart_width * value / maximum
        color = colors[index] if colors else PALETTE["blue"]
        body.append(
            f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="{PALETTE["ink"]}">{html.escape(label)}</text>'
        )
        body.append(f'<rect x="{left}" y="{y}" width="{chart_width}" height="20" fill="#eef1f4" rx="2"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="20" fill="{color}" rx="2"/>')
        body.append(
            f'<text x="{left + bar_width + 7:.2f}" y="{y + 15}" font-family="ui-monospace,monospace" font-size="11" fill="{PALETTE["ink"]}">{value:,.{value_decimals}f}{html.escape(value_suffix)}</text>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(width, height, "".join(body), title))


def save_histogram_svg(
    path: Path,
    values: np.ndarray,
    title: str,
    subtitle: str,
    q25: float,
    q75: float,
    width: int = 960,
    height: int = 520,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    bins = np.geomspace(float(values.min()), float(values.max()), 36)
    counts, edges = np.histogram(values, bins=bins)
    left, right, top, bottom = 80, 35, 92, 70
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_count = max(int(counts.max()), 1)
    log_min, log_max = math.log10(edges[0]), math.log10(edges[-1])

    def x_coord(value: float) -> float:
        return left + chart_width * (math.log10(value) - log_min) / max(log_max - log_min, 1e-12)

    body = [
        f'<text x="24" y="34" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="{PALETTE["ink"]}">{html.escape(title)}</text>',
        f'<text x="24" y="60" font-family="Arial,sans-serif" font-size="13" fill="{PALETTE["muted"]}">{html.escape(subtitle)}</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - fraction)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        body.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="ui-monospace,monospace" font-size="11" fill="{PALETTE["muted"]}">{int(max_count * fraction):,}</text>'
        )
    for count, edge0, edge1 in zip(counts, edges[:-1], edges[1:]):
        x0, x1 = x_coord(float(edge0)), x_coord(float(edge1))
        bar_height = chart_height * count / max_count
        body.append(
            f'<rect x="{x0 + 0.5:.2f}" y="{top + chart_height - bar_height:.2f}" width="{max(x1 - x0 - 1.0, 0.5):.2f}" height="{bar_height:.2f}" fill="{PALETTE["blue"]}"/>'
        )
    for value, color, label in ((q25, PALETTE["gold"], "Q25"), (q75, PALETTE["orange"], "Q75")):
        x = x_coord(value)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + chart_height}" stroke="{color}" stroke-width="2" stroke-dasharray="6 4"/>')
        body.append(
            f'<text x="{x + 4:.2f}" y="{top + 14}" font-family="Arial,sans-serif" font-size="11" fill="{color}">{label} {value:.1f}</text>'
        )
    for exponent in range(math.floor(log_min), math.ceil(log_max) + 1):
        value = 10.0**exponent
        if edges[0] <= value <= edges[-1]:
            x = x_coord(value)
            body.append(f'<line x1="{x:.2f}" y1="{top + chart_height}" x2="{x:.2f}" y2="{top + chart_height + 6}" stroke="{PALETTE["ink"]}"/>')
            body.append(
                f'<text x="{x:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="11" fill="{PALETTE["ink"]}">{value:g}</text>'
            )
    body.append(
        f'<text x="{left + chart_width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="{PALETTE["ink"]}">Nuclear area (px², log scale)</text>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(width, height, "".join(body), title))


def save_grouped_bar_svg(
    path: Path,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    width: int = 960,
    height: int = 520,
) -> None:
    left, right, top, bottom = 85, 35, 105, 105
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = max((value for _, values, _ in series for value in values), default=1.0)
    maximum = max(maximum, 1.0)
    group_width = chart_width / max(len(categories), 1)
    bar_width = min(42.0, group_width * 0.72 / max(len(series), 1))
    body = [
        f'<text x="24" y="34" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="{PALETTE["ink"]}">{html.escape(title)}</text>',
        f'<text x="24" y="60" font-family="Arial,sans-serif" font-size="13" fill="{PALETTE["muted"]}">{html.escape(subtitle)}</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - fraction)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="{PALETTE["grid"]}"/>')
        body.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="ui-monospace,monospace" font-size="11" fill="{PALETTE["muted"]}">{int(maximum * fraction):,}</text>'
        )
    for category_index, category in enumerate(categories):
        group_center = left + group_width * (category_index + 0.5)
        group_total_width = bar_width * len(series)
        for series_index, (_, values, color) in enumerate(series):
            value = values[category_index]
            bar_height = chart_height * value / maximum
            x = group_center - group_total_width / 2 + series_index * bar_width
            body.append(
                f'<rect x="{x + 2:.2f}" y="{top + chart_height - bar_height:.2f}" width="{bar_width - 4:.2f}" height="{bar_height:.2f}" fill="{color}" rx="2"/>'
            )
            body.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{top + chart_height - bar_height - 5:.2f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10" fill="{PALETTE["ink"]}">{value:,.0f}</text>'
            )
        body.append(
            f'<text x="{group_center:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="{PALETTE["ink"]}">{html.escape(category.replace("_", " "))}</text>'
        )
    legend_x = left
    for name, _, color in series:
        body.append(f'<rect x="{legend_x}" y="{height - 46}" width="14" height="14" fill="{color}"/>')
        body.append(
            f'<text x="{legend_x + 20}" y="{height - 34}" font-family="Arial,sans-serif" font-size="12" fill="{PALETTE["ink"]}">{html.escape(name)}</text>'
        )
        legend_x += 145
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(width, height, "".join(body), title))


def render_qc_overlay(
    output_path: Path,
    key: str,
    combined_raw: np.ndarray,
    combined_mask: np.ndarray,
    bf_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    nucleus_rows: list[dict[str, Any]],
    overlay_alpha: float,
) -> None:
    base = robust_uint8_rgb(combined_raw).astype(np.float64)
    max_nucleus = int(nuclear_mask.max())
    color_lut = np.zeros((max_nucleus + 1, 3), dtype=np.uint8)
    for row in nucleus_rows:
        color_lut[int(row["nucleus_id"])] = OVERLAY_COLORS.get(
            str(row["overall_alignment_status"]),
            (215, 169, 40),
        )
    nuclear_pixels = nuclear_mask > 0
    colors = color_lut[nuclear_mask]
    base[nuclear_pixels] = (1.0 - overlay_alpha) * base[nuclear_pixels] + overlay_alpha * colors[nuclear_pixels]

    combined_boundary = cv2.dilate(label_boundary(combined_mask).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    bf_boundary = cv2.dilate(label_boundary(bf_mask).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    nuclear_boundary = label_boundary(nuclear_mask)
    base[combined_boundary] = (0, 190, 225)
    base[bf_boundary] = (255, 190, 35)
    base[nuclear_boundary] = colors[nuclear_boundary]
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    header_height = 86
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 9), f"{key} | Nuclei-to-cell alignment QC", fill=(32, 37, 43), font=font)
    draw.line((12, 31, 45, 31), fill=(0, 190, 225), width=3)
    draw.text((52, 25), "Combined boundary", fill=(32, 37, 43), font=font)
    draw.line((185, 31, 218, 31), fill=(255, 190, 35), width=3)
    draw.text((225, 25), "BF boundary", fill=(32, 37, 43), font=font)
    legend_items = [
        ("matched_consensus", "strict consensus"),
        ("matched_one_way", "one-way match"),
        ("crosses_both", "crosses boundary"),
        ("bf_combined_assignment_conflict", "assignment conflict"),
        ("unmatched_both", "unmatched"),
        ("low_containment", "low containment"),
    ]
    x = 12
    for status, label in legend_items:
        color = OVERLAY_COLORS[status]
        draw.rectangle((x, 53, x + 12, 65), fill=color)
        draw.text((x + 18, 52), label, fill=(32, 37, 43), font=font)
        x += 148
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output_path)


def field_summary(
    key: str,
    nucleus_rows: list[dict[str, Any]],
    combined_cells: list[dict[str, Any]],
    bf_cells: list[dict[str, Any]],
    min_cell_area: float,
    high_density: bool,
) -> dict[str, Any]:
    total_nuclei = len(nucleus_rows)
    strict = sum(bool(row["strict_consensus"]) for row in nucleus_rows)
    acceptable = sum(bool(row["acceptable_alignment"]) for row in nucleus_rows)
    actionable = sum(bool(row["actionable_mismatch"]) for row in nucleus_rows)
    details = key_details(key)

    def summarize_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
        countable = [row for row in rows if float(row["cell_area_px2"]) >= min_cell_area]
        statuses = summarize_counts(countable, "nuclear_status")
        ratios = np.asarray([float(row["nucleus_to_cell_area_ratio"]) for row in countable], dtype=float)
        cytoplasm_ratios = np.asarray(
            [
                float(row["nucleus_to_cytoplasm_area_ratio"])
                for row in countable
                if math.isfinite(float(row["nucleus_to_cytoplasm_area_ratio"]))
            ],
            dtype=float,
        )
        core_cytoplasm_ratios = np.asarray(
            [
                float(row["core_nucleus_to_cytoplasm_area_ratio"])
                for row in countable
                if row.get("core_nucleus_to_cytoplasm_area_ratio", "") != ""
                and math.isfinite(float(row["core_nucleus_to_cytoplasm_area_ratio"]))
            ],
            dtype=float,
        )
        return {
            "n_cells": len(rows),
            "n_countable_cells": len(countable),
            "n_confirmed_multinucleated": statuses.get("confirmed_multinucleated", 0),
            "n_candidate_multinucleated": statuses.get("candidate_multinucleated", 0),
            "n_single_nucleus": statuses.get("single_nucleus", 0),
            "n_no_confident_nucleus": statuses.get("no_confident_nucleus", 0),
            "median_nucleus_to_cell_ratio": float(np.median(ratios)) if ratios.size else 0.0,
            "median_nucleus_to_cytoplasm_ratio": (
                float(np.median(cytoplasm_ratios)) if cytoplasm_ratios.size else 0.0
            ),
            "median_core_nucleus_to_cytoplasm_ratio": (
                float(np.median(core_cytoplasm_ratios)) if core_cytoplasm_ratios.size else 0.0
            ),
        }

    combined_summary = summarize_cells(combined_cells)
    bf_summary = summarize_cells(bf_cells)
    areas = np.asarray([float(row["nuclear_area_px2"]) for row in nucleus_rows], dtype=float)
    row: dict[str, Any] = {
        "key": key,
        **details,
        "high_density": high_density,
        "n_nuclei": total_nuclei,
        "median_nuclear_area_px2": float(np.median(areas)) if areas.size else 0.0,
        "n_strict_consensus": strict,
        "strict_consensus_rate": strict / total_nuclei if total_nuclei else 0.0,
        "n_acceptable_alignment": acceptable,
        "acceptable_alignment_rate": acceptable / total_nuclei if total_nuclei else 0.0,
        "n_actionable_mismatch": actionable,
        "actionable_mismatch_rate": actionable / total_nuclei if total_nuclei else 0.0,
        "n_combined_cross_boundary": sum(
            row["combined_match_category"] == "crosses_cells" for row in nucleus_rows
        ),
        "n_bf_cross_boundary": sum(row["bf_match_category"] == "crosses_cells" for row in nucleus_rows),
        "n_combined_unmatched": sum(row["combined_match_category"] == "unmatched" for row in nucleus_rows),
        "n_bf_unmatched": sum(row["bf_match_category"] == "unmatched" for row in nucleus_rows),
    }
    for prefix, summary in (("combined", combined_summary), ("bf", bf_summary)):
        for name, value in summary.items():
            row[f"{prefix}_{name}"] = value
    return row


def make_summary_markdown(summary: dict[str, Any]) -> str:
    nuclei = summary["nuclei"]
    alignment = summary["alignment"]
    combined_cells = summary["cells"]["Combined"]
    bf_cells = summary["cells"]["Brightfield"]
    size = nuclei["size_model"]
    core = nuclei["intensity_supported_core"]
    normal_density = summary["density_strata"]["non_high_density"]
    high_density = summary["density_strata"]["high_density"]
    concordance = summary["cross_method_cell_concordance"]
    unit_note = (
        f"Physical calibration used: {summary['units']['pixel_size_um']} µm/pixel."
        if summary["units"]["pixel_size_um"] is not None
        else "No valid microscopy pixel-size calibration was available; size is reported in pixels and pixels²."
    )
    size_interpretation = (
        f"The log-area model selected {size['selected_components']} components and the adjacent modes meet the preset separation criterion."
        if size["cleanly_separable_modes"]
        else (
            f"The log-area model selected {size['selected_components']} components, but they do not meet the preset clean-separation criterion; "
            "small/medium/large quantile bins are descriptive rather than distinct biological classes."
        )
    )
    core_note = (
        f"Intensity-supported cores were available for {core['n_nuclei_with_core']:,} nuclei; "
        f"their median area was {core['area_px2']['median']:.1f} px² and median core/extent fraction was "
        f"{core['core_to_extent_fraction']['median']:.1%}."
        if core["n_nuclei_with_core"]
        else "No intensity-supported nucleus-core masks were available for this run."
    )
    return f"""# Nuclear morphometry and cell-mask alignment summary

Generated from `{summary['source']['run_root']}` on {summary['generated_at']}.

## Technical summary

- The analysis covered **{summary['inventory']['n_fields']:,} fields**, **{nuclei['total']:,} nuclei**, **{combined_cells['countable_cells']:,} countable Combined cells**, and **{bf_cells['countable_cells']:,} countable BF cells**.
- **{alignment['strict_consensus_count']:,} nuclei ({alignment['strict_consensus_rate']:.2%})** were fully contained in both cell masks and assigned to reciprocal-best BF/Combined cell pairs.
- **{alignment['actionable_mismatch_count']:,} nuclei ({alignment['actionable_mismatch_rate']:.2%})** had an actionable alignment issue: unmatched or low-overlap assignment, a cell-boundary crossing, or a BF/Combined cell-pair conflict.
- Combined masks contained **{combined_cells['confirmed_multinucleated_cells']:,} confirmed multinucleated cells ({combined_cells['confirmed_multinucleated_rate']:.2%})**; BF masks contained **{bf_cells['confirmed_multinucleated_cells']:,} ({bf_cells['confirmed_multinucleated_rate']:.2%})**.
- Median extent-based nucleus-to-cytoplasm area ratio was **{combined_cells['median_nucleus_to_cytoplasm_ratio']:.3f}** by Combined masks and **{bf_cells['median_nucleus_to_cytoplasm_ratio']:.3f}** by BF masks; the corresponding exposure-robust core-based values were **{combined_cells['median_core_nucleus_to_cytoplasm_ratio']:.3f}** and **{bf_cells['median_core_nucleus_to_cytoplasm_ratio']:.3f}**.
- Among **{concordance['reciprocal_countable_cell_pairs']:,} reciprocal BF/Combined cell pairs**, exact four-class nuclear-status agreement was **{concordance['exact_status_concordance_rate']:.2%}**. Confirmed-multinucleation Jaccard agreement was **{concordance['confirmed_multinucleation_jaccard']:.2%}**.
- Median nuclear area was **{nuclei['area_px2']['median']:.1f} px²** (Q25–Q75: {nuclei['area_px2']['q25']:.1f}–{nuclei['area_px2']['q75']:.1f} px²). {size_interpretation}
- {core_note}
- Density strongly modified reliability: non-high-density fields had **{normal_density['strict_consensus_rate']:.2%}** strict consensus and **{normal_density['actionable_mismatch_rate']:.2%}** actionable mismatches, versus **{high_density['strict_consensus_rate']:.2%}** and **{high_density['actionable_mismatch_rate']:.2%}** in high-density fields.

## Metric definitions

- A high-confidence nucleus is at least {summary['parameters']['min_nucleus_area']} px², does not touch the image edge, has solidity ≥0.60, is ≥{summary['parameters']['high_containment']:.0%} contained in one cell mask, and its centroid lies in that same cell.
- Strict consensus requires high-containment assignments in both Combined and BF plus a reciprocal-best overlap between the two assigned cell masks.
- Confirmed multinucleation requires at least two high-confidence nuclei assigned to one cell. Candidate multinucleation requires at least two moderate (≥{summary['parameters']['moderate_containment']:.0%}) assignments.
- Nuclear-to-cell ratio is nuclear pixels inside the cell divided by cell-mask area. Nuclear-to-cytoplasm ratio is nuclear pixels divided by non-nuclear cell-mask pixels.
- {unit_note}

## Primary outputs

- `nucleus_features.csv`: per-nucleus size, shape, edge, and fluorescence-intensity measurements.
- `nucleus_cell_mapping.csv`: Combined/BF overlap, containment, cell-pair consensus, confidence, and failure category.
- `cell_nuclear_summary.csv`: per-cell nucleus counts, multinucleation status, nuclear-to-cell ratio, and nuclear-to-cytoplasm ratio.
- `segmentation_mismatch.csv`: every non-strict nucleus, with actionable versus review-only flags.
- `field_summary.csv`: field-level rates and cell/nucleus totals.
- `density_stratified_summary.csv`: reliability, multinucleation, and nuclear-to-cell ratios split by the production high-density call.
- `cell_pair_nuclear_status_concordance.csv`: matched BF/Combined cell pairs and cross-method multinucleation agreement.
- `qc_overlays/`: Combined images with Combined, BF, and status-colored nuclear boundaries.

## Limitations and interpretation

- These are agreement and morphology measurements between predicted masks, not accuracy estimates against manually annotated ground truth.
- BF and Combined are two segmentations of related cell morphology; reciprocal-best overlap is used because their instance label IDs are unrelated.
- Cells at the image border and nuclei below the minimum-area threshold remain in the detailed tables but are excluded from confirmed multinucleation calls.
- A high nuclear-to-cell ratio can indicate biology, an undersized cell mask, an oversized nuclear mask, or both; the overlay and BF/Combined comparison should be reviewed together.

## Recommended calibration step

Review a stratified sample of strict matches, boundary crossings, unmatched nuclei, BF/Combined conflicts, and multinucleated calls in `qc_overlays/`. If a calibrated pixel size or manual truth set becomes available, rerun with `--pixel-size-um` and tune the containment thresholds against those annotations.
"""


def main() -> int:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.input_root = args.input_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    masks_by_profile = {profile: index_mask_files(args.run_root, profile) for profile in PROFILE_STEMS}
    nucleus_core_masks = index_nucleus_core_files(args.run_root)
    raw_by_profile = {profile: index_raw_files(args.input_root, profile) for profile in PROFILE_STEMS}
    key_sets = {profile: set(paths) for profile, paths in masks_by_profile.items()}
    all_mask_keys = set.union(*key_sets.values())
    common_mask_keys = set.intersection(*key_sets.values())
    if common_mask_keys != all_mask_keys:
        missing = {
            profile: sorted(all_mask_keys - keys)
            for profile, keys in key_sets.items()
            if all_mask_keys - keys
        }
        raise RuntimeError(f"Mask key mismatch across profiles: {missing}")
    missing_raw = {
        profile: sorted(common_mask_keys - set(raw_by_profile[profile]))
        for profile in PROFILE_STEMS
        if common_mask_keys - set(raw_by_profile[profile])
    }
    if missing_raw:
        raise RuntimeError(f"Missing raw images for mask keys: {missing_raw}")
    keys = sorted(common_mask_keys)
    if args.max_fields is not None:
        keys = keys[: args.max_fields]
    if not keys:
        raise RuntimeError("No complete Nuclei/Brightfield/Combined fields found")

    classification = load_combined_classification(args.run_root)
    density_calls = load_density_calls(args.run_root)
    all_nucleus_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_field_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for field_index, key in enumerate(keys, start=1):
        print(f"[{field_index}/{len(keys)}] {key}", flush=True)
        nuclear_mask = read_mask(masks_by_profile["Nuclei"][key])
        nucleus_core_mask = read_mask(nucleus_core_masks[key]) if key in nucleus_core_masks else None
        combined_mask = read_mask(masks_by_profile["Combined"][key])
        bf_mask = read_mask(masks_by_profile["Brightfield"][key])
        if not (nuclear_mask.shape == combined_mask.shape == bf_mask.shape):
            raise ValueError(
                f"Mask shape mismatch for {key}: nuclei={nuclear_mask.shape}, "
                f"combined={combined_mask.shape}, bf={bf_mask.shape}"
            )
        if nucleus_core_mask is not None:
            if nucleus_core_mask.shape != nuclear_mask.shape:
                raise ValueError(
                    f"Nucleus core/extent shape mismatch for {key}: "
                    f"core={nucleus_core_mask.shape}, extent={nuclear_mask.shape}"
                )
            invalid_core = (nucleus_core_mask > 0) & (nucleus_core_mask != nuclear_mask)
            if np.any(invalid_core):
                raise ValueError(
                    f"Nucleus core pixels do not preserve extent instance IDs for {key}: "
                    f"{int(np.count_nonzero(invalid_core))} invalid pixels"
                )
        nuclear_raw = as_gray(read_image(raw_by_profile["Nuclei"][key]))
        combined_raw = read_image(raw_by_profile["Combined"][key])
        high_density_field = bool(density_calls.get(key, False))
        if nuclear_raw.shape != nuclear_mask.shape or combined_raw.shape[:2] != combined_mask.shape:
            raise ValueError(
                f"Raw/mask shape mismatch for {key}: nuclei_raw={nuclear_raw.shape}, "
                f"nuclei_mask={nuclear_mask.shape}, combined_raw={combined_raw.shape}, "
                f"combined_mask={combined_mask.shape}"
            )

        nuclear_stats = compute_label_stats(nuclear_mask)
        nucleus_core_stats = compute_label_stats(nucleus_core_mask) if nucleus_core_mask is not None else None
        combined_stats = compute_label_stats(combined_mask)
        bf_stats = compute_label_stats(bf_mask)
        if nuclear_stats.count != int(nuclear_mask.max()):
            raise ValueError(f"Non-contiguous nuclear labels in {masks_by_profile['Nuclei'][key]}")
        combined_matches = compute_nucleus_target_matches(
            nuclear_mask, combined_mask, nuclear_stats, combined_stats, args
        )
        bf_matches = compute_nucleus_target_matches(nuclear_mask, bf_mask, nuclear_stats, bf_stats, args)
        pair_rows, pair_lookup, combined_to_bf, bf_to_combined = compute_cell_correspondence(
            combined_mask, bf_mask, combined_stats, bf_stats, key
        )
        all_pair_rows.extend(pair_rows)
        intensity_means, intensity_stds = intensity_arrays(nuclear_raw, nuclear_mask, nuclear_stats)

        field_nucleus_rows: list[dict[str, Any]] = []
        details = key_details(key)
        for nuclear_label_value in nuclear_stats.labels:
            nucleus_id = int(nuclear_label_value)
            area = float(nuclear_stats.areas[nucleus_id])
            core_area = (
                float(nucleus_core_stats.areas[nucleus_id])
                if nucleus_core_stats is not None
                and nucleus_id < len(nucleus_core_stats.areas)
                and nucleus_core_stats.areas[nucleus_id] > 0
                else float("nan")
            )
            row: dict[str, Any] = {
                "key": key,
                **details,
                "high_density_field": high_density_field,
                "nucleus_id": nucleus_id,
                "nuclear_area_px2": area,
                "nucleus_core_present": math.isfinite(core_area),
                "nucleus_core_area_px2": core_area if math.isfinite(core_area) else "",
                "nucleus_core_equivalent_diameter_px": (
                    math.sqrt(4.0 * core_area / math.pi) if math.isfinite(core_area) else ""
                ),
                "nucleus_core_to_extent_fraction": core_area / area if math.isfinite(core_area) else "",
                "nucleus_edge_sensitive_area_px2": area - core_area if math.isfinite(core_area) else "",
                "equivalent_diameter_px": math.sqrt(4.0 * area / math.pi),
                "major_axis_px": float(nuclear_stats.major_axis[nucleus_id]),
                "minor_axis_px": float(nuclear_stats.minor_axis[nucleus_id]),
                "axis_ratio": float(nuclear_stats.major_axis[nucleus_id])
                / max(float(nuclear_stats.minor_axis[nucleus_id]), 1e-12),
                "eccentricity": float(nuclear_stats.eccentricity[nucleus_id]),
                "orientation_deg": float(nuclear_stats.orientation_deg[nucleus_id]),
                "centroid_y": float(nuclear_stats.centroids[nucleus_id, 0]),
                "centroid_x": float(nuclear_stats.centroids[nucleus_id, 1]),
                "bbox_y0": int(nuclear_stats.y0[nucleus_id]),
                "bbox_x0": int(nuclear_stats.x0[nucleus_id]),
                "bbox_y1": int(nuclear_stats.y1[nucleus_id]),
                "bbox_x1": int(nuclear_stats.x1[nucleus_id]),
                "border_touch": bool(nuclear_stats.border_touch[nucleus_id]),
            }
            row.update(object_shape_metrics(nuclear_mask, nuclear_stats, nucleus_id))
            row.update(
                object_intensity_metrics(
                    nuclear_raw,
                    nuclear_mask,
                    nuclear_stats,
                    nucleus_id,
                    intensity_means[nucleus_id],
                    intensity_stds[nucleus_id],
                )
            )
            row["fragment_suspect"] = area < args.min_nucleus_area
            row["shape_irregular_flag"] = row["solidity"] < 0.75 or row["circularity"] < 0.35
            row["low_intensity_flag"] = (
                math.isfinite(float(row["local_intensity_snr"]))
                and float(row["local_intensity_snr"]) < 1.0
            )
            row["quality_nucleus"] = (
                area >= args.min_nucleus_area
                and not row["border_touch"]
                and float(row["solidity"]) >= 0.60
            )
            row["morphology_quality_score"] = morphology_score(row, args.min_nucleus_area)
            if args.pixel_size_um is not None:
                row["nuclear_area_um2"] = area * args.pixel_size_um**2
                row["equivalent_diameter_um"] = row["equivalent_diameter_px"] * args.pixel_size_um
                row["major_axis_um"] = row["major_axis_px"] * args.pixel_size_um
                row["minor_axis_um"] = row["minor_axis_px"] * args.pixel_size_um
            else:
                row["nuclear_area_um2"] = ""
                row["equivalent_diameter_um"] = ""
                row["major_axis_um"] = ""
                row["minor_axis_um"] = ""

            combined_match = combined_matches[nucleus_id]
            bf_match = bf_matches[nucleus_id]
            for prefix, match in (("combined", combined_match), ("bf", bf_match)):
                for field_name, value in match.items():
                    if field_name == "primary_cell_id":
                        row[f"{prefix}_cell_id"] = value
                    else:
                        row[f"{prefix}_{field_name}"] = value
            consensus, pair = classify_consensus(
                int(combined_match["primary_cell_id"]),
                int(bf_match["primary_cell_id"]),
                pair_lookup,
            )
            row["bf_combined_consensus"] = consensus
            row["assigned_cell_pair_overlap_px"] = int(pair["overlap_px"]) if pair else 0
            row["assigned_cell_pair_iou"] = float(pair["cell_iou"]) if pair else 0.0
            row["assigned_cell_pair_overlap_of_smaller_fraction"] = (
                float(pair["overlap_of_smaller_cell_fraction"]) if pair else 0.0
            )
            status = overall_alignment_status(combined_match, bf_match, consensus)
            row["overall_alignment_status"] = status
            row["strict_consensus"] = status == "matched_consensus"
            row["acceptable_alignment"] = (
                combined_match["match_category"] != "crosses_cells"
                and bf_match["match_category"] != "crosses_cells"
                and float(combined_match["primary_nucleus_fraction"]) >= args.moderate_containment
                and float(bf_match["primary_nucleus_fraction"]) >= args.moderate_containment
                and consensus in {"reciprocal_best", "one_way_best"}
            )
            row["actionable_mismatch"] = (
                combined_match["match_category"] in {"unmatched", "crosses_cells"}
                or bf_match["match_category"] in {"unmatched", "crosses_cells"}
                or consensus in {"conflict_no_overlap", "overlapping_nonbest"}
                or float(combined_match["primary_nucleus_fraction"]) < args.moderate_containment
                or float(bf_match["primary_nucleus_fraction"]) < args.moderate_containment
            )
            row["review_recommended"] = not row["strict_consensus"]
            overall_score = (
                0.20 * float(row["morphology_quality_score"])
                + 0.35 * float(combined_match["assignment_score"])
                + 0.30 * float(bf_match["assignment_score"])
                + 0.15 * consensus_score(consensus)
            )
            row["overall_confidence_score"] = overall_score
            row["overall_confidence_category"] = (
                "high" if overall_score >= 0.80 else "medium" if overall_score >= 0.60 else "low"
            )
            field_nucleus_rows.append(row)

        combined_cell_rows = cell_summary_rows(
            key,
            "Combined",
            combined_mask,
            combined_stats,
            nuclear_mask,
            nucleus_core_mask,
            field_nucleus_rows,
            "combined",
            args,
            classification,
            pair_lookup,
            combined_to_bf,
            bf_to_combined,
        )
        bf_cell_rows = cell_summary_rows(
            key,
            "Brightfield",
            bf_mask,
            bf_stats,
            nuclear_mask,
            nucleus_core_mask,
            field_nucleus_rows,
            "bf",
            args,
            classification,
            pair_lookup,
            combined_to_bf,
            bf_to_combined,
        )
        for cell_row in combined_cell_rows:
            cell_row["high_density_field"] = high_density_field
        for cell_row in bf_cell_rows:
            cell_row["high_density_field"] = high_density_field
        all_nucleus_rows.extend(field_nucleus_rows)
        all_cell_rows.extend(combined_cell_rows)
        all_cell_rows.extend(bf_cell_rows)
        all_field_rows.append(
            field_summary(
                key,
                field_nucleus_rows,
                combined_cell_rows,
                bf_cell_rows,
                args.min_cell_area,
                high_density_field,
            )
        )
        inventory_rows.append(
            {
                "key": key,
                "high_density": high_density_field,
                "height": nuclear_mask.shape[0],
                "width": nuclear_mask.shape[1],
                "n_nuclei": nuclear_stats.count,
                "n_nucleus_cores": nucleus_core_stats.count if nucleus_core_stats is not None else 0,
                "n_combined_cells": combined_stats.count,
                "n_bf_cells": bf_stats.count,
                "nuclear_mask_fraction": float(np.mean(nuclear_mask > 0)),
                "nucleus_core_mask_fraction": (
                    float(np.mean(nucleus_core_mask > 0)) if nucleus_core_mask is not None else ""
                ),
                "combined_mask_fraction": float(np.mean(combined_mask > 0)),
                "bf_mask_fraction": float(np.mean(bf_mask > 0)),
                "nuclei_mask_path": str(masks_by_profile["Nuclei"][key]),
                "nucleus_core_mask_path": str(nucleus_core_masks[key]) if key in nucleus_core_masks else "",
                "combined_mask_path": str(masks_by_profile["Combined"][key]),
                "bf_mask_path": str(masks_by_profile["Brightfield"][key]),
                "nuclei_raw_path": str(raw_by_profile["Nuclei"][key]),
                "combined_raw_path": str(raw_by_profile["Combined"][key]),
                "bf_raw_path": str(raw_by_profile["Brightfield"][key]),
            }
        )
        if not args.skip_overlays:
            render_qc_overlay(
                args.out_dir / "qc_overlays" / f"{key}_nuclear_cell_alignment_qc.png",
                key,
                combined_raw,
                combined_mask,
                bf_mask,
                nuclear_mask,
                field_nucleus_rows,
                args.overlay_alpha,
            )

    eligible_size_rows = [
        row
        for row in all_nucleus_rows
        if not row["fragment_suspect"] and not row["border_touch"]
    ]
    eligible_areas = np.asarray([float(row["nuclear_area_px2"]) for row in eligible_size_rows], dtype=float)
    q25, q75 = np.quantile(eligible_areas, (0.25, 0.75))
    gmm_result = fit_gmm_1d(np.log(eligible_areas), max_components=3)
    selected_gmm = gmm_result["selected"]
    components = gmm_posterior_components(np.log(eligible_areas), selected_gmm)
    component_names = {
        1: ["single_mode"],
        2: ["smaller_mode", "larger_mode"],
        3: ["small_mode", "medium_mode", "large_mode"],
    }[int(selected_gmm["n_components"])]
    for row, component in zip(eligible_size_rows, components):
        row["gmm_size_component"] = int(component) + 1
        row["gmm_size_class"] = component_names[int(component)]
    for row in all_nucleus_rows:
        area = float(row["nuclear_area_px2"])
        row["size_quantile_class"] = "small" if area < q25 else "large" if area > q75 else "medium"
        row.setdefault("gmm_size_component", "")
        row.setdefault("gmm_size_class", "excluded_fragment_or_border")

    rows_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_nucleus_rows:
        rows_by_key[str(row["key"])].append(row)
    for rows in rows_by_key.values():
        log_area = np.log(np.asarray([float(row["nuclear_area_px2"]) for row in rows], dtype=float))
        median = float(np.median(log_area))
        mad = float(np.median(np.abs(log_area - median)))
        scale = max(1.4826 * mad, 1e-9)
        for row, value in zip(rows, log_area):
            row["within_field_log_area_robust_z"] = (float(value) - median) / scale

    nucleus_feature_fields = [
        "key", "well", "site", "day", "hour", "minute", "elapsed_hours", "high_density_field", "nucleus_id",
        "nuclear_area_px2", "nuclear_area_um2", "equivalent_diameter_px", "equivalent_diameter_um",
        "nucleus_core_present", "nucleus_core_area_px2", "nucleus_core_equivalent_diameter_px",
        "nucleus_core_to_extent_fraction", "nucleus_edge_sensitive_area_px2",
        "major_axis_px", "major_axis_um", "minor_axis_px", "minor_axis_um", "axis_ratio",
        "eccentricity", "orientation_deg", "perimeter_px", "convex_area_px2", "solidity", "extent",
        "circularity", "centroid_y", "centroid_x", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "border_touch", "nuclear_intensity_mean", "nuclear_intensity_std", "nuclear_intensity_p10",
        "nuclear_intensity_median", "nuclear_intensity_p90", "local_background_median",
        "local_background_mad", "local_intensity_snr", "fragment_suspect", "shape_irregular_flag",
        "low_intensity_flag", "quality_nucleus", "morphology_quality_score", "size_quantile_class",
        "gmm_size_component", "gmm_size_class", "within_field_log_area_robust_z",
    ]
    mapping_fields = [
        "key", "well", "site", "day", "hour", "minute", "elapsed_hours", "high_density_field", "nucleus_id",
        "nuclear_area_px2", "nucleus_core_area_px2", "nucleus_core_to_extent_fraction",
        "quality_nucleus", "combined_cell_id", "combined_primary_overlap_px",
        "combined_primary_nucleus_fraction", "combined_primary_cell_fraction", "combined_primary_iou",
        "combined_secondary_cell_id", "combined_secondary_overlap_px", "combined_secondary_nucleus_fraction",
        "combined_total_cell_overlap_px", "combined_total_cell_coverage_fraction", "combined_outside_cell_fraction",
        "combined_n_intersected_cells", "combined_n_significant_cells", "combined_centroid_cell_id",
        "combined_centroid_in_primary", "combined_centroid_boundary_distance_px", "combined_nearest_cell_id",
        "combined_nearest_cell_distance_px", "combined_match_category", "combined_assignment_score",
        "bf_cell_id", "bf_primary_overlap_px", "bf_primary_nucleus_fraction", "bf_primary_cell_fraction",
        "bf_primary_iou", "bf_secondary_cell_id", "bf_secondary_overlap_px", "bf_secondary_nucleus_fraction",
        "bf_total_cell_overlap_px", "bf_total_cell_coverage_fraction", "bf_outside_cell_fraction",
        "bf_n_intersected_cells", "bf_n_significant_cells", "bf_centroid_cell_id", "bf_centroid_in_primary",
        "bf_centroid_boundary_distance_px", "bf_nearest_cell_id", "bf_nearest_cell_distance_px",
        "bf_match_category", "bf_assignment_score", "bf_combined_consensus", "assigned_cell_pair_overlap_px",
        "assigned_cell_pair_iou", "assigned_cell_pair_overlap_of_smaller_fraction", "overall_alignment_status",
        "strict_consensus", "acceptable_alignment", "actionable_mismatch", "review_recommended",
        "overall_confidence_score", "overall_confidence_category",
    ]
    write_csv(args.out_dir / "nucleus_features.csv", all_nucleus_rows, nucleus_feature_fields)
    write_csv(args.out_dir / "nucleus_cell_mapping.csv", all_nucleus_rows, mapping_fields)
    write_csv(args.out_dir / "cell_nuclear_summary.csv", all_cell_rows)
    write_csv(args.out_dir / "cell_mask_correspondence.csv", all_pair_rows)
    write_csv(args.out_dir / "field_summary.csv", all_field_rows)
    write_csv(args.out_dir / "input_inventory.csv", inventory_rows)
    cross_method_rows = cross_method_nuclear_status_rows(all_cell_rows)
    write_csv(args.out_dir / "cell_pair_nuclear_status_concordance.csv", cross_method_rows)
    mismatch_rows = [row for row in all_nucleus_rows if row["review_recommended"]]
    write_csv(args.out_dir / "segmentation_mismatch.csv", mismatch_rows, mapping_fields)

    total_nuclei = len(all_nucleus_rows)
    strict_count = sum(bool(row["strict_consensus"]) for row in all_nucleus_rows)
    acceptable_count = sum(bool(row["acceptable_alignment"]) for row in all_nucleus_rows)
    actionable_count = sum(bool(row["actionable_mismatch"]) for row in all_nucleus_rows)

    cell_summary: dict[str, Any] = {}
    for profile in ("Combined", "Brightfield"):
        profile_rows = [row for row in all_cell_rows if row["cell_mask_profile"] == profile]
        countable = [row for row in profile_rows if row["cell_countable_by_area"]]
        statuses = summarize_counts(countable, "nuclear_status")
        confirmed = statuses.get("confirmed_multinucleated", 0)
        ratios = np.asarray([float(row["nucleus_to_cell_area_ratio"]) for row in countable], dtype=float)
        cytoplasm_ratios = np.asarray(
            [
                float(row["nucleus_to_cytoplasm_area_ratio"])
                for row in countable
                if math.isfinite(float(row["nucleus_to_cytoplasm_area_ratio"]))
            ],
            dtype=float,
        )
        core_cell_ratios = np.asarray(
            [
                float(row["core_nucleus_to_cell_area_ratio"])
                for row in countable
                if row.get("core_nucleus_to_cell_area_ratio", "") != ""
                and math.isfinite(float(row["core_nucleus_to_cell_area_ratio"]))
            ],
            dtype=float,
        )
        core_cytoplasm_ratios = np.asarray(
            [
                float(row["core_nucleus_to_cytoplasm_area_ratio"])
                for row in countable
                if row.get("core_nucleus_to_cytoplasm_area_ratio", "") != ""
                and math.isfinite(float(row["core_nucleus_to_cytoplasm_area_ratio"]))
            ],
            dtype=float,
        )
        cell_summary[profile] = {
            "all_cells": len(profile_rows),
            "countable_cells": len(countable),
            "status_counts": statuses,
            "confirmed_multinucleated_cells": confirmed,
            "confirmed_multinucleated_rate": confirmed / len(countable) if countable else 0.0,
            "median_nucleus_to_cell_ratio": float(np.median(ratios)) if ratios.size else 0.0,
            "q25_nucleus_to_cell_ratio": float(np.quantile(ratios, 0.25)) if ratios.size else 0.0,
            "q75_nucleus_to_cell_ratio": float(np.quantile(ratios, 0.75)) if ratios.size else 0.0,
            "median_nucleus_to_cytoplasm_ratio": (
                float(np.median(cytoplasm_ratios)) if cytoplasm_ratios.size else 0.0
            ),
            "q25_nucleus_to_cytoplasm_ratio": (
                float(np.quantile(cytoplasm_ratios, 0.25)) if cytoplasm_ratios.size else 0.0
            ),
            "q75_nucleus_to_cytoplasm_ratio": (
                float(np.quantile(cytoplasm_ratios, 0.75)) if cytoplasm_ratios.size else 0.0
            ),
            "median_core_nucleus_to_cell_ratio": (
                float(np.median(core_cell_ratios)) if core_cell_ratios.size else 0.0
            ),
            "median_core_nucleus_to_cytoplasm_ratio": (
                float(np.median(core_cytoplasm_ratios)) if core_cytoplasm_ratios.size else 0.0
            ),
        }

    density_strata: dict[str, Any] = {}
    density_summary_rows: list[dict[str, Any]] = []
    for high_density_value, stratum_name in ((False, "non_high_density"), (True, "high_density")):
        stratum_nuclei = [
            row for row in all_nucleus_rows if bool(row["high_density_field"]) == high_density_value
        ]
        stratum: dict[str, Any] = {
            "n_fields": sum(bool(row["high_density"]) == high_density_value for row in all_field_rows),
            "n_nuclei": len(stratum_nuclei),
            "strict_consensus_count": sum(bool(row["strict_consensus"]) for row in stratum_nuclei),
            "actionable_mismatch_count": sum(bool(row["actionable_mismatch"]) for row in stratum_nuclei),
        }
        stratum["strict_consensus_rate"] = (
            stratum["strict_consensus_count"] / stratum["n_nuclei"] if stratum["n_nuclei"] else 0.0
        )
        stratum["actionable_mismatch_rate"] = (
            stratum["actionable_mismatch_count"] / stratum["n_nuclei"] if stratum["n_nuclei"] else 0.0
        )
        for profile in ("Combined", "Brightfield"):
            cells = [
                row
                for row in all_cell_rows
                if row["cell_mask_profile"] == profile
                and row["cell_countable_by_area"]
                and bool(row["high_density_field"]) == high_density_value
            ]
            confirmed = sum(row["nuclear_status"] == "confirmed_multinucleated" for row in cells)
            ratios = np.asarray([float(row["nucleus_to_cell_area_ratio"]) for row in cells], dtype=float)
            cytoplasm_ratios = np.asarray(
                [
                    float(row["nucleus_to_cytoplasm_area_ratio"])
                    for row in cells
                    if math.isfinite(float(row["nucleus_to_cytoplasm_area_ratio"]))
                ],
                dtype=float,
            )
            core_cytoplasm_ratios = np.asarray(
                [
                    float(row["core_nucleus_to_cytoplasm_area_ratio"])
                    for row in cells
                    if row.get("core_nucleus_to_cytoplasm_area_ratio", "") != ""
                    and math.isfinite(float(row["core_nucleus_to_cytoplasm_area_ratio"]))
                ],
                dtype=float,
            )
            prefix = profile.lower()
            stratum[f"{prefix}_countable_cells"] = len(cells)
            stratum[f"{prefix}_confirmed_multinucleated_cells"] = confirmed
            stratum[f"{prefix}_confirmed_multinucleated_rate"] = confirmed / len(cells) if cells else 0.0
            stratum[f"{prefix}_median_nucleus_to_cell_ratio"] = float(np.median(ratios)) if ratios.size else 0.0
            stratum[f"{prefix}_median_nucleus_to_cytoplasm_ratio"] = (
                float(np.median(cytoplasm_ratios)) if cytoplasm_ratios.size else 0.0
            )
            stratum[f"{prefix}_median_core_nucleus_to_cytoplasm_ratio"] = (
                float(np.median(core_cytoplasm_ratios)) if core_cytoplasm_ratios.size else 0.0
            )
        density_strata[stratum_name] = stratum
        density_summary_rows.append({"density_stratum": stratum_name, **stratum})
    write_csv(args.out_dir / "density_stratified_summary.csv", density_summary_rows)

    exact_status_concordance = sum(bool(row["exact_status_concordance"]) for row in cross_method_rows)
    confirmed_both = sum(
        bool(row["combined_confirmed_multinucleated"]) and bool(row["bf_confirmed_multinucleated"])
        for row in cross_method_rows
    )
    confirmed_combined_only = sum(
        bool(row["combined_confirmed_multinucleated"]) and not bool(row["bf_confirmed_multinucleated"])
        for row in cross_method_rows
    )
    confirmed_bf_only = sum(
        not bool(row["combined_confirmed_multinucleated"]) and bool(row["bf_confirmed_multinucleated"])
        for row in cross_method_rows
    )
    confirmed_union = confirmed_both + confirmed_combined_only + confirmed_bf_only
    cross_method_summary = {
        "reciprocal_countable_cell_pairs": len(cross_method_rows),
        "exact_status_concordance_count": exact_status_concordance,
        "exact_status_concordance_rate": (
            exact_status_concordance / len(cross_method_rows) if cross_method_rows else 0.0
        ),
        "confirmed_multinucleated_both": confirmed_both,
        "confirmed_multinucleated_combined_only": confirmed_combined_only,
        "confirmed_multinucleated_bf_only": confirmed_bf_only,
        "confirmed_multinucleation_jaccard": confirmed_both / confirmed_union if confirmed_union else 0.0,
    }

    all_areas = np.asarray([float(row["nuclear_area_px2"]) for row in all_nucleus_rows], dtype=float)
    core_areas = np.asarray(
        [float(row["nucleus_core_area_px2"]) for row in all_nucleus_rows if row["nucleus_core_present"]],
        dtype=float,
    )
    core_fractions = np.asarray(
        [float(row["nucleus_core_to_extent_fraction"]) for row in all_nucleus_rows if row["nucleus_core_present"]],
        dtype=float,
    )
    nucleus_core_summary: dict[str, Any] = {
        "masks_available_for_fields": len(set(keys) & set(nucleus_core_masks)),
        "n_nuclei_with_core": int(core_areas.size),
        "n_nuclei_without_core": int(total_nuclei - core_areas.size),
    }
    if core_areas.size:
        nucleus_core_summary.update(
            {
                "area_px2": {
                    "q10": float(np.quantile(core_areas, 0.10)),
                    "median": float(np.median(core_areas)),
                    "q90": float(np.quantile(core_areas, 0.90)),
                },
                "core_to_extent_fraction": {
                    "q10": float(np.quantile(core_fractions, 0.10)),
                    "median": float(np.median(core_fractions)),
                    "q90": float(np.quantile(core_fractions, 0.90)),
                },
            }
        )
    mismatch_status_counts = summarize_counts(all_nucleus_rows, "overall_alignment_status")
    validation_checks = {
        "nucleus_row_count_matches_inventory": total_nuclei == sum(int(row["n_nuclei"]) for row in inventory_rows),
        "all_nucleus_fractions_in_range": all(
            0.0 <= float(row[field]) <= 1.0 + 1e-9
            for row in all_nucleus_rows
            for field in (
                "combined_primary_nucleus_fraction",
                "combined_total_cell_coverage_fraction",
                "bf_primary_nucleus_fraction",
                "bf_total_cell_coverage_fraction",
            )
        ),
        "strict_plus_review_equals_total": strict_count + len(mismatch_rows) == total_nuclei,
        "field_counts_equal_total": sum(int(row["n_nuclei"]) for row in all_field_rows) == total_nuclei,
        "all_requested_profiles_present": all(len(masks_by_profile[p]) >= len(keys) for p in PROFILE_STEMS),
        "reciprocal_cell_pairs_unique": len(cross_method_rows)
        == len({(row["key"], row["combined_cell_id"], row["bf_cell_id"]) for row in cross_method_rows}),
    }
    if not all(validation_checks.values()):
        raise RuntimeError(f"Validation failure: {validation_checks}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_root": args.run_root,
            "input_root": args.input_root,
            "script": Path(__file__).resolve(),
            "python": sys.executable,
        },
        "units": {
            "pixel_size_um": args.pixel_size_um,
            "size_unit": "micrometers" if args.pixel_size_um is not None else "pixels",
            "area_unit": "square micrometers" if args.pixel_size_um is not None else "square pixels",
        },
        "parameters": {
            "min_nucleus_area": args.min_nucleus_area,
            "min_cell_area": args.min_cell_area,
            "high_containment": args.high_containment,
            "moderate_containment": args.moderate_containment,
            "unmatched_overlap": args.unmatched_overlap,
            "significant_intersection_fraction": args.significant_intersection_fraction,
            "significant_secondary_fraction": args.significant_secondary_fraction,
            "min_significant_overlap_px": args.min_significant_overlap_px,
            "composite_confidence_weights": {
                "morphology": 0.20,
                "combined_assignment": 0.35,
                "bf_assignment": 0.30,
                "bf_combined_consensus": 0.15,
            },
        },
        "inventory": {
            "n_fields": len(keys),
            "field_keys": keys,
            "shape": [inventory_rows[0]["height"], inventory_rows[0]["width"]],
            "density_calls_available": len(density_calls) > 0,
            "nucleus_core_masks_available": len(nucleus_core_masks),
        },
        "nuclei": {
            "total": total_nuclei,
            "quality_nuclei": sum(bool(row["quality_nucleus"]) for row in all_nucleus_rows),
            "fragment_suspects": sum(bool(row["fragment_suspect"]) for row in all_nucleus_rows),
            "border_touching": sum(bool(row["border_touch"]) for row in all_nucleus_rows),
            "shape_irregular": sum(bool(row["shape_irregular_flag"]) for row in all_nucleus_rows),
            "area_px2": {
                "min": float(all_areas.min()),
                "q10": float(np.quantile(all_areas, 0.10)),
                "q25": float(np.quantile(all_areas, 0.25)),
                "median": float(np.median(all_areas)),
                "q75": float(np.quantile(all_areas, 0.75)),
                "q90": float(np.quantile(all_areas, 0.90)),
                "max": float(all_areas.max()),
            },
            "descriptive_size_thresholds_px2": {"small_below": float(q25), "large_above": float(q75)},
            "size_model": gmm_summary(gmm_result),
            "intensity_supported_core": nucleus_core_summary,
        },
        "alignment": {
            "strict_consensus_count": strict_count,
            "strict_consensus_rate": strict_count / total_nuclei,
            "acceptable_alignment_count": acceptable_count,
            "acceptable_alignment_rate": acceptable_count / total_nuclei,
            "actionable_mismatch_count": actionable_count,
            "actionable_mismatch_rate": actionable_count / total_nuclei,
            "review_recommended_count": len(mismatch_rows),
            "review_recommended_rate": len(mismatch_rows) / total_nuclei,
            "status_counts": mismatch_status_counts,
            "combined_match_categories": summarize_counts(all_nucleus_rows, "combined_match_category"),
            "bf_match_categories": summarize_counts(all_nucleus_rows, "bf_match_category"),
            "bf_combined_consensus_categories": summarize_counts(all_nucleus_rows, "bf_combined_consensus"),
        },
        "cells": cell_summary,
        "density_strata": density_strata,
        "cross_method_cell_concordance": cross_method_summary,
        "validation": validation_checks,
        "outputs": {
            "nucleus_features": args.out_dir / "nucleus_features.csv",
            "nucleus_cell_mapping": args.out_dir / "nucleus_cell_mapping.csv",
            "cell_nuclear_summary": args.out_dir / "cell_nuclear_summary.csv",
            "cell_mask_correspondence": args.out_dir / "cell_mask_correspondence.csv",
            "segmentation_mismatch": args.out_dir / "segmentation_mismatch.csv",
            "field_summary": args.out_dir / "field_summary.csv",
            "density_stratified_summary": args.out_dir / "density_stratified_summary.csv",
            "cell_pair_nuclear_status_concordance": args.out_dir / "cell_pair_nuclear_status_concordance.csv",
            "qc_overlays": args.out_dir / "qc_overlays",
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "analysis_summary.md").write_text(make_summary_markdown(json_ready(summary)))

    plots_dir = args.out_dir / "plots"
    save_histogram_svg(
        plots_dir / "nuclear_area_distribution.svg",
        eligible_areas,
        "Nuclear area distribution",
        f"Eligible non-border nuclei (n={eligible_areas.size:,}); Q25/Q75 define descriptive size bins",
        float(q25),
        float(q75),
    )
    status_labels = sorted(
        mismatch_status_counts,
        key=lambda label: (-mismatch_status_counts[label], label),
    )
    status_values = [float(mismatch_status_counts[label]) for label in status_labels]
    status_colors = [
        PALETTE["olive"] if label == "matched_consensus" else
        PALETTE["blue"] if label == "matched_one_way" else
        PALETTE["pink"] if label.startswith("crosses") else
        PALETTE["red"] if "unmatched" in label else
        PALETTE["orange"] if "conflict" in label else PALETTE["gold"]
        for label in status_labels
    ]
    save_horizontal_bar_svg(
        plots_dir / "alignment_status_counts.svg",
        "Nucleus-to-cell alignment status",
        f"All nuclei across {len(keys)} fields; exact counts by strict precedence-based category",
        [label.replace("_", " ") for label in status_labels],
        status_values,
        status_colors,
        width=1080,
    )
    cell_categories = [
        "no_confident_nucleus",
        "single_nucleus",
        "candidate_multinucleated",
        "confirmed_multinucleated",
    ]
    save_grouped_bar_svg(
        plots_dir / "cell_nuclear_status_by_mask.svg",
        "Cell nuclear status by cell-mask method",
        f"Countable cells only (area ≥ {args.min_cell_area:g} px²)",
        ["No confident nucleus", "Single nucleus", "Candidate multi", "Confirmed multi"],
        [
            (
                "Combined",
                [float(cell_summary["Combined"]["status_counts"].get(category, 0)) for category in cell_categories],
                PALETTE["blue"],
            ),
            (
                "Brightfield",
                [float(cell_summary["Brightfield"]["status_counts"].get(category, 0)) for category in cell_categories],
                PALETTE["gold"],
            ),
        ],
    )
    field_chart_rows = sorted(
        all_field_rows,
        key=lambda row: (-float(row["actionable_mismatch_rate"]), str(row["key"])),
    )
    save_horizontal_bar_svg(
        plots_dir / "field_actionable_mismatch_rates.svg",
        "Actionable alignment issue rate by field",
        "Unmatched/low-overlap nuclei, boundary crossings, or BF–Combined assignment conflicts",
        [str(row["key"]) for row in field_chart_rows],
        [100.0 * float(row["actionable_mismatch_rate"]) for row in field_chart_rows],
        [PALETTE["orange"]] * len(field_chart_rows),
        value_suffix="%",
        value_decimals=1,
        width=1080,
    )
    chart_map = [
        {
            "section": "Nuclear size",
            "question": "How broad is the nuclear-size distribution?",
            "chart": "histogram on log-scaled nuclear area",
            "fields": ["nuclear_area_px2"],
            "supported_claim": "Size spread and descriptive Q25/Q75 thresholds",
            "palette": "single-root blue with gold/orange references",
            "path": plots_dir / "nuclear_area_distribution.svg",
        },
        {
            "section": "Mask alignment",
            "question": "Which nucleus-to-cell alignment outcomes are most common?",
            "chart": "horizontal bar",
            "fields": ["overall_alignment_status", "count"],
            "supported_claim": "Exact composition of strict matches and failure modes",
            "palette": "status-aware categorical roots",
            "path": plots_dir / "alignment_status_counts.svg",
        },
        {
            "section": "Multinucleation",
            "question": "How do BF and Combined differ in cell nuclear status?",
            "chart": "grouped bar",
            "fields": ["cell_mask_profile", "nuclear_status", "count"],
            "supported_claim": "Mask-method sensitivity of multinucleation calls",
            "palette": "hard two-root blue/gold",
            "path": plots_dir / "cell_nuclear_status_by_mask.svg",
        },
        {
            "section": "Field heterogeneity",
            "question": "Which fields have the highest actionable issue rate?",
            "chart": "horizontal bar",
            "fields": ["key", "actionable_mismatch_rate"],
            "supported_claim": "Field-level concentration of alignment problems",
            "palette": "single-root orange",
            "path": plots_dir / "field_actionable_mismatch_rates.svg",
        },
    ]
    write_json(args.out_dir / "chart_map.json", chart_map)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
