#!/usr/bin/env python3
"""Generate conservative shape- and intensity-supported nucleus-split candidates.

This script never overwrites the production Nuclei masks.  It identifies large,
shape-suspicious Cellpose extents with two stable fluorescence peaks, creates
marker-watershed alternatives inside the original foreground, and retains the
original instance ID for the larger child while appending IDs for extra children.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


UTILS_PATH = Path(__file__).with_name("nuclei_segmentation_utils.py")
UTILS_SPEC = importlib.util.spec_from_file_location("nuclei_segmentation_utils_local", UTILS_PATH)
if UTILS_SPEC is None or UTILS_SPEC.loader is None:
    raise ImportError(f"Cannot load nucleus utilities from {UTILS_PATH}")
UTILS_MODULE = importlib.util.module_from_spec(UTILS_SPEC)
sys.modules[UTILS_SPEC.name] = UTILS_MODULE
UTILS_SPEC.loader.exec_module(UTILS_MODULE)
NucleusCoreConfig = UTILS_MODULE.NucleusCoreConfig
build_nucleus_core_seeds = UTILS_MODULE.build_nucleus_core_seeds

SCHEMA_PATH = Path(__file__).with_name("shape_strict_schema.py")
SCHEMA_SPEC = importlib.util.spec_from_file_location("shape_strict_schema_local", SCHEMA_PATH)
if SCHEMA_SPEC is None or SCHEMA_SPEC.loader is None:
    raise ImportError(f"Cannot load shape-strict schema from {SCHEMA_PATH}")
SCHEMA_MODULE = importlib.util.module_from_spec(SCHEMA_SPEC)
sys.modules[SCHEMA_SPEC.name] = SCHEMA_MODULE
SCHEMA_SPEC.loader.exec_module(SCHEMA_MODULE)
SHAPE_MULTIPEAK_DIAGNOSTIC_FIELDS = SCHEMA_MODULE.SHAPE_MULTIPEAK_DIAGNOSTIC_FIELDS


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
PERTURBATION_SIGMAS = (0.8, 1.0, 1.2, 1.5)


@dataclass(frozen=True)
class SplitConfig:
    tag: str
    min_parent_area: int
    max_parent_solidity: float
    max_parent_circularity: float
    min_parent_axis_ratio: float
    min_peak_distance: float
    min_peak_snr: float
    max_third_peak_ratio: float
    min_valley_drop_snr: float
    min_valley_drop_fraction: float
    min_stability_fraction: float
    min_child_area: int
    min_child_fraction: float
    min_shape_gain: float
    min_cell_support_fraction: float


SPLIT_CONFIGS = (
    SplitConfig(
        "shape_sensitive",
        140,
        0.94,
        0.72,
        1.40,
        6.0,
        2.5,
        1.10,
        0.7,
        0.10,
        0.50,
        25,
        0.15,
        0.00,
        0.00,
    ),
    SplitConfig(
        "shape_balanced",
        170,
        0.92,
        0.65,
        1.55,
        7.0,
        3.0,
        0.90,
        1.0,
        0.15,
        0.75,
        35,
        0.20,
        0.03,
        0.50,
    ),
    SplitConfig(
        "shape_evidence",
        190,
        0.90,
        0.58,
        1.70,
        8.0,
        3.5,
        0.80,
        1.3,
        0.18,
        0.80,
        35,
        0.22,
        0.05,
        0.50,
    ),
    SplitConfig(
        "shape_strict",
        190,
        0.90,
        0.58,
        1.70,
        8.0,
        3.5,
        0.80,
        1.3,
        0.18,
        0.80,
        35,
        0.22,
        0.05,
        1.00,
    ),
    SplitConfig(
        "shape_ultra",
        220,
        0.87,
        0.52,
        1.90,
        9.0,
        4.0,
        0.70,
        1.6,
        0.22,
        1.00,
        40,
        0.25,
        0.07,
        1.00,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen existing nuclear masks for stable two-peak merged-nucleus candidates."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell-run-root", type=Path, required=True)
    parser.add_argument(
        "--classification-root",
        type=Path,
        help="Classification directory linked into candidate outputs. Defaults to <cell-run-root>/classification_fusion.",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--keys", nargs="*")
    parser.add_argument(
        "--field-record",
        type=Path,
        help="Direct per-field JSON record. Bypasses all result-directory discovery.",
    )
    parser.add_argument(
        "--cell-mask-branch",
        choices=("original", "nucleated"),
        default="original",
        help="Select original or nucleated-only Combined/Brightfield masks from --field-record.",
    )
    parser.add_argument(
        "--production-shard",
        action="store_true",
        help="Omit per-field convenience symlinks and reports rebuilt by the finalizer.",
    )
    parser.add_argument(
        "--candidate-tags",
        nargs="+",
        choices=tuple(config.tag for config in SPLIT_CONFIGS),
        help="Generate only the requested split candidates. Defaults to the complete calibration panel.",
    )
    parser.add_argument("--background-margin", type=int, default=10)
    parser.add_argument("--broad-min-distance", type=int, default=5)
    parser.add_argument("--broad-min-peak-snr", type=float, default=2.0)
    parser.add_argument("--peak-match-distance", type=float, default=3.0)
    parser.add_argument("--max-peaks", type=int, default=3)
    return parser.parse_args()


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def index_files(
    directory: Path,
    pattern: str,
    requested_keys: set[str] | None = None,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    if requested_keys:
        paths = {
            path
            for key in requested_keys
            for path in directory.glob(pattern.replace("*", f"*{key}*", 1))
        }
    else:
        paths = set(directory.glob(pattern))
    for path in sorted(paths):
        key = key_from_path(path)
        if requested_keys and key not in requested_keys:
            continue
        if key in result:
            raise ValueError(f"Duplicate field {key} in {directory}")
        result[key] = path
    return result


def index_raw(input_root: Path, requested_keys: set[str] | None = None) -> dict[str, Path]:
    directory = input_root / "Nuclei"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ranks = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}
    result: dict[str, Path] = {}
    if requested_keys:
        paths = {path for key in requested_keys for path in directory.glob(f"*{key}*")}
    else:
        paths = set(directory.iterdir())
    for path in sorted(paths):
        if not path.is_file() or path.suffix.lower() not in ranks:
            continue
        try:
            key = key_from_path(path)
        except ValueError:
            continue
        if requested_keys and key not in requested_keys:
            continue
        previous = result.get(key)
        if previous is None or ranks[path.suffix.lower()] < ranks[previous.suffix.lower()]:
            result[key] = path
    return result


def direct_paths_from_json(
    path: Path,
    requested_keys: set[str],
    cell_mask_branch: str,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict[str, Path]]:
    payload = json.loads(path.read_text())
    key = str(payload.get("key", ""))
    if KEY_RE.fullmatch(key) is None:
        raise ValueError(f"Invalid key in field record {path}: {key!r}")
    if requested_keys and requested_keys != {key}:
        raise ValueError(f"Requested keys {sorted(requested_keys)} do not match field record key {key}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Missing profiles object in {path}")
    mask_field = "nucleated_mask" if cell_mask_branch == "nucleated" else "original_mask"
    extent = Path(str(profiles.get("Nuclei", {}).get("extent_mask", "")))
    raw = Path(str(profiles.get("Nuclei", {}).get("raw", "")))
    combined = Path(str(profiles.get("Combined", {}).get(mask_field, "")))
    brightfield = Path(str(profiles.get("Brightfield", {}).get(mask_field, "")))
    for label, candidate in {
        "nuclei_extent": extent,
        "nuclei_raw": raw,
        "combined_mask": combined,
        "brightfield_mask": brightfield,
    }.items():
        if not candidate.is_file():
            raise FileNotFoundError(f"{label} is missing for {key}: {candidate}")
    return {key: extent}, {key: combined}, {key: brightfield}, {key: raw}


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected 2D integer mask at {path}; got {labels.shape} {labels.dtype}")
    return labels.astype(np.int32, copy=False)


def read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = np.asarray(tifffile.imread(path))
    else:
        with Image.open(path) as opened:
            image = np.asarray(opened)
    image = np.squeeze(image)
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[-1] >= 3:
        arr = image[..., :3].astype(np.float32, copy=False)
    elif image.ndim == 3 and 3 <= image.shape[0] <= 4:
        arr = np.moveaxis(image[:3], 0, -1).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Cannot convert image shape {image.shape} to grayscale")
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(
        np.float32
    )


def write_mask(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if int(labels.max()) > np.iinfo(np.uint16).max else np.uint16
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")
    tifffile.imwrite(temporary, labels.astype(dtype, copy=False), compression="zlib")
    os.replace(temporary, path)


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if not rows and fields is None:
        temporary.write_text("")
        os.replace(temporary, path)
        return
    resolved_fields = list(fields or ())
    if fields is None:
        for row in rows:
            for field in row:
                if field not in resolved_fields:
                    resolved_fields.append(field)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def binary_shape_metrics(binary: np.ndarray) -> dict[str, float]:
    binary = np.asarray(binary, dtype=bool)
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "axis_ratio": float("inf"),
            "eccentricity": 1.0,
            "shape_score": 0.0,
        }
    yy, xx = np.nonzero(binary)
    centered_y = yy.astype(np.float64) - float(yy.mean())
    centered_x = xx.astype(np.float64) - float(xx.mean())
    cov_yy = float(np.mean(centered_y**2))
    cov_xx = float(np.mean(centered_x**2))
    cov_yx = float(np.mean(centered_y * centered_x))
    trace = cov_xx + cov_yy
    delta = math.sqrt(max((cov_xx - cov_yy) ** 2 + 4.0 * cov_yx**2, 0.0))
    major = max((trace + delta) / 2.0, 0.0)
    minor = max((trace - delta) / 2.0, 0.0)
    axis_ratio = math.sqrt(major / max(minor, 1e-12)) if major > 0 else 1.0
    eccentricity = math.sqrt(max(0.0, 1.0 - minor / max(major, 1e-12))) if major > 0 else 0.0
    contours, _hierarchy = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    points = np.concatenate(contours, axis=0) if contours else np.zeros((0, 1, 2), dtype=np.int32)
    if points.size:
        hull = cv2.convexHull(points)
        hull_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        convex_area = max(int(np.count_nonzero(hull_mask)), area)
    else:
        convex_area = area
    solidity = area / convex_area
    circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    shape_score = (
        0.50 * min(max(solidity, 0.0), 1.0)
        + 0.35 * min(max(circularity, 0.0), 1.0)
        + 0.15 * min(max(1.0 / max(axis_ratio, 1.0), 0.0), 1.0)
    )
    return {
        "area": float(area),
        "perimeter": perimeter,
        "solidity": solidity,
        "circularity": circularity,
        "axis_ratio": axis_ratio,
        "eccentricity": eccentricity,
        "shape_score": shape_score,
    }


def best_overlap_maps(
    combined: np.ndarray, brightfield: np.ndarray
) -> tuple[dict[int, int], dict[int, int]]:
    overlap = (combined > 0) & (brightfield > 0)
    if not np.any(overlap):
        return {}, {}
    base = int(brightfield.max()) + 1
    codes = combined[overlap].astype(np.int64) * base + brightfield[overlap].astype(np.int64)
    unique, counts = np.unique(codes, return_counts=True)
    best_combined: dict[int, tuple[int, int]] = {}
    best_bf: dict[int, tuple[int, int]] = {}
    for code, count_value in zip(unique, counts):
        combined_id = int(code // base)
        bf_id = int(code % base)
        count = int(count_value)
        if count > best_combined.get(combined_id, (0, 0))[1]:
            best_combined[combined_id] = (bf_id, count)
        if count > best_bf.get(bf_id, (0, 0))[1]:
            best_bf[bf_id] = (combined_id, count)
    return (
        {label: value[0] for label, value in best_combined.items()},
        {label: value[0] for label, value in best_bf.items()},
    )


def child_cell_assignment(child: np.ndarray, cell_mask: np.ndarray) -> dict[str, Any]:
    area = int(np.count_nonzero(child))
    values = cell_mask[child]
    foreground = values[values > 0]
    if foreground.size:
        counts = np.bincount(foreground.astype(np.int64, copy=False))
        primary = int(np.argmax(counts))
        fraction = float(counts[primary] / area)
    else:
        primary = 0
        fraction = 0.0
    yy, xx = np.nonzero(child)
    cy = int(round(float(yy.mean())))
    cx = int(round(float(xx.mean())))
    centroid_label = int(cell_mask[cy, cx])
    return {
        "primary": primary,
        "fraction": fraction,
        "centroid_label": centroid_label,
        "supported": primary > 0 and fraction >= 0.80 and centroid_label == primary,
    }


def detect_peaks(
    z_image: np.ndarray,
    object_mask: np.ndarray,
    min_distance: int,
    threshold_abs: float,
    max_peaks: int,
) -> np.ndarray:
    coordinates = peak_local_max(
        z_image,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(threshold_abs),
        labels=object_mask.astype(np.uint8),
        exclude_border=False,
        num_peaks=max_peaks,
    )
    if not len(coordinates):
        return np.zeros((0, 2), dtype=np.int32)
    values = z_image[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(values)[::-1]
    return coordinates[order].astype(np.int32, copy=False)


def peak_pair_stability(
    reference: np.ndarray,
    variants: list[np.ndarray],
    max_distance: float,
) -> float:
    if len(reference) < 2:
        return 0.0
    target = reference[:2].astype(np.float64)
    stable = 0
    for coordinates in variants:
        if len(coordinates) < 2:
            continue
        available = set(range(len(coordinates)))
        matched = 0
        for point in target:
            choices = sorted(
                (
                    (float(np.linalg.norm(coordinates[index].astype(np.float64) - point)), index)
                    for index in available
                ),
                key=lambda item: item[0],
            )
            if choices and choices[0][0] <= max_distance:
                matched += 1
                available.remove(choices[0][1])
        stable += int(matched == 2)
    return stable / len(variants) if variants else 0.0


def line_valley_metrics(z_image: np.ndarray, object_mask: np.ndarray, peaks: np.ndarray) -> tuple[float, float]:
    p0 = peaks[0]
    p1 = peaks[1]
    distance = float(np.linalg.norm(p0.astype(np.float64) - p1.astype(np.float64)))
    samples = max(int(math.ceil(distance * 3.0)), 21)
    yy = np.rint(np.linspace(float(p0[0]), float(p1[0]), samples)).astype(np.int32)
    xx = np.rint(np.linspace(float(p0[1]), float(p1[1]), samples)).astype(np.int32)
    profile = z_image[yy, xx].astype(np.float64)
    inside = object_mask[yy, xx]
    profile[~inside] = 0.0
    edge = max(2, samples // 10)
    interior = profile[edge:-edge] if samples > 2 * edge else profile
    valley = float(np.min(interior)) if interior.size else float(np.min(profile))
    peak_floor = float(min(z_image[tuple(p0)], z_image[tuple(p1)]))
    drop = peak_floor - valley
    return drop, drop / max(peak_floor, 1e-6)


def shape_gate(config: SplitConfig, parent: dict[str, float]) -> bool:
    return (
        parent["solidity"] <= config.max_parent_solidity
        or parent["circularity"] <= config.max_parent_circularity
        or parent["axis_ratio"] >= config.min_parent_axis_ratio
    )


def config_accepts(config: SplitConfig, metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["parent_area"] >= config.min_parent_area
        and shape_gate(config, metrics["parent_shape"])
        and metrics["n_peaks"] >= 2
        and metrics["peak_distance"] >= config.min_peak_distance
        and metrics["min_peak_snr"] >= config.min_peak_snr
        and metrics["third_peak_ratio"] <= config.max_third_peak_ratio
        and metrics["valley_drop_snr"] >= config.min_valley_drop_snr
        and metrics["valley_drop_fraction"] >= config.min_valley_drop_fraction
        and metrics["stability_fraction"] >= config.min_stability_fraction
        and metrics["min_child_area"] >= config.min_child_area
        and metrics["min_child_fraction"] >= config.min_child_fraction
        and metrics["shape_gain"] >= config.min_shape_gain
        and metrics["cell_support_fraction"] >= config.min_cell_support_fraction
    )


def main() -> int:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.cell_run_root = args.cell_run_root.resolve()
    args.classification_root = (
        args.classification_root.resolve()
        if args.classification_root is not None
        else args.cell_run_root / "classification_fusion"
    )
    args.input_root = args.input_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    requested_keys = set(args.keys or [])
    selected_tags = set(args.candidate_tags or [config.tag for config in SPLIT_CONFIGS])
    configs = tuple(config for config in SPLIT_CONFIGS if config.tag in selected_tags)

    if args.field_record is not None:
        extent_paths, combined_paths, bf_paths, raw_paths = direct_paths_from_json(
            args.field_record, requested_keys, args.cell_mask_branch
        )
        record_source = "field_record"
    else:
        extent_paths = index_files(
            args.run_root / "Nuclei" / "segmentations", "*_cp_masks.tif", requested_keys
        )
        combined_paths = index_files(
            args.cell_run_root / "Combined" / "segmentations", "*_cp_masks.tif", requested_keys
        )
        bf_paths = index_files(
            args.cell_run_root / "Brightfield" / "segmentations", "*_cp_masks.tif", requested_keys
        )
        raw_paths = index_raw(args.input_root, requested_keys)
        record_source = "directory_discovery"
    keys = sorted(set(extent_paths) & set(combined_paths) & set(bf_paths) & set(raw_paths))
    if args.keys:
        requested = set(args.keys)
        keys = [key for key in keys if key in requested]
        missing = sorted(requested - set(keys))
        if missing:
            raise SystemExit(f"Requested keys are incomplete: {missing}")
    if not keys:
        raise SystemExit("No common fields")
    print(f"record_source={record_source} n_selected={len(keys)}", flush=True)

    candidate_roots: dict[str, Path] = {}
    for config in configs:
        root = args.out_root / "candidates" / config.tag
        candidate_roots[config.tag] = root
        if not args.production_shard:
            link_directory(args.cell_run_root / "Combined", root / "Combined")
            link_directory(args.cell_run_root / "Brightfield", root / "Brightfield")
            link_directory(args.cell_run_root / "qc", root / "qc")
            link_directory(args.classification_root, root / "classification_fusion")

    diagnostic_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    summary_accumulator: dict[str, dict[str, int]] = {
        config.tag: defaultdict(int) for config in configs
    }
    total_baseline_nuclei = 0
    total_evaluated = 0
    core_config = NucleusCoreConfig()

    for field_index, key in enumerate(keys, start=1):
        print(f"[{field_index}/{len(keys)}] {key}", flush=True)
        extent = read_mask(extent_paths[key])
        combined = read_mask(combined_paths[key])
        brightfield = read_mask(bf_paths[key])
        raw = read_gray(raw_paths[key])
        if not (extent.shape == combined.shape == brightfield.shape == raw.shape):
            raise ValueError(f"Shape mismatch for {key}")
        total_baseline_nuclei += int(extent.max())
        combined_to_bf, bf_to_combined = best_overlap_maps(combined, brightfield)
        objects = ndimage.find_objects(extent)
        candidate_outputs = {config.tag: extent.copy() for config in configs}
        next_ids = {config.tag: int(extent.max()) + 1 for config in configs}

        for nucleus_id, obj_slice in enumerate(objects, start=1):
            if obj_slice is None:
                continue
            obj = extent[obj_slice] == nucleus_id
            parent_shape = binary_shape_metrics(obj)
            if parent_shape["area"] < 140:
                continue
            broad_shape_suspicious = (
                parent_shape["solidity"] <= 0.94
                or parent_shape["circularity"] <= 0.72
                or parent_shape["axis_ratio"] >= 1.40
            )
            if not broad_shape_suspicious:
                continue
            total_evaluated += 1

            margin = max(1, int(args.background_margin))
            y0 = max(0, int(obj_slice[0].start) - margin)
            y1 = min(extent.shape[0], int(obj_slice[0].stop) + margin)
            x0 = max(0, int(obj_slice[1].start) - margin)
            x1 = min(extent.shape[1], int(obj_slice[1].stop) + margin)
            window_raw = raw[y0:y1, x0:x1]
            window_labels = extent[y0:y1, x0:x1]
            background = window_raw[window_labels == 0]
            if background.size < 20:
                background = window_raw.ravel()
                background = background[background <= np.percentile(background, 35)]
            background_median = float(np.median(background)) if background.size else float(np.median(window_raw))
            background_mad = float(np.median(np.abs(background - background_median))) if background.size else 0.0
            background_sigma = max(1.4826 * background_mad, 0.5)
            crop_y0 = int(obj_slice[0].start) - y0
            crop_y1 = int(obj_slice[0].stop) - y0
            crop_x0 = int(obj_slice[1].start) - x0
            crop_x1 = int(obj_slice[1].stop) - x0
            z_variants: list[np.ndarray] = []
            peak_variants: list[np.ndarray] = []
            for sigma in PERTURBATION_SIGMAS:
                smoothed = ndimage.gaussian_filter(window_raw.astype(np.float32), sigma=float(sigma))
                z_window = (smoothed - background_median) / background_sigma
                z_crop = z_window[crop_y0:crop_y1, crop_x0:crop_x1]
                z_variants.append(z_crop)
                peak_variants.append(
                    detect_peaks(
                        z_crop,
                        obj,
                        args.broad_min_distance,
                        args.broad_min_peak_snr,
                        args.max_peaks,
                    )
                )
            reference_index = PERTURBATION_SIGMAS.index(1.0)
            z_image = z_variants[reference_index]
            peaks = peak_variants[reference_index]
            if len(peaks) < 2:
                diagnostic_rows.append(
                    {
                        "key": key,
                        "nucleus_id": nucleus_id,
                        "parent_area": parent_shape["area"],
                        "parent_solidity": parent_shape["solidity"],
                        "parent_circularity": parent_shape["circularity"],
                        "parent_axis_ratio": parent_shape["axis_ratio"],
                        "n_peaks": len(peaks),
                        "accepted_configs": "",
                    }
                )
                continue

            markers = np.zeros(obj.shape, dtype=np.int32)
            markers[tuple(peaks[0])] = 1
            markers[tuple(peaks[1])] = 2
            split_labels = watershed(-z_image, markers=markers, mask=obj, compactness=0.001)
            # A disconnected parent component without a marker is left at zero by
            # watershed.  Reject it rather than silently changing nuclear foreground.
            if np.any(obj & (split_labels == 0)):
                diagnostic_rows.append(
                    {
                        "key": key,
                        "nucleus_id": nucleus_id,
                        "parent_area": parent_shape["area"],
                        "parent_solidity": parent_shape["solidity"],
                        "parent_circularity": parent_shape["circularity"],
                        "parent_axis_ratio": parent_shape["axis_ratio"],
                        "n_peaks": len(peaks),
                        "watershed_complete": False,
                        "accepted_configs": "",
                    }
                )
                continue
            children = [split_labels == child_id for child_id in (1, 2)]
            child_shapes = [binary_shape_metrics(child) for child in children]
            child_areas = [int(shape["area"]) for shape in child_shapes]
            weighted_child_shape_score = sum(
                area * shape["shape_score"] for area, shape in zip(child_areas, child_shapes)
            ) / max(sum(child_areas), 1)
            shape_gain = weighted_child_shape_score - parent_shape["shape_score"]
            peak_values = [float(z_image[tuple(peaks[index])]) for index in range(len(peaks))]
            peak_distance = float(np.linalg.norm(peaks[0].astype(float) - peaks[1].astype(float)))
            third_peak_ratio = (
                peak_values[2] / max(min(peak_values[0], peak_values[1]), 1e-6)
                if len(peak_values) >= 3
                else 0.0
            )
            valley_drop_snr, valley_drop_fraction = line_valley_metrics(z_image, obj, peaks)
            stability = peak_pair_stability(peaks, peak_variants, args.peak_match_distance)

            child_assignments: list[dict[str, Any]] = []
            supported_children = 0
            assigned_pairs: list[tuple[int, int]] = []
            for child in children:
                combined_assignment = child_cell_assignment(child, combined[obj_slice])
                bf_assignment = child_cell_assignment(child, brightfield[obj_slice])
                reciprocal = (
                    combined_assignment["primary"] > 0
                    and bf_assignment["primary"] > 0
                    and combined_to_bf.get(combined_assignment["primary"])
                    == bf_assignment["primary"]
                    and bf_to_combined.get(bf_assignment["primary"])
                    == combined_assignment["primary"]
                )
                supported = bool(
                    combined_assignment["supported"]
                    and bf_assignment["supported"]
                    and reciprocal
                )
                supported_children += int(supported)
                assigned_pairs.append(
                    (combined_assignment["primary"], bf_assignment["primary"])
                )
                child_assignments.append(
                    {
                        "combined": combined_assignment,
                        "brightfield": bf_assignment,
                        "reciprocal": reciprocal,
                        "supported": supported,
                    }
                )

            metrics: dict[str, Any] = {
                "parent_area": int(parent_shape["area"]),
                "parent_shape": parent_shape,
                "n_peaks": len(peaks),
                "peak_distance": peak_distance,
                "min_peak_snr": min(peak_values[0], peak_values[1]),
                "third_peak_ratio": third_peak_ratio,
                "valley_drop_snr": valley_drop_snr,
                "valley_drop_fraction": valley_drop_fraction,
                "stability_fraction": stability,
                "child_areas": child_areas,
                "min_child_area": min(child_areas),
                "min_child_fraction": min(child_areas) / max(int(parent_shape["area"]), 1),
                "shape_gain": shape_gain,
                "cell_support_fraction": supported_children / 2.0,
                "children_same_cell_pair": len(set(assigned_pairs)) == 1,
            }
            accepted_tags = [config.tag for config in configs if config_accepts(config, metrics)]
            diagnostic_rows.append(
                {
                    "key": key,
                    "nucleus_id": nucleus_id,
                    "parent_area": metrics["parent_area"],
                    "parent_solidity": parent_shape["solidity"],
                    "parent_circularity": parent_shape["circularity"],
                    "parent_axis_ratio": parent_shape["axis_ratio"],
                    "parent_eccentricity": parent_shape["eccentricity"],
                    "parent_shape_score": parent_shape["shape_score"],
                    "n_peaks": metrics["n_peaks"],
                    "watershed_complete": True,
                    "peak1_y": int(peaks[0, 0] + int(obj_slice[0].start)),
                    "peak1_x": int(peaks[0, 1] + int(obj_slice[1].start)),
                    "peak2_y": int(peaks[1, 0] + int(obj_slice[0].start)),
                    "peak2_x": int(peaks[1, 1] + int(obj_slice[1].start)),
                    "peak_distance": peak_distance,
                    "min_peak_snr": metrics["min_peak_snr"],
                    "third_peak_ratio": third_peak_ratio,
                    "valley_drop_snr": valley_drop_snr,
                    "valley_drop_fraction": valley_drop_fraction,
                    "stability_fraction": stability,
                    "child1_area": child_areas[0],
                    "child2_area": child_areas[1],
                    "child1_solidity": child_shapes[0]["solidity"],
                    "child2_solidity": child_shapes[1]["solidity"],
                    "child1_circularity": child_shapes[0]["circularity"],
                    "child2_circularity": child_shapes[1]["circularity"],
                    "child1_axis_ratio": child_shapes[0]["axis_ratio"],
                    "child2_axis_ratio": child_shapes[1]["axis_ratio"],
                    "shape_gain": shape_gain,
                    "cell_support_fraction": metrics["cell_support_fraction"],
                    "children_same_cell_pair": metrics["children_same_cell_pair"],
                    "accepted_configs": ";".join(accepted_tags),
                }
            )

            for config in configs:
                if config.tag not in accepted_tags:
                    continue
                output = candidate_outputs[config.tag]
                next_id = next_ids[config.tag]
                larger_index = int(np.argmax(child_areas))
                smaller_index = 1 - larger_index
                view = output[obj_slice]
                view[obj] = 0
                view[children[larger_index]] = nucleus_id
                view[children[smaller_index]] = next_id
                next_ids[config.tag] += 1
                accumulator = summary_accumulator[config.tag]
                accumulator["n_split_parents"] += 1
                accumulator["n_added_nuclei"] += 1
                event_rows.append(
                    {
                        "candidate": config.tag,
                        "key": key,
                        "parent_nucleus_id": nucleus_id,
                        "retained_child_id": nucleus_id,
                        "new_child_id": next_id,
                        "parent_area": metrics["parent_area"],
                        "parent_solidity": parent_shape["solidity"],
                        "parent_circularity": parent_shape["circularity"],
                        "parent_axis_ratio": parent_shape["axis_ratio"],
                        "parent_shape_score": parent_shape["shape_score"],
                        "peak1_y": int(peaks[0, 0] + int(obj_slice[0].start)),
                        "peak1_x": int(peaks[0, 1] + int(obj_slice[1].start)),
                        "peak2_y": int(peaks[1, 0] + int(obj_slice[0].start)),
                        "peak2_x": int(peaks[1, 1] + int(obj_slice[1].start)),
                        "peak_distance": peak_distance,
                        "min_peak_snr": metrics["min_peak_snr"],
                        "third_peak_ratio": third_peak_ratio,
                        "valley_drop_snr": valley_drop_snr,
                        "valley_drop_fraction": valley_drop_fraction,
                        "stability_fraction": stability,
                        "retained_child_area": child_areas[larger_index],
                        "new_child_area": child_areas[smaller_index],
                        "min_child_fraction": metrics["min_child_fraction"],
                        "weighted_child_shape_score": weighted_child_shape_score,
                        "shape_gain": shape_gain,
                        "cell_support_fraction": metrics["cell_support_fraction"],
                        "children_same_cell_pair": metrics["children_same_cell_pair"],
                        "child1_combined_cell": child_assignments[0]["combined"]["primary"],
                        "child1_bf_cell": child_assignments[0]["brightfield"]["primary"],
                        "child1_cell_supported": child_assignments[0]["supported"],
                        "child2_combined_cell": child_assignments[1]["combined"]["primary"],
                        "child2_bf_cell": child_assignments[1]["brightfield"]["primary"],
                        "child2_cell_supported": child_assignments[1]["supported"],
                    }
                )

        for config in configs:
            root = candidate_roots[config.tag]
            output = candidate_outputs[config.tag]
            # Per-field labels are contiguous because new IDs are appended after the field maximum.
            if len(np.unique(output)) - 1 != int(output.max()):
                raise RuntimeError(f"Non-contiguous labels in {config.tag}/{key}")
            write_mask(root / "Nuclei" / "segmentations" / extent_paths[key].name, output)
            core, _diagnostics = build_nucleus_core_seeds(raw, output, core_config)
            core_name = extent_paths[key].name.replace("_cp_masks.tif", "_core_masks.tif")
            write_mask(root / "Nuclei" / "nucleus_core_seeds" / core_name, core)
            summary_accumulator[config.tag]["n_fields"] += 1
            summary_accumulator[config.tag]["n_candidate_nuclei"] += int(output.max())

    write_rows(
        args.out_root / "shape_multipeak_diagnostics.csv",
        diagnostic_rows,
        fields=SHAPE_MULTIPEAK_DIAGNOSTIC_FIELDS,
    )
    write_rows(args.out_root / "split_events.csv", event_rows)
    write_json(args.out_root / "split_configs.json", [asdict(config) for config in configs])

    summaries: list[dict[str, Any]] = []
    for config in configs:
        rows = [row for row in event_rows if row["candidate"] == config.tag]
        raw = summary_accumulator[config.tag]

        def median(field: str) -> float:
            return float(np.median([float(row[field]) for row in rows])) if rows else 0.0

        summary = {
            "candidate": config.tag,
            **asdict(config),
            **dict(raw),
            "n_baseline_nuclei": total_baseline_nuclei,
            "n_shape_suspicious_evaluated": total_evaluated,
            "split_parent_fraction": raw.get("n_split_parents", 0) / total_baseline_nuclei,
            "count_increase_fraction": raw.get("n_added_nuclei", 0) / total_baseline_nuclei,
            "median_peak_distance": median("peak_distance"),
            "median_min_peak_snr": median("min_peak_snr"),
            "median_valley_drop_snr": median("valley_drop_snr"),
            "median_valley_drop_fraction": median("valley_drop_fraction"),
            "median_stability_fraction": median("stability_fraction"),
            "median_shape_gain": median("shape_gain"),
            "median_cell_support_fraction": median("cell_support_fraction"),
            "same_cell_pair_fraction": (
                float(np.mean([str(row["children_same_cell_pair"]).lower() == "true" for row in rows]))
                if rows
                else 0.0
            ),
        }
        summaries.append(summary)
        write_json(candidate_roots[config.tag] / "split_summary.json", summary)
    if not args.production_shard:
        write_rows(args.out_root / "candidate_summary.csv", summaries)

    lines = [
        "# Shape-aware nucleus split screen",
        "",
        f"- fields: {len(keys)}",
        f"- baseline nuclei: {total_baseline_nuclei}",
        f"- broad shape-suspicious objects evaluated: {total_evaluated}",
        "- production masks were not overwritten",
        "",
        "| candidate | split parents | count increase | median valley drop | median stability | median shape gain | median cell support |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['candidate']} | {summary.get('n_split_parents', 0)} | "
            f"{summary['count_increase_fraction']:.2%} | {summary['median_valley_drop_snr']:.2f} | "
            f"{summary['median_stability_fraction']:.1%} | {summary['median_shape_gain']:.3f} | "
            f"{summary['median_cell_support_fraction']:.1%} |"
        )
    if not args.production_shard:
        (args.out_root / "screen_report.md").write_text("\n".join(lines) + "\n")
        print(f"candidate_summary={args.out_root / 'candidate_summary.csv'}", flush=True)
    print(f"split_events={args.out_root / 'split_events.csv'}", flush=True)
    if not args.production_shard:
        print(f"report={args.out_root / 'screen_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
