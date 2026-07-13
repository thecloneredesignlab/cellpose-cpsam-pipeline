#!/usr/bin/env python3
"""Full-cohort calibration for Dead and Combined-blue image channels.

The scanner is intentionally independent of Cellpose.  It streams paired TIFFs,
records per-image robust intensity statistics, and accumulates fixed-grid
histograms for raw blue plus two blue-excess transforms. Multiple scan shards
can be merged into a versioned calibration artifact consumed downstream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
IMAGE_SUFFIXES = {".tif", ".tiff"}
DEAD_HIST_BINS = 262_144
DEAD_HIST_RANGE = (0.0, 4096.0)
BLUE_HIST_BINS = 256
BLUE_HIST_RANGE = (-0.5, 255.5)
EXCESS_HIST_BINS = 1024
EXCESS_HIST_RANGE = (0.0, 256.0)
STAT_QUANTILES = (0.1, 1.0, 5.0, 50.0, 90.0, 95.0, 99.0, 99.5, 99.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan one deterministic shard of paired images.")
    scan.add_argument("--input-root", type=Path, required=True)
    scan.add_argument("--dead-dir", type=Path)
    scan.add_argument("--combined-dir", type=Path)
    scan.add_argument("--out-dir", type=Path, required=True)
    scan.add_argument("--shard-index", type=int, required=True, help="One-based shard index.")
    scan.add_argument("--shard-count", type=int, required=True)
    scan.add_argument("--max-images", type=int)
    scan.add_argument("--correlation-sample-points", type=int, default=50_000)

    merge = subparsers.add_parser("merge", help="Merge completed scan shards.")
    merge.add_argument("--scan-dir", type=Path, required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    merge.add_argument("--expected-shards", type=int)
    merge.add_argument("--expected-pairs", type=int)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for empty table: {path}")
    columns = fieldnames or list(rows[0])
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate key {key}: {result[key]} and {path}")
        result[key] = path.resolve()
    return result


def scalar_dead(raw: np.ndarray) -> np.ndarray:
    arr = np.squeeze(raw)
    if arr.ndim != 2:
        raise ValueError(f"Dead image must be scalar 2D, got {raw.shape}")
    return arr


def combined_blue(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim != 3:
        raise ValueError(f"Combined image must be RGB, got {arr.shape}")
    if arr.shape[-1] >= 3:
        return arr[..., 2]
    if arr.shape[0] >= 3:
        return arr[2, ...]
    raise ValueError(f"Cannot locate blue channel in Combined image {arr.shape}")


def combined_blue_excess(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw)
    if arr.ndim != 3:
        raise ValueError(f"Combined image must be RGB, got {arr.shape}")
    if arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.float32, copy=False)
    elif arr.shape[0] >= 3:
        rgb = np.moveaxis(arr[:3, ...], 0, -1).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Cannot locate RGB channels in Combined image {arr.shape}")
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess_mean = np.clip(blue - 0.5 * (red + green), 0.0, None)
    excess_max = np.clip(blue - np.maximum(red, green), 0.0, None)
    return excess_mean.astype(np.float32, copy=False), excess_max.astype(np.float32, copy=False)


def fixed_histogram(
    image: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    data = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("Non-finite intensity values")
    low, high = value_range
    clipped = np.clip(data, low, np.nextafter(high, low))
    histogram, _edges = np.histogram(clipped, bins=bins, range=value_range)
    return histogram.astype(np.uint64)


def dtype_max(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    return float(np.nanmax(image))


def image_stats(prefix: str, image: np.ndarray) -> dict[str, Any]:
    data = np.asarray(image, dtype=np.float64)
    percentiles = np.percentile(data, STAT_QUANTILES)
    stats: dict[str, Any] = {
        f"{prefix}_dtype": str(image.dtype),
        f"{prefix}_height": int(image.shape[0]),
        f"{prefix}_width": int(image.shape[1]),
        f"{prefix}_min": float(np.min(data)),
        f"{prefix}_max": float(np.max(data)),
        f"{prefix}_mean": float(np.mean(data)),
        f"{prefix}_std": float(np.std(data)),
        f"{prefix}_zero_fraction": float(np.mean(data == 0)),
        f"{prefix}_saturation_fraction": float(np.mean(data >= dtype_max(image))),
    }
    for quantile, value in zip(STAT_QUANTILES, percentiles):
        label = str(quantile).replace(".", "p")
        stats[f"{prefix}_q{label}"] = float(value)
    return stats


def deterministic_sample_pair(
    first: np.ndarray,
    second: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if first.shape != second.shape:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    x = np.asarray(first, dtype=np.float64).ravel()
    y = np.asarray(second, dtype=np.float64).ravel()
    stride = max(1, int(math.ceil(x.size / max(max_points, 1))))
    return x[::stride], y[::stride]


def paired_stats(
    dead: np.ndarray,
    blue: np.ndarray,
    max_points: int,
    prefix: str,
) -> dict[str, Any]:
    x, y = deterministic_sample_pair(dead, blue, max_points)
    if x.size < 3:
        return {
            f"{prefix}_shape_equal": False,
            f"{prefix}_pearson": "",
            f"{prefix}_linear_r2": "",
            f"{prefix}_linear_rmse": "",
            f"{prefix}_sample_points": 0,
        }
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    pearson = float(np.corrcoef(x, y)[0, 1]) if x_std > 0 and y_std > 0 else 0.0
    design = np.column_stack([y, np.ones(y.size, dtype=np.float64)])
    slope, intercept = np.linalg.lstsq(design, x, rcond=None)[0]
    predicted = slope * y + intercept
    residual = x - predicted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((x - float(np.mean(x))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        f"{prefix}_shape_equal": True,
        f"{prefix}_pearson": pearson,
        f"{prefix}_linear_slope": float(slope),
        f"{prefix}_linear_intercept": float(intercept),
        f"{prefix}_linear_r2": float(r2),
        f"{prefix}_linear_rmse": float(np.sqrt(np.mean(residual * residual))),
        f"{prefix}_sample_points": int(x.size),
    }


def file_list_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def scan(args: argparse.Namespace) -> int:
    if args.shard_count < 1 or not 1 <= args.shard_index <= args.shard_count:
        raise SystemExit("Require shard_count >= 1 and 1 <= shard_index <= shard_count")
    dead_dir = (args.dead_dir or args.input_root / "Dead").resolve()
    combined_dir = (args.combined_dir or args.input_root / "Combined").resolve()
    dead_index = index_images(dead_dir)
    combined_index = index_images(combined_dir)
    common_keys = sorted(set(dead_index) & set(combined_index))
    selected = [
        key for offset, key in enumerate(common_keys) if offset % args.shard_count == args.shard_index - 1
    ]
    if args.max_images is not None:
        selected = selected[: args.max_images]
    if not selected:
        raise SystemExit(f"No paired keys selected for shard {args.shard_index}/{args.shard_count}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
    rows: list[dict[str, Any]] = []
    dead_hist = np.zeros(DEAD_HIST_BINS, dtype=np.uint64)
    blue_hist = np.zeros(BLUE_HIST_BINS, dtype=np.uint64)
    excess_mean_hist = np.zeros(EXCESS_HIST_BINS, dtype=np.uint64)
    excess_max_hist = np.zeros(EXCESS_HIST_BINS, dtype=np.uint64)
    failures: list[dict[str, str]] = []
    for position, key in enumerate(selected, start=1):
        dead_path = dead_index[key]
        combined_path = combined_index[key]
        try:
            dead = scalar_dead(tifffile.imread(dead_path))
            combined_raw = tifffile.imread(combined_path)
            blue = combined_blue(combined_raw)
            excess_mean, excess_max = combined_blue_excess(combined_raw)
            dead_hist += fixed_histogram(dead, DEAD_HIST_BINS, DEAD_HIST_RANGE)
            blue_hist += fixed_histogram(blue, BLUE_HIST_BINS, BLUE_HIST_RANGE)
            excess_mean_hist += fixed_histogram(excess_mean, EXCESS_HIST_BINS, EXCESS_HIST_RANGE)
            excess_max_hist += fixed_histogram(excess_max, EXCESS_HIST_BINS, EXCESS_HIST_RANGE)
            rows.append(
                {
                    "key": key,
                    "dead_path": str(dead_path),
                    "combined_path": str(combined_path),
                    "dead_file_bytes": dead_path.stat().st_size,
                    "combined_file_bytes": combined_path.stat().st_size,
                    **image_stats("dead", dead),
                    **image_stats("combined_blue", blue),
                    **image_stats("combined_blue_excess_mean", excess_mean),
                    **image_stats("combined_blue_excess_max", excess_max),
                    **paired_stats(dead, blue, args.correlation_sample_points, "dead_vs_blue"),
                    **paired_stats(
                        dead,
                        excess_mean,
                        args.correlation_sample_points,
                        "dead_vs_blue_excess_mean",
                    ),
                    **paired_stats(
                        dead,
                        excess_max,
                        args.correlation_sample_points,
                        "dead_vs_blue_excess_max",
                    ),
                }
            )
            print(f"scan={position}/{len(selected)} key={key}", flush=True)
        except Exception as exc:
            failures.append({"key": key, "error": repr(exc)})
            print(f"FAILED key={key} error={exc!r}", flush=True)

    if failures:
        write_rows(args.out_dir / f"{stem}_failures.csv", failures)
        raise SystemExit(f"Calibration shard failed for {len(failures)} images")
    write_rows(args.out_dir / f"{stem}_images.csv", rows)
    np.savez_compressed(
        args.out_dir / f"{stem}_histograms.npz",
        dead_hist=dead_hist,
        combined_blue_hist=blue_hist,
        combined_blue_excess_mean_hist=excess_mean_hist,
        combined_blue_excess_max_hist=excess_max_hist,
    )
    manifest = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "input_root": str(args.input_root.resolve()),
        "dead_dir": str(dead_dir),
        "combined_dir": str(combined_dir),
        "n_dead_images": len(dead_index),
        "n_combined_images": len(combined_index),
        "n_common_pairs": len(common_keys),
        "n_missing_dead": len(set(combined_index) - set(dead_index)),
        "n_missing_combined": len(set(dead_index) - set(combined_index)),
        "n_selected": len(selected),
        "n_completed": len(rows),
        "dead_pixel_count": int(dead_hist.sum()),
        "combined_blue_pixel_count": int(blue_hist.sum()),
        "combined_blue_excess_mean_pixel_count": int(excess_mean_hist.sum()),
        "combined_blue_excess_max_pixel_count": int(excess_max_hist.sum()),
        "selected_file_digest": file_list_digest(
            [path for key in selected for path in (dead_index[key], combined_index[key])]
        ),
    }
    write_json(args.out_dir / f"{stem}_manifest.json", manifest)
    (args.out_dir / f"{stem}_SUCCESS").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    print(f"scan_complete={len(rows)} manifest={args.out_dir / f'{stem}_manifest.json'}")
    return 0


def histogram_quantile(
    histogram: np.ndarray,
    percentile: float,
    value_range: tuple[float, float],
) -> float:
    total = int(histogram.sum())
    if total <= 0:
        return 0.0
    target = percentile / 100.0 * max(total - 1, 0)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    index = int(np.searchsorted(cumulative, target, side="right"))
    low, high = value_range
    width = (high - low) / histogram.size
    return float(low + (index + 0.5) * width)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def channel_calibration(
    rows: list[dict[str, str]],
    hist: np.ndarray,
    prefix: str,
    value_range: tuple[float, float],
) -> dict[str, Any]:
    q1 = [float(row[f"{prefix}_q1p0"]) for row in rows]
    q99 = [float(row[f"{prefix}_q99p0"]) for row in rows]
    q995 = [float(row[f"{prefix}_q99p5"]) for row in rows]
    global_q1 = histogram_quantile(hist, 1.0, value_range)
    global_q995 = histogram_quantile(hist, 99.5, value_range)
    if prefix == "dead":
        # Dead images are floating-point fluorescence with acquisition-dependent
        # background offsets. Preserve a cohort-wide signal span and let the
        # downstream map apply only a bounded per-image background correction.
        fixed_low = global_q1
        fixed_high = global_q995
    else:
        fixed_low = float(np.median(q1))
        fixed_high = float(np.percentile(q99, 75))
    if fixed_high <= fixed_low:
        fixed_high = float(np.percentile(q995, 75))
    return {
        "normalization": "cohort_fixed",
        "recommended_fixed_low": fixed_low,
        "recommended_fixed_high": fixed_high,
        "global_pixel_quantiles": {
            str(q): histogram_quantile(hist, q, value_range) for q in STAT_QUANTILES
        },
        "per_image_q1_distribution": distribution(q1),
        "per_image_q99_distribution": distribution(q99),
        "per_image_q99p5_distribution": distribution(q995),
        "bounded_background_low_range": [
            float(np.percentile(q1, 5)),
            float(np.percentile(q1, 95)),
        ],
        "recommended_background_span": fixed_high - fixed_low,
        "pixel_count": int(hist.sum()),
        "nonzero_bin_count": int(np.count_nonzero(hist)),
        "histogram_bins": int(hist.size),
        "histogram_range": list(value_range),
    }


def robust_outlier(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        return np.zeros(values.size, dtype=bool)
    return np.abs(values - median) / (1.4826 * mad) > 6.0


def render_qc(
    out_dir: Path,
    rows: list[dict[str, str]],
    dead_hist: np.ndarray,
    blue_hist: np.ndarray,
    excess_mean_hist: np.ndarray,
    excess_max_hist: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    dead_q99 = np.asarray([float(row["dead_q99p0"]) for row in rows])
    blue_q99 = np.asarray([float(row["combined_blue_q99p0"]) for row in rows])
    correlation = np.asarray([float(row["dead_vs_blue_pearson"]) for row in rows])
    excess_correlation = np.asarray(
        [float(row["dead_vs_blue_excess_mean_pearson"]) for row in rows]
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for histogram, label, axis in (
        (dead_hist, "Dead", axes[0, 0]),
        (blue_hist, "Combined blue", axes[0, 1]),
        (excess_mean_hist, "Combined blue excess mean", axes[0, 2]),
    ):
        if label == "Dead":
            value_range = DEAD_HIST_RANGE
        elif label == "Combined blue":
            value_range = BLUE_HIST_RANGE
        else:
            value_range = EXCESS_HIST_RANGE
        low, high = value_range
        centers = np.linspace(low, high, histogram.size, endpoint=False)
        step = max(1, histogram.size // 16_384)
        x = centers[::step]
        axis.plot(x, histogram[::step] / max(float(histogram.sum()), 1.0), lw=1)
        axis.set_yscale("log")
        axis.set_title(f"{label} global intensity histogram")
        axis.set_xlabel("Raw intensity")
        axis.set_ylabel("Pixel fraction")
    axes[1, 0].scatter(dead_q99, blue_q99, s=5, alpha=0.4)
    axes[1, 0].set_xlabel("Dead per-image q99")
    axes[1, 0].set_ylabel("Combined-blue per-image q99")
    axes[1, 0].set_title("Per-image high-signal coverage")
    axes[1, 1].hist(correlation[np.isfinite(correlation)], bins=50, alpha=0.65, label="raw B")
    axes[1, 1].hist(
        excess_correlation[np.isfinite(excess_correlation)],
        bins=50,
        alpha=0.65,
        label="B excess mean",
    )
    axes[1, 1].set_xlabel("Pixel-aligned Pearson correlation")
    axes[1, 1].set_ylabel("Images")
    axes[1, 1].set_title("Dead vs Combined-blue dependence")
    axes[1, 1].legend()
    max_centers = np.linspace(
        EXCESS_HIST_RANGE[0], EXCESS_HIST_RANGE[1], excess_max_hist.size, endpoint=False
    )
    step = max(1, excess_max_hist.size // 16_384)
    axes[1, 2].plot(
        max_centers[::step],
        excess_max_hist[::step] / max(float(excess_max_hist.sum()), 1.0),
        lw=1,
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("B - max(R,G)")
    axes[1, 2].set_ylabel("Pixel fraction")
    axes[1, 2].set_title("Combined blue excess max histogram")
    figure.tight_layout()
    figure.savefig(out_dir / "dead_combined_blue_calibration_qc.png", dpi=180)
    plt.close(figure)


def merge(args: argparse.Namespace) -> int:
    csv_paths = sorted(args.scan_dir.glob("shard_*_images.csv"))
    histogram_paths = sorted(args.scan_dir.glob("shard_*_histograms.npz"))
    manifest_paths = sorted(args.scan_dir.glob("shard_*_manifest.json"))
    if not csv_paths or len(csv_paths) != len(histogram_paths) or len(csv_paths) != len(manifest_paths):
        raise SystemExit(
            f"Incomplete calibration shards: csv={len(csv_paths)} hist={len(histogram_paths)} "
            f"manifest={len(manifest_paths)}"
        )
    if args.expected_shards is not None and len(csv_paths) != args.expected_shards:
        raise SystemExit(f"Expected {args.expected_shards} shards, found {len(csv_paths)}")

    rows: list[dict[str, str]] = []
    dead_hist = np.zeros(DEAD_HIST_BINS, dtype=np.uint64)
    blue_hist = np.zeros(BLUE_HIST_BINS, dtype=np.uint64)
    excess_mean_hist = np.zeros(EXCESS_HIST_BINS, dtype=np.uint64)
    excess_max_hist = np.zeros(EXCESS_HIST_BINS, dtype=np.uint64)
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    for csv_path, histogram_path in zip(csv_paths, histogram_paths):
        with csv_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
        payload = np.load(histogram_path)
        dead_hist += payload["dead_hist"].astype(np.uint64)
        blue_hist += payload["combined_blue_hist"].astype(np.uint64)
        excess_mean_hist += payload["combined_blue_excess_mean_hist"].astype(np.uint64)
        excess_max_hist += payload["combined_blue_excess_max_hist"].astype(np.uint64)
    rows.sort(key=lambda row: row["key"])
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate keys found while merging calibration shards")
    if args.expected_pairs is not None and len(rows) != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} pairs, found {len(rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dead_calibration = channel_calibration(rows, dead_hist, "dead", DEAD_HIST_RANGE)
    blue_calibration = channel_calibration(rows, blue_hist, "combined_blue", BLUE_HIST_RANGE)
    excess_mean_calibration = channel_calibration(
        rows,
        excess_mean_hist,
        "combined_blue_excess_mean",
        EXCESS_HIST_RANGE,
    )
    excess_max_calibration = channel_calibration(
        rows,
        excess_max_hist,
        "combined_blue_excess_max",
        EXCESS_HIST_RANGE,
    )
    correlations = [float(row["dead_vs_blue_pearson"]) for row in rows]
    r2_values = [float(row["dead_vs_blue_linear_r2"]) for row in rows]
    excess_mean_correlations = [
        float(row["dead_vs_blue_excess_mean_pearson"]) for row in rows
    ]
    excess_mean_r2 = [float(row["dead_vs_blue_excess_mean_linear_r2"]) for row in rows]
    excess_max_correlations = [
        float(row["dead_vs_blue_excess_max_pearson"]) for row in rows
    ]
    excess_max_r2 = [float(row["dead_vs_blue_excess_max_linear_r2"]) for row in rows]
    calibration = {
        "schema_version": 1,
        "n_pairs": len(rows),
        "n_shards": len(csv_paths),
        "input_root": manifests[0]["input_root"],
        "dead": dead_calibration,
        "combined_blue": blue_calibration,
        "combined_blue_excess_mean": excess_mean_calibration,
        "combined_blue_excess_max": excess_max_calibration,
        "dead_vs_combined_blue_dependence": {
            "pearson_distribution": distribution(correlations),
            "linear_r2_distribution": distribution(r2_values),
            "near_deterministic_fraction_r2_ge_0p995": float(np.mean(np.asarray(r2_values) >= 0.995)),
        },
        "dead_vs_combined_blue_excess_mean_dependence": {
            "pearson_distribution": distribution(excess_mean_correlations),
            "linear_r2_distribution": distribution(excess_mean_r2),
            "near_deterministic_fraction_r2_ge_0p995": float(
                np.mean(np.asarray(excess_mean_r2) >= 0.995)
            ),
        },
        "dead_vs_combined_blue_excess_max_dependence": {
            "pearson_distribution": distribution(excess_max_correlations),
            "linear_r2_distribution": distribution(excess_max_r2),
            "near_deterministic_fraction_r2_ge_0p995": float(
                np.mean(np.asarray(excess_max_r2) >= 0.995)
            ),
        },
        "validation": {
            "all_shards_present": args.expected_shards is None or len(csv_paths) == args.expected_shards,
            "expected_pair_count": args.expected_pairs,
            "pair_count_matches": args.expected_pairs is None or len(rows) == args.expected_pairs,
            "keys_unique": len(keys) == len(set(keys)),
            "all_shapes_equal": all(
                row["dead_vs_blue_shape_equal"] == "True" for row in rows
            ),
        },
    }
    write_json(args.out_dir / "dead_combined_blue_calibration.json", calibration)
    write_json(args.out_dir / "dead_preprocess_calibration.json", dead_calibration)
    write_json(args.out_dir / "combined_blue_preprocess_calibration.json", blue_calibration)
    write_json(
        args.out_dir / "combined_blue_excess_mean_preprocess_calibration.json",
        excess_mean_calibration,
    )
    write_json(
        args.out_dir / "combined_blue_excess_max_preprocess_calibration.json",
        excess_max_calibration,
    )

    dead_q1 = np.asarray([float(row["dead_q1p0"]) for row in rows])
    dead_q99 = np.asarray([float(row["dead_q99p0"]) for row in rows])
    blue_q99 = np.asarray([float(row["combined_blue_q99p0"]) for row in rows])
    correlation = np.asarray(correlations)
    # High-signal quantiles are biological readouts and are deliberately not
    # used to label calibration outliers. Only extreme Dead-background drift is
    # a preprocessing calibration outlier; signal-coverage flags remain columns.
    outlier_mask = robust_outlier(dead_q1)
    enriched_rows: list[dict[str, Any]] = []
    for row, is_outlier in zip(rows, outlier_mask):
        enriched_rows.append(
            {
                **row,
                "dead_calibration_low": dead_calibration["recommended_fixed_low"],
                "dead_calibration_high": dead_calibration["recommended_fixed_high"],
                "combined_blue_calibration_low": blue_calibration["recommended_fixed_low"],
                "combined_blue_calibration_high": blue_calibration["recommended_fixed_high"],
                "combined_blue_excess_mean_calibration_low": excess_mean_calibration[
                    "recommended_fixed_low"
                ],
                "combined_blue_excess_mean_calibration_high": excess_mean_calibration[
                    "recommended_fixed_high"
                ],
                "combined_blue_excess_max_calibration_low": excess_max_calibration[
                    "recommended_fixed_low"
                ],
                "combined_blue_excess_max_calibration_high": excess_max_calibration[
                    "recommended_fixed_high"
                ],
                "calibration_group": "global",
                "calibration_outlier": bool(is_outlier),
                "low_raw_blue_dependence_flag": bool(
                    float(row["dead_vs_blue_pearson"]) < 0.20
                ),
                "low_blue_excess_dependence_flag": bool(
                    float(row["dead_vs_blue_excess_mean_pearson"]) < 0.20
                ),
            }
        )
    write_rows(args.out_dir / "dead_combined_blue_image_calibration_map.csv", enriched_rows)
    write_rows(
        args.out_dir / "calibration_outliers.csv",
        [row for row in enriched_rows if row["calibration_outlier"]],
        list(enriched_rows[0]),
    )
    np.savez_compressed(
        args.out_dir / "dead_combined_blue_global_histograms.npz",
        dead_hist=dead_hist,
        combined_blue_hist=blue_hist,
        combined_blue_excess_mean_hist=excess_mean_hist,
        combined_blue_excess_max_hist=excess_max_hist,
    )
    render_qc(
        args.out_dir,
        rows,
        dead_hist,
        blue_hist,
        excess_mean_hist,
        excess_max_hist,
    )
    write_json(args.out_dir / "scan_manifests.json", manifests)
    (args.out_dir / "_SUCCESS").write_text(json.dumps(calibration["validation"], sort_keys=True) + "\n")
    print(f"merge_complete={len(rows)} calibration={args.out_dir / 'dead_combined_blue_calibration.json'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "scan":
        return scan(args)
    if args.command == "merge":
        return merge(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
