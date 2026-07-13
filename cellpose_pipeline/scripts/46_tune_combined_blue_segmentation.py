#!/usr/bin/env python3
"""GPU screen of Combined-blue dead-signal segmentation candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
REQUIRED_CELLPOSE_VERSION = "4.2.1.1"


@dataclass(frozen=True)
class Candidate:
    tag: str
    model: str
    signal_transform: str
    diameter: float
    cellprob_threshold: float
    flow_threshold: float = 0.0
    min_size: int = 10
    preprocess: str = "fixed"
    background_sigma: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-set", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--candidate-tag", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--match-distance", type=float, default=20.0)
    parser.add_argument("--min-overlap-pixels", type=int, default=3)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"Cannot extract key from {path}")
    return match.group(1)


def index_files(directory: Path, pattern: str = "*.tif") -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate key {key} in {directory}")
        result[key] = path.resolve()
    return result


def blue_signals(raw: np.ndarray) -> dict[str, np.ndarray]:
    if raw.ndim != 3:
        raise ValueError(f"Expected Combined RGB image, got {raw.shape}")
    rgb = raw[..., :3].astype(np.float32, copy=False)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {
        "blue": blue,
        "blue_excess_mean": np.clip(blue - 0.5 * (red + green), 0.0, None),
        "blue_excess_max": np.clip(blue - np.maximum(red, green), 0.0, None),
    }


def calibration_for(payload: dict[str, Any], transform: str) -> tuple[float, float]:
    key = {
        "blue": "combined_blue",
        "blue_excess_mean": "combined_blue_excess_mean",
        "blue_excess_max": "combined_blue_excess_max",
    }[transform]
    block = payload[key]
    low = float(block["recommended_fixed_low"])
    high = float(block["recommended_fixed_high"])
    if high <= low:
        raise ValueError(f"Invalid calibration for {transform}: {low}-{high}")
    return low, high


def fixed_normalize(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((signal.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def background_correct(normalized: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    background = cv2.GaussianBlur(normalized.astype(np.float32), (k, k), sigma)
    corrected = np.clip(normalized - background, 0.0, None)
    high = float(np.percentile(corrected, 99.8))
    return np.clip(corrected / max(high, 1e-6), 0.0, 1.0)


def prepare_signal(signal: np.ndarray, candidate: Candidate, calibration: dict[str, Any]) -> np.ndarray:
    low, high = calibration_for(calibration, candidate.signal_transform)
    normalized = fixed_normalize(signal, low, high)
    if candidate.preprocess == "fixed":
        return normalized.astype(np.float32, copy=False)
    if candidate.preprocess == "background":
        return background_correct(normalized, candidate.background_sigma).astype(np.float32, copy=False)
    raise ValueError(candidate.preprocess)


def coarse_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for transform in ("blue", "blue_excess_mean", "blue_excess_max"):
        for model in ("cpsam_v2", "cpsam"):
            for diameter in (18.0, 22.0):
                for cellprob in (-3.0, -2.5):
                    tag = f"{transform}_{model}_d{int(diameter)}_cp{str(cellprob).replace('-', 'm').replace('.', 'p')}"
                    candidates.append(
                        Candidate(tag, model, transform, diameter, cellprob)
                    )
    for model in ("cpsam_v2", "cpsam"):
        candidates.append(
            Candidate(
                f"blue_{model}_d22_cpm2p5_bg12",
                model,
                "blue",
                22.0,
                -2.5,
                preprocess="background",
                background_sigma=12.0,
            )
        )
    return candidates


def fine_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for transform in ("blue_excess_mean", "blue_excess_max"):
        for model in ("cpsam_v2", "cpsam"):
            for diameter in (18.0, 20.0, 22.0):
                for cellprob in (-3.25, -3.0, -2.75, -2.5, -2.25):
                    tag = f"fine_{transform}_{model}_d{int(diameter)}_cp{str(cellprob).replace('-', 'm').replace('.', 'p')}"
                    candidates.append(Candidate(tag, model, transform, diameter, cellprob))
    return candidates


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if int(mask.max()) > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, mask.astype(dtype, copy=False))


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}: {path}")
    return mask.astype(np.int32, copy=False)


def object_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int32)
    if labels.size == 0:
        return labels, np.zeros(1, dtype=np.int64), np.zeros((1, 2), dtype=np.float64)
    max_label = int(mask.max())
    areas = np.bincount(mask.ravel(), minlength=max_label + 1)
    yy, xx = np.indices(mask.shape)
    sums_y = np.bincount(mask.ravel(), weights=yy.ravel(), minlength=max_label + 1)
    sums_x = np.bincount(mask.ravel(), weights=xx.ravel(), minlength=max_label + 1)
    centroids = np.zeros((max_label + 1, 2), dtype=np.float64)
    centroids[labels, 0] = sums_y[labels] / np.maximum(areas[labels], 1)
    centroids[labels, 1] = sums_x[labels] / np.maximum(areas[labels], 1)
    return labels, areas, centroids


def match_masks(
    dead: np.ndarray,
    blue: np.ndarray,
    max_distance: float,
    min_overlap: int,
) -> dict[str, float | int]:
    dead_labels, dead_areas, dead_centroids = object_stats(dead)
    blue_labels, blue_areas, blue_centroids = object_stats(blue)
    candidates: list[tuple[float, int, int, int, float]] = []
    foreground = (dead > 0) & (blue > 0)
    if np.any(foreground):
        pairs = np.column_stack([dead[foreground], blue[foreground]])
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (dead_label, blue_label), overlap in zip(unique_pairs, counts):
            dead_label = int(dead_label)
            blue_label = int(blue_label)
            if overlap < min_overlap:
                continue
            union = int(dead_areas[dead_label] + blue_areas[blue_label] - overlap)
            iou = float(overlap / max(union, 1))
            candidates.append((iou + 1.0, dead_label, blue_label, int(overlap), iou))
    if dead_labels.size and blue_labels.size:
        from scipy.spatial import cKDTree

        tree = cKDTree(dead_centroids[dead_labels])
        distances, indexes = tree.query(blue_centroids[blue_labels], k=1)
        for blue_label, distance, index in zip(blue_labels, distances, indexes):
            if float(distance) <= max_distance:
                dead_label = int(dead_labels[int(index)])
                candidates.append((1.0 - float(distance) / max_distance, dead_label, int(blue_label), 0, 0.0))
    candidates.sort(reverse=True)
    used_dead: set[int] = set()
    used_blue: set[int] = set()
    matches: list[tuple[int, int, int, float]] = []
    for _score, dead_label, blue_label, overlap, iou in candidates:
        if dead_label in used_dead or blue_label in used_blue:
            continue
        used_dead.add(dead_label)
        used_blue.add(blue_label)
        matches.append((dead_label, blue_label, overlap, iou))
    precision = len(matches) / blue_labels.size if blue_labels.size else 0.0
    recall = len(matches) / dead_labels.size if dead_labels.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "dead_objects": int(dead_labels.size),
        "blue_objects": int(blue_labels.size),
        "matched_objects": len(matches),
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": f1,
        "median_match_iou": float(np.median([row[3] for row in matches])) if matches else 0.0,
    }


def mask_signal_support(mask: np.ndarray, signal: np.ndarray, strong_threshold: float) -> dict[str, float]:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if labels.size == 0:
        return {"strong_signal_fraction": 0.0, "median_object_p90": 0.0}
    values: list[float] = []
    for label in labels:
        object_values = signal[mask == int(label)]
        values.append(float(np.percentile(object_values, 90)))
    array = np.asarray(values)
    return {
        "strong_signal_fraction": float(np.mean(array >= strong_threshold)),
        "median_object_p90": float(np.median(array)),
    }


def cell_support_fraction(mask: np.ndarray, cell_mask: np.ndarray) -> float:
    labels, _areas, centroids = object_stats(mask)
    if labels.size == 0:
        return 0.0
    yy = np.clip(np.rint(centroids[labels, 0]).astype(int), 0, mask.shape[0] - 1)
    xx = np.clip(np.rint(centroids[labels, 1]).astype(int), 0, mask.shape[1] - 1)
    return float(np.mean(cell_mask[yy, xx] > 0))


def aggregate_candidate(rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    dead_total = sum(int(row["dead_objects"]) for row in rows)
    blue_total = sum(int(row["blue_objects"]) for row in rows)
    matched = sum(int(row["matched_objects"]) for row in rows)
    precision = matched / blue_total if blue_total else 0.0
    recall = matched / dead_total if dead_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_ratio = blue_total / max(dead_total, 1)
    count_penalty = math.exp(-0.20 * abs(math.log(max(count_ratio, 1e-6))))
    cell_support = float(np.median([float(row["cell_support_fraction"]) for row in rows]))
    strong_signal = float(np.median([float(row["strong_signal_fraction"]) for row in rows]))
    composite = f1 * count_penalty * math.sqrt(max(cell_support, 0.0)) * (0.5 + 0.5 * strong_signal)
    return {
        **asdict(candidate),
        "n_fields": len(rows),
        "dead_objects": dead_total,
        "blue_objects": blue_total,
        "matched_objects": matched,
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": f1,
        "count_ratio_blue_to_dead": count_ratio,
        "median_cell_support_fraction": cell_support,
        "median_strong_signal_fraction": strong_signal,
        "median_mask_fraction": float(np.median([float(row["mask_fraction"]) for row in rows])),
        "min_field_match_f1": min(float(row["match_f1"]) for row in rows),
        "composite_score": composite,
    }


def main() -> int:
    args = parse_args()
    version = metadata.version("cellpose")
    if version != REQUIRED_CELLPOSE_VERSION:
        raise SystemExit(f"Expected cellpose=={REQUIRED_CELLPOSE_VERSION}, found {version}")
    from cellpose import models

    calibration = json.loads(args.calibration_json.read_text())
    combined_index = index_files(args.input_root / "Combined")
    dead_mask_index = index_files(args.baseline_run / "Dead" / "segmentations", "*_cp_masks.tif")
    combined_cell_index = index_files(
        args.baseline_run / "Combined" / "segmentations", "*_cp_masks.tif"
    )
    keys = sorted(set(combined_index) & set(dead_mask_index) & set(combined_cell_index))
    if args.limit is not None:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit("No common Combined/dead-mask/Combined-cell fields")
    candidates = coarse_candidates() if args.candidate_set == "coarse" else fine_candidates()
    if args.candidate_tag:
        wanted = set(args.candidate_tag)
        candidates = [candidate for candidate in candidates if candidate.tag in wanted]
    if not candidates:
        raise SystemExit("No candidates selected")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "candidate_configs.json", [asdict(candidate) for candidate in candidates])
    model_cache: dict[str, Any] = {}
    field_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        model = model_cache.get(candidate.model)
        if model is None:
            print(f"loading_model={candidate.model}", flush=True)
            model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=candidate.model)
            model_cache[candidate.model] = model
            print(f"model_device={getattr(model, 'device', 'unknown')}", flush=True)
        low, high = calibration_for(calibration, candidate.signal_transform)
        for field_index, key in enumerate(keys, start=1):
            mask_path = args.out_root / "candidates" / candidate.tag / "segmentations" / f"{key}_cp_masks.tif"
            raw_rgb = tifffile.imread(combined_index[key])
            signal = blue_signals(raw_rgb)[candidate.signal_transform]
            start = time.time()
            if mask_path.exists() and not args.force:
                mask = read_mask(mask_path)
                elapsed = 0.0
            else:
                prepared = prepare_signal(signal, candidate, calibration)
                mask, *_ = model.eval(
                    prepared,
                    channel_axis=None,
                    normalize=False,
                    diameter=candidate.diameter,
                    flow_threshold=candidate.flow_threshold,
                    cellprob_threshold=candidate.cellprob_threshold,
                    min_size=candidate.min_size,
                )
                mask = np.asarray(mask, dtype=np.int32)
                write_mask(mask_path, mask)
                elapsed = time.time() - start
            dead_mask = read_mask(dead_mask_index[key])
            cell_mask = read_mask(combined_cell_index[key])
            match = match_masks(dead_mask, mask, args.match_distance, args.min_overlap_pixels)
            signal_metrics = mask_signal_support(mask, signal, high)
            row = {
                "candidate": candidate.tag,
                "key": key,
                **asdict(candidate),
                "calibration_low": low,
                "calibration_high": high,
                "elapsed_sec": elapsed,
                "mask_fraction": float(np.mean(mask > 0)),
                "cell_support_fraction": cell_support_fraction(mask, cell_mask),
                **match,
                **signal_metrics,
                "mask_path": str(mask_path),
            }
            field_rows.append(row)
            print(
                f"candidate={candidate_index}/{len(candidates)} field={field_index}/{len(keys)} "
                f"tag={candidate.tag} key={key} objects={match['blue_objects']} f1={match['match_f1']:.3f}",
                flush=True,
            )
    write_rows(args.out_root / "candidate_field_metrics.csv", field_rows)
    summaries = [
        aggregate_candidate(
            [row for row in field_rows if row["candidate"] == candidate.tag], candidate
        )
        for candidate in candidates
    ]
    summaries.sort(key=lambda row: float(row["composite_score"]), reverse=True)
    write_rows(args.out_root / "candidate_summary.csv", summaries)
    write_json(args.out_root / "best_candidate.json", summaries[0])
    (args.out_root / "_SUCCESS").write_text(json.dumps(summaries[0], sort_keys=True) + "\n")
    print(f"screen_complete={len(candidates)} best={summaries[0]['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
