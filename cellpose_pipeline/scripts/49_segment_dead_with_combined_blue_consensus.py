#!/usr/bin/env python3
"""Segment Dead and Combined blue-excess signals and build a conservative consensus mask."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = load_module("cellpose_workflow18_consensus", SCRIPT_DIR / "18_run_segmentation_classification_workflow.py")
blue_utils = load_module("combined_blue_tune46_consensus", SCRIPT_DIR / "46_tune_combined_blue_segmentation.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dead-image", type=Path, required=True)
    parser.add_argument("--combined-image", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--calibration-map", type=Path, required=True)
    parser.add_argument("--dead-calibration-mode", choices=("global", "bounded-background"), default="bounded-background")
    parser.add_argument("--dead-model", default="cpsam_v2")
    parser.add_argument("--dead-diameter", type=float, default=22.0)
    parser.add_argument("--dead-cellprob-threshold", type=float, default=-2.75)
    parser.add_argument("--blue-model", default="cpsam")
    parser.add_argument("--blue-transform", choices=("blue_excess_mean", "blue_excess_max"), default="blue_excess_mean")
    parser.add_argument("--blue-diameter", type=float, default=22.0)
    parser.add_argument("--blue-cellprob-threshold", type=float, default=-3.0)
    parser.add_argument("--match-distance", type=float, default=20.0)
    parser.add_argument("--min-overlap-pixels", type=int, default=3)
    parser.add_argument("--rescue-min-area", type=int, default=20)
    parser.add_argument("--rescue-max-area", type=int, default=2500)
    parser.add_argument("--rescue-min-dead-p90", type=float, default=15.0)
    parser.add_argument("--rescue-min-dead-p90-delta", type=float, default=1.0)
    parser.add_argument("--rescue-min-dead-snr", type=float, default=1.5)
    parser.add_argument("--rescue-max-primary-overlap", type=float, default=0.10)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if int(mask.max()) > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, mask.astype(dtype, copy=False))


def dead_profile(args: argparse.Namespace, key: str, calibration: dict[str, Any], per_key: dict[str, dict[str, str]]) -> Any:
    block = calibration.get("dead", calibration)
    low = float(block["recommended_fixed_low"])
    high = float(block["recommended_fixed_high"])
    if args.dead_calibration_mode == "bounded-background":
        row = per_key.get(key)
        if row is None:
            raise KeyError(f"No full-cohort calibration row for {key}")
        bounds = block["bounded_background_low_range"]
        low = float(np.clip(float(row["dead_q1p0"]), float(bounds[0]), float(bounds[1])))
        high = low + float(block["recommended_background_span"])
    return replace(
        workflow.FOLDER_PROFILES["Dead"],
        model=args.dead_model,
        diameter=args.dead_diameter,
        cellprob_threshold=args.dead_cellprob_threshold,
        fixed_low_value=low,
        fixed_high_value=high,
    )


def match_pairs(dead: np.ndarray, blue: np.ndarray, max_distance: float, min_overlap: int) -> list[dict[str, Any]]:
    dead_labels, dead_areas, dead_centroids = blue_utils.object_stats(dead)
    blue_labels, blue_areas, blue_centroids = blue_utils.object_stats(blue)
    candidates: dict[tuple[int, int], tuple[float, int, float, float]] = {}
    foreground = (dead > 0) & (blue > 0)
    if np.any(foreground):
        pairs, counts = np.unique(np.column_stack((dead[foreground], blue[foreground])), axis=0, return_counts=True)
        for (dead_id, blue_id), overlap in zip(pairs, counts):
            dead_id, blue_id, overlap = int(dead_id), int(blue_id), int(overlap)
            if overlap < min_overlap:
                continue
            union = int(dead_areas[dead_id] + blue_areas[blue_id] - overlap)
            iou = overlap / max(union, 1)
            distance = float(np.linalg.norm(dead_centroids[dead_id] - blue_centroids[blue_id]))
            candidates[(dead_id, blue_id)] = (2.0 + iou, overlap, iou, distance)
    if dead_labels.size and blue_labels.size:
        tree = cKDTree(dead_centroids[dead_labels])
        distances, indexes = tree.query(blue_centroids[blue_labels], k=1)
        for blue_id, distance, index in zip(blue_labels, distances, indexes):
            if float(distance) > max_distance:
                continue
            dead_id = int(dead_labels[int(index)])
            candidates.setdefault(
                (dead_id, int(blue_id)),
                (1.0 - float(distance) / max_distance, 0, 0.0, float(distance)),
            )
    ordered = sorted(((score, dead_id, blue_id, overlap, iou, distance) for (dead_id, blue_id), (score, overlap, iou, distance) in candidates.items()), reverse=True)
    used_dead: set[int] = set()
    used_blue: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _score, dead_id, blue_id, overlap, iou, distance in ordered:
        if dead_id in used_dead or blue_id in used_blue:
            continue
        used_dead.add(dead_id)
        used_blue.add(blue_id)
        matches.append({"dead_id": dead_id, "blue_id": blue_id, "overlap_pixels": overlap, "iou": iou, "centroid_distance": distance})
    return matches


def local_dead_evidence(raw: np.ndarray, blue_mask: np.ndarray, label_id: int, obj_slice: tuple[slice, slice], margin: int = 14) -> dict[str, float]:
    obj = blue_mask[obj_slice] == label_id
    area = int(obj.sum())
    y0, y1 = obj_slice[0].start or 0, obj_slice[0].stop or raw.shape[0]
    x0, x1 = obj_slice[1].start or 0, obj_slice[1].stop or raw.shape[1]
    yy0, yy1 = max(0, y0 - margin), min(raw.shape[0], y1 + margin)
    xx0, xx1 = max(0, x0 - margin), min(raw.shape[1], x1 + margin)
    object_values = raw[obj_slice][obj]
    window_labels = blue_mask[yy0:yy1, xx0:xx1]
    background = raw[yy0:yy1, xx0:xx1][window_labels == 0]
    if background.size < 25:
        background = raw.ravel()
    bg_median = float(np.median(background))
    bg_mad = float(np.median(np.abs(background - bg_median)))
    bg_sigma = max(1.4826 * bg_mad, 1e-6)
    p90 = float(np.percentile(object_values, 90))
    return {
        "area": area,
        "dead_mean": float(np.mean(object_values)),
        "dead_p90": p90,
        "dead_bg_median": bg_median,
        "dead_p90_delta": p90 - bg_median,
        "dead_snr": (p90 - bg_median) / bg_sigma,
    }


def main() -> int:
    args = parse_args()
    from cellpose import models

    key_dead = workflow.extract_image_key(args.dead_image)
    key_combined = workflow.extract_image_key(args.combined_image)
    if key_dead != key_combined:
        raise SystemExit(f"Dead/Combined key mismatch: {key_dead} vs {key_combined}")
    key = key_dead
    calibration = json.loads(args.calibration_json.read_text())
    per_key = workflow.load_dead_calibration_map(args.calibration_map)
    profile = dead_profile(args, key, calibration, per_key)
    final_path = args.out_root / "segmentations" / f"{args.dead_image.stem}_cp_masks.tif"
    if final_path.exists() and not args.force:
        print(f"Using existing consensus segmentation: {final_path}")
        return 0
    raw_dead = np.squeeze(tifffile.imread(args.dead_image)).astype(np.float32, copy=False)
    raw_combined = tifffile.imread(args.combined_image)
    if raw_dead.ndim != 2 or raw_combined.shape[:2] != raw_dead.shape:
        raise ValueError(f"Dead/Combined shape mismatch: {raw_dead.shape} vs {raw_combined.shape}")
    model_cache: dict[str, Any] = {}
    elapsed: dict[str, float] = {}

    def get_model(name: str) -> Any:
        if name not in model_cache:
            model_cache[name] = models.CellposeModel(gpu=args.use_gpu, pretrained_model=name)
            print(f"model={name} device={getattr(model_cache[name], 'device', 'unknown')}", flush=True)
        return model_cache[name]

    start = time.time()
    dead_unfiltered, *_ = get_model(args.dead_model).eval(
        workflow.prepare_for_cellpose(raw_dead, profile),
        channel_axis=None,
        normalize=False,
        diameter=profile.diameter,
        flow_threshold=profile.flow_threshold,
        cellprob_threshold=profile.cellprob_threshold,
        min_size=profile.min_size,
    )
    dead_primary = workflow.postprocess_dead_raw_signal(raw_dead, np.asarray(dead_unfiltered), profile).astype(np.int32, copy=False)
    elapsed["dead_sec"] = time.time() - start

    blue_candidate = blue_utils.Candidate(
        tag="production_combined_blue",
        model=args.blue_model,
        signal_transform=args.blue_transform,
        diameter=args.blue_diameter,
        cellprob_threshold=args.blue_cellprob_threshold,
    )
    blue_signal = blue_utils.blue_signals(raw_combined)[args.blue_transform]
    blue_prepared = blue_utils.prepare_signal(blue_signal, blue_candidate, calibration)
    start = time.time()
    combined_blue, *_ = get_model(args.blue_model).eval(
        blue_prepared,
        channel_axis=None,
        normalize=False,
        diameter=args.blue_diameter,
        flow_threshold=0.0,
        cellprob_threshold=args.blue_cellprob_threshold,
        min_size=10,
    )
    combined_blue = np.asarray(combined_blue, dtype=np.int32)
    elapsed["combined_blue_sec"] = time.time() - start

    matches = match_pairs(dead_primary, combined_blue, args.match_distance, args.min_overlap_pixels)
    matched_dead = {int(row["dead_id"]) for row in matches}
    matched_blue = {int(row["blue_id"]) for row in matches}
    final = np.zeros(dead_primary.shape, dtype=np.uint32)
    provenance: list[dict[str, Any]] = []
    next_id = 1
    for dead_id in np.unique(dead_primary):
        dead_id = int(dead_id)
        if dead_id == 0:
            continue
        final[dead_primary == dead_id] = next_id
        matched = next((row for row in matches if int(row["dead_id"]) == dead_id), None)
        provenance.append(
            {
                "key": key,
                "final_id": next_id,
                "source": "dual_channel" if dead_id in matched_dead else "dead_primary_only",
                "dead_id": dead_id,
                "blue_id": int(matched["blue_id"]) if matched else 0,
                "overlap_pixels": int(matched["overlap_pixels"]) if matched else 0,
                "iou": float(matched["iou"]) if matched else 0.0,
                "centroid_distance": float(matched["centroid_distance"]) if matched else -1.0,
                "rescue_reason": "",
            }
        )
        next_id += 1
    rescued = 0
    rejected_blue_only = 0
    blue_only_rows: list[dict[str, Any]] = []
    for blue_id, obj_slice in enumerate(ndi.find_objects(combined_blue), start=1):
        if obj_slice is None or blue_id in matched_blue:
            continue
        evidence = local_dead_evidence(raw_dead, combined_blue, blue_id, obj_slice)
        obj = combined_blue == blue_id
        primary_overlap = float(np.mean(dead_primary[obj] > 0)) if np.any(obj) else 1.0
        keep = (
            args.rescue_min_area <= evidence["area"] <= args.rescue_max_area
            and evidence["dead_p90"] >= args.rescue_min_dead_p90
            and evidence["dead_p90_delta"] >= args.rescue_min_dead_p90_delta
            and evidence["dead_snr"] >= args.rescue_min_dead_snr
            and primary_overlap <= args.rescue_max_primary_overlap
        )
        candidate_row = {
            "key": key,
            "blue_id": blue_id,
            "primary_overlap_fraction": primary_overlap,
            **evidence,
            "selected_for_rescue": False,
            "decision": "failed_rescue_thresholds",
        }
        if not keep:
            rejected_blue_only += 1
            blue_only_rows.append(candidate_row)
            continue
        pixels = obj & (final == 0)
        if int(pixels.sum()) < args.rescue_min_area:
            rejected_blue_only += 1
            candidate_row["decision"] = "insufficient_unoccupied_area"
            blue_only_rows.append(candidate_row)
            continue
        final[pixels] = next_id
        provenance.append(
            {
                "key": key,
                "final_id": next_id,
                "source": "combined_blue_rescue",
                "dead_id": 0,
                "blue_id": blue_id,
                "overlap_pixels": 0,
                "iou": 0.0,
                "centroid_distance": -1.0,
                "rescue_reason": "relaxed_dead_raw_evidence",
                **evidence,
            }
        )
        next_id += 1
        rescued += 1
        candidate_row["selected_for_rescue"] = True
        candidate_row["decision"] = "rescued"
        blue_only_rows.append(candidate_row)

    write_mask(final_path, final)
    write_mask(args.out_root / "intermediate" / "dead_primary" / f"{key}_cp_masks.tif", dead_primary)
    write_mask(args.out_root / "intermediate" / "combined_blue" / f"{key}_cp_masks.tif", combined_blue)
    provenance_columns = [
        "key", "final_id", "source", "dead_id", "blue_id", "overlap_pixels", "iou", "centroid_distance",
        "rescue_reason", "area", "dead_mean", "dead_p90", "dead_bg_median", "dead_p90_delta", "dead_snr",
    ]
    write_rows(args.out_root / "object_provenance" / f"{key}.csv", provenance, provenance_columns)
    blue_only_columns = [
        "key", "blue_id", "primary_overlap_fraction", "area", "dead_mean", "dead_p90", "dead_bg_median",
        "dead_p90_delta", "dead_snr", "selected_for_rescue", "decision",
    ]
    write_rows(args.out_root / "blue_only_candidates" / f"{key}.csv", blue_only_rows, blue_only_columns)
    summary = {
        "key": key,
        "dead_image": str(args.dead_image),
        "combined_image": str(args.combined_image),
        "final_mask": str(final_path),
        "dead_primary_objects": int(dead_primary.max()),
        "combined_blue_objects": int(combined_blue.max()),
        "matched_objects": len(matches),
        "dead_primary_only_objects": int(dead_primary.max()) - len(matches),
        "combined_blue_rescued_objects": rescued,
        "combined_blue_rejected_objects": rejected_blue_only,
        "final_objects": int(final.max()),
        "final_mask_fraction": float(np.mean(final > 0)),
        "dead_profile": asdict(profile),
        "blue_profile": asdict(blue_candidate),
        "consensus_parameters": {
            "match_distance": args.match_distance,
            "min_overlap_pixels": args.min_overlap_pixels,
            "rescue_min_area": args.rescue_min_area,
            "rescue_max_area": args.rescue_max_area,
            "rescue_min_dead_p90": args.rescue_min_dead_p90,
            "rescue_min_dead_p90_delta": args.rescue_min_dead_p90_delta,
            "rescue_min_dead_snr": args.rescue_min_dead_snr,
            "rescue_max_primary_overlap": args.rescue_max_primary_overlap,
        },
        **elapsed,
    }
    write_json(args.out_root / "metadata" / f"{key}.json", summary)
    write_rows(args.out_root / "field_summaries" / f"{key}.csv", [summary], list(summary))
    print(f"consensus_complete key={key} final={summary['final_objects']} rescued={rescued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
