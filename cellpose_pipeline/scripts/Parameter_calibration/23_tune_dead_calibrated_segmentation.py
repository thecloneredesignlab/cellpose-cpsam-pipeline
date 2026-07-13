#!/usr/bin/env python3
"""GPU screen of cohort-calibrated Dead segmentation candidates on large test."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_SCRIPT_DIR = SCRIPT_DIR.parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = load_module("cellpose_workflow01", PRODUCTION_SCRIPT_DIR / "01_segment_images.py")
blue_tune = load_module("combined_blue_tune22", SCRIPT_DIR / "22_tune_combined_blue_segmentation.py")


@dataclass(frozen=True)
class Candidate:
    tag: str
    calibration_mode: str
    diameter: float
    cellprob_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--calibration-map", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-tag", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def candidates() -> list[Candidate]:
    result: list[Candidate] = []
    parameter_pairs = ((22.0, -3.0), (20.0, -3.0), (24.0, -3.0), (22.0, -2.75), (22.0, -3.25))
    for mode in ("legacy", "global", "bounded-background"):
        for diameter, cellprob in parameter_pairs:
            tag = f"{mode}_d{int(diameter)}_cp{str(cellprob).replace('-', 'm').replace('.', 'p')}"
            result.append(Candidate(tag, mode, diameter, cellprob))
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_mask(path: Path) -> np.ndarray:
    return np.squeeze(tifffile.imread(path)).astype(np.int32, copy=False)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if int(mask.max()) > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, mask.astype(dtype, copy=False))


def centroid_support(mask: np.ndarray, support_mask: np.ndarray) -> float:
    labels, _areas, centroids = blue_tune.object_stats(mask)
    if labels.size == 0:
        return 0.0
    yy = np.clip(np.rint(centroids[labels, 0]).astype(int), 0, mask.shape[0] - 1)
    xx = np.clip(np.rint(centroids[labels, 1]).astype(int), 0, mask.shape[1] - 1)
    return float(np.mean(support_mask[yy, xx] > 0))


def candidate_profile(candidate: Candidate, key: str, calibration: dict[str, Any], per_key: dict[str, dict[str, str]]) -> Any:
    profile = workflow.FOLDER_PROFILES["Dead"]
    if candidate.calibration_mode == "legacy":
        low, high = 7.6, 55.0
    else:
        dead = calibration.get("dead", calibration)
        low = float(dead["recommended_fixed_low"])
        high = float(dead["recommended_fixed_high"])
        if candidate.calibration_mode == "bounded-background":
            bounds = dead["bounded_background_low_range"]
            image_low = float(per_key[key]["dead_q1p0"])
            low = float(np.clip(image_low, float(bounds[0]), float(bounds[1])))
            high = low + float(dead["recommended_background_span"])
    return replace(
        profile,
        diameter=candidate.diameter,
        cellprob_threshold=candidate.cellprob_threshold,
        fixed_low_value=low,
        fixed_high_value=high,
    )


def aggregate(rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    dead_total = sum(int(row["baseline_objects"]) for row in rows)
    candidate_total = sum(int(row["candidate_objects"]) for row in rows)
    matched = sum(int(row["matched_objects"]) for row in rows)
    precision = matched / max(candidate_total, 1)
    recall = matched / max(dead_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    ratio = candidate_total / max(dead_total, 1)
    stability = math.exp(-abs(math.log(max(ratio, 1e-9))))
    cell_support = float(np.median([float(row["cell_support_fraction"]) for row in rows]))
    nucleus_support = float(np.median([float(row["nucleus_support_fraction"]) for row in rows]))
    score = f1 * math.sqrt(max(cell_support, 0.0)) * (0.75 + 0.25 * nucleus_support) * stability
    return {
        **asdict(candidate),
        "n_fields": len(rows),
        "baseline_objects": dead_total,
        "candidate_objects": candidate_total,
        "matched_objects": matched,
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": f1,
        "count_ratio_to_legacy": ratio,
        "median_cell_support_fraction": cell_support,
        "median_nucleus_support_fraction": nucleus_support,
        "median_mask_fraction": float(np.median([float(row["mask_fraction"]) for row in rows])),
        "median_filter_retention": float(np.median([float(row["filter_retention"]) for row in rows])),
        "composite_score": score,
    }


def main() -> int:
    args = parse_args()
    from cellpose import models

    calibration = json.loads(args.calibration_json.read_text())
    per_key = workflow.load_dead_calibration_map(args.calibration_map)
    dead_index = blue_tune.index_files(args.input_root / "Dead")
    baseline_index = blue_tune.index_files(args.baseline_run / "Dead" / "segmentations", "*_cp_masks.tif")
    cell_index = blue_tune.index_files(args.baseline_run / "Combined" / "segmentations", "*_cp_masks.tif")
    nucleus_index = blue_tune.index_files(args.baseline_run / "Nuclei" / "segmentations", "*_cp_masks.tif")
    keys = sorted(set(dead_index) & set(baseline_index) & set(cell_index) & set(nucleus_index) & set(per_key))
    if args.limit is not None:
        keys = keys[: args.limit]
    selected = candidates()
    if args.candidate_tag:
        wanted = set(args.candidate_tag)
        selected = [candidate for candidate in selected if candidate.tag in wanted]
    if not keys or not selected:
        raise SystemExit("No fields or candidates selected")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "candidate_configs.json", [asdict(candidate) for candidate in selected])
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model="cpsam_v2")
    print(f"model_device={getattr(model, 'device', 'unknown')}", flush=True)
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected, start=1):
        for field_index, key in enumerate(keys, start=1):
            raw = tifffile.imread(dead_index[key]).astype(np.float32, copy=False)
            profile = candidate_profile(candidate, key, calibration, per_key)
            mask_path = args.out_root / "candidates" / candidate.tag / "segmentations" / f"{key}_cp_masks.tif"
            if mask_path.exists() and not args.force:
                filtered = read_mask(mask_path)
                unfiltered_count = int(filtered.max())
                elapsed = 0.0
            else:
                prepared = workflow.prepare_for_cellpose(raw, profile)
                start = time.time()
                unfiltered, *_ = model.eval(
                    prepared,
                    channel_axis=None,
                    normalize=False,
                    diameter=profile.diameter,
                    flow_threshold=profile.flow_threshold,
                    cellprob_threshold=profile.cellprob_threshold,
                    min_size=profile.min_size,
                )
                elapsed = time.time() - start
                unfiltered = np.asarray(unfiltered, dtype=np.int32)
                unfiltered_count = int(unfiltered.max())
                filtered = workflow.postprocess_dead_raw_signal(raw, unfiltered, profile).astype(np.int32, copy=False)
                write_mask(mask_path, filtered)
            baseline = read_mask(baseline_index[key])
            match = blue_tune.match_masks(baseline, filtered, 20.0, 3)
            row = {
                "candidate": candidate.tag,
                "key": key,
                **asdict(candidate),
                "fixed_low": profile.fixed_low_value,
                "fixed_high": profile.fixed_high_value,
                "elapsed_sec": elapsed,
                "unfiltered_objects": unfiltered_count,
                "filter_retention": int(filtered.max()) / max(unfiltered_count, 1),
                "mask_fraction": float(np.mean(filtered > 0)),
                "cell_support_fraction": centroid_support(filtered, read_mask(cell_index[key])),
                "nucleus_support_fraction": centroid_support(filtered, read_mask(nucleus_index[key])),
                "baseline_objects": match["dead_objects"],
                "candidate_objects": match["blue_objects"],
                "matched_objects": match["matched_objects"],
                "match_f1": match["match_f1"],
                "mask_path": str(mask_path),
            }
            rows.append(row)
            print(
                f"candidate={candidate_index}/{len(selected)} field={field_index}/{len(keys)} "
                f"tag={candidate.tag} key={key} objects={row['candidate_objects']} f1={row['match_f1']:.3f}",
                flush=True,
            )
    write_rows(args.out_root / "candidate_field_metrics.csv", rows)
    summaries = [aggregate([row for row in rows if row["candidate"] == candidate.tag], candidate) for candidate in selected]
    summaries.sort(key=lambda row: float(row["composite_score"]), reverse=True)
    write_rows(args.out_root / "candidate_summary.csv", summaries)
    write_json(args.out_root / "best_candidate.json", summaries[0])
    (args.out_root / "_SUCCESS").write_text(json.dumps(summaries[0], sort_keys=True) + "\n")
    print(f"dead_calibrated_screen_complete={len(selected)} best={summaries[0]['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
