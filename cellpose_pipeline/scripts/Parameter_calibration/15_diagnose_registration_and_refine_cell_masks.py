#!/usr/bin/env python3
"""Diagnose global channel offsets and generate nucleus-aware cell-mask candidates.

The refinement is deliberately local.  It never splits or merges cell labels and it
does not impose one nucleus per cell.  A nuclear region is reassigned only when its
intensity-supported core identifies a sufficiently strong BF/Combined cell pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
FOCUS_KEYS = {
    "E3_3_04d06h00m",
    "E3_2_04d22h00m",
    "C2_2_04d04h00m",
    "C11_1_03d02h00m",
    "G8_2_04d20h00m",
    "A11_1_00d12h00m",
}


@dataclass(frozen=True)
class RepairConfig:
    tag: str
    min_core_primary_fraction: float
    consensus: str
    require_both_direct: bool
    region: str
    fill_background: bool
    max_nearest_distance: float
    max_changed_pixels_per_profile: int = 0
    min_remaining_label_area: int = 0


REPAIR_CONFIGS = (
    RepairConfig("strict_relabel", 0.80, "reciprocal", True, "extent", False, 0.0),
    RepairConfig("strict_fill", 0.80, "reciprocal", True, "extent", True, 0.0),
    RepairConfig("balanced_relabel", 0.65, "reciprocal", True, "extent", False, 0.0),
    RepairConfig("balanced_fill", 0.65, "reciprocal", False, "extent", True, 3.0),
    RepairConfig("balanced_oneway", 0.65, "oneway", False, "extent", True, 3.0),
    RepairConfig("core4_fill", 0.65, "reciprocal", False, "core_dilate4", True, 3.0),
    RepairConfig("safe1_relabel", 0.80, "reciprocal", True, "extent", False, 0.0, 1, 35),
    RepairConfig("safe3_relabel", 0.80, "reciprocal", True, "extent", False, 0.0, 3, 35),
    RepairConfig("safe5_relabel", 0.80, "reciprocal", True, "extent", False, 0.0, 5, 35),
    RepairConfig("safe10_relabel", 0.80, "reciprocal", True, "extent", False, 0.0, 10, 35),
)


@dataclass(frozen=True)
class CoreSupport:
    primary_label: int
    primary_fraction: float
    secondary_label: int
    secondary_fraction: float
    centroid_label: int
    nearest_label: int
    nearest_distance: float


@dataclass(frozen=True)
class LocalRepairPlan:
    selected: np.ndarray
    changed_pixels: int
    filled_pixels: int
    relabeled_pixels: int
    source_label_counts: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Nuclei-to-cell registration and generate conservative cell-mask repair candidates."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--keys", nargs="*")
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def index_masks(mask_dir: Path, pattern: str) -> dict[str, Path]:
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    result: dict[str, Path] = {}
    for path in sorted(mask_dir.glob(pattern)):
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate key {key}: {result[key]} and {path}")
        result[key] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer mask at {path}, got {labels.shape} {labels.dtype}")
    return labels.astype(np.int32, copy=False)


def write_mask(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = int(labels.max()) if labels.size else 0
    dtype = np.uint32 if maximum > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, labels.astype(dtype, copy=False), compression="zlib")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def link_directory(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Cannot replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def label_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def instance_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(labels)
    if not yy.size:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 2), dtype=np.float64)
    ids = labels[yy, xx].astype(np.int64, copy=False)
    maximum = int(ids.max())
    areas = np.bincount(ids, minlength=maximum + 1).astype(np.float64)
    sum_y = np.bincount(ids, weights=yy, minlength=maximum + 1)
    sum_x = np.bincount(ids, weights=xx, minlength=maximum + 1)
    valid = np.flatnonzero(areas > 0)
    valid = valid[valid > 0]
    centers = np.column_stack((sum_y[valid] / areas[valid], sum_x[valid] / areas[valid]))
    return valid.astype(np.int32), centers


def registration_metric(
    cell_mask: np.ndarray,
    centers: np.ndarray,
    dy: int,
    dx: int,
    boundary_distance: np.ndarray,
    background_distance: np.ndarray,
) -> dict[str, float | int]:
    height, width = cell_mask.shape
    source_y = np.rint(centers[:, 0]).astype(np.int64) - int(dy)
    source_x = np.rint(centers[:, 1]).astype(np.int64) - int(dx)
    valid = (source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width)
    sampled_labels = np.zeros(len(centers), dtype=np.int32)
    sampled_clearance = np.zeros(len(centers), dtype=np.float64)
    sampled_background = np.full(len(centers), 8.0, dtype=np.float64)
    sampled_labels[valid] = cell_mask[source_y[valid], source_x[valid]]
    sampled_clearance[valid] = boundary_distance[source_y[valid], source_x[valid]]
    sampled_background[valid] = background_distance[source_y[valid], source_x[valid]]
    inside = sampled_labels > 0
    inside_rate = float(np.mean(inside)) if len(inside) else 0.0
    clearance = float(np.mean(np.minimum(sampled_clearance[inside], 8.0) / 8.0)) if np.any(inside) else 0.0
    background = float(np.mean(np.minimum(sampled_background[~inside], 8.0) / 8.0)) if np.any(~inside) else 0.0
    score = inside_rate + 0.15 * clearance - 0.10 * background - 0.001 * (dy * dy + dx * dx)
    return {
        "dy": int(dy),
        "dx": int(dx),
        "score": score,
        "centroid_inside_rate": inside_rate,
        "inside_boundary_clearance_score": clearance,
        "outside_background_distance_score": background,
    }


def diagnose_registration(cell_mask: np.ndarray, core_mask: np.ndarray, max_shift: int) -> dict[str, Any]:
    _ids, centers = instance_centroids(core_mask)
    if not len(centers):
        raise ValueError("Cannot diagnose registration without nucleus-core centroids")
    boundary_distance = ndimage.distance_transform_edt(~label_boundary(cell_mask))
    background_distance = ndimage.distance_transform_edt(cell_mask == 0)
    candidates = [
        registration_metric(cell_mask, centers, dy, dx, boundary_distance, background_distance)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    ]
    candidates.sort(key=lambda row: (-float(row["score"]), abs(int(row["dy"])) + abs(int(row["dx"]))))
    baseline = next(row for row in candidates if row["dy"] == 0 and row["dx"] == 0)
    best = candidates[0]
    second = candidates[1]
    return {
        "n_core_centroids": len(centers),
        "baseline_score": baseline["score"],
        "baseline_centroid_inside_rate": baseline["centroid_inside_rate"],
        "best_dy": best["dy"],
        "best_dx": best["dx"],
        "best_score": best["score"],
        "best_centroid_inside_rate": best["centroid_inside_rate"],
        "score_gain": float(best["score"]) - float(baseline["score"]),
        "inside_rate_gain": float(best["centroid_inside_rate"]) - float(baseline["centroid_inside_rate"]),
        "best_to_second_margin": float(best["score"]) - float(second["score"]),
        "best_at_search_edge": abs(int(best["dy"])) == max_shift or abs(int(best["dx"])) == max_shift,
    }


def best_overlap_maps(
    combined: np.ndarray,
    brightfield: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    overlap = (combined > 0) & (brightfield > 0)
    if not np.any(overlap):
        return {}, {}
    base = int(brightfield.max()) + 1
    codes = combined[overlap].astype(np.int64) * base + brightfield[overlap].astype(np.int64)
    unique, counts = np.unique(codes, return_counts=True)
    best_combined: dict[int, tuple[int, int]] = {}
    best_brightfield: dict[int, tuple[int, int]] = {}
    for code, count_value in zip(unique, counts):
        combined_id = int(code // base)
        brightfield_id = int(code % base)
        count = int(count_value)
        if count > best_combined.get(combined_id, (0, 0))[1]:
            best_combined[combined_id] = (brightfield_id, count)
        if count > best_brightfield.get(brightfield_id, (0, 0))[1]:
            best_brightfield[brightfield_id] = (combined_id, count)
    return (
        {label: value[0] for label, value in best_combined.items()},
        {label: value[0] for label, value in best_brightfield.items()},
    )


def nearest_label_context(cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(cell_mask > 0):
        return np.zeros(cell_mask.shape, dtype=np.int32), np.full(cell_mask.shape, np.inf)
    distance, indices = ndimage.distance_transform_edt(cell_mask == 0, return_indices=True)
    nearest = cell_mask[indices[0], indices[1]].astype(np.int32, copy=False)
    return nearest, distance


def core_support(
    cell_mask: np.ndarray,
    core_mask: np.ndarray,
    nucleus_id: int,
    obj_slice: tuple[slice, slice],
    nearest_labels: np.ndarray,
    nearest_distance: np.ndarray,
) -> CoreSupport:
    core = core_mask[obj_slice] == nucleus_id
    area = int(np.count_nonzero(core))
    if area <= 0:
        return CoreSupport(0, 0.0, 0, 0.0, 0, 0, float("inf"))
    labels = cell_mask[obj_slice][core]
    foreground = labels[labels > 0]
    if foreground.size:
        counts = np.bincount(foreground.astype(np.int64, copy=False))
        order = np.argsort(counts)[::-1]
        order = order[counts[order] > 0]
        primary = int(order[0])
        secondary = int(order[1]) if len(order) > 1 else 0
        primary_fraction = float(counts[primary] / area)
        secondary_fraction = float(counts[secondary] / area) if secondary else 0.0
    else:
        primary = secondary = 0
        primary_fraction = secondary_fraction = 0.0
    yy, xx = np.nonzero(core)
    y = int(round(float(yy.mean() + (obj_slice[0].start or 0))))
    x = int(round(float(xx.mean() + (obj_slice[1].start or 0))))
    y = min(max(y, 0), cell_mask.shape[0] - 1)
    x = min(max(x, 0), cell_mask.shape[1] - 1)
    centroid_label = int(cell_mask[y, x])
    return CoreSupport(
        primary,
        primary_fraction,
        secondary,
        secondary_fraction,
        centroid_label,
        int(nearest_labels[y, x]),
        float(nearest_distance[y, x]),
    )


def target_label(support: CoreSupport) -> int:
    if support.primary_label > 0:
        return support.primary_label
    if support.centroid_label > 0:
        return support.centroid_label
    return support.nearest_label


def pair_consensus(
    combined_id: int,
    brightfield_id: int,
    combined_to_bf: dict[int, int],
    bf_to_combined: dict[int, int],
) -> str:
    if combined_id <= 0 or brightfield_id <= 0:
        return "missing"
    combined_best = combined_to_bf.get(combined_id) == brightfield_id
    brightfield_best = bf_to_combined.get(brightfield_id) == combined_id
    if combined_best and brightfield_best:
        return "reciprocal"
    if combined_best or brightfield_best:
        return "oneway"
    return "conflict"


def consensus_allowed(actual: str, requested: str) -> bool:
    if requested == "reciprocal":
        return actual == "reciprocal"
    if requested == "oneway":
        return actual in {"reciprocal", "oneway"}
    raise ValueError(f"Unsupported consensus rule: {requested}")


def repair_region(
    extent_mask: np.ndarray,
    core_mask: np.ndarray,
    nucleus_id: int,
    obj_slice: tuple[slice, slice],
    mode: str,
) -> np.ndarray:
    extent = extent_mask[obj_slice] == nucleus_id
    if mode == "extent":
        return extent
    if mode == "core_dilate4":
        core = core_mask[obj_slice] == nucleus_id
        return ndimage.binary_dilation(core, iterations=4) & extent
    raise ValueError(f"Unsupported repair region: {mode}")


def plan_local_repair(
    output: np.ndarray,
    obj_slice: tuple[slice, slice],
    region: np.ndarray,
    target: int,
    fill_background: bool,
) -> LocalRepairPlan:
    if target <= 0:
        return LocalRepairPlan(np.zeros(region.shape, dtype=bool), 0, 0, 0, ())
    view = output[obj_slice]
    selected = region if fill_background else region & (view > 0)
    changed = selected & (view != target)
    filled = changed & (view == 0)
    relabeled = changed & (view > 0)
    source_values = view[relabeled]
    if source_values.size:
        source_labels, source_counts = np.unique(source_values, return_counts=True)
        source_label_counts = tuple(
            (int(label), int(count)) for label, count in zip(source_labels, source_counts)
        )
    else:
        source_label_counts = ()
    return LocalRepairPlan(
        selected=selected.copy(),
        changed_pixels=int(np.count_nonzero(changed)),
        filled_pixels=int(np.count_nonzero(filled)),
        relabeled_pixels=int(np.count_nonzero(relabeled)),
        source_label_counts=source_label_counts,
    )


def repair_plan_allowed(
    plan: LocalRepairPlan,
    label_areas: np.ndarray,
    config: RepairConfig,
) -> bool:
    if (
        config.max_changed_pixels_per_profile > 0
        and plan.changed_pixels > config.max_changed_pixels_per_profile
    ):
        return False
    if config.min_remaining_label_area > 0:
        for source_label, removed_pixels in plan.source_label_counts:
            if int(label_areas[source_label]) - removed_pixels < config.min_remaining_label_area:
                return False
    return True


def apply_local_repair_plan(
    output: np.ndarray,
    obj_slice: tuple[slice, slice],
    target: int,
    plan: LocalRepairPlan,
    label_areas: np.ndarray,
) -> None:
    if target <= 0 or plan.changed_pixels <= 0:
        return
    view = output[obj_slice]
    view[plan.selected] = target
    for source_label, removed_pixels in plan.source_label_counts:
        label_areas[source_label] -= removed_pixels
    label_areas[target] += plan.changed_pixels


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    extent_paths = index_masks(args.run_root / "Nuclei" / "segmentations", "*_cp_masks.tif")
    core_paths = index_masks(args.run_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif")
    combined_paths = index_masks(args.run_root / "Combined" / "segmentations", "*_cp_masks.tif")
    brightfield_paths = index_masks(args.run_root / "Brightfield" / "segmentations", "*_cp_masks.tif")
    keys = sorted(set(extent_paths) & set(core_paths) & set(combined_paths) & set(brightfield_paths))
    if args.keys:
        requested = set(args.keys)
        keys = [key for key in keys if key in requested]
        missing = sorted(requested - set(keys))
        if missing:
            raise SystemExit(f"Requested keys are missing one or more masks: {missing}")
    if not keys:
        raise SystemExit("No common fields found")

    candidate_roots: dict[str, Path] = {}
    for config in REPAIR_CONFIGS:
        candidate_root = args.out_root / "candidates" / config.tag
        candidate_roots[config.tag] = candidate_root
        link_directory(args.run_root / "Nuclei", candidate_root / "Nuclei")
        if (args.run_root / "qc").is_dir():
            link_directory(args.run_root / "qc", candidate_root / "qc")
        if (args.run_root / "classification_fusion").is_dir():
            link_directory(args.run_root / "classification_fusion", candidate_root / "classification_fusion")

    registration_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    summary_accumulator: dict[str, dict[str, int]] = {
        config.tag: defaultdict(int) for config in REPAIR_CONFIGS
    }
    total_core_objects = 0

    for field_index, key in enumerate(keys, start=1):
        print(f"[{field_index}/{len(keys)}] {key}", flush=True)
        extent = read_mask(extent_paths[key])
        core = read_mask(core_paths[key])
        combined = read_mask(combined_paths[key])
        brightfield = read_mask(brightfield_paths[key])
        if not (extent.shape == core.shape == combined.shape == brightfield.shape):
            raise ValueError(f"Shape mismatch for {key}")
        invalid_core = (core > 0) & (core != extent)
        if np.any(invalid_core):
            raise ValueError(f"Core IDs are not preserved inside extent for {key}")

        core_ids, _centers = instance_centroids(core)
        total_core_objects += len(core_ids)
        for profile, mask in (("Combined", combined), ("Brightfield", brightfield)):
            diagnostics = diagnose_registration(mask, core, args.max_shift)
            registration_rows.append(
                {
                    "key": key,
                    "profile": profile,
                    "focus_key": key in FOCUS_KEYS,
                    **diagnostics,
                }
            )

        combined_to_bf, bf_to_combined = best_overlap_maps(combined, brightfield)
        combined_nearest, combined_distance = nearest_label_context(combined)
        bf_nearest, bf_distance = nearest_label_context(brightfield)
        objects = ndimage.find_objects(extent)
        supports: list[dict[str, Any]] = []
        for nucleus_id, obj_slice in enumerate(objects, start=1):
            if obj_slice is None or not np.any(core[obj_slice] == nucleus_id):
                continue
            combined_support = core_support(
                combined, core, nucleus_id, obj_slice, combined_nearest, combined_distance
            )
            bf_support = core_support(
                brightfield, core, nucleus_id, obj_slice, bf_nearest, bf_distance
            )
            combined_target = target_label(combined_support)
            bf_target = target_label(bf_support)
            consensus = pair_consensus(
                combined_target, bf_target, combined_to_bf, bf_to_combined
            )
            supports.append(
                {
                    "nucleus_id": nucleus_id,
                    "obj_slice": obj_slice,
                    "combined": combined_support,
                    "brightfield": bf_support,
                    "combined_target": combined_target,
                    "brightfield_target": bf_target,
                    "consensus": consensus,
                }
            )

        for config in REPAIR_CONFIGS:
            combined_out = combined.copy()
            brightfield_out = brightfield.copy()
            combined_label_areas = np.bincount(combined_out.ravel()).astype(np.int64)
            brightfield_label_areas = np.bincount(brightfield_out.ravel()).astype(np.int64)
            candidate_summary = summary_accumulator[config.tag]
            candidate_summary["n_fields"] += 1
            for support_row in supports:
                combined_support = support_row["combined"]
                bf_support = support_row["brightfield"]
                combined_direct = (
                    combined_support.primary_fraction >= config.min_core_primary_fraction
                )
                bf_direct = bf_support.primary_fraction >= config.min_core_primary_fraction
                if config.require_both_direct:
                    support_ok = combined_direct and bf_direct
                else:
                    support_ok = combined_direct or bf_direct
                if not support_ok or not consensus_allowed(support_row["consensus"], config.consensus):
                    continue
                if not combined_direct and combined_support.nearest_distance > config.max_nearest_distance:
                    continue
                if not bf_direct and bf_support.nearest_distance > config.max_nearest_distance:
                    continue

                nucleus_id = int(support_row["nucleus_id"])
                obj_slice = support_row["obj_slice"]
                region = repair_region(extent, core, nucleus_id, obj_slice, config.region)
                combined_plan = plan_local_repair(
                    combined_out,
                    obj_slice,
                    region,
                    int(support_row["combined_target"]),
                    config.fill_background,
                )
                bf_plan = plan_local_repair(
                    brightfield_out,
                    obj_slice,
                    region,
                    int(support_row["brightfield_target"]),
                    config.fill_background,
                )
                if not repair_plan_allowed(combined_plan, combined_label_areas, config):
                    continue
                if not repair_plan_allowed(bf_plan, brightfield_label_areas, config):
                    continue
                combined_changed = combined_plan.changed_pixels
                combined_filled = combined_plan.filled_pixels
                combined_relabeled = combined_plan.relabeled_pixels
                bf_changed = bf_plan.changed_pixels
                bf_filled = bf_plan.filled_pixels
                bf_relabeled = bf_plan.relabeled_pixels
                if combined_changed + bf_changed <= 0:
                    continue
                apply_local_repair_plan(
                    combined_out,
                    obj_slice,
                    int(support_row["combined_target"]),
                    combined_plan,
                    combined_label_areas,
                )
                apply_local_repair_plan(
                    brightfield_out,
                    obj_slice,
                    int(support_row["brightfield_target"]),
                    bf_plan,
                    brightfield_label_areas,
                )
                candidate_summary["n_repaired_nuclei"] += 1
                candidate_summary["combined_changed_pixels"] += combined_changed
                candidate_summary["combined_filled_pixels"] += combined_filled
                candidate_summary["combined_relabeled_pixels"] += combined_relabeled
                candidate_summary["brightfield_changed_pixels"] += bf_changed
                candidate_summary["brightfield_filled_pixels"] += bf_filled
                candidate_summary["brightfield_relabeled_pixels"] += bf_relabeled
                repair_rows.append(
                    {
                        "candidate": config.tag,
                        "key": key,
                        "nucleus_id": nucleus_id,
                        "pair_consensus": support_row["consensus"],
                        "combined_target": support_row["combined_target"],
                        "combined_core_primary_fraction": combined_support.primary_fraction,
                        "combined_nearest_distance": combined_support.nearest_distance,
                        "combined_changed_pixels": combined_changed,
                        "combined_filled_pixels": combined_filled,
                        "combined_relabeled_pixels": combined_relabeled,
                        "brightfield_target": support_row["brightfield_target"],
                        "brightfield_core_primary_fraction": bf_support.primary_fraction,
                        "brightfield_nearest_distance": bf_support.nearest_distance,
                        "brightfield_changed_pixels": bf_changed,
                        "brightfield_filled_pixels": bf_filled,
                        "brightfield_relabeled_pixels": bf_relabeled,
                    }
                )

            candidate_root = candidate_roots[config.tag]
            write_mask(
                candidate_root / "Combined" / "segmentations" / combined_paths[key].name,
                combined_out,
            )
            write_mask(
                candidate_root / "Brightfield" / "segmentations" / brightfield_paths[key].name,
                brightfield_out,
            )

    write_rows(args.out_root / "registration_diagnostics.csv", registration_rows)
    write_rows(args.out_root / "repair_events.csv", repair_rows)
    write_json(args.out_root / "repair_configs.json", [asdict(config) for config in REPAIR_CONFIGS])

    image_pixels = 0
    if keys:
        image_pixels = int(read_mask(extent_paths[keys[0]]).size) * len(keys)
    repair_summaries: list[dict[str, Any]] = []
    for config in REPAIR_CONFIGS:
        raw_summary = dict(summary_accumulator[config.tag])
        summary = {
            "candidate": config.tag,
            **asdict(config),
            **raw_summary,
            "n_core_objects": total_core_objects,
            "repaired_nucleus_fraction": (
                raw_summary.get("n_repaired_nuclei", 0) / total_core_objects if total_core_objects else 0.0
            ),
            "combined_changed_field_fraction": (
                raw_summary.get("combined_changed_pixels", 0) / image_pixels if image_pixels else 0.0
            ),
            "brightfield_changed_field_fraction": (
                raw_summary.get("brightfield_changed_pixels", 0) / image_pixels if image_pixels else 0.0
            ),
        }
        repair_summaries.append(summary)
        write_json(candidate_roots[config.tag] / "repair_summary.json", summary)
    write_rows(args.out_root / "repair_summary.csv", repair_summaries)

    lines = [
        "# Registration diagnostic and nucleus-aware cell-mask candidates",
        "",
        f"- run root: `{args.run_root}`",
        f"- fields: {len(keys)}",
        f"- nucleus cores: {total_core_objects}",
        f"- integer shift search: ±{args.max_shift} px",
        "",
        "## Registration diagnostics",
        "",
        "| profile | stratum | median best dy | median best dx | median score gain | median inside-rate gain | nonzero best shift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile in ("Combined", "Brightfield"):
        for stratum_name, predicate in (
            ("all", lambda row: True),
            ("focus", lambda row: bool(row["focus_key"])),
            ("nonfocus", lambda row: not bool(row["focus_key"])),
        ):
            selected = [
                row for row in registration_rows if row["profile"] == profile and predicate(row)
            ]
            if not selected:
                continue
            lines.append(
                f"| {profile} | {stratum_name} | {np.median([row['best_dy'] for row in selected]):.1f} | "
                f"{np.median([row['best_dx'] for row in selected]):.1f} | "
                f"{np.median([row['score_gain'] for row in selected]):.4f} | "
                f"{np.median([row['inside_rate_gain'] for row in selected]):.4f} | "
                f"{np.mean([(row['best_dy'] != 0 or row['best_dx'] != 0) for row in selected]):.1%} |"
            )
    lines.extend(
        [
            "",
            "## Repair candidates",
            "",
            "| candidate | repaired nuclei | repaired fraction | Combined changed field fraction | BF changed field fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in repair_summaries:
        lines.append(
            f"| {summary['candidate']} | {summary.get('n_repaired_nuclei', 0)} | "
            f"{float(summary['repaired_nucleus_fraction']):.2%} | "
            f"{float(summary['combined_changed_field_fraction']):.3%} | "
            f"{float(summary['brightfield_changed_field_fraction']):.3%} |"
        )
    (args.out_root / "diagnostic_report.md").write_text("\n".join(lines) + "\n")
    print(f"registration_diagnostics={args.out_root / 'registration_diagnostics.csv'}", flush=True)
    print(f"repair_events={args.out_root / 'repair_events.csv'}", flush=True)
    print(f"repair_summary={args.out_root / 'repair_summary.csv'}", flush=True)
    print(f"report={args.out_root / 'diagnostic_report.md'}", flush=True)


if __name__ == "__main__":
    main()
