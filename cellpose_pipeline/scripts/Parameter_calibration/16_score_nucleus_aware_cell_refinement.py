#!/usr/bin/env python3
"""Score nucleus-aware cell-mask refinements with independent guardrails.

The nucleus-to-cell alignment statistics are useful but partially circular because
the candidate masks were refined using nucleus cores.  This scorer therefore also
audits raw-image boundary support, per-label area changes, label connectivity, and
BF/Combined cell-mask agreement before recommending a conservative candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
PROFILES = ("Combined", "Brightfield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nucleus-aware cell-mask candidates with alignment and image-based guardrails."
    )
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-changed-fraction", type=float, default=0.010)
    parser.add_argument("--max-repaired-nucleus-fraction", type=float, default=0.45)
    parser.add_argument("--max-multinucleation-rate-delta", type=float, default=0.010)
    parser.add_argument("--max-concordance-loss", type=float, default=0.020)
    parser.add_argument("--min-bf-local-edge-ratio", type=float, default=0.90)
    parser.add_argument("--max-fragmentation-increase-rate", type=float, default=0.010)
    parser.add_argument("--max-area-change-q95", type=float, default=0.50)
    parser.add_argument("--near-best-benefit-fraction", type=float, default=0.90)
    return parser.parse_args()


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def index_masks(run_root: Path, profile: str) -> dict[str, Path]:
    mask_dir = run_root / profile / "segmentations"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    result: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*_cp_masks.tif")):
        key = key_from_path(path)
        if key in result:
            raise ValueError(f"Duplicate {profile} mask for {key}")
        result[key] = path
    return result


def index_raw(input_root: Path, profile: str) -> dict[str, Path]:
    raw_dir = input_root / profile
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw-image directory: {raw_dir}")
    suffix_rank = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}
    result: dict[str, Path] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffix_rank:
            continue
        try:
            key = key_from_path(path)
        except ValueError:
            continue
        previous = result.get(key)
        if previous is None or suffix_rank[path.suffix.lower()] < suffix_rank[previous.suffix.lower()]:
            result[key] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected 2D integer mask at {path}; got {labels.shape} {labels.dtype}")
    return labels.astype(np.int32, copy=False)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.ndim == 3 and image.shape[-1] >= 3:
        arr = image[..., :3].astype(np.float64, copy=False)
    elif image.ndim == 3 and image.shape[0] >= 3 and image.shape[0] <= 4:
        arr = np.moveaxis(image[:3], 0, -1).astype(np.float64, copy=False)
    else:
        raise ValueError(f"Cannot convert image shape {image.shape} to grayscale")
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def normalized_gradient(image: np.ndarray) -> np.ndarray:
    gray = as_gray(image)
    finite = gray[np.isfinite(gray)]
    if not finite.size:
        return np.zeros(gray.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    scaled = np.clip((gray - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    smooth = ndimage.gaussian_filter(scaled, sigma=1.0)
    dy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
    dx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
    return np.hypot(dy, dx).astype(np.float32, copy=False)


def label_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def union_slice(a: tuple[slice, slice] | None, b: tuple[slice, slice] | None) -> tuple[slice, slice] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        slice(min(a[0].start or 0, b[0].start or 0), max(a[0].stop or 0, b[0].stop or 0)),
        slice(min(a[1].start or 0, b[1].start or 0), max(a[1].stop or 0, b[1].stop or 0)),
    )


def component_count(labels: np.ndarray, label_id: int, obj_slice: tuple[slice, slice] | None) -> int:
    if obj_slice is None:
        return 0
    _components, count = ndimage.label(labels[obj_slice] == label_id, structure=np.ones((3, 3), dtype=bool))
    return int(count)


def new_profile_accumulator() -> dict[str, Any]:
    return {
        "total_pixels": 0,
        "changed_pixels": 0,
        "baseline_foreground_pixels": 0,
        "candidate_foreground_pixels": 0,
        "local_baseline_edge_sum": 0.0,
        "local_baseline_edge_count": 0,
        "local_candidate_edge_sum": 0.0,
        "local_candidate_edge_count": 0,
        "global_baseline_edge_sum": 0.0,
        "global_baseline_edge_count": 0,
        "global_candidate_edge_sum": 0.0,
        "global_candidate_edge_count": 0,
        "field_local_edge_ratios": [],
        "area_relative_changes": [],
        "affected_labels": 0,
        "lost_labels": 0,
        "gained_labels": 0,
        "labels_with_more_components": 0,
        "extra_components": 0,
    }


def audit_mask_pair(
    baseline: np.ndarray,
    candidate: np.ndarray,
    gradient: np.ndarray,
    accumulator: dict[str, Any],
) -> None:
    if baseline.shape != candidate.shape or baseline.shape != gradient.shape:
        raise ValueError("Mask/raw shape mismatch during mask audit")
    changed = baseline != candidate
    accumulator["total_pixels"] += int(baseline.size)
    accumulator["changed_pixels"] += int(np.count_nonzero(changed))
    accumulator["baseline_foreground_pixels"] += int(np.count_nonzero(baseline))
    accumulator["candidate_foreground_pixels"] += int(np.count_nonzero(candidate))

    baseline_boundary = label_boundary(baseline)
    candidate_boundary = label_boundary(candidate)
    accumulator["global_baseline_edge_sum"] += float(gradient[baseline_boundary].sum())
    accumulator["global_baseline_edge_count"] += int(np.count_nonzero(baseline_boundary))
    accumulator["global_candidate_edge_sum"] += float(gradient[candidate_boundary].sum())
    accumulator["global_candidate_edge_count"] += int(np.count_nonzero(candidate_boundary))
    if np.any(changed):
        local = ndimage.binary_dilation(changed, iterations=3)
        baseline_local = baseline_boundary & local
        candidate_local = candidate_boundary & local
        baseline_sum = float(gradient[baseline_local].sum())
        candidate_sum = float(gradient[candidate_local].sum())
        baseline_count = int(np.count_nonzero(baseline_local))
        candidate_count = int(np.count_nonzero(candidate_local))
        accumulator["local_baseline_edge_sum"] += baseline_sum
        accumulator["local_baseline_edge_count"] += baseline_count
        accumulator["local_candidate_edge_sum"] += candidate_sum
        accumulator["local_candidate_edge_count"] += candidate_count
        baseline_mean = baseline_sum / baseline_count if baseline_count else 0.0
        candidate_mean = candidate_sum / candidate_count if candidate_count else 0.0
        if baseline_mean > 0:
            accumulator["field_local_edge_ratios"].append(candidate_mean / baseline_mean)

    maximum = max(int(baseline.max()), int(candidate.max()))
    baseline_areas = np.bincount(baseline.ravel(), minlength=maximum + 1)
    candidate_areas = np.bincount(candidate.ravel(), minlength=maximum + 1)
    affected = np.unique(np.concatenate((baseline[changed], candidate[changed])))
    affected = affected[affected > 0]
    baseline_slices = ndimage.find_objects(baseline, max_label=maximum)
    candidate_slices = ndimage.find_objects(candidate, max_label=maximum)
    for label_value in affected:
        label_id = int(label_value)
        baseline_area = int(baseline_areas[label_id])
        candidate_area = int(candidate_areas[label_id])
        accumulator["affected_labels"] += 1
        if baseline_area == 0:
            accumulator["gained_labels"] += 1
            continue
        if candidate_area == 0:
            accumulator["lost_labels"] += 1
            accumulator["area_relative_changes"].append(1.0)
            continue
        accumulator["area_relative_changes"].append(abs(candidate_area - baseline_area) / baseline_area)
        obj_slice = union_slice(baseline_slices[label_id - 1], candidate_slices[label_id - 1])
        before_components = component_count(baseline, label_id, obj_slice)
        after_components = component_count(candidate, label_id, obj_slice)
        if after_components > before_components:
            accumulator["labels_with_more_components"] += 1
            accumulator["extra_components"] += after_components - before_components


def finalize_profile_audit(accumulator: dict[str, Any]) -> dict[str, Any]:
    def mean(sum_name: str, count_name: str) -> float:
        count = int(accumulator[count_name])
        return float(accumulator[sum_name]) / count if count else 0.0

    local_baseline = mean("local_baseline_edge_sum", "local_baseline_edge_count")
    local_candidate = mean("local_candidate_edge_sum", "local_candidate_edge_count")
    global_baseline = mean("global_baseline_edge_sum", "global_baseline_edge_count")
    global_candidate = mean("global_candidate_edge_sum", "global_candidate_edge_count")
    area_changes = np.asarray(accumulator["area_relative_changes"], dtype=float)
    field_ratios = np.asarray(accumulator["field_local_edge_ratios"], dtype=float)
    affected = int(accumulator["affected_labels"])
    return {
        "changed_pixel_fraction": accumulator["changed_pixels"] / accumulator["total_pixels"],
        "foreground_area_delta_fraction": (
            (accumulator["candidate_foreground_pixels"] - accumulator["baseline_foreground_pixels"])
            / accumulator["baseline_foreground_pixels"]
            if accumulator["baseline_foreground_pixels"]
            else 0.0
        ),
        "local_boundary_edge_ratio": local_candidate / local_baseline if local_baseline else 0.0,
        "median_field_local_boundary_edge_ratio": float(np.median(field_ratios)) if field_ratios.size else 0.0,
        "q10_field_local_boundary_edge_ratio": float(np.quantile(field_ratios, 0.10)) if field_ratios.size else 0.0,
        "global_boundary_edge_ratio": global_candidate / global_baseline if global_baseline else 0.0,
        "affected_labels": affected,
        "median_affected_label_area_change": float(np.median(area_changes)) if area_changes.size else 0.0,
        "q95_affected_label_area_change": float(np.quantile(area_changes, 0.95)) if area_changes.size else 0.0,
        "lost_labels": int(accumulator["lost_labels"]),
        "gained_labels": int(accumulator["gained_labels"]),
        "labels_with_more_components": int(accumulator["labels_with_more_components"]),
        "fragmentation_increase_rate": (
            accumulator["labels_with_more_components"] / affected if affected else 0.0
        ),
        "extra_components": int(accumulator["extra_components"]),
    }


def json_load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_rate(value: float, total: int) -> float:
    return float(value) / int(total) if total else 0.0


def summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    alignment = summary["alignment"]
    total = int(summary["nuclei"]["total"])
    status_counts = alignment["status_counts"]
    crossing_union = sum(int(value) for name, value in status_counts.items() if name.startswith("crosses_"))
    combined_crossing = int(alignment["combined_match_categories"].get("crosses_cells", 0))
    bf_crossing = int(alignment["bf_match_categories"].get("crosses_cells", 0))
    combined = summary["cells"]["Combined"]
    brightfield = summary["cells"]["Brightfield"]
    concordance = summary["cross_method_cell_concordance"]
    high = summary["density_strata"]["high_density"]
    non_high = summary["density_strata"]["non_high_density"]
    return {
        "nuclei": total,
        "strict_rate": float(alignment["strict_consensus_rate"]),
        "actionable_rate": float(alignment["actionable_mismatch_rate"]),
        "crossing_union_count": crossing_union,
        "crossing_union_rate": safe_rate(crossing_union, total),
        "combined_crossing_event_rate": safe_rate(combined_crossing, total),
        "bf_crossing_event_rate": safe_rate(bf_crossing, total),
        "high_density_actionable_rate": float(high["actionable_mismatch_rate"]),
        "non_high_density_actionable_rate": float(non_high["actionable_mismatch_rate"]),
        "combined_all_cells": int(combined["all_cells"]),
        "bf_all_cells": int(brightfield["all_cells"]),
        "combined_countable_cells": int(combined["countable_cells"]),
        "bf_countable_cells": int(brightfield["countable_cells"]),
        "combined_multinucleated_rate": float(combined["confirmed_multinucleated_rate"]),
        "bf_multinucleated_rate": float(brightfield["confirmed_multinucleated_rate"]),
        "combined_extent_nc_ratio": float(combined["median_nucleus_to_cytoplasm_ratio"]),
        "bf_extent_nc_ratio": float(brightfield["median_nucleus_to_cytoplasm_ratio"]),
        "combined_core_nc_ratio": float(combined["median_core_nucleus_to_cytoplasm_ratio"]),
        "bf_core_nc_ratio": float(brightfield["median_core_nucleus_to_cytoplasm_ratio"]),
        "reciprocal_countable_pairs": int(concordance["reciprocal_countable_cell_pairs"]),
        "exact_status_concordance_rate": float(concordance["exact_status_concordance_rate"]),
        "multinucleation_jaccard": float(concordance["confirmed_multinucleation_jaccard"]),
        "validation_ok": all(bool(value) for value in summary["validation"].values()),
    }


def field_comparisons(
    baseline_path: Path,
    candidate_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    baseline = {row["key"]: row for row in csv_rows(baseline_path)}
    result: list[dict[str, Any]] = []
    for candidate, path in sorted(candidate_paths.items()):
        current = {row["key"]: row for row in csv_rows(path)}
        if set(current) != set(baseline):
            raise RuntimeError(f"Field-summary key mismatch for {candidate}")
        for key in sorted(baseline):
            before = baseline[key]
            after = current[key]
            before_actionable = float(before["actionable_mismatch_rate"])
            after_actionable = float(after["actionable_mismatch_rate"])
            before_crossing = int(before["n_combined_cross_boundary"]) + int(before["n_bf_cross_boundary"])
            after_crossing = int(after["n_combined_cross_boundary"]) + int(after["n_bf_cross_boundary"])
            result.append(
                {
                    "candidate": candidate,
                    "key": key,
                    "high_density": after["high_density"],
                    "n_nuclei": int(after["n_nuclei"]),
                    "baseline_actionable_rate": before_actionable,
                    "candidate_actionable_rate": after_actionable,
                    "actionable_rate_reduction_pp": 100.0 * (before_actionable - after_actionable),
                    "baseline_crossing_events": before_crossing,
                    "candidate_crossing_events": after_crossing,
                    "crossing_events_reduced": before_crossing - after_crossing,
                }
            )
    return result


def main() -> int:
    args = parse_args()
    args.baseline_run_root = args.baseline_run_root.resolve()
    args.baseline_summary = args.baseline_summary.resolve()
    args.candidates_root = args.candidates_root.resolve()
    args.input_root = args.input_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = json_load(args.baseline_summary)
    baseline_metrics = summary_metrics(baseline_summary)
    candidate_dirs = {
        path.name: path
        for path in sorted(args.candidates_root.iterdir())
        if path.is_dir() and (path / "alignment" / "summary.json").is_file()
    }
    if not candidate_dirs:
        raise SystemExit(f"No validated candidate directories found under {args.candidates_root}")

    baseline_masks = {profile: index_masks(args.baseline_run_root, profile) for profile in PROFILES}
    raw_paths = {profile: index_raw(args.input_root, profile) for profile in PROFILES}
    candidate_masks = {
        candidate: {profile: index_masks(root, profile) for profile in PROFILES}
        for candidate, root in candidate_dirs.items()
    }
    keys = sorted(set.intersection(*(set(paths) for paths in baseline_masks.values())))
    for profile in PROFILES:
        if set(keys) - set(raw_paths[profile]):
            raise RuntimeError(f"Missing {profile} raw images: {sorted(set(keys) - set(raw_paths[profile]))}")
    for candidate, by_profile in candidate_masks.items():
        for profile in PROFILES:
            if set(by_profile[profile]) != set(keys):
                raise RuntimeError(f"Mask-key mismatch for {candidate}/{profile}")

    audit_accumulators = {
        candidate: {profile: new_profile_accumulator() for profile in PROFILES}
        for candidate in candidate_dirs
    }
    cross_agreement: dict[str, dict[str, int]] = {
        candidate: defaultdict(int) for candidate in candidate_dirs
    }
    baseline_cross = defaultdict(int)
    for field_index, key in enumerate(keys, start=1):
        print(f"[audit {field_index}/{len(keys)}] {key}", flush=True)
        baseline_field = {profile: read_mask(baseline_masks[profile][key]) for profile in PROFILES}
        gradients = {
            profile: normalized_gradient(read_image(raw_paths[profile][key])) for profile in PROFILES
        }
        baseline_overlap = (baseline_field["Combined"] > 0) & (baseline_field["Brightfield"] > 0)
        baseline_union = (baseline_field["Combined"] > 0) | (baseline_field["Brightfield"] > 0)
        baseline_cross["intersection"] += int(np.count_nonzero(baseline_overlap))
        baseline_cross["union"] += int(np.count_nonzero(baseline_union))
        for candidate in candidate_dirs:
            current: dict[str, np.ndarray] = {}
            for profile in PROFILES:
                current[profile] = read_mask(candidate_masks[candidate][profile][key])
                audit_mask_pair(
                    baseline_field[profile],
                    current[profile],
                    gradients[profile],
                    audit_accumulators[candidate][profile],
                )
            overlap = (current["Combined"] > 0) & (current["Brightfield"] > 0)
            union = (current["Combined"] > 0) | (current["Brightfield"] > 0)
            cross_agreement[candidate]["intersection"] += int(np.count_nonzero(overlap))
            cross_agreement[candidate]["union"] += int(np.count_nonzero(union))

    baseline_foreground_jaccard = (
        baseline_cross["intersection"] / baseline_cross["union"] if baseline_cross["union"] else 0.0
    )
    candidate_summaries = {
        candidate: json_load(root / "alignment" / "summary.json")
        for candidate, root in candidate_dirs.items()
    }
    rows: list[dict[str, Any]] = []
    for candidate, root in candidate_dirs.items():
        metrics = summary_metrics(candidate_summaries[candidate])
        repair = json_load(root / "repair_summary.json")
        audits = {
            profile: finalize_profile_audit(audit_accumulators[candidate][profile])
            for profile in PROFILES
        }
        candidate_foreground_jaccard = (
            cross_agreement[candidate]["intersection"] / cross_agreement[candidate]["union"]
            if cross_agreement[candidate]["union"]
            else 0.0
        )
        average_changed = 0.5 * (
            audits["Combined"]["changed_pixel_fraction"]
            + audits["Brightfield"]["changed_pixel_fraction"]
        )
        action_reduction = baseline_metrics["actionable_rate"] - metrics["actionable_rate"]
        crossing_reduction = baseline_metrics["crossing_union_rate"] - metrics["crossing_union_rate"]
        strict_gain = metrics["strict_rate"] - baseline_metrics["strict_rate"]
        row: dict[str, Any] = {
            "candidate": candidate,
            "actionable_rate": metrics["actionable_rate"],
            "actionable_rate_reduction_pp": 100.0 * action_reduction,
            "strict_rate": metrics["strict_rate"],
            "strict_rate_gain_pp": 100.0 * strict_gain,
            "crossing_union_rate": metrics["crossing_union_rate"],
            "crossing_union_rate_reduction_pp": 100.0 * crossing_reduction,
            "high_density_actionable_rate": metrics["high_density_actionable_rate"],
            "high_density_actionable_reduction_pp": 100.0 * (
                baseline_metrics["high_density_actionable_rate"] - metrics["high_density_actionable_rate"]
            ),
            "non_high_density_actionable_rate": metrics["non_high_density_actionable_rate"],
            "non_high_density_actionable_reduction_pp": 100.0 * (
                baseline_metrics["non_high_density_actionable_rate"] - metrics["non_high_density_actionable_rate"]
            ),
            "repaired_nucleus_fraction": float(repair["repaired_nucleus_fraction"]),
            "average_changed_pixel_fraction": average_changed,
            "combined_changed_pixel_fraction": audits["Combined"]["changed_pixel_fraction"],
            "bf_changed_pixel_fraction": audits["Brightfield"]["changed_pixel_fraction"],
            "combined_local_edge_ratio": audits["Combined"]["local_boundary_edge_ratio"],
            "bf_local_edge_ratio": audits["Brightfield"]["local_boundary_edge_ratio"],
            "combined_median_field_local_edge_ratio": audits["Combined"]["median_field_local_boundary_edge_ratio"],
            "bf_median_field_local_edge_ratio": audits["Brightfield"]["median_field_local_boundary_edge_ratio"],
            "combined_global_edge_ratio": audits["Combined"]["global_boundary_edge_ratio"],
            "bf_global_edge_ratio": audits["Brightfield"]["global_boundary_edge_ratio"],
            "combined_q95_affected_label_area_change": audits["Combined"]["q95_affected_label_area_change"],
            "bf_q95_affected_label_area_change": audits["Brightfield"]["q95_affected_label_area_change"],
            "combined_fragmentation_increase_rate": audits["Combined"]["fragmentation_increase_rate"],
            "bf_fragmentation_increase_rate": audits["Brightfield"]["fragmentation_increase_rate"],
            "combined_lost_labels": audits["Combined"]["lost_labels"],
            "bf_lost_labels": audits["Brightfield"]["lost_labels"],
            "combined_gained_labels": audits["Combined"]["gained_labels"],
            "bf_gained_labels": audits["Brightfield"]["gained_labels"],
            "foreground_jaccard": candidate_foreground_jaccard,
            "foreground_jaccard_gain_pp": 100.0 * (
                candidate_foreground_jaccard - baseline_foreground_jaccard
            ),
            "combined_multinucleated_rate": metrics["combined_multinucleated_rate"],
            "combined_multinucleated_rate_delta_pp": 100.0 * (
                metrics["combined_multinucleated_rate"]
                - baseline_metrics["combined_multinucleated_rate"]
            ),
            "bf_multinucleated_rate": metrics["bf_multinucleated_rate"],
            "bf_multinucleated_rate_delta_pp": 100.0 * (
                metrics["bf_multinucleated_rate"] - baseline_metrics["bf_multinucleated_rate"]
            ),
            "exact_status_concordance_rate": metrics["exact_status_concordance_rate"],
            "exact_status_concordance_delta_pp": 100.0 * (
                metrics["exact_status_concordance_rate"]
                - baseline_metrics["exact_status_concordance_rate"]
            ),
            "multinucleation_jaccard": metrics["multinucleation_jaccard"],
            "multinucleation_jaccard_delta_pp": 100.0 * (
                metrics["multinucleation_jaccard"] - baseline_metrics["multinucleation_jaccard"]
            ),
            "combined_extent_nc_ratio": metrics["combined_extent_nc_ratio"],
            "bf_extent_nc_ratio": metrics["bf_extent_nc_ratio"],
            "combined_core_nc_ratio": metrics["combined_core_nc_ratio"],
            "bf_core_nc_ratio": metrics["bf_core_nc_ratio"],
            "combined_all_cell_delta": metrics["combined_all_cells"] - baseline_metrics["combined_all_cells"],
            "bf_all_cell_delta": metrics["bf_all_cells"] - baseline_metrics["bf_all_cells"],
            "validation_ok": metrics["validation_ok"],
        }
        reasons: list[str] = []
        if not metrics["validation_ok"]:
            reasons.append("analysis validation failed")
        if action_reduction <= 0:
            reasons.append("no actionable-mismatch improvement")
        if average_changed > args.max_changed_fraction:
            reasons.append("changed-pixel fraction above limit")
        if float(repair["repaired_nucleus_fraction"]) > args.max_repaired_nucleus_fraction:
            reasons.append("repaired-nucleus fraction above limit")
        if any(audits[profile]["lost_labels"] or audits[profile]["gained_labels"] for profile in PROFILES):
            reasons.append("cell labels gained or lost")
        if max(audits[p]["fragmentation_increase_rate"] for p in PROFILES) > args.max_fragmentation_increase_rate:
            reasons.append("label fragmentation above limit")
        if max(audits[p]["q95_affected_label_area_change"] for p in PROFILES) > args.max_area_change_q95:
            reasons.append("affected-cell area change above limit")
        if abs(metrics["combined_multinucleated_rate"] - baseline_metrics["combined_multinucleated_rate"]) > args.max_multinucleation_rate_delta:
            reasons.append("Combined multinucleation shift above limit")
        if abs(metrics["bf_multinucleated_rate"] - baseline_metrics["bf_multinucleated_rate"]) > args.max_multinucleation_rate_delta:
            reasons.append("BF multinucleation shift above limit")
        if metrics["exact_status_concordance_rate"] < baseline_metrics["exact_status_concordance_rate"] - args.max_concordance_loss:
            reasons.append("exact BF/Combined status concordance loss")
        if metrics["multinucleation_jaccard"] < baseline_metrics["multinucleation_jaccard"] - args.max_concordance_loss:
            reasons.append("multinucleation Jaccard loss")
        if audits["Brightfield"]["local_boundary_edge_ratio"] < args.min_bf_local_edge_ratio:
            reasons.append("BF raw-edge support below limit")
        row["eligible"] = not reasons
        row["guardrail_failures"] = "; ".join(reasons)
        row["benefit_per_1pct_changed"] = (
            (100.0 * action_reduction) / (100.0 * average_changed) if average_changed else 0.0
        )
        rows.append(row)

    eligible = [row for row in rows if row["eligible"]]
    selected: dict[str, Any] | None = None
    if eligible:
        best_benefit = max(float(row["actionable_rate_reduction_pp"]) for row in eligible)
        near_best = [
            row
            for row in eligible
            if float(row["actionable_rate_reduction_pp"])
            >= args.near_best_benefit_fraction * best_benefit
        ]
        selected = min(
            near_best,
            key=lambda row: (
                float(row["average_changed_pixel_fraction"]),
                -float(row["bf_local_edge_ratio"]),
                str(row["candidate"]),
            ),
        )
    for row in rows:
        row["recommended"] = selected is not None and row["candidate"] == selected["candidate"]

    rows.sort(key=lambda row: (not bool(row["eligible"]), -float(row["actionable_rate_reduction_pp"])))
    write_rows(args.out_dir / "candidate_comparison.csv", rows)
    field_rows = field_comparisons(
        args.baseline_summary.parent / "field_summary.csv",
        {
            candidate: root / "alignment" / "field_summary.csv"
            for candidate, root in candidate_dirs.items()
        },
    )
    write_rows(args.out_dir / "field_comparison.csv", field_rows)

    decision = {
        "selected_candidate": selected["candidate"] if selected else None,
        "selection_rule": (
            "Among candidates passing every guardrail, retain those reaching at least "
            f"{args.near_best_benefit_fraction:.0%} of the largest actionable-mismatch reduction, "
            "then choose the smallest changed-pixel fraction."
        ),
        "baseline": {
            **baseline_metrics,
            "foreground_jaccard": baseline_foreground_jaccard,
        },
        "guardrails": {
            "max_changed_fraction": args.max_changed_fraction,
            "max_repaired_nucleus_fraction": args.max_repaired_nucleus_fraction,
            "max_multinucleation_rate_delta": args.max_multinucleation_rate_delta,
            "max_concordance_loss": args.max_concordance_loss,
            "min_bf_local_edge_ratio": args.min_bf_local_edge_ratio,
            "max_fragmentation_increase_rate": args.max_fragmentation_increase_rate,
            "max_area_change_q95": args.max_area_change_q95,
        },
    }
    (args.out_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# Nucleus-aware cell-mask refinement validation",
        "",
        "Alignment improvement is reported as a diagnostic, but it is not treated as independent accuracy evidence because nucleus cores were used to construct the candidates. Raw BF boundary support, label topology, area stability, and BF/Combined concordance are therefore enforced as guardrails.",
        "",
        f"Baseline actionable mismatch: **{baseline_metrics['actionable_rate']:.2%}**; baseline boundary-crossing union: **{baseline_metrics['crossing_union_rate']:.2%}**; baseline BF/Combined foreground Jaccard: **{baseline_foreground_jaccard:.2%}**.",
        "",
        "| candidate | actionable | reduction | changed pixels | BF local edge | area Q95 C/BF | fragmentation C/BF | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['candidate']} | {row['actionable_rate']:.2%} | {row['actionable_rate_reduction_pp']:.2f} pp | "
            f"{row['average_changed_pixel_fraction']:.3%} | {row['bf_local_edge_ratio']:.3f} | "
            f"{row['combined_q95_affected_label_area_change']:.1%}/{row['bf_q95_affected_label_area_change']:.1%} | "
            f"{row['combined_fragmentation_increase_rate']:.2%}/{row['bf_fragmentation_increase_rate']:.2%} | "
            f"{'yes' if row['eligible'] else 'no'} |"
        )
    markdown.extend(["", "## Decision", ""])
    if selected is None:
        markdown.append("No candidate passed every guardrail; retain the baseline masks.")
    else:
        markdown.append(
            f"Recommended candidate: **{selected['candidate']}**. {decision['selection_rule']}"
        )
    failures = [row for row in rows if row["guardrail_failures"]]
    if failures:
        markdown.extend(["", "## Guardrail failures", ""])
        for row in failures:
            markdown.append(f"- `{row['candidate']}`: {row['guardrail_failures']}")
    markdown.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "The selected candidate remains a prediction-calibration result, not a ground-truth accuracy estimate. Visual review and a small manual annotation set are required before treating the repaired cell boundaries as biologically validated.",
        ]
    )
    (args.out_dir / "validation_report.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
