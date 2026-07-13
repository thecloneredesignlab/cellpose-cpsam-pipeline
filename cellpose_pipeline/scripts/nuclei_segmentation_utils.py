#!/usr/bin/env python3
"""Shared utilities for conservative nuclear masks and nuclear QC metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NucleusCoreConfig:
    """Parameters for converting a nuclear extent mask into a robust core seed."""

    min_extent_area: int = 15
    min_core_area: int = 5
    local_bg_margin: int = 10
    snr_threshold: float = 2.0
    object_quantile: float = 0.35
    erode_px: int = 1
    smooth_sigma: float = 1.0


def odd_kernel_from_sigma(sigma: float) -> int:
    size = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    return size if size % 2 else size + 1


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32, copy=False)
    size = odd_kernel_from_sigma(sigma)
    return cv2.GaussianBlur(image.astype(np.float32), (size, size), float(sigma))


def robust_rescale(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.8,
) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    low, high = np.percentile(data, [low_percentile, high_percentile])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def smooth_sharpen(
    normalized: np.ndarray,
    smooth_radius: float,
    sharpen_radius: float,
) -> np.ndarray:
    """Band-pass a normalized image using small and large Gaussian scales."""

    small = gaussian_blur(normalized, smooth_radius) if smooth_radius > 0 else normalized
    if sharpen_radius <= 0:
        return np.clip(small, 0.0, 1.0).astype(np.float32, copy=False)
    large = gaussian_blur(normalized, sharpen_radius)
    return robust_rescale(small - large, 1.0, 99.8)


def instance_centroids(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64, copy=False)
    maximum = int(labels.max()) if labels.size else 0
    if maximum <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    yy, xx = np.nonzero(labels)
    ids = labels[yy, xx]
    areas = np.bincount(ids, minlength=maximum + 1).astype(np.float64)
    sum_y = np.bincount(ids, weights=yy, minlength=maximum + 1)
    sum_x = np.bincount(ids, weights=xx, minlength=maximum + 1)
    valid = np.flatnonzero(areas > 0)
    valid = valid[valid > 0]
    return np.column_stack((sum_y[valid] / areas[valid], sum_x[valid] / areas[valid]))


def summarize_instances(labels: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(np.int64, copy=False)
    counts = np.bincount(labels.ravel())
    areas = counts[1:]
    areas = areas[areas > 0]
    if not areas.size:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
            "tiny_fraction_lt35": 0.0,
        }
    return {
        "n_objects": int(areas.size),
        "mask_fraction": float(np.count_nonzero(labels) / labels.size),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "area_p90": float(np.percentile(areas, 90)),
        "tiny_fraction_lt35": float(np.mean(areas < 35)),
    }


def _largest_signal_component(binary: np.ndarray, signal: np.ndarray) -> np.ndarray:
    components, count = ndimage.label(binary)
    if count <= 1:
        return binary
    component_ids = np.arange(1, count + 1)
    signal_sums = ndimage.sum(signal, components, component_ids)
    areas = ndimage.sum(np.ones(binary.shape, dtype=np.uint8), components, component_ids)
    best = int(component_ids[np.lexsort((areas, signal_sums))[-1]])
    return components == best


def build_nucleus_core_seeds(
    raw_image: np.ndarray,
    extent_labels: np.ndarray,
    config: NucleusCoreConfig = NucleusCoreConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one conservative, intensity-supported seed for each retained extent.

    Instance identity is deliberately preserved: this step shrinks and filters extents
    but does not split one Cellpose instance into multiple nuclei.
    """

    raw = np.squeeze(raw_image).astype(np.float32, copy=False)
    labels = np.squeeze(extent_labels).astype(np.int32, copy=False)
    if raw.ndim != 2 or labels.ndim != 2 or raw.shape != labels.shape:
        raise ValueError(f"Expected matching 2D raw/mask arrays, got {raw.shape} and {labels.shape}")

    smoothed = gaussian_blur(raw, config.smooth_sigma)
    core_labels = np.zeros(labels.shape, dtype=np.int32)
    core_fractions: list[float] = []
    local_snrs: list[float] = []
    fallback_count = 0
    filtered_count = 0
    retained_count = 0
    margin = max(1, int(config.local_bg_margin))

    for label_id, obj_slice in enumerate(ndimage.find_objects(labels), start=1):
        if obj_slice is None:
            continue
        obj = labels[obj_slice] == label_id
        extent_area = int(np.count_nonzero(obj))
        if extent_area < int(config.min_extent_area):
            filtered_count += 1
            continue

        y0 = max(0, int(obj_slice[0].start) - margin)
        y1 = min(labels.shape[0], int(obj_slice[0].stop) + margin)
        x0 = max(0, int(obj_slice[1].start) - margin)
        x1 = min(labels.shape[1], int(obj_slice[1].stop) + margin)
        window_labels = labels[y0:y1, x0:x1]
        window_raw = smoothed[y0:y1, x0:x1]
        background = window_raw[window_labels == 0]
        if background.size < 20:
            background = window_raw.ravel()
            background = background[background <= np.percentile(background, 35)]
        bg_median = float(np.median(background)) if background.size else float(np.median(window_raw))
        bg_mad = float(np.median(np.abs(background - bg_median))) if background.size else 0.0
        bg_sigma = max(1.4826 * bg_mad, 0.5)

        obj_signal = smoothed[obj_slice][obj]
        object_floor = float(np.quantile(obj_signal, float(config.object_quantile)))
        signal_floor = bg_median + float(config.snr_threshold) * bg_sigma
        threshold = max(object_floor, signal_floor)
        core = obj & (smoothed[obj_slice] >= threshold)

        if config.erode_px > 0:
            eroded = ndimage.binary_erosion(obj, iterations=int(config.erode_px))
            eroded_core = core & eroded
            if np.count_nonzero(eroded_core) >= int(config.min_core_area):
                core = eroded_core

        core = _largest_signal_component(core, smoothed[obj_slice])
        if np.count_nonzero(core) < int(config.min_core_area):
            fallback_count += 1
            fallback = ndimage.binary_erosion(obj, iterations=1)
            if np.count_nonzero(fallback) < int(config.min_core_area):
                fallback = obj
            target = min(
                int(np.count_nonzero(fallback)),
                max(int(config.min_core_area), int(round(extent_area * 0.25))),
            )
            values = smoothed[obj_slice][fallback]
            if values.size > target:
                cutoff = float(np.partition(values, values.size - target)[values.size - target])
                fallback = fallback & (smoothed[obj_slice] >= cutoff)
                fallback = _largest_signal_component(fallback, smoothed[obj_slice])
            core = fallback

        core_area = int(np.count_nonzero(core))
        if core_area < int(config.min_core_area):
            filtered_count += 1
            continue
        target_view = core_labels[obj_slice]
        target_view[core] = label_id
        retained_count += 1
        core_fractions.append(core_area / extent_area)
        local_snrs.append((float(np.percentile(obj_signal, 90)) - bg_median) / bg_sigma)

    diagnostics: dict[str, Any] = {
        "n_core_objects": retained_count,
        "n_filtered_extents": filtered_count,
        "n_core_fallbacks": fallback_count,
        "median_core_to_extent_fraction": float(np.median(core_fractions)) if core_fractions else 0.0,
        "median_local_p90_snr": float(np.median(local_snrs)) if local_snrs else 0.0,
    }
    return core_labels, diagnostics


def centroid_match_metrics(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    max_distance: float = 6.0,
) -> dict[str, float | int]:
    reference = instance_centroids(reference_labels)
    candidate = instance_centroids(candidate_labels)
    if not len(reference) or not len(candidate):
        matches = 0
    else:
        distances, indices = cKDTree(reference).query(candidate, k=1)
        pairs = sorted(
            (float(distance), int(candidate_id), int(reference_id))
            for candidate_id, (distance, reference_id) in enumerate(zip(distances, indices))
            if distance <= max_distance
        )
        used_candidate: set[int] = set()
        used_reference: set[int] = set()
        matches = 0
        for _distance, candidate_id, reference_id in pairs:
            if candidate_id in used_candidate or reference_id in used_reference:
                continue
            used_candidate.add(candidate_id)
            used_reference.add(reference_id)
            matches += 1
    precision = matches / len(candidate) if len(candidate) else 0.0
    recall = matches / len(reference) if len(reference) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "center_matches": matches,
        "center_precision": precision,
        "center_recall": recall,
        "center_f1": f1,
    }


def cell_overlap_metrics(
    nucleus_labels: np.ndarray,
    cell_labels: np.ndarray,
    significant_fraction: float = 0.05,
    min_significant_pixels: int = 3,
) -> dict[str, float | int]:
    nuclei = nucleus_labels.astype(np.int32, copy=False)
    cells = cell_labels.astype(np.int32, copy=False)
    if nuclei.shape != cells.shape:
        raise ValueError(f"Nucleus/cell shapes differ: {nuclei.shape} versus {cells.shape}")

    centroids = instance_centroids(nuclei)
    containments: list[float] = []
    crossing = 0
    unmatched = 0
    centroid_inside = 0
    objects = list(ndimage.find_objects(nuclei))
    for index, obj_slice in enumerate(objects, start=1):
        if obj_slice is None:
            continue
        obj = nuclei[obj_slice] == index
        area = int(np.count_nonzero(obj))
        overlaps = cells[obj_slice][obj]
        counts = np.bincount(overlaps)
        foreground = counts[1:]
        best = int(foreground.max()) if foreground.size else 0
        containments.append(best / area if area else 0.0)
        threshold = max(int(min_significant_pixels), int(np.ceil(area * significant_fraction)))
        if np.count_nonzero(foreground >= threshold) > 1:
            crossing += 1
        if best == 0:
            unmatched += 1
        if index - 1 < len(centroids):
            y = min(max(int(round(centroids[index - 1, 0])), 0), cells.shape[0] - 1)
            x = min(max(int(round(centroids[index - 1, 1])), 0), cells.shape[1] - 1)
            centroid_inside += int(cells[y, x] > 0)

    count = len(containments)
    return {
        "n_nuclei_evaluated": count,
        "centroid_inside_rate": centroid_inside / count if count else 0.0,
        "median_containment": float(np.median(containments)) if containments else 0.0,
        "high_containment_rate": float(np.mean(np.asarray(containments) >= 0.80)) if containments else 0.0,
        "crossing_rate": crossing / count if count else 0.0,
        "unmatched_rate": unmatched / count if count else 0.0,
    }
