"""Dataset-adapted four-block features for the independent V4 cell-state axis.

The module consumes Brightfield-derived measurements, Nuclei raw/mask support,
and the raw Dead fluorescence plane.  It deliberately has no API for an
existing Dead segmentation, current classifier outputs, or trajectory data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


FEATURE_SCHEMA_VERSION = "multimodal_cell_state_v4_features_v1"
SHAPE_COLUMNS = (
    "area_px2",
    "roundness",
    "aspect_ratio",
    "extent",
    "solidity",
)
BRIGHTFIELD_COLUMNS = (
    "bf_object_median",
    "bf_object_iqr",
    "bf_object_p90_minus_p10",
    "bf_object_mean_minus_ring_background",
    "bf_interior_minus_boundary_mean",
    "bf_boundary_gradient_p95",
    "bf_laplacian_variance",
    "bf_edge_density",
    "bf_glcm_contrast",
    "bf_glcm_entropy",
    "bf_glcm_correlation",
)
NUCLEI_COLUMNS = (
    "nucleus_absent",
    "nuclei_count_excess",
    "nuclei_area_fraction_if_present",
    "largest_nucleus_fraction",
    "nucleus_area_cv",
    "small_nuclear_fragment_fraction",
    "nuclei_spatial_dispersion",
)
NUCLEI_QC_COLUMNS = (
    "nuclei_cell_to_ring_contrast",
    "nuclei_signal_positive_fraction",
)
NUCLEI_MEASUREMENT_STATUSES = (
    "present_supported",
    "absent_supported",
    "possible_nuclei_mask_miss",
    "low_quality_nuclei_signal",
)
DEAD_COLUMNS = (
    "dead_signal_absent",
    "dead_cell_to_ring_median_contrast",
    "dead_cell_to_ring_p90_contrast",
    "dead_object_p90_minus_p10",
    "dead_signal_positive_fraction",
    "dead_integrated_excess_per_area",
    "dead_largest_positive_component_fraction",
    "dead_positive_component_count_excess",
    "dead_signal_spatial_dispersion",
)
DEAD_QC_COLUMNS = ("dead_signal_saturation_fraction",)
DEAD_MEASUREMENT_STATUSES = (
    "dead_signal_present_supported",
    "dead_signal_absent_supported",
    "dead_signal_saturated",
    "dead_signal_background_uncertain",
)
MEASUREMENT_STATUSES = NUCLEI_MEASUREMENT_STATUSES
OUTPUT_FEATURE_COLUMNS = (
    SHAPE_COLUMNS
    + BRIGHTFIELD_COLUMNS
    + NUCLEI_COLUMNS
    + NUCLEI_QC_COLUMNS
    + DEAD_COLUMNS
    + DEAD_QC_COLUMNS
)


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def scalar_first_plane(raw: np.ndarray, *, label: str) -> np.ndarray:
    """Resolve one microscopy plane without silently treating pages as channels."""

    value = np.asarray(raw)
    value = np.squeeze(value)
    if value.ndim == 2:
        plane = value
    elif value.ndim == 3 and value.shape[-1] in (1, 3):
        plane = value[..., 0] if value.shape[-1] == 1 else np.mean(value, axis=-1)
    else:
        raise ValueError(f"{label} must resolve to one 2-D plane, got shape {value.shape}")
    plane = np.asarray(plane, dtype=np.float64)
    if not plane.size or not np.any(np.isfinite(plane)):
        raise ValueError(f"{label} contains no finite pixels")
    return plane


def validate_labels(mask: np.ndarray, *, label: str) -> np.ndarray:
    value = np.squeeze(np.asarray(mask))
    if value.ndim != 2:
        raise ValueError(f"{label} must resolve to one 2-D plane, got shape {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain finite numeric labels")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or np.min(rounded, initial=0) < 0:
        raise ValueError(f"{label} must contain nonnegative integer labels")
    return rounded.astype(np.int64, copy=False)


def robust_background_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 16:
        return float("nan"), float("nan")
    median = float(np.median(finite))
    mad_scale = 1.4826 * float(np.median(np.abs(finite - median)))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    scale = mad_scale if mad_scale > 0 else iqr_scale
    if not math.isfinite(scale) or scale <= 0:
        return median, float("nan")
    return median, scale


def nucleus_features_for_object(
    object_mask: np.ndarray,
    nuclei_mask: np.ndarray,
    nuclei_raw: np.ndarray,
    *,
    minimum_contrast: float = 3.0,
    minimum_positive_fraction: float = 0.05,
    ring_radius: int = 3,
) -> dict[str, Any]:
    """Return zero-aware topology and raw-signal support for one cell mask."""

    obj = np.asarray(object_mask, dtype=bool)
    if obj.shape != nuclei_mask.shape or obj.shape != nuclei_raw.shape:
        raise ValueError("Object, Nuclei mask, and Nuclei raw shapes differ")
    area = int(np.count_nonzero(obj))
    if area < 1:
        raise ValueError("Cannot calculate Nuclei features for an empty object")

    overlapping = nuclei_mask[obj]
    positive = overlapping[overlapping > 0]
    labels, counts = np.unique(positive, return_counts=True)
    counts = counts.astype(np.float64, copy=False)
    nuclei_count = int(labels.size)
    overlap_area = float(np.sum(counts))

    ring = ndimage.binary_dilation(obj, structure=_disk(ring_radius), border_value=0) & ~obj
    cell_values = nuclei_raw[obj & np.isfinite(nuclei_raw)]
    ring_values = nuclei_raw[ring & np.isfinite(nuclei_raw)]
    background_values = ring_values
    if background_values.size < 16:
        background_values = nuclei_raw[(~obj) & np.isfinite(nuclei_raw)]
    background_median, background_scale = robust_background_scale(background_values)
    qc_valid = (
        cell_values.size >= 4
        and math.isfinite(background_median)
        and math.isfinite(background_scale)
        and background_scale > 0
    )
    if qc_valid:
        cell_median = float(np.median(cell_values))
        contrast = (cell_median - background_median) / background_scale
        positive_fraction = float(
            np.mean(cell_values > background_median + minimum_contrast * background_scale)
        )
    else:
        contrast = float("nan")
        positive_fraction = float("nan")

    if nuclei_count > 0:
        status = "present_supported"
    elif not qc_valid:
        status = "low_quality_nuclei_signal"
    elif contrast >= minimum_contrast and positive_fraction >= minimum_positive_fraction:
        status = "possible_nuclei_mask_miss"
    else:
        status = "absent_supported"

    conditional_missing = nuclei_count == 0
    if conditional_missing:
        area_fraction = largest_fraction = area_cv = fragment_fraction = dispersion = float("nan")
    else:
        area_fraction = overlap_area / float(area)
        largest = float(np.max(counts))
        largest_fraction = largest / overlap_area
        area_cv = float(np.std(counts, ddof=0) / np.mean(counts)) if counts.size > 1 else 0.0
        small_limit = max(4.0, 0.25 * largest)
        fragment_fraction = float(np.sum(counts[counts <= small_limit]) / overlap_area)

        object_y, object_x = np.nonzero(obj)
        cell_y = float(np.mean(object_y))
        cell_x = float(np.mean(object_x))
        squared = []
        weights = []
        for nucleus_label, weight in zip(labels, counts, strict=True):
            selected = obj & (nuclei_mask == int(nucleus_label))
            yy, xx = np.nonzero(selected)
            if yy.size:
                squared.append((float(np.mean(yy)) - cell_y) ** 2 + (float(np.mean(xx)) - cell_x) ** 2)
                weights.append(float(weight))
        equivalent_radius = math.sqrt(area / math.pi)
        dispersion = (
            math.sqrt(float(np.average(squared, weights=weights))) / equivalent_radius
            if squared and equivalent_radius > 0
            else 0.0
        )

    return {
        "nuclei_count": nuclei_count,
        "nuclei_overlap_area_px2": int(overlap_area),
        "nuclei_measurement_status": status,
        "nucleus_absent": 1 if status == "absent_supported" else 0,
        "nuclei_count_excess": math.log1p(max(nuclei_count - 1, 0)),
        "nuclei_area_fraction_if_present": area_fraction,
        "largest_nucleus_fraction": largest_fraction,
        "nucleus_area_cv": area_cv,
        "small_nuclear_fragment_fraction": fragment_fraction,
        "nuclei_spatial_dispersion": dispersion,
        "nuclei_cell_to_ring_contrast": contrast,
        "nuclei_signal_positive_fraction": positive_fraction,
    }


def dead_features_for_object(
    object_mask: np.ndarray,
    dead_raw: np.ndarray,
    *,
    minimum_contrast: float = 3.0,
    minimum_positive_fraction: float = 0.02,
    minimum_component_pixels: int = 3,
    saturation_fraction_threshold: float = 0.05,
    saturation_value: float | None = None,
    ring_radius: int = 3,
) -> dict[str, Any]:
    """Return zero-aware raw Dead-fluorescence evidence for one cell.

    Absence is represented by a separate indicator and never treated as a
    numeric synonym for live.  Positive-signal topology is conditional on a
    supported signal and is left missing otherwise for neutral imputation by
    the projection layer.
    """

    obj = np.asarray(object_mask, dtype=bool)
    raw = np.asarray(dead_raw, dtype=np.float64)
    if obj.shape != raw.shape:
        raise ValueError("Object mask and Dead raw shapes differ")
    area = int(np.count_nonzero(obj))
    if area < 1:
        raise ValueError("Cannot calculate Dead features for an empty object")
    if minimum_component_pixels < 1:
        raise ValueError("Dead component minimum must be positive")

    ring = ndimage.binary_dilation(
        obj, structure=_disk(ring_radius), border_value=0
    ) & ~obj
    cell_values = raw[obj & np.isfinite(raw)]
    ring_values = raw[ring & np.isfinite(raw)]
    background_values = ring_values
    if background_values.size < 16:
        background_values = raw[(~obj) & np.isfinite(raw)]
    background_median, background_scale = robust_background_scale(background_values)
    qc_valid = (
        cell_values.size >= 4
        and math.isfinite(background_median)
        and math.isfinite(background_scale)
        and background_scale > 0
    )

    if saturation_value is not None and math.isfinite(float(saturation_value)):
        saturation_fraction = float(np.mean(cell_values >= float(saturation_value)))
    else:
        saturation_fraction = 0.0
    saturated = saturation_fraction >= saturation_fraction_threshold

    if qc_valid:
        threshold = background_median + minimum_contrast * background_scale
        cell_median = float(np.median(cell_values))
        cell_p10, cell_p90 = np.quantile(cell_values, [0.1, 0.9])
        median_contrast = (cell_median - background_median) / background_scale
        p90_contrast = (float(cell_p90) - background_median) / background_scale
        p90_minus_p10 = (float(cell_p90) - float(cell_p10)) / background_scale
        positive_mask = obj & np.isfinite(raw) & (raw > threshold)
        positive_count = int(np.count_nonzero(positive_mask))
        positive_fraction = positive_count / float(area)
        integrated_excess = float(
            np.sum(np.maximum(raw[obj & np.isfinite(raw)] - threshold, 0.0))
            / (float(area) * background_scale)
        )
    else:
        median_contrast = p90_contrast = p90_minus_p10 = float("nan")
        positive_fraction = integrated_excess = float("nan")
        positive_mask = np.zeros_like(obj, dtype=bool)
        positive_count = 0

    if saturated:
        status = "dead_signal_saturated"
    elif not qc_valid:
        status = "dead_signal_background_uncertain"
    elif positive_fraction >= minimum_positive_fraction:
        status = "dead_signal_present_supported"
    else:
        status = "dead_signal_absent_supported"

    if status == "dead_signal_present_supported":
        components, _ = ndimage.label(positive_mask)
        all_component_sizes = np.bincount(components.ravel())
        retained_labels = np.flatnonzero(
            all_component_sizes >= minimum_component_pixels
        )
        retained_labels = retained_labels[retained_labels != 0]
        component_sizes = all_component_sizes[retained_labels]
        retained_mask = np.isin(components, retained_labels) & positive_mask
        retained_count = int(np.count_nonzero(retained_mask))
        if component_sizes.size and retained_count:
            largest_fraction = float(np.max(component_sizes) / retained_count)
            component_count_excess = math.log1p(max(int(component_sizes.size) - 1, 0))
            object_y, object_x = np.nonzero(obj)
            cell_y, cell_x = float(np.mean(object_y)), float(np.mean(object_x))
            yy, xx = np.nonzero(retained_mask)
            equivalent_radius = math.sqrt(area / math.pi)
            dispersion = (
                math.sqrt(
                    float(np.mean((yy - cell_y) ** 2 + (xx - cell_x) ** 2))
                )
                / equivalent_radius
                if yy.size and equivalent_radius > 0
                else 0.0
            )
        else:
            largest_fraction = component_count_excess = dispersion = 0.0
    else:
        largest_fraction = component_count_excess = dispersion = float("nan")

    return {
        "dead_measurement_status": status,
        "dead_signal_absent": 1 if status == "dead_signal_absent_supported" else 0,
        "dead_cell_to_ring_median_contrast": median_contrast,
        "dead_cell_to_ring_p90_contrast": p90_contrast,
        "dead_object_p90_minus_p10": p90_minus_p10,
        "dead_signal_positive_fraction": positive_fraction,
        "dead_integrated_excess_per_area": integrated_excess,
        "dead_largest_positive_component_fraction": largest_fraction,
        "dead_positive_component_count_excess": component_count_excess,
        "dead_signal_spatial_dispersion": dispersion,
        "dead_signal_saturation_fraction": saturation_fraction,
    }


def derive_brightfield_features(row: dict[str, Any]) -> dict[str, float]:
    """Select complementary Brightfield evidence from the frozen V1 extractor."""

    def finite(name: str) -> float:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"Brightfield feature is nonfinite: {name}")
        return value

    result = {name: finite(name) for name in BRIGHTFIELD_COLUMNS if name != "bf_object_p90_minus_p10"}
    result["bf_object_p90_minus_p10"] = finite("bf_object_p90") - finite("bf_object_p10")
    return {name: result[name] for name in BRIGHTFIELD_COLUMNS}
