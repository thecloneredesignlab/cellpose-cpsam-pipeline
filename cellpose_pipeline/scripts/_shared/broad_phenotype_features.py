"""Deterministic Brightfield morphology features for broad-phenotype models.

The primary object universe is supplied by a Combined instance mask.  The
intensity source is the paired Brightfield raw image.  This module deliberately
has no knowledge of current cell-state predictions, Dead-channel measurements,
or trajectory-refinement outputs.

All geometry is reported in pixels.  Brightfield values are converted to one
scalar plane and normalized deterministically to approximately ``[0, 1]``
before intensity and texture measurements are calculated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


FEATURE_SCHEMA_VERSION = "broad_phenotype_features_v1"
FEATURE_CODE_VERSION = "bf_combined_mask_v1"
CONFIG_SCHEMA_VERSION = "broad_phenotype_feature_config_v1"

NUMERIC_FEATURE_COLUMNS = (
    "area_px2",
    "perimeter_px",
    "major_axis_px",
    "minor_axis_px",
    "equivalent_diameter_px",
    "aspect_ratio",
    "roundness",
    "extent",
    "solidity",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "bf_object_mean",
    "bf_object_median",
    "bf_object_sd",
    "bf_object_mad",
    "bf_object_p10",
    "bf_object_p90",
    "bf_object_iqr",
    "bf_global_background_median",
    "bf_global_background_mad",
    "bf_global_background_iqr",
    "bf_object_mean_minus_global_background",
    "bf_object_mean_over_global_background",
    "bf_object_robust_z_global_background",
    "bf_object_median_minus_global_background",
    "bf_object_iqr_over_background_iqr",
    "bf_ring_background_mean",
    "bf_object_mean_minus_ring_background",
    "bf_object_mean_over_ring_background",
    "bf_boundary_mean",
    "bf_interior_mean",
    "bf_interior_minus_boundary_mean",
    "bf_boundary_gradient_mean",
    "bf_boundary_gradient_p95",
    "bf_glcm_contrast",
    "bf_glcm_entropy",
    "bf_glcm_asm",
    "bf_glcm_idm",
    "bf_glcm_correlation",
    "bf_tenengrad_mean",
    "bf_tenengrad_p95",
    "bf_laplacian_variance",
    "bf_sobel_magnitude_mean",
    "bf_edge_density",
    "nuclei_comparator_enabled",
    "nuclei_count",
    "nuclei_overlap_area_px2",
    "nuclei_area_fraction",
)


@dataclass(frozen=True)
class BroadPhenotypeFeatureParameters:
    """Versioned numerical conventions used by the extractor."""

    epsilon: float = 1e-8
    object_crop_padding: int = 6
    object_ring_radius: int = 3
    global_background_dilation_radius: int = 5
    glcm_levels: int = 16
    glcm_min_area: int = 4
    glcm_min_pairs: int = 8
    edge_percentile: float = 90.0
    boundary_connectivity: int = 4
    glcm_offsets: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )


DEFAULT_PARAMETERS = BroadPhenotypeFeatureParameters()


def feature_schema_sha256() -> str:
    """Return a stable hash for feature names, order, and numerical settings."""

    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_code_version": FEATURE_CODE_VERSION,
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "parameters": asdict(DEFAULT_PARAMETERS),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_number(
    value: Any,
    field: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if integer and number != math.floor(number):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return int(number) if integer else number


def _validate_ordered_subset(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty JSON array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must contain nonempty feature names")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicate feature names")
    unknown = [item for item in value if item not in NUMERIC_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"{field} contains unknown features: {unknown}")
    positions = [NUMERIC_FEATURE_COLUMNS.index(item) for item in value]
    if positions != sorted(positions):
        raise ValueError(f"{field} must preserve numeric_feature_columns order")
    return tuple(value)


def load_feature_config(path: Any) -> dict[str, Any]:
    """Read and strictly validate the shared extractor/model feature contract."""

    from pathlib import Path

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Broad-phenotype feature config must be a JSON object: {config_path}")
    required = {
        "schema_version",
        "method_version",
        "feature_schema_version",
        "numeric_feature_columns",
        "primary_model_feature_columns",
        "nuclei_comparator_feature_columns",
        "parameters",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(
            f"Broad-phenotype feature config keys mismatch: missing={missing}, unknown={unknown}"
        )
    expected_scalars = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "method_version": FEATURE_CODE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected:
            raise ValueError(
                f"Broad-phenotype feature config {field} mismatch: "
                f"expected={expected!r}, observed={payload[field]!r}"
            )
    if payload["numeric_feature_columns"] != list(NUMERIC_FEATURE_COLUMNS):
        raise ValueError(
            "Broad-phenotype numeric_feature_columns must match the extractor schema exactly"
        )
    primary = _validate_ordered_subset(
        payload["primary_model_feature_columns"],
        "primary_model_feature_columns",
    )
    comparator = _validate_ordered_subset(
        payload["nuclei_comparator_feature_columns"],
        "nuclei_comparator_feature_columns",
    )
    if not set(primary).issubset(comparator):
        raise ValueError(
            "nuclei_comparator_feature_columns must include every primary_model_feature_columns entry"
        )
    prohibited_primary_prefixes = ("centroid_", "bbox_")
    prohibited_primary = [
        feature for feature in primary if feature.startswith(prohibited_primary_prefixes)
    ]
    if prohibited_primary:
        raise ValueError(
            "primary_model_feature_columns contains localization fields: "
            f"{prohibited_primary}"
        )
    nucleus_columns = {
        "nuclei_count",
        "nuclei_overlap_area_px2",
        "nuclei_area_fraction",
    }
    if not nucleus_columns.issubset(comparator):
        raise ValueError(
            "nuclei_comparator_feature_columns must include all nucleus comparator measurements"
        )
    selected_nucleus_columns = nucleus_columns.intersection(primary)
    if selected_nucleus_columns and selected_nucleus_columns != nucleus_columns:
        raise ValueError(
            "primary_model_feature_columns must include either all or none of the "
            "nucleus support measurements"
        )
    if "nuclei_comparator_enabled" in comparator:
        raise ValueError(
            "nuclei_comparator_enabled is a provenance flag, not a model feature"
        )
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON object")
    parameter_fields = {
        "epsilon",
        "object_crop_padding",
        "object_ring_radius",
        "global_background_dilation_radius",
        "glcm_levels",
        "glcm_min_area",
        "glcm_min_pairs",
        "edge_percentile",
        "boundary_connectivity",
        "glcm_offsets",
    }
    missing_parameters = sorted(parameter_fields - set(parameters))
    unknown_parameters = sorted(set(parameters) - parameter_fields)
    if missing_parameters or unknown_parameters:
        raise ValueError(
            "Broad-phenotype feature parameters mismatch: "
            f"missing={missing_parameters}, unknown={unknown_parameters}"
        )
    offsets_value = parameters["glcm_offsets"]
    if not isinstance(offsets_value, list) or not offsets_value:
        raise ValueError("parameters.glcm_offsets must be a nonempty JSON array")
    offsets: list[tuple[int, int]] = []
    for index, offset in enumerate(offsets_value):
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError(f"parameters.glcm_offsets[{index}] must contain [dy, dx]")
        dy = _validate_number(offset[0], f"parameters.glcm_offsets[{index}][0]", integer=True)
        dx = _validate_number(offset[1], f"parameters.glcm_offsets[{index}][1]", integer=True)
        if dy == 0 and dx == 0:
            raise ValueError("parameters.glcm_offsets must not contain [0, 0]")
        offsets.append((int(dy), int(dx)))
    if len(offsets) != len(set(offsets)):
        raise ValueError("parameters.glcm_offsets must not contain duplicates")
    boundary_connectivity = _validate_number(
        parameters["boundary_connectivity"],
        "parameters.boundary_connectivity",
        integer=True,
    )
    if boundary_connectivity != 4:
        raise ValueError("Version 1 requires parameters.boundary_connectivity=4")
    resolved_parameters = BroadPhenotypeFeatureParameters(
        epsilon=float(_validate_number(parameters["epsilon"], "parameters.epsilon", minimum=1e-15)),
        object_crop_padding=int(
            _validate_number(
                parameters["object_crop_padding"],
                "parameters.object_crop_padding",
                integer=True,
                minimum=0,
            )
        ),
        object_ring_radius=int(
            _validate_number(
                parameters["object_ring_radius"],
                "parameters.object_ring_radius",
                integer=True,
                minimum=1,
            )
        ),
        global_background_dilation_radius=int(
            _validate_number(
                parameters["global_background_dilation_radius"],
                "parameters.global_background_dilation_radius",
                integer=True,
                minimum=0,
            )
        ),
        glcm_levels=int(
            _validate_number(
                parameters["glcm_levels"],
                "parameters.glcm_levels",
                integer=True,
                minimum=2,
                maximum=256,
            )
        ),
        glcm_min_area=int(
            _validate_number(
                parameters["glcm_min_area"],
                "parameters.glcm_min_area",
                integer=True,
                minimum=1,
            )
        ),
        glcm_min_pairs=int(
            _validate_number(
                parameters["glcm_min_pairs"],
                "parameters.glcm_min_pairs",
                integer=True,
                minimum=1,
            )
        ),
        edge_percentile=float(
            _validate_number(
                parameters["edge_percentile"],
                "parameters.edge_percentile",
                minimum=0,
                maximum=100,
            )
        ),
        boundary_connectivity=int(boundary_connectivity),
        glcm_offsets=tuple(offsets),
    )
    raw_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    semantic_payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "method_version": FEATURE_CODE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "primary_model_feature_columns": list(primary),
        "nuclei_comparator_feature_columns": list(comparator),
        "parameters": parameters_dict(resolved_parameters),
    }
    semantic_text = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
    return {
        "path": config_path,
        "sha256": raw_sha256,
        "semantic_sha256": hashlib.sha256(semantic_text.encode("utf-8")).hexdigest(),
        "primary_model_feature_columns": primary,
        "nuclei_comparator_feature_columns": comparator,
        "parameters": resolved_parameters,
        "payload": payload,
    }


def parameters_dict(
    parameters: BroadPhenotypeFeatureParameters = DEFAULT_PARAMETERS,
) -> dict[str, Any]:
    """Return JSON-serializable feature parameters."""

    payload = asdict(parameters)
    payload["glcm_offsets"] = [list(offset) for offset in parameters.glcm_offsets]
    return payload


def validate_instance_mask(mask: np.ndarray, *, label: str = "instance mask") -> np.ndarray:
    """Validate and normalize a two-dimensional nonnegative integer mask."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.int64, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        if array.size and int(np.min(array)) < 0:
            raise ValueError(f"{label} contains negative labels")
        return array.astype(np.int64, copy=False)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must contain integer labels, got dtype {array.dtype}")
    finite = np.isfinite(array)
    if not bool(np.all(finite)):
        raise ValueError(f"{label} contains non-finite labels")
    rounded = np.rint(array)
    if not bool(np.array_equal(array, rounded)):
        raise ValueError(f"{label} contains non-integer labels")
    if array.size and float(np.min(rounded)) < 0:
        raise ValueError(f"{label} contains negative labels")
    maximum = float(np.max(rounded)) if rounded.size else 0.0
    if maximum > float(np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains labels outside int64 range")
    return rounded.astype(np.int64, copy=False)


def scalar_brightfield(raw: np.ndarray) -> np.ndarray:
    """Convert a 2-D or channel-last RGB Brightfield image to one float plane."""

    image = np.asarray(raw)
    if image.ndim == 2:
        scalar = image
    elif image.ndim == 3 and image.shape[2] == 1:
        scalar = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        scalar = np.mean(image.astype(np.float64, copy=False), axis=2)
    else:
        raise ValueError(
            "Brightfield raw image must be 2-D, HxWx1, or channel-last HxWx3; "
            f"got shape {image.shape}"
        )
    scalar = scalar.astype(np.float64, copy=False)
    if scalar.size == 0 or not bool(np.any(np.isfinite(scalar))):
        raise ValueError("Brightfield raw image contains no finite pixels")
    return scalar


def normalize_brightfield(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize Brightfield values without data-dependent percentile clipping."""

    scalar = scalar_brightfield(raw)
    finite = scalar[np.isfinite(scalar)]
    input_min = float(np.min(finite))
    input_max = float(np.max(finite))
    normalized = scalar.copy()
    if input_min >= 0.0 and input_max <= 1.0:
        scale = "already_0_1"
    elif input_min >= 0.0 and input_max > 0.0:
        normalized /= input_max
        scale = "divide_by_input_max"
    elif input_max > input_min:
        normalized = (normalized - input_min) / (input_max - input_min)
        scale = "minmax"
    else:
        normalized[np.isfinite(normalized)] = 0.0
        scale = "constant_nonpositive_to_zero"
    return normalized, {
        "normalization": scale,
        "input_min": input_min,
        "input_max": input_max,
    }


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _four_neighbor_structure() -> np.ndarray:
    return np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )


def _finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    return float(np.quantile(finite, probability, method="linear"))


def _median_absolute_deviation(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def _iqr(values: np.ndarray) -> float:
    return _finite_quantile(values, 0.75) - _finite_quantile(values, 0.25)


def _background_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    parameters: BroadPhenotypeFeatureParameters,
) -> dict[str, Any]:
    finite = np.isfinite(image)
    excluded = ndimage.binary_dilation(
        mask > 0,
        structure=_disk(parameters.global_background_dilation_radius),
        border_value=0,
    )
    usable = finite & ~excluded
    status = "outside_dilated_foreground"
    if not bool(np.any(usable)):
        usable = finite & (mask == 0)
        status = "fallback_outside_foreground"
    if not bool(np.any(usable)):
        usable = finite
        status = "fallback_all_finite_pixels"
    values = image[usable]
    return {
        "values": values,
        "median": float(np.median(values)),
        "mad": _median_absolute_deviation(values),
        "iqr": _iqr(values),
        "usable_fraction": float(np.mean(usable)),
        "status": status,
    }


def _gradient_images(image: np.ndarray) -> dict[str, np.ndarray | float]:
    finite_image = np.where(np.isfinite(image), image, 0.0)
    gx = ndimage.sobel(finite_image, axis=1, mode="constant", cval=0.0)
    gy = ndimage.sobel(finite_image, axis=0, mode="constant", cval=0.0)
    tenengrad = gx * gx + gy * gy
    magnitude = np.sqrt(tenengrad)
    laplacian = ndimage.laplace(finite_image, mode="constant", cval=0.0)
    return {
        "gx": gx,
        "gy": gy,
        "tenengrad": tenengrad,
        "magnitude": magnitude,
        "laplacian": laplacian,
    }


def _object_perimeter(object_mask: np.ndarray) -> float:
    area = int(np.count_nonzero(object_mask))
    horizontal_adjacencies = int(np.count_nonzero(object_mask[:, :-1] & object_mask[:, 1:]))
    vertical_adjacencies = int(np.count_nonzero(object_mask[:-1, :] & object_mask[1:, :]))
    return float(4 * area - 2 * horizontal_adjacencies - 2 * vertical_adjacencies)


def _convex_pixel_area(object_mask: np.ndarray) -> float:
    boundary = object_mask & ~ndimage.binary_erosion(
        object_mask,
        structure=_four_neighbor_structure(),
        border_value=0,
    )
    yy, xx = np.nonzero(boundary)
    if not yy.size:
        return 0.0
    corners = np.concatenate(
        (
            np.column_stack((xx, yy)),
            np.column_stack((xx + 1, yy)),
            np.column_stack((xx, yy + 1)),
            np.column_stack((xx + 1, yy + 1)),
        ),
        axis=0,
    ).astype(np.float64, copy=False)
    corners = np.unique(corners, axis=0)
    if corners.shape[0] < 3:
        return float(np.count_nonzero(object_mask))
    try:
        return float(ConvexHull(corners).volume)
    except QhullError:
        return float(np.count_nonzero(object_mask))


def _geometry_features(
    object_mask: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> dict[str, float | int]:
    yy, xx = np.nonzero(object_mask)
    area = int(yy.size)
    if area == 0:
        raise ValueError("Cannot compute geometry for an empty object")
    centroid_x_local = float(np.mean(xx))
    centroid_y_local = float(np.mean(yy))
    centered = np.column_stack((xx - centroid_x_local, yy - centroid_y_local))
    covariance = centered.T @ centered / float(area)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minor_axis = float(4.0 * math.sqrt(float(eigenvalues[0])))
    major_axis = float(4.0 * math.sqrt(float(eigenvalues[-1])))
    if minor_axis > 0.0:
        aspect_ratio = major_axis / minor_axis
    elif major_axis == 0.0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = float("nan")
    perimeter = _object_perimeter(object_mask)
    convex_area = _convex_pixel_area(object_mask)
    bbox_height, bbox_width = object_mask.shape
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "equivalent_diameter_px": float(math.sqrt(4.0 * area / math.pi)),
        "aspect_ratio": aspect_ratio,
        "roundness": float(4.0 * math.pi * area / (perimeter * perimeter))
        if perimeter > 0.0
        else float("nan"),
        "extent": float(area / float(bbox_height * bbox_width)),
        "solidity": float(area / convex_area) if convex_area > 0.0 else float("nan"),
        "centroid_x_px": centroid_x_local + x_offset,
        "centroid_y_px": centroid_y_local + y_offset,
        "bbox_x_min": x_offset,
        "bbox_y_min": y_offset,
        "bbox_x_max": x_offset + bbox_width,
        "bbox_y_max": y_offset + bbox_height,
    }


def _glcm_metrics(
    image: np.ndarray,
    object_mask: np.ndarray,
    parameters: BroadPhenotypeFeatureParameters,
) -> dict[str, float]:
    names = (
        "bf_glcm_contrast",
        "bf_glcm_entropy",
        "bf_glcm_asm",
        "bf_glcm_idm",
        "bf_glcm_correlation",
    )
    missing = {name: float("nan") for name in names}
    if int(np.count_nonzero(object_mask)) < parameters.glcm_min_area:
        return missing
    levels = int(parameters.glcm_levels)
    clipped = np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, 1.0)
    quantized = np.floor(clipped * (levels - 1)).astype(np.int64)
    quantized[~np.isfinite(image)] = -1
    per_offset: list[dict[str, float]] = []
    height, width = image.shape
    for dy, dx in parameters.glcm_offsets:
        y1_start = max(0, -dy)
        y1_stop = min(height, height - dy)
        x1_start = max(0, -dx)
        x1_stop = min(width, width - dx)
        if y1_start >= y1_stop or x1_start >= x1_stop:
            continue
        y2_start = y1_start + dy
        y2_stop = y1_stop + dy
        x2_start = x1_start + dx
        x2_stop = x1_stop + dx
        mask_a = object_mask[y1_start:y1_stop, x1_start:x1_stop]
        mask_b = object_mask[y2_start:y2_stop, x2_start:x2_stop]
        values_a = quantized[y1_start:y1_stop, x1_start:x1_stop]
        values_b = quantized[y2_start:y2_stop, x2_start:x2_stop]
        valid = mask_a & mask_b & (values_a >= 0) & (values_b >= 0)
        if int(np.count_nonzero(valid)) < parameters.glcm_min_pairs:
            continue
        linear = values_a[valid] * levels + values_b[valid]
        counts = np.bincount(linear, minlength=levels * levels).reshape(levels, levels)
        probability = counts.astype(np.float64) / float(np.sum(counts))
        ii, jj = np.indices(probability.shape, dtype=np.float64)
        difference_squared = (ii - jj) ** 2
        nonzero = probability[probability > 0.0]
        row_marginal = np.sum(probability, axis=1)
        column_marginal = np.sum(probability, axis=0)
        indices = np.arange(levels, dtype=np.float64)
        mean_i = float(np.sum(indices * row_marginal))
        mean_j = float(np.sum(indices * column_marginal))
        sd_i = float(np.sqrt(np.sum(((indices - mean_i) ** 2) * row_marginal)))
        sd_j = float(np.sqrt(np.sum(((indices - mean_j) ** 2) * column_marginal)))
        correlation = (
            float(np.sum((ii - mean_i) * (jj - mean_j) * probability) / (sd_i * sd_j))
            if sd_i > 0.0 and sd_j > 0.0
            else float("nan")
        )
        per_offset.append(
            {
                "bf_glcm_contrast": float(np.sum(difference_squared * probability)),
                "bf_glcm_entropy": float(-np.sum(nonzero * np.log2(nonzero))),
                "bf_glcm_asm": float(np.sum(probability * probability)),
                "bf_glcm_idm": float(np.sum(probability / (1.0 + difference_squared))),
                "bf_glcm_correlation": correlation,
            }
        )
    if not per_offset:
        return missing
    result: dict[str, float] = {}
    for name in names:
        values = np.asarray([record[name] for record in per_offset], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[name] = float(np.mean(finite)) if finite.size else float("nan")
    return result


def _nucleus_comparator(
    object_mask: np.ndarray,
    nucleus_crop: np.ndarray | None,
) -> dict[str, float | int]:
    if nucleus_crop is None:
        return {
            "nuclei_comparator_enabled": 0,
            "nuclei_count": float("nan"),
            "nuclei_overlap_area_px2": float("nan"),
            "nuclei_area_fraction": float("nan"),
        }
    overlapping = nucleus_crop[object_mask]
    positive = overlapping[overlapping > 0]
    overlap_area = int(positive.size)
    object_area = int(np.count_nonzero(object_mask))
    return {
        "nuclei_comparator_enabled": 1,
        "nuclei_count": int(np.unique(positive).size),
        "nuclei_overlap_area_px2": overlap_area,
        "nuclei_area_fraction": float(overlap_area / object_area),
    }


def extract_broad_phenotype_features(
    brightfield_raw: np.ndarray,
    combined_mask: np.ndarray,
    *,
    nuclei_mask: np.ndarray | None = None,
    parameters: BroadPhenotypeFeatureParameters = DEFAULT_PARAMETERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate one stable wide feature row per positive Combined mask label."""

    mask = validate_instance_mask(combined_mask, label="Combined mask")
    image, normalization = normalize_brightfield(brightfield_raw)
    if image.shape != mask.shape:
        raise ValueError(
            f"Brightfield raw shape {image.shape} does not match Combined mask shape {mask.shape}"
        )
    normalized_nuclei: np.ndarray | None = None
    if nuclei_mask is not None:
        normalized_nuclei = validate_instance_mask(nuclei_mask, label="Nuclei comparator mask")
        if normalized_nuclei.shape != mask.shape:
            raise ValueError(
                "Nuclei comparator mask shape "
                f"{normalized_nuclei.shape} does not match Combined mask shape {mask.shape}"
            )
    foreground_y, foreground_x = np.nonzero(mask)
    foreground_labels = mask[foreground_y, foreground_x]
    labels, inverse = np.unique(foreground_labels, return_inverse=True)
    if labels.size:
        y0_by_object = np.full(labels.size, mask.shape[0], dtype=np.int64)
        x0_by_object = np.full(labels.size, mask.shape[1], dtype=np.int64)
        y1_by_object = np.zeros(labels.size, dtype=np.int64)
        x1_by_object = np.zeros(labels.size, dtype=np.int64)
        np.minimum.at(y0_by_object, inverse, foreground_y)
        np.minimum.at(x0_by_object, inverse, foreground_x)
        np.maximum.at(y1_by_object, inverse, foreground_y + 1)
        np.maximum.at(x1_by_object, inverse, foreground_x + 1)
    else:
        y0_by_object = x0_by_object = y1_by_object = x1_by_object = np.zeros(
            0,
            dtype=np.int64,
        )
    background = _background_statistics(image, mask, parameters)
    gradients = _gradient_images(image)
    magnitude = np.asarray(gradients["magnitude"])
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    edge_threshold = (
        float(np.percentile(finite_magnitude, parameters.edge_percentile))
        if finite_magnitude.size
        else float("nan")
    )
    rows: list[dict[str, Any]] = []
    for object_index, label_value in enumerate(labels):
        label = int(label_value)
        padding = max(
            int(parameters.object_crop_padding),
            int(parameters.object_ring_radius),
            1,
        )
        y_min = max(0, int(y0_by_object[object_index]) - padding)
        y_max = min(mask.shape[0], int(y1_by_object[object_index]) + padding)
        x_min = max(0, int(x0_by_object[object_index]) - padding)
        x_max = min(mask.shape[1], int(x1_by_object[object_index]) + padding)
        mask_crop = mask[y_min:y_max, x_min:x_max]
        image_crop = image[y_min:y_max, x_min:x_max]
        object_mask = mask_crop == label
        object_y, object_x = np.nonzero(object_mask)
        bbox_y_min = int(np.min(object_y))
        bbox_y_max = int(np.max(object_y)) + 1
        bbox_x_min = int(np.min(object_x))
        bbox_x_max = int(np.max(object_x)) + 1
        geometry_object = object_mask[
            bbox_y_min:bbox_y_max,
            bbox_x_min:bbox_x_max,
        ]
        row: dict[str, Any] = {"mask_label": label}
        row.update(
            _geometry_features(
                geometry_object,
                x_offset=x_min + bbox_x_min,
                y_offset=y_min + bbox_y_min,
            )
        )
        object_values = image_crop[object_mask & np.isfinite(image_crop)]
        if not object_values.size:
            raise ValueError(f"Brightfield object label {label} contains no finite pixels")
        object_mean = float(np.mean(object_values))
        object_median = float(np.median(object_values))
        object_iqr = _iqr(object_values)
        row.update(
            {
                "bf_object_mean": object_mean,
                "bf_object_median": object_median,
                "bf_object_sd": float(np.std(object_values, ddof=0)),
                "bf_object_mad": _median_absolute_deviation(object_values),
                "bf_object_p10": _finite_quantile(object_values, 0.10),
                "bf_object_p90": _finite_quantile(object_values, 0.90),
                "bf_object_iqr": object_iqr,
                "bf_global_background_median": background["median"],
                "bf_global_background_mad": background["mad"],
                "bf_global_background_iqr": background["iqr"],
                "bf_object_mean_minus_global_background": object_mean
                - float(background["median"]),
                "bf_object_mean_over_global_background": object_mean
                / (float(background["median"]) + parameters.epsilon),
                "bf_object_robust_z_global_background": (
                    object_mean - float(background["median"])
                )
                / (1.4826 * float(background["mad"]) + parameters.epsilon),
                "bf_object_median_minus_global_background": object_median
                - float(background["median"]),
                "bf_object_iqr_over_background_iqr": object_iqr
                / (float(background["iqr"]) + parameters.epsilon),
            }
        )
        ring = ndimage.binary_dilation(
            object_mask,
            structure=_disk(parameters.object_ring_radius),
            border_value=0,
        ) & (mask_crop == 0)
        ring_values = image_crop[ring & np.isfinite(image_crop)]
        ring_mean = (
            float(np.mean(ring_values))
            if ring_values.size
            else float(background["median"])
        )
        row.update(
            {
                "bf_ring_background_mean": ring_mean,
                "bf_object_mean_minus_ring_background": object_mean - ring_mean,
                "bf_object_mean_over_ring_background": object_mean
                / (ring_mean + parameters.epsilon),
            }
        )
        boundary = object_mask & ~ndimage.binary_erosion(
            object_mask,
            structure=_four_neighbor_structure(),
            border_value=0,
        )
        interior = object_mask & ~boundary
        boundary_values = image_crop[boundary & np.isfinite(image_crop)]
        interior_values = image_crop[interior & np.isfinite(image_crop)]
        if not interior_values.size:
            interior_values = object_values
        boundary_mean = (
            float(np.mean(boundary_values)) if boundary_values.size else float("nan")
        )
        interior_mean = float(np.mean(interior_values))
        magnitude_crop = magnitude[y_min:y_max, x_min:x_max]
        boundary_gradient = magnitude_crop[boundary & np.isfinite(magnitude_crop)]
        tenengrad_crop = np.asarray(gradients["tenengrad"])[y_min:y_max, x_min:x_max]
        laplacian_crop = np.asarray(gradients["laplacian"])[y_min:y_max, x_min:x_max]
        tenengrad_values = tenengrad_crop[object_mask & np.isfinite(tenengrad_crop)]
        magnitude_values = magnitude_crop[object_mask & np.isfinite(magnitude_crop)]
        laplacian_values = laplacian_crop[object_mask & np.isfinite(laplacian_crop)]
        row.update(
            {
                "bf_boundary_mean": boundary_mean,
                "bf_interior_mean": interior_mean,
                "bf_interior_minus_boundary_mean": interior_mean - boundary_mean,
                "bf_boundary_gradient_mean": float(np.mean(boundary_gradient))
                if boundary_gradient.size
                else float("nan"),
                "bf_boundary_gradient_p95": _finite_quantile(boundary_gradient, 0.95),
                "bf_tenengrad_mean": float(np.mean(tenengrad_values))
                if tenengrad_values.size
                else float("nan"),
                "bf_tenengrad_p95": _finite_quantile(tenengrad_values, 0.95),
                "bf_laplacian_variance": float(np.var(laplacian_values, ddof=1))
                if laplacian_values.size > 1
                else float("nan"),
                "bf_sobel_magnitude_mean": float(np.mean(magnitude_values))
                if magnitude_values.size
                else float("nan"),
                "bf_edge_density": float(np.mean(magnitude_values > edge_threshold))
                if magnitude_values.size and math.isfinite(edge_threshold)
                else float("nan"),
            }
        )
        row.update(_glcm_metrics(image_crop, object_mask, parameters))
        nucleus_crop = (
            normalized_nuclei[y_min:y_max, x_min:x_max]
            if normalized_nuclei is not None
            else None
        )
        row.update(_nucleus_comparator(object_mask, nucleus_crop))
        missing = set(NUMERIC_FEATURE_COLUMNS) - set(row)
        extra = set(row) - ({"mask_label"} | set(NUMERIC_FEATURE_COLUMNS))
        if missing or extra:
            raise RuntimeError(
                f"Internal broad-phenotype feature schema mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        rows.append(row)
    diagnostics = {
        **normalization,
        "object_count": len(rows),
        "positive_labels_sha256": hashlib.sha256(
            "\n".join(str(int(label)) for label in labels).encode("utf-8")
        ).hexdigest(),
        "background_status": background["status"],
        "usable_background_fraction": background["usable_fraction"],
        "edge_threshold": edge_threshold,
        "nuclei_comparator_enabled": normalized_nuclei is not None,
        "feature_schema_sha256": feature_schema_sha256(),
    }
    return rows, diagnostics
