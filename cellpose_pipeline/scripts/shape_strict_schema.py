"""Canonical CSV schemas shared by shape-strict workers and finalizers."""

from __future__ import annotations


SHAPE_MULTIPEAK_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "key",
    "nucleus_id",
    "parent_area",
    "parent_solidity",
    "parent_circularity",
    "parent_axis_ratio",
    "parent_eccentricity",
    "parent_shape_score",
    "n_peaks",
    "watershed_complete",
    "peak1_y",
    "peak1_x",
    "peak2_y",
    "peak2_x",
    "peak_distance",
    "min_peak_snr",
    "third_peak_ratio",
    "valley_drop_snr",
    "valley_drop_fraction",
    "stability_fraction",
    "child1_area",
    "child2_area",
    "child1_solidity",
    "child2_solidity",
    "child1_circularity",
    "child2_circularity",
    "child1_axis_ratio",
    "child2_axis_ratio",
    "shape_gain",
    "cell_support_fraction",
    "children_same_cell_pair",
    "accepted_configs",
)

SHAPE_MULTIPEAK_REQUIRED_FIELDS: tuple[str, ...] = (
    "key",
    "nucleus_id",
    "parent_area",
    "parent_solidity",
    "parent_circularity",
    "parent_axis_ratio",
    "n_peaks",
    "accepted_configs",
)
