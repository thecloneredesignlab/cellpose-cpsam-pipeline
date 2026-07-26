#!/usr/bin/env python3
"""Shared density-aware late-death trajectory model and calibration entrypoint.

The optimization does not treat the current classifier's repeated live calls as
ground truth.  Frozen d0 live objects and untreated trajectories are safety
anchors; strong blue-positive objects and tracked post-blue remnants are death
anchors.  E9 site 1 at day 5 is the expert-supplied field-level development
constraint.  E9 sites 2-4 and F9 are never used to rank parameter sets.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


RAW_FEATURES = {
    "core_nc_percentile": ("core_nucleus_to_cytoplasm", "high"),
    "extent_nc_percentile": ("extent_nucleus_to_cytoplasm", "high"),
    "area_depletion_percentile": ("mask_area", "low"),
    "cytoplasm_depletion_percentile": ("core_cytoplasm_area", "low"),
    "red_mass_depletion_percentile": ("red_mass_proxy", "low"),
    "cytoplasm_red_mass_depletion_percentile": (
        "cytoplasm_red_mass_proxy",
        "low",
    ),
    "circularity_percentile": ("mask_circularity", "high"),
    "solidity_percentile": ("mask_solidity", "high"),
    "cell_dead_snr_percentile": ("cell_dead_snr", "high"),
    "cell_dead_positive_fraction_percentile": (
        "cell_dead_positive_fraction",
        "high",
    ),
    "cell_dead_core_enrichment_percentile": (
        "cell_dead_core_enrichment",
        "high",
    ),
}
MODEL_FEATURES = tuple(RAW_FEATURES)
FIELD_RATIO_COLUMNS = (
    "field_count_ratio_to_peak",
    "field_area_ratio_to_peak",
    "field_cytoplasm_ratio_to_peak",
    "field_red_mass_ratio_to_peak",
    "field_mask_fraction_ratio_to_peak",
)
E9_SENTINEL_KEY = "E9_1_05d00h00m"
DENSITY_ANCHORS = (0.10, 0.30, 0.50, 0.70, 0.90)
APPROVED_FIELD_CONFIGURATION = {
    "area_ratio_max": 0.25,
    "count_ratio_max": 0.60,
    "mask_fraction_ratio_max": 0.45,
    "minimum_field_signals": 4,
    "minimum_site_fraction": 0.75,
    "persistence_frames": 3,
    "recovery_frames": 3,
    "recovery_signal_max": 1,
    "red_mass_ratio_max": 0.25,
}
APPROVED_OBJECT_CONFIGURATION = {
    "feature_threshold": 0.50,
    "healthy_threshold": 0.45,
    "minimum_death_signals": 2,
    "minimum_healthy_signals": 3,
    "branch_match_distance_px": 10.0,
    "unmatched_minimum_death_signals": 3,
    "temporal_minimum_support_frames": 2,
}
APPROVED_LATE_MIN_HOURS = 72.0
APPROVED_CONFIGURATION_SOURCE = (
    "death_classification_consensus_optimization_20260725_213646/"
    "optimization/best_configuration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260723)
    parser.add_argument("--target-e9-dead-fraction", type=float, default=0.95)
    parser.add_argument("--target-holdout-dead-fraction", type=float, default=0.90)
    parser.add_argument("--max-control-false-positive-rate", type=float, default=0.01)
    parser.add_argument("--late-min-hours", type=float, default=72.0)
    parser.add_argument(
        "--max-nonfocus-rows-per-field",
        type=int,
        default=750,
        help=(
            "Deterministic per-field cap outside E9/F9. All blue-supported and "
            "temporal-remnant anchors are retained before sampling."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def finite_numeric(values: pd.Series, fallback: float = 0.0) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if result.notna().any():
        return result.fillna(float(result.median()))
    return pd.Series(fallback, index=values.index, dtype=float)


def load_dataset(
    root: Path,
    max_nonfocus_rows_per_field: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_path = root / "feature_cache" / "shard_inventory.csv"
    fields_path = root / "feature_cache" / "field_trajectory_metrics.csv"
    inventory = pd.read_csv(inventory_path)
    fields = pd.read_csv(fields_path)
    if inventory.empty or fields.empty:
        raise ValueError("Trajectory inventory and field metrics must be non-empty")
    needed = {
        "cohort",
        "branch",
        "image_id",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "combined_mask_id",
        "centroid_y",
        "centroid_x",
        "final_state",
        "rgb_state",
        "classification_confidence",
        "countable",
        "proxy_type",
        "temporal_track_confident",
        "temporal_support_frames",
        "temporal_match_confidence",
        "sentinel",
        "development_anchor",
        "holdout_anchor",
        "replicate_diagnostic",
        "treated",
        "ploidy",
        "cyclophosphamide",
        "doxorubicin_nm",
        "border_touching",
        "mask_area",
        "mask_solidity",
        "mask_circularity",
        "core_cytoplasm_area",
        "core_nucleus_to_cytoplasm",
        "extent_nucleus_to_cytoplasm",
        "red_mass_proxy",
        "cytoplasm_red_mass_proxy",
        "cell_dead_snr",
        "cell_dead_positive_fraction",
        "cell_dead_core_enrichment",
        "source_feature_path",
        "cell_mask_path",
        "nucleus_core_mask_path",
        "nucleus_extent_mask_path",
        "combined_raw_path",
        "dead_raw_path",
    }
    frames: list[pd.DataFrame] = []
    for seen, row in enumerate(inventory.itertuples(index=False), start=1):
        path = Path(row.shard_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = needed - header
        if missing:
            raise ValueError(f"Shard {path} missing fields: {sorted(missing)}")
        frame = pd.read_csv(path, usecols=sorted(needed))
        well = str(row.key).split("_", 1)[0]
        if (
            str(row.cohort) != "d0_frozen"
            and well not in {"E9", "F9"}
            and len(frame) > max_nonfocus_rows_per_field
        ):
            positive = frame["proxy_type"].isin(
                {"blue_supported_dead", "temporal_dead_remnant"}
            )
            anchors = frame.loc[positive]
            remainder = frame.loc[~positive]
            keep_remainder = max(
                0,
                max_nonfocus_rows_per_field - len(anchors),
            )
            if len(remainder) > keep_remainder:
                row_seed = (
                    seed
                    + sum(ord(character) for character in str(row.key))
                    + (0 if str(row.branch) == "original" else 1_000_003)
                )
                remainder = remainder.sample(
                    keep_remainder,
                    random_state=row_seed,
                )
            frame = pd.concat([anchors, remainder], ignore_index=True)
        frames.append(frame)
        if seen == 1 or seen % 250 == 0 or seen == len(inventory):
            print(f"loaded_shards={seen}/{len(inventory)}", flush=True)
    data = pd.concat(frames, ignore_index=True)
    for column in (
        "countable",
        "sentinel",
        "development_anchor",
        "holdout_anchor",
        "replicate_diagnostic",
        "treated",
        "border_touching",
        "cyclophosphamide",
    ):
        data[column] = as_bool(data[column])
    for column in ("site", "elapsed_hours", *[value[0] for value in RAW_FEATURES.values()]):
        data[column] = finite_numeric(data[column])
    merge_columns = [
        "branch",
        "key",
        *FIELD_RATIO_COLUMNS,
        "field_countable_count",
        "field_current_dead_fraction",
        "field_median_core_nc",
    ]
    data = data.merge(
        fields[merge_columns].drop_duplicates(["branch", "key"]),
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    if data[list(FIELD_RATIO_COLUMNS)].isna().any().any():
        raise ValueError("Object rows are missing field-trajectory ratios")
    return data, fields


def assign_density(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = data.copy()
    field_rows = result[
        ["branch", "key", "cohort", "field_countable_count", "field_mask_fraction_ratio_to_peak"]
    ].drop_duplicates(["branch", "key"])
    field_rows["density_score"] = (
        np.log1p(finite_numeric(field_rows["field_countable_count"]))
        + 2.5 * finite_numeric(field_rows["field_mask_fraction_ratio_to_peak"])
    )
    definitions: dict[str, Any] = {}
    field_rows["density_bin"] = ""
    field_rows["density_percentile"] = np.nan
    field_rows["density_anchor"] = np.nan
    for branch, branch_rows in field_rows.groupby("branch"):
        reference = branch_rows.loc[branch_rows["cohort"].eq("d0_frozen"), "density_score"]
        if len(reference) < 50:
            raise ValueError(f"Insufficient d0 fields for density calibration: {branch}")
        q1, q2 = np.quantile(reference, [1 / 3, 2 / 3])
        definitions[str(branch)] = {
            "low_middle": float(q1),
            "middle_high": float(q2),
            "continuous_reference_count": int(len(reference)),
            "continuous_anchor_percentiles": list(DENSITY_ANCHORS),
        }
        selector = field_rows["branch"].eq(branch)
        density_percentiles = empirical_density_percentile(
            reference.to_numpy(float),
            field_rows.loc[selector, "density_score"].to_numpy(float),
        )
        field_rows.loc[selector, "density_percentile"] = density_percentiles
        field_rows.loc[selector, "density_anchor"] = [
            nearest_density_anchor(value) for value in density_percentiles
        ]
        field_rows.loc[selector, "density_bin"] = np.select(
            [
                field_rows.loc[selector, "density_score"] <= q1,
                field_rows.loc[selector, "density_score"] <= q2,
            ],
            ["low", "middle"],
            default="high",
        )
    result = result.merge(
        field_rows[
            [
                "branch",
                "key",
                "density_score",
                "density_percentile",
                "density_anchor",
                "density_bin",
            ]
        ],
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    return result, definitions


def empirical_percentile(
    reference: np.ndarray,
    values: np.ndarray,
    direction: str,
    *,
    reference_sorted: bool = False,
) -> np.ndarray:
    reference = reference[np.isfinite(reference)]
    if not reference_sorted:
        reference = np.sort(reference)
    if reference.size < 20:
        raise ValueError(f"Insufficient d0 live reference values: n={reference.size}")
    observed = np.nan_to_num(
        values,
        nan=float(np.median(reference)),
        posinf=float(reference[-1]),
        neginf=float(reference[0]),
    )
    percentile = np.searchsorted(reference, observed, side="right") / reference.size
    percentile = np.clip(percentile, 0.0, 1.0)
    return 1.0 - percentile if direction == "low" else percentile


def empirical_density_percentile(
    reference: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Map density onto a continuous d0 reference percentile."""
    reference = np.sort(reference[np.isfinite(reference)])
    if reference.size < 20:
        raise ValueError(
            f"Insufficient d0 density reference values: n={reference.size}"
        )
    observed = np.nan_to_num(
        values,
        nan=float(np.median(reference)),
        posinf=float(reference[-1]),
        neginf=float(reference[0]),
    )
    result = np.searchsorted(reference, observed, side="right") / reference.size
    return np.clip(result, 0.0, 1.0)


def nearest_density_anchor(value: float) -> float:
    return min(DENSITY_ANCHORS, key=lambda anchor: abs(anchor - float(value)))


def density_anchor_weights(value: float) -> tuple[tuple[float, float], ...]:
    """Return adjacent anchor weights for continuous density interpolation."""
    density = float(np.clip(value, 0.0, 1.0))
    anchors = np.asarray(DENSITY_ANCHORS, dtype=float)
    if density <= anchors[0]:
        return ((float(anchors[0]), 1.0),)
    if density >= anchors[-1]:
        return ((float(anchors[-1]), 1.0),)
    upper_index = int(np.searchsorted(anchors, density, side="right"))
    lower = float(anchors[upper_index - 1])
    upper = float(anchors[upper_index])
    upper_weight = (density - lower) / (upper - lower)
    return ((lower, 1.0 - upper_weight), (upper, upper_weight))


def reference_time_group(
    cohort: str,
    elapsed_hours: float,
    untreated_live: bool,
) -> str:
    if str(cohort) == "d0_frozen" or float(elapsed_hours) <= 0:
        return "d0"
    if not untreated_live:
        return "d0"
    hours = float(elapsed_hours)
    if hours <= 48:
        return "untreated_24_48"
    if hours <= 72:
        return "untreated_50_72"
    return "untreated_74_96"


def target_time_group(elapsed_hours: float) -> str:
    hours = float(elapsed_hours)
    if hours <= 24:
        return "d0"
    if hours <= 48:
        return "untreated_24_48"
    if hours <= 72:
        return "untreated_50_72"
    return "untreated_74_96"


def apply_empirical_feature_calibration(
    data: pd.DataFrame,
    arrays: dict[tuple[str, float, str, str], np.ndarray],
    *,
    error_context: str,
) -> pd.DataFrame:
    """Apply the fixed density interpolation without per-field frame writes."""
    result = data.copy()
    hours = finite_numeric(result["elapsed_hours"]).to_numpy(float)
    result["_calibration_time_group"] = np.select(
        [hours <= 24.0, hours <= 48.0, hours <= 72.0],
        ["d0", "untreated_24_48", "untreated_50_72"],
        default="untreated_74_96",
    )
    anchors = np.asarray(DENSITY_ANCHORS, dtype=float)
    for (branch, time_group), indices in result.groupby(
        ["branch", "_calibration_time_group"],
        sort=False,
    ).groups.items():
        density = np.clip(
            finite_numeric(result.loc[indices, "density_percentile"]).to_numpy(
                float
            ),
            0.0,
            1.0,
        )
        upper_index = np.searchsorted(anchors, density, side="right")
        lower_index = np.clip(upper_index - 1, 0, len(anchors) - 1)
        upper_index = np.clip(upper_index, 0, len(anchors) - 1)
        lower_anchor = anchors[lower_index]
        upper_anchor = anchors[upper_index]
        same_anchor = lower_index == upper_index
        upper_weight = np.zeros(len(indices), dtype=float)
        interpolated = ~same_anchor
        upper_weight[interpolated] = (
            density[interpolated] - lower_anchor[interpolated]
        ) / (
            upper_anchor[interpolated] - lower_anchor[interpolated]
        )
        lower_weight = 1.0 - upper_weight
        for output, (raw, direction) in RAW_FEATURES.items():
            target = pd.to_numeric(
                result.loc[indices, raw],
                errors="coerce",
            ).to_numpy(float)
            calibrated = np.zeros(len(indices), dtype=float)
            used_weight = np.zeros(len(indices), dtype=float)
            for anchor in anchors:
                weights = (
                    np.where(lower_anchor == anchor, lower_weight, 0.0)
                    + np.where(upper_anchor == anchor, upper_weight, 0.0)
                )
                active = weights > 0.0
                if not active.any():
                    continue
                reference = arrays.get(
                    (str(branch), float(anchor), str(time_group), raw)
                )
                if reference is None:
                    reference = arrays.get(
                        (str(branch), float(anchor), "d0", raw)
                    )
                if reference is None:
                    continue
                calibrated[active] += weights[active] * empirical_percentile(
                    reference,
                    target[active],
                    direction,
                    reference_sorted=True,
                )
                used_weight[active] += weights[active]
            if np.any(used_weight <= 0.0):
                missing_count = int((used_weight <= 0.0).sum())
                raise ValueError(
                    f"Missing continuous reference for {error_context}: "
                    f"branch={branch}, time_group={time_group}, "
                    f"raw_feature={raw}, rows={missing_count}"
                )
            result.loc[indices, output] = calibrated / used_weight
    result = result.drop(columns="_calibration_time_group")
    if result[list(MODEL_FEATURES)].isna().any().any():
        raise ValueError(
            f"{error_context} calibrated object features contain missing values"
        )
    return result


def calibrate_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = data.copy()
    references: dict[str, Any] = {}
    eligible = result["countable"] & ~result["border_touching"]
    d0_live = result["cohort"].eq("d0_frozen") & result["proxy_type"].eq(
        "d0_live_anchor"
    )
    untreated_live = result["proxy_type"].eq("untreated_live_anchor")
    reference_rows = result.loc[eligible & (d0_live | untreated_live)].copy()
    reference_rows["reference_time_group"] = [
        reference_time_group(
            cohort,
            elapsed,
            proxy == "untreated_live_anchor",
        )
        for cohort, elapsed, proxy in zip(
            reference_rows["cohort"],
            finite_numeric(reference_rows["elapsed_hours"]),
            reference_rows["proxy_type"],
        )
    ]
    arrays: dict[tuple[str, float, str, str], np.ndarray] = {}
    for (branch, anchor, time_group), rows in reference_rows.groupby(
        ["branch", "density_anchor", "reference_time_group"],
        sort=False,
    ):
        reference_key = f"{branch}:{float(anchor):.2f}:{time_group}"
        references[reference_key] = {}
        for output, (raw, direction) in RAW_FEATURES.items():
            values = pd.to_numeric(rows[raw], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size < 20:
                continue
            arrays[(str(branch), float(anchor), str(time_group), raw)] = np.sort(
                values
            )
            references[reference_key][output] = {
                "raw_feature": raw,
                "direction": direction,
                "count": int(len(values)),
                "quantiles": {
                    str(q): float(np.quantile(values, q))
                    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
                },
            }
    result = apply_empirical_feature_calibration(
        result,
        arrays,
        error_context="calibration",
    )
    return result, references


def field_parameter_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for values in product(
        (0.60, 0.70, 0.80),
        (0.25, 0.35, 0.45),
        (0.25, 0.35, 0.45),
        (0.45, 0.60, 0.75),
        (3, 4),
        (0.50, 0.75),
        (2, 3),
    ):
        configs.append(
            dict(
                zip(
                    (
                        "count_ratio_max",
                        "area_ratio_max",
                        "red_mass_ratio_max",
                        "mask_fraction_ratio_max",
                        "minimum_field_signals",
                        "minimum_site_fraction",
                        "persistence_frames",
                    ),
                    values,
                )
            )
        )
        configs[-1]["recovery_frames"] = 3
        configs[-1]["recovery_signal_max"] = 1
    return configs


def apply_field_configuration(
    fields: pd.DataFrame,
    config: dict[str, Any],
    late_min_hours: float,
) -> pd.DataFrame:
    result = fields.copy().sort_values(
        ["branch", "well", "site", "elapsed_hours", "key"]
    )
    ratio_values = {
        column: finite_numeric(result[column], fallback=1.0)
        for column in FIELD_RATIO_COLUMNS
    }
    signals = np.column_stack(
        (
            ratio_values["field_count_ratio_to_peak"] <= config["count_ratio_max"],
            ratio_values["field_area_ratio_to_peak"] <= config["area_ratio_max"],
            ratio_values["field_cytoplasm_ratio_to_peak"] <= config["area_ratio_max"],
            ratio_values["field_red_mass_ratio_to_peak"]
            <= config["red_mass_ratio_max"],
            ratio_values["field_mask_fraction_ratio_to_peak"]
            <= config["mask_fraction_ratio_max"],
        )
    ).sum(axis=1)
    result["field_collapse_signal_count"] = signals
    late = finite_numeric(result["elapsed_hours"]).ge(late_min_hours)
    result["field_collapse_preliminary"] = (
        late & (signals >= int(config["minimum_field_signals"]))
    )
    concordance = result.groupby(
        ["branch", "well", "elapsed_hours"],
        sort=False,
    )["field_collapse_preliminary"].transform("mean")
    result["field_site_concordance"] = concordance
    eligible = late & (
        result["field_collapse_preliminary"]
        | (
            concordance.ge(float(config["minimum_site_fraction"]))
            & (signals >= int(config["minimum_field_signals"]) - 1)
        )
    )
    persistence = int(config["persistence_frames"])
    result["field_global_late_death"] = False
    result["field_state_transition"] = "inactive"
    for _group, indices in result.groupby(["branch", "well", "site"], sort=False).groups.items():
        ordered = result.loc[indices].sort_values("elapsed_hours")
        active = False
        entry_streak = 0
        recovery_streak = 0
        states: list[bool] = []
        transitions: list[str] = []
        for row_index in ordered.index:
            qualifies = bool(eligible.loc[row_index])
            signal_count = int(signals[result.index.get_loc(row_index)])
            if not active:
                entry_streak = entry_streak + 1 if qualifies else 0
                if entry_streak >= persistence:
                    active = True
                    recovery_streak = 0
                    transitions.append("entered")
                else:
                    transitions.append("inactive")
            else:
                recovered = (
                    signal_count <= int(config.get("recovery_signal_max", 1))
                    and float(result.loc[row_index, "field_site_concordance"])
                    < float(config["minimum_site_fraction"])
                )
                recovery_streak = recovery_streak + 1 if recovered else 0
                if recovery_streak >= int(config.get("recovery_frames", 3)):
                    active = False
                    entry_streak = 0
                    transitions.append("recovered")
                else:
                    transitions.append("active")
            states.append(active)
        result.loc[ordered.index, "field_global_late_death"] = states
        result.loc[ordered.index, "field_state_transition"] = transitions
    return result


def apply_branch_field_consensus(fields: pd.DataFrame) -> pd.DataFrame:
    """Create one shared field gate while preserving branch diagnostics."""
    required = {"branch", "key", "field_global_late_death"}
    missing = required - set(fields.columns)
    if missing:
        raise ValueError(f"Field consensus missing columns: {sorted(missing)}")
    result = fields.copy()
    state = (
        result[["branch", "key", "field_global_late_death"]]
        .drop_duplicates(["branch", "key"])
        .pivot(index="key", columns="branch", values="field_global_late_death")
    )
    needed = {"original", "nucleated_only"}
    if not needed <= set(state.columns):
        raise ValueError(
            f"Field consensus requires both branches; found {sorted(state.columns)}"
        )
    state = state[list(sorted(needed))].astype(bool)
    state["field_branch_raw_discordant"] = (
        state["original"] != state["nucleated_only"]
    )
    state["field_branch_consensus_late_death"] = (
        state["original"] & state["nucleated_only"]
    )
    result = result.merge(
        state[
            [
                "field_branch_raw_discordant",
                "field_branch_consensus_late_death",
            ]
        ].reset_index(),
        on="key",
        how="left",
        validate="many_to_one",
    )
    result["field_branch_raw_global"] = result[
        "field_global_late_death"
    ].astype(bool)
    result["field_global_late_death"] = result[
        "field_branch_consensus_late_death"
    ].astype(bool)
    return result


def object_parameter_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for feature_threshold, minimum_death_signals, healthy_threshold, healthy_signals in product(
        (0.60, 0.70, 0.80),
        (2, 3),
        (0.45, 0.55, 0.65),
        (3, 4),
    ):
        configs.append(
            {
                "feature_threshold": feature_threshold,
                "minimum_death_signals": minimum_death_signals,
                "healthy_threshold": healthy_threshold,
                "minimum_healthy_signals": healthy_signals,
                "branch_match_distance_px": 10.0,
                "unmatched_minimum_death_signals": 3,
                "temporal_minimum_support_frames": 2,
            }
        )
    for feature_threshold, healthy_gap in product(
        (0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57),
        (0.05, 0.06),
    ):
        candidate = {
            "feature_threshold": feature_threshold,
            "minimum_death_signals": 2,
            "healthy_threshold": round(
                feature_threshold - healthy_gap,
                2,
            ),
            "minimum_healthy_signals": 3,
            "branch_match_distance_px": 10.0,
            "unmatched_minimum_death_signals": 3,
            "temporal_minimum_support_frames": 2,
        }
        if candidate not in configs:
            configs.append(candidate)
    return configs


def object_evidence(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    nc = np.maximum(
        data["core_nc_percentile"].to_numpy(float),
        data["extent_nc_percentile"].to_numpy(float),
    )
    area = np.maximum(
        data["area_depletion_percentile"].to_numpy(float),
        data["cytoplasm_depletion_percentile"].to_numpy(float),
    )
    red_mass = np.maximum(
        data["red_mass_depletion_percentile"].to_numpy(float),
        data["cytoplasm_red_mass_depletion_percentile"].to_numpy(float),
    )
    shape = np.maximum(
        data["circularity_percentile"].to_numpy(float),
        data["solidity_percentile"].to_numpy(float),
    )
    dead_channel = np.maximum.reduce(
        (
            data["cell_dead_snr_percentile"].to_numpy(float),
            data["cell_dead_positive_fraction_percentile"].to_numpy(float),
            data["cell_dead_core_enrichment_percentile"].to_numpy(float),
        )
    )
    feature_threshold = float(config["feature_threshold"])
    death_matrix = np.column_stack(
        (
            nc >= feature_threshold,
            area >= feature_threshold,
            red_mass >= feature_threshold,
            shape >= feature_threshold,
            dead_channel >= feature_threshold,
        )
    )
    healthy_threshold = float(config["healthy_threshold"])
    healthy_matrix = np.column_stack(
        (
            nc <= healthy_threshold,
            area <= healthy_threshold,
            red_mass <= healthy_threshold,
            shape <= healthy_threshold,
            dead_channel <= healthy_threshold,
        )
    )
    result["death_signal_count"] = death_matrix.sum(axis=1)
    result["healthy_signal_count"] = healthy_matrix.sum(axis=1)
    result["death_score"] = (
        0.24 * nc
        + 0.22 * area
        + 0.26 * red_mass
        + 0.08 * shape
        + 0.20 * dead_channel
    )
    result["strong_live_evidence"] = (
        result["healthy_signal_count"] >= int(config["minimum_healthy_signals"])
    )
    result["late_dead_object_evidence"] = (
        result["death_signal_count"] >= int(config["minimum_death_signals"])
    ) & ~result["strong_live_evidence"]
    return result


def attach_branch_object_consensus(
    work: pd.DataFrame,
    match_distance_px: float,
) -> pd.DataFrame:
    """Match the two immutable segmentation views without changing either mask."""
    result = work.copy()
    result["branch_partner_matched"] = False
    result["branch_partner_distance"] = np.inf
    result["branch_partner_current_dead"] = False
    result["branch_partner_object_evidence"] = False
    result["branch_partner_strong_live"] = False
    result["branch_partner_temporal_remnant"] = False
    result["branch_partner_death_signal_count"] = 0
    result["branch_object_evidence_agree"] = False
    cached_columns = {
        "_branch_row_id",
        "_branch_partner_row_id",
        "_branch_partner_distance",
        "_branch_match_distance_px",
    }
    if cached_columns <= set(result.columns):
        cached_distance = finite_numeric(
            result["_branch_match_distance_px"],
            fallback=float(match_distance_px),
        )
        if not np.allclose(
            cached_distance.to_numpy(float),
            float(match_distance_px),
        ):
            raise ValueError(
                "Cached branch-object matches use a different distance threshold"
            )
        partner_lookup = result.set_index("_branch_row_id", verify_integrity=True)
        valid = finite_numeric(
            result["_branch_partner_row_id"],
            fallback=-1.0,
        ).ge(0)
        partner_ids = result.loc[valid, "_branch_partner_row_id"].astype(int)
        missing_partner_ids = set(partner_ids) - set(partner_lookup.index)
        if missing_partner_ids:
            raise ValueError(
                "Cached branch-object matches reference unknown row ids: "
                f"{sorted(missing_partner_ids)[:10]}"
            )
        result.loc[valid, "branch_partner_matched"] = True
        result.loc[valid, "branch_partner_distance"] = finite_numeric(
            result.loc[valid, "_branch_partner_distance"],
            fallback=np.inf,
        ).to_numpy(float)
        partner_fields = {
            "branch_partner_current_dead": "current_dead_call",
            "branch_partner_object_evidence": "late_dead_object_evidence",
            "branch_partner_strong_live": "strong_live_evidence",
            "branch_partner_death_signal_count": "death_signal_count",
        }
        for output, source in partner_fields.items():
            result.loc[valid, output] = partner_ids.map(
                partner_lookup[source]
            ).to_numpy()
        result.loc[valid, "branch_partner_temporal_remnant"] = (
            partner_ids.map(partner_lookup["proxy_type"])
            .astype(str)
            .eq("temporal_dead_remnant")
            .to_numpy()
        )
        own_evidence = result.loc[
            valid,
            "late_dead_object_evidence",
        ].astype(bool)
        partner_evidence = result.loc[
            valid,
            "branch_partner_object_evidence",
        ].astype(bool)
        result.loc[valid, "branch_object_evidence_agree"] = (
            own_evidence.to_numpy() == partner_evidence.to_numpy()
        )
        return result
    for (_cohort, _key), indices in result.groupby(
        ["cohort", "key"],
        sort=False,
    ).groups.items():
        group = result.loc[indices]
        original = group.loc[group["branch"].eq("original")]
        nucleated = group.loc[group["branch"].eq("nucleated_only")]
        if original.empty or nucleated.empty:
            continue
        original_xy = original[["centroid_y", "centroid_x"]].to_numpy(float)
        nucleated_xy = nucleated[["centroid_y", "centroid_x"]].to_numpy(float)
        original_distances, original_to_nucleated = cKDTree(
            nucleated_xy
        ).query(original_xy, k=1)
        _nucleated_distances, nucleated_to_original = cKDTree(
            original_xy
        ).query(nucleated_xy, k=1)
        for original_position, nucleated_position in enumerate(
            original_to_nucleated
        ):
            if nucleated_to_original[nucleated_position] != original_position:
                continue
            distance = float(original_distances[original_position])
            if distance > float(match_distance_px):
                continue
            original_index = original.index[original_position]
            nucleated_index = nucleated.index[nucleated_position]
            for own_index, partner_index in (
                (original_index, nucleated_index),
                (nucleated_index, original_index),
            ):
                result.loc[own_index, "branch_partner_matched"] = True
                result.loc[own_index, "branch_partner_distance"] = distance
                result.loc[own_index, "branch_partner_current_dead"] = bool(
                    result.loc[partner_index, "current_dead_call"]
                )
                result.loc[
                    own_index, "branch_partner_object_evidence"
                ] = bool(result.loc[partner_index, "late_dead_object_evidence"])
                result.loc[own_index, "branch_partner_strong_live"] = bool(
                    result.loc[partner_index, "strong_live_evidence"]
                )
                result.loc[
                    own_index, "branch_partner_temporal_remnant"
                ] = str(result.loc[partner_index, "proxy_type"]) == (
                    "temporal_dead_remnant"
                )
                result.loc[
                    own_index, "branch_partner_death_signal_count"
                ] = int(result.loc[partner_index, "death_signal_count"])
                result.loc[
                    own_index, "branch_object_evidence_agree"
                ] = bool(
                    result.loc[own_index, "late_dead_object_evidence"]
                    == result.loc[partner_index, "late_dead_object_evidence"]
                )
    return result


def precompute_branch_object_matches(
    data: pd.DataFrame,
    match_distance_px: float,
) -> pd.DataFrame:
    """Cache mutual-nearest dual-view matches for parameter-grid reuse."""
    required = {"cohort", "key", "branch", "centroid_y", "centroid_x"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Branch-object match cache missing columns: {sorted(missing)}"
        )
    result = data.copy()
    result["_branch_row_id"] = np.arange(len(result), dtype=np.int64)
    result["_branch_partner_row_id"] = -1
    result["_branch_partner_distance"] = np.inf
    result["_branch_match_distance_px"] = float(match_distance_px)
    for (_cohort, _key), indices in result.groupby(
        ["cohort", "key"],
        sort=False,
    ).groups.items():
        group = result.loc[indices]
        original = group.loc[group["branch"].eq("original")]
        nucleated = group.loc[group["branch"].eq("nucleated_only")]
        if original.empty or nucleated.empty:
            continue
        original_xy = original[["centroid_y", "centroid_x"]].to_numpy(float)
        nucleated_xy = nucleated[["centroid_y", "centroid_x"]].to_numpy(float)
        original_distances, original_to_nucleated = cKDTree(
            nucleated_xy
        ).query(original_xy, k=1)
        _nucleated_distances, nucleated_to_original = cKDTree(
            original_xy
        ).query(nucleated_xy, k=1)
        original_positions = np.arange(len(original), dtype=np.int64)
        mutual = (
            nucleated_to_original[original_to_nucleated]
            == original_positions
        )
        within_distance = original_distances <= float(match_distance_px)
        valid_original_positions = original_positions[
            mutual & within_distance
        ]
        if valid_original_positions.size == 0:
            continue
        valid_nucleated_positions = original_to_nucleated[
            valid_original_positions
        ]
        original_indices = original.index.to_numpy()[
            valid_original_positions
        ]
        nucleated_indices = nucleated.index.to_numpy()[
            valid_nucleated_positions
        ]
        original_row_ids = result.loc[
            original_indices,
            "_branch_row_id",
        ].to_numpy(np.int64)
        nucleated_row_ids = result.loc[
            nucleated_indices,
            "_branch_row_id",
        ].to_numpy(np.int64)
        distances = original_distances[valid_original_positions]
        result.loc[
            original_indices,
            "_branch_partner_row_id",
        ] = nucleated_row_ids
        result.loc[
            original_indices,
            "_branch_partner_distance",
        ] = distances
        result.loc[
            nucleated_indices,
            "_branch_partner_row_id",
        ] = original_row_ids
        result.loc[
            nucleated_indices,
            "_branch_partner_distance",
        ] = distances
    return result


def mapped_partner_values(
    work: pd.DataFrame,
    column: str,
    *,
    default: Any,
) -> pd.Series:
    """Map a current-row value from each cached mutual-nearest partner."""
    result = pd.Series(default, index=work.index)
    needed = {"_branch_row_id", "_branch_partner_row_id", column}
    if not needed <= set(work.columns):
        return result
    lookup = work.set_index("_branch_row_id", verify_integrity=True)[column]
    valid = finite_numeric(
        work["_branch_partner_row_id"],
        fallback=-1.0,
    ).ge(0)
    partner_ids = work.loc[valid, "_branch_partner_row_id"].astype(int)
    result.loc[valid] = partner_ids.map(lookup).to_numpy()
    return result


def classification_calls(
    data: pd.DataFrame,
    fields: pd.DataFrame,
    object_config: dict[str, Any],
    late_min_hours: float,
    apply_treatment_scope: bool,
) -> pd.DataFrame:
    field_columns = [
        "branch",
        "key",
        "field_global_late_death",
        "field_collapse_signal_count",
        "field_site_concordance",
    ]
    for optional in (
        "field_branch_raw_discordant",
        "field_branch_raw_global",
        "field_branch_consensus_late_death",
        "field_state_transition",
    ):
        if optional in fields:
            field_columns.append(optional)
    field_state = fields[field_columns].drop_duplicates(["branch", "key"])
    work = data.merge(
        field_state,
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    if "field_branch_raw_discordant" not in work:
        work["field_branch_raw_discordant"] = False
    evidence = object_evidence(work, object_config)
    for column in evidence:
        work[column] = evidence[column]
    current_dead = work["final_state"].astype(str).eq("dead")
    work["current_dead_call"] = current_dead
    has_both_branches = {"original", "nucleated_only"} <= set(
        work["branch"].astype(str)
    )
    if "_branch_partner_row_id" not in work and has_both_branches:
        work = precompute_branch_object_matches(
            work,
            float(object_config.get("branch_match_distance_px", 10.0)),
        )
    work = attach_branch_object_consensus(
        work,
        float(object_config.get("branch_match_distance_px", 10.0)),
    )
    temporal = work["proxy_type"].eq("temporal_dead_remnant")
    temporal_track_confident = (
        as_bool(work["temporal_track_confident"])
        if "temporal_track_confident" in work
        else pd.Series(False, index=work.index)
    )
    temporal_support = (
        finite_numeric(work["temporal_support_frames"])
        if "temporal_support_frames" in work
        else pd.Series(0.0, index=work.index)
    )
    eligible = (
        work["cohort"].eq("trajectory")
        & finite_numeric(work["elapsed_hours"]).ge(late_min_hours)
        & work["countable"]
        & ~work["border_touching"]
        & work["final_state"].astype(str).eq("live")
    )
    if apply_treatment_scope:
        eligible &= work["treated"]
    matched_consensus = (
        work["branch_partner_matched"]
        & work["branch_partner_object_evidence"]
        & work["late_dead_object_evidence"]
    )
    unmatched_strong = (
        ~work["branch_partner_matched"]
        & (
            work["death_signal_count"]
            >= int(object_config.get("unmatched_minimum_death_signals", 3))
        )
    )
    global_rescue = (
        eligible
        & work["field_global_late_death"].fillna(False)
        & work["late_dead_object_evidence"]
        & ~work["strong_live_evidence"]
        & ~work["branch_partner_strong_live"]
        & (matched_consensus | unmatched_strong)
    )
    temporal_consensus = (
        (
            work["branch_partner_matched"]
            & (
                work["branch_partner_temporal_remnant"]
                | work["branch_partner_object_evidence"]
            )
        )
        | (
            ~work["branch_partner_matched"]
            & (
                work["death_signal_count"]
                >= int(object_config.get("unmatched_minimum_death_signals", 3))
            )
        )
    )
    temporal_rescue = (
        eligible
        & temporal
        & temporal_track_confident
        & (
            temporal_support
            >= int(object_config.get("temporal_minimum_support_frames", 2))
        )
        & ~work["strong_live_evidence"]
        & ~work["branch_partner_strong_live"]
        & temporal_consensus
    )
    work["temporal_carryforward_call"] = temporal_rescue
    work["global_late_death_rescue_call"] = global_rescue
    partner_temporal_rescue = mapped_partner_values(
        work,
        "temporal_carryforward_call",
        default=False,
    ).astype(bool)
    partner_global_rescue = mapped_partner_values(
        work,
        "global_late_death_rescue_call",
        default=False,
    ).astype(bool)
    matched = work["branch_partner_matched"].astype(bool)
    work["temporal_carryforward_call"] = (
        work["temporal_carryforward_call"].astype(bool)
        | (matched & partner_temporal_rescue)
    )
    work["global_late_death_rescue_call"] = (
        work["global_late_death_rescue_call"].astype(bool)
        | (matched & partner_global_rescue)
    )
    work["late_death_rescue_call"] = (
        work["temporal_carryforward_call"]
        | work["global_late_death_rescue_call"]
    )
    work["final_dead_call"] = current_dead | work["late_death_rescue_call"]
    work["branch_partner_late_death_rescue_call"] = mapped_partner_values(
        work,
        "late_death_rescue_call",
        default=False,
    ).astype(bool)
    work["branch_partner_final_dead_call"] = mapped_partner_values(
        work,
        "final_dead_call",
        default=False,
    ).astype(bool)
    work["branch_late_death_rescue_call_agree"] = (
        matched
        & (
            work["late_death_rescue_call"].astype(bool)
            == work["branch_partner_late_death_rescue_call"].astype(bool)
        )
    )
    work["branch_final_dead_call_agree"] = (
        matched
        & (
            work["final_dead_call"].astype(bool)
            == work["branch_partner_final_dead_call"].astype(bool)
        )
    )
    work["branch_final_call_discordant"] = (
        matched & ~work["branch_final_dead_call_agree"]
    )
    work["branch_discordant_uncertain"] = (
        eligible
        & (
            work["field_branch_raw_discordant"].fillna(False)
            | (
                work["branch_partner_matched"]
                & ~work["branch_object_evidence_agree"]
            )
        )
        & ~work["late_death_rescue_call"]
    )
    work["track_uncertain"] = (
        eligible
        & temporal
        & ~work["late_death_rescue_call"]
    )
    work["field_evidence_uncertain"] = (
        eligible
        & work["field_global_late_death"].fillna(False)
        & work["late_dead_object_evidence"]
        & ~work["late_death_rescue_call"]
    )
    work["late_death_uncertain"] = (
        work["branch_discordant_uncertain"]
        | work["track_uncertain"]
        | work["field_evidence_uncertain"]
        | (
            work["branch_final_call_discordant"]
            & ~work["final_dead_call"].astype(bool)
        )
    )
    work["classification_tier"] = np.select(
        [
            current_dead,
            work["late_death_rescue_call"],
            work["late_death_uncertain"],
        ],
        ["confirmed_dead", "probable_dead", "uncertain"],
        default="confirmed_live",
    )
    return work


def countable_dead_fraction(rows: pd.DataFrame) -> float:
    eligible = rows["countable"] & ~rows["border_touching"] & ~rows["final_state"].eq("artifact")
    if not eligible.any():
        return float("nan")
    return float(rows.loc[eligible, "final_dead_call"].mean())


def optimize_field_state(
    fields: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trials: list[dict[str, Any]] = []
    configurations: dict[int, dict[str, Any]] = {}
    for index, config in enumerate(field_parameter_grid(), start=1):
        state = apply_field_configuration(fields, config, args.late_min_hours)
        state = apply_branch_field_consensus(state)
        dev = state.loc[
            state["key"].eq(E9_SENTINEL_KEY)
            & state["branch"].eq("original")
        ]
        if len(dev) != 1:
            raise ValueError(f"Expected one original E9 sentinel field, found {len(dev)}")
        untreated_late = state.loc[
            ~as_bool(state["treated"])
            & finite_numeric(state["elapsed_hours"]).ge(args.late_min_hours)
        ]
        treated_non_e9 = state.loc[
            as_bool(state["treated"])
            & ~state["well"].isin(["E9", "F9"])
            & finite_numeric(state["elapsed_hours"]).ge(args.late_min_hours)
        ]
        dev_active = bool(dev.iloc[0]["field_global_late_death"])
        untreated_activation = float(
            untreated_late["field_global_late_death"].mean()
        ) if len(untreated_late) else 0.0
        non_e9_activation = float(
            treated_non_e9["field_global_late_death"].mean()
        ) if len(treated_non_e9) else 0.0
        trials.append(
            {
                "trial": f"field_{index:04d}",
                "development_e9_active": dev_active,
                "untreated_late_field_activation_rate": untreated_activation,
                "treated_non_e9_field_activation_rate": non_e9_activation,
                "configuration": json.dumps(config, sort_keys=True),
            }
        )
        configurations[index] = config
    trial_frame = pd.DataFrame(trials)
    trial_frame["passes_development"] = (
        trial_frame["development_e9_active"]
        & trial_frame["untreated_late_field_activation_rate"].le(0.01)
    )
    trial_frame = trial_frame.sort_values(
        [
            "passes_development",
            "untreated_late_field_activation_rate",
            "treated_non_e9_field_activation_rate",
            "trial",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    best_trial = str(trial_frame.iloc[0]["trial"])
    best_index = int(best_trial.rsplit("_", 1)[1])
    best_config = configurations[best_index]
    best_state = apply_field_configuration(fields, best_config, args.late_min_hours)
    best_state = apply_branch_field_consensus(best_state)
    return best_config, best_state, trial_frame


def optimize_objects(
    data: pd.DataFrame,
    fields: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trials: list[dict[str, Any]] = []
    configurations: dict[int, dict[str, Any]] = {}
    parameter_grid = object_parameter_grid()
    match_distances = {
        float(config.get("branch_match_distance_px", 10.0))
        for config in parameter_grid
    }
    if len(match_distances) != 1:
        raise ValueError(
            "Object grid must use one branch-match distance for cached matching"
        )
    matched_data = precompute_branch_object_matches(
        data,
        match_distances.pop(),
    )
    for index, config in enumerate(parameter_grid, start=1):
        counterfactual = classification_calls(
            matched_data,
            fields,
            config,
            args.late_min_hours,
            apply_treatment_scope=False,
        )
        dev = counterfactual.loc[
            counterfactual["key"].eq(E9_SENTINEL_KEY)
            & counterfactual["branch"].eq("original")
        ]
        dev_fraction = countable_dead_fraction(dev)
        controls = counterfactual.loc[
            counterfactual["proxy_type"].eq("untreated_live_anchor")
        ]
        control_fpr = float(
            controls["late_death_rescue_call"].mean()
        ) if len(controls) else float("nan")
        new_calls_outside_e9 = int(
            counterfactual.loc[
                counterfactual["treated"]
                & ~counterfactual["well"].isin(["E9", "F9"]),
                "late_death_rescue_call",
            ].sum()
        )
        stability = perturbation_stability_metrics(
            counterfactual,
            config,
        )
        trials.append(
            {
                "trial": f"object_{index:04d}",
                "development_e9_dead_fraction": dev_fraction,
                "untreated_live_anchor_counterfactual_fpr": control_fpr,
                "new_calls_outside_e9_f9": new_calls_outside_e9,
                "object_evidence_stability_rate": stability[
                    "object_evidence_stability_rate"
                ],
                "object_evidence_flip_rate": stability[
                    "object_evidence_flip_rate"
                ],
                "configuration": json.dumps(config, sort_keys=True),
            }
        )
        configurations[index] = config
    trial_frame = pd.DataFrame(trials)
    trial_frame["passes_development"] = (
        trial_frame["development_e9_dead_fraction"].ge(
            args.target_e9_dead_fraction
        )
        & trial_frame["untreated_live_anchor_counterfactual_fpr"].lt(
            args.max_control_false_positive_rate
        )
    )
    trial_frame["passes_parameter_stability"] = trial_frame[
        "object_evidence_stability_rate"
    ].ge(0.995)
    trial_frame["passes_production_candidate"] = (
        trial_frame["passes_development"]
        & trial_frame["passes_parameter_stability"]
    )
    trial_frame = trial_frame.sort_values(
        [
            "passes_production_candidate",
            "passes_development",
            "object_evidence_stability_rate",
            "development_e9_dead_fraction",
            "untreated_live_anchor_counterfactual_fpr",
            "new_calls_outside_e9_f9",
            "trial",
        ],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    best_trial = str(trial_frame.iloc[0]["trial"])
    best_index = int(best_trial.rsplit("_", 1)[1])
    best_config = configurations[best_index]
    predictions = classification_calls(
        matched_data,
        fields,
        best_config,
        args.late_min_hours,
        apply_treatment_scope=True,
    )
    return best_config, predictions, trial_frame


def field_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = predictions.loc[
        predictions["countable"]
        & ~predictions["border_touching"]
        & ~predictions["final_state"].astype(str).eq("artifact")
    ].copy()
    return (
        eligible.groupby(
            [
                "branch",
                "key",
                "well",
                "site",
                "elapsed_hours",
                "development_anchor",
                "holdout_anchor",
                "replicate_diagnostic",
                "treated",
                "field_global_late_death",
                "field_branch_raw_global",
                "field_branch_raw_discordant",
                "field_branch_consensus_late_death",
            ],
            as_index=False,
        )
        .agg(
            countable_objects=("combined_mask_id", "size"),
            baseline_dead=("current_dead_call", "sum"),
            temporal_rescued=("temporal_carryforward_call", "sum"),
            global_rescued=("global_late_death_rescue_call", "sum"),
            total_rescued=("late_death_rescue_call", "sum"),
            final_dead=("final_dead_call", "sum"),
            uncertain=("late_death_uncertain", "sum"),
            branch_discordant_uncertain=("branch_discordant_uncertain", "sum"),
            track_uncertain=("track_uncertain", "sum"),
        )
        .assign(
            baseline_dead_fraction=lambda frame: frame["baseline_dead"]
            / frame["countable_objects"],
            final_dead_fraction=lambda frame: frame["final_dead"]
            / frame["countable_objects"],
        )
    )


def anchor_metrics(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    object_config: dict[str, Any],
) -> dict[str, Any]:
    e9_dev = summary.loc[
        summary["key"].eq(E9_SENTINEL_KEY) & summary["branch"].eq("original")
    ]
    e9_holdout = summary.loc[
        summary["well"].eq("E9")
        & summary["holdout_anchor"]
        & finite_numeric(summary["elapsed_hours"]).eq(120.0)
    ]
    f9 = summary.loc[
        summary["well"].eq("F9")
        & finite_numeric(summary["elapsed_hours"]).eq(120.0)
    ]
    blue = predictions["proxy_type"].eq("blue_supported_dead")
    temporal = (
        predictions["proxy_type"].eq("temporal_dead_remnant")
        & predictions["cohort"].eq("trajectory")
        & finite_numeric(predictions["elapsed_hours"]).ge(args.late_min_hours)
        & predictions["treated"]
        & predictions["countable"]
        & ~predictions["border_touching"]
        & predictions["final_state"].astype(str).eq("live")
        & as_bool(predictions["temporal_track_confident"])
        & finite_numeric(predictions["temporal_support_frames"]).ge(
            int(object_config.get("temporal_minimum_support_frames", 2))
        )
    )
    d0_changed = predictions.loc[
        predictions["cohort"].eq("d0_frozen"),
        "late_death_rescue_call",
    ]
    metrics = {
        "target_e9_dead_fraction": float(args.target_e9_dead_fraction),
        "target_holdout_dead_fraction": float(
            args.target_holdout_dead_fraction
        ),
        "development_e9_original_day5_dead_fraction": float(
            e9_dev.iloc[0]["final_dead_fraction"]
        ),
        "development_e9_original_day5_baseline_dead_fraction": float(
            e9_dev.iloc[0]["baseline_dead_fraction"]
        ),
        "holdout_e9_day5_field_count": int(len(e9_holdout)),
        "holdout_e9_day5_min_dead_fraction": float(
            e9_holdout["final_dead_fraction"].min()
        ) if len(e9_holdout) else float("nan"),
        "holdout_e9_day5_median_dead_fraction": float(
            e9_holdout["final_dead_fraction"].median()
        ) if len(e9_holdout) else float("nan"),
        "f9_day5_field_count": int(len(f9)),
        "f9_day5_min_dead_fraction": float(
            f9["final_dead_fraction"].min()
        ) if len(f9) else float("nan"),
        "f9_day5_median_dead_fraction": float(
            f9["final_dead_fraction"].median()
        ) if len(f9) else float("nan"),
        "blue_supported_anchor_preservation": float(
            predictions.loc[blue, "final_dead_call"].mean()
        ) if blue.any() else float("nan"),
        "temporal_remnant_anchor_recall": float(
            predictions.loc[temporal, "final_dead_call"].mean()
        ) if temporal.any() else float("nan"),
        "temporal_remnant_anchor_count": int(temporal.sum()),
        "changed_d0_object_count": int(d0_changed.sum()),
    }
    metrics["development_pass"] = (
        metrics["development_e9_original_day5_dead_fraction"]
        >= args.target_e9_dead_fraction
    )
    metrics["holdout_pass"] = (
        math.isfinite(metrics["holdout_e9_day5_min_dead_fraction"])
        and metrics["holdout_e9_day5_min_dead_fraction"]
        >= args.target_holdout_dead_fraction
    )
    metrics["d0_pass"] = metrics["changed_d0_object_count"] == 0
    return metrics


def branch_consensus_metrics(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    matched = predictions["branch_partner_matched"].astype(bool)
    matched_agreement = (
        float(predictions.loc[matched, "branch_object_evidence_agree"].mean())
        if matched.any()
        else float("nan")
    )
    matched_original = matched & predictions["branch"].eq("original")
    matched_rescue_agreement = (
        float(
            predictions.loc[
                matched_original,
                "branch_late_death_rescue_call_agree",
            ].mean()
        )
        if matched_original.any()
        else float("nan")
    )
    matched_final_agreement = (
        float(
            predictions.loc[
                matched_original,
                "branch_final_dead_call_agree",
            ].mean()
        )
        if matched_original.any()
        else float("nan")
    )
    field_state = (
        summary[
            [
                "branch",
                "key",
                "field_global_late_death",
                "countable_objects",
                "total_rescued",
                "final_dead_fraction",
            ]
        ]
        .drop_duplicates(["branch", "key"])
        .pivot(index="key", columns="branch")
    )
    state_original = field_state[("field_global_late_death", "original")].astype(
        bool
    )
    state_nucleated = field_state[
        ("field_global_late_death", "nucleated_only")
    ].astype(bool)
    dead_original = field_state[("final_dead_fraction", "original")].astype(float)
    dead_nucleated = field_state[
        ("final_dead_fraction", "nucleated_only")
    ].astype(float)
    final_difference = (dead_original - dead_nucleated).abs()
    rescue_original = (
        field_state[("total_rescued", "original")].astype(float)
        / field_state[("countable_objects", "original")]
        .astype(float)
        .replace(0, np.nan)
    ).fillna(0.0)
    rescue_nucleated = (
        field_state[("total_rescued", "nucleated_only")].astype(float)
        / field_state[("countable_objects", "nucleated_only")]
        .astype(float)
        .replace(0, np.nan)
    ).fillna(0.0)
    rescue_difference = (rescue_original - rescue_nucleated).abs()
    raw_fields = summary[
        ["branch", "key", "field_branch_raw_discordant"]
    ].drop_duplicates(["branch", "key"])
    return {
        "matched_object_count": int(matched.sum()),
        "matched_object_evidence_agreement": matched_agreement,
        "matched_pair_count": int(matched_original.sum()),
        "matched_pair_rescue_call_agreement": matched_rescue_agreement,
        "matched_pair_final_dead_call_agreement": matched_final_agreement,
        "final_field_state_mismatch_count": int(
            (state_original != state_nucleated).sum()
        ),
        "final_field_state_mismatch_rate": float(
            (state_original != state_nucleated).mean()
        ),
        "raw_field_discordance_rate": float(
            raw_fields["field_branch_raw_discordant"].astype(bool).mean()
        ),
        "field_rescue_fraction_abs_diff_median": float(
            rescue_difference.median()
        ),
        "field_rescue_fraction_abs_diff_gt_0p10_count": int(
            rescue_difference.gt(0.10).sum()
        ),
        "field_rescue_fraction_abs_diff_gt_0p10_rate": float(
            rescue_difference.gt(0.10).mean()
        ),
        "field_rescue_fraction_abs_diff_gt_0p25_count": int(
            rescue_difference.gt(0.25).sum()
        ),
        "field_rescue_fraction_abs_diff_gt_0p25_rate": float(
            rescue_difference.gt(0.25).mean()
        ),
        "field_dead_fraction_abs_diff_median": float(
            final_difference.median()
        ),
        "field_dead_fraction_abs_diff_gt_0p10_count": int(
            final_difference.gt(0.10).sum()
        ),
        "field_dead_fraction_abs_diff_gt_0p10_rate": float(
            final_difference.gt(0.10).mean()
        ),
        "field_dead_fraction_abs_diff_gt_0p25_count": int(
            final_difference.gt(0.25).sum()
        ),
        "field_dead_fraction_abs_diff_gt_0p25_rate": float(
            final_difference.gt(0.25).mean()
        ),
    }


def perturbation_stability_metrics(
    predictions: pd.DataFrame,
    object_config: dict[str, Any],
    delta: float = 0.05,
) -> dict[str, Any]:
    baseline_evidence = object_evidence(predictions, object_config)
    baseline = baseline_evidence["late_dead_object_evidence"].astype(bool)
    high_confidence = (
        baseline_evidence["death_signal_count"]
        >= int(object_config["minimum_death_signals"]) + 1
    ) | (
        baseline_evidence["healthy_signal_count"]
        >= int(object_config["minimum_healthy_signals"]) + 1
    )
    stable = np.ones(len(predictions), dtype=bool)
    feature_frame = predictions[list(MODEL_FEATURES)].copy()
    for direction in (-1.0, 1.0):
        perturbed = feature_frame.copy()
        for column in MODEL_FEATURES:
            perturbed[column] = np.clip(
                finite_numeric(feature_frame[column]) + direction * delta,
                0.0,
                1.0,
            )
        calls = object_evidence(perturbed, object_config)[
            "late_dead_object_evidence"
        ].astype(bool)
        stable &= calls.to_numpy(bool) == baseline.to_numpy(bool)
    evaluated = stable[high_confidence.to_numpy(bool)]
    stability_rate = float(evaluated.mean()) if evaluated.size else float("nan")
    return {
        "feature_percentile_perturbation": float(delta),
        "high_confidence_evaluated_count": int(evaluated.size),
        "object_evidence_stable_count": int(evaluated.sum()),
        "object_evidence_stability_rate": stability_rate,
        "object_evidence_flip_rate": float(1.0 - stability_rate),
    }


def convergence_report(
    metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
    perturbation_metrics: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "D0_SAFETY": {
            "pass": bool(metrics["changed_d0_object_count"] == 0)
            and bool(
                metrics["untreated_live_anchor_counterfactual_fpr"] <= 0.005
            ),
            "changed_d0_object_count": int(metrics["changed_d0_object_count"]),
            "untreated_live_anchor_counterfactual_fpr": float(
                metrics["untreated_live_anchor_counterfactual_fpr"]
            ),
            "maximum_fpr": 0.005,
        },
        "POSITIVE_ANCHOR_RETENTION": {
            "pass": bool(metrics["blue_supported_anchor_preservation"] == 1.0)
            and bool(metrics["temporal_remnant_anchor_recall"] >= 0.99),
            "blue_supported_anchor_preservation": float(
                metrics["blue_supported_anchor_preservation"]
            ),
            "temporal_remnant_anchor_recall": float(
                metrics["temporal_remnant_anchor_recall"]
            ),
        },
        "LATE_TREATED_ANCHORS": {
            "pass": bool(metrics["development_pass"])
            and bool(metrics["holdout_pass"])
            and bool(
                metrics["f9_day5_median_dead_fraction"]
                >= metrics["target_holdout_dead_fraction"]
            ),
            "development_e9_day5_dead_fraction": float(
                metrics["development_e9_original_day5_dead_fraction"]
            ),
            "holdout_e9_day5_min_dead_fraction": float(
                metrics["holdout_e9_day5_min_dead_fraction"]
            ),
            "f9_day5_median_dead_fraction": float(
                metrics["f9_day5_median_dead_fraction"]
            ),
            "development_target": float(
                metrics["target_e9_dead_fraction"]
            ),
            "holdout_and_replicate_target": float(
                metrics["target_holdout_dead_fraction"]
            ),
            "metric_semantics": (
                "operational_late_treated_anchors_not_manual_ground_truth"
            ),
        },
        "BRANCH_CONSENSUS": {
            "pass": bool(
                branch_metrics["matched_object_evidence_agreement"] >= 0.99
            )
            and bool(branch_metrics["final_field_state_mismatch_rate"] <= 0.01)
            and bool(
                branch_metrics["matched_pair_rescue_call_agreement"] >= 0.99
            )
            and bool(
                branch_metrics["matched_pair_final_dead_call_agreement"] >= 0.99
            ),
            **branch_metrics,
        },
        "TEMPORAL_STABILITY": {
            "pass": bool(metrics["temporal_remnant_anchor_recall"] >= 0.99),
            "temporal_remnant_anchor_recall": float(
                metrics["temporal_remnant_anchor_recall"]
            ),
            "temporal_remnant_anchor_count": int(
                metrics["temporal_remnant_anchor_count"]
            ),
        },
        "PARAMETER_STABILITY": {
            "pass": bool(
                perturbation_metrics["object_evidence_stability_rate"] >= 0.995
            ),
            **perturbation_metrics,
        },
        "PERTURBATION_ROBUSTNESS": {
            "pass": bool(
                perturbation_metrics["object_evidence_flip_rate"] <= 0.005
            ),
            **perturbation_metrics,
        },
    }
    decision = "GO" if all(value["pass"] for value in gates.values()) else "NO_GO"
    return {
        "schema_version": 1,
        "metric_semantics": (
            "operational_proxies_without_manual_biological_ground_truth"
        ),
        "decision": decision,
        "gates": gates,
    }


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < args.target_e9_dead_fraction <= 1:
        raise ValueError("--target-e9-dead-fraction must be in (0, 1]")
    if not 0 < args.target_holdout_dead_fraction <= 1:
        raise ValueError("--target-holdout-dead-fraction must be in (0, 1]")
    if args.max_nonfocus_rows_per_field <= 0:
        raise ValueError("--max-nonfocus-rows-per-field must be positive")
    data, fields = load_dataset(
        args.dataset_root,
        args.max_nonfocus_rows_per_field,
        args.seed,
    )
    data, density_definitions = assign_density(data)
    data, references = calibrate_features(data)

    field_config, field_states, field_trials = optimize_field_state(fields, args)
    field_trials.to_csv(args.out_dir / "field_parameter_trials.csv", index=False)
    field_states.to_csv(args.out_dir / "selected_field_states.csv", index=False)
    object_config, predictions, object_trials = optimize_objects(
        data,
        field_states,
        args,
    )
    object_trials.to_csv(args.out_dir / "object_parameter_trials.csv", index=False)
    summary = field_summary(predictions)
    summary.to_csv(args.out_dir / "late_death_field_summary.csv", index=False)
    metrics = anchor_metrics(
        predictions,
        summary,
        args,
        object_config,
    )
    selected_object_trial = object_trials.iloc[0]
    metrics["untreated_live_anchor_counterfactual_fpr"] = float(
        selected_object_trial["untreated_live_anchor_counterfactual_fpr"]
    )
    metrics["control_fpr_pass"] = (
        math.isfinite(metrics["untreated_live_anchor_counterfactual_fpr"])
        and metrics["untreated_live_anchor_counterfactual_fpr"]
        < args.max_control_false_positive_rate
    )
    branch_metrics = branch_consensus_metrics(predictions, summary)
    perturbation_metrics = perturbation_stability_metrics(
        predictions,
        object_config,
    )
    go_no_go = convergence_report(
        metrics,
        branch_metrics,
        perturbation_metrics,
    )
    write_json(args.out_dir / "FULL_CLASSIFICATION_GO_NO_GO.json", go_no_go)

    output_columns = [
        "cohort",
        "branch",
        "image_id",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "combined_mask_id",
        "centroid_y",
        "centroid_x",
        "final_state",
        "proxy_type",
        "countable",
        "border_touching",
        "development_anchor",
        "holdout_anchor",
        "replicate_diagnostic",
        "treated",
        "density_bin",
        "density_percentile",
        "density_anchor",
        "field_global_late_death",
        "field_branch_raw_global",
        "field_branch_raw_discordant",
        "field_branch_consensus_late_death",
        "field_state_transition",
        "field_collapse_signal_count",
        "field_site_concordance",
        "death_signal_count",
        "healthy_signal_count",
        "death_score",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "branch_partner_matched",
        "branch_partner_distance",
        "branch_partner_current_dead",
        "branch_partner_object_evidence",
        "branch_partner_strong_live",
        "branch_partner_temporal_remnant",
        "branch_partner_death_signal_count",
        "branch_object_evidence_agree",
        "branch_partner_late_death_rescue_call",
        "branch_partner_final_dead_call",
        "branch_late_death_rescue_call_agree",
        "branch_final_dead_call_agree",
        "branch_final_call_discordant",
        "temporal_track_confident",
        "temporal_support_frames",
        "temporal_match_confidence",
        "current_dead_call",
        "temporal_carryforward_call",
        "global_late_death_rescue_call",
        "late_death_rescue_call",
        "final_dead_call",
        "late_death_uncertain",
        "branch_discordant_uncertain",
        "track_uncertain",
        "field_evidence_uncertain",
        "classification_tier",
        *MODEL_FEATURES,
        "source_feature_path",
        "cell_mask_path",
        "nucleus_core_mask_path",
        "nucleus_extent_mask_path",
        "combined_raw_path",
        "dead_raw_path",
    ]
    predictions_path = args.out_dir / "late_death_predictions.csv.gz"
    predictions[output_columns].to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )
    production_pass = bool(go_no_go["decision"] == "GO")
    configuration = {
        "schema_version": 3,
        "metric_semantics": "operational_anchors_not_biological_ground_truth",
        "method": (
            "dual_branch_consensus_with_continuous_density_time_calibration_"
            "multiframe_tracking_and_recoverable_field_state"
        ),
        "field_configuration": field_config,
        "object_configuration": object_config,
        "late_min_hours": args.late_min_hours,
        "target_e9_dead_fraction": args.target_e9_dead_fraction,
        "target_holdout_dead_fraction": args.target_holdout_dead_fraction,
        "max_control_false_positive_rate": args.max_control_false_positive_rate,
        "anchor_metrics": metrics,
        "branch_consensus_metrics": branch_metrics,
        "perturbation_stability_metrics": perturbation_metrics,
        "go_no_go": go_no_go,
        "density_definitions": density_definitions,
        "reference_quantiles": references,
        "production_integration": "approved" if production_pass else "blocked",
        "predictions": str(predictions_path),
        "field_summary": str(args.out_dir / "late_death_field_summary.csv"),
        "decision_layers": [
            "preserve every current confirmed-dead object",
            "require a high-confidence multi-frame post-blue remnant without strong-live evidence",
            "require both frozen segmentation views to agree on the final field gate",
            "calibrate object evidence continuously across d0 density and untreated time drift",
            "measure Dead-channel evidence inside every existing cell mask without changing segmentation",
            "within a collapsed treated field, require object evidence and branch consensus",
            "route branch, track, and field evidence conflicts to uncertainty",
            "never modify frozen d0 classifications",
        ],
    }
    write_json(args.out_dir / "best_configuration.json", configuration)
    write_json(args.out_dir / "anchor_metrics.json", metrics)
    print(f"selected_field_configuration={json.dumps(field_config, sort_keys=True)}")
    print(f"selected_object_configuration={json.dumps(object_config, sort_keys=True)}")
    print(
        "e9_day5_baseline_dead_fraction="
        f"{metrics['development_e9_original_day5_baseline_dead_fraction']:.8f}"
    )
    print(
        "e9_day5_final_dead_fraction="
        f"{metrics['development_e9_original_day5_dead_fraction']:.8f}"
    )
    print(
        "e9_holdout_day5_min_dead_fraction="
        f"{metrics['holdout_e9_day5_min_dead_fraction']:.8f}"
    )
    print(f"production_integration={configuration['production_integration']}")
    print(f"go_no_go_decision={go_no_go['decision']}")
    print(f"best_configuration={args.out_dir / 'best_configuration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
