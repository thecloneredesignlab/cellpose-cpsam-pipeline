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
APPROVED_FIELD_CONFIGURATION = {
    "area_ratio_max": 0.25,
    "count_ratio_max": 0.60,
    "mask_fraction_ratio_max": 0.45,
    "minimum_field_signals": 4,
    "minimum_site_fraction": 0.75,
    "persistence_frames": 3,
    "red_mass_ratio_max": 0.25,
}
APPROVED_OBJECT_CONFIGURATION = {
    "feature_threshold": 0.60,
    "healthy_threshold": 0.65,
    "minimum_death_signals": 2,
    "minimum_healthy_signals": 4,
}
APPROVED_LATE_MIN_HOURS = 72.0
APPROVED_CONFIGURATION_SOURCE = (
    "late_dead_trajectory_optimization_20260723_024151/optimization_v3/"
    "best_configuration.json"
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
    for branch, branch_rows in field_rows.groupby("branch"):
        reference = branch_rows.loc[branch_rows["cohort"].eq("d0_frozen"), "density_score"]
        if len(reference) < 50:
            raise ValueError(f"Insufficient d0 fields for density calibration: {branch}")
        q1, q2 = np.quantile(reference, [1 / 3, 2 / 3])
        definitions[str(branch)] = {"low_middle": float(q1), "middle_high": float(q2)}
        selector = field_rows["branch"].eq(branch)
        field_rows.loc[selector, "density_bin"] = np.select(
            [
                field_rows.loc[selector, "density_score"] <= q1,
                field_rows.loc[selector, "density_score"] <= q2,
            ],
            ["low", "middle"],
            default="high",
        )
    result = result.merge(
        field_rows[["branch", "key", "density_score", "density_bin"]],
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    return result, definitions


def empirical_percentile(
    reference: np.ndarray,
    values: np.ndarray,
    direction: str,
) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
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


def calibrate_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = data.copy()
    references: dict[str, Any] = {}
    d0_live = result.loc[
        result["cohort"].eq("d0_frozen")
        & result["proxy_type"].eq("d0_live_anchor")
        & result["countable"]
        & ~result["border_touching"]
    ]
    for (branch, density_bin), indices in result.groupby(
        ["branch", "density_bin"]
    ).groups.items():
        reference_rows = d0_live.loc[
            d0_live["branch"].eq(branch) & d0_live["density_bin"].eq(density_bin)
        ]
        reference_key = f"{branch}:{density_bin}"
        references[reference_key] = {}
        for output, (raw, direction) in RAW_FEATURES.items():
            reference = pd.to_numeric(reference_rows[raw], errors="coerce").to_numpy(float)
            target = pd.to_numeric(result.loc[indices, raw], errors="coerce").to_numpy(float)
            result.loc[indices, output] = empirical_percentile(
                reference,
                target,
                direction,
            )
            finite_reference = reference[np.isfinite(reference)]
            references[reference_key][output] = {
                "raw_feature": raw,
                "direction": direction,
                "count": int(len(finite_reference)),
                "quantiles": {
                    str(q): float(np.quantile(finite_reference, q))
                    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
                },
            }
    if result[list(MODEL_FEATURES)].isna().any().any():
        raise ValueError("Density-calibrated object features contain missing values")
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
    for _group, indices in result.groupby(["branch", "well", "site"], sort=False).groups.items():
        ordered = result.loc[indices].sort_values("elapsed_hours")
        rolling = (
            eligible.loc[ordered.index]
            .astype(int)
            .rolling(persistence, min_periods=persistence)
            .sum()
            .ge(persistence)
        )
        absorbing = rolling.cummax()
        result.loc[ordered.index, "field_global_late_death"] = absorbing.to_numpy(bool)
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
            }
        )
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
    feature_threshold = float(config["feature_threshold"])
    death_matrix = np.column_stack(
        (
            nc >= feature_threshold,
            area >= feature_threshold,
            red_mass >= feature_threshold,
            shape >= feature_threshold,
        )
    )
    healthy_threshold = float(config["healthy_threshold"])
    healthy_matrix = np.column_stack(
        (
            nc <= healthy_threshold,
            area <= healthy_threshold,
            red_mass <= healthy_threshold,
            shape <= healthy_threshold,
        )
    )
    result["death_signal_count"] = death_matrix.sum(axis=1)
    result["healthy_signal_count"] = healthy_matrix.sum(axis=1)
    result["death_score"] = 0.30 * nc + 0.28 * area + 0.32 * red_mass + 0.10 * shape
    result["strong_live_evidence"] = (
        result["healthy_signal_count"] >= int(config["minimum_healthy_signals"])
    )
    result["late_dead_object_evidence"] = (
        result["death_signal_count"] >= int(config["minimum_death_signals"])
    ) & ~result["strong_live_evidence"]
    return result


def classification_calls(
    data: pd.DataFrame,
    fields: pd.DataFrame,
    object_config: dict[str, Any],
    late_min_hours: float,
    apply_treatment_scope: bool,
) -> pd.DataFrame:
    field_state = fields[
        [
            "branch",
            "key",
            "field_global_late_death",
            "field_collapse_signal_count",
            "field_site_concordance",
        ]
    ].drop_duplicates(["branch", "key"])
    work = data.merge(
        field_state,
        on=["branch", "key"],
        how="left",
        validate="many_to_one",
    )
    evidence = object_evidence(work, object_config)
    for column in evidence:
        work[column] = evidence[column]
    current_dead = work["final_state"].astype(str).eq("dead")
    temporal = work["proxy_type"].eq("temporal_dead_remnant")
    eligible = (
        work["cohort"].eq("trajectory")
        & finite_numeric(work["elapsed_hours"]).ge(late_min_hours)
        & work["countable"]
        & ~work["border_touching"]
        & work["final_state"].astype(str).eq("live")
    )
    if apply_treatment_scope:
        eligible &= work["treated"]
    global_rescue = (
        eligible
        & work["field_global_late_death"].fillna(False)
        & work["late_dead_object_evidence"]
    )
    temporal_rescue = eligible & temporal
    work["current_dead_call"] = current_dead
    work["temporal_carryforward_call"] = temporal_rescue
    work["global_late_death_rescue_call"] = global_rescue
    work["late_death_rescue_call"] = temporal_rescue | global_rescue
    work["final_dead_call"] = current_dead | work["late_death_rescue_call"]
    work["late_death_uncertain"] = (
        eligible
        & work["field_global_late_death"].fillna(False)
        & ~work["late_dead_object_evidence"]
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
    return best_config, best_state, trial_frame


def optimize_objects(
    data: pd.DataFrame,
    fields: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trials: list[dict[str, Any]] = []
    configurations: dict[int, dict[str, Any]] = {}
    for index, config in enumerate(object_parameter_grid(), start=1):
        counterfactual = classification_calls(
            data,
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
        trials.append(
            {
                "trial": f"object_{index:04d}",
                "development_e9_dead_fraction": dev_fraction,
                "untreated_live_anchor_counterfactual_fpr": control_fpr,
                "new_calls_outside_e9_f9": new_calls_outside_e9,
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
    trial_frame = trial_frame.sort_values(
        [
            "passes_development",
            "development_e9_dead_fraction",
            "untreated_live_anchor_counterfactual_fpr",
            "new_calls_outside_e9_f9",
            "trial",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    best_trial = str(trial_frame.iloc[0]["trial"])
    best_index = int(best_trial.rsplit("_", 1)[1])
    best_config = configurations[best_index]
    predictions = classification_calls(
        data,
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
    )
    d0_changed = predictions.loc[
        predictions["cohort"].eq("d0_frozen"),
        "late_death_rescue_call",
    ]
    metrics = {
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
    metrics = anchor_metrics(predictions, summary, args)
    selected_object_trial = object_trials.iloc[0]
    metrics["untreated_live_anchor_counterfactual_fpr"] = float(
        selected_object_trial["untreated_live_anchor_counterfactual_fpr"]
    )
    metrics["control_fpr_pass"] = (
        math.isfinite(metrics["untreated_live_anchor_counterfactual_fpr"])
        and metrics["untreated_live_anchor_counterfactual_fpr"]
        < args.max_control_false_positive_rate
    )

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
        "field_global_late_death",
        "field_collapse_signal_count",
        "field_site_concordance",
        "death_signal_count",
        "healthy_signal_count",
        "death_score",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "current_dead_call",
        "temporal_carryforward_call",
        "global_late_death_rescue_call",
        "late_death_rescue_call",
        "final_dead_call",
        "late_death_uncertain",
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
    production_pass = bool(
        metrics["development_pass"]
        and metrics["holdout_pass"]
        and metrics["d0_pass"]
        and metrics["control_fpr_pass"]
        and metrics["blue_supported_anchor_preservation"] == 1.0
        and metrics["temporal_remnant_anchor_recall"] >= 0.99
    )
    configuration = {
        "schema_version": 2,
        "metric_semantics": "operational_anchors_not_biological_ground_truth",
        "method": "density_aware_field_collapse_with_object_evidence_and_absorbing_time_state",
        "field_configuration": field_config,
        "object_configuration": object_config,
        "late_min_hours": args.late_min_hours,
        "target_e9_dead_fraction": args.target_e9_dead_fraction,
        "target_holdout_dead_fraction": args.target_holdout_dead_fraction,
        "max_control_false_positive_rate": args.max_control_false_positive_rate,
        "anchor_metrics": metrics,
        "density_definitions": density_definitions,
        "reference_quantiles": references,
        "production_integration": "approved" if production_pass else "blocked",
        "predictions": str(predictions_path),
        "field_summary": str(args.out_dir / "late_death_field_summary.csv"),
        "decision_layers": [
            "preserve every current confirmed-dead object",
            "preserve a tracked post-blue remnant",
            "detect a persistent multi-site field collapse from count, area, cytoplasm, red-mass, and mask-coverage trajectories",
            "within a collapsed treated field, reclassify only objects with multi-signal late-death evidence and no strong live evidence",
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
    print(f"best_configuration={args.out_dir / 'best_configuration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
