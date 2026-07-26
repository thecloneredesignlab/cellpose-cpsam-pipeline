#!/usr/bin/env python3
"""Summarize failed operational death-consensus gates without manual labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "_shared"
    / "late_dead_trajectory_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "late_dead_trajectory_model_diagnostics",
    MODEL_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load late-death model: {MODEL_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--late-min-hours", type=float, default=72.0)
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def main() -> int:
    args = parse_args()
    root = args.optimization_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(root / "late_death_predictions.csv.gz")
    temporal_anchor = (
        predictions["proxy_type"].eq("temporal_dead_remnant")
        & predictions["cohort"].eq("trajectory")
        & MODEL.finite_numeric(predictions["elapsed_hours"]).ge(
            args.late_min_hours
        )
        & as_bool(predictions["treated"])
        & as_bool(predictions["countable"])
        & ~as_bool(predictions["border_touching"])
        & predictions["final_state"].astype(str).eq("live")
    )
    anchors = predictions.loc[temporal_anchor].copy()
    high_confidence_anchor = (
        as_bool(anchors["temporal_track_confident"])
        & MODEL.finite_numeric(anchors["temporal_support_frames"]).ge(2)
    )
    anchors["high_confidence_multiframe_anchor"] = (
        high_confidence_anchor.to_numpy(bool)
    )
    eligible_anchors = anchors.loc[
        anchors["high_confidence_multiframe_anchor"]
    ].copy()
    eligible_anchors["missed"] = ~as_bool(
        eligible_anchors["final_dead_call"]
    )
    missed = eligible_anchors.loc[eligible_anchors["missed"]].copy()
    reason_columns = {
        "own_strong_live_veto": as_bool(missed["strong_live_evidence"]),
        "partner_strong_live_veto": as_bool(
            missed["branch_partner_strong_live"]
        ),
        "branch_partner_unmatched": ~as_bool(
            missed["branch_partner_matched"]
        ),
        "partner_lacks_temporal_or_object_support": ~(
            as_bool(missed["branch_partner_temporal_remnant"])
            | as_bool(missed["branch_partner_object_evidence"])
        ),
        "track_not_confident": ~as_bool(missed["temporal_track_confident"]),
        "support_below_two_frames": MODEL.finite_numeric(
            missed["temporal_support_frames"]
        ).lt(2),
        "own_object_evidence_absent": ~as_bool(
            missed["late_dead_object_evidence"]
        ),
    }
    for name, values in reason_columns.items():
        missed[name] = values.to_numpy(bool)
    reason_counts = {
        name: int(missed[name].sum())
        for name in reason_columns
    }
    reason_combinations = Counter(
        "|".join(
            name
            for name in reason_columns
            if bool(row[name])
        )
        or "no_recorded_veto"
        for _index, row in missed.iterrows()
    )
    missed_columns = [
        "cohort",
        "branch",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "combined_mask_id",
        "temporal_support_frames",
        "temporal_match_confidence",
        "death_signal_count",
        "healthy_signal_count",
        "death_score",
        "branch_partner_matched",
        "branch_partner_distance",
        "branch_partner_temporal_remnant",
        "branch_partner_object_evidence",
        "branch_partner_strong_live",
        *reason_columns,
    ]
    missed[missed_columns].to_csv(
        args.out_dir / "missed_temporal_anchors.csv",
        index=False,
    )
    trial_rows = pd.read_csv(root / "object_parameter_trials.csv")
    stability_rows = []
    for trial in trial_rows.itertuples(index=False):
        configuration = json.loads(str(trial.configuration))
        stability = MODEL.perturbation_stability_metrics(
            predictions,
            configuration,
        )
        stability_rows.append(
            {
                "trial": str(trial.trial),
                "passes_development": str(
                    trial.passes_development
                ).strip().lower() == "true",
                "development_e9_dead_fraction": float(
                    trial.development_e9_dead_fraction
                ),
                "untreated_live_anchor_counterfactual_fpr": float(
                    trial.untreated_live_anchor_counterfactual_fpr
                ),
                "new_calls_outside_e9_f9": int(
                    trial.new_calls_outside_e9_f9
                ),
                **stability,
                "configuration": json.dumps(
                    configuration,
                    sort_keys=True,
                ),
            }
        )
    stability_frame = pd.DataFrame(stability_rows).sort_values(
        [
            "passes_development",
            "object_evidence_stability_rate",
            "development_e9_dead_fraction",
            "new_calls_outside_e9_f9",
            "trial",
        ],
        ascending=[False, False, False, True, True],
    )
    stability_frame.to_csv(
        args.out_dir / "configuration_stability.csv",
        index=False,
    )
    stable_development = stability_frame.loc[
        stability_frame["passes_development"]
        & stability_frame["object_evidence_stability_rate"].ge(0.995)
    ]
    fine_rows = []
    for feature_threshold, healthy_gap in product(
        (0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58),
        (0.04, 0.05, 0.06),
    ):
        healthy_threshold = round(feature_threshold - healthy_gap, 2)
        configuration = {
            "feature_threshold": feature_threshold,
            "minimum_death_signals": 2,
            "healthy_threshold": healthy_threshold,
            "minimum_healthy_signals": 3,
            "branch_match_distance_px": 10.0,
            "unmatched_minimum_death_signals": 3,
            "temporal_minimum_support_frames": 2,
        }
        fine_rows.append(
            {
                "trial": (
                    f"fine_feature_{feature_threshold:.2f}_"
                    f"healthy_{healthy_threshold:.2f}"
                ),
                **MODEL.perturbation_stability_metrics(
                    predictions,
                    configuration,
                ),
                "configuration": json.dumps(
                    configuration,
                    sort_keys=True,
                ),
            }
        )
    fine_stability = pd.DataFrame(fine_rows).sort_values(
        [
            "object_evidence_stability_rate",
            "trial",
        ],
        ascending=[False, True],
    )
    fine_stability.to_csv(
        args.out_dir / "fine_configuration_stability.csv",
        index=False,
    )
    summary = {
        "schema_version": 1,
        "metric_semantics": (
            "operational_proxies_without_manual_biological_ground_truth"
        ),
        "temporal_remnant_candidate_count": int(len(anchors)),
        "low_confidence_candidate_count": int(
            (~anchors["high_confidence_multiframe_anchor"]).sum()
        ),
        "temporal_anchor_count": int(len(eligible_anchors)),
        "temporal_anchor_retained": int(
            (~eligible_anchors["missed"]).sum()
        ),
        "temporal_anchor_missed": int(eligible_anchors["missed"].sum()),
        "temporal_anchor_recall": (
            float((~eligible_anchors["missed"]).mean())
            if len(eligible_anchors)
            else None
        ),
        "miss_reason_counts": reason_counts,
        "miss_reason_combinations": dict(reason_combinations.most_common()),
        "missed_by_branch": {
            str(key): int(value)
            for key, value in missed["branch"].value_counts().items()
        },
        "missed_by_well": {
            str(key): int(value)
            for key, value in missed["well"].value_counts().items()
        },
        "missed_temporal_anchors_csv": str(
            args.out_dir / "missed_temporal_anchors.csv"
        ),
        "stable_development_configuration_count": int(
            len(stable_development)
        ),
        "best_stable_development_trial": (
            str(stable_development.iloc[0]["trial"])
            if len(stable_development)
            else None
        ),
        "configuration_stability_csv": str(
            args.out_dir / "configuration_stability.csv"
        ),
        "fine_stable_configuration_count": int(
            fine_stability["object_evidence_stability_rate"].ge(0.995).sum()
        ),
        "best_fine_stability_trial": str(
            fine_stability.iloc[0]["trial"]
        ),
        "best_fine_stability_rate": float(
            fine_stability.iloc[0]["object_evidence_stability_rate"]
        ),
        "fine_configuration_stability_csv": str(
            args.out_dir / "fine_configuration_stability.csv"
        ),
    }
    (args.out_dir / "no_go_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
