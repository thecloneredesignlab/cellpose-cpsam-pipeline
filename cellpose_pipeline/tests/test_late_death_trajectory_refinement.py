from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import tifffile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODEL = load_module(
    "late_dead_trajectory_model_test",
    SCRIPT_DIR / "_shared" / "late_dead_trajectory_model.py",
)
REFINEMENT = load_module(
    "late_dead_trajectory_refinement_test",
    SCRIPT_DIR / "14_apply_late_dead_trajectory_refinement.py",
)
try:
    DATASET_BUILDER = load_module(
        "late_dead_trajectory_dataset_test",
        SCRIPT_DIR / "13_build_late_dead_trajectory_dataset.py",
    )
except ModuleNotFoundError:
    DATASET_BUILDER = None


class LateDeathTrajectoryRefinementTests(unittest.TestCase):
    @unittest.skipUnless(
        DATASET_BUILDER is not None,
        "The local lightweight test environment does not include scikit-image.",
    )
    def test_cell_conditioned_dead_features_use_existing_masks(self) -> None:
        cell_labels = np.zeros((10, 10), dtype=np.int32)
        cell_labels[3:7, 3:7] = 1
        nucleus_labels = np.zeros_like(cell_labels)
        nucleus_labels[4:6, 4:6] = 1
        yy, xx = np.indices(cell_labels.shape)
        dead_raw = np.where((yy + xx) % 2 == 0, 9.0, 11.0)
        dead_raw[cell_labels == 1] = 20.0
        dead_raw[nucleus_labels == 1] = 40.0

        features = DATASET_BUILDER.cell_conditioned_dead_features(
            cell_labels,
            nucleus_labels,
            dead_raw,
        )

        self.assertEqual(features["combined_mask_id"].tolist(), [1])
        self.assertGreater(float(features.loc[0, "cell_dead_snr"]), 0.0)
        self.assertEqual(float(features.loc[0, "cell_dead_positive_fraction"]), 1.0)
        self.assertGreater(
            float(features.loc[0, "cell_dead_core_enrichment"]),
            0.0,
        )

    def test_continuous_density_anchor_weights(self) -> None:
        self.assertEqual(MODEL.density_anchor_weights(0.0), ((0.1, 1.0),))
        self.assertEqual(MODEL.density_anchor_weights(1.0), ((0.9, 1.0),))
        weights = MODEL.density_anchor_weights(0.40)
        self.assertEqual(tuple(anchor for anchor, _weight in weights), (0.3, 0.5))
        self.assertAlmostEqual(sum(weight for _anchor, weight in weights), 1.0)
        self.assertAlmostEqual(weights[0][1], 0.5)
        self.assertAlmostEqual(weights[1][1], 0.5)

    def test_vectorized_feature_calibration_matches_scalar_interpolation(self) -> None:
        densities = [0.0, 0.10, 0.20, 0.40, 0.90, 1.0]
        rows = []
        arrays = {}
        for index, density in enumerate(densities):
            row = {
                "branch": "original",
                "key": f"E2_1_{index:02d}d00h00m",
                "elapsed_hours": 0.0,
                "density_percentile": density,
            }
            for _output, (raw, _direction) in MODEL.RAW_FEATURES.items():
                row[raw] = 25.0 + index
            rows.append(row)
        for anchor in MODEL.DENSITY_ANCHORS:
            for _output, (raw, _direction) in MODEL.RAW_FEATURES.items():
                arrays[("original", anchor, "d0", raw)] = np.linspace(
                    anchor * 10.0,
                    100.0 + anchor * 10.0,
                    101,
                )
        calibrated = MODEL.apply_empirical_feature_calibration(
            pd.DataFrame(rows),
            arrays,
            error_context="test",
        )
        for row_index, density in enumerate(densities):
            for output, (raw, direction) in MODEL.RAW_FEATURES.items():
                expected = 0.0
                used_weight = 0.0
                for anchor, weight in MODEL.density_anchor_weights(density):
                    expected += weight * MODEL.empirical_percentile(
                        arrays[("original", anchor, "d0", raw)],
                        np.asarray([rows[row_index][raw]], dtype=float),
                        direction,
                        reference_sorted=True,
                    )[0]
                    used_weight += weight
                self.assertAlmostEqual(
                    float(calibrated.loc[row_index, output]),
                    expected / used_weight,
                    places=12,
                )

    def test_field_state_can_recover_after_sustained_normalization(self) -> None:
        rows = []
        for site in range(1, 5):
            for offset, collapsed in enumerate(
                (True, True, True, True, False, False, False)
            ):
                rows.append(
                    {
                        "branch": "original",
                        "well": "E9",
                        "site": site,
                        "elapsed_hours": 72.0 + 2.0 * offset,
                        "key": f"E9_{site}_{72 + 2 * offset:03d}h",
                        "field_count_ratio_to_peak": 0.50 if collapsed else 1.0,
                        "field_area_ratio_to_peak": 0.20 if collapsed else 1.0,
                        "field_cytoplasm_ratio_to_peak": (
                            0.20 if collapsed else 1.0
                        ),
                        "field_red_mass_ratio_to_peak": (
                            0.20 if collapsed else 1.0
                        ),
                        "field_mask_fraction_ratio_to_peak": (
                            0.40 if collapsed else 1.0
                        ),
                    }
                )
        fields = MODEL.apply_field_configuration(
            pd.DataFrame(rows),
            REFINEMENT.FIELD_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
        )
        site_one = fields.loc[fields["site"].eq(1)].sort_values("elapsed_hours")
        self.assertIn("entered", set(site_one["field_state_transition"]))
        self.assertEqual(site_one.iloc[-1]["field_state_transition"], "recovered")
        self.assertFalse(bool(site_one.iloc[-1]["field_global_late_death"]))

    def test_field_consensus_requires_both_segmentation_views(self) -> None:
        fields = pd.DataFrame(
            {
                "branch": [
                    "original",
                    "nucleated_only",
                    "original",
                    "nucleated_only",
                ],
                "key": ["E9_1_t1", "E9_1_t1", "E9_1_t2", "E9_1_t2"],
                "field_global_late_death": [True, False, True, True],
            }
        )
        consensus = MODEL.apply_branch_field_consensus(fields)
        first = consensus.loc[consensus["key"].eq("E9_1_t1")]
        second = consensus.loc[consensus["key"].eq("E9_1_t2")]
        self.assertTrue(first["field_branch_raw_discordant"].astype(bool).all())
        self.assertFalse(first["field_global_late_death"].astype(bool).any())
        self.assertFalse(second["field_branch_raw_discordant"].astype(bool).any())
        self.assertTrue(second["field_global_late_death"].astype(bool).all())
        self.assertEqual(
            first.loc[first["branch"].eq("original"), "field_branch_raw_global"].iloc[0],
            True,
        )

    def test_temporal_rescue_requires_consensus_and_respects_live_veto(self) -> None:
        base = {
            "cohort": "trajectory",
            "key": "E9_1_05d00h00m",
            "well": "E9",
            "site": 1,
            "elapsed_hours": 120.0,
            "treated": True,
            "countable": True,
            "border_touching": False,
            "final_state": "live",
            "proxy_type": "temporal_dead_remnant",
            "temporal_track_confident": True,
            "temporal_support_frames": 3,
            "temporal_match_confidence": 0.95,
            "centroid_y": 50.0,
            "centroid_x": 50.0,
        }
        rows = []
        for branch in ("original", "nucleated_only"):
            row = dict(base, branch=branch, combined_mask_id=1)
            row.update({feature: 0.10 for feature in MODEL.MODEL_FEATURES})
            rows.append(row)
        fields = pd.DataFrame(
            {
                "branch": ["original", "nucleated_only"],
                "key": [base["key"], base["key"]],
                "field_global_late_death": [True, True],
                "field_collapse_signal_count": [5, 5],
                "field_site_concordance": [1.0, 1.0],
                "field_branch_raw_discordant": [False, False],
                "field_branch_raw_global": [True, True],
                "field_branch_consensus_late_death": [True, True],
            }
        )
        vetoed = MODEL.classification_calls(
            pd.DataFrame(rows),
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        self.assertTrue(vetoed["strong_live_evidence"].astype(bool).all())
        self.assertFalse(vetoed["late_death_rescue_call"].astype(bool).any())

        death_supported = pd.DataFrame(rows)
        for feature in MODEL.MODEL_FEATURES:
            death_supported[feature] = 0.95
        rescued = MODEL.classification_calls(
            death_supported,
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        self.assertTrue(rescued["branch_partner_matched"].astype(bool).all())
        self.assertTrue(rescued["temporal_carryforward_call"].astype(bool).all())
        self.assertTrue(rescued["late_death_rescue_call"].astype(bool).all())

    def test_cached_branch_matches_preserve_classification_calls(self) -> None:
        rows = []
        for branch, x_offset in (
            ("original", 0.0),
            ("nucleated_only", 0.5),
        ):
            for object_id, x_position, percentile in (
                (1, 20.0, 0.95),
                (2, 80.0, 0.10),
            ):
                row = {
                    "cohort": "trajectory",
                    "branch": branch,
                    "key": "E9_1_05d00h00m",
                    "well": "E9",
                    "site": 1,
                    "elapsed_hours": 120.0,
                    "treated": True,
                    "combined_mask_id": object_id,
                    "centroid_y": 50.0,
                    "centroid_x": x_position + x_offset,
                    "countable": True,
                    "border_touching": False,
                    "final_state": "live",
                    "proxy_type": (
                        "temporal_dead_remnant"
                        if object_id == 1
                        else "unlabeled"
                    ),
                    "temporal_track_confident": object_id == 1,
                    "temporal_support_frames": 3 if object_id == 1 else 0,
                    "temporal_match_confidence": 0.95 if object_id == 1 else 0.0,
                }
                row.update(
                    {feature: percentile for feature in MODEL.MODEL_FEATURES}
                )
                rows.append(row)
        objects = pd.DataFrame(rows)
        fields = pd.DataFrame(
            {
                "branch": ["original", "nucleated_only"],
                "key": ["E9_1_05d00h00m", "E9_1_05d00h00m"],
                "field_global_late_death": [True, True],
                "field_collapse_signal_count": [5, 5],
                "field_site_concordance": [1.0, 1.0],
                "field_branch_raw_discordant": [False, False],
                "field_branch_raw_global": [True, True],
                "field_branch_consensus_late_death": [True, True],
            }
        )
        direct = MODEL.classification_calls(
            objects,
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        cached = MODEL.classification_calls(
            MODEL.precompute_branch_object_matches(objects, 10.0),
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        columns = [
            "branch_partner_matched",
            "branch_partner_distance",
            "branch_partner_current_dead",
            "branch_partner_object_evidence",
            "branch_partner_strong_live",
            "branch_partner_temporal_remnant",
            "branch_partner_death_signal_count",
            "branch_object_evidence_agree",
            "temporal_carryforward_call",
            "global_late_death_rescue_call",
            "late_death_rescue_call",
            "final_dead_call",
            "late_death_uncertain",
            "classification_tier",
        ]
        pd.testing.assert_frame_equal(
            direct[columns].reset_index(drop=True),
            cached[columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_confirmed_temporal_rescue_propagates_across_matched_views(self) -> None:
        rows = []
        for branch, proxy_type, x_offset in (
            ("original", "temporal_dead_remnant", 0.0),
            ("nucleated_only", "unlabeled", 0.5),
        ):
            row = {
                "cohort": "trajectory",
                "branch": branch,
                "key": "E9_1_05d00h00m",
                "well": "E9",
                "site": 1,
                "elapsed_hours": 120.0,
                "treated": True,
                "combined_mask_id": 1,
                "centroid_y": 50.0,
                "centroid_x": 50.0 + x_offset,
                "countable": True,
                "border_touching": False,
                "final_state": "live",
                "proxy_type": proxy_type,
                "temporal_track_confident": proxy_type == "temporal_dead_remnant",
                "temporal_support_frames": (
                    3 if proxy_type == "temporal_dead_remnant" else 0
                ),
                "temporal_match_confidence": (
                    0.95 if proxy_type == "temporal_dead_remnant" else 0.0
                ),
            }
            row.update({feature: 0.95 for feature in MODEL.MODEL_FEATURES})
            rows.append(row)
        fields = pd.DataFrame(
            {
                "branch": ["original", "nucleated_only"],
                "key": ["E9_1_05d00h00m", "E9_1_05d00h00m"],
                "field_global_late_death": [False, False],
                "field_collapse_signal_count": [0, 0],
                "field_site_concordance": [0.0, 0.0],
                "field_branch_raw_discordant": [False, False],
                "field_branch_raw_global": [False, False],
                "field_branch_consensus_late_death": [False, False],
            }
        )
        calls = MODEL.classification_calls(
            pd.DataFrame(rows),
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        self.assertTrue(calls["temporal_carryforward_call"].astype(bool).all())
        self.assertTrue(
            calls["branch_late_death_rescue_call_agree"].astype(bool).all()
        )
        self.assertTrue(calls["branch_final_dead_call_agree"].astype(bool).all())

    def test_persistent_field_collapse_and_object_evidence_rescue(self) -> None:
        rows = []
        for site in range(1, 5):
            for elapsed_hours in (0.0, 72.0, 74.0, 76.0):
                collapsed = elapsed_hours >= 72.0
                rows.append(
                    {
                        "branch": "original",
                        "well": "E9",
                        "site": site,
                        "elapsed_hours": elapsed_hours,
                        "key": f"E9_{site}_{int(elapsed_hours):03d}h",
                        "field_count_ratio_to_peak": 0.50 if collapsed else 1.0,
                        "field_area_ratio_to_peak": 0.20 if collapsed else 1.0,
                        "field_cytoplasm_ratio_to_peak": 0.20 if collapsed else 1.0,
                        "field_red_mass_ratio_to_peak": 0.20 if collapsed else 1.0,
                        "field_mask_fraction_ratio_to_peak": (
                            0.40 if collapsed else 1.0
                        ),
                    }
                )
        fields = MODEL.apply_field_configuration(
            pd.DataFrame(rows),
            REFINEMENT.FIELD_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
        )
        terminal = fields.loc[
            fields["elapsed_hours"].eq(76.0),
            "field_global_late_death",
        ]
        self.assertTrue(terminal.astype(bool).all())

        objects = fields[
            ["branch", "well", "site", "elapsed_hours", "key"]
        ].copy()
        objects["cohort"] = "trajectory"
        objects.loc[objects["elapsed_hours"].eq(0.0), "cohort"] = "d0_frozen"
        objects["treated"] = True
        objects["countable"] = True
        objects["border_touching"] = False
        objects["final_state"] = "live"
        objects["proxy_type"] = "unlabeled"
        for feature in MODEL.MODEL_FEATURES:
            objects[feature] = 0.90
        calls = MODEL.classification_calls(
            objects,
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        self.assertFalse(
            calls.loc[
                calls["elapsed_hours"].lt(76.0),
                "late_death_rescue_call",
            ].astype(bool).any()
        )
        self.assertTrue(
            calls.loc[
                calls["elapsed_hours"].eq(76.0),
                "late_death_rescue_call",
            ].astype(bool).all()
        )

        untreated = objects.copy()
        untreated["treated"] = False
        untreated_calls = MODEL.classification_calls(
            untreated,
            fields,
            REFINEMENT.OBJECT_CONFIGURATION,
            REFINEMENT.LATE_MIN_HOURS,
            apply_treatment_scope=True,
        )
        self.assertFalse(untreated_calls["late_death_rescue_call"].astype(bool).any())

    def test_restore_pre_refinement_state_is_idempotent(self) -> None:
        previously_refined = pd.DataFrame(
            {
                "state": ["dead"],
                "final_state": ["dead"],
                "final_reason": ["late_death_trajectory_rescue"],
                "classification_confidence": ["medium"],
                "pre_late_death_state": ["live"],
                "pre_late_death_final_reason": ["rgb_live"],
                "pre_late_death_classification_confidence": ["high"],
                "late_death_rescue_call": [True],
                "late_death_score": [0.9],
            }
        )
        restored = REFINEMENT.restore_pre_refinement_state(
            previously_refined,
            "final_state",
        )
        self.assertEqual(restored.loc[0, "state"], "live")
        self.assertEqual(restored.loc[0, "final_state"], "live")
        self.assertEqual(restored.loc[0, "final_reason"], "rgb_live")
        self.assertEqual(restored.loc[0, "classification_confidence"], "high")
        self.assertNotIn("late_death_rescue_call", restored)
        self.assertNotIn("late_death_score", restored)

        captured = REFINEMENT.capture_pre_refinement_state(
            restored,
            "final_state",
        )
        self.assertEqual(captured.loc[0, "pre_late_death_state"], "live")
        self.assertEqual(
            captured.loc[0, "pre_late_death_classification_confidence"],
            "high",
        )

    def test_production_rejects_calibration_configuration_drift(self) -> None:
        approved = {
            "production_integration": "approved",
            "field_configuration": dict(REFINEMENT.FIELD_CONFIGURATION),
            "object_configuration": dict(REFINEMENT.OBJECT_CONFIGURATION),
            "late_min_hours": REFINEMENT.LATE_MIN_HOURS,
        }
        REFINEMENT.validate_approved_calibration_configuration(approved)
        drifted = json.loads(json.dumps(approved))
        drifted["object_configuration"]["feature_threshold"] += 0.10
        with self.assertRaisesRegex(
            RuntimeError,
            "object configuration does not match",
        ):
            REFINEMENT.validate_approved_calibration_configuration(drifted)

    def test_prepared_reference_artifacts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = REFINEMENT.prepared_paths(Path(temporary))
            references = {
                ("original", 0.1, "d0", "area"): np.asarray(
                    [1.0, 2.0, 3.0]
                ),
                ("nucleated_only", 0.9, "late", "red_mass_proxy"): np.asarray(
                    [4.0, 5.0]
                ),
            }
            REFINEMENT.write_reference_artifacts(paths, references)
            restored = REFINEMENT.load_reference_artifacts(paths)
            self.assertEqual(set(restored), set(references))
            for key, expected in references.items():
                np.testing.assert_array_equal(restored[key], expected)

    def test_well_failure_traceback_is_written_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classification_root = root / "classification"
            dataset_root = root / "dataset"
            classification_root.mkdir()
            paths = REFINEMENT.prepared_paths(dataset_root)
            paths["root"].mkdir(parents=True)
            paths["receipt"].write_text('{"schema_version": 1}\n')
            inventory = pd.DataFrame(
                {
                    "well": ["E2", "E2"],
                    "branch": ["original", "nucleated_only"],
                    "key": ["E2_1_t0", "E2_1_t0"],
                    "shard_path": ["original.csv.gz", "nucleated.csv.gz"],
                }
            )
            inventory.to_csv(paths["inventory"], index=False)
            pd.DataFrame({"well": ["E2"]}).to_csv(
                paths["field_states"],
                index=False,
            )
            manifest = pd.DataFrame(
                {
                    "array_index": [1],
                    "well": ["E2"],
                    "original_fields": [1],
                    "nucleated_only_fields": [1],
                    "total_shards": [2],
                }
            )
            args = SimpleNamespace(
                classification_root=classification_root,
                dataset_root=dataset_root,
                overlay_alpha=0.55,
                force=False,
            )
            with (
                mock.patch.object(
                    REFINEMENT,
                    "validate_prepared_state",
                    return_value=(paths, {"schema_version": 1}, manifest),
                ),
                mock.patch.object(
                    REFINEMENT,
                    "load_reference_artifacts",
                    return_value={},
                ),
                mock.patch.object(
                    REFINEMENT,
                    "refine_group",
                    side_effect=MemoryError("simulated well failure"),
                ),
            ):
                with self.assertRaisesRegex(MemoryError, "simulated"):
                    REFINEMENT.run_one_well(args, 1)
            failure_path = REFINEMENT.well_artifact_paths(
                classification_root,
                1,
                "E2",
            )["failure"]
            self.assertTrue(failure_path.is_file())
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure["well"], "E2")
            self.assertEqual(failure["status"], "FAILED")
            self.assertIn("MemoryError", failure["error"])
            self.assertIn("simulated well failure", failure["traceback"])

    def test_production_receipt_reports_operational_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            consensus_path = (
                root
                / "classification_consensus"
                / "summaries"
                / "cell_count_summary.csv"
            )
            consensus_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "key": ["E2_1_00d00h00m", "E9_1_05d00h00m"],
                    "branch_dead_fraction_abs_diff": [0.0, 0.005],
                    "late_death_rescue_count": [0, 10],
                    "total_cell_count": [100, 100],
                    "nucleated_only_late_death_rescue_count": [0, 10],
                    "nucleated_only_total_cell_count": [100, 100],
                }
            ).to_csv(consensus_path, index=False)
            status_rows = []
            for branch in ("original", "nucleated_only"):
                for key in ("E2_1_00d00h00m", "E9_1_05d00h00m"):
                    status_rows.append(
                        {
                            "branch": branch,
                            "key": key,
                            "field_global_late_death": key.startswith("E9"),
                            "objects": 100,
                            "rescued": 0 if key.startswith("E2") else 10,
                            "uncertain": 0,
                            "matched_objects": 100,
                            "matched_rescue_call_agree": 100,
                            "matched_final_dead_call_agree": 100,
                        }
                    )
            receipt_path, receipt = REFINEMENT.write_production_go_no_go(
                root,
                pd.DataFrame(status_rows),
                consensus_path,
                {
                    "decision": "GO",
                    "metric_semantics": (
                        "operational_proxies_without_manual_biological_ground_truth"
                    ),
                },
                {
                    "verified": True,
                    "receipt_path": "freeze.json",
                    "source_run_root": "frozen",
                    "verified_file_count": 34,
                },
                expected_fields_per_branch=2,
            )
            self.assertTrue(receipt_path.is_file())
            self.assertEqual(receipt["decision"], "GO")
            self.assertFalse(receipt["biological_accuracy_claimed"])
            self.assertTrue(receipt["gates"]["D0_INVARIANCE"]["pass"])
            self.assertTrue(
                receipt["gates"]["FULL_COHORT_COMPLETENESS"]["pass"]
            )

    def test_field_outputs_are_updated_and_force_rerun_stays_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "classification_fusion"
            image_id = "SUM159_AC_E9_1_05d00h00m"
            key = "E9_1_05d00h00m"
            feature_path = (
                out_dir
                / "features"
                / f"{image_id}_per_cell_fusion_features.csv"
            )
            prediction_path = (
                out_dir / "predictions" / f"{image_id}_per_cell_predictions.csv"
            )
            summary_path = out_dir / "summaries" / f"{image_id}_summary.csv"
            for path in (feature_path, prediction_path, summary_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "combined_mask_id": [1],
                    "state": ["live"],
                    "final_state": ["live"],
                    "final_reason": ["rgb_live"],
                    "classification_confidence": ["high"],
                }
            ).to_csv(feature_path, index=False)
            pd.DataFrame(
                {
                    "mask_id": [1],
                    "state": ["live"],
                    "final_reason": ["rgb_live"],
                    "classification_confidence": ["high"],
                }
            ).to_csv(prediction_path, index=False)
            pd.DataFrame(
                {
                    "image_id": [image_id],
                    "key": [key],
                    "well": ["E9"],
                    "site": [1],
                    "elapsed_hours": [120.0],
                    "total_masks": [1],
                    "total_cell_count": [1],
                    "live_cell_count": [1],
                    "dead_cell_count": [0],
                    "uncertain_count": [0],
                    "transitional_count": [0],
                    "artifact_count": [0],
                    "dead_fraction": [0.0],
                    "live_fraction": [1.0],
                    "supplemental_dead_object_count": [0],
                    "object_aware_dead_count": [0],
                }
            ).to_csv(summary_path, index=False)

            cell_mask = np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                dtype=np.uint32,
            )
            core_mask = np.where(cell_mask > 0, 1, 0).astype(np.uint32)
            combined_raw = np.zeros((4, 4, 3), dtype=np.uint8)
            combined_raw[..., 0] = 100
            nuclei_raw = np.zeros((4, 4), dtype=np.uint8)
            dead_raw = np.zeros((4, 4), dtype=np.uint8)
            paths = {
                "cell_mask": root / "cell_mask.tif",
                "core_mask": root / "core_mask.tif",
                "combined_raw": root / "combined_raw.tif",
                "nuclei_raw": root / "nuclei_raw.tif",
                "dead_raw": root / "dead_raw.tif",
            }
            tifffile.imwrite(paths["cell_mask"], cell_mask)
            tifffile.imwrite(paths["core_mask"], core_mask)
            tifffile.imwrite(paths["combined_raw"], combined_raw)
            tifffile.imwrite(paths["nuclei_raw"], nuclei_raw)
            tifffile.imwrite(paths["dead_raw"], dead_raw)
            cell_state_path = (
                out_dir / "masks" / "cell_state" / f"{image_id}_cell_state_masks.tif"
            )
            confirmed_path = (
                out_dir
                / "masks"
                / "confirmed_dead"
                / f"{image_id}_confirmed_dead_masks.tif"
            )
            cell_state_path.parent.mkdir(parents=True, exist_ok=True)
            confirmed_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(cell_state_path, cell_mask)
            tifffile.imwrite(confirmed_path, np.zeros_like(cell_mask))
            record_path = root / "record.json"
            record_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "Nuclei": {
                                "raw": str(paths["nuclei_raw"]),
                            }
                        }
                    }
                )
            )

            row = {
                "branch": "original",
                "key": key,
                "image_id": image_id,
                "combined_mask_id": 1,
                "field_global_late_death": True,
                "field_collapse_signal_count": 5,
                "field_site_concordance": 1.0,
                "density_bin": "middle",
                "death_signal_count": 4,
                "healthy_signal_count": 0,
                "death_score": 0.9,
                "strong_live_evidence": False,
                "late_dead_object_evidence": True,
                "temporal_carryforward_call": False,
                "global_late_death_rescue_call": True,
                "late_death_rescue_call": True,
                "late_death_uncertain": False,
                "current_dead_call": False,
                "final_dead_call": True,
                "source_feature_path": str(feature_path),
                "cell_mask_path": str(paths["cell_mask"]),
                "nucleus_core_mask_path": str(paths["core_mask"]),
                "combined_raw_path": str(paths["combined_raw"]),
                "dead_raw_path": str(paths["dead_raw"]),
                "field_record_path": str(record_path),
            }
            row.update({feature: 0.9 for feature in MODEL.MODEL_FEATURES})
            field_rows = pd.DataFrame([row])
            REFINEMENT.WORKER_ARGS = {
                "classification_root": str(root),
                "overlay_alpha": 0.55,
                "force": True,
            }
            first = REFINEMENT.update_field_outputs(field_rows)
            self.assertEqual(first["rescued"], 1)
            updated = pd.read_csv(feature_path)
            self.assertEqual(updated.loc[0, "pre_late_death_state"], "live")
            self.assertEqual(updated.loc[0, "final_state"], "dead")
            self.assertTrue(bool(updated.loc[0, "late_death_rescue_call"]))
            summary = pd.read_csv(summary_path)
            self.assertEqual(int(summary.loc[0, "dead_cell_count"]), 1)
            self.assertEqual(int(summary.loc[0, "live_cell_count"]), 0)

            second = REFINEMENT.update_field_outputs(field_rows)
            self.assertEqual(second["rescued"], 1)
            rerun = pd.read_csv(feature_path)
            self.assertEqual(rerun.loc[0, "pre_late_death_state"], "live")
            self.assertEqual(rerun.loc[0, "final_state"], "dead")
            self.assertFalse(
                any(column.endswith(("_x", "_y")) for column in rerun.columns)
            )


if __name__ == "__main__":
    unittest.main()
