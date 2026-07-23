from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


class LateDeathTrajectoryRefinementTests(unittest.TestCase):
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
