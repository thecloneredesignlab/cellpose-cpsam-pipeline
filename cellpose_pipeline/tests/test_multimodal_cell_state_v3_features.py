from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "41_extract_multimodal_cell_state_v3_features.py"
)
HELPER = (
    ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "_shared"
    / "multimodal_cell_state_v3_features.py"
)
CONFIG = ROOT / "cellpose_pipeline" / "configs" / "multimodal_cell_state_v3.json"

SPEC = importlib.util.spec_from_file_location("v3_feature_test_helper", HELPER)
assert SPEC is not None and SPEC.loader is not None
FEATURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FEATURES
SPEC.loader.exec_module(FEATURES)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV3FeatureTests(unittest.TestCase):
    def test_zero_aware_present_absent_and_possible_miss(self) -> None:
        shape = (20, 24)
        nuclei_mask = np.zeros(shape, dtype=np.uint16)
        raw = np.zeros(shape, dtype=np.float64)

        present = np.zeros(shape, dtype=bool)
        present[2:9, 2:9] = True
        nuclei_mask[4:7, 4:7] = 1
        raw[4:7, 4:7] = 20
        present_result = FEATURES.nucleus_features_for_object(present, nuclei_mask, raw)
        self.assertEqual(
            present_result["nuclei_measurement_status"], "present_supported"
        )
        self.assertEqual(present_result["nucleus_absent"], 0)
        self.assertAlmostEqual(
            present_result["nuclei_area_fraction_if_present"], 9 / 49
        )
        self.assertEqual(present_result["largest_nucleus_fraction"], 1.0)

        absent = np.zeros(shape, dtype=bool)
        absent[11:18, 2:9] = True
        absent_result = FEATURES.nucleus_features_for_object(absent, nuclei_mask, raw)
        self.assertEqual(
            absent_result["nuclei_measurement_status"], "low_quality_nuclei_signal"
        )
        self.assertEqual(absent_result["nucleus_absent"], 0)
        self.assertTrue(np.isnan(absent_result["nuclei_area_fraction_if_present"]))

        rng = np.random.default_rng(17)
        noisy = rng.normal(10, 1, size=shape)
        quiet_absent = FEATURES.nucleus_features_for_object(absent, nuclei_mask, noisy)
        self.assertEqual(quiet_absent["nuclei_measurement_status"], "absent_supported")
        self.assertEqual(quiet_absent["nucleus_absent"], 1)

        missed = noisy.copy()
        missed[absent] += 12
        miss_result = FEATURES.nucleus_features_for_object(absent, nuclei_mask, missed)
        self.assertEqual(
            miss_result["nuclei_measurement_status"], "possible_nuclei_mask_miss"
        )
        self.assertEqual(miss_result["nucleus_absent"], 0)
        self.assertTrue(np.isnan(miss_result["nuclei_area_fraction_if_present"]))

    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        shape = (24, 28)
        combined = np.zeros(shape, dtype=np.uint16)
        combined[3:11, 3:11] = 1
        combined[13:21, 15:23] = 2
        nuclei = np.zeros(shape, dtype=np.uint16)
        nuclei[5:8, 5:8] = 11
        rng = np.random.default_rng(8)
        raw = rng.normal(100, 4, size=shape).astype(np.float32)
        raw[combined == 2] += 50
        for name, value in (
            ("combined.tif", combined),
            ("nuclei_mask.tif", nuclei),
            ("nuclei_raw.tif", raw),
        ):
            tifffile.imwrite(root / name, value)
        record = {
            "key": "A10_1_01d02h00m",
            "profiles": {
                "Combined": {"original_mask": str(root / "combined.tif")},
                "Nuclei": {
                    "raw": str(root / "nuclei_raw.tif"),
                    "core_mask": str(root / "nuclei_mask.tif"),
                },
                "Dead": {"raw": str(root / "forbidden_dead.tif")},
            },
        }
        record_path = root / "record.json"
        record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        broad_path = root / "broad.tsv"
        base = {
            "key": record["key"],
            "well": "A10",
            "branch": "original",
            "area_px2": "64",
            "roundness": "0.7",
            "aspect_ratio": "1.2",
            "extent": "0.8",
            "solidity": "0.9",
            "bf_object_median": "0.5",
            "bf_object_iqr": "0.1",
            "bf_object_p10": "0.4",
            "bf_object_p90": "0.7",
            "bf_object_mean_minus_ring_background": "0.05",
            "bf_interior_minus_boundary_mean": "-0.01",
            "bf_boundary_gradient_p95": "0.4",
            "bf_laplacian_variance": "0.03",
            "bf_edge_density": "0.2",
            "bf_glcm_contrast": "1.1",
            "bf_glcm_entropy": "2.2",
            "bf_glcm_correlation": "0.3",
        }
        rows = []
        for label in (1, 2):
            row = dict(base)
            row.update(
                cell_id=f"original|{record['key']}|{label}", mask_label=str(label)
            )
            rows.append(row)
        with broad_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        selected_path = root / "cells.tsv"
        with selected_path.open("w", newline="", encoding="utf-8") as handle:
            selected_rows = [
                {
                    "cell_id": row["cell_id"],
                    "key": row["key"],
                    "well": row["well"],
                    "mask_label": row["mask_label"],
                    "split": "development",
                }
                for row in rows
            ]
            writer = csv.DictWriter(
                handle,
                fieldnames=list(selected_rows[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(selected_rows)
        return record_path, broad_path, selected_path

    def test_field_generation_atomic_reuse_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, broad, selected = self.make_fixture(root)
            output = root / "generation"
            command = [
                sys.executable,
                str(SCRIPT),
                "--field-record",
                str(record),
                "--broad-feature-tsv",
                str(broad),
                "--selected-cells",
                str(selected),
                "--config",
                str(CONFIG),
                "--output-dir",
                str(output),
            ]
            created = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.assertEqual(created.returncode, 0, created.stdout)
            self.assertIn("generation_status=created", created.stdout)
            rows = read_tsv(output / "features.tsv")
            self.assertEqual(len(rows), 2)
            self.assertNotIn(
                "nuclei_overlap_area_px2",
                json.loads(CONFIG.read_text())["feature_blocks"]["nuclei"][
                    "candidate_columns"
                ],
            )
            self.assertEqual(rows[0]["nuclei_measurement_status"], "present_supported")
            self.assertEqual(
                rows[1]["nuclei_measurement_status"], "possible_nuclei_mask_miss"
            )
            receipt = json.loads((output / "feature_receipt.json").read_text())
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(receipt["forbidden_inputs_read"], [])
            self.assertEqual(
                receipt["inputs"]["implementation"]["sha256"],
                hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            )
            reused = subprocess.run(
                [*command, "--overwrite"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(reused.returncode, 0, reused.stdout)
            self.assertIn("generation_status=verified_reuse", reused.stdout)
            with (output / "features.tsv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            rejected = subprocess.run(
                [*command, "--overwrite"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact hash differs", rejected.stdout)

    def test_config_freezes_equal_blocks_and_blinding(self) -> None:
        value = json.loads(CONFIG.read_text())
        self.assertEqual(
            list(value["feature_blocks"]), ["shape", "brightfield", "nuclei"]
        )
        self.assertEqual(value["block_equalization"]["target_fraction_each"], 1 / 3)
        self.assertFalse(value["projection_audit"]["dead_island_required"])
        self.assertIn("Dead", value["blinding"]["forbidden_before_model_freeze"])
        self.assertEqual(
            value["projection_grid"]["primary_profile"], "v3_pruned_block_equal"
        )


if __name__ == "__main__":
    unittest.main()
