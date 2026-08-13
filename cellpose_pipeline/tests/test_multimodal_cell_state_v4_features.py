from __future__ import annotations

import csv
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
SCRIPT = ROOT / "cellpose_pipeline/scripts/48_extract_multimodal_cell_state_v4_features.py"
HELPER = ROOT / "cellpose_pipeline/scripts/_shared/multimodal_cell_state_v4_features.py"
CONFIG = ROOT / "cellpose_pipeline/configs/multimodal_cell_state_v4.json"
SPEC = importlib.util.spec_from_file_location("v4_features", HELPER)
assert SPEC is not None and SPEC.loader is not None
FEATURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FEATURES
SPEC.loader.exec_module(FEATURES)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV4FeatureTests(unittest.TestCase):
    def test_dead_zero_aware_states_and_topology(self) -> None:
        shape = (32, 36)
        obj = np.zeros(shape, dtype=bool)
        obj[10:22, 12:24] = True
        rng = np.random.default_rng(20260813)
        background = rng.normal(100, 2, size=shape)

        present = background.copy()
        present[12:16, 14:18] += 40
        present[17:20, 20:23] += 25
        result = FEATURES.dead_features_for_object(obj, present)
        self.assertEqual(result["dead_measurement_status"], "dead_signal_present_supported")
        self.assertEqual(result["dead_signal_absent"], 0)
        self.assertGreater(result["dead_signal_positive_fraction"], 0)
        self.assertGreater(result["dead_largest_positive_component_fraction"], 0)

        absent = FEATURES.dead_features_for_object(obj, background)
        self.assertEqual(absent["dead_measurement_status"], "dead_signal_absent_supported")
        self.assertEqual(absent["dead_signal_absent"], 1)
        self.assertTrue(np.isnan(absent["dead_largest_positive_component_fraction"]))

        saturated_raw = background.copy()
        saturated_raw[obj] = 65535
        saturated = FEATURES.dead_features_for_object(
            obj, saturated_raw, saturation_value=65535
        )
        self.assertEqual(saturated["dead_measurement_status"], "dead_signal_saturated")
        self.assertEqual(saturated["dead_signal_absent"], 0)

        flat = np.full(shape, 100.0)
        uncertain = FEATURES.dead_features_for_object(obj, flat)
        self.assertEqual(
            uncertain["dead_measurement_status"], "dead_signal_background_uncertain"
        )
        self.assertEqual(uncertain["dead_signal_absent"], 0)

    def test_config_freezes_four_equal_blocks_and_raw_dead_only(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            list(value["feature_blocks"]), ["shape", "brightfield", "nuclei", "dead"]
        )
        self.assertEqual(value["block_equalization"]["target_fraction_each"], 0.25)
        self.assertEqual(value["dead_zero_aware"]["raw_source"], "profiles.Dead.raw_first_tiff_page")
        self.assertTrue(value["dead_zero_aware"]["temporal_normalization_forbidden"])
        self.assertIn("Dead_raw", value["blinding"]["allowed_before_model_freeze"])
        self.assertIn(
            "Existing_Dead_segmentation",
            value["blinding"]["forbidden_before_model_freeze"],
        )
        self.assertNotIn(
            "nuclei_overlap_area_px2",
            value["feature_blocks"]["nuclei"]["candidate_columns"],
        )

    def test_field_extractor_reads_raw_dead_and_not_dead_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (30, 34)
            combined = np.zeros(shape, dtype=np.uint16)
            combined[3:14, 3:14] = 1
            combined[16:27, 18:29] = 2
            nuclei_mask = np.zeros(shape, dtype=np.uint16)
            nuclei_mask[6:10, 6:10] = 11
            rng = np.random.default_rng(7)
            nuclei_raw = rng.normal(100, 3, size=shape).astype(np.float32)
            dead_raw = rng.normal(50, 2, size=shape).astype(np.float32)
            dead_raw[combined == 2] += 30
            for name, array in (
                ("combined.tif", combined),
                ("nuclei_mask.tif", nuclei_mask),
                ("nuclei_raw.tif", nuclei_raw),
                ("dead_raw.tif", dead_raw),
            ):
                tifffile.imwrite(root / name, array)
            record = {
                "key": "A10_1_01d02h00m",
                "profiles": {
                    "Combined": {"original_mask": str(root / "combined.tif")},
                    "Nuclei": {
                        "raw": str(root / "nuclei_raw.tif"),
                        "core_mask": str(root / "nuclei_mask.tif"),
                    },
                    "Dead": {
                        "raw": str(root / "dead_raw.tif"),
                        "segmentation": str(root / "forbidden_dead_segmentation.tif"),
                    },
                },
            }
            (root / "record.json").write_text(json.dumps(record) + "\n")
            broad_fields = [
                "cell_id", "key", "well", "branch", "mask_label", "area_px2",
                "roundness", "aspect_ratio", "extent", "solidity",
                "bf_object_median", "bf_object_iqr", "bf_object_p10", "bf_object_p90",
                "bf_object_mean_minus_ring_background", "bf_interior_minus_boundary_mean",
                "bf_boundary_gradient_p95", "bf_laplacian_variance", "bf_edge_density",
                "bf_glcm_contrast", "bf_glcm_entropy", "bf_glcm_correlation",
            ]
            broad_rows = []
            for label in (1, 2):
                broad_rows.append(
                    {
                        "cell_id": f"original|{record['key']}|{label}",
                        "key": record["key"], "well": "A10", "branch": "original",
                        "mask_label": str(label), "area_px2": "121", "roundness": ".7",
                        "aspect_ratio": "1.2", "extent": ".8", "solidity": ".9",
                        "bf_object_median": ".5", "bf_object_iqr": ".1",
                        "bf_object_p10": ".4", "bf_object_p90": ".7",
                        "bf_object_mean_minus_ring_background": ".05",
                        "bf_interior_minus_boundary_mean": "-.01",
                        "bf_boundary_gradient_p95": ".4", "bf_laplacian_variance": ".03",
                        "bf_edge_density": ".2", "bf_glcm_contrast": "1.1",
                        "bf_glcm_entropy": "2.2", "bf_glcm_correlation": ".3",
                    }
                )
            with (root / "broad.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=broad_fields, delimiter="\t", lineterminator="\n")
                writer.writeheader(); writer.writerows(broad_rows)
            with (root / "cells.tsv").open("w", newline="", encoding="utf-8") as handle:
                rows = [
                    {"cell_id": row["cell_id"], "key": row["key"], "well": "A10", "mask_label": row["mask_label"], "split": "development"}
                    for row in broad_rows
                ]
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)
            output = root / "generation"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--field-record", str(root / "record.json"),
                 "--broad-feature-tsv", str(root / "broad.tsv"), "--selected-cells", str(root / "cells.tsv"),
                 "--config", str(CONFIG), "--output-dir", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            rows = read_tsv(output / "features.tsv")
            self.assertEqual(len(rows), 2)
            self.assertIn(rows[1]["dead_measurement_status"], {
                "dead_signal_present_supported", "dead_signal_saturated"
            })
            receipt = json.loads((output / "feature_receipt.json").read_text())
            self.assertEqual(receipt["forbidden_inputs_read"], [])
            self.assertEqual(
                receipt["input_contract"]["dead_raw"],
                "profiles.Dead.raw_first_tiff_page",
            )
            self.assertEqual(
                receipt["input_contract"]["existing_dead_segmentation"], "not_read"
            )


if __name__ == "__main__":
    unittest.main()
