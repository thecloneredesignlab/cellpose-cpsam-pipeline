from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FEATURES = load_module(
    "broad_phenotype_features_test",
    PIPELINE_ROOT / "scripts" / "_shared" / "broad_phenotype_features.py",
)
EXTRACTOR = load_module(
    "broad_phenotype_extractor_test",
    PIPELINE_ROOT / "scripts" / "16_extract_broad_phenotype_features.py",
)


class BroadPhenotypeFeatureTests(unittest.TestCase):
    def synthetic_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.indices((18, 20))
        raw = (20 + yy * 2 + xx * 3).astype(np.uint16)
        mask = np.zeros(raw.shape, dtype=np.int32)
        mask[3:6, 4:8] = 7
        mask[10:14, 12:16] = 2
        return raw, mask

    def test_shared_config_locks_schema_parameters_and_model_allowlists(self) -> None:
        config = FEATURES.load_feature_config(EXTRACTOR.DEFAULT_CONFIG)

        self.assertEqual(len(FEATURES.NUMERIC_FEATURE_COLUMNS), 52)
        self.assertEqual(config["payload"]["method_version"], "bf_combined_mask_v1")
        self.assertEqual(config["parameters"].object_ring_radius, 3)
        self.assertEqual(config["parameters"].global_background_dilation_radius, 5)
        self.assertEqual(config["parameters"].glcm_levels, 16)
        self.assertEqual(config["parameters"].glcm_min_pairs, 8)
        self.assertEqual(
            config["parameters"].glcm_offsets,
            ((0, 1), (1, 0), (1, 1), (1, -1)),
        )
        primary = config["primary_model_feature_columns"]
        comparator = config["nuclei_comparator_feature_columns"]
        self.assertTrue(set(primary).issubset(comparator))
        self.assertFalse(
            any(name.startswith(("centroid_", "bbox_")) for name in primary)
        )
        self.assertTrue(
            {"nuclei_count", "nuclei_overlap_area_px2", "nuclei_area_fraction"}
            .issubset(primary)
        )
        self.assertTrue(
            {"nuclei_count", "nuclei_overlap_area_px2", "nuclei_area_fraction"}
            .issubset(comparator)
        )
        self.assertRegex(config["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(config["semantic_sha256"], r"^[0-9a-f]{64}$")

    def test_geometry_and_brightfield_families_are_deterministic(self) -> None:
        raw, mask = self.synthetic_inputs()
        first, first_diagnostics = FEATURES.extract_broad_phenotype_features(raw, mask)
        second, second_diagnostics = FEATURES.extract_broad_phenotype_features(raw, mask)

        self.assertEqual([row["mask_label"] for row in first], [2, 7])
        self.assertEqual(first_diagnostics, second_diagnostics)
        for left, right in zip(first, second):
            self.assertEqual(set(left), {"mask_label", *FEATURES.NUMERIC_FEATURE_COLUMNS})
            self.assertEqual(left["mask_label"], right["mask_label"])
            for column in FEATURES.NUMERIC_FEATURE_COLUMNS:
                a = float(left[column])
                b = float(right[column])
                if math.isnan(a):
                    self.assertTrue(math.isnan(b), column)
                else:
                    self.assertEqual(a, b, column)

        by_label = {row["mask_label"]: row for row in first}
        rectangle = by_label[7]
        self.assertEqual(rectangle["area_px2"], 12)
        self.assertEqual(rectangle["perimeter_px"], 14.0)
        self.assertEqual(rectangle["bbox_x_min"], 4)
        self.assertEqual(rectangle["bbox_x_max"], 8)
        self.assertEqual(rectangle["bbox_y_min"], 3)
        self.assertEqual(rectangle["bbox_y_max"], 6)
        self.assertAlmostEqual(rectangle["extent"], 1.0)
        self.assertAlmostEqual(rectangle["solidity"], 1.0)
        self.assertAlmostEqual(
            rectangle["roundness"],
            4.0 * math.pi * 12.0 / (14.0**2),
        )
        self.assertGreater(rectangle["major_axis_px"], rectangle["minor_axis_px"])
        self.assertGreater(rectangle["bf_object_p90"], rectangle["bf_object_p10"])
        self.assertGreater(rectangle["bf_boundary_gradient_mean"], 0.0)
        self.assertTrue(math.isfinite(rectangle["bf_glcm_contrast"]))
        self.assertTrue(math.isfinite(rectangle["bf_glcm_entropy"]))
        self.assertEqual(rectangle["nuclei_comparator_enabled"], 0)
        self.assertTrue(math.isnan(rectangle["nuclei_count"]))

    def test_nucleus_comparator_counts_overlapping_labels_only(self) -> None:
        raw = np.arange(144, dtype=np.uint16).reshape(12, 12)
        mask = np.zeros((12, 12), dtype=np.int32)
        mask[2:8, 2:8] = 4
        nuclei = np.zeros_like(mask)
        nuclei[3:5, 3:5] = 10
        nuclei[6:8, 6:8] = 20
        nuclei[0:2, 0:2] = 30

        rows, diagnostics = FEATURES.extract_broad_phenotype_features(
            raw,
            mask,
            nuclei_mask=nuclei,
        )

        self.assertTrue(diagnostics["nuclei_comparator_enabled"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["nuclei_comparator_enabled"], 1)
        self.assertEqual(row["nuclei_count"], 2)
        self.assertEqual(row["nuclei_overlap_area_px2"], 8)
        self.assertAlmostEqual(row["nuclei_area_fraction"], 8.0 / 36.0)

    def test_invalid_masks_and_dimension_mismatches_fail(self) -> None:
        raw = np.ones((8, 8), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "does not match"):
            FEATURES.extract_broad_phenotype_features(
                raw,
                np.zeros((7, 8), dtype=np.int32),
            )
        noninteger = np.zeros((8, 8), dtype=np.float64)
        noninteger[2, 2] = 1.5
        with self.assertRaisesRegex(ValueError, "non-integer"):
            FEATURES.extract_broad_phenotype_features(raw, noninteger)
        negative = np.zeros((8, 8), dtype=np.int32)
        negative[2, 2] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            FEATURES.extract_broad_phenotype_features(raw, negative)
        with self.assertRaisesRegex(ValueError, "channel-last"):
            FEATURES.extract_broad_phenotype_features(
                np.zeros((3, 8, 8), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.int32),
            )

    def test_cli_uses_brightfield_and_selected_combined_branch_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, original = self.synthetic_inputs()
            nucleated = original.copy()
            nucleated[10:14, 14:16] = 0
            nuclei = np.zeros_like(original)
            nuclei[3:5, 5:7] = 1
            raw_path = root / "bf.tif"
            original_path = root / "original.tif"
            nucleated_path = root / "nucleated.tif"
            nuclei_path = root / "nuclei_core.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(original_path, original)
            tifffile.imwrite(nucleated_path, nucleated)
            tifffile.imwrite(nuclei_path, nuclei)
            record_path = root / "A2_1_00d00h00m.json"
            record_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "key": "A2_1_00d00h00m",
                        "profiles": {
                            "Brightfield": {"raw": str(raw_path)},
                            "Combined": {
                                "raw": str(root / "must_not_be_read_combined_raw.tif"),
                                "original_mask": str(original_path),
                                "nucleated_mask": str(nucleated_path),
                            },
                            "Nuclei": {"core_mask": str(nuclei_path)},
                            "Dead": {
                                "raw": str(root / "must_not_be_read_dead_raw.tif"),
                                "mask": str(root / "must_not_be_read_dead_mask.tif"),
                                "current_classification": str(
                                    root / "must_not_be_read_current_classification.tsv"
                                ),
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_dir = root / "shadow"

            status = EXTRACTOR.main(
                ["--field-record", str(record_path), "--out-dir", str(out_dir)]
            )

            self.assertEqual(status, 0)
            feature_path = (
                out_dir
                / "shards"
                / "A2"
                / "A2_1_00d00h00m__original_broad_phenotype_features.tsv"
            )
            receipt_path = (
                out_dir / "receipts" / "A2" / "A2_1_00d00h00m__original.json"
            )
            self.assertTrue(feature_path.is_file())
            self.assertTrue(receipt_path.is_file())
            with feature_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), EXTRACTOR.OUTPUT_COLUMNS)
            self.assertEqual([int(row["mask_label"]) for row in rows], [2, 7])
            self.assertEqual(rows[0]["cell_id"], "original|A2_1_00d00h00m|2")
            self.assertEqual(rows[0]["nuclei_comparator_enabled"], "0")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(receipt["row_count"], 2)
            self.assertEqual(receipt["input_contract"]["dead"], "not_read")
            self.assertEqual(receipt["input_contract"]["current_classification"], "not_read")
            self.assertEqual(receipt["forbidden_inputs_read"], [])
            first_hash = EXTRACTOR.sha256_file(feature_path)
            self.assertEqual(
                EXTRACTOR.main(
                    ["--field-record", str(record_path), "--out-dir", str(out_dir)]
                ),
                0,
            )
            self.assertEqual(EXTRACTOR.sha256_file(feature_path), first_hash)

            self.assertEqual(
                EXTRACTOR.main(
                    [
                        "--field-record",
                        str(record_path),
                        "--out-dir",
                        str(out_dir),
                        "--branch",
                        "nucleated",
                        "--include-nuclei-comparator",
                    ]
                ),
                0,
            )
            nucleated_feature_path = (
                out_dir
                / "shards"
                / "A2"
                / "A2_1_00d00h00m__nucleated_broad_phenotype_features.tsv"
            )
            with nucleated_feature_path.open(encoding="utf-8", newline="") as handle:
                nucleated_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(nucleated_rows), 2)
            self.assertTrue(
                all(row["nuclei_comparator_enabled"] == "1" for row in nucleated_rows)
            )
            self.assertFalse(list(out_dir.rglob("*.tmp.*")))

    def test_cli_preserves_conflicting_existing_generation_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = np.arange(100, dtype=np.uint16).reshape(10, 10)
            mask = np.zeros((10, 10), dtype=np.uint16)
            mask[2:8, 2:8] = 1
            raw_path = root / "bf.tif"
            mask_path = root / "mask.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(mask_path, mask)
            record_path = root / "record.json"
            record_path.write_text(
                json.dumps(
                    {
                        "key": "B3_2_01d02h30m",
                        "profiles": {
                            "Brightfield": {"raw": str(raw_path)},
                            "Combined": {
                                "original_mask": str(mask_path),
                                "nucleated_mask": str(mask_path),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "out"
            self.assertEqual(
                EXTRACTOR.main(
                    ["--field-record", str(record_path), "--out-dir", str(out_dir)]
                ),
                0,
            )
            feature_path, _receipt_path = EXTRACTOR.output_paths(
                out_dir,
                "B3_2_01d02h30m",
                "B3",
                "original",
            )
            feature_path.write_text("corrupt\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                EXTRACTOR.main(
                    ["--field-record", str(record_path), "--out-dir", str(out_dir)]
                )


if __name__ == "__main__":
    unittest.main()
