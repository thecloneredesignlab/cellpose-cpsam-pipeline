from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysisi"
    / "04_plot_well_counts_over_time.py"
)
SPEC = importlib.util.spec_from_file_location("plot_well_counts_over_time", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


class PlotWellCountsTests(unittest.TestCase):
    def test_plate_map_has_expected_wells_and_conditions(self) -> None:
        plate_map = PLOTTER.load_plate_map(PLOTTER.DEFAULT_PLATE_MAP)
        self.assertEqual(len(plate_map), 80)
        self.assertEqual(set(plate_map["plate_row"]), set("ABCDEFGH"))
        self.assertEqual(set(plate_map["plate_column"]), set(range(2, 12)))

        by_well = plate_map.set_index("well")
        self.assertEqual(float(by_well.loc["A2", "doxorubicin_nm"]), 0.0)
        self.assertEqual(float(by_well.loc["H11", "doxorubicin_nm"]), 800.0)
        self.assertTrue(bool(by_well.loc["A2", "cyclophosphamide"]))
        self.assertFalse(bool(by_well.loc["C2", "cyclophosphamide"]))
        self.assertEqual(by_well.loc["E2", "ploidy"], "4N")
        self.assertEqual(int(by_well.loc["B2", "replicate"]), 2)

    def test_auto_source_prefers_primary_fusion_over_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fusion_summary = root / "classification_fusion" / "summaries" / "cell_count_summary.csv"
            fusion_summary.parent.mkdir(parents=True)
            fusion_summary.write_text("image_id\n")
            legacy_predictions = root / "Combined" / "classification" / "predictions"
            legacy_predictions.mkdir(parents=True)

            source = PLOTTER.resolve_input_source(root, "auto")
            self.assertEqual(source.branch, "fusion")
            self.assertEqual(source.path, fusion_summary.resolve())

            legacy_source = PLOTTER.resolve_input_source(root, "legacy-combined")
            self.assertEqual(legacy_source.branch, "legacy-combined")
            self.assertEqual(legacy_source.path, legacy_predictions.resolve())

    def test_auto_source_prefers_authoritative_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            consensus_summary = (
                root
                / "classification_consensus"
                / "summaries"
                / "cell_count_summary.csv"
            )
            fusion_summary = (
                root
                / "classification_fusion"
                / "summaries"
                / "cell_count_summary.csv"
            )
            consensus_summary.parent.mkdir(parents=True)
            fusion_summary.parent.mkdir(parents=True)
            consensus_summary.write_text("image_id\n")
            fusion_summary.write_text("image_id\n")

            source = PLOTTER.resolve_input_source(root, "auto")

            self.assertEqual(source.branch, "fusion-consensus")
            self.assertEqual(source.path, consensus_summary.resolve())

    def test_fusion_aggregation_uses_countable_cell_denominator(self) -> None:
        fieldnames = [
            "image_id",
            "well",
            "site",
            "elapsed_hours",
            "total_masks",
            "total_cell_count",
            "live_cell_count",
            "dead_cell_count",
            "artifact_count",
            "uncertain_count",
            "transitional_count",
        ]
        rows = [
            {
                "image_id": "SUM159_AC_A2_1_00d00h00m",
                "well": "A2",
                "site": 1,
                "elapsed_hours": 0,
                "total_masks": 10,
                "total_cell_count": 9,
                "live_cell_count": 6,
                "dead_cell_count": 2,
                "artifact_count": 1,
                "uncertain_count": 1,
                "transitional_count": 0,
            },
            {
                "image_id": "SUM159_AC_A2_2_00d00h00m",
                "well": "A2",
                "site": 2,
                "elapsed_hours": 0,
                "total_masks": 9,
                "total_cell_count": 8,
                "live_cell_count": 5,
                "dead_cell_count": 2,
                "artifact_count": 1,
                "uncertain_count": 0,
                "transitional_count": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "cell_count_summary.csv"
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = PLOTTER.load_fusion_summary(summary_path)
            plate_map = PLOTTER.load_plate_map(PLOTTER.DEFAULT_PLATE_MAP)
            PLOTTER.validate_summary(
                summary,
                plate_map,
                strict_completeness=False,
                expected_timepoints=1,
                expected_sites=2,
            )
            aggregated = PLOTTER.aggregate_counts(summary, plate_map)

        self.assertEqual(len(aggregated), 1)
        record = aggregated.iloc[0]
        self.assertEqual(int(record["total_masks"]), 19)
        self.assertEqual(int(record["total_cell_count"]), 17)
        self.assertEqual(int(record["live_count"]), 11)
        self.assertEqual(int(record["dead_count"]), 4)
        self.assertAlmostEqual(float(record["live_fraction"]), 11 / 17)
        self.assertAlmostEqual(float(record["dead_fraction"]), 4 / 17)

    def test_legacy_loader_preserves_intermediate_state_counts(self) -> None:
        fieldnames = [
            "image_id",
            "total_masks",
            "live_count",
            "dead_count",
            "transitional_count",
            "artifact_count",
            "uncertain_count",
        ]
        rows = [
            {
                "image_id": "SUM159_AC_A2_1_00d00h00m",
                "total_masks": 10,
                "live_count": 5,
                "dead_count": 2,
                "transitional_count": 1,
                "artifact_count": 1,
                "uncertain_count": 1,
            },
            {
                "image_id": "SUM159_AC_A2_2_00d00h00m",
                "total_masks": 8,
                "live_count": 4,
                "dead_count": 1,
                "transitional_count": 1,
                "artifact_count": 1,
                "uncertain_count": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_dir = Path(tmpdir)
            for row in rows:
                path = predictions_dir / f"{row['image_id']}_summary.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)

            summary = PLOTTER.load_legacy_summary(predictions_dir, workers=2)
            plate_map = PLOTTER.load_plate_map(PLOTTER.DEFAULT_PLATE_MAP)
            PLOTTER.validate_summary(
                summary,
                plate_map,
                strict_completeness=False,
                expected_timepoints=1,
                expected_sites=2,
            )
            aggregated = PLOTTER.aggregate_counts(summary, plate_map)

        record = aggregated.iloc[0]
        self.assertEqual(int(record["total_cell_count"]), 16)
        self.assertEqual(int(record["transitional_count"]), 2)
        self.assertEqual(int(record["uncertain_count"]), 2)


if __name__ == "__main__":
    unittest.main()
