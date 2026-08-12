import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "29_compare_current_vs_reference_cell_state.py"
)
SPEC = importlib.util.spec_from_file_location("reference_cell_state_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(
    path: Path,
    rows: list[dict[str, object]],
    *,
    delimiter: str = "\t",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class ReferenceCellStateComparisonTests(unittest.TestCase):
    def build_normalized_fixture(self, root: Path) -> dict[str, Path]:
        current = root / "current_input" / "current.tsv"
        reference = root / "reference_input" / "reference.tsv"
        output = root / "comparison_output"
        current_rows = [
            {
                "cell_id": f"original|A1_1_0d00h00m|{label}",
                "key": "A1_1_0d00h00m",
                "mask_label": label,
                "well": "A1",
                "elapsed_hours": 0,
                "current_final_state": state,
                "current_prediction_status": "ok",
            }
            for label, state in enumerate(
                ("live", "dead", "uncertain", "artifact", "live"), start=1
            )
        ]
        reference_values = (
            ("live_cell", "ok"),
            ("dead_cell", "ok"),
            ("multinucleated_cell", "ok"),
            ("", "unavailable_missing_features"),
            ("multinucleated_cell", "ok"),
        )
        reference_rows = [
            {
                "model_id": "reference_model_test",
                "cell_id": f"original|A1_1_0d00h00m|{label}",
                "reference_cell_state_class_id": class_id,
                "prediction_status": status,
            }
            for label, (class_id, status) in enumerate(reference_values, start=1)
        ]
        write_table(current, current_rows)
        write_table(reference, reference_rows)
        return {"current": current, "reference": reference, "output": output}

    def run_normalized(self, fixture: dict[str, Path]) -> int:
        return COMPARE.main(
            [
                "--current-predictions",
                str(fixture["current"]),
                "--reference-predictions",
                str(fixture["reference"]),
                "--output-root",
                str(fixture["output"]),
            ]
        )

    def test_precomparison_outputs_preserve_classes_and_abstentions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            self.assertEqual(self.run_normalized(fixture), 0)

            output = fixture["output"]
            receipt = json.loads((output / "comparison_receipt.json").read_text())
            self.assertEqual(receipt["status"], "PRECOMPARISON_COMPLETE")
            self.assertEqual(
                receipt["scientific_claim"], "DESCRIPTIVE_ASSOCIATION_ONLY"
            )
            self.assertFalse(receipt["accuracy_claimed"])
            self.assertFalse(receipt["ground_truth_included"])
            self.assertFalse(receipt["results_fed_back_to_either_model"])
            self.assertFalse(
                receipt["multinucleated_cell_renamed_or_collapsed_to_live"]
            )
            self.assertEqual(receipt["row_count"], 5)
            self.assertEqual(receipt["reference_unavailable_count"], 1)
            self.assertEqual(receipt["current_dead_endpoint_abstention_count"], 2)

            paired = read_tsv(output / "paired_cell_predictions.tsv")
            self.assertEqual(len(paired), 5)
            self.assertEqual(
                paired[4]["reference_cell_state_class_id"], "multinucleated_cell"
            )
            self.assertEqual(paired[4]["reference_dead_endpoint_call"], "not_dead")
            self.assertEqual(paired[2]["current_dead_endpoint_call"], "abstain")
            self.assertEqual(paired[3]["reference_dead_endpoint_call"], "abstain")

            cross = read_tsv(output / "cross_classification_counts.tsv")
            self.assertEqual(len(cross), 12)
            lookup = {
                (row["current_final_state"], row["reference_cell_state_class_id"]): int(
                    row["count"]
                )
                for row in cross
            }
            self.assertEqual(lookup[("live", "live_cell")], 1)
            self.assertEqual(lookup[("live", "multinucleated_cell")], 1)
            self.assertEqual(lookup[("artifact", "dead_cell")], 0)

            subset = read_tsv(output / "dead_endpoint_descriptive_subset.tsv")
            self.assertEqual(len(subset), 3)
            self.assertTrue(
                all(row["dead_endpoint_pair_status"] != "abstention" for row in subset)
            )
            fractions = read_tsv(output / "well_time_class_fractions.tsv")
            self.assertEqual(len(fractions), 7)
            reference_multinucleated = next(
                row
                for row in fractions
                if row["method"] == "reference"
                and row["class_id"] == "multinucleated_cell"
            )
            self.assertEqual(reference_multinucleated["count"], "2")
            self.assertEqual(
                reference_multinucleated["class_fraction_denominator"],
                "reference_available_predictions",
            )

    def test_manifest_mode_verifies_feature_prediction_identity_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            feature = root / "current_features" / "A1_1_0d00h00m_features.csv"
            prediction = root / "current_predictions" / "A1_1_0d00h00m_predictions.csv"
            manifest = root / "current_manifest" / "manifest.tsv"
            reference = root / "reference_input" / "reference.tsv"
            output = root / "comparison_output"
            write_table(
                feature,
                [
                    {
                        "key": "A1_1_0d00h00m",
                        "well": "A1",
                        "elapsed_hours": 0,
                        "combined_mask_id": 1,
                        "final_state": "live",
                    },
                    {
                        "key": "A1_1_0d00h00m",
                        "well": "A1",
                        "elapsed_hours": 0,
                        "combined_mask_id": 2,
                        "final_state": "dead",
                    },
                ],
                delimiter=",",
            )
            write_table(
                prediction,
                [
                    {"key": "A1_1_0d00h00m", "mask_id": 1, "state": "live"},
                    {"key": "A1_1_0d00h00m", "mask_id": 2, "state": "dead"},
                ],
                delimiter=",",
            )
            write_table(
                manifest,
                [
                    {
                        "key": "A1_1_0d00h00m",
                        "feature_path": str(feature),
                        "feature_sha256": sha256(feature),
                        "prediction_path": str(prediction),
                        "prediction_sha256": sha256(prediction),
                    }
                ],
            )
            write_table(
                reference,
                [
                    {
                        "model_id": "reference_model_test",
                        "cell_id": "original|A1_1_0d00h00m|1",
                        "reference_cell_state_class_id": "live_cell",
                        "prediction_status": "ok",
                    },
                    {
                        "model_id": "reference_model_test",
                        "cell_id": "original|A1_1_0d00h00m|2",
                        "reference_cell_state_class_id": "dead_cell",
                        "prediction_status": "ok",
                    },
                ],
            )
            self.assertEqual(
                COMPARE.main(
                    [
                        "--current-manifest",
                        str(manifest),
                        "--reference-predictions",
                        str(reference),
                        "--output-root",
                        str(output),
                    ]
                ),
                0,
            )
            receipt = json.loads((output / "comparison_receipt.json").read_text())
            self.assertEqual(
                receipt["current_input_mode"],
                "sha256_verified_per_field_feature_prediction_manifest",
            )
            self.assertEqual(
                receipt["inputs"]["current"]["verified_source_file_count"], 2
            )
            universe = read_tsv(output / "comparison_cell_universe.tsv")
            self.assertEqual(
                [row["cell_id"] for row in universe],
                [
                    "original|A1_1_0d00h00m|1",
                    "original|A1_1_0d00h00m|2",
                ],
            )

    def test_stable_identity_mismatch_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            reference_rows = read_tsv(fixture["reference"])
            reference_rows[1]["cell_id"] = "original|A1_1_0d00h00m|99"
            write_table(fixture["reference"], reference_rows)
            current_before = sha256(fixture["current"])
            reference_before = sha256(fixture["reference"])
            with self.assertRaisesRegex(ValueError, "stable-cell join/order mismatch"):
                self.run_normalized(fixture)
            self.assertFalse((fixture["output"] / "comparison_receipt.json").exists())
            self.assertEqual(sha256(fixture["current"]), current_before)
            self.assertEqual(sha256(fixture["reference"]), reference_before)

    def test_unfrozen_current_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            rows = read_tsv(fixture["current"])
            rows[0]["current_final_state"] = "transitional"
            write_table(fixture["current"], rows)
            with self.assertRaisesRegex(ValueError, "Unknown current final state"):
                self.run_normalized(fixture)
            self.assertFalse((fixture["output"] / "comparison_receipt.json").exists())

    def test_output_cannot_be_inside_an_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            fixture["output"] = fixture["current"].parent / "comparison"
            with self.assertRaisesRegex(
                ValueError, "must not write into an input directory"
            ):
                self.run_normalized(fixture)


if __name__ == "__main__":
    unittest.main()
