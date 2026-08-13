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
        return {
            "current_root": current.parent,
            "current": current,
            "reference": reference,
            "output": output,
        }

    def build_manifest_fixture(self, root: Path) -> dict[str, Path]:
        current_root = root / "current_axis"
        feature = current_root / "features" / "A1_1_0d00h00m_features.csv"
        prediction = (
            current_root / "predictions" / "A1_1_0d00h00m_predictions.csv"
        )
        manifest = current_root / "manifests" / "manifest.tsv"
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
                }
            ],
            delimiter=",",
        )
        write_table(
            prediction,
            [{"key": "A1_1_0d00h00m", "mask_id": 1, "state": "live"}],
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
                }
            ],
        )
        return {
            "current_root": current_root,
            "feature": feature,
            "prediction": prediction,
            "manifest": manifest,
            "reference": reference,
            "output": output,
        }

    def run_normalized(self, fixture: dict[str, Path]) -> int:
        return COMPARE.main(
            [
                "--current-root",
                str(fixture["current_root"]),
                "--current-predictions",
                str(fixture["current"]),
                "--reference-predictions",
                str(fixture["reference"]),
                "--output-root",
                str(fixture["output"]),
            ]
        )

    def run_manifest(self, fixture: dict[str, Path]) -> int:
        return COMPARE.main(
            [
                "--current-root",
                str(fixture["current_root"]),
                "--current-manifest",
                str(fixture["manifest"]),
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
            self.assertEqual(receipt["implementation"], str(SCRIPT.resolve()))
            self.assertEqual(receipt["implementation_sha256"], sha256(SCRIPT))
            self.assertEqual(
                receipt["inputs"]["current_root"]["path"],
                str(fixture["current_root"]),
            )
            self.assertRegex(
                receipt["inputs"]["current_root"][
                    "confined_input_binding_sha256"
                ],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                receipt["inputs"]["current_root"]["identity_sha256"],
                receipt["inputs"]["current_root"][
                    "confined_input_binding_sha256"
                ],
            )
            self.assertFalse(
                receipt["inputs"]["current_root"]["symlink_components_allowed"]
            )
            self.assertNotIn("device", receipt["inputs"]["current_root"])
            self.assertNotIn("inode", receipt["inputs"]["current_root"])

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

            self.assertEqual(self.run_normalized(fixture), 0)
            receipt_path = output / "comparison_receipt.json"
            original_receipt = receipt_path.read_bytes()
            tampered = json.loads(original_receipt)
            tampered["implementation_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(tampered) + "\n")
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                self.run_normalized(fixture)
            self.assertEqual(json.loads(receipt_path.read_text())["implementation_sha256"], "0" * 64)

    def test_partial_generation_is_preserved_and_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            fixture["output"].mkdir()
            sentinel = fixture["output"] / "join_audit.json"
            sentinel.write_text("partial\n")
            with self.assertRaisesRegex(RuntimeError, "Incomplete or unexpected"):
                self.run_normalized(fixture)
            self.assertEqual(sentinel.read_text(), "partial\n")
            self.assertEqual(list(fixture["output"].iterdir()), [sentinel])

    def test_manifest_mode_verifies_feature_prediction_identity_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            current_root = root / "current_axis"
            feature = current_root / "current_features" / "A1_1_0d00h00m_features.csv"
            prediction = current_root / "current_predictions" / "A1_1_0d00h00m_predictions.csv"
            manifest = current_root / "current_manifest" / "manifest.tsv"
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
                        "--current-root",
                        str(current_root),
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
            source_identities = receipt["inputs"]["current_root"][
                "confined_source_files"
            ]
            self.assertEqual(
                [item["role"] for item in source_identities],
                ["feature", "prediction"],
            )
            self.assertEqual(
                [item["sha256"] for item in source_identities],
                [sha256(feature), sha256(prediction)],
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
                ValueError, "must not write into the frozen current root"
            ):
                self.run_normalized(fixture)

    def test_current_root_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_normalized_fixture(Path(temporary).resolve())
            with self.assertRaises(SystemExit) as raised:
                COMPARE.parse_args(
                    [
                        "--current-predictions",
                        str(fixture["current"]),
                        "--reference-predictions",
                        str(fixture["reference"]),
                        "--output-root",
                        str(fixture["output"]),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)

    def test_current_root_binding_is_portable_content_identity(self) -> None:
        root = Path("/frozen/current-axis")
        current = root / "manifest.tsv"
        sources = [
            {
                "path": str(root / "features" / "field.csv"),
                "sha256": "2" * 64,
                "role": "feature",
            }
        ]
        observed = COMPARE.current_root_binding_sha256(
            root, current, "1" * 64, sources
        )
        expected_payload = {
            "current_input_path": str(current),
            "current_input_sha256": "1" * 64,
            "current_root_path": str(root),
            "confined_source_files": sources,
        }
        expected = hashlib.sha256(
            json.dumps(
                expected_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(observed, expected)
        self.assertNotIn("device", json.dumps(expected_payload))
        self.assertNotIn("inode", json.dumps(expected_payload))

    def test_primary_current_input_cannot_escape_current_root(self) -> None:
        for mode in ("normalized", "manifest"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                fixture = self.build_normalized_fixture(root)
                confined_root = root / "different_current_root"
                confined_root.mkdir()
                fixture["current_root"] = confined_root
                with self.assertRaisesRegex(ValueError, "escaped the frozen current root"):
                    if mode == "normalized":
                        self.run_normalized(fixture)
                    else:
                        fixture["manifest"] = fixture["current"]
                        self.run_manifest(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_manifest_source_file_cannot_escape_current_root(self) -> None:
        for role in ("feature", "prediction"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                fixture = self.build_manifest_fixture(root)
                escaped = root / "outside" / fixture[role].name
                escaped.parent.mkdir()
                escaped.write_bytes(fixture[role].read_bytes())
                rows = read_tsv(fixture["manifest"])
                rows[0][f"{role}_path"] = str(escaped)
                rows[0][f"{role}_sha256"] = sha256(escaped)
                write_table(fixture["manifest"], rows)
                with self.assertRaisesRegex(
                    ValueError, "escaped the frozen current root"
                ):
                    self.run_manifest(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_current_inputs_reject_symlink_and_symlink_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_normalized_fixture(root)
            linked_input = fixture["current_root"] / "current_link.tsv"
            linked_input.symlink_to(fixture["current"].name)
            fixture["current"] = linked_input
            with self.assertRaisesRegex(ValueError, "has a symlink component"):
                self.run_normalized(fixture)
            self.assertFalse(fixture["output"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_manifest_fixture(root)
            linked_manifest = fixture["manifest"].with_name("manifest_link.tsv")
            linked_manifest.symlink_to(fixture["manifest"].name)
            fixture["manifest"] = linked_manifest
            with self.assertRaisesRegex(ValueError, "has a symlink component"):
                self.run_manifest(fixture)
            self.assertFalse(fixture["output"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_manifest_fixture(root)
            alias = fixture["current_root"] / "feature_alias"
            alias.symlink_to(fixture["feature"].parent, target_is_directory=True)
            linked_feature = alias / fixture["feature"].name
            rows = read_tsv(fixture["manifest"])
            rows[0]["feature_path"] = str(linked_feature)
            write_table(fixture["manifest"], rows)
            with self.assertRaisesRegex(ValueError, "has a symlink component"):
                self.run_manifest(fixture)
            self.assertFalse(fixture["output"].exists())


if __name__ == "__main__":
    unittest.main()
