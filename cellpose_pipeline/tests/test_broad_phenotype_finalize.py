import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "cellpose_pipeline" / "scripts" / "19_finalize_broad_phenotype_shadow.py"
SPEC = importlib.util.spec_from_file_location("broad_phenotype_finalize", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FINALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINALIZE
SPEC.loader.exec_module(FINALIZE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class BroadPhenotypeFinalizeTests(unittest.TestCase):
    def build_fixture(self, root: Path, probability_override: str | None = None) -> dict[str, Path]:
        cpa = root / "cpa"
        cpa.mkdir(parents=True)
        project = cpa / "project.yml"
        project.write_text(
            "schema_version: cell_phenotype_annotator_project_v1\n"
            "project_id: broad_shadow_test\n"
            "classes_file: classes.tsv\n"
            "cells_file: cells.tsv\n"
            "runs_dir: ../runs\n"
        )
        write_tsv(
            cpa / "classes.tsv",
            [
                {"class_id": "class_a", "display_name": "A", "color": "#000000", "trainable": "true", "role": "phenotype", "order": 1},
                {"class_id": "class_b", "display_name": "B", "color": "#ffffff", "trainable": "true", "role": "phenotype", "order": 2},
                {"class_id": "uncertain", "display_name": "U", "color": "#999999", "trainable": "false", "role": "uncertainty", "order": 3},
            ],
        )
        write_tsv(
            cpa / "cells.tsv",
            [
                {"cell_id": "original|A01_1_0d0h0m|1", "image_id": "A01_1_0d0h0m", "mask_label": 1},
                {"cell_id": "original|A01_1_0d0h0m|2", "image_id": "A01_1_0d0h0m", "mask_label": 2},
                {"cell_id": "original|A01_1_0d0h0m|3", "image_id": "A01_1_0d0h0m", "mask_label": 3},
            ],
        )
        model_dir = root / "runs" / "model_test"
        prediction_dir = model_dir / "predictions" / "prediction_test"
        prediction_dir.mkdir(parents=True)
        (model_dir / "model.rds").write_bytes(b"model")
        write_tsv(
            prediction_dir / "all_cell_predictions.tsv",
            [
                {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|1", "predicted_class_id": "class_a", "prediction_status": "ok"},
                {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|2", "predicted_class_id": "class_b", "prediction_status": "ok"},
                {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|3", "predicted_class_id": "", "prediction_status": "unavailable_missing_features"},
            ],
        )
        probabilities = [
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|1", "class_id": "class_a", "probability": "0.8", "prediction_status": "ok"},
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|2", "class_id": "class_a", "probability": "0.2", "prediction_status": "ok"},
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|3", "class_id": "class_a", "probability": "NA", "prediction_status": "unavailable_missing_features"},
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|1", "class_id": "class_b", "probability": "0.2", "prediction_status": "ok"},
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|2", "class_id": "class_b", "probability": probability_override or "0.8", "prediction_status": "ok"},
            {"model_id": "model_test", "cell_id": "original|A01_1_0d0h0m|3", "class_id": "class_b", "probability": "NA", "prediction_status": "unavailable_missing_features"},
        ]
        write_tsv(prediction_dir / "all_cell_probabilities.tsv", probabilities)
        model_manifest = {
            "model_id": "model_test",
            "project_id": "broad_shadow_test",
            "class_ids": ["class_a", "class_b"],
            "artifact_file_sha256": {
                "model": sha256(model_dir / "model.rds"),
            },
        }
        (model_dir / "model_manifest.json").write_text(json.dumps(model_manifest))
        prediction_manifest = {
            "model_id": "model_test",
            "project_id": "broad_shadow_test",
            "model_sha256": sha256(model_dir / "model.rds"),
            "model_manifest_sha256": sha256(model_dir / "model_manifest.json"),
            "artifact_file_sha256": {
                "all_cell_predictions": sha256(prediction_dir / "all_cell_predictions.tsv"),
                "all_cell_probabilities": sha256(prediction_dir / "all_cell_probabilities.tsv"),
            },
        }
        (prediction_dir / "prediction_manifest.json").write_text(json.dumps(prediction_manifest))
        legacy = root / "legacy.json"
        legacy.write_text(json.dumps({"decision": "NO_GO", "metric": "legacy"}))
        return {
            "project": project,
            "model_dir": model_dir,
            "prediction_dir": prediction_dir,
            "legacy": legacy,
        }

    def run_main(self, root: Path, fixture: dict[str, Path]) -> None:
        arguments = [
            str(SCRIPT),
            "--shadow-root", str(root),
            "--project", str(fixture["project"]),
            "--prediction-dir", str(fixture["prediction_dir"]),
            "--model-dir", str(fixture["model_dir"]),
            "--legacy-no-go", str(fixture["legacy"]),
        ]
        with unittest.mock.patch.object(sys, "argv", arguments):
            self.assertEqual(FINALIZE.main(), 0)

    def test_exact_coverage_and_shadow_only_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_fixture(root)
            self.run_main(root, fixture)
            receipt = json.loads((root / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json").read_text())
            self.assertEqual(receipt["technical_decision"], "GO")
            self.assertEqual(receipt["promotion_decision"], "NO_GO")
            self.assertEqual(receipt["overall_decision"], "SHADOW_ONLY")
            self.assertTrue(receipt["legacy_viability_gate"]["ignored"])
            self.assertEqual(receipt["legacy_viability_gate"]["recorded_decision"], "NO_GO")
            self.assertFalse(receipt["published_shadow_axis"]["overwrites_viability_state"])
            self.assertEqual(receipt["prediction_generation"]["cell_count"], 3)
            self.assertEqual(receipt["prediction_generation"]["unavailable_count"], 1)
            output = root / "predictions" / "broad_phenotype_predictions.tsv"
            self.assertTrue(output.is_file())
            self.assertIn("broad_phenotype_class_id", output.read_text().splitlines()[0])

    def test_probability_sum_mismatch_fails_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_fixture(root, probability_override="0.7")
            with self.assertRaisesRegex(ValueError, "sum to one"):
                self.run_main(root, fixture)
            self.assertFalse((root / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json").exists())
            self.assertFalse((root / "predictions" / "broad_phenotype_predictions.tsv").exists())

    def test_json_project_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_fixture(root)
            fixture["project"].write_text(
                json.dumps(
                    {
                        "schema_version": "cell_phenotype_annotator_project_v1",
                        "project_id": "broad_shadow_test",
                        "classes_file": "classes.tsv",
                        "cells_file": "cells.tsv",
                        "runs_dir": "../runs",
                    }
                )
            )
            self.run_main(root, fixture)
            self.assertTrue((root / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json").is_file())

    def test_prediction_identity_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = self.build_fixture(root)
            path = fixture["prediction_dir"] / "all_cell_predictions.tsv"
            text = path.read_text().replace("original|A01_1_0d0h0m|2", "wrong", 1)
            path.write_text(text)
            manifest_path = fixture["prediction_dir"] / "prediction_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_file_sha256"]["all_cell_predictions"] = sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.run_main(root, fixture)


if __name__ == "__main__":
    unittest.main()
