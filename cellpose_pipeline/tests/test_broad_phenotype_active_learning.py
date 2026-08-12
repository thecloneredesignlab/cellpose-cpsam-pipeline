from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cellpose_pipeline" / "scripts" / "24_prepare_broad_phenotype_active_learning_project.py"
RUNNER_SCRIPT = ROOT / "cellpose_pipeline" / "scripts" / "20_run_cellphenotypeannotator_stage.py"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACTIVE = load_script("broad_phenotype_active_learning_test", SCRIPT)
RUNNER = load_script("broad_phenotype_active_learning_runner_test", RUNNER_SCRIPT)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class BroadPhenotypeActiveLearningTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        shadow = root / "shadow"
        representative = shadow / "projection_input" / "representative_umap"
        representative.mkdir(parents=True)
        runs = representative / "runs"
        runs.mkdir()
        project_id = "fixture_development"
        run_id = "run_fixture"
        class_sha = "class_fixture_sha"
        classifier_sha = "classifier_fixture_sha"
        feature_columns = ["area_px2", "bf_object_mean"]
        class_ids = ["round", "flat"]
        cell_ids = [f"original|A01_1_00d00h00m|{index}" for index in range(1, 5)]

        classes = representative / "classes.tsv"
        write_tsv(
            classes,
            ["class_id", "display_name", "color", "trainable", "role", "order"],
            [
                {"class_id": "round", "display_name": "Round", "color": "#00aa00", "trainable": "true", "role": "phenotype", "order": 1},
                {"class_id": "flat", "display_name": "Flat", "color": "#aa0000", "trainable": "true", "role": "phenotype", "order": 2},
            ],
        )
        cells = representative / "cells.tsv"
        write_tsv(
            cells,
            ["cell_id", "image_id", "well", "split"],
            [
                {"cell_id": cell_id, "image_id": "A01_1_00d00h00m", "well": "A01", "split": "development"}
                for cell_id in cell_ids
            ],
        )
        features = representative / "features.tsv"
        write_tsv(
            features,
            ["cell_id", *feature_columns],
            [
                {"cell_id": cell_id, "area_px2": 10 + index, "bf_object_mean": index / 10}
                for index, cell_id in enumerate(cell_ids, 1)
            ],
        )
        images = representative / "images.tsv"
        write_tsv(images, ["image_id", "channel_id"], [{"image_id": "A01_1_00d00h00m", "channel_id": "brightfield"}])
        project = representative / "project.yml"
        project.write_text(
            json.dumps(
                {
                    "schema_version": "cell_phenotype_annotator_project_v1",
                    "project_id": project_id,
                    "classes_file": "classes.tsv",
                    "cells_file": "cells.tsv",
                    "features_file": "features.tsv",
                    "images_file": "images.tsv",
                    "runs_dir": "runs",
                    "projection": {"mode": "compute_umap", "feature_columns": feature_columns},
                    "annotation": {"title": "Fixture"},
                    "review": {
                        "strata": ["well"],
                        "group_column": "image_id",
                        "quotas": {"default_per_class": 2},
                        "max_per_group": 3,
                        "seed": 1,
                        "displays": [],
                    },
                    "classifier": {
                        "feature_columns": feature_columns,
                        "group_column": "well",
                        "allow_ungrouped": False,
                    },
                }
            )
        )

        model_dir = runs / project_id / run_id / "model" / "model_fixture"
        model_dir.mkdir(parents=True)
        model_hashes: dict[str, str] = {}
        for role, filename in ACTIVE.MODEL_ARTIFACTS.items():
            artifact = model_dir / filename
            artifact.write_text(f"fixture {role}\n")
            model_hashes[role] = ACTIVE.sha256_file(artifact)
        model_manifest = model_dir / "model_manifest.json"
        model_manifest.write_text(
            json.dumps(
                {
                    "schema_version": ACTIVE.CLASSIFIER_SCHEMA_VERSION,
                    "model_id": "model_fixture",
                    "project_id": project_id,
                    "run_id": run_id,
                    "class_config_sha256": class_sha,
                    "classifier_config_sha256": classifier_sha,
                    "classifier_config": {
                        "feature_columns": feature_columns,
                        "group_column": "well",
                        "allow_ungrouped": False,
                    },
                    "feature_columns": feature_columns,
                    "class_ids": class_ids,
                    "artifact_file_sha256": model_hashes,
                },
                sort_keys=True,
            )
        )

        prediction_dir = model_dir / "predictions" / "prediction_fixture"
        prediction_dir.mkdir(parents=True)
        predictions = prediction_dir / "all_cell_predictions.tsv"
        prediction_rows = []
        probability_rows = []
        for index, cell_id in enumerate(cell_ids):
            round_probability = 0.55 + index * 0.1
            flat_probability = 1 - round_probability
            predicted = "round" if round_probability >= flat_probability else "flat"
            prediction_rows.append(
                {"model_id": "model_fixture", "cell_id": cell_id, "predicted_class_id": predicted, "prediction_status": "ok"}
            )
            for class_id, probability in (("round", round_probability), ("flat", flat_probability)):
                probability_rows.append(
                    {"model_id": "model_fixture", "cell_id": cell_id, "class_id": class_id, "probability": probability, "prediction_status": "ok"}
                )
        write_tsv(predictions, ["model_id", "cell_id", "predicted_class_id", "prediction_status"], prediction_rows)
        probabilities = prediction_dir / "all_cell_probabilities.tsv"
        write_tsv(probabilities, ["model_id", "cell_id", "class_id", "probability", "prediction_status"], probability_rows)
        prediction_manifest = prediction_dir / "prediction_manifest.json"
        prediction_manifest.write_text(
            json.dumps(
                {
                    "schema_version": ACTIVE.CLASSIFIER_SCHEMA_VERSION,
                    "prediction_id": prediction_dir.name,
                    "model_id": "model_fixture",
                    "model_sha256": ACTIVE.sha256_file(model_dir / "model.rds"),
                    "model_manifest_sha256": ACTIVE.sha256_file(model_manifest),
                    "project_id": project_id,
                    "run_id": run_id,
                    "classifier_config_sha256": classifier_sha,
                    "artifact_file_sha256": {
                        "all_cell_predictions": ACTIVE.sha256_file(predictions),
                        "all_cell_probabilities": ACTIVE.sha256_file(probabilities),
                    },
                },
                sort_keys=True,
            )
        )

        history = shadow / "review_history" / "accepted" / "review_submission_fixture"
        history.mkdir(parents=True)
        submission = history / "review_submission.json"
        submission.write_text("{}\n")
        reviewed_labels = history / "reviewed_labels.tsv"
        write_tsv(
            reviewed_labels,
            ["cell_id", "review_status", "class_id", "reviewer_id", "reviewed_at", "confidence"],
            [
                {"cell_id": cell_ids[0], "review_status": "confirmed", "class_id": "round", "reviewer_id": "tester", "reviewed_at": "2026-08-12T00:00:00Z", "confidence": 1},
                {"cell_id": cell_ids[1], "review_status": "unreviewed", "class_id": "", "reviewer_id": "", "reviewed_at": "", "confidence": ""},
            ],
        )
        review_manifest = history / "review_import_manifest.json"
        review_manifest.write_text(
            json.dumps(
                {
                    "schema_version": ACTIVE.REVIEW_IMPORT_SCHEMA_VERSION,
                    "identity": {"project_id": project_id, "run_id": run_id, "class_config_sha256": class_sha},
                    "submission_id": "review_submission_fixture",
                    "artifact_file_sha256": {
                        "review_submission": ACTIVE.sha256_file(submission),
                        "reviewed_labels": ACTIVE.sha256_file(reviewed_labels),
                    },
                },
                sort_keys=True,
            )
        )

        umap_dir = runs / project_id / run_id / "projection" / "projection_fixture"
        umap_dir.mkdir(parents=True)
        umap_hashes: dict[str, str] = {}
        for role, filename in ACTIVE.UMAP_ARTIFACTS.items():
            artifact = umap_dir / filename
            if role == "umap":
                write_tsv(
                    artifact,
                    ["cell_id", "Dim1", "Dim2"],
                    [{"cell_id": cell_id, "Dim1": index, "Dim2": -index} for index, cell_id in enumerate(cell_ids, 1)],
                )
            else:
                artifact.write_text(f"fixture {role}\n")
            umap_hashes[role] = ACTIVE.sha256_file(artifact)
        (umap_dir / "umap_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": ACTIVE.UMAP_SCHEMA_VERSION,
                    "project_id": project_id,
                    "run_id": run_id,
                    "projection_id": umap_dir.name,
                    "artifact_file_sha256": umap_hashes,
                },
                sort_keys=True,
            )
        )
        return {
            "shadow": shadow,
            "project": project,
            "prediction_dir": prediction_dir,
            "reviewed_labels": reviewed_labels,
            "cells": cells,
        }

    def argv(self, fixture: dict[str, Path]) -> list[str]:
        return [
            "--project", str(fixture["project"]),
            "--shadow-root", str(fixture["shadow"]),
            "--prediction-dir", str(fixture["prediction_dir"]),
            "--reviewed-labels", str(fixture["reviewed_labels"]),
            "--queue-id", "uncertainty_v1",
            "--max-cells", "2",
            "--max-per-group", "2",
            "--seed", "20260812",
        ]

    def test_active_learning_project_is_frozen_and_development_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            self.assertEqual(ACTIVE.main(self.argv(fixture)), 0)
            output = fixture["shadow"] / "active_learning" / "uncertainty_v1"
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(ACTIVE.main(self.argv(fixture)), 0)
            self.assertEqual(first, {path.name: path.read_bytes() for path in output.iterdir()})

            project = json.loads((output / "project.yml").read_text())
            queue = project["review"]["queue"]
            self.assertEqual(queue["strategy"], "model_uncertainty")
            self.assertEqual(queue["max_cells"], 2)
            self.assertFalse(queue["allow_subset"])
            self.assertNotIn("quotas", project["review"])
            self.assertEqual(project["classifier"]["feature_columns"], ["area_px2", "bf_object_mean"])
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["heldout_cell_count"], 0)
            self.assertFalse(receipt["automatic_cross_review_id_merge"])
            self.assertFalse(receipt["automatic_retraining"])
            self.assertEqual(receipt["input_identity"]["history_excluded_cell_count"], 1)

    def test_heldout_and_tampered_prediction_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            rows: list[dict[str, str]] = []
            with fixture["cells"].open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["split"] = "heldout"
            write_tsv(fixture["cells"], fields, rows)
            with self.assertRaisesRegex(ValueError, "Heldout"):
                ACTIVE.main(self.argv(fixture))

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            prediction = fixture["prediction_dir"] / "all_cell_predictions.tsv"
            prediction.write_text(prediction.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                ACTIVE.main(self.argv(fixture))

    def test_review_history_identity_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            manifest_path = fixture["reviewed_labels"].parent / "review_import_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["identity"]["run_id"] = "run_other"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "parent identity mismatch"):
                ACTIVE.main(self.argv(fixture))

    def test_shadow_internal_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            alias = fixture["shadow"] / "prediction_alias"
            alias.symlink_to(fixture["prediction_dir"], target_is_directory=True)
            argv = self.argv(fixture)
            argv[argv.index("--prediction-dir") + 1] = str(alias)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                ACTIVE.main(argv)

    def test_stage_wrapper_hashes_queue_inputs_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            ACTIVE.main(self.argv(fixture))
            project_path = (
                fixture["shadow"] / "active_learning" / "uncertainty_v1" / "project.yml"
            )
            project = json.loads(project_path.read_text())
            before = RUNNER.review_queue_input_identities(
                project_path, project, fixture["shadow"]
            )
            self.assertTrue(any(row["role"].startswith("queue_prediction") for row in before))
            self.assertTrue(any(row["role"].startswith("queue_review_history") for row in before))

            probability = fixture["prediction_dir"] / "all_cell_probabilities.tsv"
            probability.write_text(probability.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                RUNNER.review_queue_input_identities(
                    project_path, project, fixture["shadow"]
                )

    def test_stage_wrapper_binds_exact_review_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            review_dir = shadow / "runs" / "review_fixture"
            review_dir.mkdir(parents=True)
            review_id = "review_fixture"
            manifest = review_dir / "review_manifest.json"
            manifest.write_text(json.dumps({"identity": {"review_id": review_id}}))
            stdout = "\n".join(
                [
                    f"review_id: {review_id}",
                    "selection_mode: active_queue",
                    f"manifest: {manifest}",
                ]
            )
            identity = RUNNER.stage_output_identity(
                Namespace(stage="review-build", check_config=False, dry_run=False),
                stdout,
                shadow,
            )
            self.assertEqual(identity["review_id"], review_id)
            self.assertEqual(identity["selection_mode"], "active_queue")
            self.assertEqual(identity["files"][0]["sha256"], RUNNER.sha256_file(manifest))


if __name__ == "__main__":
    unittest.main()
