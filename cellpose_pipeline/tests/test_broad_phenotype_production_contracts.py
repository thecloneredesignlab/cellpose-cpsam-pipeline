from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    REPO_ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "23_validate_broad_phenotype_production_contracts.py"
)
FEATURE_CONFIG = (
    REPO_ROOT / "cellpose_pipeline" / "configs" / "broad_phenotype_features_v1.json"
)
CLASSES = REPO_ROOT / "cellpose_pipeline" / "configs" / "broad_phenotype_classes_v1.tsv"
MODEL_ARTIFACTS = {
    "model": "model.rds",
    "training_rows": "training_rows.tsv",
    "outer_folds": "outer_fold_assignments.tsv",
    "fold_summary": "fold_summary.tsv",
    "fold_metrics_by_class": "fold_metrics_by_class.tsv",
    "feature_preprocessor": "feature_preprocessor_manifest.tsv",
    "out_of_fold_predictions": "out_of_fold_predictions.tsv",
    "out_of_fold_probabilities": "out_of_fold_probabilities.tsv",
    "metrics_by_class": "metrics_by_class.tsv",
    "metrics_summary": "metrics_summary.tsv",
    "confusion_matrix": "confusion_matrix.tsv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object, *, ensure_ascii: bool = True) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
        ).encode()
    ).hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


class ModelAcceptanceContractTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path | str]:
        shadow = root / "shadow"
        representative = shadow / "projection_input" / "representative_umap"
        representative.mkdir(parents=True)
        classes = representative / "classes.tsv"
        classes.write_bytes(CLASSES.read_bytes())
        feature_config = root / "feature_config.json"
        feature_config.write_bytes(FEATURE_CONFIG.read_bytes())
        config = json.loads(feature_config.read_text())
        primary = config["primary_model_feature_columns"]
        cells = representative / "cells.tsv"
        features = representative / "features.tsv"
        images = representative / "images.tsv"
        write_tsv(cells, ["cell_id"], [["original|A01_1_0d0h0m|1"]])
        write_tsv(features, ["cell_id", *primary], [["original|A01_1_0d0h0m|1", *([1] * len(primary))]])
        write_tsv(images, ["image_id"], [["A01_1_0d0h0m"]])
        project_id, run_id, model_id = "broad_phenotype_development", "run_fixture", "model_fixture"
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
                    "classifier": {"feature_columns": primary},
                },
                indent=2,
            )
            + "\n"
        )
        reviewed = representative / "runs" / project_id / run_id / "review" / "reviewed_labels.tsv"
        reviewed.parent.mkdir(parents=True)
        reviewed.write_text("cell_id\noriginal|A01_1_0d0h0m|1\n")
        class_config_sha = "c" * 64
        review_manifest = reviewed.parent / "review_import_manifest.json"
        review_manifest.write_text(
            json.dumps(
                {
                    "identity": {
                        "project_id": project_id,
                        "run_id": run_id,
                        "class_config_sha256": class_config_sha,
                    }
                },
                sort_keys=True,
            )
            + "\n"
        )
        stdout_log = shadow / "workflow_status" / "cpa_stages" / "train.stdout.log"
        stderr_log = shadow / "workflow_status" / "cpa_stages" / "train.stderr.log"
        stdout_log.parent.mkdir(parents=True)
        stdout_log.write_text("trained\n")
        stderr_log.write_text("")
        files = []
        for role, path in (
            ("dispatch_project", project),
            ("classes_file", classes),
            ("cells_file", cells),
            ("features_file", features),
            ("images_file", images),
            ("review_import_generation:reviewed_labels.tsv", reviewed),
            ("review_import_generation:review_import_manifest.json", review_manifest),
        ):
            files.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        identity = {
            "schema_version": "cellphenotypeannotator_stage_input_identity_v1",
            "dependency": {"fixture": True},
            "files": files,
        }
        identity["sha256"] = canonical_hash(identity)
        command = [
            "Rscript",
            "cell-phenotype-annotator",
            "train",
            str(project.resolve()),
            "--reviewed-labels",
            str(reviewed.resolve()),
        ]
        invocation = {"command": command, "input_identity": identity}
        train_receipt = stdout_log.parent / "train.run.fixture.json"
        train_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "cellphenotypeannotator_stage_receipt_v1",
                    "stage": "train",
                    "status": "complete",
                    "returncode": 0,
                    "shadow_root": str(shadow.resolve()),
                    "project": str(project.resolve()),
                    "project_sha256": sha256(project),
                    "command": command,
                    "command_sha256": canonical_hash(command, ensure_ascii=False),
                    "invocation_sha256": canonical_hash(invocation, ensure_ascii=False),
                    "input_identity": identity,
                    "stdout_log": str(stdout_log.resolve()),
                    "stdout_sha256": sha256(stdout_log),
                    "stderr_log": str(stderr_log.resolve()),
                    "stderr_sha256": sha256(stderr_log),
                },
                sort_keys=True,
            )
            + "\n"
        )
        preflight = shadow / "workflow_status" / "submission_preflight.tsv"
        write_tsv(
            preflight,
            ["property", "value"],
            [
                ["feature_config_sha256", sha256(feature_config)],
                ["classes_sha256", sha256(classes)],
            ],
        )
        model_dir = representative / "runs" / project_id / run_id / "model" / model_id
        model_dir.mkdir(parents=True)
        artifact_hashes = {}
        for role, filename in MODEL_ARTIFACTS.items():
            path = model_dir / filename
            path.write_text(f"{role}\n")
            artifact_hashes[role] = sha256(path)
        class_ids = sorted(
            row["class_id"]
            for row in csv.DictReader(CLASSES.read_text().splitlines(), delimiter="\t")
            if row["trainable"].lower() == "true"
        )
        classifier_config_sha = "d" * 64
        model_manifest = model_dir / "model_manifest.json"
        model_manifest.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "project_id": project_id,
                    "run_id": run_id,
                    "class_config_sha256": class_config_sha,
                    "classifier_config_sha256": classifier_config_sha,
                    "feature_columns": primary,
                    "class_ids": class_ids,
                    "artifact_file_sha256": artifact_hashes,
                },
                sort_keys=True,
            )
            + "\n"
        )
        acceptance = shadow / "workflow_status" / "model_acceptance" / "model_acceptance.json"
        acceptance.parent.mkdir(parents=True)
        acceptance.write_text(
            json.dumps(
                {
                    "schema_version": "broad_phenotype_model_acceptance_v1",
                    "status": "ACCEPTED",
                    "accepted": True,
                    "shadow_root": str(shadow.resolve()),
                    "project": {
                        "path": str(project.resolve()),
                        "sha256": sha256(project),
                        "project_id": project_id,
                        "run_id": run_id,
                        "class_config_sha256": class_config_sha,
                        "classifier_config_sha256": classifier_config_sha,
                        "feature_columns": primary,
                        "class_ids": class_ids,
                    },
                    "model": {
                        "dir": str(model_dir.resolve()),
                        "manifest": str(model_manifest.resolve()),
                        "manifest_sha256": sha256(model_manifest),
                        "model_id": model_id,
                        "project_id": project_id,
                        "run_id": run_id,
                        "class_config_sha256": class_config_sha,
                        "classifier_config_sha256": classifier_config_sha,
                        "feature_columns": primary,
                        "class_ids": class_ids,
                    },
                    "train_receipt": {
                        "path": str(train_receipt.resolve()),
                        "sha256": sha256(train_receipt),
                        "input_identity_sha256": identity["sha256"],
                        "reviewed_labels_sha256": sha256(reviewed),
                        "review_import_manifest_sha256": sha256(review_manifest),
                    },
                    "frozen_inputs": {
                        "submission_preflight": str(preflight.resolve()),
                        "submission_preflight_sha256": sha256(preflight),
                        "feature_config": str(feature_config.resolve()),
                        "feature_config_sha256": sha256(feature_config),
                        "classes_file": str(classes.resolve()),
                        "classes_sha256": sha256(classes),
                    },
                    "semantic_verification": {
                        "reference_generation_verified": True,
                        "expected_model_directory_verified": True,
                        "train_receipt_input_identity_verified": True,
                        "model_train_parent_verified": True,
                        "frozen_hashes_verified": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return {
            "shadow": shadow,
            "project": project,
            "model": model_dir,
            "model_manifest": model_manifest,
            "train_receipt": train_receipt,
            "feature_config": feature_config,
            "classes": classes,
            "preflight": preflight,
            "acceptance": acceptance,
        }

    def run_verify(self, fixture: dict[str, Path | str], expected_sha: str | None = None):
        acceptance = Path(fixture["acceptance"])
        return subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "verify-model-acceptance",
                "--shadow-root",
                str(fixture["shadow"]),
                "--project",
                str(fixture["project"]),
                "--model-dir",
                str(fixture["model"]),
                "--train-receipt",
                str(fixture["train_receipt"]),
                "--feature-config",
                str(fixture["feature_config"]),
                "--classes-file",
                str(fixture["classes"]),
                "--submission-preflight",
                str(fixture["preflight"]),
                "--acceptance-receipt",
                str(acceptance),
                "--expected-sha256",
                expected_sha or sha256(acceptance),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_acceptance_positive_and_receipt_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            accepted = self.run_verify(fixture)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            frozen_sha = sha256(Path(fixture["acceptance"]))
            with Path(fixture["acceptance"]).open("a") as handle:
                handle.write(" \n")
            rejected = self.run_verify(fixture, expected_sha=frozen_sha)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("receipt SHA mismatch", rejected.stderr)

    def test_same_shadow_wrong_model_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            wrong = Path(fixture["model"]).parent / "model_wrong"
            wrong.mkdir()
            for path in Path(fixture["model"]).iterdir():
                wrong.joinpath(path.name).write_bytes(path.read_bytes())
            fixture["model"] = wrong
            rejected = self.run_verify(fixture)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("canonical model generation", rejected.stderr)

    def test_different_project_model_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            manifest_path = Path(fixture["model_manifest"])
            manifest = json.loads(manifest_path.read_text())
            manifest["project_id"] = "different_project"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            acceptance_path = Path(fixture["acceptance"])
            acceptance = json.loads(acceptance_path.read_text())
            acceptance["model"]["manifest_sha256"] = sha256(manifest_path)
            acceptance["model"]["project_id"] = "different_project"
            acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
            rejected = self.run_verify(fixture)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("project_id differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
