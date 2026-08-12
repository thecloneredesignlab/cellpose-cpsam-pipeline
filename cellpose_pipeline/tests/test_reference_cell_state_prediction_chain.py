#!/usr/bin/env python3
"""Contract tests for the isolated reference-cell-state prediction chain."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ACCEPT = REPO / "cellpose_pipeline" / "scripts" / "30_accept_reference_cell_state_model.R"
PREDICT = REPO / "cellpose_pipeline" / "scripts" / "31_predict_reference_cell_state_shard.R"
MERGE = REPO / "cellpose_pipeline" / "scripts" / "32_merge_reference_cell_state_predictions.py"
LOCK = REPO / "cellpose_pipeline" / "configs" / "cellphenotypeannotator_dependency.lock.tsv"
REFERENCE = Path(
    os.environ.get(
        "CPA_REFERENCE_ROOT", "/Users/4482173/Documents/GitHub/cell-phenotype-annotator"
    )
).expanduser().resolve()
FEATURES = (
    "area_px2",
    "perimeter_px",
    "roundness",
    "aspect_ratio",
    "extent",
    "solidity",
    "equivalent_diameter_px",
    "major_axis_px",
    "minor_axis_px",
)
CLASSES = ("dead_cell", "live_cell", "multinucleated_cell")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_hash(cell_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(cell_ids).encode()).hexdigest()


def write_tsv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class ReferenceMergeTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        parent = root / "parent"
        shadow = root / "reference"
        parent.mkdir()
        shadow.mkdir()
        key, well, branch = "A1_1_0d0h0m", "A1", "original"
        cell_ids = [f"{branch}|{key}|1", f"{branch}|{key}|2"]
        feature_columns = [
            "cell_id", "key", "well", "branch", "image_id", "mask_label",
            "feature_schema_version", *FEATURES,
        ]
        feature = parent / "features" / f"{key}__{branch}_broad_phenotype_features.tsv"
        rows = []
        for label, cell_id in enumerate(cell_ids, 1):
            rows.append(
                [cell_id, key, well, branch, key, label, "broad_phenotype_features_v1"]
                + [label + index / 10 for index in range(len(FEATURES))]
            )
        write_tsv(feature, feature_columns, rows)
        feature_receipt = parent / "features" / f"{key}__{branch}.json"
        feature_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "broad_phenotype_feature_receipt_v1",
                    "status": "COMPLETE",
                    "key": key,
                    "well": well,
                    "branch": branch,
                    "feature_schema_version": "broad_phenotype_features_v1",
                    "feature_columns": feature_columns,
                    "feature_tsv": str(feature.resolve()),
                    "feature_tsv_sha256": sha256(feature),
                    "row_count": 2,
                    "cell_id_sha256": cell_hash(cell_ids),
                }
            )
            + "\n"
        )
        feature_manifest = parent / "workflow_status" / "feature_inventory" / "original_feature_manifest.tsv"
        write_tsv(
            feature_manifest,
            ["key", "feature_path", "receipt_path"],
            [[key, feature.resolve(), feature_receipt.resolve()]],
        )
        cells = parent / "cpa" / "cells.tsv"
        write_tsv(cells, ["cell_id"], [[cell_id] for cell_id in cell_ids])

        parent_import = shadow / "projection_input" / "representative_umap" / "parent_import_manifest.json"
        parent_import.parent.mkdir(parents=True)
        parent_import.write_text(
            json.dumps(
                {
                    "schema_version": "reference_cell_state_parent_import_v1",
                    "parent": {"shadow_root": str(parent.resolve())},
                }
            )
            + "\n"
        )
        acceptance = shadow / "workflow_status" / "model_acceptance" / "model_acceptance.json"
        acceptance.parent.mkdir(parents=True)
        acceptance_payload = {
            "schema_version": "reference_cell_state_model_acceptance_v1",
            "status": "ACCEPTED",
            "accepted": True,
            "shadow_root": str(shadow.resolve()),
            "model": {
                "model_id": "model_reference",
                "manifest_sha256": "b" * 64,
            },
            "dependency": {"lock_sha256": "c" * 64},
            "frozen_inputs": {
                "parent_import_manifest": str(parent_import.resolve()),
                "parent_import_manifest_sha256": sha256(parent_import),
                "parent_shadow_root": str(parent.resolve()),
            },
        }
        acceptance.write_text(json.dumps(acceptance_payload) + "\n")

        prediction_root = shadow / "prediction_shards"
        prediction = prediction_root / "shards" / well / f"{key}__{branch}_reference_cell_state_predictions.tsv"
        prediction_columns = [
            "model_id", "cell_id", "predicted_class_id", "prediction_status",
            *[f"probability__{class_id}" for class_id in CLASSES],
        ]
        write_tsv(
            prediction,
            prediction_columns,
            [
                ["model_reference", cell_ids[0], "dead_cell", "ok", ".8", ".1", ".1"],
                ["model_reference", cell_ids[1], "", "unavailable_missing_features", "", "", ""],
            ],
        )
        prediction_receipt = prediction_root / "receipts" / well / f"{key}__{branch}.json"
        prediction_receipt.parent.mkdir(parents=True)
        prediction_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "reference_cell_state_shard_prediction_v1",
                    "status": "COMPLETE",
                    "key": key,
                    "well": well,
                    "branch": branch,
                    "model_id": "model_reference",
                    "class_ids": list(CLASSES),
                    "feature_tsv_sha256": sha256(feature),
                    "feature_receipt_sha256": sha256(feature_receipt),
                    "model_sha256": "a" * 64,
                    "model_manifest_sha256": "b" * 64,
                    "model_acceptance_receipt": str(acceptance.resolve()),
                    "model_acceptance_sha256": sha256(acceptance),
                    "parent_import_manifest_sha256": sha256(parent_import),
                    "dependency_lock_sha256": "c" * 64,
                    "implementation_sha256": "d" * 64,
                    "source_identity": {"mode": "fixture"},
                    "runtime_identity": {"r_version": "fixture"},
                    "row_count": 2,
                    "cell_id_sha256": cell_hash(cell_ids),
                    "ok_count": 1,
                    "unavailable_count": 1,
                    "max_probability_sum_error": 0,
                    "output_columns": prediction_columns,
                    "prediction_tsv": str(prediction.resolve()),
                    "prediction_tsv_sha256": sha256(prediction),
                }
            )
            + "\n"
        )
        return {
            "parent": parent,
            "shadow": shadow,
            "parent_import": parent_import,
            "acceptance": acceptance,
            "feature_manifest": feature_manifest,
            "prediction_root": prediction_root,
            "cells": cells,
            "prediction_receipt": prediction_receipt,
        }

    def run_merge(self, fixture: dict[str, Path]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3", str(MERGE),
                "--feature-manifest", str(fixture["feature_manifest"]),
                "--prediction-root", str(fixture["prediction_root"]),
                "--shadow-root", str(fixture["shadow"]),
                "--parent-import-manifest", str(fixture["parent_import"]),
                "--model-acceptance-receipt", str(fixture["acceptance"]),
                "--model-acceptance-sha256", sha256(fixture["acceptance"]),
                "--cells", str(fixture["cells"]),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_merge_publishes_exact_independent_four_column_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            completed = self.run_merge(fixture)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = fixture["shadow"] / "predictions" / "reference_cell_state_predictions.tsv"
            rows = read_tsv(output)
            self.assertEqual(
                list(rows[0]),
                ["model_id", "cell_id", "reference_cell_state_class_id", "prediction_status"],
            )
            self.assertEqual(rows[0]["reference_cell_state_class_id"], "dead_cell")
            self.assertEqual(rows[1]["prediction_status"], "unavailable_missing_features")
            receipt = json.loads(
                (fixture["shadow"] / "REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json").read_text()
            )
            self.assertEqual(receipt["published_shadow_axis"]["semantic_axis"], "reference_cell_state")
            self.assertFalse(receipt["isolation_contract"]["current_classification_read"])
            repeated = self.run_merge(fixture)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("already_complete=1", repeated.stdout)

    def test_merge_rejects_wrong_class_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            receipt = json.loads(fixture["prediction_receipt"].read_text())
            receipt["class_ids"] = ["live_cell", "dead_cell", "multinucleated_cell"]
            fixture["prediction_receipt"].write_text(json.dumps(receipt) + "\n")
            completed = self.run_merge(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("wrong reference class_ids", completed.stderr)

    def test_merge_rejects_feature_manifest_outside_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            outside = Path(directory) / "outside.tsv"
            outside.write_text(fixture["feature_manifest"].read_text())
            fixture["feature_manifest"] = outside
            completed = self.run_merge(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must resolve inside", completed.stderr)


class StaticIsolationContractTests(unittest.TestCase):
    def test_new_scripts_do_not_mutate_legacy_implementations(self) -> None:
        accept_text = ACCEPT.read_text()
        predict_text = PREDICT.read_text()
        merge_text = MERGE.read_text()
        self.assertIn('classifier_feature_columns', accept_text)
        self.assertIn('REFERENCE_FEATURES <- c(', accept_text)
        self.assertIn('31_predict_reference_cell_state_shard.R', predict_text)
        self.assertNotIn('21_predict_broad_phenotype_shard.R', predict_text)
        self.assertIn('REFERENCE_CLASS_IDS <- c("dead_cell", "live_cell", "multinucleated_cell")', predict_text)
        self.assertNotIn('--legacy-no-go', merge_text)
        self.assertIn('reference_cell_state_predictions.tsv', merge_text)
        self.assertIn('current_classification_read', merge_text)


class ReferencePredictionRuntimeTests(unittest.TestCase):
    def prerequisites(self) -> str | None:
        if shutil.which("Rscript") is None:
            return "Rscript is unavailable"
        if not REFERENCE.is_dir() or not (REFERENCE / ".git").exists():
            return "pinned reference checkout is unavailable"
        packages = subprocess.run(
            [
                "Rscript",
                "-e",
                'quit(status=ifelse(all(vapply(c("glmnet","digest","jsonlite"), requireNamespace, logical(1), quietly=TRUE)),0,1))',
            ],
            check=False,
        )
        if packages.returncode:
            return "required R packages are unavailable"
        status = subprocess.run(
            ["git", "-C", str(REFERENCE), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            capture_output=True,
            check=False,
        )
        if status.returncode or status.stdout.strip():
            return "pinned reference checkout is absent or dirty"
        return None

    def test_reference_predictor_enforces_and_applies_exact_three_class_model(self) -> None:
        reason = self.prerequisites()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            shadow = root / "reference"
            model_dir = shadow / "runs" / "fixture" / "model" / "model_reference"
            parent.mkdir()
            model_dir.mkdir(parents=True)
            key, well, branch = "A1_1_0d0h0m", "A1", "original"
            cell_ids = [f"{branch}|{key}|{label}" for label in range(1, 5)]
            feature_columns = [
                "cell_id", "key", "well", "branch", "image_id", "mask_label",
                "feature_schema_version", *FEATURES,
            ]
            feature = parent / "features.tsv"
            numeric = [
                [-2.0 + index / 100 for index in range(9)],
                [0.0 + index / 100 for index in range(9)],
                [2.0 + index / 100 for index in range(9)],
                [1.0 + index / 100 for index in range(9)],
            ]
            numeric[-1][-1] = ""
            write_tsv(
                feature,
                feature_columns,
                [
                    [cell_ids[row], key, well, branch, key, row + 1,
                     "broad_phenotype_features_v1", *numeric[row]]
                    for row in range(4)
                ],
            )
            feature_receipt = parent / "feature.json"
            feature_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "broad_phenotype_feature_receipt_v1",
                        "status": "COMPLETE",
                        "key": key,
                        "well": well,
                        "branch": branch,
                        "feature_schema_version": "broad_phenotype_features_v1",
                        "feature_columns": feature_columns,
                        "feature_tsv": str(feature.resolve()),
                        "feature_tsv_sha256": sha256(feature),
                        "row_count": 4,
                        "cell_id_sha256": cell_hash(cell_ids),
                    }
                )
                + "\n"
            )
            make_model = root / "make_model.R"
            make_model.write_text(
                textwrap.dedent(
                    r'''
                    args <- commandArgs(trailingOnly=TRUE)
                    reference_root <- args[[1]]
                    model_dir <- args[[2]]
                    source_files <- sort(list.files(file.path(reference_root, "R"), pattern="[.]R$", full.names=TRUE), method="radix")
                    api <- new.env(parent=baseenv())
                    for (path in source_files) sys.source(path, envir=api)
                    feature_names <- c("area_px2","perimeter_px","roundness","aspect_ratio","extent","solidity","equivalent_diameter_px","major_axis_px","minor_axis_px")
                    centers <- c(-3, 0, 3)
                    x <- do.call(rbind, lapply(centers, function(center) {
                      do.call(rbind, lapply(seq_len(5), function(index) center + seq_along(feature_names)/100 + index/1000))
                    }))
                    colnames(x) <- feature_names
                    classes <- c("dead_cell","live_cell","multinucleated_cell")
                    y <- factor(rep(classes, each=5), levels=classes)
                    preprocessor <- api$cpa_classifier_fit_preprocessor(x, "median_impute", "drop_with_manifest")
                    transformed <- api$cpa_classifier_apply_preprocessor(x, preprocessor)$x
                    fit <- glmnet::glmnet(x=transformed, y=y, family="multinomial", type.multinomial="grouped", alpha=1, lambda=0.05, standardize=FALSE)
                    runtime <- api$cpa_classifier_runtime_identity()
                    source_identity <- api$cpa_source_checkout_identity(reference_root)
                    model <- list(
                      schema_version=api$cpa_classifier_schema_version(), model_id="model_reference",
                      class_ids=classes, feature_columns=feature_names,
                      preprocessor=preprocessor, fit=fit, lambda=0.05,
                      classifier_config_sha256="fixture_config", training_features_sha256="fixture_training",
                      runtime_identity=runtime
                    )
                    saveRDS(model, file.path(model_dir,"model.rds"), version=3L)
                    artifact_names <- c("training_rows","outer_folds","fold_summary","fold_metrics_by_class","feature_preprocessor","out_of_fold_predictions","out_of_fold_probabilities","metrics_by_class","metrics_summary","confusion_matrix")
                    artifact_files <- c("training_rows.tsv","outer_fold_assignments.tsv","fold_summary.tsv","fold_metrics_by_class.tsv","feature_preprocessor_manifest.tsv","out_of_fold_predictions.tsv","out_of_fold_probabilities.tsv","metrics_by_class.tsv","metrics_summary.tsv","confusion_matrix.tsv")
                    for (name in artifact_files) writeLines(c("fixture","1"), file.path(model_dir,name))
                    hashes <- c(list(model=api$cpa_hash_file(file.path(model_dir,"model.rds"))), setNames(lapply(file.path(model_dir,artifact_files), api$cpa_hash_file), artifact_names))
                    manifest <- list(
                      schema_version=api$cpa_classifier_schema_version(), model_id="model_reference",
                      project_id="reference_cell_state_development", run_id="fixture",
                      class_config_sha256="fixture_classes", classifier_config_sha256="fixture_config",
                      training_features_sha256="fixture_training", feature_columns=as.list(feature_names),
                      class_ids=as.list(classes), runtime_identity=runtime, source_identity=source_identity,
                      artifact_file_sha256=hashes
                    )
                    api$cpa_write_text_atomic(api$cpa_canonical_json(manifest, pretty=TRUE), file.path(model_dir,"model_manifest.json"))
                    '''
                ).strip()
                + "\n"
            )
            made = subprocess.run(
                ["Rscript", str(make_model), str(REFERENCE), str(model_dir)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(made.returncode, 0, made.stderr)
            parent_import = shadow / "projection_input" / "representative_umap" / "parent_import_manifest.json"
            parent_import.parent.mkdir(parents=True)
            parent_import.write_text(
                json.dumps(
                    {
                        "schema_version": "reference_cell_state_parent_import_v1",
                        "parent": {"shadow_root": str(parent.resolve())},
                    }
                )
                + "\n"
            )
            acceptance = shadow / "workflow_status" / "model_acceptance" / "model_acceptance.json"
            acceptance.parent.mkdir(parents=True)
            acceptance.write_text(
                json.dumps(
                    {
                        "schema_version": "reference_cell_state_model_acceptance_v1",
                        "status": "ACCEPTED",
                        "accepted": True,
                        "shadow_root": str(shadow.resolve()),
                        "model": {
                            "dir": str(model_dir.resolve()),
                            "model_id": "model_reference",
                            "manifest_sha256": sha256(model_dir / "model_manifest.json"),
                            "feature_columns": list(FEATURES),
                            "class_ids": list(CLASSES),
                        },
                        "dependency": {"lock_sha256": sha256(LOCK)},
                        "frozen_inputs": {
                            "parent_import_manifest": str(parent_import.resolve()),
                            "parent_import_manifest_sha256": sha256(parent_import),
                            "parent_shadow_root": str(parent.resolve()),
                        },
                    }
                )
                + "\n"
            )
            output = shadow / "prediction_shards" / "shards" / well / f"{key}__{branch}_reference_cell_state_predictions.tsv"
            receipt = shadow / "prediction_shards" / "receipts" / well / f"{key}__{branch}.json"
            output.parent.mkdir(parents=True)
            receipt.parent.mkdir(parents=True)
            command = [
                "Rscript", str(PREDICT),
                "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK),
                "--model-dir", str(model_dir),
                "--model-acceptance-receipt", str(acceptance),
                "--model-acceptance-sha256", sha256(acceptance),
                "--feature-shard", str(feature),
                "--feature-receipt", str(feature_receipt),
                "--output-tsv", str(output.resolve()),
                "--output-receipt", str(receipt.resolve()),
            ]
            predicted = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(predicted.returncode, 0, predicted.stderr)
            rows = read_tsv(output)
            self.assertEqual(list(rows[0])[:4], ["model_id", "cell_id", "predicted_class_id", "prediction_status"])
            self.assertEqual(list(rows[0])[4:], [f"probability__{class_id}" for class_id in CLASSES])
            self.assertEqual(rows[-1]["prediction_status"], "ok")
            repeated = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("already_complete=1", repeated.stdout)


if __name__ == "__main__":
    unittest.main()
