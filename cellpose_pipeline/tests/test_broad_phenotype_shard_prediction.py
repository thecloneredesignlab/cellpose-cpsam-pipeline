#!/usr/bin/env python3
"""Tests for audited, shard-safe broad-phenotype inference."""

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
PREDICT = REPO / "cellpose_pipeline" / "scripts" / "21_predict_broad_phenotype_shard.R"
MERGE = REPO / "cellpose_pipeline" / "scripts" / "22_merge_broad_phenotype_predictions.py"
LOCK = REPO / "cellpose_pipeline" / "configs" / "cellphenotypeannotator_dependency.lock.tsv"
REFERENCE = Path(
    os.environ.get(
        "CPA_REFERENCE_ROOT",
        "/Users/4482173/Documents/GitHub/cell-phenotype-annotator",
    )
).expanduser().resolve()


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


class ShardMergeTests(unittest.TestCase):
    def make_merge_fixture(self, root: Path) -> dict[str, Path]:
        shadow = root / "shadow"
        prediction_root = shadow / "prediction_shards"
        feature_root = shadow / "features"
        shadow.mkdir()
        prediction_root.mkdir()
        feature_root.mkdir()
        key = "A1_1_0d0h0m"
        well = "A1"
        branch = "original"
        cell_ids = [f"{branch}|{key}|1", f"{branch}|{key}|2"]
        feature_columns = [
            "cell_id",
            "key",
            "well",
            "branch",
            "image_id",
            "mask_label",
            "feature_schema_version",
            "area_px2",
        ]
        feature_path = feature_root / f"{key}__{branch}_broad_phenotype_features.tsv"
        write_tsv(
            feature_path,
            feature_columns,
            [
                [cell_ids[0], key, well, branch, key, 1, "broad_phenotype_features_v1", 10],
                [cell_ids[1], key, well, branch, key, 2, "broad_phenotype_features_v1", 20],
            ],
        )
        feature_receipt_path = feature_root / f"{key}__{branch}.json"
        feature_receipt = {
            "schema_version": "broad_phenotype_feature_receipt_v1",
            "status": "COMPLETE",
            "key": key,
            "well": well,
            "branch": branch,
            "feature_schema_version": "broad_phenotype_features_v1",
            "feature_columns": feature_columns,
            "feature_tsv": str(feature_path),
            "feature_tsv_sha256": sha256(feature_path),
            "row_count": 2,
            "cell_id_sha256": cell_hash(cell_ids),
        }
        feature_receipt_path.write_text(json.dumps(feature_receipt) + "\n")
        feature_manifest = shadow / "workflow_status" / "feature_inventory" / "feature_manifest.tsv"
        write_tsv(
            feature_manifest,
            ["key", "feature_path", "receipt_path"],
            [[key, feature_path, feature_receipt_path]],
        )
        cells = shadow / "cpa" / "cells.tsv"
        write_tsv(cells, ["cell_id"], [[value] for value in cell_ids])

        class_ids = ["typical", "rounded"]
        prediction_columns = [
            "model_id",
            "cell_id",
            "predicted_class_id",
            "prediction_status",
            "probability__typical",
            "probability__rounded",
        ]
        prediction_path = (
            prediction_root
            / "shards"
            / well
            / f"{key}__{branch}_broad_phenotype_predictions.tsv"
        )
        write_tsv(
            prediction_path,
            prediction_columns,
            [
                ["model_test", cell_ids[0], "rounded", "ok", "0.25", "0.75"],
                ["model_test", cell_ids[1], "", "unavailable_missing_features", "", ""],
            ],
        )
        prediction_receipt_path = (
            prediction_root / "receipts" / well / f"{key}__{branch}.json"
        )
        prediction_receipt_path.parent.mkdir(parents=True)
        model_acceptance = shadow / "workflow_status" / "model_acceptance" / "model_acceptance.json"
        model_acceptance.parent.mkdir(parents=True)
        model_acceptance.write_text('{"accepted":true}\n')
        model_acceptance_sha = sha256(model_acceptance)
        prediction_receipt = {
            "schema_version": "broad_phenotype_shard_prediction_v1",
            "status": "COMPLETE",
            "key": key,
            "well": well,
            "branch": branch,
            "model_id": "model_test",
            "class_ids": class_ids,
            "feature_tsv_sha256": sha256(feature_path),
            "feature_receipt_sha256": sha256(feature_receipt_path),
            "model_sha256": "a" * 64,
            "model_manifest_sha256": "b" * 64,
            "model_acceptance_receipt": str(model_acceptance.resolve()),
            "model_acceptance_sha256": model_acceptance_sha,
            "dependency_lock_sha256": "c" * 64,
            "implementation_sha256": "f" * 64,
            "source_identity": {"mode": "source_checkout", "source_sha256": "d" * 64},
            "runtime_identity": {"r_version": "fixture", "glmnet_version": "fixture"},
            "row_count": 2,
            "cell_id_sha256": cell_hash(cell_ids),
            "ok_count": 1,
            "unavailable_count": 1,
            "max_probability_sum_error": 0,
            "output_columns": prediction_columns,
            "prediction_tsv": str(prediction_path),
            "prediction_tsv_sha256": sha256(prediction_path),
        }
        prediction_receipt_path.write_text(json.dumps(prediction_receipt) + "\n")
        legacy = root / "legacy_no_go.json"
        legacy.write_text(json.dumps({"overall_decision": "NO_GO"}) + "\n")
        return {
            "shadow": shadow,
            "prediction_root": prediction_root,
            "feature_manifest": feature_manifest,
            "cells": cells,
            "legacy": legacy,
            "prediction_path": prediction_path,
            "prediction_receipt": prediction_receipt_path,
            "model_acceptance": model_acceptance,
        }

    def run_merge(self, fixture: dict[str, Path]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                shutil.which("python3") or "python3",
                str(MERGE),
                "--feature-manifest",
                str(fixture["feature_manifest"]),
                "--prediction-root",
                str(fixture["prediction_root"]),
                "--shadow-root",
                str(fixture["shadow"]),
                "--model-acceptance-receipt",
                str(fixture["model_acceptance"]),
                "--model-acceptance-sha256",
                sha256(fixture["model_acceptance"]),
                "--cells",
                str(fixture["cells"]),
                "--legacy-no-go",
                str(fixture["legacy"]),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_stream_merge_writes_isolated_four_column_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_merge_fixture(Path(directory))
            completed = self.run_merge(fixture)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = fixture["shadow"] / "predictions" / "broad_phenotype_predictions.tsv"
            rows = read_tsv(output)
            self.assertEqual(
                list(rows[0]),
                [
                    "model_id",
                    "cell_id",
                    "broad_phenotype_class_id",
                    "prediction_status",
                ],
            )
            self.assertEqual(rows[0]["broad_phenotype_class_id"], "rounded")
            self.assertEqual(rows[1]["prediction_status"], "unavailable_missing_features")
            receipt = json.loads(
                (fixture["shadow"] / "BROAD_PHENOTYPE_SHADOW_GO_NO_GO.json").read_text()
            )
            self.assertEqual(receipt["overall_decision"], "SHADOW_ONLY")
            self.assertEqual(receipt["technical_decision"], "GO")
            self.assertEqual(receipt["promotion_decision"], "NO_GO")
            self.assertTrue(receipt["legacy_viability_gate"]["ignored"])
            self.assertEqual(receipt["ok_count"], 1)
            self.assertEqual(receipt["unavailable_count"], 1)
            repeated = self.run_merge(fixture)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("already_complete=1", repeated.stdout)
            prediction_receipt = json.loads(fixture["prediction_receipt"].read_text())
            prediction_receipt["implementation_sha256"] = "e" * 64
            fixture["prediction_receipt"].write_text(json.dumps(prediction_receipt) + "\n")
            changed_generation = self.run_merge(fixture)
            self.assertNotEqual(changed_generation.returncode, 0)
            self.assertIn("after complete", changed_generation.stderr)

    def test_stream_merge_rejects_prediction_receipt_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_merge_fixture(Path(directory))
            receipt = json.loads(fixture["prediction_receipt"].read_text())
            receipt["ok_count"] = 2
            fixture["prediction_receipt"].write_text(json.dumps(receipt) + "\n")
            completed = self.run_merge(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("ok_count mismatch", completed.stderr)


class CanonicalParityTests(unittest.TestCase):
    def prerequisites(self) -> str | None:
        if shutil.which("Rscript") is None:
            return "Rscript is unavailable"
        if not REFERENCE.is_dir() or not (REFERENCE / ".git").exists():
            return "pinned reference checkout is unavailable"
        package_check = subprocess.run(
            [
                "Rscript",
                "-e",
                'quit(status=ifelse(all(vapply(c("glmnet","digest","jsonlite"), requireNamespace, logical(1), quietly=TRUE)),0,1))',
            ],
            check=False,
        )
        if package_check.returncode:
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

    def test_r_shard_matches_reference_canonical_prediction(self) -> None:
        reason = self.prerequisites()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            key, well, branch = "A1_1_0d0h0m", "A1", "original"
            cell_ids = [f"{branch}|{key}|{label}" for label in range(1, 5)]
            columns = [
                "cell_id",
                "key",
                "well",
                "branch",
                "image_id",
                "mask_label",
                "feature_schema_version",
                "x1",
                "x2",
            ]
            feature = root / "feature.tsv"
            write_tsv(
                feature,
                columns,
                [
                    [cell_ids[0], key, well, branch, key, 1, "broad_phenotype_features_v1", -2, -1],
                    [cell_ids[1], key, well, branch, key, 2, "broad_phenotype_features_v1", -1, -2],
                    [cell_ids[2], key, well, branch, key, 3, "broad_phenotype_features_v1", 2, 1],
                    [cell_ids[3], key, well, branch, key, 4, "broad_phenotype_features_v1", 1, ""],
                ],
            )
            feature_receipt = root / "feature.json"
            feature_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "broad_phenotype_feature_receipt_v1",
                        "status": "COMPLETE",
                        "key": key,
                        "well": well,
                        "branch": branch,
                        "feature_schema_version": "broad_phenotype_features_v1",
                        "feature_columns": columns,
                        "feature_tsv": str(feature),
                        "feature_tsv_sha256": sha256(feature),
                        "row_count": len(cell_ids),
                        "cell_id_sha256": cell_hash(cell_ids),
                    }
                )
                + "\n"
            )
            fixture_script = root / "make_model.R"
            fixture_script.write_text(
                textwrap.dedent(
                    r'''
                    args <- commandArgs(trailingOnly=TRUE)
                    reference_root <- args[[1]]
                    model_dir <- args[[2]]
                    canonical_predictions <- args[[3]]
                    canonical_probabilities <- args[[4]]
                    source_files <- sort(list.files(file.path(reference_root, "R"), pattern="[.]R$", full.names=TRUE), method="radix")
                    api <- new.env(parent=baseenv())
                    for (path in source_files) sys.source(path, envir=api)
                    x <- rbind(c(-3,-2),c(-2,-3),c(-1,-2),c(-2,-1),c(3,2),c(2,3),c(1,2),c(2,1))
                    colnames(x) <- c("x1","x2")
                    y <- factor(c(rep("alpha",4),rep("beta",4)), levels=c("alpha","beta"))
                    preprocessor <- api$cpa_classifier_fit_preprocessor(x, "fail", "drop_with_manifest")
                    transformed <- api$cpa_classifier_apply_preprocessor(x, preprocessor)$x
                    fit <- glmnet::glmnet(x=transformed, y=y, family="multinomial", type.multinomial="grouped", alpha=1, lambda=0.05, standardize=FALSE)
                    runtime <- api$cpa_classifier_runtime_identity()
                    source_identity <- api$cpa_source_checkout_identity(reference_root)
                    model <- list(
                      schema_version=api$cpa_classifier_schema_version(), model_id="model_parity",
                      class_ids=c("alpha","beta"), feature_columns=c("x1","x2"),
                      preprocessor=preprocessor, fit=fit, lambda=0.05,
                      classifier_config_sha256="fixture_config", training_features_sha256="fixture_training",
                      runtime_identity=runtime
                    )
                    saveRDS(model, file.path(model_dir,"model.rds"), version=3L)
                    artifact_names <- c(
                      "training_rows","outer_folds","fold_summary","fold_metrics_by_class",
                      "feature_preprocessor","out_of_fold_predictions","out_of_fold_probabilities",
                      "metrics_by_class","metrics_summary","confusion_matrix"
                    )
                    artifact_files <- c(
                      "training_rows.tsv","outer_fold_assignments.tsv","fold_summary.tsv","fold_metrics_by_class.tsv",
                      "feature_preprocessor_manifest.tsv","out_of_fold_predictions.tsv","out_of_fold_probabilities.tsv",
                      "metrics_by_class.tsv","metrics_summary.tsv","confusion_matrix.tsv"
                    )
                    for (name in artifact_files) writeLines(c("fixture","1"), file.path(model_dir,name))
                    hashes <- c(list(model=api$cpa_hash_file(file.path(model_dir,"model.rds"))),
                                setNames(lapply(file.path(model_dir,artifact_files), api$cpa_hash_file), artifact_names))
                    manifest <- list(
                      schema_version=api$cpa_classifier_schema_version(), model_id="model_parity",
                      project_id="fixture", run_id="fixture", class_config_sha256="fixture_classes",
                      classifier_config_sha256="fixture_config", training_features_sha256="fixture_training",
                      feature_columns=as.list(c("x1","x2")), class_ids=as.list(c("alpha","beta")),
                      runtime_identity=runtime, source_identity=source_identity, artifact_file_sha256=hashes
                    )
                    api$cpa_write_text_atomic(api$cpa_canonical_json(manifest, pretty=TRUE), file.path(model_dir,"model_manifest.json"))
                    features <- data.frame(cell_id=c("original|A1_1_0d0h0m|1","original|A1_1_0d0h0m|2","original|A1_1_0d0h0m|3","original|A1_1_0d0h0m|4"),
                                           x1=c(-2,-1,2,1), x2=c(-1,-2,1,NA), check.names=FALSE, stringsAsFactors=FALSE)
                    validation <- list(tables=list(features=features), cells=data.frame(cell_id=features$cell_id, stringsAsFactors=FALSE))
                    canonical <- api$cpa_predict_classifier_cells(validation, list(model=model))
                    api$cpa_write_tsv_atomic(canonical$predictions, canonical_predictions)
                    api$cpa_write_tsv_atomic(canonical$probabilities, canonical_probabilities)
                    '''
                ).strip()
                + "\n"
            )
            canonical_predictions = root / "canonical_predictions.tsv"
            canonical_probabilities = root / "canonical_probabilities.tsv"
            make = subprocess.run(
                [
                    "Rscript",
                    str(fixture_script),
                    str(REFERENCE),
                    str(model_dir),
                    str(canonical_predictions),
                    str(canonical_probabilities),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(make.returncode, 0, make.stderr)
            output = root / "shard_predictions.tsv"
            receipt = root / "shard_predictions.json"
            model_acceptance = root / "model_acceptance.json"
            model_acceptance.write_text('{"accepted":true}\n')
            predict_command = [
                    "Rscript",
                    str(PREDICT),
                    "--reference-root",
                    str(REFERENCE),
                    "--dependency-lock",
                    str(LOCK),
                    "--model-dir",
                    str(model_dir),
                    "--model-acceptance-receipt",
                    str(model_acceptance),
                    "--model-acceptance-sha256",
                    sha256(model_acceptance),
                    "--feature-shard",
                    str(feature),
                    "--feature-receipt",
                    str(feature_receipt),
                    "--output-tsv",
                    str(output),
                    "--output-receipt",
                    str(receipt),
                ]
            predict = subprocess.run(
                predict_command,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(predict.returncode, 0, predict.stderr)
            repeated = subprocess.run(
                predict_command,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("already_complete=1", repeated.stdout)
            raw_receipt = receipt.read_text()
            self.assertEqual(raw_receipt.count('"ok_count"'), 1)
            self.assertEqual(raw_receipt.count('"unavailable_count"'), 1)
            self.assertEqual(raw_receipt.count('"max_probability_sum_error"'), 1)
            receipt_payload = json.loads(raw_receipt)
            self.assertEqual(receipt_payload["ok_count"], 3)
            self.assertEqual(receipt_payload["unavailable_count"], 1)
            self.assertLessEqual(receipt_payload["max_probability_sum_error"], 1e-6)
            self.assertEqual(
                receipt_payload["output_columns"][:4],
                ["model_id", "cell_id", "predicted_class_id", "prediction_status"],
            )
            observed = read_tsv(output)
            canonical_prediction_rows = read_tsv(canonical_predictions)
            canonical_probability_rows = read_tsv(canonical_probabilities)
            long_probability = {
                (row["cell_id"], row["class_id"]): row["probability"]
                for row in canonical_probability_rows
            }
            self.assertEqual(len(observed), len(canonical_prediction_rows))
            for wide, canonical in zip(observed, canonical_prediction_rows, strict=True):
                self.assertEqual(wide["model_id"], canonical["model_id"])
                self.assertEqual(wide["cell_id"], canonical["cell_id"])
                self.assertEqual(wide["predicted_class_id"], canonical["predicted_class_id"])
                self.assertEqual(wide["prediction_status"], canonical["prediction_status"])
                for class_id in ("alpha", "beta"):
                    expected = long_probability[(wide["cell_id"], class_id)]
                    actual = wide[f"probability__{class_id}"]
                    if expected == "":
                        self.assertEqual(actual, "")
                    else:
                        self.assertAlmostEqual(float(actual), float(expected), places=14)

            # Even a colluding rewrite of both the wide TSV and its recorded
            # hash must not be accepted: inference is recomputed first.
            tampered = read_tsv(output)
            tampered[0]["probability__alpha"] = "0.5"
            tampered[0]["probability__beta"] = "0.5"
            tampered[0]["predicted_class_id"] = "alpha"
            write_tsv(
                output,
                list(tampered[0]),
                [[row[column] for column in tampered[0]] for row in tampered],
            )
            tampered_receipt = json.loads(receipt.read_text())
            tampered_receipt["prediction_tsv_sha256"] = sha256(output)
            receipt.write_text(json.dumps(tampered_receipt) + "\n")
            rejected = subprocess.run(
                predict_command,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("freshly computed canonical wide output", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
