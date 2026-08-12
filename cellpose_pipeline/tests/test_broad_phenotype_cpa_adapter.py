from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "cellpose_pipeline" / "scripts"
REFERENCE_ROOT = Path(
    os.environ.get(
        "CPA_REFERENCE_ROOT",
        str(ROOT.parent / "cell-phenotype-annotator"),
    )
).expanduser().resolve()
LOCK_PATH = ROOT / "cellpose_pipeline" / "configs" / "cellphenotypeannotator_dependency.lock.tsv"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_script("broad_phenotype_cpa_adapter_test", "17_build_broad_phenotype_cpa_inputs.py")
PROJECTION = load_script("broad_phenotype_projection_test", "18_prepare_broad_phenotype_projection.py")
RUNNER = load_script("broad_phenotype_runner_test", "20_run_cellphenotypeannotator_stage.py")


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class BroadPhenotypeCPAAdapterTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        source = root / "source"
        feature_root = root / "feature_root"
        manifest_rows: list[dict[str, object]] = []
        feature_manifest_rows: list[dict[str, object]] = []
        numeric = ["area_px2", "centroid_x_px", "bf_object_mean", "nuclei_count"]
        for well, offset in (("A01", 0), ("B01", 10)):
            key = f"{well}_1_00d00h00m"
            field = source / key
            field.mkdir(parents=True)
            combined = np.zeros((12, 12, 3), dtype=np.uint8)
            combined[..., 0] = 30 + offset
            combined[..., 1] = 60 + offset
            combined[..., 2] = 90 + offset
            scalar = np.arange(144, dtype=np.uint16).reshape(12, 12) + offset
            mask = np.zeros((12, 12), dtype=np.uint16)
            mask[1:4, 1:4] = 1
            mask[5:8, 1:4] = 2
            mask[1:4, 6:10] = 3
            paths = {
                "combined_raw": field / "combined.tif",
                "brightfield_raw": field / "brightfield.tif",
                "dead_raw": field / "dead.tif",
                "nuclei_raw": field / "nuclei.tif",
                "combined_mask": field / "combined_mask.tif",
                "nucleated_combined_mask": field / "nucleated_mask.tif",
            }
            tifffile.imwrite(paths["combined_raw"], combined, photometric="rgb")
            for role in ("brightfield_raw", "nuclei_raw"):
                # Live Incucyte TIFFs expose a full-resolution primary IFD and
                # a one-pixel auxiliary IFD to R's tiff reader.  Exercise that
                # exact contract so the adapter must write page_index=1.
                with tifffile.TiffWriter(paths[role]) as writer:
                    writer.write(scalar)
                    writer.write(np.zeros((1, 1), dtype=scalar.dtype))
            tifffile.imwrite(paths["dead_raw"], scalar)
            tifffile.imwrite(paths["combined_mask"], mask)
            tifffile.imwrite(paths["nucleated_combined_mask"], mask)
            manifest_rows.append({"key": key, **{role: str(path) for role, path in paths.items()}})

            shard = feature_root / "shards" / well / f"{key}__original_broad_phenotype_features.tsv"
            rows = []
            for label in (1, 2, 3):
                rows.append(
                    {
                        "cell_id": f"original|{key}|{label}",
                        "key": key,
                        "well": well,
                        "site": 1,
                        "elapsed_hours": 0,
                        "branch": "original",
                        "image_id": key,
                        "mask_label": label,
                        "area_px2": label + 8,
                        "centroid_x_px": label + 1,
                        "bf_object_mean": 0.1 * label + offset,
                        "nuclei_count": label % 2,
                    }
                )
            fields = [
                "cell_id",
                "key",
                "well",
                "site",
                "elapsed_hours",
                "branch",
                "image_id",
                "mask_label",
                *numeric,
            ]
            write_tsv(shard, fields, rows)
            receipt = feature_root / "receipts" / well / f"{key}__original.json"
            feature_manifest_rows.append(
                {"key": key, "feature_path": str(shard), "receipt_path": str(receipt)}
            )

        field_manifest = root / "field_manifest.tsv"
        manifest_fields = [
            "key",
            "combined_raw",
            "brightfield_raw",
            "dead_raw",
            "nuclei_raw",
            "combined_mask",
            "nucleated_combined_mask",
        ]
        write_tsv(field_manifest, manifest_fields, manifest_rows)
        feature_manifest = root / "feature_manifest.tsv"
        write_tsv(feature_manifest, ["key", "feature_path", "receipt_path"], feature_manifest_rows)
        classes = root / "classes.tsv"
        write_tsv(
            classes,
            ["class_id", "display_name", "color", "description", "shortcut", "trainable", "role", "order"],
            [
                {"class_id": "round", "display_name": "Round", "color": "#00aa00", "description": "", "shortcut": "r", "trainable": "true", "role": "phenotype", "order": 1},
                {"class_id": "flat", "display_name": "Flat", "color": "#aa0000", "description": "", "shortcut": "f", "trainable": "true", "role": "phenotype", "order": 2},
            ],
        )
        feature_config = root / "feature_config.json"
        feature_config.write_text(
            json.dumps(
                {
                    "schema_version": "broad_phenotype_feature_config_v1",
                    "method_version": "bf_combined_mask_v1",
                    "feature_schema_version": "broad_phenotype_features_v1",
                    "numeric_feature_columns": numeric,
                    "primary_model_feature_columns": ["area_px2", "bf_object_mean"],
                    "nuclei_comparator_feature_columns": ["area_px2", "bf_object_mean", "nuclei_count"],
                    "parameters": {"fixture_parameter": 1},
                }
            )
        )
        config = ADAPTER.load_feature_config(feature_config)
        implementation = {
            "extractor_sha256": ADAPTER.sha256_file(
                SCRIPT_ROOT / "16_extract_broad_phenotype_features.py"
            ),
            "feature_module_sha256": ADAPTER.sha256_file(
                SCRIPT_ROOT / "_shared" / "broad_phenotype_features.py"
            ),
        }
        positive_labels_sha256 = hashlib.sha256(b"1\n2\n3").hexdigest()
        for row in feature_manifest_rows:
            shard = Path(str(row["feature_path"]))
            receipt = Path(str(row["receipt_path"]))
            key = str(row["key"])
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "broad_phenotype_feature_receipt_v1",
                        "status": "COMPLETE",
                        "key": key,
                        "branch": "original",
                        "feature_schema_version": config["feature_schema_version"],
                        "feature_code_version": config["method_version"],
                        "feature_schema_sha256": config["feature_schema_sha256"],
                        "feature_config_sha256": config["raw_sha256"],
                        "feature_config_semantic_sha256": config["semantic_sha256"],
                        "feature_tsv": str(shard),
                        "feature_tsv_sha256": ADAPTER.sha256_file(shard),
                        "numeric_feature_columns": config["numeric_feature_columns"],
                        "primary_model_feature_columns": config["primary_model_feature_columns"],
                        "nuclei_comparator_feature_columns": config["nuclei_comparator_feature_columns"],
                        "feature_parameters": config["parameters"],
                        "feature_columns": fields,
                        "row_count": 3,
                        "missing_value_counts": {
                            column: 0 for column in config["numeric_feature_columns"]
                        },
                        "diagnostics": {
                            "object_count": 3,
                            "positive_labels_sha256": positive_labels_sha256,
                        },
                        "implementation": implementation,
                        "forbidden_inputs_read": [],
                        "input_contract": {"dead": "not_read", "current_classification": "not_read"},
                    }
                )
            )
        return field_manifest, feature_manifest, classes, feature_config, feature_root

    def test_streamed_adapter_and_representative_project_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            field_manifest, feature_manifest, classes, feature_config, _ = self.make_fixture(root)
            shadow = root / "shadow"
            argv = [
                "--field-manifest", str(field_manifest),
                "--feature-manifest", str(feature_manifest),
                "--classes-file", str(classes),
                "--feature-config", str(feature_config),
                "--shadow-root", str(shadow),
                "--branch", "original",
                "--heldout-wells", "B01",
                "--outer-folds", "2",
                "--inner-folds", "2",
            ]
            self.assertEqual(ADAPTER.main(argv), 0)
            first_hash = ADAPTER.sha256_file(shadow / "cpa" / "features.tsv")
            self.assertEqual(ADAPTER.main(argv), 0)
            self.assertEqual(first_hash, ADAPTER.sha256_file(shadow / "cpa" / "features.tsv"))

            cells = read_tsv(shadow / "cpa" / "cells.tsv")
            self.assertEqual(len(cells), 6)
            self.assertEqual(cells[0]["cell_id"], "original|A01_1_00d00h00m|1")
            project = json.loads((shadow / "cpa" / "project.yml").read_text())
            self.assertEqual(project["projection"]["feature_columns"], ["area_px2", "bf_object_mean"])
            self.assertEqual(project["classifier"]["feature_columns"], ["area_px2", "bf_object_mean"])
            self.assertEqual(project["classifier"]["alpha"], 0.5)
            self.assertEqual(project["classifier"]["group_column"], "well")
            self.assertEqual(project["review"]["group_column"], "well")
            self.assertEqual(project["review"]["quota_scope"], "global")
            self.assertEqual(project["review"]["quotas"]["default_per_class"], 50)
            self.assertEqual(project["review"]["quotas"]["explicit_unassigned"], 0)
            self.assertEqual(project["review"]["shortage_policy"], "fail")
            self.assertEqual(
                [display["display_id"] for display in project["review"]["displays"]],
                ["brightfield", "nuclei"],
            )
            image_rows = read_tsv(shadow / "cpa" / "images.tsv")
            self.assertEqual({row["channel_id"] for row in image_rows}, {"brightfield", "nuclei"})
            self.assertEqual({row["page_index"] for row in image_rows}, {"1"})
            path_audit = read_tsv(
                shadow / "workflow_status" / "adapter" / "path_dimension_audit.tsv"
            )
            review_rows = [
                row for row in path_audit if row["asset_role"] in {"brightfield_raw", "nuclei_raw"}
            ]
            self.assertEqual({row["pages"] for row in review_rows}, {"2"})
            self.assertEqual({row["page_index"] for row in review_rows}, {"1"})
            policy = {row["column"]: row for row in read_tsv(shadow / "workflow_status" / "adapter" / "feature_policy.tsv")}
            self.assertEqual(policy["centroid_x_px"]["included"], "false")
            self.assertEqual(policy["nuclei_count"]["usage"], "nuclei_comparator_only")

            self.assertEqual(
                PROJECTION.main(
                    [
                        "--project", str(shadow / "cpa" / "project.yml"),
                        "--shadow-root", str(shadow),
                        "--max-cells-per-field", "3",
                        "--max-cells-per-well", "3",
                    ]
                ),
                0,
            )
            representative = shadow / "projection_input" / "representative_umap"
            representative_cells = read_tsv(representative / "cells.tsv")
            self.assertEqual(len(representative_cells), 3)
            self.assertEqual({row["well"] for row in representative_cells}, {"A01"})
            representative_features = read_tsv(representative / "features.tsv")
            self.assertEqual(list(representative_features[0]), ["cell_id", "area_px2", "bf_object_mean"])
            self.assertEqual({row["image_id"] for row in read_tsv(representative / "images.tsv")}, {"A01_1_00d00h00m"})
            representative_project = json.loads((representative / "project.yml").read_text())
            self.assertEqual(representative_project["cells_file"], "cells.tsv")
            self.assertEqual(representative_project["classifier"]["group_column"], "well")
            manifest = json.loads((shadow / "projection_input" / "projection_input_manifest.json").read_text())
            self.assertEqual(manifest["representative_heldout_cell_count"], 0)
            self.assertFalse(manifest["heldout_evaluation_performed"])

    def test_primary_allowlist_rejects_localization_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, config, _ = self.make_fixture(root)
            payload = json.loads(config.read_text())
            payload["primary_model_feature_columns"] = ["area_px2", "centroid_x_px"]
            config.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "localization"):
                ADAPTER.load_feature_config(config)

    def test_plate_map_split_is_condition_balanced_and_deterministic(self) -> None:
        plate_map = (
            SCRIPT_ROOT
            / "analysisi"
            / "resources"
            / "SUM159_AC_Experiment1_PlateMap.csv"
        )
        _, rows = ADAPTER.read_table(plate_map)
        wells = [row["well"] for row in rows]
        first = ADAPTER.condition_balanced_splits(wells, plate_map, 20260812)
        second = ADAPTER.condition_balanced_splits(wells, plate_map, 20260812)
        self.assertEqual(first, second)
        splits, condition_rows, well_conditions, metadata = first
        self.assertEqual(len(splits), 80)
        self.assertEqual(sum(value == "heldout" for value in splits.values()), 16)
        self.assertEqual(sum(value == "development" for value in splits.values()), 64)
        self.assertEqual(len(condition_rows), 40)
        self.assertEqual(len(well_conditions), 80)
        selected = [
            row for row in condition_rows if row["selected_for_heldout"] == "true"
        ]
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            {row["doxorubicin_nm"] for row in selected},
            {"0", "12.5", "100", "800"},
        )
        self.assertTrue(
            all(int(row["development_support_count"]) >= 1 for row in condition_rows)
        )
        self.assertEqual(metadata["heldout_replicate_1_count"], 8)
        self.assertEqual(metadata["heldout_replicate_2_count"], 8)
        self.assertEqual(
            sorted(well for well, split in splits.items() if split == "heldout"),
            sorted(
                "A2,A8,B11,B5,C2,C8,D11,D5,E11,E5,F2,F8,G11,G5,H2,H8".split(",")
            ),
        )
        self.assertEqual(
            metadata["split_identity_sha256"],
            "2f50f785e3944152ed083cdb47e4fb3c099fa09cec473e600a32eef9b00906f8",
        )
        self.assertEqual(
            metadata["strategy"], "condition_balanced_paired_replicate_v1"
        )

        shuffled = Path(tempfile.mkdtemp()) / "shuffled.csv"
        try:
            shuffled_rows = list(rows)
            random.Random(9182).shuffle(shuffled_rows)
            write_tsv(shuffled, list(rows[0]), shuffled_rows)
            shuffled_result = ADAPTER.condition_balanced_splits(
                wells, shuffled, 20260812
            )
            self.assertEqual(first[0], shuffled_result[0])
            self.assertEqual(first[1], shuffled_result[1])
            self.assertEqual(
                metadata["split_identity_sha256"],
                shuffled_result[3]["split_identity_sha256"],
            )
            self.assertEqual(
                metadata["plate_map_semantic_sha256"],
                shuffled_result[3]["plate_map_semantic_sha256"],
            )
        finally:
            shutil.rmtree(shuffled.parent)

    def test_plate_map_split_rejects_design_and_coordinate_drift(self) -> None:
        source = (
            SCRIPT_ROOT
            / "analysisi"
            / "resources"
            / "SUM159_AC_Experiment1_PlateMap.csv"
        )
        fields, rows = ADAPTER.read_table(source)
        wells = [row["well"] for row in rows]
        mutations = {
            "coordinate mismatch": lambda changed: changed[0].update(
                {"plate_column": "99"}
            ),
            "duplicate well": lambda changed: changed[1].update(
                {"well": changed[0]["well"]}
            ),
            "replicate pair drift": lambda changed: changed[10].update(
                {"replicate": "1"}
            ),
            "dose grid drift": lambda changed: changed[0].update(
                {"doxorubicin_nm": "0.5"}
            ),
            "ploidy drift": lambda changed: changed[0].update({"ploidy": "8N"}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                changed = [dict(row) for row in rows]
                mutate(changed)
                candidate = Path(temporary) / "plate_map.tsv"
                write_tsv(candidate, fields, changed)
                with self.assertRaises(ValueError):
                    ADAPTER.condition_balanced_splits(wells, candidate, 20260812)

    def test_tampered_shard_is_rejected_by_receipt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            field_manifest, feature_manifest, classes, feature_config, _ = self.make_fixture(root)
            feature_row = read_tsv(feature_manifest)[0]
            shard = Path(feature_row["feature_path"])
            shard.write_text(shard.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "receipt verification failed"):
                ADAPTER.main(
                    [
                        "--field-manifest", str(field_manifest),
                        "--feature-manifest", str(feature_manifest),
                        "--classes-file", str(classes),
                        "--feature-config", str(feature_config),
                        "--shadow-root", str(root / "shadow"),
                        "--branch", "original",
                        "--heldout-wells", "B01",
                        "--outer-folds", "2",
                        "--inner-folds", "2",
                    ]
                )

    def test_tampered_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            field_manifest, feature_manifest, classes, feature_config, _ = self.make_fixture(root)
            feature_row = read_tsv(feature_manifest)[0]
            receipt_path = Path(feature_row["receipt_path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["status"] = "RUNNING"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, "receipt verification failed"):
                ADAPTER.main(
                    [
                        "--field-manifest", str(field_manifest),
                        "--feature-manifest", str(feature_manifest),
                        "--classes-file", str(classes),
                        "--feature-config", str(feature_config),
                        "--shadow-root", str(root / "shadow"),
                        "--branch", "original",
                        "--heldout-wells", "B01",
                        "--outer-folds", "2",
                        "--inner-folds", "2",
                    ]
                )

    def test_projection_rejects_tampered_split_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            field_manifest, feature_manifest, classes, feature_config, _ = self.make_fixture(root)
            shadow = root / "shadow"
            self.assertEqual(
                ADAPTER.main(
                    [
                        "--field-manifest", str(field_manifest),
                        "--feature-manifest", str(feature_manifest),
                        "--classes-file", str(classes),
                        "--feature-config", str(feature_config),
                        "--shadow-root", str(shadow),
                        "--branch", "original",
                        "--heldout-wells", "B01",
                        "--outer-folds", "2",
                        "--inner-folds", "2",
                    ]
                ),
                0,
            )
            split = shadow / "workflow_status" / "adapter" / "well_split_freeze.tsv"
            split.write_text(split.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "split artifact hash mismatch"):
                PROJECTION.main(
                    [
                        "--project", str(shadow / "cpa" / "project.yml"),
                        "--shadow-root", str(shadow),
                        "--max-cells-per-field", "3",
                        "--max-cells-per-well", "3",
                    ]
                )

    def test_reference_accepts_representative_umap_contract(self) -> None:
        rscript = shutil.which("Rscript")
        reference_cli = REFERENCE_ROOT / "exec" / "cell-phenotype-annotator"
        if rscript is None or not reference_cli.is_file():
            self.skipTest("Local reference source-mode R CLI is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            field_manifest, feature_manifest, classes, feature_config, _ = self.make_fixture(root)
            shadow = root / "shadow"
            ADAPTER.main(
                [
                    "--field-manifest", str(field_manifest),
                    "--feature-manifest", str(feature_manifest),
                    "--classes-file", str(classes),
                    "--feature-config", str(feature_config),
                    "--shadow-root", str(shadow),
                    "--branch", "original",
                    "--heldout-wells", "B01",
                    "--outer-folds", "2",
                    "--inner-folds", "2",
                ]
            )
            PROJECTION.main(
                [
                    "--project", str(shadow / "cpa" / "project.yml"),
                    "--shadow-root", str(shadow),
                    "--max-cells-per-field", "3",
                    "--max-cells-per-well", "3",
                ]
            )
            project = shadow / "projection_input" / "representative_umap" / "project.yml"
            completed = subprocess.run(
                [rscript, str(reference_cli), "umap", str(project), "--check-config"],
                cwd=shadow,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class CellPhenotypeSourceRunnerTests(unittest.TestCase):
    def make_model_generation(self, shadow: Path) -> Path:
        model_dir = shadow / "runs" / "fixture" / "classifier" / "model_fixture"
        model_dir.mkdir(parents=True)
        declared: dict[str, str] = {}
        for role, filename in RUNNER.MODEL_GENERATION_ARTIFACTS.items():
            artifact = model_dir / filename
            artifact.write_text(f"immutable fixture {role}\n")
            declared[role] = RUNNER.sha256_file(artifact)
        (model_dir / "model_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "cell_phenotype_annotator_classifier_v1",
                    "model_id": "model_fixture",
                    "artifact_file_sha256": declared,
                },
                sort_keys=True,
            )
        )
        return model_dir

    def test_source_runner_verifies_lock_and_reuses_command_receipt(self) -> None:
        if not REFERENCE_ROOT.is_dir() or not LOCK_PATH.is_file():
            self.skipTest("Pinned local reference checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            shadow.mkdir()
            project = shadow / "project.json"
            for name in ("classes.tsv", "cells.tsv", "features.tsv", "images.tsv"):
                (shadow / name).write_text(f"fixture\t{name}\n")
            project.write_text(
                json.dumps(
                    {
                        "classes_file": "classes.tsv",
                        "cells_file": "cells.tsv",
                        "features_file": "features.tsv",
                        "images_file": "images.tsv",
                        "runs_dir": "runs",
                    }
                )
            )
            fake_rscript = Path(temporary) / "Rscript"
            fake_rscript.write_text("#!/bin/sh\nexit 0\n")
            fake_rscript.chmod(0o755)
            argv = [
                "--stage", "validate",
                "--validate-stage", "umap",
                "--project", str(project),
                "--shadow-root", str(shadow),
                "--reference-root", str(REFERENCE_ROOT),
                "--dependency-lock", str(LOCK_PATH),
                "--rscript", str(fake_rscript),
                "--check-config",
            ]
            self.assertEqual(RUNNER.main(argv), 0)
            receipts = list((shadow / "workflow_status" / "cpa_stages").glob("validate.umap.check_config.*.json"))
            self.assertEqual(len(receipts), 1)
            before = receipts[0].read_bytes()
            self.assertEqual(RUNNER.main(argv), 0)
            self.assertEqual(receipts[0].read_bytes(), before)
            (shadow / "features.tsv").write_text("fixture\tfeatures.tsv\nchanged\t1\n")
            self.assertEqual(RUNNER.main(argv), 0)
            receipts = list((shadow / "workflow_status" / "cpa_stages").glob("validate.umap.check_config.*.json"))
            self.assertEqual(len(receipts), 2)

    def test_model_identity_excludes_report_outputs_and_detects_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            shadow.mkdir()
            model_dir = self.make_model_generation(shadow)
            before = RUNNER.model_generation_identities(model_dir, shadow)

            # These are outputs of report/predict, not classifier-generation inputs.
            (model_dir / "report.html").write_text("generated report\n")
            predictions = model_dir / "predictions" / "prediction_fixture"
            predictions.mkdir(parents=True)
            (predictions / "prediction_manifest.json").write_text("{}\n")
            after_outputs = RUNNER.model_generation_identities(model_dir, shadow)
            self.assertEqual(before, after_outputs)

            artifact = model_dir / RUNNER.MODEL_GENERATION_ARTIFACTS["metrics_summary"]
            artifact.write_text("tampered metrics\n")
            with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                RUNNER.model_generation_identities(model_dir, shadow)

            # A legitimate new immutable generation updates both artifact and manifest;
            # its input identity must differ from the original generation.
            manifest_path = model_dir / "model_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_file_sha256"]["metrics_summary"] = RUNNER.sha256_file(artifact)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            after_generation_change = RUNNER.model_generation_identities(model_dir, shadow)
            self.assertNotEqual(before, after_generation_change)

    def test_derived_review_project_stays_beside_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "shadow"
            project_dir = shadow / "projection_input" / "representative_umap"
            project_dir.mkdir(parents=True)
            accepted = shadow / "accepted"
            accepted.mkdir()
            project_path = project_dir / "project.yml"
            project = {
                "classes_file": "classes.tsv",
                "cells_file": "cells.tsv",
                "features_file": "features.tsv",
                "images_file": "images.tsv",
                "runs_dir": "runs",
                "review": {"displays": []},
            }
            project_path.write_text(json.dumps(project))
            derived = RUNNER.write_derived_review_project(project_path, project, accepted, shadow)
            self.assertEqual(derived.parent, project_path.parent)
            payload = json.loads(derived.read_text())
            self.assertEqual(payload["cells_file"], "cells.tsv")
            self.assertEqual(payload["review"]["annotation_import_dir"], str(accepted.resolve()))


if __name__ == "__main__":
    unittest.main()
