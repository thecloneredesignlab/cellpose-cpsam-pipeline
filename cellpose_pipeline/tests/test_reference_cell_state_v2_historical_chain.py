#!/usr/bin/env python3
"""Focused contracts for V2 historical review/training/prediction adapters."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "cellpose_pipeline" / "scripts"
SHARED = SCRIPTS / "_shared" / "reference_cell_state_v2.R"
RENDER = SCRIPTS / "37_render_reference_cell_state_exact_review.py"
IMPORT = SCRIPTS / "38_import_reference_cell_state_exact_review.R"
SEED1 = SCRIPTS / "33_build_reference_cell_state_seed1_review.R"
TRAIN = SCRIPTS / "34_train_reference_cell_state_historical.R"
SEED2 = SCRIPTS / "35_build_reference_cell_state_seed2_review.R"
MERGE_REVIEWS = SCRIPTS / "36_merge_reference_cell_state_reviews.R"
ACCEPT = SCRIPTS / "30_accept_reference_cell_state_model.R"
PREDICT = SCRIPTS / "31_predict_reference_cell_state_shard.R"
CLASSES_V2 = REPO / "cellpose_pipeline" / "configs" / "reference_cell_state_classes_v2.tsv"
CONFIG = REPO / "cellpose_pipeline" / "configs" / "reference_cell_state_features_v2.json"
LOCK = REPO / "cellpose_pipeline" / "configs" / "cellphenotypeannotator_dependency.lock.tsv"
REFERENCE = Path(
    os.environ.get(
        "CPA_REFERENCE_ROOT", "/Users/4482173/Documents/GitHub/cell-phenotype-annotator"
    )
).expanduser().resolve()
FEATURE_IMPL = SCRIPTS / "_shared" / "broad_phenotype_features.py"
CURRENT12 = (
    "area_px2", "perimeter_px", "roundness", "aspect_ratio", "extent",
    "solidity", "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
    "bf_boundary_mean", "bf_interior_mean", "bf_interior_minus_boundary_mean",
)
CELLPOSE_PYTHON = Path("/Users/4482173/.pyenv/versions/CellPose/bin/python")


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        return fields, list(reader)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r_prerequisites() -> str | None:
    if shutil.which("Rscript") is None:
        return "Rscript is unavailable"
    if not REFERENCE.is_dir() or not (REFERENCE / ".git").exists():
        return "pinned reference checkout is unavailable"
    packages = subprocess.run(
        [
            "Rscript", "-e",
            'quit(status=ifelse(all(vapply(c("digest","jsonlite"), requireNamespace, logical(1), quietly=TRUE)),0,1))',
        ], check=False,
    )
    if packages.returncode:
        return "required R packages are unavailable"
    status = subprocess.run(
        ["git", "-C", str(REFERENCE), "status", "--porcelain", "--untracked-files=all"],
        text=True, capture_output=True, check=False,
    )
    if status.returncode or status.stdout.strip():
        return "pinned reference checkout is absent or dirty"
    return None


def load_feature_impl():
    spec = importlib.util.spec_from_file_location("broad_phenotype_features_v2_test", FEATURE_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load broad phenotype feature implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StaticHistoricalContracts(unittest.TestCase):
    def test_all_new_sources_parse(self) -> None:
        ast.parse(RENDER.read_text(encoding="utf-8"))
        for number in range(30, 39):
            path = next(iter(SCRIPTS.glob(f"{number}_*")), None)
            if path is None or path.suffix != ".R":
                continue
            completed = subprocess.run(
                ["Rscript", "-e", f'parse(file="{path}")'],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_config_and_classifier_freeze_exact_twelve(self) -> None:
        config = json.loads(CONFIG.read_text())
        self.assertEqual(tuple(config["classifier_feature_columns"]), CURRENT12)
        accept = (SCRIPTS / "30_accept_reference_cell_state_model.R").read_text()
        predict = (SCRIPTS / "31_predict_reference_cell_state_shard.R").read_text()
        self.assertIn("promoted_shape_plus_rfs_boundary", accept)
        self.assertIn("V1 nine-feature models are forbidden", accept)
        self.assertIn('SCHEMA_VERSION <- "reference_cell_state_shard_prediction_v2"', predict)
        self.assertIn("historical_predict_morphology_cell_state_probabilities", predict)
        self.assertNotIn("cpa_verify_classifier_generation(model_dir", predict)

    def test_shared_loader_is_selective_and_plate_contract_is_real(self) -> None:
        text = SHARED.read_text()
        self.assertIn('reference_cell_state_v2_ast_function(files$utils, "morphology_umap_row_key"', text)
        self.assertNotIn("sys.source(files$utils", text)
        self.assertIn("projection_manifest$well_split_freeze", text)
        self.assertIn("projection_manifest$well_split_freeze_sha256", text)
        self.assertIn("nrow(split) != 80L", text)
        self.assertIn('sum(split$split == "development") != 64L', text)
        self.assertIn('sum(split$split == "heldout") != 16L', text)
        self.assertIn('paste0("SUM-159-NLS-", split$ploidy)', text)
        self.assertIn('rows$source_id <- as.character(rows$well)', text)
        self.assertIn("reference_cell_state_v2_validate_canonical_cells", text)
        self.assertNotIn('"|replicate=", split$replicate', text)

    def test_review_training_leakage_guards_are_explicit(self) -> None:
        seed1 = (SCRIPTS / "33_build_reference_cell_state_seed1_review.R").read_text()
        trainer = (SCRIPTS / "34_train_reference_cell_state_historical.R").read_text()
        seed2 = (SCRIPTS / "35_build_reference_cell_state_seed2_review.R").read_text()
        merger = (SCRIPTS / "36_merge_reference_cell_state_reviews.R").read_text()
        self.assertIn("historical_exact_nominal", seed1)
        self.assertIn(
            "disclosed_no_stable_cluster_adapter_global_max500_balanced_well_max8",
            seed1,
        )
        self.assertIn("historical_core_parity_with_sampling_adaptation", seed1)
        self.assertIn("--expanded-labelability-decision", seed1)
        self.assertIn("authoritative_expanded_computational_nogo", seed1)
        self.assertIn("max_per_group = 8L", seed1)
        self.assertIn("train_morphology_cell_state_classifier_workflow", trainer)
        self.assertIn('engine = "glmnet"', trainer)
        self.assertIn('feature_profile = "promoted_shape_plus_rfs_boundary"', trainer)
        self.assertIn('status = "NOT_APPLICABLE"', trainer)
        self.assertIn('file.path(staging_dir, "training_receipt.json")', trainer)
        self.assertIn("install_receipt_mirror", trainer)
        self.assertIn("n_total = 200L, seed = 2L", seed2)
        self.assertIn("disclosed_no_stable_cluster_adapter", seed2)
        self.assertIn('source_group_column = "source_id"', seed1)
        self.assertIn('source_group_column = "source_id"', seed2)
        self.assertIn('source_id_semantics = "well"', seed1)
        self.assertIn('source_id_semantics = "well"', seed2)
        self.assertIn('cluster = as.character(cells$cluster)', seed1)
        self.assertIn('cluster = as.character(cells$cluster)', seed2)
        self.assertIn("historical_high_or_medium_blank_as_high_low_excluded", trainer)
        self.assertIn("HUMAN_ADJUDICATION_REQUIRED", merger)
        for text in (trainer, seed2, merger):
            self.assertNotIn("cpa_run_train_stage", text)
        for path in (
            SCRIPTS / "30_accept_reference_cell_state_model.R",
            SCRIPTS / "31_predict_reference_cell_state_shard.R",
            SCRIPTS / "32_merge_reference_cell_state_predictions.py",
            SCRIPTS / "33_build_reference_cell_state_seed1_review.R",
            SCRIPTS / "34_train_reference_cell_state_historical.R",
            SCRIPTS / "35_build_reference_cell_state_seed2_review.R",
            SCRIPTS / "36_merge_reference_cell_state_reviews.R",
            SCRIPTS / "37_render_reference_cell_state_exact_review.py",
            SCRIPTS / "38_import_reference_cell_state_exact_review.R",
        ):
            self.assertNotIn('"--force"', path.read_text(), path.name)


class ExactReviewRendererTests(unittest.TestCase):
    def test_renderer_preserves_exact_selection_and_two_channels(self) -> None:
        probe = subprocess.run(
            [str(CELLPOSE_PYTHON if CELLPOSE_PYTHON.is_file() else Path(sys.executable)), "-c", "import numpy,tifffile,PIL"],
            check=False,
        )
        if probe.returncode:
            self.skipTest("renderer Python image dependencies are unavailable")
        python = str(CELLPOSE_PYTHON if CELLPOSE_PYTHON.is_file() else Path(sys.executable))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project_dir = root / "reference" / "projection_input" / "representative_umap"
            project_dir.mkdir(parents=True)
            make_images = subprocess.run(
                [python, "-c", (
                    "import numpy as np,tifffile,sys,pathlib;"
                    "r=pathlib.Path(sys.argv[1]);a=np.arange(64,dtype=np.uint16).reshape(8,8);"
                    "m=np.zeros((8,8),dtype=np.uint16);m[2:6,2:6]=1;"
                    "tifffile.imwrite(r/'bf.tif',a);tifffile.imwrite(r/'nuclei.tif',np.flipud(a));"
                    "tifffile.imwrite(r/'mask.tif',m)"
                ), str(root)], check=False, capture_output=True, text=True,
            )
            self.assertEqual(make_images.returncode, 0, make_images.stderr)
            project = {
                "cells_file": "cells.tsv", "images_file": "images.tsv",
            }
            (project_dir / "project.yml").write_text(json.dumps(project) + "\n")
            cell_id = "original|A01_1_0d0h0m|1"
            write_tsv(project_dir / "cells.tsv", ["cell_id", "image_id", "mask_label"], [
                {"cell_id": cell_id, "image_id": "A01_1_0d0h0m", "mask_label": 1}
            ])
            image_rows = []
            for channel, image in (("brightfield", root / "bf.tif"), ("nuclei", root / "nuclei.tif")):
                image_rows.append({
                    "image_id": "A01_1_0d0h0m", "channel_id": channel,
                    "image_path": image, "channel_index": "", "page_index": "",
                    "display_percentile_low": 1, "display_percentile_high": 99,
                    "mask_path": root / "mask.tif", "mask_channel_index": "",
                    "width": 8, "height": 8,
                })
            write_tsv(project_dir / "images.tsv", list(image_rows[0]), image_rows)
            selection = root / "reference" / "human_review" / "seed1" / "selection"
            selection.mkdir(parents=True)
            review = selection / "seed1_review_set.tsv"
            write_tsv(review, [
                "morphology_umap_row_key", "review_default_label",
                "review_sampling_bucket", "context_key", "source_id",
            ], [{
                "morphology_umap_row_key": cell_id, "review_default_label": "uncertain",
                "review_sampling_bucket": "all_unassigned_global_balanced_fallback",
                "context_key": "SUM-159-NLS-2N", "source_id": "A01",
            }])
            ids_hash = hashlib.sha256(cell_id.encode()).hexdigest()
            manifest = selection / "seed1_review_manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "reference_cell_state_seed1_review_v2",
                "row_count": 1, "stable_id_sha256": ids_hash,
            }) + "\n")
            output = root / "reference" / "human_review" / "seed1" / "render"
            completed = subprocess.run([
                python, str(RENDER), "--project", str(project_dir / "project.yml"),
                "--review-set", str(review), "--review-manifest", str(manifest),
                "--output-dir", str(output),
            ], text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            render_manifest = json.loads((output / "exact_review_render_manifest.json").read_text())
            self.assertTrue(render_manifest["exact_selection_preserved"])
            self.assertFalse(render_manifest["secondary_sampling_performed"])
            self.assertEqual(render_manifest["channels"], ["brightfield", "nuclei_support"])
            self.assertTrue(render_manifest["all_crops_available"])
            self.assertEqual(len(list((output / "crops").glob("*.png"))), 2)
            review_html = (output / "exact_review.html").read_text(encoding="utf-8")
            for expected in (
                "localStorage",
                "I inspected this cell",
                "cells still require explicit inspection",
                "reference-cell-state-v2-exact-review-guided",
                "multinucleated_cell</b> only when at least two distinct nuclei",
            ):
                self.assertIn(expected, review_html)

            reason = r_prerequisites()
            if reason:
                self.skipTest(reason)
            identity = render_manifest["identity"]
            submission = output / "exact_review_submission.json"
            submission.write_text(json.dumps({
                "schema_version": "reference_cell_state_exact_review_submission_v2",
                "identity": identity,
                "creation_metadata": {
                    "created_at": "2026-08-12T12:00:00Z",
                    "client_version": "fixture",
                },
                "rows": [{
                    "morphology_umap_row_key": cell_id,
                    "label": "live_cell",
                    "label_confidence": "high",
                    "reviewer": "fixture-reviewer",
                    "reviewed_at": "2026-08-12T12:00:00Z",
                    "review_notes": "manual fixture",
                }],
            }) + "\n")
            imported = root / "reference" / "human_review" / "seed1" / "import"
            completed = subprocess.run([
                "Rscript", str(IMPORT),
                "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK),
                "--review-set", str(review),
                "--render-dir", str(output),
                "--submission", str(submission),
                "--output-dir", str(imported),
            ], text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            _, reviewed_rows = read_tsv(imported / "reviewed_labels.tsv")
            self.assertEqual(len(reviewed_rows), 1)
            self.assertEqual(reviewed_rows[0]["label"], "live_cell")
            self.assertEqual(reviewed_rows[0]["reviewer"], "fixture-reviewer")
            import_manifest = json.loads((imported / "review_import_manifest.json").read_text())
            self.assertTrue(import_manifest["exact_selection_preserved"])
            self.assertFalse(import_manifest["secondary_sampling_performed"])

            def run_import(
                *, render_dir: Path = output, submission_path: Path = submission,
                import_dir: Path,
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.run([
                    "Rscript", str(IMPORT),
                    "--reference-root", str(REFERENCE),
                    "--dependency-lock", str(LOCK),
                    "--review-set", str(review),
                    "--render-dir", str(render_dir),
                    "--submission", str(submission_path),
                    "--output-dir", str(import_dir),
                ], text=True, capture_output=True, check=False)

            reused = run_import(import_dir=imported)
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertIn("exact_review_import_verified_reuse=1", reused.stdout)

            manifest_path = imported / "review_import_manifest.json"
            original_manifest = manifest_path.read_bytes()
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_manifest = json.loads(original_manifest)
                tampered_manifest[field] = "0" * 64
                manifest_path.write_text(json.dumps(tampered_manifest) + "\n")
                rejected = run_import(import_dir=imported)
                self.assertNotEqual(rejected.returncode, 0, field)
                self.assertIn(f"identity differs for {field}", rejected.stderr)
                manifest_path.write_bytes(original_manifest)

            # Every render input and every crop is hash-bound at import time.
            for role, path in (
                ("project", project_dir / "project.yml"),
                ("cells", project_dir / "cells.tsv"),
                ("images", project_dir / "images.tsv"),
            ):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                rejected = run_import(import_dir=imported.parent / f"tampered_{role}")
                self.assertNotEqual(rejected.returncode, 0, role)
                self.assertIn("Render input hash mismatch", rejected.stderr)
                path.write_bytes(original)

            crop = next((output / "crops").glob("*.png"))
            crop_bytes = crop.read_bytes()
            crop.write_bytes(crop_bytes + b"tamper")
            rejected = run_import(import_dir=imported.parent / "tampered_crop")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Crop artifact hash mismatch", rejected.stderr)
            crop.write_bytes(crop_bytes)

            submission_payload = json.loads(submission.read_text())
            submission_payload["rows"][0]["reviewer"] = "different-reviewer"
            submission.write_text(json.dumps(submission_payload) + "\n")
            rejected = run_import(import_dir=imported)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Existing review import identity differs", rejected.stderr)

            # Same stable IDs are insufficient: a submission exported from one
            # crop/render generation cannot authorize a second render.
            old_submission = output / "old_render_submission.json"
            old_submission.write_text(json.dumps({
                **submission_payload,
                "identity": identity,
                "rows": [{
                    **submission_payload["rows"][0],
                    "reviewer": "fixture-reviewer",
                }],
            }) + "\n")
            second_render = output.parent / "render_padding_zero"
            rerendered = subprocess.run([
                python, str(RENDER), "--project", str(project_dir / "project.yml"),
                "--review-set", str(review), "--review-manifest", str(manifest),
                "--output-dir", str(second_render), "--padding", "0",
            ], text=True, capture_output=True, check=False)
            self.assertEqual(rerendered.returncode, 0, rerendered.stderr)
            rejected = run_import(
                render_dir=second_render, submission_path=old_submission,
                import_dir=imported.parent / "cross_render_submission",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Submission/render identity mismatch", rejected.stderr)


class PlateAndFeatureParityRuntimeTests(unittest.TestCase):
    def test_real_80_well_contract_and_canonical_nested_index(self) -> None:
        if shutil.which("Rscript") is None:
            self.skipTest("Rscript is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            project_dir = root / "shadow" / "projection_input" / "representative_umap_v2"
            parent.mkdir()
            project_dir.mkdir(parents=True)
            fields = [
                "well", "plate_row", "plate_column", "doxorubicin_nm", "ploidy",
                "cyclophosphamide", "replicate", "split", "split_strategy",
                "split_seed", "assignment_sha256",
            ]
            wells = [f"{row}{column}" for row in "ABCDEFGH" for column in range(2, 12)]
            split_rows = []
            for index, well in enumerate(wells):
                split_rows.append({
                    "well": well, "plate_row": well[0], "plate_column": int(well[1:]),
                    "doxorubicin_nm": "0" if index % 2 == 0 else "25",
                    "ploidy": "2N" if index % 2 == 0 else "4N",
                    "cyclophosphamide": "false" if index % 4 < 2 else "true",
                    "replicate": 1 if index % 2 == 0 else 2,
                    "split": "development" if index < 64 else "heldout",
                    "split_strategy": "fixture_frozen", "split_seed": 42,
                    "assignment_sha256": hashlib.sha256(well.encode()).hexdigest(),
                })
            split_path = parent / "well_split_freeze.tsv"
            write_tsv(split_path, fields, split_rows)
            projection_manifest = parent / "projection_input_manifest.json"
            projection_manifest.write_text(json.dumps({
                "well_split_freeze": str(split_path.resolve()),
                "well_split_freeze_sha256": sha(split_path),
            }) + "\n")
            cells = project_dir / "cells.tsv"
            cell_rows = []
            for index, row in enumerate(split_rows[:64]):
                well = str(row["well"])
                for label in (2, 1):
                    cell_id = f"original|{well}_1_0d0h0m|{label}"
                    cell_rows.append({
                        "cell_id": cell_id, "well": well, "site": 1,
                        "elapsed_hours": 0, "mask_label": label,
                        "context_key": f"SUM-159-NLS-{row['ploidy']}",
                        "ltee_cell_line": f"SUM-159-NLS-{row['ploidy']}",
                        "ltee_cell_line_family": "SUM-159-NLS", "source_id": well,
                        "suffix": "site=1|elapsed_hours=0",
                        "condition": (
                            f"doxorubicin_nm={row['doxorubicin_nm']}|ploidy={row['ploidy']}|"
                            f"cyclophosphamide={row['cyclophosphamide']}"
                        ),
                        "feature_row_index": 2 if label == 2 else 1,
                        "segmentation_qc_cell_id": cell_id,
                        "segmentation_object_id": cell_id,
                        "cluster": index % 3,
                    })
            cell_fields = list(cell_rows[0])
            write_tsv(cells, cell_fields, cell_rows)
            project = project_dir / "project.yml"
            project.write_text(json.dumps({"cells_file": "cells.tsv"}) + "\n")
            parent_import = project_dir / "parent_import_manifest.json"
            parent_import.write_text(json.dumps({
                "parent": {
                    "projection_input_manifest": str(projection_manifest.resolve()),
                    "input_file_sha256": {
                        "projection_input_manifest": sha(projection_manifest),
                    },
                },
            }) + "\n")
            program = (
                f'source("{SHARED}"); '
                f'contract <- reference_cell_state_v2_resolve_plate_contract("{project}"); '
                f'cells <- reference_cell_state_v2_read_table("{cells}", "cells"); '
                'valid <- reference_cell_state_v2_validate_canonical_cells(cells, contract); '
                'stopifnot(nrow(contract$rows)==80L, contract$development_well_count==64L, '
                'setequal(unique(contract$rows$context_key), c("SUM-159-NLS-2N","SUM-159-NLS-4N")), '
                '!any(grepl("replicate=", contract$rows$condition, fixed=TRUE)), '
                'identical(as.integer(valid$feature_row_index[1:2]), c(2L,1L)))'
            )
            completed = subprocess.run(
                ["Rscript", "-e", program], text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_python_bf_features_equal_pinned_historical_r_at_1e_12(self) -> None:
        reason = r_prerequisites()
        if reason:
            self.skipTest(reason)
        python = CELLPOSE_PYTHON if CELLPOSE_PYTHON.is_file() else Path(sys.executable)
        python_program = f'''
import importlib.util,json,sys,numpy as np
spec=importlib.util.spec_from_file_location("bpf_fixture",r"{FEATURE_IMPL}")
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
raw=np.array([[0,5,10,15,20],[2,7,12,17,22],[4,9,14,19,24],[6,11,16,21,26],[8,13,18,23,28]],dtype=float)
mask=np.zeros((5,5),dtype=np.int64);mask[1:4,1:4]=1
rows,_=module.extract_broad_phenotype_features(raw,mask)
print(json.dumps([rows[0][x] for x in ("bf_boundary_mean","bf_interior_mean","bf_interior_minus_boundary_mean")]))
'''
        python_result = subprocess.run(
            [str(python), "-c", python_program], text=True, capture_output=True, check=False,
        )
        if python_result.returncode:
            self.skipTest(f"Python feature dependency unavailable: {python_result.stderr.strip()}")
        observed = json.loads(python_result.stdout)
        r_program = f'''
          source("{SHARED}")
          loaded <- reference_cell_state_v2_load_historical_classifier("{REFERENCE}")
          raw <- matrix(c(0,5,10,15,20,2,7,12,17,22,4,9,14,19,24,6,11,16,21,26,8,13,18,23,28), nrow=5, byrow=TRUE)
          mask <- matrix(FALSE, 5, 5); mask[2:4,2:4] <- TRUE
          norm <- loaded$api$morphology_candidate_normalize_raw_image(raw)$matrix
          values <- loaded$api$morphology_candidate_boundary_interior_summary(norm, mask)$values
          cat(jsonlite::toJSON(as.numeric(values), auto_unbox=TRUE))
        '''
        completed = subprocess.run(
            ["Rscript", "-e", r_program], text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected = json.loads(completed.stdout)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(observed, expected)), 1e-12)


class HistoricalReviewExecutionTests(unittest.TestCase):
    def test_nonempty_seed1_and_diagnostic_only_seed2_execute_and_reuse(self) -> None:
        reason = r_prerequisites()
        if reason:
            self.skipTest(reason)
        glmnet = subprocess.run(
            ["Rscript", "-e", 'quit(status=ifelse(requireNamespace("glmnet",quietly=TRUE),0,1))'],
            check=False,
        )
        if glmnet.returncode:
            self.skipTest("glmnet is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            shadow = root / "shadow"
            project_dir = shadow / "projection_input" / "representative_umap_v2"
            parent.mkdir()
            project_dir.mkdir(parents=True)
            split_fields = [
                "well", "plate_row", "plate_column", "doxorubicin_nm", "ploidy",
                "cyclophosphamide", "replicate", "split", "split_strategy",
                "split_seed", "assignment_sha256",
            ]
            wells = [f"{row}{column}" for row in "ABCDEFGH" for column in range(2, 12)]
            split_rows = []
            for index, well in enumerate(wells):
                split_rows.append({
                    "well": well, "plate_row": well[0], "plate_column": int(well[1:]),
                    "doxorubicin_nm": "0" if index % 2 == 0 else "25",
                    "ploidy": "2N" if index % 2 == 0 else "4N",
                    "cyclophosphamide": "false" if index % 4 < 2 else "true",
                    "replicate": 1 if index % 2 == 0 else 2,
                    "split": "development" if index < 64 else "heldout",
                    "split_strategy": "fixture_frozen", "split_seed": 42,
                    "assignment_sha256": hashlib.sha256(well.encode()).hexdigest(),
                })
            split_path = parent / "well_split_freeze.tsv"
            write_tsv(split_path, split_fields, split_rows)
            projection_manifest = parent / "projection_input_manifest.json"
            projection_manifest.write_text(json.dumps({
                "well_split_freeze": str(split_path.resolve()),
                "well_split_freeze_sha256": sha(split_path),
            }) + "\n")
            cell_rows: list[dict[str, object]] = []
            feature_rows: list[dict[str, object]] = []
            diagnostic_rows: list[dict[str, object]] = []
            for well_index, plate in enumerate(split_rows[:64]):
                well = str(plate["well"])
                for label in range(1, 17):
                    cell_id = f"original|{well}_1_0d0h0m|{label}"
                    context = f"SUM-159-NLS-{plate['ploidy']}"
                    condition = (
                        f"doxorubicin_nm={plate['doxorubicin_nm']}|ploidy={plate['ploidy']}|"
                        f"cyclophosphamide={plate['cyclophosphamide']}"
                    )
                    cell_rows.append({
                        "cell_id": cell_id, "well": well, "site": 1,
                        "elapsed_hours": 0, "mask_label": label,
                        "context_key": context, "ltee_cell_line": context,
                        "ltee_cell_line_family": "SUM-159-NLS", "source_id": well,
                        "suffix": "site=1|elapsed_hours=0", "condition": condition,
                        "feature_row_index": label, "segmentation_qc_cell_id": cell_id,
                        "segmentation_object_id": cell_id, "cluster": 1,
                    })
                    class_index = (label - 1) % 3
                    base = (class_index + 1) * 20 + well_index / 100 + label / 1000
                    feature_rows.append({
                        "cell_id": cell_id,
                        "area_px2": base * base, "perimeter_px": base * 2,
                        "roundness": 0.3 + class_index * 0.2 + label / 10000,
                        "aspect_ratio": 1 + class_index * 0.5 + label / 10000,
                        "extent": 0.4 + class_index * 0.15 + label / 10000,
                        "solidity": 0.5 + class_index * 0.15 + label / 10000,
                        "equivalent_diameter_px": base, "major_axis_px": base * 1.2,
                        "minor_axis_px": base * 0.8,
                        "bf_boundary_mean": 0.15 + class_index * 0.25 + label / 10000,
                        "bf_interior_mean": 0.2 + class_index * 0.25 + label / 10000,
                        "bf_interior_minus_boundary_mean": 0.05,
                    })
                    diagnostic_rows.append({
                        "cell_id": cell_id, "cluster": 1,
                        "cluster_source": "optimize_dbscan_v4_one_cluster_fallback",
                    })
            write_tsv(project_dir / "cells.tsv", list(cell_rows[0]), cell_rows)
            write_tsv(project_dir / "features.tsv", list(feature_rows[0]), feature_rows)
            project_path = project_dir / "project.yml"
            project_path.write_text(json.dumps({
                "cells_file": "cells.tsv", "features_file": "features.tsv",
                "classifier": {"feature_columns": list(CURRENT12)},
            }) + "\n")
            (project_dir / "parent_import_manifest.json").write_text(json.dumps({
                "parent": {
                    "shadow_root": str(parent.resolve()),
                    "projection_input_manifest": str(projection_manifest.resolve()),
                    "input_file_sha256": {"projection_input_manifest": sha(projection_manifest)},
                },
            }) + "\n")
            historical = project_dir / "historical_projection"
            historical.mkdir()
            clusters = historical / "diagnostic_clusters.tsv"
            write_tsv(clusters, list(diagnostic_rows[0]), diagnostic_rows)
            diagnostic_manifest = historical / "historical_projection_manifest.json"
            diagnostic_manifest.write_text(json.dumps({
                "schema_version": "reference_cell_state_historical_projection_v2",
                "status": "COMPLETE",
                "diagnostic_cluster": {"one_cluster_fallback": True, "role": "metadata_only"},
                "output_file_sha256": {"diagnostic_clusters.tsv": sha(clusters)},
            }) + "\n")

            seed1_selection = shadow / "human_review" / "seed1" / "selection"
            seed1_command = [
                "Rscript", str(SEED1), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK), "--project", str(project_path),
                "--diagnostic-cluster-manifest", str(diagnostic_manifest),
                "--output-dir", str(seed1_selection),
            ]
            selected = subprocess.run(seed1_command, text=True, capture_output=True, check=False)
            self.assertEqual(selected.returncode, 0, selected.stderr)
            _, seed1_rows = read_tsv(seed1_selection / "seed1_review_set.tsv")
            self.assertEqual(len(seed1_rows), 500)
            counts: dict[str, int] = {}
            for row in seed1_rows:
                counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
            self.assertLessEqual(max(counts.values()), 8)
            selected_again = subprocess.run(seed1_command, text=True, capture_output=True, check=False)
            self.assertEqual(selected_again.returncode, 0, selected_again.stderr)
            self.assertIn("seed1_review_verified_reuse=1", selected_again.stdout)
            seed1_manifest = json.loads((seed1_selection / "seed1_review_manifest.json").read_text())
            self.assertEqual(seed1_manifest["historical_parity_scope"],
                             "historical_core_parity_with_sampling_adaptation")
            self.assertEqual(seed1_manifest["implementation_sha256"], sha(SEED1))
            self.assertEqual(seed1_manifest["shared_implementation_sha256"], sha(SHARED))
            seed1_manifest_path = seed1_selection / "seed1_review_manifest.json"
            seed1_manifest_bytes = seed1_manifest_path.read_bytes()
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_seed1_manifest = dict(seed1_manifest)
                tampered_seed1_manifest[field] = "0" * 64
                seed1_manifest_path.write_text(json.dumps(tampered_seed1_manifest) + "\n")
                rejected_seed1_reuse = subprocess.run(
                    seed1_command, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_seed1_reuse.returncode, 0, field)
                self.assertIn(f"identity differs for {field}", rejected_seed1_reuse.stderr)
                seed1_manifest_path.write_bytes(seed1_manifest_bytes)

            # A multi-cluster projection that fails the preregistered
            # size/stability gates must use blind review without falsely
            # claiming a one-cluster fallback.
            decision = historical / "labelability_decision.json"
            expanded_manifest = historical / "expanded_projection_manifest.json"
            decision.write_text(json.dumps({
                "schema_version": "reference_cell_state_expanded_labelability_decision_v1",
                "status": "COMPLETE",
                "selected_annotation_profile": "expanded39",
                "computational_gate": "FAIL",
                "overall_labelability": "NO_GO_FOR_POLYGON_ANNOTATION_USE_500_CELL_BLIND_REVIEW",
            }) + "\n")
            expanded_manifest.write_text(json.dumps({
                "schema_version": "reference_cell_state_expanded_projection_comparison_v1",
                "status": "COMPLETE",
                "selected_annotation_profile": "expanded39",
                "classifier_boundary": {
                    "final_classifier_feature_source": "classifier12",
                    "expanded39_allowed_in_final_classifier": False,
                },
            }) + "\n")
            standard_parent_import = (project_dir / "parent_import_manifest.json").read_bytes()
            base_lineage = shadow / "expanded_base_lineage"
            base_lineage.mkdir()
            base_project = base_lineage / "project.yml"
            base_project.write_bytes(project_path.read_bytes())
            (base_lineage / "cells.tsv").write_bytes((project_dir / "cells.tsv").read_bytes())
            (base_lineage / "parent_import_manifest.json").write_bytes(standard_parent_import)
            (project_dir / "parent_import_manifest.json").write_text(json.dumps({
                "schema_version": "reference_cell_state_expanded_annotation_parent_import_v1",
                "status": "COMPLETE",
                "inputs": {
                    "base_project": {"path": str(base_project), "sha256": sha(base_project)},
                },
            }) + "\n")
            one_cluster_manifest_bytes = diagnostic_manifest.read_bytes()
            diagnostic_manifest.write_text(json.dumps({
                "schema_version": "reference_cell_state_historical_projection_expanded_v1",
                "status": "COMPLETE",
                "selected_annotation_profile": "expanded39",
                "expanded_feature_role": "annotation_geometry_and_human_morphology_evidence_only",
                "expanded_features_allowed_in_final_classifier": False,
                "computational_labelability_gate": "FAIL",
                "labelability_decision_sha256": sha(decision),
                "expanded_projection_manifest_sha256": sha(expanded_manifest),
                "output_file_sha256": {
                    "diagnostic_clusters.tsv": sha(clusters),
                    "expanded_projection_manifest.json": sha(expanded_manifest),
                    "labelability_decision.json": sha(decision),
                },
            }) + "\n")
            expanded_selection = shadow / "human_review" / "seed1_expanded" / "selection"
            expanded_command = [
                "Rscript", str(SEED1), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK), "--project", str(project_path),
                "--expanded-labelability-decision", str(decision),
                "--output-dir", str(expanded_selection),
            ]
            expanded_selected = subprocess.run(
                expanded_command, text=True, capture_output=True, check=False,
            )
            self.assertEqual(expanded_selected.returncode, 0, expanded_selected.stderr)
            _, expanded_rows = read_tsv(expanded_selection / "seed1_review_set.tsv")
            self.assertEqual(len(expanded_rows), 500)
            expanded_review_manifest = json.loads(
                (expanded_selection / "seed1_review_manifest.json").read_text()
            )
            self.assertEqual(
                expanded_review_manifest["selection_input_mode"],
                "authoritative_expanded_computational_nogo",
            )
            self.assertTrue(expanded_review_manifest["computational_nogo_fallback_proven"])
            self.assertFalse(expanded_review_manifest["one_cluster_fallback_proven"])
            self.assertFalse(
                expanded_review_manifest["sampling_adaptation"]["reference_native_sampling_claimed"]
            )
            expanded_reuse = subprocess.run(
                expanded_command, text=True, capture_output=True, check=False,
            )
            self.assertEqual(expanded_reuse.returncode, 0, expanded_reuse.stderr)
            self.assertIn("seed1_review_verified_reuse=1", expanded_reuse.stdout)
            diagnostic_manifest.write_bytes(one_cluster_manifest_bytes)
            (project_dir / "parent_import_manifest.json").write_bytes(standard_parent_import)

            fields, seed1_rows = read_tsv(seed1_selection / "seed1_review_label_template.tsv")
            classes = ("live_cell", "dead_cell", "multinucleated_cell")
            for row in seed1_rows:
                row["label"] = classes[(int(row["mask_label"]) - 1) % 3]
                row["reviewer"] = "fixture-reviewer"
                row["reviewed_at"] = "2026-08-12T12:00:00Z"
                row["label_confidence"] = "high"
            seed1_import = shadow / "human_review" / "seed1" / "import"
            reviewed_path = seed1_import / "reviewed_labels.tsv"
            write_tsv(reviewed_path, fields, seed1_rows)
            render_manifest = shadow / "human_review" / "seed1" / "render" / "exact_review_render_manifest.json"
            render_manifest.parent.mkdir(parents=True)
            selection_manifest = seed1_selection / "seed1_review_manifest.json"
            render_manifest.write_text(json.dumps({
                "inputs": {"review_manifest": {
                    "path": str(selection_manifest.resolve()), "sha256": sha(selection_manifest),
                }},
            }) + "\n")
            (seed1_import / "review_import_manifest.json").write_text(json.dumps({
                "schema_version": "reference_cell_state_exact_review_import_v2",
                "status": "COMPLETE", "exact_selection_preserved": True,
                "secondary_sampling_performed": False,
                "reviewed_labels_sha256": sha(reviewed_path),
                "render_manifest": str(render_manifest.resolve()),
                "render_manifest_sha256": sha(render_manifest),
                "implementation": str(IMPORT.resolve()),
                "implementation_sha256": sha(IMPORT),
                "shared_implementation": str(SHARED.resolve()),
                "shared_implementation_sha256": sha(SHARED),
            }) + "\n")
            initial_model = shadow / "models" / "initial"
            initial_receipt = shadow / "workflow_status" / "training_v2" / "initial_train_receipt.json"
            train_command = [
                "Rscript", str(TRAIN), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK), "--feature-config", str(CONFIG),
                "--project", str(project_path), "--reviewed-labels", str(reviewed_path),
                "--training-stage", "initial", "--output-dir", str(initial_model),
                "--output-receipt", str(initial_receipt),
            ]
            trained = subprocess.run(train_command, text=True, capture_output=True, check=False)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            trained_again = subprocess.run(train_command, text=True, capture_output=True, check=False)
            self.assertEqual(trained_again.returncode, 0, trained_again.stderr)
            self.assertIn("historical_train_verified_reuse=1", trained_again.stdout)
            authoritative_initial_receipt = initial_model / "training_receipt.json"
            initial_authority_bytes = authoritative_initial_receipt.read_bytes()
            initial_mirror_bytes = initial_receipt.read_bytes()
            initial_payload = json.loads(initial_authority_bytes)
            self.assertEqual(initial_payload["implementation_sha256"], sha(TRAIN))
            self.assertEqual(initial_payload["shared_implementation_sha256"], sha(SHARED))
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_initial_payload = dict(initial_payload)
                tampered_initial_payload[field] = "0" * 64
                bad_receipt = (
                    json.dumps(tampered_initial_payload, indent=2, sort_keys=True) + "\n"
                ).encode()
                authoritative_initial_receipt.write_bytes(bad_receipt)
                initial_receipt.write_bytes(bad_receipt)
                rejected_train_reuse = subprocess.run(
                    train_command, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_train_reuse.returncode, 0, field)
                self.assertIn("training generation identity changed", rejected_train_reuse.stderr)
                authoritative_initial_receipt.write_bytes(initial_authority_bytes)
                initial_receipt.write_bytes(initial_mirror_bytes)

            seed2_selection = shadow / "human_review" / "seed2" / "selection"
            seed2_command = [
                "Rscript", str(SEED2), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK), "--project", str(project_path),
                "--diagnostic-cluster-manifest", str(diagnostic_manifest),
                "--initial-model-dir", str(initial_model),
                "--seed1-reviewed-labels", str(reviewed_path),
                "--output-dir", str(seed2_selection),
            ]
            targeted = subprocess.run(seed2_command, text=True, capture_output=True, check=False)
            self.assertEqual(targeted.returncode, 0, targeted.stderr)
            _, seed2_rows = read_tsv(seed2_selection / "seed2_review_set.tsv")
            self.assertEqual(len(seed2_rows), 200)
            seed2_manifest = json.loads((seed2_selection / "seed2_review_manifest.json").read_text())
            self.assertEqual(seed2_manifest["historical_parity_scope"],
                             "historical_core_parity_with_sampling_adaptation")
            self.assertFalse(seed2_manifest["fallback_sampling_adapter"]["reference_native_sampling_claimed"])
            self.assertEqual(seed2_manifest["implementation_sha256"], sha(SEED2))
            self.assertEqual(seed2_manifest["shared_implementation_sha256"], sha(SHARED))
            targeted_again = subprocess.run(seed2_command, text=True, capture_output=True, check=False)
            self.assertEqual(targeted_again.returncode, 0, targeted_again.stderr)
            self.assertIn("seed2_review_verified_reuse=1", targeted_again.stdout)
            seed2_manifest_path = seed2_selection / "seed2_review_manifest.json"
            seed2_manifest_bytes = seed2_manifest_path.read_bytes()
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_seed2_manifest = dict(seed2_manifest)
                tampered_seed2_manifest[field] = "0" * 64
                seed2_manifest_path.write_text(json.dumps(tampered_seed2_manifest) + "\n")
                rejected_seed2_reuse = subprocess.run(
                    seed2_command, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_seed2_reuse.returncode, 0, field)
                self.assertIn("generation identity differs", rejected_seed2_reuse.stderr)
                seed2_manifest_path.write_bytes(seed2_manifest_bytes)

            final_model = shadow / "models" / "final"
            final_receipt = shadow / "workflow_status" / "training_v2" / "final_train_receipt.json"
            final_train_command = [
                *train_command[:],
            ]
            final_train_command[final_train_command.index("initial")] = "final"
            final_train_command[final_train_command.index(str(initial_model))] = str(final_model)
            final_train_command[final_train_command.index(str(initial_receipt))] = str(final_receipt)
            final_trained = subprocess.run(
                final_train_command, text=True, capture_output=True, check=False,
            )
            self.assertEqual(final_trained.returncode, 0, final_trained.stderr)
            acceptance = shadow / "workflow_status" / "model_acceptance" / "accepted.json"
            accept_command = [
                "Rscript", str(ACCEPT), "--shadow-root", str(shadow),
                "--project", str(project_path), "--model-dir", str(final_model),
                "--train-receipt", str(final_receipt), "--feature-config", str(CONFIG),
                "--classes-file", str(CLASSES_V2),
                "--parent-import-manifest", str(project_dir / "parent_import_manifest.json"),
                "--reference-root", str(REFERENCE), "--dependency-lock", str(LOCK),
                "--output-receipt", str(acceptance),
            ]
            accepted = subprocess.run(accept_command, text=True, capture_output=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            accepted_again = subprocess.run(accept_command, text=True, capture_output=True, check=False)
            self.assertEqual(accepted_again.returncode, 0, accepted_again.stderr)
            self.assertIn("model_acceptance_verified_reuse=1", accepted_again.stdout)
            acceptance_payload = json.loads(acceptance.read_text())
            self.assertFalse(acceptance_payload["model"]["unit_adaptation"]["valid_microscopy_calibration"])
            self.assertFalse(acceptance_payload["model"]["unit_adaptation"]["physical_micron_parity_claimed"])
            self.assertEqual(acceptance_payload["shared_implementation_sha256"], sha(SHARED))
            acceptance_bytes = acceptance.read_bytes()
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_acceptance = dict(acceptance_payload)
                tampered_acceptance[field] = "0" * 64
                acceptance.write_text(json.dumps(tampered_acceptance) + "\n")
                rejected_acceptance_reuse = subprocess.run(
                    accept_command, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_acceptance_reuse.returncode, 0, field)
                self.assertIn("acceptance identity differs", rejected_acceptance_reuse.stderr)
                acceptance.write_bytes(acceptance_bytes)

            feature_generation = parent / "feature_shards" / "A2_1_0d0h0m__original"
            feature_generation.mkdir(parents=True)
            feature_shard = feature_generation / "features.tsv"
            shard_rows = []
            for row in feature_rows[:16]:
                label = int(str(row["cell_id"]).rsplit("|", 1)[1])
                shard_rows.append({
                    "cell_id": row["cell_id"], "key": "A2_1_0d0h0m", "well": "A2",
                    "branch": "original", "image_id": "A2_1_0d0h0m", "mask_label": label,
                    "feature_schema_version": "broad_phenotype_features_v1",
                    **{name: row[name] for name in CURRENT12},
                })
            write_tsv(feature_shard, list(shard_rows[0]), shard_rows)
            feature_receipt = feature_generation / "receipt.json"
            feature_receipt.write_text(json.dumps({
                "schema_version": "broad_phenotype_feature_receipt_v1", "status": "COMPLETE",
                "key": "A2_1_0d0h0m", "well": "A2", "branch": "original",
                "feature_schema_version": "broad_phenotype_features_v1",
                "feature_columns": list(shard_rows[0]),
                "feature_tsv": str(feature_shard.resolve()),
                "feature_tsv_sha256": sha(feature_shard), "row_count": len(shard_rows),
                "cell_id_sha256": hashlib.sha256(
                    "\n".join(str(row["cell_id"]) for row in shard_rows).encode()
                ).hexdigest(),
            }) + "\n")
            prediction_generation = (
                shadow / "prediction_shards" / "shards" / "A2" /
                "A2_1_0d0h0m__original"
            )
            prediction_tsv = prediction_generation / "reference_cell_state_predictions.tsv"
            prediction_receipt = prediction_generation / "prediction_receipt.json"
            predict_command = [
                "Rscript", str(PREDICT), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK), "--model-dir", str(final_model),
                "--model-acceptance-receipt", str(acceptance),
                "--model-acceptance-sha256", sha(acceptance),
                "--feature-shard", str(feature_shard),
                "--feature-receipt", str(feature_receipt),
                "--output-tsv", str(prediction_tsv),
                "--output-receipt", str(prediction_receipt),
            ]
            predicted = subprocess.run(predict_command, text=True, capture_output=True, check=False)
            self.assertEqual(predicted.returncode, 0, predicted.stderr)
            self.assertTrue(prediction_tsv.is_file())
            self.assertTrue(prediction_receipt.is_file())
            predicted_again = subprocess.run(predict_command, text=True, capture_output=True, check=False)
            self.assertEqual(predicted_again.returncode, 0, predicted_again.stderr)
            self.assertIn("shard_prediction_already_complete=1", predicted_again.stdout)
            prediction_receipt_bytes = prediction_receipt.read_bytes()
            prediction_payload = json.loads(prediction_receipt_bytes)
            self.assertEqual(prediction_payload["implementation_sha256"], sha(PREDICT))
            self.assertEqual(prediction_payload["shared_implementation_sha256"], sha(SHARED))
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_prediction_payload = dict(prediction_payload)
                tampered_prediction_payload[field] = "0" * 64
                prediction_receipt.write_text(json.dumps(tampered_prediction_payload) + "\n")
                rejected_prediction_reuse = subprocess.run(
                    predict_command, text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_prediction_reuse.returncode, 0, field)
                self.assertIn(f"differs for {field}", rejected_prediction_reuse.stderr)
                prediction_receipt.write_bytes(prediction_receipt_bytes)

    def test_conflict_barrier_then_adjudication_complete_and_reuse(self) -> None:
        reason = r_prerequisites()
        if reason:
            self.skipTest(reason)
        probe = subprocess.run(
            [str(CELLPOSE_PYTHON if CELLPOSE_PYTHON.is_file() else Path(sys.executable)),
             "-c", "import numpy,tifffile,PIL"], check=False,
        )
        if probe.returncode:
            self.skipTest("renderer Python image dependencies are unavailable")
        python = str(CELLPOSE_PYTHON if CELLPOSE_PYTHON.is_file() else Path(sys.executable))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shadow = root / "shadow"
            project_dir = shadow / "projection_input" / "representative_umap_v2"
            project_dir.mkdir(parents=True)
            made = subprocess.run([
                python, "-c", (
                    "import numpy as np,tifffile,sys,pathlib;"
                    "r=pathlib.Path(sys.argv[1]);a=np.arange(64,dtype=np.uint16).reshape(8,8);"
                    "m=np.zeros((8,8),dtype=np.uint16);m[2:6,2:6]=1;"
                    "tifffile.imwrite(r/'bf.tif',a);tifffile.imwrite(r/'nuclei.tif',np.flipud(a));"
                    "tifffile.imwrite(r/'mask.tif',m)"
                ), str(root),
            ], text=True, capture_output=True, check=False)
            self.assertEqual(made.returncode, 0, made.stderr)
            (project_dir / "project.yml").write_text(json.dumps({
                "cells_file": "cells.tsv", "images_file": "images.tsv",
            }) + "\n")
            cell_id = "original|A02_1_0d0h0m|1"
            write_tsv(project_dir / "cells.tsv", ["cell_id", "image_id", "mask_label"], [{
                "cell_id": cell_id, "image_id": "A02_1_0d0h0m", "mask_label": 1,
            }])
            image_rows = []
            for channel, path in (("brightfield", root / "bf.tif"), ("nuclei", root / "nuclei.tif")):
                image_rows.append({
                    "image_id": "A02_1_0d0h0m", "channel_id": channel,
                    "image_path": path, "channel_index": "", "page_index": "",
                    "display_percentile_low": 1, "display_percentile_high": 99,
                    "mask_path": root / "mask.tif", "mask_channel_index": "",
                    "width": 8, "height": 8,
                })
            write_tsv(project_dir / "images.tsv", list(image_rows[0]), image_rows)

            imported_paths: dict[int, Path] = {}
            for seed, label in ((1, "live_cell"), (2, "dead_cell")):
                selection = shadow / "human_review" / f"seed{seed}" / "selection"
                selection.mkdir(parents=True)
                review = selection / f"seed{seed}_review_set.tsv"
                write_tsv(review, [
                    "morphology_umap_row_key", "review_default_label",
                    "review_sampling_bucket", "context_key", "source_id",
                ], [{
                    "morphology_umap_row_key": cell_id,
                    "review_default_label": "uncertain",
                    "review_sampling_bucket": "fixture_manual_review",
                    "context_key": "SUM-159-NLS-2N", "source_id": "A02",
                }])
                manifest = selection / f"seed{seed}_review_manifest.json"
                schema = (
                    "reference_cell_state_seed1_review_v2" if seed == 1
                    else "reference_cell_state_seed2_multinucleated_review_v2"
                )
                manifest.write_text(json.dumps({
                    "schema_version": schema, "status": "HUMAN_REVIEW_REQUIRED",
                    "row_count": 1, "stable_id_sha256": hashlib.sha256(cell_id.encode()).hexdigest(),
                    "output_file_sha256": {"review_set": sha(review)},
                }) + "\n")
                render = shadow / "human_review" / f"seed{seed}" / "render"
                rendered = subprocess.run([
                    python, str(RENDER), "--project", str(project_dir / "project.yml"),
                    "--review-set", str(review), "--review-manifest", str(manifest),
                    "--output-dir", str(render),
                ], text=True, capture_output=True, check=False)
                self.assertEqual(rendered.returncode, 0, rendered.stderr)
                identity = json.loads((render / "exact_review_render_manifest.json").read_text())["identity"]
                submission = render / "exact_review_submission.json"
                submission.write_text(json.dumps({
                    "schema_version": "reference_cell_state_exact_review_submission_v2",
                    "identity": identity,
                    "creation_metadata": {
                        "created_at": "2026-08-12T12:00:00Z", "client_version": "fixture",
                    },
                    "rows": [{
                        "morphology_umap_row_key": cell_id, "label": label,
                        "label_confidence": "high", "reviewer": f"reviewer-{seed}",
                        "reviewed_at": "2026-08-12T12:00:00Z", "review_notes": "fixture",
                    }],
                }) + "\n")
                imported = shadow / "human_review" / f"seed{seed}" / "import"
                imported_result = subprocess.run([
                    "Rscript", str(IMPORT), "--reference-root", str(REFERENCE),
                    "--dependency-lock", str(LOCK), "--review-set", str(review),
                    "--render-dir", str(render), "--submission", str(submission),
                    "--output-dir", str(imported),
                ], text=True, capture_output=True, check=False)
                self.assertEqual(imported_result.returncode, 0, imported_result.stderr)
                imported_paths[seed] = imported / "reviewed_labels.tsv"

            merged = shadow / "human_review" / "merged"
            base_command = [
                "Rscript", str(MERGE_REVIEWS), "--reference-root", str(REFERENCE),
                "--dependency-lock", str(LOCK),
                "--seed1-reviewed-labels", str(imported_paths[1]),
                "--seed2-reviewed-labels", str(imported_paths[2]),
                "--output-dir", str(merged),
            ]
            barrier_result = subprocess.run(base_command, text=True, capture_output=True, check=False)
            self.assertNotEqual(barrier_result.returncode, 0)
            self.assertIn("requires explicit human adjudication", barrier_result.stderr)
            barrier = shadow / "human_review" / "merged_adjudication_barrier"
            self.assertTrue((barrier / "review_adjudication_barrier_manifest.json").is_file())
            adjudication = shadow / "human_review" / "adjudication.tsv"
            adjudication_fields = [
                "morphology_umap_row_key", "label", "reviewer", "reviewed_at",
                "label_confidence", "adjudication_notes",
            ]
            valid_adjudication = {
                "morphology_umap_row_key": cell_id, "label": "multinucleated_cell",
                "reviewer": "adjudicator", "reviewed_at": "2026-08-12T13:00:00Z",
                "label_confidence": "high", "adjudication_notes": "manual conflict decision",
            }
            invalid_cases = (
                ("reviewer", "adjudicator\x01", "control characters"),
                ("reviewed_at", "2026-08-12 13:00:00", "RFC3339"),
                ("label_confidence", "HIGH", "confidence"),
                ("label", " live_cell", "must be trimmed"),
                ("adjudication_notes", "", "missing values"),
            )
            for field, value, error_text in invalid_cases:
                row = dict(valid_adjudication)
                row[field] = value
                write_tsv(adjudication, adjudication_fields, [row])
                rejected = subprocess.run(
                    [*base_command, "--adjudication", str(adjudication)],
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected.returncode, 0, field)
                self.assertIn(error_text, rejected.stderr, field)
                self.assertFalse(merged.exists(), field)
            write_tsv(adjudication, adjudication_fields, [valid_adjudication])
            completed = subprocess.run(
                [*base_command, "--adjudication", str(adjudication)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((merged / "review_merge_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertTrue(manifest["complete_human_evidence_chain_revalidated"])
            _, merged_rows = read_tsv(merged / "merged_reviewed_labels.tsv")
            self.assertEqual(merged_rows[0]["label"], "multinucleated_cell")
            reused = subprocess.run(
                [*base_command, "--adjudication", str(adjudication)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertIn("review_merge_verified_reuse=1", reused.stdout)
            merge_manifest_path = merged / "review_merge_manifest.json"
            merge_manifest_bytes = merge_manifest_path.read_bytes()
            merge_manifest = json.loads(merge_manifest_bytes)
            self.assertEqual(merge_manifest["implementation_sha256"], sha(MERGE_REVIEWS))
            self.assertEqual(merge_manifest["shared_implementation_sha256"], sha(SHARED))
            for field in ("implementation_sha256", "shared_implementation_sha256"):
                tampered_merge_manifest = dict(merge_manifest)
                tampered_merge_manifest[field] = "0" * 64
                merge_manifest_path.write_text(json.dumps(tampered_merge_manifest) + "\n")
                rejected_reuse = subprocess.run(
                    [*base_command, "--adjudication", str(adjudication)],
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(rejected_reuse.returncode, 0, field)
                self.assertIn("generation identity differs", rejected_reuse.stderr)
                merge_manifest_path.write_bytes(merge_manifest_bytes)


if __name__ == "__main__":
    unittest.main()
