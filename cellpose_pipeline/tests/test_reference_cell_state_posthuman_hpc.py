from __future__ import annotations

import os
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_HPC = REPO_ROOT / "cellpose_pipeline" / "Docker" / "hpc"
HOST_HPC = REPO_ROOT / "cellpose_pipeline" / "hpc"
SIF_PATH = (
    "/share/lab_crd/taoli/Docker/"
    "cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
)
SIF_SHA256 = "a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
WRAPPERS = (
    "run_reference_cell_state_cpa_stage.sh",
    "run_reference_cell_state_model_acceptance.sh",
    "run_reference_cell_state_predict_array_task.sh",
    "run_reference_cell_state_finalize.sh",
)


class ReferenceCellStatePostHumanHpcTests(unittest.TestCase):
    def test_host_delegates_and_workers_are_executable_and_shell_valid(self) -> None:
        for name in WRAPPERS:
            worker = DOCKER_HPC / name
            delegate = HOST_HPC / name
            self.assertTrue(os.access(worker, os.X_OK), worker)
            self.assertTrue(os.access(delegate, os.X_OK), delegate)
            delegate_text = delegate.read_text(encoding="utf-8")
            self.assertIn(f"cellpose_pipeline/Docker/hpc/{name}", delegate_text)
            self.assertIn('exec "$TARGET" "$@"', delegate_text)
            for script in (worker, delegate):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        with (DOCKER_HPC / "hpc_script_mapping.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            mapping = {
                row["source_path"]: row["target_path"]
                for row in csv.DictReader(handle, delimiter="\t")
            }
        for name in WRAPPERS:
            self.assertEqual(mapping.get(name), name)

    def test_every_computational_worker_locks_latest_cpu_sif(self) -> None:
        for name in WRAPPERS:
            text = (DOCKER_HPC / name).read_text(encoding="utf-8")
            self.assertIn(SIF_PATH, text)
            self.assertIn(SIF_SHA256, text)
            self.assertIn(
                '[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_HPC_CONTAINER_IMAGE" ]]',
                text,
            )
            self.assertIn("HPC_CONTAINER_GPU=0", text)
            self.assertIn("HPC_PROJECT_ROOT_BIND_MODE=ro", text)
            self.assertIn("HPC_CONTAINER_NO_MOUNT=/share", text)
            self.assertIn('HPC_CONTAINER_NO_MOUNT+x', text)
            self.assertIn('case "${HPC_CONTAINER_BINDS+x}', text)
            self.assertIn("differs from the frozen", text)
            self.assertIn("hpc_container_apptainer_runtime.sh", text)
            self.assertIn("broad_phenotype_worker_verify_container_identity", text)
            self.assertNotIn("conda activate", text)
        for name in WRAPPERS[:3]:
            text = (DOCKER_HPC / name).read_text(encoding="utf-8")
            self.assertIn(
                '[[ "$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT" ]]',
                text,
            )

    def test_workers_reject_noncanonical_sif_and_cpa_paths_before_inputs(self) -> None:
        for name in WRAPPERS:
            completed = subprocess.run(
                ["bash", str(DOCKER_HPC / name)],
                env={**os.environ, "HPC_CONTAINER_IMAGE": "/tmp/not-latest.sif"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("must equal the frozen latest SIF path", completed.stderr)
        for name in WRAPPERS[:3]:
            environment = dict(os.environ)
            environment.pop("HPC_CONTAINER_IMAGE", None)
            environment["CPA_REFERENCE_ROOT"] = "/tmp/not-the-reference"
            completed = subprocess.run(
                ["bash", str(DOCKER_HPC / name)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("must equal the frozen read-only checkout", completed.stderr)

    def test_cpa_worker_is_reference_root_confined_and_post_human_only(self) -> None:
        text = (DOCKER_HPC / WRAPPERS[0]).read_text(encoding="utf-8")
        self.assertIn("REFERENCE_SHADOW_ROOT is required", text)
        self.assertIn("DATASET_ROOT is required", text)
        self.assertIn("SOURCE_SEGMENTATION_ROOT is required", text)
        self.assertIn("require_inside_reference_shadow", text)
        self.assertIn(
            "validate|annotation-import|review-build|review-import|train",
            text,
        )
        self.assertNotIn("validate|umap|annotate|", text)
        self.assertNotIn("|predict|report", text)
        self.assertIn("20_run_cellphenotypeannotator_stage.py", text)
        self.assertIn("current_classifier_access=forbidden", text)
        self.assertIn("reference_posthuman_container_blinding=PASS", text)
        self.assertIn('dataset / "Dead"', text)
        self.assertIn('dataset / "Combined"', text)
        self.assertIn('path.name.startswith("classification_")', text)
        self.assertIn(
            "$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,"
            "$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,"
            "$NUCLEI_ROOT:$NUCLEI_ROOT:ro,"
            "$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,"
            "$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro",
            text,
        )
        self.assertNotIn("broad_phenotype_classes_v1.tsv", text)
        self.assertNotIn("broad_phenotype_features_v1.json", text)

    def test_model_acceptance_uses_reference_contract_and_parent_import(self) -> None:
        text = (DOCKER_HPC / WRAPPERS[1]).read_text(encoding="utf-8")
        self.assertIn("30_accept_reference_cell_state_model.R", text)
        self.assertIn("reference_cell_state_features_v1.json", text)
        self.assertIn("reference_cell_state_classes_v1.tsv", text)
        self.assertIn('--parent-import-manifest "$PARENT_IMPORT_MANIFEST"', text)
        self.assertIn("MODEL_ACCEPTANCE_SHA256_FILE", text)
        self.assertIn(
            'expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,'
            '$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"',
            text,
        )
        self.assertNotIn("23_validate_broad_phenotype_production_contracts.py", text)
        self.assertNotIn("LEGACY_NO_GO", text)

    def test_model_acceptance_rejects_external_output_before_creating_it(self) -> None:
        worker = DOCKER_HPC / "run_reference_cell_state_model_acceptance.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow = root / "reference"
            project_dir = shadow / "projection_input" / "representative_umap"
            model_dir = project_dir / "runs" / "model"
            model_dir.mkdir(parents=True)
            project = project_dir / "project.yml"
            parent_import = project_dir / "parent_import_manifest.json"
            train_receipt = project_dir / "train_receipt.json"
            for path in (project, parent_import, train_receipt):
                path.write_text("{}\n", encoding="utf-8")
            escaped = root / "outside" / "model_acceptance.json"
            environment = dict(os.environ)
            for name in (
                "HPC_CONTAINER_BINDS",
                "HPC_CONTAINER_FORWARD_PREFIXES",
                "HPC_CONTAINER_NO_MOUNT",
            ):
                environment.pop(name, None)
            environment.update(
                {
                    "REFERENCE_SHADOW_ROOT": str(shadow),
                    "PROJECT_FILE": str(project),
                    "PARENT_IMPORT_MANIFEST": str(parent_import),
                    "CPA_MODEL_DIR": str(model_dir),
                    "CPA_TRAIN_RECEIPT": str(train_receipt),
                    "MODEL_ACCEPTANCE_RECEIPT": str(escaped),
                    "MODEL_ACCEPTANCE_SHA256_FILE": str(escaped.with_suffix(".sha256")),
                }
            )
            completed = subprocess.run(
                ["bash", str(worker)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("must be inside REFERENCE_SHADOW_ROOT", completed.stderr)
            self.assertFalse(escaped.parent.exists())

    def test_model_acceptance_rejects_ambient_bind_and_mount_overrides(self) -> None:
        worker = DOCKER_HPC / "run_reference_cell_state_model_acceptance.sh"
        wrong_mount = subprocess.run(
            ["bash", str(worker)],
            env={**os.environ, "HPC_CONTAINER_NO_MOUNT": "/tmp"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(wrong_mount.returncode, 2)
        self.assertIn("must be exactly /share", wrong_mount.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow = root / "reference"
            project_dir = shadow / "projection_input" / "representative_umap"
            model_dir = project_dir / "runs" / "model"
            model_dir.mkdir(parents=True)
            project = project_dir / "project.yml"
            parent_import = project_dir / "parent_import_manifest.json"
            train_receipt = project_dir / "train_receipt.json"
            for path in (project, parent_import, train_receipt):
                path.write_text("{}\n", encoding="utf-8")
            output_dir = shadow / "workflow_status" / "model_acceptance"
            environment = dict(os.environ)
            environment.pop("HPC_CONTAINER_FORWARD_PREFIXES", None)
            environment.update(
                {
                    "HPC_CONTAINER_NO_MOUNT": "/share",
                    "HPC_CONTAINER_BINDS": "/:/ambient:ro",
                    "REFERENCE_SHADOW_ROOT": str(shadow),
                    "PROJECT_FILE": str(project),
                    "PARENT_IMPORT_MANIFEST": str(parent_import),
                    "CPA_MODEL_DIR": str(model_dir),
                    "CPA_TRAIN_RECEIPT": str(train_receipt),
                    "MODEL_ACCEPTANCE_RECEIPT": str(output_dir / "model_acceptance.json"),
                    "MODEL_ACCEPTANCE_SHA256_FILE": str(output_dir / "model_acceptance.sha256"),
                }
            )
            ambient_bind = subprocess.run(
                ["bash", str(worker)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(ambient_bind.returncode, 2, ambient_bind.stderr)
            self.assertIn(
                "HPC_CONTAINER_BINDS differs from the frozen model-acceptance allowlist",
                ambient_bind.stderr,
            )

    def test_prediction_worker_reads_parent_features_and_writes_reference_namespace(self) -> None:
        text = (DOCKER_HPC / WRAPPERS[2]).read_text(encoding="utf-8")
        self.assertIn("PARENT_BROAD_SHADOW_ROOT is required", text)
        self.assertIn(
            'expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,'
            '$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,'
            '$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"',
            text,
        )
        self.assertIn("31_predict_reference_cell_state_shard.R", text)
        self.assertIn(
            "${field_key}__${shard_branch}_reference_cell_state_predictions.tsv",
            text,
        )
        self.assertIn("SLURM_ARRAY_TASK_ID must be a one-based positive integer", text)
        self.assertIn("probability__multinucleated_cell", text)
        self.assertNotIn("21_predict_broad_phenotype_shard.R", text)
        self.assertNotIn("LEGACY_NO_GO", text)

    def test_finalize_publishes_exact_four_column_axis_before_optional_comparison(self) -> None:
        text = (DOCKER_HPC / WRAPPERS[3]).read_text(encoding="utf-8")
        self.assertIn("32_merge_reference_cell_state_predictions.py", text)
        self.assertIn(
            "predictions/reference_cell_state_predictions.tsv", text
        )
        self.assertIn(
            "model_id\\tcell_id\\treference_cell_state_class_id\\tprediction_status",
            text,
        )
        self.assertIn('RUN_REFERENCE_COMPARISON="${RUN_REFERENCE_COMPARISON:-0}"', text)
        self.assertIn("29_compare_current_vs_reference_cell_state.py", text)
        self.assertIn("comparison_interpretation=descriptive_cross_classification_not_accuracy", text)
        self.assertIn("legacy_no_go_enforcement=not_read", text)
        self.assertIn(
            'expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,'
            '$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro"',
            text,
        )
        self.assertNotIn("--legacy-no-go", text)
        self.assertNotIn("19_finalize_broad_phenotype_shadow.py", text)
        self.assertNotIn("22_merge_broad_phenotype_predictions.py", text)

    def test_merged_schema_constant_matches_wrapper_contract(self) -> None:
        merger = (
            REPO_ROOT
            / "cellpose_pipeline"
            / "scripts"
            / "32_merge_reference_cell_state_predictions.py"
        ).read_text(encoding="utf-8")
        ordered = (
            '"model_id",\n'
            '    "cell_id",\n'
            '    "reference_cell_state_class_id",\n'
            '    "prediction_status",'
        )
        self.assertIn(ordered, merger)
        self.assertNotIn('parser.add_argument("--legacy-no-go"', merger)


if __name__ == "__main__":
    unittest.main()
