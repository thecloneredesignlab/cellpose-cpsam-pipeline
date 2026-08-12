from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_HPC = REPO_ROOT / "cellpose_pipeline" / "hpc"
DOCKER_HPC = REPO_ROOT / "cellpose_pipeline" / "Docker" / "hpc"
SIF_PATH = "/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
SIF_SHA256 = "a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"


WRAPPERS = (
    "run_broad_phenotype_feature_array_task.sh",
    "run_broad_phenotype_input_preflight.sh",
    "run_broad_phenotype_model_acceptance.sh",
    "run_broad_phenotype_predict_array_task.sh",
    "run_broad_phenotype_cpa_adapter.sh",
    "run_broad_phenotype_cpa_stage.sh",
    "run_broad_phenotype_finalize.sh",
    "submit_broad_phenotype_shadow_full.sh",
)


class BroadPhenotypeHpcContractTests(unittest.TestCase):
    def test_host_and_container_counterparts_are_executable_and_mapped(self) -> None:
        for relative in WRAPPERS:
            host = HOST_HPC / relative
            container = DOCKER_HPC / relative
            self.assertTrue(os.access(host, os.X_OK), host)
            self.assertTrue(os.access(container, os.X_OK), container)
            host_text = host.read_text()
            self.assertIn(f"cellpose_pipeline/Docker/hpc/{relative}", host_text)
            self.assertIn('exec "$TARGET" "$@"', host_text)

        calibration_relative = "Parameter_calibration/29_run_broad_phenotype_shadow_test.sh"
        self.assertTrue(os.access(HOST_HPC / calibration_relative, os.X_OK))
        self.assertTrue(os.access(DOCKER_HPC / calibration_relative, os.X_OK))

        rows = list(
            csv.DictReader(
                (DOCKER_HPC / "hpc_script_mapping.tsv").read_text().splitlines(),
                delimiter="\t",
            )
        )
        mapped = {
            row["source_path"]: row["target_path"] for row in rows
        }
        for relative in (*WRAPPERS, calibration_relative):
            self.assertEqual(mapped.get(relative), relative)

    def test_latest_sif_and_container_execution_are_locked(self) -> None:
        computational_workers = (
            "run_broad_phenotype_feature_array_task.sh",
            "run_broad_phenotype_input_preflight.sh",
            "run_broad_phenotype_model_acceptance.sh",
            "run_broad_phenotype_predict_array_task.sh",
            "run_broad_phenotype_cpa_adapter.sh",
            "run_broad_phenotype_cpa_stage.sh",
            "run_broad_phenotype_finalize.sh",
        )
        for name in computational_workers:
            text = (DOCKER_HPC / name).read_text()
            self.assertIn(SIF_PATH, text)
            self.assertIn(SIF_SHA256, text)
            self.assertIn("broad_phenotype_container_identity.sh", text)
            self.assertIn("broad_phenotype_worker_verify_container_identity", text)
            self.assertNotIn('sha256sum "$HPC_CONTAINER_IMAGE"', text)
            self.assertNotIn("worker_sha256", text)
            self.assertIn("HPC_CONTAINER_GPU=0", text)
            self.assertIn("hpc_container_apptainer_runtime.sh", text)
            self.assertNotIn("conda activate", text)

        submitter = (DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh").read_text()
        self.assertIn("broad_phenotype_capture_container_identity", submitter)
        self.assertIn("HPC_CONTAINER_IDENTITY_FILE_SHA256", submitter)
        self.assertNotIn('sha256sum "$HPC_CONTAINER_IMAGE"', submitter)

        rscript_shim = DOCKER_HPC / "bin" / "Rscript"
        self.assertTrue(os.access(rscript_shim, os.X_OK))
        self.assertIn('hpc_apptainer_exec "Rscript" "$@"', rscript_shim.read_text())
        command_rows = list(
            csv.DictReader(
                (DOCKER_HPC / "container_command_mapping.tsv").read_text().splitlines(),
                delimiter="\t",
            )
        )
        self.assertIn({"host_command": "Rscript", "container_command": "Rscript"}, command_rows)
        runtime_rows = list(
            csv.DictReader(
                (DOCKER_HPC / "sif_runtime_commands.tsv").read_text().splitlines(),
                delimiter="\t",
            )
        )
        self.assertIn(
            {"command": "Rscript", "version_args": "--version", "required": "TRUE"},
            runtime_rows,
        )

    def test_formal_dag_has_j0_human_barrier_and_no_node_or_gpu_pin(self) -> None:
        text = (DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh").read_text()
        self.assertIn('SHADOW_ROOT="${SHADOW_ROOT:-$RESULTS_ROOT/broad_phenotype_shadow_$RUN_STAMP}"', text)
        self.assertIn('DATASET_ROOT:?DATASET_ROOT must be the explicit raw-image dataset root', text)
        self.assertIn('append_bind "$DATASET_ROOT:$DATASET_ROOT:ro"', text)
        self.assertIn('append_bind "$RESULTS_ROOT:$RESULTS_ROOT:ro"', text)
        self.assertIn('append_bind "$SHADOW_ROOT:$SHADOW_ROOT"', text)
        self.assertNotIn('append_bind "$RESULTS_ROOT:$RESULTS_ROOT"\n', text)
        self.assertIn('INCLUDE_NUCLEI_COMPARATOR="${INCLUDE_NUCLEI_COMPARATOR:-1}"', text)
        self.assertIn('preflight_job="$(submit_job input-preflight', text)
        self.assertIn('--dependency "afterok:$preflight_job"', text)
        self.assertIn('--dependency "afterok:$feature_job"', text)
        self.assertIn('--dependency "afterok:$adapter_job"', text)
        self.assertIn('--dependency "afterok:$validate_job"', text)
        self.assertIn('--dependency "afterok:$umap_job"', text)
        self.assertIn('CPA_VALIDATE_STAGE=umap', text)
        self.assertIn('BROAD_PHENOTYPE_QOS-xxlarge', text)
        self.assertIn('base_args+=(--qos "$BROAD_PHENOTYPE_QOS")', text)
        self.assertIn("export GIT_OPTIONAL_LOCKS=0", text)
        self.assertIn('status --porcelain --untracked-files=all', text)
        self.assertIn('compare_frozen feature_config_sha256', text)
        self.assertIn('compare_frozen include_nuclei_comparator', text)
        self.assertIn('compare_frozen max_umap_fields_per_well', text)
        self.assertIn('compare_frozen hpc_container_identity_file_sha256', text)
        self.assertIn('compare_frozen execution_scope', text)
        self.assertIn('compare_frozen split_mode', text)
        self.assertIn('compare_frozen heldout_wells', text)
        self.assertIn('Formal Slurm runs require the complete 27200-field universe', text)
        self.assertIn('Formal Slurm runs forbid TASK_LIST_INPUT', text)
        self.assertIn('Formal Slurm runs forbid the TEST_CPA_STAGE_MODE override', text)
        self.assertIn('--export=ALL,CPA_STAGE=validate,CPA_VALIDATE_STAGE=umap,CPA_STAGE_MODE=', text)
        self.assertIn('--export=ALL,CPA_STAGE=umap,CPA_STAGE_MODE=', text)
        self.assertIn('--export=ALL,CPA_STAGE=annotate,CPA_STAGE_MODE=', text)
        self.assertIn('Formal Slurm runs forbid HELDOUT_WELLS', text)
        self.assertIn('compare_frozen plate_map_sha256', text)
        self.assertIn('MAX_UMAP_FIELDS_PER_WELL="${MAX_UMAP_FIELDS_PER_WELL:-20}"', text)
        self.assertIn('ADAPTER_MEM="${ADAPTER_MEM:-128G}"', text)
        self.assertIn('ADAPTER_TIME="${ADAPTER_TIME:-12:00:00}"', text)
        self.assertIn('REVIEW_MEM="${REVIEW_MEM:-256G}"', text)
        self.assertIn('REVIEW_TIME="${REVIEW_TIME:-12:00:00}"', text)
        self.assertIn("validate_qos_walltime_limits", text)
        self.assertIn("Requested wall-time exceeds QOS MaxWall before any job was submitted", text)
        self.assertIn("qos_max_wall\\t$QOS_MAX_WALL", text)
        self.assertIn('compare_frozen qos_max_wall "$QOS_MAX_WALL"', text)
        self.assertIn('human_barrier=annotation_region_submission_required', text)
        self.assertIn('legacy_no_go_enforcement\\tinformational_only', text)
        self.assertIn('RESUME_STAGE" == "predict-sharded"', text)
        self.assertIn('finalize-sharded)', text)
        self.assertNotIn("--nodelist", text)
        self.assertNotIn("--node=", text)
        self.assertNotIn("--gres=", text)
        self.assertNotIn("--gpus=", text)
        for inherited_pin in (
            "SBATCH_NODELIST",
            "SBATCH_NODES",
            "SBATCH_CONSTRAINT",
            "SBATCH_EXCLUDE",
            "SBATCH_HOSTLIST",
        ):
            self.assertIn(f"-u {inherited_pin}", text)

        phase_a = text[text.index('if [[ "$RESUME_STAGE" == "phase-a" ]]', text.index("base_args=")) :]
        phase_a = phase_a[: phase_a.index("if [[ \"$RESUME_STAGE\" == \"predict-sharded\" ]]")]
        self.assertIn("CPA_STAGE=annotate", phase_a)
        self.assertNotIn("CPA_STAGE=annotation-import", phase_a)
        self.assertNotIn("CPA_STAGE=review-build", phase_a)

    def test_calibration_is_host_locked_and_confined(self) -> None:
        text = (
            DOCKER_HPC / "Parameter_calibration" / "29_run_broad_phenotype_shadow_test.sh"
        ).read_text()
        self.assertIn('required_host="hpctpa3pc0009"', text)
        self.assertIn('observed_host="$(hostname -s)"', text)
        self.assertIn('$RESULTS_ROOT/Tests_and_Parameters_calibration', text)
        self.assertIn("EXECUTION_MODE=direct_test", text)
        self.assertIn("HPC_CONTAINER_GPU=0", text)
        self.assertIn('test_direct_concurrency=${DIRECT_TEST_CONCURRENCY:-4}', text)
        self.assertIn('DATASET_ROOT:?DATASET_ROOT must be the explicit raw-image dataset root', text)
        self.assertIn('HELDOUT_WELLS="$(awk', text)
        self.assertIn('export RUN_STAMP SHADOW_ROOT TASK_LIST_INPUT EXPECTED_FIELDS HELDOUT_WELLS', text)

    def test_frozen_container_identity_rejects_metadata_drift_without_worker_sif_hash(self) -> None:
        helper = DOCKER_HPC / "util" / "broad_phenotype_container_identity.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.sif"
            image.write_bytes(b"small test-only SIF identity fixture")
            expected_sha = hashlib.sha256(image.read_bytes()).hexdigest()
            receipt = root / "workflow_status" / "hpc_container_identity.json"
            capture = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; broad_phenotype_capture_container_identity "$2" "$3" "$4"',
                    "_",
                    str(helper),
                    str(image),
                    expected_sha,
                    str(receipt),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(capture.returncode, 0, capture.stderr)
            receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["expected_sha256"], expected_sha)
            self.assertEqual(payload["container_realpath"], str(image.resolve()))
            self.assertEqual(
                set(payload["stat"]),
                {"dev", "inode", "size", "mtime_ns", "ctime_ns"},
            )
            verify_command = [
                "bash",
                "-c",
                'source "$1"; broad_phenotype_verify_container_identity "$2" "$3" "$4" "$5"',
                "_",
                str(helper),
                str(image),
                expected_sha,
                str(receipt),
                receipt_sha,
            ]
            accepted = subprocess.run(
                verify_command, text=True, capture_output=True, check=False
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            stat_before = image.stat()
            os.utime(
                image,
                ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000_000),
            )
            rejected = subprocess.run(
                verify_command, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("identity metadata mismatch", rejected.stderr)

    def test_feature_adapter_stage_and_sharded_prediction_contracts(self) -> None:
        feature = (DOCKER_HPC / "run_broad_phenotype_feature_array_task.sh").read_text()
        self.assertIn('--config "$FEATURE_CONFIG"', feature)
        self.assertIn('--field-record "$FIELD_RECORD"', feature)
        self.assertIn('INCLUDE_NUCLEI_COMPARATOR:-0', feature)

        adapter = (DOCKER_HPC / "run_broad_phenotype_cpa_adapter.sh").read_text()
        self.assertIn('--feature-config "$FEATURE_CONFIG"', adapter)
        self.assertIn('original|nucleated) cpa_branch="$BRANCH"', adapter)
        self.assertIn('--feature-manifest "$feature_manifest"', adapter)
        self.assertIn('args+=(--plate-map "$PLATE_MAP")', adapter)
        self.assertNotIn('--feature-config "$FEATURE_CONFIG"\n  --plate-map', adapter)
        self.assertIn("18_prepare_broad_phenotype_projection.py", adapter)
        self.assertIn('--max-fields-per-well "${MAX_UMAP_FIELDS_PER_WELL:-20}"', adapter)
        self.assertIn("23_validate_broad_phenotype_production_contracts.py", adapter)

        stage = (DOCKER_HPC / "run_broad_phenotype_cpa_stage.sh").read_text()
        self.assertIn("7d1e23077efb85d503ca5333df76acab1ae3640c", stage)
        self.assertIn('append_bind "$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"', stage)
        self.assertIn('--annotation-import-dir "$CPA_ANNOTATION_IMPORT_DIR"', stage)
        self.assertIn('--reviewed-labels "$CPA_REVIEWED_LABELS"', stage)
        self.assertIn('must be staged inside SHADOW_ROOT', stage)

        predict = (DOCKER_HPC / "run_broad_phenotype_predict_array_task.sh").read_text()
        self.assertIn("21_predict_broad_phenotype_shard.R", predict)
        self.assertIn("key\\tfeature_path\\treceipt_path", predict)
        self.assertIn("__${shard_branch}_broad_phenotype_predictions.tsv", predict)
        self.assertIn('Rscript "${args[@]}"', predict)
        self.assertIn("MODEL_ACCEPTANCE_RECEIPT", predict)
        self.assertIn("verify-model-acceptance", predict)

        acceptance = (DOCKER_HPC / "run_broad_phenotype_model_acceptance.sh").read_text()
        self.assertIn("25_accept_broad_phenotype_model.R", acceptance)
        self.assertIn("verify-train-receipt", acceptance)
        self.assertIn("model_acceptance.sha256", (DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh").read_text())

        finalize = (DOCKER_HPC / "run_broad_phenotype_finalize.sh").read_text()
        self.assertIn("FINALIZE_MODE:-canonical", finalize)
        self.assertIn("22_merge_broad_phenotype_predictions.py", finalize)
        self.assertIn('--feature-manifest "$FEATURE_MANIFEST"', finalize)
        self.assertIn('--prediction-root "$PREDICTION_ROOT"', finalize)
        self.assertIn('--cells "$CELLS_FILE"', finalize)
        merge_script = (
            REPO_ROOT / "cellpose_pipeline" / "scripts" / "22_merge_broad_phenotype_predictions.py"
        ).read_text()
        self.assertIn('"prediction_status"', merge_script)

    def _make_resolver_fixture(self, root: Path) -> dict[str, Path | dict[str, str]]:
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        log = root / "apptainer.log"
        sha = fake_bin / "sha256sum"
        sha.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = "{root / "locked.sif"}" ]; then printf "%s  %s\\n" "{SIF_SHA256}" "$1"; '
            'else exec shasum -a 256 "$1"; fi\n'
        )
        sha.chmod(0o755)
        apptainer = fake_bin / "apptainer"
        apptainer.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_APPTAINER_LOG\"\n"
            "[[ \"${1:-}\" == exec ]] && shift\n"
            "while (($#)); do\n"
            "  case \"$1\" in\n"
            "    --cleanenv|--nv) shift ;;\n"
            "    --home|--env|--bind|--pwd) shift 2 ;;\n"
            f"    *.sif) shift; [[ \"${{1:-}}\" == python ]] && shift && exec {sys.executable!s} \"$@\"; exec \"$@\" ;;\n"
            "    *) echo \"unexpected apptainer argument: $1\" >&2; exit 2 ;;\n"
            "  esac\n"
            "done\n"
        )
        apptainer.chmod(0o755)
        sif = root / "locked.sif"
        sif.write_bytes(b"test-only fake SIF")

        live = root / "live"
        legacy = root / "legacy"
        classification = root / "classification"
        manifest_root = root / "stage06"
        classification.mkdir()
        key = "A01_1_0d0h0m"
        path_fields = {
            "combined_raw": "Combined/raw.tif",
            "combined_mask": "Combined/original_mask.tif",
            "nucleated_combined_mask": "Combined/nucleated_mask.tif",
            "brightfield_raw": "Brightfield/raw.tif",
            "brightfield_mask": "Brightfield/original_mask.tif",
            "nucleated_brightfield_mask": "Brightfield/nucleated_mask.tif",
            "dead_raw": "Dead/raw.tif",
            "dead_mask": "Dead/mask.tif",
            "nuclei_raw": "Nuclei/raw.tif",
            "nuclei_extent_mask": "Nuclei/extent_mask.tif",
            "nuclei_core_mask": "Nuclei/core_mask.tif",
        }
        for relative in path_fields.values():
            path = live / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())

        def old(relative: str) -> str:
            return str(legacy / relative)

        record = {
            "schema_version": "postsegmentation_field_manifest_v1",
            "key": key,
            "profiles": {
                "Combined": {
                    "stem": "combined_stem_must_not_be_resolved",
                    "raw": old(path_fields["combined_raw"]),
                    "original_mask": old(path_fields["combined_mask"]),
                    "nucleated_mask": old(path_fields["nucleated_combined_mask"]),
                },
                "Brightfield": {
                    "stem": "brightfield_stem_must_not_be_resolved",
                    "raw": old(path_fields["brightfield_raw"]),
                    "original_mask": old(path_fields["brightfield_mask"]),
                    "nucleated_mask": old(path_fields["nucleated_brightfield_mask"]),
                },
                "Dead": {
                    "stem": "dead_stem_must_not_be_resolved",
                    "raw": old(path_fields["dead_raw"]),
                    "mask": old(path_fields["dead_mask"]),
                },
                "Nuclei": {
                    "stem": "nuclei_stem_must_not_be_resolved",
                    "raw": old(path_fields["nuclei_raw"]),
                    "extent_mask": old(path_fields["nuclei_extent_mask"]),
                    "core_mask": old(path_fields["nuclei_core_mask"]),
                },
            },
        }
        record_path = manifest_root / "records" / "A01" / f"{key}.json"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        manifest_path = manifest_root / "field_manifest.tsv"
        manifest_fields = ["key", "record_json", *path_fields]
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow(
                {
                    "key": key,
                    "record_json": str(record_path),
                    **{column: old(relative) for column, relative in path_fields.items()},
                }
            )
        task_list = root / "tasks.txt"
        task_list.write_text(key + "\n")
        resolved = root / "shadow" / "workflow_status" / "resolved_stage06_manifest"
        resolved.parent.mkdir(parents=True)
        identity_file = resolved.parent / "hpc_container_identity.json"
        sif_stat = sif.stat()
        identity_file.write_text(
            json.dumps(
                {
                    "schema_version": "broad_phenotype_hpc_container_identity_v1",
                    "container_realpath": str(sif.resolve()),
                    "expected_sha256": SIF_SHA256,
                    "stat": {
                        "dev": sif_stat.st_dev,
                        "inode": sif_stat.st_ino,
                        "size": sif_stat.st_size,
                        "mtime_ns": sif_stat.st_mtime_ns,
                        "ctime_ns": sif_stat.st_ctime_ns,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_APPTAINER_LOG": str(log),
            "PROJECT_DIR": str(REPO_ROOT),
            "HPC_CONTAINER_IMAGE": str(sif),
            "HPC_CONTAINER_IDENTITY_FILE": str(identity_file),
            "HPC_CONTAINER_IDENTITY_FILE_SHA256": hashlib.sha256(
                identity_file.read_bytes()
            ).hexdigest(),
            "HPC_CONTAINER_BINDS": "",
            "SOURCE_FIELD_MANIFEST_FILE": str(manifest_path),
            "TASK_LIST": str(task_list),
            "RESOLVED_FIELD_MANIFEST_DIR": str(resolved),
            "DATASET_ROOT": str(live),
            "CLASSIFICATION_ROOT": str(classification),
            "SOURCE_SEGMENTATION_ROOT": str(live),
            "STALE_DATASET_ROOT": str(root / "stale-dataset"),
            "STALE_SEGMENTATION_ROOT": str(root / "stale-segmentation"),
            "BRANCH": "original",
            "INCLUDE_NUCLEI_COMPARATOR": "1",
            "LEGACY_PATH_PREFIX": str(legacy) + "/",
            "LIVE_PATH_PREFIX": str(live) + "/",
        }
        return {
            "env": env,
            "log": log,
            "live": live,
            "legacy": legacy,
            "manifest": manifest_path,
            "record": record_path,
            "resolved": resolved,
            "path_fields": path_fields,
        }

    def test_input_preflight_resolves_only_paths_and_atomically_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_resolver_fixture(Path(temporary))
            source_record_hash = hashlib.sha256(Path(fixture["record"]).read_bytes()).hexdigest()
            completed = subprocess.run(
                [str(DOCKER_HPC / "run_broad_phenotype_input_preflight.sh")],
                env=fixture["env"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolved = Path(fixture["resolved"])
            self.assertTrue((resolved / "field_manifest.tsv").is_file())
            self.assertFalse(list(resolved.parent.glob(resolved.name + ".tmp.*")))
            payload = json.loads((resolved / "records" / "A01" / "A01_1_0d0h0m.json").read_text())
            self.assertEqual(payload["profiles"]["Combined"]["stem"], "combined_stem_must_not_be_resolved")
            self.assertEqual(payload["profiles"]["Nuclei"]["stem"], "nuclei_stem_must_not_be_resolved")
            self.assertTrue(
                payload["profiles"]["Brightfield"]["raw"].startswith(
                    str(Path(fixture["live"]).resolve())
                )
            )
            self.assertEqual(
                hashlib.sha256(Path(fixture["record"]).read_bytes()).hexdigest(),
                source_record_hash,
            )
            audit = list(
                csv.DictReader(
                    (resolved / "path_resolution_manifest.tsv").read_text().splitlines(),
                    delimiter="\t",
                )
            )
            self.assertTrue(audit)
            self.assertEqual({row["ambiguity"] for row in audit}, {"0"})
            self.assertNotIn("/stem", {row["json_pointer"] for row in audit})
            receipt = json.loads((resolved / "path_resolution_receipt.json").read_text())
            self.assertEqual(receipt["ambiguity_count"], 0)
            self.assertFalse(receipt["source_write_performed"])
            identities = list(csv.DictReader(
                (resolved / "input_identity_manifest.tsv").read_text().splitlines(),
                delimiter="\t",
            ))
            self.assertEqual(
                {row["asset_role"] for row in identities},
                {"field_record", "brightfield_raw", "combined_mask", "nuclei_mask"},
            )
            self.assertTrue(all(len(row["file_sha256"]) == 64 for row in identities))
            self.assertGreaterEqual(Path(fixture["log"]).read_text().count("exec"), 2)

    def test_input_preflight_rejects_ambiguity_and_cleans_staging_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_resolver_fixture(Path(temporary))
            ambiguous = Path(fixture["legacy"]) / fixture["path_fields"]["brightfield_raw"]
            ambiguous.parent.mkdir(parents=True, exist_ok=True)
            ambiguous.write_bytes(b"ambiguous legacy copy")
            completed = subprocess.run(
                [str(DOCKER_HPC / "run_broad_phenotype_input_preflight.sh")],
                env=fixture["env"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Path resolution ambiguity", completed.stderr)
            resolved = Path(fixture["resolved"])
            self.assertFalse(resolved.exists())
            self.assertFalse(list(resolved.parent.glob(resolved.name + ".tmp.*")))

    def test_input_preflight_exact_stale_root_mapping_allows_missing_optional_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_resolver_fixture(Path(temporary))
            env = dict(fixture["env"])
            env["STALE_DATASET_ROOT"] = str(fixture["legacy"])
            optional = Path(fixture["live"]) / fixture["path_fields"][
                "nucleated_brightfield_mask"
            ]
            optional.unlink()
            completed = subprocess.run(
                [str(DOCKER_HPC / "run_broad_phenotype_input_preflight.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolved = Path(fixture["resolved"])
            rows = list(
                csv.DictReader(
                    (resolved / "path_resolution_manifest.tsv").read_text().splitlines(),
                    delimiter="\t",
                )
            )
            self.assertIn(
                "exact_stale_dataset_root_to_declared_dataset_root",
                {row["resolution_rule"] for row in rows},
            )
            optional_rows = [
                row
                for row in rows
                if row["json_pointer"]
                in {
                    "/manifest/nucleated_brightfield_mask",
                    "/profiles/Brightfield/nucleated_mask",
                }
            ]
            self.assertTrue(optional_rows)
            self.assertEqual({row["status"] for row in optional_rows}, {"missing_optional"})
            with (resolved / "field_manifest.tsv").open(newline="") as handle:
                manifest_row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(manifest_row["nucleated_brightfield_mask"], "")
            record = json.loads(
                (resolved / "records" / "A01" / "A01_1_0d0h0m.json").read_text()
            )
            self.assertEqual(record["profiles"]["Brightfield"]["nucleated_mask"], "")

    def test_feature_input_gate_rejects_post_j0_asset_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_resolver_fixture(Path(temporary))
            completed = subprocess.run(
                [str(DOCKER_HPC / "run_broad_phenotype_input_preflight.sh")],
                env=fixture["env"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolved = Path(fixture["resolved"])
            shadow = resolved.parents[1]
            key = "A01_1_0d0h0m"
            well = "A01"
            with (resolved / "field_manifest.tsv").open(newline="") as handle:
                field = next(csv.DictReader(handle, delimiter="\t"))
            record_path = resolved / "records" / well / f"{key}.json"
            record = json.loads(record_path.read_text())
            feature = shadow / "features" / "original" / "shards" / well / (
                f"{key}__original_broad_phenotype_features.tsv"
            )
            feature.parent.mkdir(parents=True)
            feature.write_text("cell_id\noriginal|A01_1_0d0h0m|1\n")
            receipt = shadow / "features" / "original" / "receipts" / well / f"{key}__original.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "broad_phenotype_feature_receipt_v1",
                        "status": "COMPLETE",
                        "key": key,
                        "branch": "original",
                        "include_nuclei_comparator": True,
                        "feature_tsv": str(feature.resolve()),
                        "feature_tsv_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                        "field_record": str(record_path.resolve()),
                        "field_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                        "brightfield_raw": field["brightfield_raw"],
                        "bf_raw_sha256": hashlib.sha256(Path(field["brightfield_raw"]).read_bytes()).hexdigest(),
                        "combined_mask": field["combined_mask"],
                        "combined_mask_field": "original_mask",
                        "combined_mask_sha256": hashlib.sha256(Path(field["combined_mask"]).read_bytes()).hexdigest(),
                        "nuclei_mask": record["profiles"]["Nuclei"]["core_mask"],
                        "nuclei_mask_field": "core_mask",
                        "nuclei_mask_sha256": hashlib.sha256(Path(record["profiles"]["Nuclei"]["core_mask"]).read_bytes()).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            feature_manifest = shadow / "workflow_status" / "feature_inventory" / "original_feature_manifest.tsv"
            feature_manifest.parent.mkdir(parents=True)
            with feature_manifest.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["key", "feature_path", "receipt_path"])
                writer.writerow([key, feature, receipt])
            validator = REPO_ROOT / "cellpose_pipeline" / "scripts" / "23_validate_broad_phenotype_production_contracts.py"
            gate_args = [
                str(validator),
                "feature-inputs",
                "--field-manifest",
                str(resolved / "field_manifest.tsv"),
                "--input-identity-manifest",
                str(resolved / "input_identity_manifest.tsv"),
                "--feature-manifest",
                str(feature_manifest),
                "--shadow-root",
                str(shadow),
                "--branch",
                "original",
                "--include-nuclei-comparator",
                "1",
                "--output-receipt",
                str(shadow / "workflow_status" / "feature_input_alignment" / "original.json"),
                "--output-audit",
                str(shadow / "workflow_status" / "feature_input_alignment" / "original.tsv"),
            ]
            accepted = subprocess.run(
                [sys.executable, *gate_args], text=True, capture_output=True, check=False
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            Path(field["brightfield_raw"]).write_bytes(b"tampered after J0")
            rejected = subprocess.run(
                [sys.executable, *gate_args], text=True, capture_output=True, check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("changed after J0", rejected.stderr)

    def test_formal_phase_a_dry_run_materializes_expected_dag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            dataset = root / "raw"
            classification = root / "classification"
            stage06 = classification / "workflow_status" / "postsegmentation_manifest"
            for directory in (results, dataset, classification):
                directory.mkdir()
            stage06.mkdir(parents=True)
            (stage06 / "field_manifest.tsv").write_text(
                "key\n"
                + "".join(
                    f"A01_{field_index}_0d0h0m\n"
                    for field_index in range(1, 27201)
                ),
                encoding="utf-8",
            )
            legacy_no_go = root / "legacy.json"
            legacy_no_go.write_text('{"overall_decision":"NO_GO"}\n', encoding="utf-8")
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            fake_sif = root / "locked.sif"
            fake_sif.write_bytes(b"dry-run SIF placeholder")
            fake_sha = fake_bin / "sha256sum"
            fake_sha.write_text(
                "#!/bin/sh\n"
                f'if [ "$1" = "{fake_sif}" ]; then printf "%s  %s\\n" "{SIF_SHA256}" "$1"; '
                'else exec shasum -a 256 "$1"; fi\n'
            )
            fake_sha.chmod(0o755)
            fake_apptainer = fake_bin / "apptainer"
            fake_apptainer.write_text("#!/bin/sh\nexit 0\n")
            fake_apptainer.chmod(0o755)
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            fake_git = fake_bin / "git"
            fake_git.write_text(
                "#!/bin/sh\n"
                f'if [ "$1" = "-C" ] && [ "$2" = "{REPO_ROOT}" ] && [ "$3" = "status" ]; then exit 0; fi\n'
                f'exec "{real_git}" "$@"\n'
            )
            fake_git.chmod(0o755)
            reference_root = Path(
                "/Users/4482173/Documents/GitHub/cell-phenotype-annotator"
            )
            self.assertTrue(reference_root.is_dir())
            run_stamp = "20990101_000000"
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "PROJECT_DIR": str(REPO_ROOT),
                "HPC_CONTAINER_IMAGE": str(fake_sif),
                "HPC_CONTAINER_BINDS": "",
                "CPA_REFERENCE_ROOT": str(reference_root),
                "RESULTS_ROOT": str(results),
                "DATASET_ROOT": str(dataset),
                "FIELD_MANIFEST_DIR": str(stage06),
                "CLASSIFICATION_ROOT": str(classification),
                "LEGACY_NO_GO": str(legacy_no_go),
                "EXPECTED_FIELDS": "27200",
                "RUN_STAMP": run_stamp,
                "DRY_RUN_SUBMIT": "1",
                "EXECUTION_MODE": "slurm",
                "RESUME_STAGE": "phase-a",
            }
            for ambient_name in (
                "TASK_LIST_INPUT",
                "HELDOUT_WELLS",
                "CPA_STAGE_MODE",
                "TEST_CPA_STAGE_MODE",
            ):
                env.pop(ambient_name, None)
            completed = subprocess.run(
                [str(DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("input_preflight_job_id=DRYRUN_input-preflight", completed.stdout)
            self.assertIn("feature_job_id=DRYRUN_feature", completed.stdout)
            self.assertIn("annotate_job_id=DRYRUN_annotate", completed.stdout)
            self.assertIn(
                "human_barrier=annotation_region_submission_required", completed.stdout
            )
            shadow = results / f"broad_phenotype_shadow_{run_stamp}"
            dag = (shadow / "workflow_status" / "slurm_dag.tsv").read_text()
            self.assertIn("input_preflight\tDRYRUN_input-preflight", dag)
            self.assertIn("feature\tDRYRUN_feature\tDRYRUN_input-preflight", dag)
            self.assertIn("human_annotation_barrier\tnot_submitted\tannotate", dag)
            preflight_path = shadow / "workflow_status" / "submission_preflight.tsv"
            preflight_rows = list(
                csv.DictReader(preflight_path.read_text().splitlines(), delimiter="\t")
            )
            preflight = {row["property"]: row["value"] for row in preflight_rows}
            self.assertEqual(preflight["path_resolution_ambiguity_count"], "pending_j0")
            self.assertEqual(preflight["legacy_no_go_enforcement"], "informational_only")
            self.assertEqual(preflight["project_git_status"], "clean")
            self.assertEqual(preflight["include_nuclei_comparator"], "1")
            self.assertEqual(preflight["execution_scope"], "formal_full")
            self.assertEqual(preflight["split_mode"], "plate_map_preregistered")
            self.assertEqual(preflight["heldout_wells"], "plate_map_preregistered")
            self.assertEqual(preflight["cpa_stage_mode"], "")
            identity_path = Path(preflight["hpc_container_identity_file"])
            self.assertEqual(identity_path, shadow / "workflow_status" / "hpc_container_identity.json")
            self.assertEqual(
                preflight["hpc_container_identity_file_sha256"],
                hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(preflight["canonical_task_list_sha256"], hashlib.sha256(
                (shadow / "workflow_status" / "field_tasks.txt").read_bytes()
            ).hexdigest())

            calibration_parent = results / "Tests_and_Parameters_calibration"
            calibration_parent.mkdir()
            calibration_tasks = root / "calibration_tasks.txt"
            calibration_tasks.write_text(
                "A01_1_0d0h0m\nA02_1_0d0h0m\n", encoding="utf-8"
            )
            failed_shadow = calibration_parent / "broad_phenotype_shadow_test_invalid"
            invalid_direct_env = {
                **env,
                "RUN_STAMP": "20990101_000003",
                "SHADOW_ROOT": str(failed_shadow),
                "EXECUTION_MODE": "direct_test",
                "TASK_LIST_INPUT": str(calibration_tasks),
                "EXPECTED_FIELDS": "2",
                "HELDOUT_WELLS": "A02",
                "DIRECT_TEST_CONCURRENCY": "0",
            }
            invalid_direct = subprocess.run(
                [str(DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh")],
                env=invalid_direct_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(invalid_direct.returncode, 0)
            self.assertIn("DIRECT_TEST_CONCURRENCY", invalid_direct.stderr)
            self.assertFalse(
                failed_shadow.exists(),
                "host-only bootstrap failure must remove only the owned shadow root",
            )

            resolved_stage06 = shadow / "workflow_status" / "resolved_stage06_manifest"
            resolved_stage06.mkdir(parents=True)
            (resolved_stage06 / "field_manifest.tsv").write_text(
                "key\nA01_1_0d0h0m\n", encoding="utf-8"
            )
            (resolved_stage06 / "path_resolution_manifest.tsv").write_text(
                "key\tambiguity\nA01_1_0d0h0m\t0\n", encoding="utf-8"
            )
            canonical_task_list = shadow / "workflow_status" / "field_tasks.txt"
            canonical_before = canonical_task_list.read_bytes()
            retry_list = root / "retry.txt"
            retry_list.write_text("A01_1_0d0h0m\n", encoding="utf-8")
            retry_env = {
                **env,
                "RUN_STAMP": "20990101_000001",
                "SHADOW_ROOT": str(shadow),
                "RESUME_STAGE": "feature-retry",
                "RETRY_TASK_LIST": str(retry_list),
            }
            retry = subprocess.run(
                [str(DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh")],
                env=retry_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertIn("resume_job_id=DRYRUN_feature-retry", retry.stdout)
            self.assertEqual(canonical_task_list.read_bytes(), canonical_before)
            retry_receipts = list(
                (shadow / "workflow_status").glob(
                    "resume_submission_feature-retry_20990101_000001_*.tsv"
                )
            )
            self.assertEqual(len(retry_receipts), 1)
            self.assertIn(str(retry_list), retry_receipts[0].read_text())

            drift_config = root / "drift_feature_config.json"
            drift_config.write_text('{"drift":true}\n', encoding="utf-8")
            drift_env = {
                **env,
                "RUN_STAMP": "20990101_000002",
                "SHADOW_ROOT": str(shadow),
                "RESUME_STAGE": "adapter",
                "FEATURE_CONFIG": str(drift_config),
            }
            drift = subprocess.run(
                [str(DOCKER_HPC / "submit_broad_phenotype_shadow_full.sh")],
                env=drift_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(drift.returncode, 0)
            self.assertIn("Resume provenance drift for feature_config_sha256", drift.stderr)


if __name__ == "__main__":
    unittest.main()
