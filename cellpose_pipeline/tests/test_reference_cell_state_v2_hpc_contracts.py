from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER_HPC = ROOT / "cellpose_pipeline" / "Docker" / "hpc"
HOST_HPC = ROOT / "cellpose_pipeline" / "hpc"
CONTAINER = ROOT / "container"
V2_NAMES = (
    "submit_reference_cell_state_shadow_v2.sh",
    "run_reference_cell_state_phase_a_v2.sh",
    "run_reference_cell_state_cpa_stage_v2.sh",
    "run_reference_cell_state_model_acceptance_v2.sh",
    "run_reference_cell_state_predict_array_task_v2.sh",
    "run_reference_cell_state_finalize_v2.sh",
    "run_reference_cell_state_compare_v2.sh",
)


class ReferenceCellStateV2HpcContracts(unittest.TestCase):
    def _make_single_job_submitter_fixture(self, temporary: str) -> dict[str, Path]:
        # macOS exposes TemporaryDirectory paths through both /var and
        # /private/var; freeze the canonical spelling because the submitter
        # deliberately compares canonical roots byte-for-byte.
        fixture = Path(temporary).resolve()
        project = fixture / "project"
        docker_hpc = project / "cellpose_pipeline" / "Docker" / "hpc"
        util = docker_hpc / "util"
        configs = project / "cellpose_pipeline" / "configs"
        util.mkdir(parents=True)
        configs.mkdir(parents=True)

        for relative in (
            "submit_reference_cell_state_shadow_v2.sh",
            "reference_cell_state_v2_sif_identity.tsv",
            "util/reference_cell_state_v2_contract.sh",
            "util/broad_phenotype_container_identity.sh",
        ):
            source = DOCKER_HPC / relative
            destination = docker_hpc / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name in (
            "reference_cell_state_features_v2.json",
            "reference_cell_state_classes_v2.tsv",
        ):
            shutil.copy2(ROOT / "cellpose_pipeline" / "configs" / name, configs / name)

        cpa_root = fixture / "cpa-reference"
        cpa_root.mkdir()
        (cpa_root / "README").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=cpa_root, check=True)
        subprocess.run(["git", "add", "README"], cwd=cpa_root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=V2 contract test",
                "-c",
                "user.email=v2-contract-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            cwd=cpa_root,
            check=True,
        )
        cpa_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cpa_root, text=True
        ).strip()
        cpa_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=cpa_root, text=True
        ).strip()

        experiment = fixture / "experiment"
        results = experiment / "results"
        dataset = experiment / "20260626_SUM159_AC_Exp1_SeparateImages"
        segmentation = results / "full_fusion_shape_strict_20260711_155940"
        parent = results / "broad_phenotype_shadow_20260812_075437"
        parent_project = parent / "projection_input" / "representative_umap"
        parent_run = parent_project / "runs" / "fixture"
        feature_inventory = parent / "workflow_status" / "feature_inventory"
        parent_cpa = parent / "cpa"
        for directory in (
            results,
            dataset,
            segmentation,
            parent_run,
            feature_inventory,
            parent_cpa,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (parent_project / "project.yml").write_text("name: fixture\n", encoding="utf-8")
        (parent / "projection_input" / "projection_input_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (parent_run / "umap_manifest.json").write_text("{}\n", encoding="utf-8")
        (feature_inventory / "original_feature_manifest.tsv").write_text(
            "well\tfield\nA01\t1\n", encoding="utf-8"
        )
        (parent_cpa / "cells.tsv").write_text("cell_id\nfixture\n", encoding="utf-8")

        sif = fixture / "reference-v2.sif"
        sif.write_bytes(b"reference-v2-submit-recovery-fixture\n")
        sif_sha = hashlib.sha256(sif.read_bytes()).hexdigest()
        sif_bytes = sif.stat().st_size

        contract_path = util / "reference_cell_state_v2_contract.sh"
        contract_text = contract_path.read_text(encoding="utf-8")
        replacements = {
            "/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/"
            "SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/"
            "20260619_SUM159_Doxorubicin_Cyclophosphamide": str(experiment),
            "/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/"
            "7d1e23077efb85d503ca5333df76acab1ae3640c": str(cpa_root),
            "7d1e23077efb85d503ca5333df76acab1ae3640c": cpa_commit,
            "7cd003ae4be29a5851ba305d92249cd9f8ff888e": cpa_tree,
            "/share/lab_crd/taoli/Docker/"
            "cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models-reference-v2-parity.sif": str(
                sif
            ),
        }
        for old, new in replacements.items():
            self.assertIn(old, contract_text)
            contract_text = contract_text.replace(old, new)
        contract_path.write_text(contract_text, encoding="utf-8")
        submitter_path = docker_hpc / "submit_reference_cell_state_shadow_v2.sh"
        submitter_text = submitter_path.read_text(encoding="utf-8")
        frozen_experiment = (
            "/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/"
            "SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/"
            "20260619_SUM159_Doxorubicin_Cyclophosphamide"
        )
        self.assertIn(frozen_experiment, submitter_text)
        submitter_path.write_text(
            submitter_text.replace(frozen_experiment, str(experiment)),
            encoding="utf-8",
        )

        identity_path = docker_hpc / "reference_cell_state_v2_sif_identity.tsv"
        identity_rows = []
        for line in identity_path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                identity_rows.append(line)
                continue
            key, value = line.split("\t", 1)
            value = {
                "status": "VERIFIED",
                "image_path": str(sif),
                "image_sha256": sif_sha,
                "image_bytes": str(sif_bytes),
                "runtime_rootfs_read_only": "verified",
            }.get(key, value)
            identity_rows.append(f"{key}\t{value}")
        identity_path.write_text("\n".join(identity_rows) + "\n", encoding="utf-8")

        subprocess.run(["git", "init", "-q"], cwd=project, check=True)
        subprocess.run(["git", "add", "."], cwd=project, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=V2 contract test",
                "-c",
                "user.email=v2-contract-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            cwd=project,
            check=True,
        )

        stub_bin = fixture / "stub-bin"
        stub_bin.mkdir()
        sbatch_count = fixture / "sbatch-count.txt"
        sbatch_count.write_text("0\n", encoding="utf-8")
        sbatch_pause = fixture / "sbatch-pause"
        sbatch_entered = fixture / "sbatch-entered"
        sacct_states = fixture / "sacct-states.tsv"
        sacct_states.write_text("", encoding="utf-8")
        fail_summary_once = fixture / "fail-summary-once"
        system_mv = shutil.which("mv")
        system_sha256sum = shutil.which("sha256sum")
        self.assertIsNotNone(system_mv)
        self.assertIsNotNone(system_sha256sum)
        (stub_bin / "sbatch").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f'count_file="{sbatch_count}"\n'
            f'pause_file="{sbatch_pause}"\n'
            f'entered_file="{sbatch_entered}"\n'
            "if [[ -f \"$pause_file\" ]]; then : > \"$entered_file\"; while [[ -f \"$pause_file\" ]]; do sleep 0.05; done; fi\n"
            "count=0\n"
            "if [[ -f \"$count_file\" ]]; then read -r count < \"$count_file\"; fi\n"
            "count=$((count + 1))\n"
            "printf '%s\\n' \"$count\" > \"$count_file\"\n"
            "printf '%s\\n' \"$((41000 + count))\"\n",
            encoding="utf-8",
        )
        (stub_bin / "sacct").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f'states_file="{sacct_states}"\n'
            "job_id=\n"
            "while (($#)); do\n"
            "  if [[ \"$1\" == --jobs ]]; then shift; job_id=\"$1\"; fi\n"
            "  shift\n"
            "done\n"
            "[[ \"$job_id\" =~ ^[0-9]+$ ]] || exit 2\n"
            "awk -F '\\t' -v id=\"$job_id\" '$1==id || index($1,id \"_\")==1{print $1 \"|\" $1 \"|\" $2}' \"$states_file\"\n",
            encoding="utf-8",
        )
        system_flock = shutil.which("flock")
        if system_flock is None:
            # macOS lacks the flock CLI; fcntl.flock on inherited fd 9 has the
            # same kernel-released open-file-description lifetime required by
            # the Linux submitter contract.
            (stub_bin / "flock").write_text(
                "#!/bin/bash\nset -euo pipefail\n"
                "[[ \"${1:-}\" == -x && \"${2:-}\" == 9 ]]\n"
                "/usr/bin/python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'\n",
                encoding="utf-8",
            )
        else:
            (stub_bin / "flock").symlink_to(system_flock)
        (stub_bin / "mv").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            'destination="${!#}"\n'
            f'fail_marker="{fail_summary_once}"\n'
            "case \"$destination\" in\n"
            "  */workflow_status/submission_*_v2.tsv)\n"
            "    case \"$destination\" in\n"
            "      *_jobs_v2.tsv|*_attempts_v2.tsv) ;;\n"
            "      *)\n"
            "        if [[ -f \"$fail_marker\" ]]; then\n"
            "          /bin/rm -f -- \"$fail_marker\"\n"
            "          exit 86\n"
            "        fi\n"
            "        ;;\n"
            "    esac\n"
            "    ;;\n"
            "esac\n"
            f'exec "{system_mv}" "$@"\n',
            encoding="utf-8",
        )
        (stub_bin / "stat").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "if [[ \"${1:-}\" == -c && \"${2:-}\" == %s && $# -eq 3 ]]; then\n"
            "  exec /usr/bin/python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' \"$3\"\n"
            "fi\n"
            "exec /usr/bin/stat \"$@\"\n",
            encoding="utf-8",
        )
        (stub_bin / "sha256sum").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f'exec "{system_sha256sum}" "$@"\n',
            encoding="utf-8",
        )
        for stub in stub_bin.iterdir():
            stub.chmod(0o755)

        shadow_root = results / "reference_cell_state_shadow_v2_fault_recovery"
        return {
            "project": project,
            "submitter": docker_hpc / "submit_reference_cell_state_shadow_v2.sh",
            "experiment": experiment,
            "results": results,
            "dataset": dataset,
            "segmentation": segmentation,
            "parent": parent,
            "cpa_root": cpa_root,
            "sif": sif,
            "stub_bin": stub_bin,
            "sbatch_count": sbatch_count,
            "sbatch_pause": sbatch_pause,
            "sbatch_entered": sbatch_entered,
            "sacct_states": sacct_states,
            "fail_summary_once": fail_summary_once,
            "shadow_root": shadow_root,
        }

    def _run_single_job_submitter(
        self,
        fixture: dict[str, Path],
        stage: str,
        *,
        dependency: str = "",
        region_submission: Path | None = None,
        shadow_root: Path | str | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": f'{fixture["stub_bin"]}:{os.environ["PATH"]}',
            "HOME": str(fixture["project"].parent),
            "PROJECT_DIR": str(fixture["project"]),
            "EXECUTION_MODE": "slurm",
            "V2_STAGE": stage,
            "DRY_RUN_SUBMIT": "0",
            "RUN_STAMP": "fault_recovery",
            "EXPERIMENT_ROOT": str(fixture["experiment"]),
            "RESULTS_ROOT": str(fixture["results"]),
            "PARENT_BROAD_SHADOW_ROOT": str(fixture["parent"]),
            "DATASET_ROOT": str(fixture["dataset"]),
            "SOURCE_SEGMENTATION_ROOT": str(fixture["segmentation"]),
            "HPC_CONTAINER_IMAGE": str(fixture["sif"]),
            "CPA_REFERENCE_ROOT": str(fixture["cpa_root"]),
            "EXPECTED_PARENT_CELL_COUNT": "32000",
            "DEPENDENCY_JOB_ID": dependency,
            "REFERENCE_QOS": "xxlarge",
            "REFERENCE_SHADOW_ROOT": str(shadow_root or fixture["shadow_root"]),
        }
        if region_submission is not None:
            environment["REGION_SUBMISSION"] = str(region_submission)
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            ["bash", str(fixture["submitter"])],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

    def _write_stable_phase_a_receipt(self, fixture: dict[str, Path]) -> Path:
        root = fixture["shadow_root"]
        artifacts: dict[str, Path] = {}

        def create(key: str, relative: str, content: str) -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            artifacts[key] = path
            return path

        project = create("project_file", "projection_input/representative_umap_v2/project.yml", "project_id: fixture\n")
        historical = create(
            "historical_projection_manifest",
            "projection_input/representative_umap_v2/historical_projection/historical_projection_manifest.json",
            '{"diagnostic_cluster":{"one_cluster_fallback":false},'
            '"schema_version":"reference_cell_state_historical_projection_v2",'
            '"status":"COMPLETE"}\n',
        )
        runtime = create("runtime_package_receipt", "workflow_status/reference_v2_runtime_packages.tsv", "package\tversion\nR\t4.2.3\n")
        annotation = create(
            "annotation_manifest",
            "projection_input/representative_umap_v2/runs/fixture/annotation_manifest.json",
            "{}\n",
        )
        workspace = create("morphology_workspace", "morphology_reference/annotation_workspace.html", "<html>fixture</html>\n")
        plate = fixture["project"] / "cellpose_pipeline" / "configs" / "reference_cell_state_classes_v2.tsv"
        dependency = fixture["project"] / "cellpose_pipeline" / "configs" / "reference_cell_state_features_v2.json"
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        overlay_image = root / "morphology_reference" / "umap_morphology_overlay.png"
        overlay_image.write_bytes(b"fixture-overlay")
        overlay = create(
            "morphology_overlay_manifest",
            "morphology_reference/overlay_manifest.json",
            "{\n"
            '  "schema_version": "reference_morphology_workspace_v2",\n'
            '  "status": "COMPLETE",\n'
            '  "output_artifact_sha256": {\n'
            f'    "annotation_workspace.html": "{sha(workspace)}",\n'
            f'    "umap_morphology_overlay.png": "{sha(overlay_image)}"\n'
            "  }\n"
            "}\n",
        )
        rows = {
            "schema_version": "reference_cell_state_phase_a_receipt_v2",
            "status": "COMPLETE",
            "reference_shadow_root": str(root),
            "project_file": str(project),
            "project_sha256": sha(project),
            "plate_map": str(plate),
            "plate_map_sha256": sha(plate),
            "historical_projection_manifest": str(historical),
            "historical_projection_manifest_sha256": sha(historical),
            "runtime_package_receipt": str(runtime),
            "runtime_package_receipt_sha256": sha(runtime),
            "annotation_dir": str(annotation.parent),
            "annotation_manifest": str(annotation),
            "annotation_manifest_sha256": sha(annotation),
            "morphology_overlay_manifest": str(overlay),
            "morphology_overlay_manifest_sha256": sha(overlay),
            "morphology_workspace": str(workspace),
            "morphology_workspace_sha256": sha(workspace),
            "dependency_lock": str(dependency),
            "dependency_lock_sha256": sha(dependency),
            "hpc_container_image": str(fixture["sif"]),
            "hpc_container_sha256": sha(fixture["sif"]),
            "one_cluster_fallback": "false",
            "method_parity_status": "historical_core_parity_with_pinned_stable_polygon_sampling",
            "review_sampling_contract": "pinned_reference_stable_polygon_seed1_seed2_sampling",
            "human_barrier": "region_submission_required",
            "human_workspace": str(workspace),
            "human_submission_expected_path": str(annotation.parent / "region_submission.json"),
        }
        receipt = root / "workflow_status" / "reference_cell_state_phase_a_v2" / "PHASE_A_COMPLETE.tsv"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            "property\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in rows.items()),
            encoding="utf-8",
        )
        return receipt

    @staticmethod
    def _set_sacct_states(fixture: dict[str, Path], states: dict[str, str]) -> None:
        fixture["sacct_states"].write_text(
            "".join(f"{job_id}\t{state}\n" for job_id, state in states.items()),
            encoding="utf-8",
        )

    def test_phase_a_single_job_recovers_after_ledger_before_summary_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            fixture["fail_summary_once"].write_text("fail once\n", encoding="utf-8")

            interrupted = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(interrupted.returncode, 86, interrupted.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")
            summary = (
                fixture["shadow_root"]
                / "workflow_status"
                / "submission_phase-a_v2.tsv"
            )
            ledger = (
                fixture["shadow_root"]
                / "workflow_status"
                / "submission_phase-a_jobs_v2.tsv"
            )
            self.assertFalse(summary.exists())
            with ledger.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["job_name"], "reference_cell_state_v2_phase_a")
            self.assertEqual(rows[0]["job_id"], "41001")
            self.assertEqual(rows[0]["dependency_job_id"], "none")

            fixture["sacct_states"].write_text("41001\tPENDING\n", encoding="utf-8")
            resumed = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")
            self.assertTrue(summary.is_file())
            summary_text = summary.read_text(encoding="utf-8")
            self.assertIn("stage\tphase-a\n", summary_text)
            self.assertIn("job_id\t41001\n", summary_text)

    def test_posthuman_single_job_recovers_after_ledger_before_summary_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            phase_a = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(phase_a.returncode, 0, phase_a.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")

            self._write_stable_phase_a_receipt(fixture)
            region_submission = fixture["shadow_root"] / "human_review" / "regions.json"
            region_submission.parent.mkdir(parents=True)
            region_submission.write_text("{}\n", encoding="utf-8")
            fixture["fail_summary_once"].write_text("fail once\n", encoding="utf-8")

            interrupted = self._run_single_job_submitter(
                fixture,
                "post-region",
                dependency="41001",
                region_submission=region_submission,
            )
            self.assertEqual(interrupted.returncode, 86, interrupted.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "2")
            summary = (
                fixture["shadow_root"]
                / "workflow_status"
                / "submission_post-region_v2.tsv"
            )
            ledger = (
                fixture["shadow_root"]
                / "workflow_status"
                / "submission_post-region_jobs_v2.tsv"
            )
            self.assertFalse(summary.exists())
            with ledger.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["job_name"], "reference_cell_state_v2_post-region"
            )
            self.assertEqual(rows[0]["job_id"], "41002")
            self.assertEqual(rows[0]["dependency_job_id"], "41001")

            fixture["sacct_states"].write_text(
                "41001\tPENDING\n41002\tRUNNING\n", encoding="utf-8"
            )
            resumed = self._run_single_job_submitter(
                fixture,
                "post-region",
                dependency="41001",
                region_submission=region_submission,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "2")
            self.assertTrue(summary.is_file())
            summary_text = summary.read_text(encoding="utf-8")
            self.assertIn("stage\tpost-region\n", summary_text)
            self.assertIn("job_id\t41002\n", summary_text)

    def test_slurm_failed_job_gets_append_only_attempts_and_completed_missing_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            initial = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(initial.returncode, 0, initial.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")

            self._set_sacct_states(fixture, {"41001": "PENDING"})
            pending = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(pending.returncode, 0, pending.stderr)
            self.assertIn("slurm_attempt_active=reference_cell_state_v2_phase_a:41001:PENDING", pending.stdout)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")

            self._set_sacct_states(fixture, {"41001": "FAILED"})
            retried = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(retried.returncode, 0, retried.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "2")
            attempts = fixture["shadow_root"] / "workflow_status" / "submission_phase-a_attempts_v2.tsv"
            with attempts.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["attempt"] for row in rows], ["1", "2"])
            self.assertEqual(rows[1]["job_id"], "41002")
            self.assertEqual(rows[1]["trigger_state"], "FAILED")
            self.assertEqual(rows[1]["prior_job_id"], "41001")

            self._set_sacct_states(fixture, {"41002": "RUNNING"})
            active_retry = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(active_retry.returncode, 0, active_retry.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "2")

            self._set_sacct_states(fixture, {"41002": "TIMEOUT"})
            retry_three = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(retry_three.returncode, 0, retry_three.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "3")
            with attempts.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["attempt"] for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[2]["prior_job_id"], "41002")
            self.assertEqual(rows[2]["trigger_state"], "TIMEOUT")

            self._set_sacct_states(fixture, {"41003": "COMPLETED"})
            missing = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(missing.returncode, 2)
            self.assertIn("COMPLETED but the V2 product is missing", missing.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "3")

    def test_summary_gap_then_failed_job_retries_and_reenters_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            fixture["fail_summary_once"].write_text("fail once\n", encoding="utf-8")
            interrupted = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(interrupted.returncode, 86, interrupted.stderr)

            self._set_sacct_states(fixture, {"41001": "FAILED"})
            retry = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(retry.returncode, 0, retry.stderr)
            summary = fixture["shadow_root"] / "workflow_status" / "submission_phase-a_v2.tsv"
            self.assertIn("job_id\t41002\n", summary.read_text(encoding="utf-8"))

            self._set_sacct_states(fixture, {"41002": "PENDING"})
            reentered = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(reentered.returncode, 0, reentered.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "2")

            self._set_sacct_states(fixture, {"41002": "FAILED"})
            third = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "3")
            self.assertIn("job_id\t41002\n", summary.read_text(encoding="utf-8"))

    def test_predict_dependency_drift_waits_then_retries_against_latest_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            manifest = fixture["parent"] / "workflow_status" / "feature_inventory" / "original_feature_manifest.tsv"
            manifest.write_text(
                "well\tfield\n" + "".join(f"A01\t{index}\n" for index in range(1, 27201)),
                encoding="utf-8",
            )
            phase = self._run_single_job_submitter(fixture, "phase-a")
            self.assertEqual(phase.returncode, 0, phase.stderr)
            self._write_stable_phase_a_receipt(fixture)

            submitted = self._run_single_job_submitter(fixture, "predict", dependency="41001")
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "4")

            self._set_sacct_states(
                fixture,
                {"41002": "FAILED", "41003_[1-27200]": "PENDING", "41004": "PENDING"},
            )
            blocked = self._run_single_job_submitter(fixture, "predict", dependency="41001")
            self.assertEqual(blocked.returncode, 2)
            self.assertIn("dependency/array differs", blocked.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "5")

            self._set_sacct_states(
                fixture,
                {"41005": "PENDING", "41003_[1-27200]": "CANCELLED", "41004": "CANCELLED"},
            )
            recovered = self._run_single_job_submitter(fixture, "predict", dependency="41001")
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "7")
            attempts = fixture["shadow_root"] / "workflow_status" / "submission_predict_attempts_v2.tsv"
            with attempts.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            predict_rows = [row for row in rows if row["logical_node"] == "reference_cell_state_v2_predict"]
            finalize_rows = [row for row in rows if row["logical_node"] == "reference_cell_state_v2_finalize"]
            self.assertEqual(predict_rows[-1]["job_id"], "41006")
            self.assertEqual(predict_rows[-1]["dependency_job_id"], "41005")
            self.assertEqual(finalize_rows[-1]["job_id"], "41007")
            self.assertEqual(finalize_rows[-1]["dependency_job_id"], "41006")

    def test_public_submitter_rejects_ambient_frozen_reexec_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            rejected = self._run_single_job_submitter(
                fixture,
                "phase-a",
                extra_environment={"REFERENCE_CELL_STATE_V2_FROZEN_REEXEC": "1"},
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("Ambient/incomplete frozen-reexec assertion is forbidden", rejected.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "0")
            self.assertFalse(fixture["shadow_root"].exists())

    def test_forged_external_token_cannot_authorize_live_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            token = Path(temporary).resolve() / "forged-token.tsv"
            token.write_text("property\tvalue\n", encoding="utf-8")
            rejected = self._run_single_job_submitter(
                fixture,
                "phase-a",
                extra_environment={
                    "REFERENCE_CELL_STATE_V2_FROZEN_REEXEC": "1",
                    "REFERENCE_CELL_STATE_V2_FROZEN_TOKEN": "a" * 64,
                    "REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE": str(token),
                },
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("not bound to this extracted submitter", rejected.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "0")

    def test_existing_root_bootstrap_rejects_namespace_and_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            results = fixture["results"]
            calibration = results / "Tests_and_Parameters_calibration"
            calibration.mkdir()
            formal_test_root = calibration / "reference_cell_state_shadow_v2_test_wrong_mode"
            nested_root = results / "nested" / "reference_cell_state_shadow_v2_nested"
            outside_root = Path(temporary).resolve() / "reference_cell_state_shadow_v2_outside"
            for root in (formal_test_root, nested_root, outside_root):
                root.mkdir(parents=True)
            cases = (
                (formal_test_root, "Formal V2 continuation root escaped"),
                (nested_root, "Formal V2 continuation root escaped"),
                (outside_root, "Formal V2 continuation root escaped"),
            )
            for root, message in cases:
                with self.subTest(root=root):
                    rejected = self._run_single_job_submitter(
                        fixture, "phase-a", shadow_root=root
                    )
                    self.assertEqual(rejected.returncode, 2)
                    self.assertIn(message, rejected.stderr)
                    self.assertNotIn("resume input is unavailable", rejected.stderr)

            escaped_target = fixture["experiment"] / "outside" / "reference_cell_state_shadow_v2_escape"
            escaped_target.mkdir(parents=True)
            lexical_escape = results / "nested" / ".." / ".." / "outside" / escaped_target.name
            escaped = self._run_single_job_submitter(
                fixture, "phase-a", shadow_root=str(lexical_escape)
            )
            self.assertEqual(escaped.returncode, 2)
            self.assertIn("Formal V2 continuation root escaped", escaped.stderr)

            valid = results / "reference_cell_state_shadow_v2_symlink_workflow"
            valid.mkdir()
            external_workflow = Path(temporary).resolve() / "external-workflow"
            external_workflow.mkdir()
            (valid / "workflow_status").symlink_to(external_workflow, target_is_directory=True)
            symlinked = self._run_single_job_submitter(
                fixture, "phase-a", shadow_root=valid
            )
            self.assertEqual(symlinked.returncode, 2)
            self.assertIn("workflow_status must be a real in-root directory", symlinked.stderr)

    def test_concurrent_submitters_serialize_and_refresh_summary_under_flock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            fixture["sbatch_pause"].write_text("pause\n", encoding="utf-8")
            self._set_sacct_states(fixture, {"41001": "PENDING"})

            environment = {
                "PATH": f'{fixture["stub_bin"]}:{os.environ["PATH"]}',
                "HOME": str(fixture["project"].parent),
                "PROJECT_DIR": str(fixture["project"]),
                "EXECUTION_MODE": "slurm",
                "V2_STAGE": "phase-a",
                "DRY_RUN_SUBMIT": "0",
                "RUN_STAMP": "fault_recovery",
                "EXPERIMENT_ROOT": str(fixture["experiment"]),
                "RESULTS_ROOT": str(fixture["results"]),
                "PARENT_BROAD_SHADOW_ROOT": str(fixture["parent"]),
                "DATASET_ROOT": str(fixture["dataset"]),
                "SOURCE_SEGMENTATION_ROOT": str(fixture["segmentation"]),
                "HPC_CONTAINER_IMAGE": str(fixture["sif"]),
                "CPA_REFERENCE_ROOT": str(fixture["cpa_root"]),
                "EXPECTED_PARENT_CELL_COUNT": "32000",
                "DEPENDENCY_JOB_ID": "",
                "REFERENCE_QOS": "xxlarge",
                "REFERENCE_SHADOW_ROOT": str(fixture["shadow_root"]),
            }
            first = subprocess.Popen(
                ["bash", str(fixture["submitter"])],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            for _ in range(200):
                if fixture["sbatch_entered"].exists():
                    break
                import time
                time.sleep(0.025)
            self.assertTrue(fixture["sbatch_entered"].exists())
            second = subprocess.Popen(
                ["bash", str(fixture["submitter"])],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            fixture["sbatch_pause"].unlink()
            first_stdout, first_stderr = first.communicate(timeout=20)
            second_stdout, second_stderr = second.communicate(timeout=20)
            self.assertEqual(first.returncode, 0, first_stderr)
            self.assertEqual(second.returncode, 0, second_stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")
            self.assertIn("slurm_attempt_active=reference_cell_state_v2_phase_a:41001:PENDING", second_stdout)
            summary = fixture["shadow_root"] / "workflow_status" / "submission_phase-a_v2.tsv"
            self.assertEqual(summary.read_text(encoding="utf-8").count("job_id\t41001\n"), 1)

    def test_failed_single_stage_rejects_changed_external_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._make_single_job_submitter_fixture(temporary)
            initial = self._run_single_job_submitter(fixture, "phase-a", dependency="123")
            self.assertEqual(initial.returncode, 0, initial.stderr)
            self._set_sacct_states(fixture, {"41001": "FAILED"})
            changed = self._run_single_job_submitter(fixture, "phase-a", dependency="124")
            self.assertEqual(changed.returncode, 2)
            self.assertIn("changed the immutable external dependency", changed.stderr)
            self.assertEqual(fixture["sbatch_count"].read_text().strip(), "1")

    def test_workers_and_delegates_are_executable_and_parse(self) -> None:
        paths = [DOCKER_HPC / name for name in V2_NAMES]
        paths += [HOST_HPC / name for name in V2_NAMES]
        paths += [
            DOCKER_HPC
            / "Parameter_calibration"
            / "31_run_reference_cell_state_shadow_v2_test.sh",
            HOST_HPC
            / "Parameter_calibration"
            / "31_run_reference_cell_state_shadow_v2_test.sh",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            self.assertTrue(os.access(path, os.X_OK), path)
            completed = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_mapping_covers_every_v2_hpc_entrypoint(self) -> None:
        with (DOCKER_HPC / "hpc_script_mapping.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        mapping = {row["source_path"]: row["target_path"] for row in rows}
        for name in V2_NAMES:
            self.assertEqual(mapping.get(name), name)
        calibration = "Parameter_calibration/31_run_reference_cell_state_shadow_v2_test.sh"
        self.assertEqual(mapping.get(calibration), calibration)

    def test_new_parity_sif_identity_is_exact_and_verified(self) -> None:
        identity = {}
        for line in (
            DOCKER_HPC / "reference_cell_state_v2_sif_identity.tsv"
        ).read_text(encoding="utf-8").splitlines()[1:]:
            key, value = line.split("\t")
            identity[key] = value
        self.assertEqual(identity["status"], "VERIFIED")
        self.assertEqual(
            identity["image_path"],
            "/share/lab_crd/taoli/Docker/"
            "cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models-reference-v2-parity.sif",
        )
        self.assertEqual(identity["r_locked_package_count"], "66")
        self.assertEqual(
            identity["image_sha256"],
            "b3fda3cf5de4934c7533471f99b5d137d6f0431aa9dfa145b5da27d9e1687752",
        )
        self.assertEqual(identity["image_bytes"], "6932799488")
        self.assertEqual(
            identity["filesystem_immutability_mode"],
            "shared_filesystem_mode_bits_unavailable",
        )
        self.assertEqual(identity["runtime_rootfs_read_only"], "verified")
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"}"; '
                "reference_cell_state_v2_load_sif_identity",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_every_worker_rehashes_sif_and_probes_read_only_rootfs(self) -> None:
        contract = (
            DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('observed_sif="$(reference_cell_state_v2_sha256 "$sif")"', contract)
        self.assertIn("reference_cell_state_v2_verify_container_rootfs_read_only", contract)
        self.assertIn(
            "hpc_apptainer_exec /usr/bin/bash --noprofile --norc -c", contract
        )
        self.assertIn("V2 SIF root filesystem unexpectedly accepted a write", contract)
        for name in V2_NAMES[1:]:
            text = (DOCKER_HPC / name).read_text(encoding="utf-8")
            self.assertIn("reference_cell_state_v2_require_runtime_identity", text)
            self.assertIn("reference_cell_state_v2_verify_container_rootfs_read_only", text)

    def test_formal_submitter_clears_scheduler_environment_and_is_unpinned(self) -> None:
        text = (DOCKER_HPC / V2_NAMES[0]).read_text(encoding="utf-8")
        self.assertIn("/usr/bin/env -i", text)
        self.assertIn("ambient_sbatch_names", text)
        self.assertIn("reference_cell_state_shadow_v2_$RUN_STAMP", text)
        self.assertIn("reference_cell_state_shadow_v2_test_$RUN_STAMP", text)
        self.assertIn("--qos", text)
        for forbidden in (
            "--nodelist",
            "--constraint",
            "--exclude",
            "--gres",
            "--export=ALL",
            "--export=NONE",
        ):
            self.assertNotIn(forbidden, text)

    def test_submission_ledger_recovers_each_predict_dag_boundary(self) -> None:
        contract = DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "predict_jobs.tsv"

            def shell(body: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["bash", "-c", f'source "{contract}"; {body}'],
                    text=True,
                    capture_output=True,
                    check=False,
                )

            initialized = shell(
                f'reference_cell_state_v2_ledger_init_or_verify "{ledger}"; '
                f'reference_cell_state_v2_ledger_record "{ledger}" '
                'reference_cell_state_v2_accept 41001 none none'
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr)

            # New shells model an orchestrator dying immediately after each
            # successful sbatch/atomic-ledger boundary.
            after_accept = shell(
                f'reference_cell_state_v2_ledger_init_or_verify "{ledger}"; '
                f'test "$(reference_cell_state_v2_ledger_reuse "{ledger}" '
                'reference_cell_state_v2_accept none none)" = 41001; '
                f'! reference_cell_state_v2_ledger_reuse "{ledger}" '
                'reference_cell_state_v2_predict 41001 "1-27200%64"; '
                f'reference_cell_state_v2_ledger_record "{ledger}" '
                'reference_cell_state_v2_predict 41002 41001 "1-27200%64"'
            )
            self.assertEqual(after_accept.returncode, 0, after_accept.stderr)

            after_array = shell(
                f'reference_cell_state_v2_ledger_init_or_verify "{ledger}"; '
                f'test "$(reference_cell_state_v2_ledger_reuse "{ledger}" '
                'reference_cell_state_v2_predict 41001 "1-27200%64")" = 41002; '
                f'! reference_cell_state_v2_ledger_reuse "{ledger}" '
                'reference_cell_state_v2_finalize 41002 none; '
                f'reference_cell_state_v2_ledger_record "{ledger}" '
                'reference_cell_state_v2_finalize 41003 41002 none'
            )
            self.assertEqual(after_array.returncode, 0, after_array.stderr)

            final_resume = shell(
                f'reference_cell_state_v2_ledger_init_or_verify "{ledger}"; '
                f'test "$(reference_cell_state_v2_ledger_reuse "{ledger}" '
                'reference_cell_state_v2_finalize 41002 none)" = 41003'
            )
            self.assertEqual(final_resume.returncode, 0, final_resume.stderr)
            with ledger.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["job_id"] for row in rows], ["41001", "41002", "41003"])

    def test_sacct_array_aggregation_requires_exact_task_coverage(self) -> None:
        contract = DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)

            def aggregate(name: str, content: str) -> subprocess.CompletedProcess[str]:
                output = base / f"{name}.tsv"
                output.write_text(content, encoding="utf-8")
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'source "{contract}"; '
                        f'reference_cell_state_v2_aggregate_sacct_state 700 "1-4%2" "{output}"',
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )

            mixed_active = aggregate(
                "mixed-active",
                "700|700|COMPLETED\n"
                "700_1|700_1|FAILED\n"
                "700_2|700_2|RUNNING\n"
                "700_[3-4]|700_[3-4]|PENDING\n",
            )
            self.assertEqual(mixed_active.returncode, 0, mixed_active.stderr)
            self.assertEqual(mixed_active.stdout.strip(), "RUNNING")

            all_complete = aggregate(
                "all-complete", "700_[1-4]|700_[1-4]|COMPLETED\n"
            )
            self.assertEqual(all_complete.returncode, 0, all_complete.stderr)
            self.assertEqual(all_complete.stdout.strip(), "COMPLETED")

            terminal_failure = aggregate(
                "terminal-failure",
                "700_1|700_1|COMPLETED\n"
                "700_2|700_2|CANCELLED by 123\n"
                "700_[3-4]|700_[3-4]|FAILED+\n",
            )
            self.assertEqual(terminal_failure.returncode, 0, terminal_failure.stderr)
            self.assertEqual(terminal_failure.stdout.strip(), "FAILED")

            missing = aggregate(
                "missing", "700_[1-3]|700_[1-3]|COMPLETED\n"
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("coverage is incomplete", missing.stderr)

    def test_calibration_is_a30_direct_and_confined(self) -> None:
        text = (
            DOCKER_HPC
            / "Parameter_calibration"
            / "31_run_reference_cell_state_shadow_v2_test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("required_host=hpctpa3pc0009", text)
        self.assertIn("Tests_and_Parameters_calibration", text)
        self.assertIn("EXECUTION_MODE=direct_test", text)
        self.assertIn("$RESULTS_ROOT/broad_phenotype_shadow_20260812_075437", text)
        self.assertIn(
            'EXPECTED_PARENT_CELL_COUNT="${EXPECTED_PARENT_CELL_COUNT:-32000}"', text
        )
        self.assertNotIn("broad_phenotype_shadow_test_20260812_073844", text)
        self.assertIn("resume_existing_through_frozen_archive", text)
        self.assertNotIn("will not be resumed or overwritten", text)
        self.assertNotIn("sbatch", text.lower())

    def test_phase_a_is_historical_v2_and_current_axis_blinded(self) -> None:
        text = (DOCKER_HPC / "run_reference_cell_state_phase_a_v2.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "28_build_reference_cell_state_historical_projection.R",
            "--method-version v2",
            "reference_cell_state_classes_v2.tsv",
            "reference_cell_state_features_v2.json",
            "--plate-map",
            "for stage in validate umap annotate",
            "--cluster-balance 0.2",
            "--minimum-cluster-representatives 2",
            'dataset / "Dead"',
            'dataset / "Combined"',
            "current_classifier_read_performed",
        ):
            self.assertIn(required, text)
        self.assertNotIn("reference_cell_state_classes_v1.tsv", text)

    def test_exact_human_review_chain_has_no_second_sampling_stage(self) -> None:
        text = (DOCKER_HPC / "run_reference_cell_state_cpa_stage_v2.sh").read_text(
            encoding="utf-8"
        )
        for script in range(33, 39):
            self.assertIn(f"/{script}_", text)
        self.assertIn("exact_review_submission.json", text)
        self.assertIn("HUMAN_ADJUDICATION_REQUIRED", (ROOT / "cellpose_pipeline" / "scripts" / "36_merge_reference_cell_state_reviews.R").read_text(encoding="utf-8"))
        self.assertNotIn("--stage review-build", text)
        self.assertNotIn("--stage review-import", text)
        self.assertIn("post-seed2|post-adjudication", text)
        self.assertIn("merged_adjudication_barrier", text)
        self.assertIn("reference_cell_state_v2_require_posthuman_action_branch", text)
        self.assertIn('import_exact_review 1', text)
        self.assertIn('import_exact_review 2', text)
        self.assertNotIn("merged_adjudicated", text)

    def test_prediction_dag_and_comparison_are_isolated(self) -> None:
        submitter = (DOCKER_HPC / V2_NAMES[0]).read_text(encoding="utf-8")
        self.assertIn("reference_cell_state_v2_accept", submitter)
        self.assertIn("reference_cell_state_v2_predict", submitter)
        self.assertIn("reference_cell_state_v2_finalize", submitter)
        self.assertIn('"$accept_job" "1-$task_count%64"', submitter)
        self.assertIn('"$predict_job"', submitter)
        phase = (DOCKER_HPC / "run_reference_cell_state_phase_a_v2.sh").read_text(
            encoding="utf-8"
        )
        posthuman = (
            DOCKER_HPC / "run_reference_cell_state_cpa_stage_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("CURRENT_CLASSIFICATION_ROOT", phase)
        self.assertNotIn("CURRENT_CLASSIFICATION_ROOT", posthuman)
        comparison = (DOCKER_HPC / "run_reference_cell_state_compare_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT"',
            submitter,
        )
        self.assertIn(
            'reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT"',
            comparison,
        )
        self.assertIn("CURRENT_CLASSIFICATION_ROOT", comparison)
        self.assertIn("descriptive_cross_classification_not_accuracy", comparison)
        self.assertIn('expected_binds="$comparison_parent:$comparison_parent:rw,$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:ro,$CURRENT_CLASSIFICATION_ROOT:$CURRENT_CLASSIFICATION_ROOT:ro"', comparison)
        self.assertIn('--current-root "$CURRENT_CLASSIFICATION_ROOT"', comparison)
        self.assertIn("reference-v2-comparison-bind-probe", comparison)
        self.assertIn("Comparison bind write-policy probe failed", comparison)
        self.assertNotIn('mkdir "$COMPARISON_ROOT"', comparison)
        compare_script = (
            ROOT / "cellpose_pipeline" / "scripts" / "29_compare_current_vs_reference_cell_state.py"
        ).read_text(encoding="utf-8")
        self.assertIn("comparison_verified_reuse=1", compare_script)
        self.assertIn("os.replace(staging_root, output_root)", compare_script)
        self.assertNotIn('"--force"', compare_script)

    def test_prediction_workers_match_atomic_generation_and_final_layout(self) -> None:
        predict = (
            DOCKER_HPC / "run_reference_cell_state_predict_array_task_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'generation_dir="$PREDICTION_ROOT/shards/$well/${field_key}__${shard_branch}"',
            predict,
        )
        self.assertIn('output_tsv="$generation_dir/reference_cell_state_predictions.tsv"', predict)
        self.assertIn('output_receipt="$generation_dir/prediction_receipt.json"', predict)
        self.assertNotIn("$PREDICTION_ROOT/receipts/", predict)
        self.assertIn(
            "probability__live_cell\\tprobability__dead_cell\\t"
            "probability__multinucleated_cell",
            predict,
        )
        finalize = (
            DOCKER_HPC / "run_reference_cell_state_finalize_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'REFERENCE_RECEIPT="$REFERENCE_SHADOW_ROOT/predictions/'
            'REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json"',
            finalize,
        )
        submitter = (DOCKER_HPC / V2_NAMES[0]).read_text(encoding="utf-8")
        self.assertIn('PREDICTION_ROOT="$REFERENCE_SHADOW_ROOT/prediction_shards_v2"', submitter)
        self.assertIn("post-adjudication", submitter)

    def test_worker_retries_delegate_to_full_verified_reuse_contracts(self) -> None:
        phase = (DOCKER_HPC / "run_reference_cell_state_phase_a_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("reference_cell_state_phase_a_v2_verified_reuse=1", phase)
        self.assertIn('reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT"', phase)
        for evidence in (
            "annotation_manifest",
            "morphology_overlay_manifest",
            "morphology_workspace",
            "seed1_selection_manifest",
            "seed1_review_set",
            "seed1_render_manifest",
            "seed1_exact_review_html",
            "seed1_crop_status",
        ):
            self.assertIn(evidence, phase)
        self.assertIn("--overwrite", phase)
        posthuman = (
            DOCKER_HPC / "run_reference_cell_state_cpa_stage_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("finish_posthuman_stage", posthuman)
        self.assertNotIn("exact-render output already exists", posthuman)
        self.assertNotIn("Seed1 generation already exists", posthuman)
        for script, marker in (
            ("26_prepare_reference_cell_state_project.py", "generation_status=verified_reuse"),
            ("33_build_reference_cell_state_seed1_review.R", "seed1_review_verified_reuse=1"),
            ("34_train_reference_cell_state_historical.R", "historical_train_verified_reuse=1"),
            ("35_build_reference_cell_state_seed2_review.R", "seed2_review_verified_reuse=1"),
            ("36_merge_reference_cell_state_reviews.R", "review_merge_verified_reuse=1"),
            ("37_render_reference_cell_state_exact_review.py", "exact_review_verified_reuse=1"),
            ("38_import_reference_cell_state_exact_review.R", "exact_review_import_verified_reuse=1"),
        ):
            self.assertIn(
                marker,
                (ROOT / "cellpose_pipeline" / "scripts" / script).read_text(encoding="utf-8"),
                script,
            )
        accept = (
            DOCKER_HPC / "run_reference_cell_state_model_acceptance_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("require_new_output", accept)
        self.assertIn("Existing V2 model-acceptance SHA sidecar differs", accept)
        predict = (
            DOCKER_HPC / "run_reference_cell_state_predict_array_task_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("retry must use a fresh V2 root", predict)
        finalize = (
            DOCKER_HPC / "run_reference_cell_state_finalize_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("already exists and cannot be overwritten", finalize)
        contract = (
            DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("Historical projection branch is not boolean", contract)
        self.assertIn("Phase A receipt branch differs from historical manifest", contract)
        submitter = (DOCKER_HPC / V2_NAMES[0]).read_text(encoding="utf-8")
        self.assertIn('case "$V2_STAGE" in', submitter)
        self.assertIn('predict) reference_cell_state_v2_audit_final_predictions', submitter)

    def test_scientific_parity_status_is_branch_specific_and_disclosed(self) -> None:
        phase = (DOCKER_HPC / "run_reference_cell_state_phase_a_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "historical_core_parity_with_pinned_stable_polygon_sampling", phase
        )
        self.assertIn(
            "historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation",
            phase,
        )
        self.assertIn("not_pinned_reference_exact_sampling", phase)

    def test_v2_root_guard_rejects_a_v1_root(self) -> None:
        contract = DOCKER_HPC / "util" / "reference_cell_state_v2_contract.sh"
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            v1 = results / "reference_cell_state_shadow_20260812_000000"
            v1.mkdir(parents=True)
            command = (
                f'source "{contract}"; '
                f'reference_cell_state_v2_require_shadow_root "{v1}" formal "{results}"'
            )
            completed = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 2)

    def test_r_lock_is_exact_66_package_reference_v2_parity_closure(self) -> None:
        with (CONTAINER / "locks" / "r-packages.lock.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 66)
        versions = {row["package"]: row["version"] for row in rows}
        self.assertEqual(versions["generics"], "0.1.4")
        self.assertEqual(versions["dbscan"], "1.2.3")
        generator = (CONTAINER / "scripts" / "prepare_r_package_context.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"dbscan",', generator)
        for required in ('"dplyr",', '"stringr",', '"ggplot2",', '"purrr",', '"tidyr",'):
            self.assertIn(required, generator)
        verifier = (CONTAINER / "scripts" / "verify_r_environment.R").read_text(
            encoding="utf-8"
        )
        self.assertIn("dbscan_reference_call_fixture=PASS", verifier)
        self.assertIn("reference_tidy_namespace_fixture=PASS", verifier)


if __name__ == "__main__":
    unittest.main()
