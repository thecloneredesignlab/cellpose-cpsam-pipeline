from __future__ import annotations

import os
import stat
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER_HPC = ROOT / "cellpose_pipeline" / "Docker" / "hpc"
HOST_HPC = ROOT / "cellpose_pipeline" / "hpc"
SUBMITTER = DOCKER_HPC / "submit_reference_cell_state_shadow.sh"
WORKER = DOCKER_HPC / "run_reference_cell_state_phase_a.sh"
CALIBRATION = (
    DOCKER_HPC
    / "Parameter_calibration"
    / "30_run_reference_cell_state_shadow_test.sh"
)


class ReferenceCellStateHpcContracts(unittest.TestCase):
    def test_entrypoints_exist_and_parse(self) -> None:
        paths = [
            SUBMITTER,
            WORKER,
            CALIBRATION,
            HOST_HPC / "submit_reference_cell_state_shadow.sh",
            HOST_HPC / "run_reference_cell_state_phase_a.sh",
            HOST_HPC
            / "Parameter_calibration"
            / "30_run_reference_cell_state_shadow_test.sh",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            completed = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_worker_uses_only_reference_inputs_and_stops_at_barrier(self) -> None:
        text = WORKER.read_text(encoding="utf-8")
        for required in (
            "26_prepare_reference_cell_state_project.py",
            "27_build_reference_morphology_workspace.py",
            "for stage in validate umap annotate",
            "human_barrier=region_submission_required",
            "PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro",
            "BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro",
            "NUCLEI_ROOT:$NUCLEI_ROOT:ro",
            "COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro",
            "CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro",
        ):
            self.assertIn(required, text)
        for forbidden in (
            "CLASSIFICATION_ROOT:$CLASSIFICATION_ROOT",
            "/Dead/",
            "Combined RGB",
            "final_state",
            "classification_confidence",
        ):
            self.assertNotIn(forbidden, text)

    def test_submitter_is_cpu_only_unpinned_and_node_local(self) -> None:
        text = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("HPC_CONTAINER_GPU=0", text)
        self.assertIn("Node-local archive SHA-256 mismatch", text)
        self.assertIn("HPC_PROJECT_ROOT_SOURCE", text)
        self.assertIn("--qos", text)
        self.assertIn("--chdir", text)
        self.assertNotIn("--gres", text)
        self.assertNotIn("--nodelist", text)
        self.assertNotIn("--constraint", text)
        self.assertNotIn("--export=ALL", text)
        self.assertNotIn("--export=NONE", text)

    def test_submitter_has_independent_output_and_no_current_bind(self) -> None:
        text = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("reference_cell_state_shadow_$RUN_STAMP", text)
        self.assertIn("current_classification_root_bound\\tfalse", text)
        self.assertIn("dead_channel_bound\\tfalse", text)
        self.assertNotIn("CLASSIFICATION_ROOT=", text)
        self.assertNotIn("LEGACY_NO_GO=", text)
        self.assertNotIn("$DATASET_ROOT:$DATASET_ROOT", text)
        self.assertNotIn("$SOURCE_SEGMENTATION_ROOT:$SOURCE_SEGMENTATION_ROOT", text)
        self.assertIn("HPC_CONTAINER_NO_MOUNT=/share", text)
        self.assertIn(
            '"$HPC_CONTAINER_IMAGE" == "$DEFAULT_SIF"',
            text,
        )
        self.assertIn(
            '"$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT"',
            text,
        )
        runtime = (DOCKER_HPC / "util" / "hpc_container_apptainer_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('command+=(--no-mount "${HPC_CONTAINER_NO_MOUNT}")', runtime)

    def test_calibration_hard_pins_compute_node_and_test_root(self) -> None:
        text = CALIBRATION.read_text(encoding="utf-8")
        self.assertIn('required_host="hpctpa3pc0009"', text)
        self.assertIn("Tests_and_Parameters_calibration", text)
        self.assertIn("direct_test_no_sbatch", text)
        self.assertIn("broad_phenotype_shadow_test_20260812_073844", text)
        self.assertNotIn("sbatch", text.replace("no_sbatch", ""))

    def test_forbidden_ambient_overrides_are_rejected(self) -> None:
        text = SUBMITTER.read_text(encoding="utf-8")
        for name in (
            "HPC_CONTAINER_BINDS",
            "HPC_CONTAINER_RUNTIME_ROOT",
            "HPC_CONTAINER_FORWARD_PREFIXES",
            "HPC_PROJECT_ROOT_SOURCE",
            "HPC_CONTAINER_NO_MOUNT",
            "CPA_DEPENDENCY_LOCK",
            "REFERENCE_CELL_STATE_PROJECT_BUILDER",
            "REFERENCE_MORPHOLOGY_WORKSPACE_SCRIPT",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
