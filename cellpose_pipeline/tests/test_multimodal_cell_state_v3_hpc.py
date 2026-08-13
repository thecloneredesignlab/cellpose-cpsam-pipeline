from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "cellpose_pipeline" / "Docker" / "hpc"
HOST = ROOT / "cellpose_pipeline" / "hpc"


class MultimodalCellStateV3HpcTests(unittest.TestCase):
    def test_calibration_is_direct_node_locked_blinded_and_latest_sif_gated(
        self,
    ) -> None:
        launcher = (
            DOCKER / "Parameter_calibration" / "33_run_multimodal_cell_state_v3_test.sh"
        )
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("required_host=hpctpa3pc0009", text)
        self.assertIn('[[ -z "${SLURM_JOB_ID:-}" ]]', text)
        self.assertNotIn("sbatch ", text)
        self.assertIn("HPC_CONTAINER_GPU=0", text)
        self.assertIn("HPC_CONTAINER_NO_MOUNT=/share", text)
        self.assertIn("reference_cell_state_v2_require_runtime_identity", text)
        self.assertIn("REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256", text)
        self.assertIn("multimodal_cell_state_v3_test_*", text)
        self.assertNotIn("--nodelist", text)
        bind_line = next(
            line for line in text.splitlines() if line.startswith("expected_binds=")
        )
        self.assertIn("$BRIGHTFIELD_ROOT", bind_line)
        self.assertIn("$NUCLEI_RAW_ROOT", bind_line)
        self.assertIn("$COMBINED_MASK_ROOT", bind_line)
        self.assertIn("$NUCLEI_MASK_ROOT", bind_line)
        self.assertIn("$NUCLEI_CORE_ROOT", bind_line)
        self.assertNotIn("$DATASET_ROOT:", bind_line)
        self.assertNotIn("Dead", bind_line)

    def test_launcher_and_mapping_are_exact(self) -> None:
        host = (
            HOST / "Parameter_calibration" / "33_run_multimodal_cell_state_v3_test.sh"
        )
        target = "cellpose_pipeline/Docker/hpc/Parameter_calibration/33_run_multimodal_cell_state_v3_test.sh"
        self.assertIn(target, host.read_text(encoding="utf-8"))
        mapping = (DOCKER / "hpc_script_mapping.tsv").read_text(encoding="utf-8")
        row = "Parameter_calibration/33_run_multimodal_cell_state_v3_test.sh\tParameter_calibration/33_run_multimodal_cell_state_v3_test.sh\tpresent"
        self.assertEqual(mapping.splitlines().count(row), 1)
        result = subprocess.run(
            [
                "bash",
                "-n",
                str(host),
                str(
                    DOCKER
                    / "Parameter_calibration"
                    / "33_run_multimodal_cell_state_v3_test.sh"
                ),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_driver_stops_at_blind_human_barrier(self) -> None:
        driver = (
            ROOT
            / "cellpose_pipeline"
            / "scripts"
            / "47_run_multimodal_cell_state_v3_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"human_barrier": "blind_exact_review_submission_required"', driver
        )
        self.assertIn('"dead": "not_read"', driver)
        self.assertIn('"heldout": "not_read"', driver)
        self.assertNotIn("region_submission.json", driver)
        self.assertIn("ThreadPoolExecutor", driver)


if __name__ == "__main__":
    unittest.main()
