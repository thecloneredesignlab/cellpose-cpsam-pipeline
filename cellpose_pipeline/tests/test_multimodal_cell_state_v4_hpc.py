from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "cellpose_pipeline/Docker/hpc"
HOST = ROOT / "cellpose_pipeline/hpc"


class MultimodalCellStateV4HpcTests(unittest.TestCase):
    def test_calibration_is_direct_a30_no_gpu_latest_sif_and_exact_allowlist(self) -> None:
        launcher = DOCKER / "Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn('required_host="${V4_REQUIRED_HOST:-hpctpa3pc0028}"', text)
        self.assertIn('[[ -z "${SLURM_JOB_ID:-}" ]]', text)
        self.assertNotIn("sbatch ", text)
        self.assertIn("HPC_CONTAINER_GPU=0", text)
        self.assertIn("HPC_CONTAINER_NO_MOUNT=/share", text)
        self.assertIn("reference_cell_state_v2_require_runtime_identity", text)
        self.assertIn("REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256", text)
        self.assertIn("multimodal_cell_state_v4_test_*", text)
        self.assertNotIn("--nodelist", text)
        bind_line = next(line for line in text.splitlines() if line.startswith("expected_binds="))
        for name in ("$BRIGHTFIELD_ROOT", "$NUCLEI_RAW_ROOT", "$DEAD_RAW_ROOT", "$COMBINED_MASK_ROOT", "$NUCLEI_MASK_ROOT"):
            self.assertIn(name, bind_line)
        self.assertNotIn("$DATASET_ROOT:", bind_line)
        self.assertIn('segmentation / "Dead"', text)
        self.assertIn("raw_dead_visible=1 existing_dead_segmentation_visible=0", text)
        for action in ("post-anchor", "post-region", "post-review"):
            self.assertIn(action, text)
        self.assertIn("ANCHOR_SUBMISSION", text)
        self.assertIn("REGION_SUBMISSION", text)
        self.assertIn("BROAD_REVIEW_SUBMISSION", text)
        self.assertIn('case "$submission" in "$V4_SHADOW_ROOT"/*)', text)

    def test_host_delegate_mapping_and_shell_syntax(self) -> None:
        host = HOST / "Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh"
        target = "cellpose_pipeline/Docker/hpc/Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh"
        self.assertIn(target, host.read_text(encoding="utf-8"))
        row = "Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh\tParameter_calibration/34_run_multimodal_cell_state_v4_test.sh\tpresent"
        mapping = (DOCKER / "hpc_script_mapping.tsv").read_text(encoding="utf-8")
        self.assertEqual(mapping.splitlines().count(row), 1)
        result = subprocess.run(
            ["bash", "-n", str(host), str(DOCKER / "Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh")],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_driver_stops_at_independent_three_channel_anchor_barrier(self) -> None:
        text = (ROOT / "cellpose_pipeline/scripts/54_run_multimodal_cell_state_v4_calibration.py").read_text(encoding="utf-8")
        self.assertIn('"human_barrier": "independent_three_channel_anchor_submission_required"', text)
        self.assertIn('"dead_raw": "read_as_independent_measurement"', text)
        self.assertIn('"existing_dead_segmentation": "not_read"', text)
        self.assertIn('"annotation_workspace"', text)
        self.assertIn("ThreadPoolExecutor", text)
        self.assertNotIn("region_submission.json", text)
        posthuman = (
            ROOT / "cellpose_pipeline/scripts/59_run_multimodal_cell_state_v4_posthuman.py"
        ).read_text(encoding="utf-8")
        self.assertIn('ACTIONS = ("post-anchor", "post-region", "post-review")', posthuman)
        self.assertIn("broad_dead_region_submission_required", posthuman)
        self.assertIn("broad_region_three_channel_review_submission_required", posthuman)
        self.assertIn("model_training_requires_separate_technical_acceptance", posthuman)
        builder = (
            ROOT / "cellpose_pipeline/scripts/51_prepare_multimodal_cell_state_v4_project.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"group_column": "source_id"', builder)
        self.assertIn('"allow_ungrouped": False', builder)
        self.assertEqual(builder.count('"role": "phenotype"'), 2)
        self.assertNotIn('"role": "death_primary"', builder)


if __name__ == "__main__":
    unittest.main()
