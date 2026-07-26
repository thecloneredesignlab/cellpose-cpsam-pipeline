from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = (
    REPO_ROOT
    / "cellpose_pipeline"
    / "configs"
    / "segmentation_freeze_v3_20260725.json"
)


class SegmentationFreezeTests(unittest.TestCase):
    def test_frozen_segmentation_files_match_recorded_hashes(self) -> None:
        payload = json.loads(FREEZE_PATH.read_text())
        self.assertTrue(payload["classification_only"])
        self.assertTrue(payload["policy"]["forbid_run_segmentation_jobs"])
        self.assertTrue(payload["policy"]["forbid_overwrite_segmentation_results"])
        self.assertGreater(len(payload["sha256"]), 20)
        failures = []
        for relative_path, expected in payload["sha256"].items():
            path = REPO_ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing:{relative_path}")
                continue
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected:
                failures.append(
                    f"changed:{relative_path}:expected={expected}:observed={observed}"
                )
        self.assertEqual(failures, [])

    def test_classification_submitter_does_not_invoke_segmentation_workers(
        self,
    ) -> None:
        submitter = (
            REPO_ROOT
            / "cellpose_pipeline"
            / "hpc"
            / "submit_classification_only_full.sh"
        ).read_text()
        forbidden_workers = (
            "run_cellpose_cpsam_array_task.sh",
            "run_dead_calibration_array_task.sh",
            "run_dead_calibration_merge.sh",
            "run_dead_consensus_merge.sh",
            "run_nucleated_branch_array_task.sh",
            "run_nucleated_branch_merge.sh",
            "run_shape_strict_array_task.sh",
            "run_shape_strict_finalize.sh",
        )
        for worker in forbidden_workers:
            self.assertNotIn(worker, submitter)
        self.assertIn("classification_only=1", submitter)
        self.assertIn("SOURCE_RUN_ROOT", submitter)
        self.assertIn("--array \"$ARRAY_SPEC\"", submitter)
        self.assertNotIn("%256", submitter)


if __name__ == "__main__":
    unittest.main()
