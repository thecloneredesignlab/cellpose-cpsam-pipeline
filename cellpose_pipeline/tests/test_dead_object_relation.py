from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "08_fuse_multichannel_classification.py"
)
SPEC = importlib.util.spec_from_file_location("fusion_classification", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
FUSION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUSION
SPEC.loader.exec_module(FUSION)


class DeadObjectRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            dead_uncertain_min_combined_overlap_fraction=0.45,
            dead_live_rgb_min_combined_overlap_fraction=0.60,
            dead_live_rgb_max_area=500,
        )

    def same_cell(self, rgb_state: str, area: int, overlap: float) -> bool:
        evidence = SimpleNamespace(combined_overlap_fraction=overlap)
        cell = {"rgb_state": rgb_state, "area": area}
        return FUSION.dead_object_has_same_cell_support(evidence, cell, self.args)

    def test_figure5_case3_large_cell_low_coverage_is_adjacent(self) -> None:
        self.assertFalse(self.same_cell("uncertain", 1761, 0.064736))

    def test_figure5_case6_compact_cell_supported_coverage_is_same_cell(self) -> None:
        self.assertTrue(self.same_cell("uncertain", 413, 0.343826))

    def test_rgb_live_is_never_rewritten_by_object_relation(self) -> None:
        self.assertFalse(self.same_cell("live", 100, 1.0))

    def test_rgb_dead_remains_same_cell(self) -> None:
        self.assertTrue(self.same_cell("dead", 2000, 0.01))

    def test_uncertain_cell_requires_full_or_compact_support(self) -> None:
        self.assertTrue(self.same_cell("uncertain", 1800, 0.45))
        self.assertFalse(self.same_cell("uncertain", 450, 0.29))

    def test_nucleus_evidence_refines_confirmed_overlap_relation(self) -> None:
        objects = [
            SimpleNamespace(label=1, association_relation="adjacent_dead"),
            SimpleNamespace(label=2, association_relation="adjacent_dead"),
            SimpleNamespace(label=3, association_relation="adjacent_dead"),
        ]
        result = SimpleNamespace(objects=objects)
        evidence = {
            1: {"nucleus_supported_relation": "probable_live_dead_overlap_multi_nucleus"},
            2: {"nucleus_supported_relation": "live_with_death_signal_single_nucleus"},
            3: {"nucleus_supported_relation": "adjacent_or_overlapping_dead_nucleus_unresolved"},
        }
        FUSION.apply_nucleus_aware_overlap_relations(result, evidence)
        self.assertEqual(
            [item.association_relation for item in objects],
            [
                "overlapping_live_dead_multi_nucleus",
                "live_with_death_signal",
                "adjacent_or_overlapping_dead_uncertain",
            ],
        )

    def test_overlap_mask_preserves_cell_and_confirmed_dead_ids(self) -> None:
        combined = np.array([[0, 4, 4], [0, 4, 4]], dtype=np.int32)
        dead = np.array([[0, 0, 7], [0, 0, 7]], dtype=np.int32)
        cell, confirmed, overlap = FUSION.build_cell_dead_overlap_masks(
            combined,
            dead,
            [{"mask_id": 4, "state": "live"}],
            [{"dead_mask_id": 7, "confirmed_dead_object": True}],
        )
        self.assertEqual(overlap.shape, (2, 2, 3))
        self.assertEqual(int(cell[0, 2]), 4)
        self.assertEqual(int(confirmed[0, 2]), 7)
        self.assertEqual((int(overlap[0, 0, 2]), int(overlap[1, 0, 2])), (4, 7))


if __name__ == "__main__":
    unittest.main()
