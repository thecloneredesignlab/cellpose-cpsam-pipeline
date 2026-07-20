from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TIMEPOINT = load_module(
    "timepoint_selection_test",
    PIPELINE_ROOT / "scripts" / "_shared" / "timepoint_selection.py",
)
MANIFEST = load_module(
    "postsegmentation_manifest_test",
    PIPELINE_ROOT / "scripts" / "06_build_postsegmentation_field_manifest.py",
)
AUDIT = load_module(
    "dead_classification_audit_test",
    PIPELINE_ROOT
    / "scripts"
    / "Parameter_calibration"
    / "25_audit_dead_classification_without_ground_truth.py",
)


class TimepointSelectionTests(unittest.TestCase):
    def test_d0_aliases_normalize_to_exact_token(self) -> None:
        for value in ("d0", "day0", "0", "00d00h00m"):
            self.assertEqual(TIMEPOINT.normalize_timepoint(value), "00d00h00m")

    def test_select_keys_excludes_other_timepoints(self) -> None:
        keys = [
            "A2_1_00d00h00m",
            "A2_1_00d12h00m",
            "B3_4_00d00h00m",
            "B3_4_01d00h00m",
        ]
        self.assertEqual(
            TIMEPOINT.select_keys(keys, "d0"),
            ["A2_1_00d00h00m", "B3_4_00d00h00m"],
        )

    def test_extracts_key_from_raw_and_mask_names(self) -> None:
        for name in (
            "SUM159_AC_A2_1_00d00h00m.tif",
            "SUM159_AC_Exp1_Dead_A2_1_00d00h00m_cp_masks.tif",
        ):
            self.assertEqual(
                TIMEPOINT.extract_key_and_timepoint(name),
                ("A2_1_00d00h00m", "00d00h00m"),
            )

    def test_invalid_timepoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TIMEPOINT.normalize_timepoint("day-zero-ish")

    def test_manifest_validation_allows_unselected_timepoints(self) -> None:
        selected = "A2_1_00d00h00m"
        index = {
            selected: Path("selected.tif"),
            "A2_1_00d12h00m": Path("unselected.tif"),
        }
        MANIFEST.validate_keys("raw_combined", index, {selected}, allow_extra=True)

    def test_manifest_task_list_mode_still_rejects_extra_keys(self) -> None:
        selected = "A2_1_00d00h00m"
        index = {
            selected: Path("selected.tif"),
            "A2_1_00d12h00m": Path("unselected.tif"),
        }
        with self.assertRaisesRegex(ValueError, "extra"):
            MANIFEST.validate_keys("raw_combined", index, {selected})

    def test_manifest_validation_rejects_missing_selected_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            MANIFEST.validate_keys("raw_combined", {}, {"A2_1_00d00h00m"})

    def test_audit_selector_ignores_aggregate_and_other_timepoint_files(self) -> None:
        paths = [
            Path("cell_count_summary.csv"),
            Path("SUM159_AC_A2_1_00d00h00m_summary.csv"),
            Path("SUM159_AC_A2_1_00d12h00m_summary.csv"),
        ]
        self.assertEqual(
            AUDIT.selected_paths(paths, "00d00h00m"),
            [Path("SUM159_AC_A2_1_00d00h00m_summary.csv")],
        )


if __name__ == "__main__":
    unittest.main()
