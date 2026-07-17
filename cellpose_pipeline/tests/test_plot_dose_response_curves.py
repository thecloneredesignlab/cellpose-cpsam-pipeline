from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysisi"
    / "07_plot_dose_response_curves.py"
)
SPEC = importlib.util.spec_from_file_location("plot_dose_response_curves", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
DOSE_RESPONSE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOSE_RESPONSE
SPEC.loader.exec_module(DOSE_RESPONSE)


def well_time_rows(
    well: str,
    plate_row: str,
    plate_column: int,
    dose: float,
    replicate: int,
    live_counts: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "well": well,
            "elapsed_hours": float(time),
            "live_count": float(live_count),
            "plate_row": plate_row,
            "plate_column": plate_column,
            "doxorubicin_nm": dose,
            "ploidy": "2N",
            "cyclophosphamide": True,
            "replicate": replicate,
        }
        for time, live_count in zip((0, 1, 2), live_counts)
    ]


class DoseResponseTests(unittest.TestCase):
    def test_auc_is_baseline_corrected_and_normalized_to_paired_vehicle(self) -> None:
        rows = []
        rows.extend(well_time_rows("A2", "A", 2, 0.0, 1, [100, 200, 300]))
        rows.extend(well_time_rows("A3", "A", 3, 10.0, 1, [100, 100, 100]))
        rows.extend(well_time_rows("B2", "B", 2, 0.0, 2, [200, 300, 400]))
        rows.extend(well_time_rows("B3", "B", 3, 10.0, 2, [200, 100, 0]))
        data = pd.DataFrame(rows)

        raw = DOSE_RESPONSE.extract_raw_responses(
            data,
            metric="auc",
            start_hours=0.0,
            endpoint_hours=2.0,
            endpoint_window_hours=0.0,
        )
        response = DOSE_RESPONSE.normalize_to_paired_vehicle(raw).set_index("well")

        self.assertAlmostEqual(float(response.loc["A2", "raw_response"]), 4.0)
        self.assertAlmostEqual(float(response.loc["A3", "raw_response"]), 2.0)
        self.assertAlmostEqual(float(response.loc["A3", "normalized_response"]), 0.5)
        self.assertAlmostEqual(float(response.loc["B2", "raw_response"]), 3.0)
        self.assertAlmostEqual(float(response.loc["B3", "raw_response"]), 1.0)
        self.assertAlmostEqual(float(response.loc["B3", "normalized_response"]), 1 / 3)

    def test_hill_fit_recovers_known_parameters(self) -> None:
        doses = np.array([0, 3.125, 6.25, 12.5, 25, 50, 100, 200, 400, 800], dtype=float)
        expected_bottom = 0.08
        expected_top = 1.02
        expected_ec50 = 42.0
        expected_slope = 2.25
        base_response = expected_bottom + (expected_top - expected_bottom) / (
            1.0 + (doses / expected_ec50) ** expected_slope
        )
        rows = []
        for replicate, offset in ((1, -0.006), (2, 0.006)):
            for dose, value in zip(doses, base_response):
                rows.append(
                    {
                        "well": f"R{replicate}_{dose:g}",
                        "doxorubicin_nm": dose,
                        "normalized_response": value + offset,
                    }
                )
        data = pd.DataFrame(rows)

        fit = DOSE_RESPONSE.fit_hill_curve(
            data,
            condition="synthetic",
            cyclophosphamide=False,
            ploidy="2N",
        )

        self.assertAlmostEqual(fit.bottom, expected_bottom, delta=0.015)
        self.assertAlmostEqual(fit.top, expected_top, delta=0.015)
        self.assertAlmostEqual(fit.ec50_nm, expected_ec50, delta=1.5)
        self.assertAlmostEqual(fit.hill_slope, expected_slope, delta=0.12)
        self.assertGreater(fit.r_squared, 0.999)

    def test_endpoint_window_averages_baseline_normalized_live_counts(self) -> None:
        rows = well_time_rows("A2", "A", 2, 0.0, 1, [100, 150, 200])
        data = pd.DataFrame(rows)

        raw = DOSE_RESPONSE.extract_raw_responses(
            data,
            metric="endpoint",
            start_hours=0.0,
            endpoint_hours=2.0,
            endpoint_window_hours=1.0,
        ).set_index("well")

        self.assertAlmostEqual(float(raw.loc["A2", "raw_response"]), 1.75)

    def test_run_directory_resolves_branch_well_time_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            csv_path = (
                run_dir
                / "analysis"
                / "well_count_timecourses"
                / "fusion"
                / "well_time_cell_state_counts.csv"
            )
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("well,elapsed_hours\n")

            resolved = DOSE_RESPONSE.resolve_well_time_csv(run_dir, "fusion")

        self.assertEqual(resolved.csv_path, csv_path.resolve())
        self.assertEqual(resolved.run_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
