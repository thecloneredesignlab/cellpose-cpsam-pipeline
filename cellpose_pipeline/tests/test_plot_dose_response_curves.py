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
    def test_default_analysis_specs_include_auc_day4_and_day5(self) -> None:
        data = pd.DataFrame({"elapsed_hours": [0.0, 168.0]})

        specs = DOSE_RESPONSE.build_analysis_specs(
            data,
            primary_metric="auc",
            start_hours=None,
            endpoint_hours=None,
            endpoint_window_hours=0.0,
            additional_endpoint_days=[4.0, 5.0],
        )

        self.assertEqual([spec.output_key for spec in specs], ["auc", "day4", "day5"])
        self.assertEqual([spec.metric for spec in specs], ["auc", "endpoint", "endpoint"])
        self.assertEqual([spec.endpoint_hours for spec in specs], [168.0, 96.0, 120.0])
        self.assertEqual([spec.endpoint_day for spec in specs], [None, 4.0, 5.0])

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

    def test_endpoint_gr_value_has_expected_biological_scale(self) -> None:
        self.assertAlmostEqual(DOSE_RESPONSE.endpoint_gr_value(4.0, 4.0), 1.0)
        self.assertAlmostEqual(
            DOSE_RESPONSE.endpoint_gr_value(2.0, 4.0),
            np.sqrt(2.0) - 1.0,
        )
        self.assertAlmostEqual(DOSE_RESPONSE.endpoint_gr_value(1.0, 4.0), 0.0)
        self.assertAlmostEqual(DOSE_RESPONSE.endpoint_gr_value(0.0, 4.0), -1.0)
        with self.assertRaises(ValueError):
            DOSE_RESPONSE.endpoint_gr_value(2.0, 1.0)

    def test_gr_fit_recovers_known_fixed_top_parameters(self) -> None:
        doses = np.array([0, 3.125, 6.25, 12.5, 25, 50, 100, 200, 400, 800], dtype=float)
        expected_gr_inf = -0.2
        expected_gec50 = 42.0
        expected_slope = 2.1
        base_response = DOSE_RESPONSE.gr_model_parameterized(
            doses,
            expected_gr_inf,
            np.log10(expected_gec50),
            expected_slope,
        )
        rows = []
        for replicate, offset in ((1, -0.006), (2, 0.006)):
            for dose, value in zip(doses, base_response):
                rows.append(
                    {
                        "well": f"R{replicate}_{dose:g}",
                        "doxorubicin_nm": dose,
                        "gr_value": value if dose == 0 else value + offset,
                    }
                )
        data = pd.DataFrame(rows)

        fit = DOSE_RESPONSE.fit_gr_curve(
            data,
            condition="synthetic",
            cyclophosphamide=False,
            ploidy="2N",
            analysis_key="day4",
            endpoint_day=4.0,
            endpoint_hours=96.0,
        )

        self.assertAlmostEqual(fit.gr_inf, expected_gr_inf, delta=0.02)
        self.assertAlmostEqual(fit.gec50_nm, expected_gec50, delta=1.0)
        self.assertAlmostEqual(fit.hill_slope, expected_slope, delta=0.1)
        self.assertGreater(fit.r_squared, 0.999)

    def test_paired_gr_difference_is_four_n_minus_two_n(self) -> None:
        rows = pd.DataFrame(
            [
                {"doxorubicin_nm": 25.0, "replicate": 1, "ploidy": "2N", "gr_value": 0.6},
                {"doxorubicin_nm": 25.0, "replicate": 1, "ploidy": "4N", "gr_value": 0.8},
                {"doxorubicin_nm": 25.0, "replicate": 2, "ploidy": "2N", "gr_value": 0.7},
                {"doxorubicin_nm": 25.0, "replicate": 2, "ploidy": "4N", "gr_value": 0.65},
            ]
        )

        paired = DOSE_RESPONSE.paired_gr_differences(rows).sort_values("replicate")

        np.testing.assert_allclose(paired["delta_gr"], [0.2, -0.05])

    def test_gr_bootstrap_returns_curve_bands_and_delta_interval(self) -> None:
        doses = np.array([0, 3.125, 6.25, 12.5, 25, 50, 100, 200, 400, 800], dtype=float)
        rows = []
        fits = []
        for ploidy, gr_inf, gec50 in (("2N", -0.3, 35.0), ("4N", -0.15, 48.0)):
            base_response = DOSE_RESPONSE.gr_model_parameterized(
                doses,
                gr_inf,
                np.log10(gec50),
                2.0,
            )
            for replicate, offset in ((1, -0.008), (2, 0.008)):
                for dose, value in zip(doses, base_response):
                    rows.append(
                        {
                            "analysis_key": "day4",
                            "endpoint_day": 4.0,
                            "endpoint_hours": 96.0,
                            "condition": "Doxorubicin alone",
                            "cyclophosphamide": False,
                            "ploidy": ploidy,
                            "replicate": replicate,
                            "doxorubicin_nm": dose,
                            "gr_value": value if dose == 0 else value + offset,
                        }
                    )
        response = pd.DataFrame(rows)
        for ploidy in ("2N", "4N"):
            fits.append(
                DOSE_RESPONSE.fit_gr_curve(
                    response[response["ploidy"] == ploidy],
                    condition="Doxorubicin alone",
                    cyclophosphamide=False,
                    ploidy=ploidy,
                    analysis_key="day4",
                    endpoint_day=4.0,
                    endpoint_hours=96.0,
                )
            )

        bands, intervals, delta = DOSE_RESPONSE.bootstrap_gr_curves(
            response,
            nominal_fits=fits,
            iterations=20,
            seed=17,
        )

        self.assertEqual(set(bands["series"]), {"2N", "4N", "delta"})
        self.assertEqual(set(intervals["ploidy"]), {"2N", "4N"})
        self.assertGreater(float(delta.iloc[0]["mean_delta_gr_log_dose"]), 0.0)
        self.assertEqual(int(delta.iloc[0]["bootstrap_successes"]), 20)

    def test_excess_lethal_fraction_uses_matched_control(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "analysis_key": "day4",
                    "endpoint_day": 4.0,
                    "endpoint_hours": 96.0,
                    "well": "A2",
                    "plate_row": "A",
                    "doxorubicin_nm": 0.0,
                    "ploidy": "2N",
                    "cyclophosphamide": False,
                    "replicate": 1,
                    "live_count": 90,
                    "dead_count": 10,
                    "transitional_count": 0,
                    "uncertain_count": 0,
                },
                {
                    "analysis_key": "day4",
                    "endpoint_day": 4.0,
                    "endpoint_hours": 96.0,
                    "well": "A3",
                    "plate_row": "A",
                    "doxorubicin_nm": 25.0,
                    "ploidy": "2N",
                    "cyclophosphamide": False,
                    "replicate": 1,
                    "live_count": 40,
                    "dead_count": 10,
                    "transitional_count": 5,
                    "uncertain_count": 5,
                },
            ]
        )

        response = DOSE_RESPONSE.normalize_death_to_matched_control(rows).set_index("well")

        self.assertAlmostEqual(float(response.loc["A3", "lethal_fraction"]), 0.2)
        self.assertAlmostEqual(float(response.loc["A3", "lethal_fraction_lower"]), 1 / 6)
        self.assertAlmostEqual(float(response.loc["A3", "lethal_fraction_upper"]), 1 / 3)
        self.assertAlmostEqual(
            float(response.loc["A3", "excess_lethal_fraction"]),
            1.0 - 0.8 / 0.9,
        )
        self.assertEqual(response.loc["A3", "control_well"], "A2")

    def test_death_fit_recovers_known_fixed_zero_parameters(self) -> None:
        doses = np.array([0, 3.125, 6.25, 12.5, 25, 50, 100, 200, 400, 800], dtype=float)
        expected_lf_inf = 0.72
        expected_lec50 = 48.0
        expected_slope = 2.2
        base_response = DOSE_RESPONSE.lethal_fraction_model_parameterized(
            doses,
            expected_lf_inf,
            np.log10(expected_lec50),
            expected_slope,
        )
        rows = []
        for replicate, offset in ((1, -0.005), (2, 0.005)):
            for dose, value in zip(doses, base_response):
                rows.append(
                    {
                        "well": f"R{replicate}_{dose:g}",
                        "doxorubicin_nm": dose,
                        "excess_lethal_fraction": value if dose == 0 else value + offset,
                    }
                )
        data = pd.DataFrame(rows)

        fit = DOSE_RESPONSE.fit_death_curve(
            data,
            condition="synthetic",
            cyclophosphamide=False,
            ploidy="2N",
            analysis_key="day4",
            endpoint_day=4.0,
            endpoint_hours=96.0,
        )

        self.assertAlmostEqual(fit.lf_inf, expected_lf_inf, delta=0.02)
        self.assertAlmostEqual(fit.lec50_nm, expected_lec50, delta=1.0)
        self.assertAlmostEqual(fit.hill_slope, expected_slope, delta=0.1)
        self.assertGreater(fit.r_squared, 0.999)

    def test_paired_death_difference_is_four_n_minus_two_n(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "doxorubicin_nm": 25.0,
                    "replicate": 1,
                    "ploidy": "2N",
                    "excess_lethal_fraction": 0.1,
                },
                {
                    "doxorubicin_nm": 25.0,
                    "replicate": 1,
                    "ploidy": "4N",
                    "excess_lethal_fraction": 0.3,
                },
                {
                    "doxorubicin_nm": 25.0,
                    "replicate": 2,
                    "ploidy": "2N",
                    "excess_lethal_fraction": 0.2,
                },
                {
                    "doxorubicin_nm": 25.0,
                    "replicate": 2,
                    "ploidy": "4N",
                    "excess_lethal_fraction": 0.15,
                },
            ]
        )

        paired = DOSE_RESPONSE.paired_death_differences(rows).sort_values("replicate")

        np.testing.assert_allclose(paired["delta_excess_lf"], [0.2, -0.05])

    def test_death_bootstrap_returns_curve_bands_and_delta_interval(self) -> None:
        doses = np.array([0, 3.125, 6.25, 12.5, 25, 50, 100, 200, 400, 800], dtype=float)
        rows = []
        fits = []
        for ploidy, lf_inf, lec50 in (("2N", 0.65, 45.0), ("4N", 0.78, 35.0)):
            base_response = DOSE_RESPONSE.lethal_fraction_model_parameterized(
                doses,
                lf_inf,
                np.log10(lec50),
                2.0,
            )
            for replicate, offset in ((1, -0.006), (2, 0.006)):
                for dose, value in zip(doses, base_response):
                    rows.append(
                        {
                            "analysis_key": "day4",
                            "endpoint_day": 4.0,
                            "endpoint_hours": 96.0,
                            "condition": "Doxorubicin alone",
                            "cyclophosphamide": False,
                            "ploidy": ploidy,
                            "replicate": replicate,
                            "doxorubicin_nm": dose,
                            "excess_lethal_fraction": value if dose == 0 else value + offset,
                        }
                    )
        response = pd.DataFrame(rows)
        for ploidy in ("2N", "4N"):
            fits.append(
                DOSE_RESPONSE.fit_death_curve(
                    response[response["ploidy"] == ploidy],
                    condition="Doxorubicin alone",
                    cyclophosphamide=False,
                    ploidy=ploidy,
                    analysis_key="day4",
                    endpoint_day=4.0,
                    endpoint_hours=96.0,
                )
            )

        bands, intervals, delta = DOSE_RESPONSE.bootstrap_death_curves(
            response,
            nominal_fits=fits,
            iterations=20,
            seed=18,
        )

        self.assertEqual(set(bands["series"]), {"2N", "4N", "delta"})
        self.assertEqual(set(intervals["ploidy"]), {"2N", "4N"})
        self.assertGreater(
            float(delta.iloc[0]["mean_delta_excess_lf_log_dose"]),
            0.0,
        )
        self.assertEqual(int(delta.iloc[0]["bootstrap_successes"]), 20)

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
