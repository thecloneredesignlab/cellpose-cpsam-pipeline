#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import t as student_t


REQUIRED_COLUMNS = {
    "well",
    "elapsed_hours",
    "live_count",
    "dead_count",
    "plate_row",
    "plate_column",
    "doxorubicin_nm",
    "ploidy",
    "cyclophosphamide",
    "replicate",
}
PLOIDY_STYLES = {
    "2N": {"color": "#007C83", "marker": "o"},
    "4N": {"color": "#A33A65", "marker": "s"},
}
CONDITION_NAMES = {
    False: "Doxorubicin alone",
    True: "Doxorubicin + cyclophosphamide",
}
CONDITION_FILE_STEMS = {
    False: "doxorubicin_alone_2N_vs_4N_hill",
    True: "doxorubicin_plus_cyclophosphamide_2N_vs_4N_hill",
}
GR_CONDITION_FILE_STEMS = {
    False: "doxorubicin_alone_2N_vs_4N_gr",
    True: "doxorubicin_plus_cyclophosphamide_2N_vs_4N_gr",
}
DEATH_CONDITION_FILE_STEMS = {
    False: "doxorubicin_alone_2N_vs_4N_excess_lethal_fraction",
    True: "doxorubicin_plus_cyclophosphamide_2N_vs_4N_excess_lethal_fraction",
}


@dataclass(frozen=True)
class ResolvedInput:
    csv_path: Path
    run_dir: Path | None


@dataclass(frozen=True)
class HillFit:
    condition: str
    cyclophosphamide: bool
    ploidy: str
    n_observations: int
    n_doses: int
    bottom: float
    bottom_ci_low: float
    bottom_ci_high: float
    top: float
    top_ci_low: float
    top_ci_high: float
    ec50_nm: float
    ec50_ci_low_nm: float
    ec50_ci_high_nm: float
    hill_slope: float
    hill_slope_ci_low: float
    hill_slope_ci_high: float
    r_squared: float


@dataclass(frozen=True)
class GRFit:
    analysis_key: str
    endpoint_day: float | None
    endpoint_hours: float
    condition: str
    cyclophosphamide: bool
    ploidy: str
    n_observations: int
    n_doses: int
    gr_inf: float
    gec50_nm: float
    gr50_nm: float
    hill_slope: float
    gr_max: float
    highest_dose_nm: float
    r_squared: float


@dataclass(frozen=True)
class DeathFit:
    analysis_key: str
    endpoint_day: float | None
    endpoint_hours: float
    condition: str
    cyclophosphamide: bool
    ploidy: str
    n_observations: int
    n_doses: int
    lf_inf: float
    lec50_nm: float
    lf50_nm: float
    hill_slope: float
    lf_max: float
    highest_dose_nm: float
    r_squared: float


@dataclass(frozen=True)
class AnalysisSpec:
    output_key: str
    metric: str
    start_hours: float
    endpoint_hours: float
    endpoint_window_hours: float
    endpoint_day: float | None


@dataclass(frozen=True)
class AnalysisResult:
    spec: AnalysisSpec
    response: pd.DataFrame
    start_hours: float
    endpoint_hours: float


def format_number(value: float) -> str:
    return f"{value:g}"


def day_output_key(day: float) -> str:
    text = format_number(day).replace(".", "p")
    return f"day{text}"


def build_analysis_specs(
    data: pd.DataFrame,
    primary_metric: str,
    start_hours: float | None,
    endpoint_hours: float | None,
    endpoint_window_hours: float,
    additional_endpoint_days: list[float],
) -> list[AnalysisSpec]:
    observed_start = float(data["elapsed_hours"].min())
    observed_endpoint = float(data["elapsed_hours"].max())
    start = observed_start if start_hours is None else float(start_hours)
    primary_endpoint = observed_endpoint if endpoint_hours is None else float(endpoint_hours)
    specs = [
        AnalysisSpec(
            output_key=primary_metric,
            metric=primary_metric,
            start_hours=start,
            endpoint_hours=primary_endpoint,
            endpoint_window_hours=endpoint_window_hours if primary_metric == "endpoint" else 0.0,
            endpoint_day=None,
        )
    ]
    seen = {(primary_metric, primary_endpoint, specs[0].endpoint_window_hours)}
    for day in additional_endpoint_days:
        day = float(day)
        if day <= 0:
            raise SystemExit(f"Additional endpoint days must be positive, found {day:g}")
        endpoint = day * 24.0
        identity = ("endpoint", endpoint, 0.0)
        if identity in seen:
            continue
        specs.append(
            AnalysisSpec(
                output_key=day_output_key(day),
                metric="endpoint",
                start_hours=start,
                endpoint_hours=endpoint,
                endpoint_window_hours=0.0,
                endpoint_day=day,
            )
        )
        seen.add(identity)
    return specs


def branch_directory(branch: str) -> str:
    return branch


def resolve_well_time_csv(input_path: Path, branch: str) -> ResolvedInput:
    if input_path.is_file():
        return ResolvedInput(csv_path=input_path.resolve(), run_dir=None)
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    filename = "well_time_cell_state_counts.csv"
    direct_candidates = [
        input_path / filename,
        input_path / "analysis" / "well_count_timecourses" / branch_directory(branch) / filename,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            run_dir = input_path if candidate != input_path / filename else None
            return ResolvedInput(csv_path=candidate.resolve(), run_dir=run_dir)

    nested_candidates: list[Path] = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("matplotlib_"):
                continue
            candidate = (
                Path(entry.path)
                / "analysis"
                / "well_count_timecourses"
                / branch_directory(branch)
                / filename
            )
            if candidate.is_file():
                nested_candidates.append(candidate)
    if len(nested_candidates) == 1:
        return ResolvedInput(csv_path=nested_candidates[0].resolve(), run_dir=nested_candidates[0].parents[3])
    if len(nested_candidates) > 1:
        joined = "\n".join(str(path) for path in nested_candidates)
        raise SystemExit(f"Multiple well-time tables found. Pass a run directory or CSV explicitly:\n{joined}")
    raise SystemExit(
        f"No {filename} found for branch {branch}. Run 04_plot_well_counts_over_time.py first."
    )


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "with"}:
        return True
    if text in {"0", "false", "no", "without"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_well_time_table(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise SystemExit(f"Well-time table is missing required columns: {', '.join(missing)}")
    data = data.copy()
    data["well"] = data["well"].astype(str)
    data["plate_row"] = data["plate_row"].astype(str)
    data["ploidy"] = data["ploidy"].astype(str)
    data["cyclophosphamide"] = data["cyclophosphamide"].map(parse_bool)
    for column in ("transitional_count", "uncertain_count"):
        if column not in data:
            data[column] = 0
    numeric_columns = (
        "elapsed_hours",
        "live_count",
        "dead_count",
        "transitional_count",
        "uncertain_count",
        "plate_column",
        "doxorubicin_nm",
        "replicate",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    state_count_columns = ["live_count", "dead_count", "transitional_count", "uncertain_count"]
    if (data[state_count_columns] < 0).any().any():
        raise SystemExit("Well-time table contains negative cell-state counts")
    if data.duplicated(["well", "elapsed_hours"]).any():
        duplicates = data.loc[
            data.duplicated(["well", "elapsed_hours"], keep=False),
            ["well", "elapsed_hours"],
        ].head(10)
        raise SystemExit(f"Duplicate well/time rows found:\n{duplicates.to_string(index=False)}")
    unexpected_ploidy = sorted(set(data["ploidy"]) - set(PLOIDY_STYLES))
    if unexpected_ploidy:
        raise SystemExit(f"Unexpected ploidy labels: {', '.join(unexpected_ploidy)}")

    metadata_columns = [
        "plate_row",
        "plate_column",
        "doxorubicin_nm",
        "ploidy",
        "cyclophosphamide",
        "replicate",
    ]
    inconsistent = [
        well
        for well, rows in data.groupby("well")
        if any(rows[column].nunique(dropna=False) != 1 for column in metadata_columns)
    ]
    if inconsistent:
        raise SystemExit(f"Treatment metadata changes over time for wells: {', '.join(inconsistent)}")
    return data.sort_values(["well", "elapsed_hours"]).reset_index(drop=True)


def select_time_range(
    data: pd.DataFrame,
    start_hours: float | None,
    endpoint_hours: float | None,
) -> tuple[pd.DataFrame, float, float]:
    observed_min = float(data["elapsed_hours"].min())
    observed_max = float(data["elapsed_hours"].max())
    start = observed_min if start_hours is None else float(start_hours)
    endpoint = observed_max if endpoint_hours is None else float(endpoint_hours)
    if endpoint <= start:
        raise SystemExit(f"Endpoint ({endpoint:g} h) must be after start ({start:g} h)")
    tolerance = 1e-8
    if start < observed_min - tolerance or endpoint > observed_max + tolerance:
        raise SystemExit(
            f"Requested range {start:g}-{endpoint:g} h is outside observed range "
            f"{observed_min:g}-{observed_max:g} h"
        )
    selected = data[
        (data["elapsed_hours"] >= start - tolerance)
        & (data["elapsed_hours"] <= endpoint + tolerance)
    ].copy()
    for boundary, label in ((start, "start"), (endpoint, "endpoint")):
        missing = [
            well
            for well, rows in selected.groupby("well")
            if not np.isclose(rows["elapsed_hours"].to_numpy(), boundary).any()
        ]
        if missing:
            raise SystemExit(f"{label.capitalize()} time {boundary:g} h is missing for wells: {', '.join(missing[:10])}")
    return selected, start, endpoint


def extract_raw_responses(
    data: pd.DataFrame,
    metric: str,
    start_hours: float,
    endpoint_hours: float,
    endpoint_window_hours: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metadata_columns = [
        "well",
        "plate_row",
        "plate_column",
        "doxorubicin_nm",
        "ploidy",
        "cyclophosphamide",
        "replicate",
    ]
    for well, rows in data.groupby("well"):
        rows = rows.sort_values("elapsed_hours")
        baseline_rows = rows[np.isclose(rows["elapsed_hours"], start_hours)]
        baseline_live = float(baseline_rows.iloc[0]["live_count"])
        if baseline_live <= 0:
            raise SystemExit(f"Baseline live-cell count must be positive for well {well}")
        normalized_live = rows["live_count"].to_numpy(dtype=float) / baseline_live
        times = rows["elapsed_hours"].to_numpy(dtype=float)
        if metric == "auc":
            raw_response = float(np.trapz(normalized_live, times))
        else:
            window_start = endpoint_hours - endpoint_window_hours
            endpoint_values = normalized_live[times >= window_start - 1e-8]
            if endpoint_values.size == 0:
                raise SystemExit(f"No endpoint observations found for well {well}")
            raw_response = float(endpoint_values.mean())
        metadata = rows.iloc[0][metadata_columns].to_dict()
        records.append(
            {
                **metadata,
                "baseline_live_count": baseline_live,
                "raw_response": raw_response,
                "metric": metric,
                "start_hours": start_hours,
                "endpoint_hours": endpoint_hours,
                "endpoint_window_hours": endpoint_window_hours if metric == "endpoint" else 0.0,
            }
        )
    return pd.DataFrame(records)


def normalize_to_paired_vehicle(raw_response: pd.DataFrame) -> pd.DataFrame:
    normalized = raw_response.copy()
    vehicle_by_row: dict[str, float] = {}
    for plate_row, rows in normalized.groupby("plate_row"):
        vehicle = rows[np.isclose(rows["doxorubicin_nm"], 0.0)]
        if len(vehicle) != 1:
            raise SystemExit(
                f"Expected exactly one 0 nM vehicle well in plate row {plate_row}, found {len(vehicle)}"
            )
        value = float(vehicle.iloc[0]["raw_response"])
        if value <= 0:
            raise SystemExit(f"Vehicle response must be positive in plate row {plate_row}")
        vehicle_by_row[str(plate_row)] = value
    normalized["vehicle_response"] = normalized["plate_row"].map(vehicle_by_row)
    normalized["normalized_response"] = normalized["raw_response"] / normalized["vehicle_response"]
    normalized["condition"] = normalized["cyclophosphamide"].map(CONDITION_NAMES)
    return normalized.sort_values(
        ["cyclophosphamide", "ploidy", "doxorubicin_nm", "replicate"]
    ).reset_index(drop=True)


def endpoint_gr_value(treated_fold_growth: float, vehicle_fold_growth: float) -> float:
    if treated_fold_growth < 0:
        raise ValueError("Treated fold growth must be nonnegative")
    if vehicle_fold_growth <= 1:
        raise ValueError("Vehicle fold growth must exceed 1 to calculate GR")
    if treated_fold_growth == 0:
        return -1.0
    growth_rate_ratio = math.log2(treated_fold_growth) / math.log2(vehicle_fold_growth)
    return 2.0**growth_rate_ratio - 1.0


def add_gr_values(response: pd.DataFrame) -> pd.DataFrame:
    if set(response["metric"]) != {"endpoint"}:
        raise ValueError("GR values require exact endpoint fold-growth responses")
    invalid_vehicle = response[response["vehicle_response"] <= 1.0]
    if not invalid_vehicle.empty:
        wells = ", ".join(invalid_vehicle["well"].astype(str).head(10))
        raise SystemExit(
            "GR requires matched vehicle wells to grow above baseline; "
            f"invalid vehicle growth for wells: {wells}"
        )
    invalid_treated = response[response["raw_response"] < 0.0]
    if not invalid_treated.empty:
        wells = ", ".join(invalid_treated["well"].astype(str).head(10))
        raise SystemExit(f"Negative treated fold growth found for wells: {wells}")

    gr_response = response.copy()
    gr_response["gr_value"] = [
        endpoint_gr_value(float(treated), float(vehicle))
        for treated, vehicle in zip(
            gr_response["raw_response"],
            gr_response["vehicle_response"],
        )
    ]
    return gr_response


def validate_response_design(response: pd.DataFrame, allow_incomplete: bool) -> None:
    expected_doses = frozenset(response["doxorubicin_nm"].unique())
    problems: list[str] = []
    for (cyclophosphamide, ploidy), rows in response.groupby(["cyclophosphamide", "ploidy"]):
        observed_doses = frozenset(rows["doxorubicin_nm"].unique())
        if observed_doses != expected_doses:
            problems.append(f"{CONDITION_NAMES[bool(cyclophosphamide)]}/{ploidy}: inconsistent dose set")
        counts = rows.groupby("doxorubicin_nm")["well"].nunique()
        if not (counts == 2).all():
            problems.append(
                f"{CONDITION_NAMES[bool(cyclophosphamide)]}/{ploidy}: replicate counts "
                f"{sorted(int(value) for value in counts.unique())}"
            )
    if problems and not allow_incomplete:
        raise SystemExit("Incomplete dose-response design:\n" + "\n".join(problems))
    if problems:
        print("WARNING: " + "; ".join(problems), flush=True)
    print(
        f"response_design wells={response['well'].nunique()} doses={len(expected_doses)} "
        f"conditions={response['cyclophosphamide'].nunique()} ploidies={response['ploidy'].nunique()}",
        flush=True,
    )


def summarize_responses(response: pd.DataFrame) -> pd.DataFrame:
    summary = (
        response.groupby(
            ["condition", "cyclophosphamide", "ploidy", "doxorubicin_nm"],
            as_index=False,
        )
        .agg(
            n_wells=("well", "nunique"),
            mean_response=("normalized_response", "mean"),
            std_response=("normalized_response", "std"),
            min_response=("normalized_response", "min"),
            max_response=("normalized_response", "max"),
        )
        .sort_values(["cyclophosphamide", "ploidy", "doxorubicin_nm"])
        .reset_index(drop=True)
    )
    summary["sem_response"] = summary["std_response"] / np.sqrt(summary["n_wells"])
    return summary


def hill_model_parameterized(
    dose_nm: np.ndarray | float,
    bottom: float,
    amplitude: float,
    log10_ec50: float,
    hill_slope: float,
) -> np.ndarray:
    dose = np.asarray(dose_nm, dtype=float)
    ec50 = 10.0**log10_ec50
    return bottom + amplitude / (1.0 + (dose / ec50) ** hill_slope)


def evaluate_hill(dose_nm: np.ndarray | float, fit: HillFit) -> np.ndarray:
    dose = np.asarray(dose_nm, dtype=float)
    return fit.bottom + (fit.top - fit.bottom) / (1.0 + (dose / fit.ec50_nm) ** fit.hill_slope)


def fit_hill_curve(rows: pd.DataFrame, condition: str, cyclophosphamide: bool, ploidy: str) -> HillFit:
    doses = rows["doxorubicin_nm"].to_numpy(dtype=float)
    responses = rows["normalized_response"].to_numpy(dtype=float)
    positive_doses = doses[doses > 0]
    if len(np.unique(doses)) < 5 or positive_doses.size == 0:
        raise RuntimeError(f"At least five doses including a positive dose are required for {condition}/{ploidy}")

    vehicle_values = responses[np.isclose(doses, 0.0)]
    top_initial = float(np.median(vehicle_values)) if vehicle_values.size else float(np.max(responses))
    high_dose_cutoff = float(np.quantile(positive_doses, 0.7))
    bottom_initial = max(0.0, float(np.median(responses[doses >= high_dose_cutoff])))
    amplitude_initial = max(0.1, top_initial - bottom_initial)
    half_response = bottom_initial + amplitude_initial / 2.0
    positive_rows = np.flatnonzero(doses > 0)
    closest = positive_rows[np.argmin(np.abs(responses[positive_rows] - half_response))]
    ec50_initial = float(doses[closest])

    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())
    response_upper = max(2.0, float(responses.max()) + 0.5)
    lower_bounds = [0.0, 0.01, math.log10(minimum_positive / 100.0), 0.05]
    upper_bounds = [response_upper, response_upper + 1.0, math.log10(maximum_dose * 100.0), 10.0]
    parameters, covariance = curve_fit(
        hill_model_parameterized,
        doses,
        responses,
        p0=[bottom_initial, amplitude_initial, math.log10(ec50_initial), 1.5],
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )
    bottom, amplitude, log10_ec50, hill_slope = (float(value) for value in parameters)
    top = bottom + amplitude
    predictions = hill_model_parameterized(doses, *parameters)
    residual_sum_squares = float(np.sum((responses - predictions) ** 2))
    total_sum_squares = float(np.sum((responses - responses.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")

    degrees_freedom = max(1, len(responses) - len(parameters))
    critical_value = float(student_t.ppf(0.975, degrees_freedom))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    bottom_se, _, log10_ec50_se, hill_slope_se = (float(value) for value in standard_errors)
    top_variance = float(covariance[0, 0] + covariance[1, 1] + 2.0 * covariance[0, 1])
    top_se = math.sqrt(max(0.0, top_variance))
    ec50 = 10.0**log10_ec50
    ec50_ci_low = 10.0 ** (log10_ec50 - critical_value * log10_ec50_se)
    ec50_ci_high = 10.0 ** (log10_ec50 + critical_value * log10_ec50_se)

    return HillFit(
        condition=condition,
        cyclophosphamide=cyclophosphamide,
        ploidy=ploidy,
        n_observations=len(rows),
        n_doses=rows["doxorubicin_nm"].nunique(),
        bottom=bottom,
        bottom_ci_low=max(0.0, bottom - critical_value * bottom_se),
        bottom_ci_high=bottom + critical_value * bottom_se,
        top=top,
        top_ci_low=top - critical_value * top_se,
        top_ci_high=top + critical_value * top_se,
        ec50_nm=ec50,
        ec50_ci_low_nm=ec50_ci_low,
        ec50_ci_high_nm=ec50_ci_high,
        hill_slope=hill_slope,
        hill_slope_ci_low=max(0.0, hill_slope - critical_value * hill_slope_se),
        hill_slope_ci_high=hill_slope + critical_value * hill_slope_se,
        r_squared=r_squared,
    )


def fit_all_curves(response: pd.DataFrame) -> tuple[list[HillFit], pd.DataFrame]:
    fits: list[HillFit] = []
    for cyclophosphamide in (False, True):
        condition = CONDITION_NAMES[cyclophosphamide]
        for ploidy in ("2N", "4N"):
            rows = response[
                (response["cyclophosphamide"] == cyclophosphamide)
                & (response["ploidy"] == ploidy)
            ]
            if rows.empty:
                raise SystemExit(f"No response rows found for {condition}/{ploidy}")
            fits.append(fit_hill_curve(rows, condition, cyclophosphamide, ploidy))
    fit_table = pd.DataFrame([asdict(fit) for fit in fits])
    return fits, fit_table


def summarize_gr_responses(response: pd.DataFrame) -> pd.DataFrame:
    summary = (
        response.groupby(
            [
                "analysis_key",
                "endpoint_day",
                "endpoint_hours",
                "condition",
                "cyclophosphamide",
                "ploidy",
                "doxorubicin_nm",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_wells=("well", "nunique"),
            mean_gr=("gr_value", "mean"),
            std_gr=("gr_value", "std"),
            min_gr=("gr_value", "min"),
            max_gr=("gr_value", "max"),
        )
        .sort_values(["analysis_key", "cyclophosphamide", "ploidy", "doxorubicin_nm"])
        .reset_index(drop=True)
    )
    summary["sem_gr"] = summary["std_gr"] / np.sqrt(summary["n_wells"])
    return summary


def gr_model_parameterized(
    dose_nm: np.ndarray | float,
    gr_inf: float,
    log10_gec50: float,
    hill_slope: float,
) -> np.ndarray:
    dose = np.asarray(dose_nm, dtype=float)
    gec50 = 10.0**log10_gec50
    return gr_inf + (1.0 - gr_inf) / (1.0 + (dose / gec50) ** hill_slope)


def gr50_from_parameters(gr_inf: float, gec50_nm: float, hill_slope: float) -> float:
    if gr_inf >= 0.5:
        return float("nan")
    ratio = 0.5 / (0.5 - gr_inf)
    return gec50_nm * ratio ** (1.0 / hill_slope)


def evaluate_gr_fit(dose_nm: np.ndarray | float, fit: GRFit) -> np.ndarray:
    return gr_model_parameterized(
        dose_nm,
        fit.gr_inf,
        math.log10(fit.gec50_nm),
        fit.hill_slope,
    )


def fit_gr_curve(
    rows: pd.DataFrame,
    condition: str,
    cyclophosphamide: bool,
    ploidy: str,
    analysis_key: str,
    endpoint_day: float | None,
    endpoint_hours: float,
) -> GRFit:
    doses = rows["doxorubicin_nm"].to_numpy(dtype=float)
    responses = rows["gr_value"].to_numpy(dtype=float)
    positive_doses = doses[doses > 0]
    if len(np.unique(doses)) < 5 or positive_doses.size == 0:
        raise RuntimeError(f"At least five doses including a positive dose are required for {condition}/{ploidy}")

    high_dose_cutoff = float(np.quantile(positive_doses, 0.7))
    gr_inf_initial = float(np.clip(np.median(responses[doses >= high_dose_cutoff]), -0.95, 0.95))
    half_response = gr_inf_initial + (1.0 - gr_inf_initial) / 2.0
    positive_rows = np.flatnonzero(doses > 0)
    closest = positive_rows[np.argmin(np.abs(responses[positive_rows] - half_response))]
    gec50_initial = float(doses[closest])
    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        parameters, _ = curve_fit(
            gr_model_parameterized,
            doses,
            responses,
            p0=[gr_inf_initial, math.log10(gec50_initial), 1.5],
            bounds=(
                [-1.0, math.log10(minimum_positive / 100.0), 0.05],
                [1.0, math.log10(maximum_dose * 100.0), 10.0],
            ),
            maxfev=100000,
        )
    gr_inf, log10_gec50, hill_slope = (float(value) for value in parameters)
    gec50_nm = 10.0**log10_gec50
    predictions = gr_model_parameterized(doses, *parameters)
    residual_sum_squares = float(np.sum((responses - predictions) ** 2))
    total_sum_squares = float(np.sum((responses - responses.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")
    highest_dose = float(doses.max())
    gr_max = float(rows.loc[np.isclose(rows["doxorubicin_nm"], highest_dose), "gr_value"].mean())

    return GRFit(
        analysis_key=analysis_key,
        endpoint_day=endpoint_day,
        endpoint_hours=endpoint_hours,
        condition=condition,
        cyclophosphamide=cyclophosphamide,
        ploidy=ploidy,
        n_observations=len(rows),
        n_doses=rows["doxorubicin_nm"].nunique(),
        gr_inf=gr_inf,
        gec50_nm=gec50_nm,
        gr50_nm=gr50_from_parameters(gr_inf, gec50_nm, hill_slope),
        hill_slope=hill_slope,
        gr_max=gr_max,
        highest_dose_nm=highest_dose,
        r_squared=r_squared,
    )


def fit_all_gr_curves(response: pd.DataFrame) -> tuple[list[GRFit], pd.DataFrame]:
    fits: list[GRFit] = []
    analysis_keys = response["analysis_key"].drop_duplicates().tolist()
    for analysis_key in analysis_keys:
        analysis_rows = response[response["analysis_key"] == analysis_key]
        endpoint_day_value = analysis_rows.iloc[0]["endpoint_day"]
        endpoint_day = None if pd.isna(endpoint_day_value) else float(endpoint_day_value)
        endpoint_hours = float(analysis_rows.iloc[0]["endpoint_hours"])
        for cyclophosphamide in (False, True):
            condition = CONDITION_NAMES[cyclophosphamide]
            for ploidy in ("2N", "4N"):
                rows = analysis_rows[
                    (analysis_rows["cyclophosphamide"] == cyclophosphamide)
                    & (analysis_rows["ploidy"] == ploidy)
                ]
                if rows.empty:
                    raise SystemExit(
                        f"No GR response rows found for {analysis_key}/{condition}/{ploidy}"
                    )
                fits.append(
                    fit_gr_curve(
                        rows,
                        condition=condition,
                        cyclophosphamide=cyclophosphamide,
                        ploidy=ploidy,
                        analysis_key=str(analysis_key),
                        endpoint_day=endpoint_day,
                        endpoint_hours=endpoint_hours,
                    )
                )
    return fits, pd.DataFrame([asdict(fit) for fit in fits])


def curve_dose_grid(doses: np.ndarray, points: int = 320) -> np.ndarray:
    positive_doses = np.asarray(doses, dtype=float)
    positive_doses = positive_doses[positive_doses > 0]
    if positive_doses.size == 0:
        raise ValueError("At least one positive dose is required")
    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.0, minimum_positive, 30),
                np.geomspace(minimum_positive, maximum_dose, points),
            ]
        )
    )


def mean_delta_gr_log_dose(doses: np.ndarray, delta_gr: np.ndarray) -> float:
    doses = np.asarray(doses, dtype=float)
    delta_gr = np.asarray(delta_gr, dtype=float)
    positive = doses > 0
    log_doses = np.log10(doses[positive])
    if log_doses.size < 2:
        raise ValueError("At least two positive doses are required to integrate delta GR")
    return float(np.trapz(delta_gr[positive], log_doses) / (log_doses[-1] - log_doses[0]))


def paired_gr_differences(rows: pd.DataFrame) -> pd.DataFrame:
    paired = rows.pivot(
        index=["doxorubicin_nm", "replicate"],
        columns="ploidy",
        values="gr_value",
    )
    missing = [ploidy for ploidy in ("2N", "4N") if ploidy not in paired.columns]
    if missing or paired[["2N", "4N"]].isna().any().any():
        raise ValueError("Each dose/replicate must contain paired 2N and 4N GR values")
    paired = paired.reset_index()
    paired["delta_gr"] = paired["4N"] - paired["2N"]
    return paired


def percentile_interval(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    low, median, high = np.percentile(finite, [2.5, 50.0, 97.5])
    return float(low), float(median), float(high)


def bootstrap_gr_curves(
    response: pd.DataFrame,
    nominal_fits: list[GRFit],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if iterations <= 0:
        raise ValueError("GR bootstrap iterations must be positive")

    rng = np.random.default_rng(seed)
    fit_by_key = {
        (fit.analysis_key, fit.cyclophosphamide, fit.ploidy): fit
        for fit in nominal_fits
    }
    band_records: list[dict[str, object]] = []
    interval_records: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []

    groups = response[["analysis_key", "cyclophosphamide"]].drop_duplicates()
    for group in groups.itertuples(index=False):
        analysis_key = str(group.analysis_key)
        cyclophosphamide = bool(group.cyclophosphamide)
        rows = response[
            (response["analysis_key"] == analysis_key)
            & (response["cyclophosphamide"] == cyclophosphamide)
        ]
        condition = CONDITION_NAMES[cyclophosphamide]
        replicate_sets = {
            ploidy: set(rows.loc[rows["ploidy"] == ploidy, "replicate"].astype(int))
            for ploidy in ("2N", "4N")
        }
        if replicate_sets["2N"] != replicate_sets["4N"] or not replicate_sets["2N"]:
            raise SystemExit(
                f"GR bootstrap requires matching 2N/4N replicate IDs for {analysis_key}/{condition}"
            )
        replicates = np.array(sorted(replicate_sets["2N"]), dtype=int)
        doses = np.sort(rows["doxorubicin_nm"].unique().astype(float))
        grid = curve_dose_grid(doses)
        curve_samples: dict[str, list[np.ndarray]] = {"2N": [], "4N": [], "delta": []}
        parameter_samples: dict[str, dict[str, list[float]]] = {
            ploidy: {
                "gr_inf": [],
                "gec50_nm": [],
                "gr50_nm": [],
                "hill_slope": [],
                "gr_max": [],
            }
            for ploidy in ("2N", "4N")
        }
        delta_integrals: list[float] = []

        for _ in range(iterations):
            sampled_replicates = rng.choice(replicates, size=len(replicates), replace=True)
            sampled = pd.concat(
                [rows[rows["replicate"] == replicate] for replicate in sampled_replicates],
                ignore_index=True,
            )
            try:
                sampled_fits = {
                    ploidy: fit_gr_curve(
                        sampled[sampled["ploidy"] == ploidy],
                        condition=condition,
                        cyclophosphamide=cyclophosphamide,
                        ploidy=ploidy,
                        analysis_key=analysis_key,
                        endpoint_day=fit_by_key[
                            (analysis_key, cyclophosphamide, ploidy)
                        ].endpoint_day,
                        endpoint_hours=fit_by_key[
                            (analysis_key, cyclophosphamide, ploidy)
                        ].endpoint_hours,
                    )
                    for ploidy in ("2N", "4N")
                }
            except (RuntimeError, ValueError, FloatingPointError):
                continue

            curves = {
                ploidy: evaluate_gr_fit(grid, sampled_fits[ploidy])
                for ploidy in ("2N", "4N")
            }
            curve_samples["2N"].append(curves["2N"])
            curve_samples["4N"].append(curves["4N"])
            delta_curve = curves["4N"] - curves["2N"]
            curve_samples["delta"].append(delta_curve)
            delta_integrals.append(mean_delta_gr_log_dose(grid, delta_curve))
            for ploidy, fit in sampled_fits.items():
                for parameter in parameter_samples[ploidy]:
                    parameter_samples[ploidy][parameter].append(float(getattr(fit, parameter)))

        successes = len(delta_integrals)
        if successes < max(1, math.ceil(iterations * 0.5)):
            raise RuntimeError(
                f"Only {successes}/{iterations} GR bootstrap fits succeeded for "
                f"{analysis_key}/{condition}"
            )

        nominal_curves = {
            ploidy: evaluate_gr_fit(
                grid,
                fit_by_key[(analysis_key, cyclophosphamide, ploidy)],
            )
            for ploidy in ("2N", "4N")
        }
        nominal_curves["delta"] = nominal_curves["4N"] - nominal_curves["2N"]
        for series in ("2N", "4N", "delta"):
            samples = np.vstack(curve_samples[series])
            low, median, high = np.percentile(samples, [2.5, 50.0, 97.5], axis=0)
            for dose, estimate, boot_median, ci_low, ci_high in zip(
                grid,
                nominal_curves[series],
                median,
                low,
                high,
            ):
                band_records.append(
                    {
                        "analysis_key": analysis_key,
                        "condition": condition,
                        "cyclophosphamide": cyclophosphamide,
                        "series": series,
                        "doxorubicin_nm": dose,
                        "estimate": estimate,
                        "bootstrap_median": boot_median,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_successes": successes,
                    }
                )

        for ploidy in ("2N", "4N"):
            record: dict[str, object] = {
                "analysis_key": analysis_key,
                "condition": condition,
                "cyclophosphamide": cyclophosphamide,
                "ploidy": ploidy,
                "bootstrap_successes": successes,
            }
            for parameter, values in parameter_samples[ploidy].items():
                low, median, high = percentile_interval(values)
                record[f"{parameter}_bootstrap_median"] = median
                record[f"{parameter}_ci_low"] = low
                record[f"{parameter}_ci_high"] = high
            interval_records.append(record)

        nominal_delta = mean_delta_gr_log_dose(grid, nominal_curves["delta"])
        delta_low, delta_median, delta_high = percentile_interval(delta_integrals)
        delta_records.append(
            {
                "analysis_key": analysis_key,
                "condition": condition,
                "cyclophosphamide": cyclophosphamide,
                "mean_delta_gr_log_dose": nominal_delta,
                "mean_delta_gr_bootstrap_median": delta_median,
                "mean_delta_gr_ci_low": delta_low,
                "mean_delta_gr_ci_high": delta_high,
                "bootstrap_successes": successes,
            }
        )

    return (
        pd.DataFrame(band_records),
        pd.DataFrame(interval_records),
        pd.DataFrame(delta_records),
    )


def normalize_death_to_matched_control(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    response = endpoint_rows.copy()
    response["classified_live_dead_count"] = response["live_count"] + response["dead_count"]
    response["ambiguous_count"] = response["transitional_count"] + response["uncertain_count"]
    response["interpretable_count"] = (
        response["classified_live_dead_count"] + response["ambiguous_count"]
    )
    invalid = response[response["classified_live_dead_count"] <= 0]
    if not invalid.empty:
        wells = ", ".join(invalid["well"].astype(str).head(10))
        raise SystemExit(f"Endpoint lethal fraction requires live or dead cells for wells: {wells}")

    response["fractional_viability"] = (
        response["live_count"] / response["classified_live_dead_count"]
    )
    response["lethal_fraction"] = (
        response["dead_count"] / response["classified_live_dead_count"]
    )
    response["lethal_fraction_lower"] = (
        response["dead_count"] / response["interpretable_count"]
    )
    response["lethal_fraction_upper"] = (
        response["dead_count"] + response["ambiguous_count"]
    ) / response["interpretable_count"]

    controls = response[np.isclose(response["doxorubicin_nm"], 0.0)].copy()
    control_counts = controls.groupby(["analysis_key", "plate_row"]).size()
    invalid_controls = control_counts[control_counts != 1]
    if not invalid_controls.empty:
        details = ", ".join(
            f"{analysis}/{plate_row}={count}"
            for (analysis, plate_row), count in invalid_controls.items()
        )
        raise SystemExit(
            "Expected exactly one endpoint 0 nM control per analysis/plate row; "
            f"found {details}"
        )
    controls = controls[
        [
            "analysis_key",
            "plate_row",
            "well",
            "fractional_viability",
            "lethal_fraction",
        ]
    ].rename(
        columns={
            "well": "control_well",
            "fractional_viability": "control_fractional_viability",
            "lethal_fraction": "control_lethal_fraction",
        }
    )
    response = response.merge(
        controls,
        on=["analysis_key", "plate_row"],
        how="left",
        validate="many_to_one",
    )
    missing_controls = response[response["control_well"].isna()]
    if not missing_controls.empty:
        groups = missing_controls[["analysis_key", "plate_row"]].drop_duplicates()
        details = ", ".join(
            f"{row.analysis_key}/{row.plate_row}"
            for row in groups.itertuples(index=False)
        )
        raise SystemExit(f"Missing endpoint 0 nM controls for: {details}")
    invalid_control_fv = response[response["control_fractional_viability"] <= 0]
    if not invalid_control_fv.empty:
        wells = ", ".join(invalid_control_fv["control_well"].astype(str).drop_duplicates())
        raise SystemExit(f"Matched controls have zero fractional viability: {wells}")

    response["normalized_fractional_viability"] = (
        response["fractional_viability"] / response["control_fractional_viability"]
    )
    response["excess_lethal_fraction"] = 1.0 - response["normalized_fractional_viability"]
    response["condition"] = response["cyclophosphamide"].map(CONDITION_NAMES)
    return response.sort_values(
        ["analysis_key", "cyclophosphamide", "ploidy", "doxorubicin_nm", "replicate"]
    ).reset_index(drop=True)


def extract_endpoint_death_responses(
    data: pd.DataFrame,
    endpoint_results: list[AnalysisResult],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "well",
        "plate_row",
        "plate_column",
        "doxorubicin_nm",
        "ploidy",
        "cyclophosphamide",
        "replicate",
        "live_count",
        "dead_count",
        "transitional_count",
        "uncertain_count",
    ]
    for result in endpoint_results:
        endpoint_rows = data[
            np.isclose(data["elapsed_hours"], result.endpoint_hours)
        ][columns].copy()
        if endpoint_rows["well"].nunique() != data["well"].nunique():
            raise SystemExit(
                f"Endpoint {result.endpoint_hours:g} h is missing death counts for one or more wells"
            )
        endpoint_rows.insert(0, "analysis_key", result.spec.output_key)
        endpoint_rows.insert(1, "endpoint_day", result.spec.endpoint_day)
        endpoint_rows.insert(2, "endpoint_hours", result.endpoint_hours)
        frames.append(endpoint_rows)
    return normalize_death_to_matched_control(pd.concat(frames, ignore_index=True))


def summarize_death_responses(response: pd.DataFrame) -> pd.DataFrame:
    summary = (
        response.groupby(
            [
                "analysis_key",
                "endpoint_day",
                "endpoint_hours",
                "condition",
                "cyclophosphamide",
                "ploidy",
                "doxorubicin_nm",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_wells=("well", "nunique"),
            mean_excess_lf=("excess_lethal_fraction", "mean"),
            std_excess_lf=("excess_lethal_fraction", "std"),
            min_excess_lf=("excess_lethal_fraction", "min"),
            max_excess_lf=("excess_lethal_fraction", "max"),
            mean_raw_lf=("lethal_fraction", "mean"),
            mean_control_lf=("control_lethal_fraction", "mean"),
        )
        .sort_values(["analysis_key", "cyclophosphamide", "ploidy", "doxorubicin_nm"])
        .reset_index(drop=True)
    )
    summary["sem_excess_lf"] = summary["std_excess_lf"] / np.sqrt(summary["n_wells"])
    return summary


def lethal_fraction_model_parameterized(
    dose_nm: np.ndarray | float,
    lf_inf: float,
    log10_lec50: float,
    hill_slope: float,
) -> np.ndarray:
    dose = np.asarray(dose_nm, dtype=float)
    lec50 = 10.0**log10_lec50
    scaled = (dose / lec50) ** hill_slope
    return lf_inf * scaled / (1.0 + scaled)


def lf50_from_parameters(lf_inf: float, lec50_nm: float, hill_slope: float) -> float:
    if lf_inf <= 0.5:
        return float("nan")
    ratio = 0.5 / (lf_inf - 0.5)
    return lec50_nm * ratio ** (1.0 / hill_slope)


def evaluate_death_fit(dose_nm: np.ndarray | float, fit: DeathFit) -> np.ndarray:
    return lethal_fraction_model_parameterized(
        dose_nm,
        fit.lf_inf,
        math.log10(fit.lec50_nm),
        fit.hill_slope,
    )


def fit_death_curve(
    rows: pd.DataFrame,
    condition: str,
    cyclophosphamide: bool,
    ploidy: str,
    analysis_key: str,
    endpoint_day: float | None,
    endpoint_hours: float,
) -> DeathFit:
    doses = rows["doxorubicin_nm"].to_numpy(dtype=float)
    responses = rows["excess_lethal_fraction"].to_numpy(dtype=float)
    positive_doses = doses[doses > 0]
    if len(np.unique(doses)) < 5 or positive_doses.size == 0:
        raise RuntimeError(f"At least five doses including a positive dose are required for {condition}/{ploidy}")

    high_dose_cutoff = float(np.quantile(positive_doses, 0.7))
    lf_inf_initial = float(np.clip(np.median(responses[doses >= high_dose_cutoff]), 0.01, 0.99))
    half_response = lf_inf_initial / 2.0
    positive_rows = np.flatnonzero(doses > 0)
    closest = positive_rows[np.argmin(np.abs(responses[positive_rows] - half_response))]
    lec50_initial = float(doses[closest])
    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        parameters, _ = curve_fit(
            lethal_fraction_model_parameterized,
            doses,
            responses,
            p0=[lf_inf_initial, math.log10(lec50_initial), 1.5],
            bounds=(
                [0.0, math.log10(minimum_positive / 100.0), 0.05],
                [1.0, math.log10(maximum_dose * 100.0), 10.0],
            ),
            maxfev=100000,
        )
    lf_inf, log10_lec50, hill_slope = (float(value) for value in parameters)
    lec50_nm = 10.0**log10_lec50
    predictions = lethal_fraction_model_parameterized(doses, *parameters)
    residual_sum_squares = float(np.sum((responses - predictions) ** 2))
    total_sum_squares = float(np.sum((responses - responses.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")
    highest_dose = float(doses.max())
    lf_max = float(
        rows.loc[
            np.isclose(rows["doxorubicin_nm"], highest_dose),
            "excess_lethal_fraction",
        ].mean()
    )

    return DeathFit(
        analysis_key=analysis_key,
        endpoint_day=endpoint_day,
        endpoint_hours=endpoint_hours,
        condition=condition,
        cyclophosphamide=cyclophosphamide,
        ploidy=ploidy,
        n_observations=len(rows),
        n_doses=rows["doxorubicin_nm"].nunique(),
        lf_inf=lf_inf,
        lec50_nm=lec50_nm,
        lf50_nm=lf50_from_parameters(lf_inf, lec50_nm, hill_slope),
        hill_slope=hill_slope,
        lf_max=lf_max,
        highest_dose_nm=highest_dose,
        r_squared=r_squared,
    )


def fit_all_death_curves(response: pd.DataFrame) -> tuple[list[DeathFit], pd.DataFrame]:
    fits: list[DeathFit] = []
    analysis_keys = response["analysis_key"].drop_duplicates().tolist()
    for analysis_key in analysis_keys:
        analysis_rows = response[response["analysis_key"] == analysis_key]
        endpoint_day_value = analysis_rows.iloc[0]["endpoint_day"]
        endpoint_day = None if pd.isna(endpoint_day_value) else float(endpoint_day_value)
        endpoint_hours = float(analysis_rows.iloc[0]["endpoint_hours"])
        for cyclophosphamide in (False, True):
            condition = CONDITION_NAMES[cyclophosphamide]
            for ploidy in ("2N", "4N"):
                rows = analysis_rows[
                    (analysis_rows["cyclophosphamide"] == cyclophosphamide)
                    & (analysis_rows["ploidy"] == ploidy)
                ]
                if rows.empty:
                    raise SystemExit(
                        f"No death response rows found for {analysis_key}/{condition}/{ploidy}"
                    )
                fits.append(
                    fit_death_curve(
                        rows,
                        condition=condition,
                        cyclophosphamide=cyclophosphamide,
                        ploidy=ploidy,
                        analysis_key=str(analysis_key),
                        endpoint_day=endpoint_day,
                        endpoint_hours=endpoint_hours,
                    )
                )
    return fits, pd.DataFrame([asdict(fit) for fit in fits])


def paired_death_differences(rows: pd.DataFrame) -> pd.DataFrame:
    paired = rows.pivot(
        index=["doxorubicin_nm", "replicate"],
        columns="ploidy",
        values="excess_lethal_fraction",
    )
    missing = [ploidy for ploidy in ("2N", "4N") if ploidy not in paired.columns]
    if missing or paired[["2N", "4N"]].isna().any().any():
        raise ValueError(
            "Each dose/replicate must contain paired 2N and 4N excess lethal fractions"
        )
    paired = paired.reset_index()
    paired["delta_excess_lf"] = paired["4N"] - paired["2N"]
    return paired


def bootstrap_death_curves(
    response: pd.DataFrame,
    nominal_fits: list[DeathFit],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if iterations <= 0:
        raise ValueError("Death bootstrap iterations must be positive")

    rng = np.random.default_rng(seed)
    fit_by_key = {
        (fit.analysis_key, fit.cyclophosphamide, fit.ploidy): fit
        for fit in nominal_fits
    }
    band_records: list[dict[str, object]] = []
    interval_records: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []

    groups = response[["analysis_key", "cyclophosphamide"]].drop_duplicates()
    for group in groups.itertuples(index=False):
        analysis_key = str(group.analysis_key)
        cyclophosphamide = bool(group.cyclophosphamide)
        rows = response[
            (response["analysis_key"] == analysis_key)
            & (response["cyclophosphamide"] == cyclophosphamide)
        ]
        condition = CONDITION_NAMES[cyclophosphamide]
        replicate_sets = {
            ploidy: set(rows.loc[rows["ploidy"] == ploidy, "replicate"].astype(int))
            for ploidy in ("2N", "4N")
        }
        if replicate_sets["2N"] != replicate_sets["4N"] or not replicate_sets["2N"]:
            raise SystemExit(
                "Death bootstrap requires matching 2N/4N replicate IDs for "
                f"{analysis_key}/{condition}"
            )
        replicates = np.array(sorted(replicate_sets["2N"]), dtype=int)
        doses = np.sort(rows["doxorubicin_nm"].unique().astype(float))
        grid = curve_dose_grid(doses)
        curve_samples: dict[str, list[np.ndarray]] = {"2N": [], "4N": [], "delta": []}
        parameter_samples: dict[str, dict[str, list[float]]] = {
            ploidy: {
                "lf_inf": [],
                "lec50_nm": [],
                "lf50_nm": [],
                "hill_slope": [],
                "lf_max": [],
            }
            for ploidy in ("2N", "4N")
        }
        delta_integrals: list[float] = []

        for _ in range(iterations):
            sampled_replicates = rng.choice(replicates, size=len(replicates), replace=True)
            sampled = pd.concat(
                [rows[rows["replicate"] == replicate] for replicate in sampled_replicates],
                ignore_index=True,
            )
            try:
                sampled_fits = {
                    ploidy: fit_death_curve(
                        sampled[sampled["ploidy"] == ploidy],
                        condition=condition,
                        cyclophosphamide=cyclophosphamide,
                        ploidy=ploidy,
                        analysis_key=analysis_key,
                        endpoint_day=fit_by_key[
                            (analysis_key, cyclophosphamide, ploidy)
                        ].endpoint_day,
                        endpoint_hours=fit_by_key[
                            (analysis_key, cyclophosphamide, ploidy)
                        ].endpoint_hours,
                    )
                    for ploidy in ("2N", "4N")
                }
            except (RuntimeError, ValueError, FloatingPointError):
                continue

            curves = {
                ploidy: evaluate_death_fit(grid, sampled_fits[ploidy])
                for ploidy in ("2N", "4N")
            }
            curve_samples["2N"].append(curves["2N"])
            curve_samples["4N"].append(curves["4N"])
            delta_curve = curves["4N"] - curves["2N"]
            curve_samples["delta"].append(delta_curve)
            delta_integrals.append(mean_delta_gr_log_dose(grid, delta_curve))
            for ploidy, fit in sampled_fits.items():
                for parameter in parameter_samples[ploidy]:
                    parameter_samples[ploidy][parameter].append(float(getattr(fit, parameter)))

        successes = len(delta_integrals)
        if successes < max(1, math.ceil(iterations * 0.5)):
            raise RuntimeError(
                f"Only {successes}/{iterations} death bootstrap fits succeeded for "
                f"{analysis_key}/{condition}"
            )

        nominal_curves = {
            ploidy: evaluate_death_fit(
                grid,
                fit_by_key[(analysis_key, cyclophosphamide, ploidy)],
            )
            for ploidy in ("2N", "4N")
        }
        nominal_curves["delta"] = nominal_curves["4N"] - nominal_curves["2N"]
        for series in ("2N", "4N", "delta"):
            samples = np.vstack(curve_samples[series])
            low, median, high = np.percentile(samples, [2.5, 50.0, 97.5], axis=0)
            for dose, estimate, boot_median, ci_low, ci_high in zip(
                grid,
                nominal_curves[series],
                median,
                low,
                high,
            ):
                band_records.append(
                    {
                        "analysis_key": analysis_key,
                        "condition": condition,
                        "cyclophosphamide": cyclophosphamide,
                        "series": series,
                        "doxorubicin_nm": dose,
                        "estimate": estimate,
                        "bootstrap_median": boot_median,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_successes": successes,
                    }
                )

        for ploidy in ("2N", "4N"):
            record: dict[str, object] = {
                "analysis_key": analysis_key,
                "condition": condition,
                "cyclophosphamide": cyclophosphamide,
                "ploidy": ploidy,
                "bootstrap_successes": successes,
            }
            for parameter, values in parameter_samples[ploidy].items():
                low, median, high = percentile_interval(values)
                record[f"{parameter}_bootstrap_median"] = median
                record[f"{parameter}_ci_low"] = low
                record[f"{parameter}_ci_high"] = high
            interval_records.append(record)

        nominal_delta = mean_delta_gr_log_dose(grid, nominal_curves["delta"])
        delta_low, delta_median, delta_high = percentile_interval(delta_integrals)
        delta_records.append(
            {
                "analysis_key": analysis_key,
                "condition": condition,
                "cyclophosphamide": cyclophosphamide,
                "mean_delta_excess_lf_log_dose": nominal_delta,
                "mean_delta_excess_lf_bootstrap_median": delta_median,
                "mean_delta_excess_lf_ci_low": delta_low,
                "mean_delta_excess_lf_ci_high": delta_high,
                "bootstrap_successes": successes,
            }
        )

    return (
        pd.DataFrame(band_records),
        pd.DataFrame(interval_records),
        pd.DataFrame(delta_records),
    )


def metric_description(
    metric: str,
    start_hours: float,
    endpoint_hours: float,
    window_hours: float,
    endpoint_day: float | None,
) -> str:
    if metric == "auc":
        return (
            f"Live-cell AUC from {start_hours:g}-{endpoint_hours:g} h; each trajectory divided by its "
            "baseline, then by its paired 0 nM control"
        )
    if endpoint_day is not None and window_hours == 0:
        return (
            f"Day {format_number(endpoint_day)} ({endpoint_hours:g} h) live-cell response; "
            "divided by baseline and paired 0 nM control"
        )
    window = f"mean over the final {window_hours:g} h" if window_hours > 0 else f"at {endpoint_hours:g} h"
    return f"Live-cell growth {window}; divided by baseline and paired 0 nM control"


def fit_annotation(fit: HillFit) -> str:
    return (
        f"{fit.ploidy}: EC50 {fit.ec50_nm:.1f} nM "
        f"[{fit.ec50_ci_low_nm:.1f}, {fit.ec50_ci_high_nm:.1f}]\n"
        f"Hill {fit.hill_slope:.2f} "
        f"[{fit.hill_slope_ci_low:.2f}, {fit.hill_slope_ci_high:.2f}], "
        f"R2 {fit.r_squared:.3f}"
    )


def plot_condition(
    response: pd.DataFrame,
    summary: pd.DataFrame,
    fits: list[HillFit],
    cyclophosphamide: bool,
    metric: str,
    start_hours: float,
    endpoint_hours: float,
    endpoint_window_hours: float,
    endpoint_day: float | None,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    condition = CONDITION_NAMES[cyclophosphamide]
    condition_response = response[response["cyclophosphamide"] == cyclophosphamide]
    condition_summary = summary[summary["cyclophosphamide"] == cyclophosphamide]
    condition_fits = {fit.ploidy: fit for fit in fits if fit.cyclophosphamide == cyclophosphamide}
    doses = np.sort(condition_response["doxorubicin_nm"].unique().astype(float))
    positive_doses = doses[doses > 0]
    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())
    curve_doses = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, minimum_positive, 40),
                np.geomspace(minimum_positive, maximum_dose, 360),
            ]
        )
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for ploidy in ("2N", "4N"):
        style = PLOIDY_STYLES[ploidy]
        points = condition_response[condition_response["ploidy"] == ploidy]
        means = condition_summary[condition_summary["ploidy"] == ploidy].sort_values("doxorubicin_nm")
        fit = condition_fits[ploidy]
        ax.scatter(
            points["doxorubicin_nm"],
            points["normalized_response"],
            s=27,
            marker=style["marker"],
            facecolors=style["color"],
            edgecolors="white",
            linewidths=0.55,
            alpha=0.62,
            zorder=3,
        )
        lower = means["mean_response"] - means["min_response"]
        upper = means["max_response"] - means["mean_response"]
        ax.errorbar(
            means["doxorubicin_nm"],
            means["mean_response"],
            yerr=np.vstack([lower, upper]),
            fmt=style["marker"],
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.4,
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
        ax.plot(
            curve_doses,
            evaluate_hill(curve_doses, fit),
            color=style["color"],
            linewidth=2.1,
            label=f"{ploidy} Hill fit",
            zorder=2,
        )

    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
    ax.set_xscale("symlog", base=2, linthresh=minimum_positive, linscale=0.75)
    ax.set_xlim(-minimum_positive * 0.18, maximum_dose * 1.18)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{dose:g}" for dose in doses])
    response_maximum = float(condition_response["normalized_response"].max())
    ax.set_ylim(-0.03, max(1.2, response_maximum * 1.12))
    ax.set_xlabel("Doxorubicin concentration (nM)", fontsize=10)
    ylabel = "Relative live-cell AUC" if metric == "auc" else "Relative live-cell growth"
    ax.set_ylabel(f"{ylabel}\n(paired vehicle = 1)", fontsize=10)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.32)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(
        metric_description(metric, start_hours, endpoint_hours, endpoint_window_hours, endpoint_day),
        fontsize=8.5,
        pad=8,
    )
    fig.suptitle(condition, fontsize=15, y=0.98)

    annotations = "\n\n".join(fit_annotation(condition_fits[ploidy]) for ploidy in ("2N", "4N"))
    ax.text(
        0.985,
        0.97,
        annotations,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        linespacing=1.2,
        bbox={"boxstyle": "square,pad=0.45", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )
    fig.text(
        0.5,
        0.018,
        "Small filled points: replicate wells (n=2/dose); open points and bars: mean and range; brackets: 95% CI",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def format_ci(
    estimate: float,
    ci_low: float,
    ci_high: float,
    digits: int = 2,
    suffix: str = "",
) -> str:
    if not math.isfinite(estimate):
        return "not reached"
    value = f"{estimate:.{digits}f}{suffix}"
    if math.isfinite(ci_low) and math.isfinite(ci_high):
        return f"{value} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"
    return value


def gr_fit_annotation(row: pd.Series) -> str:
    ploidy = str(row["ploidy"]) if "ploidy" in row else str(row.name)
    return (
        f"{ploidy}: GR50 "
        f"{format_ci(float(row['gr50_nm']), float(row['gr50_nm_ci_low']), float(row['gr50_nm_ci_high']), 1, ' nM')}\n"
        f"GRmax "
        f"{format_ci(float(row['gr_max']), float(row['gr_max_ci_low']), float(row['gr_max_ci_high']), 2)}"
    )


def endpoint_panel_title(rows: pd.DataFrame) -> str:
    endpoint_hours = float(rows.iloc[0]["endpoint_hours"])
    endpoint_day = rows.iloc[0]["endpoint_day"]
    if pd.notna(endpoint_day):
        return f"Day {format_number(float(endpoint_day))} ({endpoint_hours:g} h)"
    return f"{endpoint_hours:g} h endpoint"


def configure_dose_axis(ax: plt.Axes, doses: np.ndarray) -> None:
    positive_doses = doses[doses > 0]
    minimum_positive = float(positive_doses.min())
    maximum_dose = float(positive_doses.max())
    ax.set_xscale("symlog", base=2, linthresh=minimum_positive, linscale=0.75)
    ax.set_xlim(-minimum_positive * 0.18, maximum_dose * 1.18)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{dose:g}" for dose in doses])
    ax.grid(True, which="major", linewidth=0.45, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_gr_condition(
    response: pd.DataFrame,
    summary: pd.DataFrame,
    fits: list[GRFit],
    fit_table: pd.DataFrame,
    bands: pd.DataFrame,
    delta_summary: pd.DataFrame,
    cyclophosphamide: bool,
    out_png: Path,
    out_pdf: Path,
    bootstrap_iterations: int,
    dpi: int,
) -> None:
    condition = CONDITION_NAMES[cyclophosphamide]
    condition_response = response[response["cyclophosphamide"] == cyclophosphamide]
    analysis_keys = condition_response["analysis_key"].drop_duplicates().tolist()
    n_columns = len(analysis_keys)
    fig, axes = plt.subplots(
        2,
        n_columns,
        figsize=(max(8.2, 7.2 * n_columns), 8.4),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": [2.05, 1.0]},
    )

    condition_bands = bands[bands["cyclophosphamide"] == cyclophosphamide]
    gr_band_values = condition_bands[
        condition_bands["series"].isin(["2N", "4N"])
    ][["ci_low", "ci_high"]].to_numpy(dtype=float)
    gr_minimum = min(-1.05, float(np.nanmin(gr_band_values)))
    gr_maximum = max(
        1.12,
        float(np.nanmax(gr_band_values)),
        float(condition_response["gr_value"].max()) * 1.05,
    )
    paired_differences = [
        paired_gr_differences(
            condition_response[condition_response["analysis_key"] == analysis_key]
        )
        for analysis_key in analysis_keys
    ]
    delta_band_values = condition_bands[condition_bands["series"] == "delta"][
        ["ci_low", "ci_high"]
    ].to_numpy(dtype=float)
    maximum_delta = max(
        0.12,
        float(np.nanmax(np.abs(delta_band_values))),
        max(float(frame["delta_gr"].abs().max()) for frame in paired_differences),
    )
    delta_limit = maximum_delta * 1.18

    fit_by_key = {
        (fit.analysis_key, fit.ploidy): fit
        for fit in fits
        if fit.cyclophosphamide == cyclophosphamide
    }
    for column, analysis_key in enumerate(analysis_keys):
        top_ax = axes[0, column]
        delta_ax = axes[1, column]
        analysis_response = condition_response[
            condition_response["analysis_key"] == analysis_key
        ]
        analysis_summary = summary[
            (summary["analysis_key"] == analysis_key)
            & (summary["cyclophosphamide"] == cyclophosphamide)
        ]
        doses = np.sort(analysis_response["doxorubicin_nm"].unique().astype(float))

        for ploidy in ("2N", "4N"):
            style = PLOIDY_STYLES[ploidy]
            points = analysis_response[analysis_response["ploidy"] == ploidy]
            means = analysis_summary[
                analysis_summary["ploidy"] == ploidy
            ].sort_values("doxorubicin_nm")
            band = condition_bands[
                (condition_bands["analysis_key"] == analysis_key)
                & (condition_bands["series"] == ploidy)
            ].sort_values("doxorubicin_nm")
            fit = fit_by_key[(str(analysis_key), ploidy)]
            top_ax.fill_between(
                band["doxorubicin_nm"].to_numpy(dtype=float),
                band["ci_low"].to_numpy(dtype=float),
                band["ci_high"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
            top_ax.plot(
                band["doxorubicin_nm"],
                evaluate_gr_fit(band["doxorubicin_nm"].to_numpy(dtype=float), fit),
                color=style["color"],
                linewidth=2.0,
                label=f"{ploidy} GR fit",
                zorder=2,
            )
            top_ax.scatter(
                points["doxorubicin_nm"],
                points["gr_value"],
                s=26,
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.62,
                zorder=3,
            )
            lower = means["mean_gr"] - means["min_gr"]
            upper = means["max_gr"] - means["mean_gr"]
            top_ax.errorbar(
                means["doxorubicin_nm"],
                means["mean_gr"],
                yerr=np.vstack([lower, upper]),
                fmt=style["marker"],
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.3,
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=2.4,
                zorder=4,
            )

        top_ax.axhline(1.0, color="#686868", linewidth=0.8, linestyle="--", alpha=0.75)
        top_ax.axhline(0.0, color="#202020", linewidth=0.9, alpha=0.8)
        top_ax.set_ylim(gr_minimum, gr_maximum)
        top_ax.set_title(endpoint_panel_title(analysis_response), fontsize=11, pad=8)
        configure_dose_axis(top_ax, doses)
        if column == 0:
            top_ax.set_ylabel(
                "Growth-rate inhibition (GR)\n1 = control growth; 0 = cytostasis",
                fontsize=9.5,
            )
            top_ax.legend(loc="lower left", frameon=False, fontsize=8.5)
        else:
            top_ax.tick_params(labelleft=False)

        fit_rows = fit_table[
            (fit_table["analysis_key"] == analysis_key)
            & (fit_table["cyclophosphamide"] == cyclophosphamide)
        ].set_index("ploidy")
        annotations = "\n\n".join(
            gr_fit_annotation(fit_rows.loc[ploidy])
            for ploidy in ("2N", "4N")
        )
        top_ax.text(
            0.985,
            0.97,
            annotations,
            transform=top_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            linespacing=1.18,
            bbox={
                "boxstyle": "square,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.92,
            },
        )

        paired = paired_differences[column]
        delta_means = (
            paired.groupby("doxorubicin_nm", as_index=False)
            .agg(
                mean_delta=("delta_gr", "mean"),
                min_delta=("delta_gr", "min"),
                max_delta=("delta_gr", "max"),
            )
            .sort_values("doxorubicin_nm")
        )
        delta_band = condition_bands[
            (condition_bands["analysis_key"] == analysis_key)
            & (condition_bands["series"] == "delta")
        ].sort_values("doxorubicin_nm")
        delta_ax.fill_between(
            delta_band["doxorubicin_nm"].to_numpy(dtype=float),
            delta_band["ci_low"].to_numpy(dtype=float),
            delta_band["ci_high"].to_numpy(dtype=float),
            color="#555555",
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        delta_ax.plot(
            delta_band["doxorubicin_nm"],
            delta_band["estimate"],
            color="#303030",
            linewidth=1.8,
            zorder=2,
        )
        delta_ax.scatter(
            paired["doxorubicin_nm"],
            paired["delta_gr"],
            s=22,
            color="#666666",
            edgecolors="white",
            linewidths=0.45,
            alpha=0.66,
            zorder=3,
        )
        delta_lower = delta_means["mean_delta"] - delta_means["min_delta"]
        delta_upper = delta_means["max_delta"] - delta_means["mean_delta"]
        delta_ax.errorbar(
            delta_means["doxorubicin_nm"],
            delta_means["mean_delta"],
            yerr=np.vstack([delta_lower, delta_upper]),
            fmt="D",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor="#303030",
            markeredgewidth=1.1,
            ecolor="#505050",
            elinewidth=0.9,
            capsize=2.2,
            zorder=4,
        )
        delta_ax.axhline(0.0, color="#202020", linewidth=0.9)
        delta_ax.set_ylim(-delta_limit, delta_limit)
        configure_dose_axis(delta_ax, doses)
        delta_ax.set_xlabel("Doxorubicin concentration (nM)", fontsize=9.5)
        if column == 0:
            delta_ax.set_ylabel("Delta GR (4N - 2N)", fontsize=9.5)
        else:
            delta_ax.tick_params(labelleft=False)

        delta_row = delta_summary[
            (delta_summary["analysis_key"] == analysis_key)
            & (delta_summary["cyclophosphamide"] == cyclophosphamide)
        ].iloc[0]
        delta_ax.text(
            0.985,
            0.94,
            "Mean Delta GR across log-dose\n"
            + format_ci(
                float(delta_row["mean_delta_gr_log_dose"]),
                float(delta_row["mean_delta_gr_ci_low"]),
                float(delta_row["mean_delta_gr_ci_high"]),
                3,
            ),
            transform=delta_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.92,
            },
        )

    control_description = (
        "matched cyclophosphamide-only controls (0 nM DOX)"
        if cyclophosphamide
        else "matched untreated controls (0 nM DOX, no cyclophosphamide)"
    )
    fig.suptitle(f"GR dose response: {condition}", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.94,
        f"Each well is normalized to its baseline and {control_description}",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.022,
        (
            f"Filled points: paired replicate wells (n=2/dose); open points and bars: mean and range; "
            f"bands: 95% paired-replicate bootstrap ({bootstrap_iterations} resamples). "
            "Delta GR > 0 favors 4N; Delta GR < 0 favors 2N."
        ),
        ha="center",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.12, wspace=0.16, hspace=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def death_fit_annotation(row: pd.Series) -> str:
    ploidy = str(row["ploidy"]) if "ploidy" in row else str(row.name)
    return (
        f"{ploidy}: LF50 "
        f"{format_ci(float(row['lf50_nm']), float(row['lf50_nm_ci_low']), float(row['lf50_nm_ci_high']), 1, ' nM')}\n"
        f"LEC50 "
        f"{format_ci(float(row['lec50_nm']), float(row['lec50_nm_ci_low']), float(row['lec50_nm_ci_high']), 1, ' nM')}\n"
        f"LFmax "
        f"{format_ci(float(row['lf_max']), float(row['lf_max_ci_low']), float(row['lf_max_ci_high']), 2)}"
    )


def plot_death_condition(
    response: pd.DataFrame,
    summary: pd.DataFrame,
    fits: list[DeathFit],
    fit_table: pd.DataFrame,
    bands: pd.DataFrame,
    delta_summary: pd.DataFrame,
    cyclophosphamide: bool,
    out_png: Path,
    out_pdf: Path,
    bootstrap_iterations: int,
    dpi: int,
) -> None:
    condition = CONDITION_NAMES[cyclophosphamide]
    condition_response = response[response["cyclophosphamide"] == cyclophosphamide]
    analysis_keys = condition_response["analysis_key"].drop_duplicates().tolist()
    n_columns = len(analysis_keys)
    fig, axes = plt.subplots(
        2,
        n_columns,
        figsize=(max(8.2, 7.2 * n_columns), 8.4),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": [2.05, 1.0]},
    )

    condition_bands = bands[bands["cyclophosphamide"] == cyclophosphamide]
    lf_band_values = condition_bands[
        condition_bands["series"].isin(["2N", "4N"])
    ][["ci_low", "ci_high"]].to_numpy(dtype=float)
    lf_minimum = min(
        -0.1,
        float(np.nanmin(lf_band_values)),
        float(condition_response["excess_lethal_fraction"].min()) * 1.05,
    )
    lf_maximum = max(
        1.02,
        float(np.nanmax(lf_band_values)),
        float(condition_response["excess_lethal_fraction"].max()) * 1.05,
    )
    paired_differences = [
        paired_death_differences(
            condition_response[condition_response["analysis_key"] == analysis_key]
        )
        for analysis_key in analysis_keys
    ]
    delta_band_values = condition_bands[condition_bands["series"] == "delta"][
        ["ci_low", "ci_high"]
    ].to_numpy(dtype=float)
    maximum_delta = max(
        0.12,
        float(np.nanmax(np.abs(delta_band_values))),
        max(float(frame["delta_excess_lf"].abs().max()) for frame in paired_differences),
    )
    delta_limit = maximum_delta * 1.18

    fit_by_key = {
        (fit.analysis_key, fit.ploidy): fit
        for fit in fits
        if fit.cyclophosphamide == cyclophosphamide
    }
    for column, analysis_key in enumerate(analysis_keys):
        top_ax = axes[0, column]
        delta_ax = axes[1, column]
        analysis_response = condition_response[
            condition_response["analysis_key"] == analysis_key
        ]
        analysis_summary = summary[
            (summary["analysis_key"] == analysis_key)
            & (summary["cyclophosphamide"] == cyclophosphamide)
        ]
        doses = np.sort(analysis_response["doxorubicin_nm"].unique().astype(float))

        for ploidy in ("2N", "4N"):
            style = PLOIDY_STYLES[ploidy]
            points = analysis_response[analysis_response["ploidy"] == ploidy]
            means = analysis_summary[
                analysis_summary["ploidy"] == ploidy
            ].sort_values("doxorubicin_nm")
            band = condition_bands[
                (condition_bands["analysis_key"] == analysis_key)
                & (condition_bands["series"] == ploidy)
            ].sort_values("doxorubicin_nm")
            fit = fit_by_key[(str(analysis_key), ploidy)]
            top_ax.fill_between(
                band["doxorubicin_nm"].to_numpy(dtype=float),
                band["ci_low"].to_numpy(dtype=float),
                band["ci_high"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
            top_ax.plot(
                band["doxorubicin_nm"],
                evaluate_death_fit(
                    band["doxorubicin_nm"].to_numpy(dtype=float),
                    fit,
                ),
                color=style["color"],
                linewidth=2.0,
                label=f"{ploidy} lethal-fraction fit",
                zorder=2,
            )
            top_ax.scatter(
                points["doxorubicin_nm"],
                points["excess_lethal_fraction"],
                s=26,
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.62,
                zorder=3,
            )
            lower = means["mean_excess_lf"] - means["min_excess_lf"]
            upper = means["max_excess_lf"] - means["mean_excess_lf"]
            top_ax.errorbar(
                means["doxorubicin_nm"],
                means["mean_excess_lf"],
                yerr=np.vstack([lower, upper]),
                fmt=style["marker"],
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.3,
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=2.4,
                zorder=4,
            )

        top_ax.axhline(0.0, color="#202020", linewidth=0.9)
        top_ax.axhline(1.0, color="#686868", linewidth=0.8, linestyle="--", alpha=0.75)
        top_ax.set_ylim(lf_minimum, lf_maximum)
        top_ax.set_title(endpoint_panel_title(analysis_response), fontsize=11, pad=8)
        configure_dose_axis(top_ax, doses)
        if column == 0:
            top_ax.set_ylabel(
                "Excess lethal fraction\n0 = matched control; 1 = complete death",
                fontsize=9.5,
            )
            top_ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        else:
            top_ax.tick_params(labelleft=False)

        fit_rows = fit_table[
            (fit_table["analysis_key"] == analysis_key)
            & (fit_table["cyclophosphamide"] == cyclophosphamide)
        ].set_index("ploidy")
        annotations = "\n\n".join(
            death_fit_annotation(fit_rows.loc[ploidy])
            for ploidy in ("2N", "4N")
        )
        top_ax.text(
            0.015,
            0.97,
            annotations,
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            linespacing=1.15,
            bbox={
                "boxstyle": "square,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.92,
            },
        )

        paired = paired_differences[column]
        delta_means = (
            paired.groupby("doxorubicin_nm", as_index=False)
            .agg(
                mean_delta=("delta_excess_lf", "mean"),
                min_delta=("delta_excess_lf", "min"),
                max_delta=("delta_excess_lf", "max"),
            )
            .sort_values("doxorubicin_nm")
        )
        delta_band = condition_bands[
            (condition_bands["analysis_key"] == analysis_key)
            & (condition_bands["series"] == "delta")
        ].sort_values("doxorubicin_nm")
        delta_ax.fill_between(
            delta_band["doxorubicin_nm"].to_numpy(dtype=float),
            delta_band["ci_low"].to_numpy(dtype=float),
            delta_band["ci_high"].to_numpy(dtype=float),
            color="#555555",
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        delta_ax.plot(
            delta_band["doxorubicin_nm"],
            delta_band["estimate"],
            color="#303030",
            linewidth=1.8,
            zorder=2,
        )
        delta_ax.scatter(
            paired["doxorubicin_nm"],
            paired["delta_excess_lf"],
            s=22,
            color="#666666",
            edgecolors="white",
            linewidths=0.45,
            alpha=0.66,
            zorder=3,
        )
        delta_lower = delta_means["mean_delta"] - delta_means["min_delta"]
        delta_upper = delta_means["max_delta"] - delta_means["mean_delta"]
        delta_ax.errorbar(
            delta_means["doxorubicin_nm"],
            delta_means["mean_delta"],
            yerr=np.vstack([delta_lower, delta_upper]),
            fmt="D",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor="#303030",
            markeredgewidth=1.1,
            ecolor="#505050",
            elinewidth=0.9,
            capsize=2.2,
            zorder=4,
        )
        delta_ax.axhline(0.0, color="#202020", linewidth=0.9)
        delta_ax.set_ylim(-delta_limit, delta_limit)
        configure_dose_axis(delta_ax, doses)
        delta_ax.set_xlabel("Doxorubicin concentration (nM)", fontsize=9.5)
        if column == 0:
            delta_ax.set_ylabel("Delta excess LF (4N - 2N)", fontsize=9.5)
        else:
            delta_ax.tick_params(labelleft=False)

        delta_row = delta_summary[
            (delta_summary["analysis_key"] == analysis_key)
            & (delta_summary["cyclophosphamide"] == cyclophosphamide)
        ].iloc[0]
        delta_ax.text(
            0.985,
            0.94,
            "Mean Delta excess LF across log-dose\n"
            + format_ci(
                float(delta_row["mean_delta_excess_lf_log_dose"]),
                float(delta_row["mean_delta_excess_lf_ci_low"]),
                float(delta_row["mean_delta_excess_lf_ci_high"]),
                3,
            ),
            transform=delta_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.92,
            },
        )

    control_description = (
        "matched cyclophosphamide-only controls (0 nM DOX)"
        if cyclophosphamide
        else "matched untreated controls (0 nM DOX, no cyclophosphamide)"
    )
    fig.suptitle(f"Excess lethal-fraction dose response: {condition}", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.94,
        f"Fractional viability is background-corrected against {control_description}",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.022,
        (
            f"Filled points: paired replicate wells (n=2/dose); open points and bars: mean and range; "
            f"bands: 95% paired-replicate bootstrap ({bootstrap_iterations} resamples). "
            "Delta excess LF > 0 means more death in 4N; < 0 means more death in 2N."
        ),
        ha="center",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.12, wspace=0.16, hspace=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized SUM159 dose responses and compare 2N versus 4N "
            "using conventional Hill curves, endpoint GR curves, and excess "
            "lethal-fraction curves."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Production run, branch well-count output directory, or well_time_cell_state_counts.csv.",
    )
    parser.add_argument(
        "--branch",
        choices=("fusion", "fusion-nucleated-only", "legacy-combined"),
        default="fusion",
    )
    parser.add_argument(
        "--metric",
        choices=("auc", "endpoint"),
        default="auc",
        help="AUC uses the full selected live-cell trajectory; endpoint uses the final value/window.",
    )
    parser.add_argument("--start-hours", type=float, default=None)
    parser.add_argument("--endpoint-hours", type=float, default=None)
    parser.add_argument(
        "--endpoint-window-hours",
        type=float,
        default=0.0,
        help="For --metric endpoint, average this many hours ending at the endpoint.",
    )
    parser.add_argument(
        "--additional-endpoint-days",
        type=float,
        nargs="*",
        default=[4.0, 5.0],
        help=(
            "Also fit exact endpoint responses on these days (default: 4 5). "
            "Pass the option with no values to disable additional endpoints."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--skip-gr-comparison",
        action="store_true",
        help="Do not generate the cross-endpoint GR figures and tables.",
    )
    parser.add_argument(
        "--gr-bootstrap-iterations",
        type=int,
        default=500,
        help="Paired replicate bootstrap resamples for GR confidence bands (default: 500).",
    )
    parser.add_argument(
        "--gr-bootstrap-seed",
        type=int,
        default=20260717,
        help="Random seed for paired GR bootstrap resampling.",
    )
    parser.add_argument(
        "--skip-death-comparison",
        action="store_true",
        help="Do not generate endpoint excess lethal-fraction figures and tables.",
    )
    parser.add_argument(
        "--death-bootstrap-iterations",
        type=int,
        default=500,
        help="Paired replicate bootstrap resamples for death-response bands (default: 500).",
    )
    parser.add_argument(
        "--death-bootstrap-seed",
        type=int,
        default=20260718,
        help="Random seed for paired death-response bootstrap resampling.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def run_analysis(
    data: pd.DataFrame,
    spec: AnalysisSpec,
    out_dir: Path,
    allow_incomplete: bool,
    dpi: int,
) -> AnalysisResult:
    selected, start_hours, endpoint_hours = select_time_range(
        data,
        spec.start_hours,
        spec.endpoint_hours,
    )
    if spec.metric == "endpoint" and spec.endpoint_window_hours > endpoint_hours - start_hours:
        raise SystemExit(
            f"Endpoint window for {spec.output_key} cannot exceed its selected time range"
        )
    raw_response = extract_raw_responses(
        selected,
        metric=spec.metric,
        start_hours=start_hours,
        endpoint_hours=endpoint_hours,
        endpoint_window_hours=spec.endpoint_window_hours,
    )
    response = normalize_to_paired_vehicle(raw_response)
    response.insert(0, "analysis_key", spec.output_key)
    response.insert(1, "endpoint_day", spec.endpoint_day)
    validate_response_design(response, allow_incomplete)
    summary = summarize_responses(response)
    summary.insert(0, "analysis_key", spec.output_key)
    summary.insert(1, "metric", spec.metric)
    summary.insert(2, "start_hours", start_hours)
    summary.insert(3, "endpoint_hours", endpoint_hours)
    summary.insert(4, "endpoint_day", spec.endpoint_day)
    fits, fit_table = fit_all_curves(response)
    fit_table.insert(0, "analysis_key", spec.output_key)
    fit_table.insert(1, "metric", spec.metric)
    fit_table.insert(2, "start_hours", start_hours)
    fit_table.insert(3, "endpoint_hours", endpoint_hours)
    fit_table.insert(4, "endpoint_day", spec.endpoint_day)

    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / "dose_response_normalized_values.csv"
    summary_path = out_dir / "dose_response_summary.csv"
    fit_path = out_dir / "hill_fit_parameters.csv"
    response.to_csv(response_path, index=False)
    summary.to_csv(summary_path, index=False)
    fit_table.to_csv(fit_path, index=False)

    for cyclophosphamide in (False, True):
        stem = CONDITION_FILE_STEMS[cyclophosphamide]
        plot_condition(
            response,
            summary,
            fits,
            cyclophosphamide=cyclophosphamide,
            metric=spec.metric,
            start_hours=start_hours,
            endpoint_hours=endpoint_hours,
            endpoint_window_hours=spec.endpoint_window_hours,
            endpoint_day=spec.endpoint_day,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            dpi=dpi,
        )
    print(
        f"analysis={spec.output_key} metric={spec.metric} "
        f"time_range_hours={start_hours:g}-{endpoint_hours:g}",
        flush=True,
    )
    for fit in fits:
        print(
            f"fit analysis={spec.output_key} condition={fit.condition!r} ploidy={fit.ploidy} "
            f"ec50_nm={fit.ec50_nm:.6g} "
            f"hill_slope={fit.hill_slope:.6g} r_squared={fit.r_squared:.6g}",
            flush=True,
        )
    print(f"normalized_values={response_path}", flush=True)
    print(f"dose_summary={summary_path}", flush=True)
    print(f"hill_parameters={fit_path}", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    return AnalysisResult(
        spec=spec,
        response=response,
        start_hours=start_hours,
        endpoint_hours=endpoint_hours,
    )


def run_gr_comparison(
    endpoint_results: list[AnalysisResult],
    out_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    dpi: int,
) -> None:
    if not endpoint_results:
        print("GR comparison skipped: no exact endpoint analyses were requested", flush=True)
        return

    gr_response = pd.concat(
        [add_gr_values(result.response) for result in endpoint_results],
        ignore_index=True,
    )
    gr_summary = summarize_gr_responses(gr_response)
    gr_fits, gr_fit_table = fit_all_gr_curves(gr_response)
    bands, fit_intervals, delta_summary = bootstrap_gr_curves(
        gr_response,
        nominal_fits=gr_fits,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    gr_fit_table = gr_fit_table.merge(
        fit_intervals,
        on=["analysis_key", "condition", "cyclophosphamide", "ploidy"],
        how="left",
        validate="one_to_one",
    )

    paired_frames: list[pd.DataFrame] = []
    groups = gr_response[["analysis_key", "condition", "cyclophosphamide"]].drop_duplicates()
    for group in groups.itertuples(index=False):
        group_rows = gr_response[
            (gr_response["analysis_key"] == group.analysis_key)
            & (gr_response["cyclophosphamide"] == group.cyclophosphamide)
        ]
        paired = paired_gr_differences(group_rows)
        paired.insert(0, "analysis_key", group.analysis_key)
        paired.insert(1, "condition", group.condition)
        paired.insert(2, "cyclophosphamide", group.cyclophosphamide)
        paired_frames.append(paired)
    paired_table = pd.concat(paired_frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / "gr_normalized_values.csv"
    summary_path = out_dir / "gr_dose_summary.csv"
    fit_path = out_dir / "gr_fit_parameters.csv"
    band_path = out_dir / "gr_bootstrap_curve_bands.csv"
    delta_path = out_dir / "gr_delta_summary.csv"
    paired_path = out_dir / "gr_paired_differences.csv"
    gr_response.to_csv(response_path, index=False)
    gr_summary.to_csv(summary_path, index=False)
    gr_fit_table.to_csv(fit_path, index=False)
    bands.to_csv(band_path, index=False)
    delta_summary.to_csv(delta_path, index=False)
    paired_table.to_csv(paired_path, index=False)

    for cyclophosphamide in (False, True):
        stem = GR_CONDITION_FILE_STEMS[cyclophosphamide]
        plot_gr_condition(
            response=gr_response,
            summary=gr_summary,
            fits=gr_fits,
            fit_table=gr_fit_table,
            bands=bands,
            delta_summary=delta_summary,
            cyclophosphamide=cyclophosphamide,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            bootstrap_iterations=bootstrap_iterations,
            dpi=dpi,
        )

    for fit in gr_fits:
        gr50_text = f"{fit.gr50_nm:.6g}" if math.isfinite(fit.gr50_nm) else "not_reached"
        print(
            f"gr_fit analysis={fit.analysis_key} condition={fit.condition!r} "
            f"ploidy={fit.ploidy} gr50_nm={gr50_text} gr_max={fit.gr_max:.6g} "
            f"gec50_nm={fit.gec50_nm:.6g} hill_slope={fit.hill_slope:.6g}",
            flush=True,
        )
    print(f"gr_normalized_values={response_path}", flush=True)
    print(f"gr_dose_summary={summary_path}", flush=True)
    print(f"gr_fit_parameters={fit_path}", flush=True)
    print(f"gr_delta_summary={delta_path}", flush=True)
    print(f"gr_output_dir={out_dir}", flush=True)


def run_death_comparison(
    data: pd.DataFrame,
    endpoint_results: list[AnalysisResult],
    out_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    dpi: int,
) -> None:
    if not endpoint_results:
        print(
            "Death comparison skipped: no exact endpoint analyses were requested",
            flush=True,
        )
        return

    death_response = extract_endpoint_death_responses(data, endpoint_results)
    death_summary = summarize_death_responses(death_response)
    death_fits, death_fit_table = fit_all_death_curves(death_response)
    bands, fit_intervals, delta_summary = bootstrap_death_curves(
        death_response,
        nominal_fits=death_fits,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    death_fit_table = death_fit_table.merge(
        fit_intervals,
        on=["analysis_key", "condition", "cyclophosphamide", "ploidy"],
        how="left",
        validate="one_to_one",
    )

    paired_frames: list[pd.DataFrame] = []
    groups = death_response[
        ["analysis_key", "condition", "cyclophosphamide"]
    ].drop_duplicates()
    for group in groups.itertuples(index=False):
        group_rows = death_response[
            (death_response["analysis_key"] == group.analysis_key)
            & (death_response["cyclophosphamide"] == group.cyclophosphamide)
        ]
        paired = paired_death_differences(group_rows)
        paired.insert(0, "analysis_key", group.analysis_key)
        paired.insert(1, "condition", group.condition)
        paired.insert(2, "cyclophosphamide", group.cyclophosphamide)
        paired_frames.append(paired)
    paired_table = pd.concat(paired_frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / "death_fraction_values.csv"
    summary_path = out_dir / "death_dose_summary.csv"
    fit_path = out_dir / "death_fit_parameters.csv"
    band_path = out_dir / "death_bootstrap_curve_bands.csv"
    delta_path = out_dir / "death_delta_summary.csv"
    paired_path = out_dir / "death_paired_differences.csv"
    death_response.to_csv(response_path, index=False)
    death_summary.to_csv(summary_path, index=False)
    death_fit_table.to_csv(fit_path, index=False)
    bands.to_csv(band_path, index=False)
    delta_summary.to_csv(delta_path, index=False)
    paired_table.to_csv(paired_path, index=False)

    for cyclophosphamide in (False, True):
        stem = DEATH_CONDITION_FILE_STEMS[cyclophosphamide]
        plot_death_condition(
            response=death_response,
            summary=death_summary,
            fits=death_fits,
            fit_table=death_fit_table,
            bands=bands,
            delta_summary=delta_summary,
            cyclophosphamide=cyclophosphamide,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            bootstrap_iterations=bootstrap_iterations,
            dpi=dpi,
        )

    for fit in death_fits:
        lf50_text = f"{fit.lf50_nm:.6g}" if math.isfinite(fit.lf50_nm) else "not_reached"
        print(
            f"death_fit analysis={fit.analysis_key} condition={fit.condition!r} "
            f"ploidy={fit.ploidy} lf50_nm={lf50_text} lf_max={fit.lf_max:.6g} "
            f"lec50_nm={fit.lec50_nm:.6g} hill_slope={fit.hill_slope:.6g}",
            flush=True,
        )
    print(f"death_fraction_values={response_path}", flush=True)
    print(f"death_dose_summary={summary_path}", flush=True)
    print(f"death_fit_parameters={fit_path}", flush=True)
    print(f"death_delta_summary={delta_path}", flush=True)
    print(f"death_output_dir={out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.endpoint_window_hours < 0:
        raise SystemExit("--endpoint-window-hours must be nonnegative")
    if not args.skip_gr_comparison and args.gr_bootstrap_iterations <= 0:
        raise SystemExit("--gr-bootstrap-iterations must be positive")
    if not args.skip_death_comparison and args.death_bootstrap_iterations <= 0:
        raise SystemExit("--death-bootstrap-iterations must be positive")
    resolved = resolve_well_time_csv(args.input_path.resolve(), args.branch)
    data = load_well_time_table(resolved.csv_path)
    specs = build_analysis_specs(
        data,
        primary_metric=args.metric,
        start_hours=args.start_hours,
        endpoint_hours=args.endpoint_hours,
        endpoint_window_hours=args.endpoint_window_hours,
        additional_endpoint_days=args.additional_endpoint_days,
    )
    if args.out_dir is not None:
        base_out_dir = args.out_dir.resolve()
    elif resolved.run_dir is not None:
        base_out_dir = resolved.run_dir / "analysis" / "dose_response" / args.branch
    else:
        base_out_dir = resolved.csv_path.parent / "dose_response"

    print(f"input_csv={resolved.csv_path}", flush=True)
    print(f"analyses={','.join(spec.output_key for spec in specs)}", flush=True)
    results: list[AnalysisResult] = []
    for spec in specs:
        results.append(
            run_analysis(
                data,
                spec,
                out_dir=base_out_dir / spec.output_key,
                allow_incomplete=args.allow_incomplete,
                dpi=args.dpi,
            )
        )
    exact_endpoint_results = [
        result
        for result in results
        if result.spec.metric == "endpoint"
        and math.isclose(result.spec.endpoint_window_hours, 0.0)
    ]
    if not args.skip_gr_comparison:
        run_gr_comparison(
            exact_endpoint_results,
            out_dir=base_out_dir / "gr",
            bootstrap_iterations=args.gr_bootstrap_iterations,
            bootstrap_seed=args.gr_bootstrap_seed,
            dpi=args.dpi,
        )
    if not args.skip_death_comparison:
        run_death_comparison(
            data,
            exact_endpoint_results,
            out_dir=base_out_dir / "death",
            bootstrap_iterations=args.death_bootstrap_iterations,
            bootstrap_seed=args.death_bootstrap_seed,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
