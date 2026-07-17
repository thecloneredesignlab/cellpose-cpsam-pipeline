#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as student_t


REQUIRED_COLUMNS = {
    "well",
    "elapsed_hours",
    "live_count",
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
    for column in ("elapsed_hours", "live_count", "plate_column", "doxorubicin_nm", "replicate"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data["live_count"] < 0).any():
        raise SystemExit("Well-time table contains negative live-cell counts")
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


def metric_description(metric: str, start_hours: float, endpoint_hours: float, window_hours: float) -> str:
    if metric == "auc":
        return (
            f"Live-cell AUC from {start_hours:g}-{endpoint_hours:g} h; each trajectory divided by its "
            "baseline, then by its paired 0 nM control"
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
    ax.set_title(metric_description(metric, start_hours, endpoint_hours, endpoint_window_hours), fontsize=8.5, pad=8)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized SUM159 dose responses and compare 2N versus 4N Hill curves."
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
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.endpoint_window_hours < 0:
        raise SystemExit("--endpoint-window-hours must be nonnegative")
    resolved = resolve_well_time_csv(args.input_path.resolve(), args.branch)
    data = load_well_time_table(resolved.csv_path)
    selected, start_hours, endpoint_hours = select_time_range(
        data,
        args.start_hours,
        args.endpoint_hours,
    )
    if args.metric == "endpoint" and args.endpoint_window_hours > endpoint_hours - start_hours:
        raise SystemExit("--endpoint-window-hours cannot exceed the selected time range")

    raw_response = extract_raw_responses(
        selected,
        metric=args.metric,
        start_hours=start_hours,
        endpoint_hours=endpoint_hours,
        endpoint_window_hours=args.endpoint_window_hours,
    )
    response = normalize_to_paired_vehicle(raw_response)
    validate_response_design(response, args.allow_incomplete)
    summary = summarize_responses(response)
    fits, fit_table = fit_all_curves(response)

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    elif resolved.run_dir is not None:
        out_dir = resolved.run_dir / "analysis" / "dose_response" / args.branch / args.metric
    else:
        out_dir = resolved.csv_path.parent / f"dose_response_{args.metric}"
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
            metric=args.metric,
            start_hours=start_hours,
            endpoint_hours=endpoint_hours,
            endpoint_window_hours=args.endpoint_window_hours,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            dpi=args.dpi,
        )

    print(f"input_csv={resolved.csv_path}", flush=True)
    print(f"metric={args.metric} time_range_hours={start_hours:g}-{endpoint_hours:g}", flush=True)
    for fit in fits:
        print(
            f"fit condition={fit.condition!r} ploidy={fit.ploidy} ec50_nm={fit.ec50_nm:.6g} "
            f"hill_slope={fit.hill_slope:.6g} r_squared={fit.r_squared:.6g}",
            flush=True,
        )
    print(f"normalized_values={response_path}", flush=True)
    print(f"dose_summary={summary_path}", flush=True)
    print(f"hill_parameters={fit_path}", flush=True)
    print(f"output_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
