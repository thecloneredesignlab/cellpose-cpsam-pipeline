#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


IMAGE_ID_RE = re.compile(
    r"(?:^|_)(?P<well>[A-H][0-9]{1,2})_(?P<site>[0-9]+)_"
    r"(?P<day>[0-9]{2})d(?P<hour>[0-9]{2})h(?P<minute>[0-9]{2})m$"
)
PLATE_ROWS = tuple("ABCDEFGH")
PLATE_COLUMNS = tuple(range(1, 13))
DEFAULT_PLATE_MAP = Path(__file__).resolve().parent / "resources" / "SUM159_AC_Experiment1_PlateMap.csv"
STATE_COUNT_COLUMNS = (
    "live_count",
    "dead_count",
    "transitional_count",
    "artifact_count",
    "uncertain_count",
)


@dataclass(frozen=True)
class InputSource:
    branch: str
    kind: str
    path: Path
    run_dir: Path


def well_sort_key(well: str) -> tuple[str, int]:
    match = re.match(r"^([A-H])([0-9]{1,2})$", well)
    if not match:
        return (well, 0)
    return (match.group(1), int(match.group(2)))


def parse_image_id(image_id: str) -> dict[str, object] | None:
    match = IMAGE_ID_RE.search(image_id)
    if not match:
        return None
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    return {
        "well": match.group("well"),
        "site": int(match.group("site")),
        "elapsed_hours": day * 24 + hour + minute / 60,
    }


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def fusion_summary_candidates(result_root: Path, branch_dir: str) -> list[Path]:
    if result_root.is_file():
        if (
            result_root.name == "cell_count_summary.csv"
            and result_root.parent.name == "summaries"
            and result_root.parent.parent.name == branch_dir
        ):
            return [result_root]
        return []

    direct: list[Path] = []
    if result_root.name == branch_dir:
        direct.append(result_root / "summaries" / "cell_count_summary.csv")
    direct.append(result_root / branch_dir / "summaries" / "cell_count_summary.csv")
    existing = [path for path in direct if path.is_file()]
    if existing:
        return unique_paths(existing)

    nested: list[Path] = []
    with os.scandir(result_root) as entries:
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("matplotlib_"):
                continue
            candidate = Path(entry.path) / branch_dir / "summaries" / "cell_count_summary.csv"
            if candidate.is_file():
                nested.append(candidate)
    return unique_paths(nested)


def legacy_combined_candidates(result_root: Path) -> list[Path]:
    if not result_root.is_dir():
        return []

    direct: list[Path] = []
    if result_root.name == "Combined":
        direct.append(result_root / "classification" / "predictions")
    direct.append(result_root / "Combined" / "classification" / "predictions")
    existing = [path for path in direct if path.is_dir()]
    if existing:
        return unique_paths(existing)

    nested: list[Path] = []
    with os.scandir(result_root) as entries:
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("matplotlib_"):
                continue
            candidate = Path(entry.path) / "Combined" / "classification" / "predictions"
            if candidate.is_dir():
                nested.append(candidate)
    return unique_paths(nested)


def require_single_candidate(candidates: list[Path], description: str) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No {description} found")
    joined = "\n".join(str(path) for path in candidates)
    raise SystemExit(f"Multiple {description} candidates found. Pass the run directory explicitly:\n{joined}")


def source_from_fusion_summary(summary_path: Path, branch: str) -> InputSource:
    branch_dir = summary_path.parent.parent
    return InputSource(branch=branch, kind="fusion", path=summary_path, run_dir=branch_dir.parent)


def source_from_legacy_predictions(predictions_dir: Path) -> InputSource:
    combined_dir = predictions_dir.parent.parent
    run_dir = combined_dir.parent if combined_dir.name == "Combined" else combined_dir
    return InputSource(
        branch="legacy-combined",
        kind="legacy",
        path=predictions_dir,
        run_dir=run_dir,
    )


def resolve_input_source(result_root: Path, requested_branch: str) -> InputSource:
    if not result_root.exists():
        raise SystemExit(f"Result path does not exist: {result_root}")

    if requested_branch in {"auto", "fusion-consensus"}:
        candidates = fusion_summary_candidates(result_root, "classification_consensus")
        if candidates:
            return source_from_fusion_summary(
                require_single_candidate(
                    candidates,
                    "classification_consensus summary",
                ),
                "fusion-consensus",
            )
        if requested_branch == "fusion-consensus":
            require_single_candidate(candidates, "classification_consensus summary")

    if requested_branch in {"auto", "fusion"}:
        candidates = fusion_summary_candidates(result_root, "classification_fusion")
        if candidates:
            return source_from_fusion_summary(
                require_single_candidate(candidates, "classification_fusion summary"),
                "fusion",
            )
        if requested_branch == "fusion":
            require_single_candidate(candidates, "classification_fusion summary")

    if requested_branch in {"auto", "fusion-nucleated-only"}:
        candidates = fusion_summary_candidates(result_root, "classification_fusion_nucleated_only")
        if candidates:
            return source_from_fusion_summary(
                require_single_candidate(candidates, "classification_fusion_nucleated_only summary"),
                "fusion-nucleated-only",
            )
        if requested_branch == "fusion-nucleated-only":
            require_single_candidate(candidates, "classification_fusion_nucleated_only summary")

    if requested_branch in {"auto", "legacy-combined"}:
        candidates = legacy_combined_candidates(result_root)
        if candidates:
            return source_from_legacy_predictions(
                require_single_candidate(candidates, "legacy Combined classification/predictions directory")
            )
        if requested_branch == "legacy-combined":
            require_single_candidate(candidates, "legacy Combined classification/predictions directory")

    raise SystemExit(
        "No supported cell-state summary found. Expected classification_consensus/"
        "summaries/cell_count_summary.csv, classification_fusion/summaries/"
        "cell_count_summary.csv, or Combined/classification/predictions/."
    )


def load_fusion_summary(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    required = {
        "image_id",
        "well",
        "site",
        "elapsed_hours",
        "total_masks",
        "total_cell_count",
        "live_cell_count",
        "dead_cell_count",
        "artifact_count",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise SystemExit(f"Fusion summary is missing required columns: {', '.join(missing)}")

    for column in ("transitional_count", "uncertain_count"):
        if column not in summary:
            summary[column] = 0

    normalized = summary[
        [
            "image_id",
            "well",
            "site",
            "elapsed_hours",
            "total_masks",
            "total_cell_count",
            "live_cell_count",
            "dead_cell_count",
            "transitional_count",
            "artifact_count",
            "uncertain_count",
        ]
    ].rename(columns={"live_cell_count": "live_count", "dead_cell_count": "dead_count"})
    return coerce_summary_types(normalized)


def read_legacy_summary_file(path: Path) -> tuple[dict[str, object] | None, bool]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return None, True

    image_id = row.get("image_id") or path.stem.removesuffix("_summary")
    parsed = parse_image_id(image_id)
    if parsed is None:
        return None, True

    total_masks = int(float(row.get("total_masks") or 0))
    artifact_count = int(float(row.get("artifact_count") or 0))
    return (
        {
            **parsed,
            "image_id": image_id,
            "total_masks": total_masks,
            "total_cell_count": total_masks - artifact_count,
            "live_count": int(float(row.get("live_count") or 0)),
            "dead_count": int(float(row.get("dead_count") or 0)),
            "transitional_count": int(float(row.get("transitional_count") or 0)),
            "artifact_count": artifact_count,
            "uncertain_count": int(float(row.get("uncertain_count") or 0)),
        },
        False,
    )


def load_legacy_summary(predictions_dir: Path, workers: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    skipped = 0
    paths: list[Path] = []
    with os.scandir(predictions_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith("_summary.csv"):
                paths.append(Path(entry.path))

    if not paths:
        raise SystemExit(f"No *_summary.csv files found in {predictions_dir}")
    print(f"legacy_summary_files_found={len(paths)} workers={workers}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(read_legacy_summary_file, path) for path in paths]
        for seen, future in enumerate(as_completed(futures), start=1):
            row, was_skipped = future.result()
            if was_skipped:
                skipped += 1
            elif row is not None:
                rows.append(row)
            if seen == 1 or seen % 500 == 0 or seen == len(paths):
                print(f"read_legacy_summary_files={seen}/{len(paths)}", flush=True)

    if not rows:
        raise SystemExit(f"No parseable *_summary.csv files found in {predictions_dir}")
    if skipped:
        print(f"skipped_legacy_summary_files={skipped}", flush=True)
    return coerce_summary_types(pd.DataFrame(rows))


def coerce_summary_types(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["image_id"] = result["image_id"].astype(str)
    result["well"] = result["well"].astype(str)
    numeric_columns = ["site", "elapsed_hours", "total_masks", "total_cell_count", *STATE_COUNT_COLUMNS]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    integer_columns = ["site", "total_masks", "total_cell_count", *STATE_COUNT_COLUMNS]
    for column in integer_columns:
        result[column] = result[column].astype("int64")
    return result


def load_plate_map(path: Path) -> pd.DataFrame:
    plate_map = pd.read_csv(path, dtype={"well": str, "plate_row": str, "ploidy": str})
    required = {
        "well",
        "plate_row",
        "plate_column",
        "doxorubicin_nm",
        "ploidy",
        "cyclophosphamide",
        "replicate",
    }
    missing = sorted(required - set(plate_map.columns))
    if missing:
        raise SystemExit(f"Plate map is missing required columns: {', '.join(missing)}")
    if plate_map["well"].duplicated().any():
        duplicates = sorted(plate_map.loc[plate_map["well"].duplicated(), "well"].unique())
        raise SystemExit(f"Duplicate wells in plate map: {', '.join(duplicates)}")

    plate_map["plate_column"] = pd.to_numeric(plate_map["plate_column"], errors="raise").astype(int)
    plate_map["doxorubicin_nm"] = pd.to_numeric(plate_map["doxorubicin_nm"], errors="raise")
    plate_map["replicate"] = pd.to_numeric(plate_map["replicate"], errors="raise").astype(int)
    plate_map["cyclophosphamide"] = plate_map["cyclophosphamide"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "with"}
    )
    expected_wells = plate_map["plate_row"] + plate_map["plate_column"].astype(str)
    invalid = plate_map.loc[plate_map["well"] != expected_wells, ["well", "plate_row", "plate_column"]]
    if not invalid.empty:
        raise SystemExit(f"Plate-map well coordinates are inconsistent:\n{invalid.to_string(index=False)}")
    if not set(plate_map["plate_row"]).issubset(PLATE_ROWS):
        raise SystemExit("Plate map contains rows outside A-H")
    if not set(plate_map["plate_column"]).issubset(PLATE_COLUMNS):
        raise SystemExit("Plate map contains columns outside 1-12")
    return plate_map.sort_values(["plate_row", "plate_column"]).reset_index(drop=True)


def validate_summary(
    summary: pd.DataFrame,
    plate_map: pd.DataFrame,
    strict_completeness: bool,
    expected_timepoints: int | None,
    expected_sites: int | None,
) -> None:
    if summary.empty:
        raise SystemExit("Cell-state summary is empty")
    if summary[["image_id", "well", "site", "elapsed_hours"]].isna().any().any():
        raise SystemExit("Cell-state summary contains missing identifiers")
    if summary["image_id"].duplicated().any():
        duplicates = summary.loc[summary["image_id"].duplicated(keep=False), "image_id"].head(10)
        raise SystemExit(f"Duplicate image_id values found:\n{duplicates.to_string(index=False)}")

    duplicate_key = summary.duplicated(["well", "site", "elapsed_hours"], keep=False)
    if duplicate_key.any():
        examples = summary.loc[duplicate_key, ["well", "site", "elapsed_hours"]].head(10)
        raise SystemExit(f"Duplicate well/site/time records found:\n{examples.to_string(index=False)}")

    count_columns = ["total_masks", "total_cell_count", *STATE_COUNT_COLUMNS]
    if (summary[count_columns] < 0).any().any():
        raise SystemExit("Cell-state summary contains negative counts")

    state_sum = summary[list(STATE_COUNT_COLUMNS)].sum(axis=1)
    mask_mismatch = state_sum != summary["total_masks"]
    if mask_mismatch.any():
        raise SystemExit(f"State counts do not sum to total_masks in {int(mask_mismatch.sum())} rows")

    countable_sum = summary[["live_count", "dead_count", "transitional_count", "uncertain_count"]].sum(axis=1)
    countable_mismatch = countable_sum != summary["total_cell_count"]
    if countable_mismatch.any():
        raise SystemExit(
            f"Countable state counts do not sum to total_cell_count in {int(countable_mismatch.sum())} rows"
        )

    mapped_wells = set(plate_map["well"])
    observed_wells = set(summary["well"])
    unknown_wells = sorted(observed_wells - mapped_wells, key=well_sort_key)
    if unknown_wells:
        raise SystemExit(f"Summary contains wells absent from the plate map: {', '.join(unknown_wells)}")

    n_timepoints = int(summary["elapsed_hours"].nunique())
    site_counts = summary.groupby(["well", "elapsed_hours"])["site"].nunique()
    print(
        "validation "
        f"rows={len(summary)} wells={len(observed_wells)} timepoints={n_timepoints} "
        f"sites_per_well_time={sorted(int(value) for value in site_counts.unique())}",
        flush=True,
    )

    if expected_timepoints is not None and n_timepoints != expected_timepoints:
        raise SystemExit(f"Expected {expected_timepoints} timepoints, found {n_timepoints}")
    if expected_sites is not None and not (site_counts == expected_sites).all():
        bad = site_counts[site_counts != expected_sites].head(10)
        raise SystemExit(
            f"Expected {expected_sites} sites per well/time; mismatches include:\n{bad.to_string()}"
        )

    if strict_completeness:
        missing_wells = sorted(mapped_wells - observed_wells, key=well_sort_key)
        if missing_wells:
            raise SystemExit(f"Plate-map wells missing from summary: {', '.join(missing_wells)}")
        expected_time_values = frozenset(summary["elapsed_hours"].unique())
        bad_time_wells = [
            well
            for well, rows in summary.groupby("well")
            if frozenset(rows["elapsed_hours"].unique()) != expected_time_values
        ]
        if bad_time_wells:
            raise SystemExit(f"Wells have incomplete or inconsistent time courses: {', '.join(bad_time_wells)}")
        expected_site_values = frozenset(summary["site"].unique())
        bad_site_groups = [
            (well, elapsed_hours)
            for (well, elapsed_hours), rows in summary.groupby(["well", "elapsed_hours"])
            if frozenset(rows["site"].unique()) != expected_site_values
        ]
        if bad_site_groups:
            examples = ", ".join(f"{well}@{elapsed:g}h" for well, elapsed in bad_site_groups[:10])
            raise SystemExit(f"Well/time groups have incomplete or inconsistent site sets: {examples}")


def aggregate_counts(summary: pd.DataFrame, plate_map: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["well", "elapsed_hours"], as_index=False)
        .agg(
            n_sites=("site", "nunique"),
            n_images=("image_id", "count"),
            total_masks=("total_masks", "sum"),
            total_cell_count=("total_cell_count", "sum"),
            live_count=("live_count", "sum"),
            dead_count=("dead_count", "sum"),
            transitional_count=("transitional_count", "sum"),
            artifact_count=("artifact_count", "sum"),
            uncertain_count=("uncertain_count", "sum"),
        )
        .merge(plate_map, on="well", how="left", validate="many_to_one")
    )
    grouped = grouped.sort_values(["plate_row", "plate_column", "elapsed_hours"]).reset_index(drop=True)
    denominator = grouped["total_cell_count"].where(grouped["total_cell_count"] != 0, pd.NA)
    grouped["live_fraction"] = grouped["live_count"] / denominator
    grouped["dead_fraction"] = grouped["dead_count"] / denominator
    grouped["binary_classified_count"] = (
        grouped["live_count"] + grouped["dead_count"]
    )
    grouped["classification_coverage"] = (
        grouped["binary_classified_count"] / denominator
    )
    return grouped


def dose_label(value: float) -> str:
    return f"{float(value):g} nM"


def source_description(branch: str) -> str:
    if branch == "fusion-consensus":
        return (
            "Authoritative dual-view death-classification consensus "
            "(original cell masks with nucleated-only diagnostics)"
        )
    if branch == "fusion":
        return "Primary multichannel fusion on original Combined cell masks"
    if branch == "fusion-nucleated-only":
        return "Nucleated-only multichannel fusion sensitivity branch"
    return "Legacy Combined RGB-only classifier"


def set_y_limits(
    axes_by_well: dict[str, plt.Axes],
    aggregated: pd.DataFrame,
    mode: str,
) -> None:
    if mode == "independent":
        return
    if mode == "shared":
        maximum = float(
            aggregated[
                ["live_count", "dead_count", "uncertain_count", "total_cell_count"]
            ]
            .max()
            .max()
        )
        limit = max(1.0, maximum * 1.05)
        for ax in axes_by_well.values():
            ax.set_ylim(0, limit)
        return
    for plate_row, row_data in aggregated.groupby("plate_row"):
        maximum = float(
            row_data[
                ["live_count", "dead_count", "uncertain_count", "total_cell_count"]
            ]
            .max()
            .max()
        )
        limit = max(1.0, maximum * 1.05)
        for well, ax in axes_by_well.items():
            if well.startswith(str(plate_row)):
                ax.set_ylim(0, limit)


def style_data_axis(ax: plt.Axes, well: str, data: pd.DataFrame, minimum_time: float) -> None:
    ax.plot(data["elapsed_hours"], data["live_count"], color="#c9332c", linewidth=1.25, label="live")
    ax.plot(data["elapsed_hours"], data["dead_count"], color="#2367c9", linewidth=1.25, label="dead")
    ax.plot(
        data["elapsed_hours"],
        data["uncertain_count"],
        color="#d28b00",
        linewidth=1.05,
        label="uncertain",
    )
    ax.plot(
        data["elapsed_hours"],
        data["total_cell_count"],
        color="#303030",
        linewidth=0.9,
        linestyle=":",
        label="total countable",
    )
    ax.text(
        0.025,
        0.96,
        well,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#252525",
    )
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlim(left=max(0, minimum_time))


def plot_plate_counts(
    aggregated: pd.DataFrame,
    plate_map: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    branch: str,
    layout: str,
    y_axis: str,
    dpi: int,
) -> None:
    columns = PLATE_COLUMNS if layout == "plate" else tuple(range(2, 12))
    fig_width = 29 if layout == "plate" else 25
    fig, axes = plt.subplots(
        len(PLATE_ROWS),
        len(columns),
        figsize=(fig_width, 16.5),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.865, bottom=0.065, wspace=0.20, hspace=0.28)

    minimum_time = float(aggregated["elapsed_hours"].min())
    data_wells = set(aggregated["well"])
    map_by_well = plate_map.set_index("well")
    axes_by_well: dict[str, plt.Axes] = {}

    for row_index, plate_row in enumerate(PLATE_ROWS):
        for column_index, plate_column in enumerate(columns):
            ax = axes[row_index, column_index]
            well = f"{plate_row}{plate_column}"
            if well not in data_wells:
                ax.axis("off")
                continue
            data = aggregated[aggregated["well"] == well].sort_values("elapsed_hours")
            style_data_axis(ax, well, data, minimum_time)
            axes_by_well[well] = ax
            metadata = map_by_well.loc[well]
            if bool(metadata["cyclophosphamide"]):
                ax.set_facecolor("#fffbea")

    set_y_limits(axes_by_well, aggregated, y_axis)
    first_data_column = columns.index(2)
    if y_axis != "independent":
        for row_index in range(len(PLATE_ROWS)):
            for column_index in range(len(columns)):
                if column_index != first_data_column and axes[row_index, column_index].axison:
                    axes[row_index, column_index].tick_params(labelleft=False)

    for column_index, plate_column in enumerate(columns):
        bbox = axes[0, column_index].get_position()
        column_rows = plate_map[plate_map["plate_column"] == plate_column]
        if column_rows.empty:
            label = f"Column {plate_column}\n(no data)"
        else:
            label = f"Column {plate_column}\nDoxo {dose_label(float(column_rows.iloc[0]['doxorubicin_nm']))}"
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.012, label, ha="center", va="bottom", fontsize=7.5)

    for row_index, plate_row in enumerate(PLATE_ROWS):
        row_metadata = plate_map[plate_map["plate_row"] == plate_row].iloc[0]
        bbox = axes[row_index, first_data_column].get_position()
        cyclo_label = "+ cyclophosphamide" if bool(row_metadata["cyclophosphamide"]) else "no cyclophosphamide"
        label = f"{plate_row}\n{row_metadata['ploidy']}, {cyclo_label}\nreplicate {int(row_metadata['replicate'])}"
        fig.text(
            bbox.x0 - 0.012,
            (bbox.y0 + bbox.y1) / 2,
            label,
            ha="right",
            va="center",
            fontsize=7.5,
            linespacing=1.25,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "#fff3ad" if bool(row_metadata["cyclophosphamide"]) else "#f3f3f3",
                "edgecolor": "#b8b8b8",
                "linewidth": 0.5,
            },
        )

    first_axis = next(iter(axes_by_well.values()))
    handles, labels = first_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "SUM-159 classified and uncertain cell counts over time by well",
        y=0.992,
        fontsize=14,
    )
    site_counts = sorted(int(value) for value in aggregated["n_sites"].unique())
    state_note = (
        "artifacts excluded; live + dead + uncertain = total countable"
        if branch != "legacy-combined"
        else "artifacts excluded; legacy live + dead + transitional + uncertain = total countable"
    )
    fig.text(
        0.5,
        0.938,
        f"{source_description(branch)} | sites summed per well/time: {site_counts} | {state_note}",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.supxlabel("Elapsed time (hours)", fontsize=10, y=0.018)
    fig.supylabel("Count across imaging sites", fontsize=10, x=0.012)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_grid_counts(
    aggregated: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    branch: str,
    max_cols: int,
    y_axis: str,
    dpi: int,
) -> None:
    wells = sorted(aggregated["well"].unique(), key=well_sort_key)
    n_cols = min(max_cols, len(wells))
    n_rows = math.ceil(len(wells) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12, n_cols * 2.35), max(6, n_rows * 1.9)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes.flat)
    axes_by_well: dict[str, plt.Axes] = {}
    minimum_time = float(aggregated["elapsed_hours"].min())
    for ax, well in zip(axes_flat, wells):
        data = aggregated[aggregated["well"] == well].sort_values("elapsed_hours")
        style_data_axis(ax, str(well), data, minimum_time)
        axes_by_well[str(well)] = ax
    for ax in axes_flat[len(wells) :]:
        ax.axis("off")
    set_y_limits(axes_by_well, aggregated, y_axis)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.supxlabel("Elapsed time (hours)", fontsize=10)
    fig.supylabel("Count across imaging sites", fontsize=10)
    fig.suptitle(
        f"SUM-159 classified and uncertain counts: {source_description(branch)}",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot live/dead CellposeSAM cell counts over time, one subplot per well."
    )
    parser.add_argument(
        "result_root",
        type=Path,
        help="Production run, fusion branch, merged fusion summary, or legacy Combined run.",
    )
    parser.add_argument(
        "--branch",
        choices=(
            "auto",
            "fusion-consensus",
            "fusion",
            "fusion-nucleated-only",
            "legacy-combined",
        ),
        default="auto",
        help=(
            "Classification source. Auto prefers the authoritative dual-view "
            "consensus summary."
        ),
    )
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--layout", choices=("plate", "compact", "grid"), default="plate")
    parser.add_argument("--y-axis", choices=("shared", "row", "independent"), default="shared")
    parser.add_argument("--max-cols", type=int, default=8, help="Maximum columns for --layout grid.")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent readers for legacy summaries.")
    parser.add_argument("--strict-completeness", action="store_true")
    parser.add_argument("--expected-timepoints", type=int, default=None)
    parser.add_argument("--expected-sites", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    plate_map_path = args.plate_map.resolve()
    source = resolve_input_source(result_root, args.branch)
    plate_map = load_plate_map(plate_map_path)

    print(f"source_branch={source.branch}", flush=True)
    print(f"source_path={source.path}", flush=True)
    print(f"plate_map={plate_map_path}", flush=True)
    if source.kind == "fusion":
        summary = load_fusion_summary(source.path)
    else:
        summary = load_legacy_summary(source.path, args.workers)

    validate_summary(
        summary,
        plate_map,
        strict_completeness=args.strict_completeness,
        expected_timepoints=args.expected_timepoints,
        expected_sites=args.expected_sites,
    )
    aggregated = aggregate_counts(summary, plate_map)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else source.run_dir / "analysis" / "well_count_timecourses" / source.branch
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_dir / "per_image_cell_state_counts.csv"
    aggregated_csv = out_dir / "well_time_cell_state_counts.csv"
    out_png = out_dir / "well_live_dead_counts_over_time.png"
    out_pdf = out_dir / "well_live_dead_counts_over_time.pdf"

    summary.sort_values(["well", "site", "elapsed_hours"]).to_csv(per_image_csv, index=False)
    aggregated.to_csv(aggregated_csv, index=False)
    if args.layout in {"plate", "compact"}:
        plot_plate_counts(
            aggregated,
            plate_map,
            out_png,
            out_pdf,
            branch=source.branch,
            layout=args.layout,
            y_axis=args.y_axis,
            dpi=args.dpi,
        )
    else:
        plot_grid_counts(
            aggregated,
            out_png,
            out_pdf,
            branch=source.branch,
            max_cols=args.max_cols,
            y_axis=args.y_axis,
            dpi=args.dpi,
        )

    print(
        f"images={len(summary)} wells={aggregated['well'].nunique()} "
        f"timepoints={aggregated['elapsed_hours'].nunique()}",
        flush=True,
    )
    print(f"per_image_csv={per_image_csv}", flush=True)
    print(f"aggregated_csv={aggregated_csv}", flush=True)
    print(f"plot_png={out_png}", flush=True)
    print(f"plot_pdf={out_pdf}", flush=True)


if __name__ == "__main__":
    main()
