#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


IMAGE_ID_RE = re.compile(
    r"(?:^|_)(?P<well>[A-H][0-9]{1,2})_(?P<site>[0-9]+)_(?P<day>[0-9]{2})d(?P<hour>[0-9]{2})h(?P<minute>[0-9]{2})m$"
)


def find_predictions_dir(result_root: Path) -> Path:
    direct = result_root / "classification" / "predictions"
    if direct.is_dir():
        return direct

    candidates: list[Path] = []
    with os.scandir(result_root) as entries:
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("matplotlib_"):
                continue
            predictions = Path(entry.path) / "classification" / "predictions"
            if predictions.is_dir():
                candidates.append(predictions)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No classification/predictions directory found under {result_root}")
    joined = "\n".join(str(path) for path in candidates)
    raise SystemExit(f"Multiple prediction directories found. Pass the run directory explicitly:\n{joined}")


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


def well_sort_key(well: str) -> tuple[str, int]:
    match = re.match(r"^([A-H])([0-9]{1,2})$", well)
    if not match:
        return (well, 0)
    return (match.group(1), int(match.group(2)))


def read_summary_file(path: Path) -> tuple[dict[str, object] | None, bool]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if not row:
        return None, True
    parsed = parse_image_id(row.get("image_id", path.stem.removesuffix("_summary")))
    if parsed is None:
        return None, True
    return (
        {
            **parsed,
            "image_id": row.get("image_id", ""),
            "total_masks": int(float(row.get("total_masks") or 0)),
            "live_count": int(float(row.get("live_count") or 0)),
            "dead_count": int(float(row.get("dead_count") or 0)),
            "transitional_count": int(float(row.get("transitional_count") or 0)),
            "artifact_count": int(float(row.get("artifact_count") or 0)),
            "uncertain_count": int(float(row.get("uncertain_count") or 0)),
        },
        False,
    )


def load_summary_rows(predictions_dir: Path, workers: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    skipped = 0
    paths: list[Path] = []
    with os.scandir(predictions_dir) as entries:
        for entry in entries:
            if entry.name.endswith("_summary.csv"):
                paths.append(Path(entry.path))

    if not paths:
        raise SystemExit(f"No *_summary.csv files found in {predictions_dir}")
    print(f"summary_files_found={len(paths)} workers={workers}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(read_summary_file, path) for path in paths]
        for seen, future in enumerate(as_completed(futures), start=1):
            row, was_skipped = future.result()
            if was_skipped:
                skipped += 1
            elif row is not None:
                rows.append(row)
            if seen == 1 or seen % 500 == 0 or seen == len(paths):
                print(f"read_summary_files={seen}/{len(paths)}", flush=True)

    if not rows:
        raise SystemExit(f"No parseable *_summary.csv files found in {predictions_dir}")
    if skipped:
        print(f"skipped_summary_files={skipped}", flush=True)
    print(f"loaded_summary_files={len(rows)}", flush=True)
    return pd.DataFrame(rows)


def aggregate_counts(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["well", "elapsed_hours"], as_index=False)
        .agg(
            n_sites=("site", "nunique"),
            n_images=("image_id", "count"),
            total_masks=("total_masks", "sum"),
            live_count=("live_count", "sum"),
            dead_count=("dead_count", "sum"),
            transitional_count=("transitional_count", "sum"),
            artifact_count=("artifact_count", "sum"),
            uncertain_count=("uncertain_count", "sum"),
        )
    )
    grouped["_well_row"] = grouped["well"].map(lambda well: well_sort_key(str(well))[0])
    grouped["_well_col"] = grouped["well"].map(lambda well: well_sort_key(str(well))[1])
    grouped = grouped.sort_values(["_well_row", "_well_col", "elapsed_hours"]).drop(columns=["_well_row", "_well_col"])
    grouped["live_fraction"] = grouped["live_count"] / grouped["total_masks"].where(grouped["total_masks"] != 0, pd.NA)
    grouped["dead_fraction"] = grouped["dead_count"] / grouped["total_masks"].where(grouped["total_masks"] != 0, pd.NA)
    return grouped


def plot_counts(aggregated: pd.DataFrame, out_png: Path, out_pdf: Path, max_cols: int) -> None:
    wells = sorted(aggregated["well"].unique(), key=well_sort_key)
    n_wells = len(wells)
    n_cols = min(max_cols, n_wells)
    n_rows = math.ceil(n_wells / n_cols)
    fig_width = max(12, n_cols * 2.35)
    fig_height = max(6, n_rows * 1.9)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True)
    if not isinstance(axes, (list, pd.Series)):
        axes_flat = list(getattr(axes, "flat", [axes]))
    else:
        axes_flat = list(axes)
    if hasattr(axes, "flat"):
        axes_flat = list(axes.flat)

    for ax, well in zip(axes_flat, wells):
        data = aggregated[aggregated["well"] == well].sort_values("elapsed_hours")
        ax.plot(data["elapsed_hours"], data["live_count"], color="#c9332c", linewidth=1.4, label="live")
        ax.plot(data["elapsed_hours"], data["dead_count"], color="#2367c9", linewidth=1.4, label="dead")
        ax.set_title(well, fontsize=8, pad=3)
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.set_xlim(left=max(0, float(aggregated["elapsed_hours"].min())))

    for ax in axes_flat[n_wells:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.supxlabel("Elapsed time (hours)", fontsize=10)
    fig.supylabel("Segmented cell count", fontsize=10)
    fig.suptitle("CellposeSAM classified cell counts over time by well", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.965))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot live/dead CellposeSAM cell counts over time, one subplot per well.")
    parser.add_argument("result_root", type=Path, help="Result root or run directory containing classification/predictions.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12, help="Concurrent summary CSV readers.")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    predictions_dir = find_predictions_dir(result_root)
    run_dir = predictions_dir.parents[1]
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "qc" / "well_count_timecourses"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"predictions_dir={predictions_dir}", flush=True)
    summary = load_summary_rows(predictions_dir, args.workers)
    aggregated = aggregate_counts(summary)

    per_image_csv = out_dir / "per_image_cell_state_counts.csv"
    aggregated_csv = out_dir / "well_time_cell_state_counts.csv"
    out_png = out_dir / "well_live_dead_counts_over_time.png"
    out_pdf = out_dir / "well_live_dead_counts_over_time.pdf"

    summary.sort_values(["well", "site", "elapsed_hours"]).to_csv(per_image_csv, index=False)
    aggregated.to_csv(aggregated_csv, index=False)
    plot_counts(aggregated, out_png, out_pdf, args.max_cols)

    print(f"images={len(summary)} wells={aggregated['well'].nunique()} timepoints={aggregated['elapsed_hours'].nunique()}", flush=True)
    print(f"per_image_csv={per_image_csv}", flush=True)
    print(f"aggregated_csv={aggregated_csv}", flush=True)
    print(f"plot_png={out_png}", flush=True)
    print(f"plot_pdf={out_pdf}", flush=True)


if __name__ == "__main__":
    main()
