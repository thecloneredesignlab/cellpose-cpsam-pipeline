#!/usr/bin/env python3
"""Render exhaustive E9/d0 QC and a self-contained trajectory report."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries


LIVE_COLOR = (35, 205, 95)
DEAD_COLOR = (176, 74, 214)
UNCERTAIN_COLOR = (255, 214, 10)
ARTIFACT_COLOR = (145, 150, 160)
CLASSIFICATION_BOUNDARY_WIDTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qc-wells", default="E9,F9")
    parser.add_argument("--key-hours", default="0,24,48,72,96,120,144,168")
    parser.add_argument("--render-all-e9", action="store_true")
    parser.add_argument("--render-all-d0", action="store_true")
    parser.add_argument("--max-report-images", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.ndim == 3 and values.shape[0] in {3, 4} and values.shape[-1] not in {3, 4}:
        values = np.moveaxis(values, 0, -1)
    values = values[:, :, :3].astype(np.float32)
    output = np.zeros_like(values, dtype=np.uint8)
    for channel in range(3):
        plane = values[:, :, channel]
        low, high = np.percentile(plane, [0.5, 99.5])
        if high <= low:
            high = low + 1.0
        output[:, :, channel] = np.clip(
            (plane - low) / (high - low) * 255,
            0,
            255,
        ).astype(np.uint8)
    return output


def normalize_scalar(array: np.ndarray) -> np.ndarray:
    values = np.squeeze(np.asarray(array)).astype(np.float32)
    low, high = np.percentile(values, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return np.stack((scaled, scaled, scaled), axis=2)


def read_labels(path: str) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels: {path}, shape={labels.shape}")
    return labels.astype(np.int32, copy=False)


def tint(
    image: np.ndarray,
    labels: np.ndarray,
    selected: set[int],
    color: tuple[int, int, int],
    mode: str = "outer",
) -> np.ndarray:
    output = image.copy()
    if not selected:
        return output
    mask = np.isin(labels, np.fromiter(selected, dtype=np.int32))
    boundary = find_boundaries(mask, mode=mode)
    for _ in range(CLASSIFICATION_BOUNDARY_WIDTH - 1):
        padded = np.pad(boundary, 1, mode="constant", constant_values=False)
        boundary = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    output[boundary] = np.asarray(color, dtype=np.uint8)
    return output


def label_ids(rows: pd.DataFrame, selector: pd.Series) -> set[int]:
    return set(rows.loc[selector, "combined_mask_id"].astype(int).tolist())


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
        Path(
            "/System/Library/Fonts/Supplemental/"
            + ("Arial Bold.ttf" if bold else "Arial.ttf")
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def panel(image: np.ndarray, title: str, width: int = 620) -> Image.Image:
    source = Image.fromarray(image)
    scale = width / source.width
    resized = source.resize(
        (width, max(1, int(round(source.height * scale)))),
        Image.Resampling.BILINEAR,
    )
    header_height = 64
    canvas = Image.new("RGB", (width, resized.height + header_height), "white")
    canvas.paste(resized, (0, header_height))
    ImageDraw.Draw(canvas).text(
        (12, 15),
        title,
        fill="black",
        font=font(27, True),
    )
    return canvas


def compose(panels: list[Image.Image], footer: list[str]) -> Image.Image:
    columns = 3
    rows = math.ceil(len(panels) / columns)
    panel_width = max(item.width for item in panels)
    panel_height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * panel_height + 100),
        "white",
    )
    for index, item in enumerate(panels):
        canvas.paste(
            item,
            ((index % columns) * panel_width, (index // columns) * panel_height),
        )
    draw = ImageDraw.Draw(canvas)
    y = rows * panel_height + 8
    for line in footer:
        draw.text((10, y), line, fill="black", font=ImageFont.load_default())
        y += 16
    return canvas


def render_field_panel_images(
    rows: pd.DataFrame,
    include_nuclei: bool = True,
    include_evidence: bool = True,
) -> tuple[list[tuple[np.ndarray, str]], dict[str, Any]]:
    first = rows.iloc[0]
    combined = normalize_rgb(tifffile.imread(str(first["combined_raw_path"])))
    dead_channel = normalize_scalar(tifffile.imread(str(first["dead_raw_path"])))
    cells = read_labels(str(first["cell_mask_path"]))
    live_ids = label_ids(
        rows,
        rows["final_state"].astype(str).eq("live"),
    )
    baseline_dead_ids = label_ids(rows, rows["current_dead_call"])
    rescued_ids = label_ids(rows, rows["late_death_rescue_call"])
    final_dead_ids = label_ids(rows, rows["final_dead_call"])
    residual_live_ids = label_ids(
        rows,
        rows["final_state"].astype(str).eq("live") & ~rows["final_dead_call"],
    )
    artifact_ids = label_ids(
        rows,
        rows["final_state"].astype(str).eq("artifact"),
    )
    uncertain_ids = label_ids(rows, rows["late_death_uncertain"])
    strong_live_ids = label_ids(rows, rows["strong_live_evidence"])
    evidence_ids = label_ids(rows, rows["late_dead_object_evidence"])

    baseline = tint(combined, cells, live_ids, LIVE_COLOR)
    baseline = tint(baseline, cells, baseline_dead_ids, DEAD_COLOR)
    baseline = tint(baseline, cells, artifact_ids, ARTIFACT_COLOR)
    final = tint(combined, cells, residual_live_ids, LIVE_COLOR)
    final = tint(final, cells, final_dead_ids, DEAD_COLOR)
    final = tint(final, cells, rescued_ids, DEAD_COLOR)
    final = tint(final, cells, uncertain_ids, UNCERTAIN_COLOR)
    final = tint(final, cells, artifact_ids, ARTIFACT_COLOR)
    evidence = tint(combined, cells, evidence_ids, DEAD_COLOR)
    evidence = tint(evidence, cells, strong_live_ids, LIVE_COLOR)
    panels = [
        (combined, "Combined RGB · raw"),
        (dead_channel, "Dead channel · raw"),
        (
            baseline,
            "Previous classification",
        ),
        (
            final,
            "Final classification",
        ),
    ]
    if include_evidence:
        panels.append(
            (
                evidence,
                "Late-death object evidence",
            )
        )
    if include_nuclei:
        nuclei = read_labels(str(first["nucleus_core_mask_path"]))
        nuclei_overlay = combined.copy()
        nuclei_overlay[
            find_boundaries(nuclei > 0, mode="outer")
        ] = np.asarray(
            (0, 255, 255),
            dtype=np.uint8,
        )
        nuclei_overlay = tint(
            nuclei_overlay,
            cells,
            final_dead_ids,
            DEAD_COLOR,
        )
        panels.append(
            (
                nuclei_overlay,
                "Nuclei with final dead boundaries",
            )
        )
    countable = rows["countable"] & ~rows["border_touching"]
    baseline_dead = int(rows.loc[countable, "current_dead_call"].sum())
    final_dead = int(rows.loc[countable, "final_dead_call"].sum())
    denominator = int(countable.sum())
    global_state = bool(rows["field_global_late_death"].iloc[0])
    statistics = {
        "first": first,
        "countable_objects": denominator,
        "baseline_dead": baseline_dead,
        "final_dead": final_dead,
        "global_late_death": global_state,
        "rescued": len(rescued_ids),
        "residual_live": len(residual_live_ids),
        "uncertain": len(uncertain_ids),
    }
    return panels, statistics


def render_field(rows: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    panel_images, statistics = render_field_panel_images(rows, include_nuclei=True)
    first = statistics["first"]
    denominator = int(statistics["countable_objects"])
    baseline_dead = int(statistics["baseline_dead"])
    final_dead = int(statistics["final_dead"])
    global_state = bool(statistics["global_late_death"])
    footer = [
        (
            f"key={first['key']} | baseline_dead={baseline_dead}/{denominator} "
            f"({baseline_dead / max(denominator, 1):.1%}) | final_dead={final_dead}/{denominator} "
            f"({final_dead / max(denominator, 1):.1%})"
        ),
        (
            f"global_late_death={global_state} | rescued={statistics['rescued']} | "
            f"residual_live={statistics['residual_live']} | uncertain={statistics['uncertain']}"
        ),
        "Operational proxy QC; the percentages are not manually annotated biological accuracy.",
    ]
    panels = [panel(image, title) for image, title in panel_images]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compose(panels, footer).save(out_path, optimize=True)
    return {
        "branch": str(first["branch"]),
        "cohort": str(first["cohort"]),
        "key": str(first["key"]),
        "well": str(first["well"]),
        "site": int(first["site"]),
        "elapsed_hours": float(first["elapsed_hours"]),
        "global_late_death": global_state,
        "countable_objects": denominator,
        "baseline_dead": baseline_dead,
        "final_dead": final_dead,
        "rescued": int(statistics["rescued"]),
        "residual_live": int(statistics["residual_live"]),
        "uncertain": int(statistics["uncertain"]),
        "qc_path": str(out_path),
    }


def plot_timecourses(summary: pd.DataFrame, out_path: Path) -> None:
    subset = summary.loc[
        summary["branch"].eq("original") & summary["well"].isin(["E9", "F9"])
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row_index, well in enumerate(("E9", "F9")):
        for column_index, metric in enumerate(
            ("baseline_dead_fraction", "final_dead_fraction")
        ):
            axis = axes[row_index, column_index]
            rows = subset.loc[subset["well"].eq(well)]
            for site, site_rows in rows.groupby("site"):
                axis.plot(
                    site_rows["elapsed_hours"],
                    site_rows[metric],
                    linewidth=1.4,
                    label=f"site {int(site)}",
                )
            axis.set_title(
                f"{well}: {'current' if metric.startswith('baseline') else 'refined'} death fraction"
            )
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Elapsed hours")
            if column_index == 0:
                axis.set_ylabel("Death fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_e9_feature_map(predictions: pd.DataFrame, out_path: Path) -> None:
    rows = predictions.loc[
        predictions["branch"].eq("original")
        & predictions["key"].eq("E9_1_05d00h00m")
        & predictions["countable"]
        & ~predictions["border_touching"]
    ].copy()
    nc = np.maximum(
        rows["core_nc_percentile"].to_numpy(float),
        rows["extent_nc_percentile"].to_numpy(float),
    )
    red = np.maximum(
        rows["red_mass_depletion_percentile"].to_numpy(float),
        rows["cytoplasm_red_mass_depletion_percentile"].to_numpy(float),
    )
    colors = np.where(
        rows["late_death_rescue_call"],
        "#c218d4",
        np.where(rows["current_dead_call"], "#d62728", "#2ca02c"),
    )
    fig, axis = plt.subplots(figsize=(7.5, 6))
    axis.scatter(red, nc, c=colors, s=18, alpha=0.7, edgecolors="none")
    axis.set_xlabel("Red-mass depletion percentile")
    axis.set_ylabel("Nucleus-to-cytoplasm percentile")
    axis.set_title("E9_1 day 5: object evidence and final classification")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.text(
        0.02,
        0.02,
        "red=current dead; magenta=rescued; green=residual live",
        transform=axis.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(
        index=False,
        border=0,
        classes="data-table",
        float_format=lambda value: f"{value:.6g}",
    )


def build_html(
    out_path: Path,
    configuration: dict[str, Any],
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    report_rows = inventory.loc[
        inventory["well"].isin(["E9", "F9"])
        & inventory["elapsed_hours"].isin([72.0, 96.0, 120.0, 144.0])
    ].head(40)
    figures = "\n".join(
        (
            f'<figure><a href="{html.escape(Path(row.qc_path).relative_to(out_path.parent).as_posix())}">'
            f'<img src="{html.escape(Path(row.qc_path).relative_to(out_path.parent).as_posix())}" loading="lazy"></a>'
            f"<figcaption>{html.escape(str(row.key))}: current {int(row.baseline_dead)}/"
            f"{int(row.countable_objects)}, refined {int(row.final_dead)}/"
            f"{int(row.countable_objects)}, rescued {int(row.rescued)}.</figcaption></figure>"
        )
        for row in report_rows.itertuples(index=False)
    )
    e9_day5 = summary.loc[
        summary["branch"].eq("original")
        & summary["well"].eq("E9")
        & summary["elapsed_hours"].eq(120.0)
    ]
    f9_day5 = summary.loc[
        summary["branch"].eq("original")
        & summary["well"].eq("F9")
        & summary["elapsed_hours"].eq(120.0)
    ]
    metrics = pd.DataFrame(
        [
            {
                "group": "E9 original day 5",
                "fields": len(e9_day5),
                "current_death_fraction_median": e9_day5[
                    "baseline_dead_fraction"
                ].median(),
                "refined_death_fraction_median": e9_day5[
                    "final_dead_fraction"
                ].median(),
                "refined_death_fraction_min": e9_day5[
                    "final_dead_fraction"
                ].min(),
            },
            {
                "group": "F9 original day 5",
                "fields": len(f9_day5),
                "current_death_fraction_median": f9_day5[
                    "baseline_dead_fraction"
                ].median(),
                "refined_death_fraction_median": f9_day5[
                    "final_dead_fraction"
                ].median(),
                "refined_death_fraction_min": f9_day5[
                    "final_dead_fraction"
                ].min(),
            },
        ]
    )
    approved = configuration["production_integration"] == "approved"
    status_class = "pass" if approved else "blocked"
    status_text = (
        "APPROVED by the declared operational gates"
        if approved
        else "BLOCKED: at least one operational gate failed"
    )
    css = """
body{font-family:Arial,sans-serif;max-width:1500px;margin:0 auto;padding:28px;color:#20242a}
h1,h2,h3{color:#17365d}.notice{padding:14px;background:#fff4ce;border-left:5px solid #e0a800}
.pass{padding:14px;background:#e8f5e9;border-left:5px solid #2e7d32}.blocked{padding:14px;background:#ffebee;border-left:5px solid #c62828}
.data-table{border-collapse:collapse;width:100%;font-size:12px}.data-table th,.data-table td{border:1px solid #ddd;padding:6px}.data-table th{background:#eef3f8}
figure{margin:24px 0;border:1px solid #ddd;padding:12px;background:#fafafa}figure img{width:100%;height:auto}figcaption{margin-top:8px}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart-grid img{width:100%;height:auto;border:1px solid #ddd}
code,pre{background:#f4f4f4;padding:3px 5px}pre{white-space:pre-wrap}
"""
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Late-death trajectory refinement</title><style>{css}</style></head><body>
<h1>Late-death trajectory refinement</h1>
<div class="notice"><strong>Interpretation boundary:</strong> all rates are operational anchor metrics. Without manual object-level ground truth, they are not true biological sensitivity or specificity.</div>
<h2>Decision</h2><div class="{status_class}"><strong>{status_text}</strong></div>
<h2>Why the previous method failed</h2>
<p>The previous model used repeated outputs from the same blue-centered classifier as stable-live labels and limited static recovery to 0.5% per field. In E9_1 day 5 this allowed only one new call despite a field-wide collapse phenotype.</p>
<h2>Method</h2><ol>{''.join(f'<li>{html.escape(str(value))}</li>' for value in configuration['decision_layers'])}</ol>
<h2>Day-5 comparison</h2>{table_html(metrics)}
<div class="chart-grid"><img src="charts/e9_f9_timecourses.png"><img src="charts/e9_day5_feature_map.png"></div>
<h2>Operational anchor metrics</h2><pre>{html.escape(json.dumps(configuration['anchor_metrics'], indent=2, sort_keys=True))}</pre>
<h2>Selected parameters</h2>
<h3>Field collapse</h3><pre>{html.escape(json.dumps(configuration['field_configuration'], indent=2, sort_keys=True))}</pre>
<h3>Object evidence</h3><pre>{html.escape(json.dumps(configuration['object_configuration'], indent=2, sort_keys=True))}</pre>
<h2>QC panels</h2>{figures}
<h2>Exhaustive QC inventories</h2>
<p>The machine-readable inventory lists every rendered E9 field and, when requested, all 320 frozen d0 fields: <code>qc_inventory.csv</code>.</p>
</body></html>"""
    out_path.write_text(body)


def main() -> int:
    args = parse_args()
    args.optimization_root = args.optimization_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    configuration = json.loads(
        (args.optimization_root / "best_configuration.json").read_text()
    )
    predictions = pd.read_csv(
        args.optimization_root / "late_death_predictions.csv.gz"
    )
    summary = pd.read_csv(
        args.optimization_root / "late_death_field_summary.csv"
    )
    for column in (
        "countable",
        "border_touching",
        "current_dead_call",
        "late_death_rescue_call",
        "final_dead_call",
        "late_death_uncertain",
        "strong_live_evidence",
        "late_dead_object_evidence",
        "field_global_late_death",
    ):
        predictions[column] = bool_series(predictions[column])
    qc_wells = {
        value.strip() for value in args.qc_wells.split(",") if value.strip()
    }
    key_hours = {
        float(value.strip())
        for value in args.key_hours.split(",")
        if value.strip()
    }
    field_rows = predictions[
        ["branch", "cohort", "key", "well", "site", "elapsed_hours"]
    ].drop_duplicates()
    selected = field_rows.loc[
        field_rows["branch"].eq("original")
        & field_rows["well"].isin(qc_wells)
        & field_rows["elapsed_hours"].isin(key_hours)
    ].copy()
    if args.render_all_e9:
        selected = pd.concat(
            [
                selected,
                field_rows.loc[
                    field_rows["branch"].eq("original")
                    & field_rows["well"].eq("E9")
                ],
            ],
            ignore_index=True,
        ).drop_duplicates(["branch", "key"])
    if args.render_all_d0:
        selected = pd.concat(
            [
                selected,
                field_rows.loc[
                    field_rows["branch"].eq("original")
                    & field_rows["cohort"].eq("d0_frozen")
                ],
            ],
            ignore_index=True,
        ).drop_duplicates(["branch", "key"])
    selected = selected.sort_values(["cohort", "well", "site", "elapsed_hours"])
    selected_pairs = pd.MultiIndex.from_frame(
        selected[["branch", "key"]].drop_duplicates()
    )
    prediction_pairs = pd.MultiIndex.from_frame(predictions[["branch", "key"]])
    selected_predictions = predictions.loc[
        prediction_pairs.isin(selected_pairs)
    ].copy()
    grouped_predictions = {
        (branch, key): rows.copy()
        for (branch, key), rows in selected_predictions.groupby(
            ["branch", "key"],
            sort=False,
        )
    }
    selected_rows = list(selected.itertuples(index=False))

    def render_selected(row: Any) -> dict[str, Any]:
        rows = grouped_predictions[(row.branch, row.key)]
        category = "d0" if row.cohort == "d0_frozen" else row.well
        out_path = (
            args.out_dir
            / "qc"
            / category
            / f"{row.branch}__{row.key}__late_death_qc.png"
        )
        return render_field(rows, out_path)

    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, rendered in enumerate(
            executor.map(render_selected, selected_rows),
            start=1,
        ):
            inventory.append(rendered)
            if (
                completed == 1
                or completed % 25 == 0
                or completed == len(selected_rows)
            ):
                row = selected_rows[completed - 1]
                print(
                    f"rendered_qc={completed}/{len(selected_rows)} key={row.key}",
                    flush=True,
                )
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(args.out_dir / "qc_inventory.csv", index=False)
    plot_timecourses(summary, args.out_dir / "charts" / "e9_f9_timecourses.png")
    plot_e9_feature_map(
        predictions,
        args.out_dir / "charts" / "e9_day5_feature_map.png",
    )
    report_path = args.out_dir / "LATE_DEATH_TRAJECTORY_REFINEMENT_REPORT.html"
    build_html(report_path, configuration, summary, inventory_frame)
    print(f"qc_inventory={args.out_dir / 'qc_inventory.csv'}")
    print(f"html_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
