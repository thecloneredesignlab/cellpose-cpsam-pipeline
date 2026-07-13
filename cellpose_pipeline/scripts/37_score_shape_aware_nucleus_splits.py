#!/usr/bin/env python3
"""Score shape-aware nucleus-split candidates without manual ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.spatial import cKDTree


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing-data-only nucleus split candidates.")
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-count-increase-fraction", type=float, default=0.01)
    parser.add_argument("--max-multinucleation-rate-delta", type=float, default=0.005)
    parser.add_argument("--max-concordance-loss", type=float, default=0.01)
    parser.add_argument("--max-actionable-rate-increase", type=float, default=0.001)
    parser.add_argument("--min-median-stability", type=float, default=0.80)
    parser.add_argument("--min-p10-stability", type=float, default=0.75)
    parser.add_argument("--min-median-shape-gain", type=float, default=0.03)
    parser.add_argument("--min-both-cell-supported-fraction", type=float, default=0.50)
    parser.add_argument("--min-splits-for-selection", type=int, default=5)
    return parser.parse_args()


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(path)
    return match.group(1)


def index_masks(run_root: Path, core: bool = False) -> dict[str, Path]:
    directory = (
        run_root / "Nuclei" / "nucleus_core_seeds"
        if core
        else run_root / "Nuclei" / "segmentations"
    )
    pattern = "*_core_masks.tif" if core else "*_cp_masks.tif"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        key = key_from_path(path)
        if key in result:
            raise ValueError(f"Duplicate field {key} in {directory}")
        result[key] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {labels.shape}")
    return labels.astype(np.int32, copy=False)


def json_load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label_count(labels: np.ndarray) -> int:
    counts = np.bincount(labels.ravel())
    return int(np.count_nonzero(counts[1:]))


def core_density_metrics(labels: np.ndarray) -> dict[str, Any]:
    yy, xx = np.nonzero(labels)
    if not yy.size:
        return {"count": 0, "median_nn": float("inf"), "high_density": False}
    ids = labels[yy, xx].astype(np.int64, copy=False)
    maximum = int(ids.max())
    areas = np.bincount(ids, minlength=maximum + 1).astype(np.float64)
    sum_y = np.bincount(ids, weights=yy, minlength=maximum + 1)
    sum_x = np.bincount(ids, weights=xx, minlength=maximum + 1)
    valid = np.flatnonzero(areas > 0)
    valid = valid[valid > 0]
    centroids = np.column_stack((sum_y[valid] / areas[valid], sum_x[valid] / areas[valid]))
    if len(centroids) >= 2:
        distances, _indices = cKDTree(centroids).query(centroids, k=2)
        median_nn = float(np.median(distances[:, 1]))
    else:
        median_nn = float("inf")
    count = len(valid)
    return {
        "count": count,
        "median_nn": median_nn,
        "high_density": count >= 4000 or median_nn <= 16.0,
    }


def summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    alignment = summary["alignment"]
    combined = summary["cells"]["Combined"]
    brightfield = summary["cells"]["Brightfield"]
    concordance = summary["cross_method_cell_concordance"]
    return {
        "nuclei": int(summary["nuclei"]["total"]),
        "quality_nuclei": int(summary["nuclei"]["quality_nuclei"]),
        "fragment_suspects": int(summary["nuclei"]["fragment_suspects"]),
        "shape_irregular": int(summary["nuclei"]["shape_irregular"]),
        "median_area": float(summary["nuclei"]["area_px2"]["median"]),
        "actionable_rate": float(alignment["actionable_mismatch_rate"]),
        "strict_rate": float(alignment["strict_consensus_rate"]),
        "combined_multinucleated_rate": float(combined["confirmed_multinucleated_rate"]),
        "bf_multinucleated_rate": float(brightfield["confirmed_multinucleated_rate"]),
        "combined_extent_nc": float(combined["median_nucleus_to_cytoplasm_ratio"]),
        "bf_extent_nc": float(brightfield["median_nucleus_to_cytoplasm_ratio"]),
        "combined_core_nc": float(combined["median_core_nucleus_to_cytoplasm_ratio"]),
        "bf_core_nc": float(brightfield["median_core_nucleus_to_cytoplasm_ratio"]),
        "exact_concordance": float(concordance["exact_status_concordance_rate"]),
        "multinucleation_jaccard": float(concordance["confirmed_multinucleation_jaccard"]),
        "validation_ok": all(bool(value) for value in summary["validation"].values()),
    }


def quantile(rows: list[dict[str, str]], field: str, q: float) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return float(np.quantile(values, q)) if values.size else 0.0


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = json_load(args.baseline_summary)
    baseline_metrics = summary_metrics(baseline_summary)
    baseline_extent_paths = index_masks(args.baseline_run_root)
    baseline_core_paths = index_masks(args.baseline_run_root, core=True)
    all_events = csv_rows(args.screen_root / "split_events.csv")
    candidate_dirs = {
        path.name: path
        for path in sorted(args.candidates_root.iterdir())
        if path.is_dir() and (path / "alignment" / "summary.json").is_file()
    }
    if not candidate_dirs:
        raise SystemExit("No validated candidates")

    baseline_density = {
        key: core_density_metrics(read_mask(path)) for key, path in baseline_core_paths.items()
    }
    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for candidate, root in candidate_dirs.items():
        summary = json_load(root / "alignment" / "summary.json")
        metrics = summary_metrics(summary)
        split_summary = json_load(root / "split_summary.json")
        events = [row for row in all_events if row["candidate"] == candidate]
        extent_paths = index_masks(root)
        core_paths = index_masks(root, core=True)
        if set(extent_paths) != set(baseline_extent_paths) or set(core_paths) != set(baseline_core_paths):
            raise RuntimeError(f"Field-key mismatch for {candidate}")
        foreground_equal = True
        all_contiguous = True
        density_flip_keys: list[str] = []
        total_candidate_core = 0
        total_baseline_core = 0
        for key in sorted(extent_paths):
            baseline_extent = read_mask(baseline_extent_paths[key])
            candidate_extent = read_mask(extent_paths[key])
            baseline_core = read_mask(baseline_core_paths[key])
            candidate_core = read_mask(core_paths[key])
            equal = bool(np.array_equal(baseline_extent > 0, candidate_extent > 0))
            contiguous = label_count(candidate_extent) == int(candidate_extent.max())
            foreground_equal &= equal
            all_contiguous &= contiguous
            base_density = baseline_density[key]
            current_density = core_density_metrics(candidate_core)
            total_baseline_core += int(base_density["count"])
            total_candidate_core += int(current_density["count"])
            if bool(base_density["high_density"]) != bool(current_density["high_density"]):
                density_flip_keys.append(key)
            field_rows.append(
                {
                    "candidate": candidate,
                    "key": key,
                    "baseline_nuclei": label_count(baseline_extent),
                    "candidate_nuclei": label_count(candidate_extent),
                    "nuclei_added": label_count(candidate_extent) - label_count(baseline_extent),
                    "extent_foreground_equal": equal,
                    "candidate_labels_contiguous": contiguous,
                    "baseline_core_count": int(base_density["count"]),
                    "candidate_core_count": int(current_density["count"]),
                    "baseline_median_nn": float(base_density["median_nn"]),
                    "candidate_median_nn": float(current_density["median_nn"]),
                    "baseline_high_density": bool(base_density["high_density"]),
                    "candidate_high_density": bool(current_density["high_density"]),
                }
            )

        n_splits = len(events)
        child_below35 = sum(
            min(int(row["retained_child_area"]), int(row["new_child_area"])) < 35 for row in events
        )
        both_cell_supported_fraction = (
            float(np.mean([float(row["cell_support_fraction"]) >= 1.0 for row in events]))
            if events
            else 0.0
        )
        core_added = total_candidate_core - total_baseline_core
        row: dict[str, Any] = {
            "candidate": candidate,
            "n_split_parents": n_splits,
            "count_increase_fraction": n_splits / baseline_metrics["nuclei"],
            "core_objects_added": core_added,
            "core_added_per_split": core_added / n_splits if n_splits else 0.0,
            "split_children_below35": child_below35,
            "extent_foreground_equal": foreground_equal,
            "all_labels_contiguous": all_contiguous,
            "density_call_flips": len(density_flip_keys),
            "density_flip_keys": ";".join(density_flip_keys),
            "median_peak_distance": quantile(events, "peak_distance", 0.50),
            "median_min_peak_snr": quantile(events, "min_peak_snr", 0.50),
            "median_valley_drop_snr": quantile(events, "valley_drop_snr", 0.50),
            "median_valley_drop_fraction": quantile(events, "valley_drop_fraction", 0.50),
            "median_stability_fraction": quantile(events, "stability_fraction", 0.50),
            "p10_stability_fraction": quantile(events, "stability_fraction", 0.10),
            "median_shape_gain": quantile(events, "shape_gain", 0.50),
            "p10_shape_gain": quantile(events, "shape_gain", 0.10),
            "both_cell_supported_fraction": both_cell_supported_fraction,
            "same_cell_pair_fraction": (
                float(np.mean([str(row["children_same_cell_pair"]).lower() == "true" for row in events]))
                if events
                else 0.0
            ),
            "actionable_rate": metrics["actionable_rate"],
            "actionable_rate_delta_pp": 100.0 * (
                metrics["actionable_rate"] - baseline_metrics["actionable_rate"]
            ),
            "strict_rate": metrics["strict_rate"],
            "strict_rate_delta_pp": 100.0 * (metrics["strict_rate"] - baseline_metrics["strict_rate"]),
            "shape_irregular": metrics["shape_irregular"],
            "shape_irregular_delta": metrics["shape_irregular"] - baseline_metrics["shape_irregular"],
            "fragment_suspect_delta": metrics["fragment_suspects"] - baseline_metrics["fragment_suspects"],
            "combined_multinucleated_rate": metrics["combined_multinucleated_rate"],
            "combined_multinucleated_rate_delta_pp": 100.0 * (
                metrics["combined_multinucleated_rate"]
                - baseline_metrics["combined_multinucleated_rate"]
            ),
            "bf_multinucleated_rate": metrics["bf_multinucleated_rate"],
            "bf_multinucleated_rate_delta_pp": 100.0 * (
                metrics["bf_multinucleated_rate"] - baseline_metrics["bf_multinucleated_rate"]
            ),
            "exact_concordance_rate": metrics["exact_concordance"],
            "exact_concordance_delta_pp": 100.0 * (
                metrics["exact_concordance"] - baseline_metrics["exact_concordance"]
            ),
            "multinucleation_jaccard": metrics["multinucleation_jaccard"],
            "multinucleation_jaccard_delta_pp": 100.0 * (
                metrics["multinucleation_jaccard"] - baseline_metrics["multinucleation_jaccard"]
            ),
            "combined_extent_nc_delta": metrics["combined_extent_nc"] - baseline_metrics["combined_extent_nc"],
            "bf_extent_nc_delta": metrics["bf_extent_nc"] - baseline_metrics["bf_extent_nc"],
            "combined_core_nc_delta": metrics["combined_core_nc"] - baseline_metrics["combined_core_nc"],
            "bf_core_nc_delta": metrics["bf_core_nc"] - baseline_metrics["bf_core_nc"],
            "analysis_validation_ok": metrics["validation_ok"],
            "screen_split_count": int(split_summary.get("n_split_parents", 0)),
        }
        reasons: list[str] = []
        if n_splits <= 0:
            reasons.append("no split candidates")
        if not metrics["validation_ok"]:
            reasons.append("alignment validation failed")
        if not foreground_equal:
            reasons.append("nuclear foreground changed")
        if not all_contiguous:
            reasons.append("non-contiguous labels")
        if n_splits / baseline_metrics["nuclei"] > args.max_count_increase_fraction:
            reasons.append("count increase above limit")
        if child_below35 > 0:
            reasons.append("split child below 35 px")
        if core_added < n_splits:
            reasons.append("one or more split children lack an intensity core")
        if density_flip_keys:
            reasons.append("density call changed")
        if row["median_stability_fraction"] < args.min_median_stability:
            reasons.append("median peak stability below limit")
        if row["p10_stability_fraction"] < args.min_p10_stability:
            reasons.append("P10 peak stability below limit")
        if row["median_shape_gain"] < args.min_median_shape_gain:
            reasons.append("median shape gain below limit")
        if both_cell_supported_fraction < args.min_both_cell_supported_fraction:
            reasons.append("two-child BF/Combined support below limit")
        if metrics["actionable_rate"] > baseline_metrics["actionable_rate"] + args.max_actionable_rate_increase:
            reasons.append("actionable mismatch increased")
        if abs(metrics["combined_multinucleated_rate"] - baseline_metrics["combined_multinucleated_rate"]) > args.max_multinucleation_rate_delta:
            reasons.append("Combined multinucleation shift above limit")
        if abs(metrics["bf_multinucleated_rate"] - baseline_metrics["bf_multinucleated_rate"]) > args.max_multinucleation_rate_delta:
            reasons.append("BF multinucleation shift above limit")
        if metrics["exact_concordance"] < baseline_metrics["exact_concordance"] - args.max_concordance_loss:
            reasons.append("exact concordance loss")
        if metrics["multinucleation_jaccard"] < baseline_metrics["multinucleation_jaccard"] - args.max_concordance_loss:
            reasons.append("multinucleation Jaccard loss")
        row["eligible"] = not reasons
        row["guardrail_failures"] = "; ".join(reasons)
        row["evidence_score"] = (
            float(row["median_shape_gain"])
            + 0.10 * float(row["median_valley_drop_fraction"])
            + 0.08 * float(row["both_cell_supported_fraction"])
            + 0.05 * float(row["median_stability_fraction"])
        )
        rows.append(row)

    selectable = [
        row
        for row in rows
        if row["eligible"] and int(row["n_split_parents"]) >= args.min_splits_for_selection
    ]
    selected = max(
        selectable,
        key=lambda row: (float(row["evidence_score"]), int(row["n_split_parents"])),
        default=None,
    )
    for row in rows:
        row["recommended"] = selected is not None and row["candidate"] == selected["candidate"]
    rows.sort(key=lambda row: (not bool(row["eligible"]), -float(row["evidence_score"])))
    write_rows(args.out_dir / "candidate_comparison.csv", rows)
    write_rows(args.out_dir / "field_audit.csv", field_rows)
    decision = {
        "selected_candidate": selected["candidate"] if selected else None,
        "selection_rule": (
            "Pass every foreground, core, density, stability, morphology, cell-support, "
            "multinucleation, and concordance guardrail; require at least "
            f"{args.min_splits_for_selection} split parents; then maximize the evidence score."
        ),
        "baseline": baseline_metrics,
        "guardrails": {
            "max_count_increase_fraction": args.max_count_increase_fraction,
            "max_multinucleation_rate_delta": args.max_multinucleation_rate_delta,
            "max_concordance_loss": args.max_concordance_loss,
            "max_actionable_rate_increase": args.max_actionable_rate_increase,
            "min_median_stability": args.min_median_stability,
            "min_p10_stability": args.min_p10_stability,
            "min_median_shape_gain": args.min_median_shape_gain,
            "min_both_cell_supported_fraction": args.min_both_cell_supported_fraction,
        },
    }
    (args.out_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Shape-aware nucleus split validation",
        "",
        "These candidates use only existing images and predicted masks. Evidence scores measure internal robustness, not ground-truth accuracy.",
        "",
        "| candidate | splits | shape gain | valley fraction | stable P10 | both cells supported | actionable delta | multi Jaccard delta | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['n_split_parents']} | {row['median_shape_gain']:.3f} | "
            f"{row['median_valley_drop_fraction']:.1%} | {row['p10_stability_fraction']:.1%} | "
            f"{row['both_cell_supported_fraction']:.1%} | {row['actionable_rate_delta_pp']:+.3f} pp | "
            f"{row['multinucleation_jaccard_delta_pp']:+.3f} pp | {'yes' if row['eligible'] else 'no'} |"
        )
    lines.extend(["", "## Decision", ""])
    if selected is None:
        lines.append("No candidate passed every guardrail with enough split events; retain the baseline nuclei masks.")
    else:
        lines.append(f"Recommended optional candidate: **{selected['candidate']}**. {decision['selection_rule']}")
    failures = [row for row in rows if row["guardrail_failures"]]
    if failures:
        lines.extend(["", "## Guardrail failures", ""])
        for row in failures:
            lines.append(f"- `{row['candidate']}`: {row['guardrail_failures']}")
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "Do not overwrite the production extent/core masks. A selected candidate is suitable only as a sensitivity layer until external truth becomes available.",
        ]
    )
    (args.out_dir / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
