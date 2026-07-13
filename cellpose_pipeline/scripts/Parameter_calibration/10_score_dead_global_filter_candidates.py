#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def load_tune_module() -> Any:
    script = Path(__file__).with_name("09_largetest_dead_global_filter_tune.py")
    spec = importlib.util.spec_from_file_location("dead_global_filter_tune", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing Dead global-filter candidate masks without rerunning Cellpose.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing candidate_masks/ from script 23.")
    parser.add_argument("--write-key-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune = load_tune_module()
    images = tune.iter_dead_images(args.input_root)
    baseline = tune.read_baseline_summary(args.baseline_run)
    density_bins = tune.density_bins_from_baseline(baseline)
    selected_keys = {tune.F4_LOW_SIGNAL_KEY, tune.A11_ARTIFACT_KEY}
    selected_keys.update(key for key, density in density_bins.items() if density == "high")

    nuclei_masks: dict[str, np.ndarray] = {}
    for (folder, key), row in baseline.items():
        if folder == "Nuclei":
            mask_path = Path(row["mask_path"])
            if mask_path.exists():
                nuclei_masks[key] = tifffile.imread(mask_path)

    rows: list[dict[str, Any]] = []
    by_combined: dict[str, list[dict[str, Any]]] = {}

    for seg_config in tune.DEFAULT_SEG_CONFIGS:
        for image_path in images:
            key = tune.extract_key(image_path.name)
            raw = tifffile.imread(image_path).astype(np.float32, copy=False)
            candidate_path = args.run_root / "candidate_masks" / seg_config.tag / f"{image_path.stem}_candidate_masks.tif"
            if not candidate_path.exists():
                raise FileNotFoundError(candidate_path)
            candidate = tifffile.imread(candidate_path)
            candidate_stats = tune.summarize_mask(candidate)
            nuc = baseline.get(("Nuclei", key), {})
            nuclei_count = int(float(nuc.get("n_objects", 0) or 0))
            candidate_ratio = candidate_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")
            base_records = tune.object_records(raw, candidate, nuclei_masks.get(key), tune.DEFAULT_FILTER_CONFIGS[0])
            prepared = None
            raw_base = None
            for filter_config in tune.DEFAULT_FILTER_CONFIGS:
                combined_tag = f"{seg_config.tag}__{filter_config.tag}"
                records = tune.apply_filter(base_records, filter_config)
                filtered = tune.relabel_filtered(candidate.astype(np.int32, copy=False), records)
                filtered_stats = tune.summarize_mask(filtered)
                filtered_ratio = filtered_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")
                mask_path = args.run_root / "filtered_masks" / combined_tag / f"{image_path.stem}_filtered_masks.tif"
                global_overlay_path = args.run_root / "overlays_global" / combined_tag / f"{image_path.stem}_overlay.png"
                raw_overlay_path = args.run_root / "overlays_raw_fullrange" / combined_tag / f"{image_path.stem}_overlay.png"

                if args.write_key_overlays and key in selected_keys:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    global_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(mask_path, filtered.astype(np.uint16 if filtered.max() < 65535 else np.uint32, copy=False))
                    if prepared is None:
                        prepared = tune.fixed_normalize(raw, seg_config)
                    if raw_base is None:
                        raw_base = tune.raw_display(raw)
                    tune.make_overlay(prepared, filtered).save(global_overlay_path)
                    tune.make_overlay(raw_base, filtered).save(raw_overlay_path)

                roi_stats = (
                    tune.a11_artifact_roi_stats(filtered)
                    if key == tune.A11_ARTIFACT_KEY
                    else {"a11_artifact_roi_mask_fraction": 0.0, "a11_artifact_roi_objects": 0}
                )
                row = {
                    "stage": args.run_root.name,
                    "combined_config": combined_tag,
                    "seg_config": seg_config.tag,
                    "filter_config": filter_config.tag,
                    "key": key,
                    "density_bin": density_bins.get(key, "unknown"),
                    "image": image_path.name,
                    **{f"seg_{k}": v for k, v in tune.asdict(seg_config).items()},
                    **{f"filter_{k}": v for k, v in tune.asdict(filter_config).items()},
                    "baseline_nuclei_count": nuclei_count,
                    "candidate_n_objects": candidate_stats["n_objects"],
                    "candidate_mask_fraction": candidate_stats["mask_fraction"],
                    "candidate_object_to_nuclei_ratio": f"{candidate_ratio:.6f}",
                    "filtered_n_objects": filtered_stats["n_objects"],
                    "filtered_mask_fraction": filtered_stats["mask_fraction"],
                    "filtered_area_median": filtered_stats["area_median"],
                    "filtered_area_p90": filtered_stats["area_p90"],
                    "filtered_max_area": filtered_stats["max_area"],
                    "filtered_large_fraction_gt1800": filtered_stats["large_fraction_gt1800"],
                    "filtered_large_fraction_gt2500": filtered_stats["large_fraction_gt2500"],
                    "filtered_tiny_fraction_lt20": filtered_stats["tiny_fraction_lt20"],
                    "filtered_object_to_nuclei_ratio": f"{filtered_ratio:.6f}",
                    "kept_fraction_of_candidates": f"{filtered_stats['n_objects'] / candidate_stats['n_objects']:.6f}" if candidate_stats["n_objects"] else "0.000000",
                    "raw_frac_gt10": float((raw > 10).mean()),
                    "raw_frac_gt12": float((raw > 12).mean()),
                    "raw_frac_gt15": float((raw > 15).mean()),
                    "raw_frac_gt20": float((raw > 20).mean()),
                    **roi_stats,
                    "mask_path": str(mask_path),
                    "global_overlay_path": str(global_overlay_path),
                    "raw_overlay_path": str(raw_overlay_path),
                    "object_stats_path": "",
                }
                rows.append(row)
                by_combined.setdefault(combined_tag, []).append(row)

    tune.write_rows(args.run_root / "summary.csv", rows)
    score_table = [tune.score_rows(config_rows) for config_rows in by_combined.values()]
    score_table.sort(key=lambda row: float(row["score"]))
    tune.write_rows(args.run_root / "config_scores.csv", score_table)

    if args.make_contact_sheets:
        for score_row in score_table[:16]:
            tag = score_row["combined_config"]
            raw_paths = [
                Path(row["raw_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in selected_keys and Path(row["raw_overlay_path"]).exists()
            ]
            tune.write_contact_sheet(
                raw_paths,
                args.run_root / "contact_sheets_raw_fullrange" / f"{tag}_F4_A11_high.png",
                f"{tag} raw full range F4 A11 high",
            )
            global_paths = [
                Path(row["global_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in selected_keys and Path(row["global_overlay_path"]).exists()
            ]
            tune.write_contact_sheet(
                global_paths,
                args.run_root / "contact_sheets_global" / f"{tag}_F4_A11_high.png",
                f"{tag} fixed global display F4 A11 high",
            )

    lines = [
        "# Dead Global-Filter Candidate Score Report",
        "",
        f"- input_root: `{args.input_root}`",
        f"- baseline_run: `{args.baseline_run}`",
        f"- run_root: `{args.run_root}`",
        f"- rows: {len(rows)}",
        "",
        "| rank | config | score | F4 mask | F4 ratio | A11 count | A11 artifact ROI | high mask p75 | strong count median |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(score_table[:16], start=1):
        lines.append(
            f"| {idx} | {row['combined_config']} | {row['score']} | {row['f4_mask_fraction']} | "
            f"{row['f4_object_ratio']} | {row['a11_count']} | {row['a11_artifact_roi_mask_fraction']} | "
            f"{row['high_mask_fraction_p75']} | {row['strong_signal_count_median']} |"
        )
    (args.run_root / "retune_report.md").write_text("\n".join(lines) + "\n")
    print(f"summary={args.run_root / 'summary.csv'}", flush=True)
    print(f"scores={args.run_root / 'config_scores.csv'}", flush=True)
    print(f"report={args.run_root / 'retune_report.md'}", flush=True)


if __name__ == "__main__":
    main()
