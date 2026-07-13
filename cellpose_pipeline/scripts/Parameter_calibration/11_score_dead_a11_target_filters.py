#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


A11_TARGET_KEY = "A11_1_00d12h00m"
F4_CONTROL_KEY = "F4_1_02d00h00m"
G8_HIGH_DEAD_KEY = "G8_1_03d10h00m"
E3_HIGH_DEAD_KEY = "E3_3_04d06h00m"
KEY_OVERLAY_SET = {A11_TARGET_KEY, F4_CONTROL_KEY, G8_HIGH_DEAD_KEY, E3_HIGH_DEAD_KEY}


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
    parser = argparse.ArgumentParser(
        description="Retune Dead object-level filters to target lower A11 false positives without rerunning Cellpose."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", default="dead_a11_target_filters")
    parser.add_argument("--seg-tag", action="append", default=["g55_d22_cp-3p0"])
    parser.add_argument("--a11-target-count", type=float, default=20.0)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_filter_configs(tune: Any) -> list[Any]:
    FilterConfig = tune.FilterConfig
    return [
        FilterConfig(
            "weak_raw_area2500",
            max_area=2500,
            min_mean_delta=0.20,
            min_p90_delta=0.70,
            min_p90_abs=10.0,
            min_snr=1.25,
        ),
        FilterConfig(
            "strict_area2500",
            max_area=2500,
            min_mean_delta=0.45,
            min_p90_delta=1.50,
            min_p90_abs=11.0,
            min_snr=2.25,
        ),
        FilterConfig(
            "a11_target_p11p5_s2p75",
            max_area=2500,
            min_mean_delta=0.55,
            min_p90_delta=1.80,
            min_p90_abs=11.5,
            min_snr=2.75,
        ),
        FilterConfig(
            "a11_target_p12_s3p0",
            max_area=2500,
            min_mean_delta=0.60,
            min_p90_delta=2.00,
            min_p90_abs=12.0,
            min_snr=3.00,
        ),
        FilterConfig(
            "a11_target_p12p5_s3p25",
            max_area=2500,
            min_mean_delta=0.70,
            min_p90_delta=2.20,
            min_p90_abs=12.5,
            min_snr=3.25,
        ),
        FilterConfig(
            "a11_target_p13_s3p5",
            max_area=2500,
            min_mean_delta=0.80,
            min_p90_delta=2.50,
            min_p90_abs=13.0,
            min_snr=3.50,
        ),
        FilterConfig(
            "a11_target_p15_s3p0",
            max_area=2500,
            min_mean_delta=0.60,
            min_p90_delta=2.00,
            min_p90_abs=15.0,
            min_snr=3.00,
        ),
        FilterConfig(
            "a11_target_p12_area1800",
            max_area=1800,
            min_mean_delta=0.60,
            min_p90_delta=2.00,
            min_p90_abs=12.0,
            min_snr=3.00,
        ),
        FilterConfig(
            "a11_target_p13_area1800",
            max_area=1800,
            min_mean_delta=0.80,
            min_p90_delta=2.50,
            min_p90_abs=13.0,
            min_snr=3.50,
        ),
    ]


def custom_score(config_rows: list[dict[str, Any]], a11_target_count: float) -> dict[str, Any]:
    def one_value(key: str, field: str) -> float:
        values = [float(row[field]) for row in config_rows if row["key"] == key]
        return float(np.median(values)) if values else float("nan")

    first = config_rows[0]
    a11_count = one_value(A11_TARGET_KEY, "filtered_n_objects")
    a11_mask = one_value(A11_TARGET_KEY, "filtered_mask_fraction")
    a11_roi = one_value(A11_TARGET_KEY, "a11_artifact_roi_mask_fraction")
    f4_count = one_value(F4_CONTROL_KEY, "filtered_n_objects")
    f4_mask = one_value(F4_CONTROL_KEY, "filtered_mask_fraction")
    g8_count = one_value(G8_HIGH_DEAD_KEY, "filtered_n_objects")
    g8_mask = one_value(G8_HIGH_DEAD_KEY, "filtered_mask_fraction")
    e3_count = one_value(E3_HIGH_DEAD_KEY, "filtered_n_objects")
    e3_mask = one_value(E3_HIGH_DEAD_KEY, "filtered_mask_fraction")

    a11_target_penalty = abs(a11_count - a11_target_count) / 20.0
    a11_artifact_penalty = a11_roi * 20.0 + max(0.0, a11_mask - 0.004) * 20.0
    f4_penalty = max(0.0, f4_count - 25.0) / 25.0 + max(0.0, f4_mask - 0.003) * 20.0
    high_signal_penalty = max(0.0, 380.0 - g8_count) / 160.0 + max(0.0, 1050.0 - e3_count) / 350.0
    score = a11_target_penalty + a11_artifact_penalty + f4_penalty + high_signal_penalty

    return {
        "combined_config": first["combined_config"],
        "seg_config": first["seg_config"],
        "filter_config": first["filter_config"],
        "score": f"{score:.6f}",
        "a11_count": f"{a11_count:.3f}",
        "a11_mask_fraction": f"{a11_mask:.6f}",
        "a11_artifact_roi_mask_fraction": f"{a11_roi:.6f}",
        "f4_count": f"{f4_count:.3f}",
        "f4_mask_fraction": f"{f4_mask:.6f}",
        "g8_count": f"{g8_count:.3f}",
        "g8_mask_fraction": f"{g8_mask:.6f}",
        "e3_count": f"{e3_count:.3f}",
        "e3_mask_fraction": f"{e3_mask:.6f}",
    }


def main() -> None:
    args = parse_args()
    tune = load_tune_module()
    out_root = args.out_root.resolve() / args.stage
    out_root.mkdir(parents=True, exist_ok=True)

    seg_configs = [config for config in tune.DEFAULT_SEG_CONFIGS if config.tag in set(args.seg_tag)]
    if not seg_configs:
        raise SystemExit(f"No matching seg configs for {args.seg_tag}")
    filter_configs = target_filter_configs(tune)
    (out_root / "filter_configs.json").write_text(json.dumps([asdict(config) for config in filter_configs], indent=2) + "\n")

    images = tune.iter_dead_images(args.input_root)
    baseline = tune.read_baseline_summary(args.baseline_run)
    density_bins = tune.density_bins_from_baseline(baseline)

    nuclei_masks: dict[str, np.ndarray] = {}
    for (folder, key), row in baseline.items():
        if folder == "Nuclei":
            mask_path = Path(row["mask_path"])
            if mask_path.exists():
                nuclei_masks[key] = tifffile.imread(mask_path)

    rows: list[dict[str, Any]] = []
    by_combined: dict[str, list[dict[str, Any]]] = {}

    for seg_config in seg_configs:
        for image_path in images:
            key = tune.extract_key(image_path.name)
            raw = tifffile.imread(image_path).astype(np.float32, copy=False)
            candidate_path = args.candidate_run_root / "candidate_masks" / seg_config.tag / f"{image_path.stem}_candidate_masks.tif"
            if not candidate_path.exists():
                raise FileNotFoundError(candidate_path)
            candidate = tifffile.imread(candidate_path)
            candidate_stats = tune.summarize_mask(candidate)
            base_records = tune.object_records(raw, candidate, nuclei_masks.get(key), filter_configs[0])
            prepared = tune.fixed_normalize(raw, seg_config)
            raw_base = tune.raw_display(raw)

            nuc = baseline.get(("Nuclei", key), {})
            nuclei_count = int(float(nuc.get("n_objects", 0) or 0))
            candidate_ratio = candidate_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")

            for filter_config in filter_configs:
                combined_tag = f"{seg_config.tag}__{filter_config.tag}"
                records = tune.apply_filter(base_records, filter_config)
                filtered = tune.relabel_filtered(candidate.astype(np.int32, copy=False), records)
                filtered_stats = tune.summarize_mask(filtered)
                filtered_ratio = filtered_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")

                mask_path = out_root / "filtered_masks" / combined_tag / f"{image_path.stem}_filtered_masks.tif"
                global_overlay_path = out_root / "overlays_global" / combined_tag / f"{image_path.stem}_overlay.png"
                raw_overlay_path = out_root / "overlays_raw_fullrange" / combined_tag / f"{image_path.stem}_overlay.png"
                if key in KEY_OVERLAY_SET:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    global_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(mask_path, filtered.astype(np.uint16 if filtered.max() < 65535 else np.uint32, copy=False))
                    tune.make_overlay(prepared, filtered).save(global_overlay_path)
                    tune.make_overlay(raw_base, filtered).save(raw_overlay_path)

                roi_stats = (
                    tune.a11_artifact_roi_stats(filtered)
                    if key == A11_TARGET_KEY
                    else {"a11_artifact_roi_mask_fraction": 0.0, "a11_artifact_roi_objects": 0}
                )
                row = {
                    "stage": args.stage,
                    "combined_config": combined_tag,
                    "seg_config": seg_config.tag,
                    "filter_config": filter_config.tag,
                    "key": key,
                    "density_bin": density_bins.get(key, "unknown"),
                    "image": image_path.name,
                    **{f"seg_{k}": v for k, v in asdict(seg_config).items()},
                    **{f"filter_{k}": v for k, v in asdict(filter_config).items()},
                    "baseline_nuclei_count": nuclei_count,
                    "candidate_n_objects": candidate_stats["n_objects"],
                    "candidate_mask_fraction": candidate_stats["mask_fraction"],
                    "candidate_object_to_nuclei_ratio": f"{candidate_ratio:.6f}",
                    "filtered_n_objects": filtered_stats["n_objects"],
                    "filtered_mask_fraction": filtered_stats["mask_fraction"],
                    "filtered_area_median": filtered_stats["area_median"],
                    "filtered_area_p90": filtered_stats["area_p90"],
                    "filtered_max_area": filtered_stats["max_area"],
                    "filtered_object_to_nuclei_ratio": f"{filtered_ratio:.6f}",
                    "raw_frac_gt10": float((raw > 10).mean()),
                    "raw_frac_gt12": float((raw > 12).mean()),
                    "raw_frac_gt15": float((raw > 15).mean()),
                    "raw_frac_gt20": float((raw > 20).mean()),
                    **roi_stats,
                    "mask_path": str(mask_path),
                    "global_overlay_path": str(global_overlay_path),
                    "raw_overlay_path": str(raw_overlay_path),
                }
                rows.append(row)
                by_combined.setdefault(combined_tag, []).append(row)

    write_rows(out_root / "summary.csv", rows)
    score_rows = [custom_score(config_rows, args.a11_target_count) for config_rows in by_combined.values()]
    score_rows.sort(key=lambda row: float(row["score"]))
    write_rows(out_root / "config_scores.csv", score_rows)

    if args.make_contact_sheets:
        for score_row in score_rows[:8]:
            tag = score_row["combined_config"]
            raw_paths = [
                Path(row["raw_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in KEY_OVERLAY_SET and Path(row["raw_overlay_path"]).exists()
            ]
            tune.write_contact_sheet(
                raw_paths,
                out_root / "contact_sheets_raw_fullrange" / f"{tag}_four_key.png",
                f"{tag} raw full range four key Dead controls",
            )
            global_paths = [
                Path(row["global_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in KEY_OVERLAY_SET and Path(row["global_overlay_path"]).exists()
            ]
            tune.write_contact_sheet(
                global_paths,
                out_root / "contact_sheets_global" / f"{tag}_four_key.png",
                f"{tag} fixed global display four key Dead controls",
            )

    report = [
        "# Dead A11 Target Filter Report",
        "",
        f"- input_root: `{args.input_root}`",
        f"- baseline_run: `{args.baseline_run}`",
        f"- candidate_run_root: `{args.candidate_run_root}`",
        f"- out_root: `{out_root}`",
        f"- a11_target_count: {args.a11_target_count}",
        "",
        "| rank | config | score | A11 count | F4 count | G8 count | E3 count | A11 ROI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(score_rows[:12], start=1):
        report.append(
            f"| {idx} | {row['combined_config']} | {row['score']} | {row['a11_count']} | "
            f"{row['f4_count']} | {row['g8_count']} | {row['e3_count']} | {row['a11_artifact_roi_mask_fraction']} |"
        )
    (out_root / "retune_report.md").write_text("\n".join(report) + "\n")

    print(f"summary={out_root / 'summary.csv'}", flush=True)
    print(f"scores={out_root / 'config_scores.csv'}", flush=True)
    print(f"report={out_root / 'retune_report.md'}", flush=True)


if __name__ == "__main__":
    main()
