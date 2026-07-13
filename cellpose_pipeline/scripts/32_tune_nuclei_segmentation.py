#!/usr/bin/env python3
"""Tune conservative nuclear segmentation on fluorescence/BF/Combined evidence.

The search separates network inference settings from mask reconstruction settings.
Cellpose flows are therefore computed once per image/preprocessing/diameter and reused
for several cell-probability, flow-QC, and minimum-area combinations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from cellpose import dynamics, models
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nuclei_segmentation_utils import (
    NucleusCoreConfig,
    build_nucleus_core_seeds,
    cell_overlap_metrics,
    centroid_match_metrics,
    gaussian_blur,
    robust_rescale,
    smooth_sharpen,
    summarize_instances,
)


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
DEFAULT_KEYS = (
    "H10_2_01d12h00m",
    "A11_1_00d12h00m",
    "F7_1_01d14h00m",
    "A4_2_05d20h00m",
    "E3_3_04d06h00m",
    "C2_2_04d04h00m",
)


@dataclass(frozen=True)
class InferenceConfig:
    tag: str
    model: str = "cpsam_v2"
    diameter: float = 24.0
    normalization: str = "percentile"
    low_percentile: float = 0.5
    high_percentile: float = 99.9
    fixed_low_value: float = 8.0
    fixed_high_value: float = 160.0
    preprocess: str = "percentile"
    background_sigma: float = 0.0
    tophat_kernel: int = 0
    blur_sigma: float = 0.0
    smooth_radius: float = 0.0
    sharpen_radius: float = 0.0


@dataclass(frozen=True)
class MaskConfig:
    tag: str
    cellprob_threshold: float
    flow_threshold: float
    min_size: int


DEFAULT_INFERENCE_CONFIGS = (
    InferenceConfig(tag="pct0p5_99p9_d24"),
    InferenceConfig(tag="pct0p5_99p9_d22", diameter=22),
    InferenceConfig(
        tag="bandpass_s2_r8_d24",
        preprocess="smooth_sharpen",
        smooth_radius=2.0,
        sharpen_radius=8.0,
    ),
    InferenceConfig(
        tag="bg18_blur1_d24",
        preprocess="background",
        background_sigma=18.0,
        blur_sigma=1.0,
    ),
    InferenceConfig(
        tag="bg24_blur1_d22",
        diameter=22,
        preprocess="background",
        background_sigma=24.0,
        blur_sigma=1.0,
    ),
    InferenceConfig(
        tag="tophat41_blur1_d24",
        preprocess="tophat",
        tophat_kernel=41,
        blur_sigma=1.0,
    ),
    InferenceConfig(
        tag="fixed8_160_bandpass_s2_r8_d24",
        normalization="fixed",
        preprocess="smooth_sharpen",
        smooth_radius=2.0,
        sharpen_radius=8.0,
    ),
)


DEFAULT_MASK_CONFIGS = (
    MaskConfig("cp-2p75_f0_min5", -2.75, 0.0, 5),
    MaskConfig("cp-2p0_f0_min15", -2.0, 0.0, 15),
    MaskConfig("cp-1p5_f0_min20", -1.5, 0.0, 20),
    MaskConfig("cp-1p0_f0_min15", -1.0, 0.0, 15),
    MaskConfig("cp0_f0_min15", 0.0, 0.0, 15),
    MaskConfig("cp-1p5_f0p4_min20", -1.5, 0.4, 20),
    MaskConfig("cp-1p0_f0p4_min25", -1.0, 0.4, 25),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune nuclear Cellpose masks using multichannel QC evidence.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--inference-config-json", type=Path)
    parser.add_argument("--mask-config-json", type=Path)
    parser.add_argument("--inference-tags", nargs="*")
    parser.add_argument("--mask-tags", nargs="*")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--center-match-distance", type=float, default=6.0)
    parser.add_argument("--core-min-extent-area", type=int, default=15)
    parser.add_argument("--core-min-area", type=int, default=5)
    parser.add_argument("--core-bg-margin", type=int, default=10)
    parser.add_argument("--core-snr", type=float, default=2.0)
    parser.add_argument("--core-object-quantile", type=float, default=0.35)
    parser.add_argument("--core-erode-px", type=int, default=1)
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract key from {path}")
    return match.group(1)


def index_images(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for image_path in sorted(path.glob("*.tif")):
        result[extract_key(image_path)] = image_path
    return result


def index_masks(run_root: Path, profile: str) -> dict[str, Path]:
    mask_dir = run_root / profile / "segmentations"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing {profile} masks: {mask_dir}")
    return {extract_key(path): path for path in sorted(mask_dir.glob("*_cp_masks.tif"))}


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {mask.shape}")
    return mask.astype(np.int32, copy=False)


def load_configs(args: argparse.Namespace) -> tuple[list[InferenceConfig], list[MaskConfig]]:
    if args.inference_config_json:
        inference = [InferenceConfig(**row) for row in json.loads(args.inference_config_json.read_text())]
    else:
        inference = list(DEFAULT_INFERENCE_CONFIGS)
    if args.mask_config_json:
        masks = [MaskConfig(**row) for row in json.loads(args.mask_config_json.read_text())]
    else:
        masks = list(DEFAULT_MASK_CONFIGS)
    if args.inference_tags:
        selected = set(args.inference_tags)
        inference = [config for config in inference if config.tag in selected]
    if args.mask_tags:
        selected = set(args.mask_tags)
        masks = [config for config in masks if config.tag in selected]
    if not inference or not masks:
        raise SystemExit("No inference or mask configurations remain after filtering")
    return inference, masks


def normalize_raw(raw: np.ndarray, config: InferenceConfig) -> np.ndarray:
    data = raw.astype(np.float32, copy=False)
    if config.normalization == "fixed":
        low, high = float(config.fixed_low_value), float(config.fixed_high_value)
    elif config.normalization == "percentile":
        low, high = np.percentile(data, [config.low_percentile, config.high_percentile])
    else:
        raise ValueError(f"Unsupported normalization: {config.normalization}")
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def prepare_image(raw: np.ndarray, config: InferenceConfig) -> np.ndarray:
    normalized = normalize_raw(raw, config)
    if config.preprocess == "percentile":
        prepared = normalized
    elif config.preprocess == "background":
        prepared = robust_rescale(normalized - gaussian_blur(normalized, config.background_sigma), 1.0, 99.8)
    elif config.preprocess == "tophat":
        size = int(config.tophat_kernel)
        size = size + 1 if size % 2 == 0 else size
        array = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        prepared = robust_rescale(cv2.morphologyEx(array, cv2.MORPH_TOPHAT, kernel), 1.0, 99.8)
    elif config.preprocess == "smooth_sharpen":
        prepared = smooth_sharpen(normalized, config.smooth_radius, config.sharpen_radius)
    else:
        raise ValueError(f"Unsupported preprocess: {config.preprocess}")
    if config.blur_sigma > 0:
        prepared = gaussian_blur(prepared, config.blur_sigma)
    return np.clip(prepared, 0.0, 1.0).astype(np.float32, copy=False)


def boundary(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    return foreground & ~cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def write_preview(
    raw: np.ndarray,
    extent: np.ndarray,
    core: np.ndarray,
    out_path: Path,
    scale: float,
) -> None:
    gray = np.clip(robust_rescale(raw, 0.5, 99.8) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[boundary(extent)] = np.array([255, 45, 45], dtype=np.uint8)
    rgb[boundary(core)] = np.array([35, 225, 235], dtype=np.uint8)
    image = Image.fromarray(rgb)
    if 0 < scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_density_calls(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            row["key"]: row.get("high_density", "").strip().lower() in {"1", "true", "yes"}
            for row in csv.DictReader(handle)
        }


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def median(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return float(np.median(values)) if values else float("nan")


def percentile(rows: list[dict[str, Any]], field: str, q: float) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return float(np.percentile(values, q)) if values else float("nan")


def score_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate"], []).append(row)

    scores: list[dict[str, Any]] = []
    for candidate, candidate_rows in grouped.items():
        low = [row for row in candidate_rows if not row["high_density"]]
        high = [row for row in candidate_rows if row["high_density"]]
        center_f1 = median(candidate_rows, "center_f1")
        center_recall = median(candidate_rows, "center_recall")
        center_recall_p10 = percentile(candidate_rows, "center_recall", 10)
        core_center_f1 = median(candidate_rows, "core_center_f1")
        core_center_recall = median(candidate_rows, "core_center_recall")
        core_center_recall_p10 = percentile(candidate_rows, "core_center_recall", 10)
        area_ratio = median(candidate_rows, "extent_area_to_baseline")
        mask_ratio = median(candidate_rows, "extent_fraction_to_baseline")
        high_count_ratio = median(high or candidate_rows, "core_count_to_baseline_core")
        all_core_count_ratio_p10 = percentile(candidate_rows, "core_count_to_baseline_core", 10)
        low_cell_error_values = [
            abs(math.log(max(float(row["count_to_cell_mean"]), 1e-6)))
            for row in (low or candidate_rows)
            if math.isfinite(float(row["count_to_cell_mean"]))
        ]
        low_cell_error = float(np.median(low_cell_error_values)) if low_cell_error_values else 0.0
        crossing = 0.5 * (
            median(candidate_rows, "extent_combined_crossing_rate")
            + median(candidate_rows, "extent_brightfield_crossing_rate")
        )
        low_core_inside = 0.5 * (
            median(low or candidate_rows, "core_combined_centroid_inside_rate")
            + median(low or candidate_rows, "core_brightfield_centroid_inside_rate")
        )
        tiny = median(candidate_rows, "extent_tiny_fraction_lt35")
        target_area_ratio = 0.78
        score = (
            3.0 * (1.0 - center_f1)
            + 4.0 * max(0.0, 0.92 - center_recall)
            + 4.0 * max(0.0, 0.90 - center_recall_p10)
            + 2.0 * (1.0 - core_center_f1)
            + 4.0 * max(0.0, 0.92 - core_center_recall)
            + 5.0 * max(0.0, 0.90 - core_center_recall_p10)
            + 1.0 * abs(math.log(max(area_ratio, 1e-6) / target_area_ratio))
            + 0.5 * abs(math.log(max(mask_ratio, 1e-6) / target_area_ratio))
            + 1.0 * low_cell_error
            + 1.2 * crossing
            + 0.6 * (1.0 - low_core_inside)
            + 0.8 * tiny
            + 3.0 * max(0.0, 0.90 - high_count_ratio)
            + 1.5 * max(0.0, high_count_ratio - 1.08)
            + 5.0 * max(0.0, 0.90 - all_core_count_ratio_p10)
        )
        first = candidate_rows[0]
        scores.append(
            {
                "candidate": candidate,
                "inference_tag": first["inference_tag"],
                "mask_tag": first["mask_tag"],
                "n_fields": len(candidate_rows),
                "score": score,
                "center_f1_median": center_f1,
                "center_recall_median": center_recall,
                "center_recall_p10": center_recall_p10,
                "core_center_f1_median": core_center_f1,
                "core_center_recall_median": core_center_recall,
                "core_center_recall_p10": core_center_recall_p10,
                "extent_area_to_baseline_median": area_ratio,
                "extent_fraction_to_baseline_median": mask_ratio,
                "high_count_to_baseline_median": high_count_ratio,
                "all_core_count_to_baseline_p10": all_core_count_ratio_p10,
                "low_count_to_cell_log_error_median": low_cell_error,
                "extent_crossing_rate_mean_masks": crossing,
                "low_core_centroid_inside_rate_mean_masks": low_core_inside,
                "tiny_fraction_lt35_median": tiny,
                "core_to_extent_fraction_median": median(candidate_rows, "median_core_to_extent_fraction"),
            }
        )
    return sorted(scores, key=lambda row: float(row["score"]))


def write_contact_sheet(
    candidate: str,
    candidate_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    paths = [Path(row["preview_path"]) for row in candidate_rows if row.get("preview_path")]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    columns = 2
    rows = math.ceil(len(images) / columns)
    label_height = 36
    sheet = Image.new("RGB", (columns * width, 40 + rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), candidate, fill="black", font=font)
    for index, (image, row) in enumerate(zip(images, candidate_rows)):
        grid_row, grid_col = divmod(index, columns)
        x = grid_col * width
        y = 40 + grid_row * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text(
            (x + 5, y + 5),
            f"{row['key']} n={row['extent_n_objects']} area={float(row['extent_area_median']):.1f}",
            fill="black",
            font=font,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    for image in images:
        image.close()


def main() -> None:
    args = parse_args()
    inference_configs, mask_configs = load_configs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "inference_configs.json").write_text(
        json.dumps([asdict(config) for config in inference_configs], indent=2) + "\n"
    )
    (args.out_dir / "mask_configs.json").write_text(
        json.dumps([asdict(config) for config in mask_configs], indent=2) + "\n"
    )

    raw_index = index_images(args.input_root / "Nuclei")
    baseline_nuclei = index_masks(args.baseline_run, "Nuclei")
    combined_index = index_masks(args.baseline_run, "Combined")
    brightfield_index = index_masks(args.baseline_run, "Brightfield")
    common_keys = sorted(set(raw_index) & set(baseline_nuclei) & set(combined_index) & set(brightfield_index))
    selected_keys = common_keys if args.all_images else [key for key in args.keys if key in common_keys]
    missing = sorted(set(args.keys) - set(selected_keys)) if not args.all_images else []
    if missing:
        raise SystemExit(f"Requested keys are missing one or more inputs: {missing}")
    if not selected_keys:
        raise SystemExit("No common fields selected")

    density = read_density_calls(args.baseline_run / "qc" / "density_calls.csv")
    core_config = NucleusCoreConfig(
        min_extent_area=args.core_min_extent_area,
        min_core_area=args.core_min_area,
        local_bg_margin=args.core_bg_margin,
        snr_threshold=args.core_snr,
        object_quantile=args.core_object_quantile,
        erode_px=args.core_erode_px,
    )

    print(f"selected_keys={selected_keys}", flush=True)
    print(f"n_inference_configs={len(inference_configs)}", flush=True)
    print(f"n_mask_configs={len(mask_configs)}", flush=True)
    model_cache: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    summary_path = args.out_dir / "field_metrics.csv"
    for inference_index, inference_config in enumerate(inference_configs, start=1):
        model = model_cache.get(inference_config.model)
        if model is None:
            model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=inference_config.model)
            model_cache[inference_config.model] = model
            print(f"model={inference_config.model} device={model.device}", flush=True)

        for key_index, key in enumerate(selected_keys, start=1):
            raw = np.squeeze(tifffile.imread(raw_index[key]))
            baseline = read_mask(baseline_nuclei[key])
            combined = read_mask(combined_index[key])
            brightfield = read_mask(brightfield_index[key])
            prepared = prepare_image(raw, inference_config)
            print(
                f"inference {inference_index}/{len(inference_configs)} field {key_index}/{len(selected_keys)} "
                f"tag={inference_config.tag} key={key}",
                flush=True,
            )
            start = time.time()
            _empty, flows, _styles = model.eval(
                prepared,
                channel_axis=None,
                normalize=False,
                diameter=inference_config.diameter,
                flow_threshold=0.0,
                cellprob_threshold=-2.75,
                min_size=5,
                compute_masks=False,
            )
            inference_seconds = time.time() - start
            dP, cellprob = flows[1], flows[2]
            baseline_stats = summarize_instances(baseline)
            baseline_core, _baseline_core_diagnostics = build_nucleus_core_seeds(raw, baseline, core_config)
            baseline_core_stats = summarize_instances(baseline_core)
            combined_stats = summarize_instances(combined)
            brightfield_stats = summarize_instances(brightfield)
            cell_mean = 0.5 * (combined_stats["n_objects"] + brightfield_stats["n_objects"])
            niter = max(1, int(round(200.0 * inference_config.diameter / 30.0)))

            for mask_config in mask_configs:
                start = time.time()
                extent = dynamics.resize_and_compute_masks(
                    dP,
                    cellprob,
                    niter=niter,
                    cellprob_threshold=mask_config.cellprob_threshold,
                    flow_threshold=mask_config.flow_threshold,
                    min_size=mask_config.min_size,
                    device=model.device,
                ).astype(np.int32, copy=False)
                mask_seconds = time.time() - start
                core, core_diagnostics = build_nucleus_core_seeds(raw, extent, core_config)
                extent_stats = summarize_instances(extent)
                core_stats = summarize_instances(core)
                center_stats = centroid_match_metrics(baseline, extent, args.center_match_distance)
                core_center_stats = centroid_match_metrics(baseline_core, core, args.center_match_distance)
                extent_combined = cell_overlap_metrics(extent, combined)
                extent_brightfield = cell_overlap_metrics(extent, brightfield)
                core_combined = cell_overlap_metrics(core, combined)
                core_brightfield = cell_overlap_metrics(core, brightfield)
                candidate = f"{inference_config.tag}__{mask_config.tag}"
                preview_path = args.out_dir / "previews" / candidate / f"{key}.png"
                if args.save_previews:
                    write_preview(raw, extent, core, preview_path, args.preview_scale)
                else:
                    preview_path = Path("")
                extent_path = args.out_dir / "masks" / "extent" / candidate / f"{raw_index[key].stem}_cp_masks.tif"
                core_path = args.out_dir / "masks" / "core" / candidate / f"{raw_index[key].stem}_core_masks.tif"
                if args.save_masks:
                    extent_path.parent.mkdir(parents=True, exist_ok=True)
                    core_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(extent_path, extent.astype(np.uint16 if extent.max() < 65535 else np.uint32), compression="zlib")
                    tifffile.imwrite(core_path, core.astype(np.uint16 if core.max() < 65535 else np.uint32), compression="zlib")
                else:
                    extent_path = Path("")
                    core_path = Path("")

                row: dict[str, Any] = {
                    "candidate": candidate,
                    "inference_tag": inference_config.tag,
                    "mask_tag": mask_config.tag,
                    "key": key,
                    "high_density": bool(density.get(key, False)),
                    "raw_path": str(raw_index[key]),
                    "inference_seconds": inference_seconds,
                    "mask_seconds": mask_seconds,
                    **asdict(inference_config),
                    "mask_cellprob_threshold": mask_config.cellprob_threshold,
                    "mask_flow_threshold": mask_config.flow_threshold,
                    "mask_min_size": mask_config.min_size,
                    **prefixed("extent", extent_stats),
                    **prefixed("core", core_stats),
                    **center_stats,
                    **prefixed("core", core_center_stats),
                    **core_diagnostics,
                    **prefixed("extent_combined", extent_combined),
                    **prefixed("extent_brightfield", extent_brightfield),
                    **prefixed("core_combined", core_combined),
                    **prefixed("core_brightfield", core_brightfield),
                    "baseline_nuclei_count": baseline_stats["n_objects"],
                    "baseline_nuclei_area_median": baseline_stats["area_median"],
                    "baseline_nuclei_mask_fraction": baseline_stats["mask_fraction"],
                    "baseline_core_count": baseline_core_stats["n_objects"],
                    "baseline_core_mask_fraction": baseline_core_stats["mask_fraction"],
                    "combined_cell_count": combined_stats["n_objects"],
                    "brightfield_cell_count": brightfield_stats["n_objects"],
                    "count_to_baseline": safe_ratio(extent_stats["n_objects"], baseline_stats["n_objects"]),
                    "core_count_to_baseline_core": safe_ratio(core_stats["n_objects"], baseline_core_stats["n_objects"]),
                    "count_to_cell_mean": safe_ratio(extent_stats["n_objects"], cell_mean),
                    "extent_area_to_baseline": safe_ratio(extent_stats["area_median"], baseline_stats["area_median"]),
                    "extent_fraction_to_baseline": safe_ratio(extent_stats["mask_fraction"], baseline_stats["mask_fraction"]),
                    "preview_path": str(preview_path) if str(preview_path) != "." else "",
                    "extent_mask_path": str(extent_path) if str(extent_path) != "." else "",
                    "core_mask_path": str(core_path) if str(core_path) != "." else "",
                }
                result_rows.append(row)
                print(
                    f"candidate={candidate} key={key} n={extent_stats['n_objects']} "
                    f"area={extent_stats['area_median']:.1f} center_f1={center_stats['center_f1']:.3f}",
                    flush=True,
                )
            write_rows(summary_path, result_rows)

    write_rows(summary_path, result_rows)
    score_rows = score_candidates(result_rows)
    scores_path = args.out_dir / "config_scores.csv"
    write_rows(scores_path, score_rows)

    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        rows_by_candidate.setdefault(row["candidate"], []).append(row)
    if args.save_previews:
        for score_row in score_rows[:12]:
            candidate = score_row["candidate"]
            write_contact_sheet(candidate, rows_by_candidate[candidate], args.out_dir / "contact_sheets" / f"{candidate}.png")

    report = [
        "# Nuclear segmentation tuning report",
        "",
        f"- input root: `{args.input_root}`",
        f"- baseline run: `{args.baseline_run}`",
        f"- fields: {len(selected_keys)}",
        f"- inference configurations: {len(inference_configs)}",
        f"- mask configurations per inference: {len(mask_configs)}",
        f"- core configuration: `{asdict(core_config)}`",
        "",
        "## Top candidates",
        "",
        "| rank | candidate | score | extent center F1 | core recall | core recall P10 | area/baseline | high core count/baseline | core count P10 | crossing | core/extent |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(score_rows[:15], start=1):
        report.append(
            f"| {rank} | {row['candidate']} | {float(row['score']):.4f} | "
            f"{float(row['center_f1_median']):.3f} | {float(row['core_center_recall_median']):.3f} | "
            f"{float(row['core_center_recall_p10']):.3f} | "
            f"{float(row['extent_area_to_baseline_median']):.3f} | "
            f"{float(row['high_count_to_baseline_median']):.3f} | "
            f"{float(row['all_core_count_to_baseline_p10']):.3f} | "
            f"{float(row['extent_crossing_rate_mean_masks']):.3f} | "
            f"{float(row['core_to_extent_fraction_median']):.3f} |"
        )
    report.extend(
        [
            "",
            "The score is a calibration heuristic, not manual ground truth. It favors retention of baseline nuclear centers,",
            "a 15-30% reduction of inflated extent area, stable high-density counts, fewer cell-boundary crossings,",
            "low fragment rates, and conservative core seeds inside BF/Combined cell masks.",
            "",
            "Overlay colors: red = nuclear extent boundary; cyan = conservative core-seed boundary.",
        ]
    )
    report_path = args.out_dir / "tuning_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"field_metrics={summary_path}", flush=True)
    print(f"config_scores={scores_path}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
