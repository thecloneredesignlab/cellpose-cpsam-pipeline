#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from cellpose import models
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
TIME_KEY_RE = re.compile(r"(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")


@dataclass(frozen=True)
class TuneConfig:
    tag: str
    folder: str
    model: str
    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    low_percentile: float
    high_percentile: float
    preprocess: str = "percentile"
    gamma: float = 1.0
    clahe_clip: float = 2.0
    clahe_tile: int = 16
    blur_sigma: float = 0.0
    background_sigma: float = 0.0
    tophat_kernel: int = 0
    invert: bool = False


DEAD_ROUND1_CONFIGS = [
    TuneConfig("dead_baseline_d30_cp-2p25_min10_pct0p5-99p9", "Dead", "cpsam_v2", 30, 0.0, -2.25, 10, 0.5, 99.9),
    TuneConfig("dead_raw_d26_cp-2p75_min5_pct0p1-99p7", "Dead", "cpsam_v2", 26, 0.0, -2.75, 5, 0.1, 99.7),
    TuneConfig("dead_raw_d26_cp-3p25_min5_pct0p1-99p7", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.1, 99.7),
    TuneConfig("dead_raw_d22_cp-3p25_min5_pct0p1-99p5", "Dead", "cpsam_v2", 22, 0.0, -3.25, 5, 0.1, 99.5),
    TuneConfig("dead_gamma0p65_d26_cp-2p75_min5_pct0p1-99p5", "Dead", "cpsam_v2", 26, 0.0, -2.75, 5, 0.1, 99.5, gamma=0.65),
    TuneConfig("dead_gamma0p65_d26_cp-3p25_min5_pct0p1-99p5", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.1, 99.5, gamma=0.65),
    TuneConfig("dead_gamma0p5_d22_cp-3p25_min5_pct0p1-99p5", "Dead", "cpsam_v2", 22, 0.0, -3.25, 5, 0.1, 99.5, gamma=0.5),
    TuneConfig("dead_clahe16_d26_cp-2p75_min5_pct0p5-99p5", "Dead", "cpsam_v2", 26, 0.0, -2.75, 5, 0.5, 99.5, preprocess="clahe", clahe_clip=2.0, clahe_tile=16),
    TuneConfig("dead_clahe16_d26_cp-3p25_min5_pct0p5-99p5", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.5, 99.5, preprocess="clahe", clahe_clip=2.0, clahe_tile=16),
    TuneConfig("dead_clahe32_d30_cp-2p75_min8_pct0p5-99p5", "Dead", "cpsam_v2", 30, 0.0, -2.75, 8, 0.5, 99.5, preprocess="clahe", clahe_clip=2.0, clahe_tile=32),
    TuneConfig("dead_clahe32_d30_cp-3p25_min8_pct0p5-99p5", "Dead", "cpsam_v2", 30, 0.0, -3.25, 8, 0.5, 99.5, preprocess="clahe", clahe_clip=2.0, clahe_tile=32),
    TuneConfig("dead_clahe16_gamma0p75_d26_cp-3p25_min5", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.5, 99.5, preprocess="clahe", gamma=0.75, clahe_clip=2.0, clahe_tile=16),
    TuneConfig("dead_bg35_d26_cp-2p75_min5_pct0p5-99p7", "Dead", "cpsam_v2", 26, 0.0, -2.75, 5, 0.5, 99.7, preprocess="background", background_sigma=35),
    TuneConfig("dead_bg35_d26_cp-3p25_min5_pct0p5-99p7", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.5, 99.7, preprocess="background", background_sigma=35),
    TuneConfig("dead_bg55_d30_cp-3p25_min5_pct0p5-99p7", "Dead", "cpsam_v2", 30, 0.0, -3.25, 5, 0.5, 99.7, preprocess="background", background_sigma=55),
    TuneConfig("dead_tophat61_d26_cp-2p75_min5_pct0p1-99p8", "Dead", "cpsam_v2", 26, 0.0, -2.75, 5, 0.1, 99.8, preprocess="tophat", tophat_kernel=61),
    TuneConfig("dead_tophat61_d26_cp-3p25_min5_pct0p1-99p8", "Dead", "cpsam_v2", 26, 0.0, -3.25, 5, 0.1, 99.8, preprocess="tophat", tophat_kernel=61),
    TuneConfig("dead_tophat81_gamma0p75_d30_cp-3p25_min5", "Dead", "cpsam_v2", 30, 0.0, -3.25, 5, 0.1, 99.8, preprocess="tophat", gamma=0.75, tophat_kernel=81),
    TuneConfig("dead_cpsam_clahe16_d26_cp-3p25_min5", "Dead", "cpsam", 26, 0.0, -3.25, 5, 0.5, 99.5, preprocess="clahe", clahe_clip=2.0, clahe_tile=16),
    TuneConfig("dead_cpsam_bg35_d26_cp-3p25_min5", "Dead", "cpsam", 26, 0.0, -3.25, 5, 0.5, 99.7, preprocess="background", background_sigma=35),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune Cellpose profiles on SUM159 largetest.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--stage", default="dead_round1")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--folders", nargs="+", default=["Dead"])
    parser.add_argument("--density-source", choices=("baseline_nuclei", "manifest_order"), default="baseline_nuclei")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_key(name: str) -> str:
    match = TIME_KEY_RE.search(name)
    if not match:
        raise ValueError(f"Cannot extract key from {name}")
    return match.group(1)


def iter_images(input_root: Path, folder: str) -> list[Path]:
    root = input_root / folder
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_configs(args: argparse.Namespace) -> list[TuneConfig]:
    if args.config_json:
        payload = json.loads(args.config_json.read_text())
        return [TuneConfig(**row) for row in payload]
    allowed = set(args.folders)
    return [config for config in DEAD_ROUND1_CONFIGS if config.folder in allowed]


def percentile_normalize(image: np.ndarray, low_percentile: float, high_percentile: float, invert: bool) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    low, high = np.percentile(data, [low_percentile, high_percentile])
    if high <= low:
        high = low + 1.0
    norm = np.clip((data - low) / (high - low), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    return norm.astype(np.float32, copy=False)


def apply_clahe(norm: np.ndarray, clip: float, tile: int) -> np.ndarray:
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    out = clahe.apply(arr).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def odd_kernel_from_sigma(sigma: float) -> int:
    k = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    return k if k % 2 else k + 1


def apply_background(norm: np.ndarray, sigma: float) -> np.ndarray:
    k = odd_kernel_from_sigma(sigma)
    background = cv2.GaussianBlur(norm.astype(np.float32), (k, k), float(sigma))
    corrected = norm - background
    corrected -= float(corrected.min())
    high = float(np.percentile(corrected, 99.8))
    if high <= 0:
        high = float(corrected.max()) or 1.0
    return np.clip(corrected / high, 0.0, 1.0).astype(np.float32, copy=False)


def apply_tophat(norm: np.ndarray, kernel: int) -> np.ndarray:
    k = int(kernel)
    if k <= 0:
        return norm
    if k % 2 == 0:
        k += 1
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, struct).astype(np.float32)
    high = float(np.percentile(out, 99.8))
    if high <= 0:
        high = float(out.max()) or 1.0
    return np.clip(out / high, 0.0, 1.0).astype(np.float32, copy=False)


def preprocess_image(image: np.ndarray, config: TuneConfig) -> np.ndarray:
    norm = percentile_normalize(image, config.low_percentile, config.high_percentile, config.invert)
    if config.preprocess == "percentile":
        out = norm
    elif config.preprocess == "clahe":
        out = apply_clahe(norm, config.clahe_clip, config.clahe_tile)
    elif config.preprocess == "background":
        out = apply_background(norm, config.background_sigma)
    elif config.preprocess == "tophat":
        out = apply_tophat(norm, config.tophat_kernel)
    else:
        raise ValueError(f"Unsupported preprocess: {config.preprocess}")
    if config.blur_sigma > 0:
        k = odd_kernel_from_sigma(config.blur_sigma)
        out = cv2.GaussianBlur(out.astype(np.float32), (k, k), float(config.blur_sigma))
    if config.gamma != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), float(config.gamma)).astype(np.float32, copy=False)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    labels = mask.astype(np.int64, copy=False)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def make_overlay(norm: np.ndarray, mask: np.ndarray) -> Image.Image:
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    pixels = mask > 0
    rgb[pixels, 0] = np.maximum(rgb[pixels, 0], 220)
    rgb[pixels, 2] = np.maximum(rgb[pixels, 2], 170)
    rgb[boundary_mask(mask)] = np.array([255, 28, 28], dtype=np.uint8)
    return Image.fromarray(rgb)


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    n_objects = int(mask.max()) if mask.size else 0
    areas = np.bincount(mask.ravel())[1:]
    if areas.size == 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
            "small_fraction_lt50": 0.0,
            "tiny_fraction_lt20": 0.0,
            "large_fraction_gt5000": 0.0,
            "area_cv": 0.0,
        }
    return {
        "n_objects": n_objects,
        "mask_fraction": float((mask > 0).sum() / mask.size),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "area_p90": float(np.percentile(areas, 90)),
        "small_fraction_lt50": float((areas < 50).mean()),
        "tiny_fraction_lt20": float((areas < 20).mean()),
        "large_fraction_gt5000": float((areas > 5000).mean()),
        "area_cv": float(np.std(areas) / (np.mean(areas) + 1e-9)),
    }


def read_baseline_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    summary = path / "segmentation_summary.csv"
    with summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = extract_key(Path(row["image_path"]).name)
            rows[(row["folder_profile"], key)] = row
    return rows


def density_bins_from_baseline(baseline: dict[tuple[str, str], dict[str, Any]]) -> dict[str, str]:
    nuclei = []
    for (folder, key), row in baseline.items():
        if folder == "Nuclei":
            nuclei.append((key, int(float(row["n_objects"]))))
    nuclei.sort(key=lambda item: item[1])
    bins: dict[str, str] = {}
    n = len(nuclei)
    for idx, (key, _count) in enumerate(nuclei):
        if idx < n / 3:
            bins[key] = "low"
        elif idx < 2 * n / 3:
            bins[key] = "middle"
        else:
            bins[key] = "high"
    return bins


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(paths: list[Path], out_path: Path, title: str, columns: int = 4) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 360, 260
    label_h = 42
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h) + 34), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for idx, path in enumerate(paths):
        row, col = divmod(idx, columns)
        x = col * thumb_w
        y = 34 + row * (thumb_h + label_h)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 4), f"{path.parent.name}/{path.stem}"[:64], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def score_config(rows: list[dict[str, Any]], baseline: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    folder = rows[0]["folder"]
    by_density: dict[str, list[dict[str, Any]]] = {"low": [], "middle": [], "high": []}
    for row in rows:
        by_density.setdefault(row["density_bin"], []).append(row)

    high = by_density.get("high", [])
    all_rows = rows
    if folder == "Nuclei":
        ratio_field = "object_to_brightfield_ratio"
        ratio_label = "object_to_brightfield"
        target_ratio = 1.0
        low_ratio_floor = 0.8
        high_mask_p75_limit = 0.35
        high_mask_max_limit = 0.50
    elif folder == "Brightfield":
        ratio_field = "object_to_nuclei_ratio"
        ratio_label = "object_to_nuclei"
        target_ratio = 1.0
        low_ratio_floor = 0.8
        high_mask_p75_limit = 0.85
        high_mask_max_limit = 0.98
    else:
        ratio_field = "object_to_nuclei_ratio"
        ratio_label = "object_to_nuclei"
        target_ratio = 0.25
        low_ratio_floor = 0.12
        high_mask_p75_limit = 0.22
        high_mask_max_limit = 0.42

    high_ratios = [float(r[ratio_field]) for r in high if math.isfinite(float(r[ratio_field]))]
    high_mask_fraction = [float(r["mask_fraction"]) for r in high]
    high_tiny = [float(r["tiny_fraction_lt20"]) for r in high]
    high_large = [float(r["large_fraction_gt5000"]) for r in high]
    all_ratios = [float(r[ratio_field]) for r in all_rows if math.isfinite(float(r[ratio_field]))]

    # Preference: recover high-density Dead objects without making masks cover too much area,
    # without many tiny specks, and without huge saturated blobs dominating.
    high_ratio = median(high_ratios)
    all_ratio = median(all_ratios)
    high_ratio_p25 = percentile(high_ratios, 25)
    high_ratio_p75 = percentile(high_ratios, 75)
    high_mask = median(high_mask_fraction)
    high_mask_p75 = percentile(high_mask_fraction, 75)
    high_mask_max = max(high_mask_fraction) if high_mask_fraction else float("nan")
    f4_rows = [row for row in rows if row["key"] == "F4_1_02d00h00m"]
    f4_mask = median([float(row["mask_fraction"]) for row in f4_rows])
    f4_ratio = median([float(row[ratio_field]) for row in f4_rows if math.isfinite(float(row[ratio_field]))])
    tiny = median(high_tiny)
    large = median(high_large)
    ratio_penalty = abs(high_ratio - target_ratio)
    low_recall_penalty = max(0.0, low_ratio_floor - high_ratio) * 2.0
    over_mask_penalty = max(0.0, high_mask_p75 - high_mask_p75_limit) * 1.5 + max(0.0, high_mask_max - high_mask_max_limit) * 2.5
    if folder == "Dead":
        over_mask_penalty += max(0.0, f4_mask - 0.30) * 4.0
    speck_penalty = tiny * 0.75
    large_penalty = large * 1.5
    score = ratio_penalty + low_recall_penalty + over_mask_penalty + speck_penalty + large_penalty
    return {
        "config": rows[0]["config"],
        "folder": folder,
        "model": rows[0]["model"],
        "n_images": len(rows),
        "score": f"{score:.6f}",
        "ratio_label": ratio_label,
        "target_ratio": f"{target_ratio:.6f}",
        "high_object_ratio_median": f"{high_ratio:.6f}",
        "high_object_ratio_p25": f"{high_ratio_p25:.6f}",
        "high_object_ratio_p75": f"{high_ratio_p75:.6f}",
        "all_object_ratio_median": f"{all_ratio:.6f}",
        "high_mask_fraction_median": f"{high_mask:.6f}",
        "high_mask_fraction_p75": f"{high_mask_p75:.6f}",
        "high_mask_fraction_max": f"{high_mask_max:.6f}",
        "f4_mask_fraction": f"{f4_mask:.6f}",
        "f4_object_ratio": f"{f4_ratio:.6f}",
        "high_tiny_fraction_lt20_median": f"{tiny:.6f}",
        "high_large_fraction_gt5000_median": f"{large:.6f}",
        "count_median": f"{median([float(r['n_objects']) for r in rows]):.3f}",
        "high_count_median": f"{median([float(r['n_objects']) for r in high]):.3f}",
        "area_median_median": f"{median([float(r['area_median']) for r in rows]):.3f}",
    }


def main() -> None:
    args = parse_args()
    configs = load_configs(args)
    out_root = args.out_root.resolve() / args.stage
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "configs.json").write_text(json.dumps([asdict(config) for config in configs], indent=2) + "\n")

    baseline = read_baseline_summary(args.baseline_run)
    density_bins = density_bins_from_baseline(baseline)

    configs_by_model: dict[str, list[TuneConfig]] = {}
    for config in configs:
        configs_by_model.setdefault(config.model, []).append(config)

    rows: list[dict[str, Any]] = []
    by_config_rows: dict[str, list[dict[str, Any]]] = {}
    for model_name, model_configs in configs_by_model.items():
        print(f"loading_model={model_name}", flush=True)
        model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=model_name)
        print(f"model_device={model.device}", flush=True)
        for config in model_configs:
            images = iter_images(args.input_root, config.folder)
            for image_path in images:
                key = extract_key(image_path.name)
                raw = tifffile.imread(image_path)
                mask_path = out_root / "masks" / config.folder / model_name / config.tag / f"{image_path.stem}_masks.tif"
                overlay_path = out_root / "overlays" / config.folder / model_name / config.tag / f"{image_path.stem}_overlay.png"
                prepared_path = out_root / "prepared_previews" / config.folder / model_name / config.tag / f"{image_path.stem}_prepared.png"
                if args.skip_existing and mask_path.exists() and overlay_path.exists():
                    mask = tifffile.imread(mask_path)
                    elapsed = 0.0
                    print(f"skip_existing\t{model_name}\t{config.tag}\t{image_path.name}", flush=True)
                    prepared = preprocess_image(raw, config)
                else:
                    print(f"run\t{model_name}\t{config.tag}\t{image_path.name}", flush=True)
                    prepared = preprocess_image(raw, config)
                    start = time.time()
                    mask, *_ = model.eval(
                        prepared,
                        channel_axis=None,
                        normalize=False,
                        diameter=config.diameter,
                        flow_threshold=config.flow_threshold,
                        cellprob_threshold=config.cellprob_threshold,
                        min_size=config.min_size,
                    )
                    elapsed = time.time() - start
                    mask = np.asarray(mask)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    prepared_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(mask_path, mask.astype(np.uint16 if mask.max() < 65535 else np.uint32, copy=False))
                    make_overlay(prepared, mask).save(overlay_path)
                    Image.fromarray(np.clip(prepared * 255.0, 0, 255).astype(np.uint8)).save(prepared_path)
                stats = summarize_mask(mask)
                nuc = baseline.get(("Nuclei", key), {})
                bf = baseline.get(("Brightfield", key), {})
                nuclei_count = int(float(nuc.get("n_objects", 0) or 0))
                brightfield_count = int(float(bf.get("n_objects", 0) or 0))
                object_to_nuc = stats["n_objects"] / nuclei_count if nuclei_count else float("inf")
                object_to_bf = stats["n_objects"] / brightfield_count if brightfield_count else float("inf")
                row = {
                    "stage": args.stage,
                    "folder": config.folder,
                    "model": model_name,
                    "config": config.tag,
                    "key": key,
                    "density_bin": density_bins.get(key, "unknown"),
                    "image": image_path.name,
                    "elapsed_sec": f"{elapsed:.3f}",
                    **asdict(config),
                    **stats,
                    "baseline_nuclei_count": nuclei_count,
                    "baseline_brightfield_count": brightfield_count,
                    "object_to_nuclei_ratio": f"{object_to_nuc:.6f}",
                    "object_to_brightfield_ratio": f"{object_to_bf:.6f}",
                    "dead_to_nuclei_ratio": f"{object_to_nuc:.6f}",
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "prepared_path": str(prepared_path),
                }
                rows.append(row)
                by_config_rows.setdefault(config.tag, []).append(row)

    write_rows(out_root / "summary.csv", rows)
    score_rows = [score_config(config_rows, baseline) for config_rows in by_config_rows.values()]
    score_rows.sort(key=lambda row: float(row["score"]))
    write_rows(out_root / "config_scores.csv", score_rows)

    if args.make_contact_sheets:
        high_keys = {key for key, density in density_bins.items() if density == "high"}
        selected_keys = set(high_keys) | {"A11_1_00d12h00m"}
        n_sheets = len(score_rows) if len(score_rows) <= 32 else 12
        for score_row in score_rows[:n_sheets]:
            tag = score_row["config"]
            overlay_paths = [
                Path(row["overlay_path"])
                for row in by_config_rows[tag]
                if row["key"] in selected_keys and Path(row["overlay_path"]).exists()
            ]
            write_contact_sheet(
                overlay_paths,
                out_root / "contact_sheets" / f"{tag}_high_plus_A11.png",
                f"{tag} high density plus A11",
                columns=4,
            )

    report_lines = [
        "# Largetest Retune Report",
        "",
        f"- stage: `{args.stage}`",
        f"- input_root: `{args.input_root}`",
        f"- baseline_run: `{args.baseline_run}`",
        f"- configs: {len(configs)}",
        f"- rows: {len(rows)}",
        "",
        "## Top Configs",
        "",
        "| rank | folder | config | model | score | ratio label | high ratio | high mask median | high mask p75 | high mask max | high tiny frac | high count median |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(score_rows[:12], start=1):
        report_lines.append(
            f"| {idx} | {row['folder']} | {row['config']} | {row['model']} | {row['score']} | "
            f"{row['ratio_label']} | {row['high_object_ratio_median']} | {row['high_mask_fraction_median']} | "
            f"{row['high_mask_fraction_p75']} | {row['high_mask_fraction_max']} | "
            f"{row['high_tiny_fraction_lt20_median']} | {row['high_count_median']} |"
        )
    report_lines.extend(
        [
            "",
            "Scoring is heuristic. It favors improved high-density Dead recall while penalizing excessive mask fraction, tiny speckles, and huge saturated blobs.",
            "",
            "## Files",
            "",
            "- `summary.csv`",
            "- `config_scores.csv`",
            "- `contact_sheets/`",
            "- `overlays/`",
            "- `prepared_previews/`",
        ]
    )
    (out_root / "retune_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"summary={out_root / 'summary.csv'}", flush=True)
    print(f"scores={out_root / 'config_scores.csv'}", flush=True)
    print(f"report={out_root / 'retune_report.md'}", flush=True)


if __name__ == "__main__":
    main()
