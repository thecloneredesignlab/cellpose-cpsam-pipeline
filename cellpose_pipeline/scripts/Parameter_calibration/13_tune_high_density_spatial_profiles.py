#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from importlib import metadata
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


REQUIRED_CELLPOSE_VERSION = "4.2.1.1"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
KEY_RE = re.compile(r"(?:SUM159_AC_)?(?:Exp1_)?(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")
HIGH_DENSITY_KEY_SET = {"C2_2_04d04h00m", "E3_2_04d22h00m", "E3_3_04d06h00m"}
KEY_OVERLAY_SET = {
    "A11_1_00d12h00m",
    "B2_1_04d04h00m",
    "B3_2_04d18h00m",
    "C2_2_04d04h00m",
    "E3_2_04d22h00m",
    "E3_3_04d06h00m",
    "F4_1_02d00h00m",
    "G8_1_03d10h00m",
}


@dataclass(frozen=True)
class CandidateConfig:
    profile: str
    tag: str
    transform: str
    model: str = "cpsam"
    diameter: float = 25.0
    flow_threshold: float = 0.0
    cellprob_threshold: float = -1.75
    min_size: int = 20
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    clahe_clip: float = 0.0
    clahe_tile: int = 16
    blur_sigma: float = 0.0
    channel_axis: int | None = None


DEFAULT_CONFIGS = [
    CandidateConfig("Brightfield", "bf_current", "gray"),
    CandidateConfig("Brightfield", "bf_hd_d22_cp2p25_min10", "gray", diameter=22, cellprob_threshold=-2.25, min_size=10),
    CandidateConfig("Brightfield", "bf_hd_d20_cp2p5_min8", "gray", diameter=20, cellprob_threshold=-2.5, min_size=8),
    CandidateConfig("Brightfield", "bf_hd_d18_cp2p5_min8", "gray", diameter=18, cellprob_threshold=-2.5, min_size=8),
    CandidateConfig("Brightfield", "bf_hd_d20_cp2p75_min5", "gray", diameter=20, cellprob_threshold=-2.75, min_size=5),
    CandidateConfig(
        "Brightfield",
        "bf_hd_d20_cp2p5_min8_clahe",
        "gray",
        diameter=20,
        cellprob_threshold=-2.5,
        min_size=8,
        clahe_clip=1.5,
        clahe_tile=16,
    ),
    CandidateConfig("Combined", "combined_luma_current", "rgb_luma"),
    CandidateConfig("Combined", "combined_ch1_d22_cp2p25_min10", "rgb_ch1", diameter=22, cellprob_threshold=-2.25, min_size=10),
    CandidateConfig("Combined", "combined_ch1_d20_cp2p5_min8", "rgb_ch1", diameter=20, cellprob_threshold=-2.5, min_size=8),
    CandidateConfig("Combined", "combined_ch2_d20_cp2p5_min8", "rgb_ch2", diameter=20, cellprob_threshold=-2.5, min_size=8),
    CandidateConfig(
        "Combined",
        "combined_mix12_70_30_d20_cp2p5_min8",
        "rgb_mix12_70_30",
        diameter=20,
        cellprob_threshold=-2.5,
        min_size=8,
    ),
    CandidateConfig(
        "Combined",
        "combined_mix12_50_50_d20_cp2p5_min8",
        "rgb_mix12_50_50",
        diameter=20,
        cellprob_threshold=-2.5,
        min_size=8,
    ),
    CandidateConfig("Combined", "combined_ch1_d18_cp2p75_min5", "rgb_ch1", diameter=18, cellprob_threshold=-2.75, min_size=5),
    CandidateConfig(
        "Combined",
        "combined_mix12_70_30_d18_cp2p75_min5",
        "rgb_mix12_70_30",
        diameter=18,
        cellprob_threshold=-2.75,
        min_size=5,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune high-density Brightfield/Combined profiles using Nuclei-guided spatial QC."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root with Brightfield, Combined, and Nuclei folders.")
    parser.add_argument("--nuclei-run", type=Path, required=True, help="Workflow run containing segmentations/Nuclei masks.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", default="high_density_spatial_tuning")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nuclei-count-threshold", type=int, default=4000)
    parser.add_argument("--nuclei-mask-fraction-threshold", type=float, default=0.22)
    parser.add_argument("--nuclei-median-nn-threshold", type=float, default=16.0)
    return parser.parse_args()


def require_cellpose_version() -> str:
    version = metadata.version("cellpose")
    if version != REQUIRED_CELLPOSE_VERSION:
        raise RuntimeError(f"Expected cellpose=={REQUIRED_CELLPOSE_VERSION}, got {version}")
    return version


def extract_key(path_or_name: Path | str) -> str:
    name = Path(path_or_name).name
    match = KEY_RE.search(name)
    if not match:
        raise ValueError(f"Cannot extract key from {name}")
    return match.group(1)


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def map_images(input_root: Path, profile: str) -> dict[str, Path]:
    return {extract_key(path): path for path in iter_images(input_root / profile)}


def map_nuclei_masks(nuclei_run: Path) -> dict[str, Path]:
    mask_dir = nuclei_run / "segmentations" / "Nuclei"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Nuclei mask directory not found: {mask_dir}")
    return {extract_key(path): path for path in sorted(mask_dir.glob("*_cp_masks.tif"))}


def normalize_percentile(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    data = arr.astype(np.float32, copy=False)
    low, high = np.percentile(data, [low_pct, high_pct])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def apply_clahe(norm: np.ndarray, clip: float, tile: int) -> np.ndarray:
    if clip <= 0:
        return norm
    arr = np.clip(norm * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return np.clip(clahe.apply(arr).astype(np.float32) / 255.0, 0.0, 1.0)


def odd_kernel_from_sigma(sigma: float) -> int:
    k = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    return k if k % 2 else k + 1


def transform_image(raw: np.ndarray, config: CandidateConfig) -> np.ndarray:
    data = raw.astype(np.float32, copy=False)
    if config.transform == "gray":
        if data.ndim != 2:
            data = data[..., 0]
        transformed = data
    elif config.transform == "rgb":
        transformed = data
    else:
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(f"{config.transform} expects RGB channel-last input, got {raw.shape}")
        if config.transform == "rgb_luma":
            transformed = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
        elif config.transform == "rgb_ch0":
            transformed = data[..., 0]
        elif config.transform == "rgb_ch1":
            transformed = data[..., 1]
        elif config.transform == "rgb_ch2":
            transformed = data[..., 2]
        elif config.transform == "rgb_mix12_70_30":
            transformed = 0.7 * data[..., 1] + 0.3 * data[..., 2]
        elif config.transform == "rgb_mix12_50_50":
            transformed = 0.5 * data[..., 1] + 0.5 * data[..., 2]
        elif config.transform == "rgb_min12":
            transformed = np.minimum(data[..., 1], data[..., 2])
        else:
            raise ValueError(f"Unsupported transform: {config.transform}")
    if config.channel_axis is not None:
        return normalize_percentile(transformed, config.low_percentile, config.high_percentile)
    norm = normalize_percentile(transformed, config.low_percentile, config.high_percentile)
    norm = apply_clahe(norm, config.clahe_clip, config.clahe_tile)
    if config.blur_sigma > 0:
        k = odd_kernel_from_sigma(config.blur_sigma)
        norm = cv2.GaussianBlur(norm.astype(np.float32), (k, k), float(config.blur_sigma))
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_rgb_for_display(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Cannot display image shape={raw.shape}")
    if raw.dtype.kind in "ui":
        high = float(np.iinfo(raw.dtype).max)
        return np.clip(arr / max(high, 1.0), 0.0, 1.0)
    low, high = np.percentile(arr, [1, 99])
    return np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)


def stable_color(mask_id: int) -> np.ndarray:
    rng = np.random.default_rng(mask_id * 1103515245 + 12345)
    color = rng.uniform(0.15, 1.0, size=3).astype(np.float32)
    return color / max(float(color.max()), 1e-6)


def make_mask_overlay(raw: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    overlay = normalize_rgb_for_display(raw)
    for mask_id in np.unique(mask):
        mask_id = int(mask_id)
        if mask_id == 0:
            continue
        pixels = mask == mask_id
        overlay[pixels] = (1.0 - alpha) * overlay[pixels] + alpha * stable_color(mask_id)
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def object_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)
    objects = ndi.find_objects(labels)
    ids: list[int] = []
    centroids: list[tuple[float, float]] = []
    for label_id, obj_slice in enumerate(objects, start=1):
        if obj_slice is None:
            continue
        obj = labels[obj_slice] == label_id
        if not np.any(obj):
            continue
        yy, xx = np.nonzero(obj)
        y0 = obj_slice[0].start or 0
        x0 = obj_slice[1].start or 0
        ids.append(label_id)
        centroids.append((float(yy.mean() + y0), float(xx.mean() + x0)))
    if not ids:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)
    return np.asarray(ids, dtype=np.int32), np.asarray(centroids, dtype=np.float32)


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    labels = mask.astype(np.int64, copy=False)
    areas = np.bincount(labels.ravel())[1:]
    n_objects = int(labels.max()) if labels.size else 0
    if areas.size == 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p90": 0.0,
            "area_max": 0.0,
            "small_fraction_lt60": 0.0,
            "large_fraction_gt2500": 0.0,
        }
    return {
        "n_objects": n_objects,
        "mask_fraction": float((labels > 0).sum() / labels.size),
        "area_median": float(np.median(areas)),
        "area_p90": float(np.percentile(areas, 90)),
        "area_max": float(areas.max()),
        "small_fraction_lt60": float((areas < 60).mean()),
        "large_fraction_gt2500": float((areas > 2500).mean()),
    }


def nuclei_density_metrics(nuclei_mask: np.ndarray) -> dict[str, float | int]:
    stats = summarize_mask(nuclei_mask)
    _, centers = object_centroids(nuclei_mask)
    if len(centers) >= 2:
        dist, _ = cKDTree(centers).query(centers, k=2)
        median_nn = float(np.median(dist[:, 1]))
    else:
        median_nn = float("nan")
    return {
        "nuclei_count": int(stats["n_objects"]),
        "nuclei_mask_fraction": float(stats["mask_fraction"]),
        "nuclei_area_median": float(stats["area_median"]),
        "nuclei_median_nn_px": median_nn,
    }


def spatial_qc(cell_mask: np.ndarray, nuclei_mask: np.ndarray) -> dict[str, Any]:
    nuclei_ids, nuclei_centers = object_centroids(nuclei_mask)
    nuclei_count = int(len(nuclei_ids))
    cell_stats = summarize_mask(cell_mask)
    if nuclei_count == 0:
        return {
            **cell_stats,
            "nuclei_count": 0,
            "covered_centroid_count": 0,
            "uncovered_centroid_count": 0,
            "uncovered_centroid_fraction": 0.0,
            "overlap_covered_nuclei_count": 0,
            "overlap_uncovered_nuclei_count": 0,
            "overlap_uncovered_nuclei_fraction": 0.0,
            "multi_nuclei_cell_count": 0,
            "extra_nuclei_in_multi_cells": 0,
            "extra_nuclei_per_nucleus": 0.0,
            "cell_count_to_nuclei_count": float("nan"),
        }

    yy = np.clip(np.rint(nuclei_centers[:, 0]).astype(np.int64), 0, cell_mask.shape[0] - 1)
    xx = np.clip(np.rint(nuclei_centers[:, 1]).astype(np.int64), 0, cell_mask.shape[1] - 1)
    cell_at_centroid = cell_mask[yy, xx].astype(np.int64, copy=False)
    covered = cell_at_centroid > 0
    binc = np.bincount(cell_at_centroid[covered], minlength=int(cell_mask.max()) + 1)
    multi_counts = binc[binc >= 2]
    overlap_labels = np.unique(nuclei_mask[(cell_mask > 0) & (nuclei_mask > 0)])
    overlap_labels = overlap_labels[overlap_labels > 0]
    overlap_covered = int(len(overlap_labels))
    extra_nuclei = int(np.maximum(multi_counts - 1, 0).sum()) if multi_counts.size else 0
    return {
        **cell_stats,
        "nuclei_count": nuclei_count,
        "covered_centroid_count": int(covered.sum()),
        "uncovered_centroid_count": int((~covered).sum()),
        "uncovered_centroid_fraction": float((~covered).sum() / nuclei_count),
        "overlap_covered_nuclei_count": overlap_covered,
        "overlap_uncovered_nuclei_count": int(nuclei_count - overlap_covered),
        "overlap_uncovered_nuclei_fraction": float((nuclei_count - overlap_covered) / nuclei_count),
        "multi_nuclei_cell_count": int(len(multi_counts)),
        "extra_nuclei_in_multi_cells": extra_nuclei,
        "extra_nuclei_per_nucleus": float(extra_nuclei / nuclei_count),
        "cell_count_to_nuclei_count": float(cell_stats["n_objects"] / nuclei_count) if nuclei_count else float("nan"),
    }


def make_spatial_overlay(raw: np.ndarray, cell_mask: np.ndarray, nuclei_mask: np.ndarray) -> Image.Image:
    image = make_mask_overlay(raw, cell_mask, alpha=0.45).convert("RGB")
    draw = ImageDraw.Draw(image)
    _, centers = object_centroids(nuclei_mask)
    if len(centers):
        yy = np.clip(np.rint(centers[:, 0]).astype(np.int64), 0, cell_mask.shape[0] - 1)
        xx = np.clip(np.rint(centers[:, 1]).astype(np.int64), 0, cell_mask.shape[1] - 1)
        covered = cell_mask[yy, xx] > 0
        for (y, x), ok in zip(centers, covered):
            color = (0, 255, 255) if ok else (255, 0, 0)
            r = 2 if ok else 4
            draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r), outline=color, width=1)
    return image


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if str(row.get(key, "")).lower() not in {"", "nan"}]
    return float(np.mean(vals)) if vals else float("nan")


def score_candidate(rows: list[dict[str, Any]], density_by_key: dict[str, bool]) -> dict[str, Any]:
    high_rows = [row for row in rows if density_by_key.get(row["key"], False)]
    low_rows = [row for row in rows if not density_by_key.get(row["key"], False)]
    target_rows = high_rows or rows

    uncovered = mean_float(target_rows, "uncovered_centroid_fraction")
    extra = mean_float(target_rows, "extra_nuclei_per_nucleus")
    count_ratios = [max(float(row["cell_count_to_nuclei_count"]), 1e-6) for row in target_rows]
    count_penalty = float(np.mean([abs(math.log(x)) for x in count_ratios])) if count_ratios else float("nan")
    small = max(0.0, mean_float(target_rows, "small_fraction_lt60") - 0.18)
    large = mean_float(target_rows, "large_fraction_gt2500")
    low_overseg = 0.0
    if low_rows:
        low_ratios = [float(row["cell_count_to_nuclei_count"]) for row in low_rows]
        low_overseg = float(np.mean([max(0.0, x - 1.20) for x in low_ratios]))
    score = 2.0 * uncovered + 1.2 * extra + 0.35 * count_penalty + 0.3 * small + 0.2 * large + 0.8 * low_overseg
    first = rows[0]
    return {
        "profile": first["profile"],
        "config": first["config"],
        "score": f"{score:.6f}",
        "high_density_rows": len(high_rows),
        "mean_hd_uncovered_centroid_fraction": f"{mean_float(target_rows, 'uncovered_centroid_fraction'):.6f}",
        "mean_hd_extra_nuclei_per_nucleus": f"{mean_float(target_rows, 'extra_nuclei_per_nucleus'):.6f}",
        "mean_hd_cell_count_to_nuclei_count": f"{mean_float(target_rows, 'cell_count_to_nuclei_count'):.6f}",
        "mean_hd_large_fraction_gt2500": f"{mean_float(target_rows, 'large_fraction_gt2500'):.6f}",
        "mean_low_overseg_penalty": f"{low_overseg:.6f}",
    }


def write_contact_sheet(paths: list[Path], out_path: Path, title: str, columns: int = 3) -> None:
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    thumb_w, thumb_h = 420, 310
    label_h = 38
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
        draw.text((x + 4, y + 4), f"{path.parent.name}/{path.stem}"[:76], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    version = require_cellpose_version()
    out_root = args.out_root.resolve() / args.stage
    out_root.mkdir(parents=True, exist_ok=True)

    image_maps = {
        "Brightfield": map_images(args.input_root, "Brightfield"),
        "Combined": map_images(args.input_root, "Combined"),
    }
    nuclei_masks = map_nuclei_masks(args.nuclei_run)
    common_keys = sorted(set(nuclei_masks) & (set(image_maps["Brightfield"]) | set(image_maps["Combined"])))
    if not common_keys:
        raise SystemExit("No common keys found between input images and nuclei masks.")

    density_rows: list[dict[str, Any]] = []
    density_by_key: dict[str, bool] = {}
    for key in common_keys:
        nuclei_mask = tifffile.imread(nuclei_masks[key])
        density = nuclei_density_metrics(nuclei_mask)
        high_density = (
            int(density["nuclei_count"]) >= args.nuclei_count_threshold
            or float(density["nuclei_mask_fraction"]) >= args.nuclei_mask_fraction_threshold
            or (
                not math.isnan(float(density["nuclei_median_nn_px"]))
                and float(density["nuclei_median_nn_px"]) <= args.nuclei_median_nn_threshold
            )
        )
        density_by_key[key] = high_density
        density_rows.append(
            {
                "key": key,
                **density,
                "high_density": high_density,
                "is_focus_key": key in HIGH_DENSITY_KEY_SET,
            }
        )
    write_rows(out_root / "density_calls.csv", density_rows)

    model_cache: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    by_config: dict[tuple[str, str], list[dict[str, Any]]] = {}
    print(f"cellpose_version={version}", flush=True)
    print(f"out_root={out_root}", flush=True)
    print(f"n_keys={len(common_keys)}", flush=True)

    for config in DEFAULT_CONFIGS:
        image_map = image_maps[config.profile]
        model = model_cache.get(config.model)
        if model is None:
            model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=config.model)
            model_cache[config.model] = model
            print(f"model={config.model} device={model.device}", flush=True)
        for key in common_keys:
            image_path = image_map.get(key)
            if image_path is None:
                continue
            nuclei_mask = tifffile.imread(nuclei_masks[key])
            raw = tifffile.imread(image_path)
            mask_path = out_root / "masks" / config.profile / config.tag / f"{image_path.stem}_cp_masks.tif"
            if args.skip_existing and mask_path.exists():
                cell_mask = tifffile.imread(mask_path)
                elapsed = 0.0
                print(f"skip\t{config.profile}\t{config.tag}\t{key}", flush=True)
            else:
                prepared = transform_image(raw, config)
                print(f"run\t{config.profile}\t{config.tag}\t{key}\tshape={prepared.shape}", flush=True)
                start = time.time()
                cell_mask, *_ = model.eval(
                    prepared,
                    channel_axis=config.channel_axis,
                    normalize=False,
                    diameter=config.diameter,
                    flow_threshold=config.flow_threshold,
                    cellprob_threshold=config.cellprob_threshold,
                    min_size=config.min_size,
                )
                elapsed = time.time() - start
                cell_mask = np.asarray(cell_mask)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                dtype = np.uint32 if int(cell_mask.max()) > np.iinfo(np.uint16).max else np.uint16
                tifffile.imwrite(mask_path, cell_mask.astype(dtype, copy=False))

            qc = spatial_qc(cell_mask, nuclei_mask)
            overlay_path = out_root / "overlays" / config.profile / config.tag / f"{image_path.stem}_spatial_overlay.png"
            if key in KEY_OVERLAY_SET:
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                make_spatial_overlay(raw, cell_mask, nuclei_mask).save(overlay_path)

            row = {
                "profile": config.profile,
                "config": config.tag,
                "key": key,
                "image": image_path.name,
                "high_density": density_by_key[key],
                "focus_key": key in HIGH_DENSITY_KEY_SET,
                **{f"param_{k}": v for k, v in asdict(config).items()},
                "elapsed_sec": f"{elapsed:.3f}",
                **qc,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
            all_rows.append(row)
            by_config.setdefault((config.profile, config.tag), []).append(row)

    write_rows(out_root / "spatial_qc_summary.csv", all_rows)
    score_rows = [score_candidate(rows, density_by_key) for rows in by_config.values()]
    score_rows.sort(key=lambda row: (row["profile"], float(row["score"])))
    write_rows(out_root / "config_scores.csv", score_rows)

    if args.make_contact_sheets:
        for (profile, tag), rows in by_config.items():
            paths = [
                Path(row["overlay_path"])
                for row in rows
                if row["key"] in KEY_OVERLAY_SET and Path(row["overlay_path"]).exists()
            ]
            write_contact_sheet(paths, out_root / "contact_sheets" / profile / f"{tag}.png", f"{profile} {tag}")

    best_by_profile: dict[str, dict[str, Any]] = {}
    for row in score_rows:
        best_by_profile.setdefault(row["profile"], row)
    lines = [
        "# High-Density Spatial Profile Tuning",
        "",
        f"- input_root: `{args.input_root}`",
        f"- nuclei_run: `{args.nuclei_run}`",
        f"- out_root: `{out_root}`",
        f"- cellpose_version: `{version}`",
        "",
        "## Density calls",
        "",
        "| key | nuclei_count | nuclei_mask_fraction | median_nn_px | high_density |",
        "|---|---:|---:|---:|---|",
    ]
    for row in density_rows:
        if row["high_density"] or row["is_focus_key"]:
            lines.append(
                f"| {row['key']} | {row['nuclei_count']} | {float(row['nuclei_mask_fraction']):.4f} | "
                f"{float(row['nuclei_median_nn_px']):.2f} | {row['high_density']} |"
            )
    lines.extend(["", "## Best config by profile", ""])
    lines.append(
        "| profile | config | score | uncovered nuclei frac | extra nuclei per nucleus | cell/nuclei | large frac |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for profile in sorted(best_by_profile):
        row = best_by_profile[profile]
        lines.append(
            f"| {profile} | {row['config']} | {row['score']} | "
            f"{row['mean_hd_uncovered_centroid_fraction']} | {row['mean_hd_extra_nuclei_per_nucleus']} | "
            f"{row['mean_hd_cell_count_to_nuclei_count']} | {row['mean_hd_large_fraction_gt2500']} |"
        )
    lines.extend(["", "## Full ranking", ""])
    lines.append(
        "| profile | config | score | uncovered nuclei frac | extra nuclei per nucleus | cell/nuclei | low overseg penalty |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in score_rows:
        lines.append(
            f"| {row['profile']} | {row['config']} | {row['score']} | "
            f"{row['mean_hd_uncovered_centroid_fraction']} | {row['mean_hd_extra_nuclei_per_nucleus']} | "
            f"{row['mean_hd_cell_count_to_nuclei_count']} | {row['mean_low_overseg_penalty']} |"
        )
    (out_root / "tuning_report.md").write_text("\n".join(lines) + "\n")
    print(f"density_calls={out_root / 'density_calls.csv'}", flush=True)
    print(f"spatial_qc_summary={out_root / 'spatial_qc_summary.csv'}", flush=True)
    print(f"config_scores={out_root / 'config_scores.csv'}", flush=True)
    print(f"report={out_root / 'tuning_report.md'}", flush=True)


if __name__ == "__main__":
    main()
