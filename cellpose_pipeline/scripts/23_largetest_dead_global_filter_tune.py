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

import numpy as np
import tifffile
from cellpose import models
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


IMAGE_SUFFIXES = {".tif", ".tiff"}
TIME_KEY_RE = re.compile(r"(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")
A11_ARTIFACT_KEY = "A11_1_00d12h00m"
F4_LOW_SIGNAL_KEY = "F4_1_02d00h00m"


@dataclass(frozen=True)
class SegConfig:
    tag: str
    model: str
    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    low_value: float
    high_value: float


@dataclass(frozen=True)
class FilterConfig:
    tag: str
    min_area: int = 20
    max_area: int = 2500
    max_aspect: float = 4.0
    min_mean_delta: float = 0.25
    min_p90_delta: float = 0.8
    min_p90_abs: float = 10.0
    min_snr: float = 1.5
    min_nuclei_overlap: float = 0.0
    bg_margin: int = 14


DEFAULT_SEG_CONFIGS = [
    SegConfig("g20_d22_cp-2p5", "cpsam_v2", 22, 0.0, -2.5, 12, 7.6, 20.0),
    SegConfig("g20_d22_cp-3p5", "cpsam_v2", 22, 0.0, -3.5, 12, 7.6, 20.0),
    SegConfig("g30_d22_cp-2p5", "cpsam_v2", 22, 0.0, -2.5, 12, 7.6, 30.0),
    SegConfig("g30_d22_cp-3p5", "cpsam_v2", 22, 0.0, -3.5, 12, 7.6, 30.0),
    SegConfig("g40_d22_cp-3p0", "cpsam_v2", 22, 0.0, -3.0, 12, 7.6, 40.0),
    SegConfig("g40_d22_cp-4p0", "cpsam_v2", 22, 0.0, -4.0, 12, 7.6, 40.0),
    SegConfig("g55_d22_cp-3p0", "cpsam_v2", 22, 0.0, -3.0, 12, 7.6, 55.0),
    SegConfig("g55_d22_cp-4p0", "cpsam_v2", 22, 0.0, -4.0, 12, 7.6, 55.0),
    SegConfig("g80_d22_cp-4p0", "cpsam_v2", 22, 0.0, -4.0, 12, 7.6, 80.0),
    SegConfig("g30_d20_cp-3p5", "cpsam_v2", 20, 0.0, -3.5, 12, 7.6, 30.0),
    SegConfig("g40_d20_cp-4p0", "cpsam_v2", 20, 0.0, -4.0, 12, 7.6, 40.0),
]


DEFAULT_FILTER_CONFIGS = [
    FilterConfig("weak_raw_area2500", max_area=2500, min_mean_delta=0.20, min_p90_delta=0.70, min_p90_abs=10.0, min_snr=1.25),
    FilterConfig("balanced_area2500", max_area=2500, min_mean_delta=0.30, min_p90_delta=1.00, min_p90_abs=10.5, min_snr=1.75),
    FilterConfig("strict_area2500", max_area=2500, min_mean_delta=0.45, min_p90_delta=1.50, min_p90_abs=11.0, min_snr=2.25),
    FilterConfig("balanced_area1800", max_area=1800, min_mean_delta=0.30, min_p90_delta=1.00, min_p90_abs=10.5, min_snr=1.75),
    FilterConfig("strict_area1800", max_area=1800, min_mean_delta=0.45, min_p90_delta=1.50, min_p90_abs=11.0, min_snr=2.25),
    FilterConfig("nuc_balanced_area2500", max_area=2500, min_mean_delta=0.25, min_p90_delta=0.85, min_p90_abs=10.2, min_snr=1.50, min_nuclei_overlap=0.02),
    FilterConfig("nuc_strict_area2500", max_area=2500, min_mean_delta=0.35, min_p90_delta=1.20, min_p90_abs=10.8, min_snr=2.00, min_nuclei_overlap=0.02),
    FilterConfig("nuc_balanced_area1800", max_area=1800, min_mean_delta=0.25, min_p90_delta=0.85, min_p90_abs=10.2, min_snr=1.50, min_nuclei_overlap=0.02),
    FilterConfig("nuc_weak_p9p5_area1800", max_area=1800, min_mean_delta=0.05, min_p90_delta=0.25, min_p90_abs=9.5, min_snr=0.75, min_nuclei_overlap=0.02),
    FilterConfig("nuc_weak_p9p8_area1800", max_area=1800, min_mean_delta=0.10, min_p90_delta=0.40, min_p90_abs=9.8, min_snr=1.00, min_nuclei_overlap=0.02),
    FilterConfig("nuc_mid_p10_area1800", max_area=1800, min_mean_delta=0.15, min_p90_delta=0.55, min_p90_abs=10.0, min_snr=1.20, min_nuclei_overlap=0.02),
    FilterConfig("raw_weak_p9p8_area1800", max_area=1800, min_mean_delta=0.10, min_p90_delta=0.40, min_p90_abs=9.8, min_snr=1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Dead segmentation using fixed global scaling and raw-evidence filters.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", default="dead_global_filter_round1")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-all-filtered-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-object-stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_key(name: str) -> str:
    match = TIME_KEY_RE.search(name)
    if not match:
        raise ValueError(f"Cannot extract key from {name}")
    return match.group(1)


def iter_dead_images(input_root: Path) -> list[Path]:
    root = input_root / "Dead"
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


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


def global_intensity_rows(images: list[Path]) -> list[dict[str, Any]]:
    rows = []
    samples = []
    for path in images:
        raw = tifffile.imread(path).astype(np.float32, copy=False)
        samples.append(raw.ravel()[::97])
        qs = np.percentile(raw, [0, 0.1, 1, 50, 90, 95, 99, 99.5, 99.9, 100])
        rows.append(
            {
                "image": path.name,
                "key": extract_key(path.name),
                "p0": f"{qs[0]:.6f}",
                "p0_1": f"{qs[1]:.6f}",
                "p1": f"{qs[2]:.6f}",
                "p50": f"{qs[3]:.6f}",
                "p90": f"{qs[4]:.6f}",
                "p95": f"{qs[5]:.6f}",
                "p99": f"{qs[6]:.6f}",
                "p99_5": f"{qs[7]:.6f}",
                "p99_9": f"{qs[8]:.6f}",
                "p100": f"{qs[9]:.6f}",
                "frac_gt10": f"{float((raw > 10).mean()):.8f}",
                "frac_gt12": f"{float((raw > 12).mean()):.8f}",
                "frac_gt15": f"{float((raw > 15).mean()):.8f}",
                "frac_gt20": f"{float((raw > 20).mean()):.8f}",
                "frac_gt30": f"{float((raw > 30).mean()):.8f}",
            }
        )
    all_sample = np.concatenate(samples)
    qs = np.percentile(all_sample, [0, 0.1, 1, 5, 50, 90, 95, 99, 99.5, 99.8, 99.9, 99.95, 100])
    rows.insert(
        0,
        {
            "image": "__GLOBAL_SAMPLE__",
            "key": "__GLOBAL_SAMPLE__",
            "p0": f"{qs[0]:.6f}",
            "p0_1": f"{qs[1]:.6f}",
            "p1": f"{qs[2]:.6f}",
            "p50": f"{qs[4]:.6f}",
            "p90": f"{qs[5]:.6f}",
            "p95": f"{qs[6]:.6f}",
            "p99": f"{qs[7]:.6f}",
            "p99_5": f"{qs[8]:.6f}",
            "p99_9": f"{qs[10]:.6f}",
            "p100": f"{qs[12]:.6f}",
            "frac_gt10": "",
            "frac_gt12": "",
            "frac_gt15": "",
            "frac_gt20": "",
            "frac_gt30": "",
        },
    )
    return rows


def fixed_normalize(raw: np.ndarray, config: SegConfig) -> np.ndarray:
    denom = max(config.high_value - config.low_value, 1e-6)
    return np.clip((raw.astype(np.float32, copy=False) - config.low_value) / denom, 0.0, 1.0)


def raw_display(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype(np.float32, copy=False)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if low >= 0.0:
        return np.clip(arr / max(high, 1.0), 0.0, 1.0)
    lo, hi = np.percentile(arr, [0.1, 99.9])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    labels = mask.astype(np.int64, copy=False)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def make_overlay(base_norm: np.ndarray, mask: np.ndarray) -> Image.Image:
    gray = np.clip(base_norm * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    pixels = mask > 0
    rgb[pixels, 0] = np.maximum(rgb[pixels, 0], 220)
    rgb[pixels, 2] = np.maximum(rgb[pixels, 2], 160)
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
            "area_p90": 0.0,
            "max_area": 0,
            "large_fraction_gt1800": 0.0,
            "large_fraction_gt2500": 0.0,
            "tiny_fraction_lt20": 0.0,
        }
    return {
        "n_objects": n_objects,
        "mask_fraction": float((mask > 0).sum() / mask.size),
        "area_median": float(np.median(areas)),
        "area_p90": float(np.percentile(areas, 90)),
        "max_area": int(areas.max()),
        "large_fraction_gt1800": float((areas > 1800).mean()),
        "large_fraction_gt2500": float((areas > 2500).mean()),
        "tiny_fraction_lt20": float((areas < 20).mean()),
    }


def object_records(raw: np.ndarray, labels: np.ndarray, nuclei_mask: np.ndarray | None, config: FilterConfig) -> list[dict[str, Any]]:
    labels = labels.astype(np.int32, copy=False)
    objects = ndi.find_objects(labels)
    records = []
    h, w = labels.shape
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        obj = labels[slc] == idx
        area = int(obj.sum())
        if area == 0:
            continue
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        margin = int(config.bg_margin)
        yy0, yy1 = max(0, y0 - margin), min(h, y1 + margin)
        xx0, xx1 = max(0, x0 - margin), min(w, x1 + margin)
        raw_obj = raw[slc][obj]
        window_labels = labels[yy0:yy1, xx0:xx1]
        bg_values = raw[yy0:yy1, xx0:xx1][window_labels == 0]
        if bg_values.size < 25:
            bg_values = raw[yy0:yy1, xx0:xx1][window_labels != idx]
        if bg_values.size < 25:
            bg_values = raw.ravel()
        bg_med = float(np.median(bg_values))
        bg_mad = float(np.median(np.abs(bg_values - bg_med)))
        bg_sigma = max(bg_mad * 1.4826, 1e-6)
        mean = float(np.mean(raw_obj))
        p90 = float(np.percentile(raw_obj, 90))
        raw_max = float(np.max(raw_obj))
        bbox_h = max(1, y1 - y0)
        bbox_w = max(1, x1 - x0)
        aspect = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))
        nuc_overlap = 0.0
        if nuclei_mask is not None:
            nuc_overlap = float((nuclei_mask[slc][obj] > 0).mean())
        record = {
                "label": idx,
                "area": area,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "aspect": aspect,
                "raw_mean": mean,
                "raw_p90": p90,
                "raw_max": raw_max,
                "bg_median": bg_med,
                "bg_sigma": bg_sigma,
                "mean_delta": mean - bg_med,
                "p90_delta": p90 - bg_med,
                "snr": (p90 - bg_med) / bg_sigma,
                "nuclei_overlap": nuc_overlap,
            }
        record["keep"] = keep_record(record, config)
        records.append(record)
    return records


def keep_record(record: dict[str, Any], config: FilterConfig) -> bool:
    return (
        int(record["area"]) >= config.min_area
        and int(record["area"]) <= config.max_area
        and float(record["aspect"]) <= config.max_aspect
        and float(record["mean_delta"]) >= config.min_mean_delta
        and float(record["p90_delta"]) >= config.min_p90_delta
        and float(record["raw_p90"]) >= config.min_p90_abs
        and float(record["snr"]) >= config.min_snr
        and float(record["nuclei_overlap"]) >= config.min_nuclei_overlap
    )


def apply_filter(records: list[dict[str, Any]], config: FilterConfig) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        next_record = dict(record)
        next_record["keep"] = keep_record(next_record, config)
        filtered.append(next_record)
    return filtered


def relabel_filtered(labels: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    max_label = int(labels.max()) if labels.size else 0
    label_map = np.zeros(max_label + 1, dtype=np.uint32)
    next_id = 1
    for record in records:
        if not record["keep"]:
            continue
        label = int(record["label"])
        if label <= max_label:
            label_map[label] = next_id
        next_id += 1
    return label_map[labels]


def a11_artifact_roi_stats(mask: np.ndarray) -> dict[str, float | int]:
    h, w = mask.shape
    roi = mask[int(h * 0.74) : h, 0 : int(w * 0.32)]
    ids = np.unique(roi)
    ids = ids[ids > 0]
    return {
        "a11_artifact_roi_mask_fraction": float((roi > 0).sum() / roi.size),
        "a11_artifact_roi_objects": int(ids.size),
    }


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
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high = [row for row in rows if row["density_bin"] == "high"]
    strong_signal = [row for row in rows if float(row["raw_frac_gt20"]) >= 0.02]
    f4 = [row for row in rows if row["key"] == F4_LOW_SIGNAL_KEY]
    a11 = [row for row in rows if row["key"] == A11_ARTIFACT_KEY]
    f4_mask = median([float(row["filtered_mask_fraction"]) for row in f4])
    f4_ratio = median([float(row["filtered_object_to_nuclei_ratio"]) for row in f4])
    a11_roi = median([float(row["a11_artifact_roi_mask_fraction"]) for row in a11])
    a11_roi_objects = median([float(row["a11_artifact_roi_objects"]) for row in a11])
    a11_max_area = median([float(row["filtered_max_area"]) for row in a11])
    a11_count = median([float(row["filtered_n_objects"]) for row in a11])
    high_mask_p75 = percentile([float(row["filtered_mask_fraction"]) for row in high], 75)
    high_ratio_median = median([float(row["filtered_object_to_nuclei_ratio"]) for row in high])
    strong_count_median = median([float(row["filtered_n_objects"]) for row in strong_signal])
    all_count_median = median([float(row["filtered_n_objects"]) for row in rows])
    large_p75 = percentile([float(row["filtered_large_fraction_gt1800"]) for row in rows], 75)

    f4_penalty = max(0.0, f4_mask - 0.025) * 12.0 + max(0.0, f4_ratio - 0.08) * 1.5
    artifact_penalty = a11_roi * 8.0 + max(0.0, a11_roi_objects - 2.0) * 0.05 + max(0.0, a11_max_area - 2200.0) / 2200.0
    overmask_penalty = max(0.0, high_mask_p75 - 0.12) * 3.0 + max(0.0, high_ratio_median - 0.30) * 0.5
    large_penalty = large_p75 * 1.5
    no_signal_penalty = 0.0
    if strong_signal:
        no_signal_penalty += max(0.0, 80.0 - strong_count_median) / 160.0
    no_signal_penalty += max(0.0, 30.0 - all_count_median) / 120.0
    no_signal_penalty += max(0.0, 80.0 - a11_count) / 160.0
    score = f4_penalty + artifact_penalty + overmask_penalty + large_penalty + no_signal_penalty
    first = rows[0]
    return {
        "combined_config": first["combined_config"],
        "seg_config": first["seg_config"],
        "filter_config": first["filter_config"],
        "score": f"{score:.6f}",
        "f4_mask_fraction": f"{f4_mask:.6f}",
        "f4_object_ratio": f"{f4_ratio:.6f}",
        "a11_artifact_roi_mask_fraction": f"{a11_roi:.6f}",
        "a11_artifact_roi_objects": f"{a11_roi_objects:.3f}",
        "a11_max_area": f"{a11_max_area:.3f}",
        "a11_count": f"{a11_count:.3f}",
        "high_mask_fraction_p75": f"{high_mask_p75:.6f}",
        "high_object_ratio_median": f"{high_ratio_median:.6f}",
        "strong_signal_count_median": f"{strong_count_median:.3f}",
        "all_count_median": f"{all_count_median:.3f}",
        "large_fraction_gt1800_p75": f"{large_p75:.6f}",
    }


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve() / args.stage
    out_root.mkdir(parents=True, exist_ok=True)
    images = iter_dead_images(args.input_root)
    if not images:
        raise SystemExit(f"No Dead tif images found under {args.input_root}")

    baseline = read_baseline_summary(args.baseline_run)
    density_bins = density_bins_from_baseline(baseline)
    selected_keys = {F4_LOW_SIGNAL_KEY, A11_ARTIFACT_KEY}
    selected_keys.update(key for key, density in density_bins.items() if density == "high")
    intensity_rows = global_intensity_rows(images)
    write_rows(out_root / "raw_intensity_summary.csv", intensity_rows)
    (out_root / "seg_configs.json").write_text(json.dumps([asdict(config) for config in DEFAULT_SEG_CONFIGS], indent=2) + "\n")
    (out_root / "filter_configs.json").write_text(json.dumps([asdict(config) for config in DEFAULT_FILTER_CONFIGS], indent=2) + "\n")

    nuclei_masks: dict[str, np.ndarray] = {}
    for (folder, key), row in baseline.items():
        if folder == "Nuclei":
            mask_path = Path(row["mask_path"])
            if mask_path.exists():
                nuclei_masks[key] = tifffile.imread(mask_path)

    rows: list[dict[str, Any]] = []
    by_combined: dict[str, list[dict[str, Any]]] = {}
    model_cache: dict[str, Any] = {}
    raw_cache: dict[Path, np.ndarray] = {}
    prepared_cache: dict[tuple[str, Path], np.ndarray] = {}
    mask_cache: dict[tuple[str, Path], np.ndarray] = {}
    records_cache: dict[tuple[str, Path], list[dict[str, Any]]] = {}

    for seg_config in DEFAULT_SEG_CONFIGS:
        model = model_cache.get(seg_config.model)
        if model is None:
            print(f"loading_model={seg_config.model}", flush=True)
            model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=seg_config.model)
            model_cache[seg_config.model] = model
            print(f"model_device={model.device}", flush=True)
        for image_path in images:
            raw = raw_cache.get(image_path)
            if raw is None:
                raw = tifffile.imread(image_path).astype(np.float32, copy=False)
                raw_cache[image_path] = raw
            prepared = fixed_normalize(raw, seg_config)
            prepared_cache[(seg_config.tag, image_path)] = prepared
            mask_path = out_root / "candidate_masks" / seg_config.tag / f"{image_path.stem}_candidate_masks.tif"
            if args.skip_existing and mask_path.exists():
                candidate = tifffile.imread(mask_path)
                elapsed = 0.0
                print(f"skip_candidate\t{seg_config.tag}\t{image_path.name}", flush=True)
            else:
                print(f"run_candidate\t{seg_config.tag}\t{image_path.name}", flush=True)
                start = time.time()
                candidate, *_ = model.eval(
                    prepared,
                    channel_axis=None,
                    normalize=False,
                    diameter=seg_config.diameter,
                    flow_threshold=seg_config.flow_threshold,
                    cellprob_threshold=seg_config.cellprob_threshold,
                    min_size=seg_config.min_size,
                )
                elapsed = time.time() - start
                candidate = np.asarray(candidate)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(mask_path, candidate.astype(np.uint16 if candidate.max() < 65535 else np.uint32, copy=False))
            mask_cache[(seg_config.tag, image_path)] = candidate

    for seg_config in DEFAULT_SEG_CONFIGS:
        for filter_config in DEFAULT_FILTER_CONFIGS:
            combined_tag = f"{seg_config.tag}__{filter_config.tag}"
            for image_path in images:
                key = extract_key(image_path.name)
                raw = raw_cache[image_path]
                prepared = prepared_cache[(seg_config.tag, image_path)]
                candidate = mask_cache[(seg_config.tag, image_path)]
                cache_key = (seg_config.tag, image_path)
                base_records = records_cache.get(cache_key)
                if base_records is None:
                    nuclei_mask = nuclei_masks.get(key)
                    base_records = object_records(raw, candidate, nuclei_mask, DEFAULT_FILTER_CONFIGS[0])
                    records_cache[cache_key] = base_records
                records = apply_filter(base_records, filter_config)
                filtered = relabel_filtered(candidate.astype(np.int32, copy=False), records)

                mask_path = out_root / "filtered_masks" / combined_tag / f"{image_path.stem}_filtered_masks.tif"
                global_overlay_path = out_root / "overlays_global" / combined_tag / f"{image_path.stem}_overlay.png"
                raw_overlay_path = out_root / "overlays_raw_fullrange" / combined_tag / f"{image_path.stem}_overlay.png"
                write_filtered_output = args.write_all_filtered_outputs or key in selected_keys
                if write_filtered_output:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    global_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(mask_path, filtered.astype(np.uint16 if filtered.max() < 65535 else np.uint32, copy=False))
                    make_overlay(prepared, filtered).save(global_overlay_path)
                    make_overlay(raw_display(raw), filtered).save(raw_overlay_path)

                object_stats_path = out_root / "object_stats" / combined_tag / f"{image_path.stem}_object_stats.csv"
                if args.write_object_stats and write_filtered_output:
                    write_rows(object_stats_path, records)

                filtered_stats = summarize_mask(filtered)
                candidate_stats = summarize_mask(candidate)
                nuc = baseline.get(("Nuclei", key), {})
                nuclei_count = int(float(nuc.get("n_objects", 0) or 0))
                filtered_ratio = filtered_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")
                candidate_ratio = candidate_stats["n_objects"] / nuclei_count if nuclei_count else float("inf")
                raw_fracs = {
                    "raw_frac_gt10": float((raw > 10).mean()),
                    "raw_frac_gt12": float((raw > 12).mean()),
                    "raw_frac_gt15": float((raw > 15).mean()),
                    "raw_frac_gt20": float((raw > 20).mean()),
                }
                roi_stats = a11_artifact_roi_stats(filtered) if key == A11_ARTIFACT_KEY else {
                    "a11_artifact_roi_mask_fraction": 0.0,
                    "a11_artifact_roi_objects": 0,
                }
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
                    "filtered_large_fraction_gt1800": filtered_stats["large_fraction_gt1800"],
                    "filtered_large_fraction_gt2500": filtered_stats["large_fraction_gt2500"],
                    "filtered_tiny_fraction_lt20": filtered_stats["tiny_fraction_lt20"],
                    "filtered_object_to_nuclei_ratio": f"{filtered_ratio:.6f}",
                    "kept_fraction_of_candidates": f"{filtered_stats['n_objects'] / candidate_stats['n_objects']:.6f}" if candidate_stats["n_objects"] else "0.000000",
                    **raw_fracs,
                    **roi_stats,
                    "mask_path": str(mask_path),
                    "global_overlay_path": str(global_overlay_path),
                    "raw_overlay_path": str(raw_overlay_path),
                    "object_stats_path": str(object_stats_path),
                }
                rows.append(row)
                by_combined.setdefault(combined_tag, []).append(row)

    write_rows(out_root / "summary.csv", rows)
    score_table = [score_rows(config_rows) for config_rows in by_combined.values()]
    score_table.sort(key=lambda row: float(row["score"]))
    write_rows(out_root / "config_scores.csv", score_table)

    if args.make_contact_sheets:
        for score_row in score_table[:16]:
            tag = score_row["combined_config"]
            paths = [
                Path(row["raw_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in selected_keys and Path(row["raw_overlay_path"]).exists()
            ]
            write_contact_sheet(paths, out_root / "contact_sheets_raw_fullrange" / f"{tag}_F4_A11_high.png", f"{tag} raw full range F4 A11 high")
            paths = [
                Path(row["global_overlay_path"])
                for row in by_combined[tag]
                if row["key"] in selected_keys and Path(row["global_overlay_path"]).exists()
            ]
            write_contact_sheet(paths, out_root / "contact_sheets_global" / f"{tag}_F4_A11_high.png", f"{tag} fixed global display F4 A11 high")

    report_lines = [
        "# Dead Global-Filter Retune Report",
        "",
        f"- stage: `{args.stage}`",
        f"- input_root: `{args.input_root}`",
        f"- baseline_run: `{args.baseline_run}`",
        f"- segmentation configs: {len(DEFAULT_SEG_CONFIGS)}",
        f"- filter configs: {len(DEFAULT_FILTER_CONFIGS)}",
        f"- combined configs: {len(score_table)}",
        "",
        "## Top Configs",
        "",
        "| rank | config | score | F4 mask | F4 ratio | A11 count | A11 artifact ROI | A11 ROI objects | high mask p75 | strong count median |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(score_table[:16], start=1):
        report_lines.append(
            f"| {idx} | {row['combined_config']} | {row['score']} | {row['f4_mask_fraction']} | "
            f"{row['f4_object_ratio']} | {row['a11_count']} | {row['a11_artifact_roi_mask_fraction']} | "
            f"{row['a11_artifact_roi_objects']} | {row['high_mask_fraction_p75']} | "
            f"{row['strong_signal_count_median']} |"
        )
    report_lines.extend(
        [
            "",
            "Scoring favors raw-evidence masks, low F4 false positives, removal of the A11 lower-left artifact ROI, and nonzero recall on raw-positive images.",
            "",
            "## Files",
            "",
            "- `raw_intensity_summary.csv`",
            "- `summary.csv`",
            "- `config_scores.csv`",
            "- `contact_sheets_raw_fullrange/`",
            "- `contact_sheets_global/`",
            "- `filtered_masks/`",
            "- `object_stats/`",
        ]
    )
    (out_root / "retune_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"summary={out_root / 'summary.csv'}", flush=True)
    print(f"scores={out_root / 'config_scores.csv'}", flush=True)
    print(f"report={out_root / 'retune_report.md'}", flush=True)


if __name__ == "__main__":
    main()
