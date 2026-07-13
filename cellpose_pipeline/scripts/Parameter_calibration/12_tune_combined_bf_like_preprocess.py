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

import numpy as np
import tifffile
from cellpose import models
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".tif", ".tiff"}
TIME_KEY_RE = re.compile(r"(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")
KEY_OVERLAY_SET = {
    "A11_1_00d12h00m",
    "C11_1_03d02h00m",
    "C2_2_04d04h00m",
    "E3_3_04d06h00m",
    "F4_1_02d00h00m",
    "G8_1_03d10h00m",
}


@dataclass(frozen=True)
class CombinedConfig:
    tag: str
    transform: str
    model: str = "cpsam"
    diameter: float = 25.0
    flow_threshold: float = 0.0
    cellprob_threshold: float = -1.75
    min_size: int = 20
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    channel_axis: int | None = None


DEFAULT_CONFIGS = [
    CombinedConfig("rgb_bf", "rgb", channel_axis=2),
    CombinedConfig("gray_luma_bf", "gray_luma"),
    CombinedConfig("gray_mean_bf", "gray_mean"),
    CombinedConfig("gray_median_bf", "gray_median"),
    CombinedConfig("gray_min_bf", "gray_min"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune BF-like preprocessing for RGB Combined SUM159 images.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing Combined TIFF files.")
    parser.add_argument("--baseline-run", type=Path, required=True, help="Run dir with Brightfield/Nuclei segmentation_summary.csv.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", default="combined_bf_like_preprocess")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_key(name: str) -> str:
    match = TIME_KEY_RE.search(name)
    if not match:
        raise ValueError(f"Cannot extract key from {name}")
    return match.group(1)


def iter_images(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_baseline_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    summary = path / "segmentation_summary.csv"
    with summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["folder_profile"], extract_key(Path(row["image_path"]).name))] = row
    return rows


def normalize_percentile(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    data = arr.astype(np.float32, copy=False)
    low, high = np.percentile(data, [low_pct, high_pct])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def prepare_image(raw: np.ndarray, config: CombinedConfig) -> np.ndarray:
    rgb = raw.astype(np.float32, copy=False)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Expected RGB channel-last image, got shape={raw.shape}")
    if config.transform == "rgb":
        prepared = rgb
    elif config.transform == "gray_luma":
        prepared = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif config.transform == "gray_mean":
        prepared = rgb.mean(axis=2)
    elif config.transform == "gray_median":
        prepared = np.median(rgb, axis=2)
    elif config.transform == "gray_min":
        prepared = rgb.min(axis=2)
    else:
        raise ValueError(f"Unsupported transform: {config.transform}")
    return normalize_percentile(prepared, config.low_percentile, config.high_percentile)


def stable_color(mask_id: int) -> np.ndarray:
    rng = np.random.default_rng(mask_id * 1103515245 + 12345)
    color = rng.uniform(0.15, 1.0, size=3).astype(np.float32)
    return color / max(float(color.max()), 1e-6)


def normalize_rgb_for_display(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype(np.float32, copy=False)
    if raw.dtype.kind in "ui":
        return np.clip(arr / float(np.iinfo(raw.dtype).max), 0.0, 1.0)
    low, high = np.percentile(arr, [1, 99])
    return np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)


def make_overlay(raw_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    overlay = normalize_rgb_for_display(raw_rgb)
    for mask_id in np.unique(mask):
        mask_id = int(mask_id)
        if mask_id == 0:
            continue
        pixels = mask == mask_id
        overlay[pixels] = 0.45 * overlay[pixels] + 0.55 * stable_color(mask_id)
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    areas = np.bincount(mask.astype(np.int64, copy=False).ravel())[1:]
    n_objects = int(mask.max()) if mask.size else 0
    if areas.size == 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
            "small_fraction_lt60": 0.0,
            "large_fraction_gt2500": 0.0,
        }
    return {
        "n_objects": n_objects,
        "mask_fraction": float((mask > 0).sum() / mask.size),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "area_p90": float(np.percentile(areas, 90)),
        "small_fraction_lt60": float((areas < 60).mean()),
        "large_fraction_gt2500": float((areas > 2500).mean()),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(paths: list[Path], out_path: Path, title: str, columns: int = 3) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 420, 310
    label_h = 40
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
        draw.text((x + 4, y + 4), f"{path.stem}"[:70], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def candidate_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median(values: list[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")

    count_ratios_bf = [float(row["ratio_to_brightfield"]) for row in rows if row["ratio_to_brightfield"] not in ("", "nan")]
    count_ratios_nuc = [float(row["ratio_to_nuclei"]) for row in rows if row["ratio_to_nuclei"] not in ("", "nan")]
    small_fracs = [float(row["small_fraction_lt60"]) for row in rows]
    mask_fracs = [float(row["mask_fraction"]) for row in rows]
    count_penalty = median([abs(math.log(max(x, 1e-6))) for x in count_ratios_bf])
    count_penalty += 0.5 * median([abs(math.log(max(x, 1e-6))) for x in count_ratios_nuc])
    small_penalty = max(0.0, median(small_fracs) - 0.18)
    overmask_penalty = max(0.0, median(mask_fracs) - 0.75) * 0.5
    score = count_penalty + small_penalty + overmask_penalty
    first = rows[0]
    return {
        "config": first["config"],
        "score": f"{score:.6f}",
        "median_ratio_to_brightfield": f"{median(count_ratios_bf):.6f}",
        "median_ratio_to_nuclei": f"{median(count_ratios_nuc):.6f}",
        "median_small_fraction_lt60": f"{median(small_fracs):.6f}",
        "median_mask_fraction": f"{median(mask_fracs):.6f}",
        "median_n_objects": f"{median([float(row['n_objects']) for row in rows]):.3f}",
    }


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve() / args.stage
    out_root.mkdir(parents=True, exist_ok=True)
    images = iter_images(args.input_root)
    if not images:
        raise SystemExit(f"No images found under {args.input_root}")

    baseline = read_baseline_summary(args.baseline_run)
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model="cpsam")
    print(f"model_device={model.device}", flush=True)

    rows: list[dict[str, Any]] = []
    by_config: dict[str, list[dict[str, Any]]] = {}
    for config in DEFAULT_CONFIGS:
        for image_path in images:
            key = extract_key(image_path.name)
            raw = tifffile.imread(image_path)
            mask_path = out_root / "masks" / config.tag / f"{image_path.stem}_cp_masks.tif"
            if args.skip_existing and mask_path.exists():
                mask = tifffile.imread(mask_path)
                elapsed = 0.0
                print(f"skip\t{config.tag}\t{image_path.name}", flush=True)
            else:
                prepared = prepare_image(raw, config)
                print(f"run\t{config.tag}\t{image_path.name}\tshape={prepared.shape}", flush=True)
                start = time.time()
                mask, *_ = model.eval(
                    prepared,
                    channel_axis=config.channel_axis,
                    normalize=False,
                    diameter=config.diameter,
                    flow_threshold=config.flow_threshold,
                    cellprob_threshold=config.cellprob_threshold,
                    min_size=config.min_size,
                )
                elapsed = time.time() - start
                mask = np.asarray(mask)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(mask_path, mask.astype(np.uint16 if mask.max() < 65535 else np.uint32, copy=False))

            overlay_path = out_root / "overlays" / config.tag / f"{image_path.stem}_overlay.png"
            if key in KEY_OVERLAY_SET:
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                make_overlay(raw, mask).save(overlay_path)

            stats = summarize_mask(mask)
            bf_count = int(float(baseline.get(("Brightfield", key), {}).get("n_objects", 0) or 0))
            nuclei_count = int(float(baseline.get(("Nuclei", key), {}).get("n_objects", 0) or 0))
            row = {
                "stage": args.stage,
                "config": config.tag,
                "key": key,
                "image": image_path.name,
                **{f"param_{k}": v for k, v in asdict(config).items()},
                "baseline_brightfield_count": bf_count,
                "baseline_nuclei_count": nuclei_count,
                "ratio_to_brightfield": f"{stats['n_objects'] / bf_count:.6f}" if bf_count else "nan",
                "ratio_to_nuclei": f"{stats['n_objects'] / nuclei_count:.6f}" if nuclei_count else "nan",
                "elapsed_sec": f"{elapsed:.3f}",
                **stats,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
            rows.append(row)
            by_config.setdefault(config.tag, []).append(row)

    write_rows(out_root / "summary.csv", rows)
    scores = [candidate_score(config_rows) for config_rows in by_config.values()]
    scores.sort(key=lambda row: float(row["score"]))
    write_rows(out_root / "config_scores.csv", scores)
    if args.make_contact_sheets:
        for score_row in scores:
            tag = score_row["config"]
            paths = [
                Path(row["overlay_path"])
                for row in by_config[tag]
                if row["key"] in KEY_OVERLAY_SET and Path(row["overlay_path"]).exists()
            ]
            write_contact_sheet(paths, out_root / "contact_sheets" / f"{tag}_key_examples.png", tag)

    lines = [
        "# Combined BF-like Preprocess Tuning",
        "",
        f"- input_root: `{args.input_root}`",
        f"- baseline_run: `{args.baseline_run}`",
        f"- out_root: `{out_root}`",
        "",
        "| rank | config | score | median ratio BF | median ratio Nuclei | small frac | mask frac | median objects |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(scores, start=1):
        lines.append(
            f"| {idx} | {row['config']} | {row['score']} | {row['median_ratio_to_brightfield']} | "
            f"{row['median_ratio_to_nuclei']} | {row['median_small_fraction_lt60']} | "
            f"{row['median_mask_fraction']} | {row['median_n_objects']} |"
        )
    (out_root / "tuning_report.md").write_text("\n".join(lines) + "\n")
    print(f"summary={out_root / 'summary.csv'}", flush=True)
    print(f"scores={out_root / 'config_scores.csv'}", flush=True)
    print(f"report={out_root / 'tuning_report.md'}", flush=True)


if __name__ == "__main__":
    main()
