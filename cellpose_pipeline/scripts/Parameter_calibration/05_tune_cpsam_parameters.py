#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

from cellpose import models


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DEFAULT_FOLDERS = ("Brightfield", "Dead", "Nuclei")
DEFAULT_MODELS = ("cpsam", "cpsam_v2")


@dataclass(frozen=True)
class SegConfig:
    tag: str
    folder: str
    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    low_percentile: float
    high_percentile: float
    invert: bool = False


SCREEN_CONFIGS = [
    SegConfig("bf_inv_d30_cp0", "Brightfield", 30, 0.4, 0.0, 50, 1.0, 99.0, True),
    SegConfig("bf_inv_d45_cp0", "Brightfield", 45, 0.4, 0.0, 80, 1.0, 99.0, True),
    SegConfig("bf_inv_d60_cp-0p5", "Brightfield", 60, 0.4, -0.5, 120, 1.0, 99.0, True),
    SegConfig("bf_raw_d45_cp0", "Brightfield", 45, 0.4, 0.0, 80, 1.0, 99.0, False),
    SegConfig("bf_inv_d45_cp-1", "Brightfield", 45, 0.4, -1.0, 80, 1.0, 99.0, True),
    SegConfig("bf_inv_d30_cp-1", "Brightfield", 30, 0.4, -1.0, 50, 1.0, 99.0, True),
    SegConfig("dead_d20_cp0", "Dead", 20, 0.4, 0.0, 10, 1.0, 99.8, False),
    SegConfig("dead_d30_cp0", "Dead", 30, 0.4, 0.0, 20, 1.0, 99.8, False),
    SegConfig("dead_d40_cp0", "Dead", 40, 0.4, 0.0, 30, 1.0, 99.8, False),
    SegConfig("dead_d30_cp-1", "Dead", 30, 0.4, -1.0, 20, 1.0, 99.8, False),
    SegConfig("dead_d20_cp-1", "Dead", 20, 0.4, -1.0, 10, 1.0, 99.8, False),
    SegConfig("dead_d40_cp-1", "Dead", 40, 0.4, -1.0, 30, 1.0, 99.8, False),
    SegConfig("nuc_d18_cp0", "Nuclei", 18, 0.4, 0.0, 10, 1.0, 99.8, False),
    SegConfig("nuc_d24_cp0", "Nuclei", 24, 0.4, 0.0, 20, 1.0, 99.8, False),
    SegConfig("nuc_d30_cp0", "Nuclei", 30, 0.4, 0.0, 30, 1.0, 99.8, False),
    SegConfig("nuc_d24_cp-1", "Nuclei", 24, 0.4, -1.0, 20, 1.0, 99.8, False),
    SegConfig("nuc_d18_cp-1", "Nuclei", 18, 0.4, -1.0, 10, 1.0, 99.8, False),
    SegConfig("nuc_d30_cp-1", "Nuclei", 30, 0.4, -1.0, 30, 1.0, 99.8, False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune cpsam/cpsam_v2 segmentation parameters on SUM159 test TIFFs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "validate"), default="screen")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--folders", nargs="+", default=list(DEFAULT_FOLDERS))
    parser.add_argument("--representative-only", action="store_true", help="Use the middle sorted image per folder.")
    parser.add_argument("--config-json", type=Path, help="Optional list of SegConfig dicts for validation.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-contact-sheets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def iter_images(input_root: Path, folder: str) -> list[Path]:
    root = input_root / folder
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def representative_image(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    return [paths[len(paths) // 2]]


def load_configs(args: argparse.Namespace) -> list[SegConfig]:
    if args.config_json:
        payload = json.loads(args.config_json.read_text())
        return [SegConfig(**row) for row in payload]
    return [config for config in SCREEN_CONFIGS if config.folder in set(args.folders)]


def normalize_image(image: np.ndarray, low_percentile: float, high_percentile: float, invert: bool) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    low, high = np.percentile(data, [low_percentile, high_percentile])
    if high <= low:
        high = low + 1.0
    norm = np.clip((data - low) / (high - low), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    return norm.astype(np.float32, copy=False)


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
    mask_pixels = mask > 0
    rgb[mask_pixels, 1] = np.maximum(rgb[mask_pixels, 1], 90)
    rgb[mask_pixels, 2] = np.maximum(rgb[mask_pixels, 2], 55)
    edge = boundary_mask(mask)
    rgb[edge] = np.array([255, 32, 32], dtype=np.uint8)
    return Image.fromarray(rgb)


def write_contact_sheet(overlay_paths: list[Path], out_path: Path, title: str, columns: int = 3) -> None:
    if not overlay_paths:
        return
    thumb_w, thumb_h = 352, 260
    label_h = 42
    rows = (len(overlay_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h) + 34), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for idx, path in enumerate(overlay_paths):
        row, col = divmod(idx, columns)
        x = col * thumb_w
        y = 34 + row * (thumb_h + label_h)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + label_h))
        label = path.parent.name
        draw.text((x + 4, y + 4), label[:55], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    n_objects = int(mask.max())
    area = np.bincount(mask.ravel())[1:]
    if area.size == 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
            "small_fraction": 0.0,
            "large_fraction": 0.0,
        }
    total_pixels = mask.size
    return {
        "n_objects": n_objects,
        "mask_fraction": float((mask > 0).sum() / total_pixels),
        "area_median": float(np.median(area)),
        "area_p10": float(np.percentile(area, 10)),
        "area_p90": float(np.percentile(area, 90)),
        "small_fraction": float((area < 50).mean()),
        "large_fraction": float((area > 10000).mean()),
    }


def run_one(model: models.CellposeModel, image_path: Path, config: SegConfig) -> tuple[np.ndarray, np.ndarray, float]:
    raw = tifffile.imread(image_path)
    norm = normalize_image(raw, config.low_percentile, config.high_percentile, config.invert)
    start = time.time()
    masks, *_ = model.eval(
        norm,
        channel_axis=None,
        normalize=False,
        diameter=config.diameter,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        min_size=config.min_size,
    )
    elapsed = time.time() - start
    return norm, masks.astype(np.uint16, copy=False), elapsed


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    run_root = out_root / args.stage
    configs = load_configs(args)
    configs_by_folder = {
        folder: [config for config in configs if config.folder == folder]
        for folder in args.folders
    }

    image_map: dict[str, list[Path]] = {}
    for folder in args.folders:
        paths = iter_images(args.input_root, folder)
        image_map[folder] = representative_image(paths) if args.representative_only else paths
        if not image_map[folder]:
            raise SystemExit(f"No images found for {folder}")

    config_dump = run_root / "configs.json"
    config_dump.parent.mkdir(parents=True, exist_ok=True)
    config_dump.write_text(json.dumps([asdict(config) for config in configs], indent=2))

    rows: list[dict] = []
    for model_name in args.models:
        print(f"loading_model={model_name}", flush=True)
        model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=model_name)
        print(f"model_device={model.device}", flush=True)
        for folder in args.folders:
            for config in configs_by_folder[folder]:
                for image_path in image_map[folder]:
                    mask_path = run_root / "masks" / folder / model_name / config.tag / f"{image_path.stem}_masks.tif"
                    overlay_path = run_root / "overlays" / folder / model_name / config.tag / f"{image_path.stem}_overlay.png"
                    if args.skip_existing and mask_path.exists() and overlay_path.exists():
                        mask = tifffile.imread(mask_path)
                        elapsed = 0.0
                        print(f"skip_existing\t{model_name}\t{folder}\t{config.tag}\t{image_path.name}", flush=True)
                    else:
                        print(f"run\t{model_name}\t{folder}\t{config.tag}\t{image_path.name}", flush=True)
                        norm, mask, elapsed = run_one(model, image_path, config)
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        overlay_path.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(mask_path, mask)
                        make_overlay(norm, mask).save(overlay_path)
                    stats = summarize_mask(mask)
                    row = {
                        "stage": args.stage,
                        "folder": folder,
                        "model": model_name,
                        "config": config.tag,
                        "image": image_path.name,
                        "elapsed_sec": round(elapsed, 3),
                        **asdict(config),
                        **stats,
                        "mask_path": str(mask_path),
                        "overlay_path": str(overlay_path),
                    }
                    rows.append(row)

    summary_path = run_root / "summary.csv"
    write_rows(summary_path, rows)

    if args.make_contact_sheets:
        for folder in args.folders:
            for model_name in args.models:
                images = image_map[folder]
                for image_path in images:
                    overlay_paths = [
                        run_root / "overlays" / folder / model_name / config.tag / f"{image_path.stem}_overlay.png"
                        for config in configs_by_folder[folder]
                    ]
                    overlay_paths = [path for path in overlay_paths if path.exists()]
                    write_contact_sheet(
                        overlay_paths,
                        run_root / "contact_sheets" / folder / model_name / f"{image_path.stem}.png",
                        f"{args.stage} {folder} {model_name} {image_path.name}",
                    )

    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
