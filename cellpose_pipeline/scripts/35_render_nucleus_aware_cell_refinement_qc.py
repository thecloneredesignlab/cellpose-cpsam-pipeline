#!/usr/bin/env python3
"""Render before/after crop mosaics for nucleus-aware cell-mask refinements."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
PROFILES = ("Combined", "Brightfield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render focused before/after cell-mask QC mosaics.")
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repair-events", type=Path, required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keys", nargs="*")
    parser.add_argument("--max-crops", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=192)
    return parser.parse_args()


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def index_files(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        key = key_from_path(path)
        if key in result:
            raise ValueError(f"Duplicate field {key} in {directory}")
        result[key] = path
    return result


def index_raw(input_root: Path, profile: str) -> dict[str, Path]:
    directory = input_root / profile
    ranks = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ranks:
            continue
        try:
            key = key_from_path(path)
        except ValueError:
            continue
        previous = result.get(key)
        if previous is None or ranks[path.suffix.lower()] < ranks[previous.suffix.lower()]:
            result[key] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {mask.shape}")
    return mask.astype(np.int32, copy=False)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def robust_rgb(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    elif image.ndim == 3 and image.shape[0] >= 3 and image.shape[0] <= 4:
        image = np.moveaxis(image[:3], 0, -1)
    else:
        raise ValueError(f"Cannot convert raw image shape {image.shape} to RGB")
    arr = image.astype(np.float64, copy=False)
    lo, hi = np.percentile(arr[np.isfinite(arr)], (1.0, 99.0))
    scaled = np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def label_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def make_overlay(
    raw_rgb: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_extent: np.ndarray,
    nucleus_core: np.ndarray,
    changed: np.ndarray,
    cell_color: tuple[int, int, int],
) -> np.ndarray:
    output = raw_rgb.astype(np.float64, copy=True)
    changed_outline = ndimage.binary_dilation(binary_boundary(changed), iterations=1)
    cell_edge = label_boundary(cell_mask)
    extent_edge = label_boundary(nucleus_extent) & (nucleus_extent > 0)
    core_edge = label_boundary(nucleus_core) & (nucleus_core > 0)
    output[cell_edge] = cell_color
    output[extent_edge] = (245, 65, 65)
    output[core_edge] = (65, 230, 105)
    output[changed_outline] = (245, 35, 205)
    return np.clip(output, 0, 255).astype(np.uint8)


def crop_fixed(image: np.ndarray, center_y: float, center_x: float, size: int) -> np.ndarray:
    half = size // 2
    y0 = int(round(center_y)) - half
    x0 = int(round(center_x)) - half
    y1 = y0 + size
    x1 = x0 + size
    output = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, image.shape[0])
    src_x1 = min(x1, image.shape[1])
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    output[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    return output


def main() -> int:
    args = parse_args()
    args.baseline_run_root = args.baseline_run_root.resolve()
    args.candidate_run_root = args.candidate_run_root.resolve()
    args.input_root = args.input_root.resolve()
    args.repair_events = args.repair_events.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_masks = {
        profile: index_files(args.baseline_run_root / profile / "segmentations", "*_cp_masks.tif")
        for profile in PROFILES
    }
    candidate_masks = {
        profile: index_files(args.candidate_run_root / profile / "segmentations", "*_cp_masks.tif")
        for profile in PROFILES
    }
    extent_paths = index_files(args.baseline_run_root / "Nuclei" / "segmentations", "*_cp_masks.tif")
    core_paths = index_files(
        args.baseline_run_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif"
    )
    raw_paths = {profile: index_raw(args.input_root, profile) for profile in PROFILES}

    events_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.repair_events.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["candidate"] == args.candidate_tag:
                events_by_key[row["key"]].append(row)
    keys = sorted(events_by_key)
    if args.keys:
        requested = set(args.keys)
        keys = [key for key in keys if key in requested]

    panel_names = ("Combined before", "Combined after", "BF before", "BF after")
    font = ImageFont.load_default()
    for key in keys:
        print(f"rendering={key}", flush=True)
        extent = read_mask(extent_paths[key])
        core = read_mask(core_paths[key])
        baseline = {profile: read_mask(baseline_masks[profile][key]) for profile in PROFILES}
        candidate = {profile: read_mask(candidate_masks[profile][key]) for profile in PROFILES}
        raw = {profile: robust_rgb(read_image(raw_paths[profile][key])) for profile in PROFILES}
        changed = {profile: baseline[profile] != candidate[profile] for profile in PROFILES}
        overlays = {
            ("Combined", "before"): make_overlay(
                raw["Combined"], baseline["Combined"], extent, core, changed["Combined"], (35, 205, 235)
            ),
            ("Combined", "after"): make_overlay(
                raw["Combined"], candidate["Combined"], extent, core, changed["Combined"], (35, 205, 235)
            ),
            ("Brightfield", "before"): make_overlay(
                raw["Brightfield"], baseline["Brightfield"], extent, core, changed["Brightfield"], (255, 190, 35)
            ),
            ("Brightfield", "after"): make_overlay(
                raw["Brightfield"], candidate["Brightfield"], extent, core, changed["Brightfield"], (255, 190, 35)
            ),
        }
        ranked = sorted(
            events_by_key[key],
            key=lambda row: -(
                int(row["combined_changed_pixels"]) + int(row["brightfield_changed_pixels"])
            ),
        )[: args.max_crops]
        header = 52
        label_height = 24
        row_height = label_height + args.crop_size
        canvas = Image.new(
            "RGB",
            (4 * args.crop_size, header + len(ranked) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"{key} | {args.candidate_tag} | magenta=changed", fill=(20, 25, 30), font=font)
        draw.text(
            (8, 25),
            "cell boundary: cyan/yellow | nucleus extent: red | intensity core: green",
            fill=(65, 70, 75),
            font=font,
        )
        for column, panel_name in enumerate(panel_names):
            draw.text((column * args.crop_size + 6, 39), panel_name, fill=(20, 25, 30), font=font)
        for row_index, event in enumerate(ranked):
            nucleus_id = int(event["nucleus_id"])
            yy, xx = np.nonzero(extent == nucleus_id)
            if not yy.size:
                continue
            center_y = float(yy.mean())
            center_x = float(xx.mean())
            y = header + row_index * row_height
            descriptor = (
                f"nucleus {nucleus_id} | changed C={event['combined_changed_pixels']} "
                f"BF={event['brightfield_changed_pixels']}"
            )
            draw.text((6, y + 5), descriptor, fill=(35, 40, 45), font=font)
            panels = (
                overlays[("Combined", "before")],
                overlays[("Combined", "after")],
                overlays[("Brightfield", "before")],
                overlays[("Brightfield", "after")],
            )
            for column, panel in enumerate(panels):
                crop = crop_fixed(panel, center_y, center_x, args.crop_size)
                canvas.paste(Image.fromarray(crop), (column * args.crop_size, y + label_height))
        output_path = args.out_dir / f"{key}_{args.candidate_tag}_comparison.png"
        canvas.save(output_path, optimize=True)
    print(f"qc_dir={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
