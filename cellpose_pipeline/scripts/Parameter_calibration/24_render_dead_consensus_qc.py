#!/usr/bin/env python3
"""Render large-test QC panels for Dead/Combined-blue consensus segmentation."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dead-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=4)
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(path)
    return match.group(1)


def index(directory: Path, pattern: str) -> dict[str, Path]:
    return {extract_key(path): path for path in sorted(directory.glob(pattern))}


def boundaries(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:-1] |= mask[1:] != mask[:-1]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge & (mask > 0)


def gray_rgb(raw: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(raw, [low, high])
    norm = np.clip((raw.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.repeat(norm[..., None], 3, axis=2)


def combined_rgb(raw: np.ndarray) -> np.ndarray:
    rgb = raw[..., :3].astype(np.float32)
    lo, hi = np.percentile(rgb, [1, 99])
    return np.clip((rgb - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def resize_panel(rgb: np.ndarray, title: str, size: tuple[int, int] = (512, 360)) -> Image.Image:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 28), "black")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), title, fill="white")
    return canvas


def provenance_map(path: Path) -> dict[int, str]:
    with path.open(newline="") as handle:
        return {int(row["final_id"]): row["source"] for row in csv.DictReader(handle)}


def render_field(key: str, dead_path: Path, combined_path: Path, dead_run: Path) -> Image.Image:
    dead = np.squeeze(tifffile.imread(dead_path)).astype(np.float32, copy=False)
    combined = tifffile.imread(combined_path)
    primary = np.squeeze(tifffile.imread(dead_run / "intermediate" / "dead_primary" / f"{key}_cp_masks.tif"))
    blue = np.squeeze(tifffile.imread(dead_run / "intermediate" / "combined_blue" / f"{key}_cp_masks.tif"))
    final_path = next((dead_run / "segmentations").glob(f"*{key}_cp_masks.tif"))
    final = np.squeeze(tifffile.imread(final_path))
    sources = provenance_map(dead_run / "object_provenance" / f"{key}.csv")

    dead_panel = gray_rgb(dead)
    dead_panel[boundaries(primary)] = np.array([1.0, 0.1, 0.1])
    rgb = combined_rgb(combined)
    red, green, blue_raw = combined[..., 0].astype(np.float32), combined[..., 1].astype(np.float32), combined[..., 2].astype(np.float32)
    excess = np.clip(blue_raw - 0.5 * (red + green), 0.0, None)
    blue_panel = gray_rgb(excess, 0.0, 99.8)
    blue_panel[boundaries(blue)] = np.array([0.0, 0.9, 1.0])
    consensus = rgb.copy()
    edge = boundaries(final)
    colors = {
        "dual_channel": np.array([0.1, 1.0, 0.1]),
        "dead_primary_only": np.array([1.0, 0.15, 0.15]),
        "combined_blue_rescue": np.array([0.1, 0.5, 1.0]),
    }
    for label_id, source in sources.items():
        consensus[edge & (final == label_id)] = colors[source]

    panels = [
        resize_panel(dead_panel, "Dead raw + primary mask (red)"),
        resize_panel(rgb, "Combined RGB"),
        resize_panel(blue_panel, "Blue excess + mask (cyan)"),
        resize_panel(consensus, "Consensus: dual green / Dead red / rescue blue"),
    ]
    field = Image.new("RGB", (1024, 2 * panels[0].height + 32), "#202020")
    for idx, panel in enumerate(panels):
        field.paste(panel, ((idx % 2) * 512, (idx // 2) * panel.height + 32))
    ImageDraw.Draw(field).text((8, 8), key, fill="white")
    return field


def main() -> int:
    args = parse_args()
    dead_index = index(args.input_root / "Dead", "*.tif")
    combined_index = index(args.input_root / "Combined", "*.tif")
    metadata_keys = sorted(path.stem for path in (args.dead_run / "metadata").glob("*.json"))
    keys = [key for key in metadata_keys if key in dead_index and key in combined_index]
    if not keys:
        raise SystemExit("No consensus fields found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Image.Image]] = []
    for key in keys:
        image = render_field(key, dead_index[key], combined_index[key], args.dead_run)
        image.save(args.out_dir / f"{key}_dead_consensus_qc.png")
        rendered.append((key, image))
    for page_index in range(math.ceil(len(rendered) / args.page_size)):
        subset = rendered[page_index * args.page_size : (page_index + 1) * args.page_size]
        page = Image.new("RGB", (2048, 2 * subset[0][1].height), "#101010")
        for index_in_page, (_key, image) in enumerate(subset):
            page.paste(image, ((index_in_page % 2) * 1024, (index_in_page // 2) * image.height))
        page.save(args.out_dir / f"dead_consensus_contact_sheet_{page_index + 1:02d}.png")
    print(f"dead_consensus_qc_complete={len(rendered)} pages={math.ceil(len(rendered) / args.page_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
