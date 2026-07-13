#!/usr/bin/env python3
"""Render focused before/after QC mosaics for shape-aware nucleus splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
PROFILES = ("Nuclei", "Combined", "Brightfield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render shape-aware nucleus split comparisons.")
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--cell-run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--split-events", type=Path, required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keys", nargs="*")
    parser.add_argument(
        "--field-manifest-dir",
        type=Path,
        help="Manifest directory containing records/<well>/<key>.json for direct QC lookup.",
    )
    parser.add_argument(
        "--cell-mask-branch",
        choices=("original", "nucleated"),
        default="original",
    )
    parser.add_argument("--max-crops", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=224)
    return parser.parse_args()


def key_from_path(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(path)
    return match.group(1)


def index_files(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        key = key_from_path(path)
        if key in result:
            raise ValueError(f"Duplicate {key} in {directory}")
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


def direct_qc_indexes(
    manifest_dir: Path,
    keys: list[str],
    candidate_run_root: Path,
    cell_mask_branch: str,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, dict[str, Path]],
]:
    baseline_extent: dict[str, Path] = {}
    candidate_extent: dict[str, Path] = {}
    combined_paths: dict[str, Path] = {}
    bf_paths: dict[str, Path] = {}
    raw_paths: dict[str, dict[str, Path]] = {profile: {} for profile in PROFILES}
    mask_field = "nucleated_mask" if cell_mask_branch == "nucleated" else "original_mask"
    for key in keys:
        record_path = manifest_dir / "records" / key.split("_", 1)[0] / f"{key}.json"
        payload = json.loads(record_path.read_text())
        if payload.get("key") != key:
            raise ValueError(f"Field record key mismatch: expected={key} path={record_path}")
        profiles = payload["profiles"]
        extent = Path(profiles["Nuclei"]["extent_mask"])
        paths = {
            "baseline_extent": extent,
            "candidate_extent": candidate_run_root / "Nuclei" / "segmentations" / extent.name,
            "combined_mask": Path(profiles["Combined"][mask_field]),
            "brightfield_mask": Path(profiles["Brightfield"][mask_field]),
            "nuclei_raw": Path(profiles["Nuclei"]["raw"]),
            "combined_raw": Path(profiles["Combined"]["raw"]),
            "brightfield_raw": Path(profiles["Brightfield"]["raw"]),
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing for {key}: {path}")
        baseline_extent[key] = paths["baseline_extent"]
        candidate_extent[key] = paths["candidate_extent"]
        combined_paths[key] = paths["combined_mask"]
        bf_paths[key] = paths["brightfield_mask"]
        raw_paths["Nuclei"][key] = paths["nuclei_raw"]
        raw_paths["Combined"][key] = paths["combined_raw"]
        raw_paths["Brightfield"][key] = paths["brightfield_raw"]
    return baseline_extent, candidate_extent, combined_paths, bf_paths, raw_paths


def read_mask(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {labels.shape}")
    return labels.astype(np.int32, copy=False)


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
    elif image.ndim == 3 and 3 <= image.shape[0] <= 4:
        image = np.moveaxis(image[:3], 0, -1)
    else:
        raise ValueError(f"Unsupported raw shape: {image.shape}")
    arr = image.astype(np.float64, copy=False)
    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, (1.0, 99.0)) if finite.size else (0.0, 1.0)
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


def binary_boundary(binary: np.ndarray) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    return binary & ~ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))


def draw_cross(image: np.ndarray, y: int, x: int, color: tuple[int, int, int], radius: int = 4) -> None:
    y0 = max(0, y - radius)
    y1 = min(image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(image.shape[1], x + radius + 1)
    image[y, x0:x1] = color
    image[y0:y1, x] = color


def overlay_parent(raw: np.ndarray, parent: np.ndarray, peak1: tuple[int, int], peak2: tuple[int, int]) -> np.ndarray:
    output = raw.copy()
    output[ndimage.binary_dilation(binary_boundary(parent), iterations=1)] = (255, 255, 255)
    draw_cross(output, peak1[0], peak1[1], (245, 35, 205))
    draw_cross(output, peak2[0], peak2[1], (35, 235, 105))
    return output


def overlay_children(
    raw: np.ndarray,
    retained: np.ndarray,
    new_child: np.ndarray,
    cell_mask: np.ndarray | None = None,
    cell_color: tuple[int, int, int] = (255, 190, 35),
) -> np.ndarray:
    output = raw.copy()
    if cell_mask is not None:
        output[label_boundary(cell_mask)] = cell_color
    output[ndimage.binary_dilation(binary_boundary(retained), iterations=1)] = (245, 65, 65)
    output[ndimage.binary_dilation(binary_boundary(new_child), iterations=1)] = (35, 225, 235)
    return output


def crop_fixed(image: np.ndarray, center_y: float, center_x: float, size: int) -> np.ndarray:
    half = size // 2
    y0 = int(round(center_y)) - half
    x0 = int(round(center_x)) - half
    y1 = y0 + size
    x1 = x0 + size
    output = np.zeros((size, size, 3), dtype=image.dtype)
    sy0 = max(y0, 0)
    sx0 = max(x0, 0)
    sy1 = min(y1, image.shape[0])
    sx1 = min(x1, image.shape[1])
    dy0 = sy0 - y0
    dx0 = sx0 - x0
    output[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = image[sy0:sy1, sx0:sx1]
    return output


def event_rank(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        float(row["shape_gain"]),
        float(row["valley_drop_fraction"]),
        float(row["stability_fraction"]),
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.split_events.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["candidate"] == args.candidate_tag:
                events_by_key[row["key"]].append(row)
    keys = sorted(events_by_key)
    if args.keys:
        requested = set(args.keys)
        keys = [key for key in keys if key in requested]

    if args.field_manifest_dir is not None:
        baseline_extent, candidate_extent, combined_paths, bf_paths, raw_paths = direct_qc_indexes(
            args.field_manifest_dir,
            keys,
            args.candidate_run_root,
            args.cell_mask_branch,
        )
        print(f"record_source=field_manifest n_selected={len(keys)}", flush=True)
    else:
        baseline_extent = index_files(
            args.baseline_run_root / "Nuclei" / "segmentations", "*_cp_masks.tif"
        )
        candidate_extent = index_files(
            args.candidate_run_root / "Nuclei" / "segmentations", "*_cp_masks.tif"
        )
        combined_paths = index_files(
            args.cell_run_root / "Combined" / "segmentations", "*_cp_masks.tif"
        )
        bf_paths = index_files(
            args.cell_run_root / "Brightfield" / "segmentations", "*_cp_masks.tif"
        )
        raw_paths = {profile: index_raw(args.input_root, profile) for profile in PROFILES}
        print(f"record_source=directory_discovery n_selected={len(keys)}", flush=True)

    font = ImageFont.load_default()
    panel_names = ("parent + peaks", "candidate split", "Combined context", "BF context")
    for key in keys:
        print(f"rendering={key}", flush=True)
        before = read_mask(baseline_extent[key])
        after = read_mask(candidate_extent[key])
        combined = read_mask(combined_paths[key])
        brightfield = read_mask(bf_paths[key])
        raw = {profile: robust_rgb(read_image(raw_paths[profile][key])) for profile in PROFILES}
        ranked = sorted(events_by_key[key], key=event_rank, reverse=True)[: args.max_crops]
        header_height = 54
        row_label_height = 38
        row_height = row_label_height + args.crop_size
        canvas = Image.new(
            "RGB",
            (4 * args.crop_size, header_height + len(ranked) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 6),
            f"{key} | {args.candidate_tag} | parent=white, retained=red, new=cyan",
            fill=(25, 30, 35),
            font=font,
        )
        draw.text(
            (8, 23),
            "peak1=magenta, peak2=green; yellow/cyan lines are cell boundaries",
            fill=(70, 75, 80),
            font=font,
        )
        for column, name in enumerate(panel_names):
            draw.text((column * args.crop_size + 6, 40), name, fill=(25, 30, 35), font=font)

        for row_index, event in enumerate(ranked):
            parent_id = int(event["parent_nucleus_id"])
            retained_id = int(event["retained_child_id"])
            new_id = int(event["new_child_id"])
            parent = before == parent_id
            retained = after == retained_id
            new_child = after == new_id
            yy, xx = np.nonzero(parent)
            if not yy.size:
                continue
            center_y = float(yy.mean())
            center_x = float(xx.mean())
            peak1 = (int(event["peak1_y"]), int(event["peak1_x"]))
            peak2 = (int(event["peak2_y"]), int(event["peak2_x"]))
            panels = (
                overlay_parent(raw["Nuclei"], parent, peak1, peak2),
                overlay_children(raw["Nuclei"], retained, new_child),
                overlay_children(raw["Combined"], retained, new_child, combined, (35, 205, 235)),
                overlay_children(raw["Brightfield"], retained, new_child, brightfield, (255, 190, 35)),
            )
            y = header_height + row_index * row_height
            line1 = (
                f"parent {parent_id}: area={float(event['parent_area']):.0f}, "
                f"sol={float(event['parent_solidity']):.2f}, circ={float(event['parent_circularity']):.2f}, "
                f"axis={float(event['parent_axis_ratio']):.2f}"
            )
            line2 = (
                f"children={event['retained_child_area']}/{event['new_child_area']}, "
                f"peak d={float(event['peak_distance']):.1f}, valley={float(event['valley_drop_fraction']):.2f}, "
                f"stable={float(event['stability_fraction']):.0%}, cell={float(event['cell_support_fraction']):.0%}"
            )
            draw.text((6, y + 4), line1, fill=(35, 40, 45), font=font)
            draw.text((6, y + 20), line2, fill=(35, 40, 45), font=font)
            for column, panel in enumerate(panels):
                crop = crop_fixed(panel, center_y, center_x, args.crop_size)
                canvas.paste(
                    Image.fromarray(crop),
                    (column * args.crop_size, y + row_label_height),
                )
        output = args.out_dir / f"{key}_{args.candidate_tag}_shape_split_qc.png"
        canvas.save(output, optimize=True)
    print(f"qc_dir={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
