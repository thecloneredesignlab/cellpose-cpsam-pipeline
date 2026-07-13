#!/usr/bin/env python3
"""Render baseline-versus-selected QC sheets for high-density cell masks."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")

SCORER_PATH = Path(__file__).with_name("20_score_high_density_bf_combined.py")
SCORER_SPEC = importlib.util.spec_from_file_location("high_density_scorer_local", SCORER_PATH)
if SCORER_SPEC is None or SCORER_SPEC.loader is None:
    raise ImportError(f"Cannot load scoring utilities from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SCORER_SPEC)
sys.modules[SCORER_SPEC.name] = SCORER_MODULE
SCORER_SPEC.loader.exec_module(SCORER_MODULE)
label_boundary = SCORER_MODULE.label_boundary
nucleus_core_assignment = SCORER_MODULE.nucleus_core_assignment
read_mask = SCORER_MODULE.read_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render high-density BF/Combined candidate QC.")
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-crops", type=int, default=3)
    return parser.parse_args()


def extract_key(path_or_name: Path | str) -> str:
    match = KEY_RE.search(Path(path_or_name).name)
    if match is None:
        raise ValueError(f"Cannot extract key from {path_or_name}")
    return match.group(1)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_display(raw: np.ndarray) -> np.ndarray:
    data = raw.astype(np.float32, copy=False)
    if data.ndim == 2:
        low, high = np.percentile(data, (1, 99))
        norm = np.clip((data - low) / max(high - low, 1e-6), 0.0, 1.0)
        return np.repeat(norm[:, :, None], 3, axis=2)
    if data.ndim == 3 and data.shape[-1] >= 3:
        rgb = data[..., :3]
        output = np.zeros(rgb.shape, dtype=np.float32)
        for channel in range(3):
            low, high = np.percentile(rgb[..., channel], (1, 99))
            output[..., channel] = np.clip(
                (rgb[..., channel] - low) / max(high - low, 1e-6), 0.0, 1.0
            )
        return output
    raise ValueError(f"Cannot display image shape={raw.shape}")


def core_centroids(core: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(core)
    if not yy.size:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    ids = core[yy, xx].astype(np.int64, copy=False)
    maximum = int(ids.max())
    areas = np.bincount(ids, minlength=maximum + 1).astype(np.float64)
    sum_y = np.bincount(ids, weights=yy, minlength=maximum + 1)
    sum_x = np.bincount(ids, weights=xx, minlength=maximum + 1)
    valid = np.flatnonzero(areas > 0)
    valid = valid[valid > 0]
    return valid.astype(np.int32), sum_y[valid] / areas[valid], sum_x[valid] / areas[valid]


def overlay(raw: np.ndarray, cell: np.ndarray, core: np.ndarray) -> Image.Image:
    display = np.clip(normalize_display(raw) * 255.0, 0, 255).astype(np.uint8)
    cell_edge = label_boundary(cell)
    core_edge = label_boundary(core) & (core > 0)
    display[cell_edge] = np.array([32, 255, 32], dtype=np.uint8)
    display[core_edge] = np.array([0, 255, 255], dtype=np.uint8)
    image = Image.fromarray(display)
    draw = ImageDraw.Draw(image)
    assignment = nucleus_core_assignment(core, cell, 0.80, 0.50, 0.10)
    ids, cy, cx = core_centroids(core)
    for object_id, y, x in zip(ids, cy, cx):
        supported = bool(assignment["supported"][object_id])
        conflict = bool(assignment["conflict"][object_id])
        if supported and not conflict:
            continue
        color = (255, 0, 0) if not supported else (255, 180, 0)
        radius = 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    return image


def crop_box(center_y: int, center_x: int, height: int, width: int, size: int) -> tuple[int, int, int, int]:
    half = size // 2
    y0 = min(max(center_y - half, 0), max(height - size, 0))
    x0 = min(max(center_x - half, 0), max(width - size, 0))
    y1 = min(y0 + size, height)
    x1 = min(x0 + size, width)
    return x0, y0, x1, y1


def choose_crops(score: np.ndarray, size: int, maximum: int) -> list[tuple[int, int, int, int]]:
    height, width = score.shape
    if height <= size or width <= size:
        return [(0, 0, width, height)]
    step = max(size // 2, 32)
    integral = np.pad(score.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    candidates: list[tuple[int, int, int]] = []
    for y0 in range(0, height - size + 1, step):
        for x0 in range(0, width - size + 1, step):
            y1, x1 = y0 + size, x0 + size
            value = int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
            candidates.append((value, y0 + size // 2, x0 + size // 2))
    candidates.sort(reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    centers: list[tuple[int, int]] = []
    for _value, cy, cx in candidates:
        if any((cy - oy) ** 2 + (cx - ox) ** 2 < (size * 0.70) ** 2 for oy, ox in centers):
            continue
        selected.append(crop_box(cy, cx, height, width, size))
        centers.append((cy, cx))
        if len(selected) >= maximum:
            break
    return selected


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy().convert("RGB")
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def render_sheet(
    key: str,
    columns: list[tuple[str, Image.Image]],
    crops: list[tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    panel_w, overview_h, crop_h = 360, 270, 300
    label_h, title_h = 34, 42
    total_h = title_h + label_h + overview_h + len(crops) * (label_h + crop_h)
    sheet = Image.new("RGB", (panel_w * len(columns), total_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 12), f"High-density BF/Combined QC: {key}", fill=(0, 0, 0), font=font)
    y = title_h
    for col, (label, image) in enumerate(columns):
        x = col * panel_w
        draw.text((x + 5, y + 9), label, fill=(0, 0, 0), font=font)
        sheet.paste(fit_panel(image, panel_w, overview_h), (x, y + label_h))
    y += label_h + overview_h
    for crop_index, box in enumerate(crops, start=1):
        for col, (label, image) in enumerate(columns):
            x = col * panel_w
            draw.text(
                (x + 5, y + 9),
                f"crop {crop_index}: {box} | {label}",
                fill=(0, 0, 0),
                font=font,
            )
            sheet.paste(fit_panel(image.crop(box), panel_w, crop_h), (x, y + label_h))
        y += label_h + crop_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def render_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    panel_w, panel_h, label_h, columns = 480, 420, 30, 2
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_w, rows * (panel_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(paths):
        row, col = divmod(index, columns)
        x, y = col * panel_w, row * (panel_h + label_h)
        draw.text((x + 5, y + 8), path.stem, fill=(0, 0, 0), font=font)
        image = Image.open(path).convert("RGB")
        sheet.paste(fit_panel(image, panel_w, panel_h), (x, y + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    screen_root = args.screen_root.resolve()
    scoring_dir = (args.scoring_dir or screen_root / "scoring").resolve()
    out_dir = (args.out_dir or screen_root / "qc").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((screen_root / "screen_manifest.json").read_text())
    selected = json.loads((scoring_dir / "selected_pair.json").read_text())
    inference_rows = csv_rows(screen_root / "inference_summary.csv")
    index = {
        (row["profile"], row["config"], row["key"]): row
        for row in inference_rows
    }
    configs = manifest["selected_configs"]
    baseline_tags = {
        profile: next(
            row["tag"] for row in configs if row["profile"] == profile and row.get("baseline", False)
        )
        for profile in ("Brightfield", "Combined")
    }
    selected_tags = {
        "Brightfield": selected["brightfield"]["tag"],
        "Combined": selected["combined"]["tag"],
    }

    core_run = Path(manifest["nuclei_run_root"])
    core_dir = next(
        path
        for path in (
            core_run / "Nuclei" / "nucleus_core_seeds",
            core_run / "nucleus_core_seeds" / "Nuclei",
            core_run / "nucleus_core_seeds",
        )
        if path.is_dir()
    )
    core_paths = {extract_key(path): path for path in core_dir.glob("*_core_masks.tif")}

    qc_rows: list[dict[str, Any]] = []
    sheet_paths: list[Path] = []
    for key in manifest["selected_keys"]:
        core = read_mask(core_paths[key])
        images: dict[str, np.ndarray] = {}
        masks: dict[tuple[str, str], np.ndarray] = {}
        overlays: dict[tuple[str, str], Image.Image] = {}
        for profile in ("Brightfield", "Combined"):
            first = index[(profile, baseline_tags[profile], key)]
            images[profile] = tifffile.imread(Path(first["image_path"]))
            for tag in {baseline_tags[profile], selected_tags[profile]}:
                mask = read_mask(Path(index[(profile, tag, key)]["mask_path"]))
                masks[(profile, tag)] = mask
                overlays[(profile, tag)] = overlay(images[profile], mask, core)

        bf_base_boundary = label_boundary(masks[("Brightfield", baseline_tags["Brightfield"])])
        bf_selected_boundary = label_boundary(masks[("Brightfield", selected_tags["Brightfield"])])
        combined_base_boundary = label_boundary(masks[("Combined", baseline_tags["Combined"])])
        combined_selected_boundary = label_boundary(masks[("Combined", selected_tags["Combined"])])
        disagreement = (
            (bf_base_boundary != bf_selected_boundary)
            | (combined_base_boundary != combined_selected_boundary)
            | (bf_selected_boundary != combined_selected_boundary)
        )
        disagreement = ndimage.binary_dilation(disagreement, iterations=2)
        crops = choose_crops(disagreement, args.crop_size, args.max_crops)
        columns = [
            (
                f"BF baseline: {baseline_tags['Brightfield']}",
                overlays[("Brightfield", baseline_tags["Brightfield"])],
            ),
            (
                f"BF selected: {selected_tags['Brightfield']}",
                overlays[("Brightfield", selected_tags["Brightfield"])],
            ),
            (
                f"Combined baseline: {baseline_tags['Combined']}",
                overlays[("Combined", baseline_tags["Combined"])],
            ),
            (
                f"Combined selected: {selected_tags['Combined']}",
                overlays[("Combined", selected_tags["Combined"])],
            ),
        ]
        sheet_path = out_dir / "per_field" / f"{key}_high_density_qc.png"
        render_sheet(key, columns, crops, sheet_path)
        sheet_paths.append(sheet_path)
        qc_rows.append(
            {
                "key": key,
                "split": "calibration" if key in set(manifest["calibration_keys"]) else "validation",
                "bf_baseline": baseline_tags["Brightfield"],
                "bf_selected": selected_tags["Brightfield"],
                "combined_baseline": baseline_tags["Combined"],
                "combined_selected": selected_tags["Combined"],
                "n_crops": len(crops),
                "disagreement_fraction": float(np.mean(disagreement)),
                "sheet_path": str(sheet_path),
            }
        )
        print(f"qc_key={key} sheet={sheet_path}", flush=True)

    out_manifest = out_dir / "qc_manifest.csv"
    with out_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    render_contact_sheet(sheet_paths, out_dir / "high_density_qc_contact_sheet.png")
    print(f"qc_manifest={out_manifest}", flush=True)
    print(f"contact_sheet={out_dir / 'high_density_qc_contact_sheet.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
