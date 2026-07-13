#!/usr/bin/env python3
"""Validate rendered shape_strict QC images and build a compact contact sheet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize production shape_strict QC artifacts.")
    parser.add_argument("--shape-root", type=Path, required=True)
    parser.add_argument("--candidate-tag", default="shape_strict")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--thumb-height", type=int, default=720)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    shape_root = args.shape_root.resolve()
    qc_root = shape_root / "qc"
    selection_path = qc_root / "qc_selection.csv"
    rows: list[dict[str, str]] = []
    if selection_path.is_file() and selection_path.stat().st_size:
        with selection_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    manifest: list[dict[str, Any]] = []
    thumbnails: list[tuple[str, Image.Image]] = []
    for row in rows:
        path = Path(row["expected_qc_png"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Expected shape_strict QC image is missing: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            original_size = image.size
            image.thumbnail((args.thumb_width, args.thumb_height), Image.Resampling.LANCZOS)
            thumbnails.append((row["key"], image.copy()))
        manifest.append(
            {
                **row,
                "qc_png": str(path),
                "file_size_bytes": path.stat().st_size,
                "width_px": original_size[0],
                "height_px": original_size[1],
                "render_complete": True,
            }
        )

    manifest_fields = [
        "qc_rank", "key", "n_split_events", "max_shape_gain",
        "max_valley_drop_fraction", "min_stability_fraction", "expected_qc_png",
        "qc_png", "file_size_bytes", "width_px", "height_px", "render_complete",
    ]
    write_rows(qc_root / "qc_manifest.csv", manifest, manifest_fields)

    contact_sheet = qc_root / "shape_strict_qc_contact_sheet.png"
    if thumbnails:
        columns = max(1, int(args.columns))
        rows_count = int(math.ceil(len(thumbnails) / columns))
        tile_width = int(args.thumb_width)
        tile_height = int(args.thumb_height) + 24
        canvas = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for index, (key, image) in enumerate(thumbnails):
            row = index // columns
            column = index % columns
            x0 = column * tile_width
            y0 = row * tile_height
            draw.text((x0 + 6, y0 + 5), key, fill=(25, 30, 35), font=font)
            canvas.paste(image, (x0, y0 + 24))
        canvas.save(contact_sheet, optimize=True)
    elif contact_sheet.exists():
        contact_sheet.unlink()

    summary_path = shape_root / "shape_strict_summary.json"
    summary = json.loads(summary_path.read_text())
    expected = int(summary.get("n_qc_fields", 0))
    if expected != len(manifest):
        raise RuntimeError(f"QC count mismatch: expected={expected} rendered={len(manifest)}")
    summary["qc_manifest"] = str(qc_root / "qc_manifest.csv")
    summary["qc_contact_sheet"] = str(contact_sheet) if thumbnails else None
    summary["validation"]["qc_render_complete"] = True
    write_json(summary_path, summary)
    (shape_root / "_SUCCESS").write_text("shape_strict production stage complete\n")
    print(f"qc_manifest={qc_root / 'qc_manifest.csv'}", flush=True)
    print(f"qc_contact_sheet={contact_sheet if thumbnails else 'none'}", flush=True)
    print(f"qc_images={len(manifest)}", flush=True)
    print("shape_strict_complete=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
