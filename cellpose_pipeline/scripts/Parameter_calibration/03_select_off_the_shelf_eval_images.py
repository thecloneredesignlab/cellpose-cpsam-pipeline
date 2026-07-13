#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_inventory(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("elapsed_hours") not in ("", None)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_file:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def select_rows(rows: list[dict], n_images: int, time_bins: int) -> list[tuple[dict, int]]:
    elapsed = [float(row["elapsed_hours"]) for row in rows]
    min_t, max_t = min(elapsed), max(elapsed)
    bin_width = max((max_t - min_t) / time_bins, 1)
    bins: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        idx = min(time_bins - 1, int(math.floor((float(row["elapsed_hours"]) - min_t) / bin_width)))
        bins[idx].append(row)

    per_bin = max(1, math.ceil(n_images / time_bins))
    selected: list[tuple[dict, int]] = []
    selected_paths = set()
    for idx in range(time_bins):
        candidates = sorted(
            bins.get(idx, []),
            key=lambda row: (row["well"], row["site"], float(row["elapsed_hours"]), row["filename"]),
        )
        if not candidates:
            continue
        total_slots = max(time_bins * per_bin, 1)
        for i in range(per_bin):
            slot = idx * per_bin + i
            pos = round((slot + 0.5) * (len(candidates) - 1) / total_slots)
            row = candidates[min(len(candidates) - 1, pos)]
            if row["path"] in selected_paths:
                continue
            selected.append((row, idx))
            selected_paths.add(row["path"])
            if len(selected) >= n_images:
                return selected
    return selected[:n_images]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative images for off-the-shelf CellposeSAM evaluation.")
    parser.add_argument("--inventory", type=Path, default=Path("cellpose_pipeline/manifests/image_inventory.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("cellpose_pipeline/eval_off_the_shelf/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv"))
    parser.add_argument("--n-images", type=int, default=192)
    parser.add_argument("--time-bins", type=int, default=12)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    inventory = args.inventory if args.inventory.is_absolute() else PROJECT_ROOT / args.inventory
    out_dir = ensure_dir(args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir)
    rows = load_inventory(inventory)
    selected = select_rows(rows, args.n_images, args.time_bins)
    if not selected:
        raise SystemExit(f"No rows selected from {inventory}")

    manifest_rows = []
    for i, (row, time_bin) in enumerate(selected, start=1):
        src = PROJECT_ROOT / row["path"]
        dst = out_dir / f"{i:03d}_{Path(row['path']).name}"
        link_or_copy(src, dst, args.copy)
        manifest_rows.append(
            {
                "selected_path": str(dst.relative_to(PROJECT_ROOT)),
                "source_path": row["path"],
                "well": row["well"],
                "site": row["site"],
                "elapsed_hours": row["elapsed_hours"],
                "time_bin": time_bin,
            }
        )

    write_csv(args.manifest, manifest_rows, ["selected_path", "source_path", "well", "site", "elapsed_hours", "time_bin"])
    print(f"Selected {len(manifest_rows)} images into {out_dir}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
