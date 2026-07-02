#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import DEFAULT_RAW_DIR, IMAGE_SUFFIXES, iter_files, parse_image_name, relative_to_project, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory Incucyte TIFF/movie files for Cellpose training.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing time-course TIFFs.")
    parser.add_argument("--out", type=Path, default=Path("cellpose_pipeline/manifests/image_inventory.csv"))
    parser.add_argument("--include-size", action="store_true", help="Call stat() for each TIFF and include size_bytes.")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.is_absolute():
        raw_dir = Path.cwd() / raw_dir

    rows = []
    wells = Counter()
    sites = Counter()
    timepoints = Counter()
    for path in iter_files(raw_dir, IMAGE_SUFFIXES):
        parsed = parse_image_name(path)
        if parsed.well:
            wells[parsed.well] += 1
        if parsed.site:
            sites[parsed.site] += 1
        if parsed.elapsed_hours is not None:
            timepoints[parsed.elapsed_hours] += 1
        size_bytes = path.stat().st_size if args.include_size else ""
        rows.append(
            {
                "path": relative_to_project(path),
                "filename": path.name,
                "well": parsed.well or "",
                "site": parsed.site or "",
                "day": parsed.day if parsed.day is not None else "",
                "hour": parsed.hour if parsed.hour is not None else "",
                "minute": parsed.minute if parsed.minute is not None else "",
                "elapsed_hours": f"{parsed.elapsed_hours:.3f}" if parsed.elapsed_hours is not None else "",
                "size_bytes": size_bytes,
            }
        )

    rows.sort(key=lambda row: (row["well"], row["site"], float(row["elapsed_hours"] or -1), row["filename"]))
    write_csv(args.out, rows, ["path", "filename", "well", "site", "day", "hour", "minute", "elapsed_hours", "size_bytes"])

    print(f"Wrote {len(rows)} TIFF records to {args.out}")
    print(f"Found {len(wells)} wells, {len(sites)} sites, and {len(timepoints)} parsed timepoints.")


if __name__ == "__main__":
    main()
