#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from common import DEFAULT_RAW_DIR, IMAGE_SUFFIXES, ensure_dir, iter_files, link_or_copy, parse_image_name, relative_to_project, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative TIFFs for manual Cellpose annotation.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("cellpose_pipeline/annotation_images/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("cellpose_pipeline/manifests/annotation_selection.csv"))
    parser.add_argument("--n-images", type=int, default=96)
    parser.add_argument("--time-bins", type=int, default=12)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking them.")
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else Path.cwd() / args.raw_dir
    out_dir = ensure_dir(args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir)

    records = []
    for path in iter_files(raw_dir, IMAGE_SUFFIXES):
        parsed = parse_image_name(path)
        if parsed.elapsed_hours is None:
            continue
        records.append((path, parsed))

    if not records:
        raise SystemExit(f"No parseable TIFF timepoints found under {raw_dir}")

    min_t = min(parsed.elapsed_hours for _, parsed in records if parsed.elapsed_hours is not None)
    max_t = max(parsed.elapsed_hours for _, parsed in records if parsed.elapsed_hours is not None)
    bin_width = max((max_t - min_t) / args.time_bins, 1)

    bins: dict[int, list[tuple[Path, object]]] = defaultdict(list)
    for path, parsed in records:
        idx = min(args.time_bins - 1, int(math.floor((parsed.elapsed_hours - min_t) / bin_width)))
        bins[idx].append((path, parsed))

    per_bin = max(1, math.ceil(args.n_images / args.time_bins))
    selected = []
    selected_paths = set()
    for idx in range(args.time_bins):
        candidates = sorted(
            bins.get(idx, []),
            key=lambda item: (item[1].well or "", item[1].site or "", item[1].elapsed_hours, item[0].name),
        )
        if not candidates:
            continue
        if len(candidates) <= per_bin:
            picks = candidates
        else:
            total_slots = max(args.time_bins * per_bin, 1)
            picks = []
            for i in range(per_bin):
                slot = idx * per_bin + i
                pos = round((slot + 0.5) * (len(candidates) - 1) / total_slots)
                picks.append(candidates[min(len(candidates) - 1, pos)])
        for path, parsed in picks:
            if path in selected_paths:
                continue
            selected.append((path, parsed, idx))
            selected_paths.add(path)
        if len(selected) >= args.n_images:
            break

    if len(selected) < args.n_images:
        for path, parsed in sorted(records, key=lambda item: (item[1].well or "", item[1].site or "", item[1].elapsed_hours)):
            if path not in selected_paths:
                selected.append((path, parsed, -1))
                selected_paths.add(path)
            if len(selected) >= args.n_images:
                break

    rows = []
    for i, (src, parsed, time_bin) in enumerate(selected[: args.n_images], start=1):
        dst = out_dir / f"{i:03d}_{src.name}"
        link_or_copy(src, dst, args.copy)
        rows.append(
            {
                "selected_path": relative_to_project(dst),
                "source_path": relative_to_project(src),
                "well": parsed.well or "",
                "site": parsed.site or "",
                "elapsed_hours": f"{parsed.elapsed_hours:.3f}" if parsed.elapsed_hours is not None else "",
                "time_bin": time_bin,
            }
        )

    write_csv(args.manifest, rows, ["selected_path", "source_path", "well", "site", "elapsed_hours", "time_bin"])
    print(f"Selected {len(rows)} images into {out_dir}")
    print(f"Annotation manifest: {args.manifest}")


if __name__ == "__main__":
    main()
