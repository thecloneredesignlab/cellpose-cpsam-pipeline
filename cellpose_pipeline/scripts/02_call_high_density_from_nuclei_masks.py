#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


KEY_RE = re.compile(r"(?:SUM159_AC_)?(?:Exp1_)?(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call high-density images from Nuclei segmentation masks.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--nuclei-mask-dir", type=Path, help="Directory containing Nuclei *_cp_masks.tif files.")
    inputs.add_argument("--run-dir", type=Path, help="Workflow run directory containing Nuclei masks.")
    parser.add_argument(
        "--nuclei-core-mask-dir",
        type=Path,
        help="Optional directory containing *_core_masks.tif. These masks take priority over extent masks.",
    )
    parser.add_argument(
        "--prefer-core-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer conservative nucleus-core masks when they are present under --run-dir.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--nuclei-count-threshold", type=int, default=4000)
    parser.add_argument("--nuclei-mask-fraction-threshold", type=float, default=0.22)
    parser.add_argument("--nuclei-median-nn-threshold", type=float, default=16.0)
    parser.add_argument(
        "--enable-mask-fraction-trigger",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow mask area fraction alone to call high density. Disabled by default because fluorescence "
            "exposure inflates nuclear area; count and centroid spacing remain active."
        ),
    )
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if not match:
        return path.stem.replace("_cp_masks", "")
    return match.group(1)


def is_nuclei_mask(path: Path, mask_dir: Path) -> bool:
    stem = path.stem.replace("_cp_masks", "").replace("_core_masks", "")
    try:
        parts = path.relative_to(mask_dir).parts
    except ValueError:
        parts = path.parts
    if "Nuclei" in path.parts or "Nuclei" in parts:
        return True
    return "Exp1_" in stem and "_BF_" not in stem and "_Dead_" not in stem


def first_mask_dir(candidates: list[Path], pattern: str) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and any(is_nuclei_mask(path, candidate) for path in candidate.glob(pattern)):
            return candidate
    return None


def resolve_mask_source(args: argparse.Namespace) -> tuple[Path, str, str]:
    if args.nuclei_core_mask_dir is not None:
        core_dir = first_mask_dir([args.nuclei_core_mask_dir], "*_core_masks.tif")
        if core_dir is None:
            raise FileNotFoundError(f"No Nuclei *_core_masks.tif files under {args.nuclei_core_mask_dir}")
        return core_dir, "core", "*_core_masks.tif"

    if args.prefer_core_masks:
        core_candidates: list[Path] = []
        if args.run_dir is not None:
            core_candidates.extend(
                [
                    args.run_dir / "nucleus_core_seeds",
                    args.run_dir / "nucleus_core_seeds" / "Nuclei",
                    args.run_dir / "Nuclei" / "nucleus_core_seeds",
                    args.run_dir / "Nuclei" / "nucleus_core_seeds" / "Nuclei",
                    args.run_dir / "segmentations" / "NucleiCore",
                ]
            )
        elif args.nuclei_mask_dir is not None:
            core_candidates.extend(
                [
                    args.nuclei_mask_dir.parent / "nucleus_core_seeds",
                    args.nuclei_mask_dir.parent.parent / "nucleus_core_seeds",
                    args.nuclei_mask_dir.parent.parent / "nucleus_core_seeds" / "Nuclei",
                ]
            )
        core_dir = first_mask_dir(core_candidates, "*_core_masks.tif")
        if core_dir is not None:
            return core_dir, "core", "*_core_masks.tif"

    if args.nuclei_mask_dir is not None:
        extent_dir = first_mask_dir([args.nuclei_mask_dir], "*_cp_masks.tif")
        if extent_dir is None:
            raise FileNotFoundError(f"No Nuclei *_cp_masks.tif files under {args.nuclei_mask_dir}")
        return extent_dir, "extent", "*_cp_masks.tif"

    extent_candidates = [
        args.run_dir / "segmentations" / "Nuclei",
        args.run_dir / "segmentations",
        args.run_dir / "Nuclei" / "segmentations",
        args.run_dir / "Nuclei" / "segmentations" / "Nuclei",
        args.run_dir.parent / "Nuclei" / args.run_dir.name / "segmentations",
        args.run_dir.parent / "Nuclei" / args.run_dir.name / "segmentations" / "Nuclei",
    ]
    extent_dir = first_mask_dir(extent_candidates, "*_cp_masks.tif")
    if extent_dir is not None:
        return extent_dir, "extent", "*_cp_masks.tif"
    raise FileNotFoundError(
        "Could not find Nuclei masks under run dir. Tried: "
        + ", ".join(str(candidate) for candidate in extent_candidates)
    )


def object_centroids(labels: np.ndarray) -> np.ndarray:
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    centroids: list[tuple[float, float]] = []
    for label_id, obj_slice in enumerate(ndi.find_objects(labels), start=1):
        if obj_slice is None:
            continue
        obj = labels[obj_slice] == label_id
        if not np.any(obj):
            continue
        yy, xx = np.nonzero(obj)
        y0 = obj_slice[0].start or 0
        x0 = obj_slice[1].start or 0
        centroids.append((float(yy.mean() + y0), float(xx.mean() + x0)))
    if not centroids:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(centroids, dtype=np.float32)


def nuclei_metrics(mask: np.ndarray) -> dict[str, Any]:
    labels = mask.astype(np.int64, copy=False)
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    count = int(areas.size)
    centers = object_centroids(labels)
    if len(centers) >= 2:
        dist, _ = cKDTree(centers).query(centers, k=2)
        median_nn = float(np.median(dist[:, 1]))
    else:
        median_nn = float("nan")
    return {
        "nuclei_count": count,
        "nuclei_mask_fraction": float((labels > 0).sum() / labels.size) if labels.size else 0.0,
        "nuclei_area_median": float(np.median(areas)) if areas.size else 0.0,
        "nuclei_median_nn_px": median_nn,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    mask_dir, mask_kind, mask_pattern = resolve_mask_source(args)
    masks = sorted(path for path in mask_dir.glob(mask_pattern) if is_nuclei_mask(path, mask_dir))
    if not masks:
        raise SystemExit(f"No {mask_pattern} files found under {mask_dir}")

    rows: list[dict[str, Any]] = []
    for path in masks:
        metrics = nuclei_metrics(tifffile.imread(path))
        median_nn = float(metrics["nuclei_median_nn_px"])
        high_density_by_count = int(metrics["nuclei_count"]) >= args.nuclei_count_threshold
        high_density_by_mask_fraction = (
            float(metrics["nuclei_mask_fraction"]) >= args.nuclei_mask_fraction_threshold
        )
        high_density_by_nn = not np.isnan(median_nn) and median_nn <= args.nuclei_median_nn_threshold
        high_density = bool(
            high_density_by_count
            or high_density_by_nn
            or (args.enable_mask_fraction_trigger and high_density_by_mask_fraction)
        )
        rows.append(
            {
                "key": extract_key(path),
                "mask_path": str(path),
                "density_mask_kind": mask_kind,
                **metrics,
                "mask_fraction_trigger_enabled": args.enable_mask_fraction_trigger,
                "high_density_by_count": high_density_by_count,
                "high_density_by_nn": high_density_by_nn,
                "high_density_by_mask_fraction": high_density_by_mask_fraction,
                "high_density": high_density,
            }
        )
    write_rows(args.out_csv, rows)
    print(f"density_mask_dir={mask_dir}")
    print(f"density_mask_kind={mask_kind}")
    print(f"mask_fraction_trigger_enabled={args.enable_mask_fraction_trigger}")
    print(f"n_masks={len(rows)}")
    print(f"n_high_density={sum(1 for row in rows if row['high_density'])}")
    print(f"density_calls={args.out_csv}")


if __name__ == "__main__":
    main()
