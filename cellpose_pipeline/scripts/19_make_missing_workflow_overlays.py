#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_COLORS = {
    "live": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "dead": np.array([0.0, 0.25, 1.0], dtype=np.float32),
    "transitional": np.array([0.65, 0.0, 0.85], dtype=np.float32),
    "artifact": np.array([1.0, 0.9, 0.0], dtype=np.float32),
    "uncertain": np.array([0.55, 0.55, 0.55], dtype=np.float32),
}


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    arr = np.clip(image * 255, 0, 255).astype(np.uint8)
    tmp_path = path.with_name(f".{path.name}.tmp")
    Image.fromarray(arr).save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def stable_color(mask_id: int) -> np.ndarray:
    # Deterministic bright color without depending on global random state.
    rng = np.random.default_rng(mask_id * 1103515245 + 12345)
    color = rng.uniform(0.15, 1.0, size=3).astype(np.float32)
    return color / max(float(color.max()), 1e-6)


def load_manifest_map(manifest: Path) -> dict[str, Path]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    mapping: dict[str, Path] = {}
    for row in rows:
        for key in ("source_path", "selected_path"):
            value = row.get(key)
            if not value:
                continue
            path = Path(value)
            abs_path = path if path.is_absolute() else PROJECT_ROOT / path
            mapping[abs_path.stem] = abs_path
    return mapping


def mask_stem(mask_path: Path) -> str:
    name = mask_path.name
    suffix = "_cp_masks.tif"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return mask_path.stem


def prediction_path_for(run_dir: Path, stem: str) -> Path:
    return run_dir / "classification" / "predictions" / f"{stem}_per_cell_predictions.csv"


def segmentation_overlay_path_for(run_dir: Path, stem: str) -> Path:
    return run_dir / "qc" / "segmentation_overlays" / f"{stem}_segmentation_overlay.png"


def classification_overlay_path_for(run_dir: Path, stem: str) -> Path:
    return run_dir / "classification" / "qc" / "label_overlays" / f"{stem}_state_overlay.png"


def make_segmentation_overlay(raw_rgb: np.ndarray, masks: np.ndarray, out_path: Path, alpha: float) -> None:
    overlay = normalize_rgb(raw_rgb)
    for mask_id in np.unique(masks):
        mask_id = int(mask_id)
        if mask_id == 0:
            continue
        mask = masks == mask_id
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * stable_color(mask_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_png(out_path, overlay)


def make_classification_overlay(
    raw_rgb: np.ndarray,
    masks: np.ndarray,
    predictions: pd.DataFrame,
    out_path: Path,
    alpha: float,
) -> None:
    base = normalize_rgb(raw_rgb)
    overlay = base.copy()
    for row in predictions[["mask_id", "state"]].itertuples(index=False):
        mask_id = int(row.mask_id)
        color = STATE_COLORS.get(str(row.state), STATE_COLORS["uncertain"])
        mask = masks == mask_id
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    side_by_side = np.concatenate([base, overlay], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_png(out_path, side_by_side)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing segmentation and classification overlays for a workflow run.")
    parser.add_argument("--run-dir", type=Path, default=Path("cellpose_pipeline/workflow_runs/overnight_cpsam_192"))
    parser.add_argument("--manifest", type=Path, default=Path("cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv"))
    parser.add_argument("--force", action="store_true", help="Recompute overlays even if PNGs already exist.")
    parser.add_argument(
        "--overlay-type",
        choices=("both", "segmentation", "classification"),
        default="both",
        help="Which overlay type to backfill.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of mask files to inspect.")
    parser.add_argument("--stem", action="append", help="Only process this image stem. Can be passed repeatedly.")
    parser.add_argument("--segmentation-alpha", type=float, default=0.55)
    parser.add_argument("--classification-alpha", type=float, default=0.55)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    manifest = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    raw_by_stem = load_manifest_map(manifest)

    mask_paths = sorted((run_dir / "segmentations").glob("*_cp_masks.tif"))
    if args.stem:
        wanted = set(args.stem)
        mask_paths = [path for path in mask_paths if mask_stem(path) in wanted]
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    made_seg = 0
    made_cls = 0
    skipped_seg = 0
    skipped_cls = 0
    missing_raw = 0
    missing_predictions = 0
    read_errors = 0

    for mask_path in mask_paths:
        stem = mask_stem(mask_path)
        raw_path = raw_by_stem.get(stem)
        if raw_path is None or not raw_path.exists():
            missing_raw += 1
            print(f"missing raw image for {stem}", flush=True)
            continue

        seg_out = segmentation_overlay_path_for(run_dir, stem)
        cls_out = classification_overlay_path_for(run_dir, stem)
        pred_path = prediction_path_for(run_dir, stem)
        want_seg = args.overlay_type in ("both", "segmentation")
        want_cls = args.overlay_type in ("both", "classification")
        need_seg = want_seg and (args.force or not seg_out.exists())
        need_cls = want_cls and (args.force or (pred_path.exists() and not cls_out.exists()))

        if want_seg and not need_seg:
            skipped_seg += 1
        if want_cls and pred_path.exists() and not need_cls:
            skipped_cls += 1
        if want_cls and not pred_path.exists():
            missing_predictions += 1

        if not need_seg and not need_cls:
            continue

        try:
            raw_rgb = to_rgb(tifffile.imread(raw_path))
            masks = tifffile.imread(mask_path)
            if masks.ndim > 2:
                masks = np.squeeze(masks)
        except Exception as exc:
            read_errors += 1
            print(f"read error for {stem}: {exc!r}", flush=True)
            continue

        if need_seg:
            make_segmentation_overlay(raw_rgb, masks, seg_out, args.segmentation_alpha)
            made_seg += 1
            print(f"segmentation overlay: {seg_out}", flush=True)

        if need_cls:
            try:
                predictions = pd.read_csv(pred_path)
                make_classification_overlay(raw_rgb, masks, predictions, cls_out, args.classification_alpha)
                made_cls += 1
                print(f"classification overlay: {cls_out}", flush=True)
            except Exception as exc:
                read_errors += 1
                print(f"classification overlay error for {stem}: {exc!r}", flush=True)

    print(
        "summary "
        f"mask_files={len(mask_paths)} "
        f"made_segmentation={made_seg} skipped_segmentation={skipped_seg} "
        f"made_classification={made_cls} skipped_classification={skipped_cls} "
        f"missing_raw={missing_raw} missing_predictions={missing_predictions} read_errors={read_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
