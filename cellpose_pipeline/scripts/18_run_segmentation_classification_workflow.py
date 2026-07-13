#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nuclei_segmentation_utils import NucleusCoreConfig, build_nucleus_core_seeds


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
REQUIRED_CELLPOSE_VERSION = "4.2.1.1"
IGNORED_PROFILE_FOLDERS = {"Dead_Uncalibrated"}
IMAGE_KEY_RE = re.compile(r"(?:SUM159_AC_)?(?:Exp1_)?(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")


@dataclass(frozen=True)
class SegmentationProfile:
    folder: str
    model: str
    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    low_percentile: float
    high_percentile: float
    invert: bool = False
    channel_axis: int | None = None
    normalize: bool = False
    normalization: str = "percentile"
    fixed_low_value: float = 0.0
    fixed_high_value: float = 1.0
    input_transform: str = "identity"
    preprocess: str = "percentile"
    gamma: float = 1.0
    clahe_clip: float = 2.0
    clahe_tile: int = 16
    blur_sigma: float = 0.0
    background_sigma: float = 0.0
    tophat_kernel: int = 0
    postprocess: str = "none"
    filter_min_area: int = 20
    filter_max_area: int = 2500
    filter_max_aspect: float = 4.0
    filter_min_mean_delta: float = 0.2
    filter_min_p90_delta: float = 0.7
    filter_min_p90_abs: float = 10.0
    filter_min_snr: float = 1.25
    filter_bg_margin: int = 14


FOLDER_PROFILES: dict[str, SegmentationProfile] = {
    "Brightfield": SegmentationProfile(
        folder="Brightfield",
        model="cpsam",
        diameter=25,
        flow_threshold=0.0,
        cellprob_threshold=-1.75,
        min_size=20,
        low_percentile=1.0,
        high_percentile=99.0,
    ),
    "Combined": SegmentationProfile(
        folder="Combined",
        model="cpsam",
        diameter=25,
        flow_threshold=0.0,
        cellprob_threshold=-1.75,
        min_size=20,
        low_percentile=1.0,
        high_percentile=99.0,
        preprocess="rgb_luma",
    ),
    "Dead": SegmentationProfile(
        folder="Dead",
        model="cpsam_v2",
        diameter=22,
        flow_threshold=0.0,
        cellprob_threshold=-3.0,
        min_size=12,
        low_percentile=0.0,
        high_percentile=100.0,
        normalization="fixed",
        fixed_low_value=7.6,
        fixed_high_value=55.0,
        postprocess="dead_raw_signal",
        filter_min_mean_delta=0.60,
        filter_min_p90_delta=2.00,
        filter_min_p90_abs=15.0,
        filter_min_snr=3.00,
    ),
    "Nuclei": SegmentationProfile(
        folder="Nuclei",
        model="cpsam_v2",
        # Full 20-field calibration retained this high-recall extent profile.
        # Exposure-sensitive size correction is provided by nucleus_core_seeds,
        # rather than by raising cellprob_threshold and dropping weak nuclei.
        diameter=24,
        flow_threshold=0.0,
        cellprob_threshold=-2.75,
        min_size=5,
        low_percentile=0.5,
        high_percentile=99.9,
    ),
}


HIGH_DENSITY_FOLDER_PROFILES: dict[str, SegmentationProfile] = {
    "Brightfield": SegmentationProfile(
        folder="Brightfield",
        model="cpsam",
        # Two-round GPU calibration on nucleus-core-defined high-density fields.
        # Local contrast enhancement improves separation in stacked monolayers.
        diameter=22,
        flow_threshold=0.0,
        cellprob_threshold=-2.25,
        min_size=10,
        low_percentile=1.0,
        high_percentile=99.0,
        preprocess="clahe",
        clahe_clip=1.4,
        clahe_tile=16,
    ),
    "Combined": SegmentationProfile(
        folder="Combined",
        model="cpsam",
        # RGB luma conversion is followed by broad-background subtraction;
        # this preserves local cell boundaries in fluorescently crowded fields.
        diameter=22,
        flow_threshold=0.0,
        cellprob_threshold=-2.25,
        min_size=10,
        low_percentile=1.0,
        high_percentile=99.0,
        input_transform="rgb_luma",
        preprocess="background",
        background_sigma=12.0,
    ),
}


def high_density_calls_path(run_dir: Path, name: str) -> Path:
    return run_dir / "qc" / name


def cellpose_version() -> str:
    try:
        return metadata.version("cellpose")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("cellpose is not installed in this Python environment") from exc


def require_cellpose_version() -> str:
    version = cellpose_version()
    if version != REQUIRED_CELLPOSE_VERSION:
        raise RuntimeError(
            f"Unsupported cellpose version {version!r}; this pipeline requires "
            f"cellpose=={REQUIRED_CELLPOSE_VERSION}."
        )
    return version


def assert_cellpose_api_compatible() -> None:
    from cellpose import models

    init_params = set(inspect.signature(models.CellposeModel).parameters)
    eval_params = set(inspect.signature(models.CellposeModel.eval).parameters)
    required_init = {"gpu", "pretrained_model"}
    required_eval = {
        "channel_axis",
        "normalize",
        "diameter",
        "flow_threshold",
        "cellprob_threshold",
        "min_size",
    }
    missing_init = sorted(required_init - init_params)
    missing_eval = sorted(required_eval - eval_params)
    if missing_init or missing_eval:
        raise RuntimeError(
            "Installed Cellpose API is not compatible with this workflow. "
            f"Missing CellposeModel params={missing_init}; missing eval params={missing_eval}."
        )


def import_cellpose_model_class() -> Any:
    require_cellpose_version()
    assert_cellpose_api_compatible()
    from cellpose import models

    return models.CellposeModel


def iter_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def manual_profile(args: argparse.Namespace) -> SegmentationProfile:
    return SegmentationProfile(
        folder="manual",
        model=args.pretrained_model,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        invert=args.invert,
        channel_axis=args.channel_axis,
        normalize=args.cellpose_normalize,
        normalization=args.normalization,
        fixed_low_value=args.fixed_low_value,
        fixed_high_value=args.fixed_high_value,
        input_transform=args.input_transform,
        preprocess=args.preprocess,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        blur_sigma=args.blur_sigma,
        background_sigma=args.background_sigma,
        tophat_kernel=args.tophat_kernel,
        postprocess=args.postprocess,
        filter_min_area=args.filter_min_area,
        filter_max_area=args.filter_max_area,
        filter_max_aspect=args.filter_max_aspect,
        filter_min_mean_delta=args.filter_min_mean_delta,
        filter_min_p90_delta=args.filter_min_p90_delta,
        filter_min_p90_abs=args.filter_min_p90_abs,
        filter_min_snr=args.filter_min_snr,
        filter_bg_margin=args.filter_bg_margin,
    )


def extract_image_key(image_path: Path) -> str:
    match = IMAGE_KEY_RE.search(image_path.name)
    if not match:
        return image_path.stem
    return match.group(1)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_high_density_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"High-density calls file not found: {path}")
    keys: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "key" not in reader.fieldnames:
            raise ValueError(f"High-density calls CSV must contain a 'key' column: {path}")
        high_density_col = "high_density" if "high_density" in reader.fieldnames else None
        for row in reader:
            key = row.get("key", "").strip()
            if not key:
                continue
            if high_density_col is None or truthy(row.get(high_density_col, "")):
                keys.add(key)
    return keys


def load_dead_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Dead calibration JSON not found: {path}")
    payload = json.loads(path.read_text())
    block = payload.get("dead", payload)
    required = {"recommended_fixed_low", "recommended_fixed_high"}
    missing = required - set(block)
    if missing:
        raise ValueError(f"Dead calibration is missing {sorted(missing)}: {path}")
    return block


def load_dead_calibration_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Dead calibration map not found: {path}")
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "key" not in reader.fieldnames or "dead_q1p0" not in reader.fieldnames:
            raise ValueError(f"Dead calibration map requires key and dead_q1p0 columns: {path}")
        for row in reader:
            key = row.get("key", "").strip()
            if not key:
                continue
            if key in rows:
                raise ValueError(f"Duplicate key in Dead calibration map: {key}")
            rows[key] = row
    return rows


def calibrated_dead_profile(
    profile: SegmentationProfile,
    image_path: Path,
    args: argparse.Namespace,
) -> SegmentationProfile:
    calibration = getattr(args, "dead_calibration", None)
    if profile.folder != "Dead" or calibration is None or args.dead_calibration_mode == "off":
        return profile
    global_low = float(calibration["recommended_fixed_low"])
    global_high = float(calibration["recommended_fixed_high"])
    low, high = global_low, global_high
    if args.dead_calibration_mode == "bounded-background":
        row = getattr(args, "dead_calibration_by_key", {}).get(extract_image_key(image_path))
        bounds = calibration.get("bounded_background_low_range", [global_low, global_low])
        if row is not None and len(bounds) == 2:
            low = float(np.clip(float(row["dead_q1p0"]), float(bounds[0]), float(bounds[1])))
            span = float(calibration.get("recommended_background_span", global_high - global_low))
            high = low + max(span, 1e-6)
    return replace(profile, fixed_low_value=low, fixed_high_value=high)


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


def nuclei_density_metrics(mask: np.ndarray) -> dict[str, float | int]:
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


def call_high_density(metrics: dict[str, float | int], args: argparse.Namespace) -> bool:
    median_nn = float(metrics["nuclei_median_nn_px"])
    return (
        int(metrics["nuclei_count"]) >= args.nuclei_count_threshold
        or (not np.isnan(median_nn) and median_nn <= args.nuclei_median_nn_threshold)
        or (
            args.enable_nuclei_mask_fraction_trigger
            and float(metrics["nuclei_mask_fraction"]) >= args.nuclei_mask_fraction_threshold
        )
    )


def write_density_calls_from_nuclei_masks(
    mask_dir: Path,
    core_mask_dir: Path | None,
    out_csv: Path,
    args: argparse.Namespace,
) -> set[str]:
    core_masks = sorted(core_mask_dir.glob("*_core_masks.tif")) if core_mask_dir is not None else []
    if core_masks:
        masks = core_masks
        density_mask_kind = "core"
        density_mask_dir = core_mask_dir
    else:
        masks = sorted(mask_dir.glob("*_cp_masks.tif"))
        density_mask_kind = "extent"
        density_mask_dir = mask_dir
    if not masks:
        print(f"density_calls_skipped=no_nuclei_masks mask_dir={density_mask_dir}", flush=True)
        return set()
    rows: list[dict[str, Any]] = []
    high_density_keys: set[str] = set()
    for mask_path in masks:
        key = extract_image_key(mask_path)
        metrics = nuclei_density_metrics(tifffile.imread(mask_path))
        median_nn = float(metrics["nuclei_median_nn_px"])
        high_density_by_count = int(metrics["nuclei_count"]) >= args.nuclei_count_threshold
        high_density_by_nn = not np.isnan(median_nn) and median_nn <= args.nuclei_median_nn_threshold
        high_density_by_mask_fraction = (
            float(metrics["nuclei_mask_fraction"]) >= args.nuclei_mask_fraction_threshold
        )
        high_density = call_high_density(metrics, args)
        if high_density:
            high_density_keys.add(key)
        rows.append(
            {
                "key": key,
                "mask_path": str(mask_path),
                "density_mask_kind": density_mask_kind,
                **metrics,
                "mask_fraction_trigger_enabled": args.enable_nuclei_mask_fraction_trigger,
                "high_density_by_count": high_density_by_count,
                "high_density_by_nn": high_density_by_nn,
                "high_density_by_mask_fraction": high_density_by_mask_fraction,
                "high_density": high_density,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"density_calls={out_csv}", flush=True)
    print(f"density_mask_dir={density_mask_dir}", flush=True)
    print(f"density_mask_kind={density_mask_kind}", flush=True)
    print(f"mask_fraction_trigger_enabled={args.enable_nuclei_mask_fraction_trigger}", flush=True)
    print(f"density_calls_n_masks={len(rows)}", flush=True)
    print(f"density_calls_n_high_density={len(high_density_keys)}", flush=True)
    return high_density_keys


def resolve_profile(image_path: Path, args: argparse.Namespace) -> SegmentationProfile | None:
    folder = image_path.parent.name
    if args.profile_mode == "manual":
        return manual_profile(args)
    if folder in IGNORED_PROFILE_FOLDERS:
        return None
    if folder in FOLDER_PROFILES:
        high_density_keys = getattr(args, "high_density_keys", set())
        if (
            getattr(args, "enable_high_density_profiles", False)
            and folder in HIGH_DENSITY_FOLDER_PROFILES
            and extract_image_key(image_path) in high_density_keys
        ):
            return calibrated_dead_profile(HIGH_DENSITY_FOLDER_PROFILES[folder], image_path, args)
        return calibrated_dead_profile(FOLDER_PROFILES[folder], image_path, args)
    if args.skip_unknown_profiles:
        return None
    return manual_profile(args)


def expected_mask_path(
    segmentation_dir: Path,
    image_path: Path,
    profile: SegmentationProfile,
    flat_profile_output: bool = False,
) -> Path:
    if profile.folder in FOLDER_PROFILES and not flat_profile_output:
        return segmentation_dir / profile.folder / f"{image_path.stem}_cp_masks.tif"
    return segmentation_dir / f"{image_path.stem}_cp_masks.tif"


def expected_nucleus_core_path(
    run_dir: Path,
    image_path: Path,
    profile: SegmentationProfile,
    flat_profile_output: bool = False,
) -> Path | None:
    if profile.folder != "Nuclei":
        return None
    core_dir = run_dir / "nucleus_core_seeds"
    if not flat_profile_output:
        core_dir = core_dir / profile.folder
    return core_dir / f"{image_path.stem}_core_masks.tif"


def metadata_path(
    metadata_dir: Path,
    image_path: Path,
    profile: SegmentationProfile,
    flat_profile_output: bool = False,
) -> Path:
    if profile.folder in FOLDER_PROFILES and not flat_profile_output:
        return metadata_dir / profile.folder / f"{image_path.stem}_segmentation_metadata.json"
    return metadata_dir / f"{image_path.stem}_segmentation_metadata.json"


def segmentation_overlay_path(
    run_dir: Path,
    image_id: str,
    profile: SegmentationProfile,
    flat_profile_output: bool = False,
) -> Path:
    out_dir = run_dir / "qc" / "segmentation_overlays"
    if profile.folder in FOLDER_PROFILES and not flat_profile_output:
        out_dir = out_dir / profile.folder
    return out_dir / f"{image_id}_segmentation_overlay.png"


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return np.clip(arr / float(info.max), 0, 1)
    if float(np.nanmin(arr)) >= 0.0 and float(np.nanmax(arr)) <= 1.0:
        return np.clip(arr, 0, 1)
    if float(np.nanmin(arr)) >= 0.0:
        return np.clip(arr / max(float(np.nanmax(arr)), 1.0), 0, 1)
    lo, hi = np.percentile(arr, (1, 99))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def stable_color(mask_id: int) -> np.ndarray:
    rng = np.random.default_rng(mask_id * 1103515245 + 12345)
    color = rng.uniform(0.15, 1.0, size=3).astype(np.float32)
    return color / max(float(color.max()), 1e-6)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    arr = np.clip(image * 255, 0, 255).astype(np.uint8)
    tmp_path = path.with_name(f".{path.name}.tmp")
    Image.fromarray(arr).save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def make_segmentation_overlay(raw_rgb: np.ndarray, masks: np.ndarray, out_path: Path, alpha: float = 0.55) -> None:
    overlay = normalize_rgb(raw_rgb)
    for mask_id in np.unique(masks):
        mask_id = int(mask_id)
        if mask_id == 0:
            continue
        mask = masks == mask_id
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * stable_color(mask_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_png(out_path, overlay)


def write_segmentation_overlay(image_path: Path, mask_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"Using existing segmentation overlay: {out_path}", flush=True)
        return

    raw_rgb = to_rgb(tifffile.imread(image_path))
    masks = tifffile.imread(mask_path)
    if masks.ndim > 2:
        masks = np.squeeze(masks)
    if masks.shape != raw_rgb.shape[:2]:
        raise ValueError(f"Mask shape {masks.shape} does not match image shape {raw_rgb.shape[:2]}")

    make_segmentation_overlay(raw_rgb, masks, out_path)
    print(f"segmentation_overlay={out_path}", flush=True)


def normalize_image(image: np.ndarray, profile: SegmentationProfile) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    if profile.normalization == "fixed":
        low, high = float(profile.fixed_low_value), float(profile.fixed_high_value)
    elif profile.normalization == "percentile":
        low, high = np.percentile(data, [profile.low_percentile, profile.high_percentile])
    else:
        raise ValueError(f"Unsupported normalization mode: {profile.normalization}")
    if high <= low:
        high = low + 1.0
    normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    if profile.invert:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32, copy=False)


def odd_kernel_from_sigma(sigma: float) -> int:
    k = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    return k if k % 2 else k + 1


def apply_clahe(norm: np.ndarray, clip: float, tile: int) -> np.ndarray:
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return np.clip(clahe.apply(arr).astype(np.float32) / 255.0, 0.0, 1.0)


def apply_background(norm: np.ndarray, sigma: float) -> np.ndarray:
    k = odd_kernel_from_sigma(sigma)
    background = cv2.GaussianBlur(norm.astype(np.float32), (k, k), float(sigma))
    corrected = norm - background
    corrected -= float(corrected.min())
    high = float(np.percentile(corrected, 99.8))
    if high <= 0:
        high = float(corrected.max()) or 1.0
    return np.clip(corrected / high, 0.0, 1.0).astype(np.float32, copy=False)


def apply_tophat(norm: np.ndarray, kernel: int) -> np.ndarray:
    k = int(kernel)
    if k <= 0:
        return norm
    if k % 2 == 0:
        k += 1
    arr = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, struct).astype(np.float32)
    high = float(np.percentile(out, 99.8))
    if high <= 0:
        high = float(out.max()) or 1.0
    return np.clip(out / high, 0.0, 1.0).astype(np.float32, copy=False)


def rgb_to_grayscale(raw: np.ndarray, mode: str) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"{mode} preprocessing expects an RGB channel-last image, got {raw.shape}")
    data = raw.astype(np.float32, copy=False)
    if mode == "rgb_luma":
        return 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
    if mode == "rgb_mean":
        return data.mean(axis=2)
    if mode == "rgb_median":
        return np.median(data, axis=2)
    if mode == "rgb_min":
        return data.min(axis=2)
    raise ValueError(f"Unsupported RGB preprocessing mode: {mode}")


def prepare_for_cellpose(raw: np.ndarray, profile: SegmentationProfile) -> np.ndarray:
    if profile.normalize:
        return raw
    raw_for_norm = raw
    preprocess = profile.preprocess
    if profile.input_transform != "identity":
        raw_for_norm = rgb_to_grayscale(raw_for_norm, profile.input_transform)
    if preprocess in {"rgb_luma", "rgb_mean", "rgb_median", "rgb_min"}:
        if profile.input_transform != "identity":
            raise ValueError(
                "Use either input_transform or a legacy RGB preprocess mode, not both: "
                f"input_transform={profile.input_transform} preprocess={profile.preprocess}"
            )
        raw_for_norm = rgb_to_grayscale(raw_for_norm, preprocess)
        preprocess = "percentile"

    norm = normalize_image(raw_for_norm, profile)
    if preprocess == "percentile":
        prepared = norm
    elif preprocess == "clahe":
        prepared = apply_clahe(norm, profile.clahe_clip, profile.clahe_tile)
    elif preprocess == "background":
        prepared = apply_background(norm, profile.background_sigma)
    elif preprocess == "tophat":
        prepared = apply_tophat(norm, profile.tophat_kernel)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {profile.preprocess}")
    if profile.blur_sigma > 0:
        k = odd_kernel_from_sigma(profile.blur_sigma)
        prepared = cv2.GaussianBlur(prepared.astype(np.float32), (k, k), float(profile.blur_sigma))
    if profile.gamma != 1.0:
        prepared = np.power(np.clip(prepared, 0.0, 1.0), float(profile.gamma)).astype(np.float32, copy=False)
    return np.clip(prepared, 0.0, 1.0).astype(np.float32, copy=False)


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    n_objects = int(mask.max()) if mask.size else 0
    areas = np.bincount(mask.ravel())[1:]
    if areas.size == 0:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
        }
    return {
        "n_objects": n_objects,
        "mask_fraction": float((mask > 0).sum() / mask.size),
        "area_median": float(np.median(areas)),
        "area_p10": float(np.percentile(areas, 10)),
        "area_p90": float(np.percentile(areas, 90)),
    }


def nucleus_core_config(args: argparse.Namespace) -> NucleusCoreConfig:
    return NucleusCoreConfig(
        min_extent_area=args.nucleus_core_min_extent_area,
        min_core_area=args.nucleus_core_min_area,
        local_bg_margin=args.nucleus_core_bg_margin,
        snr_threshold=args.nucleus_core_snr,
        object_quantile=args.nucleus_core_object_quantile,
        erode_px=args.nucleus_core_erode_px,
        smooth_sigma=args.nucleus_core_smooth_sigma,
    )


def empty_nucleus_core_summary() -> dict[str, Any]:
    return {
        "nucleus_core_mask_path": "",
        "nucleus_core_n_objects": "",
        "nucleus_core_mask_fraction": "",
        "nucleus_core_area_median": "",
        "nucleus_core_area_p10": "",
        "nucleus_core_area_p90": "",
        "nucleus_core_filtered_extents": "",
        "nucleus_core_fallbacks": "",
        "nucleus_core_to_extent_fraction_median": "",
        "nucleus_core_local_p90_snr_median": "",
    }


def write_nucleus_core(
    raw: np.ndarray,
    masks: np.ndarray,
    core_mask_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = nucleus_core_config(args)
    core_masks, diagnostics = build_nucleus_core_seeds(raw, masks, config)
    write_mask(core_mask_path, core_masks)
    stats = summarize_mask(core_masks)
    flat_summary = {
        "nucleus_core_mask_path": str(core_mask_path),
        "nucleus_core_n_objects": stats["n_objects"],
        "nucleus_core_mask_fraction": stats["mask_fraction"],
        "nucleus_core_area_median": stats["area_median"],
        "nucleus_core_area_p10": stats["area_p10"],
        "nucleus_core_area_p90": stats["area_p90"],
        "nucleus_core_filtered_extents": diagnostics["n_filtered_extents"],
        "nucleus_core_fallbacks": diagnostics["n_core_fallbacks"],
        "nucleus_core_to_extent_fraction_median": diagnostics["median_core_to_extent_fraction"],
        "nucleus_core_local_p90_snr_median": diagnostics["median_local_p90_snr"],
    }
    metadata_payload = {
        "mask_path": str(core_mask_path),
        "config": asdict(config),
        "diagnostics": diagnostics,
        "summary": stats,
    }
    return flat_summary, metadata_payload


def keep_dead_object(raw: np.ndarray, labels: np.ndarray, label_id: int, obj_slice: tuple[slice, slice], profile: SegmentationProfile) -> bool:
    obj = labels[obj_slice] == label_id
    area = int(obj.sum())
    if area < profile.filter_min_area or area > profile.filter_max_area:
        return False

    y0, y1 = obj_slice[0].start, obj_slice[0].stop
    x0, x1 = obj_slice[1].start, obj_slice[1].stop
    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    aspect = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))
    if aspect > profile.filter_max_aspect:
        return False

    raw_obj = raw[obj_slice][obj]
    margin = int(profile.filter_bg_margin)
    yy0, yy1 = max(0, y0 - margin), min(raw.shape[0], y1 + margin)
    xx0, xx1 = max(0, x0 - margin), min(raw.shape[1], x1 + margin)
    window_labels = labels[yy0:yy1, xx0:xx1]
    bg_values = raw[yy0:yy1, xx0:xx1][window_labels == 0]
    if bg_values.size < 25:
        bg_values = raw[yy0:yy1, xx0:xx1][window_labels != label_id]
    if bg_values.size < 25:
        bg_values = raw.ravel()

    bg_median = float(np.median(bg_values))
    bg_mad = float(np.median(np.abs(bg_values - bg_median)))
    bg_sigma = max(bg_mad * 1.4826, 1e-6)
    raw_mean = float(np.mean(raw_obj))
    raw_p90 = float(np.percentile(raw_obj, 90))
    snr = (raw_p90 - bg_median) / bg_sigma
    return (
        (raw_mean - bg_median) >= profile.filter_min_mean_delta
        and (raw_p90 - bg_median) >= profile.filter_min_p90_delta
        and raw_p90 >= profile.filter_min_p90_abs
        and snr >= profile.filter_min_snr
    )


def postprocess_dead_raw_signal(raw: np.ndarray, masks: np.ndarray, profile: SegmentationProfile) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"Dead raw-signal postprocess expects a 2D image, got {raw.shape}")
    labels = masks.astype(np.int32, copy=False)
    max_label = int(labels.max()) if labels.size else 0
    if max_label == 0:
        return labels.astype(np.uint16, copy=False)
    label_map = np.zeros(max_label + 1, dtype=np.uint32)
    next_id = 1
    for label_id, obj_slice in enumerate(ndi.find_objects(labels), start=1):
        if obj_slice is None:
            continue
        if keep_dead_object(raw.astype(np.float32, copy=False), labels, label_id, obj_slice, profile):
            label_map[label_id] = next_id
            next_id += 1
    return label_map[labels]


def postprocess_mask(raw: np.ndarray, masks: np.ndarray, profile: SegmentationProfile) -> np.ndarray:
    if profile.postprocess == "none":
        return masks
    if profile.postprocess == "dead_raw_signal":
        return postprocess_dead_raw_signal(raw, masks, profile)
    raise ValueError(f"Unsupported postprocess mode: {profile.postprocess}")


def write_mask(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(masks.max()) if masks.size else 0
    dtype = np.uint32 if max_label > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, masks.astype(dtype, copy=False))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def append_segmentation_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_cellpose(
    image_path: Path,
    mask_path: Path,
    core_mask_path: Path | None,
    metadata_out: Path,
    summary_path: Path,
    profile: SegmentationProfile,
    args: argparse.Namespace,
    model_cache: dict[str, Any],
    cellpose_model_class: Any | None,
) -> None:
    if args.skip_existing and mask_path.exists():
        if core_mask_path is not None and not core_mask_path.exists() and not args.dry_run:
            raw_existing = tifffile.imread(image_path)
            masks_existing = np.squeeze(tifffile.imread(mask_path))
            write_nucleus_core(raw_existing, masks_existing, core_mask_path, args)
            print(f"nucleus_core_mask={core_mask_path}", flush=True)
        print(f"Using existing segmentation: {mask_path}", flush=True)
        return
    if args.dry_run:
        print(
            "DRY-RUN segmentation "
            f"image={image_path} model={profile.model} profile={profile.folder} "
            f"diameter={profile.diameter} flow_threshold={profile.flow_threshold} "
            f"cellprob_threshold={profile.cellprob_threshold} min_size={profile.min_size} "
            f"percentile={profile.low_percentile}-{profile.high_percentile} "
            f"normalization={profile.normalization} fixed={profile.fixed_low_value}-{profile.fixed_high_value} "
            f"input_transform={profile.input_transform} preprocess={profile.preprocess} gamma={profile.gamma} "
            f"postprocess={profile.postprocess} channel_axis={profile.channel_axis} normalize={profile.normalize}",
            flush=True,
        )
        return
    if cellpose_model_class is None:
        raise RuntimeError("Cellpose model class was not initialized")

    model = model_cache.get(profile.model)
    if model is None:
        print(f"loading_model={profile.model}", flush=True)
        model = cellpose_model_class(gpu=args.use_gpu, pretrained_model=profile.model)
        model_cache[profile.model] = model
        print(f"model_device={getattr(model, 'device', 'unknown')}", flush=True)

    raw = tifffile.imread(image_path)
    prepared = prepare_for_cellpose(raw, profile)
    start = time.time()
    masks, *_ = model.eval(
        prepared,
        channel_axis=profile.channel_axis,
        normalize=profile.normalize,
        diameter=profile.diameter,
        flow_threshold=profile.flow_threshold,
        cellprob_threshold=profile.cellprob_threshold,
        min_size=profile.min_size,
    )
    elapsed = time.time() - start
    masks = np.asarray(masks)
    masks = postprocess_mask(raw, masks, profile)
    write_mask(mask_path, masks)

    mask_summary = summarize_mask(masks)
    core_summary = empty_nucleus_core_summary()
    core_metadata: dict[str, Any] | None = None
    if core_mask_path is not None:
        core_summary, core_metadata = write_nucleus_core(raw, masks, core_mask_path, args)
    metadata_payload = {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "cellpose_version": cellpose_version(),
        "python_executable": sys.executable,
        "profile": asdict(profile),
        "raw_shape": list(raw.shape),
        "prepared_shape": list(prepared.shape),
        "elapsed_sec": elapsed,
        "nucleus_core": core_metadata,
        **mask_summary,
    }
    write_json(metadata_out, metadata_payload)
    append_segmentation_row(
        summary_path,
        {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "metadata_path": str(metadata_out),
            "folder_profile": profile.folder,
            "model": profile.model,
            "cellpose_version": metadata_payload["cellpose_version"],
            "python_executable": metadata_payload["python_executable"],
            "diameter": profile.diameter,
            "flow_threshold": profile.flow_threshold,
            "cellprob_threshold": profile.cellprob_threshold,
            "min_size": profile.min_size,
            "low_percentile": profile.low_percentile,
            "high_percentile": profile.high_percentile,
            "invert": profile.invert,
            "input_transform": profile.input_transform,
            "preprocess": profile.preprocess,
            "gamma": profile.gamma,
            "clahe_clip": profile.clahe_clip,
            "clahe_tile": profile.clahe_tile,
            "blur_sigma": profile.blur_sigma,
            "background_sigma": profile.background_sigma,
            "tophat_kernel": profile.tophat_kernel,
            "channel_axis": profile.channel_axis,
            "normalize": profile.normalize,
            "normalization": profile.normalization,
            "fixed_low_value": profile.fixed_low_value,
            "fixed_high_value": profile.fixed_high_value,
            "elapsed_sec": f"{elapsed:.3f}",
            "postprocess": profile.postprocess,
            "filter_min_area": profile.filter_min_area,
            "filter_max_area": profile.filter_max_area,
            "filter_max_aspect": profile.filter_max_aspect,
            "filter_min_mean_delta": profile.filter_min_mean_delta,
            "filter_min_p90_delta": profile.filter_min_p90_delta,
            "filter_min_p90_abs": profile.filter_min_p90_abs,
            "filter_min_snr": profile.filter_min_snr,
            "filter_bg_margin": profile.filter_bg_margin,
            **mask_summary,
            **core_summary,
        },
    )
    print(f"mask={mask_path}", flush=True)
    if core_mask_path is not None:
        print(f"nucleus_core_mask={core_mask_path}", flush=True)
    print(f"metadata={metadata_out}", flush=True)


def run_classifier(
    image_path: Path,
    mask_path: Path,
    classification_dir: Path,
    image_id: str,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "cellpose_pipeline/scripts/17_classify_cell_states.py",
        "--raw-image",
        str(image_path),
        "--mask-image",
        str(mask_path),
        "--image-id",
        image_id,
        "--out-dir",
        str(classification_dir),
    ]
    run_command(cmd, env, dry_run)


def select_images_for_profiles(
    images: list[Path],
    args: argparse.Namespace,
) -> tuple[list[tuple[Path, SegmentationProfile | None]], list[tuple[Path, str]]]:
    selected: list[tuple[Path, SegmentationProfile | None]] = []
    skipped: list[tuple[Path, str]] = []
    for path in images:
        image_path = path.resolve()
        profile = resolve_profile(image_path, args)
        if profile is None:
            skipped.append((image_path, image_path.parent.name))
            continue
        selected.append((image_path, profile))
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected, skipped


def process_selected_images(
    selected: list[tuple[Path, SegmentationProfile | None]],
    args: argparse.Namespace,
    run_dir: Path,
    segmentation_dir: Path,
    metadata_dir: Path,
    classification_dir: Path,
    summary_path: Path,
    env: dict[str, str],
    model_cache: dict[str, Any],
    cellpose_model_class: Any | None,
    phase: str,
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for index, (image_path, profile) in enumerate(selected, start=1):
        image_id = image_path.stem
        image_key = extract_image_key(image_path)
        if profile is None:
            continue
        is_high_density = image_key in args.high_density_keys and profile == HIGH_DENSITY_FOLDER_PROFILES.get(profile.folder)
        mask_path = (
            args.mask_image.resolve()
            if args.mask_image
            else expected_mask_path(segmentation_dir, image_path, profile, args.flat_profile_output)
        )
        core_mask_path = (
            None
            if args.mask_image
            else expected_nucleus_core_path(run_dir, image_path, profile, args.flat_profile_output)
        )
        metadata_out = metadata_path(metadata_dir, image_path, profile, args.flat_profile_output)
        auto_profile = profile.folder in FOLDER_PROFILES
        should_classify = not args.segmentation_only and (not auto_profile or args.force_classification)

        print(f"[{phase} {index}/{len(selected)}] {image_path}", flush=True)
        print(f"image_key={image_key} high_density_profile={is_high_density}", flush=True)
        print(f"profile={profile.folder} model={profile.model} params={asdict(profile)}", flush=True)
        try:
            if args.mask_image:
                print(f"Using supplied mask: {mask_path}", flush=True)
            else:
                run_cellpose(
                    image_path,
                    mask_path,
                    core_mask_path,
                    metadata_out,
                    summary_path,
                    profile,
                    args,
                    model_cache,
                    cellpose_model_class,
                )

            if not args.dry_run and not mask_path.exists():
                raise FileNotFoundError(f"Expected mask not found after segmentation: {mask_path}")

            if not args.dry_run:
                write_segmentation_overlay(
                    image_path,
                    mask_path,
                    segmentation_overlay_path(run_dir, image_id, profile, args.flat_profile_output),
                )

            if should_classify:
                run_classifier(image_path, mask_path, classification_dir, image_id, env, args.dry_run)
            else:
                print(f"classification=skipped image={image_path}", flush=True)
        except Exception as exc:
            failures.append((str(image_path), repr(exc)))
            print(f"FAILED [{phase} {index}/{len(selected)}] {image_path}: {exc!r}", flush=True)
            if not args.continue_on_error:
                raise
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cellpose segmentation and optional live/dead/transitional classification."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image-path", type=Path, action="append", help="One image to process. Can be passed repeatedly.")
    inputs.add_argument("--dir", type=Path, help="Directory of images to process.")
    parser.add_argument(
        "--mask-image",
        type=Path,
        help="Existing mask for a single --image-path. Skips segmentation and runs classification only.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively collect images when --dir is used.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed after filtering.")
    parser.add_argument("--run-name", default="segmentation_classification", help="Name under workflow_runs.")
    parser.add_argument("--out-root", type=Path, default=Path("cellpose_pipeline/workflow_runs"))
    parser.add_argument("--summary-name", default="segmentation_summary.csv", help="Segmentation summary CSV under run dir.")
    parser.add_argument("--profile-mode", choices=("auto", "manual"), default="auto")
    parser.add_argument(
        "--enable-high-density-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use high-density auto profiles. Directory mode auto-generates density calls from Nuclei masks by default.",
    )
    parser.add_argument(
        "--high-density-calls-csv",
        type=Path,
        default=None,
        help="Optional existing CSV with key and high_density columns. If omitted in directory mode, Nuclei masks are used.",
    )
    parser.add_argument(
        "--auto-generate-density-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In directory auto mode, segment Nuclei first and write qc/density_calls.csv before other profiles.",
    )
    parser.add_argument("--density-calls-name", default="density_calls.csv", help="Auto-generated density calls filename under qc/.")
    parser.add_argument(
        "--dead-calibration-json",
        type=Path,
        help="Full-cohort calibration JSON produced by script 45.",
    )
    parser.add_argument(
        "--dead-calibration-map",
        type=Path,
        help="Per-image calibration map produced by script 45.",
    )
    parser.add_argument(
        "--dead-calibration-mode",
        choices=("off", "global", "bounded-background"),
        default="bounded-background",
        help="Apply cohort-global anchors or a cohort-bounded per-image background offset.",
    )
    parser.add_argument("--nuclei-count-threshold", type=int, default=4000, help="High-density call threshold on nuclei count.")
    parser.add_argument(
        "--nuclei-mask-fraction-threshold",
        type=float,
        default=0.22,
        help="High-density call threshold on Nuclei mask fraction when its trigger is explicitly enabled.",
    )
    parser.add_argument(
        "--enable-nuclei-mask-fraction-trigger",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow nuclear mask fraction alone to trigger high density. Disabled by default because "
            "fluorescence exposure inflates nuclear extent area."
        ),
    )
    parser.add_argument(
        "--nuclei-median-nn-threshold",
        type=float,
        default=16.0,
        help="High-density call threshold on median nearest-neighbor nuclei distance in pixels.",
    )
    parser.add_argument("--nucleus-core-min-extent-area", type=int, default=15)
    parser.add_argument("--nucleus-core-min-area", type=int, default=5)
    parser.add_argument("--nucleus-core-bg-margin", type=int, default=10)
    parser.add_argument("--nucleus-core-snr", type=float, default=2.0)
    parser.add_argument("--nucleus-core-object-quantile", type=float, default=0.35)
    parser.add_argument("--nucleus-core-erode-px", type=int, default=1)
    parser.add_argument("--nucleus-core-smooth-sigma", type=float, default=1.0)
    parser.add_argument(
        "--skip-unknown-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In auto mode, skip images outside supported folder profiles instead of using manual parameters.",
    )
    parser.add_argument("--pretrained-model", default="cpsam", help="Manual-mode model fallback.")
    parser.add_argument("--channel-axis", type=int, default=2, help="Manual-mode channel axis fallback.")
    parser.add_argument("--diameter", type=float, default=30, help="Manual-mode diameter fallback.")
    parser.add_argument("--flow-threshold", type=float, default=0.4, help="Manual-mode flow threshold fallback.")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0, help="Manual-mode cell probability fallback.")
    parser.add_argument("--min-size", type=int, default=15, help="Manual-mode minimum mask size fallback.")
    parser.add_argument("--low-percentile", type=float, default=1.0, help="Manual-mode preprocessing low percentile.")
    parser.add_argument("--high-percentile", type=float, default=99.0, help="Manual-mode preprocessing high percentile.")
    parser.add_argument("--invert", action="store_true", help="Manual-mode preprocessing inversion.")
    parser.add_argument("--normalization", choices=("percentile", "fixed"), default="percentile", help="Manual-mode intensity scaling.")
    parser.add_argument("--fixed-low-value", type=float, default=0.0, help="Manual-mode fixed normalization low value.")
    parser.add_argument("--fixed-high-value", type=float, default=1.0, help="Manual-mode fixed normalization high value.")
    parser.add_argument(
        "--input-transform",
        choices=("identity", "rgb_luma", "rgb_mean", "rgb_median", "rgb_min"),
        default="identity",
        help="Manual-mode input transform applied before normalization and preprocessing.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("percentile", "clahe", "background", "tophat", "rgb_luma", "rgb_mean", "rgb_median", "rgb_min"),
        default="percentile",
        help="Manual-mode preprocessing transform after percentile scaling.",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Manual-mode gamma after preprocessing.")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="Manual-mode CLAHE clip limit.")
    parser.add_argument("--clahe-tile", type=int, default=16, help="Manual-mode CLAHE tile grid size.")
    parser.add_argument("--blur-sigma", type=float, default=0.0, help="Manual-mode Gaussian blur sigma after preprocessing.")
    parser.add_argument("--background-sigma", type=float, default=0.0, help="Manual-mode background subtraction sigma.")
    parser.add_argument("--tophat-kernel", type=int, default=0, help="Manual-mode top-hat kernel diameter.")
    parser.add_argument("--postprocess", choices=("none", "dead_raw_signal"), default="none", help="Manual-mode mask postprocessing.")
    parser.add_argument("--filter-min-area", type=int, default=20, help="Manual-mode postprocess minimum object area.")
    parser.add_argument("--filter-max-area", type=int, default=2500, help="Manual-mode postprocess maximum object area.")
    parser.add_argument("--filter-max-aspect", type=float, default=4.0, help="Manual-mode postprocess maximum bbox aspect ratio.")
    parser.add_argument("--filter-min-mean-delta", type=float, default=0.2, help="Manual-mode postprocess mean-background threshold.")
    parser.add_argument("--filter-min-p90-delta", type=float, default=0.7, help="Manual-mode postprocess p90-background threshold.")
    parser.add_argument("--filter-min-p90-abs", type=float, default=10.0, help="Manual-mode postprocess raw p90 absolute threshold.")
    parser.add_argument("--filter-min-snr", type=float, default=1.25, help="Manual-mode postprocess local p90 SNR threshold.")
    parser.add_argument("--filter-bg-margin", type=int, default=14, help="Manual-mode postprocess local background margin.")
    parser.add_argument(
        "--cellpose-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Manual-mode Cellpose internal normalization. Auto profiles force normalize=False.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-device", default=None, help="Retained for CLI compatibility; Python API auto-selects device.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true", help="Log failures and continue with the next image.")
    parser.add_argument("--segmentation-only", action="store_true", help="Skip cell-state classification.")
    parser.add_argument(
        "--flat-profile-output",
        action="store_true",
        help="Do not add a profile subdirectory under segmentations, metadata, or qc/segmentation_overlays.",
    )
    parser.add_argument(
        "--force-classification",
        action="store_true",
        help="Allow classifier after auto-profile segmentation. Default auto profiles are segmentation-only.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Check version/API/images and exit before segmentation.")
    parser.add_argument("--check-models", action="store_true", help="In preflight, instantiate models used by selected profiles.")
    parser.add_argument(
        "--density-only",
        action="store_true",
        help="Segment Nuclei and write qc/density_calls.csv, then exit before other profiles.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running segmentation/classification.")
    return parser.parse_args()


def print_profiles() -> None:
    print("supported_profiles=", flush=True)
    for name, profile in FOLDER_PROFILES.items():
        print(f"  {name}: {asdict(profile)}", flush=True)
    print("high_density_profiles=", flush=True)
    for name, profile in HIGH_DENSITY_FOLDER_PROFILES.items():
        print(f"  {name}: {asdict(profile)}", flush=True)
    print(f"ignored_folders={sorted(IGNORED_PROFILE_FOLDERS)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mask_image and (not args.image_path or len(args.image_path) != 1):
        raise SystemExit("--mask-image is only supported with exactly one --image-path")
    if args.density_only and not args.dir:
        raise SystemExit("--density-only requires --dir")
    if args.gpu_device:
        print("--gpu-device is ignored by the Cellpose Python API; use CUDA_VISIBLE_DEVICES if needed.", flush=True)
    args.high_density_calls_explicit = args.high_density_calls_csv is not None
    args.high_density_keys = load_high_density_keys(args.high_density_calls_csv) if args.high_density_calls_explicit else set()
    args.dead_calibration = load_dead_calibration(args.dead_calibration_json)
    args.dead_calibration_by_key = load_dead_calibration_map(args.dead_calibration_map)

    cellpose_model_class = None
    version = require_cellpose_version()
    assert_cellpose_api_compatible()

    if args.dir:
        images = iter_images(args.dir, args.recursive)
    else:
        images = [path for path in args.image_path]

    run_dir = args.out_root if args.run_name in ("", ".") else args.out_root / args.run_name
    segmentation_dir = run_dir / "segmentations"
    metadata_dir = run_dir / "metadata"
    classification_dir = run_dir / "classification"
    summary_path = run_dir / args.summary_name
    auto_density_csv = high_density_calls_path(run_dir, args.density_calls_name)

    if (
        args.enable_high_density_profiles
        and not args.high_density_calls_explicit
        and auto_density_csv.exists()
    ):
        args.high_density_calls_csv = auto_density_csv
        args.high_density_keys = load_high_density_keys(auto_density_csv)

    selected, skipped = select_images_for_profiles(images, args)
    if not selected and not args.mask_image:
        raise SystemExit("No images selected after profile filtering.")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLCONFIGDIR", "cellpose_pipeline/tmp/matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print(f"run_dir={run_dir}", flush=True)
    print(f"n_images={len(selected)}", flush=True)
    print(f"n_skipped={len(skipped)}", flush=True)
    print(f"cellpose_version={version}", flush=True)
    print(f"python_executable={sys.executable}", flush=True)
    print(f"high_density_profiles_enabled={args.enable_high_density_profiles}", flush=True)
    print(f"high_density_calls_csv={args.high_density_calls_csv}", flush=True)
    print(f"auto_generate_density_calls={args.auto_generate_density_calls}", flush=True)
    print(f"density_calls_name={args.density_calls_name}", flush=True)
    print(f"density_only={args.density_only}", flush=True)
    print(f"dead_calibration_json={args.dead_calibration_json}", flush=True)
    print(f"dead_calibration_map={args.dead_calibration_map}", flush=True)
    print(f"dead_calibration_mode={args.dead_calibration_mode}", flush=True)
    print(f"dead_calibration_keys={len(args.dead_calibration_by_key)}", flush=True)
    print(f"nuclei_mask_fraction_trigger_enabled={args.enable_nuclei_mask_fraction_trigger}", flush=True)
    print(f"nucleus_core_config={asdict(nucleus_core_config(args))}", flush=True)
    print(f"flat_profile_output={args.flat_profile_output}", flush=True)
    print(f"n_high_density_keys={len(args.high_density_keys)}", flush=True)
    print_profiles()
    for image_path, folder in skipped[:20]:
        print(f"skipped_profile folder={folder} image={image_path}", flush=True)
    if len(skipped) > 20:
        print(f"skipped_profile additional={len(skipped) - 20}", flush=True)

    if args.preflight_only:
        if args.check_models and selected:
            cellpose_model_class = import_cellpose_model_class()
            for model_name in sorted({profile.model for _, profile in selected if profile is not None}):
                print(f"preflight_loading_model={model_name}", flush=True)
                model = cellpose_model_class(gpu=args.use_gpu, pretrained_model=model_name)
                print(f"preflight_model_device={getattr(model, 'device', 'unknown')}", flush=True)
        print("preflight_ok=1", flush=True)
        return

    segmentation_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.mask_image:
        cellpose_model_class = import_cellpose_model_class()

    failures: list[tuple[str, str]] = []
    model_cache: dict[str, Any] = {}
    should_auto_generate_density = (
        args.enable_high_density_profiles
        and args.auto_generate_density_calls
        and not args.high_density_calls_explicit
        and args.dir is not None
        and not args.mask_image
    )
    if should_auto_generate_density:
        nuclei_selected = [(path, profile) for path, profile in selected if profile is not None and profile.folder == "Nuclei"]
        if nuclei_selected:
            print(f"density_prepass_nuclei_images={len(nuclei_selected)}", flush=True)
            failures.extend(
                process_selected_images(
                    nuclei_selected,
                    args,
                    run_dir,
                    segmentation_dir,
                    metadata_dir,
                    classification_dir,
                    summary_path,
                    env,
                    model_cache,
                    cellpose_model_class,
                    phase="density-nuclei",
                )
            )
            if not args.dry_run:
                args.high_density_calls_csv = auto_density_csv
                core_mask_dir = run_dir / "nucleus_core_seeds"
                nuclei_mask_dir = segmentation_dir
                if not args.flat_profile_output:
                    core_mask_dir = core_mask_dir / "Nuclei"
                    nuclei_mask_dir = segmentation_dir / "Nuclei"
                args.high_density_keys = write_density_calls_from_nuclei_masks(
                    nuclei_mask_dir,
                    core_mask_dir,
                    auto_density_csv,
                    args,
                )
                selected, skipped = select_images_for_profiles(images, args)
                print(f"density_calls_active={args.high_density_calls_csv}", flush=True)
                print(f"n_high_density_keys_active={len(args.high_density_keys)}", flush=True)
            else:
                print("density_calls_skipped=dry_run", flush=True)
            if args.density_only:
                print("density_only_complete=1", flush=True)
            else:
                remaining_selected = [
                    (path, profile)
                    for path, profile in selected
                    if profile is not None and profile.folder != "Nuclei"
                ]
                failures.extend(
                    process_selected_images(
                        remaining_selected,
                        args,
                        run_dir,
                        segmentation_dir,
                        metadata_dir,
                        classification_dir,
                        summary_path,
                        env,
                        model_cache,
                        cellpose_model_class,
                        phase="segmentation",
                    )
                )
        else:
            print("density_prepass_skipped=no_nuclei_images_selected", flush=True)
            if args.density_only:
                print("density_only_complete=0", flush=True)
            else:
                failures.extend(
                    process_selected_images(
                        selected,
                        args,
                        run_dir,
                        segmentation_dir,
                        metadata_dir,
                        classification_dir,
                        summary_path,
                        env,
                        model_cache,
                        cellpose_model_class,
                        phase="segmentation",
                    )
                )
    else:
        if args.density_only:
            print("density_only_skipped=auto_density_disabled_or_explicit_calls", flush=True)
        else:
            failures.extend(
                process_selected_images(
                    selected,
                    args,
                    run_dir,
                    segmentation_dir,
                    metadata_dir,
                    classification_dir,
                    summary_path,
                    env,
                    model_cache,
                    cellpose_model_class,
                    phase="segmentation",
                )
            )

    if failures:
        failure_path = run_dir / "failures.tsv"
        with failure_path.open("w") as handle:
            handle.write("image_path\terror\n")
            for image_path, error in failures:
                handle.write(f"{image_path}\t{error}\n")
        print(f"Workflow complete with {len(failures)} failures: {failure_path}", flush=True)
    else:
        print(f"Workflow complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
