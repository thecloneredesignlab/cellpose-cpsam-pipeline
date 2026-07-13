#!/usr/bin/env python3
"""Run a reproducible high-density Brightfield/Combined Cellpose screen.

Only fields marked high density by the supplied production density table are
processed by default.  Nuclear masks are not modified.  This script performs
inference and records complete configuration metadata; cross-channel and
nucleus-aware scoring is handled by 20_score_high_density_bf_combined.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from cellpose import models


REQUIRED_CELLPOSE_VERSION = "4.2.1.1"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")


@dataclass(frozen=True)
class CandidateConfig:
    profile: str
    tag: str
    transform: str
    model: str
    diameter: float
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    low_percentile: float
    high_percentile: float
    baseline: bool = False
    family: str = ""
    preprocess: str = "percentile"
    gamma: float = 1.0
    clahe_clip: float = 0.0
    clahe_tile: int = 16
    blur_sigma: float = 0.0
    background_sigma: float = 0.0
    tophat_kernel: int = 0
    channel_axis: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen high-density Brightfield and Combined segmentation candidates."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--nuclei-run-root", type=Path, required=True)
    parser.add_argument("--density-calls", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("Brightfield", "Combined"))
    parser.add_argument("--config-tags", nargs="+")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-low-density",
        action="store_true",
        help="Include requested low-density keys. The default is production high-density rows only.",
    )
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_key(path_or_name: Path | str) -> str:
    match = KEY_RE.search(Path(path_or_name).name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path_or_name}")
    return match.group(1)


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_density_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "key" not in rows[0] or "high_density" not in rows[0]:
        raise ValueError(f"Density table must contain key and high_density columns: {path}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["key"]
        if key in result:
            raise ValueError(f"Duplicate density key {key} in {path}")
        result[key] = row
    return result


def load_configs(path: Path) -> list[CandidateConfig]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Candidate config must be a non-empty JSON list: {path}")
    configs = [CandidateConfig(**row) for row in payload]
    tags = [config.tag for config in configs]
    if len(tags) != len(set(tags)):
        raise ValueError("Candidate tags must be unique")
    for profile in ("Brightfield", "Combined"):
        baselines = [config.tag for config in configs if config.profile == profile and config.baseline]
        if len(baselines) != 1:
            raise ValueError(f"Expected exactly one {profile} baseline, found {baselines}")
    return configs


def index_images(input_root: Path, profile: str) -> dict[str, Path]:
    folder = input_root / profile
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing input folder: {folder}")
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate {profile} key {key}: {result[key]} and {path}")
        result[key] = path
    return result


def index_nucleus_core_masks(run_root: Path) -> dict[str, Path]:
    candidates = (
        run_root / "Nuclei" / "nucleus_core_seeds",
        run_root / "nucleus_core_seeds" / "Nuclei",
        run_root / "nucleus_core_seeds",
    )
    mask_dir = next((path for path in candidates if path.is_dir()), None)
    if mask_dir is None:
        raise FileNotFoundError(f"Cannot locate nucleus_core_seeds under {run_root}")
    result: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*_core_masks.tif")):
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate nucleus-core key {key}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"No nucleus core masks found in {mask_dir}")
    return result


def normalize_percentile(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    low, high = np.percentile(data, (low_pct, high_pct))
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def odd_kernel_from_sigma(sigma: float) -> int:
    value = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    return value if value % 2 else value + 1


def grayscale_transform(raw: np.ndarray, transform: str) -> np.ndarray:
    data = raw.astype(np.float32, copy=False)
    if transform == "gray":
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            return data[..., 0]
        raise ValueError(f"gray transform does not support shape={data.shape}")
    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(f"{transform} expects a channel-last RGB image, got shape={data.shape}")
    rgb = data[..., :3]
    if transform == "rgb_luma":
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if transform == "rgb_mean":
        return np.mean(rgb, axis=2)
    if transform == "rgb_median":
        return np.median(rgb, axis=2)
    if transform == "rgb_min":
        return np.min(rgb, axis=2)
    if transform == "rgb_ch0":
        return rgb[..., 0]
    if transform == "rgb_ch1":
        return rgb[..., 1]
    if transform == "rgb_ch2":
        return rgb[..., 2]
    if transform == "rgb_mix12_70_30":
        return 0.7 * rgb[..., 1] + 0.3 * rgb[..., 2]
    if transform == "rgb_mix12_50_50":
        return 0.5 * rgb[..., 1] + 0.5 * rgb[..., 2]
    raise ValueError(f"Unsupported transform: {transform}")


def preprocess_image(raw: np.ndarray, config: CandidateConfig) -> np.ndarray:
    transformed = grayscale_transform(raw, config.transform)
    prepared = normalize_percentile(transformed, config.low_percentile, config.high_percentile)
    if config.preprocess == "percentile":
        pass
    elif config.preprocess == "clahe":
        data8 = np.clip(prepared * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip),
            tileGridSize=(int(config.clahe_tile), int(config.clahe_tile)),
        )
        prepared = clahe.apply(data8).astype(np.float32) / 255.0
    elif config.preprocess == "background":
        sigma = float(config.background_sigma)
        if sigma <= 0:
            raise ValueError(f"background preprocessing requires background_sigma > 0: {config.tag}")
        kernel = odd_kernel_from_sigma(sigma)
        background = cv2.GaussianBlur(prepared, (kernel, kernel), sigma)
        corrected = prepared - background
        corrected -= float(corrected.min())
        high = float(np.percentile(corrected, 99.8))
        prepared = corrected / max(high, 1e-6)
    elif config.preprocess == "tophat":
        kernel = int(config.tophat_kernel)
        if kernel <= 0:
            raise ValueError(f"tophat preprocessing requires tophat_kernel > 0: {config.tag}")
        if kernel % 2 == 0:
            kernel += 1
        data8 = np.clip(prepared * 255.0, 0, 255).astype(np.uint8)
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        top = cv2.morphologyEx(data8, cv2.MORPH_TOPHAT, struct).astype(np.float32)
        prepared = top / max(float(np.percentile(top, 99.8)), 1e-6)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {config.preprocess}")
    if config.gamma <= 0:
        raise ValueError(f"gamma must be positive: {config.tag}")
    if config.gamma != 1.0:
        prepared = np.power(np.clip(prepared, 0.0, 1.0), float(config.gamma))
    if config.blur_sigma > 0:
        kernel = odd_kernel_from_sigma(config.blur_sigma)
        prepared = cv2.GaussianBlur(prepared.astype(np.float32), (kernel, kernel), config.blur_sigma)
    return np.clip(prepared, 0.0, 1.0).astype(np.float32, copy=False)


def summarize_mask(mask: np.ndarray) -> dict[str, float | int]:
    labels = mask.astype(np.int64, copy=False)
    areas = np.bincount(labels.ravel())[1:]
    if not areas.size:
        return {
            "n_objects": 0,
            "mask_fraction": 0.0,
            "area_median": 0.0,
            "area_p10": 0.0,
            "area_p90": 0.0,
        }
    return {
        "n_objects": int(np.count_nonzero(areas)),
        "mask_fraction": float(np.mean(labels > 0)),
        "area_median": float(np.median(areas[areas > 0])),
        "area_p10": float(np.percentile(areas[areas > 0], 10)),
        "area_p90": float(np.percentile(areas[areas > 0], 90)),
    }


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = int(mask.max()) if mask.size else 0
    dtype = np.uint32 if maximum > np.iinfo(np.uint16).max else np.uint16
    tifffile.imwrite(path, mask.astype(dtype, copy=False), compression="zlib")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    version = metadata.version("cellpose")
    if version != REQUIRED_CELLPOSE_VERSION:
        raise SystemExit(f"Expected cellpose=={REQUIRED_CELLPOSE_VERSION}, found {version}")

    args.input_root = args.input_root.resolve()
    args.nuclei_run_root = args.nuclei_run_root.resolve()
    args.density_calls = args.density_calls.resolve()
    args.candidate_config = args.candidate_config.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    density_rows = read_density_rows(args.density_calls)
    nucleus_cores = index_nucleus_core_masks(args.nuclei_run_root)
    image_maps = {
        profile: index_images(args.input_root, profile)
        for profile in ("Brightfield", "Combined")
    }
    configs = load_configs(args.candidate_config)
    if args.profiles:
        requested_profiles = set(args.profiles)
        configs = [config for config in configs if config.profile in requested_profiles]
    if args.config_tags:
        requested_tags = set(args.config_tags)
        configs = [config for config in configs if config.tag in requested_tags]
        missing_tags = sorted(requested_tags - {config.tag for config in configs})
        if missing_tags:
            raise SystemExit(f"Unknown or profile-filtered config tags: {missing_tags}")
    if not configs:
        raise SystemExit("No candidate configurations selected")

    if args.keys:
        selected_keys = list(dict.fromkeys(args.keys))
        missing_density = sorted(set(selected_keys) - set(density_rows))
        if missing_density:
            raise SystemExit(f"Requested keys missing from density table: {missing_density}")
        if not args.include_low_density:
            low_requested = [key for key in selected_keys if not bool_value(density_rows[key]["high_density"])]
            if low_requested:
                raise SystemExit(
                    f"Requested keys are not production high-density fields: {low_requested}; "
                    "pass --include-low-density to override"
                )
    else:
        selected_keys = sorted(
            key for key, row in density_rows.items() if bool_value(row["high_density"])
        )
    if args.limit is not None:
        selected_keys = selected_keys[: args.limit]
    if not selected_keys:
        raise SystemExit("No fields selected")

    missing_core = sorted(set(selected_keys) - set(nucleus_cores))
    if missing_core:
        raise SystemExit(f"Selected keys missing optimized nucleus-core masks: {missing_core}")
    for profile in {config.profile for config in configs}:
        missing_images = sorted(set(selected_keys) - set(image_maps[profile]))
        if missing_images:
            raise SystemExit(f"Selected keys missing {profile} images: {missing_images}")

    split_rows = sorted(
        selected_keys,
        key=lambda key: (
            float(density_rows[key].get("nuclei_count", 0) or 0),
            key,
        ),
    )
    calibration_keys = split_rows[0::2]
    validation_keys = split_rows[1::2]
    selection_manifest = {
        "cellpose_version": version,
        "input_root": str(args.input_root),
        "nuclei_run_root": str(args.nuclei_run_root),
        "density_calls": str(args.density_calls),
        "candidate_config": str(args.candidate_config),
        "selected_keys": selected_keys,
        "calibration_keys": calibration_keys,
        "validation_keys": validation_keys,
        "selected_configs": [asdict(config) for config in configs],
    }
    (args.out_root / "screen_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.out_root / "selected_candidate_configs.json").write_text(
        json.dumps([asdict(config) for config in configs], indent=2, sort_keys=True) + "\n"
    )

    model_cache: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    print(f"cellpose_version={version}", flush=True)
    print(f"n_fields={len(selected_keys)}", flush=True)
    print(f"n_configs={len(configs)}", flush=True)
    print(f"calibration_keys={','.join(calibration_keys)}", flush=True)
    print(f"validation_keys={','.join(validation_keys)}", flush=True)

    for config_index, config in enumerate(configs, start=1):
        model = model_cache.get(config.model)
        if model is None:
            model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=config.model)
            model_cache[config.model] = model
            print(f"model={config.model} device={getattr(model, 'device', 'unknown')}", flush=True)
        for key_index, key in enumerate(selected_keys, start=1):
            image_path = image_maps[config.profile][key]
            mask_path = (
                args.out_root
                / "masks"
                / config.profile
                / config.tag
                / f"{image_path.stem}_cp_masks.tif"
            )
            metadata_path = (
                args.out_root
                / "metadata"
                / config.profile
                / config.tag
                / f"{image_path.stem}_metadata.json"
            )
            print(
                f"[{config_index}/{len(configs)}][{key_index}/{len(selected_keys)}] "
                f"profile={config.profile} config={config.tag} key={key}",
                flush=True,
            )
            if args.skip_existing and mask_path.is_file():
                mask = np.squeeze(tifffile.imread(mask_path))
                elapsed = 0.0
                status = "existing"
            else:
                raw = tifffile.imread(image_path)
                prepared = preprocess_image(raw, config)
                started = time.time()
                mask, *_ = model.eval(
                    prepared,
                    channel_axis=config.channel_axis,
                    normalize=False,
                    diameter=config.diameter,
                    flow_threshold=config.flow_threshold,
                    cellprob_threshold=config.cellprob_threshold,
                    min_size=config.min_size,
                )
                elapsed = time.time() - started
                mask = np.squeeze(np.asarray(mask))
                if mask.ndim != 2:
                    raise ValueError(f"Expected 2D mask for {config.tag}/{key}, got {mask.shape}")
                write_mask(mask_path, mask)
                status = "computed"
            stats = summarize_mask(mask)
            row = {
                "profile": config.profile,
                "config": config.tag,
                "family": config.family,
                "baseline": config.baseline,
                "key": key,
                "split": "calibration" if key in calibration_keys else "validation",
                "high_density": bool_value(density_rows[key]["high_density"]),
                "nuclei_count": density_rows[key].get("nuclei_count", ""),
                "nuclei_median_nn_px": density_rows[key].get("nuclei_median_nn_px", ""),
                "image_path": str(image_path),
                "nucleus_core_path": str(nucleus_cores[key]),
                "mask_path": str(mask_path),
                "status": status,
                "elapsed_sec": elapsed,
                **stats,
                **{f"param_{name}": value for name, value in asdict(config).items()},
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            summary_rows.append(row)

    write_rows(args.out_root / "inference_summary.csv", summary_rows)
    print(f"screen_manifest={args.out_root / 'screen_manifest.json'}", flush=True)
    print(f"inference_summary={args.out_root / 'inference_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
