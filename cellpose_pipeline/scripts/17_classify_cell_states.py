#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.measure import regionprops


STATE_COLORS = {
    "live": np.array([1.0, 0.0, 0.0]),
    "dead": np.array([0.0, 0.25, 1.0]),
    "transitional": np.array([0.65, 0.0, 0.85]),
    "artifact": np.array([1.0, 0.9, 0.0]),
    "uncertain": np.array([0.55, 0.55, 0.55]),
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


def safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(math.log((numerator + 1.0) / (denominator + 1.0)))


def classify(row: dict) -> tuple[str, str]:
    area = row["area"]
    sat = row["hsv_s_mean"]
    r_frac = row["r_frac"]
    g_frac = row["g_frac"]
    b_frac = row["b_frac"]
    red_excess = row["red_excess"]
    blue_excess = row["blue_excess"]
    purple_score = row["purple_score"]
    red_blue_balance = row["red_blue_balance"]
    log_b_over_r = row["log_b_over_r"]

    if area < 35:
        return "artifact", "area_lt_35"
    if sat < 0.08:
        return "uncertain", "low_saturation"

    # Dead cells in the first reviewed image are compact blue/cyan objects.
    if (b_frac >= 0.40 and blue_excess >= 0.055 and log_b_over_r >= 0.25) or (
        row["median_b"] >= 175 and b_frac >= 0.38 and blue_excess >= 0.035
    ):
        return "dead", "blue_cyan_rule"

    # Transitional should be conservative. In the first reviewed image there are
    # no transitional cells, so require both substantial red/blue and low green.
    if (
        min(r_frac, b_frac) >= 0.34
        and g_frac <= 0.285
        and red_blue_balance >= 0.92
        and purple_score >= 0.055
        and sat >= 0.18
    ):
        return "transitional", "strict_purple_rule"

    if r_frac >= 0.335 and red_excess >= 0.015:
        return "live", "red_rule"

    if area < 80 and sat < 0.16:
        return "artifact", "small_low_saturation"

    return "uncertain", "no_confident_rule"


def extract_features(raw_rgb: np.ndarray, masks: np.ndarray, image_id: str) -> pd.DataFrame:
    raw_float = raw_rgb.astype(np.float32)
    rgb_unit = raw_float / 255.0
    hsv = rgb2hsv(rgb_unit)
    lab = rgb2lab(rgb_unit)

    rows = []
    height, width = masks.shape
    for prop in regionprops(masks):
        mask_id = int(prop.label)
        coords = prop.coords
        pixels = raw_float[coords[:, 0], coords[:, 1], :]
        hsv_pixels = hsv[coords[:, 0], coords[:, 1], :]
        lab_pixels = lab[coords[:, 0], coords[:, 1], :]

        median_rgb = np.median(pixels, axis=0)
        mean_rgb = np.mean(pixels, axis=0)
        std_rgb = np.std(pixels, axis=0)
        q10_rgb = np.percentile(pixels, 10, axis=0)
        q90_rgb = np.percentile(pixels, 90, axis=0)
        rgb_sum = float(median_rgb.sum()) + 1e-9
        r_frac, g_frac, b_frac = (median_rgb / rgb_sum).tolist()

        y0, x0, y1, x1 = prop.bbox
        border_touch = y0 == 0 or x0 == 0 or y1 == height or x1 == width

        red_excess = r_frac - max(g_frac, b_frac)
        blue_excess = b_frac - max(r_frac, g_frac)
        purple_score = min(r_frac, b_frac) - g_frac
        red_blue_balance = 1.0 - abs(r_frac - b_frac)

        row = {
            "image_id": image_id,
            "mask_id": mask_id,
            "area": int(prop.area),
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "bbox_y0": int(y0),
            "bbox_x0": int(x0),
            "bbox_y1": int(y1),
            "bbox_x1": int(x1),
            "border_touch": bool(border_touch),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "major_axis_length": float(prop.major_axis_length),
            "minor_axis_length": float(prop.minor_axis_length),
            "median_r": float(median_rgb[0]),
            "median_g": float(median_rgb[1]),
            "median_b": float(median_rgb[2]),
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
            "std_r": float(std_rgb[0]),
            "std_g": float(std_rgb[1]),
            "std_b": float(std_rgb[2]),
            "q10_r": float(q10_rgb[0]),
            "q10_g": float(q10_rgb[1]),
            "q10_b": float(q10_rgb[2]),
            "q90_r": float(q90_rgb[0]),
            "q90_g": float(q90_rgb[1]),
            "q90_b": float(q90_rgb[2]),
            "r_frac": float(r_frac),
            "g_frac": float(g_frac),
            "b_frac": float(b_frac),
            "log_r_over_b": safe_log_ratio(median_rgb[0], median_rgb[2]),
            "log_b_over_r": safe_log_ratio(median_rgb[2], median_rgb[0]),
            "log_r_over_g": safe_log_ratio(median_rgb[0], median_rgb[1]),
            "log_b_over_g": safe_log_ratio(median_rgb[2], median_rgb[1]),
            "hsv_h_mean": float(np.mean(hsv_pixels[:, 0])),
            "hsv_s_mean": float(np.mean(hsv_pixels[:, 1])),
            "hsv_v_mean": float(np.mean(hsv_pixels[:, 2])),
            "lab_l_mean": float(np.mean(lab_pixels[:, 0])),
            "lab_a_mean": float(np.mean(lab_pixels[:, 1])),
            "lab_b_mean": float(np.mean(lab_pixels[:, 2])),
            "red_excess": float(red_excess),
            "blue_excess": float(blue_excess),
            "purple_score": float(purple_score),
            "red_blue_balance": float(red_blue_balance),
        }
        state, reason = classify(row)
        row["state"] = state
        row["state_reason"] = reason
        rows.append(row)

    return pd.DataFrame(rows)


def make_overlay(raw_rgb: np.ndarray, masks: np.ndarray, features: pd.DataFrame, out_path: Path, alpha: float = 0.55) -> None:
    base = normalize_rgb(raw_rgb)
    overlay = base.copy()
    state_by_id = dict(zip(features["mask_id"].astype(int), features["state"]))
    for mask_id, state in state_by_id.items():
        color = STATE_COLORS[state]
        mask = masks == mask_id
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    side_by_side = np.concatenate([base, overlay], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_png(out_path, side_by_side)


def write_summary(features: pd.DataFrame, out_path: Path, image_id: str) -> pd.DataFrame:
    counts = features["state"].value_counts().to_dict()
    total = len(features)
    summary = {
        "image_id": image_id,
        "total_masks": total,
    }
    for state in STATE_COLORS:
        count = int(counts.get(state, 0))
        summary[f"{state}_count"] = count
        summary[f"{state}_fraction"] = count / total if total else 0
    summary["segmented_area_total"] = int(features["area"].sum()) if total else 0
    summary_df = pd.DataFrame([summary])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB cell features and classify live/dead/transitional states.")
    parser.add_argument("--raw-image", type=Path, required=True)
    parser.add_argument("--mask-image", type=Path, required=True)
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("cellpose_pipeline/classification"))
    args = parser.parse_args()

    image_id = args.image_id or args.raw_image.stem
    raw_rgb = to_rgb(tifffile.imread(args.raw_image))
    masks = tifffile.imread(args.mask_image)
    if masks.ndim > 2:
        masks = np.squeeze(masks)
    if masks.shape != raw_rgb.shape[:2]:
        raise ValueError(f"Mask shape {masks.shape} does not match image shape {raw_rgb.shape[:2]}")

    out_dir = args.out_dir
    features = extract_features(raw_rgb, masks, image_id)

    feature_path = out_dir / "features" / f"{image_id}_per_cell_features.csv"
    prediction_path = out_dir / "predictions" / f"{image_id}_per_cell_predictions.csv"
    summary_path = out_dir / "predictions" / f"{image_id}_summary.csv"
    overlay_path = out_dir / "qc" / "label_overlays" / f"{image_id}_state_overlay.png"

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    features.to_csv(prediction_path, index=False)
    summary_df = write_summary(features, summary_path, image_id)
    make_overlay(raw_rgb, masks, features, overlay_path)

    print(f"features={feature_path}")
    print(f"predictions={prediction_path}")
    print(f"summary={summary_path}")
    print(f"overlay={overlay_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
