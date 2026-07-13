#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile


def classify_manual_match(median_rgb: np.ndarray, mask_id: int) -> tuple[str, str]:
    if not mask_id:
        return "segmentation_miss_dead_cell", ""

    rgb_frac = median_rgb / (median_rgb.sum() + 1e-9)
    r_frac, _, b_frac = rgb_frac
    log_b_over_r = float(np.log((median_rgb[2] + 1) / (median_rgb[0] + 1)))
    blue_excess = b_frac - max(r_frac, rgb_frac[1])
    is_blue_cyan = (
        (b_frac >= 0.40 and blue_excess >= 0.055 and log_b_over_r >= 0.25)
        or (median_rgb[2] >= 175 and b_frac >= 0.38 and blue_excess >= 0.035)
    )
    if is_blue_cyan:
        return "matched_dead_mask", "dead"
    return "ambiguous_or_neighbor_mask", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract orange-circled manual dead-cell annotations from an edited TIFF.")
    parser.add_argument("--annotated-image", type=Path, required=True)
    parser.add_argument("--raw-image", type=Path, required=True)
    parser.add_argument("--mask-image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("cellpose_pipeline/classification/labels/circled_dead_cells.csv"))
    args = parser.parse_args()

    annotated = tifffile.imread(args.annotated_image)
    raw = tifffile.imread(args.raw_image)
    masks = tifffile.imread(args.mask_image)

    hsv = cv2.cvtColor(annotated, cv2.COLOR_RGB2HSV)
    orange = (
        (hsv[:, :, 0] >= 8)
        & (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 100)
        & (hsv[:, :, 2] >= 120)
    ).astype("uint8")

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(orange, 8)
    components = []
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        if area >= 20 and w >= 8 and h >= 8:
            components.append((label_id, area, x, y, w, h, centroids[label_id]))

    groups = []
    for component in components:
        _, _, _, _, _, _, centroid = component
        cx, cy = centroid
        for group in groups:
            if abs(group["cx"] - cx) < 30 and abs(group["cy"] - cy) < 30:
                group["items"].append(component)
                group["cx"] = float(np.mean([item[6][0] for item in group["items"]]))
                group["cy"] = float(np.mean([item[6][1] for item in group["items"]]))
                break
        else:
            groups.append({"cx": float(cx), "cy": float(cy), "items": [component]})

    rows = []
    for manual_id, group in enumerate(sorted(groups, key=lambda item: (item["cy"], item["cx"])), 1):
        xs: list[int] = []
        ys: list[int] = []
        for _, _, x, y, w, h, _ in group["items"]:
            xs.extend([x, x + w])
            ys.extend([y, y + h])

        x0 = max(0, min(xs) - 10)
        x1 = min(annotated.shape[1], max(xs) + 10)
        y0 = max(0, min(ys) - 10)
        y1 = min(annotated.shape[0], max(ys) + 10)

        submask = masks[y0:y1, x0:x1]
        mask_ids, counts = np.unique(submask[submask > 0], return_counts=True)
        mask_id = 0 if len(mask_ids) == 0 else int(mask_ids[np.argmax(counts)])

        if mask_id:
            pixels = raw[masks == mask_id].astype(float)
            area = int((masks == mask_id).sum())
        else:
            pixels = raw[y0:y1, x0:x1].reshape(-1, 3).astype(float)
            area = 0

        median_rgb = np.median(pixels, axis=0)
        mean_rgb = np.mean(pixels, axis=0)
        rgb_frac = median_rgb / (median_rgb.sum() + 1e-9)

        match_status, mask_training_label = classify_manual_match(median_rgb, mask_id)
        rows.append(
            {
                "manual_label_id": manual_id,
                "manual_annotation": "dead_cell_circle",
                "match_status": match_status,
                "mask_training_label": mask_training_label,
                "circle_center_x": group["cx"],
                "circle_center_y": group["cy"],
                "mask_id": mask_id,
                "mask_area": area,
                "median_r": median_rgb[0],
                "median_g": median_rgb[1],
                "median_b": median_rgb[2],
                "mean_r": mean_rgb[0],
                "mean_g": mean_rgb[1],
                "mean_b": mean_rgb[2],
                "r_frac": rgb_frac[0],
                "g_frac": rgb_frac[1],
                "b_frac": rgb_frac[2],
                "log_b_over_r": float(np.log((median_rgb[2] + 1) / (median_rgb[0] + 1))),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"orange_components={len(components)}")
    print(f"grouped_circles={len(groups)}")
    print(f"wrote={args.out}")
    if rows:
        df = pd.DataFrame(rows)
        print(
            df[
                [
                    "manual_label_id",
                    "match_status",
                    "mask_training_label",
                    "mask_id",
                    "mask_area",
                    "median_r",
                    "median_g",
                    "median_b",
                    "r_frac",
                    "g_frac",
                    "b_frac",
                    "log_b_over_r",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
