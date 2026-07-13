"""Shared Combined-blue preprocessing primitives used by production and tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    tag: str
    model: str
    signal_transform: str
    diameter: float
    cellprob_threshold: float
    flow_threshold: float = 0.0
    min_size: int = 10
    preprocess: str = "fixed"
    background_sigma: float = 0.0


def blue_signals(raw: np.ndarray) -> dict[str, np.ndarray]:
    if raw.ndim != 3:
        raise ValueError(f"Expected Combined RGB image, got {raw.shape}")
    rgb = raw[..., :3].astype(np.float32, copy=False)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {
        "blue": blue,
        "blue_excess_mean": np.clip(blue - 0.5 * (red + green), 0.0, None),
        "blue_excess_max": np.clip(blue - np.maximum(red, green), 0.0, None),
    }


def calibration_for(payload: dict[str, Any], transform: str) -> tuple[float, float]:
    key = {
        "blue": "combined_blue",
        "blue_excess_mean": "combined_blue_excess_mean",
        "blue_excess_max": "combined_blue_excess_max",
    }[transform]
    block = payload[key]
    low = float(block["recommended_fixed_low"])
    high = float(block["recommended_fixed_high"])
    if high <= low:
        raise ValueError(f"Invalid calibration for {transform}: {low}-{high}")
    return low, high


def fixed_normalize(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((signal.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def background_correct(normalized: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    background = cv2.GaussianBlur(normalized.astype(np.float32), (k, k), sigma)
    corrected = np.clip(normalized - background, 0.0, None)
    high = float(np.percentile(corrected, 99.8))
    return np.clip(corrected / max(high, 1e-6), 0.0, 1.0)


def prepare_signal(signal: np.ndarray, candidate: Candidate, calibration: dict[str, Any]) -> np.ndarray:
    low, high = calibration_for(calibration, candidate.signal_transform)
    normalized = fixed_normalize(signal, low, high)
    if candidate.preprocess == "fixed":
        return normalized.astype(np.float32, copy=False)
    if candidate.preprocess == "background":
        return background_correct(normalized, candidate.background_sigma).astype(np.float32, copy=False)
    raise ValueError(candidate.preprocess)


def object_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int32)
    if labels.size == 0:
        return labels, np.zeros(1, dtype=np.int64), np.zeros((1, 2), dtype=np.float64)
    max_label = int(mask.max())
    areas = np.bincount(mask.ravel(), minlength=max_label + 1)
    yy, xx = np.indices(mask.shape)
    sums_y = np.bincount(mask.ravel(), weights=yy.ravel(), minlength=max_label + 1)
    sums_x = np.bincount(mask.ravel(), weights=xx.ravel(), minlength=max_label + 1)
    centroids = np.zeros((max_label + 1, 2), dtype=np.float64)
    centroids[labels, 0] = sums_y[labels] / np.maximum(areas[labels], 1)
    centroids[labels, 1] = sums_x[labels] / np.maximum(areas[labels], 1)
    return labels, areas, centroids
