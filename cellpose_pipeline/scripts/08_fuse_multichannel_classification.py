#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


PROFILES = ("Combined", "Brightfield", "Dead", "Nuclei")
STATE_COLORS = {
    "live": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "dead": np.array([0.0, 0.25, 1.0], dtype=np.float32),
    "transitional": np.array([0.65, 0.0, 0.85], dtype=np.float32),
    "artifact": np.array([1.0, 0.9, 0.0], dtype=np.float32),
    "uncertain": np.array([0.55, 0.55, 0.55], dtype=np.float32),
}
KEY_RE = re.compile(r"(?:SUM159_AC_)?(?:Exp1_)?(?:BF_|Dead_)?([A-H]\d+_\d+_\d+d\d+h\d+m)")
KEY_DETAIL_RE = re.compile(r"(?P<well>[A-H]\d+)_(?P<site>\d+)_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m")
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


TIMEPOINT_HELPER_PATH = Path(__file__).resolve().parent / "_shared" / "timepoint_selection.py"
TIMEPOINT_HELPER_SPEC = importlib.util.spec_from_file_location(
    "timepoint_selection_local",
    TIMEPOINT_HELPER_PATH,
)
if TIMEPOINT_HELPER_SPEC is None or TIMEPOINT_HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load timepoint selection helpers: {TIMEPOINT_HELPER_PATH}")
TIMEPOINT_HELPER = importlib.util.module_from_spec(TIMEPOINT_HELPER_SPEC)
sys.modules[TIMEPOINT_HELPER_SPEC.name] = TIMEPOINT_HELPER
TIMEPOINT_HELPER_SPEC.loader.exec_module(TIMEPOINT_HELPER)
extract_key_and_timepoint = TIMEPOINT_HELPER.extract_key_and_timepoint
normalize_timepoint = TIMEPOINT_HELPER.normalize_timepoint
select_keys = TIMEPOINT_HELPER.select_keys


@dataclass(frozen=True)
class ProfileRecord:
    profile: str
    key: str
    stem: str
    image_path: Path | None
    mask_path: Path
    run_dir: Path
    source_summary: Path | None = None
    mask_kind: str = "extent"


@dataclass
class LabelStats:
    labels: np.ndarray
    areas: np.ndarray
    centroids: np.ndarray
    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray

    @property
    def count(self) -> int:
        return int(self.labels.size)


@dataclass
class DeadEvidence:
    label: int
    assigned_combined_label: int
    match_mode: str
    score: float
    distance: float
    overlap_pixels: int
    dead_overlap_fraction: float
    combined_overlap_fraction: float
    area: int
    aspect: float
    centroid_y: float
    centroid_x: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    raw_mean: float
    raw_p90: float
    raw_max: float
    bg_median: float
    bg_sigma: float
    mean_delta: float
    p90_delta: float
    snr: float
    keep_signal: bool
    strong_direct: bool = False
    association_relation: str = ""


@dataclass
class DeadEvidenceResult:
    objects: list[DeadEvidence]
    best_by_combined: dict[int, DeadEvidence]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse Combined RGB classification with Brightfield, Dead, and Nuclei "
            "segmentation evidence, then write Combined-based QC overlays."
        )
    )
    parser.add_argument("--run-root", type=Path, help="Root containing a single run or grouped profile run dirs.")
    parser.add_argument("--combined-run", type=Path, help="Run dir containing Combined outputs.")
    parser.add_argument("--brightfield-run", type=Path, help="Run dir containing Brightfield outputs.")
    parser.add_argument("--dead-run", type=Path, help="Run dir containing Dead outputs.")
    parser.add_argument("--nuclei-run", type=Path, help="Run dir containing Nuclei outputs.")
    parser.add_argument(
        "--prefer-nucleus-core-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use conservative *_core_masks.tif for nucleus centroid/count evidence when available.",
    )
    parser.add_argument("--input-root", type=Path, help="Raw image root containing profile subfolders.")
    parser.add_argument("--out-dir", type=Path, help="Output dir. Defaults to <run-root>/classification_fusion.")
    parser.add_argument("--key", action="append", help="Only process this normalized key, e.g. A11_1_00d12h00m.")
    parser.add_argument(
        "--timepoint",
        help=(
            "Only process records at one exact timepoint discovered from the complete input/result roots. "
            "Accepts DDdHHhMMm or d0."
        ),
    )
    parser.add_argument(
        "--field-record",
        type=Path,
        help="Direct per-field JSON record. Bypasses all result-directory discovery.",
    )
    parser.add_argument(
        "--cell-mask-branch",
        choices=("original", "nucleated"),
        default="original",
        help="Select original or nucleated-only Combined/Brightfield masks from --field-record.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of Combined images after key filtering.")
    parser.add_argument("--force", action="store_true", help="Rewrite outputs if they already exist.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--per-key-only",
        action="store_true",
        help="Write per-image outputs only and skip the aggregate cell_count_summary.csv. Use for Slurm array tasks.",
    )
    parser.add_argument(
        "--merge-summaries-only",
        action="store_true",
        help="Only merge per-image summaries under --out-dir/summaries into cell_count_summary.csv.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--bf-match-distance", type=float, default=35.0)
    parser.add_argument("--nuclei-match-distance", type=float, default=35.0)
    parser.add_argument("--dead-match-distance", type=float, default=25.0)
    parser.add_argument("--min-dead-overlap-fraction", type=float, default=0.03)
    parser.add_argument("--min-combined-overlap-fraction", type=float, default=0.002)
    parser.add_argument("--min-countable-area", type=int, default=35)
    parser.add_argument("--unsupported-artifact-max-area", type=int, default=80)
    parser.add_argument("--dead-min-area", type=int, default=20)
    parser.add_argument("--dead-max-area", type=int, default=2500)
    parser.add_argument("--dead-max-aspect", type=float, default=4.0)
    parser.add_argument("--dead-min-mean-delta", type=float, default=0.60)
    parser.add_argument("--dead-min-p90-delta", type=float, default=2.00)
    parser.add_argument("--dead-min-p90-abs", type=float, default=15.0)
    parser.add_argument("--dead-min-snr", type=float, default=3.00)
    parser.add_argument("--dead-bg-margin", type=int, default=14)
    parser.add_argument("--dead-live-rgb-max-area", type=int, default=500)
    parser.add_argument("--dead-live-rgb-min-combined-overlap-fraction", type=float, default=0.60)
    parser.add_argument("--dead-live-rgb-min-p90-delta", type=float, default=20.0)
    parser.add_argument("--dead-live-rgb-min-snr", type=float, default=50.0)
    parser.add_argument("--dead-direct-min-p90-delta", type=float, default=35.0)
    parser.add_argument("--dead-direct-min-snr", type=float, default=90.0)
    parser.add_argument("--dead-uncertain-min-combined-overlap-fraction", type=float, default=0.45)
    parser.add_argument("--dead-uncertain-min-p90-delta", type=float, default=10.0)
    args = parser.parse_args()
    if not args.merge_summaries_only and not args.field_record and not args.run_root and not args.combined_run:
        raise SystemExit("Provide --field-record, --run-root, or --combined-run.")
    if args.merge_summaries_only and args.out_dir is None:
        raise SystemExit("--merge-summaries-only requires --out-dir.")
    return args


def extract_key(value: str | Path) -> str:
    text = Path(value).name
    text = (
        text.replace("_cp_masks.tif", "")
        .replace("_cp_masks.tiff", "")
        .replace("_core_masks.tif", "")
        .replace("_core_masks.tiff", "")
    )
    match = KEY_RE.search(text)
    if not match:
        raise ValueError(f"Cannot extract sample/time key from {value}")
    return match.group(1)


def key_details(key: str) -> dict[str, Any]:
    match = KEY_DETAIL_RE.fullmatch(key)
    if not match:
        return {
            "well": "",
            "site": "",
            "day": "",
            "hour": "",
            "minute": "",
            "elapsed_hours": "",
        }
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    return {
        "well": match.group("well"),
        "site": match.group("site"),
        "day": day,
        "hour": hour,
        "minute": minute,
        "elapsed_hours": day * 24 + hour + minute / 60,
    }


def profile_run_arg(args: argparse.Namespace, profile: str) -> Path | None:
    mapping = {
        "Combined": args.combined_run,
        "Brightfield": args.brightfield_run,
        "Dead": args.dead_run,
        "Nuclei": args.nuclei_run,
    }
    return mapping[profile]


def candidate_run_dirs(args: argparse.Namespace, profile: str) -> list[Path]:
    explicit = profile_run_arg(args, profile)
    if explicit is not None:
        return [explicit]
    if args.run_root is None:
        return []
    return [args.run_root / profile, args.run_root]


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            key = path.resolve()
        except FileNotFoundError:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def profile_image_fallback(input_root: Path | None, profile: str, basename: str) -> Path | None:
    if input_root is None or not basename:
        return None
    candidates = [input_root / profile / basename, input_root / basename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def mask_stem(value: str | Path) -> str:
    name = Path(value).name
    for suffix in ("_cp_masks.tif", "_cp_masks.tiff", "_core_masks.tif", "_core_masks.tiff"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem = Path(name).stem
    if stem.endswith("_cp_masks"):
        return stem[: -len("_cp_masks")]
    if stem.endswith("_core_masks"):
        return stem[: -len("_core_masks")]
    return stem


def stem_matches_profile(profile: str, stem: str) -> bool:
    if profile == "Brightfield":
        return "_BF_" in stem
    if profile == "Dead":
        return "_Dead_" in stem
    if profile == "Nuclei":
        return "Exp1_" in stem and "_BF_" not in stem and "_Dead_" not in stem
    if profile == "Combined":
        return "_BF_" not in stem and "_Dead_" not in stem and "Exp1_" not in stem
    return False


def path_matches_profile(mask_path: Path, run_dir: Path, profile: str) -> bool:
    if run_dir.name == profile or run_dir.parent.name == profile:
        return True
    try:
        parts = mask_path.relative_to(run_dir).parts
    except ValueError:
        parts = mask_path.parts
    if profile in parts:
        return True
    if "segmentations" in parts:
        idx = parts.index("segmentations")
        if idx + 1 < len(parts) and parts[idx + 1] in PROFILES:
            return parts[idx + 1] == profile
    return stem_matches_profile(profile, mask_stem(mask_path))


def mask_fallback(run_dir: Path, profile: str, basename: str, stem: str) -> Path | None:
    names = []
    if basename:
        names.append(basename)
    names.append(f"{stem}_cp_masks.tif")
    names.append(f"{stem}_cp_masks.tiff")
    candidates: list[Path] = []
    for name in names:
        candidates.extend(
            [
                run_dir / "segmentations" / profile / name,
                run_dir / "segmentations" / name,
                run_dir / profile / "segmentations" / profile / name,
                run_dir / profile / "segmentations" / name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def nucleus_core_fallback(run_dir: Path, stem: str) -> Path | None:
    names = (f"{stem}_core_masks.tif", f"{stem}_core_masks.tiff")
    candidate_dirs = (
        run_dir / "nucleus_core_seeds",
        run_dir / "nucleus_core_seeds" / "Nuclei",
        run_dir / "Nuclei" / "nucleus_core_seeds",
        run_dir / "Nuclei" / "nucleus_core_seeds" / "Nuclei",
    )
    for directory in candidate_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def segmentation_summary_files(run_dir: Path) -> list[Path]:
    summaries = list(run_dir.glob("segmentation_summary*.csv"))
    summaries.extend(
        path
        for path in run_dir.rglob("segmentation_summary*.csv")
        if "classification_fusion" not in path.parts
    )
    return sorted(unique_paths(summaries))


def segmentation_mask_files(run_dir: Path, profile: str) -> list[Path]:
    candidates: list[Path] = []
    search_dirs = [
        run_dir / "segmentations" / profile,
        run_dir / "segmentations",
        run_dir / profile / "segmentations" / profile,
        run_dir / profile / "segmentations",
    ]
    for mask_dir in unique_paths(search_dirs):
        if not mask_dir.exists():
            continue
        for mask_path in mask_dir.glob("*_cp_masks.tif"):
            if path_matches_profile(mask_path, run_dir, profile):
                candidates.append(mask_path)
    for mask_path in run_dir.rglob("*_cp_masks.tif"):
        if "classification_fusion" in mask_path.parts:
            continue
        if path_matches_profile(mask_path, run_dir, profile):
            candidates.append(mask_path)
    return sorted(unique_paths(candidates))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_records_for_profile(args: argparse.Namespace, profile: str) -> dict[str, ProfileRecord]:
    records: dict[str, ProfileRecord] = {}
    for run_dir in candidate_run_dirs(args, profile):
        if not run_dir.exists():
            continue
        for summary in segmentation_summary_files(run_dir):
            for row in read_csv_rows(summary):
                if row.get("folder_profile") != profile:
                    continue
                image_value = row.get("image_path", "")
                mask_value = row.get("mask_path", "")
                try:
                    key = extract_key(image_value or mask_value)
                except ValueError:
                    continue
                image_path = resolve_existing_path(image_value)
                if image_path is None and image_value:
                    image_path = profile_image_fallback(args.input_root, profile, Path(image_value).name)
                mask_path = resolve_existing_path(mask_value)
                stem = mask_stem(mask_value) if mask_value else ""
                if not stem and image_value:
                    stem = Path(image_value).stem
                if mask_path is None:
                    mask_path = mask_fallback(summary.parent, profile, Path(mask_value).name, stem)
                if mask_path is None:
                    mask_path = mask_fallback(run_dir, profile, Path(mask_value).name, stem)
                mask_kind = "extent"
                if profile == "Nuclei" and args.prefer_nucleus_core_masks:
                    core_value = row.get("nucleus_core_mask_path", "")
                    core_path = resolve_existing_path(core_value)
                    if core_path is None:
                        core_path = nucleus_core_fallback(summary.parent, stem)
                    if core_path is None:
                        core_path = nucleus_core_fallback(run_dir, stem)
                    if core_path is not None:
                        mask_path = core_path
                        mask_kind = "core"
                if mask_path is None:
                    continue
                records.setdefault(
                    key,
                    ProfileRecord(
                        profile=profile,
                        key=key,
                        stem=stem,
                        image_path=image_path,
                        mask_path=mask_path,
                        run_dir=summary.parent,
                        source_summary=summary,
                        mask_kind=mask_kind,
                    ),
                )

        for mask_path in segmentation_mask_files(run_dir, profile):
            try:
                key = extract_key(mask_path)
            except ValueError:
                continue
            stem = mask_stem(mask_path)
            image_path = find_raw_for_mask(args.input_root, profile, stem)
            records.setdefault(
                key,
                ProfileRecord(
                    profile=profile,
                    key=key,
                    stem=stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    run_dir=run_dir,
                    mask_kind="extent",
                ),
            )
        if profile == "Nuclei" and args.prefer_nucleus_core_masks:
            for core_path in sorted(run_dir.rglob("*_core_masks.tif")):
                if "classification_fusion" in core_path.parts:
                    continue
                try:
                    key = extract_key(core_path)
                except ValueError:
                    continue
                stem = mask_stem(core_path)
                records[key] = ProfileRecord(
                    profile=profile,
                    key=key,
                    stem=stem,
                    image_path=find_raw_for_mask(args.input_root, profile, stem),
                    mask_path=core_path,
                    run_dir=run_dir,
                    mask_kind="core",
                )
    return records


def direct_records_from_json(args: argparse.Namespace) -> dict[str, dict[str, ProfileRecord]]:
    if args.field_record is None:
        raise ValueError("--field-record is required")
    payload = json.loads(args.field_record.read_text())
    key = str(payload.get("key", ""))
    if not KEY_DETAIL_RE.fullmatch(key):
        raise ValueError(f"Invalid key in field record {args.field_record}: {key!r}")
    if args.key and set(args.key) != {key}:
        raise ValueError(f"Requested key(s) {args.key} do not match field record key {key}")
    profile_payloads = payload.get("profiles")
    if not isinstance(profile_payloads, dict):
        raise ValueError(f"Missing profiles object in {args.field_record}")

    records: dict[str, dict[str, ProfileRecord]] = {profile: {} for profile in PROFILES}
    for profile in PROFILES:
        values = profile_payloads.get(profile)
        if not isinstance(values, dict):
            raise ValueError(f"Missing profile {profile} in {args.field_record}")
        if profile in {"Combined", "Brightfield"}:
            mask_field = "nucleated_mask" if args.cell_mask_branch == "nucleated" else "original_mask"
            mask_kind = args.cell_mask_branch
        elif profile == "Dead":
            mask_field = "mask"
            mask_kind = "extent"
        else:
            mask_field = "core_mask" if args.prefer_nucleus_core_masks else "extent_mask"
            mask_kind = "core" if args.prefer_nucleus_core_masks else "extent"
        mask_path = Path(str(values.get(mask_field, "")))
        raw_path = Path(str(values.get("raw", "")))
        if not mask_path.is_file():
            raise FileNotFoundError(f"{profile} {mask_field} is missing for {key}: {mask_path}")
        if not raw_path.is_file():
            raise FileNotFoundError(f"{profile} raw image is missing for {key}: {raw_path}")
        stem = str(values.get("stem") or mask_stem(mask_path))
        records[profile][key] = ProfileRecord(
            profile=profile,
            key=key,
            stem=stem,
            image_path=raw_path,
            mask_path=mask_path,
            run_dir=mask_path.parent.parent,
            mask_kind=mask_kind,
        )
    return records


def find_raw_for_mask(input_root: Path | None, profile: str, stem: str) -> Path | None:
    if input_root is None:
        return None
    for suffix in IMAGE_SUFFIXES:
        candidate = input_root / profile / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_mask(path: Path) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got shape {mask.shape}")
    return mask.astype(np.int32, copy=False)


def read_raw_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(path)
    with Image.open(path) as image:
        return np.asarray(image)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, (1, 99))
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255, 0, 255).astype(np.uint8)
    tmp_path = path.with_name(f".{path.name}.tmp")
    Image.fromarray(arr).save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def label_stats(mask: np.ndarray) -> LabelStats:
    max_label = int(mask.max()) if mask.size else 0
    shape = max_label + 1
    if max_label == 0:
        zeros = np.zeros(shape, dtype=np.float64)
        ints = np.zeros(shape, dtype=np.int32)
        return LabelStats(np.zeros(0, dtype=np.int32), zeros, np.zeros((shape, 2)), ints, ints, ints, ints)

    ys, xs = np.nonzero(mask)
    labs = mask[ys, xs].astype(np.int64, copy=False)
    areas = np.bincount(labs, minlength=shape).astype(np.float64)
    labels = np.flatnonzero(areas > 0).astype(np.int32)
    labels = labels[labels > 0]
    sum_y = np.bincount(labs, weights=ys, minlength=shape)
    sum_x = np.bincount(labs, weights=xs, minlength=shape)
    centroids = np.zeros((shape, 2), dtype=np.float64)
    nonzero = areas > 0
    centroids[nonzero, 0] = sum_y[nonzero] / areas[nonzero]
    centroids[nonzero, 1] = sum_x[nonzero] / areas[nonzero]

    h, w = mask.shape
    y0 = np.full(shape, h, dtype=np.int32)
    x0 = np.full(shape, w, dtype=np.int32)
    y1 = np.zeros(shape, dtype=np.int32)
    x1 = np.zeros(shape, dtype=np.int32)
    np.minimum.at(y0, labs, ys)
    np.minimum.at(x0, labs, xs)
    np.maximum.at(y1, labs, ys + 1)
    np.maximum.at(x1, labs, xs + 1)
    return LabelStats(labels, areas, centroids, y0, x0, y1, x1)


def nearest_label_matches(
    source_labels: np.ndarray,
    source_centroids: np.ndarray,
    target_stats: LabelStats | None,
    max_distance: float,
) -> dict[int, tuple[int, float]]:
    matches: dict[int, tuple[int, float]] = {}
    if target_stats is None or target_stats.count == 0 or source_labels.size == 0:
        for label in source_labels:
            matches[int(label)] = (0, float("inf"))
        return matches
    target_labels = target_stats.labels.astype(np.int32, copy=False)
    target_centers = target_stats.centroids[target_labels]
    max_dist2 = max_distance * max_distance
    chunk_size = 256
    for start in range(0, source_labels.size, chunk_size):
        end = min(start + chunk_size, source_labels.size)
        src = source_centroids[start:end]
        diff = src[:, None, :] - target_centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        nearest_dist2 = dist2[np.arange(end - start), nearest_idx]
        for idx, label in enumerate(source_labels[start:end]):
            if nearest_dist2[idx] <= max_dist2:
                matches[int(label)] = (int(target_labels[nearest_idx[idx]]), math.sqrt(float(nearest_dist2[idx])))
            else:
                matches[int(label)] = (0, math.sqrt(float(nearest_dist2[idx])))
    return matches


def nuclei_counts_inside_combined(combined_mask: np.ndarray, nuclei_stats: LabelStats | None) -> dict[int, int]:
    counts: dict[int, int] = {}
    if nuclei_stats is None or nuclei_stats.count == 0:
        return counts
    h, w = combined_mask.shape
    for label in nuclei_stats.labels:
        y, x = nuclei_stats.centroids[int(label)]
        yy = int(round(float(y)))
        xx = int(round(float(x)))
        if 0 <= yy < h and 0 <= xx < w:
            combined_label = int(combined_mask[yy, xx])
            if combined_label > 0:
                counts[combined_label] = counts.get(combined_label, 0) + 1
    return counts


def dead_object_nucleus_evidence(
    result: DeadEvidenceResult,
    dead_mask: np.ndarray | None,
    nuclei_mask: np.ndarray | None,
    nuclei_stats: LabelStats | None,
    combined_nuclei_counts: dict[int, int],
    nearby_distance: float,
) -> dict[int, dict[str, Any]]:
    """Measure nucleus evidence without changing any upstream segmentation."""
    output: dict[int, dict[str, Any]] = {}
    has_nuclei = nuclei_stats is not None and nuclei_stats.count > 0
    for evidence in result.objects:
        combined_count = int(combined_nuclei_counts.get(evidence.assigned_combined_label, 0))
        nuclei_inside = 0
        nuclei_nearby = 0
        nearest_distance = float("inf")
        if has_nuclei:
            centers = nuclei_stats.centroids[nuclei_stats.labels]
            distances = np.sqrt(
                np.sum(
                    (centers - np.array([evidence.centroid_y, evidence.centroid_x], dtype=np.float64)) ** 2,
                    axis=1,
                )
            )
            if distances.size:
                nearest_distance = float(np.min(distances))
                nuclei_nearby = int(np.count_nonzero(distances <= nearby_distance))
            if dead_mask is not None:
                height, width = dead_mask.shape
                for nucleus_label in nuclei_stats.labels:
                    y, x = nuclei_stats.centroids[int(nucleus_label)]
                    yy = int(round(float(y)))
                    xx = int(round(float(x)))
                    if 0 <= yy < height and 0 <= xx < width and int(dead_mask[yy, xx]) == evidence.label:
                        nuclei_inside += 1

        nucleus_overlap_fraction = 0.0
        if dead_mask is not None and nuclei_mask is not None and evidence.area > 0:
            object_pixels = dead_mask == evidence.label
            nucleus_overlap_fraction = float(np.count_nonzero(object_pixels & (nuclei_mask > 0))) / float(
                evidence.area
            )

        if combined_count >= 2:
            interpretation = "probable_live_dead_overlap_multi_nucleus"
        elif combined_count == 1:
            interpretation = "live_with_death_signal_single_nucleus"
        else:
            interpretation = "adjacent_or_overlapping_dead_nucleus_unresolved"
        output[evidence.label] = {
            "combined_nuclei_count": combined_count,
            "dead_object_nuclei_inside_count": nuclei_inside,
            "dead_object_nearby_nuclei_count": nuclei_nearby,
            "nearest_nucleus_distance": nearest_distance,
            "dead_nucleus_overlap_fraction": nucleus_overlap_fraction,
            "nucleus_supported_relation": interpretation,
        }
    return output


def safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(math.log((numerator + 1.0) / (denominator + 1.0)))


def rgb_rule_classify(row: dict[str, Any]) -> tuple[str, str]:
    area = float(row["area"])
    sat = float(row["hsv_s_mean"])
    r_frac = float(row["r_frac"])
    g_frac = float(row["g_frac"])
    b_frac = float(row["b_frac"])
    red_excess = float(row["red_excess"])
    blue_excess = float(row["blue_excess"])
    purple_score = float(row["purple_score"])
    red_blue_balance = float(row["red_blue_balance"])
    log_b_over_r = float(row["log_b_over_r"])

    if area < 35:
        return "artifact", "area_lt_35"
    if sat < 0.08:
        return "uncertain", "low_saturation"
    if (b_frac >= 0.40 and blue_excess >= 0.055 and log_b_over_r >= 0.25) or (
        float(row["median_b"]) >= 175 and b_frac >= 0.38 and blue_excess >= 0.035
    ):
        return "dead", "blue_cyan_rule"
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


def combined_rgb_features(raw_rgb: np.ndarray, mask: np.ndarray, stats: LabelStats) -> dict[int, dict[str, Any]]:
    raw_float = raw_rgb.astype(np.float32, copy=False)
    rows: dict[int, dict[str, Any]] = {}
    for label in stats.labels:
        idx = int(label)
        y0, y1 = int(stats.y0[idx]), int(stats.y1[idx])
        x0, x1 = int(stats.x0[idx]), int(stats.x1[idx])
        obj = mask[y0:y1, x0:x1] == idx
        pixels = raw_float[y0:y1, x0:x1][obj]
        if pixels.size == 0:
            continue
        median_rgb = np.median(pixels, axis=0)
        mean_rgb = np.mean(pixels, axis=0)
        maxc = np.max(pixels, axis=1)
        minc = np.min(pixels, axis=1)
        sat = np.zeros_like(maxc, dtype=np.float32)
        nonzero = maxc > 0
        sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
        rgb_sum = float(median_rgb.sum()) + 1e-9
        r_frac, g_frac, b_frac = (median_rgb / rgb_sum).tolist()
        red_excess = float(r_frac - max(g_frac, b_frac))
        blue_excess = float(b_frac - max(r_frac, g_frac))
        purple_score = float(min(r_frac, b_frac) - g_frac)
        red_blue_balance = float(1.0 - abs(r_frac - b_frac))
        row: dict[str, Any] = {
            "combined_mask_id": idx,
            "area": int(stats.areas[idx]),
            "centroid_y": float(stats.centroids[idx, 0]),
            "centroid_x": float(stats.centroids[idx, 1]),
            "bbox_y0": y0,
            "bbox_x0": x0,
            "bbox_y1": y1,
            "bbox_x1": x1,
            "median_r": float(median_rgb[0]),
            "median_g": float(median_rgb[1]),
            "median_b": float(median_rgb[2]),
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
            "r_frac": float(r_frac),
            "g_frac": float(g_frac),
            "b_frac": float(b_frac),
            "log_b_over_r": safe_log_ratio(float(median_rgb[2]), float(median_rgb[0])),
            "log_r_over_b": safe_log_ratio(float(median_rgb[0]), float(median_rgb[2])),
            "hsv_s_mean": float(np.mean(sat)),
            "red_excess": red_excess,
            "blue_excess": blue_excess,
            "purple_score": purple_score,
            "red_blue_balance": red_blue_balance,
        }
        state, reason = rgb_rule_classify(row)
        row["rgb_state"] = state
        row["rgb_reason"] = reason
        rows[idx] = row
    return rows


def scalar_raw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3:
        return np.mean(image[:, :, :3].astype(np.float32, copy=False), axis=2)
    raise ValueError(f"Expected 2D or RGB image, got shape {image.shape}")


def dead_signal_keep(record: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        record["area"] >= args.dead_min_area
        and record["area"] <= args.dead_max_area
        and record["aspect"] <= args.dead_max_aspect
        and record["mean_delta"] >= args.dead_min_mean_delta
        and record["p90_delta"] >= args.dead_min_p90_delta
        and record["raw_p90"] >= args.dead_min_p90_abs
        and record["snr"] >= args.dead_min_snr
    )


def compute_dead_evidence(
    dead_raw: np.ndarray | None,
    dead_mask: np.ndarray | None,
    combined_mask: np.ndarray,
    combined_stats: LabelStats,
    args: argparse.Namespace,
) -> DeadEvidenceResult:
    if dead_raw is None or dead_mask is None or int(dead_mask.max()) == 0:
        return DeadEvidenceResult(objects=[], best_by_combined={})
    if dead_mask.shape != combined_mask.shape or dead_raw.shape != combined_mask.shape:
        raise ValueError(
            f"Dead image/mask shape mismatch: raw={dead_raw.shape} dead_mask={dead_mask.shape} combined={combined_mask.shape}"
        )

    stats = label_stats(dead_mask)
    dead_centers = stats.centroids[stats.labels] if stats.count else np.zeros((0, 2))
    nearest_combined = nearest_label_matches(
        stats.labels,
        dead_centers,
        combined_stats,
        args.dead_match_distance,
    )
    objects: list[DeadEvidence] = []
    best_by_combined: dict[int, DeadEvidence] = {}
    h, w = combined_mask.shape
    for label in stats.labels:
        idx = int(label)
        y0, y1 = int(stats.y0[idx]), int(stats.y1[idx])
        x0, x1 = int(stats.x0[idx]), int(stats.x1[idx])
        obj = dead_mask[y0:y1, x0:x1] == idx
        area = int(stats.areas[idx])
        if area == 0:
            continue
        raw_obj = dead_raw[y0:y1, x0:x1][obj]
        margin = int(args.dead_bg_margin)
        yy0, yy1 = max(0, y0 - margin), min(h, y1 + margin)
        xx0, xx1 = max(0, x0 - margin), min(w, x1 + margin)
        window_labels = dead_mask[yy0:yy1, xx0:xx1]
        bg_values = dead_raw[yy0:yy1, xx0:xx1][window_labels == 0]
        if bg_values.size < 25:
            bg_values = dead_raw[yy0:yy1, xx0:xx1][window_labels != idx]
        if bg_values.size < 25:
            bg_values = dead_raw.ravel()
        bg_median = float(np.median(bg_values))
        bg_mad = float(np.median(np.abs(bg_values - bg_median)))
        bg_sigma = max(bg_mad * 1.4826, 1e-6)
        raw_mean = float(np.mean(raw_obj))
        raw_p90 = float(np.percentile(raw_obj, 90))
        raw_max = float(np.max(raw_obj))
        bbox_h = max(1, y1 - y0)
        bbox_w = max(1, x1 - x0)
        aspect = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))
        base_record = {
            "area": float(area),
            "aspect": aspect,
            "raw_p90": raw_p90,
            "mean_delta": raw_mean - bg_median,
            "p90_delta": raw_p90 - bg_median,
            "snr": (raw_p90 - bg_median) / bg_sigma,
        }
        keep_signal = dead_signal_keep(base_record, args)

        overlap_labels = combined_mask[y0:y1, x0:x1][obj]
        overlap_labels = overlap_labels[overlap_labels > 0]
        overlap_label = 0
        overlap_pixels = 0
        if overlap_labels.size:
            counts = np.bincount(overlap_labels.astype(np.int64, copy=False))
            overlap_label = int(np.argmax(counts))
            overlap_pixels = int(counts[overlap_label])
        overlap_dead_fraction = overlap_pixels / area if area else 0.0
        overlap_combined_fraction = (
            overlap_pixels / float(combined_stats.areas[overlap_label]) if overlap_label > 0 else 0.0
        )
        nearest_label, nearest_dist = nearest_combined.get(idx, (0, float("inf")))

        assigned_label = 0
        match_mode = "unmatched"
        distance = nearest_dist
        if overlap_label > 0 and (
            overlap_dead_fraction >= args.min_dead_overlap_fraction
            or overlap_combined_fraction >= args.min_combined_overlap_fraction
        ):
            assigned_label = overlap_label
            match_mode = "overlap"
            if nearest_label == overlap_label:
                distance = nearest_dist
            else:
                cy, cx = stats.centroids[idx]
                ty, tx = combined_stats.centroids[overlap_label]
                distance = math.hypot(float(cy - ty), float(cx - tx))
        elif nearest_label > 0 and nearest_dist <= args.dead_match_distance:
            assigned_label = nearest_label
            match_mode = "centroid"
        score = (
            float(base_record["snr"])
            + 0.25 * float(base_record["p90_delta"])
            + 10.0 * overlap_dead_fraction
            + 5.0 * overlap_combined_fraction
            - 0.02 * min(distance, 100.0)
        )
        evidence = DeadEvidence(
            label=idx,
            assigned_combined_label=assigned_label,
            match_mode=match_mode,
            score=score,
            distance=float(distance),
            overlap_pixels=overlap_pixels,
            dead_overlap_fraction=float(overlap_dead_fraction),
            combined_overlap_fraction=float(overlap_combined_fraction),
            area=area,
            aspect=aspect,
            centroid_y=float(stats.centroids[idx, 0]),
            centroid_x=float(stats.centroids[idx, 1]),
            bbox_y0=y0,
            bbox_x0=x0,
            bbox_y1=y1,
            bbox_x1=x1,
            raw_mean=raw_mean,
            raw_p90=raw_p90,
            raw_max=raw_max,
            bg_median=bg_median,
            bg_sigma=bg_sigma,
            mean_delta=raw_mean - bg_median,
            p90_delta=raw_p90 - bg_median,
            snr=float(base_record["snr"]),
            keep_signal=keep_signal,
            strong_direct=(
                float(base_record["p90_delta"]) >= args.dead_direct_min_p90_delta
                and float(base_record["snr"]) >= args.dead_direct_min_snr
            ),
        )
        objects.append(evidence)
        if keep_signal and assigned_label > 0:
            previous = best_by_combined.get(assigned_label)
            if previous is None or evidence.score > previous.score:
                best_by_combined[assigned_label] = evidence
    return DeadEvidenceResult(objects=objects, best_by_combined=best_by_combined)


def assign_dead_object_relations(
    result: DeadEvidenceResult,
    combined_features: dict[int, dict[str, Any]],
    local_rgb_features: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[int, DeadEvidence]:
    """Classify every Dead mask without collapsing multiple objects onto one cell."""
    retained_by_combined: dict[int, list[DeadEvidence]] = {}
    for evidence in result.objects:
        confirmed = dead_object_is_confirmed(
            evidence,
            local_rgb_features.get(evidence.label, {}),
            args,
        )
        if not evidence.keep_signal:
            evidence.association_relation = "rejected_signal"
        elif evidence.assigned_combined_label <= 0:
            evidence.association_relation = "dead_only" if confirmed else "unassigned_candidate"
        else:
            retained_by_combined.setdefault(evidence.assigned_combined_label, []).append(evidence)

    best_by_combined: dict[int, DeadEvidence] = {}
    for combined_label, candidates in retained_by_combined.items():
        cell = combined_features.get(combined_label, {})
        rgb_state = str(cell.get("rgb_state", ""))
        same_cell: list[DeadEvidence] = []
        for evidence in candidates:
            confirmed = dead_object_is_confirmed(
                evidence,
                local_rgb_features.get(evidence.label, {}),
                args,
            )
            if not dead_object_has_same_cell_support(evidence, cell, args):
                evidence.association_relation = "adjacent_dead" if confirmed else "adjacent_candidate"
            else:
                evidence.association_relation = "same_cell"
                same_cell.append(evidence)

        if same_cell:
            same_cell.sort(key=lambda item: item.score, reverse=True)
            primary = same_cell[0]
            best_by_combined[combined_label] = primary
            for evidence in same_cell[1:]:
                if dead_object_is_confirmed(
                    evidence,
                    local_rgb_features.get(evidence.label, {}),
                    args,
                ):
                    evidence.association_relation = "merged_multiple_objects"
    result.best_by_combined = best_by_combined
    return best_by_combined


def dead_object_has_same_cell_support(
    evidence: DeadEvidence,
    cell: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Require cell-scale spatial support before a Dead object can change cell state.

    A confirmed Dead object remains independently countable when this returns
    False; only its transfer onto the spatially overlapping Combined cell is
    rejected.  This prevents a small shrunken object beside a large attached
    cell from turning that whole cell dead while retaining compact death events.
    """
    rgb_state = str(cell.get("rgb_state", ""))
    if rgb_state == "dead":
        return True
    if rgb_state == "live":
        return False
    if rgb_state not in {"uncertain", "transitional", "artifact"}:
        return False

    whole_cell_support = (
        evidence.combined_overlap_fraction
        >= args.dead_uncertain_min_combined_overlap_fraction
    )
    compact_overlap_min = min(
        args.dead_uncertain_min_combined_overlap_fraction,
        0.5 * args.dead_live_rgb_min_combined_overlap_fraction,
    )
    compact_cell_support = (
        int(cell.get("area", 0)) <= args.dead_live_rgb_max_area
        and evidence.combined_overlap_fraction >= compact_overlap_min
    )
    return whole_cell_support or compact_cell_support


def dead_object_is_confirmed(
    evidence: DeadEvidence,
    local_rgb: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not evidence.keep_signal:
        return False
    if evidence.strong_direct:
        return True
    return (
        str(local_rgb.get("rgb_state", "")) == "dead"
        and evidence.p90_delta >= args.dead_direct_min_p90_delta
        and evidence.snr >= args.dead_live_rgb_min_snr
    )


def apply_nucleus_aware_overlap_relations(
    result: DeadEvidenceResult,
    nucleus_evidence: dict[int, dict[str, Any]],
) -> None:
    """Refine confirmed non-cell-level objects while preserving their count."""
    relation_by_interpretation = {
        "probable_live_dead_overlap_multi_nucleus": "overlapping_live_dead_multi_nucleus",
        "live_with_death_signal_single_nucleus": "live_with_death_signal",
        "adjacent_or_overlapping_dead_nucleus_unresolved": "adjacent_or_overlapping_dead_uncertain",
    }
    for evidence in result.objects:
        if evidence.association_relation != "adjacent_dead":
            continue
        interpretation = str(
            nucleus_evidence.get(evidence.label, {}).get("nucleus_supported_relation", "")
        )
        evidence.association_relation = relation_by_interpretation.get(
            interpretation,
            "adjacent_or_overlapping_dead_uncertain",
        )


def dead_object_rows(
    image_id: str,
    key: str,
    result: DeadEvidenceResult,
    local_rgb_features: dict[int, dict[str, Any]],
    nucleus_evidence: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in result.objects:
        local_rgb = local_rgb_features.get(evidence.label, {})
        confirmed = dead_object_is_confirmed(evidence, local_rgb, args)
        supplemental = confirmed and evidence.association_relation in {
            "dead_only",
            "adjacent_dead",
            "overlapping_live_dead_multi_nucleus",
            "live_with_death_signal",
            "adjacent_or_overlapping_dead_uncertain",
            "merged_multiple_objects",
        }
        nucleus_row = nucleus_evidence.get(evidence.label, {})
        rows.append(
            {
                "image_id": image_id,
                "key": key,
                **key_details(key),
                "dead_mask_id": evidence.label,
                "source": "dead_segmentation_mask",
                "area": evidence.area,
                "aspect": evidence.aspect,
                "centroid_y": evidence.centroid_y,
                "centroid_x": evidence.centroid_x,
                "bbox_y0": evidence.bbox_y0,
                "bbox_x0": evidence.bbox_x0,
                "bbox_y1": evidence.bbox_y1,
                "bbox_x1": evidence.bbox_x1,
                "assigned_combined_mask_id": evidence.assigned_combined_label,
                "match_mode": evidence.match_mode,
                "association_relation": evidence.association_relation,
                "distance": evidence.distance,
                "overlap_pixels": evidence.overlap_pixels,
                "dead_overlap_fraction": evidence.dead_overlap_fraction,
                "combined_overlap_fraction": evidence.combined_overlap_fraction,
                "raw_mean": evidence.raw_mean,
                "raw_p90": evidence.raw_p90,
                "raw_max": evidence.raw_max,
                "bg_median": evidence.bg_median,
                "bg_sigma": evidence.bg_sigma,
                "mean_delta": evidence.mean_delta,
                "p90_delta": evidence.p90_delta,
                "snr": evidence.snr,
                "keep_signal": evidence.keep_signal,
                "strong_direct": evidence.strong_direct,
                "dual_channel_confirmed": (
                    confirmed and not evidence.strong_direct
                ),
                "confirmed_dead_object": confirmed,
                "supplemental_dead_object": supplemental,
                "local_rgb_state": local_rgb.get("rgb_state", ""),
                "local_rgb_reason": local_rgb.get("rgb_reason", ""),
                "local_median_r": local_rgb.get("median_r", ""),
                "local_median_g": local_rgb.get("median_g", ""),
                "local_median_b": local_rgb.get("median_b", ""),
                "local_r_frac": local_rgb.get("r_frac", ""),
                "local_g_frac": local_rgb.get("g_frac", ""),
                "local_b_frac": local_rgb.get("b_frac", ""),
                "local_blue_excess": local_rgb.get("blue_excess", ""),
                "local_red_excess": local_rgb.get("red_excess", ""),
                "combined_nuclei_count": nucleus_row.get("combined_nuclei_count", 0),
                "dead_object_nuclei_inside_count": nucleus_row.get("dead_object_nuclei_inside_count", 0),
                "dead_object_nearby_nuclei_count": nucleus_row.get("dead_object_nearby_nuclei_count", 0),
                "nearest_nucleus_distance": nucleus_row.get("nearest_nucleus_distance", ""),
                "dead_nucleus_overlap_fraction": nucleus_row.get("dead_nucleus_overlap_fraction", 0.0),
                "nucleus_supported_relation": nucleus_row.get("nucleus_supported_relation", ""),
            }
        )
    return rows


def should_apply_dead_override(
    row: dict[str, Any],
    dead: DeadEvidence,
    args: argparse.Namespace,
) -> tuple[bool, str, str]:
    rgb_state = str(row["rgb_state"])
    area = int(row["area"])
    if rgb_state == "dead":
        return True, "rgb_dead_with_dead_channel_support", "high"
    if (
        dead.p90_delta >= args.dead_direct_min_p90_delta
        and dead.snr >= args.dead_direct_min_snr
    ):
        return True, "strong_dead_channel_override_rgb_context", "high"
    if rgb_state == "live":
        if (
            area <= args.dead_live_rgb_max_area
            and dead.combined_overlap_fraction >= args.dead_live_rgb_min_combined_overlap_fraction
            and dead.p90_delta >= args.dead_live_rgb_min_p90_delta
            and dead.snr >= args.dead_live_rgb_min_snr
        ):
            return True, "compact_live_rgb_with_strong_dead_channel_support", "medium"
        return False, "dead_channel_rejected_live_rgb_context", "high"
    if rgb_state in {"uncertain", "transitional", "artifact"}:
        if (
            dead.combined_overlap_fraction >= args.dead_uncertain_min_combined_overlap_fraction
            and dead.p90_delta >= args.dead_uncertain_min_p90_delta
        ):
            return True, "dead_channel_override_context_supported", "medium"
        return False, "dead_channel_rejected_insufficient_cell_support", "medium"
    return False, "dead_channel_rejected_unknown_rgb_context", "medium"


def final_state_for(
    row: dict[str, Any],
    bf_supported: bool,
    nuclei_supported: bool,
    nuclei_inside_count: int,
    dead: DeadEvidence | None,
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    area = int(row["area"])
    rejected_dead: tuple[str, str] | None = None
    if dead is not None:
        keep_dead, reason, confidence = should_apply_dead_override(row, dead, args)
        if keep_dead:
            return "dead", reason, confidence
        if row["rgb_state"] == "dead":
            return "uncertain", reason, confidence
        rejected_dead = (reason, confidence)
    if area < args.min_countable_area:
        return "artifact", "area_below_countable_min", "high"
    if row["rgb_state"] == "artifact" and not bf_supported and not nuclei_supported:
        return "artifact", "rgb_artifact_without_bf_or_nuclei_support", "medium"
    if (
        row["rgb_state"] == "uncertain"
        and not bf_supported
        and not nuclei_supported
        and nuclei_inside_count == 0
        and area <= args.unsupported_artifact_max_area
    ):
        return "artifact", "small_uncertain_without_bf_or_nuclei_support", "medium"
    if row["rgb_state"] == "live" or bf_supported or nuclei_supported or nuclei_inside_count > 0:
        if rejected_dead is not None:
            return "live", rejected_dead[0], rejected_dead[1]
        return "live", "non_dead_countable_cell", "high" if row["rgb_state"] == "live" else "medium"
    return "live", "non_dead_combined_anchor_default_live", "low"


def empty_channel_fields(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mask_id": 0,
        f"{prefix}_distance": "",
        f"{prefix}_supported": False,
    }


def format_float(value: Any, digits: int = 6) -> Any:
    if value == "" or value is None:
        return ""
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return value


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def write_label_tiff(path: Path, data: np.ndarray, axes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tifffile.imwrite(
        temporary,
        data.astype(np.uint32, copy=False),
        compression="zlib",
        metadata={"axes": axes},
    )
    os.replace(temporary, path)


def build_cell_dead_overlap_masks(
    combined_mask: np.ndarray,
    dead_mask: np.ndarray | None,
    prediction_rows: list[dict[str, Any]],
    dead_object_feature_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_mask = np.zeros_like(combined_mask, dtype=np.uint32)
    max_combined = int(combined_mask.max()) if combined_mask.size else 0
    retained_cells = np.zeros(max_combined + 1, dtype=bool)
    for row in prediction_rows:
        label = int(row["mask_id"])
        if 0 < label <= max_combined and str(row["state"]) != "artifact":
            retained_cells[label] = True
    keep_cell_pixels = retained_cells[combined_mask]
    cell_mask[keep_cell_pixels] = combined_mask[keep_cell_pixels].astype(np.uint32, copy=False)

    confirmed_dead_mask = np.zeros_like(combined_mask, dtype=np.uint32)
    if dead_mask is not None and dead_mask.size:
        max_dead = int(dead_mask.max())
        confirmed_labels = np.zeros(max_dead + 1, dtype=bool)
        for row in dead_object_feature_rows:
            label = int(row["dead_mask_id"])
            if 0 < label <= max_dead and bool(row["confirmed_dead_object"]):
                confirmed_labels[label] = True
        keep_dead_pixels = confirmed_labels[dead_mask]
        confirmed_dead_mask[keep_dead_pixels] = dead_mask[keep_dead_pixels].astype(np.uint32, copy=False)

    overlap_mask = np.stack([cell_mask, confirmed_dead_mask], axis=0)
    return cell_mask, confirmed_dead_mask, overlap_mask


def label_boundaries(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & (mask > 0)


def per_key_outputs_complete(out_dir: Path, combined: ProfileRecord) -> bool:
    expected = (
        out_dir / "features" / f"{combined.stem}_per_cell_fusion_features.csv",
        out_dir / "predictions" / f"{combined.stem}_per_cell_predictions.csv",
        out_dir / "dead_objects" / f"{combined.stem}_dead_object_features.csv",
        out_dir / "summaries" / f"{combined.stem}_summary.csv",
        out_dir / "masks" / "cell_state" / f"{combined.stem}_cell_state_masks.tif",
        out_dir / "masks" / "confirmed_dead" / f"{combined.stem}_confirmed_dead_masks.tif",
        out_dir / "overlap_masks" / f"{combined.stem}_cell_dead_overlap_masks.tif",
        out_dir / "qc" / "label_overlays" / f"{combined.stem}_state_overlay.png",
        out_dir / "qc" / "dead_object_overlays" / f"{combined.stem}_dead_object_overlay.png",
        out_dir / "qc" / "overlap_state_overlays" / f"{combined.stem}_overlap_state_overlay.png",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def make_classification_overlay(
    raw_rgb: np.ndarray,
    masks: np.ndarray,
    rows: list[dict[str, Any]],
    out_path: Path,
    alpha: float,
) -> None:
    base = normalize_rgb(raw_rgb)
    overlay = base.copy()
    max_label = int(masks.max()) if masks.size else 0
    if max_label <= 0:
        side_by_side = np.concatenate([base, overlay], axis=1)
        save_rgb_png(out_path, side_by_side)
        return
    label_colors = np.zeros((max_label + 1, 3), dtype=np.float32)
    has_color = np.zeros(max_label + 1, dtype=bool)
    for row in rows:
        mask_id = int(row["mask_id"])
        if mask_id <= 0 or mask_id > max_label:
            continue
        state = str(row["state"])
        label_colors[mask_id] = STATE_COLORS.get(state, STATE_COLORS["uncertain"])
        has_color[mask_id] = True
    colored = has_color[masks]
    overlay[colored] = (1.0 - alpha) * overlay[colored] + alpha * label_colors[masks[colored]]
    side_by_side = np.concatenate([base, overlay], axis=1)
    save_rgb_png(out_path, side_by_side)


def make_dead_object_overlay(
    raw_rgb: np.ndarray,
    dead_raw: np.ndarray | None,
    dead_mask: np.ndarray | None,
    rows: list[dict[str, Any]],
    out_path: Path,
    alpha: float,
) -> None:
    combined_panel = normalize_rgb(raw_rgb)
    if dead_raw is None:
        dead_panel = np.zeros_like(combined_panel)
    else:
        gray = normalize_rgb(np.repeat(dead_raw[:, :, None], 3, axis=2))
        dead_panel = gray.copy()
    object_panel = dead_panel.copy()
    if dead_mask is not None and dead_mask.size and int(dead_mask.max()) > 0:
        max_label = int(dead_mask.max())
        label_colors = np.zeros((max_label + 1, 3), dtype=np.float32)
        has_color = np.zeros(max_label + 1, dtype=bool)
        relation_colors = {
            "same_cell": np.array([0.0, 0.35, 1.0], dtype=np.float32),
            "dead_only": np.array([1.0, 0.75, 0.0], dtype=np.float32),
            "adjacent_dead": np.array([1.0, 0.0, 0.8], dtype=np.float32),
            "overlapping_live_dead_multi_nucleus": np.array([0.0, 0.9, 1.0], dtype=np.float32),
            "live_with_death_signal": np.array([0.15, 0.65, 1.0], dtype=np.float32),
            "adjacent_or_overlapping_dead_uncertain": np.array([0.55, 0.2, 1.0], dtype=np.float32),
            "adjacent_candidate": np.array([0.55, 0.35, 0.55], dtype=np.float32),
            "merged_multiple_objects": np.array([0.0, 1.0, 1.0], dtype=np.float32),
            "unassigned_candidate": np.array([0.6, 0.6, 0.6], dtype=np.float32),
            "rejected_signal": np.array([0.25, 0.25, 0.25], dtype=np.float32),
        }
        for row in rows:
            mask_id = int(row["dead_mask_id"])
            if 0 < mask_id <= max_label:
                relation = str(row["association_relation"])
                label_colors[mask_id] = relation_colors.get(relation, relation_colors["unassigned_candidate"])
                has_color[mask_id] = True
        colored = has_color[dead_mask]
        object_panel[colored] = (
            (1.0 - alpha) * object_panel[colored]
            + alpha * label_colors[dead_mask[colored]]
        )
    save_rgb_png(out_path, np.concatenate([combined_panel, dead_panel, object_panel], axis=1))


def make_overlap_state_overlay(
    raw_rgb: np.ndarray,
    nuclei_raw: np.ndarray | None,
    nuclei_mask: np.ndarray | None,
    dead_raw: np.ndarray | None,
    cell_instance_mask: np.ndarray,
    confirmed_dead_mask: np.ndarray,
    prediction_rows: list[dict[str, Any]],
    out_path: Path,
    alpha: float,
) -> None:
    base = normalize_rgb(raw_rgb)
    nuclei_panel = (
        np.zeros_like(base)
        if nuclei_raw is None
        else normalize_rgb(np.repeat(nuclei_raw[:, :, None], 3, axis=2))
    )
    if nuclei_mask is not None and nuclei_mask.size:
        nuclei_pixels = nuclei_mask > 0
        nuclei_panel[nuclei_pixels] = (
            0.45 * nuclei_panel[nuclei_pixels]
            + 0.55 * np.array([0.0, 1.0, 0.85], dtype=np.float32)
        )
        nuclei_panel[label_boundaries(nuclei_mask)] = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    dead_panel = (
        np.zeros_like(base)
        if dead_raw is None
        else normalize_rgb(np.repeat(dead_raw[:, :, None], 3, axis=2))
    )
    confirmed_pixels = confirmed_dead_mask > 0
    dead_panel[confirmed_pixels] = (
        0.15 * dead_panel[confirmed_pixels]
        + 0.85 * np.array([0.0, 0.55, 1.0], dtype=np.float32)
    )

    max_cell = int(cell_instance_mask.max()) if cell_instance_mask.size else 0
    colors = np.zeros((max_cell + 1, 3), dtype=np.float32)
    has_color = np.zeros(max_cell + 1, dtype=bool)
    for row in prediction_rows:
        label = int(row["mask_id"])
        if 0 < label <= max_cell:
            colors[label] = STATE_COLORS.get(str(row["state"]), STATE_COLORS["uncertain"])
            has_color[label] = str(row["state"]) != "artifact"
    cell_panel = base.copy()
    cell_pixels = has_color[cell_instance_mask]
    cell_panel[cell_pixels] = (
        (1.0 - alpha) * cell_panel[cell_pixels]
        + alpha * colors[cell_instance_mask[cell_pixels]]
    )

    overlap_panel = cell_panel.copy()
    overlap_panel[confirmed_pixels] = (
        0.10 * overlap_panel[confirmed_pixels]
        + 0.90 * np.array([0.0, 0.55, 1.0], dtype=np.float32)
    )
    overlap_panel[label_boundaries(confirmed_dead_mask)] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    save_rgb_png(
        out_path,
        np.concatenate([base, nuclei_panel, dead_panel, cell_panel, overlap_panel], axis=1),
    )


def summary_for_image(image_id: str, key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {state: 0 for state in STATE_COLORS}
    for row in rows:
        counts[str(row["state"])] = counts.get(str(row["state"]), 0) + 1
    total_masks = len(rows)
    artifact_count = counts.get("artifact", 0)
    total_cells = total_masks - artifact_count
    dead_count = counts.get("dead", 0)
    live_count = counts.get("live", 0)
    summary = {
        "image_id": image_id,
        "key": key,
        **key_details(key),
        "total_masks": total_masks,
        "total_cell_count": total_cells,
        "live_cell_count": live_count,
        "dead_cell_count": dead_count,
        "artifact_count": artifact_count,
        "uncertain_count": counts.get("uncertain", 0),
        "transitional_count": counts.get("transitional", 0),
        "dead_fraction": dead_count / total_cells if total_cells else 0.0,
        "live_fraction": live_count / total_cells if total_cells else 0.0,
        "rgb_dead_count": sum(1 for row in rows if row["rgb_state"] == "dead"),
        "rgb_live_count": sum(1 for row in rows if row["rgb_state"] == "live"),
        "dead_channel_override_count": sum(
            1
            for row in rows
            if row["state"] == "dead" and int(row.get("dead_mask_id") or 0) > 0 and row["rgb_state"] != "dead"
        ),
        "strong_dead_channel_override_count": sum(
            1 for row in rows if row.get("final_reason") == "strong_dead_channel_override_rgb_context"
        ),
        "bf_supported_count": sum(1 for row in rows if str(row["bf_supported"]) == "True"),
        "nuclei_supported_count": sum(1 for row in rows if str(row["nuclei_supported"]) == "True"),
        "nuclei_inside_total": sum(int(row.get("nuclei_centroids_inside", 0) or 0) for row in rows),
    }
    return summary


def process_one(
    key: str,
    records_by_profile: dict[str, dict[str, ProfileRecord]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined = records_by_profile["Combined"][key]
    if combined.image_path is None or not combined.image_path.exists():
        raise FileNotFoundError(f"Missing Combined raw image for key={key}")
    raw_rgb = to_rgb(read_raw_image(combined.image_path))
    combined_mask = read_mask(combined.mask_path)
    if combined_mask.shape != raw_rgb.shape[:2]:
        raise ValueError(
            f"Combined mask shape {combined_mask.shape} does not match image shape {raw_rgb.shape[:2]} for {key}"
        )
    combined_stats = label_stats(combined_mask)
    features = combined_rgb_features(raw_rgb, combined_mask, combined_stats)

    bf_stats = load_optional_stats(records_by_profile, "Brightfield", key)
    nucleus_record = records_by_profile.get("Nuclei", {}).get(key)
    nuclei_mask = None
    nuclei_raw = None
    if nucleus_record is not None:
        if nucleus_record.mask_path.exists():
            nuclei_mask = read_mask(nucleus_record.mask_path)
            if nuclei_mask.shape != combined_mask.shape:
                raise ValueError(
                    f"Nuclei mask shape {nuclei_mask.shape} does not match Combined mask "
                    f"shape {combined_mask.shape} for {key}"
                )
        if nucleus_record.image_path is not None and nucleus_record.image_path.exists():
            nuclei_raw = scalar_raw(read_raw_image(nucleus_record.image_path))
    nuclei_stats = label_stats(nuclei_mask) if nuclei_mask is not None else None
    bf_matches = nearest_label_matches(
        combined_stats.labels,
        combined_stats.centroids[combined_stats.labels],
        bf_stats,
        args.bf_match_distance,
    )
    nuclei_matches = nearest_label_matches(
        combined_stats.labels,
        combined_stats.centroids[combined_stats.labels],
        nuclei_stats,
        args.nuclei_match_distance,
    )
    nuclei_inside = nuclei_counts_inside_combined(combined_mask, nuclei_stats)

    dead_raw = None
    dead_mask = None
    dead_record = records_by_profile.get("Dead", {}).get(key)
    if dead_record is not None:
        if dead_record.image_path is not None and dead_record.image_path.exists():
            dead_raw = scalar_raw(read_raw_image(dead_record.image_path))
        if dead_record.mask_path.exists():
            dead_mask = read_mask(dead_record.mask_path)
    local_dead_rgb_features = (
        combined_rgb_features(raw_rgb, dead_mask, label_stats(dead_mask))
        if dead_mask is not None and int(dead_mask.max()) > 0
        else {}
    )
    dead_result = compute_dead_evidence(dead_raw, dead_mask, combined_mask, combined_stats, args)
    dead_evidence = assign_dead_object_relations(
        dead_result,
        features,
        local_dead_rgb_features,
        args,
    )
    nucleus_evidence = dead_object_nucleus_evidence(
        dead_result,
        dead_mask,
        nuclei_mask,
        nuclei_stats,
        nuclei_inside,
        args.nuclei_match_distance,
    )
    apply_nucleus_aware_overlap_relations(dead_result, nucleus_evidence)
    object_rows = dead_object_rows(
        combined.stem,
        key,
        dead_result,
        local_dead_rgb_features,
        nucleus_evidence,
        args,
    )

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for label in combined_stats.labels:
        idx = int(label)
        if idx not in features:
            continue
        row = {
            "image_id": combined.stem,
            "key": key,
            **key_details(key),
            **features[idx],
        }
        bf_label, bf_dist = bf_matches.get(idx, (0, float("inf")))
        nuclei_label, nuclei_dist = nuclei_matches.get(idx, (0, float("inf")))
        dead = dead_evidence.get(idx)
        bf_supported = bf_label > 0
        nuclei_supported = nuclei_label > 0
        nuclei_inside_count = int(nuclei_inside.get(idx, 0))
        state, reason, confidence = final_state_for(
            row,
            bf_supported=bf_supported,
            nuclei_supported=nuclei_supported,
            nuclei_inside_count=nuclei_inside_count,
            dead=dead,
            args=args,
        )
        row.update(
            {
                "bf_mask_id": bf_label,
                "bf_distance": bf_dist,
                "bf_supported": bf_supported,
                "nuclei_mask_id": nuclei_label,
                "nuclei_distance": nuclei_dist,
                "nuclei_supported": nuclei_supported,
                "nuclei_centroids_inside": nuclei_inside_count,
                "dead_mask_id": dead.label if dead else 0,
                "dead_match_mode": dead.match_mode if dead else "",
                "dead_association_relation": dead.association_relation if dead else "",
                "dead_distance": dead.distance if dead else "",
                "dead_overlap_pixels": dead.overlap_pixels if dead else 0,
                "dead_overlap_fraction": dead.dead_overlap_fraction if dead else "",
                "dead_combined_overlap_fraction": dead.combined_overlap_fraction if dead else "",
                "dead_raw_mean": dead.raw_mean if dead else "",
                "dead_raw_p90": dead.raw_p90 if dead else "",
                "dead_bg_median": dead.bg_median if dead else "",
                "dead_p90_delta": dead.p90_delta if dead else "",
                "dead_snr": dead.snr if dead else "",
                "state": state,
                "final_state": state,
                "final_reason": reason,
                "classification_confidence": confidence,
                "countable": state != "artifact",
            }
        )
        rows.append(row)
        pred_rows.append(
            {
                "image_id": combined.stem,
                "key": key,
                "mask_id": idx,
                "state": state,
                "final_reason": reason,
                "rgb_state": row["rgb_state"],
                "dead_mask_id": row["dead_mask_id"],
                "bf_mask_id": row["bf_mask_id"],
                "nuclei_mask_id": row["nuclei_mask_id"],
                "classification_confidence": confidence,
            }
        )

    feature_fields = [
        "image_id",
        "key",
        "well",
        "site",
        "day",
        "hour",
        "minute",
        "elapsed_hours",
        "combined_mask_id",
        "area",
        "centroid_y",
        "centroid_x",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
        "median_r",
        "median_g",
        "median_b",
        "mean_r",
        "mean_g",
        "mean_b",
        "r_frac",
        "g_frac",
        "b_frac",
        "log_b_over_r",
        "log_r_over_b",
        "hsv_s_mean",
        "red_excess",
        "blue_excess",
        "purple_score",
        "red_blue_balance",
        "rgb_state",
        "rgb_reason",
        "bf_mask_id",
        "bf_distance",
        "bf_supported",
        "nuclei_mask_id",
        "nuclei_distance",
        "nuclei_supported",
        "nuclei_centroids_inside",
        "dead_mask_id",
        "dead_match_mode",
        "dead_association_relation",
        "dead_distance",
        "dead_overlap_pixels",
        "dead_overlap_fraction",
        "dead_combined_overlap_fraction",
        "dead_raw_mean",
        "dead_raw_p90",
        "dead_bg_median",
        "dead_p90_delta",
        "dead_snr",
        "state",
        "final_state",
        "final_reason",
        "classification_confidence",
        "countable",
    ]
    prediction_fields = [
        "image_id",
        "key",
        "mask_id",
        "state",
        "final_reason",
        "rgb_state",
        "dead_mask_id",
        "bf_mask_id",
        "nuclei_mask_id",
        "classification_confidence",
    ]
    dead_object_fields = [
        "image_id",
        "key",
        "well",
        "site",
        "day",
        "hour",
        "minute",
        "elapsed_hours",
        "dead_mask_id",
        "source",
        "area",
        "aspect",
        "centroid_y",
        "centroid_x",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
        "assigned_combined_mask_id",
        "match_mode",
        "association_relation",
        "distance",
        "overlap_pixels",
        "dead_overlap_fraction",
        "combined_overlap_fraction",
        "raw_mean",
        "raw_p90",
        "raw_max",
        "bg_median",
        "bg_sigma",
        "mean_delta",
        "p90_delta",
        "snr",
        "keep_signal",
        "strong_direct",
        "dual_channel_confirmed",
        "confirmed_dead_object",
        "supplemental_dead_object",
        "local_rgb_state",
        "local_rgb_reason",
        "local_median_r",
        "local_median_g",
        "local_median_b",
        "local_r_frac",
        "local_g_frac",
        "local_b_frac",
        "local_blue_excess",
        "local_red_excess",
        "combined_nuclei_count",
        "dead_object_nuclei_inside_count",
        "dead_object_nearby_nuclei_count",
        "nearest_nucleus_distance",
        "dead_nucleus_overlap_fraction",
        "nucleus_supported_relation",
    ]
    feature_path = out_dir / "features" / f"{combined.stem}_per_cell_fusion_features.csv"
    prediction_path = out_dir / "predictions" / f"{combined.stem}_per_cell_predictions.csv"
    dead_object_path = out_dir / "dead_objects" / f"{combined.stem}_dead_object_features.csv"
    summary_path = out_dir / "summaries" / f"{combined.stem}_summary.csv"
    cell_state_mask_path = (
        out_dir / "masks" / "cell_state" / f"{combined.stem}_cell_state_masks.tif"
    )
    confirmed_dead_mask_path = (
        out_dir / "masks" / "confirmed_dead" / f"{combined.stem}_confirmed_dead_masks.tif"
    )
    overlap_mask_path = out_dir / "overlap_masks" / f"{combined.stem}_cell_dead_overlap_masks.tif"
    overlay_path = out_dir / "qc" / "label_overlays" / f"{combined.stem}_state_overlay.png"
    dead_object_overlay_path = (
        out_dir / "qc" / "dead_object_overlays" / f"{combined.stem}_dead_object_overlay.png"
    )
    overlap_overlay_path = (
        out_dir / "qc" / "overlap_state_overlays" / f"{combined.stem}_overlap_state_overlay.png"
    )
    cell_state_mask, confirmed_dead_mask, overlap_mask = build_cell_dead_overlap_masks(
        combined_mask,
        dead_mask,
        pred_rows,
        object_rows,
    )
    if args.force or not feature_path.exists():
        write_rows(feature_path, rows, feature_fields)
    if args.force or not prediction_path.exists():
        write_rows(prediction_path, pred_rows, prediction_fields)
    if args.force or not dead_object_path.exists():
        write_rows(dead_object_path, object_rows, dead_object_fields)
    if args.force or not cell_state_mask_path.exists():
        write_label_tiff(cell_state_mask_path, cell_state_mask, "YX")
    if args.force or not confirmed_dead_mask_path.exists():
        write_label_tiff(confirmed_dead_mask_path, confirmed_dead_mask, "YX")
    if args.force or not overlap_mask_path.exists():
        write_label_tiff(overlap_mask_path, overlap_mask, "CYX")
    summary = summary_for_image(combined.stem, key, pred_rows_with_feature_counts(pred_rows, rows))
    summary["segmented_dead_object_count"] = len(object_rows)
    summary["retained_dead_object_count"] = sum(bool(row["keep_signal"]) for row in object_rows)
    summary["strong_dead_object_count"] = sum(
        bool(row["keep_signal"]) and bool(row["strong_direct"]) for row in object_rows
    )
    summary["confirmed_dead_object_count"] = sum(
        bool(row["confirmed_dead_object"]) for row in object_rows
    )
    summary["dead_only_object_count"] = sum(
        row["association_relation"] == "dead_only" and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["adjacent_dead_object_count"] = sum(
        row["association_relation"]
        in {
            "adjacent_dead",
            "overlapping_live_dead_multi_nucleus",
            "live_with_death_signal",
            "adjacent_or_overlapping_dead_uncertain",
        }
        and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["overlapping_live_dead_multi_nucleus_count"] = sum(
        row["association_relation"] == "overlapping_live_dead_multi_nucleus"
        and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["live_with_death_signal_count"] = sum(
        row["association_relation"] == "live_with_death_signal"
        and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["nucleus_unresolved_dead_overlap_count"] = sum(
        row["association_relation"] == "adjacent_or_overlapping_dead_uncertain"
        and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["merged_multiple_dead_object_count"] = sum(
        row["association_relation"] == "merged_multiple_objects" and bool(row["confirmed_dead_object"])
        for row in object_rows
    )
    summary["supplemental_dead_object_count"] = sum(
        bool(row["supplemental_dead_object"]) for row in object_rows
    )
    summary["object_aware_dead_count"] = (
        int(summary["dead_cell_count"]) + int(summary["supplemental_dead_object_count"])
    )
    summary["nuclei_evidence_mask_kind"] = nucleus_record.mask_kind if nucleus_record is not None else "missing"
    summary["nuclei_evidence_mask_path"] = str(nucleus_record.mask_path) if nucleus_record is not None else ""
    if args.force or not summary_path.exists():
        write_rows(summary_path, [summary], list(summary.keys()))
    if args.force or not overlay_path.exists():
        make_classification_overlay(raw_rgb, combined_mask, pred_rows, overlay_path, args.overlay_alpha)
    if args.force or not dead_object_overlay_path.exists():
        make_dead_object_overlay(
            raw_rgb,
            dead_raw,
            dead_mask,
            object_rows,
            dead_object_overlay_path,
            args.overlay_alpha,
        )
    if args.force or not overlap_overlay_path.exists():
        make_overlap_state_overlay(
            raw_rgb,
            nuclei_raw,
            nuclei_mask,
            dead_raw,
            cell_state_mask,
            confirmed_dead_mask,
            pred_rows,
            overlap_overlay_path,
            args.overlay_alpha,
        )
    print(
        f"key={key} features={feature_path} predictions={prediction_path} "
        f"dead_objects={dead_object_path} summary={summary_path} overlap_masks={overlap_mask_path} "
        f"overlay={overlay_path} dead_object_overlay={dead_object_overlay_path} "
        f"overlap_overlay={overlap_overlay_path}"
    )
    return summary


def pred_rows_with_feature_counts(pred_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mask = {int(row["combined_mask_id"]): row for row in feature_rows}
    merged = []
    for row in pred_rows:
        next_row = dict(row)
        feature = by_mask.get(int(row["mask_id"]), {})
        next_row["bf_supported"] = feature.get("bf_supported", False)
        next_row["nuclei_supported"] = feature.get("nuclei_supported", False)
        next_row["nuclei_centroids_inside"] = feature.get("nuclei_centroids_inside", 0)
        merged.append(next_row)
    return merged


def load_optional_stats(records_by_profile: dict[str, dict[str, ProfileRecord]], profile: str, key: str) -> LabelStats | None:
    record = records_by_profile.get(profile, {}).get(key)
    if record is None or not record.mask_path.exists():
        return None
    return label_stats(read_mask(record.mask_path))


def default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    if args.run_root is not None:
        return args.run_root / "classification_fusion"
    if args.combined_run is not None:
        return args.combined_run / "classification_fusion"
    raise SystemExit("Cannot infer output directory.")


def merge_summary_rows(out_dir: Path, force: bool, timepoint: str | None = None) -> Path:
    summaries_dir = out_dir / "summaries"
    summary_paths = sorted(path for path in summaries_dir.glob("*_summary.csv") if path.name != "cell_count_summary.csv")
    if timepoint is not None:
        summary_paths = [
            path
            for path in summary_paths
            if extract_key_and_timepoint(path)[1] == timepoint
        ]
    if not summary_paths:
        raise SystemExit(
            f"No per-image summary CSVs found under {summaries_dir} "
            f"for timepoint={timepoint or 'all'}"
        )

    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    for summary_path in summary_paths:
        summary_rows = read_csv_rows(summary_path)
        if len(summary_rows) != 1:
            raise SystemExit(f"Expected exactly one row in {summary_path}, found {len(summary_rows)}")
        if fieldnames is None:
            fieldnames = list(summary_rows[0].keys())
        rows.append(summary_rows[0])
    rows.sort(key=lambda row: (str(row.get("well", "")), float(row.get("elapsed_hours", 0) or 0), str(row.get("key", ""))))

    out_path = summaries_dir / "cell_count_summary.csv"
    if force or not out_path.exists():
        write_rows(out_path, rows, fieldnames or list(rows[0].keys()))
    print(f"merged_summaries={len(rows)} output={out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    selected_timepoint = normalize_timepoint(args.timepoint) if args.timepoint is not None else None
    if args.merge_summaries_only:
        merge_summary_rows(default_out_dir(args), args.force, selected_timepoint)
        return

    if args.field_record is not None:
        records_by_profile = direct_records_from_json(args)
        record_source = "field_record"
    else:
        records_by_profile = {profile: collect_records_for_profile(args, profile) for profile in PROFILES}
        record_source = "directory_discovery"
    combined_records = records_by_profile["Combined"]
    if not combined_records:
        raise SystemExit("No Combined records found. Check --run-root/--combined-run and segmentation summaries.")
    keys = sorted(combined_records)
    keys = select_keys(keys, selected_timepoint)
    if args.key:
        wanted = set(args.key)
        keys = [key for key in keys if key in wanted]
    if args.limit is not None:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit(
            f"No Combined records matched timepoint={selected_timepoint or 'all'} and the requested keys."
        )
    for profile in PROFILES:
        missing = [key for key in keys if key not in records_by_profile[profile]]
        if missing:
            raise SystemExit(
                f"{profile} is missing {len(missing)} selected records; examples: {missing[:5]}"
            )
    out_dir = default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"n_combined_records={len(combined_records)}")
    for profile in PROFILES:
        print(f"n_{profile.lower()}={len(records_by_profile[profile])}")
    print(f"n_selected={len(keys)}")
    print(f"selected_timepoint={selected_timepoint or 'all'}")
    print(f"record_source={record_source}")
    print(f"out_dir={out_dir}")
    print(f"prefer_nucleus_core_masks={args.prefer_nucleus_core_masks}")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for key in keys:
        try:
            if (
                args.per_key_only
                and not args.force
                and per_key_outputs_complete(out_dir, records_by_profile["Combined"][key])
            ):
                print(f"key={key} fusion_outputs_already_complete=1", flush=True)
                continue
            summaries.append(process_one(key, records_by_profile, out_dir, args))
        except Exception as exc:
            failures.append({"key": key, "error": repr(exc)})
            print(f"FAILED key={key}: {exc!r}", flush=True)
            if not args.continue_on_error:
                raise

    if summaries and not args.per_key_only:
        write_rows(out_dir / "summaries" / "cell_count_summary.csv", summaries, list(summaries[0].keys()))
    if failures:
        if args.per_key_only:
            for failure in failures:
                failure_key = str(failure.get("key", "unknown"))
                write_rows(out_dir / "failures" / f"{failure_key}_failure.csv", [failure], ["key", "error"])
        else:
            write_rows(out_dir / "failures.csv", failures, ["key", "error"])
        print(f"n_failures={len(failures)}")
    print(f"fusion_complete={out_dir}")


if __name__ == "__main__":
    main()
