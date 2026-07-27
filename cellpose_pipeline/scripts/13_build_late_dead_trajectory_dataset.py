#!/usr/bin/env python3
"""Build a path-backed trajectory dataset for all-time death refinement.

No upstream segmentation is rerun.  Frozen d0 fields provide the safety
reference, while complete well/site time courses provide field-collapse and
single-object trajectory evidence.  Proxy labels are operational anchors only;
they are never presented as biological ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table


FEATURE_SUFFIX = "_per_cell_fusion_features.csv"
DEFAULT_PLATE_MAP = (
    Path(__file__).resolve().parent
    / "analysisi"
    / "resources"
    / "SUM159_AC_Experiment1_PlateMap.csv"
)
BRANCHES = ("original", "nucleated_only")
BRANCH_DIRS = {
    "original": "classification_fusion",
    "nucleated_only": "classification_fusion_nucleated_only",
}
D0_BRANCH_DIRS = {
    "original": "classification_original",
    "nucleated_only": "classification_nucleated_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument(
        "--d0-audit-root",
        type=Path,
        help=(
            "Optional frozen d0 audit root used by calibration runs. When omitted, "
            "d0 fields are read from --classification-root for production."
        ),
    )
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--d0-fields", type=int, default=320)
    parser.add_argument(
        "--trajectory-wells",
        default="A9,B9,C2,D2,E2,F2,E9,F9,G9,H9",
        help=(
            "Comma-separated full-time-course wells. Defaults include the E9/F9 "
            "development/replicate pair, matched untreated controls, and treated "
            "2N/4N controls."
        ),
    )
    parser.add_argument(
        "--all-wells",
        action="store_true",
        help="Process every plate-map well present in the classification summary.",
    )
    parser.add_argument("--trajectory-min-hours", type=float, default=2.0)
    parser.add_argument("--trajectory-max-hours", type=float, default=168.0)
    parser.add_argument("--temporal-distance-px", type=float, default=10.0)
    parser.add_argument("--remnant-max-area-ratio", type=float, default=0.85)
    parser.add_argument("--remnant-max-red-ratio", type=float, default=0.90)
    parser.add_argument("--stable-live-distance-px", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260722)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "with"}


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip()
    return float(text) if text else default


def key_for_time(well: str, site: int, elapsed_hours: float) -> str:
    minutes = int(round(elapsed_hours * 60))
    day, rem = divmod(minutes, 24 * 60)
    hour, minute = divmod(rem, 60)
    return f"{well}_{site}_{day:02d}d{hour:02d}h{minute:02d}m"


def feature_path(branch_root: Path, key: str) -> Path:
    return branch_root / "features" / f"SUM159_AC_{key}{FEATURE_SUFFIX}"


def record_path(manifest_root: Path, key: str) -> Path:
    return manifest_root / "records" / key.split("_", 1)[0] / f"{key}.json"


def density_proxy_bins(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=("low", "middle", "high")).astype(str)


def dose_bin(value: float) -> str:
    if value <= 0:
        return "zero"
    if value <= 25:
        return "low"
    if value <= 200:
        return "middle"
    return "high"


def prepare_full_summary(
    summary: pd.DataFrame,
    plate_map: pd.DataFrame,
) -> pd.DataFrame:
    required_summary = {
        "image_id",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "total_cell_count",
        "dead_cell_count",
    }
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"Classification summary missing fields: {sorted(missing)}")
    summary = summary.copy()
    summary["well"] = summary["well"].astype(str)
    summary["site"] = pd.to_numeric(summary["site"], errors="raise").astype(int)
    summary["elapsed_hours"] = pd.to_numeric(summary["elapsed_hours"], errors="raise")
    plate = plate_map.copy()
    plate["well"] = plate["well"].astype(str)
    plate["cyclophosphamide"] = plate["cyclophosphamide"].map(bool_value)
    plate["doxorubicin_nm"] = pd.to_numeric(plate["doxorubicin_nm"], errors="raise")
    summary = summary.merge(plate, on="well", how="left", validate="many_to_one")
    if summary["ploidy"].isna().any():
        raise ValueError("Some classification wells are missing from the plate map")
    summary["treated"] = summary["cyclophosphamide"] | (summary["doxorubicin_nm"] > 0)
    summary["dose_bin"] = summary["doxorubicin_nm"].map(dose_bin)
    summary["day_bin"] = np.floor(summary["elapsed_hours"] / 24).astype(int)
    summary["density_proxy_bin"] = density_proxy_bins(summary["total_cell_count"])
    return summary.sort_values(["well", "site", "elapsed_hours"]).reset_index(drop=True)


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    full_summary = read_table(
        args.classification_root
        / "classification_fusion"
        / "summaries"
        / "cell_count_summary.csv"
    )
    plate_map = read_table(args.plate_map)
    full_summary = prepare_full_summary(full_summary, plate_map)
    trajectory_wells = (
        tuple(dict.fromkeys(full_summary["well"].astype(str)))
        if args.all_wells
        else tuple(
            dict.fromkeys(
                value.strip()
                for value in args.trajectory_wells.split(",")
                if value.strip()
            )
        )
    )
    if not trajectory_wells:
        raise ValueError("--trajectory-wells must contain at least one well")
    unknown_wells = sorted(set(trajectory_wells) - set(full_summary["well"].astype(str)))
    if unknown_wells:
        raise ValueError(f"Trajectory wells are absent from classification summary: {unknown_wells}")
    selected = full_summary.loc[
        full_summary["well"].isin(trajectory_wells)
        & full_summary["elapsed_hours"].between(
            args.trajectory_min_hours,
            args.trajectory_max_hours,
        )
    ].copy()
    if selected.empty:
        raise ValueError("No trajectory fields matched the requested wells/time range")
    available_keys = set(full_summary["key"].astype(str))
    selected["previous_key"] = [
        key_for_time(str(row.well), int(row.site), float(row.elapsed_hours) - 2.0)
        for row in selected.itertuples()
    ]
    selected["next_key"] = [
        key_for_time(str(row.well), int(row.site), float(row.elapsed_hours) + 2.0)
        for row in selected.itertuples()
    ]
    selected.loc[~selected["previous_key"].isin(available_keys), "previous_key"] = ""
    selected.loc[~selected["next_key"].isin(available_keys), "next_key"] = ""
    selected["previous_2_key"] = [
        key_for_time(str(row.well), int(row.site), float(row.elapsed_hours) - 4.0)
        for row in selected.itertuples()
    ]
    selected["next_2_key"] = [
        key_for_time(str(row.well), int(row.site), float(row.elapsed_hours) + 4.0)
        for row in selected.itertuples()
    ]
    selected.loc[
        ~selected["previous_2_key"].isin(available_keys),
        "previous_2_key",
    ] = ""
    selected.loc[~selected["next_2_key"].isin(available_keys), "next_2_key"] = ""
    selected["sentinel"] = selected["key"].eq("E9_1_05d00h00m")
    selected["development_anchor"] = selected["well"].eq("E9") & selected["site"].eq(1)
    selected["holdout_anchor"] = selected["well"].eq("E9") & selected["site"].ne(1)
    selected["replicate_diagnostic"] = selected["well"].eq("F9")
    selected["selection_seed"] = args.seed
    full_manifest_root = args.classification_root / "workflow_status" / "postsegmentation_manifest"
    if args.d0_audit_root is not None:
        d0_manifest_root = args.d0_audit_root / "field_manifest"
        d0_table = read_table(d0_manifest_root / "field_manifest.tsv")
        d0_source = "d0_audit"
    else:
        d0_manifest_root = full_manifest_root
        d0_table = full_summary.loc[
            full_summary["elapsed_hours"].eq(0.0),
            ["key", "well", "site", "elapsed_hours"],
        ].copy()
        d0_source = "classification"
    if len(d0_table) != 320:
        raise ValueError(f"Frozen d0 source manifest must have 320 fields, found {len(d0_table)}")
    if not 1 <= args.d0_fields <= 320:
        raise ValueError(f"--d0-fields must be between 1 and 320: {args.d0_fields}")
    if args.d0_fields < 320:
        d0_table = d0_table.assign(
            _selection_order=d0_table["key"].astype(str).map(
                lambda key: hashlib.sha256(f"{args.seed}:d0:{key}".encode()).hexdigest()
            )
        ).sort_values("_selection_order").head(args.d0_fields)

    plate_lookup = plate_map.copy()
    plate_lookup["well"] = plate_lookup["well"].astype(str)
    plate_lookup["cyclophosphamide"] = plate_lookup["cyclophosphamide"].map(bool_value)
    plate_lookup["doxorubicin_nm"] = pd.to_numeric(
        plate_lookup["doxorubicin_nm"], errors="raise"
    )
    plate_lookup = plate_lookup.set_index("well")
    rows: list[dict[str, Any]] = []
    for row in d0_table.itertuples(index=False):
        key = str(row.key)
        well = key.split("_", 1)[0]
        plate_row = plate_lookup.loc[well]
        cyclophosphamide = bool(plate_row["cyclophosphamide"])
        doxorubicin_nm = float(plate_row["doxorubicin_nm"])
        rows.append(
            {
                "cohort": "d0_frozen",
                "key": key,
                "well": well,
                "site": int(key.split("_")[1]),
                "elapsed_hours": 0.0,
                "treated": bool(cyclophosphamide or doxorubicin_nm > 0),
                "sentinel": False,
                "development_anchor": False,
                "holdout_anchor": False,
                "replicate_diagnostic": False,
                "record_path": str(record_path(d0_manifest_root, key)),
                "previous_key": "",
                "previous_2_key": "",
                "next_key": key_for_time(well, int(key.split("_")[1]), 2.0)
                if key_for_time(well, int(key.split("_")[1]), 2.0) in available_keys
                else "",
                "next_2_key": key_for_time(well, int(key.split("_")[1]), 4.0)
                if key_for_time(well, int(key.split("_")[1]), 4.0) in available_keys
                else "",
                "ploidy": str(plate_row["ploidy"]),
                "cyclophosphamide": cyclophosphamide,
                "doxorubicin_nm": doxorubicin_nm,
                "dose_bin": "d0",
                "density_proxy_bin": "",
                "selection_seed": args.seed,
                "feature_source": d0_source,
            }
        )
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "cohort": "trajectory",
                "key": str(row.key),
                "well": str(row.well),
                "site": int(row.site),
                "elapsed_hours": float(row.elapsed_hours),
                "treated": bool(row.treated),
                "sentinel": bool(row.sentinel),
                "development_anchor": bool(row.development_anchor),
                "holdout_anchor": bool(row.holdout_anchor),
                "replicate_diagnostic": bool(row.replicate_diagnostic),
                "record_path": str(record_path(full_manifest_root, str(row.key))),
                "previous_key": str(row.previous_key),
                "previous_2_key": str(row.previous_2_key),
                "next_key": str(row.next_key),
                "next_2_key": str(row.next_2_key),
                "ploidy": str(row.ploidy),
                "cyclophosphamide": bool(row.cyclophosphamide),
                "doxorubicin_nm": float(row.doxorubicin_nm),
                "dose_bin": str(row.dose_bin),
                "density_proxy_bin": str(row.density_proxy_bin),
                "selection_seed": args.seed,
                "feature_source": "classification",
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.loc[manifest["cohort"] == "d0_frozen", "key"].nunique() != args.d0_fields:
        raise ValueError("Frozen d0 manifest does not contain the requested unique key count")
    expected_trajectory = len(selected)
    if manifest.loc[manifest["cohort"] == "trajectory", "key"].nunique() != expected_trajectory:
        raise ValueError("Trajectory manifest contains duplicate or missing keys")
    for path in manifest["record_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return manifest


def read_mask(path: str | Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label image: {path}, shape={labels.shape}")
    return labels.astype(np.int32, copy=False)


def props_frame(labels: np.ndarray) -> pd.DataFrame:
    values = regionprops_table(
        labels,
        properties=(
            "label",
            "area",
            "perimeter",
            "solidity",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
        ),
    )
    frame = pd.DataFrame(values).rename(
        columns={
            "label": "combined_mask_id",
            "area": "mask_area",
            "perimeter": "mask_perimeter",
            "solidity": "mask_solidity",
            "eccentricity": "mask_eccentricity",
            "major_axis_length": "mask_major_axis",
            "minor_axis_length": "mask_minor_axis",
        }
    )
    perimeter = frame["mask_perimeter"].replace(0, np.nan)
    frame["mask_circularity"] = np.minimum(
        1.0,
        4.0 * math.pi * frame["mask_area"] / perimeter.pow(2),
    ).fillna(0.0)
    frame["mask_axis_ratio"] = (
        frame["mask_major_axis"]
        / frame["mask_minor_axis"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(99.0)
    return frame


def overlap_counts(cell_labels: np.ndarray, nucleus_labels: np.ndarray) -> np.ndarray:
    return np.bincount(
        cell_labels.ravel(),
        weights=(nucleus_labels.ravel() > 0).astype(np.float64),
        minlength=int(cell_labels.max()) + 1,
    )


def scalar_raw_image(path: str | Path) -> np.ndarray:
    image = np.squeeze(tifffile.imread(path))
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        return image[..., :3].max(axis=-1).astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[0] in {3, 4}:
        return image[:3].max(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Expected a scalar-compatible raw image: {path}, shape={image.shape}")


def cell_conditioned_dead_features(
    cell_labels: np.ndarray,
    nucleus_labels: np.ndarray,
    dead_raw: np.ndarray,
) -> pd.DataFrame:
    """Measure Dead signal inside every existing cell without changing masks."""
    if cell_labels.shape != nucleus_labels.shape or cell_labels.shape != dead_raw.shape:
        raise ValueError(
            "Cell-conditioned Dead inputs must have identical 2D shapes: "
            f"cell={cell_labels.shape}, nucleus={nucleus_labels.shape}, "
            f"dead={dead_raw.shape}"
        )
    labels = cell_labels.astype(np.int32, copy=False)
    raw = dead_raw.astype(np.float64, copy=False)
    max_label = int(labels.max())
    label_ids = np.arange(1, max_label + 1, dtype=int)
    flat_labels = labels.ravel()
    flat_raw = raw.ravel()
    counts = np.bincount(flat_labels, minlength=max_label + 1).astype(float)
    sums = np.bincount(
        flat_labels,
        weights=flat_raw,
        minlength=max_label + 1,
    )
    background = flat_raw[flat_labels == 0]
    if background.size < 32:
        background = flat_raw
    bg_median = float(np.median(background))
    bg_sigma = float(
        max(
            1e-6,
            1.4826 * np.median(np.abs(background - bg_median)),
        )
    )
    positive = raw >= (bg_median + 3.0 * bg_sigma)
    positive_counts = np.bincount(
        flat_labels,
        weights=positive.ravel().astype(float),
        minlength=max_label + 1,
    )
    nucleus = nucleus_labels > 0
    nucleus_counts = np.bincount(
        flat_labels,
        weights=nucleus.ravel().astype(float),
        minlength=max_label + 1,
    )
    nucleus_sums = np.bincount(
        flat_labels,
        weights=(raw * nucleus).ravel(),
        minlength=max_label + 1,
    )
    cytoplasm = (labels > 0) & ~nucleus
    cytoplasm_counts = np.bincount(
        flat_labels,
        weights=cytoplasm.ravel().astype(float),
        minlength=max_label + 1,
    )
    cytoplasm_sums = np.bincount(
        flat_labels,
        weights=(raw * cytoplasm).ravel(),
        minlength=max_label + 1,
    )
    maxima = ndimage.maximum(raw, labels=labels, index=label_ids)
    cell_mean = np.divide(
        sums[label_ids],
        counts[label_ids],
        out=np.full(max_label, bg_median, dtype=float),
        where=counts[label_ids] > 0,
    )
    nucleus_mean = np.divide(
        nucleus_sums[label_ids],
        nucleus_counts[label_ids],
        out=np.full(max_label, bg_median, dtype=float),
        where=nucleus_counts[label_ids] > 0,
    )
    cytoplasm_mean = np.divide(
        cytoplasm_sums[label_ids],
        cytoplasm_counts[label_ids],
        out=np.full(max_label, bg_median, dtype=float),
        where=cytoplasm_counts[label_ids] > 0,
    )
    return pd.DataFrame(
        {
            "combined_mask_id": label_ids,
            "cell_dead_mean": cell_mean,
            "cell_dead_max": np.asarray(maxima, dtype=float),
            "cell_dead_background": bg_median,
            "cell_dead_background_sigma": bg_sigma,
            "cell_dead_mean_delta": cell_mean - bg_median,
            "cell_dead_max_delta": np.asarray(maxima, dtype=float) - bg_median,
            "cell_dead_snr": (cell_mean - bg_median) / bg_sigma,
            "cell_dead_peak_snr": (
                np.asarray(maxima, dtype=float) - bg_median
            )
            / bg_sigma,
            "cell_dead_positive_fraction": np.divide(
                positive_counts[label_ids],
                counts[label_ids],
                out=np.zeros(max_label, dtype=float),
                where=counts[label_ids] > 0,
            ),
            "nuclear_dead_mean": nucleus_mean,
            "cytoplasm_dead_mean": cytoplasm_mean,
            "cell_dead_core_enrichment": (
                nucleus_mean - cytoplasm_mean
            )
            / bg_sigma,
        }
    )


def read_features(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    required = {
        "combined_mask_id",
        "area",
        "centroid_y",
        "centroid_x",
        "median_r",
        "rgb_state",
        "final_state",
        "dead_mask_id",
        "dead_p90_delta",
        "dead_snr",
        "classification_confidence",
        "countable",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table {path} missing columns: {sorted(missing)}")
    # Production refinement is intentionally rerunnable.  If a previous
    # refinement is present, reconstruct the frozen pre-refinement classifier
    # state before deriving temporal anchors and field trajectories.
    if "pre_late_death_state" in frame:
        baseline_state = frame["pre_late_death_state"].fillna("").astype(str)
        valid = baseline_state.ne("")
        frame.loc[valid, "state"] = baseline_state.loc[valid]
        frame.loc[valid, "final_state"] = baseline_state.loc[valid]
    if "pre_late_death_final_reason" in frame:
        baseline_reason = frame["pre_late_death_final_reason"].fillna("").astype(str)
        valid = baseline_reason.ne("")
        frame.loc[valid, "final_reason"] = baseline_reason.loc[valid]
    if "pre_late_death_classification_confidence" in frame:
        baseline_confidence = (
            frame["pre_late_death_classification_confidence"]
            .fillna("")
            .astype(str)
        )
        valid = baseline_confidence.ne("")
        frame.loc[valid, "classification_confidence"] = (
            baseline_confidence.loc[valid]
        )
    frame["combined_mask_id"] = pd.to_numeric(frame["combined_mask_id"], errors="raise").astype(int)
    return frame


def nearest_rows(current: pd.DataFrame, adjacent: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = pd.DataFrame(index=current.index)
    fields = (
        "final_state",
        "rgb_state",
        "dead_mask_id",
        "dead_p90_delta",
        "dead_snr",
        "classification_confidence",
        "median_r",
        "area",
    )
    if adjacent.empty or current.empty:
        output[f"{prefix}_distance"] = np.inf
        output[f"{prefix}_mutual_nearest"] = False
        for field in fields:
            output[f"{prefix}_{field}"] = "" if field.endswith("state") or field == "classification_confidence" else np.nan
        return output
    tree = cKDTree(adjacent[["centroid_y", "centroid_x"]].to_numpy(dtype=float))
    distances, indices = tree.query(
        current[["centroid_y", "centroid_x"]].to_numpy(dtype=float),
        k=1,
    )
    reverse_tree = cKDTree(current[["centroid_y", "centroid_x"]].to_numpy(dtype=float))
    _reverse_distances, reverse_indices = reverse_tree.query(
        adjacent[["centroid_y", "centroid_x"]].to_numpy(dtype=float),
        k=1,
    )
    matched = adjacent.iloc[indices].reset_index(drop=True)
    output[f"{prefix}_distance"] = distances
    output[f"{prefix}_mutual_nearest"] = reverse_indices[indices] == np.arange(len(current))
    for field in fields:
        output[f"{prefix}_{field}"] = matched[field].to_numpy()
    return output


def mutual_nearest_flags(current: pd.DataFrame, adjacent: pd.DataFrame) -> np.ndarray:
    if current.empty or adjacent.empty:
        return np.zeros(len(current), dtype=bool)
    current_xy = current[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    adjacent_xy = adjacent[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    _distances, adjacent_indices = cKDTree(adjacent_xy).query(current_xy, k=1)
    _reverse_distances, current_indices = cKDTree(current_xy).query(adjacent_xy, k=1)
    return current_indices[adjacent_indices] == np.arange(len(current))


def strong_dead(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    state = frame[f"{prefix}final_state"].astype(str)
    rgb_state = frame[f"{prefix}rgb_state"].astype(str)
    delta = pd.to_numeric(frame[f"{prefix}dead_p90_delta"], errors="coerce").fillna(0)
    snr = pd.to_numeric(frame[f"{prefix}dead_snr"], errors="coerce").fillna(0)
    dead_mask = pd.to_numeric(frame[f"{prefix}dead_mask_id"], errors="coerce").fillna(0)
    return (state == "dead") & (
        (rgb_state == "dead") | ((dead_mask > 0) & (delta >= 40.0) & (snr >= 100.0))
    )


def extract_one(task: dict[str, Any]) -> dict[str, Any]:
    cohort = str(task["cohort"])
    branch = str(task["branch"])
    key = str(task["key"])
    shard = Path(task["shard_path"])

    required_cache_fields = {
        "proxy_type",
        "temporal_track_confident",
        "temporal_support_frames",
        "temporal_match_confidence",
        "red_mass_proxy",
        "cytoplasm_red_mass_proxy",
        "core_nucleus_to_cytoplasm",
        "extent_nucleus_to_cytoplasm",
        "mask_area",
        "core_cytoplasm_area",
        "mask_circularity",
        "mask_solidity",
        "cell_dead_snr",
        "cell_dead_positive_fraction",
        "cell_dead_core_enrichment",
        "field_cell_count",
        "field_mask_fraction",
        "field_median_nn",
        "field_median_area",
        "field_median_red_mass",
        "field_median_core_nc",
        "field_current_dead_fraction",
        "source_feature_path",
        "field_record_path",
        "cell_mask_path",
        "nucleus_core_mask_path",
        "nucleus_extent_mask_path",
        "combined_raw_path",
        "dead_raw_path",
    }

    def summarize(frame: pd.DataFrame, cached: bool) -> dict[str, Any]:
        proxy_counts = frame["proxy_type"].value_counts().to_dict()
        first = frame.iloc[0] if len(frame) else {}
        return {
            "cohort": cohort,
            "branch": branch,
            "key": key,
            "shard_path": str(shard),
            "cached": cached,
            "cell_rows": int(len(frame)),
            "field_cell_count": int(first.get("field_cell_count", 0)),
            "field_countable_count": int(first.get("field_countable_count", 0)),
            "field_mask_fraction": float(first.get("field_mask_fraction", 0.0)),
            "field_median_nn": float(first.get("field_median_nn", float("inf"))),
            "field_median_area": float(first.get("field_median_area", float("nan"))),
            "field_median_cytoplasm_area": float(
                first.get("field_median_cytoplasm_area", float("nan"))
            ),
            "field_median_red_mass": float(
                first.get("field_median_red_mass", float("nan"))
            ),
            "field_median_core_nc": float(
                first.get("field_median_core_nc", float("nan"))
            ),
            "field_current_dead_fraction": float(
                first.get("field_current_dead_fraction", float("nan"))
            ),
            "temporal_dead_remnant": int(
                proxy_counts.get("temporal_dead_remnant", 0)
            ),
            "blue_supported_dead": int(proxy_counts.get("blue_supported_dead", 0)),
            "d0_live_anchor": int(proxy_counts.get("d0_live_anchor", 0)),
            "untreated_live_anchor": int(
                proxy_counts.get("untreated_live_anchor", 0)
            ),
            "unlabeled": int(proxy_counts.get("unlabeled", 0)),
        }

    if shard.is_file() and not bool(task["force"]):
        cached_header = set(pd.read_csv(shard, nrows=0).columns)
        if required_cache_fields <= cached_header:
            return summarize(pd.read_csv(shard), cached=True)

    record = json.loads(Path(task["record_path"]).read_text())
    profiles = record["profiles"]
    mask_field = "original_mask" if branch == "original" else "nucleated_mask"
    cell_mask = read_mask(profiles["Combined"][mask_field])
    core_mask = read_mask(profiles["Nuclei"]["core_mask"])
    extent_mask = read_mask(profiles["Nuclei"]["extent_mask"])
    dead_raw = scalar_raw_image(profiles["Dead"]["raw"])
    if cell_mask.shape != core_mask.shape or cell_mask.shape != extent_mask.shape:
        raise ValueError(f"Mask shape mismatch for {branch}/{key}")
    if dead_raw.shape != cell_mask.shape:
        raise ValueError(
            f"Dead raw/cell mask shape mismatch for {branch}/{key}: "
            f"dead={dead_raw.shape}, cell={cell_mask.shape}"
        )

    if cohort == "d0_frozen" and str(task.get("feature_source")) == "d0_audit":
        branch_root = Path(task["d0_audit_root"]) / D0_BRANCH_DIRS[branch]
    else:
        branch_root = Path(task["classification_root"]) / BRANCH_DIRS[branch]
    current_path = feature_path(branch_root, key)
    current = read_features(current_path)
    metrics = props_frame(cell_mask)
    core_counts = overlap_counts(cell_mask, core_mask)
    extent_counts = overlap_counts(cell_mask, extent_mask)
    labels = metrics["combined_mask_id"].to_numpy(dtype=int)
    metrics["nucleus_core_overlap"] = core_counts[labels]
    metrics["nucleus_extent_overlap"] = extent_counts[labels]
    metrics["core_cytoplasm_area"] = np.maximum(
        metrics["mask_area"] - metrics["nucleus_core_overlap"], 1.0
    )
    metrics["extent_cytoplasm_area"] = np.maximum(
        metrics["mask_area"] - metrics["nucleus_extent_overlap"], 1.0
    )
    metrics["core_nucleus_to_cytoplasm"] = (
        metrics["nucleus_core_overlap"] / metrics["core_cytoplasm_area"]
    )
    metrics["extent_nucleus_to_cytoplasm"] = (
        metrics["nucleus_extent_overlap"] / metrics["extent_cytoplasm_area"]
    )
    metrics["core_nucleus_fraction"] = metrics["nucleus_core_overlap"] / metrics["mask_area"]
    metrics["extent_nucleus_fraction"] = metrics["nucleus_extent_overlap"] / metrics["mask_area"]
    current = current.merge(metrics, on="combined_mask_id", how="left", validate="one_to_one")
    current = current.merge(
        cell_conditioned_dead_features(cell_mask, core_mask, dead_raw),
        on="combined_mask_id",
        how="left",
        validate="one_to_one",
    )
    if current["mask_area"].isna().any():
        raise ValueError(f"Feature/mask label mismatch for {branch}/{key}")
    if current[
        [
            "cell_dead_snr",
            "cell_dead_positive_fraction",
            "cell_dead_core_enrichment",
        ]
    ].isna().any().any():
        raise ValueError(f"Cell-conditioned Dead feature mismatch for {branch}/{key}")
    current["red_mass_proxy"] = (
        pd.to_numeric(current["median_r"], errors="coerce").fillna(0.0)
        * current["mask_area"]
    )
    current["cytoplasm_red_mass_proxy"] = (
        pd.to_numeric(current["median_r"], errors="coerce").fillna(0.0)
        * current["core_cytoplasm_area"]
    )

    centroids = current[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    if len(centroids) >= 2:
        nn = cKDTree(centroids).query(centroids, k=2)[0][:, 1]
        median_nn = float(np.median(nn))
    else:
        median_nn = float("inf")
    border_labels = set(np.unique(np.concatenate((cell_mask[0], cell_mask[-1], cell_mask[:, 0], cell_mask[:, -1]))))
    border_labels.discard(0)
    current["border_touching"] = current["combined_mask_id"].isin(border_labels)
    current["field_cell_count"] = len(current)
    current["field_mask_fraction"] = float(np.count_nonzero(cell_mask)) / float(cell_mask.size)
    current["field_median_nn"] = median_nn
    countable = current["countable"].astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )
    eligible_field = current.loc[countable & ~current["border_touching"]]
    current["field_countable_count"] = int(len(eligible_field))
    current["field_median_area"] = float(eligible_field["mask_area"].median())
    current["field_median_cytoplasm_area"] = float(
        eligible_field["core_cytoplasm_area"].median()
    )
    current["field_median_red_mass"] = float(
        eligible_field["red_mass_proxy"].median()
    )
    current["field_median_core_nc"] = float(
        eligible_field["core_nucleus_to_cytoplasm"].median()
    )
    current["field_current_dead_fraction"] = float(
        eligible_field["final_state"].astype(str).eq("dead").mean()
    )

    if cohort == "trajectory":
        previous_key = str(task.get("previous_key", ""))
        previous_2_key = str(task.get("previous_2_key", ""))
        next_key = str(task.get("next_key", ""))
        next_2_key = str(task.get("next_2_key", ""))
        previous = (
            read_features(feature_path(branch_root, previous_key))
            if previous_key
            else pd.DataFrame()
        )
        previous_2 = (
            read_features(feature_path(branch_root, previous_2_key))
            if previous_2_key
            else pd.DataFrame()
        )
        following = (
            read_features(feature_path(branch_root, next_key))
            if next_key
            else pd.DataFrame()
        )
        following_2 = (
            read_features(feature_path(branch_root, next_2_key))
            if next_2_key
            else pd.DataFrame()
        )
        prev_match = nearest_rows(current, previous, "prev")
        prev2_match = nearest_rows(current, previous_2, "prev2")
        next_match = nearest_rows(current, following, "next")
        next2_match = nearest_rows(current, following_2, "next2")
        current = pd.concat(
            [
                current.reset_index(drop=True),
                prev_match,
                prev2_match,
                next_match,
                next2_match,
            ],
            axis=1,
        )
        prior_strong = strong_dead(current, "prev_")
        prior_2_strong = strong_dead(current, "prev2_")
        current_strong = strong_dead(current)
        current_non_dead = current["final_state"].astype(str) != "dead"
        current["temporal_area_ratio"] = current["area"] / pd.to_numeric(
            current["prev_area"], errors="coerce"
        ).replace(0, np.nan)
        current["temporal_red_ratio"] = current["median_r"] / pd.to_numeric(
            current["prev_median_r"], errors="coerce"
        ).replace(0, np.nan)
        current["temporal_red_mass_ratio"] = current["red_mass_proxy"] / (
            pd.to_numeric(current["prev_median_r"], errors="coerce")
            * pd.to_numeric(current["prev_area"], errors="coerce")
        ).replace(0, np.nan)
        adaptive_distance = float(task["temporal_distance_px"])
        if math.isfinite(median_nn):
            adaptive_distance = min(
                adaptive_distance,
                max(3.0, 0.45 * median_nn),
            )
        temporal_remnant = (
            current_non_dead
            & prior_strong
            & current["prev_mutual_nearest"].astype(bool)
            & (current["prev_distance"] <= adaptive_distance)
            & (current["temporal_area_ratio"] <= float(task["remnant_max_area_ratio"]))
            & (
                current["temporal_red_mass_ratio"]
                <= float(task["remnant_max_red_ratio"])
            )
        )
        current["temporal_support_frames"] = (
            (
                prior_strong
                & current["prev_mutual_nearest"].astype(bool)
                & (current["prev_distance"] <= adaptive_distance)
            ).astype(int)
            + (
                prior_2_strong
                & current["prev2_mutual_nearest"].astype(bool)
                & (current["prev2_distance"] <= 1.5 * adaptive_distance)
            ).astype(int)
            + (
                current["next_mutual_nearest"].astype(bool)
                & (current["next_distance"] <= adaptive_distance)
            ).astype(int)
            + (
                current["next2_mutual_nearest"].astype(bool)
                & (current["next2_distance"] <= 1.5 * adaptive_distance)
            ).astype(int)
        )
        current["temporal_track_confident"] = (
            temporal_remnant & current["temporal_support_frames"].ge(2)
        )
        distance_score = np.clip(
            1.0
            - pd.to_numeric(current["prev_distance"], errors="coerce").fillna(
                adaptive_distance * 2
            )
            / max(adaptive_distance, 1e-6),
            0.0,
            1.0,
        )
        current["temporal_match_confidence"] = np.clip(
            0.50 * distance_score
            + 0.25
            * np.clip(current["temporal_support_frames"] / 3.0, 0.0, 1.0)
            + 0.15
            * np.clip(1.0 - current["temporal_area_ratio"].fillna(1.0), 0.0, 1.0)
            + 0.10
            * np.clip(
                1.0 - current["temporal_red_mass_ratio"].fillna(1.0),
                0.0,
                1.0,
            ),
            0.0,
            1.0,
        )
        untreated_live_anchor = (
            (not bool(task["treated"]))
            & (float(task["elapsed_hours"]) > 0.0)
            & (current["final_state"].astype(str) == "live")
            & (current["prev_final_state"].astype(str) == "live")
            & (current["next_final_state"].astype(str) == "live")
            & (pd.to_numeric(current["dead_mask_id"], errors="coerce").fillna(0) == 0)
            & (pd.to_numeric(current["prev_dead_mask_id"], errors="coerce").fillna(0) == 0)
            & (pd.to_numeric(current["next_dead_mask_id"], errors="coerce").fillna(0) == 0)
            & (current["prev_distance"] <= float(task["stable_live_distance_px"]))
            & (current["next_distance"] <= float(task["stable_live_distance_px"]))
            & current["prev_mutual_nearest"].astype(bool)
            & current["next_mutual_nearest"].astype(bool)
            & (current["classification_confidence"].astype(str) == "high")
            & (current["prev_classification_confidence"].astype(str) == "high")
            & (current["next_classification_confidence"].astype(str) == "high")
        )
        current["proxy_type"] = np.select(
            [temporal_remnant, current_strong, untreated_live_anchor],
            [
                "temporal_dead_remnant",
                "blue_supported_dead",
                "untreated_live_anchor",
            ],
            default="unlabeled",
        )
    else:
        current_strong = strong_dead(current)
        d0_live_anchor = (
            current["final_state"].astype(str).eq("live")
            & countable
            & ~current["border_touching"]
            & pd.to_numeric(current["dead_mask_id"], errors="coerce").fillna(0).eq(0)
            & current["classification_confidence"].astype(str).eq("high")
        )
        current["proxy_type"] = np.select(
            [current_strong, d0_live_anchor],
            ["d0_dead_anchor", "d0_live_anchor"],
            default="d0_other",
        )
        current["prev_distance"] = np.nan
        current["next_distance"] = np.nan
        current["prev_mutual_nearest"] = False
        current["next_mutual_nearest"] = False
        current["temporal_area_ratio"] = np.nan
        current["temporal_red_ratio"] = np.nan
        current["temporal_red_mass_ratio"] = np.nan
        current["temporal_support_frames"] = 0
        current["temporal_track_confident"] = False
        current["temporal_match_confidence"] = 0.0

    current.insert(0, "branch", branch)
    current.insert(0, "cohort", cohort)
    for field in (
        "treated",
        "sentinel",
        "development_anchor",
        "holdout_anchor",
        "replicate_diagnostic",
        "ploidy",
        "cyclophosphamide",
        "doxorubicin_nm",
        "dose_bin",
        "density_proxy_bin",
    ):
        current[field] = task.get(field, "")
    current["split_group"] = current["well"].astype(str) + "_" + current["site"].astype(str)
    current["source_feature_path"] = str(current_path)
    current["field_record_path"] = str(task["record_path"])
    current["cell_mask_path"] = str(profiles["Combined"][mask_field])
    current["nucleus_core_mask_path"] = str(profiles["Nuclei"]["core_mask"])
    current["nucleus_extent_mask_path"] = str(profiles["Nuclei"]["extent_mask"])
    current["combined_raw_path"] = str(profiles["Combined"]["raw"])
    current["dead_raw_path"] = str(profiles["Dead"]["raw"])
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(shard.name + f".tmp.{os.getpid()}")
    current.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, shard)
    return summarize(current, cached=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_field_trajectory_metrics(
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "cohort",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "treated",
        "sentinel",
        "development_anchor",
        "holdout_anchor",
        "replicate_diagnostic",
        "ploidy",
        "cyclophosphamide",
        "doxorubicin_nm",
        "dose_bin",
    ]
    metadata = manifest[metadata_columns].drop_duplicates("key")
    fields = inventory.merge(
        metadata,
        on=["cohort", "key"],
        how="left",
        validate="many_to_one",
    )
    if fields[["well", "site", "elapsed_hours"]].isna().any().any():
        raise ValueError("Inventory-to-manifest merge lost field metadata")
    fields = fields.sort_values(
        ["branch", "well", "site", "elapsed_hours", "key"]
    ).reset_index(drop=True)
    group_keys = ["branch", "well", "site"]
    ratio_sources = {
        "field_count_ratio_to_peak": "field_countable_count",
        "field_area_ratio_to_peak": "field_median_area",
        "field_cytoplasm_ratio_to_peak": "field_median_cytoplasm_area",
        "field_red_mass_ratio_to_peak": "field_median_red_mass",
        "field_mask_fraction_ratio_to_peak": "field_mask_fraction",
    }
    for output, source in ratio_sources.items():
        values = pd.to_numeric(fields[source], errors="coerce")
        running_peak = values.groupby(
            [fields[column] for column in group_keys], sort=False
        ).cummax()
        fields[f"{source}_running_peak"] = running_peak
        fields[output] = (values / running_peak.replace(0, np.nan)).clip(0, 5)
    # Retained as descriptive metadata for historical reports only. It is not
    # consumed by the all-time refinement eligibility logic.
    fields["field_late"] = pd.to_numeric(
        fields["elapsed_hours"], errors="coerce"
    ).ge(72.0)
    return fields


def main() -> int:
    args = parse_args()
    args.classification_root = args.classification_root.resolve()
    args.d0_audit_root = (
        args.d0_audit_root.resolve()
        if args.d0_audit_root is not None
        else None
    )
    args.plate_map = args.plate_map.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.d0_fields <= 0 or args.workers <= 0:
        raise ValueError("--d0-fields and --workers must be positive")
    if args.trajectory_max_hours < args.trajectory_min_hours:
        raise ValueError("--trajectory-max-hours must be >= --trajectory-min-hours")
    required_paths = [args.classification_root, args.plate_map]
    if args.d0_audit_root is not None:
        required_paths.append(args.d0_audit_root)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = args.out_dir / "manifests" / "sample_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    shard_root = args.out_dir / "feature_cache" / "shards"
    tasks: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        for branch in BRANCHES:
            task = dict(row)
            task.update(
                {
                    "branch": branch,
                    "classification_root": str(args.classification_root),
                    "d0_audit_root": (
                        str(args.d0_audit_root)
                        if args.d0_audit_root is not None
                        else ""
                    ),
                    "temporal_distance_px": args.temporal_distance_px,
                    "remnant_max_area_ratio": args.remnant_max_area_ratio,
                    "remnant_max_red_ratio": args.remnant_max_red_ratio,
                    "stable_live_distance_px": args.stable_live_distance_px,
                    "force": args.force,
                    "shard_path": str(
                        shard_root / f"{row['cohort']}__{branch}__{row['key']}.csv.gz"
                    ),
                }
            )
            tasks.append(task)

    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                inventory.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve every failed field
                failures.append(
                    {
                        "cohort": str(task["cohort"]),
                        "branch": str(task["branch"]),
                        "key": str(task["key"]),
                        "error": repr(exc),
                    }
                )
            if completed == 1 or completed % 50 == 0 or completed == len(tasks):
                print(
                    f"feature_tasks={completed}/{len(tasks)} failures={len(failures)}",
                    flush=True,
                )

    inventory_frame = pd.DataFrame(inventory)
    if not inventory_frame.empty:
        inventory_frame = inventory_frame.sort_values(["cohort", "branch", "key"])
    inventory_path = args.out_dir / "feature_cache" / "shard_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_frame.to_csv(inventory_path, index=False)
    failures_path = args.out_dir / "feature_cache" / "failures.csv"
    pd.DataFrame(failures, columns=("cohort", "branch", "key", "error")).to_csv(
        failures_path, index=False
    )
    if failures:
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)} tasks; "
            f"see {failures_path}"
        )
    field_trajectory = build_field_trajectory_metrics(inventory_frame, manifest)
    field_trajectory_path = (
        args.out_dir / "feature_cache" / "field_trajectory_metrics.csv"
    )
    field_trajectory.to_csv(field_trajectory_path, index=False)
    summary = {
        "schema_version": 2,
        "classification_root": str(args.classification_root),
        "d0_audit_root": (
            str(args.d0_audit_root)
            if args.d0_audit_root is not None
            else None
        ),
        "plate_map": str(args.plate_map),
        "sample_manifest": str(manifest_path),
        "selection_seed": args.seed,
        "trajectory_min_hours": args.trajectory_min_hours,
        "trajectory_max_hours": args.trajectory_max_hours,
        "trajectory_wells": [
            str(value)
            for value in sorted(
                manifest.loc[
                    manifest["cohort"].eq("trajectory"),
                    "well",
                ].unique()
            )
        ],
        "temporal_carryforward_definition": {
            "previous_state": "strong_dead",
            "matching": (
                "mutual_nearest_centroid_across_previous_2_and_next_2_frames"
            ),
            "maximum_centroid_distance_px": args.temporal_distance_px,
            "adaptive_distance_cap": "45_percent_of_current_field_median_nn",
            "minimum_support_frames": 2,
            "maximum_current_to_previous_area_ratio": args.remnant_max_area_ratio,
            "maximum_current_to_previous_red_mass_ratio": args.remnant_max_red_ratio,
        },
        "untreated_live_anchor_definition": {
            "matching": "mutual_nearest_centroid_to_previous_and_next",
            "maximum_centroid_distance_px": args.stable_live_distance_px,
            "scope": (
                "all untreated nonzero time points with valid previous and next "
                "neighbors"
            ),
            "states": "high-confidence live at previous, current, and next time points",
            "dead_mask_required_absent": True,
        },
        "d0_fields": int((manifest["cohort"] == "d0_frozen").sum()),
        "trajectory_fields": int((manifest["cohort"] == "trajectory").sum()),
        "branches": list(BRANCHES),
        "expected_shards": len(tasks),
        "completed_shards": len(inventory),
        "failed_shards": len(failures),
        "proxy_counts": {
            field: int(pd.to_numeric(inventory_frame.get(field, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            for field in (
                "temporal_dead_remnant",
                "blue_supported_dead",
                "d0_live_anchor",
                "untreated_live_anchor",
                "unlabeled",
            )
        },
        "field_trajectory_metrics": str(field_trajectory_path),
        "e9_development_anchor": "E9 site 1 complete trajectory",
        "e9_holdout_anchor": "E9 sites 2-4 complete trajectories",
        "f9_replicate_diagnostic": "F9 sites 1-4 complete trajectories",
        "metric_semantics": "operational_proxy_not_biological_ground_truth",
    }
    write_json(args.out_dir / "feature_cache" / "dataset_summary.json", summary)
    print(f"sample_manifest={manifest_path}")
    print(f"shard_inventory={inventory_path}")
    print(f"field_trajectory_metrics={field_trajectory_path}")
    print(f"dataset_summary={args.out_dir / 'feature_cache' / 'dataset_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
