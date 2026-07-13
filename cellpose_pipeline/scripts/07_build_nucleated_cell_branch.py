#!/usr/bin/env python3
"""Create BF/Combined masks retaining only instances with nuclear evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--key", action="append")
    parser.add_argument(
        "--field-record",
        type=Path,
        help="Direct per-field JSON record. Bypasses all full-directory indexes.",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        help="Expected key list used to validate --merge-shards-only completeness.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-extent-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain a cell when an extent centroid is inside even if no conservative core centroid is inside.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render-qc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--merge-shards-only",
        action="store_true",
        help="Merge per-key reports already written under out-root/shards without reading masks.",
    )
    parser.add_argument("--max-merge-qc-fields", type=int, default=24)
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract key from {path}")
    return match.group(1)


def index_files(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate key {key} in {directory}")
        result[key] = path.resolve()
    return result


def read_task_keys(path: Path) -> list[str]:
    keys = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    invalid = [key for key in keys if KEY_RE.fullmatch(key) is None]
    if invalid:
        raise ValueError(f"Invalid keys in {path}: {invalid[:5]}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate keys in {path}")
    if not keys:
        raise ValueError(f"No keys in {path}")
    return keys


def direct_indexes(path: Path, requested_keys: list[str] | None) -> tuple[
    dict[str, Path], dict[str, Path], dict[str, dict[str, Path]], dict[str, dict[str, Path]]
]:
    payload = json.loads(path.read_text())
    key = str(payload.get("key", ""))
    if KEY_RE.fullmatch(key) is None:
        raise ValueError(f"Invalid key in field record {path}: {key!r}")
    if requested_keys and set(requested_keys) != {key}:
        raise ValueError(f"Requested key(s) {requested_keys} do not match field record key {key}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Missing profiles object in {path}")
    nuclei = profiles.get("Nuclei", {})
    extent = Path(str(nuclei.get("extent_mask", "")))
    core = Path(str(nuclei.get("core_mask", "")))
    profile_indexes: dict[str, dict[str, Path]] = {}
    raw_indexes: dict[str, dict[str, Path]] = {}
    for profile in ("Combined", "Brightfield"):
        values = profiles.get(profile, {})
        mask = Path(str(values.get("original_mask", "")))
        raw = Path(str(values.get("raw", "")))
        if not mask.is_file():
            raise FileNotFoundError(mask)
        if not raw.is_file():
            raise FileNotFoundError(raw)
        profile_indexes[profile] = {key: mask}
        raw_indexes[profile] = {key: raw}
    if not extent.is_file():
        raise FileNotFoundError(extent)
    if not core.is_file():
        raise FileNotFoundError(core)
    return {key: extent}, {key: core}, profile_indexes, raw_indexes


def read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(tifffile.imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}: {path}")
    return mask.astype(np.int32, copy=False)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32 if int(mask.max()) > np.iinfo(np.uint16).max else np.uint16
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")
    tifffile.imwrite(temporary, mask.astype(dtype, copy=False))
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def label_centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int32)
    if labels.size == 0:
        return labels, np.zeros((0, 2), dtype=np.float64)
    max_label = int(mask.max())
    areas = np.bincount(mask.ravel(), minlength=max_label + 1)
    yy, xx = np.indices(mask.shape)
    sum_y = np.bincount(mask.ravel(), weights=yy.ravel(), minlength=max_label + 1)
    sum_x = np.bincount(mask.ravel(), weights=xx.ravel(), minlength=max_label + 1)
    centroids = np.column_stack(
        [sum_y[labels] / np.maximum(areas[labels], 1), sum_x[labels] / np.maximum(areas[labels], 1)]
    )
    return labels, centroids


def centroid_counts_inside(cell_mask: np.ndarray, nuclear_mask: np.ndarray) -> dict[int, int]:
    _labels, centroids = label_centroids(nuclear_mask)
    if centroids.size == 0:
        return {}
    yy = np.clip(np.rint(centroids[:, 0]).astype(int), 0, cell_mask.shape[0] - 1)
    xx = np.clip(np.rint(centroids[:, 1]).astype(int), 0, cell_mask.shape[1] - 1)
    cell_ids = cell_mask[yy, xx]
    result: dict[int, int] = {}
    for cell_id in cell_ids:
        cell_id = int(cell_id)
        if cell_id > 0:
            result[cell_id] = result.get(cell_id, 0) + 1
    return result


def filter_mask(
    key: str,
    profile: str,
    original: np.ndarray,
    core: np.ndarray,
    extent: np.ndarray,
    allow_extent_fallback: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    core_counts = centroid_counts_inside(original, core)
    extent_counts = centroid_counts_inside(original, extent)
    labels = np.unique(original)
    labels = labels[labels > 0].astype(np.int32)
    max_label = int(original.max()) if original.size else 0
    areas = np.bincount(original.ravel(), minlength=max_label + 1)
    mapping = np.zeros(max_label + 1, dtype=np.uint32)
    decisions: list[dict[str, Any]] = []
    next_label = 1
    for old_label in labels:
        old_label = int(old_label)
        n_core = int(core_counts.get(old_label, 0))
        n_extent = int(extent_counts.get(old_label, 0))
        keep = n_core > 0 or (allow_extent_fallback and n_extent > 0)
        if keep:
            mapping[old_label] = next_label
            new_label = next_label
            next_label += 1
            reason = "core_centroid_inside" if n_core > 0 else "extent_centroid_fallback"
        else:
            new_label = 0
            reason = "no_nucleus_centroid_inside"
        decisions.append(
            {
                "key": key,
                "profile": profile,
                "original_cell_id": old_label,
                "filtered_cell_id": new_label,
                "area": int(areas[old_label]),
                "n_core_centroids_inside": n_core,
                "n_extent_centroids_inside": n_extent,
                "keep": keep,
                "reason": reason,
            }
        )
    filtered = mapping[original]
    kept = next_label - 1
    summary = {
        "key": key,
        "profile": profile,
        "original_cells": int(labels.size),
        "retained_cells": kept,
        "removed_cells": int(labels.size) - kept,
        "retained_fraction": kept / labels.size if labels.size else 0.0,
        "removed_mask_fraction": float(np.mean((original > 0) & (filtered == 0))),
        "core_supported_cells": sum(1 for row in decisions if row["n_core_centroids_inside"] > 0),
        "extent_fallback_cells": sum(1 for row in decisions if row["reason"] == "extent_centroid_fallback"),
    }
    return filtered, decisions, summary


def normalize_rgb(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    data = raw[..., :3].astype(np.float32)
    low, high = np.percentile(data, [1, 99])
    return np.clip((data - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def render_filter_qc(
    raw_path: Path,
    original: np.ndarray,
    filtered: np.ndarray,
    core: np.ndarray,
    out_path: Path,
) -> None:
    rgb = normalize_rgb(tifffile.imread(raw_path))
    retained = filtered > 0
    removed = (original > 0) & ~retained
    core_fg = core > 0
    overlay = rgb.copy()
    overlay[retained] = 0.65 * overlay[retained] + 0.35 * np.array([0.0, 1.0, 1.0])
    overlay[removed] = 0.45 * overlay[removed] + 0.55 * np.array([1.0, 0.0, 0.8])
    overlay[core_fg] = 0.25 * overlay[core_fg] + 0.75 * np.array([1.0, 1.0, 0.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8)).save(out_path)


def aggregate_branch(field_summaries: list[dict[str, Any]], n_fields: int, allow_extent_fallback: bool) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "n_fields": n_fields,
        "allow_extent_fallback": allow_extent_fallback,
        "profiles": {},
    }
    for profile in ("Combined", "Brightfield"):
        rows = [row for row in field_summaries if row["profile"] == profile]
        aggregate["profiles"][profile] = {
            "original_cells": sum(int(row["original_cells"]) for row in rows),
            "retained_cells": sum(int(row["retained_cells"]) for row in rows),
            "removed_cells": sum(int(row["removed_cells"]) for row in rows),
            "core_supported_cells": sum(int(row["core_supported_cells"]) for row in rows),
            "extent_fallback_cells": sum(int(row["extent_fallback_cells"]) for row in rows),
        }
    return aggregate


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_shards(
    out_root: Path,
    run_root: Path,
    input_root: Path,
    render_qc: bool,
    max_qc_fields: int,
    task_list: Path | None,
) -> int:
    if task_list is not None:
        expected_keys = read_task_keys(task_list)
        shard_roots = [out_root / "shards" / key for key in expected_keys]
    else:
        shard_roots = sorted(path for path in (out_root / "shards").glob("*") if path.is_dir())
        expected_keys = [path.name for path in shard_roots]
    if not shard_roots:
        raise SystemExit(f"No nucleated-only shards found under {out_root / 'shards'}")
    field_summaries: list[dict[str, Any]] = []
    segmentation_rows: dict[str, list[dict[str, Any]]] = {"Combined": [], "Brightfield": []}
    keys: set[str] = set()
    fallback_values: set[bool] = set()
    decisions_path = out_root / "filter_decisions.csv"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_temporary = decisions_path.with_name(f".{decisions_path.name}.tmp.{os.getpid()}")
    decision_fields: list[str] | None = None
    decision_count = 0
    with decisions_temporary.open("w", newline="") as decision_handle:
        decision_writer: csv.DictWriter[str] | None = None
        for shard_number, shard_root in enumerate(shard_roots, start=1):
            success_path = shard_root / "_SUCCESS"
            if not success_path.is_file():
                raise SystemExit(f"Incomplete nucleated-only shard: {shard_root}")
            payload = json.loads(success_path.read_text())
            fallback_values.add(bool(payload["allow_extent_fallback"]))
            shard_decisions_path = shard_root / "filter_decisions.csv"
            with shard_decisions_path.open(newline="") as shard_handle:
                reader = csv.DictReader(shard_handle)
                current_fields = list(reader.fieldnames or [])
                if decision_fields is None:
                    decision_fields = current_fields
                    if decision_fields:
                        decision_writer = csv.DictWriter(
                            decision_handle, fieldnames=decision_fields, extrasaction="ignore"
                        )
                        decision_writer.writeheader()
                elif current_fields != decision_fields:
                    raise RuntimeError(
                        f"Decision schema mismatch in {shard_decisions_path}: "
                        f"expected={decision_fields} found={current_fields}"
                    )
                if decision_writer is not None:
                    for row in reader:
                        decision_writer.writerow(row)
                        decision_count += 1
            shard_field_rows = read_rows(shard_root / "field_summary.csv")
            field_summaries.extend(shard_field_rows)
            keys.update(str(row["key"]) for row in shard_field_rows)
            for profile in segmentation_rows:
                segmentation_rows[profile].extend(
                    read_rows(shard_root / profile / "segmentation_summary.csv")
                )
            if shard_number % 1000 == 0 or shard_number == len(shard_roots):
                print(f"merged_nucleated_shards={shard_number}/{len(shard_roots)}", flush=True)
    if keys != set(expected_keys):
        missing = sorted(set(expected_keys) - keys)
        extra = sorted(keys - set(expected_keys))
        raise RuntimeError(f"Nucleated shard key mismatch: missing={missing[:5]} extra={extra[:5]}")
    os.replace(decisions_temporary, decisions_path)
    if len(fallback_values) != 1:
        raise SystemExit(f"Mixed allow_extent_fallback settings across shards: {sorted(fallback_values)}")
    write_rows(out_root / "field_summary.csv", field_summaries)
    for profile, rows in segmentation_rows.items():
        rows.sort(key=lambda row: extract_key(Path(row["image_path"])))
        write_rows(out_root / profile / "segmentation_summary.csv", rows)
    aggregate = aggregate_branch(field_summaries, len(keys), fallback_values.pop())
    aggregate["filter_decision_rows"] = decision_count
    qc_keys: list[str] = []
    if render_qc and max_qc_fields > 0:
        ranked = sorted(
            field_summaries,
            key=lambda row: float(row["removed_mask_fraction"]),
            reverse=True,
        )
        for row in ranked:
            key = str(row["key"])
            if key not in qc_keys:
                qc_keys.append(key)
            if len(qc_keys) >= max_qc_fields:
                break
        core_index = index_files(run_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif")
        for profile in ("Combined", "Brightfield"):
            original_index = index_files(run_root / profile / "segmentations", "*_cp_masks.tif")
            filtered_index = index_files(out_root / profile / "segmentations", "*_cp_masks.tif")
            raw_index = index_files(input_root / profile, "*.tif")
            for key in qc_keys:
                render_filter_qc(
                    raw_index[key],
                    read_mask(original_index[key]),
                    read_mask(filtered_index[key]),
                    read_mask(core_index[key]),
                    out_root / "qc" / profile / f"{key}_nucleated_filter_overlay.png",
                )
        (out_root / "qc" / "qc_keys.txt").write_text("".join(f"{key}\n" for key in qc_keys))
    aggregate["qc_fields"] = qc_keys
    write_json_atomic(out_root / "branch_summary.json", aggregate)
    write_json_atomic(out_root / "_SUCCESS", aggregate)
    print(f"nucleated_branch_merge_complete={len(keys)} out_root={out_root}")
    return 0


def main() -> int:
    args = parse_args()
    if args.merge_shards_only:
        return merge_shards(
            args.out_root,
            args.run_root,
            args.input_root,
            args.render_qc,
            args.max_merge_qc_fields,
            args.task_list,
        )
    if args.field_record is not None:
        extent_index, core_index, profile_indexes, raw_indexes = direct_indexes(
            args.field_record, args.key
        )
        record_source = "field_record"
    else:
        extent_index = index_files(args.run_root / "Nuclei" / "segmentations", "*_cp_masks.tif")
        core_index = index_files(args.run_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif")
        profile_indexes = {
            profile: index_files(args.run_root / profile / "segmentations", "*_cp_masks.tif")
            for profile in ("Combined", "Brightfield")
        }
        raw_indexes = {
            profile: index_files(args.input_root / profile, "*.tif")
            for profile in ("Combined", "Brightfield")
        }
        record_source = "directory_discovery"
    keys = sorted(set(extent_index) & set(core_index) & set(profile_indexes["Combined"]) & set(profile_indexes["Brightfield"]))
    if args.key:
        wanted = set(args.key)
        keys = [key for key in keys if key in wanted]
    if args.limit is not None:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit("No common Nuclei/Combined/Brightfield fields")
    print(f"record_source={record_source} n_selected={len(keys)}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, Any]] = []
    field_summaries: list[dict[str, Any]] = []
    segmentation_rows: dict[str, list[dict[str, Any]]] = {"Combined": [], "Brightfield": []}
    for index, key in enumerate(keys, start=1):
        extent = read_mask(extent_index[key])
        core = read_mask(core_index[key])
        for profile in ("Combined", "Brightfield"):
            source = profile_indexes[profile][key]
            original = read_mask(source)
            if not (original.shape == extent.shape == core.shape):
                raise ValueError(f"Shape mismatch for {profile}/{key}")
            filtered, next_decisions, summary = filter_mask(
                key,
                profile,
                original,
                core,
                extent,
                args.allow_extent_fallback,
            )
            out_mask = args.out_root / profile / "segmentations" / source.name
            if args.force or not out_mask.exists():
                write_mask(out_mask, filtered)
            decisions.extend(next_decisions)
            field_summaries.append(summary)
            raw_path = raw_indexes[profile][key]
            segmentation_rows[profile].append(
                {
                    "image_path": str(raw_path),
                    "mask_path": str(out_mask),
                    "folder_profile": profile,
                    "branch": "nucleated_only",
                    "n_objects": summary["retained_cells"],
                    "mask_fraction": float(np.mean(filtered > 0)),
                }
            )
            if args.render_qc:
                render_filter_qc(
                    raw_path,
                    original,
                    filtered,
                    core,
                    args.out_root / "qc" / profile / f"{key}_nucleated_filter_overlay.png",
                )
        print(f"field={index}/{len(keys)} key={key}", flush=True)

    report_root = args.out_root / "shards" / keys[0] if len(keys) == 1 else args.out_root
    write_rows(report_root / "filter_decisions.csv", decisions)
    write_rows(report_root / "field_summary.csv", field_summaries)
    for profile, rows in segmentation_rows.items():
        write_rows(report_root / profile / "segmentation_summary.csv", rows)
    aggregate = aggregate_branch(field_summaries, len(keys), args.allow_extent_fallback)
    write_json_atomic(report_root / "branch_summary.json", aggregate)
    write_json_atomic(report_root / "_SUCCESS", aggregate)
    print(f"nucleated_branch_complete={len(keys)} out_root={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
