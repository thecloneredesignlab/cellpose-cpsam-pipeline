#!/usr/bin/env python3
"""Build direct per-field input records for post-segmentation array workers.

The production result directories contain tens of thousands of files.  Array
workers must not rediscover those files independently.  This command indexes
each source directory once, validates it against the canonical key list, and
writes one small JSON record per key in a two-level records/<well>/ layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d+d\d+h\d+m)")
KEY_FULL_RE = re.compile(r"^[A-H]\d+_\d+_\d+d\d+h\d+m$")
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SCHEMA_VERSION = 1


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
normalize_timepoint = TIMEPOINT_HELPER.normalize_timepoint
select_keys = TIMEPOINT_HELPER.select_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-list", type=Path)
    selection.add_argument(
        "--timepoint",
        help=(
            "Select every field at one exact timepoint directly from the complete input/result roots. "
            "Accepts DDdHHhMMm or d0."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--nucleated-branch-root",
        type=Path,
        help="Predicted nucleated-only output root. Defaults to <run-root>/nucleated_only.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_key(path: Path) -> str:
    match = KEY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract field key from {path}")
    return match.group(1)


def read_keys(path: Path) -> list[str]:
    keys = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    invalid = [key for key in keys if KEY_FULL_RE.fullmatch(key) is None]
    if invalid:
        raise ValueError(f"Invalid keys in {path}: {invalid[:5]}")
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate keys in {path}: {duplicates[:5]}")
    if not keys:
        raise ValueError(f"No keys in {path}")
    return keys


def regular_files(directory: Path, suffixes: set[str] | None = None) -> Iterable[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            path = Path(entry.path)
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            yield path


def index_directory(
    directory: Path,
    *,
    suffixes: set[str] | None = None,
    name_suffix: str | None = None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in regular_files(directory, suffixes):
        if name_suffix is not None and not path.name.endswith(name_suffix):
            continue
        key = extract_key(path)
        if key in result:
            raise ValueError(f"Duplicate key {key} in {directory}: {result[key]} and {path}")
        result[key] = path.resolve()
    return result


def validate_keys(
    label: str,
    index: dict[str, Path],
    expected: set[str],
    *,
    allow_extra: bool = False,
) -> None:
    found = set(index)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or (extra and not allow_extra):
        raise ValueError(
            f"{label} key mismatch: expected={len(expected)} found={len(found)} "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mask_stem(path: Path) -> str:
    name = path.name
    for suffix in ("_cp_masks.tif", "_cp_masks.tiff", "_core_masks.tif", "_core_masks.tiff"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def task_list_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    run_root = args.run_root.resolve()
    task_list = args.task_list.resolve() if args.task_list is not None else None
    selected_timepoint = normalize_timepoint(args.timepoint) if args.timepoint is not None else None
    out_dir = args.out_dir.resolve()
    branch_root = (
        args.nucleated_branch_root.resolve()
        if args.nucleated_branch_root is not None
        else run_root / "nucleated_only"
    )
    success = out_dir / "_SUCCESS"
    if success.is_file() and not args.force:
        payload = json.loads(success.read_text())
        selection_matches = (
            payload.get("timepoint") == selected_timepoint
            and (
                task_list is None
                or payload.get("task_list_sha256") == task_list_sha256(task_list)
            )
        )
        if (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("input_root") == str(input_root)
            and payload.get("run_root") == str(run_root)
            and selection_matches
        ):
            print(f"manifest_already_complete=1 out_dir={out_dir}")
            return 0
        raise RuntimeError(f"Existing manifest does not match requested sources/selection: {out_dir}")

    indexes: dict[str, dict[str, Path]] = {
        "combined_raw": index_directory(input_root / "Combined", suffixes=IMAGE_SUFFIXES),
        "brightfield_raw": index_directory(input_root / "Brightfield", suffixes=IMAGE_SUFFIXES),
        "dead_raw": index_directory(input_root / "Dead", suffixes=IMAGE_SUFFIXES),
        "nuclei_raw": index_directory(input_root / "Nuclei", suffixes=IMAGE_SUFFIXES),
        "combined_mask": index_directory(
            run_root / "Combined" / "segmentations", name_suffix="_cp_masks.tif"
        ),
        "brightfield_mask": index_directory(
            run_root / "Brightfield" / "segmentations", name_suffix="_cp_masks.tif"
        ),
        "dead_mask": index_directory(run_root / "Dead" / "segmentations", name_suffix="_cp_masks.tif"),
        "nuclei_extent_mask": index_directory(
            run_root / "Nuclei" / "segmentations", name_suffix="_cp_masks.tif"
        ),
        "nuclei_core_mask": index_directory(
            run_root / "Nuclei" / "nucleus_core_seeds", name_suffix="_core_masks.tif"
        ),
    }
    if task_list is not None:
        keys = read_keys(task_list)
        if selected_timepoint is not None:
            keys = select_keys(keys, selected_timepoint)
    else:
        keys = select_keys(indexes["combined_raw"], selected_timepoint)
    if not keys:
        raise ValueError(
            f"No fields selected from {input_root} for timepoint={selected_timepoint or 'all'}"
        )
    expected = set(keys)
    source_key_counts = {label: len(index) for label, index in indexes.items()}
    for label, index in indexes.items():
        validate_keys(label, index, expected, allow_extra=selected_timepoint is not None)
        indexes[label] = {key: index[key] for key in keys}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "field_manifest.tsv"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    fields = [
        "key",
        "record_json",
        "combined_raw",
        "combined_mask",
        "brightfield_raw",
        "brightfield_mask",
        "dead_raw",
        "dead_mask",
        "nuclei_raw",
        "nuclei_extent_mask",
        "nuclei_core_mask",
        "nucleated_combined_mask",
        "nucleated_brightfield_mask",
    ]
    with temporary_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for position, key in enumerate(keys, start=1):
            combined_mask = indexes["combined_mask"][key]
            brightfield_mask = indexes["brightfield_mask"][key]
            record_path = out_dir / "records" / key.split("_", 1)[0] / f"{key}.json"
            record = {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "profiles": {
                    "Combined": {
                        "raw": str(indexes["combined_raw"][key]),
                        "original_mask": str(combined_mask),
                        "nucleated_mask": str(
                            branch_root / "Combined" / "segmentations" / combined_mask.name
                        ),
                        "stem": mask_stem(combined_mask),
                    },
                    "Brightfield": {
                        "raw": str(indexes["brightfield_raw"][key]),
                        "original_mask": str(brightfield_mask),
                        "nucleated_mask": str(
                            branch_root / "Brightfield" / "segmentations" / brightfield_mask.name
                        ),
                        "stem": mask_stem(brightfield_mask),
                    },
                    "Dead": {
                        "raw": str(indexes["dead_raw"][key]),
                        "mask": str(indexes["dead_mask"][key]),
                        "stem": mask_stem(indexes["dead_mask"][key]),
                    },
                    "Nuclei": {
                        "raw": str(indexes["nuclei_raw"][key]),
                        "extent_mask": str(indexes["nuclei_extent_mask"][key]),
                        "core_mask": str(indexes["nuclei_core_mask"][key]),
                        "stem": mask_stem(indexes["nuclei_extent_mask"][key]),
                    },
                },
            }
            atomic_json(record_path, record)
            writer.writerow(
                {
                    "key": key,
                    "record_json": str(record_path),
                    "combined_raw": record["profiles"]["Combined"]["raw"],
                    "combined_mask": record["profiles"]["Combined"]["original_mask"],
                    "brightfield_raw": record["profiles"]["Brightfield"]["raw"],
                    "brightfield_mask": record["profiles"]["Brightfield"]["original_mask"],
                    "dead_raw": record["profiles"]["Dead"]["raw"],
                    "dead_mask": record["profiles"]["Dead"]["mask"],
                    "nuclei_raw": record["profiles"]["Nuclei"]["raw"],
                    "nuclei_extent_mask": record["profiles"]["Nuclei"]["extent_mask"],
                    "nuclei_core_mask": record["profiles"]["Nuclei"]["core_mask"],
                    "nucleated_combined_mask": record["profiles"]["Combined"]["nucleated_mask"],
                    "nucleated_brightfield_mask": record["profiles"]["Brightfield"]["nucleated_mask"],
                }
            )
            if position % 1000 == 0 or position == len(keys):
                print(f"manifest_records={position}/{len(keys)}", flush=True)
    os.replace(temporary_manifest, manifest_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_fields": len(keys),
        "timepoint": selected_timepoint,
        "input_root": str(input_root),
        "run_root": str(run_root),
        "nucleated_branch_root": str(branch_root),
        "task_list": str(task_list) if task_list is not None else "",
        "task_list_sha256": task_list_sha256(task_list) if task_list is not None else "",
        "field_manifest": str(manifest_path),
        "records_root": str(out_dir / "records"),
        "source_key_counts": source_key_counts,
        "ignored_nonselected_key_counts": {
            label: count - len(keys) for label, count in source_key_counts.items()
        },
        "validated_selected_key_sets": sorted(indexes),
    }
    atomic_json(out_dir / "manifest_summary.json", summary)
    atomic_json(success, summary)
    print(f"postsegmentation_manifest_complete=1 n_fields={len(keys)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
