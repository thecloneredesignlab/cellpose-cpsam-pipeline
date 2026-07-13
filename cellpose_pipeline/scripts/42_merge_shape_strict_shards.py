#!/usr/bin/env python3
"""Merge per-field shape_strict shards into a production result tree.

The array workers deliberately write isolated shard directories.  This finalizer
validates that every requested field completed, links the immutable per-field
masks into one result tree, and aggregates the split diagnostics and QC sample.
Upstream segmentation and classification outputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


KEY_RE = re.compile(r"^[A-H]\d+_\d+_\d+d\d+h\d+m$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge completed shape_strict field shards.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--cell-run-root",
        type=Path,
        help="Root containing the Brightfield/Combined masks used for cell support. Defaults to --run-root.",
    )
    parser.add_argument(
        "--fusion-root",
        type=Path,
        help="Merged classification directory used upstream. Defaults to <run-root>/classification_fusion.",
    )
    parser.add_argument("--shape-root", type=Path, required=True)
    parser.add_argument("--task-list", type=Path, required=True)
    parser.add_argument("--candidate-tag", default="shape_strict")
    parser.add_argument("--max-qc-fields", type=int, default=24)
    return parser.parse_args()


def read_keys(path: Path) -> list[str]:
    keys = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    invalid = [key for key in keys if not KEY_RE.fullmatch(key)]
    if invalid:
        raise ValueError(f"Invalid keys in {path}: {invalid[:5]}")
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate keys in {path}: {duplicates[:5]}")
    if not keys:
        raise ValueError(f"No keys in {path}")
    return keys


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        if not fields:
            pass
        else:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=True)


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    os.symlink(relative_source, destination)


def single_match(directory: Path, pattern: str, key: str) -> Path:
    matches = sorted(directory.glob(pattern)) if directory.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} for {key} in {directory}, found {len(matches)}")
    return matches[0]


def median(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return float(statistics.median(values)) if values else 0.0


def event_rank(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        float(row.get("shape_gain", 0.0)),
        float(row.get("valley_drop_fraction", 0.0)),
        float(row.get("stability_fraction", 0.0)),
    )


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    cell_run_root = args.cell_run_root.resolve() if args.cell_run_root else run_root
    fusion_root = args.fusion_root.resolve() if args.fusion_root else run_root / "classification_fusion"
    shape_root = args.shape_root.resolve()
    task_list = args.task_list.resolve()
    keys = read_keys(task_list)
    shards_root = shape_root / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(shards_root)

    fusion_summary = fusion_root / "summaries" / "cell_count_summary.csv"
    if not fusion_summary.is_file():
        raise FileNotFoundError(
            f"Merged upstream classification result is missing; shape_strict must run after fusion merge: {fusion_summary}"
        )

    shape_root.mkdir(parents=True, exist_ok=True)
    for name in ("Brightfield", "Combined"):
        link_directory(cell_run_root / name, shape_root / name)
    link_directory(run_root / "Dead", shape_root / "Dead")
    link_directory(fusion_root, shape_root / "classification_fusion")
    link_directory(cell_run_root / "qc", shape_root / "upstream_qc")

    events: list[dict[str, str]] = []
    field_rows: list[dict[str, Any]] = []
    split_configs: Any | None = None
    total_baseline = 0
    total_candidate = 0
    total_evaluated = 0

    diagnostics_path = shape_root / "shape_multipeak_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_temporary = diagnostics_path.with_name(
        f".{diagnostics_path.name}.tmp.{os.getpid()}"
    )
    diagnostic_fields: list[str] | None = None
    diagnostic_count = 0
    with diagnostics_temporary.open("w", newline="") as diagnostic_handle:
        diagnostic_writer: csv.DictWriter[str] | None = None
        for index, key in enumerate(keys, start=1):
            shard = shards_root / key
            marker = shard / "_SUCCESS"
            if not marker.is_file():
                raise RuntimeError(f"Incomplete shape_strict shard: {key} ({marker} missing)")
            candidate_root = shard / "candidates" / args.candidate_tag
            extent = single_match(
                candidate_root / "Nuclei" / "segmentations", f"*{key}*_cp_masks.tif", key
            )
            core = single_match(
                candidate_root / "Nuclei" / "nucleus_core_seeds", f"*{key}*_core_masks.tif", key
            )
            link_file(extent, shape_root / "Nuclei" / "segmentations" / extent.name)
            link_file(core, shape_root / "Nuclei" / "nucleus_core_seeds" / core.name)

            shard_diagnostics_path = shard / "shape_multipeak_diagnostics.csv"
            if shard_diagnostics_path.is_file() and shard_diagnostics_path.stat().st_size > 0:
                with shard_diagnostics_path.open(newline="") as shard_handle:
                    reader = csv.DictReader(shard_handle)
                    current_fields = list(reader.fieldnames or [])
                    if diagnostic_fields is None:
                        diagnostic_fields = current_fields
                        diagnostic_writer = csv.DictWriter(
                            diagnostic_handle,
                            fieldnames=diagnostic_fields,
                            extrasaction="ignore",
                        )
                        diagnostic_writer.writeheader()
                    elif set(current_fields) != set(diagnostic_fields):
                        raise RuntimeError(
                            f"Diagnostic schema mismatch in {shard_diagnostics_path}: "
                            f"expected={diagnostic_fields} found={current_fields}"
                        )
                    if diagnostic_writer is not None:
                        for row in reader:
                            diagnostic_writer.writerow(row)
                            diagnostic_count += 1
            shard_events = [
                row
                for row in read_csv(shard / "split_events.csv")
                if row.get("candidate") == args.candidate_tag
            ]
            events.extend(shard_events)
            summary_path = candidate_root / "split_summary.json"
            with summary_path.open() as handle:
                summary = json.load(handle)
            baseline = int(summary["n_baseline_nuclei"])
            candidate = int(summary["n_candidate_nuclei"])
            evaluated = int(summary["n_shape_suspicious_evaluated"])
            splits = int(summary.get("n_split_parents", 0))
            if candidate - baseline != splits:
                raise RuntimeError(
                    f"Count invariant failed for {key}: baseline={baseline} candidate={candidate} splits={splits}"
                )
            total_baseline += baseline
            total_candidate += candidate
            total_evaluated += evaluated
            field_rows.append(
                {
                    "key": key,
                    "baseline_nuclei": baseline,
                    "shape_strict_nuclei": candidate,
                    "nuclei_added": splits,
                    "count_increase_fraction": splits / baseline if baseline else 0.0,
                    "shape_suspicious_evaluated": evaluated,
                    "split_events": len(shard_events),
                    "extent_mask": str(shape_root / "Nuclei" / "segmentations" / extent.name),
                    "core_mask": str(shape_root / "Nuclei" / "nucleus_core_seeds" / core.name),
                    "shard": str(shard),
                    "complete": True,
                }
            )
            current_configs = json.loads((shard / "split_configs.json").read_text())
            if split_configs is None:
                split_configs = current_configs
            elif current_configs != split_configs:
                raise RuntimeError(f"Split-config mismatch in shard {key}")
            if index % 1000 == 0 or index == len(keys):
                print(f"merged_fields={index}/{len(keys)}", flush=True)
    os.replace(diagnostics_temporary, diagnostics_path)

    if len(events) != total_candidate - total_baseline:
        raise RuntimeError(
            f"Global count invariant failed: events={len(events)} added={total_candidate - total_baseline}"
        )

    event_fields = [
        "candidate", "key", "parent_nucleus_id", "retained_child_id", "new_child_id",
        "parent_area", "parent_solidity", "parent_circularity", "parent_axis_ratio",
        "parent_shape_score", "peak1_y", "peak1_x", "peak2_y", "peak2_x",
        "peak_distance", "min_peak_snr", "third_peak_ratio", "valley_drop_snr",
        "valley_drop_fraction", "stability_fraction", "retained_child_area",
        "new_child_area", "min_child_fraction", "weighted_child_shape_score", "shape_gain",
        "cell_support_fraction", "children_same_cell_pair", "child1_combined_cell",
        "child1_bf_cell", "child1_cell_supported", "child2_combined_cell", "child2_bf_cell",
        "child2_cell_supported",
    ]
    write_rows(shape_root / "split_events.csv", events, event_fields)
    write_rows(shape_root / "field_summary.csv", field_rows)
    write_json(shape_root / "split_configs.json", split_configs or [])

    ranked_keys: list[tuple[tuple[float, float, float], str]] = []
    events_by_key: dict[str, list[dict[str, str]]] = {}
    for row in events:
        events_by_key.setdefault(row["key"], []).append(row)
    for key, key_events in events_by_key.items():
        ranked_keys.append((max(event_rank(row) for row in key_events), key))
    ranked_keys.sort(reverse=True)
    qc_keys = [key for _rank, key in ranked_keys[: max(0, args.max_qc_fields)]]
    qc_rows = [
        {
            "qc_rank": index,
            "key": key,
            "n_split_events": len(events_by_key[key]),
            "max_shape_gain": max(float(row["shape_gain"]) for row in events_by_key[key]),
            "max_valley_drop_fraction": max(
                float(row["valley_drop_fraction"]) for row in events_by_key[key]
            ),
            "min_stability_fraction": min(
                float(row["stability_fraction"]) for row in events_by_key[key]
            ),
            "expected_qc_png": str(
                shape_root / "qc" / "shape_strict" / f"{key}_{args.candidate_tag}_shape_split_qc.png"
            ),
        }
        for index, key in enumerate(qc_keys, start=1)
    ]
    write_rows(shape_root / "qc" / "qc_selection.csv", qc_rows)
    (shape_root / "qc" / "qc_keys.txt").write_text("".join(f"{key}\n" for key in qc_keys))

    summary = {
        "candidate": args.candidate_tag,
        "run_root": str(run_root),
        "cell_run_root": str(cell_run_root),
        "fusion_root": str(fusion_root),
        "shape_root": str(shape_root),
        "task_list": str(task_list),
        "upstream_fusion_summary": str(fusion_summary),
        "n_fields": len(keys),
        "n_baseline_nuclei": total_baseline,
        "n_shape_strict_nuclei": total_candidate,
        "n_split_parents": len(events),
        "n_fields_with_splits": len(events_by_key),
        "n_shape_suspicious_evaluated": total_evaluated,
        "n_diagnostic_rows": diagnostic_count,
        "count_increase_fraction": (
            (total_candidate - total_baseline) / total_baseline if total_baseline else 0.0
        ),
        "median_peak_distance": median(events, "peak_distance"),
        "median_min_peak_snr": median(events, "min_peak_snr"),
        "median_valley_drop_snr": median(events, "valley_drop_snr"),
        "median_valley_drop_fraction": median(events, "valley_drop_fraction"),
        "median_stability_fraction": median(events, "stability_fraction"),
        "median_shape_gain": median(events, "shape_gain"),
        "median_cell_support_fraction": median(events, "cell_support_fraction"),
        "n_qc_fields": len(qc_keys),
        "validation": {
            "all_requested_shards_complete": True,
            "one_extent_and_core_mask_per_field": True,
            "count_increase_matches_split_events": True,
            "upstream_fusion_merge_complete": True,
            "upstream_outputs_unchanged": True,
        },
    }
    write_json(shape_root / "shape_strict_summary.json", summary)
    lines = [
        "# Production shape_strict summary",
        "",
        f"- fields: {summary['n_fields']}",
        f"- baseline nuclei: {summary['n_baseline_nuclei']}",
        f"- shape_strict nuclei: {summary['n_shape_strict_nuclei']}",
        f"- split parents: {summary['n_split_parents']}",
        f"- fields with splits: {summary['n_fields_with_splits']}",
        f"- count increase: {summary['count_increase_fraction']:.4%}",
        f"- suspicious shapes evaluated: {summary['n_shape_suspicious_evaluated']}",
        f"- median peak stability: {summary['median_stability_fraction']:.1%}",
        f"- median shape gain: {summary['median_shape_gain']:.3f}",
        f"- QC fields selected: {summary['n_qc_fields']}",
        "- upstream segmentation and classification outputs were not overwritten",
        "",
        "Outputs: `field_summary.csv`, `split_events.csv`, "
        "`shape_multipeak_diagnostics.csv`, `shape_strict_summary.json`, and `qc/`.",
    ]
    (shape_root / "shape_strict_report.md").write_text("\n".join(lines) + "\n")
    print(f"shape_root={shape_root}", flush=True)
    print(f"shape_summary={shape_root / 'shape_strict_summary.json'}", flush=True)
    print(f"field_summary={shape_root / 'field_summary.csv'}", flush=True)
    print(f"split_events={shape_root / 'split_events.csv'}", flush=True)
    print(f"qc_selection={shape_root / 'qc' / 'qc_selection.csv'}", flush=True)
    print("shape_strict_merge_complete=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
