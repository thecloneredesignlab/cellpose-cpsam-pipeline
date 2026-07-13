#!/usr/bin/env python3
"""Merge per-field Dead/Combined-blue consensus reports into production tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dead-run", type=Path, required=True)
    parser.add_argument("--expected-keys", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"No rows for {path}")
    columns = fieldnames or list(rows[0])
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    summary_paths = sorted((args.dead_run / "field_summaries").glob("*.csv"))
    if not summary_paths:
        raise SystemExit(f"No consensus summaries under {args.dead_run / 'field_summaries'}")
    summaries: list[dict[str, str]] = []
    for path in summary_paths:
        rows = read_rows(path)
        if len(rows) != 1:
            raise SystemExit(f"Expected one row in {path}, found {len(rows)}")
        summaries.extend(rows)
    summaries.sort(key=lambda row: row["key"])
    keys = [row["key"] for row in summaries]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate keys in consensus field summaries")
    if args.expected_keys:
        expected = sorted({line.strip() for line in args.expected_keys.read_text().splitlines() if line.strip()})
        missing = sorted(set(expected) - set(keys))
        extra = sorted(set(keys) - set(expected))
        if missing or extra:
            raise SystemExit(f"Consensus key mismatch: missing={len(missing)} extra={len(extra)}")

    provenance: list[dict[str, str]] = []
    blue_only_candidates: list[dict[str, str]] = []
    for key in keys:
        path = args.dead_run / "object_provenance" / f"{key}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing object provenance: {path}")
        provenance.extend(read_rows(path))
        blue_path = args.dead_run / "blue_only_candidates" / f"{key}.csv"
        if not blue_path.is_file():
            raise SystemExit(f"Missing blue-only candidate table: {blue_path}")
        blue_only_candidates.extend(read_rows(blue_path))
    provenance.sort(key=lambda row: (row["key"], int(row["final_id"])))
    write_rows(args.dead_run / "dead_consensus_field_summary.csv", summaries)
    write_rows(args.dead_run / "dead_consensus_object_provenance.csv", provenance)
    blue_columns = [
        "key", "blue_id", "primary_overlap_fraction", "area", "dead_mean", "dead_p90", "dead_bg_median",
        "dead_p90_delta", "dead_snr", "selected_for_rescue", "decision",
    ]
    write_rows(args.dead_run / "dead_consensus_blue_only_candidates.csv", blue_only_candidates, blue_columns)

    segmentation_rows: list[dict[str, Any]] = []
    for row in summaries:
        segmentation_rows.append(
            {
                "image_path": row["dead_image"],
                "mask_path": row["final_mask"],
                "folder_profile": "Dead",
                "segmentation_source": "dead_combined_blue_consensus",
                "n_objects": row["final_objects"],
                "mask_fraction": row["final_mask_fraction"],
                "dead_primary_objects": row["dead_primary_objects"],
                "combined_blue_objects": row["combined_blue_objects"],
                "matched_objects": row["matched_objects"],
                "combined_blue_rescued_objects": row["combined_blue_rescued_objects"],
            }
        )
    write_rows(args.dead_run / "segmentation_summary.csv", segmentation_rows)

    source_counts = Counter(row["source"] for row in provenance)
    aggregate = {
        "n_fields": len(summaries),
        "dead_primary_objects": sum(int(row["dead_primary_objects"]) for row in summaries),
        "combined_blue_objects": sum(int(row["combined_blue_objects"]) for row in summaries),
        "matched_objects": sum(int(row["matched_objects"]) for row in summaries),
        "combined_blue_rescued_objects": sum(int(row["combined_blue_rescued_objects"]) for row in summaries),
        "combined_blue_rejected_objects": sum(int(row["combined_blue_rejected_objects"]) for row in summaries),
        "final_objects": sum(int(row["final_objects"]) for row in summaries),
        "provenance_counts": dict(source_counts),
    }
    (args.dead_run / "dead_consensus_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    (args.dead_run / "_SUCCESS").write_text(json.dumps(aggregate, sort_keys=True) + "\n")
    print(f"dead_consensus_merge_complete={len(summaries)} final_objects={aggregate['final_objects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
