#!/usr/bin/env python3
"""Audit dead classification with an automated, classification-independent reference set.

This audit does not claim biological ground truth.  It reports two deliberately
conservative operational proxies:

* every RGB-live cell called dead is counted as a possible live false positive;
* high-confidence death references are defined from upstream RGB/dead-channel
  evidence before the final classification decision is inspected.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


FEATURE_SUFFIX = "_per_cell_fusion_features.csv"
PREDICTION_SUFFIX = "_per_cell_predictions.csv"
OVERLAY_SUFFIX = "_state_overlay.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        action="append",
        required=True,
        metavar="NAME=CLASSIFICATION_DIR",
        help="Repeat for every classification output to audit.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--direct-min-p90-delta", type=float, default=40.0)
    parser.add_argument("--direct-min-snr", type=float, default=100.0)
    parser.add_argument("--context-min-overlap", type=float, default=0.45)
    parser.add_argument("--context-min-p90-delta", type=float, default=10.0)
    return parser.parse_args()


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CLASSIFICATION_DIR, received: {value}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name:
        raise ValueError(f"Empty branch name: {value}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return name, path


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in {"", None} else 0


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def automated_reference_reasons(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    if as_int(row, "dead_mask_id") <= 0:
        return []
    reasons: list[str] = []
    if row.get("rgb_state") == "dead":
        reasons.append("rgb_dead_with_dead_channel_support")
    if (
        as_float(row, "dead_p90_delta") >= args.direct_min_p90_delta
        and as_float(row, "dead_snr") >= args.direct_min_snr
    ):
        reasons.append("strong_direct_dead_channel")
    if (
        row.get("rgb_state") != "live"
        and as_float(row, "dead_combined_overlap_fraction") >= args.context_min_overlap
        and as_float(row, "dead_p90_delta") >= args.context_min_p90_delta
    ):
        reasons.append("non_live_rgb_with_cell_level_dead_support")
    return reasons


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_feature_rows(branch_name: str, branch_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    feature_paths = sorted((branch_dir / "features").glob(f"*{FEATURE_SUFFIX}"))
    if not feature_paths:
        raise FileNotFoundError(f"No feature tables found under {branch_dir / 'features'}")
    for feature_path in feature_paths:
        with feature_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                reference_reasons = automated_reference_reasons(row, args)
                has_dead_evidence = as_int(row, "dead_mask_id") > 0
                reference_dead = bool(reference_reasons)
                detected = row.get("final_state") == "dead"
                conservative_live = row.get("rgb_state") == "live"
                output.append(
                    {
                        "branch": branch_name,
                        "image_id": row.get("image_id", ""),
                        "key": row.get("key", ""),
                        "combined_mask_id": as_int(row, "combined_mask_id"),
                        "rgb_state": row.get("rgb_state", ""),
                        "final_state": row.get("final_state", ""),
                        "final_reason": row.get("final_reason", ""),
                        "dead_mask_id": as_int(row, "dead_mask_id"),
                        "dead_p90_delta": as_float(row, "dead_p90_delta"),
                        "dead_snr": as_float(row, "dead_snr"),
                        "dead_combined_overlap_fraction": as_float(row, "dead_combined_overlap_fraction"),
                        "automated_reference_dead": reference_dead,
                        "automated_reference_reason": ";".join(reference_reasons),
                        "automated_reference_detected": reference_dead and detected,
                        "conservative_rgb_live_reference": conservative_live,
                        "possible_live_false_positive": conservative_live and detected,
                        "unresolved_dead_evidence": has_dead_evidence and not reference_dead,
                    }
                )
    return output


def verify_qc(branch_name: str, branch_dir: Path) -> list[dict[str, Any]]:
    feature_ids = {
        path.name[: -len(FEATURE_SUFFIX)]
        for path in (branch_dir / "features").glob(f"*{FEATURE_SUFFIX}")
    }
    prediction_ids = {
        path.name[: -len(PREDICTION_SUFFIX)]
        for path in (branch_dir / "predictions").glob(f"*{PREDICTION_SUFFIX}")
    }
    summary_ids = {
        path.name.removesuffix("_summary.csv")
        for path in (branch_dir / "summaries").glob("*_00d00h00m_summary.csv")
    }
    overlay_paths = {
        path.name[: -len(OVERLAY_SUFFIX)]: path
        for path in (branch_dir / "qc" / "label_overlays").glob(f"*{OVERLAY_SUFFIX}")
    }
    image_ids = sorted(feature_ids | prediction_ids | summary_ids | set(overlay_paths))
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        overlay = overlay_paths.get(image_id)
        qc_valid = False
        width = 0
        height = 0
        qc_error = ""
        if overlay is not None:
            try:
                with Image.open(overlay) as image:
                    width, height = image.size
                    image.verify()
                qc_valid = True
            except Exception as exc:  # pragma: no cover - data-dependent diagnostic
                qc_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "branch": branch_name,
                "image_id": image_id,
                "feature_exists": image_id in feature_ids,
                "prediction_exists": image_id in prediction_ids,
                "summary_exists": image_id in summary_ids,
                "qc_exists": overlay is not None,
                "qc_valid": qc_valid,
                "qc_width": width,
                "qc_height": height,
                "qc_path": str(overlay) if overlay is not None else "",
                "qc_error": qc_error,
                "complete": (
                    image_id in feature_ids
                    and image_id in prediction_ids
                    and image_id in summary_ids
                    and qc_valid
                ),
            }
        )
    return rows


def branch_metrics(branch: str, rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_dead = [row for row in rows if row["automated_reference_dead"]]
    reference_detected = [row for row in reference_dead if row["automated_reference_detected"]]
    conservative_live = [row for row in rows if row["conservative_rgb_live_reference"]]
    possible_live_fp = [row for row in conservative_live if row["possible_live_false_positive"]]
    reasons = Counter(
        reason
        for row in reference_dead
        for reason in str(row["automated_reference_reason"]).split(";")
        if reason
    )
    return {
        "branch": branch,
        "total_combined_cells": len(rows),
        "automated_reference_death_count": len(reference_dead),
        "automated_reference_death_detected": len(reference_detected),
        "automated_reference_death_missed": len(reference_dead) - len(reference_detected),
        "automated_reference_death_recall": len(reference_detected) / len(reference_dead) if reference_dead else 0.0,
        "conservative_rgb_live_reference_count": len(conservative_live),
        "possible_live_false_positive_count": len(possible_live_fp),
        "possible_live_false_positive_upper_bound": len(possible_live_fp) / len(conservative_live) if conservative_live else 0.0,
        "unresolved_dead_evidence_count": sum(bool(row["unresolved_dead_evidence"]) for row in rows),
        "reference_rgb_dead_count": reasons["rgb_dead_with_dead_channel_support"],
        "reference_strong_direct_count": reasons["strong_direct_dead_channel"],
        "reference_non_live_context_count": reasons["non_live_rgb_with_cell_level_dead_support"],
        "qc_image_count": len(qc_rows),
        "qc_complete_count": sum(bool(row["complete"]) for row in qc_rows),
        "qc_incomplete_count": sum(not bool(row["complete"]) for row in qc_rows),
    }


def main() -> int:
    args = parse_args()
    branches = [parse_branch(value) for value in args.branch]
    if len({name for name, _path in branches}) != len(branches):
        raise ValueError("Branch names must be unique")

    all_rows: list[dict[str, Any]] = []
    all_qc_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for name, path in branches:
        rows = read_feature_rows(name, path, args)
        qc_rows = verify_qc(name, path)
        all_rows.extend(rows)
        all_qc_rows.extend(qc_rows)
        metrics.append(branch_metrics(name, rows, qc_rows))

    per_cell_fields = list(all_rows[0])
    qc_fields = list(all_qc_rows[0])
    metric_fields = list(metrics[0])
    write_rows(args.out_dir / "automated_reference_per_cell.csv", all_rows, per_cell_fields)
    write_rows(
        args.out_dir / "automated_reference_missed.csv",
        [row for row in all_rows if row["automated_reference_dead"] and not row["automated_reference_detected"]],
        per_cell_fields,
    )
    write_rows(
        args.out_dir / "unresolved_dead_evidence.csv",
        [row for row in all_rows if row["unresolved_dead_evidence"]],
        per_cell_fields,
    )
    write_rows(args.out_dir / "qc_inventory.csv", all_qc_rows, qc_fields)
    write_rows(args.out_dir / "automated_reference_metrics.csv", metrics, metric_fields)

    for row in metrics:
        print(
            f"branch={row['branch']} reference_recall={row['automated_reference_death_recall']:.6f} "
            f"possible_live_fp_upper_bound={row['possible_live_false_positive_upper_bound']:.6f} "
            f"qc={row['qc_complete_count']}/{row['qc_image_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
