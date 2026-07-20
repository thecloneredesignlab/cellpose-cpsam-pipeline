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
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


FEATURE_SUFFIX = "_per_cell_fusion_features.csv"
PREDICTION_SUFFIX = "_per_cell_predictions.csv"
OVERLAY_SUFFIX = "_state_overlay.png"
DEAD_OBJECT_SUFFIX = "_dead_object_features.csv"
DEAD_OBJECT_OVERLAY_SUFFIX = "_dead_object_overlay.png"
OVERLAP_OVERLAY_SUFFIX = "_overlap_state_overlay.png"
CELL_STATE_MASK_SUFFIX = "_cell_state_masks.tif"
CONFIRMED_DEAD_MASK_SUFFIX = "_confirmed_dead_masks.tif"
OVERLAP_MASK_SUFFIX = "_cell_dead_overlap_masks.tif"

DEAD_OBJECT_AUDIT_FIELDS = [
    "branch",
    "image_id",
    "key",
    "dead_mask_id",
    "assigned_combined_mask_id",
    "association_relation",
    "keep_signal",
    "p90_delta",
    "snr",
    "automated_reference_dead_object",
    "automated_reference_detected",
    "supplemental_dead_object",
]

MASTER_ANNOTATION_FIELDS = [
    "analysis_branch",
    "annotation_level",
    "record_id",
    "linked_cell_record_id",
    "image_id",
    "key",
    "well",
    "site",
    "day",
    "hour",
    "minute",
    "elapsed_hours",
    "combined_mask_id",
    "dead_mask_id",
    "final_classification",
    "detailed_classification",
    "classification_confidence",
    "uncertainty_flag",
    "uncertainty_types",
    "countable",
    "confirmed_death_object",
    "supplemental_death_object",
    "association_relation",
    "nucleus_supported_relation",
    "linked_death_object_count",
    "confirmed_linked_death_object_count",
    "linked_cell_final_classification",
    "linked_cell_detailed_classification",
]

CELL_CLASSIFICATIONS = {"live", "dead", "transitional", "uncertain", "artifact"}
DEATH_RELATIONS = {
    "same_cell",
    "dead_only",
    "overlapping_live_dead_multi_nucleus",
    "live_with_death_signal",
    "adjacent_or_overlapping_dead_uncertain",
    "merged_multiple_objects",
    "adjacent_candidate",
    "unassigned_candidate",
    "rejected_signal",
}


TIMEPOINT_HELPER_PATH = Path(__file__).resolve().parents[1] / "_shared" / "timepoint_selection.py"
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
    parser.add_argument(
        "--annotation-out-dir",
        type=Path,
        help=(
            "Optional destination for final_multilevel_annotations.csv and "
            "final_annotation_summary.csv; defaults to --out-dir."
        ),
    )
    parser.add_argument("--direct-min-p90-delta", type=float, default=40.0)
    parser.add_argument("--direct-min-snr", type=float, default=100.0)
    parser.add_argument("--context-min-overlap", type=float, default=0.45)
    parser.add_argument("--context-min-p90-delta", type=float, default=10.0)
    parser.add_argument(
        "--timepoint",
        default="d0",
        help="Exact timepoint to audit from a complete classification directory (default: d0).",
    )
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


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


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


def selected_paths(paths: Any, timepoint: str) -> list[Path]:
    selected: list[Path] = []
    for path in paths:
        try:
            _key, observed = extract_key_and_timepoint(path)
        except ValueError:
            continue
        if observed == timepoint:
            selected.append(path)
    return sorted(selected)


def read_feature_rows(branch_name: str, branch_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    feature_paths = selected_paths(
        (branch_dir / "features").glob(f"*{FEATURE_SUFFIX}"),
        args.timepoint,
    )
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


def read_dead_object_rows(
    branch_name: str,
    branch_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    object_paths = selected_paths(
        (branch_dir / "dead_objects").glob(f"*{DEAD_OBJECT_SUFFIX}"),
        args.timepoint,
    )
    if not object_paths:
        raise FileNotFoundError(f"No dead-object tables found under {branch_dir / 'dead_objects'}")
    for object_path in object_paths:
        with object_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                reference_dead = (
                    as_bool(row, "keep_signal")
                    and as_float(row, "p90_delta") >= args.direct_min_p90_delta
                    and as_float(row, "snr") >= args.direct_min_snr
                )
                detected = as_bool(row, "confirmed_dead_object")
                output.append(
                    {
                        "branch": branch_name,
                        "image_id": row.get("image_id", ""),
                        "key": row.get("key", ""),
                        "dead_mask_id": as_int(row, "dead_mask_id"),
                        "assigned_combined_mask_id": as_int(row, "assigned_combined_mask_id"),
                        "association_relation": row.get("association_relation", ""),
                        "keep_signal": as_bool(row, "keep_signal"),
                        "p90_delta": as_float(row, "p90_delta"),
                        "snr": as_float(row, "snr"),
                        "automated_reference_dead_object": reference_dead,
                        "automated_reference_detected": reference_dead and detected,
                        "supplemental_dead_object": as_bool(row, "supplemental_dead_object"),
                    }
                )
    return output


def read_source_tables(
    directory: Path,
    pattern: str,
    label: str,
    timepoint: str,
) -> tuple[list[str], list[dict[str, str]]]:
    paths = selected_paths(directory.glob(pattern), timepoint)
    if not paths:
        raise FileNotFoundError(f"No {label} tables found under {directory}")
    expected_fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if not fields or len(fields) != len(set(fields)):
                raise RuntimeError(f"Invalid or duplicate {label} fields in {path}: {fields}")
            if expected_fields is None:
                expected_fields = fields
            elif fields != expected_fields:
                raise RuntimeError(
                    f"{label} schema mismatch in {path}: expected {expected_fields}, found {fields}"
                )
            for row in reader:
                if None in row:
                    raise RuntimeError(f"Unexpected extra {label} values in {path}: {row[None]}")
                rows.append({field: row.get(field, "") for field in fields})
    return expected_fields or [], rows


def cell_record_id(branch: str, key: str, combined_mask_id: int) -> str:
    return f"{branch}:cell:{key}:{combined_mask_id}"


def death_record_id(branch: str, key: str, dead_mask_id: int) -> str:
    return f"{branch}:death_object:{key}:{dead_mask_id}"


def joined_values(values: list[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def cell_detailed_classification(
    final_state: str,
    linked_confirmed_objects: list[dict[str, str]],
) -> str:
    relations = {row.get("association_relation", "") for row in linked_confirmed_objects}
    if final_state == "dead":
        if "same_cell" in relations:
            return "dead_same_cell"
        if linked_confirmed_objects:
            return "dead_with_confirmed_death_object"
        return "dead_without_linked_confirmed_object"
    if final_state == "live":
        if "overlapping_live_dead_multi_nucleus" in relations:
            return "live_overlapping_dead_multi_nucleus"
        if "live_with_death_signal" in relations:
            return "live_with_death_signal"
        if "adjacent_or_overlapping_dead_uncertain" in relations:
            return "live_with_uncertain_death_relation"
        if linked_confirmed_objects:
            return "live_with_confirmed_supplemental_death"
        return "live_without_confirmed_death"
    return final_state


def death_detailed_classification(row: dict[str, str]) -> str:
    relation = row.get("association_relation", "")
    confirmed = as_bool(row, "confirmed_dead_object")
    if relation not in DEATH_RELATIONS:
        raise RuntimeError(f"Unknown Death association_relation: {relation!r}")
    if confirmed:
        return {
            "same_cell": "confirmed_same_cell_dead",
            "dead_only": "confirmed_dead_only",
            "overlapping_live_dead_multi_nucleus": "confirmed_overlapping_live_dead_multi_nucleus",
            "live_with_death_signal": "confirmed_live_with_death_signal",
            "adjacent_or_overlapping_dead_uncertain": (
                "confirmed_adjacent_or_overlapping_uncertain"
            ),
            "merged_multiple_objects": "confirmed_merged_multiple_objects",
        }.get(relation, f"confirmed_{relation}")
    if relation == "rejected_signal":
        return "rejected_signal"
    return f"unconfirmed_{relation}"


def build_multilevel_annotations(
    branch_name: str,
    branch_dir: Path,
    timepoint: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    cell_source_fields, cell_rows = read_source_tables(
        branch_dir / "features",
        f"*{FEATURE_SUFFIX}",
        "cell feature",
        timepoint,
    )
    death_source_fields, death_rows = read_source_tables(
        branch_dir / "dead_objects",
        f"*{DEAD_OBJECT_SUFFIX}",
        "Death-object",
        timepoint,
    )

    cell_index: dict[tuple[str, int], dict[str, str]] = {}
    for row in cell_rows:
        key = row.get("key", "")
        combined_id = as_int(row, "combined_mask_id")
        identity = (key, combined_id)
        if not key or combined_id <= 0 or identity in cell_index:
            raise RuntimeError(f"Invalid or duplicate cell identity for {branch_name}: {identity}")
        cell_index[identity] = row

    death_index: dict[tuple[str, int], dict[str, str]] = {}
    linked_objects: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in death_rows:
        key = row.get("key", "")
        dead_id = as_int(row, "dead_mask_id")
        identity = (key, dead_id)
        if not key or dead_id <= 0 or identity in death_index:
            raise RuntimeError(f"Invalid or duplicate Death-object identity for {branch_name}: {identity}")
        death_index[identity] = row
        assigned_id = as_int(row, "assigned_combined_mask_id")
        if assigned_id > 0:
            cell_identity = (key, assigned_id)
            if cell_identity not in cell_index:
                raise RuntimeError(
                    f"Death object {identity} links to missing cell {cell_identity} in {branch_name}"
                )
            linked_objects.setdefault(cell_identity, []).append(row)

    cell_derived: dict[tuple[str, int], dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for identity, source in cell_index.items():
        key, combined_id = identity
        final_state = source.get("final_state") or source.get("state", "")
        if final_state not in CELL_CLASSIFICATIONS:
            raise RuntimeError(f"Unknown final cell classification {final_state!r} for {identity}")
        linked = linked_objects.get(identity, [])
        confirmed = [row for row in linked if as_bool(row, "confirmed_dead_object")]
        detailed = cell_detailed_classification(final_state, confirmed)
        confidence = source.get("classification_confidence", "")
        uncertainty_types: list[str] = []
        if final_state == "uncertain":
            uncertainty_types.append("explicit_uncertain_state")
        if confidence in {"medium", "low"}:
            uncertainty_types.append(f"{confidence}_classification_confidence")
        if any(
            row.get("association_relation") == "adjacent_or_overlapping_dead_uncertain"
            for row in confirmed
        ):
            uncertainty_types.append("uncertain_death_object_attribution")
        relations = joined_values([row.get("association_relation", "") for row in linked])
        nucleus_relations = joined_values(
            [row.get("nucleus_supported_relation", "") for row in linked]
        )
        derived = {
            "analysis_branch": branch_name,
            "annotation_level": "cell",
            "record_id": cell_record_id(branch_name, key, combined_id),
            "linked_cell_record_id": "",
            "image_id": source.get("image_id", ""),
            "key": key,
            "well": source.get("well", ""),
            "site": source.get("site", ""),
            "day": source.get("day", ""),
            "hour": source.get("hour", ""),
            "minute": source.get("minute", ""),
            "elapsed_hours": source.get("elapsed_hours", ""),
            "combined_mask_id": combined_id,
            "dead_mask_id": as_int(source, "dead_mask_id"),
            "final_classification": final_state,
            "detailed_classification": detailed,
            "classification_confidence": confidence,
            "uncertainty_flag": bool(uncertainty_types),
            "uncertainty_types": ";".join(uncertainty_types),
            "countable": as_bool(source, "countable"),
            "confirmed_death_object": bool(confirmed),
            "supplemental_death_object": any(
                as_bool(row, "supplemental_dead_object") for row in linked
            ),
            "association_relation": relations,
            "nucleus_supported_relation": nucleus_relations,
            "linked_death_object_count": len(linked),
            "confirmed_linked_death_object_count": len(confirmed),
            "linked_cell_final_classification": "",
            "linked_cell_detailed_classification": "",
        }
        cell_derived[identity] = derived
        output.append(
            {
                **derived,
                **{f"cell_{field}": source.get(field, "") for field in cell_source_fields},
                **{f"death_{field}": "" for field in death_source_fields},
            }
        )

    for identity, source in death_index.items():
        key, dead_id = identity
        confirmed = as_bool(source, "confirmed_dead_object")
        relation = source.get("association_relation", "")
        assigned_id = as_int(source, "assigned_combined_mask_id")
        linked_cell = cell_derived.get((key, assigned_id)) if assigned_id > 0 else None
        if confirmed:
            final_classification = "dead"
            confidence = (
                "uncertain"
                if relation == "adjacent_or_overlapping_dead_uncertain"
                else "confirmed"
            )
        elif relation == "rejected_signal":
            final_classification = "rejected"
            confidence = "rejected"
        else:
            final_classification = "candidate"
            confidence = "unconfirmed"
        uncertainty_types: list[str] = []
        if relation == "adjacent_or_overlapping_dead_uncertain":
            uncertainty_types.append("uncertain_spatial_attribution")
        if final_classification == "candidate":
            uncertainty_types.append("unconfirmed_death_candidate")
        derived = {
            "analysis_branch": branch_name,
            "annotation_level": "death_object",
            "record_id": death_record_id(branch_name, key, dead_id),
            "linked_cell_record_id": (
                cell_record_id(branch_name, key, assigned_id) if linked_cell else ""
            ),
            "image_id": source.get("image_id", ""),
            "key": key,
            "well": source.get("well", ""),
            "site": source.get("site", ""),
            "day": source.get("day", ""),
            "hour": source.get("hour", ""),
            "minute": source.get("minute", ""),
            "elapsed_hours": source.get("elapsed_hours", ""),
            "combined_mask_id": assigned_id,
            "dead_mask_id": dead_id,
            "final_classification": final_classification,
            "detailed_classification": death_detailed_classification(source),
            "classification_confidence": confidence,
            "uncertainty_flag": bool(uncertainty_types),
            "uncertainty_types": ";".join(uncertainty_types),
            "countable": confirmed,
            "confirmed_death_object": confirmed,
            "supplemental_death_object": as_bool(source, "supplemental_dead_object"),
            "association_relation": relation,
            "nucleus_supported_relation": source.get("nucleus_supported_relation", ""),
            "linked_death_object_count": "",
            "confirmed_linked_death_object_count": "",
            "linked_cell_final_classification": (
                linked_cell["final_classification"] if linked_cell else ""
            ),
            "linked_cell_detailed_classification": (
                linked_cell["detailed_classification"] if linked_cell else ""
            ),
        }
        output.append(
            {
                **derived,
                **{f"cell_{field}": "" for field in cell_source_fields},
                **{f"death_{field}": source.get(field, "") for field in death_source_fields},
            }
        )

    output.sort(
        key=lambda row: (
            str(row["analysis_branch"]),
            str(row["key"]),
            0 if row["annotation_level"] == "cell" else 1,
            int(row["combined_mask_id"] or 0),
            int(row["dead_mask_id"] or 0),
        )
    )
    return output, cell_source_fields, death_source_fields


def annotation_summary_rows(master_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def add(
        branch: str,
        metric_level: str,
        classification: str,
        subclassification: str,
        count: int,
        denominator_name: str,
        denominator: int,
    ) -> None:
        output.append(
            {
                "analysis_branch": branch,
                "metric_level": metric_level,
                "classification": classification,
                "subclassification": subclassification,
                "count": count,
                "denominator_name": denominator_name,
                "denominator": denominator,
                "percentage": count / denominator if denominator else 0.0,
            }
        )

    for branch in sorted({str(row["analysis_branch"]) for row in master_rows}):
        branch_rows = [row for row in master_rows if row["analysis_branch"] == branch]
        cells = [row for row in branch_rows if row["annotation_level"] == "cell"]
        objects = [row for row in branch_rows if row["annotation_level"] == "death_object"]
        countable_cells = [row for row in cells if bool(row["countable"])]
        confirmed_objects = [row for row in objects if bool(row["confirmed_death_object"])]
        supplemental_objects = [row for row in objects if bool(row["supplemental_death_object"])]

        for state in sorted(CELL_CLASSIFICATIONS):
            state_rows = [row for row in cells if row["final_classification"] == state]
            add(branch, "cell_state", state, "all", len(state_rows), "all_combined_cells", len(cells))
            countable_state_rows = [row for row in countable_cells if row["final_classification"] == state]
            add(
                branch,
                "countable_cell_state",
                state,
                "all",
                len(countable_state_rows),
                "countable_combined_cells",
                len(countable_cells),
            )
            for confidence in ("high", "medium", "low", ""):
                confidence_rows = [
                    row for row in state_rows if row["classification_confidence"] == confidence
                ]
                if confidence_rows:
                    add(
                        branch,
                        "cell_confidence_by_state",
                        state,
                        confidence or "not_recorded",
                        len(confidence_rows),
                        f"{state}_cells",
                        len(state_rows),
                    )
            uncertain_rows = [row for row in state_rows if bool(row["uncertainty_flag"])]
            add(
                branch,
                "cell_uncertainty_by_state",
                state,
                "uncertain",
                len(uncertain_rows),
                f"{state}_cells",
                len(state_rows),
            )

        for detailed, count in sorted(Counter(row["detailed_classification"] for row in cells).items()):
            state = next(
                str(row["final_classification"])
                for row in cells
                if row["detailed_classification"] == detailed
            )
            state_total = sum(row["final_classification"] == state for row in cells)
            add(
                branch,
                "cell_detailed_classification",
                state,
                str(detailed),
                count,
                f"{state}_cells",
                state_total,
            )

        for status in ("dead", "candidate", "rejected"):
            status_rows = [row for row in objects if row["final_classification"] == status]
            add(
                branch,
                "death_object_status",
                status,
                "all",
                len(status_rows),
                "all_segmented_death_objects",
                len(objects),
            )
            uncertain_rows = [row for row in status_rows if bool(row["uncertainty_flag"])]
            add(
                branch,
                "death_object_uncertainty_by_status",
                status,
                "uncertain",
                len(uncertain_rows),
                f"{status}_death_objects",
                len(status_rows),
            )

        for relation, count in sorted(
            Counter(row["association_relation"] for row in confirmed_objects).items()
        ):
            add(
                branch,
                "confirmed_death_object_relation",
                "dead",
                str(relation),
                count,
                "confirmed_death_objects",
                len(confirmed_objects),
            )

        linked_state_counts = Counter(
            row["linked_cell_final_classification"] or "no_linked_cell"
            for row in confirmed_objects
        )
        for linked_state, count in sorted(linked_state_counts.items()):
            add(
                branch,
                "confirmed_death_by_linked_cell_state",
                "dead_object",
                str(linked_state),
                count,
                "confirmed_death_objects",
                len(confirmed_objects),
            )

        dead_cells = [row for row in cells if row["final_classification"] == "dead"]
        object_aware_total = len(dead_cells) + len(supplemental_objects)
        add(
            branch,
            "object_aware_death_event_composition",
            "dead_event",
            "cell_associated_dead_cell",
            len(dead_cells),
            "object_aware_dead_events",
            object_aware_total,
        )
        for relation, count in sorted(
            Counter(row["association_relation"] for row in supplemental_objects).items()
        ):
            add(
                branch,
                "object_aware_death_event_composition",
                "dead_event",
                f"supplemental_{relation}",
                count,
                "object_aware_dead_events",
                object_aware_total,
            )

    return output


def validate_multilevel_annotations(
    master_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    record_ids = [str(row["record_id"]) for row in master_rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate record_id values in final multilevel annotation ledger")
    cell_ids = {
        str(row["record_id"])
        for row in master_rows
        if row["annotation_level"] == "cell"
    }
    missing_links = [
        str(row["record_id"])
        for row in master_rows
        if row["linked_cell_record_id"] and row["linked_cell_record_id"] not in cell_ids
    ]
    if missing_links:
        raise RuntimeError(f"Death objects link to missing cell records: {missing_links[:5]}")
    for row in master_rows:
        if bool(row["supplemental_death_object"]) and not bool(row["confirmed_death_object"]):
            raise RuntimeError(f"Supplemental object is not confirmed: {row['record_id']}")
    for row in summary_rows:
        denominator = int(row["denominator"])
        count = int(row["count"])
        percentage = float(row["percentage"])
        if count < 0 or denominator < 0 or count > denominator:
            raise RuntimeError(f"Invalid annotation summary count: {row}")
        expected = count / denominator if denominator else 0.0
        if abs(percentage - expected) > 1e-12:
            raise RuntimeError(f"Annotation summary percentage mismatch: {row}")

    for branch in sorted({str(row["analysis_branch"]) for row in master_rows}):
        branch_master = [row for row in master_rows if row["analysis_branch"] == branch]
        branch_summary = [row for row in summary_rows if row["analysis_branch"] == branch]
        cells = [row for row in branch_master if row["annotation_level"] == "cell"]
        objects = [row for row in branch_master if row["annotation_level"] == "death_object"]
        confirmed = [row for row in objects if bool(row["confirmed_death_object"])]
        supplemental = [row for row in objects if bool(row["supplemental_death_object"])]
        dead_cells = [row for row in cells if row["final_classification"] == "dead"]

        def metric_sum(metric_level: str) -> int:
            return sum(
                int(row["count"])
                for row in branch_summary
                if row["metric_level"] == metric_level
            )

        invariants = {
            "cell_state": (metric_sum("cell_state"), len(cells)),
            "death_object_status": (metric_sum("death_object_status"), len(objects)),
            "confirmed_death_object_relation": (
                metric_sum("confirmed_death_object_relation"),
                len(confirmed),
            ),
            "object_aware_death_event_composition": (
                metric_sum("object_aware_death_event_composition"),
                len(dead_cells) + len(supplemental),
            ),
        }
        for name, (observed, expected) in invariants.items():
            if observed != expected:
                raise RuntimeError(
                    f"Annotation summary invariant failed for {branch}/{name}: "
                    f"observed {observed}, expected {expected}"
                )


def verify_qc(branch_name: str, branch_dir: Path, timepoint: str) -> list[dict[str, Any]]:
    feature_ids = {
        path.name[: -len(FEATURE_SUFFIX)]
        for path in selected_paths(
            (branch_dir / "features").glob(f"*{FEATURE_SUFFIX}"), timepoint
        )
    }
    prediction_ids = {
        path.name[: -len(PREDICTION_SUFFIX)]
        for path in selected_paths(
            (branch_dir / "predictions").glob(f"*{PREDICTION_SUFFIX}"), timepoint
        )
    }
    summary_ids = {
        path.name.removesuffix("_summary.csv")
        for path in selected_paths((branch_dir / "summaries").glob("*_summary.csv"), timepoint)
    }
    overlay_paths = {
        path.name[: -len(OVERLAY_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "qc" / "label_overlays").glob(f"*{OVERLAY_SUFFIX}"), timepoint
        )
    }
    object_ids = {
        path.name[: -len(DEAD_OBJECT_SUFFIX)]
        for path in selected_paths(
            (branch_dir / "dead_objects").glob(f"*{DEAD_OBJECT_SUFFIX}"), timepoint
        )
    }
    object_overlay_paths = {
        path.name[: -len(DEAD_OBJECT_OVERLAY_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "qc" / "dead_object_overlays").glob(f"*{DEAD_OBJECT_OVERLAY_SUFFIX}"),
            timepoint,
        )
    }
    overlap_overlay_paths = {
        path.name[: -len(OVERLAP_OVERLAY_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "qc" / "overlap_state_overlays").glob(f"*{OVERLAP_OVERLAY_SUFFIX}"),
            timepoint,
        )
    }
    cell_state_mask_paths = {
        path.name[: -len(CELL_STATE_MASK_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "masks" / "cell_state").glob(f"*{CELL_STATE_MASK_SUFFIX}"),
            timepoint,
        )
    }
    confirmed_dead_mask_paths = {
        path.name[: -len(CONFIRMED_DEAD_MASK_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "masks" / "confirmed_dead").glob(f"*{CONFIRMED_DEAD_MASK_SUFFIX}"),
            timepoint,
        )
    }
    overlap_mask_paths = {
        path.name[: -len(OVERLAP_MASK_SUFFIX)]: path
        for path in selected_paths(
            (branch_dir / "overlap_masks").glob(f"*{OVERLAP_MASK_SUFFIX}"), timepoint
        )
    }
    image_ids = sorted(
        feature_ids
        | prediction_ids
        | summary_ids
        | object_ids
        | set(overlay_paths)
        | set(object_overlay_paths)
        | set(overlap_overlay_paths)
        | set(cell_state_mask_paths)
        | set(confirmed_dead_mask_paths)
        | set(overlap_mask_paths)
    )
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        overlay = overlay_paths.get(image_id)
        qc_valid = False
        width = 0
        height = 0
        qc_error = ""
        object_overlay = object_overlay_paths.get(image_id)
        object_qc_valid = False
        object_qc_width = 0
        object_qc_height = 0
        object_qc_error = ""
        overlap_overlay = overlap_overlay_paths.get(image_id)
        overlap_qc_valid = False
        overlap_qc_width = 0
        overlap_qc_height = 0
        overlap_qc_error = ""
        mask_layers_valid = False
        mask_layers_error = ""
        if overlay is not None:
            try:
                with Image.open(overlay) as image:
                    width, height = image.size
                    image.verify()
                qc_valid = True
            except Exception as exc:  # pragma: no cover - data-dependent diagnostic
                qc_error = f"{type(exc).__name__}: {exc}"
        if object_overlay is not None:
            try:
                with Image.open(object_overlay) as image:
                    object_qc_width, object_qc_height = image.size
                    image.verify()
                object_qc_valid = True
            except Exception as exc:  # pragma: no cover - data-dependent diagnostic
                object_qc_error = f"{type(exc).__name__}: {exc}"
        if overlap_overlay is not None:
            try:
                with Image.open(overlap_overlay) as image:
                    overlap_qc_width, overlap_qc_height = image.size
                    image.verify()
                overlap_qc_valid = True
            except Exception as exc:  # pragma: no cover - data-dependent diagnostic
                overlap_qc_error = f"{type(exc).__name__}: {exc}"
        cell_state_mask_path = cell_state_mask_paths.get(image_id)
        confirmed_dead_mask_path = confirmed_dead_mask_paths.get(image_id)
        overlap_mask_path = overlap_mask_paths.get(image_id)
        if cell_state_mask_path and confirmed_dead_mask_path and overlap_mask_path:
            try:
                cell_state_mask = tifffile.imread(cell_state_mask_path)
                confirmed_dead_mask = tifffile.imread(confirmed_dead_mask_path)
                overlap_mask = tifffile.imread(overlap_mask_path)
                mask_layers_valid = (
                    overlap_mask.ndim == 3
                    and overlap_mask.shape[0] == 2
                    and overlap_mask.shape[1:] == cell_state_mask.shape == confirmed_dead_mask.shape
                    and np.array_equal(overlap_mask[0], cell_state_mask)
                    and np.array_equal(overlap_mask[1], confirmed_dead_mask)
                )
                if not mask_layers_valid:
                    mask_layers_error = "Two-channel overlap TIFF does not match its component masks"
            except Exception as exc:  # pragma: no cover - data-dependent diagnostic
                mask_layers_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "branch": branch_name,
                "image_id": image_id,
                "feature_exists": image_id in feature_ids,
                "prediction_exists": image_id in prediction_ids,
                "summary_exists": image_id in summary_ids,
                "dead_object_table_exists": image_id in object_ids,
                "qc_exists": overlay is not None,
                "qc_valid": qc_valid,
                "qc_width": width,
                "qc_height": height,
                "qc_path": str(overlay) if overlay is not None else "",
                "qc_error": qc_error,
                "dead_object_qc_exists": object_overlay is not None,
                "dead_object_qc_valid": object_qc_valid,
                "dead_object_qc_width": object_qc_width,
                "dead_object_qc_height": object_qc_height,
                "dead_object_qc_path": str(object_overlay) if object_overlay is not None else "",
                "dead_object_qc_error": object_qc_error,
                "overlap_qc_exists": overlap_overlay is not None,
                "overlap_qc_valid": overlap_qc_valid,
                "overlap_qc_width": overlap_qc_width,
                "overlap_qc_height": overlap_qc_height,
                "overlap_qc_path": str(overlap_overlay) if overlap_overlay is not None else "",
                "overlap_qc_error": overlap_qc_error,
                "cell_state_mask_exists": cell_state_mask_path is not None,
                "confirmed_dead_mask_exists": confirmed_dead_mask_path is not None,
                "overlap_mask_exists": overlap_mask_path is not None,
                "mask_layers_valid": mask_layers_valid,
                "mask_layers_error": mask_layers_error,
                "complete": (
                    image_id in feature_ids
                    and image_id in prediction_ids
                    and image_id in summary_ids
                    and image_id in object_ids
                    and qc_valid
                    and object_qc_valid
                    and overlap_qc_valid
                    and mask_layers_valid
                ),
            }
        )
    return rows


def branch_metrics(
    branch: str,
    rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
    reference_objects = [row for row in object_rows if row["automated_reference_dead_object"]]
    detected_objects = [row for row in reference_objects if row["automated_reference_detected"]]
    return {
        "branch": branch,
        "total_combined_cells": len(rows),
        "automated_reference_death_count": len(reference_dead),
        "automated_reference_death_detected": len(reference_detected),
        "automated_reference_death_missed": len(reference_dead) - len(reference_detected),
        "automated_reference_death_recall": len(reference_detected) / len(reference_dead) if reference_dead else 0.0,
        "segmented_dead_object_count": len(object_rows),
        "automated_reference_dead_object_count": len(reference_objects),
        "automated_reference_dead_object_detected": len(detected_objects),
        "automated_reference_dead_object_missed": len(reference_objects) - len(detected_objects),
        "automated_reference_dead_object_recall": (
            len(detected_objects) / len(reference_objects) if reference_objects else 0.0
        ),
        "automated_reference_dead_object_unassigned": sum(
            as_int(row, "assigned_combined_mask_id") <= 0 for row in reference_objects
        ),
        "automated_reference_dead_object_supplemental": sum(
            bool(row["supplemental_dead_object"]) for row in reference_objects
        ),
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


def per_field_metrics(
    branch: str,
    rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cell_by_image: dict[str, list[dict[str, Any]]] = {}
    object_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cell_by_image.setdefault(str(row["image_id"]), []).append(row)
    for row in object_rows:
        object_by_image.setdefault(str(row["image_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for image_id in sorted(set(cell_by_image) | set(object_by_image)):
        cells = cell_by_image.get(image_id, [])
        objects = object_by_image.get(image_id, [])
        conservative_live = [row for row in cells if row["conservative_rgb_live_reference"]]
        possible_live_fp = [row for row in conservative_live if row["possible_live_false_positive"]]
        references = [row for row in objects if row["automated_reference_dead_object"]]
        detected = [row for row in references if row["automated_reference_detected"]]
        output.append(
            {
                "branch": branch,
                "image_id": image_id,
                "key": (cells or objects or [{}])[0].get("key", ""),
                "conservative_rgb_live_reference_count": len(conservative_live),
                "possible_live_false_positive_count": len(possible_live_fp),
                "possible_live_false_positive_upper_bound": (
                    len(possible_live_fp) / len(conservative_live) if conservative_live else 0.0
                ),
                "automated_reference_dead_object_count": len(references),
                "automated_reference_dead_object_detected": len(detected),
                "automated_reference_dead_object_missed": len(references) - len(detected),
                "automated_reference_dead_object_recall": (
                    len(detected) / len(references) if references else 1.0
                ),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    args.timepoint = normalize_timepoint(args.timepoint)
    branches = [parse_branch(value) for value in args.branch]
    if len({name for name, _path in branches}) != len(branches):
        raise ValueError("Branch names must be unique")

    all_rows: list[dict[str, Any]] = []
    all_object_rows: list[dict[str, Any]] = []
    all_qc_rows: list[dict[str, Any]] = []
    all_field_metrics: list[dict[str, Any]] = []
    all_annotation_rows: list[dict[str, Any]] = []
    cell_source_fields: list[str] | None = None
    death_source_fields: list[str] | None = None
    metrics: list[dict[str, Any]] = []
    for name, path in branches:
        rows = read_feature_rows(name, path, args)
        object_rows = read_dead_object_rows(name, path, args)
        qc_rows = verify_qc(name, path, args.timepoint)
        annotation_rows, branch_cell_fields, branch_death_fields = build_multilevel_annotations(
            name,
            path,
            args.timepoint,
        )
        if cell_source_fields is None:
            cell_source_fields = branch_cell_fields
            death_source_fields = branch_death_fields
        elif branch_cell_fields != cell_source_fields or branch_death_fields != death_source_fields:
            raise RuntimeError(f"Source annotation schema differs across branches at {name}")
        all_rows.extend(rows)
        all_object_rows.extend(object_rows)
        all_qc_rows.extend(qc_rows)
        all_field_metrics.extend(per_field_metrics(name, rows, object_rows))
        all_annotation_rows.extend(annotation_rows)
        metrics.append(branch_metrics(name, rows, object_rows, qc_rows))

    per_cell_fields = list(all_rows[0])
    object_fields = list(all_object_rows[0]) if all_object_rows else DEAD_OBJECT_AUDIT_FIELDS
    qc_fields = list(all_qc_rows[0])
    field_metric_fields = list(all_field_metrics[0])
    metric_fields = list(metrics[0])
    write_rows(args.out_dir / "automated_reference_per_cell.csv", all_rows, per_cell_fields)
    write_rows(
        args.out_dir / "automated_reference_missed.csv",
        [row for row in all_rows if row["automated_reference_dead"] and not row["automated_reference_detected"]],
        per_cell_fields,
    )
    write_rows(args.out_dir / "automated_reference_per_dead_object.csv", all_object_rows, object_fields)
    write_rows(
        args.out_dir / "automated_reference_dead_object_missed.csv",
        [
            row
            for row in all_object_rows
            if row["automated_reference_dead_object"] and not row["automated_reference_detected"]
        ],
        object_fields,
    )
    write_rows(args.out_dir / "per_field_metrics.csv", all_field_metrics, field_metric_fields)
    write_rows(
        args.out_dir / "unresolved_dead_evidence.csv",
        [row for row in all_rows if row["unresolved_dead_evidence"]],
        per_cell_fields,
    )
    write_rows(args.out_dir / "qc_inventory.csv", all_qc_rows, qc_fields)
    write_rows(args.out_dir / "automated_reference_metrics.csv", metrics, metric_fields)

    annotation_summary = annotation_summary_rows(all_annotation_rows)
    validate_multilevel_annotations(all_annotation_rows, annotation_summary)
    annotation_out_dir = args.annotation_out_dir or args.out_dir
    master_fields = [
        *MASTER_ANNOTATION_FIELDS,
        *(f"cell_{field}" for field in (cell_source_fields or [])),
        *(f"death_{field}" for field in (death_source_fields or [])),
    ]
    write_rows(
        annotation_out_dir / "final_multilevel_annotations.csv",
        all_annotation_rows,
        master_fields,
    )
    write_rows(
        annotation_out_dir / "final_annotation_summary.csv",
        annotation_summary,
        [
            "analysis_branch",
            "metric_level",
            "classification",
            "subclassification",
            "count",
            "denominator_name",
            "denominator",
            "percentage",
        ],
    )

    for row in metrics:
        print(
            f"branch={row['branch']} object_reference_recall="
            f"{row['automated_reference_dead_object_recall']:.6f} "
            f"possible_live_fp_upper_bound={row['possible_live_false_positive_upper_bound']:.6f} "
            f"qc={row['qc_complete_count']}/{row['qc_image_count']}"
        )
    print(
        f"multilevel_annotations={len(all_annotation_rows)} "
        f"annotation_summary_rows={len(annotation_summary)} "
        f"annotation_out_dir={annotation_out_dir} "
        f"selected_timepoint={args.timepoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
