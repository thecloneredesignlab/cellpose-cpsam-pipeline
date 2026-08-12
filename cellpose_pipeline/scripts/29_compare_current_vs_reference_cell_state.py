#!/usr/bin/env python3
"""Compare the frozen current and reference cell-state axes without coupling them.

This is a descriptive, pre-gold-standard comparison.  It joins the two frozen
outputs only by ``original|<field-key>|<Combined-mask-label>``.  It never edits
either input, never feeds a result back to either classifier, and deliberately
does not calculate accuracy, sensitivity, specificity, or other gold-standard
metrics.

The current input can be either one normalized TSV or a manifest of the
authoritative per-field fusion feature/prediction CSVs.  Manifest mode verifies
the recorded SHA-256 values and independently agrees ``final_state`` in the
feature file with ``state`` in the prediction file before comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO


SCHEMA_VERSION = "current_vs_reference_cell_state_precomparison_v1"
CONFIG_SCHEMA = "reference_cell_state_comparison_config_v1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "reference_cell_state_comparison_v1.json"
)
CELL_ID_RE = re.compile(r"^(?P<branch>[^|]+)\|(?P<key>[^|]+)\|(?P<label>[1-9][0-9]*)$")
KEY_RE = re.compile(
    r"^(?P<well>[A-H][0-9]+)_(?P<site>[0-9]+)_"
    r"(?P<day>[0-9]+)d(?P<hour>[0-9]+)h(?P<minute>[0-9]+)m$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

NORMALIZED_CURRENT_COLUMNS = (
    "cell_id",
    "key",
    "mask_label",
    "well",
    "elapsed_hours",
    "current_final_state",
    "current_prediction_status",
)
CURRENT_MANIFEST_COLUMNS = (
    "key",
    "feature_path",
    "feature_sha256",
    "prediction_path",
    "prediction_sha256",
)
REFERENCE_COLUMNS = (
    "model_id",
    "cell_id",
    "reference_cell_state_class_id",
    "prediction_status",
)
UNIVERSE_COLUMNS = (
    "cell_id",
    "key",
    "mask_label",
    "well",
    "elapsed_hours",
    "current_input_row_number",
    "reference_input_row_number",
    "join_status",
)
PAIRED_COLUMNS = (
    "cell_id",
    "key",
    "mask_label",
    "well",
    "elapsed_hours",
    "current_final_state",
    "current_prediction_status",
    "reference_cell_state_model_id",
    "reference_cell_state_class_id",
    "reference_cell_state_prediction_status",
    "current_dead_endpoint_call",
    "reference_dead_endpoint_call",
    "dead_endpoint_pair_status",
)
DEAD_SUBSET_COLUMNS = PAIRED_COLUMNS
OUTPUT_NAMES = (
    "comparison_cell_universe.tsv",
    "join_audit.json",
    "paired_cell_predictions.tsv",
    "cross_classification_counts.tsv",
    "cross_classification_row_percent.tsv",
    "cross_classification_column_percent.tsv",
    "dead_endpoint_descriptive_subset.tsv",
    "dead_endpoint_descriptive_counts.tsv",
    "well_time_class_fractions.tsv",
    "comparison_receipt.json",
)


@dataclass(frozen=True)
class CurrentRow:
    cell_id: str
    key: str
    mask_label: int
    well: str
    elapsed_hours: float
    final_state: str
    prediction_status: str
    row_number: int


@dataclass(frozen=True)
class ReferenceRow:
    model_id: str
    cell_id: str
    class_id: str
    prediction_status: str
    row_number: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    current = parser.add_mutually_exclusive_group(required=True)
    current.add_argument(
        "--current-predictions",
        type=Path,
        help="Normalized current TSV with the exact documented seven-column schema.",
    )
    current.add_argument(
        "--current-manifest",
        type=Path,
        help=(
            "TSV listing per-field feature and prediction CSVs plus their SHA-256 "
            "values."
        ),
    )
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only this stage's known output files; never remove a directory.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def require_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_separate_output(output_root: Path, input_paths: Iterable[Path]) -> None:
    for input_path in input_paths:
        parent = input_path.resolve().parent
        if output_root == input_path.resolve() or path_is_within(output_root, parent):
            raise ValueError(
                "Output root must not write into an input directory: "
                f"output={output_root} input={input_path}"
            )


def delimiter_for(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def open_table(path: Path, label: str) -> tuple[TextIO, csv.DictReader]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter=delimiter_for(path))
    fields = tuple(reader.fieldnames or ())
    if not fields:
        handle.close()
        raise ValueError(f"{label} is empty: {path}")
    if any(not field for field in fields) or len(fields) != len(set(fields)):
        handle.close()
        raise ValueError(f"{label} has blank or duplicate columns: {path}")
    return handle, reader


def exact_columns(reader: csv.DictReader, expected: Sequence[str], label: str) -> None:
    observed = tuple(reader.fieldnames or ())
    if observed != tuple(expected):
        raise ValueError(
            f"{label} columns changed: expected={list(expected)} observed={list(observed)}"
        )


def require_columns(
    reader: csv.DictReader, required: Sequence[str], label: str
) -> None:
    fields = set(reader.fieldnames or ())
    missing = [column for column in required if column not in fields]
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def parse_positive_integer(value: str, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid positive integer for {context}: {value!r}"
        ) from error
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ValueError(f"Invalid positive integer for {context}: {value!r}")
    return parsed


def parse_elapsed_hours(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid elapsed_hours for {context}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Invalid elapsed_hours for {context}: {value!r}")
    return parsed


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(value, ".12g")


def validate_key_metadata(
    key: str, well: str, elapsed_hours: float, context: str
) -> None:
    match = KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid field key for {context}: {key!r}")
    if match.group("well") != well:
        raise ValueError(f"Well/key mismatch for {context}: key={key!r} well={well!r}")
    expected = (
        int(match.group("day")) * 24
        + int(match.group("hour"))
        + int(match.group("minute")) / 60.0
    )
    if not math.isclose(expected, elapsed_hours, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"elapsed_hours/key mismatch for {context}: key={key!r} "
            f"expected={expected} observed={elapsed_hours}"
        )


def parse_cell_id(cell_id: str, expected_branch: str, context: str) -> tuple[str, int]:
    match = CELL_ID_RE.fullmatch(cell_id)
    if match is None:
        raise ValueError(f"Invalid stable cell_id for {context}: {cell_id!r}")
    if match.group("branch") != expected_branch:
        raise ValueError(
            f"Stable cell_id branch changed for {context}: expected={expected_branch!r} "
            f"observed={match.group('branch')!r}"
        )
    return match.group("key"), int(match.group("label"))


def load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read comparison config {path}: {error}") from error
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "current_states": ["live", "dead", "uncertain", "artifact"],
        "reference_classes": [
            "live_cell",
            "dead_cell",
            "multinucleated_cell",
        ],
        "current_dead_endpoint": {
            "live": "not_dead",
            "dead": "dead",
            "uncertain": "abstain",
            "artifact": "abstain",
        },
        "reference_dead_endpoint": {
            "live_cell": "not_dead",
            "dead_cell": "dead",
            "multinucleated_cell": "not_dead",
        },
        "reference_available_status": "ok",
        "reference_unavailable_status_prefix": "unavailable",
        "stable_cell_id_branch": "original",
    }
    if payload != expected:
        raise ValueError(
            "Comparison config does not match the frozen v1 semantics; in particular, "
            "multinucleated_cell must remain its own class and map only to not_dead for "
            "the descriptive dead endpoint"
        )
    return payload


def validate_grouped_order(
    key: str,
    label: int,
    previous_key: str | None,
    previous_label: int,
    closed_keys: set[str],
    context: str,
) -> tuple[str, int]:
    if previous_key is None or key != previous_key:
        if key in closed_keys:
            raise ValueError(f"Non-contiguous duplicate key in {context}: {key}")
        if previous_key is not None:
            closed_keys.add(previous_key)
        return key, label
    if label <= previous_label:
        raise ValueError(
            f"Mask labels must increase strictly within key={key} in {context}: "
            f"previous={previous_label} observed={label}"
        )
    return key, label


def iter_normalized_current(
    path: Path,
    config: dict[str, Any],
) -> Iterator[CurrentRow]:
    handle, reader = open_table(path, "normalized current predictions")
    previous_key: str | None = None
    previous_label = 0
    closed_keys: set[str] = set()
    count = 0
    try:
        exact_columns(
            reader, NORMALIZED_CURRENT_COLUMNS, "normalized current predictions"
        )
        for row_number, row in enumerate(reader, 2):
            key = (row.get("key") or "").strip()
            well = (row.get("well") or "").strip()
            label = parse_positive_integer(
                (row.get("mask_label") or "").strip(), f"current row {row_number}"
            )
            elapsed = parse_elapsed_hours(
                (row.get("elapsed_hours") or "").strip(), f"current row {row_number}"
            )
            validate_key_metadata(key, well, elapsed, f"current row {row_number}")
            cell_id = (row.get("cell_id") or "").strip()
            id_key, id_label = parse_cell_id(
                cell_id, config["stable_cell_id_branch"], f"current row {row_number}"
            )
            if (id_key, id_label) != (key, label):
                raise ValueError(
                    f"Current stable cell_id does not match key/mask_label at row {row_number}"
                )
            state = (row.get("current_final_state") or "").strip()
            if state not in config["current_states"]:
                raise ValueError(
                    f"Unknown current final state at row {row_number}: {state!r}"
                )
            status = (row.get("current_prediction_status") or "").strip()
            if status != "ok":
                raise ValueError(
                    "The frozen current axis must have one explicit final state per Combined "
                    f"object; invalid status at row {row_number}: {status!r}"
                )
            previous_key, previous_label = validate_grouped_order(
                key,
                label,
                previous_key,
                previous_label,
                closed_keys,
                "normalized current predictions",
            )
            count += 1
            yield CurrentRow(
                cell_id=cell_id,
                key=key,
                mask_label=label,
                well=well,
                elapsed_hours=elapsed,
                final_state=state,
                prediction_status=status,
                row_number=row_number,
            )
    finally:
        handle.close()
    if count == 0:
        raise ValueError("Normalized current predictions contain no rows")


def load_current_manifest(path: Path) -> list[dict[str, str]]:
    handle, reader = open_table(path, "current input manifest")
    rows: list[dict[str, str]] = []
    try:
        exact_columns(reader, CURRENT_MANIFEST_COLUMNS, "current input manifest")
        for row_number, row in enumerate(reader, 2):
            cleaned = {
                column: (row.get(column) or "").strip()
                for column in CURRENT_MANIFEST_COLUMNS
            }
            if any(not value for value in cleaned.values()):
                raise ValueError(f"Blank current manifest value at row {row_number}")
            if not SHA256_RE.fullmatch(
                cleaned["feature_sha256"]
            ) or not SHA256_RE.fullmatch(cleaned["prediction_sha256"]):
                raise ValueError(
                    f"Invalid current manifest SHA-256 at row {row_number}"
                )
            rows.append(cleaned)
    finally:
        handle.close()
    if not rows:
        raise ValueError("Current input manifest contains no fields")
    for column in ("key", "feature_path", "prediction_path"):
        values = [row[column] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Current input manifest contains duplicate {column}")
    return rows


def iter_current_manifest(
    rows: Sequence[dict[str, str]],
    config: dict[str, Any],
) -> Iterator[CurrentRow]:
    global_row_number = 1
    for manifest_index, item in enumerate(rows, 2):
        key = item["key"]
        feature_path = require_absolute_file(
            Path(item["feature_path"]),
            f"current feature file at manifest row {manifest_index}",
        )
        prediction_path = require_absolute_file(
            Path(item["prediction_path"]),
            f"current prediction file at manifest row {manifest_index}",
        )
        if sha256_file(feature_path) != item["feature_sha256"]:
            raise ValueError(f"Current feature SHA-256 mismatch for key={key}")
        if sha256_file(prediction_path) != item["prediction_sha256"]:
            raise ValueError(f"Current prediction SHA-256 mismatch for key={key}")
        feature_stat = feature_path.stat()
        prediction_stat = prediction_path.stat()
        feature_handle, feature_reader = open_table(
            feature_path, f"current features key={key}"
        )
        prediction_handle, prediction_reader = open_table(
            prediction_path, f"current predictions key={key}"
        )
        local_count = 0
        previous_label = 0
        try:
            require_columns(
                feature_reader,
                ("key", "well", "elapsed_hours", "combined_mask_id", "final_state"),
                f"current features key={key}",
            )
            require_columns(
                prediction_reader,
                ("key", "mask_id", "state"),
                f"current predictions key={key}",
            )
            sentinel = object()
            for local_row_number, pair in enumerate(
                itertools.zip_longest(
                    feature_reader, prediction_reader, fillvalue=sentinel
                ),
                2,
            ):
                feature, prediction = pair
                if feature is sentinel or prediction is sentinel:
                    raise ValueError(
                        f"Current feature/prediction row-count mismatch for key={key}"
                    )
                assert isinstance(feature, dict) and isinstance(prediction, dict)
                feature_key = (feature.get("key") or "").strip()
                prediction_key = (prediction.get("key") or "").strip()
                if feature_key != key or prediction_key != key:
                    raise ValueError(
                        f"Current manifest/file key mismatch at key={key}, "
                        f"feature={feature_key!r}, prediction={prediction_key!r}"
                    )
                feature_label = parse_positive_integer(
                    (feature.get("combined_mask_id") or "").strip(),
                    f"current features key={key} row={local_row_number}",
                )
                prediction_label = parse_positive_integer(
                    (prediction.get("mask_id") or "").strip(),
                    f"current predictions key={key} row={local_row_number}",
                )
                if feature_label != prediction_label:
                    raise ValueError(
                        f"Current feature/prediction mask-label mismatch for key={key}: "
                        f"feature={feature_label} prediction={prediction_label}"
                    )
                if feature_label <= previous_label:
                    raise ValueError(
                        f"Current labels are not strictly increasing for key={key}: "
                        f"previous={previous_label} observed={feature_label}"
                    )
                previous_label = feature_label
                feature_state = (feature.get("final_state") or "").strip()
                prediction_state = (prediction.get("state") or "").strip()
                if feature_state != prediction_state:
                    raise ValueError(
                        f"Current feature/prediction state mismatch for key={key}, "
                        f"mask_label={feature_label}: feature={feature_state!r} "
                        f"prediction={prediction_state!r}"
                    )
                if feature_state not in config["current_states"]:
                    raise ValueError(
                        f"Unknown current final state for key={key}, "
                        f"mask_label={feature_label}: {feature_state!r}"
                    )
                well = (feature.get("well") or "").strip()
                elapsed = parse_elapsed_hours(
                    (feature.get("elapsed_hours") or "").strip(),
                    f"current features key={key} row={local_row_number}",
                )
                validate_key_metadata(key, well, elapsed, f"current features key={key}")
                global_row_number += 1
                local_count += 1
                yield CurrentRow(
                    cell_id=f"{config['stable_cell_id_branch']}|{key}|{feature_label}",
                    key=key,
                    mask_label=feature_label,
                    well=well,
                    elapsed_hours=elapsed,
                    final_state=feature_state,
                    prediction_status="ok",
                    row_number=global_row_number,
                )
        finally:
            feature_handle.close()
            prediction_handle.close()
        if local_count == 0:
            raise ValueError(
                f"Current feature/prediction files contain no rows for key={key}"
            )
        for source_path, before, label in (
            (feature_path, feature_stat, "feature"),
            (prediction_path, prediction_stat, "prediction"),
        ):
            after = source_path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"Current {label} file changed while being compared for key={key}"
                )


def iter_reference(path: Path, config: dict[str, Any]) -> Iterator[ReferenceRow]:
    handle, reader = open_table(path, "reference predictions")
    count = 0
    model_id: str | None = None
    previous_key: str | None = None
    previous_label = 0
    closed_keys: set[str] = set()
    try:
        exact_columns(reader, REFERENCE_COLUMNS, "reference predictions")
        for row_number, row in enumerate(reader, 2):
            observed_model = (row.get("model_id") or "").strip()
            if not observed_model:
                raise ValueError(f"Blank reference model_id at row {row_number}")
            if model_id is None:
                model_id = observed_model
            elif observed_model != model_id:
                raise ValueError(
                    f"Multiple reference model_id values: {model_id!r}, {observed_model!r}"
                )
            cell_id = (row.get("cell_id") or "").strip()
            key, label = parse_cell_id(
                cell_id, config["stable_cell_id_branch"], f"reference row {row_number}"
            )
            status = (row.get("prediction_status") or "").strip()
            class_id = (row.get("reference_cell_state_class_id") or "").strip()
            if status == config["reference_available_status"]:
                if class_id not in config["reference_classes"]:
                    raise ValueError(
                        f"Unknown available reference class at row {row_number}: {class_id!r}"
                    )
            elif status == config[
                "reference_unavailable_status_prefix"
            ] or status.startswith(config["reference_unavailable_status_prefix"] + "_"):
                if class_id:
                    raise ValueError(
                        f"Unavailable reference prediction has a class at row {row_number}"
                    )
            else:
                raise ValueError(
                    f"Unknown reference prediction_status at row {row_number}: {status!r}"
                )
            previous_key, previous_label = validate_grouped_order(
                key,
                label,
                previous_key,
                previous_label,
                closed_keys,
                "reference predictions",
            )
            count += 1
            yield ReferenceRow(
                model_id=observed_model,
                cell_id=cell_id,
                class_id=class_id,
                prediction_status=status,
                row_number=row_number,
            )
    finally:
        handle.close()
    if count == 0:
        raise ValueError("Reference predictions contain no rows")


def dead_pair_status(current_call: str, reference_call: str) -> str:
    if "abstain" in {current_call, reference_call}:
        return "abstention"
    return "agreement" if current_call == reference_call else "disagreement"


def tsv_writer(path: Path, columns: Sequence[str]) -> tuple[TextIO, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(columns),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    return handle, writer


def write_summary_tables(
    temporary: dict[str, Path],
    config: dict[str, Any],
    cross_counts: Counter[tuple[str, str]],
    endpoint_counts: Counter[tuple[str, str]],
    group_totals: Counter[tuple[str, float]],
    group_current: Counter[tuple[str, float, str]],
    group_reference: Counter[tuple[str, float, str]],
    group_current_abstentions: Counter[tuple[str, float]],
    group_reference_unavailable: Counter[tuple[str, float]],
) -> None:
    current_states = config["current_states"]
    reference_classes = config["reference_classes"]

    handle, writer = tsv_writer(
        temporary["cross_classification_counts.tsv"],
        ("current_final_state", "reference_cell_state_class_id", "count"),
    )
    with handle:
        for current_state in current_states:
            for reference_class in reference_classes:
                writer.writerow(
                    {
                        "current_final_state": current_state,
                        "reference_cell_state_class_id": reference_class,
                        "count": cross_counts[(current_state, reference_class)],
                    }
                )

    row_denominators = Counter[str]()
    column_denominators = Counter[str]()
    for (current_state, reference_class), count in cross_counts.items():
        row_denominators[current_state] += count
        column_denominators[reference_class] += count
    handle, writer = tsv_writer(
        temporary["cross_classification_row_percent.tsv"],
        (
            "current_final_state",
            "reference_cell_state_class_id",
            "count",
            "row_denominator",
            "row_percent",
        ),
    )
    with handle:
        for current_state in current_states:
            denominator = row_denominators[current_state]
            for reference_class in reference_classes:
                count = cross_counts[(current_state, reference_class)]
                writer.writerow(
                    {
                        "current_final_state": current_state,
                        "reference_cell_state_class_id": reference_class,
                        "count": count,
                        "row_denominator": denominator,
                        "row_percent": (
                            format_number(100.0 * count / denominator)
                            if denominator
                            else "0"
                        ),
                    }
                )

    handle, writer = tsv_writer(
        temporary["cross_classification_column_percent.tsv"],
        (
            "current_final_state",
            "reference_cell_state_class_id",
            "count",
            "column_denominator",
            "column_percent",
        ),
    )
    with handle:
        for current_state in current_states:
            for reference_class in reference_classes:
                denominator = column_denominators[reference_class]
                count = cross_counts[(current_state, reference_class)]
                writer.writerow(
                    {
                        "current_final_state": current_state,
                        "reference_cell_state_class_id": reference_class,
                        "count": count,
                        "column_denominator": denominator,
                        "column_percent": (
                            format_number(100.0 * count / denominator)
                            if denominator
                            else "0"
                        ),
                    }
                )

    endpoint_order = ("dead", "not_dead", "abstain")
    handle, writer = tsv_writer(
        temporary["dead_endpoint_descriptive_counts.tsv"],
        (
            "current_dead_endpoint_call",
            "reference_dead_endpoint_call",
            "count",
            "interpretation",
        ),
    )
    with handle:
        for current_call in endpoint_order:
            for reference_call in endpoint_order:
                writer.writerow(
                    {
                        "current_dead_endpoint_call": current_call,
                        "reference_dead_endpoint_call": reference_call,
                        "count": endpoint_counts[(current_call, reference_call)],
                        "interpretation": "descriptive_pair_count_not_accuracy",
                    }
                )

    fraction_columns = (
        "well",
        "elapsed_hours",
        "method",
        "class_id",
        "count",
        "total_cell_count",
        "method_output_count",
        "endpoint_decisive_count",
        "abstention_count",
        "endpoint_coverage_fraction",
        "class_fraction",
        "class_fraction_denominator",
    )
    handle, writer = tsv_writer(
        temporary["well_time_class_fractions.tsv"], fraction_columns
    )
    with handle:
        for well, elapsed in sorted(group_totals, key=lambda item: (item[0], item[1])):
            total = group_totals[(well, elapsed)]
            current_abstentions = group_current_abstentions[(well, elapsed)]
            current_decisive = total - current_abstentions
            for state in current_states:
                count = group_current[(well, elapsed, state)]
                writer.writerow(
                    {
                        "well": well,
                        "elapsed_hours": format_number(elapsed),
                        "method": "current",
                        "class_id": state,
                        "count": count,
                        "total_cell_count": total,
                        "method_output_count": total,
                        "endpoint_decisive_count": current_decisive,
                        "abstention_count": current_abstentions,
                        "endpoint_coverage_fraction": format_number(
                            current_decisive / total
                        ),
                        "class_fraction": format_number(count / total),
                        "class_fraction_denominator": "all_joined_cells",
                    }
                )
            reference_unavailable = group_reference_unavailable[(well, elapsed)]
            reference_available = total - reference_unavailable
            for class_id in reference_classes:
                count = group_reference[(well, elapsed, class_id)]
                writer.writerow(
                    {
                        "well": well,
                        "elapsed_hours": format_number(elapsed),
                        "method": "reference",
                        "class_id": class_id,
                        "count": count,
                        "total_cell_count": total,
                        "method_output_count": reference_available,
                        "endpoint_decisive_count": reference_available,
                        "abstention_count": reference_unavailable,
                        "endpoint_coverage_fraction": format_number(
                            reference_available / total
                        ),
                        "class_fraction": (
                            format_number(count / reference_available)
                            if reference_available
                            else "0"
                        ),
                        "class_fraction_denominator": "reference_available_predictions",
                    }
                )


def temporary_paths(output_root: Path) -> dict[str, Path]:
    return {name: output_root / f".{name}.tmp.{os.getpid()}" for name in OUTPUT_NAMES}


def install_outputs(output_root: Path, temporary: dict[str, Path]) -> None:
    for name in OUTPUT_NAMES:
        os.replace(temporary[name], output_root / name)


def cleanup_temporaries(temporary: dict[str, Path]) -> None:
    for path in temporary.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = require_absolute_file(args.config, "comparison config")
    reference_path = require_absolute_file(
        args.reference_predictions, "reference predictions"
    )
    current_path = require_absolute_file(
        args.current_predictions or args.current_manifest,
        "current predictions" if args.current_predictions else "current manifest",
    )
    config = load_config(config_path)

    if not args.output_root.is_absolute():
        raise ValueError(f"output root must be an absolute path: {args.output_root}")
    output_root = args.output_root.resolve(strict=False)
    require_separate_output(output_root, (config_path, current_path, reference_path))

    manifest_rows: list[dict[str, str]] | None = None
    manifest_input_paths: list[Path] = []
    if args.current_manifest:
        manifest_rows = load_current_manifest(current_path)
        for row in manifest_rows:
            manifest_input_paths.extend(
                [
                    require_absolute_file(
                        Path(row["feature_path"]), "manifest feature"
                    ),
                    require_absolute_file(
                        Path(row["prediction_path"]), "manifest prediction"
                    ),
                ]
            )
        require_separate_output(output_root, manifest_input_paths)

    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise ValueError(f"output root is not a directory: {output_root}")
    existing = [name for name in OUTPUT_NAMES if (output_root / name).exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"Comparison outputs already exist; choose a new root or use --force: {existing}"
        )

    current_hash = sha256_file(current_path)
    reference_hash = sha256_file(reference_path)
    config_hash = sha256_file(config_path)
    current_stat = current_path.stat()
    reference_stat = reference_path.stat()
    temporary = temporary_paths(output_root)
    cleanup_temporaries(temporary)

    if args.current_predictions:
        current_rows: Iterator[CurrentRow] = iter_normalized_current(
            current_path, config
        )
        current_mode = "normalized_single_table"
        current_file_count = 1
    else:
        assert manifest_rows is not None
        current_rows = iter_current_manifest(manifest_rows, config)
        current_mode = "sha256_verified_per_field_feature_prediction_manifest"
        current_file_count = len(manifest_rows) * 2
    reference_rows = iter_reference(reference_path, config)

    cross_counts: Counter[tuple[str, str]] = Counter()
    endpoint_counts: Counter[tuple[str, str]] = Counter()
    current_state_counts: Counter[str] = Counter()
    reference_class_counts: Counter[str] = Counter()
    reference_status_counts: Counter[str] = Counter()
    group_totals: Counter[tuple[str, float]] = Counter()
    group_current: Counter[tuple[str, float, str]] = Counter()
    group_reference: Counter[tuple[str, float, str]] = Counter()
    group_current_abstentions: Counter[tuple[str, float]] = Counter()
    group_reference_unavailable: Counter[tuple[str, float]] = Counter()
    total_rows = 0
    reference_available_count = 0
    current_abstention_count = 0
    reference_unavailable_count = 0
    dead_decisive_count = 0
    dead_agreement_count = 0
    reference_model_id: str | None = None

    handles: list[TextIO] = []
    try:
        universe_handle, universe_writer = tsv_writer(
            temporary["comparison_cell_universe.tsv"], UNIVERSE_COLUMNS
        )
        paired_handle, paired_writer = tsv_writer(
            temporary["paired_cell_predictions.tsv"], PAIRED_COLUMNS
        )
        subset_handle, subset_writer = tsv_writer(
            temporary["dead_endpoint_descriptive_subset.tsv"], DEAD_SUBSET_COLUMNS
        )
        handles.extend((universe_handle, paired_handle, subset_handle))
        sentinel = object()
        for pair_index, pair in enumerate(
            itertools.zip_longest(current_rows, reference_rows, fillvalue=sentinel), 1
        ):
            current_row, reference_row = pair
            if current_row is sentinel:
                assert isinstance(reference_row, ReferenceRow)
                raise ValueError(
                    "Strict stable-cell universe mismatch: current predictions ended before "
                    f"reference row {reference_row.row_number} ({reference_row.cell_id})"
                )
            if reference_row is sentinel:
                assert isinstance(current_row, CurrentRow)
                raise ValueError(
                    "Strict stable-cell universe mismatch: reference predictions ended before "
                    f"current row {current_row.row_number} ({current_row.cell_id})"
                )
            assert isinstance(current_row, CurrentRow)
            assert isinstance(reference_row, ReferenceRow)
            if current_row.cell_id != reference_row.cell_id:
                raise ValueError(
                    "Strict stable-cell join/order mismatch at paired row "
                    f"{pair_index}: current={current_row.cell_id!r} "
                    f"reference={reference_row.cell_id!r}"
                )
            if reference_model_id is None:
                reference_model_id = reference_row.model_id
            elif reference_row.model_id != reference_model_id:
                raise AssertionError(
                    "Reference iterator failed its single-model contract"
                )

            total_rows += 1
            group = (current_row.well, current_row.elapsed_hours)
            group_totals[group] += 1
            current_state_counts[current_row.final_state] += 1
            group_current[(*group, current_row.final_state)] += 1
            current_call = config["current_dead_endpoint"][current_row.final_state]
            if current_call == "abstain":
                current_abstention_count += 1
                group_current_abstentions[group] += 1

            reference_status_counts[reference_row.prediction_status] += 1
            if reference_row.prediction_status == config["reference_available_status"]:
                reference_available_count += 1
                reference_class_counts[reference_row.class_id] += 1
                group_reference[(*group, reference_row.class_id)] += 1
                cross_counts[(current_row.final_state, reference_row.class_id)] += 1
                reference_call = config["reference_dead_endpoint"][
                    reference_row.class_id
                ]
            else:
                reference_unavailable_count += 1
                group_reference_unavailable[group] += 1
                reference_call = "abstain"

            pair_status = dead_pair_status(current_call, reference_call)
            endpoint_counts[(current_call, reference_call)] += 1
            row = {
                "cell_id": current_row.cell_id,
                "key": current_row.key,
                "mask_label": current_row.mask_label,
                "well": current_row.well,
                "elapsed_hours": format_number(current_row.elapsed_hours),
                "current_final_state": current_row.final_state,
                "current_prediction_status": current_row.prediction_status,
                "reference_cell_state_model_id": reference_row.model_id,
                "reference_cell_state_class_id": reference_row.class_id,
                "reference_cell_state_prediction_status": reference_row.prediction_status,
                "current_dead_endpoint_call": current_call,
                "reference_dead_endpoint_call": reference_call,
                "dead_endpoint_pair_status": pair_status,
            }
            universe_writer.writerow(
                {
                    "cell_id": current_row.cell_id,
                    "key": current_row.key,
                    "mask_label": current_row.mask_label,
                    "well": current_row.well,
                    "elapsed_hours": format_number(current_row.elapsed_hours),
                    "current_input_row_number": current_row.row_number,
                    "reference_input_row_number": reference_row.row_number,
                    "join_status": "matched_exact_stable_cell_id",
                }
            )
            paired_writer.writerow(row)
            if pair_status != "abstention":
                dead_decisive_count += 1
                dead_agreement_count += int(pair_status == "agreement")
                subset_writer.writerow(row)
            if total_rows % 1_000_000 == 0:
                print(f"comparison_rows={total_rows}", flush=True)

        if total_rows == 0:
            raise ValueError("Comparison cell universe is empty")
        for handle in handles:
            handle.close()
        handles.clear()

        write_summary_tables(
            temporary,
            config,
            cross_counts,
            endpoint_counts,
            group_totals,
            group_current,
            group_reference,
            group_current_abstentions,
            group_reference_unavailable,
        )

        current_after = current_path.stat()
        reference_after = reference_path.stat()
        if (current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns) != (
            current_after.st_ino,
            current_after.st_size,
            current_after.st_mtime_ns,
        ):
            raise RuntimeError("Current input changed while comparison was running")
        if (
            reference_stat.st_ino,
            reference_stat.st_size,
            reference_stat.st_mtime_ns,
        ) != (
            reference_after.st_ino,
            reference_after.st_size,
            reference_after.st_mtime_ns,
        ):
            raise RuntimeError("Reference input changed while comparison was running")

        join_audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "join_strategy": "streaming_exact_order_stable_cell_id",
            "stable_cell_id_contract": "original|<field-key>|<Combined-mask-label>",
            "current_row_count": total_rows,
            "reference_row_count": total_rows,
            "matched_row_count": total_rows,
            "current_only_count": 0,
            "reference_only_count": 0,
            "duplicate_cell_id_count": 0,
            "current_state_counts": dict(sorted(current_state_counts.items())),
            "reference_class_counts": dict(sorted(reference_class_counts.items())),
            "reference_prediction_status_counts": dict(
                sorted(reference_status_counts.items())
            ),
            "reference_available_count": reference_available_count,
            "reference_unavailable_count": reference_unavailable_count,
            "current_dead_endpoint_abstention_count": current_abstention_count,
            "dead_endpoint_jointly_decisive_count": dead_decisive_count,
            "dead_endpoint_descriptive_agreement_count": dead_agreement_count,
            "interpretation": (
                "Identity and descriptive association audit only; no human gold labels "
                "were used and no accuracy is claimed."
            ),
        }
        write_json(temporary["join_audit.json"], join_audit)

        nonreceipt_names = [
            name for name in OUTPUT_NAMES if name != "comparison_receipt.json"
        ]
        output_artifacts = {
            name: {
                "path": str(output_root / name),
                "sha256": sha256_file(temporary[name]),
                "size_bytes": temporary[name].stat().st_size,
            }
            for name in nonreceipt_names
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PRECOMPARISON_COMPLETE",
            "comparison_mode": "PRECOMPARISON_NO_GOLD_STANDARD",
            "scientific_claim": "DESCRIPTIVE_ASSOCIATION_ONLY",
            "accuracy_claimed": False,
            "ground_truth_included": False,
            "sensitivity_specificity_reported": False,
            "input_models_modified": False,
            "input_directories_written": False,
            "results_fed_back_to_either_model": False,
            "current_axis_overwritten": False,
            "reference_axis_overwritten": False,
            "multinucleated_cell_renamed_or_collapsed_to_live": False,
            "stable_cell_id_contract": "original|<field-key>|<Combined-mask-label>",
            "current_input_mode": current_mode,
            "row_count": total_rows,
            "reference_model_id": reference_model_id,
            "reference_available_count": reference_available_count,
            "reference_unavailable_count": reference_unavailable_count,
            "current_dead_endpoint_abstention_count": current_abstention_count,
            "dead_endpoint_jointly_decisive_count": dead_decisive_count,
            "inputs": {
                "config": {
                    "path": str(config_path),
                    "sha256": config_hash,
                },
                "current": {
                    "path": str(current_path),
                    "sha256": current_hash,
                    "mode": current_mode,
                    "verified_source_file_count": current_file_count,
                },
                "reference_predictions": {
                    "path": str(reference_path),
                    "sha256": reference_hash,
                    "exact_columns": list(REFERENCE_COLUMNS),
                },
            },
            "outputs": output_artifacts,
            "limitations": [
                "No prediction-blind human gold labels are included.",
                "Cross-classification tables are association tables, not confusion matrices.",
                "Dead-endpoint pair agreement is descriptive and is not accuracy.",
                "Cell rows are not treated as independent biological replicates.",
                "Reference class fractions are conditional on available reference predictions.",
            ],
        }
        write_json(temporary["comparison_receipt.json"], receipt)
        install_outputs(output_root, temporary)
    except Exception:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        cleanup_temporaries(temporary)
        raise

    print(f"comparison_status=PRECOMPARISON_COMPLETE rows={total_rows}", flush=True)
    print(f"comparison_root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
