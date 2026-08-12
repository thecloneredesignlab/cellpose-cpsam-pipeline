#!/usr/bin/env python3
"""Prepare a physical development-only project for UMAP, review, and training.

The full on-disk adapter tables remain an immutable audit universe, but are too
large for the generic R in-memory classifier.  This command streams them into a
deterministic representative project containing development wells only.  UMAP,
annotation, image review, and model training all use this single project
identity.  Heldout wells are frozen and excluded; no heldout performance claim
is made by this preparation stage.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "broad_phenotype_projection_input_v1"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
ADAPTER_SCHEMA_VERSION = "broad_phenotype_cpa_adapter_v1"
WELL_SPLIT_FIELDS = [
    "well",
    "plate_row",
    "plate_column",
    "doxorubicin_nm",
    "ploidy",
    "cyclophosphamide",
    "replicate",
    "split",
    "split_strategy",
    "split_seed",
    "assignment_sha256",
]
CONDITION_SPLIT_FIELDS = [
    "condition_id",
    "doxorubicin_nm",
    "dose_rank_within_stratum",
    "ploidy",
    "cyclophosphamide",
    "replicate_1_well",
    "replicate_2_well",
    "selected_for_heldout",
    "heldout_replicate",
    "heldout_well",
    "development_wells",
    "development_support_count",
]
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument(
        "--well-split-freeze",
        type=Path,
        help="Defaults to <shadow>/workflow_status/adapter/well_split_freeze.tsv.",
    )
    parser.add_argument("--max-cells-per-field", type=int, default=25)
    parser.add_argument(
        "--max-fields-per-well",
        type=int,
        default=20,
        help=(
            "Deterministic cap on development fields retained per well. "
            "The field cap bounds repeated image hashing/readback in the R stages."
        ),
    )
    parser.add_argument("--max-cells-per-well", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args(argv)


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be inside shadow root {root}: {resolved}") from error
    return resolved


def resolve_project_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project.parent / path
    return path.resolve()


def load_project(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise RuntimeError("Non-JSON YAML projects require PyYAML") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Project must contain one mapping: {path}")
    if value.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported project schema: {value.get('schema_version')}")
    return value


def read_tsv(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = path.open("r", newline="", encoding="utf-8-sig")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"TSV has no header: {path}")
    fields = [str(field) for field in reader.fieldnames]
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        handle.close()
        raise ValueError(f"TSV has blank or duplicate columns: {path}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {field: str(row.get(field, "") or "") for field in fields}
        finally:
            handle.close()

    return fields, rows()


def install_frozen(temporary: Path, output: Path) -> None:
    if output.exists():
        if output.is_file() and sha256_file(output) == sha256_file(temporary):
            temporary.unlink()
            return
        raise FileExistsError(f"Refusing to replace a non-identical frozen artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)


def write_text_frozen(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        install_frozen(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_tsv_frozen(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        install_frozen(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_split_freeze(
    path: Path, condition_path: Path, adapter_manifest_path: Path, shadow_root: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    if not condition_path.is_file() or not adapter_manifest_path.is_file():
        raise FileNotFoundError(
            f"Split provenance is incomplete: conditions={condition_path} "
            f"adapter_manifest={adapter_manifest_path}"
        )
    adapter = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(adapter, dict) or adapter.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed adapter manifest")
    split_metadata = adapter.get("split")
    output_hashes = adapter.get("output_sha256")
    if not isinstance(split_metadata, dict) or not isinstance(output_hashes, dict):
        raise ValueError("Adapter manifest lacks split or output hash provenance")
    for artifact in (path, condition_path):
        relative = str(artifact.relative_to(shadow_root))
        expected_hash = output_hashes.get(relative)
        if not isinstance(expected_hash, str) or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Adapter split artifact hash mismatch: {artifact}")

    fields, row_iterator = read_tsv(path)
    if fields != WELL_SPLIT_FIELDS:
        raise ValueError(
            f"Well split freeze header drift: expected={WELL_SPLIT_FIELDS} observed={fields}"
        )
    rows = list(row_iterator)
    result: dict[str, str] = {}
    for row in rows:
        well, split = row["well"], row["split"]
        if split not in {"development", "heldout"} or well in result:
            raise ValueError(f"Invalid or duplicate frozen well split: {well}={split}")
        if row["split_strategy"] != split_metadata.get("strategy"):
            raise ValueError(f"Well split strategy disagrees with adapter manifest: {well}")
        try:
            split_seed = int(row["split_seed"])
        except ValueError as error:
            raise ValueError(f"Invalid frozen split seed for well={well}") from error
        if split_seed != split_metadata.get("seed"):
            raise ValueError(f"Well split seed disagrees with adapter manifest: {well}")
        expected_assignment = stable_hash(
            split_metadata.get("policy_version", ""),
            split_metadata.get("strategy", ""),
            split_seed,
            split_metadata.get("plate_map_semantic_sha256", ""),
            well,
            row["doxorubicin_nm"],
            row["ploidy"],
            row["cyclophosphamide"],
            row["replicate"],
            split,
        )
        if row["assignment_sha256"] != expected_assignment:
            raise ValueError(f"Frozen split assignment hash mismatch: {well}")
        result[well] = split
    if "development" not in result.values() or "heldout" not in result.values():
        raise ValueError("Frozen split requires nonempty development and heldout wells")

    condition_fields, condition_iterator = read_tsv(condition_path)
    if condition_fields != CONDITION_SPLIT_FIELDS:
        raise ValueError(
            "Condition split freeze header drift: "
            f"expected={CONDITION_SPLIT_FIELDS} observed={condition_fields}"
        )
    condition_rows = list(condition_iterator)
    if split_metadata.get("strategy") == "condition_balanced_paired_replicate_v1":
        if len(rows) != 80 or len(condition_rows) != 40:
            raise ValueError("Formal condition-balanced split requires 80 wells and 40 conditions")
        if sum(value == "development" for value in result.values()) != 64:
            raise ValueError("Formal split must contain exactly 64 development wells")
        if sum(value == "heldout" for value in result.values()) != 16:
            raise ValueError("Formal split must contain exactly 16 heldout wells")
        selected = [
            row for row in condition_rows if row["selected_for_heldout"] == "true"
        ]
        if len(selected) != 16 or any(
            int(row["development_support_count"]) < 1 for row in condition_rows
        ):
            raise ValueError("Condition-balanced split coverage invariant failed")
        observed_replicates = {
            replicate: sum(row["heldout_replicate"] == replicate for row in selected)
            for replicate in ("1", "2")
        }
        if observed_replicates != {"1": 8, "2": 8}:
            raise ValueError(
                f"Heldout replicate balance changed: {observed_replicates}"
            )
        for name, expected in (
            ("condition_count", 40),
            ("heldout_condition_count", 16),
            ("development_well_count", 64),
            ("heldout_well_count", 16),
            ("heldout_replicate_1_count", 8),
            ("heldout_replicate_2_count", 8),
        ):
            if split_metadata.get(name) != expected:
                raise ValueError(f"Adapter split metadata changed for {name}")
    elif condition_rows:
        raise ValueError("Explicit test split must not claim treatment-condition rows")
    return result, split_metadata


def finite(value: str) -> bool:
    if value.strip().lower() in MISSING_TOKENS:
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


class Candidate:
    """A bounded-heap record whose heap order puts the worst rank first."""

    __slots__ = ("cell_id", "image_id", "well", "stable_rank", "field_rank")

    def __init__(
        self,
        cell_id: str,
        image_id: str,
        well: str,
        stable_rank: str,
        field_rank: int = 0,
    ) -> None:
        self.cell_id = cell_id
        self.image_id = image_id
        self.well = well
        self.stable_rank = stable_rank
        self.field_rank = field_rank

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.stable_rank, self.cell_id

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.sort_key > other.sort_key


def push_bounded(heap: list[Candidate], candidate: Candidate, cap: int) -> None:
    if len(heap) < cap:
        heapq.heappush(heap, candidate)
    elif candidate.sort_key < heap[0].sort_key:
        heapq.heapreplace(heap, candidate)


class FieldCandidate:
    """A bounded-heap field carrying only its bounded cell candidates."""

    __slots__ = ("image_id", "well", "stable_rank", "cells")

    def __init__(
        self,
        image_id: str,
        well: str,
        stable_rank: str,
        cells: list[Candidate],
    ) -> None:
        self.image_id = image_id
        self.well = well
        self.stable_rank = stable_rank
        self.cells = cells

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.stable_rank, self.image_id

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FieldCandidate):
            return NotImplemented
        return self.sort_key > other.sort_key


def push_bounded_field(
    heap: list[FieldCandidate], candidate: FieldCandidate, cap: int
) -> None:
    if len(heap) < cap:
        heapq.heappush(heap, candidate)
    elif candidate.sort_key < heap[0].sort_key:
        heapq.heapreplace(heap, candidate)


def stream_representative_selection(
    cells_path: Path,
    features_path: Path,
    primary_features: Sequence[str],
    splits: dict[str, str],
    seed: int,
    max_field: int,
    max_fields_per_well: int,
    max_well: int,
) -> tuple[
    list[str],
    int,
    int,
    dict[tuple[str, str], int],
    list[dict[str, Any]],
]:
    """Join two ordered full tables and retain only bounded development heaps.

    Script 17 emits cells and features in identical field/cell order.  Enforcing
    that contract lets this pass prove the one-to-one join without a 38.6M-row
    index.  At most ``max_field`` candidates for the current image and
    ``max_fields_per_well`` fields and ``max_well`` cells per well are
    resident.  Selecting fields before cells prevents a representative project
    from touching thousands of source images while retaining the same maximum
    number of cells per well.
    """

    cell_fields, cell_rows = read_tsv(cells_path)
    required = {"cell_id", "image_id", "key", "well", "branch", "mask_label", "split"}
    if not required.issubset(cell_fields):
        raise ValueError(f"Cells are missing metadata: {sorted(required - set(cell_fields))}")
    feature_fields, feature_rows = read_tsv(features_path)
    missing = sorted({"cell_id", *primary_features} - set(feature_fields))
    if missing:
        raise ValueError(f"Features are missing project allowlist columns: {missing}")

    sentinel = object()
    field_heap: list[Candidate] = []
    well_field_heaps: dict[str, list[FieldCandidate]] = {}
    split_counts: dict[tuple[str, str], int] = {}
    closed_images: set[str] = set()
    current_image = ""
    current_well = ""
    previous_label = 0
    cell_count = 0
    incomplete = 0

    def flush_field() -> None:
        nonlocal field_heap
        if not field_heap:
            return
        ranked_cells: list[Candidate] = []
        for rank, candidate in enumerate(
            sorted(field_heap, key=lambda item: item.sort_key), start=1
        ):
            ranked_cells.append(
                Candidate(
                    candidate.cell_id,
                    candidate.image_id,
                    candidate.well,
                    candidate.stable_rank,
                    rank,
                )
            )
        field = FieldCandidate(
            image_id=current_image,
            well=current_well,
            stable_rank=stable_hash(seed, "representative_field", current_image),
            cells=ranked_cells,
        )
        well_heap = well_field_heaps.setdefault(current_well, [])
        push_bounded_field(well_heap, field, max_fields_per_well)
        field_heap = []

    paired = itertools.zip_longest(cell_rows, feature_rows, fillvalue=sentinel)
    for position, pair in enumerate(paired, start=1):
        cell_row, feature_row = pair
        if cell_row is sentinel or feature_row is sentinel:
            raise ValueError(
                "Cell/feature one-to-one lockstep join failed: "
                f"first unmatched row={position} cells_exhausted={cell_row is sentinel} "
                f"features_exhausted={feature_row is sentinel}"
            )
        assert isinstance(cell_row, dict) and isinstance(feature_row, dict)
        cell_id = cell_row["cell_id"]
        if not cell_id or feature_row["cell_id"] != cell_id:
            raise ValueError(
                "Cell/feature one-to-one lockstep join failed: "
                f"row={position} cell_id={cell_id!r} feature_cell_id={feature_row['cell_id']!r}"
            )
        image_id = cell_row["image_id"]
        well = cell_row["well"]
        split = cell_row["split"]
        if image_id != cell_row["key"]:
            raise ValueError(f"Cell image_id/key mismatch: {cell_id}")
        if well not in splits or split != splits[well]:
            raise ValueError(f"Cell disagrees with frozen well split: {cell_id}")

        if image_id != current_image:
            if current_image:
                flush_field()
                closed_images.add(current_image)
            if image_id in closed_images:
                raise ValueError(f"Cells for one image are not contiguous: {image_id}")
            current_image = image_id
            current_well = well
            previous_label = 0
        elif well != current_well:
            raise ValueError(f"One image spans multiple wells: {image_id}")
        try:
            mask_label = int(cell_row["mask_label"])
        except ValueError as error:
            raise ValueError(f"Cell mask_label is not an integer: {cell_id}") from error
        if mask_label <= previous_label:
            raise ValueError(
                f"Cells are not strictly ordered by mask_label within image={image_id}: "
                f"previous={previous_label} current={mask_label}"
            )
        expected_cell_id = f"{cell_row['branch']}|{cell_row['key']}|{mask_label}"
        if cell_id != expected_cell_id:
            raise ValueError(f"Cell identity contract mismatch: {cell_id} != {expected_cell_id}")
        previous_label = mask_label

        split_counts[(well, split)] = split_counts.get((well, split), 0) + 1
        complete = all(finite(feature_row[column]) for column in primary_features)
        incomplete += int(not complete)
        if split == "development" and complete:
            candidate = Candidate(
                cell_id=cell_id,
                image_id=image_id,
                well=well,
                stable_rank=stable_hash(seed, "representative", cell_id),
            )
            push_bounded(field_heap, candidate, max_field)
        cell_count += 1
        if position % 1_000_000 == 0:
            print(f"projection_join_rows={position}", flush=True)
    flush_field()

    selection_rows: list[dict[str, Any]] = []
    for well in sorted(well_field_heaps):
        selected_fields = sorted(
            well_field_heaps[well], key=lambda item: item.sort_key
        )
        selected_field_rank = {
            field.image_id: rank for rank, field in enumerate(selected_fields, start=1)
        }
        well_cell_heap: list[Candidate] = []
        for field in selected_fields:
            for candidate in field.cells:
                push_bounded(well_cell_heap, candidate, max_well)
        for well_rank, candidate in enumerate(
            sorted(well_cell_heap, key=lambda item: item.sort_key), start=1
        ):
            selection_rows.append(
                {
                    "cell_id": candidate.cell_id,
                    "image_id": candidate.image_id,
                    "well": candidate.well,
                    "split": "development",
                    "stable_rank_sha256": candidate.stable_rank,
                    "field_rank": candidate.field_rank,
                    "selected_field_rank": selected_field_rank[candidate.image_id],
                    "well_rank": well_rank,
                }
            )
    return cell_fields, cell_count, incomplete, split_counts, selection_rows


def copy_selected_rows(
    source: Path,
    output: Path,
    output_fields: Sequence[str],
    selected_ids: set[str],
) -> int:
    source_fields, rows = read_tsv(source)
    missing = sorted(set(output_fields) - set(source_fields))
    if missing:
        raise ValueError(f"Selected-row source is missing columns: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_fields), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                if row["cell_id"] in selected_ids:
                    writer.writerow({field: row[field] for field in output_fields})
                    count += 1
        install_frozen(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if count != len(selected_ids):
        raise ValueError(f"Selected row coverage failed: expected={len(selected_ids)} observed={count}")
    return count


def copy_selected_images(source: Path, output: Path, image_ids: set[str]) -> int:
    fields, rows = read_tsv(source)
    if "image_id" not in fields:
        raise ValueError("Images table requires image_id")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    count = 0
    seen_images: set[str] = set()
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if row["image_id"] in image_ids:
                    writer.writerow(row)
                    count += 1
                    seen_images.add(row["image_id"])
        install_frozen(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if seen_images != image_ids:
        raise ValueError(
            f"Representative image coverage failed: missing={sorted(image_ids-seen_images)[:10]}"
        )
    return count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.max_cells_per_field < 1
        or args.max_fields_per_well < 1
        or args.max_cells_per_well < 1
    ):
        raise ValueError("Representative caps must be positive")
    shadow_root = args.shadow_root.expanduser().resolve()
    project_path = require_inside(args.project, shadow_root, "Base project")
    if not project_path.is_file():
        raise FileNotFoundError(project_path)
    split_path = require_inside(
        args.well_split_freeze
        if args.well_split_freeze
        else shadow_root / "workflow_status" / "adapter" / "well_split_freeze.tsv",
        shadow_root,
        "Well split freeze",
    )
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    adapter_root = require_inside(
        shadow_root / "workflow_status" / "adapter", shadow_root, "Adapter audit root"
    )
    condition_split_path = adapter_root / "condition_split_freeze.tsv"
    adapter_manifest_path = adapter_root / "adapter_manifest.json"
    base = load_project(project_path)
    projection = base.get("projection")
    classifier = base.get("classifier")
    if not isinstance(projection, dict) or projection.get("mode") != "compute_umap":
        raise ValueError("Base project must configure compute_umap")
    if not isinstance(classifier, dict):
        raise ValueError("Base project must configure a classifier")
    primary_features = list(projection.get("feature_columns") or [])
    if not primary_features or list(classifier.get("feature_columns") or []) != primary_features:
        raise ValueError("Projection/classifier primary allowlists must be identical and ordered")

    assets = {
        field: resolve_project_path(project_path, str(base[field]))
        for field in ("classes_file", "cells_file", "features_file", "images_file", "runs_dir")
    }
    for field, path in assets.items():
        require_inside(path, shadow_root, field)
        if field == "runs_dir":
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    splits, split_metadata = load_split_freeze(
        split_path, condition_split_path, adapter_manifest_path, shadow_root
    )
    projection_root = require_inside(shadow_root / "projection_input", shadow_root, "Projection root")
    compute_root = projection_root / "representative_umap"
    compute_root.mkdir(parents=True, exist_ok=True)
    cell_fields, cell_count, incomplete, split_counts, selection_rows = (
        stream_representative_selection(
            assets["cells_file"],
            assets["features_file"],
            primary_features,
            splits,
            args.seed,
            args.max_cells_per_field,
            args.max_fields_per_well,
            args.max_cells_per_well,
        )
    )
    observed_cell_wells = {well for well, _split in split_counts}
    if observed_cell_wells != set(splits):
        raise ValueError(
            "Full cells table well universe differs from frozen split: "
            f"missing={sorted(set(splits)-observed_cell_wells)} "
            f"extra={sorted(observed_cell_wells-set(splits))}"
        )
    selected_ids = {str(row["cell_id"]) for row in selection_rows}
    if len(selected_ids) != len(selection_rows):
        raise AssertionError("Representative selection contains duplicate cell IDs")
    if len(selected_ids) < 3:
        raise ValueError(f"Representative UMAP requires at least three cells: {len(selected_ids)}")
    if any(row["split"] != "development" for row in selection_rows):
        raise AssertionError("Heldout cells entered representative selection")
    selected_counts: dict[str, int] = {}
    for row in selection_rows:
        well = str(row["well"])
        selected_counts[well] = selected_counts.get(well, 0) + 1

    split_manifest = projection_root / "split_manifest.tsv"
    write_tsv_frozen(
        split_manifest,
        ["well", "split", "available_cells", "selected_for_umap", "heldout_exclusion_verified"],
        (
            {
                "well": well,
                "split": split,
                "available_cells": split_counts.get((well, split), 0),
                "selected_for_umap": selected_counts.get(well, 0),
                "heldout_exclusion_verified": str(
                    split != "heldout" or selected_counts.get(well, 0) == 0
                ).lower(),
            }
            for well, split in sorted(splits.items())
        ),
    )
    selection_path = compute_root / "selection.tsv"
    selection_fields = [
        "cell_id",
        "image_id",
        "well",
        "split",
        "stable_rank_sha256",
        "field_rank",
        "selected_field_rank",
        "well_rank",
    ]
    write_tsv_frozen(selection_path, selection_fields, selection_rows)

    compute_cells = compute_root / "cells.tsv"
    compute_features = compute_root / "features.tsv"
    compute_images = compute_root / "images.tsv"
    compute_classes = compute_root / "classes.tsv"
    # Second sequential pass copies only the at-most wells*cap selected rows.
    copy_selected_rows(assets["cells_file"], compute_cells, cell_fields, selected_ids)
    copy_selected_rows(
        assets["features_file"], compute_features, ["cell_id", *primary_features], selected_ids
    )
    selected_image_ids = {str(row["image_id"]) for row in selection_rows}
    image_row_count = copy_selected_images(
        assets["images_file"], compute_images, selected_image_ids
    )
    write_text_frozen(compute_classes, assets["classes_file"].read_text(encoding="utf-8"))
    compute_project = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_id": f"{base['project_id']}_development",
        "classes_file": "classes.tsv",
        "cells_file": "cells.tsv",
        "features_file": "features.tsv",
        "images_file": "images.tsv",
        "runs_dir": "runs",
        "projection": {
            **projection,
            "feature_columns": primary_features,
            "missing_policy": "fail",
            "constant_policy": "drop_with_manifest",
            "seed": args.seed,
            "n_neighbors": min(int(projection.get("n_neighbors", 30)), len(selected_ids) - 1),
            "n_threads": 1,
            "n_sgd_threads": 1,
        },
        "annotation": base.get("annotation"),
        "review": base.get("review"),
        "classifier": {
            **classifier,
            "feature_columns": primary_features,
            "group_column": "well",
            "allow_ungrouped": False,
        },
    }
    compute_project_path = compute_root / "project.yml"
    write_text_frozen(
        compute_project_path,
        json.dumps(compute_project, indent=2, sort_keys=False) + "\n",
    )
    prepare_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "development_representative_project",
        "base_project": str(project_path),
        "base_project_sha256": sha256_file(project_path),
        "well_split_freeze": str(split_path),
        "well_split_freeze_sha256": sha256_file(split_path),
        "condition_split_freeze": str(condition_split_path),
        "condition_split_freeze_sha256": sha256_file(condition_split_path),
        "adapter_manifest": str(adapter_manifest_path),
        "adapter_manifest_sha256": sha256_file(adapter_manifest_path),
        "split": split_metadata,
        "full_cell_count": cell_count,
        "incomplete_primary_feature_count": incomplete,
        "representative_development_cell_count": len(selected_ids),
        "representative_heldout_cell_count": 0,
        "heldout_exclusion_verified": True,
        "heldout_evaluation_performed": False,
        "heldout_evaluation_status": "not_run",
        "selection_policy": (
            "lockstep stream; stable-hash cell cap within field; stable-hash field cap "
            "within development well; stable-hash final cell cap within well"
        ),
        "selection_memory_bound_candidates": (
            len(splits) * args.max_fields_per_well * args.max_cells_per_field
            + args.max_cells_per_field
        ),
        "seed": args.seed,
        "max_cells_per_field": args.max_cells_per_field,
        "max_fields_per_well": args.max_fields_per_well,
        "max_cells_per_well": args.max_cells_per_well,
        "primary_features": primary_features,
        "representative_project": str(compute_project_path),
        "representative_project_sha256": sha256_file(compute_project_path),
        "representative_image_count": len(selected_image_ids),
        "representative_field_count": len(selected_image_ids),
        "representative_field_cap_verified": all(
            sum(1 for image_id in selected_image_ids if image_id.startswith(f"{well}_"))
            <= args.max_fields_per_well
            for well in splits
        ),
        "representative_image_manifest_rows": image_row_count,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
    }
    prepare_manifest_path = projection_root / "projection_input_manifest.json"
    write_text_frozen(
        prepare_manifest_path,
        json.dumps(prepare_manifest, indent=2, sort_keys=True) + "\n",
    )

    print(f"representative_project={compute_project_path}")
    print(f"projection_input_manifest={prepare_manifest_path}")
    print("heldout_selected=0")
    print("heldout_evaluation_status=not_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
