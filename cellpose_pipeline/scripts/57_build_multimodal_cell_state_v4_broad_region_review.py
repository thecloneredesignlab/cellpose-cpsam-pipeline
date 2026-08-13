#!/usr/bin/env python3
"""Freeze a blinded 500-cell V4 review across a provisional death region."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "multimodal_cell_state_v4_broad_region_review_v1"
PROJECT_ID = "multimodal_cell_state_v4_death_resolution"
REVIEW_FIELDS = [
    "morphology_umap_row_key",
    "review_default_label",
    "review_sampling_bucket",
    "context_key",
    "source_id",
    "nuclei_measurement_status_audit",
    "dead_measurement_status_audit",
    "provisional_region_class_audit",
    "normalized_distance_to_death_region_boundary_audit",
]
BUCKETS = (
    "death_region_boundary_inside",
    "death_region_boundary_outside",
    "death_region_interior",
    "death_region_exterior",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--annotation-import-dir", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total", type=int, default=500)
    parser.add_argument("--max-per-well", type=int, default=8)
    parser.add_argument("--minimum-per-boundary-side", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_sha(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, rows = list(reader.fieldnames or ()), list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Invalid TSV: {path}")
    return fields, rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=REVIEW_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 broad review contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            output[path.relative_to(root).as_posix()] = sha256(path)
    return output


def polygon_vertices(region: dict[str, Any]) -> np.ndarray:
    polygon = region.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("V4 death region has an invalid polygon")
    points = []
    for point in polygon:
        if isinstance(point, dict):
            values = (point.get("x"), point.get("y"))
        elif isinstance(point, list) and len(point) == 2:
            values = point
        else:
            raise ValueError("V4 death region polygon point is invalid")
        x, y = map(float, values)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("V4 death region polygon is nonfinite")
        points.append((x, y))
    return np.asarray(points, dtype=float)


def distances_to_polygons(points: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    output = np.full(points.shape[0], np.inf, dtype=float)
    for polygon in polygons:
        for index in range(len(polygon)):
            start = polygon[index - 1]
            end = polygon[index]
            vector = end - start
            denominator = float(np.dot(vector, vector))
            if denominator == 0:
                distances = np.sum((points - start) ** 2, axis=1)
            else:
                fraction = np.clip(((points - start) @ vector) / denominator, 0, 1)
                nearest = start + fraction[:, None] * vector
                distances = np.sum((points - nearest) ** 2, axis=1)
            output = np.minimum(output, distances)
    if not np.all(np.isfinite(output)):
        raise ValueError("Could not compute distance to the V4 death-region boundary")
    return np.sqrt(output)


def hashed(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode()).hexdigest()


def select_bucket(
    rows: list[dict[str, Any]], target: int, well_counts: Counter[str],
    max_per_well: int, seed: int,
) -> list[dict[str, Any]]:
    by_well: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_well[row["source_id"]].append(row)
    for well, values in by_well.items():
        values.sort(
            key=lambda row: (
                row["distance_sort"],
                hashed(
                    seed,
                    row["nuclei_measurement_status"],
                    row["dead_measurement_status"],
                    row["cell_id"],
                ),
            )
        )
    wells = sorted(by_well, key=lambda well: hashed(seed, well))
    chosen: list[dict[str, Any]] = []
    while len(chosen) < target:
        progressed = False
        for well in wells:
            if well_counts[well] >= max_per_well or not by_well[well]:
                continue
            chosen.append(by_well[well].pop(0))
            well_counts[well] += 1
            progressed = True
            if len(chosen) == target:
                break
        if not progressed:
            break
    return chosen


def verify_existing(output: Path, inputs: dict[str, str], implementation: str) -> None:
    manifest = load_json(output / "broad_region_review_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "HUMAN_REVIEW_REQUIRED"
        or manifest.get("inputs") != inputs
        or manifest.get("implementation_sha256") != implementation
    ):
        raise ValueError("Existing V4 broad-region review identity differs")
    observed = artifact_hashes(output, {"broad_region_review_manifest.json"})
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 broad-region review artifact set/hash differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_total < 4 or args.max_per_well < 1:
        raise ValueError("V4 broad-region review size/cap is invalid")
    shadow = args.shadow_root.expanduser().resolve(strict=True)
    project_path = args.project.expanduser().resolve(strict=True)
    annotation = args.annotation_import_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    for path in (project_path, annotation):
        if path.is_symlink() or shadow not in path.parents:
            raise ValueError(f"V4 broad-region review input escapes shadow root: {path}")
    if output == shadow or shadow not in output.parents:
        raise ValueError("V4 broad-region review output escapes shadow root")
    project = load_json(project_path)
    if project.get("project_id") != PROJECT_ID:
        raise ValueError("Broad-region review requires the death-resolution project")
    annotation_manifest_path = annotation / "annotation_import_manifest.json"
    labels_path = annotation / "provisional_labels.tsv"
    regions_path = annotation / "regions.json"
    annotation_manifest = load_json(annotation_manifest_path)
    declared = annotation_manifest.get("artifact_file_sha256", {})
    if (
        annotation_manifest.get("schema_version")
        != "cell_phenotype_annotator_annotation_import_v1"
        or set(declared) != {"region_submission", "regions", "provisional_labels"}
        or declared.get("regions") != sha256(regions_path)
        or declared.get("provisional_labels") != sha256(labels_path)
    ):
        raise ValueError("V4 annotation import authority differs")
    project_dir = project_path.parent
    _, cells = read_tsv(project_dir / str(project["cells_file"]))
    _, coordinates = read_tsv(
        project_dir / str(project["projection"]["coordinate_file"])
    )
    _, labels = read_tsv(labels_path)
    cell_ids = [row["cell_id"] for row in cells]
    coordinate_by_id = {
        row["cell_id"]: (float(row["Dim1"]), float(row["Dim2"]))
        for row in coordinates
    }
    label_by_id = {row["cell_id"]: row for row in labels}
    if (
        len(cell_ids) != len(set(cell_ids))
        or set(coordinate_by_id) != set(cell_ids)
        or set(label_by_id) != set(cell_ids)
    ):
        raise ValueError("V4 broad-region cell/coordinate/annotation universes differ")
    regions = load_json(regions_path).get("regions")
    if not isinstance(regions, list):
        raise ValueError("V4 regions.json lacks regions")
    dead_polygons = [
        polygon_vertices(region)
        for region in regions
        if region.get("assignment_state") == "class_assigned"
        and region.get("class_id") == "dead"
    ]
    if not dead_polygons:
        raise ValueError("V4 region submission contains no provisional dead polygon")
    matrix = np.asarray([coordinate_by_id[cell_id] for cell_id in cell_ids], dtype=float)
    span = np.ptp(matrix, axis=0)
    if np.any(~np.isfinite(span)) or np.any(span <= 0):
        raise ValueError("V4 death-resolution coordinates are degenerate")
    normalized = (matrix - np.min(matrix, axis=0)) / span
    normalized_polygons = [
        (polygon - np.min(matrix, axis=0)) / span for polygon in dead_polygons
    ]
    distance = distances_to_polygons(normalized, normalized_polygons)
    joined: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        label = label_by_id[cell["cell_id"]]
        inside = (
            label.get("assignment_state") == "class_assigned"
            and label.get("class_id") == "dead"
        )
        joined.append(
            {
                **cell,
                "inside": inside,
                "distance": float(distance[index]),
                "provisional_class": "dead" if inside else "outside_dead_region",
            }
        )
    inside = sorted((row for row in joined if row["inside"]), key=lambda row: row["distance"])
    outside = sorted((row for row in joined if not row["inside"]), key=lambda row: row["distance"])
    if min(len(inside), len(outside)) < 2 * args.minimum_per_boundary_side:
        raise ValueError("V4 provisional death region lacks enough inside/outside review support")
    inside_boundary_count = max(args.minimum_per_boundary_side, len(inside) // 4)
    outside_boundary_count = max(args.minimum_per_boundary_side, len(outside) // 20)
    pools = {
        "death_region_boundary_inside": inside[:inside_boundary_count],
        "death_region_boundary_outside": outside[:outside_boundary_count],
        "death_region_interior": inside[inside_boundary_count:],
        "death_region_exterior": outside[outside_boundary_count:],
    }
    for bucket, values in pools.items():
        near = "boundary" in bucket
        for row in values:
            row["distance_sort"] = row["distance"] if near else -row["distance"]
            row["bucket"] = bucket
    base, remainder = divmod(args.max_total, len(BUCKETS))
    targets = {bucket: base + (index < remainder) for index, bucket in enumerate(BUCKETS)}
    well_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for bucket_index, bucket in enumerate(BUCKETS):
        chosen = select_bucket(
            pools[bucket], targets[bucket], well_counts, args.max_per_well,
            args.seed + 1009 * bucket_index,
        )
        if len(chosen) != targets[bucket]:
            raise ValueError(f"V4 review cannot satisfy bucket/well cap: {bucket}")
        selected.extend(chosen)
    if len(selected) != args.max_total or len({row["cell_id"] for row in selected}) != args.max_total:
        raise RuntimeError("V4 broad-region selection is not exact and unique")
    selected.sort(key=lambda row: (row["source_id"], hashed(args.seed, row["cell_id"])))
    review_rows = [
        {
            "morphology_umap_row_key": row["cell_id"],
            "review_default_label": "uncertain_or_unreviewable",
            "review_sampling_bucket": row["bucket"],
            "context_key": row["context_key"],
            "source_id": row["source_id"],
            "nuclei_measurement_status_audit": row["nuclei_measurement_status"],
            "dead_measurement_status_audit": row["dead_measurement_status"],
            "provisional_region_class_audit": row["provisional_class"],
            "normalized_distance_to_death_region_boundary_audit": format(
                row["distance"], ".17g"
            ),
        }
        for row in selected
    ]
    implementation = sha256(Path(__file__).resolve())
    inputs = {
        "project": sha256(project_path),
        "cells": sha256(project_dir / str(project["cells_file"])),
        "coordinates": sha256(
            project_dir / str(project["projection"]["coordinate_file"])
        ),
        "annotation_import_manifest": sha256(annotation_manifest_path),
        "provisional_labels": sha256(labels_path),
        "regions": sha256(regions_path),
    }
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs, implementation)
        print(f"multimodal_cell_state_v4_broad_region_review={output} verified_reuse=1")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        review_path = staging / "broad_region_review_set.tsv"
        write_tsv(review_path, review_rows)
        artifacts = artifact_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "HUMAN_REVIEW_REQUIRED",
            "inputs": inputs,
            "implementation_sha256": implementation,
            "row_count": len(review_rows),
            "stable_id_sha256": stable_id_sha(
                row["morphology_umap_row_key"] for row in review_rows
            ),
            "sampling_bucket_counts": dict(Counter(row["review_sampling_bucket"] for row in review_rows)),
            "maximum_per_well": args.max_per_well,
            "observed_maximum_per_well": max(well_counts.values()),
            "polygon_role": "provisional_sampling_and_navigation_only_not_training_truth",
            "reviewer_visibility": "BF_Dead_Nuclei_same_cell_only; UMAP_polygon_bucket_condition_time_hidden",
            "output_file_sha256": artifacts,
        }
        (staging / "broad_region_review_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v4_broad_region_review={output}")
    print("human_barrier=three_channel_broad_region_review_submission_required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
