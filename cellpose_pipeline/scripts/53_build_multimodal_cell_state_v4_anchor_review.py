#!/usr/bin/env python3
"""Freeze a 300-cell V4 independent three-channel death anchor review set."""

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


SCHEMA_VERSION = "multimodal_cell_state_v4_anchor_review_v1"
IDENTITY_SCHEMA_VERSION = "multimodal_cell_state_v4_anchor_review_generation_identity_v1"
PROJECT_ID = "multimodal_cell_state_v4_development"
REVIEW_FIELDS = (
    "morphology_umap_row_key",
    "review_default_label",
    "review_sampling_bucket",
    "context_key",
    "source_id",
    "diagnostic_cluster_audit",
    "nuclei_measurement_status_audit",
    "dead_measurement_status_audit",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total", type=int, default=300)
    parser.add_argument("--max-per-well", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_id_sha(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=REVIEW_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def resolve_asset(project_path: Path, project: dict[str, Any], field: str) -> Path:
    text = str(project.get(field, "")).strip()
    if not text:
        raise ValueError(f"V4 project lacks {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"V4 project asset is unavailable or a symlink: {path}")
    return path


def farthest(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["cell_id"])
    if count >= len(ordered):
        return ordered
    coordinates = np.asarray(
        [[row["Dim1"], row["Dim2"]] for row in ordered], dtype=float
    )
    tokens = [
        hashlib.sha256(f"{seed}|{row['cell_id']}".encode()).hexdigest()
        for row in ordered
    ]
    chosen = [min(range(len(ordered)), key=lambda index: tokens[index])]
    distance = np.sum((coordinates - coordinates[chosen[0]]) ** 2, axis=1)
    distance[chosen] = -1
    while len(chosen) < count:
        maximum = float(np.max(distance))
        candidates = np.flatnonzero(np.isclose(distance, maximum, rtol=0, atol=1e-15))
        index = int(min(candidates, key=lambda item: tokens[int(item)]))
        chosen.append(index)
        distance = np.minimum(
            distance, np.sum((coordinates - coordinates[index]) ** 2, axis=1)
        )
        distance[chosen] = -1
    return [ordered[index] for index in chosen]


def select_anchor(
    rows: list[dict[str, Any]], total: int, max_per_well: int, seed: int
) -> list[dict[str, Any]]:
    by_well: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_well[row["source_id"]].append(row)
    wells = sorted(by_well)
    if total > max_per_well * len(wells):
        raise ValueError("Anchor-review total exceeds well-capacity contract")
    base, remainder = divmod(total, len(wells))
    if base > max_per_well or base + bool(remainder) > max_per_well:
        raise ValueError("Anchor-review well cap cannot support balanced allocation")
    priority = sorted(
        wells, key=lambda well: hashlib.sha256(f"{seed}|{well}".encode()).hexdigest()
    )
    quotas = {well: base + (well in set(priority[:remainder])) for well in wells}
    selected: list[dict[str, Any]] = []
    for well_index, well in enumerate(wells):
        candidates = by_well[well]
        quota = quotas[well]
        strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            strata[
                (
                    row["diagnostic_cluster"], row["nuclei_measurement_status"],
                    row["dead_measurement_status"],
                )
            ].append(row)
        first_pass: list[dict[str, Any]] = []
        for stratum_index, key in enumerate(sorted(strata)):
            first_pass.extend(
                farthest(strata[key], 1, seed + 1009 * well_index + stratum_index)
            )
        if len(first_pass) >= quota:
            chosen = farthest(first_pass, quota, seed + 2003 * well_index)
        else:
            chosen = list(first_pass)
            chosen_ids = {row["cell_id"] for row in chosen}
            remaining_rows = [
                row for row in candidates if row["cell_id"] not in chosen_ids
            ]
            chosen.extend(
                farthest(remaining_rows, quota - len(chosen), seed + 3001 * well_index)
            )
        selected.extend(chosen)
    if len(selected) != total or len({row["cell_id"] for row in selected}) != total:
        raise RuntimeError(
            "Anchor-review selection did not produce the exact unique total"
        )
    return sorted(selected, key=lambda row: (row["source_id"], row["cell_id"]))


def directory_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 anchor-review generation contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            output[path.relative_to(root).as_posix()] = sha256_file(path)
    return output


def verify_existing(
    output: Path, inputs: dict[str, Any], implementation_sha: str
) -> None:
    manifest_path = output / "anchor_review_manifest.json"
    identity_path = output / "anchor_review_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing V4 anchor-review generation is partial")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "HUMAN_REVIEW_REQUIRED"
        or manifest.get("inputs") != inputs
        or manifest.get("implementation_sha256") != implementation_sha
    ):
        raise ValueError("Existing V4 anchor-review identity differs")
    observed = directory_hashes(
        output, {"anchor_review_manifest.json", "anchor_review_generation_identity.json"}
    )
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 anchor-review artifact set/hash differs")
    expected = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "status": "HUMAN_REVIEW_REQUIRED",
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": implementation_sha,
        "output_file_sha256": observed,
    }
    if load_json(identity_path) != expected:
        raise ValueError("Existing V4 anchor-review generation identity differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_total < 1 or args.max_per_well < 1:
        raise ValueError("Anchor-review size/cap must be positive")
    implementation = Path(__file__).resolve()
    implementation_sha = sha256_file(implementation)
    shadow = args.shadow_root.expanduser().resolve(strict=True)
    project_path = args.project.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if shadow.is_symlink() or not shadow.is_dir() or shadow not in project_path.parents:
        raise ValueError("V4 project/shadow-root identity differs")
    if output == shadow or shadow not in output.parents:
        raise ValueError("V4 anchor-review output must remain inside its shadow root")
    project = load_json(project_path)
    if project.get("project_id") != PROJECT_ID:
        raise ValueError("Unsupported V4 project")
    cells_path = resolve_asset(project_path, project, "cells_file")
    projection = project.get("projection", {})
    coordinate_path = (
        project_path.parent / str(projection.get("coordinate_file", ""))
    ).resolve(strict=True)
    cluster_path = project_path.parent / "projection" / "diagnostic_clusters.tsv"
    project_manifest_path = project_path.parent / "parent_import_manifest.json"
    projection_manifest_path = (
        project_path.parent / "projection" / "projection_manifest.json"
    )
    for path in (
        coordinate_path,
        cluster_path,
        project_manifest_path,
        projection_manifest_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"V4 anchor-review input is unavailable: {path}")
    _, cells = read_tsv(cells_path)
    _, coordinates = read_tsv(coordinate_path)
    _, clusters = read_tsv(cluster_path)
    ids = [row["cell_id"] for row in cells]
    if [row["cell_id"] for row in coordinates] != ids or [
        row["cell_id"] for row in clusters
    ] != ids:
        raise ValueError("V4 anchor-review input row universes differ")
    joined = [
        {
            **cell,
            "Dim1": float(coordinate["Dim1"]),
            "Dim2": float(coordinate["Dim2"]),
            "diagnostic_cluster": cluster["diagnostic_cluster"],
        }
        for cell, coordinate, cluster in zip(cells, coordinates, clusters, strict=True)
    ]
    if any(row.get("split") != "development" for row in joined):
        raise ValueError("Heldout cells entered V4 independent anchor review")
    selected = select_anchor(joined, args.max_total, args.max_per_well, args.seed)
    review_rows = [
        {
            "morphology_umap_row_key": row["cell_id"],
            "review_default_label": "uncertain_or_unreviewable",
            "review_sampling_bucket": "multimodal_v4_anchor",
            "context_key": row["context_key"],
            "source_id": row["source_id"],
            "diagnostic_cluster_audit": row["diagnostic_cluster"],
            "nuclei_measurement_status_audit": row["nuclei_measurement_status"],
            "dead_measurement_status_audit": row["dead_measurement_status"],
        }
        for row in selected
    ]
    selected_ids = [row["morphology_umap_row_key"] for row in review_rows]
    inputs = {
        "project": {"path": str(project_path), "sha256": sha256_file(project_path)},
        "cells": {"path": str(cells_path), "sha256": sha256_file(cells_path)},
        "coordinates": {
            "path": str(coordinate_path),
            "sha256": sha256_file(coordinate_path),
        },
        "clusters": {"path": str(cluster_path), "sha256": sha256_file(cluster_path)},
        "project_manifest": {
            "path": str(project_manifest_path),
            "sha256": sha256_file(project_manifest_path),
        },
        "projection_manifest": {
            "path": str(projection_manifest_path),
            "sha256": sha256_file(projection_manifest_path),
        },
        "implementation": {"path": str(implementation), "sha256": implementation_sha},
    }
    if output.exists():
        if output.is_symlink() or not output.is_dir() or not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs, implementation_sha)
        print(f"multimodal_cell_state_v4_anchor_review={output}")
        print("generation_status=verified_reuse")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        review_path = staging / "anchor_review_set.tsv"
        write_tsv(review_path, review_rows)
        well_counts = Counter(row["source_id"] for row in selected)
        nuclei_status_counts = Counter(row["nuclei_measurement_status"] for row in selected)
        dead_status_counts = Counter(row["dead_measurement_status"] for row in selected)
        cluster_counts = Counter(row["diagnostic_cluster"] for row in selected)
        outputs = directory_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "HUMAN_REVIEW_REQUIRED",
            "method_version": "sum159_multimodal_cell_state_v4",
            "inputs": inputs,
            "implementation_sha256": implementation_sha,
            "row_count": len(review_rows),
            "stable_id_sha256": stable_id_sha(selected_ids),
            "selection": {
                "seed": args.seed,
                "maximum_total": args.max_total,
                "maximum_per_well": args.max_per_well,
                "well_count": len(well_counts),
                "observed_max_per_well": max(well_counts.values()),
                "stratification_audit": "well_by_diagnostic_cluster_by_nuclei_measurement_status_by_dead_measurement_status",
                "within_stratum": "deterministic_umap_farthest_point",
            },
            "diagnostic_cluster_counts": dict(sorted(cluster_counts.items())),
            "nuclei_measurement_status_counts": dict(sorted(nuclei_status_counts.items())),
            "dead_measurement_status_counts": dict(sorted(dead_status_counts.items())),
            "review_default_label": "uncertain_or_unreviewable",
            "review_channels": ["brightfield", "dead", "nuclei"],
            "review_primary_question": "dead_vs_non_dead_regardless_of_death_stage",
            "umap_polygon_training_authority": False,
            "umap_or_cluster_labels_displayed": False,
            "current_classifier_displayed": False,
            "heldout_read": False,
            "forbidden_inputs_read": [],
            "output_file_sha256": outputs,
        }
        manifest_path = staging / "anchor_review_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        identity = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "status": "HUMAN_REVIEW_REQUIRED",
            "manifest_sha256": sha256_file(manifest_path),
            "implementation_sha256": implementation_sha,
            "output_file_sha256": outputs,
        }
        (staging / "anchor_review_generation_identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(
                f"V4 anchor-review output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v4_anchor_review={output}")
    print(f"row_count={len(review_rows)}")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
