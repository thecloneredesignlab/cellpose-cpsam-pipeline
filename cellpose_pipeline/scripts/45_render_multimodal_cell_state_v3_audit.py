#!/usr/bin/env python3
"""Render a cluster-colored, image-linked audit atlas for V3 UMAP geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RENDER_HELPER_PATH = SCRIPT_DIR / "27_build_reference_morphology_workspace.py"
SPEC = importlib.util.spec_from_file_location(
    "v3_morphology_render_helper", RENDER_HELPER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load morphology rendering helpers: {RENDER_HELPER_PATH}")
RENDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDER
SPEC.loader.exec_module(RENDER)

SCHEMA_VERSION = "multimodal_cell_state_v3_morphology_audit_v1"
PROJECT_ID = "multimodal_cell_state_v3_development"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-representatives", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--mask-padding", type=int, default=6)
    parser.add_argument("--overlay-width", type=int, default=2400)
    parser.add_argument("--overlay-height", type=int, default=1800)
    parser.add_argument("--overlay-tile-px", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_asset(project_path: Path, project: dict[str, Any], field: str) -> Path:
    text = str(project.get(field, ""))
    if not text:
        raise ValueError(f"V3 project lacks {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"V3 project asset is unavailable or a symlink: {path}")
    return path


def deterministic_farthest(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count >= len(rows):
        return sorted(rows, key=lambda row: row["cell_id"])
    ordered = sorted(rows, key=lambda row: row["cell_id"])
    coordinates = np.asarray(
        [[float(row["Dim1"]), float(row["Dim2"])] for row in ordered]
    )
    digest_values = [
        hashlib.sha256(f"{seed}|{row['cell_id']}".encode()).hexdigest()
        for row in ordered
    ]
    first = min(range(len(ordered)), key=lambda index: digest_values[index])
    selected = [first]
    distances = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    distances[first] = -1
    while len(selected) < count:
        maximum = float(np.max(distances))
        candidates = np.flatnonzero(np.isclose(distances, maximum, rtol=0, atol=1e-15))
        chosen = min(candidates, key=lambda index: digest_values[int(index)])
        selected.append(int(chosen))
        candidate_distance = np.sum((coordinates - coordinates[chosen]) ** 2, axis=1)
        distances = np.minimum(distances, candidate_distance)
        distances[selected] = -1
    return [ordered[index] for index in selected]


def allocate_quotas(
    groups: dict[tuple[str, str], list[dict[str, Any]]], total: int
) -> dict[tuple[str, str], int]:
    keys = sorted(groups)
    if total < len(keys):
        raise ValueError(
            "Representative budget is smaller than context-cluster coverage"
        )
    total_rows = sum(len(groups[key]) for key in keys)
    quotas = {key: 1 for key in keys}
    remaining = total - len(keys)
    ideals = {key: remaining * len(groups[key]) / total_rows for key in keys}
    for key in keys:
        add = min(len(groups[key]) - 1, int(math.floor(ideals[key])))
        quotas[key] += add
    while sum(quotas.values()) < total:
        eligible = [key for key in keys if quotas[key] < len(groups[key])]
        if not eligible:
            break
        key = max(
            eligible,
            key=lambda item: (
                ideals[item] - math.floor(ideals[item]),
                len(groups[item]) - quotas[item],
                tuple(reversed(item)),
            ),
        )
        quotas[key] += 1
        ideals[key] = math.floor(ideals[key])
    return quotas


def directory_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V3 audit contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                result[relative] = sha256_file(path)
    return result


def verify_existing(
    output: Path,
    expected_inputs: dict[str, Any],
    implementation_sha: str,
    helper_sha: str,
) -> None:
    manifest_path = output / "morphology_audit_manifest.json"
    identity_path = output / "morphology_audit_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing V3 morphology audit is partial")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("inputs") != expected_inputs
    ):
        raise ValueError("Existing V3 morphology audit identity differs")
    observed = directory_hashes(
        output,
        {"morphology_audit_manifest.json", "morphology_audit_generation_identity.json"},
    )
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V3 morphology audit artifact set/hash differs")
    expected = {
        "schema_version": "multimodal_cell_state_v3_morphology_audit_generation_identity_v1",
        "status": "COMPLETE",
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": implementation_sha,
        "render_helper_sha256": helper_sha,
        "output_file_sha256": observed,
    }
    if load_json(identity_path) != expected:
        raise ValueError("Existing V3 morphology audit generation identity differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_representatives < 1 or args.mask_padding < 0:
        raise ValueError("Invalid V3 morphology audit representative/padding option")
    if min(args.overlay_width, args.overlay_height) < 400 or args.overlay_tile_px < 4:
        raise ValueError("Invalid V3 morphology audit canvas/tile size")
    implementation = Path(__file__).resolve()
    implementation_sha = sha256_file(implementation)
    helper_sha = sha256_file(RENDER_HELPER_PATH.resolve())
    shadow = args.shadow_root.expanduser().resolve()
    if not shadow.is_dir() or shadow.is_symlink():
        raise ValueError(f"V3 shadow root is unavailable: {shadow}")
    project_path = args.project.expanduser().resolve()
    if (
        not project_path.is_file()
        or project_path.is_symlink()
        or shadow not in project_path.parents
    ):
        raise ValueError("V3 project must be a plain file inside its shadow root")
    output = args.output_dir.expanduser().resolve()
    if (
        output == shadow
        or shadow not in output.parents
        or output in project_path.parents
    ):
        raise ValueError(
            "V3 morphology audit output must be separate inside its shadow root"
        )
    project = load_json(project_path)
    if (
        project.get("schema_version") != "cell_phenotype_annotator_project_v1"
        or project.get("project_id") != PROJECT_ID
    ):
        raise ValueError("Unsupported V3 project identity")
    cells_path = resolve_asset(project_path, project, "cells_file")
    images_path = resolve_asset(project_path, project, "images_file")
    projection = project.get("projection")
    if not isinstance(projection, dict) or projection.get("mode") != "existing_umap":
        raise ValueError("V3 project does not declare an existing UMAP")
    coordinate_path = project_path.parent / str(projection.get("coordinate_file", ""))
    coordinate_path = coordinate_path.resolve()
    if not coordinate_path.is_file() or coordinate_path.is_symlink():
        raise ValueError("V3 coordinate file is unavailable")
    clusters_path = project_path.parent / "projection" / "diagnostic_clusters.tsv"
    manifest_path = project_path.parent / "projection" / "projection_manifest.json"
    for path in (clusters_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"V3 projection audit input is unavailable: {path}")
    projection_manifest = load_json(manifest_path)
    if (
        projection_manifest.get("schema_version")
        != "multimodal_cell_state_v3_projection_v1"
        or projection_manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("V3 projection manifest is incomplete")
    declared = projection_manifest.get("output_file_sha256", {})
    if declared.get("umap.tsv") != sha256_file(coordinate_path) or declared.get(
        "diagnostic_clusters.tsv"
    ) != sha256_file(clusters_path):
        raise ValueError("V3 project projection hashes differ from authority")
    cell_fields, cells = read_tsv(cells_path)
    _, coordinates = read_tsv(coordinate_path)
    _, clusters = read_tsv(clusters_path)
    required_cells = {
        "cell_id",
        "image_id",
        "mask_label",
        "well",
        "context_key",
        "split",
    }
    if not required_cells.issubset(cell_fields):
        raise ValueError("V3 cells lack morphology audit identity")
    cell_ids = [row["cell_id"] for row in cells]
    if [row["cell_id"] for row in coordinates] != cell_ids or [
        row["cell_id"] for row in clusters
    ] != cell_ids:
        raise ValueError("V3 morphology audit row universe differs")
    if any(row["split"] != "development" for row in cells):
        raise ValueError("Heldout cells entered V3 morphology audit")
    joined: list[dict[str, Any]] = []
    for cell, coordinate, cluster in zip(cells, coordinates, clusters, strict=True):
        joined.append(
            {
                **cell,
                "Dim1": float(coordinate["Dim1"]),
                "Dim2": float(coordinate["Dim2"]),
                "cluster": cluster["diagnostic_cluster"],
            }
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(row["context_key"], row["cluster"])].append(row)
    quotas = allocate_quotas(groups, min(args.max_representatives, len(joined)))
    selected: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(groups)):
        for row in deterministic_farthest(
            groups[key], quotas[key], args.seed + group_index
        ):
            selected.append(
                {
                    **row,
                    "_selection_round": "context_cluster_farthest_point",
                    "_context_column": "context_key",
                    "_context_value": row["context_key"],
                }
            )
    selected.sort(
        key=lambda row: (row["context_key"], int(row["cluster"]), row["cell_id"])
    )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    image_rows = RENDER.load_images(images_path)
    inputs = {
        "project": {"path": str(project_path), "sha256": sha256_file(project_path)},
        "cells": {"path": str(cells_path), "sha256": sha256_file(cells_path)},
        "images": {"path": str(images_path), "sha256": sha256_file(images_path)},
        "coordinates": {
            "path": str(coordinate_path),
            "sha256": sha256_file(coordinate_path),
        },
        "diagnostic_clusters": {
            "path": str(clusters_path),
            "sha256": sha256_file(clusters_path),
        },
        "projection_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "implementation": {"path": str(implementation), "sha256": implementation_sha},
        "render_helper": {
            "path": str(RENDER_HELPER_PATH.resolve()),
            "sha256": helper_sha,
        },
    }
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError("Existing V3 morphology audit must be a real directory")
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs, implementation_sha, helper_sha)
        print(f"multimodal_cell_state_v3_morphology_audit={output}")
        print("generation_status=verified_reuse")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        rendered, source_assets = RENDER.render_crops(
            selected, image_rows, staging, args.mask_padding, include_cluster=True
        )
        representative_fields = RENDER.REPRESENTATIVE_FIELDS_V2
        write_tsv(staging / "representative_cells.tsv", representative_fields, rendered)
        bounds = RENDER.render_overlay(
            joined,
            rendered,
            staging,
            args.overlay_width,
            args.overlay_height,
            args.overlay_tile_px,
            cluster_coloring=True,
        )
        identity = {
            "project_id": PROJECT_ID,
            "selected_run_id": projection_manifest["selected_run_id"],
            "row_count": len(joined),
            "representative_count": len(rendered),
            "diagnostic_cluster_role": "navigation_only_not_cell_state_label",
        }
        (staging / "morphology_atlas.html").write_text(
            RENDER.atlas_html(rendered, identity, include_cluster=True),
            encoding="utf-8",
        )
        outputs = directory_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v3",
            "inputs": inputs,
            "selection": {
                "seed": args.seed,
                "maximum_representatives": args.max_representatives,
                "selected_count": len(rendered),
                "strata": "context_key_by_diagnostic_cluster",
                "within_stratum": "deterministic_umap_farthest_point_real_cells",
                "diagnostic_cluster_role": "navigation_only_not_cell_state_label",
            },
            "render": {
                "channels": ["brightfield", "nuclei_support"],
                "combined_mask_role": "localization_and_outline_only",
                "overlay_bounds": bounds,
                "source_assets": source_assets,
            },
            "heldout_read": False,
            "forbidden_inputs_read": [],
            "training_label_authority": "none_visual_audit_only",
            "output_file_sha256": outputs,
        }
        manifest_file = staging / "morphology_audit_manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation = {
            "schema_version": "multimodal_cell_state_v3_morphology_audit_generation_identity_v1",
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(manifest_file),
            "implementation_sha256": implementation_sha,
            "render_helper_sha256": helper_sha,
            "output_file_sha256": outputs,
        }
        (staging / "morphology_audit_generation_identity.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(
                f"V3 morphology audit output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v3_morphology_audit={output}")
    print(f"representatives={len(selected)}")
    print("training_label_authority=none")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
