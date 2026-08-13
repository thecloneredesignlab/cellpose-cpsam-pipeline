#!/usr/bin/env python3
"""Select and freeze a V4 death-resolution UMAP from independent anchor labels."""

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
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "multimodal_cell_state_v4_death_resolution_v1"
SUBMISSION_SCHEMA = "multimodal_cell_state_v4_review_submission_v1"
PROJECT_ID = "multimodal_cell_state_v4_death_resolution"
PROFILES = {
    "v4_balanced_four_block",
    "v4_death_resolution_dead35_nuclei25",
    "v4_death_resolution_dead30_bf30",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-project", type=Path, required=True)
    parser.add_argument("--projection-root", type=Path, required=True)
    parser.add_argument("--anchor-import-dir", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-per-class", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def directory_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 death-resolution generation contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            output[path.relative_to(root).as_posix()] = sha256(path)
    return output


def balanced_accuracy(truth: list[str], predicted: list[str]) -> float:
    recalls = []
    for label in ("dead", "non_dead"):
        indices = [index for index, value in enumerate(truth) if value == label]
        if not indices:
            raise ValueError(f"Anchor labels lack class: {label}")
        recalls.append(sum(predicted[index] == label for index in indices) / len(indices))
    return float(np.mean(recalls))


def evaluate(
    coordinates: dict[str, tuple[float, float]],
    anchors: list[dict[str, str]],
    cells: dict[str, dict[str, str]],
    neighbors: int,
) -> dict[str, float]:
    ids = [row["cell_id"] for row in anchors]
    matrix = np.asarray([coordinates[cell_id] for cell_id in ids], dtype=float)
    truth = [row["primary_death_label"] for row in anchors]
    wells = [cells[cell_id]["well"] for cell_id in ids]
    predictions: list[str] = []
    for index, point in enumerate(matrix):
        eligible = [j for j in range(len(ids)) if wells[j] != wells[index]]
        if not eligible:
            raise ValueError("Anchor review has no out-of-well neighbors")
        distances = np.sum((matrix[eligible] - point) ** 2, axis=1)
        selected = [eligible[j] for j in np.argsort(distances)[: min(neighbors, len(eligible))]]
        counts = Counter(truth[j] for j in selected)
        predictions.append(max(("dead", "non_dead"), key=lambda label: (counts[label], label)))
    accuracy = balanced_accuracy(truth, predictions)
    dead_indices = [index for index, value in enumerate(truth) if value == "dead"]
    purities = []
    dead_adjacency = {index: set() for index in dead_indices}
    for index in dead_indices:
        distances = np.sum((matrix - matrix[index]) ** 2, axis=1)
        selected = [j for j in np.argsort(distances) if j != index][: min(neighbors, len(ids) - 1)]
        purities.append(np.mean([truth[j] == "dead" for j in selected]))
        for neighbor in selected:
            if truth[neighbor] == "dead":
                dead_adjacency[index].add(neighbor)
                dead_adjacency[neighbor].add(index)
    remaining = set(dead_indices)
    component_sizes: list[int] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            unseen = dead_adjacency[node] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
        component_sizes.append(size)
    connected_fraction = max(component_sizes) / len(dead_indices)
    return {
        "leave_well_out_knn_balanced_accuracy": accuracy,
        "dead_anchor_neighbor_purity": float(np.mean(purities)),
        "dead_anchor_largest_connected_component_fraction": connected_fraction,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shadow = args.shadow_root.resolve(strict=True)
    balanced = args.balanced_project.resolve(strict=True)
    projection = args.projection_root.resolve(strict=True)
    anchor_import = args.anchor_import_dir.resolve(strict=True)
    for path in (balanced, projection, anchor_import):
        if path.is_symlink() or shadow not in path.parents:
            raise ValueError(f"V4 death-resolution input escapes shadow root: {path}")
    project = load_json(balanced)
    if project.get("project_id") != "multimodal_cell_state_v4_development":
        raise ValueError("Balanced V4 project identity differs")
    import_manifest_path = anchor_import / "review_import_manifest.json"
    reviewed_labels_path = anchor_import / "reviewed_labels.tsv"
    import_manifest = load_json(import_manifest_path)
    if (
        import_manifest.get("schema_version")
        != "multimodal_cell_state_v4_review_import_v1"
        or import_manifest.get("status") != "COMPLETE"
        or import_manifest.get("output_file_sha256", {}).get("reviewed_labels.tsv")
        != sha256(reviewed_labels_path)
    ):
        raise ValueError("V4 anchor review import is incomplete")
    _, rows = read_tsv(reviewed_labels_path)
    if not rows:
        raise ValueError("V4 anchor review import has no rows")
    anchors = [
        row for row in rows
        if row.get("primary_death_label") in {"dead", "non_dead"}
        and row.get("label_confidence") in {"high", "medium"}
        and row.get("training_eligible") == "true"
        and row.get("reviewer") and row.get("reviewed_at")
    ]
    counts = Counter(row["primary_death_label"] for row in anchors)
    if min(counts.get("dead", 0), counts.get("non_dead", 0)) < args.minimum_per_class:
        raise ValueError(f"V4 anchor class support is insufficient: {dict(counts)}")
    ids = [row["cell_id"] for row in anchors]
    if len(ids) != len(set(ids)):
        raise ValueError("V4 anchor submission has duplicate cell_id")
    project_dir = balanced.parent
    _, cell_rows = read_tsv(project_dir / project["cells_file"])
    cells = {row["cell_id"]: row for row in cell_rows}
    if any(cell_id not in cells for cell_id in ids):
        raise ValueError("V4 anchor submission includes an unknown cell_id")

    _, metric_rows = read_tsv(projection / "umap_grid_metrics.tsv")
    stability = {(row["profile"], row["run_id"]): float(row["mean_seed_neighbor_jaccard"]) for row in metric_rows}
    candidates: list[dict[str, Any]] = []
    for profile in sorted(PROFILES):
        grid = projection / "profiles" / profile / "grid"
        for path in sorted(grid.glob("*/umap.tsv")):
            _, coordinate_rows = read_tsv(path)
            coordinates = {row["cell_id"]: (float(row["Dim1"]), float(row["Dim2"])) for row in coordinate_rows}
            if len(coordinates) != len(cell_rows) or any(cell_id not in coordinates for cell_id in ids):
                raise ValueError(f"V4 candidate UMAP universe differs: {path}")
            metrics = evaluate(coordinates, anchors, cells, args.neighbors)
            seed_stability = stability[(profile, path.parent.name)]
            score = (
                0.45 * metrics["leave_well_out_knn_balanced_accuracy"]
                + 0.20 * metrics["dead_anchor_neighbor_purity"]
                + 0.20
                * metrics["dead_anchor_largest_connected_component_fraction"]
                + 0.15 * seed_stability
            )
            candidates.append({
                "profile": profile, "run_id": path.parent.name, "coordinate_path": str(path),
                "coordinate_sha256": sha256(path), **metrics,
                "mean_seed_neighbor_jaccard": seed_stability,
                "selection_score": score,
            })
    chosen = max(candidates, key=lambda row: (row["selection_score"], row["profile"], row["run_id"]))
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        manifest = load_json(output / "death_resolution_manifest.json")
        expected_inputs = manifest.get("inputs", {})
        current_inputs = {
            "balanced_project": sha256(balanced), "projection_manifest": sha256(projection / "projection_manifest.json"),
            "anchor_import_manifest": sha256(import_manifest_path),
            "anchor_reviewed_labels": sha256(reviewed_labels_path),
            "implementation": sha256(Path(__file__).resolve()),
        }
        if expected_inputs != current_inputs or manifest.get("status") != "COMPLETE" or manifest.get("output_file_sha256") != directory_hashes(output, {"death_resolution_manifest.json"}):
            raise ValueError("Existing V4 death-resolution generation differs")
        print(f"v4_death_resolution={output} verified_reuse=1")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        death_project = staging
        (death_project / "projection").mkdir(parents=True)
        for name in ("cells.tsv", "features.tsv", "images.tsv", "classes.tsv"):
            source = project_dir / name
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Balanced V4 project asset is unavailable: {source}")
            shutil.copyfile(source, death_project / name)
        chosen_path = Path(chosen["coordinate_path"])
        shutil.copyfile(chosen_path, death_project / "projection" / "umap.tsv")
        balanced_clusters = project_dir / "projection" / "diagnostic_clusters.tsv"
        shutil.copyfile(
            balanced_clusters, death_project / "projection" / "diagnostic_clusters.tsv"
        )
        updated_project = dict(project)
        updated_project["project_id"] = PROJECT_ID
        updated_project["annotation"] = dict(updated_project["annotation"])
        updated_project["annotation"]["title"] = "SUM159 V4 broad death-region annotation"
        (death_project / "project.yml").write_text(
            json.dumps(updated_project, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (death_project / "projection" / "candidate_metrics.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = list(candidates[0])
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(candidates)
        inputs = {
            "balanced_project": sha256(balanced), "projection_manifest": sha256(projection / "projection_manifest.json"),
            "anchor_import_manifest": sha256(import_manifest_path),
            "anchor_reviewed_labels": sha256(reviewed_labels_path),
            "implementation": sha256(Path(__file__).resolve()),
        }
        selected_projection_manifest = {
            "schema_version": "multimodal_cell_state_v4_projection_v1",
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v4_death_resolution",
            "primary_profile": chosen["profile"],
            "selected_run_id": chosen["run_id"],
            "selected_feature_columns": load_json(
                projection / "projection_manifest.json"
            )["selected_feature_columns"],
            "diagnostic_cluster_source": "balanced_four_block_navigation_audit_not_death_label",
            "anchor_review_import_sha256": sha256(import_manifest_path),
            "output_file_sha256": {
                "umap.tsv": sha256(death_project / "projection" / "umap.tsv"),
                "diagnostic_clusters.tsv": sha256(
                    death_project / "projection" / "diagnostic_clusters.tsv"
                ),
            },
        }
        (death_project / "projection" / "projection_manifest.json").write_text(
            json.dumps(selected_projection_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        project_outputs = directory_hashes(death_project)
        project_manifest = {
            "schema_version": "multimodal_cell_state_v4_death_resolution_project_v1",
            "status": "COMPLETE",
            "inputs": inputs,
            "selected": chosen,
            "cell_universe": {
                "row_count": len(cell_rows),
                "heldout_read": False,
            },
            "polygon_role": "provisional_broad_dead_region_navigation_and_sampling",
            "training_label_source": "independent_three_channel_reviews_only",
            "output_file_sha256": project_outputs,
        }
        (death_project / "parent_import_manifest.json").write_text(
            json.dumps(project_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = directory_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "COMPLETE", "inputs": inputs,
            "anchor_reviewed_rows": len(rows), "eligible_anchor_rows": len(anchors),
            "anchor_class_counts": dict(counts), "selected": chosen,
            "selection_rule": "0.45_leave_well_out_balanced_accuracy_plus_0.20_dead_neighbor_purity_plus_0.20_dead_largest_connected_component_fraction_plus_0.15_seed_stability",
            "polygon_role": "provisional_broad_dead_region_not_training_truth",
            "output_file_sha256": artifacts,
        }
        (staging / "death_resolution_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    print(f"v4_death_resolution={output}")
    print(f"selected_profile={chosen['profile']} selected_run_id={chosen['run_id']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
