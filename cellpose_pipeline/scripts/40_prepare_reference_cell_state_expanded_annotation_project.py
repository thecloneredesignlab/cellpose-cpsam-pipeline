#!/usr/bin/env python3
"""Build an immutable expanded-evidence annotation project from frozen V2 inputs.

The base V2 project supplies the exact 32,000 development cells, classifier12
features, Brightfield/Nuclei image registry, and historical three-class
ontology.  Script 39 supplies expanded39 coordinates, diagnostic clusters, and
the reference-repo representative selection.  This builder changes annotation
geometry only; it does not permit expanded39 features to enter model fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "reference_cell_state_expanded_annotation_parent_import_v1"
IDENTITY_SCHEMA_VERSION = (
    "reference_cell_state_expanded_annotation_project_generation_identity_v1"
)
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
PROJECT_ID = "reference_cell_state_development_v2"
EXPANDED_MANIFEST_SCHEMA = "reference_cell_state_expanded_projection_comparison_v1"
HISTORICAL_MANIFEST_SCHEMA = "reference_cell_state_historical_projection_expanded_v1"
SHAPE9 = (
    "area_px2",
    "perimeter_px",
    "roundness",
    "aspect_ratio",
    "extent",
    "solidity",
    "equivalent_diameter_px",
    "major_axis_px",
    "minor_axis_px",
)
CLASSIFIER12 = (
    *SHAPE9,
    "bf_boundary_mean",
    "bf_interior_mean",
    "bf_interior_minus_boundary_mean",
)
EXPECTED_CLASSES = ("live_cell", "dead_cell", "multinucleated_cell")
EXPECTED_CHANNELS = ("brightfield", "nuclei")
DECLARED_PROJECT_FILES = (
    "project.yml",
    "cells.tsv",
    "features.tsv",
    "images.tsv",
    "classes.tsv",
    "historical_projection/umap.tsv",
    "historical_projection/diagnostic_clusters.tsv",
    "historical_projection/historical_representatives.tsv",
    "historical_projection/expanded_projection_manifest.json",
    "historical_projection/labelability_decision.json",
    "historical_projection/historical_projection_manifest.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--base-shadow-root", type=Path, required=True)
    parser.add_argument("--expanded-projection", type=Path, required=True)
    parser.add_argument("--reference-shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        return fields, [dict(row) for row in reader]


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def require_plain(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a plain file: {path}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a real directory: {path}")
    return resolved


def require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} escapes its frozen root: {resolved}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    return resolved


def resolve_project_asset(project_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Base project lacks {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    return path


def declared_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in DECLARED_PROJECT_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"Expanded annotation project artifact is unavailable: {path}"
            )
        result[relative] = sha256_file(path)
    return result


def verify_base_project(
    project_path: Path,
    base_root: Path,
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, str]]]:
    project = load_json(project_path, "Base V2 project")
    if (
        project.get("schema_version") != PROJECT_SCHEMA_VERSION
        or project.get("project_id") != PROJECT_ID
    ):
        raise ValueError(
            "Expanded annotation requires the frozen V2 development project"
        )
    assets: dict[str, Path] = {}
    for field in ("cells_file", "features_file", "images_file", "classes_file"):
        assets[field] = require_inside(
            require_plain(
                resolve_project_asset(project_path, project.get(field), field), field
            ),
            base_root,
            field,
        )
    cell_fields, cells = read_tsv(assets["cells_file"])
    required_cells = {"cell_id", "context_key", "source_id", "well", "split", "cluster"}
    if not required_cells.issubset(cell_fields):
        raise ValueError("Base V2 cells.tsv lacks expanded annotation identity")
    cell_ids = [row["cell_id"] for row in cells]
    if (
        len(cells) != expected_count
        or len(set(cell_ids)) != expected_count
        or any(not cell_id for cell_id in cell_ids)
        or any(row["split"] != "development" for row in cells)
    ):
        raise ValueError("Base V2 development-cell universe differs")
    feature_fields, feature_rows = read_tsv(assets["features_file"])
    if (
        feature_fields != ["cell_id", *CLASSIFIER12]
        or [row["cell_id"] for row in feature_rows] != cell_ids
    ):
        raise ValueError("Base V2 classifier12 feature table differs")
    image_fields, image_rows = read_tsv(assets["images_file"])
    if not {"image_id", "channel_id"}.issubset(image_fields):
        raise ValueError("Base V2 image registry lacks identity fields")
    channels = {row["channel_id"] for row in image_rows}
    if channels != set(EXPECTED_CHANNELS):
        raise ValueError(
            f"Expanded annotation image registry is not blind: {sorted(channels)}"
        )
    class_fields, class_rows = read_tsv(assets["classes_file"])
    if (
        "class_id" not in class_fields
        or tuple(row["class_id"] for row in class_rows) != EXPECTED_CLASSES
    ):
        raise ValueError("Base V2 class ontology/order differs")
    return project, assets, cells


def verify_expanded_projection(
    expanded_root: Path,
    expected_cell_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], list[dict[str, str]]]:
    required = {
        "manifest": expanded_root / "expanded_projection_manifest.json",
        "decision": expanded_root / "labelability_decision.json",
        "umap": expanded_root / "umap.tsv",
        "clusters": expanded_root / "diagnostic_clusters.tsv",
        "representatives": expanded_root / "historical_representatives.tsv",
    }
    required = {name: require_plain(path, name) for name, path in required.items()}
    manifest = load_json(required["manifest"], "Expanded projection manifest")
    decision = load_json(required["decision"], "Expanded labelability decision")
    if (
        manifest.get("schema_version") != EXPANDED_MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("selected_annotation_profile") != "expanded39"
    ):
        raise ValueError(
            "Expanded projection manifest is not a complete expanded39 generation"
        )
    declared = manifest.get("output_file_sha256")
    if not isinstance(declared, dict):
        raise ValueError("Expanded projection manifest lacks artifact hashes")
    for name, path in required.items():
        if name == "manifest":
            continue
        relative = path.relative_to(expanded_root).as_posix()
        if declared.get(relative) != sha256_file(path):
            raise ValueError(f"Expanded projection artifact hash differs: {relative}")
    if (
        decision.get("schema_version")
        != "reference_cell_state_expanded_labelability_decision_v1"
        or decision.get("status") != "COMPLETE"
        or decision.get("selected_annotation_profile") != "expanded39"
    ):
        raise ValueError("Expanded labelability decision is incomplete")
    umap_fields, umap_rows = read_tsv(required["umap"])
    cluster_fields, cluster_rows = read_tsv(required["clusters"])
    if umap_fields != ["cell_id", "Dim1", "Dim2"]:
        raise ValueError("Expanded UMAP schema differs")
    if cluster_fields != ["cell_id", "cluster", "cluster_source"]:
        raise ValueError("Expanded diagnostic cluster schema differs")
    if [row["cell_id"] for row in umap_rows] != expected_cell_ids or [
        row["cell_id"] for row in cluster_rows
    ] != expected_cell_ids:
        raise ValueError("Expanded coordinate/cluster universe differs from base cells")
    return manifest, decision, required, cluster_rows


def validate_reuse(
    output: Path,
    expected_inputs: dict[str, Any],
    implementation_sha256: str,
) -> None:
    manifest_path = output / "parent_import_manifest.json"
    identity_path = output / "expanded_annotation_project_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing expanded annotation project is partial")
    manifest = load_json(manifest_path, "Expanded annotation parent import")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("inputs") != expected_inputs
        or manifest.get("builder_implementation_sha256") != implementation_sha256
    ):
        raise ValueError("Existing expanded annotation parent identity differs")
    observed = declared_hashes(output)
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing expanded annotation project artifact hashes differ")
    identity = load_json(identity_path, "Expanded annotation project identity")
    expected_identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "parent_import_manifest_sha256": sha256_file(manifest_path),
        "builder_implementation_sha256": implementation_sha256,
        "output_file_sha256": observed,
    }
    if identity != expected_identity:
        raise ValueError(
            "Existing expanded annotation project generation identity differs"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_cell_count < 1:
        raise ValueError("--expected-cell-count must be positive")
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = sha256_file(implementation_path)
    base_root = require_directory(args.base_shadow_root, "Base V2 shadow root")
    reference_root = require_directory(
        args.reference_shadow_root, "Expanded reference shadow root"
    )
    if (
        base_root == reference_root
        or base_root in reference_root.parents
        or reference_root in base_root.parents
    ):
        raise ValueError(
            "Base and expanded reference roots must be independent siblings"
        )
    project_path = require_inside(
        require_plain(args.base_project, "Base V2 project"),
        base_root,
        "Base V2 project",
    )
    expanded_root = require_inside(
        require_directory(args.expanded_projection, "Expanded projection"),
        reference_root,
        "Expanded projection",
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else reference_root / "projection_input" / "representative_umap_v2"
    )
    if output == reference_root or reference_root not in output.parents:
        raise ValueError(
            "Expanded annotation project must remain inside its new shadow root"
        )
    if (
        expanded_root == output
        or expanded_root in output.parents
        or output in expanded_root.parents
    ):
        raise ValueError(
            "Expanded projection evidence and annotation project must not overlap"
        )

    base_project, base_assets, base_cells = verify_base_project(
        project_path, base_root, args.expected_cell_count
    )
    cell_ids = [row["cell_id"] for row in base_cells]
    expanded_manifest, decision, expanded_files, cluster_rows = (
        verify_expanded_projection(expanded_root, cell_ids)
    )
    expected_inputs = {
        "base_project": {
            "path": str(project_path),
            "sha256": sha256_file(project_path),
        },
        "base_cells": {
            "path": str(base_assets["cells_file"]),
            "sha256": sha256_file(base_assets["cells_file"]),
        },
        "base_features": {
            "path": str(base_assets["features_file"]),
            "sha256": sha256_file(base_assets["features_file"]),
        },
        "base_images": {
            "path": str(base_assets["images_file"]),
            "sha256": sha256_file(base_assets["images_file"]),
        },
        "base_classes": {
            "path": str(base_assets["classes_file"]),
            "sha256": sha256_file(base_assets["classes_file"]),
        },
        "expanded_projection_manifest": {
            "path": str(expanded_files["manifest"]),
            "sha256": sha256_file(expanded_files["manifest"]),
        },
        "labelability_decision": {
            "path": str(expanded_files["decision"]),
            "sha256": sha256_file(expanded_files["decision"]),
        },
    }
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        if output.is_symlink() or not output.is_dir():
            raise ValueError(
                "Existing expanded annotation project must be a real directory"
            )
        validate_reuse(output, expected_inputs, implementation_sha256)
        print(f"expanded_annotation_project={output / 'project.yml'}")
        print("generation_status=verified_reuse")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        historical = staging / "historical_projection"
        historical.mkdir()
        (staging / "runs").mkdir()
        shutil.copyfile(base_assets["features_file"], staging / "features.tsv")
        shutil.copyfile(base_assets["images_file"], staging / "images.tsv")
        shutil.copyfile(base_assets["classes_file"], staging / "classes.tsv")
        shutil.copyfile(expanded_files["umap"], historical / "umap.tsv")
        shutil.copyfile(
            expanded_files["clusters"], historical / "diagnostic_clusters.tsv"
        )
        shutil.copyfile(
            expanded_files["representatives"],
            historical / "historical_representatives.tsv",
        )
        shutil.copyfile(
            expanded_files["manifest"], historical / "expanded_projection_manifest.json"
        )
        shutil.copyfile(
            expanded_files["decision"], historical / "labelability_decision.json"
        )

        clusters = {row["cell_id"]: row["cluster"] for row in cluster_rows}
        cell_fields, _ = read_tsv(base_assets["cells_file"])
        updated_cells = [
            {**row, "cluster": clusters[row["cell_id"]]} for row in base_cells
        ]
        write_tsv(staging / "cells.tsv", cell_fields, updated_cells)

        project = dict(base_project)
        project["cells_file"] = "cells.tsv"
        project["features_file"] = "features.tsv"
        project["images_file"] = "images.tsv"
        project["classes_file"] = "classes.tsv"
        project["runs_dir"] = "runs"
        project["projection"] = {
            "mode": "existing_umap",
            "coordinate_file": "historical_projection/umap.tsv",
            "allow_subset": False,
        }
        annotation = dict(project.get("annotation") or {})
        annotation["title"] = "Reference cell-state annotation: expanded39 evidence"
        project["annotation"] = annotation
        classifier = project.get("classifier")
        if not isinstance(classifier, dict) or classifier.get(
            "feature_columns"
        ) != list(CLASSIFIER12):
            raise ValueError(
                "Expanded project attempted to change the historical classifier12"
            )
        write_json(staging / "project.yml", project)

        historical_output_hashes = {
            name: sha256_file(historical / name)
            for name in (
                "umap.tsv",
                "diagnostic_clusters.tsv",
                "historical_representatives.tsv",
                "expanded_projection_manifest.json",
                "labelability_decision.json",
            )
        }
        representative_fields, representative_rows = read_tsv(
            historical / "historical_representatives.tsv"
        )
        if representative_fields != [
            "selection_rank",
            "cell_id",
            "context_key",
            "source_id",
            "cluster",
            "Dim1",
            "Dim2",
        ]:
            raise ValueError("Expanded historical representative schema differs")
        historical_manifest = {
            "schema_version": HISTORICAL_MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "selected_annotation_profile": "expanded39",
            "expanded_feature_role": "annotation_geometry_and_human_morphology_evidence_only",
            "expanded_features_allowed_in_final_classifier": False,
            "expanded_projection_manifest_file": "expanded_projection_manifest.json",
            "expanded_projection_manifest_sha256": historical_output_hashes[
                "expanded_projection_manifest.json"
            ],
            "labelability_decision_file": "labelability_decision.json",
            "labelability_decision_sha256": historical_output_hashes[
                "labelability_decision.json"
            ],
            "computational_labelability_gate": decision["computational_gate"],
            "morphology_overlay_gate": decision["morphology_overlay_gate"],
            "diagnostic_cluster": expanded_manifest["profiles"]["expanded39"],
            "representative_selection": {
                "role": "authoritative_rendering_cell_list",
                "outer_function_name": "get_all_cell_lines_overlay_representatives",
                "inner_function_name": "get_spatially_uniform_representatives",
                "seed": 1,
                "total_n": 300,
                "balance": 0.2,
                "minimum_cluster_representatives": 2,
                "selected_count": len(representative_rows),
                "output_file": "historical_representatives.tsv",
                "output_sha256": historical_output_hashes[
                    "historical_representatives.tsv"
                ],
            },
            "classifier_boundary": {
                "feature_columns": list(CLASSIFIER12),
                "expanded39_allowed": False,
            },
            "blinding_contract": {
                "dead": "not_read",
                "combined_rgb": "not_read",
                "current_classification": "not_read",
                "trajectory": "not_read",
                "heldout": "not_read",
            },
            "output_file_sha256": historical_output_hashes,
        }
        write_json(
            historical / "historical_projection_manifest.json", historical_manifest
        )

        output_hashes = declared_hashes(staging)
        parent_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "expanded39_annotation_geometry_classifier12_model",
            "inputs": expected_inputs,
            "builder_implementation_sha256": implementation_sha256,
            "cell_universe": {
                "row_count": len(cell_ids),
                "ordered_cell_id_sha256": sha256_text_parts(cell_ids),
                "split": "development",
                "heldout_read": False,
            },
            "projection": {
                "selected_profile": "expanded39",
                "expanded_projection_manifest_sha256": sha256_file(
                    historical / "expanded_projection_manifest.json"
                ),
                "computational_labelability_gate": decision["computational_gate"],
                "morphology_overlay_gate": decision["morphology_overlay_gate"],
            },
            "classifier": {
                "feature_columns": list(CLASSIFIER12),
                "expanded_features_allowed": False,
            },
            "output_file_sha256": output_hashes,
        }
        manifest_path = staging / "parent_import_manifest.json"
        write_json(manifest_path, parent_manifest)
        identity = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "status": "COMPLETE",
            "parent_import_manifest_sha256": sha256_file(manifest_path),
            "builder_implementation_sha256": implementation_sha256,
            "output_file_sha256": output_hashes,
        }
        write_json(
            staging / "expanded_annotation_project_generation_identity.json", identity
        )
        if sha256_file(implementation_path) != implementation_sha256:
            raise RuntimeError(
                "Expanded annotation project builder changed during execution"
            )
        if output.exists():
            raise FileExistsError(
                f"Expanded annotation output appeared during staging: {output}"
            )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"expanded_annotation_project={output / 'project.yml'}")
    print(f"computational_labelability_gate={decision['computational_gate']}")
    print("expanded_features_in_classifier=0")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
