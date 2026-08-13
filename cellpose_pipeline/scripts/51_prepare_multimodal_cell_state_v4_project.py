#!/usr/bin/env python3
"""Create an immutable CPA exploration project for the selected V4 geometry.

The initial CPA UMAP is the balanced four-block view.  Its polygons are
provisional navigation annotations; independent three-channel anchor labels
select the later death-resolution view, and reviewed cells remain the only
training authority.
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


SCHEMA_VERSION = "multimodal_cell_state_v4_project_import_v1"
IDENTITY_SCHEMA_VERSION = "multimodal_cell_state_v4_project_generation_identity_v1"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
PROJECT_ID = "multimodal_cell_state_v4_development"
PROJECTION_SCHEMA = "multimodal_cell_state_v4_projection_v1"
MERGE_SCHEMA = "multimodal_cell_state_v4_feature_merge_v1"
EXPECTED_CLASSES = ("non_dead", "dead")
BASE_CHANNELS = ("brightfield", "nuclei")
EXPECTED_CHANNELS = ("brightfield", "dead", "nuclei")
DECLARED_FILES = (
    "project.yml",
    "cells.tsv",
    "features.tsv",
    "images.tsv",
    "classes.tsv",
    "projection/umap.tsv",
    "projection/diagnostic_clusters.tsv",
    "projection/projection_manifest.json",
    "projection/calibration_decision.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--base-shadow-root", type=Path, required=True)
    parser.add_argument("--feature-merge", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    value = hashlib.sha256()
    for item in values:
        value.update(item.encode())
        value.update(b"\n")
    return value.hexdigest()


def require_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a plain file: {path}")
    return value


def require_dir(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a real directory: {path}")
    return value


def require_inside(path: Path, root: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if value == root or root not in value.parents or path.is_symlink():
        raise ValueError(f"{label} escapes its frozen root: {value}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def resolve_asset(project_path: Path, value: Any, label: str) -> Path:
    text = str(value or "")
    if not text:
        raise ValueError(f"Base project lacks {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    return require_file(path, label)


def artifact_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 project contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                result[relative] = sha256_file(path)
    return result


def verify_existing(
    output: Path, expected_inputs: dict[str, Any], implementation_sha: str
) -> None:
    manifest_path = output / "parent_import_manifest.json"
    identity_path = output / "project_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing V4 project is partial")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("inputs") != expected_inputs
        or manifest.get("implementation_sha256") != implementation_sha
    ):
        raise ValueError("Existing V4 project identity differs")
    observed = artifact_hashes(
        output, {"parent_import_manifest.json", "project_generation_identity.json"}
    )
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 project artifact set/hash differs")
    expected = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "parent_import_manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": implementation_sha,
        "output_file_sha256": observed,
    }
    if load_json(identity_path) != expected:
        raise ValueError("Existing V4 project generation identity differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_cell_count < 1:
        raise ValueError("--expected-cell-count must be positive")
    implementation = require_file(Path(__file__), "V4 project implementation")
    implementation_sha = sha256_file(implementation)
    base_root = require_dir(args.base_shadow_root, "Base shadow root")
    shadow_root = require_dir(args.shadow_root, "V4 shadow root")
    if (
        base_root == shadow_root
        or base_root in shadow_root.parents
        or shadow_root in base_root.parents
    ):
        raise ValueError("Base and V4 roots must be independent siblings")
    base_project_path = require_inside(
        require_file(args.base_project, "Base project"), base_root, "Base project"
    )
    feature_root = require_inside(
        require_dir(args.feature_merge, "V4 feature merge"),
        shadow_root,
        "V4 feature merge",
    )
    projection_root = require_inside(
        require_dir(args.projection, "V4 projection"), shadow_root, "V4 projection"
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else shadow_root / "projection_input" / "multimodal_cell_state_v4"
    )
    if output == shadow_root or shadow_root not in output.parents:
        raise ValueError("V4 project output must remain inside its shadow root")
    if any(
        root == output or root in output.parents or output in root.parents
        for root in (feature_root, projection_root)
    ):
        raise ValueError(
            "V4 project output must not overlap feature/projection evidence"
        )

    base_project = load_json(base_project_path)
    if base_project.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError("Base CPA project schema differs")
    base_assets = {
        field: require_inside(
            resolve_asset(base_project_path, base_project.get(field), field),
            base_root,
            field,
        )
        for field in ("cells_file", "images_file", "classes_file")
    }
    cell_fields, base_cells = read_tsv(base_assets["cells_file"])
    required_cells = {
        "cell_id",
        "image_id",
        "mask_label",
        "key",
        "well",
        "site",
        "elapsed_hours",
        "context_key",
        "source_id",
        "split",
    }
    if not required_cells.issubset(cell_fields):
        raise ValueError(
            f"Base cells lack V4 identity: {sorted(required_cells-set(cell_fields))}"
        )
    cell_ids = [row["cell_id"] for row in base_cells]
    if (
        len(base_cells) != args.expected_cell_count
        or len(set(cell_ids)) != len(cell_ids)
        or any(row["split"] != "development" for row in base_cells)
    ):
        raise ValueError("Base V4 development-cell universe differs")
    image_fields, images = read_tsv(base_assets["images_file"])
    if not {"image_id", "channel_id"}.issubset(image_fields):
        raise ValueError("Base image registry lacks image/channel identity")
    if {row["channel_id"] for row in images} != set(BASE_CHANNELS):
        raise ValueError("Base image registry is not the blind Brightfield/Nuclei pair")

    merge_manifest_path = require_file(
        feature_root / "feature_merge_manifest.json", "V4 feature merge manifest"
    )
    merge_features_path = require_file(
        feature_root / "features.tsv", "V4 merged features"
    )
    merge_manifest = load_json(merge_manifest_path)
    if (
        merge_manifest.get("schema_version") != MERGE_SCHEMA
        or merge_manifest.get("status") != "COMPLETE"
        or merge_manifest.get("heldout_read") is not False
    ):
        raise ValueError("V4 feature merge manifest is incomplete")
    if merge_manifest.get("output_file_sha256", {}).get("features.tsv") != sha256_file(
        merge_features_path
    ):
        raise ValueError("V4 merged feature hash differs")
    feature_fields, feature_rows = read_tsv(merge_features_path)
    if [row["cell_id"] for row in feature_rows] != cell_ids:
        raise ValueError("V4 merged features are not in base cell lockstep")

    projection_manifest_path = require_file(
        projection_root / "projection_manifest.json", "V4 projection manifest"
    )
    decision_path = require_file(
        projection_root / "calibration_decision.json", "V4 calibration decision"
    )
    umap_path = require_file(projection_root / "umap.tsv", "V4 selected UMAP")
    clusters_path = require_file(
        projection_root / "diagnostic_clusters.tsv", "V4 diagnostic clusters"
    )
    projection_manifest = load_json(projection_manifest_path)
    decision = load_json(decision_path)
    if (
        projection_manifest.get("schema_version") != PROJECTION_SCHEMA
        or projection_manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("V4 projection manifest is incomplete")
    if (
        decision.get("representation_gate") != "GO_TO_INDEPENDENT_ANCHOR_REVIEW"
        or decision.get("balanced_dead_cell_island_required") is not False
    ):
        raise ValueError(
            "V4 representation has not passed its blind label-mapping gate"
        )
    declared_projection = projection_manifest.get("output_file_sha256", {})
    for path in (decision_path, umap_path, clusters_path):
        relative = path.relative_to(projection_root).as_posix()
        if declared_projection.get(relative) != sha256_file(path):
            raise ValueError(f"V4 projection artifact hash differs: {relative}")
    _, umap_rows = read_tsv(umap_path)
    _, cluster_rows = read_tsv(clusters_path)
    if [row["cell_id"] for row in umap_rows] != cell_ids or [
        row["cell_id"] for row in cluster_rows
    ] != cell_ids:
        raise ValueError("V4 UMAP/cluster rows differ from the cell universe")
    selected = projection_manifest.get("selected_feature_columns")
    if (
        not isinstance(selected, list)
        or not selected
        or any(name not in feature_fields for name in selected)
    ):
        raise ValueError("V4 projection selected feature columns are invalid")

    inputs = {
        "base_project": {
            "path": str(base_project_path),
            "sha256": sha256_file(base_project_path),
        },
        "base_cells": {
            "path": str(base_assets["cells_file"]),
            "sha256": sha256_file(base_assets["cells_file"]),
        },
        "base_images": {
            "path": str(base_assets["images_file"]),
            "sha256": sha256_file(base_assets["images_file"]),
        },
        "feature_merge_manifest": {
            "path": str(merge_manifest_path),
            "sha256": sha256_file(merge_manifest_path),
        },
        "projection_manifest": {
            "path": str(projection_manifest_path),
            "sha256": sha256_file(projection_manifest_path),
        },
        "calibration_decision": {
            "path": str(decision_path),
            "sha256": sha256_file(decision_path),
        },
        "implementation": {"path": str(implementation), "sha256": implementation_sha},
    }
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError("Existing V4 project must be a real directory")
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs, implementation_sha)
        print(f"multimodal_cell_state_v4_project={output / 'project.yml'}")
        print("generation_status=verified_reuse")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        (staging / "projection").mkdir()
        (staging / "runs").mkdir()
        field_receipts = merge_manifest.get("identity", {}).get(
            "ordered_field_receipts"
        )
        if not isinstance(field_receipts, list) or not field_receipts:
            raise ValueError("V4 feature merge lacks ordered field receipts")
        dead_by_key: dict[str, str] = {}
        for declared in field_receipts:
            receipt_path = require_inside(
                require_file(
                    Path(str(declared["generation"])) / "feature_receipt.json",
                    "V4 field receipt",
                ),
                shadow_root,
                "V4 field receipt",
            )
            if sha256_file(receipt_path) != declared["receipt_sha256"]:
                raise ValueError("V4 field receipt hash differs")
            receipt = load_json(receipt_path)
            dead_input = receipt.get("inputs", {}).get("dead_raw", {})
            dead_path = require_file(Path(str(dead_input.get("path", ""))), "Dead raw")
            if sha256_file(dead_path) != dead_input.get("sha256"):
                raise ValueError("Dead raw identity differs from field receipt")
            key = str(receipt.get("key", ""))
            if not key or key in dead_by_key:
                raise ValueError("V4 field receipts contain blank/duplicate key")
            dead_by_key[key] = str(dead_path)
        key_by_image = {row["image_id"]: row["key"] for row in base_cells}
        image_template: dict[str, dict[str, str]] = {}
        for row in images:
            if row["channel_id"] == "nuclei":
                image_template[row["image_id"]] = row
        if set(image_template) != set(key_by_image):
            raise ValueError("Base image registry does not cover every V4 image_id")
        output_images: list[dict[str, str]] = []
        for row in images:
            output_images.append(dict(row))
            if row["channel_id"] == "nuclei":
                key = key_by_image[row["image_id"]]
                if key not in dead_by_key:
                    raise ValueError(f"No Dead raw identity for field {key}")
                dead_row = dict(row)
                dead_row.update(
                    {
                        "channel_id": "dead",
                        "image_path": dead_by_key[key],
                        "display_name": "Dead fluorescence",
                        "display_color": "#ff2d8d",
                        "display_percentile_low": "1",
                        "display_percentile_high": "99.8",
                    }
                )
                output_images.append(dead_row)
        output_images.sort(
            key=lambda row: (
                row["image_id"], EXPECTED_CHANNELS.index(row["channel_id"])
            )
        )
        write_tsv(staging / "images.tsv", image_fields, output_images)
        class_fields = [
            "class_id", "display_name", "color", "description", "shortcut",
            "trainable", "role", "order",
        ]
        write_tsv(
            staging / "classes.tsv",
            class_fields,
            [
                {
                    "class_id": "non_dead", "display_name": "Non-dead",
                    "color": "#16875b", "description": "Provisional broad non-death region; image review is authoritative",
                    "shortcut": "n", "trainable": "true", "role": "death_primary", "order": "1",
                },
                {
                    "class_id": "dead", "display_name": "Dead (all stages)",
                    "color": "#d43d51", "description": "Provisional broad death region including marker-positive and quenched late-death-like states",
                    "shortcut": "d", "trainable": "true", "role": "death_primary", "order": "2",
                },
            ],
        )
        shutil.copyfile(umap_path, staging / "projection" / "umap.tsv")
        shutil.copyfile(
            clusters_path, staging / "projection" / "diagnostic_clusters.tsv"
        )
        shutil.copyfile(
            projection_manifest_path,
            staging / "projection" / "projection_manifest.json",
        )
        shutil.copyfile(
            decision_path, staging / "projection" / "calibration_decision.json"
        )

        clusters = {row["cell_id"]: row["diagnostic_cluster"] for row in cluster_rows}
        nuclei_statuses = {
            row["cell_id"]: row["nuclei_measurement_status"] for row in feature_rows
        }
        dead_statuses = {
            row["cell_id"]: row["dead_measurement_status"] for row in feature_rows
        }
        output_cell_fields = [*cell_fields]
        for name in (
            "diagnostic_cluster", "nuclei_measurement_status",
            "dead_measurement_status",
        ):
            if name not in output_cell_fields:
                output_cell_fields.append(name)
        updated_cells = [
            {
                **row,
                "diagnostic_cluster": clusters[row["cell_id"]],
                "nuclei_measurement_status": nuclei_statuses[row["cell_id"]],
                "dead_measurement_status": dead_statuses[row["cell_id"]],
            }
            for row in base_cells
        ]
        write_tsv(staging / "cells.tsv", output_cell_fields, updated_cells)
        write_tsv(
            staging / "features.tsv",
            ["cell_id", *selected],
            [
                {"cell_id": row["cell_id"], **{name: row[name] for name in selected}}
                for row in feature_rows
            ],
        )
        project = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "cells_file": "cells.tsv",
            "features_file": "features.tsv",
            "images_file": "images.tsv",
            "classes_file": "classes.tsv",
            "runs_dir": "runs",
            "projection": {
                "mode": "existing_umap",
                "coordinate_file": "projection/umap.tsv",
                "allow_subset": False,
            },
            "annotation": {
                "title": "SUM159 multimodal V4 broad death-region annotation (three-channel review is authoritative)",
                "direct_class_limit": 8,
                "point_radius": 1.5,
                "boundary_tolerance": 1e-10,
            },
            # CPA requires an explicit grouping boundary even for validate/UMAP
            # stages.  V4 uses wells as independent sources via source_id; the
            # separate post-review trainer owns the predeclared alpha grid.
            "classifier": {
                "feature_columns": selected,
                "group_column": "source_id",
                "allow_ungrouped": False,
            },
        }
        write_json(staging / "project.yml", project)
        outputs = artifact_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v4",
            "inputs": inputs,
            "implementation_sha256": implementation_sha,
            "cell_universe": {
                "row_count": len(cell_ids),
                "ordered_cell_id_sha256": sha256_lines(cell_ids),
                "split": "development",
                "heldout_read": False,
            },
            "selected_feature_columns": selected,
            "image_channels": list(EXPECTED_CHANNELS),
            "polygon_labels_authoritative_for_training": False,
            "polygon_role": "provisional_broad_dead_region_navigation_and_sampling",
            "training_label_source": "independent_three_channel_anchor_and_boundary_review_only",
            "forbidden_inputs_read": [],
            "output_file_sha256": outputs,
        }
        manifest_path = staging / "parent_import_manifest.json"
        write_json(manifest_path, manifest)
        identity = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "status": "COMPLETE",
            "parent_import_manifest_sha256": sha256_file(manifest_path),
            "implementation_sha256": implementation_sha,
            "output_file_sha256": outputs,
        }
        write_json(staging / "project_generation_identity.json", identity)
        if sha256_file(implementation) != implementation_sha:
            raise RuntimeError("V4 project builder changed during execution")
        if output.exists():
            raise FileExistsError(
                f"V4 project output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v4_project={output / 'project.yml'}")
    print(f"selected_feature_count={len(selected)}")
    print("polygon_training_authority=0")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
