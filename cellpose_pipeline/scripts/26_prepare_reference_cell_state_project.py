#!/usr/bin/env python3
"""Create an independent reference-cell-state project from a frozen parent.

The parent is the completed, development-only representative project produced
by ``18_prepare_broad_phenotype_projection.py``.  This command does not reuse
its UMAP coordinates or broad-phenotype classes.  It preserves the exact cell
row universe, selects the historical reference method's nine promoted shape
features, installs the three reference cell-state classes, and writes a new
project configured to recompute UMAP and train a grouped nested-CV lasso.

Every parent input is read-only.  Parent hashes are sampled before validation
and checked again immediately before the staged output directory is installed
atomically.  Existing output generations and symlinked paths fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "reference_cell_state_parent_import_v1"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
PARENT_MANIFEST_SCHEMA_VERSION = "broad_phenotype_projection_input_v1"
PARENT_UMAP_SCHEMA_VERSION = "cell_phenotype_annotator_umap_v1"
FEATURE_CONFIG_SCHEMA_VERSION = "reference_cell_state_feature_config_v1"
PROJECT_ID = "reference_cell_state_development"
REFERENCE_FEATURES = (
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
REFERENCE_CLASS_IDS = (
    "multinucleated_cell",
    "dead_cell",
    "live_cell",
)
REVIEW_CHANNEL_IDS = ("brightfield", "nuclei")
UMAP_ARTIFACTS = {
    "normalized_project": "normalized_project.json",
    "input_manifest": "input_manifest.tsv",
    "join_audit": "join_audit.tsv",
    "feature_report": "feature_report.tsv",
    "row_universe_audit": "row_universe_audit.tsv",
    "feature_transform_manifest": "feature_transform_manifest.tsv",
    "umap": "umap.tsv",
}
CLASSIFIER_CONTRACT = {
    "engine": "glmnet_multinomial",
    "alpha": 1.0,
    "lambda_rule": "lambda.1se",
    "outer_folds": 5,
    "inner_folds": 5,
    "group_column": "well",
    "minimum_confidence": 0.8,
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def default_config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-project", type=Path, required=True)
    parser.add_argument("--parent-shadow-root", type=Path, required=True)
    parser.add_argument("--reference-shadow-root", type=Path, required=True)
    parser.add_argument(
        "--parent-projection-manifest",
        type=Path,
        help=(
            "Defaults to <parent-shadow-root>/projection_input/"
            "projection_input_manifest.json."
        ),
    )
    parser.add_argument(
        "--parent-umap-manifest",
        type=Path,
        help=(
            "Optional completed parent umap_manifest.json. By default the parent "
            "representative runs directory must contain exactly one generation."
        ),
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=default_config_path("reference_cell_state_classes_v1.tsv"),
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=default_config_path("reference_cell_state_features_v1.json"),
    )
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20260812)
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


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_symlink_chain(path: Path, root: Path, label: str) -> None:
    """Reject symlinks at/below a trusted lexical root.

    Operating-system aliases above the supplied root (for example macOS
    ``/var -> /private/var``) are intentionally tolerated.
    """

    lexical = Path(os.path.abspath(str(path.expanduser())))
    canonical_root = root.expanduser().resolve()
    anchor = next(
        (
            candidate
            for candidate in (lexical, *lexical.parents)
            if candidate.resolve() == canonical_root
        ),
        None,
    )
    if anchor is None:
        raise ValueError(f"{label} must not reach its root through a symlink: {lexical}")
    relative = lexical.relative_to(anchor)
    candidates = [anchor]
    candidates.extend(
        anchor / Path(*relative.parts[:index])
        for index in range(1, len(relative.parts) + 1)
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")


def require_parent_path(path: Path, root: Path, label: str, kind: str = "file") -> Path:
    reject_symlink_chain(path, root, label)
    resolved = path.expanduser().resolve()
    if not is_inside(resolved, root):
        raise ValueError(f"{label} must be inside parent shadow root {root}: {resolved}")
    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(resolved)
    if kind == "dir" and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def prepare_reference_root(path: Path) -> Path:
    lexical = Path(os.path.abspath(str(path.expanduser())))
    if lexical.exists() and lexical.is_symlink():
        raise ValueError(f"Reference shadow root must not be a symlink: {lexical}")
    if lexical.exists() and not lexical.is_dir():
        raise NotADirectoryError(lexical)
    lexical.mkdir(parents=True, exist_ok=True)
    if lexical.is_symlink():
        raise ValueError(f"Reference shadow root must not be a symlink: {lexical}")
    return lexical.resolve()


def require_plain_descendant(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    if not is_inside(resolved, root):
        raise ValueError(f"{label} must be inside reference shadow root {root}: {resolved}")
    relative = Path(os.path.abspath(str(path))).relative_to(
        Path(os.path.abspath(str(root)))
    )
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
    return resolved


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise RuntimeError(f"Non-JSON {label} requires PyYAML: {path}") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one mapping: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = path.open("r", newline="", encoding="utf-8-sig")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"TSV has no header: {path}")
    fields = [str(field) for field in reader.fieldnames]
    if any(not field for field in fields) or len(fields) != len(set(fields)):
        handle.close()
        raise ValueError(f"TSV has blank or duplicate columns: {path}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {field: str(row.get(field, "") or "") for field in fields}
        finally:
            handle.close()

    return fields, rows()


def finite(value: str) -> bool:
    if value.strip().lower() in MISSING_TOKENS:
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def resolve_project_path(project_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Parent project {label} must be a nonblank path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    return path


def load_feature_config(path: Path) -> dict[str, Any]:
    value = load_mapping(path, "Reference feature config")
    if value.get("schema_version") != FEATURE_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported reference feature config: {value.get('schema_version')}")
    for field in (
        "numeric_feature_columns",
        "projection_feature_columns",
        "classifier_feature_columns",
    ):
        if value.get(field) != list(REFERENCE_FEATURES):
            raise ValueError(f"Reference feature config {field} must equal the exact V1 order")
    if value.get("projection_transform") != "robust":
        raise ValueError("Reference projection transform must be robust")
    if value.get("nuclei_policy") != "review_display_only":
        raise ValueError("Nuclei must remain review-display-only")
    if value.get("viability_evidence_policy") != "not_read":
        raise ValueError("Viability evidence must remain outside the reference method")
    if value.get("classifier_contract") != CLASSIFIER_CONTRACT:
        raise ValueError("Reference classifier contract differs from frozen V1")
    return value


def validate_classes(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, iterator = read_tsv(path)
    expected_fields = [
        "class_id",
        "display_name",
        "color",
        "description",
        "shortcut",
        "trainable",
        "role",
        "order",
    ]
    if fields != expected_fields:
        raise ValueError(
            f"Reference classes header drift: expected={expected_fields} observed={fields}"
        )
    rows = list(iterator)
    if [row["class_id"] for row in rows] != list(REFERENCE_CLASS_IDS):
        raise ValueError("Reference classes must use the frozen priority order")
    if any(
        row["trainable"] != "true"
        or row["role"] != "phenotype"
        or row["order"] != str(index)
        for index, row in enumerate(rows, start=1)
    ):
        raise ValueError("All three reference classes must be ordered trainable phenotypes")
    return fields, rows


def validate_parent_manifest(
    path: Path,
    parent_root: Path,
    parent_project: Path,
    expected_cell_count: int,
) -> dict[str, Any]:
    manifest = load_mapping(path, "Parent projection-input manifest")
    if manifest.get("schema_version") != PARENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parent projection manifest: {manifest.get('schema_version')}")
    declared_project = manifest.get("representative_project")
    if not isinstance(declared_project, str):
        raise ValueError("Parent projection manifest lacks representative_project")
    declared_project_path = require_parent_path(
        Path(declared_project), parent_root, "Manifest representative project"
    )
    if declared_project_path != parent_project:
        raise ValueError(
            "Parent projection manifest points to a different representative project: "
            f"{declared_project_path} != {parent_project}"
        )
    expected_project_hash = manifest.get("representative_project_sha256")
    if expected_project_hash != sha256_file(parent_project):
        raise ValueError("Parent project hash disagrees with projection-input manifest")
    if manifest.get("representative_development_cell_count") != expected_cell_count:
        raise ValueError(
            "Parent projection manifest representative row count differs from the frozen "
            f"expectation: {manifest.get('representative_development_cell_count')} "
            f"!= {expected_cell_count}"
        )
    if manifest.get("representative_heldout_cell_count") != 0:
        raise ValueError("Parent representative project must contain zero heldout cells")
    if manifest.get("heldout_exclusion_verified") is not True:
        raise ValueError("Parent projection manifest did not verify heldout exclusion")
    for path_field, hash_field in (
        ("selection", "selection_sha256"),
        ("split_manifest", "split_manifest_sha256"),
    ):
        declared = manifest.get(path_field)
        expected = manifest.get(hash_field)
        if not isinstance(declared, str) or not isinstance(expected, str):
            raise ValueError(f"Parent projection manifest lacks {path_field}/{hash_field}")
        artifact = require_parent_path(Path(declared), parent_root, f"Parent {path_field}")
        if sha256_file(artifact) != expected:
            raise ValueError(f"Parent projection manifest hash mismatch: {path_field}")
    return manifest


def locate_parent_umap_manifest(
    explicit: Path | None,
    parent: dict[str, Any],
    parent_project: Path,
    parent_root: Path,
) -> Path:
    if explicit is not None:
        return require_parent_path(explicit, parent_root, "Parent UMAP manifest")
    runs_value = parent.get("runs_dir")
    runs_dir = require_parent_path(
        resolve_project_path(parent_project, runs_value, "runs_dir"),
        parent_root,
        "Parent representative runs directory",
        kind="dir",
    )
    candidates = sorted(runs_dir.rglob("umap_manifest.json"), key=lambda path: str(path))
    if len(candidates) != 1:
        raise ValueError(
            "Completed parent representative project must contain exactly one UMAP "
            f"generation unless --parent-umap-manifest is supplied; observed={len(candidates)}"
        )
    return require_parent_path(candidates[0], parent_root, "Parent UMAP manifest")


def validate_parent_umap_generation(
    manifest_path: Path,
    parent_root: Path,
    parent_project: Path,
    assets: dict[str, Path],
    parent_project_id: Any,
    expected_cell_count: int,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = load_mapping(manifest_path, "Parent UMAP manifest")
    if manifest.get("schema_version") != PARENT_UMAP_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parent UMAP schema: {manifest.get('schema_version')}")
    if manifest.get("project_id") != parent_project_id:
        raise ValueError("Parent UMAP project_id differs from its representative project")
    for field in ("configured_cell_count", "row_count"):
        if manifest.get(field) != expected_cell_count:
            raise ValueError(
                f"Parent UMAP {field} differs from expected rows: "
                f"{manifest.get(field)} != {expected_cell_count}"
            )
    if manifest.get("projection_mode") != "compute_umap":
        raise ValueError("Parent UMAP generation must have compute_umap provenance")
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict) or set(declared) != set(UMAP_ARTIFACTS):
        observed = sorted(declared) if isinstance(declared, dict) else type(declared).__name__
        raise ValueError(
            "Parent UMAP manifest artifact set differs: "
            f"expected={sorted(UMAP_ARTIFACTS)} observed={observed}"
        )
    artifact_paths: dict[str, Path] = {}
    for role, filename in UMAP_ARTIFACTS.items():
        artifact = require_parent_path(
            manifest_path.parent / filename,
            parent_root,
            f"Parent UMAP artifact {role}",
        )
        if sha256_file(artifact) != declared[role]:
            raise ValueError(f"Parent UMAP artifact hash mismatch: {role}")
        artifact_paths[f"umap_artifact:{role}"] = artifact

    input_fields, input_rows_iterator = read_tsv(
        artifact_paths["umap_artifact:input_manifest"]
    )
    required_input_fields = {"input_role", "normalized_path", "raw_sha256"}
    if not required_input_fields.issubset(input_fields):
        raise ValueError(
            "Parent UMAP input manifest lacks fields: "
            f"{sorted(required_input_fields-set(input_fields))}"
        )
    input_rows = list(input_rows_iterator)
    by_role = {row["input_role"]: row for row in input_rows}
    if len(by_role) != len(input_rows):
        raise ValueError("Parent UMAP input manifest contains duplicate input_role values")
    expected_inputs = {
        "cells": assets["cells_file"],
        "features": assets["features_file"],
        "project_config": parent_project,
    }
    if set(by_role) != set(expected_inputs):
        raise ValueError(
            "Parent UMAP input role set differs: "
            f"expected={sorted(expected_inputs)} observed={sorted(by_role)}"
        )
    for role, expected_path in expected_inputs.items():
        row = by_role[role]
        normalized_path = require_parent_path(
            Path(row["normalized_path"]), parent_root, f"Parent UMAP input {role}"
        )
        if normalized_path != expected_path:
            raise ValueError(f"Parent UMAP input path differs for {role}")
        if row["raw_sha256"] != sha256_file(expected_path):
            raise ValueError(f"Parent UMAP input raw SHA-256 differs for {role}")
    return manifest, artifact_paths


def validate_parent_umap_rows(
    umap_path: Path, expected_ids: set[str], expected_count: int
) -> dict[str, str]:
    fields, rows = read_tsv(umap_path)
    required = {"cell_id", "Dim1", "Dim2"}
    if not required.issubset(fields):
        raise ValueError(f"Parent UMAP table is missing columns: {sorted(required-set(fields))}")
    observed: set[str] = set()
    ordered_ids: list[str] = []
    for row in rows:
        cell_id = row["cell_id"]
        if not cell_id or cell_id in observed:
            raise ValueError(f"Parent UMAP contains blank or duplicate cell_id: {cell_id!r}")
        if not finite(row["Dim1"]) or not finite(row["Dim2"]):
            raise ValueError(f"Parent UMAP has nonfinite coordinates for cell_id={cell_id}")
        observed.add(cell_id)
        ordered_ids.append(cell_id)
    if len(observed) != expected_count or observed != expected_ids:
        raise ValueError(
            "Parent UMAP row universe differs from parent cells: "
            f"umap={len(observed)} cells={len(expected_ids)} "
            f"umap_only={sorted(observed-expected_ids)[:5]} "
            f"cells_only={sorted(expected_ids-observed)[:5]}"
        )
    return {
        "parent_umap_ordered_cell_id_sha256": sha256_text_parts(ordered_ids),
        "parent_umap_cell_id_sha256": sha256_text_parts(sorted(observed)),
    }


def validate_cell_feature_rows(
    cells_path: Path,
    features_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    cell_fields, cell_rows = read_tsv(cells_path)
    required_cells = {"cell_id", "image_id", "mask_label", "branch", "key", "well", "split"}
    missing_cells = sorted(required_cells - set(cell_fields))
    if missing_cells:
        raise ValueError(f"Parent cells table is missing columns: {missing_cells}")
    feature_fields, feature_rows = read_tsv(features_path)
    missing_features = sorted({"cell_id", *REFERENCE_FEATURES} - set(feature_fields))
    if missing_features:
        raise ValueError(f"Parent features table is missing reference columns: {missing_features}")

    sentinel = object()
    seen: set[str] = set()
    image_ids: set[str] = set()
    wells: set[str] = set()
    row_parts: list[str] = []
    count = 0
    for position, pair in enumerate(
        itertools.zip_longest(cell_rows, feature_rows, fillvalue=sentinel), start=1
    ):
        cell_row, feature_row = pair
        if cell_row is sentinel or feature_row is sentinel:
            raise ValueError(
                "Parent cell/feature lockstep row count differs at row "
                f"{position}: cells_exhausted={cell_row is sentinel} "
                f"features_exhausted={feature_row is sentinel}"
            )
        assert isinstance(cell_row, dict) and isinstance(feature_row, dict)
        cell_id = cell_row["cell_id"]
        if not cell_id or feature_row["cell_id"] != cell_id:
            raise ValueError(
                "Parent cell/feature lockstep identity differs at row "
                f"{position}: cells={cell_id!r} features={feature_row['cell_id']!r}"
            )
        if cell_id in seen:
            raise ValueError(f"Parent representative project contains duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        if cell_row["split"] != "development":
            raise ValueError(f"Heldout/non-development cell entered parent project: {cell_id}")
        if cell_row["image_id"] != cell_row["key"]:
            raise ValueError(f"Parent cell image_id/key mismatch: {cell_id}")
        try:
            mask_label = int(cell_row["mask_label"])
        except ValueError as error:
            raise ValueError(f"Parent mask label is not an integer: {cell_id}") from error
        expected_id = f"{cell_row['branch']}|{cell_row['key']}|{mask_label}"
        if cell_id != expected_id:
            raise ValueError(f"Parent stable cell identity changed: {cell_id} != {expected_id}")
        invalid = [name for name in REFERENCE_FEATURES if not finite(feature_row[name])]
        if invalid:
            raise ValueError(f"Parent reference features are incomplete for {cell_id}: {invalid}")
        row_parts.append(
            "\t".join([cell_id, *(feature_row[name] for name in REFERENCE_FEATURES)])
        )
        image_ids.add(cell_row["image_id"])
        wells.add(cell_row["well"])
        count += 1
    if count != expected_count:
        raise ValueError(f"Parent representative row count differs: {count} != {expected_count}")
    return {
        "cell_fields": cell_fields,
        "cell_count": count,
        "cell_id_sha256": sha256_text_parts(sorted(seen)),
        "ordered_cell_id_sha256": sha256_text_parts(row.split("\t", 1)[0] for row in row_parts),
        "ordered_cell_feature_sha256": sha256_text_parts(row_parts),
        "image_ids": image_ids,
        "cell_ids": seen,
        "well_count": len(wells),
    }


def write_reference_features(source: Path, output: Path, expected_ids: set[str]) -> int:
    fields, rows = read_tsv(source)
    if not {"cell_id", *REFERENCE_FEATURES}.issubset(fields):
        raise ValueError("Parent features changed after validation")
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cell_id", *REFERENCE_FEATURES],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if row["cell_id"] not in expected_ids:
                raise ValueError(f"Unexpected cell appeared during feature copy: {row['cell_id']}")
            writer.writerow({name: row[name] for name in ["cell_id", *REFERENCE_FEATURES]})
            count += 1
    if count != len(expected_ids):
        raise ValueError(f"Reference feature copy coverage differs: {count} != {len(expected_ids)}")
    return count


def write_review_images(source: Path, output: Path, expected_images: set[str]) -> int:
    fields, rows = read_tsv(source)
    required = {"image_id", "channel_id"}
    if not required.issubset(fields):
        raise ValueError(f"Parent images table is missing columns: {sorted(required-set(fields))}")
    selected: list[dict[str, str]] = []
    observed: dict[str, set[str]] = {image_id: set() for image_id in expected_images}
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        image_id = row["image_id"]
        channel_id = row["channel_id"]
        if image_id not in expected_images or channel_id not in REVIEW_CHANNEL_IDS:
            continue
        pair = (image_id, channel_id)
        if pair in pairs:
            raise ValueError(f"Parent images table duplicates image/channel: {pair}")
        pairs.add(pair)
        observed[image_id].add(channel_id)
        selected.append(row)
    incomplete = sorted(
        image_id
        for image_id, channels in observed.items()
        if channels != set(REVIEW_CHANNEL_IDS)
    )
    if incomplete:
        raise ValueError(
            "Every reference image requires exactly Brightfield and Nuclei review channels: "
            f"{incomplete[:10]}"
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)


def build_project(seed: int, cell_count: int) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "classes_file": "classes.tsv",
        "cells_file": "cells.tsv",
        "features_file": "features.tsv",
        "images_file": "images.tsv",
        "runs_dir": "runs",
        "projection": {
            "mode": "compute_umap",
            "feature_columns": list(REFERENCE_FEATURES),
            "transform": "robust",
            "missing_policy": "fail",
            "constant_policy": "drop_with_manifest",
            "seed": seed,
            "n_neighbors": min(30, cell_count - 1),
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_threads": 1,
            "n_sgd_threads": 1,
        },
        "annotation": {
            "title": "Reference cell-state annotation",
            "direct_class_limit": 8,
            "point_radius": 1.5,
            "boundary_tolerance": 1e-10,
        },
        "review": {
            "strata": ["well"],
            "group_column": "well",
            "quota_scope": "global",
            "quotas": {
                "default_per_class": 50,
                "unassigned": 100,
                "explicit_unassigned": 0,
            },
            "max_per_group": 8,
            "mask_padding": 6,
            "shortage_policy": "fail",
            "seed": seed,
            "unavailable_image_policy": "fail",
            "displays": [
                {
                    "display_id": "brightfield",
                    "mode": "single",
                    "channel_ids": ["brightfield"],
                    "normalization_scope": "image",
                    "gamma": 1,
                },
                {
                    "display_id": "nuclei",
                    "mode": "single",
                    "channel_ids": ["nuclei"],
                    "normalization_scope": "image",
                    "gamma": 1,
                },
            ],
        },
        "classifier": {
            "feature_columns": list(REFERENCE_FEATURES),
            "missing_policy": "median_impute",
            "constant_policy": "drop_with_manifest",
            "group_column": "well",
            "allow_ungrouped": False,
            "outer_folds": 5,
            "inner_folds": 5,
            "seed": seed,
            "engine": "glmnet_multinomial",
            "alpha": 1.0,
            "lambda_rule": "lambda.1se",
            "eligible_review_status": ["confirmed", "corrected"],
            "minimum_confidence": 0.8,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_cell_count < 3:
        raise ValueError("Reference project requires at least three expected cells")
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")

    parent_root = args.parent_shadow_root.expanduser().resolve()
    if not parent_root.is_dir():
        raise FileNotFoundError(parent_root)
    reject_symlink_chain(args.parent_shadow_root, parent_root, "Parent shadow root")
    parent_project = require_parent_path(
        args.parent_project, parent_root, "Parent representative project"
    )
    manifest_path = require_parent_path(
        args.parent_projection_manifest
        if args.parent_projection_manifest is not None
        else parent_root / "projection_input" / "projection_input_manifest.json",
        parent_root,
        "Parent projection-input manifest",
    )
    manifest_initial_sha256 = sha256_file(manifest_path)
    project_initial_sha256 = sha256_file(parent_project)
    parent_manifest = validate_parent_manifest(
        manifest_path, parent_root, parent_project, args.expected_cell_count
    )

    reference_candidate = Path(
        os.path.abspath(str(args.reference_shadow_root.expanduser()))
    ).resolve(strict=False)
    if is_inside(reference_candidate, parent_root) or is_inside(parent_root, reference_candidate):
        raise ValueError("Parent and reference shadow roots must be disjoint")
    candidate_output = reference_candidate / "projection_input" / "representative_umap"
    if os.path.lexists(candidate_output):
        raise FileExistsError(f"Refusing to replace existing reference project: {candidate_output}")

    classes_input = Path(os.path.abspath(str(args.classes_file.expanduser())))
    feature_config_input = Path(os.path.abspath(str(args.feature_config.expanduser())))
    if classes_input.is_symlink():
        raise ValueError(f"Reference classes config must not be a symlink: {classes_input}")
    if feature_config_input.is_symlink():
        raise ValueError(
            f"Reference feature config must not be a symlink: {feature_config_input}"
        )
    classes_path = classes_input.resolve()
    feature_config_path = feature_config_input.resolve()
    if not classes_path.is_file():
        raise FileNotFoundError(classes_path)
    if not feature_config_path.is_file():
        raise FileNotFoundError(feature_config_path)
    config_hashes_before = {
        "classes_file": sha256_file(classes_path),
        "feature_config": sha256_file(feature_config_path),
    }
    validate_classes(classes_path)
    feature_config = load_feature_config(feature_config_path)

    parent = load_mapping(parent_project, "Parent representative project")
    if parent.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported parent project schema: {parent.get('schema_version')}")
    projection = parent.get("projection")
    classifier = parent.get("classifier")
    if not isinstance(projection, dict) or projection.get("mode") != "compute_umap":
        raise ValueError("Parent project must be configured to compute UMAP")
    if "coordinate_file" in projection:
        raise ValueError("Parent representative project must not import old UMAP coordinates")
    if not isinstance(classifier, dict):
        raise ValueError("Parent project lacks classifier configuration")
    parent_projection_features = projection.get("feature_columns")
    if not isinstance(parent_projection_features, list) or not set(REFERENCE_FEATURES).issubset(
        parent_projection_features
    ):
        raise ValueError("Parent projection allowlist does not contain all nine reference features")

    assets = {
        field: require_parent_path(
            resolve_project_path(parent_project, parent.get(field), field),
            parent_root,
            f"Parent {field}",
        )
        for field in ("classes_file", "cells_file", "features_file", "images_file")
    }
    parent_umap_manifest_path = locate_parent_umap_manifest(
        args.parent_umap_manifest, parent, parent_project, parent_root
    )
    parent_umap, parent_umap_artifacts = validate_parent_umap_generation(
        parent_umap_manifest_path,
        parent_root,
        parent_project,
        assets,
        parent.get("project_id"),
        args.expected_cell_count,
    )
    parent_support_paths = {
        "projection_selection": require_parent_path(
            Path(parent_manifest["selection"]), parent_root, "Parent projection selection"
        ),
        "projection_split_manifest": require_parent_path(
            Path(parent_manifest["split_manifest"]),
            parent_root,
            "Parent projection split manifest",
        ),
        "umap_manifest": parent_umap_manifest_path,
        **parent_umap_artifacts,
    }
    parent_files = {
        "projection_input_manifest": manifest_path,
        "project": parent_project,
        **{field.removesuffix("_file"): path for field, path in assets.items()},
        **parent_support_paths,
    }
    parent_hashes_before = {role: sha256_file(path) for role, path in parent_files.items()}
    if parent_hashes_before["projection_input_manifest"] != manifest_initial_sha256:
        raise RuntimeError("Parent projection-input manifest changed during preflight")
    if parent_hashes_before["project"] != project_initial_sha256:
        raise RuntimeError("Parent project changed during preflight")
    if parent_hashes_before["project"] != parent_manifest["representative_project_sha256"]:
        raise RuntimeError("Parent project snapshot no longer matches its declared hash")
    row_lock = validate_cell_feature_rows(
        assets["cells_file"], assets["features_file"], args.expected_cell_count
    )
    umap_row_lock = validate_parent_umap_rows(
        parent_umap_artifacts["umap_artifact:umap"],
        row_lock["cell_ids"],
        args.expected_cell_count,
    )

    # Do not create even an empty reference directory until all parent inputs,
    # configs, and the complete row lock have passed validation.
    reference_root = prepare_reference_root(args.reference_shadow_root)
    output_parent = require_plain_descendant(
        reference_root / "projection_input", reference_root, "Reference projection root"
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = require_plain_descendant(
        output_parent / "representative_umap", reference_root, "Reference project output"
    )
    if os.path.lexists(output_root):
        raise FileExistsError(f"Refusing to replace existing reference project: {output_root}")

    staging = output_parent / f".representative_umap.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        shutil.copyfile(assets["cells_file"], staging / "cells.tsv")
        selected_ids: set[str] = set()
        _, selected_rows = read_tsv(staging / "cells.tsv")
        for row in selected_rows:
            selected_ids.add(row["cell_id"])
        if len(selected_ids) != args.expected_cell_count:
            raise RuntimeError("Staged cells table row universe changed during copy")
        write_reference_features(
            assets["features_file"], staging / "features.tsv", selected_ids
        )
        image_row_count = write_review_images(
            assets["images_file"], staging / "images.tsv", row_lock["image_ids"]
        )
        shutil.copyfile(classes_path, staging / "classes.tsv")
        (staging / "runs").mkdir()
        project = build_project(args.seed, args.expected_cell_count)
        (staging / "project.yml").write_text(
            json.dumps(project, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

        parent_hashes_after = {role: sha256_file(path) for role, path in parent_files.items()}
        if parent_hashes_after != parent_hashes_before:
            changed = sorted(
                role
                for role in parent_hashes_before
                if parent_hashes_before[role] != parent_hashes_after[role]
            )
            raise RuntimeError(f"Parent project changed during reference import: {changed}")
        if sha256_file(staging / "cells.tsv") != parent_hashes_before["cells"]:
            raise RuntimeError("Reference cells table is not an exact byte copy of its parent")
        config_hashes_after = {
            "classes_file": sha256_file(classes_path),
            "feature_config": sha256_file(feature_config_path),
        }
        if config_hashes_after != config_hashes_before:
            raise RuntimeError("Reference class/feature config changed during import")

        output_assets = {
            name: staging / name
            for name in ("classes.tsv", "cells.tsv", "features.tsv", "images.tsv", "project.yml")
        }
        import_manifest = {
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "phase": "independent_reference_cell_state_project",
            "parent": {
                "shadow_root": str(parent_root),
                "project_id": parent.get("project_id"),
                "project": str(parent_project),
                "projection_input_manifest": str(manifest_path),
                "projection_input_manifest_schema_version": parent_manifest.get(
                    "schema_version"
                ),
                "completed_umap_manifest": str(parent_umap_manifest_path),
                "completed_umap_schema_version": parent_umap.get("schema_version"),
                "completed_umap_run_id": parent_umap.get("run_id"),
                "completed_umap_projection_id": parent_umap.get("projection_id"),
                "completed_umap_cell_universe_sha256": parent_umap.get(
                    "cell_universe_sha256"
                ),
                "input_file_sha256": parent_hashes_before,
            },
            "row_lock": {
                "cell_count": row_lock["cell_count"],
                "heldout_cell_count": 0,
                "well_count": row_lock["well_count"],
                "image_count": len(row_lock["image_ids"]),
                "image_manifest_row_count": image_row_count,
                "cell_id_sha256": row_lock["cell_id_sha256"],
                "ordered_cell_id_sha256": row_lock["ordered_cell_id_sha256"],
                "ordered_cell_feature_sha256": row_lock[
                    "ordered_cell_feature_sha256"
                ],
                **umap_row_lock,
                "parent_cells_and_umap_cell_id_sets_equal": True,
                "cells_byte_identical_to_parent": True,
            },
            "reference_contract": {
                "class_ids_priority_order": list(REFERENCE_CLASS_IDS),
                "unassigned_state": "built_in_nontraining_review_state",
                "feature_columns": list(REFERENCE_FEATURES),
                "projection_mode": "compute_umap",
                "parent_umap_coordinates_imported": False,
                "projection_transform": "robust",
                "review_channel_ids": list(REVIEW_CHANNEL_IDS),
                "nuclei_policy": "review_display_only",
                "viability_evidence_policy": "not_read",
                "classifier": CLASSIFIER_CONTRACT,
            },
            "config_inputs": {
                "classes_file": str(classes_path),
                "classes_file_sha256": config_hashes_before["classes_file"],
                "feature_config": str(feature_config_path),
                "feature_config_sha256": config_hashes_before["feature_config"],
                "feature_config_schema_version": feature_config["schema_version"],
            },
            "output_file_sha256": {
                name: sha256_file(path) for name, path in output_assets.items()
            },
            "write_boundary": {
                "reference_shadow_root": str(reference_root),
                "parent_inputs_read_only": True,
                "current_classifier_modified": False,
                "existing_outputs_replaced": False,
            },
        }
        (staging / "parent_import_manifest.json").write_text(
            json.dumps(import_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if os.path.lexists(output_root):
            raise FileExistsError(f"Reference output appeared during staging: {output_root}")
        os.rename(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"reference_project={output_root / 'project.yml'}")
    print(f"parent_import_manifest={output_root / 'parent_import_manifest.json'}")
    print(f"reference_cell_count={args.expected_cell_count}")
    print("parent_umap_coordinates_imported=0")
    print("current_classifier_modified=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
