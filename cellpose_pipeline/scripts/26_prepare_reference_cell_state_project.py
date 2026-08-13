#!/usr/bin/env python3
"""Create an independent reference-cell-state project from a frozen parent.

The parent is the completed, development-only representative project produced
by ``18_prepare_broad_phenotype_projection.py``.  This command does not reuse
its UMAP coordinates or broad-phenotype classes.  It preserves the exact cell
row universe and installs the three reference cell-state classes.  The default
V1 path retains the nine-shape robust CPA UMAP.  Explicit ``--method-version
v2`` instead freezes the historical nine-shape projection plus twelve-feature
classifier contract, derives plate-map identities, calls the audited historical
R projection adapter, and imports only those newly computed coordinates.

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
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "reference_cell_state_parent_import_v1"
SCHEMA_VERSION_V2 = "reference_cell_state_parent_import_v2"
GENERATION_IDENTITY_SCHEMA_VERSION_V2 = (
    "reference_cell_state_project_generation_identity_v2"
)
GENERATION_IDENTITY_FILENAME_V2 = "reference_project_generation_identity.json"
IMPORT_MANIFEST_FILENAME = "parent_import_manifest.json"
HISTORICAL_GENERATION_IDENTITY_SCHEMA_VERSION_V2 = (
    "reference_cell_state_historical_projection_generation_identity_v2"
)
HISTORICAL_GENERATION_IDENTITY_FILENAME_V2 = (
    "historical_projection_generation_identity.json"
)
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
PARENT_MANIFEST_SCHEMA_VERSION = "broad_phenotype_projection_input_v1"
PARENT_UMAP_SCHEMA_VERSION = "cell_phenotype_annotator_umap_v1"
FEATURE_CONFIG_SCHEMA_VERSION = "reference_cell_state_feature_config_v1"
FEATURE_CONFIG_SCHEMA_VERSION_V2 = "reference_cell_state_feature_config_v2"
PROJECT_ID = "reference_cell_state_development"
PROJECT_ID_V2 = "reference_cell_state_development_v2"
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
CLASSIFIER_FEATURES_V2 = (
    *REFERENCE_FEATURES,
    "bf_boundary_mean",
    "bf_interior_mean",
    "bf_interior_minus_boundary_mean",
)
PLATE_MAP_FIELDS = (
    "well",
    "plate_row",
    "plate_column",
    "doxorubicin_nm",
    "ploidy",
    "cyclophosphamide",
    "replicate",
)
WELL_SPLIT_FIELDS = (
    *PLATE_MAP_FIELDS,
    "split",
    "split_strategy",
    "split_seed",
    "assignment_sha256",
)
PLATE_MAP_SHA256 = "cb8aa2fa4a47cfa83d2f5462420575af0726be7066e47d8a60770543156de2e9"
HISTORICAL_IDENTITY_FIELDS = (
    "context_key",
    "ltee_cell_line",
    "ltee_cell_line_family",
    "source_id",
    "suffix",
    "condition",
    "feature_row_index",
    "segmentation_qc_cell_id",
    "segmentation_object_id",
)
REFERENCE_CLASS_IDS_V1 = (
    "multinucleated_cell",
    "dead_cell",
    "live_cell",
)
REFERENCE_CLASS_IDS_V2 = (
    "live_cell",
    "dead_cell",
    "multinucleated_cell",
)
REFERENCE_CLASS_METADATA_V2 = {
    "live_cell": ("Live cell", "#16875b"),
    "dead_cell": ("Dead cell", "#d43d51"),
    "multinucleated_cell": ("Multinucleated cell", "#7257c8"),
}
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
CLASSIFIER_CONTRACT_V2 = {
    "engine": "glmnet_multinomial",
    "alpha": 1.0,
    "lambda_rule": "lambda.1se",
    "outer_folds": 5,
    "inner_folds": 5,
    "group_column": "source_id",
    "source_id_semantics": "well",
    "confidence_policy": "historical_high_or_medium_blank_as_high_low_excluded",
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
FORBIDDEN_COLUMN_RE = re.compile(
    r"(^|_)(?:class|label|dead|rgb|state|final_state|classification|confidence|trajectory|"
    r"prediction|probability|review|annotation|outcome|response)(?:_|$)",
    re.IGNORECASE,
)


def default_config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / name


def default_plate_map_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "analysisi"
        / "resources"
        / "SUM159_AC_Experiment1_PlateMap.csv"
    )


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
        help="V1/V2 classes; omitted selects the frozen classes for --method-version.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        help="V1/V2 config; omitted selects the frozen config for --method-version.",
    )
    parser.add_argument("--method-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--plate-map", type=Path, default=default_plate_map_path())
    parser.add_argument("--reference-snapshot-root", type=Path)
    parser.add_argument(
        "--historical-projection-script",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "28_build_reference_cell_state_historical_projection.R"
        ),
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--dbscan-cores", type=int, default=1)
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    parser.add_argument("--seed", type=int)
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


def sha256_named_hashes(values: dict[str, str]) -> str:
    return sha256_text_parts(
        f"{name}\t{values[name]}" for name in sorted(values)
    )


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


def load_feature_config(path: Path, method_version: str) -> dict[str, Any]:
    value = load_mapping(path, "Reference feature config")
    expected_schema = (
        FEATURE_CONFIG_SCHEMA_VERSION_V2
        if method_version == "v2"
        else FEATURE_CONFIG_SCHEMA_VERSION
    )
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"Unsupported reference feature config: {value.get('schema_version')}")
    if method_version == "v1":
        for field in (
            "numeric_feature_columns",
            "projection_feature_columns",
            "classifier_feature_columns",
        ):
            if value.get(field) != list(REFERENCE_FEATURES):
                raise ValueError(
                    f"Reference feature config {field} must equal the exact V1 order"
                )
        if value.get("projection_transform") != "robust":
            raise ValueError("Reference projection transform must be robust")
    else:
        if value.get("numeric_feature_columns") != list(CLASSIFIER_FEATURES_V2):
            raise ValueError("V2 numeric feature order must equal classifier12")
        if value.get("projection_feature_columns") != list(REFERENCE_FEATURES):
            raise ValueError("V2 projection feature order must equal shape9")
        if value.get("classifier_feature_columns") != list(CLASSIFIER_FEATURES_V2):
            raise ValueError("V2 classifier feature order must equal classifier12")
        historical = value.get("historical_projection")
        if not isinstance(historical, dict):
            raise ValueError("V2 feature config lacks historical_projection")
        required = {
            "reference_source_file": "code/lib/Utils.R",
            "reference_source_sha256": (
                "9b913da2bce7e87de3c7e9b6f084502655fd9b6514ccff1a65478b37ea2a9828"
            ),
            "size_feature_regex": "Area|Axis|Perimeter|Diameter",
            "size_feature_regex_ignore_case": False,
            "knn_k": 5,
            "pca_max_components": 10,
            "umap_seed": 42,
            "umap_n_neighbors": 15,
            "unit_adaptation": "pixel_units_constant_scale_adaptation",
            "pixel_calibration_evidence": (
                "incucyte_tiff_generic_72_dpi_only_no_ome_imagej_or_description_"
                "and_repo_declares_no_valid_microscopy_pixel_calibration"
            ),
            "historical_name_mapping_role": (
                "case_sensitive_schema_dispatch_only_not_physical_unit_claim"
            ),
            "unrecoverable_production_artifact_difference": (
                "unknown_px_to_micrometre_scale_cannot_be_recovered_and_is_not_fully_removed_"
                "for_lowercase_perimeter.µm_and_equi_diameter_because_historical_log1p_precedes_zscore"
            ),
        }
        if any(historical.get(key) != expected for key, expected in required.items()):
            raise ValueError("V2 historical projection parameters differ from frozen parity")
        representative_selection = historical.get("representative_selection")
        if representative_selection != {
            "outer_function": "get_all_cell_lines_overlay_representatives",
            "inner_function": "get_spatially_uniform_representatives",
            "seed": 1,
            "total_n": 300,
            "balance": 0.2,
            "minimum_cluster_representatives": 2,
            "context_column": "context_key",
            "cluster_role": "sampling_and_overlay_metadata_only",
        }:
            raise ValueError("V2 historical representative parameters differ from frozen parity")
    if value.get("nuclei_policy") != "review_display_only":
        raise ValueError("Nuclei must remain review-display-only")
    if value.get("viability_evidence_policy") != "not_read":
        raise ValueError("Viability evidence must remain outside the reference method")
    expected_classifier = CLASSIFIER_CONTRACT_V2 if method_version == "v2" else CLASSIFIER_CONTRACT
    if value.get("classifier_contract") != expected_classifier:
        raise ValueError(
            f"Reference classifier contract differs from frozen {method_version.upper()}"
        )
    return value


def validate_classes(
    path: Path, method_version: str
) -> tuple[list[str], list[dict[str, str]]]:
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
    expected_ids = (
        REFERENCE_CLASS_IDS_V2
        if method_version == "v2"
        else REFERENCE_CLASS_IDS_V1
    )
    if [row["class_id"] for row in rows] != list(expected_ids):
        semantics = "historical source order" if method_version == "v2" else "V1 priority order"
        raise ValueError(f"Reference classes must use the frozen {semantics}")
    if any(
        row["trainable"] != "true"
        or row["role"] != "phenotype"
        or row["order"] != str(index)
        for index, row in enumerate(rows, start=1)
    ):
        raise ValueError("All three reference classes must be ordered trainable phenotypes")
    if method_version == "v2":
        for row in rows:
            expected_display, expected_color = REFERENCE_CLASS_METADATA_V2[row["class_id"]]
            if (row["display_name"], row["color"]) != (
                expected_display,
                expected_color,
            ):
                raise ValueError(
                    "V2 classes must preserve the historical display names and colors"
                )
            if "priority" in row["description"].casefold():
                raise ValueError("V2 class descriptions must not invent priority semantics")
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
    feature_columns: Sequence[str] = REFERENCE_FEATURES,
    reject_forbidden: bool = False,
) -> dict[str, Any]:
    cell_fields, cell_rows = read_tsv(cells_path)
    required_cells = {"cell_id", "image_id", "mask_label", "branch", "key", "well", "split"}
    missing_cells = sorted(required_cells - set(cell_fields))
    if missing_cells:
        raise ValueError(f"Parent cells table is missing columns: {missing_cells}")
    feature_fields, feature_rows = read_tsv(features_path)
    if reject_forbidden:
        forbidden_cells = sorted(
            field
            for field in cell_fields
            if field != "mask_label" and FORBIDDEN_COLUMN_RE.search(field)
        )
        forbidden_features = sorted(
            field
            for field in feature_fields
            if field != "cell_id" and FORBIDDEN_COLUMN_RE.search(field)
        )
        if forbidden_cells or forbidden_features:
            raise ValueError(
                "Forbidden current/dead/RGB columns entered V2 parent: "
                f"cells={forbidden_cells} features={forbidden_features}"
            )
    missing_features = sorted({"cell_id", *feature_columns} - set(feature_fields))
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
        invalid = [name for name in feature_columns if not finite(feature_row[name])]
        if invalid:
            raise ValueError(f"Parent reference features are incomplete for {cell_id}: {invalid}")
        row_parts.append(
            "\t".join([cell_id, *(feature_row[name] for name in feature_columns)])
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


def write_reference_features(
    source: Path,
    output: Path,
    expected_ids: set[str],
    feature_columns: Sequence[str] = REFERENCE_FEATURES,
) -> int:
    fields, rows = read_tsv(source)
    if not {"cell_id", *feature_columns}.issubset(fields):
        raise ValueError("Parent features changed after validation")
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cell_id", *feature_columns],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if row["cell_id"] not in expected_ids:
                raise ValueError(f"Unexpected cell appeared during feature copy: {row['cell_id']}")
            writer.writerow({name: row[name] for name in ["cell_id", *feature_columns]})
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


def load_plate_map(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != PLATE_MAP_SHA256:
        raise ValueError(
            "V2 plate map differs from the frozen canonical snapshot: "
            f"expected={PLATE_MAP_SHA256} observed={sha256_file(path)}"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != PLATE_MAP_FIELDS:
            raise ValueError(
                f"Plate-map schema drift: expected={list(PLATE_MAP_FIELDS)} "
                f"observed={list(fields)}"
            )
        rows = [{field: str(row.get(field, "") or "") for field in fields} for row in reader]
    if len(rows) != 80 or len({row["well"] for row in rows}) != 80:
        raise ValueError("Frozen SUM159 plate map requires exactly 80 unique wells")
    by_well: dict[str, dict[str, str]] = {}
    condition_members: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        well = row["well"]
        if not re.fullmatch(r"[A-H](?:[2-9]|1[01])", well):
            raise ValueError(f"Invalid SUM159 plate-map well: {well!r}")
        if row["plate_row"] != well[0] or int(row["plate_column"]) != int(well[1:]):
            raise ValueError(f"Plate-map row/column disagrees with well: {well}")
        if row["ploidy"] not in {"2N", "4N"}:
            raise ValueError(f"Invalid plate-map ploidy: {row['ploidy']!r}")
        if row["cyclophosphamide"] not in {"true", "false"}:
            raise ValueError(
                f"Invalid plate-map cyclophosphamide: {row['cyclophosphamide']!r}"
            )
        if row["replicate"] not in {"1", "2"} or not finite(row["doxorubicin_nm"]):
            raise ValueError(f"Invalid plate-map dose/replicate for well={well}")
        if float(row["doxorubicin_nm"]) < 0:
            raise ValueError(f"Negative plate-map dose for well={well}")
        by_well[well] = row
        key = (row["doxorubicin_nm"], row["ploidy"], row["cyclophosphamide"])
        condition_members.setdefault(key, []).append(row)
    if len(condition_members) != 40 or any(
        sorted(member["replicate"] for member in members) != ["1", "2"]
        for members in condition_members.values()
    ):
        raise ValueError("Frozen SUM159 plate map requires 40 paired replicate conditions")
    return by_well, {
        "path": str(path),
        "sha256": PLATE_MAP_SHA256,
        "row_count": 80,
        "condition_count": 40,
        "paired_replicates_per_condition": 2,
    }


def canonical_number(value: str, field: str) -> str:
    if not finite(value):
        raise ValueError(f"{field} must be finite: {value!r}")
    parsed = float(value)
    return format(parsed, ".17g")


def validate_well_split_freeze(
    path: Path, plate_map: dict[str, dict[str, str]]
) -> set[str]:
    fields, iterator = read_tsv(path)
    if tuple(fields) != WELL_SPLIT_FIELDS:
        raise ValueError(
            f"Parent well split freeze schema drift: expected={list(WELL_SPLIT_FIELDS)} "
            f"observed={fields}"
        )
    rows = list(iterator)
    if len(rows) != 80 or {row["well"] for row in rows} != set(plate_map):
        raise ValueError("Parent well split freeze must cover the exact 80-well plate map")
    development: set[str] = set()
    for row in rows:
        well = row["well"]
        if any(row[field] != plate_map[well][field] for field in PLATE_MAP_FIELDS):
            raise ValueError(f"Parent well split freeze differs from plate map: {well}")
        if row["split"] not in {"development", "heldout"}:
            raise ValueError(f"Invalid parent well split: {well}={row['split']}")
        if row["split"] == "development":
            development.add(well)
    if len(development) != 64:
        raise ValueError(
            f"Parent well split freeze must contain 64 development wells: {len(development)}"
        )
    return development


def write_historical_cells(
    source: Path,
    output: Path,
    expected_ids: set[str],
    plate_map: dict[str, dict[str, str]],
    expected_development_wells: set[str],
) -> dict[str, Any]:
    fields, iterator = read_tsv(source)
    missing = sorted({"cell_id", "well", "site", "elapsed_hours", "mask_label"} - set(fields))
    if missing:
        raise ValueError(f"Parent cells lack V2 historical identity inputs: {missing}")
    overlap = sorted(set(fields) & set(HISTORICAL_IDENTITY_FIELDS))
    if overlap:
        raise ValueError(f"Parent cells already contain V2-derived identity fields: {overlap}")
    forbidden = sorted(
        field
        for field in fields
        if field != "mask_label" and FORBIDDEN_COLUMN_RE.search(field)
    )
    if forbidden:
        raise ValueError(
            f"Forbidden current/dead/RGB metadata entered V2 cells: {forbidden}"
        )
    rows = list(iterator)
    if {row["cell_id"] for row in rows} != expected_ids:
        raise ValueError("V2 cell identity derivation row universe changed")
    wells = {row["well"] for row in rows}
    if wells != expected_development_wells:
        raise ValueError(
            "V2 representative cells must contain exactly the 64 frozen development wells: "
            f"observed={len(wells)} missing={sorted(expected_development_wells-wells)[:5]} "
            f"extra={sorted(wells-expected_development_wells)[:5]}"
        )

    derived: dict[str, dict[str, str]] = {}
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        well = row["well"]
        plate = plate_map[well]
        try:
            site = int(row["site"])
            mask_label = int(row["mask_label"])
        except ValueError as error:
            raise ValueError(f"Invalid site/mask label for historical identity: {row['cell_id']}") from error
        if site < 1 or mask_label < 1:
            raise ValueError(f"Nonpositive site/mask label for historical identity: {row['cell_id']}")
        elapsed = canonical_number(row["elapsed_hours"], "elapsed_hours")
        suffix = f"site={site}|elapsed_hours={elapsed}"
        condition = (
            f"doxorubicin_nm={plate['doxorubicin_nm']}|ploidy={plate['ploidy']}|"
            f"cyclophosphamide={plate['cyclophosphamide']}"
        )
        context = f"SUM-159-NLS-{plate['ploidy']}"
        values = {
            "context_key": context,
            "ltee_cell_line": context,
            "ltee_cell_line_family": "SUM-159-NLS",
            "source_id": well,
            "suffix": suffix,
            "condition": condition,
            "feature_row_index": "",
            "segmentation_qc_cell_id": row["cell_id"],
            "segmentation_object_id": row["cell_id"],
        }
        derived[row["cell_id"]] = values
        groups.setdefault((well, suffix), []).append(row)
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: (int(row["mask_label"]), row["cell_id"]))
        for index, row in enumerate(ordered, start=1):
            derived[row["cell_id"]]["feature_row_index"] = str(index)

    output_fields = [*fields, *HISTORICAL_IDENTITY_FIELDS]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=output_fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **derived[row["cell_id"]]})
    contexts = {values["context_key"] for values in derived.values()}
    if contexts != {"SUM-159-NLS-2N", "SUM-159-NLS-4N"}:
        raise ValueError(f"V2 historical context universe differs: {sorted(contexts)}")
    return {
        "well_count": len(wells),
        "context_count": len(contexts),
        "contexts": sorted(contexts),
        "source_suffix_count": len(groups),
        "identity_fields": list(HISTORICAL_IDENTITY_FIELDS),
        "feature_row_index_policy": "one_based_mask_label_then_cell_id_within_source_id_suffix",
    }


def merge_diagnostic_cluster(cells_path: Path, clusters_path: Path) -> None:
    cell_fields, cell_iterator = read_tsv(cells_path)
    cluster_fields, cluster_iterator = read_tsv(clusters_path)
    if cluster_fields != ["cell_id", "cluster", "cluster_source"]:
        raise ValueError(f"Historical diagnostic cluster schema drift: {cluster_fields}")
    clusters = {row["cell_id"]: row for row in cluster_iterator}
    cells = list(cell_iterator)
    if len(clusters) != len(cells) or set(clusters) != {row["cell_id"] for row in cells}:
        raise ValueError("Historical diagnostic cluster universe differs from cells.tsv")
    temporary = cells_path.with_name(f".{cells_path.name}.cluster.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*cell_fields, "cluster"], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in cells:
            cluster = clusters[row["cell_id"]]["cluster"]
            try:
                int(cluster)
            except ValueError as error:
                raise ValueError(f"Diagnostic cluster is not an integer: {cluster!r}") from error
            writer.writerow({**row, "cluster": cluster})
    os.replace(temporary, cells_path)


def build_project(seed: int, cell_count: int, method_version: str = "v1", plate_map_sha256: str | None = None) -> dict[str, Any]:
    v2 = method_version == "v2"
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_id": PROJECT_ID_V2 if v2 else PROJECT_ID,
        "classes_file": "classes.tsv",
        "cells_file": "cells.tsv",
        "features_file": "features.tsv",
        "images_file": "images.tsv",
        "runs_dir": "runs",
        "projection": {
            "mode": "existing_umap" if v2 else "compute_umap",
            **(
                {
                    "coordinate_file": "historical_projection/umap.tsv",
                    "allow_subset": False,
                }
                if v2
                else {
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
                }
            ),
        },
        "annotation": {
            "title": "Reference cell-state annotation",
            "direct_class_limit": 8,
            "point_radius": 1.5,
            "boundary_tolerance": 1e-10,
        },
        "review": {
            "strata": ["context_key"] if v2 else ["well"],
            "group_column": "well",
            "quota_scope": "per_stratum" if v2 else "global",
            "quotas": {
                "default_per_class": 30 if v2 else 50,
                "unassigned": 10 if v2 else 100,
                "explicit_unassigned": 0,
            },
            "max_per_group": 8,
            "mask_padding": 6,
            "shortage_policy": "fail",
            "seed": 1 if v2 else seed,
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
            "feature_columns": list(CLASSIFIER_FEATURES_V2 if v2 else REFERENCE_FEATURES),
            "missing_policy": "median_impute",
            "constant_policy": "drop_with_manifest",
            "group_column": "source_id" if v2 else "well",
            "allow_ungrouped": False,
            "outer_folds": 5,
            "inner_folds": 5,
            "seed": 1 if v2 else seed,
            "engine": "glmnet_multinomial",
            "alpha": 1.0,
            "lambda_rule": "lambda.1se",
            "eligible_review_status": ["confirmed", "corrected"],
            **(
                {}
                if v2
                else {"minimum_confidence": 0.8}
            ),
        },
    }


def build_v2_generation_identity(
    import_manifest_path: Path,
    output_file_sha256: dict[str, str],
    builder_sha256: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    import_manifest_sha256 = sha256_file(import_manifest_path)
    output_set_sha256 = sha256_named_hashes(output_file_sha256)
    generation_id = sha256_text_parts(
        (
            GENERATION_IDENTITY_SCHEMA_VERSION_V2,
            import_manifest_sha256,
            output_set_sha256,
            builder_sha256,
            adapter_sha256,
        )
    )
    return {
        "schema_version": GENERATION_IDENTITY_SCHEMA_VERSION_V2,
        "status": "COMPLETE",
        "method_version": "v2",
        "project_id": PROJECT_ID_V2,
        "parent_import_manifest_file": IMPORT_MANIFEST_FILENAME,
        "parent_import_manifest_sha256": import_manifest_sha256,
        "declared_output_file_count": len(output_file_sha256),
        "declared_output_file_set_sha256": output_set_sha256,
        "project_builder_sha256": builder_sha256,
        "historical_projection_adapter_sha256": adapter_sha256,
        "generation_id": generation_id,
    }


def build_historical_generation_identity(
    manifest_path: Path,
    output_file_sha256: dict[str, str],
    adapter_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": HISTORICAL_GENERATION_IDENTITY_SCHEMA_VERSION_V2,
        "status": "COMPLETE",
        "historical_projection_manifest_file": (
            "historical_projection_manifest.json"
        ),
        "historical_projection_manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": adapter_sha256,
        "output_file_sha256": output_file_sha256,
    }


def validate_v2_existing_project(
    output_root: Path,
    *,
    reference_root: Path,
    parent_root: Path,
    parent: dict[str, Any],
    parent_project: Path,
    parent_manifest_path: Path,
    parent_manifest: dict[str, Any],
    parent_umap_manifest_path: Path,
    parent_umap: dict[str, Any],
    parent_hashes: dict[str, str],
    parent_support_paths: dict[str, Path],
    row_lock: dict[str, Any],
    umap_row_lock: dict[str, str],
    classes_path: Path,
    feature_config_path: Path,
    feature_config: dict[str, Any],
    config_hashes: dict[str, str],
    plate_map_manifest: dict[str, Any],
    builder_path: Path,
    builder_sha256: str,
    adapter_path: Path,
    adapter_sha256: str,
    reference_utils_path: Path,
    reference_utils_sha256: str,
    seed: int,
    expected_cell_count: int,
) -> None:
    """Fail closed unless an existing V2 directory is the same complete generation."""

    if output_root.is_symlink() or not output_root.is_dir():
        raise FileExistsError(
            f"Existing V2 reference output is not a plain directory: {output_root}"
        )
    candidates = sorted(output_root.rglob("*"), key=lambda path: str(path))
    symlinks = [path for path in candidates if path.is_symlink()]
    if symlinks:
        raise ValueError(
            f"Existing V2 reference output contains a symlink: {symlinks[0]}"
        )
    actual_files = {
        path.relative_to(output_root).as_posix(): path
        for path in candidates
        if path.is_file()
    }
    manifest_path = output_root / IMPORT_MANIFEST_FILENAME
    identity_path = output_root / GENERATION_IDENTITY_FILENAME_V2
    if not manifest_path.is_file() or not identity_path.is_file():
        raise FileExistsError(
            "Existing V2 reference output is partial: complete manifest/identity missing"
        )
    manifest = load_mapping(manifest_path, "Existing V2 parent import manifest")
    declared = manifest.get("output_file_sha256")
    if (
        not isinstance(declared, dict)
        or not declared
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for name, value in declared.items()
        )
    ):
        raise ValueError("Existing V2 import manifest has an invalid output hash map")
    expected_files = {
        *declared,
        IMPORT_MANIFEST_FILENAME,
        GENERATION_IDENTITY_FILENAME_V2,
    }
    if set(actual_files) != expected_files:
        raise ValueError(
            "Existing V2 output artifact set drifted: "
            f"missing={sorted(expected_files-set(actual_files))} "
            f"extra={sorted(set(actual_files)-expected_files)}"
        )
    expected_directories = {"runs"}
    for name in expected_files:
        for relative_parent in Path(name).parents:
            if relative_parent == Path("."):
                break
            expected_directories.add(relative_parent.as_posix())
    actual_directories = {
        path.relative_to(output_root).as_posix()
        for path in candidates
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ValueError(
            "Existing V2 output directory set drifted: "
            f"missing={sorted(expected_directories-actual_directories)} "
            f"extra={sorted(actual_directories-expected_directories)}"
        )
    for name, expected_sha256 in declared.items():
        if sha256_file(actual_files[name]) != expected_sha256:
            raise ValueError(f"Existing V2 output artifact hash drifted: {name}")

    identity = load_mapping(identity_path, "Existing V2 generation identity")
    expected_identity = build_v2_generation_identity(
        manifest_path, declared, builder_sha256, adapter_sha256
    )
    if identity != expected_identity:
        raise ValueError("Existing V2 generation identity or manifest drifted")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION_V2
        or manifest.get("status") != "COMPLETE"
        or manifest.get("method_version") != "v2"
        or manifest.get("project_id") != PROJECT_ID_V2
        or manifest.get("phase") != "independent_reference_cell_state_project"
    ):
        raise ValueError("Existing V2 import manifest is not a complete V2 generation")

    parent_receipt = manifest.get("parent")
    if not isinstance(parent_receipt, dict):
        raise ValueError("Existing V2 import manifest lacks parent identity")
    expected_parent_identity = {
        "shadow_root": str(parent_root),
        "project_id": parent.get("project_id"),
        "project": str(parent_project),
        "projection_input_manifest": str(parent_manifest_path),
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
        "input_file_sha256": parent_hashes,
    }
    for key, expected in expected_parent_identity.items():
        if parent_receipt.get(key) != expected:
            raise ValueError(f"Existing V2 parent input identity drifted: {key}")
    for role in ("well_split_freeze", "condition_split_freeze", "adapter_manifest"):
        observed = parent_receipt.get(role)
        frozen_relative = "parent_provenance/" + parent_support_paths[role].name
        frozen_path = output_root / frozen_relative
        expected = {
            "source_path": str(parent_support_paths[role]),
            "source_sha256": parent_hashes[role],
            "frozen_copy": frozen_relative,
            "frozen_copy_sha256": sha256_file(frozen_path),
        }
        if observed != expected or sha256_file(frozen_path) != parent_hashes[role]:
            raise ValueError(f"Existing V2 frozen parent provenance drifted: {role}")

    expected_row_lock = {
        "cell_count": row_lock["cell_count"],
        "heldout_cell_count": 0,
        "well_count": row_lock["well_count"],
        "image_count": len(row_lock["image_ids"]),
        "image_manifest_row_count": 2 * len(row_lock["image_ids"]),
        "cell_id_sha256": row_lock["cell_id_sha256"],
        "ordered_cell_id_sha256": row_lock["ordered_cell_id_sha256"],
        "ordered_cell_feature_sha256": row_lock["ordered_cell_feature_sha256"],
        **umap_row_lock,
        "parent_cells_and_umap_cell_id_sets_equal": True,
        "cells_byte_identical_to_parent": False,
    }
    if manifest.get("row_lock") != expected_row_lock:
        raise ValueError("Existing V2 row identity drifted")
    expected_config_inputs = {
        "classes_file": str(classes_path),
        "classes_file_sha256": config_hashes["classes_file"],
        "feature_config": str(feature_config_path),
        "feature_config_sha256": config_hashes["feature_config"],
        "feature_config_schema_version": feature_config["schema_version"],
    }
    if manifest.get("config_inputs") != expected_config_inputs:
        raise ValueError("Existing V2 config input identity drifted")
    expected_implementation = {
        "project_builder": {
            "path": str(builder_path),
            "sha256": builder_sha256,
        },
        "historical_projection_adapter": {
            "path": str(adapter_path),
            "sha256": adapter_sha256,
        },
    }
    if manifest.get("implementation") != expected_implementation:
        raise ValueError("Existing V2 implementation identity drifted")
    identity_mapping = manifest.get("identity_mapping")
    if not isinstance(identity_mapping, dict) or identity_mapping.get(
        "plate_map"
    ) != plate_map_manifest:
        raise ValueError("Existing V2 plate-map identity drifted")
    if (
        identity_mapping.get("well_count") != 64
        or identity_mapping.get("context_count") != 2
        or identity_mapping.get("contexts")
        != ["SUM-159-NLS-2N", "SUM-159-NLS-4N"]
    ):
        raise ValueError("Existing V2 historical cell identity drifted")
    expected_write_boundary = {
        "reference_shadow_root": str(reference_root),
        "parent_inputs_read_only": True,
        "current_classifier_modified": False,
        "existing_outputs_replaced": False,
    }
    if manifest.get("write_boundary") != expected_write_boundary:
        raise ValueError("Existing V2 write-boundary identity drifted")

    project_path = output_root / "project.yml"
    if load_mapping(project_path, "Existing V2 project") != build_project(
        seed, expected_cell_count, "v2", plate_map_manifest["sha256"]
    ):
        raise ValueError("Existing V2 project contract drifted")
    validate_classes(output_root / "classes.tsv", "v2")
    output_row_lock = validate_cell_feature_rows(
        output_root / "cells.tsv",
        output_root / "features.tsv",
        expected_cell_count,
        CLASSIFIER_FEATURES_V2,
        reject_forbidden=True,
    )
    for field in (
        "cell_count",
        "cell_id_sha256",
        "ordered_cell_id_sha256",
        "ordered_cell_feature_sha256",
        "well_count",
    ):
        if output_row_lock[field] != row_lock[field]:
            raise ValueError(f"Existing V2 output row lock drifted: {field}")

    projection_root = output_root / "historical_projection"
    projection_manifest_path = projection_root / "historical_projection_manifest.json"
    projection_manifest = load_mapping(
        projection_manifest_path, "Existing historical projection manifest"
    )
    if (
        projection_manifest.get("schema_version")
        != "reference_cell_state_historical_projection_v2"
        or projection_manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("Existing historical projection is not complete V2")
    expected_projection_inputs = {
        "project": (project_path, sha256_file(project_path)),
        "cells": (
            projection_root / "historical_projection_input_cells.tsv",
            sha256_file(projection_root / "historical_projection_input_cells.tsv"),
        ),
        "features": (
            output_root / "features.tsv",
            sha256_file(output_root / "features.tsv"),
        ),
        "feature_config": (feature_config_path, config_hashes["feature_config"]),
        "reference_utils": (reference_utils_path, reference_utils_sha256),
        "implementation": (adapter_path, adapter_sha256),
    }
    projection_inputs = projection_manifest.get("inputs")
    if not isinstance(projection_inputs, dict):
        raise ValueError("Existing historical projection lacks input identity")
    for role, (path, expected_sha256) in expected_projection_inputs.items():
        observed = projection_inputs.get(role)
        if observed != {"path": str(path), "sha256": expected_sha256}:
            raise ValueError(
                f"Existing historical projection input/implementation drifted: {role}"
            )
    expected_projection_outputs = {
        "umap.tsv",
        "diagnostic_clusters.tsv",
        "fixed_dbscan_clusters.tsv",
        "dbscan_scan.tsv",
        "feature_transform_manifest.tsv",
        "pca_variance.tsv",
        "preprocessed_projection_features.tsv",
        "historical_representatives.tsv",
    }
    projection_outputs = projection_manifest.get("output_file_sha256")
    if (
        not isinstance(projection_outputs, dict)
        or set(projection_outputs) != expected_projection_outputs
    ):
        raise ValueError("Existing historical projection artifact set drifted")
    actual_projection_files = {
        path.name
        for path in projection_root.iterdir()
        if path.is_file()
    }
    if actual_projection_files != {
        *expected_projection_outputs,
        "historical_projection_manifest.json",
        "historical_projection_input_cells.tsv",
        HISTORICAL_GENERATION_IDENTITY_FILENAME_V2,
    }:
        raise ValueError("Existing historical projection directory is partial or has extras")
    for name, expected_sha256 in projection_outputs.items():
        if sha256_file(projection_root / name) != expected_sha256:
            raise ValueError(f"Existing historical projection artifact drifted: {name}")
    selection = projection_manifest.get("representative_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("output_file") != "historical_representatives.tsv"
        or selection.get("output_sha256")
        != projection_outputs["historical_representatives.tsv"]
    ):
        raise ValueError("Existing historical representative manifest drifted")
    projection_identity = load_mapping(
        projection_root / HISTORICAL_GENERATION_IDENTITY_FILENAME_V2,
        "Existing historical projection generation identity",
    )
    if projection_identity != build_historical_generation_identity(
        projection_manifest_path, projection_outputs, adapter_sha256
    ):
        raise ValueError(
            "Existing historical projection generation identity or manifest drifted"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    builder_path = Path(__file__).resolve()
    builder_sha256 = sha256_file(builder_path)
    v2 = args.method_version == "v2"
    if args.seed is None:
        args.seed = 42 if v2 else 20260812
    if args.expected_cell_count < 3:
        raise ValueError("Reference project requires at least three expected cells")
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if v2 and args.seed != 42:
        raise ValueError("V2 historical projection seed is frozen at 42")
    if args.dbscan_cores < 1:
        raise ValueError("--dbscan-cores must be positive")
    if v2 and args.reference_snapshot_root is None:
        raise ValueError("V2 requires --reference-snapshot-root pointing to reference/ltee-source")

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
    output_name = "representative_umap_v2" if v2 else "representative_umap"
    candidate_output = reference_candidate / "projection_input" / output_name
    if os.path.lexists(candidate_output) and not v2:
        raise FileExistsError(f"Refusing to replace existing reference project: {candidate_output}")

    selected_classes = args.classes_file or default_config_path(
        "reference_cell_state_classes_v2.tsv"
        if v2
        else "reference_cell_state_classes_v1.tsv"
    )
    classes_input = Path(os.path.abspath(str(selected_classes.expanduser())))
    selected_feature_config = args.feature_config or default_config_path(
        "reference_cell_state_features_v2.json"
        if v2
        else "reference_cell_state_features_v1.json"
    )
    feature_config_input = Path(
        os.path.abspath(str(selected_feature_config.expanduser()))
    )
    if v2 and feature_config_input.resolve() != default_config_path(
        "reference_cell_state_features_v2.json"
    ).resolve():
        raise ValueError("V2 must use the single frozen reference_cell_state_features_v2.json")
    if v2 and classes_input.resolve() != default_config_path(
        "reference_cell_state_classes_v2.tsv"
    ).resolve():
        raise ValueError("V2 must use the frozen reference_cell_state_classes_v2.tsv")
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
    validate_classes(classes_path, args.method_version)
    feature_config = load_feature_config(feature_config_path, args.method_version)

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
    if parent.get("project_id") != "broad_phenotype_development":
        raise ValueError(
            "Reference import parent must be the broad representative project; "
            "V1/V2 reference projects are not valid parents"
        )
    parent_projection_features = projection.get("feature_columns")
    required_parent_features = CLASSIFIER_FEATURES_V2 if v2 else REFERENCE_FEATURES
    if not isinstance(parent_projection_features, list) or not set(required_parent_features).issubset(
        parent_projection_features
    ):
        raise ValueError(
            "Parent broad projection allowlist does not contain all required reference features"
        )

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
    if v2:
        for path_field, hash_field in (
            ("well_split_freeze", "well_split_freeze_sha256"),
            ("condition_split_freeze", "condition_split_freeze_sha256"),
            ("adapter_manifest", "adapter_manifest_sha256"),
        ):
            declared = parent_manifest.get(path_field)
            expected = parent_manifest.get(hash_field)
            if not isinstance(declared, str) or not isinstance(expected, str):
                raise ValueError(
                    f"V2 parent projection manifest lacks {path_field}/{hash_field}"
                )
            artifact = require_parent_path(
                Path(declared), parent_root, f"Parent {path_field}"
            )
            if sha256_file(artifact) != expected:
                raise ValueError(f"V2 parent projection manifest hash mismatch: {path_field}")
            parent_support_paths[path_field] = artifact
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
        assets["cells_file"],
        assets["features_file"],
        args.expected_cell_count,
        required_parent_features,
        reject_forbidden=v2,
    )
    umap_row_lock = validate_parent_umap_rows(
        parent_umap_artifacts["umap_artifact:umap"],
        row_lock["cell_ids"],
        args.expected_cell_count,
    )

    plate_map: dict[str, dict[str, str]] | None = None
    plate_map_manifest: dict[str, Any] | None = None
    development_wells: set[str] | None = None
    snapshot_root: Path | None = None
    adapter_script: Path | None = None
    adapter_sha256: str | None = None
    reference_utils_path: Path | None = None
    reference_utils_sha256: str | None = None
    if v2:
        plate_map_input = Path(os.path.abspath(str(args.plate_map.expanduser())))
        if plate_map_input.is_symlink():
            raise ValueError(f"V2 plate map must not be a symlink: {plate_map_input}")
        plate_map_path = plate_map_input.resolve()
        plate_map, plate_map_manifest = load_plate_map(plate_map_path)
        development_wells = validate_well_split_freeze(
            parent_support_paths["well_split_freeze"], plate_map
        )
        assert args.reference_snapshot_root is not None
        snapshot_root = args.reference_snapshot_root.expanduser().resolve()
        reference_source_file = feature_config["historical_projection"].get(
            "reference_source_file"
        )
        if not isinstance(reference_source_file, str) or not reference_source_file:
            raise ValueError("V2 feature config lacks historical reference_source_file")
        reference_utils_path = (snapshot_root / reference_source_file).resolve()
        if not reference_utils_path.is_file():
            raise FileNotFoundError(
                "--reference-snapshot-root must point directly to the pinned "
                f"reference/ltee-source directory: {snapshot_root}"
            )
        reference_utils_sha256 = sha256_file(reference_utils_path)
        configured_reference_utils_sha256 = feature_config[
            "historical_projection"
        ].get("reference_source_sha256")
        if reference_utils_sha256 != configured_reference_utils_sha256:
            raise ValueError(
                "Pinned historical reference Utils.R hash differs from V2 config: "
                f"expected={configured_reference_utils_sha256} "
                f"observed={reference_utils_sha256}"
            )
        adapter_script = args.historical_projection_script.expanduser().resolve()
        if not adapter_script.is_file():
            raise FileNotFoundError(adapter_script)
        adapter_sha256 = sha256_file(adapter_script)

    # Do not create even an empty reference directory until all parent inputs,
    # configs, and the complete row lock have passed validation.
    reference_root = prepare_reference_root(args.reference_shadow_root)
    output_parent = require_plain_descendant(
        reference_root / "projection_input", reference_root, "Reference projection root"
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = require_plain_descendant(
        output_parent / output_name, reference_root, "Reference project output"
    )
    if os.path.lexists(output_root):
        if not v2:
            raise FileExistsError(
                f"Refusing to replace existing reference project: {output_root}"
            )
        assert plate_map_manifest is not None
        assert adapter_script is not None and adapter_sha256 is not None
        assert reference_utils_path is not None and reference_utils_sha256 is not None
        validate_v2_existing_project(
            output_root,
            reference_root=reference_root,
            parent_root=parent_root,
            parent=parent,
            parent_project=parent_project,
            parent_manifest_path=manifest_path,
            parent_manifest=parent_manifest,
            parent_umap_manifest_path=parent_umap_manifest_path,
            parent_umap=parent_umap,
            parent_hashes=parent_hashes_before,
            parent_support_paths=parent_support_paths,
            row_lock=row_lock,
            umap_row_lock=umap_row_lock,
            classes_path=classes_path,
            feature_config_path=feature_config_path,
            feature_config=feature_config,
            config_hashes=config_hashes_before,
            plate_map_manifest=plate_map_manifest,
            builder_path=builder_path,
            builder_sha256=builder_sha256,
            adapter_path=adapter_script,
            adapter_sha256=adapter_sha256,
            reference_utils_path=reference_utils_path,
            reference_utils_sha256=reference_utils_sha256,
            seed=args.seed,
            expected_cell_count=args.expected_cell_count,
        )
        print(f"reference_project={output_root / 'project.yml'}")
        print(f"parent_import_manifest={output_root / IMPORT_MANIFEST_FILENAME}")
        print(f"reference_cell_count={args.expected_cell_count}")
        print("method_version=v2")
        print("parent_umap_coordinates_imported=0")
        print(f"historical_projection={output_root / 'historical_projection'}")
        print("diagnostic_cluster_role=metadata_only")
        print("generation_status=verified_reuse")
        print("current_classifier_modified=0")
        return 0

    staging = output_parent / f".{output_name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        selected_ids: set[str] = set()
        _, selected_rows = read_tsv(assets["cells_file"])
        for row in selected_rows:
            selected_ids.add(row["cell_id"])
        if len(selected_ids) != args.expected_cell_count:
            raise RuntimeError("Staged cells table row universe changed during copy")
        historical_identity_manifest: dict[str, Any] | None = None
        if v2:
            assert plate_map is not None and plate_map_manifest is not None
            assert development_wells is not None
            historical_identity_manifest = write_historical_cells(
                assets["cells_file"],
                staging / "cells.tsv",
                selected_ids,
                plate_map,
                development_wells,
            )
            provenance_dir = staging / "parent_provenance"
            provenance_dir.mkdir()
            for role in (
                "well_split_freeze",
                "condition_split_freeze",
                "adapter_manifest",
            ):
                source = parent_support_paths[role]
                shutil.copyfile(source, provenance_dir / source.name)
        else:
            shutil.copyfile(assets["cells_file"], staging / "cells.tsv")
        write_reference_features(
            assets["features_file"],
            staging / "features.tsv",
            selected_ids,
            required_parent_features,
        )
        image_row_count = write_review_images(
            assets["images_file"], staging / "images.tsv", row_lock["image_ids"]
        )
        shutil.copyfile(classes_path, staging / "classes.tsv")
        (staging / "runs").mkdir()
        project = build_project(
            args.seed,
            args.expected_cell_count,
            args.method_version,
            plate_map_manifest["sha256"] if plate_map_manifest else None,
        )
        (staging / "project.yml").write_text(
            json.dumps(project, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

        historical_projection_manifest: dict[str, Any] | None = None
        if v2:
            assert snapshot_root is not None
            assert adapter_script is not None and adapter_sha256 is not None
            command = [
                args.rscript,
                str(adapter_script),
                "--project",
                str(staging / "project.yml"),
                "--reference-snapshot-root",
                str(snapshot_root),
                "--output-dir",
                str(staging / "historical_projection"),
                "--feature-config",
                str(feature_config_path),
                "--dbscan-cores",
                str(args.dbscan_cores),
            ]
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Historical projection adapter failed:\n" + completed.stdout
                )
            historical_projection_input_cells = (
                staging
                / "historical_projection"
                / "historical_projection_input_cells.tsv"
            )
            shutil.copyfile(staging / "cells.tsv", historical_projection_input_cells)
            merge_diagnostic_cluster(
                staging / "cells.tsv",
                staging / "historical_projection" / "diagnostic_clusters.tsv",
            )
            historical_projection_manifest = load_mapping(
                staging
                / "historical_projection"
                / "historical_projection_manifest.json",
                "Historical projection manifest",
            )
            if (
                historical_projection_manifest.get("schema_version")
                != "reference_cell_state_historical_projection_v2"
                or historical_projection_manifest.get("status") != "COMPLETE"
            ):
                raise RuntimeError("Historical projection adapter did not complete V2")
            implementation_input = historical_projection_manifest.get(
                "inputs", {}
            ).get("implementation", {})
            if (
                implementation_input.get("path") != str(adapter_script)
                or implementation_input.get("sha256") != adapter_sha256
            ):
                raise RuntimeError(
                    "Historical projection manifest does not bind the invoked adapter"
                )
            projection_inputs = historical_projection_manifest.get("inputs")
            if not isinstance(projection_inputs, dict):
                raise RuntimeError("Historical projection manifest lacks input identity")
            for role, staged_path, final_path in (
                (
                    "project",
                    staging / "project.yml",
                    output_root / "project.yml",
                ),
                (
                    "cells",
                    historical_projection_input_cells,
                    output_root
                    / "historical_projection"
                    / "historical_projection_input_cells.tsv",
                ),
                (
                    "features",
                    staging / "features.tsv",
                    output_root / "features.tsv",
                ),
            ):
                observed = projection_inputs.get(role)
                if not isinstance(observed, dict):
                    raise RuntimeError(
                        f"Historical projection manifest lacks {role} input identity"
                    )
                observed["path"] = str(final_path)
                if observed.get("sha256") != sha256_file(staged_path):
                    raise RuntimeError(
                        f"Historical projection {role} input changed before relocation"
                    )
            historical_projection_manifest_path = (
                staging
                / "historical_projection"
                / "historical_projection_manifest.json"
            )
            historical_projection_manifest_path.write_text(
                json.dumps(
                    historical_projection_manifest, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            historical_projection_output_hashes = (
                historical_projection_manifest.get("output_file_sha256")
            )
            if not isinstance(historical_projection_output_hashes, dict):
                raise RuntimeError(
                    "Historical projection manifest lacks output hash identity"
                )
            historical_generation_identity = build_historical_generation_identity(
                historical_projection_manifest_path,
                historical_projection_output_hashes,
                adapter_sha256,
            )
            (
                staging
                / "historical_projection"
                / HISTORICAL_GENERATION_IDENTITY_FILENAME_V2
            ).write_text(
                json.dumps(
                    historical_generation_identity, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )

        parent_hashes_after = {role: sha256_file(path) for role, path in parent_files.items()}
        if parent_hashes_after != parent_hashes_before:
            changed = sorted(
                role
                for role in parent_hashes_before
                if parent_hashes_before[role] != parent_hashes_after[role]
            )
            raise RuntimeError(f"Parent project changed during reference import: {changed}")
        if not v2 and sha256_file(staging / "cells.tsv") != parent_hashes_before["cells"]:
            raise RuntimeError("Reference cells table is not an exact byte copy of its parent")
        config_hashes_after = {
            "classes_file": sha256_file(classes_path),
            "feature_config": sha256_file(feature_config_path),
        }
        if config_hashes_after != config_hashes_before:
            raise RuntimeError("Reference class/feature config changed during import")
        if sha256_file(builder_path) != builder_sha256:
            raise RuntimeError("Reference project builder changed during import")
        if v2 and (
            adapter_script is None
            or adapter_sha256 is None
            or sha256_file(adapter_script) != adapter_sha256
        ):
            raise RuntimeError("Historical projection adapter changed during import")

        output_assets = {
            path.relative_to(staging).as_posix(): path
            for path in sorted(staging.rglob("*"), key=lambda value: str(value))
            if path.is_file()
        }
        import_manifest = {
            "schema_version": SCHEMA_VERSION_V2 if v2 else SCHEMA_VERSION,
            **(
                {"status": "COMPLETE", "method_version": "v2"}
                if v2
                else {}
            ),
            "project_id": PROJECT_ID_V2 if v2 else PROJECT_ID,
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
                **(
                    {
                        role: {
                            "source_path": str(parent_support_paths[role]),
                            "source_sha256": parent_hashes_before[role],
                            "frozen_copy": (
                                "parent_provenance/"
                                + parent_support_paths[role].name
                            ),
                            "frozen_copy_sha256": sha256_file(
                                staging
                                / "parent_provenance"
                                / parent_support_paths[role].name
                            ),
                        }
                        for role in (
                            "well_split_freeze",
                            "condition_split_freeze",
                            "adapter_manifest",
                        )
                    }
                    if v2
                    else {}
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
                "cells_byte_identical_to_parent": not v2,
            },
            "reference_contract": {
                **(
                    {
                        "class_ids_order": list(REFERENCE_CLASS_IDS_V2),
                        "class_order_semantics": "historical_annotation_source_order",
                        "class_id_mapping": {
                            class_id: class_id for class_id in REFERENCE_CLASS_IDS_V2
                        },
                    }
                    if v2 else {
                        "class_ids_priority_order": list(REFERENCE_CLASS_IDS_V1),
                    }
                ),
                "unassigned_state": "built_in_nontraining_review_state",
                "feature_columns": list(required_parent_features),
                "projection_feature_columns": list(REFERENCE_FEATURES),
                "classifier_feature_columns": list(required_parent_features),
                "projection_mode": "existing_umap" if v2 else "compute_umap",
                "parent_umap_coordinates_imported": False,
                "projection_transform": feature_config["projection_transform"],
                "review_channel_ids": list(REVIEW_CHANNEL_IDS),
                "nuclei_policy": "review_display_only",
                "viability_evidence_policy": "not_read",
                "classifier": CLASSIFIER_CONTRACT_V2 if v2 else CLASSIFIER_CONTRACT,
                **(
                    {
                        "diagnostic_cluster_role": "metadata_only",
                        "diagnostic_cluster_authority": "optimize_dbscan_v4",
                        "fixed_dbscan_role": "audit_only_not_authoritative",
                        "unit_adaptation": {
                            "physical_unit_claim": "not_asserted",
                            "pixel_calibration_status": "unavailable_not_recoverable",
                            "pixel_calibration_evidence": feature_config[
                                "historical_projection"
                            ]["pixel_calibration_evidence"],
                            "historical_name_mapping_role": feature_config[
                                "historical_projection"
                            ]["historical_name_mapping_role"],
                            "unrecoverable_production_artifact_difference": feature_config[
                                "historical_projection"
                            ]["unrecoverable_production_artifact_difference"],
                        },
                        "historical_projection_manifest_schema_version": historical_projection_manifest.get(
                            "schema_version"
                        ),
                    }
                    if v2 and historical_projection_manifest
                    else {}
                ),
            },
            **(
                {
                    "identity_mapping": {
                        "plate_map": plate_map_manifest,
                        **(historical_identity_manifest or {}),
                        "context_interpretation": "two_ploidy_contexts_not_64_wells",
                    }
                }
                if v2
                else {}
            ),
            "config_inputs": {
                "classes_file": str(classes_path),
                "classes_file_sha256": config_hashes_before["classes_file"],
                "feature_config": str(feature_config_path),
                "feature_config_sha256": config_hashes_before["feature_config"],
                "feature_config_schema_version": feature_config["schema_version"],
            },
            **(
                {
                    "implementation": {
                        "project_builder": {
                            "path": str(builder_path),
                            "sha256": builder_sha256,
                        },
                        "historical_projection_adapter": {
                            "path": str(adapter_script),
                            "sha256": adapter_sha256,
                        },
                    }
                }
                if v2
                else {}
            ),
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
        import_manifest_path = staging / IMPORT_MANIFEST_FILENAME
        import_manifest_path.write_text(
            json.dumps(import_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if v2:
            assert adapter_sha256 is not None
            generation_identity = build_v2_generation_identity(
                import_manifest_path,
                import_manifest["output_file_sha256"],
                builder_sha256,
                adapter_sha256,
            )
            (staging / GENERATION_IDENTITY_FILENAME_V2).write_text(
                json.dumps(generation_identity, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if os.path.lexists(output_root):
            raise FileExistsError(f"Reference output appeared during staging: {output_root}")
        os.rename(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"reference_project={output_root / 'project.yml'}")
    print(f"parent_import_manifest={output_root / IMPORT_MANIFEST_FILENAME}")
    print(f"reference_cell_count={args.expected_cell_count}")
    print(f"method_version={args.method_version}")
    print("parent_umap_coordinates_imported=0")
    if v2:
        print(f"historical_projection={output_root / 'historical_projection'}")
        print("diagnostic_cluster_role=metadata_only")
        print("generation_status=created")
    print("current_classifier_modified=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
