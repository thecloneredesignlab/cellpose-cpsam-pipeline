#!/usr/bin/env python3
"""Freeze a representative-project active-learning review overlay.

This helper does not implement an active-learning selector.  It verifies the
immutable Cell Phenotype Annotator UMAP, model, prediction, and authoritative
review-history generations, then writes a queue-configured project consumed by
the pinned reference ``review-build`` command.  The scope is targeted
validation inside the development-only representative universe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "broad_phenotype_active_learning_project_v1"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
CLASSIFIER_SCHEMA_VERSION = "cell_phenotype_annotator_classifier_v1"
UMAP_SCHEMA_VERSION = "cell_phenotype_annotator_umap_v1"
REVIEW_IMPORT_SCHEMA_VERSION = "cell_phenotype_annotator_review_import_v1"
QUEUE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
MODEL_ARTIFACTS = {
    "model": "model.rds",
    "training_rows": "training_rows.tsv",
    "outer_folds": "outer_fold_assignments.tsv",
    "fold_summary": "fold_summary.tsv",
    "fold_metrics_by_class": "fold_metrics_by_class.tsv",
    "feature_preprocessor": "feature_preprocessor_manifest.tsv",
    "out_of_fold_predictions": "out_of_fold_predictions.tsv",
    "out_of_fold_probabilities": "out_of_fold_probabilities.tsv",
    "metrics_by_class": "metrics_by_class.tsv",
    "metrics_summary": "metrics_summary.tsv",
    "confusion_matrix": "confusion_matrix.tsv",
}
UMAP_ARTIFACTS = {
    "normalized_project": "normalized_project.json",
    "input_manifest": "input_manifest.tsv",
    "join_audit": "join_audit.tsv",
    "feature_report": "feature_report.tsv",
    "row_universe_audit": "row_universe_audit.tsv",
    "feature_transform_manifest": "feature_transform_manifest.tsv",
    "umap": "umap.tsv",
}
REVIEW_IMPORT_ARTIFACTS = {
    "review_submission": "review_submission.json",
    "reviewed_labels": "reviewed_labels.tsv",
}
PREDICTION_ARTIFACTS = {
    "all_cell_predictions": "all_cell_predictions.tsv",
    "all_cell_probabilities": "all_cell_probabilities.tsv",
}
LIMITATIONS = (
    "targeted_validation_only",
    "representative_development_project_only",
    "heldout_cells_excluded",
    "no_full_universe_prediction_or_review",
    "no_viability_labels_or_features",
    "no_automatic_cross_review_id_label_merge",
    "no_automatic_retraining",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--max-cells", type=int, required=True)
    parser.add_argument(
        "--uncertainty-metric",
        choices=("normalized_entropy", "least_confidence", "margin"),
        default="normalized_entropy",
    )
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--shortage-policy",
        choices=("fail", "allow_partial"),
        default="fail",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_symlink_chain(path: Path, root: Path, label: str) -> None:
    """Reject symlinks at or below ``root`` while tolerating OS path aliases.

    macOS exposes ``/var`` through the system ``/var -> /private/var`` alias.
    That alias is above a temporary shadow root and is not controlled by this
    workflow.  Locate the lexical entry that resolves to the trusted shadow
    root, then inspect only it and its descendants.  A path that reaches the
    shadow through some other symlink has no such lexical anchor and fails
    closed.
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
        raise ValueError(f"{label} must not reach shadow root through a symlink: {lexical}")
    relative = lexical.relative_to(anchor)
    candidate = anchor
    for part in (Path(), *[Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)]):
        candidate = anchor / part
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")


def require_inside(path: Path, root: Path, label: str, kind: str) -> Path:
    expanded = path.expanduser()
    reject_symlink_chain(expanded, root, label)
    resolved = expanded.resolve()
    if not is_inside(resolved, root):
        raise ValueError(f"{label} must be inside shadow root {root}: {resolved}")
    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(resolved)
    if kind == "dir" and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_project_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project.parent / path
    return path.resolve()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


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
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"TSV is empty: {path}")
    fields = [str(value) for value in reader.fieldnames]
    if any(not value for value in fields) or len(fields) != len(set(fields)):
        handle.close()
        raise ValueError(f"TSV has blank or duplicate columns: {path}")

    def rows() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {field: str(row.get(field, "") or "") for field in fields}
        finally:
            handle.close()

    return fields, rows()


def require_columns(fields: Sequence[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def artifact_identity(path: Path, shadow_root: Path, role: str) -> dict[str, Any]:
    resolved = require_inside(path, shadow_root, role, "file")
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_declared_generation(
    directory: Path,
    manifest_name: str,
    manifest: dict[str, Any],
    artifact_names: dict[str, str],
    shadow_root: Path,
    role_prefix: str,
) -> list[dict[str, Any]]:
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict) or set(declared) != set(artifact_names):
        observed = sorted(declared) if isinstance(declared, dict) else type(declared).__name__
        raise ValueError(
            f"{role_prefix} manifest artifact set mismatch: "
            f"expected={sorted(artifact_names)} observed={observed}"
        )
    identities = [
        artifact_identity(directory / manifest_name, shadow_root, f"{role_prefix}_manifest")
    ]
    for role, filename in artifact_names.items():
        identity = artifact_identity(directory / filename, shadow_root, f"{role_prefix}:{role}")
        expected = declared[role]
        if not isinstance(expected, str) or identity["sha256"] != expected:
            raise RuntimeError(
                f"{role_prefix} artifact hash mismatch: role={role} "
                f"expected={expected!r} observed={identity['sha256']}"
            )
        identity["manifest_declared_sha256"] = expected
        identities.append(identity)
    return identities


def validate_representative_project(
    project_path: Path,
    project: dict[str, Any],
    shadow_root: Path,
) -> tuple[dict[str, Path], list[str], list[str], dict[str, dict[str, str]]]:
    project_id = project.get("project_id")
    if not isinstance(project_id, str) or not project_id.endswith("_development"):
        raise ValueError(
            "Active learning requires the physical development representative project"
        )
    paths: dict[str, Path] = {}
    for field in ("classes_file", "cells_file", "features_file", "images_file"):
        value = project.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Project requires {field}")
        paths[field] = require_inside(
            resolve_project_path(project_path, value), shadow_root, field, "file"
        )
    runs_value = project.get("runs_dir", "runs")
    if not isinstance(runs_value, str) or not runs_value:
        raise ValueError("Project runs_dir must be a nonblank string")
    paths["runs_dir"] = require_inside(
        resolve_project_path(project_path, runs_value), shadow_root, "runs_dir", "dir"
    )

    projection = project.get("projection")
    classifier = project.get("classifier")
    review = project.get("review")
    if not isinstance(projection, dict) or not isinstance(classifier, dict) or not isinstance(review, dict):
        raise ValueError("Representative project requires projection, classifier, and review sections")
    projection_features = projection.get("feature_columns")
    classifier_features = classifier.get("feature_columns")
    if (
        not isinstance(projection_features, list)
        or not projection_features
        or projection_features != classifier_features
        or len(projection_features) != len(set(projection_features))
    ):
        raise ValueError("Projection/classifier feature identity is not exactly locked")
    if classifier.get("group_column") != "well" or classifier.get("allow_ungrouped") is not False:
        raise ValueError("Active learning requires the grouped representative classifier contract")

    class_fields, class_rows = read_tsv(paths["classes_file"])
    require_columns(class_fields, ("class_id", "trainable"), "Classes")
    trainable_classes: list[str] = []
    for row in class_rows:
        if row["trainable"].lower() == "true":
            trainable_classes.append(row["class_id"])
        elif row["trainable"].lower() != "false":
            raise ValueError(f"Invalid trainable class flag: {row['trainable']}")
    if len(trainable_classes) < 2 or len(trainable_classes) != len(set(trainable_classes)):
        raise ValueError("Active learning requires at least two unique trainable classes")

    feature_fields, feature_rows = read_tsv(paths["features_file"])
    require_columns(feature_fields, ("cell_id", *projection_features), "Features")
    feature_ids: set[str] = set()
    for row in feature_rows:
        cell_id = row["cell_id"]
        if not cell_id or cell_id in feature_ids:
            raise ValueError(f"Features contain a blank or duplicate cell_id: {cell_id!r}")
        feature_ids.add(cell_id)

    cell_fields, cell_rows = read_tsv(paths["cells_file"])
    require_columns(cell_fields, ("cell_id", "image_id", "well", "split"), "Cells")
    cells: dict[str, dict[str, str]] = {}
    for row in cell_rows:
        cell_id = row["cell_id"]
        if not cell_id or cell_id in cells:
            raise ValueError(f"Cells contain a blank or duplicate cell_id: {cell_id!r}")
        if row["split"] != "development":
            raise ValueError(f"Heldout/nondevelopment cell entered representative project: {cell_id}")
        cells[cell_id] = row
    if len(cells) < 3 or set(cells) != feature_ids:
        raise ValueError(
            f"Representative cell/feature universe mismatch: cells={len(cells)} features={len(feature_ids)}"
        )
    return paths, list(projection_features), trainable_classes, cells


def validate_model_and_prediction(
    prediction_dir: Path,
    project: dict[str, Any],
    project_features: Sequence[str],
    trainable_classes: Sequence[str],
    cells: dict[str, dict[str, str]],
    shadow_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prediction_dir = require_inside(prediction_dir, shadow_root, "prediction_dir", "dir")
    if prediction_dir.parent.name != "predictions" or not prediction_dir.name.startswith("prediction_"):
        raise ValueError(
            "prediction_dir must be a canonical <model_dir>/predictions/prediction_<id> generation"
        )
    model_dir = require_inside(
        prediction_dir.parent.parent, shadow_root, "prediction model_dir", "dir"
    )
    model_manifest_path = model_dir / "model_manifest.json"
    model_manifest = load_json(model_manifest_path, "Model manifest")
    if model_manifest.get("schema_version") != CLASSIFIER_SCHEMA_VERSION:
        raise ValueError("Unsupported model manifest schema_version")
    identities = validate_declared_generation(
        model_dir,
        "model_manifest.json",
        model_manifest,
        MODEL_ARTIFACTS,
        shadow_root,
        "model",
    )
    if model_manifest.get("project_id") != project.get("project_id"):
        raise ValueError("Model project_id does not match representative project")
    model_features = model_manifest.get("feature_columns")
    model_classes = model_manifest.get("class_ids")
    if model_features != list(project_features):
        raise ValueError("Model feature_columns do not exactly match representative classifier")
    if not isinstance(model_classes, list) or set(model_classes) != set(trainable_classes):
        raise ValueError("Model class universe does not exactly match trainable representative classes")
    classifier = project.get("classifier")
    model_classifier = model_manifest.get("classifier_config")
    if not isinstance(model_classifier, dict) or model_classifier.get("feature_columns") != classifier.get("feature_columns"):
        raise ValueError("Model classifier feature identity is absent or inconsistent")

    prediction_manifest_path = prediction_dir / "prediction_manifest.json"
    prediction_manifest = load_json(prediction_manifest_path, "Prediction manifest")
    if prediction_manifest.get("schema_version") != CLASSIFIER_SCHEMA_VERSION:
        raise ValueError("Unsupported prediction manifest schema_version")
    identities.extend(
        validate_declared_generation(
            prediction_dir,
            "prediction_manifest.json",
            prediction_manifest,
            PREDICTION_ARTIFACTS,
            shadow_root,
            "prediction",
        )
    )
    expected_scalar = {
        "prediction_id": prediction_dir.name,
        "model_id": model_manifest.get("model_id"),
        "model_sha256": sha256_file(model_dir / "model.rds"),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "project_id": project.get("project_id"),
        "run_id": model_manifest.get("run_id"),
        "classifier_config_sha256": model_manifest.get("classifier_config_sha256"),
    }
    mismatches = {
        field: {"expected": value, "observed": prediction_manifest.get(field)}
        for field, value in expected_scalar.items()
        if prediction_manifest.get(field) != value
    }
    if mismatches:
        raise ValueError("Prediction/model/project identity mismatch: " + json.dumps(mismatches, sort_keys=True))

    prediction_fields, prediction_rows = read_tsv(prediction_dir / "all_cell_predictions.tsv")
    require_columns(
        prediction_fields,
        ("model_id", "cell_id", "predicted_class_id", "prediction_status"),
        "Predictions",
    )
    predictions: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        cell_id = row["cell_id"]
        if cell_id in predictions or row["model_id"] != model_manifest.get("model_id"):
            raise ValueError(f"Prediction identity is duplicate or inconsistent: {cell_id}")
        status = row["prediction_status"]
        if status not in {"ok", "unavailable_missing_features"}:
            raise ValueError(f"Unsupported prediction_status for {cell_id}: {status}")
        if status == "ok" and row["predicted_class_id"] not in model_classes:
            raise ValueError(f"Available prediction has invalid class: {cell_id}")
        if status != "ok" and row["predicted_class_id"]:
            raise ValueError(f"Unavailable prediction carries a class: {cell_id}")
        predictions[cell_id] = row
    if set(predictions) != set(cells):
        raise ValueError(
            "Canonical representative prediction must cover the exact configured cell universe"
        )

    probability_fields, probability_rows = read_tsv(
        prediction_dir / "all_cell_probabilities.tsv"
    )
    require_columns(
        probability_fields,
        ("model_id", "cell_id", "class_id", "probability", "prediction_status"),
        "Probabilities",
    )
    probabilities: dict[str, dict[str, float | None]] = {cell_id: {} for cell_id in cells}
    for row in probability_rows:
        cell_id, class_id = row["cell_id"], row["class_id"]
        if cell_id not in cells or class_id not in model_classes or class_id in probabilities[cell_id]:
            raise ValueError(f"Probability key is extra or duplicate: {cell_id}/{class_id}")
        prediction = predictions[cell_id]
        if row["model_id"] != model_manifest.get("model_id") or row["prediction_status"] != prediction["prediction_status"]:
            raise ValueError(f"Probability identity/status mismatch: {cell_id}/{class_id}")
        if prediction["prediction_status"] == "ok":
            try:
                probability = float(row["probability"])
            except ValueError as error:
                raise ValueError(f"Invalid probability: {cell_id}/{class_id}") from error
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Probability is outside [0,1]: {cell_id}/{class_id}")
            probabilities[cell_id][class_id] = probability
        else:
            if row["probability"].strip().lower() not in {"", "na", "nan"}:
                raise ValueError(f"Unavailable prediction has a probability: {cell_id}/{class_id}")
            probabilities[cell_id][class_id] = None
    for cell_id, values in probabilities.items():
        if set(values) != set(model_classes):
            raise ValueError(f"Probability class universe is incomplete: {cell_id}")
        if predictions[cell_id]["prediction_status"] == "ok":
            finite_values = {key: float(value) for key, value in values.items() if value is not None}
            if abs(sum(finite_values.values()) - 1.0) > 1e-6:
                raise ValueError(f"Probabilities do not sum to one: {cell_id}")
            predicted = max(model_classes, key=lambda class_id: finite_values[class_id])
            if predicted != predictions[cell_id]["predicted_class_id"]:
                raise ValueError(f"Prediction is not the probability argmax: {cell_id}")
    return model_manifest, prediction_manifest, identities


def validate_review_history(
    reviewed_labels: Path,
    project: dict[str, Any],
    model_manifest: dict[str, Any],
    cells: dict[str, dict[str, str]],
    shadow_root: Path,
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    reviewed_labels = require_inside(reviewed_labels, shadow_root, "reviewed_labels", "file")
    accepted_dir = reviewed_labels.parent
    if reviewed_labels.name != "reviewed_labels.tsv":
        raise ValueError("reviewed_labels must be the authoritative reviewed_labels.tsv")
    manifest_path = accepted_dir / "review_import_manifest.json"
    manifest = load_json(manifest_path, "Review import manifest")
    if manifest.get("schema_version") != REVIEW_IMPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported review import manifest schema_version")
    identities = validate_declared_generation(
        accepted_dir,
        "review_import_manifest.json",
        manifest,
        REVIEW_IMPORT_ARTIFACTS,
        shadow_root,
        "review_history",
    )
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Review import manifest lacks identity")
    expected = {
        "project_id": project.get("project_id"),
        "run_id": model_manifest.get("run_id"),
        "class_config_sha256": model_manifest.get("class_config_sha256"),
    }
    mismatches = {
        field: {"expected": value, "observed": identity.get(field)}
        for field, value in expected.items()
        if identity.get(field) != value
    }
    if mismatches:
        raise ValueError("Reviewed-label parent identity mismatch: " + json.dumps(mismatches, sort_keys=True))

    fields, rows = read_tsv(reviewed_labels)
    require_columns(
        fields,
        ("cell_id", "review_status", "class_id", "reviewer_id", "reviewed_at", "confidence"),
        "Reviewed labels",
    )
    seen: set[str] = set()
    excluded: set[str] = set()
    for row in rows:
        cell_id, status = row["cell_id"], row["review_status"]
        if cell_id in seen or cell_id not in cells:
            raise ValueError(f"Reviewed label is duplicate or outside representative universe: {cell_id}")
        if status not in {"unreviewed", "confirmed", "corrected", "skipped"}:
            raise ValueError(f"Reviewed label has unsupported status: {cell_id}/{status}")
        if status in {"confirmed", "corrected", "skipped"}:
            excluded.add(cell_id)
        seen.add(cell_id)
    return manifest, excluded, identities


def validate_umap_generation(
    paths: dict[str, Path],
    project: dict[str, Any],
    run_id: str,
    cells: dict[str, dict[str, str]],
    shadow_root: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    projection_root = paths["runs_dir"] / str(project["project_id"]) / run_id / "projection"
    if not projection_root.is_dir():
        raise FileNotFoundError(f"Representative UMAP projection root is unavailable: {projection_root}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(projection_root.glob("*/umap_manifest.json")):
        manifest = load_json(path, "UMAP manifest")
        if (
            manifest.get("schema_version") == UMAP_SCHEMA_VERSION
            and manifest.get("project_id") == project.get("project_id")
            and manifest.get("run_id") == run_id
        ):
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one canonical representative UMAP generation for run_id={run_id}; "
            f"observed={len(matches)}"
        )
    directory, manifest = matches[0]
    if manifest.get("projection_id") != directory.name:
        raise ValueError("UMAP manifest projection_id/path mismatch")
    identities = validate_declared_generation(
        directory,
        "umap_manifest.json",
        manifest,
        UMAP_ARTIFACTS,
        shadow_root,
        "umap",
    )
    fields, rows = read_tsv(directory / "umap.tsv")
    require_columns(fields, ("cell_id", "Dim1", "Dim2"), "UMAP")
    observed: set[str] = set()
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in observed:
            raise ValueError(f"UMAP contains duplicate cell_id: {cell_id}")
        try:
            coordinates = (float(row["Dim1"]), float(row["Dim2"]))
        except ValueError as error:
            raise ValueError(f"UMAP contains invalid coordinates: {cell_id}") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"UMAP contains nonfinite coordinates: {cell_id}")
        observed.add(cell_id)
    if observed != set(cells):
        raise ValueError("Canonical representative UMAP does not cover the exact cell universe")
    return directory, manifest, identities


def install_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace non-identical frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tsv_bytes(fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> bytes:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return handle.getvalue().encode("utf-8")


def usage_text(output_project: Path, shadow_root: Path) -> str:
    return f"""# Broad-phenotype active-learning V1 queue

This frozen project is for **targeted validation only** inside the physical
development representative project. It excludes heldout cells and never runs
against the full 38.6M-cell universe or the viability endpoint.

Use the pinned source wrapper in this order:

```bash
python cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py \\
  --stage review-build --check-config \\
  --project {output_project} \\
  --shadow-root {shadow_root} \\
  --reference-root <PINNED_REFERENCE_ROOT> \\
  --annotation-import-dir <ACCEPTED_ANNOTATION_IMPORT_DIR>

python cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py \\
  --stage review-build \\
  --project {output_project} \\
  --shadow-root {shadow_root} \\
  --reference-root <PINNED_REFERENCE_ROOT> \\
  --annotation-import-dir <ACCEPTED_ANNOTATION_IMPORT_DIR>
```

After manual HTML review, import the submission with the same project and
annotation import directory. The reference package creates a new immutable
`review_id`; V1 does **not** merge labels across review IDs and does **not**
automatically retrain. Only one authoritative `reviewed_labels.tsv` generation
is accepted as queue history.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if QUEUE_ID_RE.fullmatch(args.queue_id) is None:
        raise ValueError(f"Invalid --queue-id: {args.queue_id}")
    if args.max_cells < 1:
        raise ValueError("--max-cells must be positive")
    if args.max_per_group is not None and args.max_per_group < 1:
        raise ValueError("--max-per-group must be positive")
    shadow_root = args.shadow_root.expanduser().resolve()
    if not shadow_root.is_dir():
        raise FileNotFoundError(shadow_root)
    project_path = require_inside(args.project, shadow_root, "project", "file")
    project = load_project(project_path)
    paths, feature_columns, class_ids, cells = validate_representative_project(
        project_path, project, shadow_root
    )
    model_manifest, prediction_manifest, prediction_inputs = validate_model_and_prediction(
        args.prediction_dir,
        project,
        feature_columns,
        class_ids,
        cells,
        shadow_root,
    )
    review_manifest, excluded_ids, history_inputs = validate_review_history(
        args.reviewed_labels,
        project,
        model_manifest,
        cells,
        shadow_root,
    )
    umap_dir, umap_manifest, umap_inputs = validate_umap_generation(
        paths,
        project,
        str(model_manifest.get("run_id", "")),
        cells,
        shadow_root,
    )
    if prediction_manifest.get("run_id") != umap_manifest.get("run_id"):
        raise ValueError("Prediction and UMAP run identity mismatch")
    available_unreviewed = len(cells) - len(excluded_ids)
    if args.shortage_policy == "fail" and available_unreviewed < args.max_cells:
        raise ValueError(
            f"Queue cannot satisfy max_cells after history exclusion: "
            f"available={available_unreviewed} requested={args.max_cells}"
        )

    output_root = require_inside(
        shadow_root / "active_learning" / args.queue_id,
        shadow_root,
        "active-learning output",
        "any",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_project = output_root / "project.yml"
    derived = json.loads(json.dumps(project))
    for field in ("classes_file", "cells_file", "features_file", "images_file", "runs_dir"):
        derived[field] = str(paths[field])
    projection = derived.get("projection")
    if isinstance(projection, dict) and projection.get("coordinate_file"):
        projection["coordinate_file"] = str(
            require_inside(
                resolve_project_path(project_path, str(projection["coordinate_file"])),
                shadow_root,
                "projection.coordinate_file",
                "file",
            )
        )
    review = derived["review"]
    for field in ("quotas", "max_total"):
        review.pop(field, None)
    review["seed"] = args.seed
    if args.max_per_group is not None:
        review["max_per_group"] = args.max_per_group
    if review.get("max_per_group") is None:
        raise ValueError("Active-learning queue requires review.max_per_group")
    reviewed_labels = require_inside(
        args.reviewed_labels, shadow_root, "reviewed_labels", "file"
    )
    prediction_dir = require_inside(
        args.prediction_dir, shadow_root, "prediction_dir", "dir"
    )
    review["queue"] = {
        "queue_id": args.queue_id,
        "strategy": "model_uncertainty",
        "max_cells": args.max_cells,
        "shortage_policy": args.shortage_policy,
        "allow_subset": False,
        "prediction_dir": str(prediction_dir),
        "uncertainty_metric": args.uncertainty_metric,
        "reviewed_labels_file": str(reviewed_labels),
        "exclude_review_statuses": ["confirmed", "corrected", "skipped"],
    }

    base_inputs = [artifact_identity(project_path, shadow_root, "base_project")]
    for field in ("classes_file", "cells_file", "features_file", "images_file"):
        base_inputs.append(artifact_identity(paths[field], shadow_root, field))
    input_rows = base_inputs + umap_inputs + prediction_inputs + history_inputs
    input_rows.sort(key=lambda row: (str(row["role"]), str(row["path"])))
    input_manifest_fields = [
        "role",
        "path",
        "size_bytes",
        "sha256",
        "manifest_declared_sha256",
    ]
    input_manifest_path = output_root / "input_manifest.tsv"
    project_payload = (json.dumps(derived, indent=2, sort_keys=False) + "\n").encode("utf-8")
    input_payload = tsv_bytes(input_manifest_fields, input_rows)
    install_frozen(output_project, project_payload)
    install_frozen(input_manifest_path, input_payload)
    usage_path = output_root / "USAGE.md"
    install_frozen(usage_path, usage_text(output_project, shadow_root).encode("utf-8"))

    input_identity = {
        "project_sha256": sha256_file(output_project),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "queue": review["queue"],
        "seed": args.seed,
        "max_per_group": review["max_per_group"],
        "representative_cell_count": len(cells),
        "history_excluded_cell_count": len(excluded_ids),
        "available_unreviewed_cell_count": available_unreviewed,
        "project_id": project["project_id"],
        "run_id": model_manifest["run_id"],
        "class_config_sha256": model_manifest["class_config_sha256"],
        "classifier_config_sha256": model_manifest["classifier_config_sha256"],
        "feature_columns": feature_columns,
        "class_ids": class_ids,
        "model_id": model_manifest["model_id"],
        "prediction_id": prediction_manifest["prediction_id"],
        "projection_id": umap_manifest["projection_id"],
        "review_history_submission_id": review_manifest.get("submission_id"),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "targeted_validation",
        "shadow_root": str(shadow_root),
        "base_project": str(project_path),
        "active_project": str(output_project),
        "input_manifest": str(input_manifest_path),
        "usage": str(usage_path),
        "umap_dir": str(umap_dir),
        "input_identity": input_identity,
        "input_identity_sha256": canonical_sha256(input_identity),
        "heldout_cell_count": 0,
        "automatic_cross_review_id_merge": False,
        "automatic_retraining": False,
        "limitations": list(LIMITATIONS),
        "output_sha256": {
            "project.yml": sha256_file(output_project),
            "input_manifest.tsv": sha256_file(input_manifest_path),
            "USAGE.md": sha256_file(usage_path),
        },
    }
    receipt_path = output_root / "receipt.json"
    install_frozen(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"active_learning_project={output_project}")
    print(f"active_learning_receipt={receipt_path}")
    print(f"queue_id={args.queue_id} strategy=model_uncertainty max_cells={args.max_cells}")
    print("scope=targeted_validation heldout_cells=0 automatic_retraining=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
