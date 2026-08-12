#!/usr/bin/env python3
"""Dispatch one Cell Phenotype Annotator stage from a pinned source checkout.

No package source is copied into this repository.  The wrapper verifies the
dependency lock, runs the reference checkout's source-mode executable, confines
all configured outputs to a shadow root, and records the exact command and
result without ever using the reference checkout as a working directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "cellphenotypeannotator_stage_receipt_v1"
STAGES = (
    "validate",
    "umap",
    "annotate",
    "annotation-import",
    "review-build",
    "review-import",
    "train",
    "predict",
    "report",
)
VALIDATION_STAGES = (
    "project",
    "all",
    "umap",
    "annotate",
    "annotation-import",
    "review-build",
    "review-import",
    "train",
    "predict",
    "report",
)
MODEL_GENERATION_ARTIFACTS = {
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
PREDICTION_GENERATION_ARTIFACTS = {
    "all_cell_predictions": "all_cell_predictions.tsv",
    "all_cell_probabilities": "all_cell_probabilities.tsv",
}
REVIEW_IMPORT_GENERATION_ARTIFACTS = {
    "review_submission": "review_submission.json",
    "reviewed_labels": "reviewed_labels.tsv",
}


def default_dependency_lock() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "cellphenotypeannotator_dependency.lock.tsv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, default=default_dependency_lock())
    parser.add_argument("--rscript", default="Rscript")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-stage", choices=VALIDATION_STAGES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--annotation-import-dir", type=Path)
    parser.add_argument("--reviewed-labels", type=Path)
    parser.add_argument("--model-dir", type=Path)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not is_relative_to(resolved, root):
        raise ValueError(f"{label} must be inside shadow root {root}: {resolved}")
    return resolved


def reject_shadow_symlink_chain(path: Path, root: Path, label: str) -> Path:
    """Reject symlinks inside shadow without rejecting macOS' ``/var`` alias."""

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
    candidates = [anchor]
    candidates.extend(
        anchor / Path(*relative.parts[:index])
        for index in range(1, len(relative.parts) + 1)
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
    return lexical


def load_mapping(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"Expected a project mapping: {path}")
    return value


def resolve_project_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project.parent / path
    return path.resolve()


def validate_project_confinement(project: Path, shadow_root: Path) -> dict[str, Any]:
    require_inside(project, shadow_root, "Project")
    if not project.is_file():
        raise FileNotFoundError(project)
    config = load_mapping(project)
    runs_dir = resolve_project_path(project, str(config.get("runs_dir", "runs")))
    require_inside(runs_dir, shadow_root, "Project runs_dir")
    return config


def file_identity(path: Path, shadow_root: Path, role: str) -> dict[str, Any]:
    if path.expanduser().is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    resolved = require_inside(path, shadow_root, role)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def directory_file_identities(
    directory: Path,
    shadow_root: Path,
    role: str,
    *,
    recursive: bool,
    exclude_top_level: Sequence[str] = (),
) -> list[dict[str, Any]]:
    resolved = require_inside(directory, shadow_root, role)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    candidates = resolved.rglob("*") if recursive else resolved.iterdir()
    identities: list[dict[str, Any]] = []
    excluded = set(exclude_top_level)
    for path in sorted(candidates, key=lambda value: str(value)):
        relative = path.relative_to(resolved)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"Input generation must not contain symlinks: {path}")
        if path.is_file():
            identities.append(file_identity(path, shadow_root, f"{role}:{relative}"))
    if not identities:
        raise ValueError(f"Input generation contains no files: {resolved}")
    return identities


def discover_generation_identities(
    runs_dir: Path,
    shadow_root: Path,
    manifest_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Hash only named immutable upstream generations, never the whole runs tree."""

    if not runs_dir.exists():
        return []
    identities: list[dict[str, Any]] = []
    seen_parents: set[Path] = set()
    for name in manifest_names:
        for manifest in sorted(runs_dir.rglob(name), key=lambda value: str(value)):
            parent = manifest.parent.resolve()
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
            # Review crop integrity is encoded in render_manifest.tsv.  Hash the
            # complete review generation; other generations have only direct
            # immutable artifacts, but recursive hashing is still bounded to
            # this one manifest-addressed directory.
            identities.extend(
                directory_file_identities(
                    parent,
                    shadow_root,
                    f"upstream_generation:{name}:{parent.relative_to(shadow_root)}",
                    recursive=name == "review_manifest.json",
                    exclude_top_level=("accepted",),
                )
            )
    return identities


def model_generation_identities(
    model_dir: Path,
    shadow_root: Path,
) -> list[dict[str, Any]]:
    """Verify and identify only the immutable classifier base generation.

    This mirrors ``cpa_verify_classifier_generation`` in the pinned reference:
    the manifest and its exact declared artifact set are inputs.  ``report.html``,
    ``predictions/``, and any other stage outputs are deliberately outside this
    identity, so report/predict can write them without mutating their own input.
    """

    resolved = require_inside(model_dir, shadow_root, "model generation")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    manifest_path = resolved / "model_manifest.json"
    manifest_identity = file_identity(
        manifest_path,
        shadow_root,
        "model_generation:model_manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Model manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Model manifest must contain one object: {manifest_path}")
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict) or set(declared) != set(MODEL_GENERATION_ARTIFACTS):
        observed = sorted(declared) if isinstance(declared, dict) else type(declared).__name__
        raise ValueError(
            "Classifier model manifest artifact hash set is incomplete or unexpected: "
            f"expected={sorted(MODEL_GENERATION_ARTIFACTS)} observed={observed}"
        )

    identities = [manifest_identity]
    for artifact_role, filename in MODEL_GENERATION_ARTIFACTS.items():
        identity = file_identity(
            resolved / filename,
            shadow_root,
            f"model_generation:{artifact_role}",
        )
        expected_sha256 = declared[artifact_role]
        if not isinstance(expected_sha256, str) or identity["sha256"] != expected_sha256:
            raise RuntimeError(
                "Classifier model artifact hash mismatch: "
                f"role={artifact_role} expected={expected_sha256!r} "
                f"observed={identity['sha256']}"
            )
        identity["manifest_declared_sha256"] = expected_sha256
        identities.append(identity)
    return identities


def declared_generation_identities(
    directory: Path,
    shadow_root: Path,
    role: str,
    manifest_name: str,
    artifact_names: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = require_inside(directory, shadow_root, role)
    if directory.expanduser().is_symlink() or not resolved.is_dir():
        raise ValueError(f"{role} must be a real directory inside shadow: {directory}")
    manifest_path = resolved / manifest_name
    manifest_identity = file_identity(manifest_path, shadow_root, f"{role}:manifest")
    manifest = load_mapping(manifest_path)
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict) or set(declared) != set(artifact_names):
        observed = sorted(declared) if isinstance(declared, dict) else type(declared).__name__
        raise ValueError(
            f"{role} manifest artifact set is incomplete or unexpected: "
            f"expected={sorted(artifact_names)} observed={observed}"
        )
    identities = [manifest_identity]
    for artifact_role, filename in artifact_names.items():
        identity = file_identity(
            resolved / filename,
            shadow_root,
            f"{role}:{artifact_role}",
        )
        expected = declared[artifact_role]
        if not isinstance(expected, str) or identity["sha256"] != expected:
            raise RuntimeError(
                f"{role} artifact hash mismatch: {artifact_role} "
                f"expected={expected!r} observed={identity['sha256']}"
            )
        identity["manifest_declared_sha256"] = expected
        identities.append(identity)
    return manifest, identities


def prediction_generation_identities(
    prediction_dir: Path,
    shadow_root: Path,
) -> list[dict[str, Any]]:
    resolved = require_inside(prediction_dir, shadow_root, "queue prediction generation")
    if resolved.parent.name != "predictions" or not resolved.is_dir():
        raise ValueError(
            "review.queue.prediction_dir must be a canonical "
            "<model_dir>/predictions/<prediction_id> directory"
        )
    manifest, identities = declared_generation_identities(
        resolved,
        shadow_root,
        "queue_prediction_generation",
        "prediction_manifest.json",
        PREDICTION_GENERATION_ARTIFACTS,
    )
    model_dir = require_inside(
        resolved.parent.parent,
        shadow_root,
        "queue prediction model generation",
    )
    model_identities = model_generation_identities(model_dir, shadow_root)
    expected = {
        "prediction_id": resolved.name,
        "model_sha256": sha256_file(model_dir / "model.rds"),
        "model_manifest_sha256": sha256_file(model_dir / "model_manifest.json"),
    }
    mismatches = {
        field: {"expected": value, "observed": manifest.get(field)}
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    if mismatches:
        raise ValueError(
            "Queue prediction/model generation identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return identities + model_identities


def review_history_generation_identities(
    reviewed_labels: Path,
    shadow_root: Path,
) -> list[dict[str, Any]]:
    resolved = require_inside(reviewed_labels, shadow_root, "queue reviewed labels")
    if resolved.name != "reviewed_labels.tsv" or not resolved.is_file():
        raise ValueError(
            "review.queue.reviewed_labels_file must be an authoritative reviewed_labels.tsv"
        )
    _, identities = declared_generation_identities(
        resolved.parent,
        shadow_root,
        "queue_review_history_generation",
        "review_import_manifest.json",
        REVIEW_IMPORT_GENERATION_ARTIFACTS,
    )
    expected_path = resolved.parent / REVIEW_IMPORT_GENERATION_ARTIFACTS["reviewed_labels"]
    if resolved != expected_path.resolve():
        raise ValueError("Queue reviewed labels do not match their manifest generation")
    return identities


def review_queue_input_identities(
    project_path: Path,
    project: dict[str, Any],
    shadow_root: Path,
) -> list[dict[str, Any]]:
    review = project.get("review")
    if not isinstance(review, dict) or review.get("queue") is None:
        return []
    queue = review.get("queue")
    if not isinstance(queue, dict):
        raise ValueError("review.queue must contain one mapping")
    strategy = queue.get("strategy")
    if strategy not in {
        "user_priority",
        "model_uncertainty",
        "model_disagreement",
        "underrepresented_stratum",
    }:
        raise ValueError(f"Unsupported review.queue.strategy: {strategy}")
    identities: list[dict[str, Any]] = []

    def queue_path(value: str, label: str) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raw = project_path.parent / raw
        return reject_shadow_symlink_chain(raw, shadow_root, label)

    score_value = queue.get("score_file")
    prediction_value = queue.get("prediction_dir")
    history_value = queue.get("reviewed_labels_file")
    if strategy == "user_priority" and not isinstance(score_value, str):
        raise ValueError("review.queue user_priority requires score_file")
    if strategy in {"model_uncertainty", "model_disagreement"} and not isinstance(
        prediction_value, str
    ):
        raise ValueError(f"review.queue {strategy} requires prediction_dir")
    if strategy == "underrepresented_stratum" and not isinstance(history_value, str):
        raise ValueError("review.queue underrepresented_stratum requires reviewed_labels_file")

    if isinstance(score_value, str):
        identities.append(
            file_identity(
                queue_path(score_value, "review.queue.score_file"),
                shadow_root,
                "queue_score_file",
            )
        )
    if isinstance(prediction_value, str):
        identities.extend(
            prediction_generation_identities(
                queue_path(prediction_value, "review.queue.prediction_dir"),
                shadow_root,
            )
        )
    if isinstance(history_value, str):
        identities.extend(
            review_history_generation_identities(
                queue_path(history_value, "review.queue.reviewed_labels_file"),
                shadow_root,
            )
        )
    identities.sort(key=lambda item: (str(item["role"]), str(item["path"])))
    return identities


def build_input_identity(
    args: argparse.Namespace,
    dispatch_project: Path,
    project: dict[str, Any],
    dependency: dict[str, Any],
    shadow_root: Path,
) -> dict[str, Any]:
    files = [file_identity(dispatch_project, shadow_root, "dispatch_project")]
    for field in ("classes_file", "cells_file", "features_file", "images_file"):
        value = project.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Project requires {field}")
        files.append(
            file_identity(resolve_project_path(dispatch_project, value), shadow_root, field)
        )
    projection = project.get("projection")
    if isinstance(projection, dict) and projection.get("coordinate_file"):
        files.append(
            file_identity(
                resolve_project_path(dispatch_project, str(projection["coordinate_file"])),
                shadow_root,
                "projection.coordinate_file",
            )
        )

    runs_dir = require_inside(
        resolve_project_path(dispatch_project, str(project.get("runs_dir", "runs"))),
        shadow_root,
        "Project runs_dir",
    )
    semantic_stage = args.validate_stage if args.stage == "validate" else args.stage
    upstream_names: list[str] = []
    if semantic_stage in {"annotate", "annotation-import", "review-build", "review-import"}:
        upstream_names.append("umap_manifest.json")
    if semantic_stage == "annotation-import":
        upstream_names.append("annotation_manifest.json")
    if semantic_stage == "review-import":
        upstream_names.append("review_manifest.json")
    files.extend(discover_generation_identities(runs_dir, shadow_root, upstream_names))
    if semantic_stage in {"review-build", "review-import"}:
        files.extend(review_queue_input_identities(dispatch_project, project, shadow_root))

    if args.submission is not None:
        # Human submissions may originate outside shadow, so bind their bytes
        # without applying shadow confinement.
        files.append(
            {
                "role": "submission",
                "path": str(args.submission),
                "size_bytes": args.submission.stat().st_size,
                "sha256": sha256_file(args.submission),
            }
        )
    if args.annotation_import_dir is not None:
        files.extend(
            directory_file_identities(
                args.annotation_import_dir,
                shadow_root,
                "annotation_import_generation",
                recursive=True,
            )
        )
    if args.reviewed_labels is not None:
        files.extend(
            directory_file_identities(
                args.reviewed_labels.parent,
                shadow_root,
                "review_import_generation",
                recursive=False,
            )
        )
    if args.model_dir is not None:
        files.extend(model_generation_identities(args.model_dir, shadow_root))

    files.sort(key=lambda item: (str(item["role"]), str(item["path"])))
    dependency_identity = {
        field: dependency[field]
        for field in (
            "root",
            "lock",
            "lock_sha256",
            "commit",
            "tree",
            "license_path",
            "license_sha256",
            "description_sha256",
            "execution_mode",
        )
    }
    identity = {
        "schema_version": "cellphenotypeannotator_stage_input_identity_v1",
        "dependency": dependency_identity,
        "files": files,
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


def load_lock(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    matches = [row for row in rows if row.get("dependency") == "cellphenotypeannotator"]
    if len(matches) != 1:
        raise ValueError(
            f"Dependency lock must contain exactly one cellphenotypeannotator row: {path}"
        )
    required = {
        "commit",
        "tree",
        "license_path",
        "license_sha256",
        "description_sha256",
        "execution_mode",
    }
    missing = sorted(field for field in required if not matches[0].get(field))
    if missing:
        raise ValueError(f"Dependency lock is missing values: {missing}")
    if matches[0]["execution_mode"] != "private_read_only_source_checkout":
        raise ValueError("Dependency execution_mode must be private_read_only_source_checkout")
    return {str(key): str(value or "") for key, value in matches[0].items()}


def git_output(reference_root: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "-C", str(reference_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Unable to verify reference checkout with git {' '.join(args)}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_reference(reference_root: Path, lock_path: Path) -> dict[str, Any]:
    reference_root = reference_root.expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    lock = load_lock(lock_path)
    cli = reference_root / "exec" / "cell-phenotype-annotator"
    description = reference_root / "DESCRIPTION"
    license_path = reference_root / lock["license_path"]
    for path in (cli, description, license_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = git_output(reference_root, "rev-parse", "HEAD")
    tree = git_output(reference_root, "rev-parse", "HEAD^{tree}")
    status = git_output(reference_root, "status", "--porcelain", "--untracked-files=all")
    observed = {
        "commit": commit,
        "tree": tree,
        "license_sha256": sha256_file(license_path),
        "description_sha256": sha256_file(description),
    }
    mismatches = {
        key: {"expected": lock[key], "observed": value}
        for key, value in observed.items()
        if lock[key] != value
    }
    if status:
        mismatches["git_status"] = {"expected": "clean", "observed": status}
    if mismatches:
        raise RuntimeError(
            "Reference checkout does not match the dependency lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "root": str(reference_root),
        "cli": str(cli),
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        **observed,
        "license_path": str(license_path),
        "execution_mode": lock["execution_mode"],
        "git_status_before": status,
    }


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"Rscript executable is unavailable: {resolved}")
        return str(resolved)
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Rscript executable is unavailable on PATH: {value}")
    return resolved


def validate_stage_arguments(args: argparse.Namespace) -> None:
    if args.validate_stage and args.stage != "validate":
        raise ValueError("--validate-stage is accepted only for --stage validate")
    if args.output_dir and args.stage != "validate":
        raise ValueError("--output-dir is accepted only for --stage validate")
    if args.submission and args.stage not in {"annotation-import", "review-import"}:
        raise ValueError("--submission is accepted only for annotation-import or review-import")
    if args.stage in {"annotation-import", "review-import"} and args.submission is None:
        raise ValueError(f"--stage {args.stage} requires --submission")
    if args.annotation_import_dir and args.stage not in {"review-build", "review-import"}:
        raise ValueError("--annotation-import-dir is accepted only for review-build or review-import")
    if args.stage in {"review-build", "review-import"} and args.annotation_import_dir is None:
        raise ValueError(
            f"{args.stage} requires --annotation-import-dir to bind the accepted annotation parent"
        )
    if args.reviewed_labels and args.stage != "train":
        raise ValueError("--reviewed-labels is accepted only for train")
    if args.stage == "train" and args.reviewed_labels is None:
        raise ValueError("train requires --reviewed-labels from an accepted review import")
    if args.model_dir and args.stage not in {"predict", "report"}:
        raise ValueError("--model-dir is accepted only for predict or report")
    if args.stage in {"predict", "report"} and args.model_dir is None:
        raise ValueError(f"{args.stage} requires --model-dir")


def write_derived_review_project(
    project_path: Path,
    project: dict[str, Any],
    annotation_import_dir: Path,
    shadow_root: Path,
) -> Path:
    annotation_import_dir = require_inside(
        annotation_import_dir, shadow_root, "Accepted annotation import"
    )
    if not annotation_import_dir.is_dir():
        raise FileNotFoundError(annotation_import_dir)
    derived = json.loads(json.dumps(project))
    review = derived.get("review")
    if not isinstance(review, dict):
        raise ValueError("Project review configuration is required for review-build")
    review["annotation_import_dir"] = str(annotation_import_dir)
    payload = json.dumps(derived, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    # Keep the derived file beside the project so every relative input/output
    # path retains exactly the same meaning.
    output = project_path.parent / f"{project_path.stem}.review.{identity}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Derived review project conflicts with existing artifact: {output}")
    if not output.exists():
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return output


def build_reference_command(
    args: argparse.Namespace,
    rscript: str,
    cli: Path,
    project: Path,
) -> list[str]:
    command = [rscript, str(cli), args.stage, str(project)]
    if args.stage == "validate":
        if args.validate_stage:
            command.extend(["--stage", args.validate_stage])
        if args.output_dir:
            command.extend(["--output-dir", str(args.output_dir.resolve())])
    if args.stage in {"annotation-import", "review-import"}:
        command.extend(["--submission", str(args.submission.resolve())])
    if args.stage == "train":
        command.extend(["--reviewed-labels", str(args.reviewed_labels.resolve())])
    if args.stage in {"predict", "report"}:
        command.extend(["--model-dir", str(args.model_dir.resolve())])
    if args.check_config:
        command.append("--check-config")
    elif args.dry_run:
        command.append("--dry-run")
    elif args.overwrite:
        command.append("--overwrite")
    return command


def atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def receipt_mode(args: argparse.Namespace) -> str:
    if args.check_config:
        return "check_config"
    if args.dry_run:
        return "dry_run"
    if args.overwrite:
        return "overwrite"
    return "run"


def stdout_value(stdout: str, field: str) -> str:
    prefix = f"{field}: "
    values = [line[len(prefix) :].strip() for line in stdout.splitlines() if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"Reference stdout did not expose exactly one {field}: value")
    return values[0]


def stage_output_identity(
    args: argparse.Namespace,
    stdout: str,
    shadow_root: Path,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"files": []}
    if args.stage == "review-build":
        evidence["review_id"] = stdout_value(stdout, "review_id")
        evidence["selection_mode"] = stdout_value(stdout, "selection_mode")
        if not args.check_config and not args.dry_run:
            manifest = require_inside(
                Path(stdout_value(stdout, "manifest")),
                shadow_root,
                "review manifest output",
            )
            identity = file_identity(manifest, shadow_root, "review_manifest_output")
            payload = load_mapping(manifest)
            manifest_identity = payload.get("identity")
            if (
                not isinstance(manifest_identity, dict)
                or manifest_identity.get("review_id") != evidence["review_id"]
            ):
                raise RuntimeError("Review manifest identity does not match reference stdout")
            evidence["files"] = [identity]
    elif args.stage == "review-import" and not args.check_config and not args.dry_run:
        output_dir = require_inside(
            Path(stdout_value(stdout, "output")),
            shadow_root,
            "review import output",
        )
        manifest, identities = declared_generation_identities(
            output_dir,
            shadow_root,
            "review_import_output",
            "review_import_manifest.json",
            REVIEW_IMPORT_GENERATION_ARTIFACTS,
        )
        evidence["review_id"] = stdout_value(stdout, "review_id")
        identity = manifest.get("identity")
        if not isinstance(identity, dict) or identity.get("review_id") != evidence["review_id"]:
            raise RuntimeError("Review-import manifest identity does not match reference stdout")
        evidence["files"] = identities
    return evidence


def reuse_receipt_if_verified(
    receipt_path: Path,
    command: list[str],
    input_identity: dict[str, Any],
) -> int | None:
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("command") != command:
        raise RuntimeError(f"Receipt command hash collision: {receipt_path}")
    if receipt.get("input_identity") != input_identity:
        raise RuntimeError(f"Stage input identity changed; refusing receipt reuse: {receipt_path}")
    for role in ("stdout", "stderr"):
        path = Path(str(receipt.get(f"{role}_log", "")))
        expected = receipt.get(f"{role}_sha256")
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise RuntimeError(f"Existing stage receipt has corrupt {role}: {receipt_path}")
    output_identity = receipt.get("output_identity")
    if isinstance(output_identity, dict):
        files = output_identity.get("files", [])
        if not isinstance(files, list):
            raise RuntimeError(f"Existing stage receipt has invalid output identity: {receipt_path}")
        for item in files:
            if not isinstance(item, dict):
                raise RuntimeError(f"Existing stage receipt has invalid output artifact: {receipt_path}")
            path = Path(str(item.get("path", "")))
            expected = item.get("sha256")
            if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
                raise RuntimeError(f"Existing stage receipt has corrupt output artifact: {path}")
    print(f"cpa_stage_reused=1 receipt={receipt_path}")
    return int(receipt.get("returncode", 1))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_stage_arguments(args)
    shadow_root = args.shadow_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    if is_relative_to(shadow_root, reference_root) or is_relative_to(reference_root, shadow_root):
        raise ValueError(
            "Shadow root and reference checkout must be disjoint; reference is strictly read-only"
        )
    shadow_root.mkdir(parents=True, exist_ok=True)
    project_path = args.project.expanduser().resolve()
    project = validate_project_confinement(project_path, shadow_root)
    if args.output_dir:
        args.output_dir = require_inside(args.output_dir, shadow_root, "Validation output directory")
    for name in ("reviewed_labels", "model_dir"):
        value = getattr(args, name)
        if value is not None:
            resolved = require_inside(value, shadow_root, name.replace("_", " "))
            if name == "model_dir" and not resolved.is_dir():
                raise FileNotFoundError(resolved)
            if name == "reviewed_labels" and not resolved.is_file():
                raise FileNotFoundError(resolved)
            setattr(args, name, resolved)
    if args.submission is not None:
        args.submission = args.submission.expanduser().resolve()
        if not args.submission.is_file():
            raise FileNotFoundError(args.submission)
    if args.annotation_import_dir is not None:
        args.annotation_import_dir = args.annotation_import_dir.expanduser().resolve()

    dependency = verify_reference(reference_root, args.dependency_lock.expanduser().resolve())
    dispatch_project = project_path
    if args.stage in {"review-build", "review-import"}:
        assert args.annotation_import_dir is not None
        dispatch_project = write_derived_review_project(
            project_path,
            project,
            args.annotation_import_dir,
            shadow_root,
        )
    dispatch_config = validate_project_confinement(dispatch_project, shadow_root)
    rscript = resolve_executable(args.rscript)
    command = build_reference_command(
        args,
        rscript,
        Path(dependency["cli"]),
        dispatch_project,
    )
    receipt_root = shadow_root / "workflow_status" / "cpa_stages"
    mode = receipt_mode(args)
    qualifier = f".{args.validate_stage or 'project'}" if args.stage == "validate" else ""
    command_sha256 = hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    input_identity = build_input_identity(
        args,
        dispatch_project,
        dispatch_config,
        dependency,
        shadow_root,
    )
    invocation_sha256 = hashlib.sha256(
        json.dumps(
            {"command": command, "input_identity": input_identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    basename = f"{args.stage}{qualifier}.{mode}.{invocation_sha256[:12]}"
    stdout_path = receipt_root / f"{basename}.stdout.log"
    stderr_path = receipt_root / f"{basename}.stderr.log"
    receipt_path = receipt_root / f"{basename}.json"
    reused = reuse_receipt_if_verified(receipt_path, command, input_identity)
    if reused is not None:
        return reused
    orphaned_logs = [path for path in (stdout_path, stderr_path) if path.exists()]
    if orphaned_logs:
        raise FileExistsError(
            "Incomplete prior stage provenance is preserved; refusing to replace logs without "
            f"a matching receipt: {[str(path) for path in orphaned_logs]}"
        )
    started = dt.datetime.now(dt.timezone.utc)
    stage_environment = dict(os.environ)
    stage_environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        command,
        cwd=shadow_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=stage_environment,
    )
    finished = dt.datetime.now(dt.timezone.utc)
    atomic_replace_text(stdout_path, completed.stdout)
    atomic_replace_text(stderr_path, completed.stderr)
    status_after = git_output(reference_root, "status", "--porcelain", "--untracked-files=all")
    if status_after != dependency["git_status_before"]:
        raise RuntimeError(
            "Reference checkout changed during stage execution; refusing to certify read-only use"
        )
    input_identity_after = build_input_identity(
        args,
        dispatch_project,
        dispatch_config,
        dependency,
        shadow_root,
    )
    if input_identity_after != input_identity:
        raise RuntimeError(
            "Stage inputs changed during execution; refusing to certify a mixed-generation result"
        )
    output_identity = (
        stage_output_identity(args, completed.stdout, shadow_root)
        if completed.returncode == 0
        else {"files": []}
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "shadow_root": str(shadow_root),
        "project": str(project_path),
        "dispatch_project": str(dispatch_project),
        "project_sha256": sha256_file(dispatch_project),
        "command": command,
        "command_sha256": command_sha256,
        "invocation_sha256": invocation_sha256,
        "input_identity": input_identity,
        "output_identity": output_identity,
        "command_shell_escaped": shlex.join(command),
        "working_directory": str(shadow_root),
        "reference": dependency,
        "reference_git_status_after": status_after,
        "stdout_log": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_log": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "human_barrier": {
            "before_annotation_import": args.stage == "annotation-import",
            "before_review_import": args.stage == "review-import",
            "authoritative_reviewed_labels_required": args.stage == "train",
        },
    }
    # The basename is command-addressed and never replaced. Reuse is handled
    # before dispatch, so a second write here indicates an external race.
    if receipt_path.exists():
        raise FileExistsError(f"Stage receipt appeared during execution: {receipt_path}")
    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp.{os.getpid()}")
    temporary_receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_receipt, receipt_path)
    finally:
        if temporary_receipt.exists():
            temporary_receipt.unlink()
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=__import__("sys").stderr)
    print(f"cpa_stage_receipt={receipt_path}")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
