#!/usr/bin/env python3
"""Validate immutable broad-phenotype production contracts.

The feature-input gate joins the resolved Stage 06 manifest, its per-field
records, the J0 input-identity freeze, and the feature receipts before the CPA
adapter can consume any shard.  The model gates validate the input-addressed
train receipt and the immutable model-acceptance receipt used by every sharded
prediction task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


FEATURE_RECEIPT_SCHEMA = "broad_phenotype_feature_receipt_v1"
FEATURE_GATE_SCHEMA = "broad_phenotype_feature_input_alignment_v1"
ACCEPTANCE_SCHEMA = "broad_phenotype_model_acceptance_v1"
STAGE_RECEIPT_SCHEMA = "cellphenotypeannotator_stage_receipt_v1"
BRANCH_MASK = {
    "original": ("combined_mask", "original_mask"),
    "nucleated": ("nucleated_combined_mask", "nucleated_mask"),
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any, *, ensure_ascii: bool = True) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def require_inside(path: Path, root: Path, label: str, *, must_exist: bool = True) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    root = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside shadow root {root}: {resolved}") from error
    return resolved


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise RuntimeError(f"{label} requires PyYAML for non-JSON YAML: {path}") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one mapping: {path}")
    return value


def read_tsv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no header: {path}")
        fields = [str(field) for field in reader.fieldnames]
        if any(not field for field in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{label} has blank or duplicate columns: {path}")
        rows = [
            {field: str(row.get(field, "") or "").strip() for field in fields}
            for row in reader
        ]
    return fields, rows


def unique_index(rows: Iterable[dict[str, str]], key_name: str, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(key_name, "")
        if not key or key in result:
            raise ValueError(f"{label} has a blank or duplicate {key_name}: {key!r}")
        result[key] = row
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def resolve_table_path(value: str, table: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = table.parent / path
    return require_file(path, label)


def resolve_project_path(project: Path, value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project.parent / path
    return require_file(path, label)


def atomic_frozen(path: Path, payload: bytes) -> None:
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


def tsv_payload(fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue().encode("utf-8")


def record_profile_path(record: dict[str, Any], profile: str, field: str, record_path: Path) -> Path:
    profiles = record.get("profiles")
    section = profiles.get(profile) if isinstance(profiles, dict) else None
    value = section.get(field) if isinstance(section, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field record lacks profiles.{profile}.{field}: {record_path}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = record_path.parent / path
    return require_file(path, f"profiles.{profile}.{field}")


def preflight_values(path: Path) -> dict[str, str]:
    fields, rows = read_tsv(path, "submission preflight")
    if fields != ["property", "value"]:
        raise ValueError(f"Submission preflight must have exact property/value columns: {path}")
    return unique_index(rows, "property", "submission preflight") | {}


def preflight_scalar(path: Path, property_name: str) -> str:
    _, rows = read_tsv(path, "submission preflight")
    matches = [row["value"] for row in rows if row.get("property") == property_name and row.get("value")]
    if len(matches) != 1:
        raise ValueError(
            f"Submission preflight must contain exactly one nonblank {property_name}: {path}"
        )
    return matches[0]


def validate_feature_inputs(args: argparse.Namespace) -> int:
    shadow = require_directory(args.shadow_root, "shadow root")
    field_manifest = require_inside(args.field_manifest, shadow, "field manifest")
    identity_manifest = require_inside(
        args.input_identity_manifest, shadow, "J0 input identity manifest"
    )
    feature_manifest = require_inside(args.feature_manifest, shadow, "feature manifest")
    output_receipt = require_inside(
        args.output_receipt, shadow, "feature alignment receipt", must_exist=False
    )
    output_audit = require_inside(
        args.output_audit, shadow, "feature alignment audit", must_exist=False
    )
    branch = args.branch
    include_nuclei = args.include_nuclei_comparator == "1"
    mask_column, mask_field = BRANCH_MASK[branch]

    manifest_fields, manifest_rows = read_tsv(field_manifest, "resolved field manifest")
    required_manifest = {"key", "record_json", "brightfield_raw", mask_column}
    missing = sorted(required_manifest - set(manifest_fields))
    if missing:
        raise ValueError(f"Resolved field manifest lacks columns: {missing}")
    fields_by_key = unique_index(manifest_rows, "key", "resolved field manifest")

    identity_fields, identity_rows = read_tsv(identity_manifest, "J0 input identity manifest")
    expected_identity_fields = [
        "key",
        "branch",
        "include_nuclei_comparator",
        "asset_role",
        "manifest_column",
        "record_json_pointer",
        "manifest_path",
        "record_path",
        "resolved_path",
        "file_size_bytes",
        "file_sha256",
        "alignment",
    ]
    if identity_fields != expected_identity_fields:
        raise ValueError(
            "J0 input identity manifest schema changed: "
            f"expected={expected_identity_fields} observed={identity_fields}"
        )
    identities: dict[tuple[str, str], dict[str, str]] = {}
    for row in identity_rows:
        identity_key = (row["key"], row["asset_role"])
        if identity_key in identities:
            raise ValueError(f"Duplicate J0 input identity row: {identity_key}")
        identities[identity_key] = row

    feature_fields, feature_rows = read_tsv(feature_manifest, "feature manifest")
    if feature_fields != ["key", "feature_path", "receipt_path"]:
        raise ValueError(
            "Feature manifest must have exact key/feature_path/receipt_path columns"
        )
    features_by_key = unique_index(feature_rows, "key", "feature manifest")
    if set(fields_by_key) != set(features_by_key):
        raise ValueError(
            "Feature/field manifest universes differ: "
            f"missing={sorted(set(fields_by_key)-set(features_by_key))[:10]} "
            f"extra={sorted(set(features_by_key)-set(fields_by_key))[:10]}"
        )

    audit_rows: list[dict[str, Any]] = []
    for key in sorted(fields_by_key):
        field = fields_by_key[key]
        record_path = resolve_table_path(field["record_json"], field_manifest, f"record_json for {key}")
        require_inside(record_path, shadow, f"resolved field record for {key}")
        record = read_json(record_path, f"field record for {key}")
        if record.get("key") != key:
            raise ValueError(f"Field record key mismatch: expected={key} observed={record.get('key')}")

        bf_record = record_profile_path(record, "Brightfield", "raw", record_path)
        bf_manifest = resolve_table_path(field["brightfield_raw"], field_manifest, f"brightfield_raw for {key}")
        combined_record = record_profile_path(record, "Combined", mask_field, record_path)
        combined_manifest = resolve_table_path(field[mask_column], field_manifest, f"{mask_column} for {key}")

        assets: list[tuple[str, str, str, Path, Path]] = [
            ("field_record", "record_json", "/", record_path, record_path),
            ("brightfield_raw", "brightfield_raw", "/profiles/Brightfield/raw", bf_manifest, bf_record),
            ("combined_mask", mask_column, f"/profiles/Combined/{mask_field}", combined_manifest, combined_record),
        ]
        nuclei_field = ""
        if include_nuclei:
            profiles = record.get("profiles")
            nuclei = profiles.get("Nuclei") if isinstance(profiles, dict) else None
            if not isinstance(nuclei, dict):
                raise ValueError(f"Nuclei comparator enabled but field record lacks Nuclei profile: {key}")
            if isinstance(nuclei.get("core_mask"), str) and nuclei["core_mask"].strip():
                nuclei_field = "core_mask"
                nuclei_column = "nuclei_core_mask"
            elif isinstance(nuclei.get("extent_mask"), str) and nuclei["extent_mask"].strip():
                nuclei_field = "extent_mask"
                nuclei_column = "nuclei_extent_mask"
            else:
                raise ValueError(f"Nuclei comparator enabled but no core/extent mask exists: {key}")
            if nuclei_column not in field or not field[nuclei_column]:
                raise ValueError(f"Resolved manifest lacks selected {nuclei_column} for {key}")
            nuclei_record = record_profile_path(record, "Nuclei", nuclei_field, record_path)
            nuclei_manifest = resolve_table_path(
                field[nuclei_column], field_manifest, f"{nuclei_column} for {key}"
            )
            assets.append(
                (
                    "nuclei_mask",
                    nuclei_column,
                    f"/profiles/Nuclei/{nuclei_field}",
                    nuclei_manifest,
                    nuclei_record,
                )
            )

        feature_row = features_by_key[key]
        feature_path = resolve_table_path(feature_row["feature_path"], feature_manifest, f"feature shard for {key}")
        receipt_path = resolve_table_path(feature_row["receipt_path"], feature_manifest, f"feature receipt for {key}")
        require_inside(feature_path, shadow, f"feature shard for {key}")
        require_inside(receipt_path, shadow, f"feature receipt for {key}")
        receipt = read_json(receipt_path, f"feature receipt for {key}")
        if receipt.get("schema_version") != FEATURE_RECEIPT_SCHEMA or receipt.get("status") != "COMPLETE":
            raise RuntimeError(f"Feature receipt is not a COMPLETE {FEATURE_RECEIPT_SCHEMA}: {receipt_path}")
        expected_receipt = {
            "key": key,
            "branch": branch,
            "include_nuclei_comparator": include_nuclei,
            "feature_tsv": str(feature_path),
            "feature_tsv_sha256": sha256_file(feature_path),
            "combined_mask_field": mask_field,
            "nuclei_mask_field": nuclei_field,
        }
        for name, expected in expected_receipt.items():
            observed = receipt.get(name)
            if observed != expected:
                raise RuntimeError(
                    f"Feature receipt mismatch for {key} {name}: expected={expected!r} observed={observed!r}"
                )

        receipt_fields = {
            "field_record": ("field_record", "field_record_sha256"),
            "brightfield_raw": ("brightfield_raw", "bf_raw_sha256"),
            "combined_mask": ("combined_mask", "combined_mask_sha256"),
            "nuclei_mask": ("nuclei_mask", "nuclei_mask_sha256"),
        }
        for asset_role, manifest_column, pointer, manifest_path, record_asset_path in assets:
            if manifest_path != record_asset_path:
                raise RuntimeError(
                    f"Resolved manifest/record path mismatch for {key} {asset_role}: "
                    f"manifest={manifest_path} record={record_asset_path}"
                )
            frozen = identities.get((key, asset_role))
            if frozen is None:
                raise RuntimeError(f"J0 input identity lacks {key} {asset_role}")
            expected_frozen = {
                "branch": branch,
                "include_nuclei_comparator": str(int(include_nuclei)),
                "manifest_column": manifest_column,
                "record_json_pointer": pointer,
                "manifest_path": str(manifest_path),
                "record_path": str(record_asset_path),
                "resolved_path": str(manifest_path),
                "alignment": "1",
            }
            for name, expected in expected_frozen.items():
                if frozen.get(name) != expected:
                    raise RuntimeError(
                        f"J0 input identity mismatch for {key} {asset_role} {name}: "
                        f"expected={expected!r} observed={frozen.get(name)!r}"
                    )
            path_field, hash_field = receipt_fields[asset_role]
            if receipt.get(path_field) != str(manifest_path):
                raise RuntimeError(
                    f"Feature receipt path differs from J0/manifest for {key} {asset_role}"
                )
            if receipt.get(hash_field) != frozen["file_sha256"]:
                raise RuntimeError(
                    f"Feature receipt SHA differs from J0 freeze for {key} {asset_role}: "
                    f"j0={frozen['file_sha256']} feature={receipt.get(hash_field)}"
                )
            observed_asset_sha = sha256_file(manifest_path)
            if observed_asset_sha != frozen["file_sha256"]:
                raise RuntimeError(
                    f"Input asset changed after J0 for {key} {asset_role}: "
                    f"frozen={frozen['file_sha256']} observed={observed_asset_sha}"
                )
            if str(manifest_path.stat().st_size) != frozen["file_size_bytes"]:
                raise RuntimeError(f"Input asset size changed after J0 for {key} {asset_role}")
            audit_rows.append(
                {
                    "key": key,
                    "asset_role": asset_role,
                    "path": str(manifest_path),
                    "sha256": frozen["file_sha256"],
                    "feature_receipt": str(receipt_path),
                    "feature_receipt_sha256": sha256_file(receipt_path),
                    "status": "aligned",
                }
            )

    audit_fields = [
        "key",
        "asset_role",
        "path",
        "sha256",
        "feature_receipt",
        "feature_receipt_sha256",
        "status",
    ]
    audit_bytes = tsv_payload(audit_fields, audit_rows)
    atomic_frozen(output_audit, audit_bytes)
    receipt = {
        "schema_version": FEATURE_GATE_SCHEMA,
        "status": "COMPLETE",
        "shadow_root": str(shadow),
        "branch": branch,
        "include_nuclei_comparator": include_nuclei,
        "field_manifest": str(field_manifest),
        "field_manifest_sha256": sha256_file(field_manifest),
        "input_identity_manifest": str(identity_manifest),
        "input_identity_manifest_sha256": sha256_file(identity_manifest),
        "feature_manifest": str(feature_manifest),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "audit": str(output_audit),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "field_count": len(fields_by_key),
        "asset_alignment_count": len(audit_rows),
        "mismatch_count": 0,
    }
    atomic_frozen(
        output_receipt,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"feature_input_alignment_complete=1 fields={len(fields_by_key)} audit={output_audit}")
    print(f"feature_input_alignment_receipt={output_receipt}")
    return 0


def train_receipt_identity(shadow: Path, project: Path, train_receipt: Path) -> dict[str, Any]:
    shadow = require_directory(shadow, "shadow root")
    project = require_inside(project, shadow, "representative project")
    train_receipt = require_inside(train_receipt, shadow, "train-stage receipt")
    receipt = read_json(train_receipt, "train-stage receipt")
    expected_scalars = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "stage": "train",
        "status": "complete",
        "returncode": 0,
        "shadow_root": str(shadow),
        "project": str(project),
        "project_sha256": sha256_file(project),
    }
    for name, expected in expected_scalars.items():
        if receipt.get(name) != expected:
            raise RuntimeError(
                f"Train receipt mismatch for {name}: expected={expected!r} observed={receipt.get(name)!r}"
            )
    command = receipt.get("command")
    identity = receipt.get("input_identity")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("Train receipt command must be a string array")
    if any(value in {"--check-config", "--dry-run"} for value in command):
        raise RuntimeError("Train receipt must certify an executed train stage, not check/dry-run mode")
    if not isinstance(identity, dict) or identity.get("schema_version") != "cellphenotypeannotator_stage_input_identity_v1":
        raise ValueError("Train receipt lacks the input-addressed identity")
    claimed_identity_sha = identity.get("sha256")
    identity_payload = dict(identity)
    identity_payload.pop("sha256", None)
    if not isinstance(claimed_identity_sha, str) or canonical_hash(identity_payload) != claimed_identity_sha:
        raise RuntimeError("Train receipt input_identity.sha256 is invalid")
    if receipt.get("command_sha256") != canonical_hash(command, ensure_ascii=False):
        raise RuntimeError("Train receipt command_sha256 is invalid")
    invocation = {"command": command, "input_identity": identity}
    if receipt.get("invocation_sha256") != canonical_hash(invocation, ensure_ascii=False):
        raise RuntimeError("Train receipt invocation_sha256 is invalid")

    project_config = load_mapping(project, "representative project")
    expected_files = {"dispatch_project": project}
    for role in ("classes_file", "cells_file", "features_file", "images_file"):
        value = project_config.get(role)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Representative project lacks {role}")
        expected_files[role] = resolve_project_path(project, value, f"project {role}")
    files = identity.get("files")
    if not isinstance(files, list) or not all(isinstance(value, dict) for value in files):
        raise ValueError("Train receipt input_identity.files must be an object array")
    for role, expected_path in expected_files.items():
        matches = [item for item in files if item.get("role") == role]
        if len(matches) != 1:
            raise RuntimeError(f"Train receipt must bind exactly one {role}")
        observed = matches[0]
        if observed.get("path") != str(expected_path) or observed.get("sha256") != sha256_file(expected_path):
            raise RuntimeError(f"Train receipt {role} differs from the current representative project")
        if observed.get("size_bytes") != expected_path.stat().st_size:
            raise RuntimeError(f"Train receipt {role} size differs from the current representative project")

    reviewed_labels = None
    for index, value in enumerate(command[:-1]):
        if value == "--reviewed-labels":
            if reviewed_labels is not None:
                raise RuntimeError("Train command supplies --reviewed-labels more than once")
            reviewed_labels = require_inside(Path(command[index + 1]), shadow, "reviewed labels")
    if reviewed_labels is None or not reviewed_labels.is_file():
        raise RuntimeError("Train receipt command lacks an in-shadow --reviewed-labels file")
    review_manifest = require_file(
        reviewed_labels.parent / "review_import_manifest.json", "review import manifest"
    )
    require_inside(review_manifest, shadow, "review import manifest")
    for required_path, label in (
        (reviewed_labels, "reviewed labels"),
        (review_manifest, "review import manifest"),
    ):
        matches = [
            item
            for item in files
            if str(item.get("role", "")).startswith("review_import_generation:")
            and item.get("path") == str(required_path)
        ]
        if len(matches) != 1 or matches[0].get("sha256") != sha256_file(required_path):
            raise RuntimeError(f"Train receipt does not bind the authoritative {label}")
    for log_role in ("stdout", "stderr"):
        log_path = require_inside(Path(str(receipt.get(f"{log_role}_log", ""))), shadow, f"train {log_role} log")
        if receipt.get(f"{log_role}_sha256") != sha256_file(log_path):
            raise RuntimeError(f"Train receipt {log_role} log hash is invalid")
    return {
        "receipt": receipt,
        "receipt_path": train_receipt,
        "receipt_sha256": sha256_file(train_receipt),
        "input_identity_sha256": claimed_identity_sha,
        "reviewed_labels": reviewed_labels,
        "reviewed_labels_sha256": sha256_file(reviewed_labels),
        "review_import_manifest": review_manifest,
        "review_import_manifest_sha256": sha256_file(review_manifest),
        "review_import_identity": read_json(review_manifest, "review import manifest").get("identity"),
    }


def validate_train_receipt_command(args: argparse.Namespace) -> int:
    identity = train_receipt_identity(
        args.shadow_root.expanduser(), args.project.expanduser(), args.train_receipt.expanduser()
    )
    print(f"train_receipt_verified=1 receipt={identity['receipt_path']}")
    print(f"train_receipt_sha256={identity['receipt_sha256']}")
    return 0


def trainable_class_ids(classes_path: Path) -> list[str]:
    fields, rows = read_tsv(classes_path, "classes file")
    required = {"class_id", "trainable"}
    if not required.issubset(fields):
        raise ValueError(f"Classes file lacks columns: {sorted(required-set(fields))}")
    class_ids = sorted(row["class_id"] for row in rows if row["trainable"].lower() == "true")
    if len(class_ids) < 2 or len(class_ids) != len(set(class_ids)):
        raise ValueError("Classes file must contain unique IDs for at least two trainable classes")
    return class_ids


def verify_model_acceptance(args: argparse.Namespace) -> int:
    shadow = require_directory(args.shadow_root, "shadow root")
    project = require_inside(args.project, shadow, "representative project")
    model_dir = require_inside(args.model_dir, shadow, "CPA model directory")
    train_receipt = require_inside(args.train_receipt, shadow, "train-stage receipt")
    feature_config = require_file(args.feature_config, "feature config")
    classes_file = require_file(args.classes_file, "classes source")
    preflight = require_inside(args.submission_preflight, shadow, "submission preflight")
    acceptance_path = require_inside(args.acceptance_receipt, shadow, "model acceptance receipt")
    expected_sha = args.expected_sha256.strip().lower()
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("--expected-sha256 must be one lowercase SHA-256 digest")
    observed_sha = sha256_file(acceptance_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Model acceptance receipt SHA mismatch: expected={expected_sha} observed={observed_sha}"
        )

    train = train_receipt_identity(shadow, project, train_receipt)
    project_config = load_mapping(project, "representative project")
    feature_payload = read_json(feature_config, "feature config")
    primary_features = feature_payload.get("primary_model_feature_columns")
    classifier = project_config.get("classifier")
    if not isinstance(primary_features, list) or not all(isinstance(value, str) for value in primary_features):
        raise ValueError("Feature config primary_model_feature_columns must be a string array")
    if not isinstance(classifier, dict) or classifier.get("feature_columns") != primary_features:
        raise RuntimeError("Representative project classifier features differ from frozen feature config")
    project_classes = resolve_project_path(
        project, str(project_config.get("classes_file", "")), "project classes_file"
    )
    if sha256_file(project_classes) != sha256_file(classes_file):
        raise RuntimeError("Representative project classes differ from the frozen source classes")

    model_manifest_path = require_file(model_dir / "model_manifest.json", "model manifest")
    model_manifest = read_json(model_manifest_path, "model manifest")
    current_class_ids = trainable_class_ids(classes_file)
    if model_manifest.get("project_id") != project_config.get("project_id"):
        raise RuntimeError("Model manifest project_id differs from the representative project")
    if model_manifest.get("feature_columns") != primary_features:
        raise RuntimeError("Model manifest feature_columns differ from the frozen allowlist")
    if model_manifest.get("class_ids") != current_class_ids:
        raise RuntimeError("Model manifest class_ids differ from the frozen trainable classes")
    expected_model_dir = (
        project.parent
        / str(project_config.get("runs_dir", "runs"))
        / str(model_manifest.get("project_id", ""))
        / str(model_manifest.get("run_id", ""))
        / "model"
        / str(model_manifest.get("model_id", ""))
    ).resolve()
    if model_dir != expected_model_dir:
        raise RuntimeError(
            "CPA model directory is not the representative project's canonical model generation: "
            f"expected={expected_model_dir} observed={model_dir}"
        )
    declared_artifacts = model_manifest.get("artifact_file_sha256")
    if not isinstance(declared_artifacts, dict) or set(declared_artifacts) != set(MODEL_GENERATION_ARTIFACTS):
        raise RuntimeError("Model manifest artifact hash set is incomplete or unexpected")
    for role, filename in MODEL_GENERATION_ARTIFACTS.items():
        artifact = require_file(model_dir / filename, f"model artifact {role}")
        if declared_artifacts.get(role) != sha256_file(artifact):
            raise RuntimeError(f"Model artifact hash mismatch for {role}")
    review_identity = train.get("review_import_identity")
    if not isinstance(review_identity, dict):
        raise RuntimeError("Train receipt review-import parent lacks an identity object")
    for field in ("project_id", "run_id", "class_config_sha256"):
        if review_identity.get(field) != model_manifest.get(field):
            raise RuntimeError(
                f"Model {field} differs from the train receipt's authoritative review import"
            )

    frozen_expected = {
        "feature_config_sha256": sha256_file(feature_config),
        "classes_sha256": sha256_file(classes_file),
    }
    for property_name, expected in frozen_expected.items():
        observed = preflight_scalar(preflight, property_name)
        if observed != expected:
            raise RuntimeError(
                f"Frozen submission preflight mismatch for {property_name}: expected={expected} observed={observed}"
            )

    acceptance = read_json(acceptance_path, "model acceptance receipt")
    expected_top = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": "ACCEPTED",
        "accepted": True,
        "shadow_root": str(shadow),
    }
    for name, expected in expected_top.items():
        if acceptance.get(name) != expected:
            raise RuntimeError(
                f"Model acceptance mismatch for {name}: expected={expected!r} observed={acceptance.get(name)!r}"
            )
    expected_sections = {
        "project": {
            "path": str(project),
            "sha256": sha256_file(project),
            "project_id": project_config.get("project_id"),
            "feature_columns": primary_features,
            "class_ids": current_class_ids,
        },
        "model": {
            "dir": str(model_dir),
            "manifest": str(model_manifest_path),
            "manifest_sha256": sha256_file(model_manifest_path),
            "model_id": model_manifest.get("model_id"),
            "project_id": model_manifest.get("project_id"),
            "run_id": model_manifest.get("run_id"),
            "class_config_sha256": model_manifest.get("class_config_sha256"),
            "classifier_config_sha256": model_manifest.get("classifier_config_sha256"),
            "feature_columns": primary_features,
            "class_ids": current_class_ids,
        },
        "train_receipt": {
            "path": str(train_receipt),
            "sha256": train["receipt_sha256"],
            "input_identity_sha256": train["input_identity_sha256"],
            "reviewed_labels_sha256": train["reviewed_labels_sha256"],
            "review_import_manifest_sha256": train["review_import_manifest_sha256"],
        },
        "frozen_inputs": {
            "submission_preflight": str(preflight),
            "submission_preflight_sha256": sha256_file(preflight),
            "feature_config": str(feature_config),
            "feature_config_sha256": sha256_file(feature_config),
            "classes_file": str(classes_file),
            "classes_sha256": sha256_file(classes_file),
        },
    }
    for section_name, fields in expected_sections.items():
        section = acceptance.get(section_name)
        if not isinstance(section, dict):
            raise RuntimeError(f"Model acceptance lacks {section_name}")
        for name, expected in fields.items():
            if section.get(name) != expected:
                raise RuntimeError(
                    f"Model acceptance {section_name}.{name} mismatch: "
                    f"expected={expected!r} observed={section.get(name)!r}"
                )
    project_section = acceptance["project"]
    for field in ("run_id", "class_config_sha256", "classifier_config_sha256"):
        if not isinstance(project_section.get(field), str) or not project_section[field]:
            raise RuntimeError(f"Model acceptance project.{field} is blank")
        if project_section[field] != model_manifest.get(field):
            raise RuntimeError(f"Accepted model {field} differs from the validated representative project")
    verification = acceptance.get("semantic_verification")
    required_checks = {
        "reference_generation_verified",
        "expected_model_directory_verified",
        "train_receipt_input_identity_verified",
        "model_train_parent_verified",
        "frozen_hashes_verified",
    }
    if not isinstance(verification, dict) or any(verification.get(name) is not True for name in required_checks):
        raise RuntimeError("Model acceptance semantic verification flags are incomplete")
    print(f"model_acceptance_verified=1 receipt={acceptance_path}")
    print(f"model_acceptance_sha256={observed_sha}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    feature = commands.add_parser("feature-inputs")
    feature.add_argument("--field-manifest", type=Path, required=True)
    feature.add_argument("--input-identity-manifest", type=Path, required=True)
    feature.add_argument("--feature-manifest", type=Path, required=True)
    feature.add_argument("--shadow-root", type=Path, required=True)
    feature.add_argument("--branch", choices=sorted(BRANCH_MASK), required=True)
    feature.add_argument("--include-nuclei-comparator", choices=("0", "1"), required=True)
    feature.add_argument("--output-receipt", type=Path, required=True)
    feature.add_argument("--output-audit", type=Path, required=True)
    feature.set_defaults(run=validate_feature_inputs)

    train = commands.add_parser("verify-train-receipt")
    train.add_argument("--shadow-root", type=Path, required=True)
    train.add_argument("--project", type=Path, required=True)
    train.add_argument("--train-receipt", type=Path, required=True)
    train.set_defaults(run=validate_train_receipt_command)

    model = commands.add_parser("verify-model-acceptance")
    model.add_argument("--shadow-root", type=Path, required=True)
    model.add_argument("--project", type=Path, required=True)
    model.add_argument("--model-dir", type=Path, required=True)
    model.add_argument("--train-receipt", type=Path, required=True)
    model.add_argument("--feature-config", type=Path, required=True)
    model.add_argument("--classes-file", type=Path, required=True)
    model.add_argument("--submission-preflight", type=Path, required=True)
    model.add_argument("--acceptance-receipt", type=Path, required=True)
    model.add_argument("--expected-sha256", required=True)
    model.set_defaults(run=verify_model_acceptance)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
