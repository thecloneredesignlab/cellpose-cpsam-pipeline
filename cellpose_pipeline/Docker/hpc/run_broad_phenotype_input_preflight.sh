#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
LEGACY_PATH_PREFIX="${LEGACY_PATH_PREFIX:-/share/lab_crd/lab_crd/}"
LIVE_PATH_PREFIX="${LIVE_PATH_PREFIX:-/share/lab_crd/}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export HPC_PROJECT_ROOT
: "${SOURCE_FIELD_MANIFEST_FILE:?SOURCE_FIELD_MANIFEST_FILE is required}"
: "${TASK_LIST:?TASK_LIST is required}"
: "${RESOLVED_FIELD_MANIFEST_DIR:?RESOLVED_FIELD_MANIFEST_DIR is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT is required}"
: "${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT is required}"
: "${BRANCH:?BRANCH is required}"
: "${INCLUDE_NUCLEI_COMPARATOR:?INCLUDE_NUCLEI_COMPARATOR is required}"
: "${STALE_DATASET_ROOT:?STALE_DATASET_ROOT is required for exact path migration}"
: "${STALE_SEGMENTATION_ROOT:?STALE_SEGMENTATION_ROOT is required for exact path migration}"
case "$BRANCH" in original|nucleated) ;; *) echo "BRANCH must be original or nucleated" >&2; exit 2 ;; esac
case "$INCLUDE_NUCLEI_COMPARATOR" in 0|1) ;; *) echo "INCLUDE_NUCLEI_COMPARATOR must be 0 or 1" >&2; exit 2 ;; esac

append_bind() {
  local bind="$1"
  case ",${HPC_CONTAINER_BINDS:-}," in
    *",$bind,"*) ;;
    *) HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:+$HPC_CONTAINER_BINDS,}$bind" ;;
  esac
}
source_manifest_parent="$(dirname "$SOURCE_FIELD_MANIFEST_FILE")"
append_bind "$source_manifest_parent:$source_manifest_parent:ro"
append_bind "$(dirname "$RESOLVED_FIELD_MANIFEST_DIR"):$(dirname "$RESOLVED_FIELD_MANIFEST_DIR")"
append_bind "$DATASET_ROOT:$DATASET_ROOT:ro"
append_bind "$SOURCE_SEGMENTATION_ROOT:$SOURCE_SEGMENTATION_ROOT:ro"
append_bind "$CLASSIFICATION_ROOT:$CLASSIFICATION_ROOT:ro"
export HPC_CONTAINER_BINDS

source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"
[[ -s "$SOURCE_FIELD_MANIFEST_FILE" && -s "$TASK_LIST" ]] || {
  echo "Stage 06 source manifest or task list is unavailable" >&2
  exit 2
}
[[ ! -e "$RESOLVED_FIELD_MANIFEST_DIR" ]] || {
  echo "Refusing to replace an existing resolved Stage 06 manifest: $RESOLVED_FIELD_MANIFEST_DIR" >&2
  exit 2
}
staging_dir="${RESOLVED_FIELD_MANIFEST_DIR}.tmp.${SLURM_JOB_ID:-manual}.$$"
[[ "$staging_dir" == "${RESOLVED_FIELD_MANIFEST_DIR}.tmp."* && ! -e "$staging_dir" ]] || {
  echo "Unsafe or existing preflight staging directory: $staging_dir" >&2
  exit 2
}
cleanup_staging() {
  if [[ -d "$staging_dir" && "$staging_dir" == "${RESOLVED_FIELD_MANIFEST_DIR}.tmp."* ]]; then
    rm -rf -- "$staging_dir"
  fi
}
trap cleanup_staging EXIT

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "source_field_manifest_file=$SOURCE_FIELD_MANIFEST_FILE"
echo "resolved_field_manifest_dir=$RESOLVED_FIELD_MANIFEST_DIR"
echo "legacy_path_prefix=$LEGACY_PATH_PREFIX"
echo "live_path_prefix=$LIVE_PATH_PREFIX"
echo "source_segmentation_root=$SOURCE_SEGMENTATION_ROOT"
echo "stale_dataset_root=$STALE_DATASET_ROOT"
echo "stale_segmentation_root=$STALE_SEGMENTATION_ROOT"
echo "branch=$BRANCH"
echo "include_nuclei_comparator=$INCLUDE_NUCLEI_COMPARATOR"

python -I -c 'import sys; print("container_python=" + sys.executable)'
python -I - "$SOURCE_FIELD_MANIFEST_FILE" "$TASK_LIST" "$staging_dir" "$RESOLVED_FIELD_MANIFEST_DIR" "$LEGACY_PATH_PREFIX" "$LIVE_PATH_PREFIX" "$DATASET_ROOT" "$SOURCE_SEGMENTATION_ROOT" "$STALE_DATASET_ROOT" "$STALE_SEGMENTATION_ROOT" "$BRANCH" "$INCLUDE_NUCLEI_COMPARATOR" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

(
    source_manifest, task_file, output_root, final_root, legacy_prefix, live_prefix,
    dataset_root, segmentation_root, stale_dataset_root, stale_segmentation_root,
    branch, include_nuclei_comparator,
) = sys.argv[1:]
source_manifest = Path(source_manifest).resolve()
task_file = Path(task_file).resolve()
output_root = Path(output_root).resolve()
final_root = Path(final_root).resolve()
dataset_root = Path(dataset_root).resolve()
segmentation_root = Path(segmentation_root).resolve()
include_nuclei_comparator = include_nuclei_comparator == "1"
keys = [line.strip() for line in task_file.read_text().splitlines() if line.strip()]
if not keys or len(keys) != len(set(keys)):
    raise SystemExit("Task list must be nonempty and contain unique keys")
selected = set(keys)

digest_cache = {}
def digest(path):
    path = Path(path).resolve()
    if path in digest_cache:
        return digest_cache[path]
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    digest_cache[path] = h.hexdigest()
    return digest_cache[path]

def candidate_paths(path):
    raw_abs = str(path)
    candidates = [(path, "identity")]
    if raw_abs.startswith(stale_dataset_root.rstrip("/") + "/"):
        suffix = raw_abs[len(stale_dataset_root.rstrip("/")):].lstrip("/")
        candidates.append((dataset_root / suffix, "exact_stale_dataset_root_to_declared_dataset_root"))
    if raw_abs.startswith(stale_segmentation_root.rstrip("/") + "/"):
        suffix = raw_abs[len(stale_segmentation_root.rstrip("/")):].lstrip("/")
        candidates.append((segmentation_root / suffix, "exact_stale_segmentation_root_to_declared_source_root"))
    if raw_abs.startswith(legacy_prefix):
        candidates.append((Path(live_prefix + raw_abs[len(legacy_prefix):]), "legacy_double_prefix_to_live_single_prefix"))
    unique = []
    seen = set()
    for candidate, rule in candidates:
        normalized = str(candidate)
        if normalized not in seen:
            unique.append((candidate, rule)); seen.add(normalized)
    return unique

def resolve_path(value, anchor, key, pointer, audit, required):
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise SystemExit(f"Required path is blank for {key} {pointer}")
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    raw_abs = str(path)
    candidates = candidate_paths(path)
    existing = []
    for candidate, rule in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if all(resolved != item[0] for item in existing):
                existing.append((resolved, rule))
    if len(existing) > 1 or (required and len(existing) != 1):
        raise SystemExit(
            f"Path resolution ambiguity for {key} {pointer}: existing_candidates={len(existing)} raw={raw_abs}"
        )
    selected_path = existing[0][0] if existing else None
    selected_rule = existing[0][1] if existing else ""
    for candidate, rule in candidates:
        candidate_resolved = candidate.resolve() if candidate.exists() else candidate
        selected = selected_path is not None and candidate_resolved == selected_path and rule == selected_rule
        audit.append({
            "key": key,
            "json_pointer": pointer,
            "original_path": raw_abs,
            "candidate_path": str(candidate),
            "resolution_rule": rule,
            "selected": str(int(selected)),
            "status": "selected" if selected else "candidate_exists_not_selected" if candidate.exists() else "missing_optional" if not required else "missing_required",
            "resolved_path": str(selected_path) if selected else "",
            "file_size_bytes": selected_path.stat().st_size if selected else "",
            "file_sha256": digest(selected_path) if selected else "",
            "existing_candidate_count": str(len(existing)),
            "ambiguity": "0",
        })
    return str(selected_path) if selected_path is not None else ""

with source_manifest.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if not reader.fieldnames or "key" not in reader.fieldnames:
        raise SystemExit("Stage 06 manifest must contain a key column")
    fieldnames = list(reader.fieldnames)
    source_rows = {}
    for row in reader:
        key = str(row.get("key", "")).strip()
        if key in selected:
            if key in source_rows:
                raise SystemExit(f"Duplicate Stage 06 manifest key: {key}")
            source_rows[key] = row
if set(source_rows) != selected:
    missing = sorted(selected - set(source_rows))
    raise SystemExit(f"Task keys missing from Stage 06 manifest: {missing[:10]}")

output_root.mkdir(parents=True, exist_ok=False)
audit_rows, output_rows, record_hash_rows, identity_rows = [], [], [], []
profile_path_fields = {
    "Combined": ("raw", "original_mask", "nucleated_mask"),
    "Brightfield": ("raw", "original_mask", "nucleated_mask"),
    "Dead": ("raw", "mask"),
    "Nuclei": ("raw", "extent_mask", "core_mask"),
}
manifest_to_record = {
    "combined_raw": ("Combined", "raw"),
    "combined_mask": ("Combined", "original_mask"),
    "nucleated_combined_mask": ("Combined", "nucleated_mask"),
    "brightfield_raw": ("Brightfield", "raw"),
    "brightfield_mask": ("Brightfield", "original_mask"),
    "nucleated_brightfield_mask": ("Brightfield", "nucleated_mask"),
    "dead_raw": ("Dead", "raw"),
    "dead_mask": ("Dead", "mask"),
    "nuclei_raw": ("Nuclei", "raw"),
    "nuclei_extent_mask": ("Nuclei", "extent_mask"),
    "nuclei_core_mask": ("Nuclei", "core_mask"),
}
selected_mask_column = "combined_mask" if branch == "original" else "nucleated_combined_mask"
selected_mask_field = "original_mask" if branch == "original" else "nucleated_mask"
required_manifest_columns = {"combined_raw", "brightfield_raw", "dead_raw", "nuclei_raw", selected_mask_column}
for key in keys:
    source_row = source_rows[key]
    well = key.split("_", 1)[0]
    standard_record = source_manifest.parent / "records" / well / f"{key}.json"
    record_candidates = [(standard_record, "canonical_source_manifest_record")]
    raw_record = str(source_row.get("record_json", "")).strip()
    if raw_record:
        raw_record_path = Path(raw_record)
        if not raw_record_path.is_absolute():
            raw_record_path = source_manifest.parent / raw_record_path
        record_candidates.extend(candidate_paths(raw_record_path))
    existing_records = []
    for candidate, rule in record_candidates:
        if candidate.exists() and candidate.resolve() not in existing_records:
            existing_records.append(candidate.resolve())
    if len(existing_records) != 1:
        raise SystemExit(f"Record resolution ambiguity for {key}: existing_candidates={len(existing_records)}")
    source_record = existing_records[0]
    seen_record_candidates = set()
    selected_record_emitted = False
    for candidate, rule in record_candidates:
        candidate_identity = (str(candidate), rule)
        if candidate_identity in seen_record_candidates:
            continue
        seen_record_candidates.add(candidate_identity)
        candidate_resolved = candidate.resolve() if candidate.exists() else candidate
        selected_record = (
            not selected_record_emitted
            and candidate.exists()
            and candidate_resolved == source_record
        )
        selected_record_emitted = selected_record_emitted or selected_record
        audit_rows.append({
            "key": key,
            "json_pointer": "/manifest/record_json",
            "original_path": raw_record or str(standard_record),
            "candidate_path": str(candidate),
            "resolution_rule": rule,
            "selected": str(int(selected_record)),
            "status": "selected" if selected_record else "candidate_exists_not_selected" if candidate.exists() else "missing_required",
            "resolved_path": str(source_record) if selected_record else "",
            "file_size_bytes": source_record.stat().st_size if selected_record else "",
            "file_sha256": digest(source_record) if selected_record else "",
            "existing_candidate_count": str(len(existing_records)),
            "ambiguity": "0",
        })
    record = json.loads(source_record.read_text(encoding="utf-8"))
    if str(record.get("key", "")) != key:
        raise SystemExit(f"Record key mismatch: expected={key} observed={record.get('key')}")
    profiles = record.get("profiles")
    if not isinstance(profiles, dict):
        raise SystemExit(f"Record profiles must be a mapping: {source_record}")
    for profile_name, fields in profile_path_fields.items():
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            continue
        for field in fields:
            value = profile.get(field)
            if isinstance(value, str) and value.strip():
                required = field == "raw" or (profile_name == "Combined" and field == selected_mask_field)
                profile[field] = resolve_path(
                    value, source_record.parent, key, f"/profiles/{profile_name}/{field}", audit_rows, required
                )
    output_record = output_root / "records" / well / f"{key}.json"
    final_record = final_root / "records" / well / f"{key}.json"
    output_record.parent.mkdir(parents=True, exist_ok=True)
    output_row = dict(source_row)
    output_row["record_json"] = str(final_record)
    for column, value in list(output_row.items()):
        if column in {"key", "record_json"} or not str(value or "").strip():
            continue
        output_row[column] = resolve_path(
            value, source_manifest.parent, key, f"/manifest/{column}", audit_rows,
            column in required_manifest_columns,
        )

    for column, (profile_name, profile_field) in manifest_to_record.items():
        manifest_value = str(output_row.get(column, "") or "")
        profile = profiles.get(profile_name)
        record_value = str(profile.get(profile_field, "") or "") if isinstance(profile, dict) else ""
        if manifest_value != record_value:
            raise SystemExit(
                f"Resolved manifest/record mismatch for {key} {column}: manifest={manifest_value} record={record_value}"
            )

    nuclei_role = ""
    if include_nuclei_comparator:
        nuclei = profiles.get("Nuclei")
        if not isinstance(nuclei, dict):
            raise SystemExit(f"Nuclei comparator enabled but Nuclei profile missing: {key}")
        if str(nuclei.get("core_mask", "")).strip():
            nuclei_role, nuclei_column, nuclei_field = "nuclei_mask", "nuclei_core_mask", "core_mask"
        elif str(nuclei.get("extent_mask", "")).strip():
            nuclei_role, nuclei_column, nuclei_field = "nuclei_mask", "nuclei_extent_mask", "extent_mask"
        else:
            raise SystemExit(f"Nuclei comparator enabled but core/extent masks are unavailable: {key}")
        if str(output_row.get(nuclei_column, "")) != str(nuclei[nuclei_field]):
            raise SystemExit(f"Selected Nuclei comparator manifest/record mismatch for {key}")

    output_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_rows.append(output_row)
    record_hash_rows.append({
        "key": key,
        "source_record": str(source_record),
        "source_record_sha256": digest(source_record),
        "resolved_record": str(final_record),
        "resolved_record_sha256": digest(output_record),
    })
    identity_assets = [
        ("field_record", "record_json", "/", final_record, final_record, digest(output_record), output_record.stat().st_size),
        ("brightfield_raw", "brightfield_raw", "/profiles/Brightfield/raw", Path(output_row["brightfield_raw"]), Path(profiles["Brightfield"]["raw"]), digest(Path(output_row["brightfield_raw"])), Path(output_row["brightfield_raw"]).stat().st_size),
        ("combined_mask", selected_mask_column, f"/profiles/Combined/{selected_mask_field}", Path(output_row[selected_mask_column]), Path(profiles["Combined"][selected_mask_field]), digest(Path(output_row[selected_mask_column])), Path(output_row[selected_mask_column]).stat().st_size),
    ]
    if nuclei_role:
        identity_assets.append(("nuclei_mask", nuclei_column, f"/profiles/Nuclei/{nuclei_field}", Path(output_row[nuclei_column]), Path(profiles["Nuclei"][nuclei_field]), digest(Path(output_row[nuclei_column])), Path(output_row[nuclei_column]).stat().st_size))
    for role, column, pointer, manifest_path, record_path, asset_sha, asset_size in identity_assets:
        identity_rows.append({
            "key": key,
            "branch": branch,
            "include_nuclei_comparator": str(int(include_nuclei_comparator)),
            "asset_role": role,
            "manifest_column": column,
            "record_json_pointer": pointer,
            "manifest_path": str(manifest_path),
            "record_path": str(record_path),
            "resolved_path": str(manifest_path),
            "file_size_bytes": asset_size,
            "file_sha256": asset_sha,
            "alignment": "1",
        })

resolved_manifest = output_root / "field_manifest.tsv"
with resolved_manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(output_rows)
audit_path = output_root / "path_resolution_manifest.tsv"
audit_fields = ["key", "json_pointer", "original_path", "candidate_path", "resolution_rule", "selected", "status", "resolved_path", "file_size_bytes", "file_sha256", "existing_candidate_count", "ambiguity"]
with audit_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=audit_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(audit_rows)
hash_path = output_root / "record_hash_manifest.tsv"
with hash_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(record_hash_rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(record_hash_rows)
identity_path = output_root / "input_identity_manifest.tsv"
identity_fields = ["key", "branch", "include_nuclei_comparator", "asset_role", "manifest_column", "record_json_pointer", "manifest_path", "record_path", "resolved_path", "file_size_bytes", "file_sha256", "alignment"]
with identity_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=identity_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(identity_rows)
receipt = {
    "schema_version": "broad_phenotype_stage06_path_resolution_v2",
    "source_manifest": str(source_manifest),
    "source_manifest_sha256": digest(source_manifest),
    "resolved_manifest": str(final_root / "field_manifest.tsv"),
    "resolved_manifest_sha256": digest(resolved_manifest),
    "input_identity_manifest": str(final_root / "input_identity_manifest.tsv"),
    "input_identity_manifest_sha256": digest(identity_path),
    "path_resolution_manifest_sha256": digest(audit_path),
    "record_hash_manifest_sha256": digest(hash_path),
    "task_list": str(task_file),
    "task_count": len(keys),
    "legacy_prefix": legacy_prefix,
    "live_prefix": live_prefix,
    "dataset_root": str(dataset_root),
    "source_segmentation_root": str(segmentation_root),
    "stale_dataset_root": stale_dataset_root,
    "stale_segmentation_root": stale_segmentation_root,
    "branch": branch,
    "include_nuclei_comparator": include_nuclei_comparator,
    "path_rows": len(audit_rows),
    "input_identity_rows": len(identity_rows),
    "ambiguity_count": 0,
    "source_write_performed": False,
}
(output_root / "path_resolution_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(f"resolved_field_manifest={final_root / 'field_manifest.tsv'}")
print(f"path_resolution_manifest={final_root / 'path_resolution_manifest.tsv'}")
print(f"input_identity_manifest={final_root / 'input_identity_manifest.tsv'}")
print("path_resolution_ambiguity_count=0")
PY
mv "$staging_dir" "$RESOLVED_FIELD_MANIFEST_DIR"
trap - EXIT
