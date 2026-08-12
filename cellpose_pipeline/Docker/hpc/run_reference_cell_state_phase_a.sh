#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${PARENT_BROAD_SHADOW_ROOT:?PARENT_BROAD_SHADOW_ROOT is required}"
: "${EXPECTED_PARENT_CELL_COUNT:?EXPECTED_PARENT_CELL_COUNT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT is required}"
: "${PARENT_PROJECT:?PARENT_PROJECT is required}"
: "${PARENT_PROJECTION_INPUT_MANIFEST:?PARENT_PROJECTION_INPUT_MANIFEST is required}"
: "${PARENT_UMAP_MANIFEST:?PARENT_UMAP_MANIFEST is required}"
: "${EXPECTED_PARENT_PROJECT_SHA256:?EXPECTED_PARENT_PROJECT_SHA256 is required}"
: "${EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256:?EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256 is required}"
: "${EXPECTED_PARENT_UMAP_MANIFEST_SHA256:?EXPECTED_PARENT_UMAP_MANIFEST_SHA256 is required}"
: "${PARENT_IMAGES:?PARENT_IMAGES is required}"
: "${EXPECTED_PARENT_IMAGES_SHA256:?EXPECTED_PARENT_IMAGES_SHA256 is required}"

BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_ROOT="$DATASET_ROOT/Nuclei"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"

HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_HPC_CONTAINER_IMAGE" && \
   "$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT" ]] || {
  echo "Reference Phase A requires the pinned latest SIF and reference checkout paths" >&2
  exit 2
}
export HPC_CONTAINER_IMAGE CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT
export HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT

for root_path in \
  "$REFERENCE_SHADOW_ROOT" \
  "$PARENT_BROAD_SHADOW_ROOT" \
  "$BRIGHTFIELD_ROOT" \
  "$NUCLEI_ROOT" \
  "$COMBINED_MASK_ROOT" \
  "$CPA_REFERENCE_ROOT"; do
  [[ "$root_path" == /* && -d "$root_path" && ! -L "$root_path" ]] || {
    echo "Required reference-cell-state root is not an absolute real directory: $root_path" >&2
    exit 2
  }
done
[[ "$EXPECTED_PARENT_CELL_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_PARENT_CELL_COUNT must be a positive integer" >&2
  exit 2
}

append_bind() {
  local bind="$1"
  case ",${HPC_CONTAINER_BINDS:-}," in
    *",$bind,"*) ;;
    *) HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:+$HPC_CONTAINER_BINDS,}$bind" ;;
  esac
}
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
[[ "${HPC_CONTAINER_BINDS:-}" == "$expected_binds" ]] || {
  echo "HPC_CONTAINER_BINDS differs from the frozen reference-cell-state allowlist" >&2
  exit 2
}
HPC_CONTAINER_BINDS="$expected_binds"
expected_forward_prefixes="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES"
[[ "${HPC_CONTAINER_FORWARD_PREFIXES:-}" == "$expected_forward_prefixes" ]] || {
  echo "HPC_CONTAINER_FORWARD_PREFIXES differs from the frozen reference-cell-state allowlist" >&2
  exit 2
}
HPC_CONTAINER_FORWARD_PREFIXES="$expected_forward_prefixes"
export HPC_CONTAINER_BINDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" \
  "$EXPECTED_HPC_CONTAINER_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT

DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
PROJECT_BUILDER="$PROJECT_DIR/cellpose_pipeline/scripts/26_prepare_reference_cell_state_project.py"
CPA_STAGE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py"
MORPHOLOGY_WORKSPACE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/27_build_reference_morphology_workspace.py"
PROJECT_FILE="$REFERENCE_SHADOW_ROOT/projection_input/representative_umap/project.yml"
PHASE_ROOT="$REFERENCE_SHADOW_ROOT/workflow_status/reference_cell_state_phase_a"
PHASE_RECEIPT="$PHASE_ROOT/PHASE_A_COMPLETE.tsv"

for required_path in \
  "$DEPENDENCY_LOCK" \
  "$PROJECT_BUILDER" \
  "$CPA_STAGE_SCRIPT" \
  "$MORPHOLOGY_WORKSPACE_SCRIPT" \
  "$PARENT_PROJECT"; do
  [[ -f "$required_path" && ! -L "$required_path" ]] || {
    echo "Required reference-cell-state Phase A file is unavailable: $required_path" >&2
    exit 2
  }
done
for frozen_pair in \
  "$PARENT_PROJECT:$EXPECTED_PARENT_PROJECT_SHA256" \
  "$PARENT_PROJECTION_INPUT_MANIFEST:$EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256" \
  "$PARENT_UMAP_MANIFEST:$EXPECTED_PARENT_UMAP_MANIFEST_SHA256" \
  "$PARENT_IMAGES:$EXPECTED_PARENT_IMAGES_SHA256"; do
  frozen_path="${frozen_pair%:*}"
  frozen_expected="${frozen_pair##*:}"
  frozen_observed="$(sha256sum "$frozen_path")"
  frozen_observed="${frozen_observed%%[[:space:]]*}"
  [[ "$frozen_expected" =~ ^[0-9a-f]{64}$ && "$frozen_observed" == "$frozen_expected" ]] || {
    echo "Parent reference input changed before Phase A: $frozen_path" >&2
    exit 2
  }
done
[[ ! -e "$PHASE_RECEIPT" && ! -L "$PHASE_RECEIPT" ]] || {
  echo "Refusing to replace an existing reference-cell-state Phase A receipt: $PHASE_RECEIPT" >&2
  exit 2
}

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p "$PHASE_ROOT"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "reference_cell_state_phase_a_started_utc=$started_utc"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "parent_broad_shadow_root=$PARENT_BROAD_SHADOW_ROOT"
echo "expected_parent_cell_count=$EXPECTED_PARENT_CELL_COUNT"
echo "reference_training_inputs=brightfield_combined_mask_shape_only"
echo "reference_review_support=nuclei"
echo "forbidden_inputs=dead_combined_rgb_current_classification_current_predictions_trajectory"
echo "site_share_mount_disabled=1"

hpc_apptainer_exec python -I - \
  "$DATASET_ROOT" "$REFERENCE_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT" <<'PY'
from pathlib import Path
import sys

dataset, reference_root, parent_broad_root = map(Path, sys.argv[1:])
for forbidden in (dataset / "Dead", dataset / "Combined"):
    if forbidden.exists():
        raise SystemExit(f"forbidden channel remained visible in SIF: {forbidden}")
results = dataset.parent / "results"
if results.exists():
    allowed = {reference_root.resolve(), parent_broad_root.resolve()}
    current_roots = [
        path
        for path in results.iterdir()
        if path.name.startswith("classification_") and path.resolve() not in allowed
    ]
    if current_roots:
        raise SystemExit(
            f"current classification roots remained visible in SIF: {current_roots[:3]}"
        )
print("reference_container_blinding=PASS")
PY

hpc_apptainer_exec python -I "$PROJECT_BUILDER" \
  --parent-project "$PARENT_PROJECT" \
  --parent-shadow-root "$PARENT_BROAD_SHADOW_ROOT" \
  --reference-shadow-root "$REFERENCE_SHADOW_ROOT" \
  --expected-cell-count "$EXPECTED_PARENT_CELL_COUNT" \
  --seed "${REFERENCE_CELL_STATE_SEED:-20260812}"

hpc_apptainer_exec python -I - \
  "$PROJECT_FILE" "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT" <<'PY'
import csv
import json
import os
import pathlib
import sys

project_file, brightfield_root, nuclei_root, mask_root = map(pathlib.Path, sys.argv[1:])
project = json.loads(project_file.read_text(encoding="utf-8"))
images_path = (project_file.parent / project["images_file"]).resolve()
allowed = {
    "brightfield": pathlib.Path(brightfield_root).resolve(),
    "nuclei": pathlib.Path(nuclei_root).resolve(),
}
mask_root = pathlib.Path(mask_root).resolve()
seen = set()
with images_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row_number, row in enumerate(reader, start=2):
        channel = row.get("channel_id", "")
        if channel not in allowed:
            raise SystemExit(f"forbidden review channel at row {row_number}: {channel!r}")
        image = pathlib.Path(row.get("image_path", "")).resolve(strict=True)
        mask = pathlib.Path(row.get("mask_path", "")).resolve(strict=True)
        if not image.is_relative_to(allowed[channel]):
            raise SystemExit(f"review image escaped frozen channel root at row {row_number}: {image}")
        if not mask.is_relative_to(mask_root):
            raise SystemExit(f"review mask escaped Combined segmentation root at row {row_number}: {mask}")
        identity = (row.get("image_id", ""), channel)
        if identity in seen:
            raise SystemExit(f"duplicate image/channel identity at row {row_number}: {identity}")
        seen.add(identity)
if not seen:
    raise SystemExit("reference images.tsv contains no review images")
print(f"reference_review_path_contract=PASS image_channel_rows={len(seen)}")
print("dead_channel_visible=0")
print("combined_rgb_visible=0")
PY

for stage in validate umap annotate; do
  stage_args=(
    "$CPA_STAGE_SCRIPT"
    --stage "$stage"
    --project "$PROJECT_FILE"
    --shadow-root "$REFERENCE_SHADOW_ROOT"
    --reference-root "$CPA_REFERENCE_ROOT"
    --dependency-lock "$DEPENDENCY_LOCK"
    --rscript Rscript
  )
  if [[ "$stage" == "validate" ]]; then
    stage_args+=(--validate-stage umap)
  fi
  hpc_apptainer_exec python -I "${stage_args[@]}"
done

annotation_manifests=()
while IFS= read -r annotation_manifest; do
  annotation_manifests+=("$annotation_manifest")
done < <(
  find "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap/runs" \
    -type f -name annotation_manifest.json -print | LC_ALL=C sort
)
[[ "${#annotation_manifests[@]}" -eq 1 ]] || {
  echo "Expected exactly one annotation_manifest.json after reference Phase A; observed=${#annotation_manifests[@]}" >&2
  printf 'annotation_manifest_candidate=%s\n' "${annotation_manifests[@]:-}" >&2
  exit 2
}
ANNOTATION_DIR="$(dirname "${annotation_manifests[0]}")"
ANNOTATION_PAYLOAD="$ANNOTATION_DIR/annotation_payload.json"
ANNOTATION_HTML="$ANNOTATION_DIR/annotation.html"
for annotation_file in "$ANNOTATION_PAYLOAD" "$ANNOTATION_HTML"; do
  [[ -s "$annotation_file" && ! -L "$annotation_file" ]] || {
    echo "Required immutable annotation artifact is unavailable: $annotation_file" >&2
    exit 2
  }
done

hpc_apptainer_exec python -I "$MORPHOLOGY_WORKSPACE_SCRIPT" \
  --project "$PROJECT_FILE" \
  --shadow-root "$REFERENCE_SHADOW_ROOT" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output-dir "$REFERENCE_SHADOW_ROOT/morphology_reference" \
  --max-representatives "${REFERENCE_MAX_REPRESENTATIVES:-300}" \
  --seed "${REFERENCE_CELL_STATE_SEED:-20260812}"

WORKSPACE_HTML="$REFERENCE_SHADOW_ROOT/morphology_reference/annotation_workspace.html"
WORKSPACE_MANIFEST="$REFERENCE_SHADOW_ROOT/morphology_reference/overlay_manifest.json"
for workspace_file in "$WORKSPACE_HTML" "$WORKSPACE_MANIFEST"; do
  [[ -s "$workspace_file" && ! -L "$workspace_file" ]] || {
    echo "Required morphology-reference artifact is unavailable: $workspace_file" >&2
    exit 2
  }
done

for frozen_pair in \
  "$PARENT_PROJECT:$EXPECTED_PARENT_PROJECT_SHA256" \
  "$PARENT_PROJECTION_INPUT_MANIFEST:$EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256" \
  "$PARENT_UMAP_MANIFEST:$EXPECTED_PARENT_UMAP_MANIFEST_SHA256" \
  "$PARENT_IMAGES:$EXPECTED_PARENT_IMAGES_SHA256"; do
  frozen_path="${frozen_pair%:*}"
  frozen_expected="${frozen_pair##*:}"
  frozen_observed="$(sha256sum "$frozen_path")"
  frozen_observed="${frozen_observed%%[[:space:]]*}"
  [[ "$frozen_observed" == "$frozen_expected" ]] || {
    echo "Parent reference input changed during Phase A: $frozen_path" >&2
    exit 2
  }
done

project_sha256="$(sha256sum "$PROJECT_FILE")"; project_sha256="${project_sha256%%[[:space:]]*}"
annotation_manifest_sha256="$(sha256sum "${annotation_manifests[0]}")"; annotation_manifest_sha256="${annotation_manifest_sha256%%[[:space:]]*}"
annotation_payload_sha256="$(sha256sum "$ANNOTATION_PAYLOAD")"; annotation_payload_sha256="${annotation_payload_sha256%%[[:space:]]*}"
annotation_html_sha256="$(sha256sum "$ANNOTATION_HTML")"; annotation_html_sha256="${annotation_html_sha256%%[[:space:]]*}"
workspace_html_sha256="$(sha256sum "$WORKSPACE_HTML")"; workspace_html_sha256="${workspace_html_sha256%%[[:space:]]*}"
workspace_manifest_sha256="$(sha256sum "$WORKSPACE_MANIFEST")"; workspace_manifest_sha256="${workspace_manifest_sha256%%[[:space:]]*}"
for final_sha in "$project_sha256" "$annotation_manifest_sha256" "$annotation_payload_sha256" "$annotation_html_sha256" "$workspace_html_sha256" "$workspace_manifest_sha256"; do
  [[ "$final_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Unable to freeze one Phase A output SHA-256" >&2; exit 2; }
done

finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
receipt_tmp="$PHASE_ROOT/.PHASE_A_COMPLETE.tsv.tmp.$$"
{
  printf 'property\tvalue\n'
  printf 'schema_version\treference_cell_state_phase_a_receipt_v1\n'
  printf 'status\tCOMPLETE\n'
  printf 'started_utc\t%s\n' "$started_utc"
  printf 'finished_utc\t%s\n' "$finished_utc"
  printf 'reference_shadow_root\t%s\n' "$REFERENCE_SHADOW_ROOT"
  printf 'parent_broad_shadow_root\t%s\n' "$PARENT_BROAD_SHADOW_ROOT"
  printf 'expected_parent_cell_count\t%s\n' "$EXPECTED_PARENT_CELL_COUNT"
  printf 'project_file\t%s\n' "$PROJECT_FILE"
  printf 'project_sha256\t%s\n' "$project_sha256"
  printf 'annotation_dir\t%s\n' "$ANNOTATION_DIR"
  printf 'annotation_manifest_sha256\t%s\n' "$annotation_manifest_sha256"
  printf 'annotation_payload_sha256\t%s\n' "$annotation_payload_sha256"
  printf 'annotation_html_sha256\t%s\n' "$annotation_html_sha256"
  printf 'morphology_workspace\t%s\n' "$WORKSPACE_HTML"
  printf 'morphology_workspace_sha256\t%s\n' "$workspace_html_sha256"
  printf 'morphology_reference_manifest_sha256\t%s\n' "$workspace_manifest_sha256"
  printf 'hpc_container_image\t%s\n' "$HPC_CONTAINER_IMAGE"
  printf 'hpc_container_sha256\t%s\n' "$EXPECTED_HPC_CONTAINER_SHA256"
  printf 'cpa_reference_root\t%s\n' "$CPA_REFERENCE_ROOT"
  printf 'human_barrier\tregion_submission_required\n'
  printf 'region_submission_expected_path\t%s\n' "$ANNOTATION_DIR/region_submission.json"
  printf 'current_classification_write_performed\tfalse\n'
  printf 'current_classification_read_performed\tfalse\n'
  printf 'dead_channel_read_performed\tfalse\n'
  printf 'site_share_mount_disabled\ttrue\n'
} > "$receipt_tmp"
mv "$receipt_tmp" "$PHASE_RECEIPT"

echo "reference_cell_state_phase_a_complete=1"
echo "morphology_workspace=$WORKSPACE_HTML"
echo "annotation_dir=$ANNOTATION_DIR"
echo "region_submission_expected_path=$ANNOTATION_DIR/region_submission.json"
echo "human_barrier=region_submission_required"
