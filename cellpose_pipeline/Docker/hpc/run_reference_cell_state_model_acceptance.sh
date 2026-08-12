#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"

HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_HPC_CONTAINER_IMAGE" ]] || {
  echo "HPC_CONTAINER_IMAGE must equal the frozen latest SIF path: $DEFAULT_HPC_CONTAINER_IMAGE" >&2
  exit 2
}
[[ "$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT" ]] || {
  echo "CPA_REFERENCE_ROOT must equal the frozen read-only checkout: $DEFAULT_CPA_REFERENCE_ROOT" >&2
  exit 2
}
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
case "${HPC_CONTAINER_NO_MOUNT+x}:${HPC_CONTAINER_NO_MOUNT:-}" in
  :|x:/share) ;;
  *) echo "HPC_CONTAINER_NO_MOUNT must be exactly /share for reference-cell-state work" >&2; exit 2 ;;
esac
HPC_CONTAINER_NO_MOUNT=/share
export HPC_CONTAINER_IMAGE CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT
export HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
PROJECT_FILE="${PROJECT_FILE:-$REFERENCE_SHADOW_ROOT/projection_input/representative_umap/project.yml}"
PARENT_IMPORT_MANIFEST="${PARENT_IMPORT_MANIFEST:-$(dirname "$PROJECT_FILE")/parent_import_manifest.json}"
: "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"
: "${CPA_TRAIN_RECEIPT:?CPA_TRAIN_RECEIPT is required}"
MODEL_ACCEPTANCE_RECEIPT="${MODEL_ACCEPTANCE_RECEIPT:-$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance/model_acceptance.json}"
MODEL_ACCEPTANCE_SHA256_FILE="${MODEL_ACCEPTANCE_SHA256_FILE:-$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance/model_acceptance.sha256}"

resolved_reference_shadow="$(realpath "$REFERENCE_SHADOW_ROOT")"
[[ "$REFERENCE_SHADOW_ROOT" == /* && -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
  echo "REFERENCE_SHADOW_ROOT must be an absolute, existing, non-symlink directory" >&2
  exit 2
}
require_inside_reference_shadow() {
  local path="$1" label="$2" must_exist="${3:-1}"
  local resolved parent nearest
  [[ "$path" == /* ]] || { echo "$label must be absolute: $path" >&2; exit 2; }
  if [[ "$must_exist" == "1" ]]; then
    [[ -e "$path" && ! -L "$path" ]] || { echo "$label is unavailable or is a symlink: $path" >&2; exit 2; }
    resolved="$(realpath "$path")"
  else
    parent="$(dirname "$path")"
    case "/$path/" in *'/../'*|*'/./'*) echo "$label output path is not canonical: $path" >&2; exit 2 ;; esac
    nearest="$parent"
    while [[ ! -e "$nearest" ]]; do nearest="$(dirname "$nearest")"; done
    [[ -d "$nearest" && ! -L "$nearest" ]] || { echo "$label has an unsafe existing ancestor: $nearest" >&2; exit 2; }
    case "$(realpath "$nearest")" in "$resolved_reference_shadow"|"$resolved_reference_shadow"/*) ;; *) echo "$label must be inside REFERENCE_SHADOW_ROOT" >&2; exit 2 ;; esac
    mkdir -p "$parent"
    [[ ! -L "$parent" && ! -L "$path" ]] || { echo "$label output path crosses a symlink: $path" >&2; exit 2; }
    resolved="$(realpath "$parent")/$(basename "$path")"
  fi
  case "$resolved" in
    "$resolved_reference_shadow"/*) ;;
    *) echo "$label must resolve inside REFERENCE_SHADOW_ROOT: $resolved" >&2; exit 2 ;;
  esac
}

for confined in "$PROJECT_FILE" "$PARENT_IMPORT_MANIFEST" "$CPA_MODEL_DIR" "$CPA_TRAIN_RECEIPT"; do
  require_inside_reference_shadow "$confined" "input:$confined"
done
require_inside_reference_shadow "$MODEL_ACCEPTANCE_RECEIPT" MODEL_ACCEPTANCE_RECEIPT 0
require_inside_reference_shadow "$MODEL_ACCEPTANCE_SHA256_FILE" MODEL_ACCEPTANCE_SHA256_FILE 0

expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) echo "HPC_CONTAINER_BINDS differs from the frozen model-acceptance allowlist" >&2; exit 2 ;;
esac
expected_forward_prefixes="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES"
case "${HPC_CONTAINER_FORWARD_PREFIXES+x}:${HPC_CONTAINER_FORWARD_PREFIXES:-}" in
  :) HPC_CONTAINER_FORWARD_PREFIXES="$expected_forward_prefixes" ;;
  "x:$expected_forward_prefixes") ;;
  *) echo "HPC_CONTAINER_FORWARD_PREFIXES differs from the frozen reference-cell-state allowlist" >&2; exit 2 ;;
esac
export HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT
DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
FEATURE_CONFIG="${FEATURE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_features_v1.json}"
CLASSES_FILE="${CLASSES_FILE:-$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_classes_v1.tsv}"
ACCEPT_SCRIPT="${REFERENCE_CELL_STATE_MODEL_ACCEPT_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/30_accept_reference_cell_state_model.R}"
for required in "$PROJECT_FILE" "$PARENT_IMPORT_MANIFEST" "$CPA_MODEL_DIR" "$CPA_TRAIN_RECEIPT" \
  "$CPA_REFERENCE_ROOT" "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$ACCEPT_SCRIPT"; do
  [[ -e "$required" && ! -L "$required" ]] || {
    echo "Required reference-cell-state model-acceptance input is unavailable: $required" >&2
    exit 2
  }
done

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "site_share_mount_disabled=1"
echo "classifier_axis=reference_cell_state"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "project_file=$PROJECT_FILE"
echo "parent_import_manifest=$PARENT_IMPORT_MANIFEST"
echo "cpa_model_dir=$CPA_MODEL_DIR"
echo "cpa_train_receipt=$CPA_TRAIN_RECEIPT"
echo "model_acceptance_receipt=$MODEL_ACCEPTANCE_RECEIPT"
echo "feature_contract=historical_promoted_shape_9"
echo "class_ids=dead_cell,live_cell,multinucleated_cell"
echo "annotation_priority=multinucleated_cell,dead_cell,live_cell"

Rscript "$ACCEPT_SCRIPT" \
  --shadow-root "$REFERENCE_SHADOW_ROOT" \
  --project "$PROJECT_FILE" \
  --model-dir "$CPA_MODEL_DIR" \
  --train-receipt "$CPA_TRAIN_RECEIPT" \
  --feature-config "$FEATURE_CONFIG" \
  --classes-file "$CLASSES_FILE" \
  --parent-import-manifest "$PARENT_IMPORT_MANIFEST" \
  --reference-root "$CPA_REFERENCE_ROOT" \
  --dependency-lock "$DEPENDENCY_LOCK" \
  --output-receipt "$MODEL_ACCEPTANCE_RECEIPT"

[[ -s "$MODEL_ACCEPTANCE_RECEIPT" && ! -L "$MODEL_ACCEPTANCE_RECEIPT" ]] || {
  echo "Model acceptance did not produce its immutable receipt" >&2
  exit 2
}
acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
acceptance_sha256="${acceptance_sha256%%[[:space:]]*}"
[[ "$acceptance_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Unable to calculate model-acceptance SHA-256" >&2
  exit 2
}
sha_tmp="${MODEL_ACCEPTANCE_SHA256_FILE}.tmp.$$"
printf '%s\n' "$acceptance_sha256" > "$sha_tmp"
if [[ -e "$MODEL_ACCEPTANCE_SHA256_FILE" ]]; then
  cmp -s "$sha_tmp" "$MODEL_ACCEPTANCE_SHA256_FILE" || {
    rm -f "$sha_tmp"
    echo "Existing model-acceptance SHA sidecar differs and was preserved" >&2
    exit 2
  }
  rm -f "$sha_tmp"
else
  mv "$sha_tmp" "$MODEL_ACCEPTANCE_SHA256_FILE"
fi

echo "reference_cell_state_model_acceptance_complete=1"
echo "model_acceptance_sha256=$acceptance_sha256"
echo "model_acceptance_sha256_file=$MODEL_ACCEPTANCE_SHA256_FILE"
