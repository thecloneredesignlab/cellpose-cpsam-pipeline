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
: "${PARENT_BROAD_SHADOW_ROOT:?PARENT_BROAD_SHADOW_ROOT is required}"
: "${FEATURE_MANIFEST:?FEATURE_MANIFEST is required}"
: "${PREDICTION_ROOT:?PREDICTION_ROOT is required}"
: "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"
: "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-9][0-9]*$ ]] || {
  echo "SLURM_ARRAY_TASK_ID must be a one-based positive integer" >&2
  exit 2
}
BRANCH="${BRANCH:-}"
case "$BRANCH" in ""|original|nucleated) ;; *) echo "BRANCH must be empty, original, or nucleated" >&2; exit 2 ;; esac

for root_path in "$REFERENCE_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$root_path" == /* && -d "$root_path" && ! -L "$root_path" ]] || {
    echo "Required root is not an absolute real directory: $root_path" >&2
    exit 2
  }
done
resolved_reference_shadow="$(realpath "$REFERENCE_SHADOW_ROOT")"
resolved_parent_shadow="$(realpath "$PARENT_BROAD_SHADOW_ROOT")"
[[ "$resolved_reference_shadow" != "$resolved_parent_shadow" ]] || {
  echo "Reference and parent broad shadow roots must be disjoint" >&2
  exit 2
}

require_inside_root() {
  local path="$1" root="$2" label="$3" must_exist="${4:-1}"
  local resolved_root resolved parent nearest
  resolved_root="$(realpath "$root")"
  [[ "$path" == /* ]] || { echo "$label must be absolute: $path" >&2; exit 2; }
  if [[ "$must_exist" == "1" ]]; then
    [[ -e "$path" && ! -L "$path" ]] || { echo "$label is unavailable or a symlink: $path" >&2; exit 2; }
    resolved="$(realpath "$path")"
  else
    parent="$(dirname "$path")"
    case "/$path/" in *'/../'*|*'/./'*) echo "$label output path is not canonical: $path" >&2; exit 2 ;; esac
    nearest="$parent"
    while [[ ! -e "$nearest" ]]; do nearest="$(dirname "$nearest")"; done
    [[ -d "$nearest" && ! -L "$nearest" ]] || { echo "$label has an unsafe existing ancestor: $nearest" >&2; exit 2; }
    case "$(realpath "$nearest")" in "$resolved_root"|"$resolved_root"/*) ;; *) echo "$label ancestor escaped its frozen root" >&2; exit 2 ;; esac
    mkdir -p "$parent"
    [[ ! -L "$parent" && ! -L "$path" ]] || { echo "$label output crosses a symlink: $path" >&2; exit 2; }
    resolved="$(realpath "$parent")/$(basename "$path")"
  fi
  case "$resolved" in "$resolved_root"/*) ;; *) echo "$label escaped its frozen root: $resolved" >&2; exit 2 ;; esac
}

require_inside_root "$FEATURE_MANIFEST" "$PARENT_BROAD_SHADOW_ROOT" FEATURE_MANIFEST
require_inside_root "$CPA_MODEL_DIR" "$REFERENCE_SHADOW_ROOT" CPA_MODEL_DIR
require_inside_root "$MODEL_ACCEPTANCE_RECEIPT" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_RECEIPT
require_inside_root "$MODEL_ACCEPTANCE_SHA256_FILE" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_SHA256_FILE
require_inside_root "$PREDICTION_ROOT" "$REFERENCE_SHADOW_ROOT" PREDICTION_ROOT 0

header="$(sed -n '1p' "$FEATURE_MANIFEST" | tr -d '\r')"
[[ "$header" == $'key\tfeature_path\treceipt_path' ]] || {
  echo "FEATURE_MANIFEST must have the exact key/feature_path/receipt_path header" >&2
  exit 2
}
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FEATURE_MANIFEST" | tr -d '\r')"
IFS=$'\t' read -r field_key feature_shard feature_receipt extra <<< "$row"
[[ -z "${extra:-}" && "$field_key" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]] || {
  echo "Invalid feature-manifest row for array task $SLURM_ARRAY_TASK_ID: $row" >&2
  exit 2
}
[[ -s "$feature_shard" && -s "$feature_receipt" ]] || {
  echo "Feature shard or receipt is unavailable: feature=$feature_shard receipt=$feature_receipt" >&2
  exit 2
}
require_inside_root "$feature_shard" "$PARENT_BROAD_SHADOW_ROOT" feature_shard
require_inside_root "$feature_receipt" "$PARENT_BROAD_SHADOW_ROOT" feature_receipt

well="${field_key%%_*}"
feature_name="${feature_shard##*/}"
case "$feature_name" in
  "${field_key}__original_broad_phenotype_features.tsv") shard_branch=original ;;
  "${field_key}__nucleated_broad_phenotype_features.tsv") shard_branch=nucleated ;;
  *)
    echo "Parent feature filename does not encode the frozen key/branch contract: $feature_shard" >&2
    exit 2
    ;;
esac
if [[ -n "$BRANCH" && "$BRANCH" != "$shard_branch" ]]; then
  echo "Feature shard branch differs from BRANCH: shard=$shard_branch requested=$BRANCH" >&2
  exit 2
fi
output_tsv="$PREDICTION_ROOT/shards/$well/${field_key}__${shard_branch}_reference_cell_state_predictions.tsv"
output_receipt="$PREDICTION_ROOT/receipts/$well/${field_key}__${shard_branch}.json"
mkdir -p "$(dirname "$output_tsv")" "$(dirname "$output_receipt")"
require_inside_root "$output_tsv" "$REFERENCE_SHADOW_ROOT" output_tsv 0
require_inside_root "$output_receipt" "$REFERENCE_SHADOW_ROOT" output_receipt 0

acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
[[ "$acceptance_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Invalid model-acceptance SHA sidecar" >&2
  exit 2
}
observed_acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
observed_acceptance_sha256="${observed_acceptance_sha256%%[[:space:]]*}"
[[ "$observed_acceptance_sha256" == "$acceptance_sha256" ]] || {
  echo "Model-acceptance receipt SHA mismatch" >&2
  exit 2
}

expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) echo "HPC_CONTAINER_BINDS differs from the frozen reference prediction allowlist" >&2; exit 2 ;;
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
PREDICT_SCRIPT="${REFERENCE_CELL_STATE_SHARD_PREDICT_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/31_predict_reference_cell_state_shard.R}"
for required in "$DEPENDENCY_LOCK" "$PREDICT_SCRIPT"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    echo "Required reference-cell-state prediction code/input is unavailable: $required" >&2
    exit 2
  }
done

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"
args=(
  "$PREDICT_SCRIPT"
  --reference-root "$CPA_REFERENCE_ROOT"
  --dependency-lock "$DEPENDENCY_LOCK"
  --model-dir "$CPA_MODEL_DIR"
  --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT"
  --model-acceptance-sha256 "$acceptance_sha256"
  --feature-shard "$feature_shard"
  --feature-receipt "$feature_receipt"
  --output-tsv "$output_tsv"
  --output-receipt "$output_receipt"
)
case "${FORCE_PREDICT:-0}" in 0) ;; 1) args+=(--force) ;; *) echo "FORCE_PREDICT must be 0 or 1" >&2; exit 2 ;; esac

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "site_share_mount_disabled=1"
echo "classifier_axis=reference_cell_state"
echo "parent_broad_shadow_access=read_only_features_only"
echo "field_key=$field_key"
echo "branch=$shard_branch"
echo "feature_shard=$feature_shard"
echo "prediction_tsv=$output_tsv"
echo "prediction_receipt=$output_receipt"
echo "model_acceptance_sha256=$acceptance_sha256"
printf 'predict_arg=%s\n' "${args[@]}"

Rscript --version
Rscript "${args[@]}"

expected_prefix=$'model_id\tcell_id\tpredicted_class_id\tprediction_status\tprobability__dead_cell\tprobability__live_cell\tprobability__multinucleated_cell'
observed_header="$(sed -n '1p' "$output_tsv" | tr -d '\r')"
[[ "$observed_header" == "$expected_prefix" ]] || {
  echo "Reference shard prediction schema changed: $observed_header" >&2
  exit 2
}
echo "reference_cell_state_prediction_task_complete=1"
