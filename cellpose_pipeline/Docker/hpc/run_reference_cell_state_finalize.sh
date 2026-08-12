#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_HPC_CONTAINER_IMAGE" ]] || {
  echo "HPC_CONTAINER_IMAGE must equal the frozen latest SIF path: $DEFAULT_HPC_CONTAINER_IMAGE" >&2
  exit 2
}
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
case "${HPC_CONTAINER_NO_MOUNT+x}:${HPC_CONTAINER_NO_MOUNT:-}" in
  :|x:/share) ;;
  *) echo "HPC_CONTAINER_NO_MOUNT must be exactly /share for reference-cell-state work" >&2; exit 2 ;;
esac
HPC_CONTAINER_NO_MOUNT=/share
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${PARENT_BROAD_SHADOW_ROOT:?PARENT_BROAD_SHADOW_ROOT is required}"
: "${FEATURE_MANIFEST:?FEATURE_MANIFEST is required}"
: "${PREDICTION_ROOT:?PREDICTION_ROOT is required}"
: "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required}"
PROJECT_FILE="${PROJECT_FILE:-$REFERENCE_SHADOW_ROOT/projection_input/representative_umap/project.yml}"
PARENT_IMPORT_MANIFEST="${PARENT_IMPORT_MANIFEST:-$(dirname "$PROJECT_FILE")/parent_import_manifest.json}"
RUN_REFERENCE_COMPARISON="${RUN_REFERENCE_COMPARISON:-0}"
case "$RUN_REFERENCE_COMPARISON" in 0|1) ;; *) echo "RUN_REFERENCE_COMPARISON must be 0 or 1" >&2; exit 2 ;; esac
case "${FORCE_FINALIZE:-0}" in 0|1) ;; *) echo "FORCE_FINALIZE must be 0 or 1" >&2; exit 2 ;; esac
case "${FORCE_COMPARISON:-0}" in 0|1) ;; *) echo "FORCE_COMPARISON must be 0 or 1" >&2; exit 2 ;; esac

for root_path in "$REFERENCE_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT"; do
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

require_inside_root "$PROJECT_FILE" "$REFERENCE_SHADOW_ROOT" PROJECT_FILE
require_inside_root "$PARENT_IMPORT_MANIFEST" "$REFERENCE_SHADOW_ROOT" PARENT_IMPORT_MANIFEST
require_inside_root "$PREDICTION_ROOT" "$REFERENCE_SHADOW_ROOT" PREDICTION_ROOT
require_inside_root "$MODEL_ACCEPTANCE_RECEIPT" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_RECEIPT
require_inside_root "$MODEL_ACCEPTANCE_SHA256_FILE" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_SHA256_FILE
require_inside_root "$FEATURE_MANIFEST" "$PARENT_BROAD_SHADOW_ROOT" FEATURE_MANIFEST
if [[ -n "${CELLS_FILE:-}" ]]; then
  require_inside_root "$CELLS_FILE" "$PARENT_BROAD_SHADOW_ROOT" CELLS_FILE
fi

model_acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
[[ "$model_acceptance_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Invalid model-acceptance SHA sidecar" >&2
  exit 2
}
observed_acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
observed_acceptance_sha256="${observed_acceptance_sha256%%[[:space:]]*}"
[[ "$observed_acceptance_sha256" == "$model_acceptance_sha256" ]] || {
  echo "Model-acceptance SHA mismatch before reference-cell-state finalize" >&2
  exit 2
}

if [[ "$RUN_REFERENCE_COMPARISON" == "1" ]]; then
  : "${CURRENT_CLASSIFICATION_ROOT:?CURRENT_CLASSIFICATION_ROOT is required when comparison is enabled}"
  : "${COMPARISON_ROOT:?COMPARISON_ROOT is required when comparison is enabled}"
  [[ "$CURRENT_CLASSIFICATION_ROOT" == /* && -d "$CURRENT_CLASSIFICATION_ROOT" && ! -L "$CURRENT_CLASSIFICATION_ROOT" ]] || {
    echo "CURRENT_CLASSIFICATION_ROOT must be an absolute real directory" >&2
    exit 2
  }
  [[ "$COMPARISON_ROOT" == /* ]] || { echo "COMPARISON_ROOT must be absolute" >&2; exit 2; }
  case "/$COMPARISON_ROOT/" in *'/../'*|*'/./'*) echo "COMPARISON_ROOT must be canonical" >&2; exit 2 ;; esac
  comparison_parent="$(dirname "$COMPARISON_ROOT")"
  [[ -d "$comparison_parent" && ! -L "$comparison_parent" && ! -L "$COMPARISON_ROOT" ]] || {
    echo "COMPARISON_ROOT must have a real, existing parent directory" >&2
    exit 2
  }
  resolved_current_root="$(realpath "$CURRENT_CLASSIFICATION_ROOT")"
  resolved_comparison_root="$(realpath "$comparison_parent")/$(basename "$COMPARISON_ROOT")"
  for protected_root in "$resolved_reference_shadow" "$resolved_parent_shadow" "$resolved_current_root"; do
    case "$resolved_comparison_root/" in "$protected_root/"*|*/../*) echo "COMPARISON_ROOT must be a third independent output root" >&2; exit 2 ;; esac
    case "$protected_root/" in "$resolved_comparison_root/"*) echo "COMPARISON_ROOT must not contain an input root" >&2; exit 2 ;; esac
  done
  mkdir -p "$COMPARISON_ROOT"
  if [[ -n "${CURRENT_PREDICTIONS:-}" && -n "${CURRENT_MANIFEST:-}" ]] || \
     [[ -z "${CURRENT_PREDICTIONS:-}" && -z "${CURRENT_MANIFEST:-}" ]]; then
    echo "Set exactly one of CURRENT_PREDICTIONS or CURRENT_MANIFEST for comparison" >&2
    exit 2
  fi
  current_input="${CURRENT_PREDICTIONS:-${CURRENT_MANIFEST:-}}"
  require_inside_root "$current_input" "$CURRENT_CLASSIFICATION_ROOT" current_comparison_input
fi

expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro"
if [[ "$RUN_REFERENCE_COMPARISON" == "1" ]]; then
  expected_binds="$expected_binds,$CURRENT_CLASSIFICATION_ROOT:$CURRENT_CLASSIFICATION_ROOT:ro,$COMPARISON_ROOT:$COMPARISON_ROOT:rw"
fi
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) echo "HPC_CONTAINER_BINDS differs from the frozen reference finalize allowlist" >&2; exit 2 ;;
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
FINALIZE_SCRIPT="${REFERENCE_CELL_STATE_FINALIZE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/32_merge_reference_cell_state_predictions.py}"
COMPARISON_SCRIPT="${REFERENCE_CELL_STATE_COMPARISON_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/29_compare_current_vs_reference_cell_state.py}"
COMPARISON_CONFIG="${REFERENCE_CELL_STATE_COMPARISON_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_comparison_v1.json}"
[[ -f "$FINALIZE_SCRIPT" && ! -L "$FINALIZE_SCRIPT" ]] || {
  echo "Reference-cell-state finalize implementation is unavailable: $FINALIZE_SCRIPT" >&2
  exit 2
}
if [[ "$RUN_REFERENCE_COMPARISON" == "1" ]]; then
  for required in "$COMPARISON_SCRIPT" "$COMPARISON_CONFIG"; do
    [[ -f "$required" && ! -L "$required" ]] || { echo "Comparison code/config unavailable: $required" >&2; exit 2; }
  done
fi

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"
args=(
  "$FINALIZE_SCRIPT"
  --feature-manifest "$FEATURE_MANIFEST"
  --prediction-root "$PREDICTION_ROOT"
  --shadow-root "$REFERENCE_SHADOW_ROOT"
  --parent-import-manifest "$PARENT_IMPORT_MANIFEST"
  --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT"
  --model-acceptance-sha256 "$model_acceptance_sha256"
)
if [[ -n "${CELLS_FILE:-}" ]]; then args+=(--cells "$CELLS_FILE"); fi
if [[ "${FORCE_FINALIZE:-0}" == "1" ]]; then args+=(--force); fi

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
echo "parent_broad_shadow_root=$PARENT_BROAD_SHADOW_ROOT"
echo "parent_broad_shadow_access=read_only_features_only"
echo "feature_manifest=$FEATURE_MANIFEST"
echo "prediction_root=$PREDICTION_ROOT"
echo "model_acceptance_sha256=$model_acceptance_sha256"
echo "legacy_no_go_enforcement=not_read"
printf 'finalize_arg=%s\n' "${args[@]}"

python -I "${args[@]}"

REFERENCE_PREDICTIONS="$REFERENCE_SHADOW_ROOT/predictions/reference_cell_state_predictions.tsv"
REFERENCE_RECEIPT="$REFERENCE_SHADOW_ROOT/REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json"
for output in "$REFERENCE_PREDICTIONS" "$REFERENCE_RECEIPT"; do
  [[ -s "$output" && ! -L "$output" ]] || { echo "Required reference finalize output is unavailable: $output" >&2; exit 2; }
done
expected_header=$'model_id\tcell_id\treference_cell_state_class_id\tprediction_status'
observed_header="$(sed -n '1p' "$REFERENCE_PREDICTIONS" | tr -d '\r')"
[[ "$observed_header" == "$expected_header" ]] || {
  echo "Merged reference-cell-state output must have the exact four-column schema: $observed_header" >&2
  exit 2
}

echo "reference_cell_state_finalize_complete=1"
echo "reference_predictions=$REFERENCE_PREDICTIONS"
echo "reference_predictions_sha256=$(sha256sum "$REFERENCE_PREDICTIONS" | awk '{print $1}')"
echo "reference_receipt=$REFERENCE_RECEIPT"

if [[ "$RUN_REFERENCE_COMPARISON" == "0" ]]; then
  echo "comparison_performed=false"
  echo "comparison_reason=independent_reference_result_frozen_without_current_classifier_read"
  exit 0
fi

compare_args=(
  "$COMPARISON_SCRIPT"
  --reference-predictions "$REFERENCE_PREDICTIONS"
  --output-root "$COMPARISON_ROOT"
  --config "$COMPARISON_CONFIG"
)
if [[ -n "${CURRENT_PREDICTIONS:-}" ]]; then
  compare_args+=(--current-predictions "$CURRENT_PREDICTIONS")
else
  compare_args+=(--current-manifest "$CURRENT_MANIFEST")
fi
if [[ "${FORCE_COMPARISON:-0}" == "1" ]]; then compare_args+=(--force); fi
printf 'comparison_arg=%s\n' "${compare_args[@]}"
python -I "${compare_args[@]}"
[[ -s "$COMPARISON_ROOT/comparison_receipt.json" ]] || {
  echo "Comparison receipt was not produced" >&2
  exit 2
}
echo "comparison_performed=true"
echo "comparison_root=$COMPARISON_ROOT"
echo "comparison_interpretation=descriptive_cross_classification_not_accuracy"
