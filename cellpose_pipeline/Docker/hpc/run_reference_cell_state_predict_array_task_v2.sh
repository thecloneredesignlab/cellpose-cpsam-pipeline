#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_load_sif_identity
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"
mode=formal
case "$REFERENCE_SHADOW_ROOT" in "$RESULTS_ROOT"/Tests_and_Parameters_calibration/*) mode=calibration ;; esac
reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" "$mode" "$RESULTS_ROOT"
METHOD_PARITY_STATUS="$(reference_cell_state_v2_method_parity_status "$REFERENCE_SHADOW_ROOT")"
reference_cell_state_v2_require_inside "$FEATURE_MANIFEST" "$PARENT_BROAD_SHADOW_ROOT" FEATURE_MANIFEST
reference_cell_state_v2_require_inside "$CPA_MODEL_DIR" "$REFERENCE_SHADOW_ROOT" CPA_MODEL_DIR
reference_cell_state_v2_require_inside "$MODEL_ACCEPTANCE_RECEIPT" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_RECEIPT
reference_cell_state_v2_require_inside "$MODEL_ACCEPTANCE_SHA256_FILE" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_SHA256_FILE
[[ "$PREDICTION_ROOT" == "$REFERENCE_SHADOW_ROOT/prediction_shards_v2" ]] || {
  reference_cell_state_v2_abort "PREDICTION_ROOT must be the canonical V2 prediction root"
  exit 2
}
mkdir -p "$PREDICTION_ROOT"

header="$(sed -n '1p' "$FEATURE_MANIFEST" | tr -d '\r')"
[[ "$header" == $'key\tfeature_path\treceipt_path' ]] || {
  reference_cell_state_v2_abort "FEATURE_MANIFEST must have exact key/feature_path/receipt_path columns"
  exit 2
}
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FEATURE_MANIFEST" | tr -d '\r')"
IFS=$'\t' read -r field_key feature_shard feature_receipt extra <<< "$row"
[[ -z "${extra:-}" && "$field_key" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]] || {
  reference_cell_state_v2_abort "Invalid V2 feature-manifest row: $row"
  exit 2
}
reference_cell_state_v2_require_inside "$feature_shard" "$PARENT_BROAD_SHADOW_ROOT" feature_shard
reference_cell_state_v2_require_inside "$feature_receipt" "$PARENT_BROAD_SHADOW_ROOT" feature_receipt
feature_name="${feature_shard##*/}"
case "$feature_name" in
  "${field_key}__original_broad_phenotype_features.tsv") shard_branch=original ;;
  *) reference_cell_state_v2_abort "Formal V2 predictions require original-branch feature shards: $feature_name"; exit 2 ;;
esac
well="${field_key%%_*}"
generation_dir="$PREDICTION_ROOT/shards/$well/${field_key}__${shard_branch}"
output_tsv="$generation_dir/reference_cell_state_predictions.tsv"
output_receipt="$generation_dir/prediction_receipt.json"
[[ ! -L "$generation_dir" && ( ! -e "$generation_dir" || -d "$generation_dir" ) ]] || {
  reference_cell_state_v2_abort "V2 prediction generation is not a regular directory"
  exit 2
}
# Script 31 atomically installs the complete generation directory.  Only its
# parent may exist before the R worker starts.
mkdir -p "$(dirname "$generation_dir")"

acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
[[ "$acceptance_sha256" =~ ^[0-9a-f]{64}$ && \
   "$(reference_cell_state_v2_sha256 "$MODEL_ACCEPTANCE_RECEIPT")" == "$acceptance_sha256" ]] || {
  reference_cell_state_v2_abort "V2 model-acceptance receipt/hash mismatch"
  exit 2
}

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the V2 prediction allowlist"; exit 2 ;;
esac
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT
export HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
reference_cell_state_v2_verify_container_rootfs_read_only
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT
DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
PREDICT_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/31_predict_reference_cell_state_shard.R"
unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
hpc_apptainer_exec Rscript "$PREDICT_SCRIPT" \
  --reference-root "$CPA_REFERENCE_ROOT" \
  --dependency-lock "$DEPENDENCY_LOCK" \
  --model-dir "$CPA_MODEL_DIR" \
  --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT" \
  --model-acceptance-sha256 "$acceptance_sha256" \
  --feature-shard "$feature_shard" \
  --feature-receipt "$feature_receipt" \
  --output-tsv "$output_tsv" \
  --output-receipt "$output_receipt"

expected_header=$'model_id\tcell_id\tpredicted_class_id\tprediction_status\tprobability__live_cell\tprobability__dead_cell\tprobability__multinucleated_cell'
[[ "$(sed -n '1p' "$output_tsv" | tr -d '\r')" == "$expected_header" ]] || {
  reference_cell_state_v2_abort "V2 shard prediction schema drifted"
  exit 2
}
echo "reference_cell_state_prediction_v2_task_complete=1"
echo "field_key=$field_key"
echo "method_parity_status=$METHOD_PARITY_STATUS"
echo "legacy_no_go_enforcement=not_read"
