#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${PARENT_BROAD_SHADOW_ROOT:?PARENT_BROAD_SHADOW_ROOT is required}"
: "${FEATURE_MANIFEST:?FEATURE_MANIFEST is required}"
: "${PREDICTION_ROOT:?PREDICTION_ROOT is required}"
: "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required}"
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

PROJECT_FILE="${PROJECT_FILE:-$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/project.yml}"
PARENT_IMPORT_MANIFEST="${PARENT_IMPORT_MANIFEST:-$(dirname "$PROJECT_FILE")/parent_import_manifest.json}"
CELLS_FILE="${CELLS_FILE:-$PARENT_BROAD_SHADOW_ROOT/cpa/cells.tsv}"
for input in "$PROJECT_FILE" "$PARENT_IMPORT_MANIFEST" "$PREDICTION_ROOT" \
  "$MODEL_ACCEPTANCE_RECEIPT" "$MODEL_ACCEPTANCE_SHA256_FILE"; do
  reference_cell_state_v2_require_inside "$input" "$REFERENCE_SHADOW_ROOT" finalize_input
done
reference_cell_state_v2_require_inside "$FEATURE_MANIFEST" "$PARENT_BROAD_SHADOW_ROOT" FEATURE_MANIFEST
reference_cell_state_v2_require_inside "$CELLS_FILE" "$PARENT_BROAD_SHADOW_ROOT" CELLS_FILE
[[ ! -L "$REFERENCE_SHADOW_ROOT/predictions" && \
   ( ! -e "$REFERENCE_SHADOW_ROOT/predictions" || -d "$REFERENCE_SHADOW_ROOT/predictions" ) ]] || {
  reference_cell_state_v2_abort "V2 merged result path is not a regular generation directory"
  exit 2
}

acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
[[ "$acceptance_sha256" =~ ^[0-9a-f]{64}$ && \
   "$(reference_cell_state_v2_sha256 "$MODEL_ACCEPTANCE_RECEIPT")" == "$acceptance_sha256" ]] || {
  reference_cell_state_v2_abort "V2 model acceptance receipt/hash mismatch before merge"
  exit 2
}

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the V2 finalize allowlist"; exit 2 ;;
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
FINALIZE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/32_merge_reference_cell_state_predictions.py"
unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
hpc_apptainer_exec python -I "$FINALIZE_SCRIPT" \
  --feature-manifest "$FEATURE_MANIFEST" \
  --prediction-root "$PREDICTION_ROOT" \
  --shadow-root "$REFERENCE_SHADOW_ROOT" \
  --parent-import-manifest "$PARENT_IMPORT_MANIFEST" \
  --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT" \
  --model-acceptance-sha256 "$acceptance_sha256" \
  --cells "$CELLS_FILE"

REFERENCE_PREDICTIONS="$REFERENCE_SHADOW_ROOT/predictions/reference_cell_state_predictions.tsv"
REFERENCE_RECEIPT="$REFERENCE_SHADOW_ROOT/predictions/REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json"
for output in "$REFERENCE_PREDICTIONS" "$REFERENCE_RECEIPT"; do
  [[ -s "$output" && ! -L "$output" ]] || {
    reference_cell_state_v2_abort "Required V2 merge output is unavailable: $output"
    exit 2
  }
done
expected_header=$'model_id\tcell_id\treference_cell_state_class_id\tprediction_status'
[[ "$(sed -n '1p' "$REFERENCE_PREDICTIONS" | tr -d '\r')" == "$expected_header" ]] || {
  reference_cell_state_v2_abort "Merged V2 result must retain the exact independent four-column axis"
  exit 2
}
echo "reference_cell_state_finalize_v2_complete=1"
echo "reference_predictions=$REFERENCE_PREDICTIONS"
echo "comparison_performed=false"
echo "method_parity_status=$METHOD_PARITY_STATUS"
echo "legacy_no_go_enforcement=not_read"
