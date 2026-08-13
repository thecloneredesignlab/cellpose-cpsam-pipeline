#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"
: "${CPA_TRAIN_RECEIPT:?CPA_TRAIN_RECEIPT is required}"
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
MODEL_ACCEPTANCE_RECEIPT="${MODEL_ACCEPTANCE_RECEIPT:-$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance_v2/model_acceptance.json}"
MODEL_ACCEPTANCE_SHA256_FILE="${MODEL_ACCEPTANCE_SHA256_FILE:-$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance_v2/model_acceptance.sha256}"
for input in "$PROJECT_FILE" "$PARENT_IMPORT_MANIFEST" "$CPA_MODEL_DIR" "$CPA_TRAIN_RECEIPT"; do
  reference_cell_state_v2_require_inside "$input" "$REFERENCE_SHADOW_ROOT" model_acceptance_input
done
reference_cell_state_v2_require_recoverable_output \
  "$MODEL_ACCEPTANCE_RECEIPT" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_RECEIPT
reference_cell_state_v2_require_recoverable_output \
  "$MODEL_ACCEPTANCE_SHA256_FILE" "$REFERENCE_SHADOW_ROOT" MODEL_ACCEPTANCE_SHA256_FILE
[[ ! -L "$MODEL_ACCEPTANCE_RECEIPT" && ! -L "$MODEL_ACCEPTANCE_SHA256_FILE" ]] || {
  reference_cell_state_v2_abort "V2 model-acceptance output may not be a symlink"
  exit 2
}

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the V2 model-acceptance allowlist"; exit 2 ;;
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
FEATURE_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_features_v2.json"
CLASSES_FILE="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_classes_v2.tsv"
ACCEPT_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/30_accept_reference_cell_state_model.R"
for required in "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$ACCEPT_SCRIPT"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    reference_cell_state_v2_abort "Required V2 model-acceptance file is unavailable: $required"
    exit 2
  }
done

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$(dirname "$MODEL_ACCEPTANCE_RECEIPT")"
hpc_apptainer_exec Rscript "$ACCEPT_SCRIPT" \
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
reference_cell_state_v2_test_maybe_fail model_acceptance_after_receipt
[[ -s "$MODEL_ACCEPTANCE_RECEIPT" && ! -L "$MODEL_ACCEPTANCE_RECEIPT" ]] || {
  reference_cell_state_v2_abort "V2 model acceptance did not produce its receipt"
  exit 2
}
acceptance_sha256="$(reference_cell_state_v2_sha256 "$MODEL_ACCEPTANCE_RECEIPT")"
if [[ -e "$MODEL_ACCEPTANCE_SHA256_FILE" ]]; then
  recorded_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
  [[ "$recorded_sha256" == "$acceptance_sha256" ]] || {
    reference_cell_state_v2_abort "Existing V2 model-acceptance SHA sidecar differs from the verified receipt"
    exit 2
  }
else
  tmp="${MODEL_ACCEPTANCE_SHA256_FILE}.tmp.$$"
  printf '%s\n' "$acceptance_sha256" > "$tmp"
  [[ ! -e "$MODEL_ACCEPTANCE_SHA256_FILE" ]] || {
    reference_cell_state_v2_abort "V2 model-acceptance SHA sidecar appeared during recovery"
    exit 2
  }
  mv "$tmp" "$MODEL_ACCEPTANCE_SHA256_FILE"
fi
echo "reference_cell_state_model_acceptance_v2_complete=1"
echo "model_acceptance_sha256=$acceptance_sha256"
echo "method_parity_status=$METHOD_PARITY_STATUS"
echo "legacy_no_go_enforcement=not_read"
