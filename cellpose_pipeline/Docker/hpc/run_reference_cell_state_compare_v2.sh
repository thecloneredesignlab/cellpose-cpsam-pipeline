#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${CURRENT_CLASSIFICATION_ROOT:?CURRENT_CLASSIFICATION_ROOT is required only for the optional comparison}"
: "${COMPARISON_ROOT:?COMPARISON_ROOT is required and must be a third independent root}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_load_sif_identity
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"
reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" formal "$RESULTS_ROOT"
METHOD_PARITY_STATUS="$(reference_cell_state_v2_method_parity_status "$REFERENCE_SHADOW_ROOT")"
reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT" >/dev/null || {
  reference_cell_state_v2_abort "Comparison requires an intact final V2 SHADOW_ONLY technical GO generation"
  exit 2
}

REFERENCE_PREDICTIONS="$REFERENCE_SHADOW_ROOT/predictions/reference_cell_state_predictions.tsv"
reference_cell_state_v2_require_inside "$REFERENCE_PREDICTIONS" "$REFERENCE_SHADOW_ROOT" REFERENCE_PREDICTIONS
[[ "$CURRENT_CLASSIFICATION_ROOT" == /* && -d "$CURRENT_CLASSIFICATION_ROOT" && ! -L "$CURRENT_CLASSIFICATION_ROOT" ]] || {
  reference_cell_state_v2_abort "CURRENT_CLASSIFICATION_ROOT must be an absolute real directory"
  exit 2
}
comparison_parent="$(dirname "$COMPARISON_ROOT")"
[[ "$COMPARISON_ROOT" == "$RESULTS_ROOT/reference_cell_state_comparison_v2_"* && \
   -d "$comparison_parent" && ! -L "$comparison_parent" && ! -L "$COMPARISON_ROOT" && \
   ( ! -e "$COMPARISON_ROOT" || -d "$COMPARISON_ROOT" ) ]] || {
  reference_cell_state_v2_abort "COMPARISON_ROOT must be a new or verifiable results/reference_cell_state_comparison_v2_* generation"
  exit 2
}
for protected in "$REFERENCE_SHADOW_ROOT" "$CURRENT_CLASSIFICATION_ROOT"; do
  case "$COMPARISON_ROOT/" in "$protected/"*) reference_cell_state_v2_abort "Comparison root overlaps an input root"; exit 2 ;; esac
  case "$protected/" in "$COMPARISON_ROOT/"*) reference_cell_state_v2_abort "Comparison root contains an input root"; exit 2 ;; esac
done
if [[ -n "${CURRENT_PREDICTIONS:-}" && -z "${CURRENT_MANIFEST:-}" ]]; then
  current_flag=--current-predictions current_input="$CURRENT_PREDICTIONS"
elif [[ -n "${CURRENT_MANIFEST:-}" && -z "${CURRENT_PREDICTIONS:-}" ]]; then
  current_flag=--current-manifest current_input="$CURRENT_MANIFEST"
else
  reference_cell_state_v2_abort "Set exactly one of CURRENT_PREDICTIONS or CURRENT_MANIFEST"
  exit 2
fi
reference_cell_state_v2_require_inside "$current_input" "$CURRENT_CLASSIFICATION_ROOT" current_comparison_input

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
# Parent write mount first; the two more-specific input roots are appended last
# as read-only mounts so nested destinations cannot inherit parent writability.
expected_binds="$comparison_parent:$comparison_parent:rw,$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:ro,$CURRENT_CLASSIFICATION_ROOT:$CURRENT_CLASSIFICATION_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the post-freeze comparison allowlist"; exit 2 ;;
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
comparison_probe="$comparison_parent/.reference-v2-comparison-write-probe.$$"
reference_probe="$REFERENCE_SHADOW_ROOT/.reference-v2-input-write-probe.$$"
current_probe="$CURRENT_CLASSIFICATION_ROOT/.reference-v2-input-write-probe.$$"
hpc_apptainer_exec /bin/sh -eu -c '
  comparison_probe="$1"; reference_probe="$2"; current_probe="$3"
  : > "$comparison_probe"
  rm -f -- "$comparison_probe"
  if : > "$reference_probe" 2>/dev/null; then rm -f -- "$reference_probe"; exit 31; fi
  if : > "$current_probe" 2>/dev/null; then rm -f -- "$current_probe"; exit 32; fi
' reference-v2-comparison-bind-probe "$comparison_probe" "$reference_probe" "$current_probe" || {
  reference_cell_state_v2_abort "Comparison bind write-policy probe failed"
  exit 2
}
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT
COMPARISON_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/29_compare_current_vs_reference_cell_state.py"
COMPARISON_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_comparison_v1.json"
hpc_apptainer_exec python -I "$COMPARISON_SCRIPT" \
  --current-root "$CURRENT_CLASSIFICATION_ROOT" \
  "$current_flag" "$current_input" \
  --reference-predictions "$REFERENCE_PREDICTIONS" \
  --output-root "$COMPARISON_ROOT" \
  --config "$COMPARISON_CONFIG"
[[ -s "$COMPARISON_ROOT/comparison_receipt.json" ]] || {
  reference_cell_state_v2_abort "Optional V2 comparison did not produce its receipt"
  exit 2
}
echo "reference_cell_state_comparison_v2_complete=1"
echo "comparison_interpretation=descriptive_cross_classification_not_accuracy"
echo "comparison_feedback_to_reference_model=false"
echo "method_parity_status=$METHOD_PARITY_STATUS"
