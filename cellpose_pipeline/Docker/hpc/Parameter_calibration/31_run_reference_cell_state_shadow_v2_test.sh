#!/usr/bin/env bash
set -euo pipefail

required_host=hpctpa3pc0009
observed_host="$(hostname -s)"
[[ "$observed_host" == "$required_host" ]] || {
  echo "Reference-cell-state V2 calibration requires login-node ssh followed by ssh to $required_host; observed=$observed_host" >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "Reference-cell-state V2 calibration is direct and cannot run inside Slurm" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd -P)}"
EXPERIMENT_ROOT="/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/results}"
DATASET_ROOT="${DATASET_ROOT:-$EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$EXPERIMENT_ROOT/results/full_fusion_shape_strict_20260711_155940}"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$RESULTS_ROOT/broad_phenotype_shadow_20260812_075437}"
EXPECTED_PARENT_CELL_COUNT="${EXPECTED_PARENT_CELL_COUNT:-32000}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
V2_STAGE="${V2_STAGE:-phase-a}"
submitter="$SCRIPT_DIR/../submit_reference_cell_state_shadow_v2.sh"

[[ "$RESULTS_ROOT" == "$EXPERIMENT_ROOT/results" && -d "$RESULTS_ROOT/Tests_and_Parameters_calibration" ]] || {
  echo "Calibration must write only under the frozen experiment results/Tests_and_Parameters_calibration root" >&2
  exit 2
}
[[ -d "$PARENT_BROAD_SHADOW_ROOT" && ! -L "$PARENT_BROAD_SHADOW_ROOT" ]] || {
  echo "Completed broad-phenotype calibration parent is unavailable: $PARENT_BROAD_SHADOW_ROOT" >&2
  exit 2
}
[[ -x "$submitter" ]] || { echo "V2 submitter is unavailable: $submitter" >&2; exit 2; }

if [[ "$V2_STAGE" == phase-a ]]; then
  REFERENCE_SHADOW_ROOT="${REFERENCE_SHADOW_ROOT:-$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_$RUN_STAMP}"
  case "$REFERENCE_SHADOW_ROOT" in
    "$RESULTS_ROOT"/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_*) ;;
    *) echo "Calibration Phase A root is outside the V2 test namespace: $REFERENCE_SHADOW_ROOT" >&2; exit 2 ;;
  esac
  if [[ -e "$REFERENCE_SHADOW_ROOT" || -L "$REFERENCE_SHADOW_ROOT" ]]; then
    [[ -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
      echo "Existing V2 calibration root is not a real directory: $REFERENCE_SHADOW_ROOT" >&2
      exit 2
    }
    calibration_root_disposition=resume_existing_through_frozen_archive
  else
    calibration_root_disposition=create_new
  fi
else
  : "${REFERENCE_SHADOW_ROOT:?A staged V2 calibration continuation requires its existing V2 test root}"
  calibration_root_disposition=continue_existing_stage
fi

echo "test_host=$observed_host"
echo "test_execution=direct_test_no_slurm_submission"
echo "test_gpu=0"
echo "test_stage=$V2_STAGE"
echo "test_reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "test_root_disposition=$calibration_root_disposition"
echo "test_results_policy=Tests_and_Parameters_calibration_only"

export PROJECT_DIR EXPERIMENT_ROOT RESULTS_ROOT DATASET_ROOT SOURCE_SEGMENTATION_ROOT
export PARENT_BROAD_SHADOW_ROOT EXPECTED_PARENT_CELL_COUNT RUN_STAMP V2_STAGE REFERENCE_SHADOW_ROOT
export EXECUTION_MODE=direct_test
exec "$submitter"
