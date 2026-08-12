#!/usr/bin/env bash
set -euo pipefail

required_host="hpctpa3pc0009"
observed_host="$(hostname -s)"
[[ "$observed_host" == "$required_host" ]] || {
  echo "Reference-cell-state calibration must run only after login-node ssh to $required_host; observed=$observed_host" >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "Reference-cell-state calibration is direct and must not run inside Slurm" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
: "${RESULTS_ROOT:?RESULTS_ROOT must be the explicit experiment results directory}"
: "${DATASET_ROOT:?DATASET_ROOT must be the explicit raw-image dataset root}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT must be explicit}"

DEFAULT_PARENT="$RESULTS_ROOT/Tests_and_Parameters_calibration/broad_phenotype_shadow_test_20260812_073844"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$DEFAULT_PARENT}"
EXPECTED_PARENT_CELL_COUNT="${EXPECTED_PARENT_CELL_COUNT:-100}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
REFERENCE_SHADOW_ROOT="$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_test_$RUN_STAMP"
submitter="$SCRIPT_DIR/../submit_reference_cell_state_shadow.sh"

[[ "$(basename "$RESULTS_ROOT")" == "results" ]] || {
  echo "RESULTS_ROOT must end in /results: $RESULTS_ROOT" >&2
  exit 2
}
[[ -s "$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/project.yml" ]] || {
  echo "Completed parent calibration project is unavailable: $PARENT_BROAD_SHADOW_ROOT" >&2
  exit 2
}
[[ ! -e "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
  echo "Refusing existing calibration target: $REFERENCE_SHADOW_ROOT" >&2
  exit 2
}
[[ -x "$submitter" ]] || {
  echo "Reference-cell-state submitter is unavailable: $submitter" >&2
  exit 2
}

echo "test_host=$observed_host"
echo "test_execution=direct_test_no_sbatch"
echo "test_gpu=0"
echo "test_parent_broad_shadow_root=$PARENT_BROAD_SHADOW_ROOT"
echo "test_expected_parent_cell_count=$EXPECTED_PARENT_CELL_COUNT"
echo "test_reference_shadow_root=$REFERENCE_SHADOW_ROOT"

export PROJECT_DIR RESULTS_ROOT DATASET_ROOT SOURCE_SEGMENTATION_ROOT
export PARENT_BROAD_SHADOW_ROOT EXPECTED_PARENT_CELL_COUNT RUN_STAMP REFERENCE_SHADOW_ROOT
export EXECUTION_MODE=direct_test HPC_CONTAINER_GPU=0
exec "$submitter"
