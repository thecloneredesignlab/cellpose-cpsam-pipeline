#!/usr/bin/env bash
set -euo pipefail

required_host="hpctpa3pc0009"
observed_host="$(hostname -s)"
if [[ "$observed_host" != "$required_host" ]]; then
  echo "Broad-phenotype calibration must be run only after login-node ssh to $required_host; observed=$observed_host" >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Broad-phenotype calibration is a direct-node test and must not run inside a Slurm allocation" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
: "${RESULTS_ROOT:?RESULTS_ROOT must be the explicit experiment results directory}"
: "${DATASET_ROOT:?DATASET_ROOT must be the explicit raw-image dataset root}"
: "${FIELD_MANIFEST_DIR:?FIELD_MANIFEST_DIR must be the explicit Stage 06 manifest root}"
: "${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT must be the explicit immutable current-classification root}"
: "${LEGACY_NO_GO:?LEGACY_NO_GO is required for informational provenance only}"

[[ "$(basename "$RESULTS_ROOT")" == "results" ]] || {
  echo "RESULTS_ROOT must end in /results: $RESULTS_ROOT" >&2
  exit 2
}
source_manifest="${SOURCE_FIELD_MANIFEST_FILE:-$FIELD_MANIFEST_DIR/field_manifest.tsv}"
[[ -s "$source_manifest" ]] || { echo "Stage 06 field manifest is unavailable: $source_manifest" >&2; exit 2; }

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
calibration_root="$RESULTS_ROOT/Tests_and_Parameters_calibration"
SHADOW_ROOT="$calibration_root/broad_phenotype_shadow_test_$RUN_STAMP"
bootstrap_root="$calibration_root/broad_phenotype_shadow_test_inputs_$RUN_STAMP"
mkdir -p "$bootstrap_root"
TASK_LIST_INPUT="$bootstrap_root/field_tasks.txt"
TEST_WELL_COUNT="${TEST_WELL_COUNT:-2}"
TEST_FIELDS_PER_WELL="${TEST_FIELDS_PER_WELL:-4}"
[[ "$TEST_WELL_COUNT" =~ ^[1-9][0-9]*$ && "$TEST_WELL_COUNT" -ge 2 && "$TEST_FIELDS_PER_WELL" =~ ^[1-9][0-9]*$ ]] || {
  echo "TEST_WELL_COUNT must be at least 2 and TEST_FIELDS_PER_WELL must be positive" >&2
  exit 2
}

awk -F '\t' -v max_wells="$TEST_WELL_COUNT" -v per_well="$TEST_FIELDS_PER_WELL" '
  NR == 1 { next }
  {
    key=$1
    split(key, parts, "_")
    well=parts[1]
    if (!(well in selected) && selected_count < max_wells) {
      selected[well]=1
      selected_count++
    }
    if ((well in selected) && counts[well] < per_well) {
      print key
      counts[well]++
    }
  }
  END {
    if (selected_count < max_wells) exit 3
    for (well in selected) if (counts[well] < per_well) exit 4
  }
' "$source_manifest" > "$TASK_LIST_INPUT" || {
  echo "Unable to select the requested cross-well calibration field universe" >&2
  exit 2
}
EXPECTED_FIELDS="$(awk 'NF {n++} END {print n+0}' "$TASK_LIST_INPUT")"
expected_requested=$((TEST_WELL_COUNT * TEST_FIELDS_PER_WELL))
[[ "$EXPECTED_FIELDS" -eq "$expected_requested" ]] || {
  echo "Calibration field selection mismatch: expected=$expected_requested observed=$EXPECTED_FIELDS" >&2
  exit 2
}
HELDOUT_WELLS="$(awk -F '_' 'NF && !seen[$1]++ {heldout=$1} END {print heldout}' "$TASK_LIST_INPUT")"
[[ "$HELDOUT_WELLS" =~ ^[A-H][0-9]+$ ]] || {
  echo "Unable to select one explicit calibration heldout well" >&2
  exit 2
}

submitter="$SCRIPT_DIR/../submit_broad_phenotype_shadow_full.sh"
[[ -x "$submitter" ]] || { echo "SIF-backed broad-phenotype submitter is unavailable: $submitter" >&2; exit 2; }

echo "test_host=$observed_host"
echo "test_output_root=$SHADOW_ROOT"
echo "test_input_root=$bootstrap_root"
echo "test_field_count=$EXPECTED_FIELDS"
echo "test_well_count=$TEST_WELL_COUNT"
echo "test_fields_per_well=$TEST_FIELDS_PER_WELL"
echo "test_heldout_well=$HELDOUT_WELLS"
echo "test_direct_concurrency=${DIRECT_TEST_CONCURRENCY:-4}"
echo "test_gpu=0"
echo "test_execution=direct_same_sif"

export PROJECT_DIR RESULTS_ROOT DATASET_ROOT FIELD_MANIFEST_DIR CLASSIFICATION_ROOT LEGACY_NO_GO
export RUN_STAMP SHADOW_ROOT TASK_LIST_INPUT EXPECTED_FIELDS HELDOUT_WELLS
export CPA_STAGE_MODE=
unset TEST_CPA_STAGE_MODE
export EXECUTION_MODE=direct_test RESUME_STAGE=phase-a HPC_CONTAINER_GPU=0
exec "$submitter"
