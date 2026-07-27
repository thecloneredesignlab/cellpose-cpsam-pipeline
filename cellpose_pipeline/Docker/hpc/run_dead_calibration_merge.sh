#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
CALIBRATION_SHARD_COUNT="${CALIBRATION_SHARD_COUNT:?CALIBRATION_SHARD_COUNT is required}"
CALIBRATION_EXPECTED_PAIRS="${CALIBRATION_EXPECTED_PAIRS:-}"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dead_calibration_merge_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR" "$CALIBRATION_ROOT/final"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python
cd "$PROJECT_DIR"

PAIR_ARGS=()
if [[ -n "$CALIBRATION_EXPECTED_PAIRS" ]]; then
  PAIR_ARGS+=(--expected-pairs "$CALIBRATION_EXPECTED_PAIRS")
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/03_calibrate_dead_combined_blue.py merge \
  --scan-dir "$CALIBRATION_ROOT/scan" \
  --out-dir "$CALIBRATION_ROOT/final" \
  --expected-shards "$CALIBRATION_SHARD_COUNT" \
  "${PAIR_ARGS[@]}"

echo "dead_calibration_merge_complete=1"
echo "dead_calibration_json=$CALIBRATION_ROOT/final/dead_combined_blue_calibration.json"
