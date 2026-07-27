#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
CALIBRATION_SHARD_COUNT="${CALIBRATION_SHARD_COUNT:-32}"
CALIBRATION_PARALLELISM="${CALIBRATION_PARALLELISM:-16}"
CALIBRATION_EXPECTED_PAIRS="${CALIBRATION_EXPECTED_PAIRS:-}"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dead_calibration_grab_$$"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

mkdir -p "$CALIBRATION_ROOT/scan" "$CALIBRATION_ROOT/final" "$CALIBRATION_ROOT/logs" "$MPLCONFIGDIR"
cd "$PROJECT_DIR"

echo "host=$(hostname)"
echo "input_root=$INPUT_ROOT"
echo "calibration_root=$CALIBRATION_ROOT"
echo "calibration_shard_count=$CALIBRATION_SHARD_COUNT"
echo "calibration_parallelism=$CALIBRATION_PARALLELISM"

pids=()
for shard_index in $(seq 1 "$CALIBRATION_SHARD_COUNT"); do
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/03_calibrate_dead_combined_blue.py scan \
    --input-root "$INPUT_ROOT" \
    --out-dir "$CALIBRATION_ROOT/scan" \
    --shard-index "$shard_index" \
    --shard-count "$CALIBRATION_SHARD_COUNT" \
    > "$CALIBRATION_ROOT/logs/scan_${shard_index}.log" 2>&1 &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "$CALIBRATION_PARALLELISM" ]]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

PAIR_ARGS=()
if [[ -n "$CALIBRATION_EXPECTED_PAIRS" ]]; then
  PAIR_ARGS+=(--expected-pairs "$CALIBRATION_EXPECTED_PAIRS")
fi
"$PYTHON_BIN" -I cellpose_pipeline/scripts/03_calibrate_dead_combined_blue.py merge \
  --scan-dir "$CALIBRATION_ROOT/scan" \
  --out-dir "$CALIBRATION_ROOT/final" \
  --expected-shards "$CALIBRATION_SHARD_COUNT" \
  "${PAIR_ARGS[@]}"

echo "dead_calibration_grab_complete=1"
echo "dead_calibration_json=$CALIBRATION_ROOT/final/dead_combined_blue_calibration.json"
