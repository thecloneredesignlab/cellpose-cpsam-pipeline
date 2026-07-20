#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_NAME="${RUN_NAME:-.}"
EXPECTED_KEYS="${EXPECTED_KEYS:-}"
DEAD_RUN="$RUN_ROOT/Dead"
if [[ "$RUN_NAME" != "." && -n "$RUN_NAME" ]]; then
  DEAD_RUN="$DEAD_RUN/$RUN_NAME"
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

ARGS=(
  cellpose_pipeline/scripts/05_merge_dead_consensus_outputs.py
  --dead-run "$DEAD_RUN"
)
if [[ -n "$EXPECTED_KEYS" ]]; then
  ARGS+=(--expected-keys "$EXPECTED_KEYS")
fi

cd "$PROJECT_DIR"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "dead_run=$DEAD_RUN"
echo "expected_keys=${EXPECTED_KEYS:-unset}"
"$PYTHON_BIN" -I "${ARGS[@]}"
