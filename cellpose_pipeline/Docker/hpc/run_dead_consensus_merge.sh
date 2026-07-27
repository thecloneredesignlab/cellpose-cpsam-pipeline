#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_NAME="${RUN_NAME:-.}"
EXPECTED_KEYS="${EXPECTED_KEYS:-}"
DEAD_RUN="$RUN_ROOT/Dead"
if [[ "$RUN_NAME" != "." && -n "$RUN_NAME" ]]; then
  DEAD_RUN="$DEAD_RUN/$RUN_NAME"
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

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
