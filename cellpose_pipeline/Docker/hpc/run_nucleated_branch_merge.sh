#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
BRANCH_ROOT="${BRANCH_ROOT:-$RUN_ROOT/nucleated_only}"
INPUT_ROOT="${INPUT_ROOT:-$RUN_ROOT}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

cd "$PROJECT_DIR"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "run_root=$RUN_ROOT"
echo "branch_root=$BRANCH_ROOT"
ARGS=(
  cellpose_pipeline/scripts/07_build_nucleated_cell_branch.py
  --run-root "$RUN_ROOT"
  --input-root "$INPUT_ROOT"
  --out-root "$BRANCH_ROOT"
  --merge-shards-only
)
if [[ -n "${TASK_LIST_BRANCH:-}" ]]; then
  ARGS+=(--task-list "$TASK_LIST_BRANCH")
fi
"$PYTHON_BIN" -I "${ARGS[@]}"
