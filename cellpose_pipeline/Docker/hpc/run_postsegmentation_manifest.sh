#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
TASK_LIST_FUSION="${TASK_LIST_FUSION:?TASK_LIST_FUSION is required}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$RUN_ROOT/workflow_status/postsegmentation_manifest}"
NUCLEATED_BRANCH_ROOT="${NUCLEATED_BRANCH_ROOT:-$RUN_ROOT/nucleated_only}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

cd "$PROJECT_DIR"
ARGS=(
  cellpose_pipeline/scripts/06_build_postsegmentation_field_manifest.py
  --input-root "$INPUT_ROOT"
  --run-root "$RUN_ROOT"
  --task-list "$TASK_LIST_FUSION"
  --out-dir "$FIELD_MANIFEST_DIR"
  --nucleated-branch-root "$NUCLEATED_BRANCH_ROOT"
)
if [[ "${FORCE_FIELD_MANIFEST:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "input_root=$INPUT_ROOT"
echo "run_root=$RUN_ROOT"
echo "task_list_fusion=$TASK_LIST_FUSION"
echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "nucleated_branch_root=$NUCLEATED_BRANCH_ROOT"
"$PYTHON_BIN" -I "${ARGS[@]}"
