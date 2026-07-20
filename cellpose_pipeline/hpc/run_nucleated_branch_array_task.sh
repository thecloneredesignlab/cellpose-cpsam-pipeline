#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
BRANCH_ROOT="${BRANCH_ROOT:-$RUN_ROOT/nucleated_only}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$RUN_ROOT/workflow_status/postsegmentation_manifest}"
: "${TASK_LIST_BRANCH:?TASK_LIST_BRANCH is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

KEY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST_BRANCH" | tr -d '[:space:]')"
if [[ ! "$KEY" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]]; then
  echo "Invalid or missing key for task ${SLURM_ARRAY_TASK_ID}: $KEY" >&2
  exit 2
fi
FIELD_RECORD="$FIELD_MANIFEST_DIR/records/${KEY%%_*}/$KEY.json"
if [[ ! -s "$FIELD_RECORD" ]]; then
  echo "Missing field record for $KEY: $FIELD_RECORD" >&2
  exit 2
fi
if [[ -s "$BRANCH_ROOT/shards/$KEY/_SUCCESS" && "${FORCE_NUCLEATED_BRANCH:-0}" != "1" ]]; then
  echo "nucleated_branch_shard_already_complete=1 key=$KEY"
  exit 0
fi
if [[ "${FORCE_NUCLEATED_BRANCH:-0}" == "1" ]]; then
  rm -f "$BRANCH_ROOT/shards/$KEY/_SUCCESS"
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

ARGS=(
  cellpose_pipeline/scripts/07_build_nucleated_cell_branch.py
  --run-root "$RUN_ROOT"
  --input-root "$INPUT_ROOT"
  --out-root "$BRANCH_ROOT"
  --key "$KEY"
  --field-record "$FIELD_RECORD"
)
if [[ "${RENDER_NUCLEATED_QC:-0}" == "1" ]]; then
  ARGS+=(--render-qc)
else
  ARGS+=(--no-render-qc)
fi
if [[ "${FORCE_NUCLEATED_BRANCH:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

cd "$PROJECT_DIR"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)"
echo "key=$KEY"
echo "run_root=$RUN_ROOT"
echo "branch_root=$BRANCH_ROOT"
echo "field_record=$FIELD_RECORD"
"$PYTHON_BIN" -I "${ARGS[@]}"
