#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/cellpose-cpsam-pipeline_v2}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CELL_RUN_ROOT="${CELL_RUN_ROOT:-$RUN_ROOT}"
FUSION_ROOT="${FUSION_ROOT:-$RUN_ROOT/classification_fusion}"
SHAPE_ROOT="${SHAPE_ROOT:-$RUN_ROOT/shape_strict}"
TASK_LIST_SHAPE="${TASK_LIST_SHAPE:?TASK_LIST_SHAPE is required}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$RUN_ROOT/workflow_status/postsegmentation_manifest}"
CELL_MASK_BRANCH="${CELL_MASK_BRANCH:-original}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

SHAPE_KEY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST_SHAPE" | tr -d '[:space:]')"
if [[ ! "$SHAPE_KEY" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]]; then
  echo "Invalid or missing shape_strict key for array task ${SLURM_ARRAY_TASK_ID}: $SHAPE_KEY" >&2
  exit 2
fi
SHARD_ROOT="$SHAPE_ROOT/shards/$SHAPE_KEY"
FIELD_RECORD="$FIELD_MANIFEST_DIR/records/${SHAPE_KEY%%_*}/$SHAPE_KEY.json"
if [[ ! -s "$FIELD_RECORD" ]]; then
  echo "Missing field record for $SHAPE_KEY: $FIELD_RECORD" >&2
  exit 2
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

mkdir -p "$SHARD_ROOT"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)"
echo "shape_key=$SHAPE_KEY"
echo "run_root=$RUN_ROOT"
echo "cell_run_root=$CELL_RUN_ROOT"
echo "fusion_root=$FUSION_ROOT"
echo "shape_root=$SHAPE_ROOT"
echo "shard_root=$SHARD_ROOT"
echo "field_record=$FIELD_RECORD"
echo "cell_mask_branch=$CELL_MASK_BRANCH"

if [[ -f "$SHARD_ROOT/_SUCCESS" && "${FORCE_SHAPE_STRICT:-0}" != "1" ]]; then
  echo "shape_strict_shard_already_complete=1"
  exit 0
fi
if [[ "${FORCE_SHAPE_STRICT:-0}" == "1" ]]; then
  rm -f "$SHARD_ROOT/_SUCCESS"
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/09_apply_shape_aware_nucleus_splits.py \
  --run-root "$RUN_ROOT" \
  --cell-run-root "$CELL_RUN_ROOT" \
  --classification-root "$FUSION_ROOT" \
  --input-root "$INPUT_ROOT" \
  --out-root "$SHARD_ROOT" \
  --keys "$SHAPE_KEY" \
  --field-record "$FIELD_RECORD" \
  --cell-mask-branch "$CELL_MASK_BRANCH" \
  --production-shard \
  --candidate-tags shape_strict

SUCCESS_TMP="$SHARD_ROOT/._SUCCESS.tmp.${SLURM_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID}"
printf 'key=%s\n' "$SHAPE_KEY" > "$SUCCESS_TMP"
mv -f "$SUCCESS_TMP" "$SHARD_ROOT/_SUCCESS"
echo "shape_strict_shard_complete=1"
