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

mkdir -p "$SHAPE_ROOT/qc/shape_strict"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "run_root=$RUN_ROOT"
echo "cell_run_root=$CELL_RUN_ROOT"
echo "fusion_root=$FUSION_ROOT"
echo "shape_root=$SHAPE_ROOT"
echo "task_list_shape=$TASK_LIST_SHAPE"
echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "cell_mask_branch=$CELL_MASK_BRANCH"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/42_merge_shape_strict_shards.py \
  --run-root "$RUN_ROOT" \
  --cell-run-root "$CELL_RUN_ROOT" \
  --fusion-root "$FUSION_ROOT" \
  --shape-root "$SHAPE_ROOT" \
  --task-list "$TASK_LIST_SHAPE" \
  --candidate-tag shape_strict \
  --max-qc-fields "${SHAPE_MAX_QC_FIELDS:-24}"

mapfile -t QC_KEYS < "$SHAPE_ROOT/qc/qc_keys.txt"
if [[ "${#QC_KEYS[@]}" -gt 0 ]]; then
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/38_render_shape_aware_nucleus_split_qc.py \
    --baseline-run-root "$RUN_ROOT" \
    --candidate-run-root "$SHAPE_ROOT" \
    --cell-run-root "$CELL_RUN_ROOT" \
    --input-root "$INPUT_ROOT" \
    --split-events "$SHAPE_ROOT/split_events.csv" \
    --candidate-tag shape_strict \
    --out-dir "$SHAPE_ROOT/qc/shape_strict" \
    --field-manifest-dir "$FIELD_MANIFEST_DIR" \
    --cell-mask-branch "$CELL_MASK_BRANCH" \
    --max-crops "${SHAPE_MAX_CROPS_PER_FIELD:-8}" \
    --crop-size "${SHAPE_QC_CROP_SIZE:-224}" \
    --keys "${QC_KEYS[@]}"
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/43_finalize_shape_strict_qc.py \
  --shape-root "$SHAPE_ROOT" \
  --candidate-tag shape_strict

echo "shape_strict_finalize_complete=1"
