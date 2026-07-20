#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
RUN_ROOT="${RUN_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940}"
RUN_NAME="${RUN_NAME:-.}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/classification_fusion}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$RUN_ROOT/workflow_status/postsegmentation_manifest}"
CELL_MASK_BRANCH="${CELL_MASK_BRANCH:-original}"

: "${TASK_LIST_FUSION:?TASK_LIST_FUSION must point to the newline-delimited fusion key list}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

FUSION_KEY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST_FUSION" | tr -d '[:space:]')"
if [[ -z "$FUSION_KEY" ]]; then
  echo "No fusion key found for task ${SLURM_ARRAY_TASK_ID} in ${TASK_LIST_FUSION}" >&2
  exit 2
fi

if [[ ! "$FUSION_KEY" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]]; then
  echo "Invalid fusion key: $FUSION_KEY" >&2
  exit 2
fi
FIELD_RECORD="$FIELD_MANIFEST_DIR/records/${FUSION_KEY%%_*}/$FUSION_KEY.json"
if [[ ! -s "$FIELD_RECORD" ]]; then
  echo "Missing field record for $FUSION_KEY: $FIELD_RECORD" >&2
  exit 2
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_fusion_mpl_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_DIR"

ARGS=(
  cellpose_pipeline/scripts/08_fuse_multichannel_classification.py
  --run-root "$RUN_ROOT"
  --input-root "$INPUT_ROOT"
  --out-dir "$OUT_DIR"
  --key "$FUSION_KEY"
  --field-record "$FIELD_RECORD"
  --cell-mask-branch "$CELL_MASK_BRANCH"
  --per-key-only
)

if [[ "$RUN_NAME" != "." && -n "$RUN_NAME" ]]; then
  if [[ -z "${COMBINED_RUN:-}" && -d "$RUN_ROOT/Combined/$RUN_NAME" ]]; then
    COMBINED_RUN="$RUN_ROOT/Combined/$RUN_NAME"
  fi
  if [[ -z "${BRIGHTFIELD_RUN:-}" && -d "$RUN_ROOT/Brightfield/$RUN_NAME" ]]; then
    BRIGHTFIELD_RUN="$RUN_ROOT/Brightfield/$RUN_NAME"
  fi
  if [[ -z "${DEAD_RUN:-}" && -d "$RUN_ROOT/Dead/$RUN_NAME" ]]; then
    DEAD_RUN="$RUN_ROOT/Dead/$RUN_NAME"
  fi
  if [[ -z "${NUCLEI_RUN:-}" && -d "$RUN_ROOT/Nuclei/$RUN_NAME" ]]; then
    NUCLEI_RUN="$RUN_ROOT/Nuclei/$RUN_NAME"
  fi
fi

if [[ -n "${COMBINED_RUN:-}" ]]; then
  ARGS+=(--combined-run "$COMBINED_RUN")
fi
if [[ -n "${BRIGHTFIELD_RUN:-}" ]]; then
  ARGS+=(--brightfield-run "$BRIGHTFIELD_RUN")
fi
if [[ -n "${DEAD_RUN:-}" ]]; then
  ARGS+=(--dead-run "$DEAD_RUN")
fi
if [[ -n "${NUCLEI_RUN:-}" ]]; then
  ARGS+=(--nuclei-run "$NUCLEI_RUN")
fi

if [[ "${FORCE_FUSION:-0}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "run_root=$RUN_ROOT"
echo "run_name=$RUN_NAME"
echo "out_dir=$OUT_DIR"
echo "task_list_fusion=$TASK_LIST_FUSION"
echo "fusion_key=$FUSION_KEY"
echo "field_record=$FIELD_RECORD"
echo "cell_mask_branch=$CELL_MASK_BRANCH"
echo "combined_run=${COMBINED_RUN:-unset}"
echo "brightfield_run=${BRIGHTFIELD_RUN:-unset}"
echo "dead_run=${DEAD_RUN:-unset}"
echo "nuclei_run=${NUCLEI_RUN:-unset}"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python_bin=$PYTHON_BIN"
printf 'fusion_arg=%s\n' "${ARGS[@]}"

"$PYTHON_BIN" -I "${ARGS[@]}"
