#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/cellpose-cpsam-pipeline_v2}"
INPUT_ROOT="${INPUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
RUN_ROOT="${RUN_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/20260626_SUM159_AC_Exp1_SeparateImages_cpsam_v2_profiles}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/classification_fusion}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_fusion_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_DIR"

ARGS=(
  cellpose_pipeline/scripts/29_fuse_multichannel_classification.py
  --run-root "$RUN_ROOT"
  --input-root "$INPUT_ROOT"
  --out-dir "$OUT_DIR"
)

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
if [[ "${CONTINUE_ON_ERROR:-1}" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi
if [[ -n "${FUSION_LIMIT:-}" ]]; then
  ARGS+=(--limit "$FUSION_LIMIT")
fi
if [[ -n "${FUSION_KEY:-}" ]]; then
  ARGS+=(--key "$FUSION_KEY")
fi

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "run_root=$RUN_ROOT"
echo "out_dir=$OUT_DIR"
echo "combined_run=${COMBINED_RUN:-unset}"
echo "brightfield_run=${BRIGHTFIELD_RUN:-unset}"
echo "dead_run=${DEAD_RUN:-unset}"
echo "nuclei_run=${NUCLEI_RUN:-unset}"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python_bin=$PYTHON_BIN"
printf 'fusion_arg=%s\n' "${ARGS[@]}"

"$PYTHON_BIN" -I "${ARGS[@]}"
