#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline_v2}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
CELL_RUN_ROOT="${CELL_RUN_ROOT:-$RESULTS_ROOT/nucleus_aware_cell_refinement_safe_hpc_20260711/candidates/safe10_relabel}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/shape_aware_nucleus_split_hpc_20260711}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "run_root=$RUN_ROOT"
echo "cell_run_root=$CELL_RUN_ROOT"
echo "out_root=$OUT_ROOT"

extra_args=()
if [[ -n "${KEYS:-}" ]]; then
  key_string="${KEYS//,/ }"
  key_string="${key_string//:/ }"
  read -r -a key_values <<< "$key_string"
  extra_args+=(--keys "${key_values[@]}")
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/09_apply_shape_aware_nucleus_splits.py \
  --run-root "$RUN_ROOT" \
  --cell-run-root "$CELL_RUN_ROOT" \
  --input-root "$INPUT_ROOT" \
  --out-root "$OUT_ROOT" \
  "${extra_args[@]}"

echo "shape_aware_nucleus_split_screen_complete=1"
