#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline_v2}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/nuclear_cell_alignment_analysis}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nuclei_alignment_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "run_root=$RUN_ROOT"
echo "out_dir=$OUT_DIR"
echo "python_bin=$PYTHON_BIN"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/31_analyze_nuclear_cell_alignment.py \
  --run-root "$RUN_ROOT" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$OUT_DIR"

echo "nuclei_alignment_refresh_complete=1"
