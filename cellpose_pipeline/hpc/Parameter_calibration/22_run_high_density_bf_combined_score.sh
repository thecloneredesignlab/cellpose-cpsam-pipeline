#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/high_density_bf_combined_optimization_hpc_20260711}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "out_root=$OUT_ROOT"
echo "python_bin=$PYTHON_BIN"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/20_score_high_density_bf_combined.py \
  --screen-root "$OUT_ROOT" \
  --boundary-tolerance "${BOUNDARY_TOLERANCE:-2.0}" \
  --minimum-improvement "${MINIMUM_IMPROVEMENT:-0.01}"

echo "score_complete=1"
