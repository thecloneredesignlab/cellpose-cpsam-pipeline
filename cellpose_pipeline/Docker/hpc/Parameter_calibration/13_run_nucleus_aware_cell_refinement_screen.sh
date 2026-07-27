#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/nucleus_aware_cell_refinement_hpc_20260711}"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

mkdir -p "$OUT_ROOT"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "run_root=$RUN_ROOT"
echo "out_root=$OUT_ROOT"
echo "python_bin=$PYTHON_BIN"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/15_diagnose_registration_and_refine_cell_masks.py \
  --run-root "$RUN_ROOT" \
  --out-root "$OUT_ROOT" \
  --max-shift 6

echo "nucleus_aware_cell_refinement_screen_complete=1"
