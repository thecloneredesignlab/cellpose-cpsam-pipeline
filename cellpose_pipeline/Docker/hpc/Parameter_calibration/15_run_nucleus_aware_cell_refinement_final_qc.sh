#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
BASELINE_ROOT="${BASELINE_ROOT:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
SCREEN_ROOT="${SCREEN_ROOT:-$RESULTS_ROOT/nucleus_aware_cell_refinement_safe_hpc_20260711}"
SELECTED_CANDIDATE="${SELECTED_CANDIDATE:-safe10_relabel}"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT_DIR"
candidate_root="$SCREEN_ROOT/candidates/$SELECTED_CANDIDATE"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "selected_candidate=$SELECTED_CANDIDATE"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
  --run-root "$candidate_root" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$candidate_root/alignment"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/17_render_nucleus_aware_cell_refinement_qc.py \
  --baseline-run-root "$BASELINE_ROOT" \
  --candidate-run-root "$candidate_root" \
  --input-root "$INPUT_ROOT" \
  --repair-events "$SCREEN_ROOT/repair_events.csv" \
  --candidate-tag "$SELECTED_CANDIDATE" \
  --out-dir "$SCREEN_ROOT/validation/qc_comparisons" \
  --max-crops 6 \
  --crop-size 192

echo "nucleus_aware_cell_refinement_final_qc_complete=1"
