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
SCREEN_ROOT="${SCREEN_ROOT:-$RESULTS_ROOT/nucleus_aware_cell_refinement_hpc_20260711}"
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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "baseline_root=$BASELINE_ROOT"
echo "screen_root=$SCREEN_ROOT"

if [[ -n "${CANDIDATES:-}" ]]; then
  candidate_string="${CANDIDATES//,/ }"
  candidate_string="${candidate_string//:/ }"
  read -r -a candidates <<< "$candidate_string"
else
  candidates=(
    strict_relabel
    strict_fill
    balanced_relabel
    balanced_fill
    balanced_oneway
    core4_fill
    safe1_relabel
    safe3_relabel
    safe5_relabel
    safe10_relabel
  )
fi

for candidate in "${candidates[@]}"; do
  candidate_root="$SCREEN_ROOT/candidates/$candidate"
  echo "validating_candidate=$candidate"
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
    --run-root "$candidate_root" \
    --input-root "$INPUT_ROOT" \
    --out-dir "$candidate_root/alignment" \
    --skip-overlays
done

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/16_score_nucleus_aware_cell_refinement.py \
  --baseline-run-root "$BASELINE_ROOT" \
  --baseline-summary "$BASELINE_ROOT/nuclear_cell_alignment_analysis/summary.json" \
  --candidates-root "$SCREEN_ROOT/candidates" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$SCREEN_ROOT/validation"

echo "nucleus_aware_cell_refinement_validation_complete=1"
