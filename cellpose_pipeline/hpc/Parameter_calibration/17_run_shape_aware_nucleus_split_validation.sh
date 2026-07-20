#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
BASELINE_ROOT="${BASELINE_ROOT:-$RESULTS_ROOT/nucleus_aware_cell_refinement_safe_hpc_20260711/candidates/safe10_relabel}"
SCREEN_ROOT="${SCREEN_ROOT:-$RESULTS_ROOT/shape_aware_nucleus_split_hpc_20260711}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT_DIR"

if [[ -n "${CANDIDATES:-}" ]]; then
  candidate_string="${CANDIDATES//,/ }"
  candidate_string="${candidate_string//:/ }"
  read -r -a candidates <<< "$candidate_string"
else
  candidates=(
    shape_sensitive
    shape_balanced
    shape_evidence
    shape_strict
    shape_ultra
  )
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "baseline_root=$BASELINE_ROOT"
echo "screen_root=$SCREEN_ROOT"

for candidate in "${candidates[@]}"; do
  candidate_root="$SCREEN_ROOT/candidates/$candidate"
  echo "validating_candidate=$candidate"
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
    --run-root "$candidate_root" \
    --input-root "$INPUT_ROOT" \
    --out-dir "$candidate_root/alignment" \
    --skip-overlays
done

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/18_score_shape_aware_nucleus_splits.py \
  --baseline-run-root "$BASELINE_ROOT" \
  --baseline-summary "$BASELINE_ROOT/alignment/summary.json" \
  --screen-root "$SCREEN_ROOT" \
  --candidates-root "$SCREEN_ROOT/candidates" \
  --out-dir "$SCREEN_ROOT/validation"

echo "shape_aware_nucleus_split_validation_complete=1"
