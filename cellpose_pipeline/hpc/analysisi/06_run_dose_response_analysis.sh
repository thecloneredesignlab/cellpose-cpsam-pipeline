#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
DOSE_DPI="${DOSE_DPI:-220}"
EXPECTED_FILES="${EXPECTED_FILES:-123}"
FORCE_DOSE_RESPONSE="${FORCE_DOSE_RESPONSE:-0}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_dose_response_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT/analysis" "$RESULT_ROOT/workflow_status/dose_response"

for branch in fusion-consensus fusion fusion-nucleated-only; do
  required="$RESULT_ROOT/analysis/well_count_timecourses/$branch/well_time_cell_state_counts.csv"
  [[ -s "$required" ]] || {
    echo "Required well-count table is missing: $required" >&2
    exit 2
  }
done

FINAL_DIR="$RESULT_ROOT/analysis/dose_response"
if [[ -e "$FINAL_DIR" && "$FORCE_DOSE_RESPONSE" != "1" ]]; then
  echo "Refusing to overwrite existing dose-response output without FORCE_DOSE_RESPONSE=1: $FINAL_DIR" >&2
  exit 2
fi
STAGING_DIR=$(mktemp -d "$RESULT_ROOT/analysis/.dose_response.tmp.XXXXXX")
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "result_root=$RESULT_ROOT"
echo "python_bin=$PYTHON_BIN"
echo "dose_dpi=$DOSE_DPI"

for branch in fusion-consensus fusion fusion-nucleated-only; do
  echo "dose_response_branch=$branch"
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/07_plot_dose_response_curves.py \
    "$RESULT_ROOT" \
    --branch "$branch" \
    --metric auc \
    --out-dir "$STAGING_DIR/$branch" \
    --dpi "$DOSE_DPI"
done

actual_files=$(find "$STAGING_DIR" -type f | wc -l)
if [[ "$actual_files" -ne "$EXPECTED_FILES" ]]; then
  echo "Expected $EXPECTED_FILES dose-response files but found $actual_files" >&2
  exit 2
fi
for branch in fusion-consensus fusion fusion-nucleated-only; do
  for relative in \
    auc/hill_fit_parameters.csv \
    day4/hill_fit_parameters.csv \
    day5/hill_fit_parameters.csv \
    gr/gr_delta_summary.csv \
    death/death_delta_summary.csv; do
    test -s "$STAGING_DIR/$branch/$relative"
  done
done

if [[ -e "$FINAL_DIR" ]]; then
  backup="$RESULT_ROOT/analysis/dose_response.backup.$(date +%Y%m%d_%H%M%S)"
  mv "$FINAL_DIR" "$backup"
  echo "previous_dose_response_backup=$backup"
fi
mv "$STAGING_DIR" "$FINAL_DIR"
trap - EXIT
touch "$RESULT_ROOT/workflow_status/dose_response/_SUCCESS"

echo "dose_response_complete=1"
echo "dose_response_root=$FINAL_DIR"
echo "dose_response_files=$actual_files"
