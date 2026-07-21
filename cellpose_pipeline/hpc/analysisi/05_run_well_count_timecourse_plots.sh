#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
EXPECTED_TIMEPOINTS="${EXPECTED_TIMEPOINTS:-85}"
EXPECTED_SITES="${EXPECTED_SITES:-4}"
PLOT_DPI="${PLOT_DPI:-200}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_well_counts_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT/analysis/well_count_timecourses"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "result_root=$RESULT_ROOT"
echo "python_bin=$PYTHON_BIN"
echo "expected_timepoints=$EXPECTED_TIMEPOINTS"
echo "expected_sites=$EXPECTED_SITES"

for branch in fusion fusion-nucleated-only; do
  out_dir="$RESULT_ROOT/analysis/well_count_timecourses/$branch"
  echo "plot_branch=$branch"
  echo "plot_out_dir=$out_dir"
  "$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/04_plot_well_counts_over_time.py \
    "$RESULT_ROOT" \
    --branch "$branch" \
    --out-dir "$out_dir" \
    --layout plate \
    --y-axis shared \
    --strict-completeness \
    --expected-timepoints "$EXPECTED_TIMEPOINTS" \
    --expected-sites "$EXPECTED_SITES" \
    --dpi "$PLOT_DPI"
done

echo "well_count_timecourse_plots_complete=1"
