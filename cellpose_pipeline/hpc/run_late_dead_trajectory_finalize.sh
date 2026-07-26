#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT is required}"
WORK_DIR="${WORK_DIR:-$CLASSIFICATION_ROOT/workflow_status/late_death_trajectory}"
EXPECTED_FIELDS_PER_BRANCH="${EXPECTED_FIELDS_PER_BRANCH:-27200}"
EXPECTED_WELLS="${EXPECTED_WELLS:-80}"
CALIBRATION_GO_NO_GO="${CALIBRATION_GO_NO_GO:?CALIBRATION_GO_NO_GO is required}"
SEGMENTATION_FREEZE_RECEIPT="${SEGMENTATION_FREEZE_RECEIPT:?SEGMENTATION_FREEZE_RECEIPT is required}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_late_death_finalize_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

for required in \
  "$WORK_DIR/prepared_refinement/_SUCCESS" \
  "$WORK_DIR/prepared_refinement/PREPARED.json" \
  "$WORK_DIR/prepared_refinement/well_manifest.tsv" \
  "$CALIBRATION_GO_NO_GO" \
  "$SEGMENTATION_FREEZE_RECEIPT"; do
  [[ -f "$required" ]] || {
    echo "Required late-death finalize input is missing: $required" >&2
    exit 2
  }
done

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "work_dir=$WORK_DIR"
echo "expected_fields_per_branch=$EXPECTED_FIELDS_PER_BRANCH"
echo "expected_wells=$EXPECTED_WELLS"

"$PYTHON_BIN" -I \
  cellpose_pipeline/scripts/14_apply_late_dead_trajectory_refinement.py \
  --classification-root "$CLASSIFICATION_ROOT" \
  --dataset-root "$WORK_DIR" \
  --calibration-go-no-go "$CALIBRATION_GO_NO_GO" \
  --segmentation-freeze-receipt "$SEGMENTATION_FREEZE_RECEIPT" \
  --mode finalize \
  --workers 1 \
  --expected-fields-per-branch "$EXPECTED_FIELDS_PER_BRANCH" \
  --expected-wells "$EXPECTED_WELLS"

touch "$WORK_DIR/_SUCCESS"
echo "late_death_production_refinement_complete=1"
echo "production_configuration=$CLASSIFICATION_ROOT/late_death_refinement/production_configuration.json"
echo "refinement_summary=$CLASSIFICATION_ROOT/late_death_refinement/refinement_summary.csv"
