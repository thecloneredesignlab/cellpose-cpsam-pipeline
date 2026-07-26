#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT is required}"
PLATE_MAP="${PLATE_MAP:-$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv}"
WORK_DIR="${WORK_DIR:-$CLASSIFICATION_ROOT/workflow_status/late_death_trajectory}"
EXPECTED_FIELDS_PER_BRANCH="${EXPECTED_FIELDS_PER_BRANCH:-27200}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
FORCE_LATE_DEATH="${FORCE_LATE_DEATH:-0}"
CALIBRATION_GO_NO_GO="${CALIBRATION_GO_NO_GO:?CALIBRATION_GO_NO_GO is required}"
SEGMENTATION_FREEZE_RECEIPT="${SEGMENTATION_FREEZE_RECEIPT:?SEGMENTATION_FREEZE_RECEIPT is required}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_late_death_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$WORK_DIR"

for required in \
  "$CLASSIFICATION_ROOT/classification_fusion/summaries/cell_count_summary.csv" \
  "$CLASSIFICATION_ROOT/classification_fusion_nucleated_only/summaries/cell_count_summary.csv" \
  "$CLASSIFICATION_ROOT/workflow_status/postsegmentation_manifest" \
  "$CALIBRATION_GO_NO_GO" \
  "$SEGMENTATION_FREEZE_RECEIPT" \
  "$PLATE_MAP"; do
  [[ -e "$required" ]] || {
    echo "Required late-death production input is missing: $required" >&2
    exit 2
  }
done

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "work_dir=$WORK_DIR"
echo "plate_map=$PLATE_MAP"
echo "workers=$WORKERS"
echo "expected_fields_per_branch=$EXPECTED_FIELDS_PER_BRANCH"
echo "force_late_death=$FORCE_LATE_DEATH"

BUILDER_ARGS=(
  cellpose_pipeline/scripts/13_build_late_dead_trajectory_dataset.py
  --classification-root "$CLASSIFICATION_ROOT"
  --plate-map "$PLATE_MAP"
  --out-dir "$WORK_DIR"
  --all-wells
  --trajectory-min-hours 2
  --trajectory-max-hours 168
  --d0-fields 320
  --workers "$WORKERS"
  --seed 260723
)
REFINEMENT_ARGS=(
  cellpose_pipeline/scripts/14_apply_late_dead_trajectory_refinement.py
  --classification-root "$CLASSIFICATION_ROOT"
  --dataset-root "$WORK_DIR"
  --calibration-go-no-go "$CALIBRATION_GO_NO_GO"
  --segmentation-freeze-receipt "$SEGMENTATION_FREEZE_RECEIPT"
  --workers "$WORKERS"
  --expected-fields-per-branch "$EXPECTED_FIELDS_PER_BRANCH"
)
if [[ "$FORCE_LATE_DEATH" == "1" ]]; then
  BUILDER_ARGS+=(--force)
  REFINEMENT_ARGS+=(--force)
fi

echo "step=01_build_production_late_death_trajectory_dataset"
"$PYTHON_BIN" -I "${BUILDER_ARGS[@]}"

echo "step=02_apply_frozen_late_death_trajectory_model"
"$PYTHON_BIN" -I "${REFINEMENT_ARGS[@]}"

touch "$WORK_DIR/_SUCCESS"
echo "late_death_production_refinement_complete=1"
echo "production_configuration=$CLASSIFICATION_ROOT/late_death_refinement/production_configuration.json"
echo "refinement_summary=$CLASSIFICATION_ROOT/late_death_refinement/refinement_summary.csv"
