#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT is required}"
WORK_DIR="${WORK_DIR:-$CLASSIFICATION_ROOT/workflow_status/late_death_trajectory}"
EXPECTED_FIELDS_PER_BRANCH="${EXPECTED_FIELDS_PER_BRANCH:-27200}"
EXPECTED_WELLS="${EXPECTED_WELLS:-80}"
FORCE_LATE_DEATH="${FORCE_LATE_DEATH:-0}"
CALIBRATION_GO_NO_GO="${CALIBRATION_GO_NO_GO:?CALIBRATION_GO_NO_GO is required}"
SEGMENTATION_FREEZE_RECEIPT="${SEGMENTATION_FREEZE_RECEIPT:?SEGMENTATION_FREEZE_RECEIPT is required}"
WELL_INDEX="${WELL_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ ! "$WELL_INDEX" =~ ^[1-9][0-9]*$ ]]; then
  echo "WELL_INDEX or SLURM_ARRAY_TASK_ID must be a positive integer" >&2
  exit 2
fi
if (( WELL_INDEX > EXPECTED_WELLS )); then
  echo "Well array index exceeds EXPECTED_WELLS: $WELL_INDEX > $EXPECTED_WELLS" >&2
  exit 2
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_late_death_well_${SLURM_ARRAY_JOB_ID:-manual}_${WELL_INDEX}"
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
    echo "Required prepared well-task input is missing: $required" >&2
    exit 2
  }
done

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=$WELL_INDEX"
echo "host=$(hostname)"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "work_dir=$WORK_DIR"

ARGS=(
  cellpose_pipeline/scripts/14_apply_late_dead_trajectory_refinement.py
  --classification-root "$CLASSIFICATION_ROOT"
  --dataset-root "$WORK_DIR"
  --calibration-go-no-go "$CALIBRATION_GO_NO_GO"
  --segmentation-freeze-receipt "$SEGMENTATION_FREEZE_RECEIPT"
  --mode well
  --well-index "$WELL_INDEX"
  --workers 1
  --expected-fields-per-branch "$EXPECTED_FIELDS_PER_BRANCH"
  --expected-wells "$EXPECTED_WELLS"
)
if [[ "$FORCE_LATE_DEATH" == "1" ]]; then
  ARGS+=(--force)
fi

"$PYTHON_BIN" -I "${ARGS[@]}"
