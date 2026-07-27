#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
CALIBRATION_SHARD_COUNT="${CALIBRATION_SHARD_COUNT:?CALIBRATION_SHARD_COUNT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

mkdir -p "$CALIBRATION_ROOT/scan"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)"
echo "input_root=$INPUT_ROOT"
echo "calibration_root=$CALIBRATION_ROOT"
echo "calibration_shard_count=$CALIBRATION_SHARD_COUNT"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/03_calibrate_dead_combined_blue.py scan \
  --input-root "$INPUT_ROOT" \
  --out-dir "$CALIBRATION_ROOT/scan" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$CALIBRATION_SHARD_COUNT"
