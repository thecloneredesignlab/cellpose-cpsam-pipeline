#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 INPUT_ROOT RUN_NAME OUT_ROOT PROJECT_DIR" >&2
  exit 2
fi

INPUT_ROOT=$1
RUN_NAME=$2
OUT_ROOT=$3
PROJECT_DIR=$4
DENSITY_CALLS_NAME="${DENSITY_CALLS_NAME:-density_calls.csv}"

if [[ "${ENABLE_HIGH_DENSITY_PROFILES:-1}" == "0" ]]; then
  echo "density_prepass_skipped=high_density_profiles_disabled"
  exit 0
fi

if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
  RUN_DIR="$OUT_ROOT"
else
  RUN_DIR="$OUT_ROOT/$RUN_NAME"
fi
DENSITY_CALLS_CSV="$RUN_DIR/qc/$DENSITY_CALLS_NAME"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

GPU_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--use-gpu --gpu-device 0)
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_density_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "run_name=$RUN_NAME"
echo "out_root=$OUT_ROOT"
echo "density_calls_csv=$DENSITY_CALLS_CSV"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python_bin=$PYTHON_BIN"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || echo "nvidia_smi_unavailable=1"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --dir "$INPUT_ROOT" \
  --recursive \
  --run-name "$RUN_NAME" \
  --out-root "$OUT_ROOT" \
  --profile-mode auto \
  --skip-unknown-profiles \
  --enable-high-density-profiles \
  --density-only \
  --segmentation-only \
  --continue-on-error \
  --density-calls-name "$DENSITY_CALLS_NAME" \
  "${GPU_ARGS[@]}"

if [[ ! -s "$DENSITY_CALLS_CSV" ]]; then
  echo "Density calls CSV was not created or is empty: $DENSITY_CALLS_CSV" >&2
  exit 3
fi

echo "density_prepass_complete=1"
