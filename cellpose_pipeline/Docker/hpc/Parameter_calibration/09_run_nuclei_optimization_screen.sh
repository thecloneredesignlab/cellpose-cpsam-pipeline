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
BASELINE_RUN="${BASELINE_RUN:-$RESULTS_ROOT/largetest_full_fusion_20260708_213504}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/nuclei_optimization_screen_hpc_20260710}"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nuclei_optimization_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "baseline_run=$BASELINE_RUN"
echo "out_dir=$OUT_DIR"
echo "python_bin=$PYTHON_BIN"
echo "runtime=apptainer_sif"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi
"$PYTHON_BIN" -I -c 'from importlib import metadata; import torch; print("cellpose_version", metadata.version("cellpose")); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count())'

"$PYTHON_BIN" cellpose_pipeline/scripts/Parameter_calibration/14_tune_nuclei_segmentation.py \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$BASELINE_RUN" \
  --out-dir "$OUT_DIR" \
  --use-gpu \
  --save-previews \
  --no-save-masks

echo "nuclei_optimization_screen_complete=1"
