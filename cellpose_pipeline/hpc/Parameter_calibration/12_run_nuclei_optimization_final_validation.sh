#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
BASELINE_RUN="${BASELINE_RUN:-$RESULTS_ROOT/largetest_full_fusion_20260708_213504}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nuclei_final_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR/Nuclei"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "baseline_run=$BASELINE_RUN"
echo "out_dir=$OUT_DIR"
echo "python_bin=$PYTHON_BIN"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi
"$PYTHON_BIN" -I -c 'from importlib import metadata; import torch; print("cellpose_version", metadata.version("cellpose")); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count())'

"$PYTHON_BIN" -I cellpose_pipeline/scripts/01_segment_images.py \
  --dir "$INPUT_ROOT/Nuclei" \
  --run-name . \
  --out-root "$OUT_DIR/Nuclei" \
  --profile-mode auto \
  --skip-unknown-profiles \
  --flat-profile-output \
  --segmentation-only \
  --no-enable-high-density-profiles \
  --no-auto-generate-density-calls \
  --use-gpu

if [[ ! -e "$OUT_DIR/Combined" ]]; then
  ln -s "$BASELINE_RUN/Combined" "$OUT_DIR/Combined"
fi
if [[ ! -e "$OUT_DIR/Brightfield" ]]; then
  ln -s "$BASELINE_RUN/Brightfield" "$OUT_DIR/Brightfield"
fi
if [[ -d "$BASELINE_RUN/classification_fusion" && ! -e "$OUT_DIR/classification_fusion" ]]; then
  ln -s "$BASELINE_RUN/classification_fusion" "$OUT_DIR/classification_fusion"
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/02_call_high_density_from_nuclei_masks.py \
  --run-dir "$OUT_DIR" \
  --out-csv "$OUT_DIR/qc/density_calls.csv" \
  --nuclei-count-threshold 4000 \
  --nuclei-mask-fraction-threshold 0.22 \
  --nuclei-median-nn-threshold 16.0 \
  --no-enable-mask-fraction-trigger

"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
  --run-root "$OUT_DIR" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$OUT_DIR/nuclear_cell_alignment_analysis"

echo "nuclei_optimization_final_validation_complete=1"
