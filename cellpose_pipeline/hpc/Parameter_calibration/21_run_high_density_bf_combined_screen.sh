#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
NUCLEI_RUN_ROOT="${NUCLEI_RUN_ROOT:-$RESULTS_ROOT/nuclei_optimization_final_hpc_20260710}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/high_density_bf_combined_optimization_hpc_20260711}"
DENSITY_CALLS="${DENSITY_CALLS:-$NUCLEI_RUN_ROOT/qc/density_calls.csv}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/high_density_bf_combined_candidates.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hd_cell_screen_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$OUT_ROOT" "$MPLCONFIGDIR"
cd "$PROJECT_DIR"

EXTRA_ARGS=()
GPU_ARGS=(--use-gpu)
if [[ "${USE_GPU:-1}" == "0" ]]; then
  GPU_ARGS=(--no-use-gpu)
fi
if [[ -n "${KEYS:-}" ]]; then
  read -r -a KEY_ARRAY <<< "${KEYS//,/ }"
  EXTRA_ARGS+=(--keys "${KEY_ARRAY[@]}")
fi
if [[ -n "${CONFIG_TAGS:-}" ]]; then
  read -r -a CONFIG_ARRAY <<< "${CONFIG_TAGS//,/ }"
  EXTRA_ARGS+=(--config-tags "${CONFIG_ARRAY[@]}")
fi
if [[ -n "${PROFILES:-}" ]]; then
  read -r -a PROFILE_ARRAY <<< "${PROFILES//,/ }"
  EXTRA_ARGS+=(--profiles "${PROFILE_ARRAY[@]}")
fi
if [[ "${INCLUDE_LOW_DENSITY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--include-low-density)
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "nuclei_run_root=$NUCLEI_RUN_ROOT"
echo "density_calls=$DENSITY_CALLS"
echo "candidate_config=$CANDIDATE_CONFIG"
echo "out_root=$OUT_ROOT"
echo "keys=${KEYS:-production_high_density}"
echo "config_tags=${CONFIG_TAGS:-all}"
echo "python_bin=$PYTHON_BIN"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || echo "nvidia_smi_unavailable=1"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/19_tune_high_density_bf_combined.py \
  --input-root "$INPUT_ROOT" \
  --nuclei-run-root "$NUCLEI_RUN_ROOT" \
  --density-calls "$DENSITY_CALLS" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --out-root "$OUT_ROOT" \
  "${GPU_ARGS[@]}" \
  --skip-existing \
  "${EXTRA_ARGS[@]}"

echo "screen_complete=1"
