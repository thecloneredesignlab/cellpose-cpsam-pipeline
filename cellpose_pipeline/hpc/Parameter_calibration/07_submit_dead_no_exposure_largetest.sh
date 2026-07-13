#!/bin/bash
#SBATCH --job-name=dead_noexpo_largetest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00
#SBATCH --qos=small
#SBATCH --gres=gpu:a30:1

set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline_v2}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_DIR/SUM159_AC_Exp1_SeparateImages_largetest}"
BASELINE_RUN="${BASELINE_RUN:-$PROJECT_DIR/segmentation_parameter_tuning/largetest_full_workflow_updated_profiles_a30_20260707_233255}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_DIR/cellpose_pipeline/configs/dead_no_exposure_configs.json}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/segmentation_parameter_tuning/dead_no_exposure_largetest_a30_${RUN_STAMP}}"
STAGE="${STAGE:-dead_no_exposure_round1}"

mkdir -p "$OUT_ROOT"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dead_noexpo_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "baseline_run=$BASELINE_RUN"
echo "config_json=$CONFIG_JSON"
echo "out_root=$OUT_ROOT"
echo "stage=$STAGE"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python=$CONDA_PREFIX/bin/python"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"$CONDA_PREFIX/bin/python" -I cellpose_pipeline/scripts/Parameter_calibration/08_largetest_retune.py \
  --input-root "$INPUT_ROOT" \
  --out-root "$OUT_ROOT" \
  --baseline-run "$BASELINE_RUN" \
  --stage "$STAGE" \
  --config-json "$CONFIG_JSON" \
  --folders Dead \
  --use-gpu \
  --make-contact-sheets
