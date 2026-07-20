#!/bin/bash
set -euo pipefail

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results}"

RUN_LABEL="${RUN_LABEL:-20260626_SUM159_AC_Exp1_SeparateImages_cpsam_v2_full}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/${RUN_LABEL}_${RUN_STAMP}}"

if [[ -e "$OUT_ROOT" ]]; then
  suffix=1
  while [[ -e "${OUT_ROOT}_${suffix}" ]]; do
    suffix=$((suffix + 1))
  done
  OUT_ROOT="${OUT_ROOT}_${suffix}"
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory does not exist: $PROJECT_DIR" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input root does not exist: $INPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULTS_ROOT"

export PROJECT_DIR
export INPUT_ROOT
export OUT_ROOT
export EXPECTED_SUBDIRS="${EXPECTED_SUBDIRS:-3}"
export EXPECTED_IMAGES="${EXPECTED_IMAGES:-}"
export SKIP_SUBDIRS="${SKIP_SUBDIRS:-Dead_Uncalibrated}"

export SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-cpsam_v2_full_${RUN_STAMP}}"
export SBATCH_QOS="${SBATCH_QOS:-small}"
export SBATCH_TIME="${SBATCH_TIME:-1-00:00:00}"
export SBATCH_CPUS="${SBATCH_CPUS:-2}"
export SBATCH_MEM="${SBATCH_MEM:-8G}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:a30:1}"

export USE_GPU="${USE_GPU:-1}"
export SEGMENTATION_ONLY="${SEGMENTATION_ONLY:-0}"
export FORCE_CLASSIFICATION="${FORCE_CLASSIFICATION:-1}"

echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "results_root=$RESULTS_ROOT"
echo "out_root=$OUT_ROOT"
echo "skip_subdirs=$SKIP_SUBDIRS"
echo "expected_subdirs=$EXPECTED_SUBDIRS"
echo "expected_images=${EXPECTED_IMAGES:-unset}"
echo "sbatch_job_name=$SBATCH_JOB_NAME"
echo "sbatch_qos=$SBATCH_QOS"
echo "sbatch_time=$SBATCH_TIME"
echo "sbatch_cpus=$SBATCH_CPUS"
echo "sbatch_mem=$SBATCH_MEM"
echo "sbatch_gres=$SBATCH_GRES"
echo "segmentation_only=$SEGMENTATION_ONLY"
echo "force_classification=$FORCE_CLASSIFICATION"

bash "$PROJECT_DIR/cellpose_pipeline/hpc/orchestrate_cellpose_cpsam_full_array.sh"
