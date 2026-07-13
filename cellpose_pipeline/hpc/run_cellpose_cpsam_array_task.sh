#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/cellpose-cpsam-pipeline_v2}"
OUT_ROOT="${OUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/20260626_SUM159_AC_Exp1_SeparateImages_cpsam_v2_profiles}"
RUN_NAME="${RUN_NAME:-.}"
INPUT_ROOT="${INPUT_ROOT:-}"

: "${TASK_LIST:?TASK_LIST must point to the tab-delimited image/group task list}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

TASK_ROW="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")"
IFS=$'\t' read -r IMAGE_PATH GROUP_NAME EXTRA_FIELD <<< "$TASK_ROW"
if [[ -z "$IMAGE_PATH" ]]; then
  echo "No image path found for task ${SLURM_ARRAY_TASK_ID} in ${TASK_LIST}" >&2
  exit 2
fi
if [[ -n "${EXTRA_FIELD:-}" ]]; then
  echo "Task list row has more than two tab-delimited fields: ${TASK_ROW}" >&2
  exit 2
fi
if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image path does not exist: ${IMAGE_PATH}" >&2
  exit 2
fi
if [[ -n "${GROUP_NAME:-}" && ( "$GROUP_NAME" == "." || "$GROUP_NAME" == ".." || "$GROUP_NAME" == *"/"* ) ]]; then
  echo "Invalid output group name: ${GROUP_NAME}" >&2
  exit 2
fi

TASK_OUT_ROOT="$OUT_ROOT"
if [[ -n "${GROUP_NAME:-}" ]]; then
  TASK_OUT_ROOT="$OUT_ROOT/$GROUP_NAME"
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

mkdir -p "$TASK_OUT_ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_mpl_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$MPLCONFIGDIR"

GPU_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--use-gpu --gpu-device 0)
fi

CLASSIFICATION_ARGS=()
if [[ "${SEGMENTATION_ONLY:-0}" == "1" ]]; then
  CLASSIFICATION_ARGS+=(--segmentation-only)
fi
if [[ "${FORCE_CLASSIFICATION:-1}" == "1" ]]; then
  CLASSIFICATION_ARGS+=(--force-classification)
fi

HIGH_DENSITY_ARGS=()
if [[ "${ENABLE_HIGH_DENSITY_PROFILES:-1}" == "0" ]]; then
  HIGH_DENSITY_ARGS+=(--no-enable-high-density-profiles)
else
  HIGH_DENSITY_ARGS+=(--enable-high-density-profiles)
  if [[ -n "${HIGH_DENSITY_CALLS_CSV:-}" ]]; then
    HIGH_DENSITY_ARGS+=(--high-density-calls-csv "$HIGH_DENSITY_CALLS_CSV")
  fi
fi

DEAD_CALIBRATION_ARGS=()
if [[ "$GROUP_NAME" == "Dead" && -n "${DEAD_CALIBRATION_JSON:-}" ]]; then
  DEAD_CALIBRATION_ARGS+=(--dead-calibration-json "$DEAD_CALIBRATION_JSON")
  if [[ -n "${DEAD_CALIBRATION_MAP:-}" ]]; then
    DEAD_CALIBRATION_ARGS+=(--dead-calibration-map "$DEAD_CALIBRATION_MAP")
  fi
  DEAD_CALIBRATION_ARGS+=(--dead-calibration-mode "${DEAD_CALIBRATION_MODE:-bounded-background}")
fi

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "task_list=$TASK_LIST"
echo "image_path=$IMAGE_PATH"
echo "group_name=${GROUP_NAME:-unset}"
echo "out_root=$OUT_ROOT"
echo "task_out_root=$TASK_OUT_ROOT"
echo "run_name=$RUN_NAME"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python_bin=$PYTHON_BIN"
echo "pythonpath=${PYTHONPATH:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "enable_high_density_profiles=${ENABLE_HIGH_DENSITY_PROFILES:-1}"
echo "high_density_calls_csv=${HIGH_DENSITY_CALLS_CSV:-unset}"

if [[ "$GROUP_NAME" == "Dead" && "${ENABLE_DEAD_CONSENSUS:-0}" == "1" ]]; then
  : "${INPUT_ROOT:?INPUT_ROOT is required for Dead consensus}"
  : "${DEAD_CALIBRATION_JSON:?DEAD_CALIBRATION_JSON is required for Dead consensus}"
  : "${DEAD_CALIBRATION_MAP:?DEAD_CALIBRATION_MAP is required for Dead consensus}"
  IMAGE_NAME="$(basename "$IMAGE_PATH")"
  KEY="$IMAGE_NAME"
  KEY="${KEY%.tif}"
  KEY="${KEY%.tiff}"
  KEY="${KEY#SUM159_AC_}"
  KEY="${KEY#Exp1_}"
  KEY="${KEY#Dead_}"
  COMBINED_IMAGE="$INPUT_ROOT/Combined/SUM159_AC_${KEY}.tif"
  if [[ ! -f "$COMBINED_IMAGE" ]]; then
    mapfile -t COMBINED_MATCHES < <(find "$INPUT_ROOT/Combined" -maxdepth 1 -type f \( -iname "*${KEY}.tif" -o -iname "*${KEY}.tiff" \))
    if [[ "${#COMBINED_MATCHES[@]}" -ne 1 ]]; then
      echo "Expected one Combined image for Dead key $KEY, found ${#COMBINED_MATCHES[@]}" >&2
      exit 2
    fi
    COMBINED_IMAGE="${COMBINED_MATCHES[0]}"
  fi
  DEAD_OUT_ROOT="$TASK_OUT_ROOT"
  if [[ "$RUN_NAME" != "." && -n "$RUN_NAME" ]]; then
    DEAD_OUT_ROOT="$TASK_OUT_ROOT/$RUN_NAME"
  fi
  DEAD_CONSENSUS_ARGS=(
    cellpose_pipeline/scripts/49_segment_dead_with_combined_blue_consensus.py
    --dead-image "$IMAGE_PATH"
    --combined-image "$COMBINED_IMAGE"
    --out-root "$DEAD_OUT_ROOT"
    --calibration-json "$DEAD_CALIBRATION_JSON"
    --calibration-map "$DEAD_CALIBRATION_MAP"
    --dead-calibration-mode "${DEAD_CALIBRATION_MODE:-bounded-background}"
    --dead-model "${DEAD_MODEL:-cpsam_v2}"
    --dead-diameter "${DEAD_DIAMETER:-22}"
    --dead-cellprob-threshold "${DEAD_CELLPROB_THRESHOLD:--2.75}"
    --blue-model "${DEAD_BLUE_MODEL:-cpsam_v2}"
    --blue-transform "${DEAD_BLUE_TRANSFORM:-blue_excess_mean}"
    --blue-diameter "${DEAD_BLUE_DIAMETER:-22}"
    --blue-cellprob-threshold "${DEAD_BLUE_CELLPROB_THRESHOLD:--3.0}"
  )
  if [[ "${USE_GPU:-1}" == "1" ]]; then
    DEAD_CONSENSUS_ARGS+=(--use-gpu)
  else
    DEAD_CONSENSUS_ARGS+=(--no-use-gpu)
  fi
  if [[ "${FORCE_DEAD_CONSENSUS:-0}" == "1" ]]; then
    DEAD_CONSENSUS_ARGS+=(--force)
  fi
  echo "dead_consensus=1"
  echo "dead_consensus_key=$KEY"
  echo "combined_image=$COMBINED_IMAGE"
  "$PYTHON_BIN" -I "${DEAD_CONSENSUS_ARGS[@]}"
  exit 0
fi

nvidia-smi || echo "nvidia_smi_unavailable=1"

"$PYTHON_BIN" -I cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --image-path "$IMAGE_PATH" \
  --run-name "$RUN_NAME" \
  --out-root "$TASK_OUT_ROOT" \
  --summary-name "segmentation_summary_task_${SLURM_ARRAY_TASK_ID}.csv" \
  --profile-mode auto \
  --skip-unknown-profiles \
  --flat-profile-output \
  "${HIGH_DENSITY_ARGS[@]}" \
  "${DEAD_CALIBRATION_ARGS[@]}" \
  "${CLASSIFICATION_ARGS[@]}" \
  "${GPU_ARGS[@]}"
