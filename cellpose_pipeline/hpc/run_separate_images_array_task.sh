#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MANIFEST RUN_NAME OUT_ROOT PROJECT_DIR" >&2
  exit 2
fi

MANIFEST=$1
RUN_NAME=$2
OUT_ROOT=$3
PROJECT_DIR=$4
PYTHON_BIN=${PYTHON_BIN:-python}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

IMAGE_PATH=$(sed -n "${TASK_ID}p" "$MANIFEST")
if [[ -z "$IMAGE_PATH" ]]; then
  echo "No image found for SLURM_ARRAY_TASK_ID=$TASK_ID in $MANIFEST" >&2
  exit 1
fi

GPU_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--use-gpu)
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

cd "$PROJECT_DIR"
echo "task_id=$TASK_ID"
echo "image_path=$IMAGE_PATH"
echo "run_name=$RUN_NAME"
echo "out_root=$OUT_ROOT"
echo "python=$PYTHON_BIN"

exec "$PYTHON_BIN" cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --image-path "$IMAGE_PATH" \
  --run-name "$RUN_NAME" \
  --out-root "$OUT_ROOT" \
  --summary-name "segmentation_summary_task_${TASK_ID}.csv" \
  --profile-mode auto \
  --skip-unknown-profiles \
  "${HIGH_DENSITY_ARGS[@]}" \
  "${CLASSIFICATION_ARGS[@]}" \
  "${GPU_ARGS[@]}"
