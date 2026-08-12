#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"
sif_verification_source="frozen_identity_metadata"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export HPC_PROJECT_ROOT
: "${SHADOW_ROOT:?SHADOW_ROOT is required}"
: "${LEGACY_NO_GO:?LEGACY_NO_GO is required for informational provenance only}"
FINALIZE_MODE="${FINALIZE_MODE:-canonical}"
case "$FINALIZE_MODE" in canonical|sharded) ;; *) echo "FINALIZE_MODE must be canonical or sharded" >&2; exit 2 ;; esac

if [[ "$FINALIZE_MODE" == "canonical" ]]; then
  : "${PROJECT_FILE:?PROJECT_FILE is required for canonical finalize}"
  : "${PREDICTION_DIR:?PREDICTION_DIR is required for canonical finalize}"
  FINALIZE_SCRIPT="${BROAD_PHENOTYPE_FINALIZE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/19_finalize_broad_phenotype_shadow.py}"
  required_inputs=("$PROJECT_FILE" "$PREDICTION_DIR" "$LEGACY_NO_GO" "$FINALIZE_SCRIPT")
else
  : "${FEATURE_MANIFEST:?FEATURE_MANIFEST is required for sharded finalize}"
  : "${PREDICTION_ROOT:?PREDICTION_ROOT is required for sharded finalize}"
  : "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required for sharded finalize}"
  : "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required for sharded finalize}"
  FINALIZE_SCRIPT="${BROAD_PHENOTYPE_SHARDED_FINALIZE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/22_merge_broad_phenotype_predictions.py}"
  required_inputs=("$FEATURE_MANIFEST" "$PREDICTION_ROOT" "$MODEL_ACCEPTANCE_RECEIPT" "$MODEL_ACCEPTANCE_SHA256_FILE" "$LEGACY_NO_GO" "$FINALIZE_SCRIPT")
fi
for required_path in "${required_inputs[@]}"; do
  [[ -e "$required_path" ]] || {
    echo "Required broad-phenotype finalize input is unavailable: $required_path" >&2
    exit 2
  }
done

if [[ "$FINALIZE_MODE" == "sharded" ]]; then
  resolved_shadow="$(cd "$SHADOW_ROOT" && pwd -P)"
  for confined_path in "$FEATURE_MANIFEST" "$PREDICTION_ROOT" "$MODEL_ACCEPTANCE_RECEIPT" "$MODEL_ACCEPTANCE_SHA256_FILE"; do
    if [[ -d "$confined_path" ]]; then
      resolved_confined="$(cd "$confined_path" && pwd -P)"
    else
      resolved_confined="$(cd "$(dirname "$confined_path")" && pwd -P)/$(basename "$confined_path")"
    fi
    case "$resolved_confined" in "$resolved_shadow"/*) ;; *) echo "Sharded finalize input must resolve inside SHADOW_ROOT: $resolved_confined" >&2; exit 2 ;; esac
  done
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"

if [[ "$FINALIZE_MODE" == "canonical" ]]; then
  args=(
    "$FINALIZE_SCRIPT"
    --shadow-root "$SHADOW_ROOT"
    --project "$PROJECT_FILE"
    --prediction-dir "$PREDICTION_DIR"
    --legacy-no-go "$LEGACY_NO_GO"
  )
  if [[ -n "${MODEL_DIR:-}" ]]; then
    [[ -d "$MODEL_DIR" ]] || { echo "MODEL_DIR is unavailable: $MODEL_DIR" >&2; exit 2; }
    args+=(--model-dir "$MODEL_DIR")
  fi
else
  model_acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
  [[ "$model_acceptance_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid model-acceptance SHA sidecar" >&2; exit 2; }
  observed_model_acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
  observed_model_acceptance_sha256="${observed_model_acceptance_sha256%%[[:space:]]*}"
  [[ "$model_acceptance_sha256" == "$observed_model_acceptance_sha256" ]] || { echo "Model-acceptance SHA mismatch before sharded finalize" >&2; exit 2; }
  args=(
    "$FINALIZE_SCRIPT"
    --feature-manifest "$FEATURE_MANIFEST"
    --prediction-root "$PREDICTION_ROOT"
    --shadow-root "$SHADOW_ROOT"
    --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT"
    --model-acceptance-sha256 "$model_acceptance_sha256"
    --legacy-no-go "$LEGACY_NO_GO"
  )
  if [[ -n "${CELLS_FILE:-}" ]]; then
    [[ -s "$CELLS_FILE" ]] || { echo "CELLS_FILE is unavailable: $CELLS_FILE" >&2; exit 2; }
    args+=(--cells "$CELLS_FILE")
  fi
fi
if [[ "${FORCE_FINALIZE:-0}" == "1" ]]; then
  args+=(--force)
elif [[ "${FORCE_FINALIZE:-0}" != "0" ]]; then
  echo "FORCE_FINALIZE must be 0 or 1" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=$sif_verification_source"
echo "hpc_container_gpu=0"
echo "shadow_root=$SHADOW_ROOT"
echo "finalize_mode=$FINALIZE_MODE"
echo "project_file=${PROJECT_FILE:-}"
echo "prediction_dir=${PREDICTION_DIR:-}"
echo "feature_manifest=${FEATURE_MANIFEST:-}"
echo "prediction_root=${PREDICTION_ROOT:-}"
echo "model_acceptance_receipt=${MODEL_ACCEPTANCE_RECEIPT:-}"
echo "model_acceptance_sha256=${model_acceptance_sha256:-}"
echo "cells_file=${CELLS_FILE:-}"
echo "model_dir=${MODEL_DIR:-}"
echo "legacy_no_go=$LEGACY_NO_GO"
echo "legacy_no_go_enforcement=informational_only"
printf 'finalize_arg=%s\n' "${args[@]}"

python -I -c 'import sys; print("container_python=" + sys.executable)'
python -I "${args[@]}"
