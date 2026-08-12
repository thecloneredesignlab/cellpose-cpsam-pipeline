#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT

: "${FEATURE_MANIFEST:?FEATURE_MANIFEST is required}"
: "${PREDICTION_ROOT:?PREDICTION_ROOT is required}"
: "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"
: "${SHADOW_ROOT:?SHADOW_ROOT is required}"
: "${PROJECT_FILE:?PROJECT_FILE is required}"
: "${CPA_TRAIN_RECEIPT:?CPA_TRAIN_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
BRANCH="${BRANCH:-}"
case "$BRANCH" in ""|original|nucleated) ;; *) echo "BRANCH must be empty, original, or nucleated" >&2; exit 2 ;; esac

append_bind() {
  local bind="$1"
  case ",${HPC_CONTAINER_BINDS:-}," in
    *",$bind,"*) ;;
    *) HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:+$HPC_CONTAINER_BINDS,}$bind" ;;
  esac
}
append_bind "$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
append_bind "$SHADOW_ROOT:$SHADOW_ROOT"
export HPC_CONTAINER_BINDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export HPC_PROJECT_ROOT
DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
FEATURE_CONFIG="${FEATURE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json}"
CLASSES_FILE="${CLASSES_FILE:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_classes_v1.tsv}"
PREDICT_SCRIPT="${BROAD_PHENOTYPE_SHARD_PREDICT_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/21_predict_broad_phenotype_shard.R}"
CONTRACT_VALIDATOR="${BROAD_PHENOTYPE_INPUT_CONTRACT_VALIDATOR:-$PROJECT_DIR/cellpose_pipeline/scripts/23_validate_broad_phenotype_production_contracts.py}"
for required_path in "$FEATURE_MANIFEST" "$CPA_MODEL_DIR" "$PROJECT_FILE" "$CPA_TRAIN_RECEIPT" "$MODEL_ACCEPTANCE_RECEIPT" "$MODEL_ACCEPTANCE_SHA256_FILE" "$CPA_REFERENCE_ROOT" "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$PREDICT_SCRIPT" "$CONTRACT_VALIDATOR"; do
  [[ -e "$required_path" ]] || { echo "Required sharded prediction input is unavailable: $required_path" >&2; exit 2; }
done

require_inside_shadow() {
  local input_path="$1" label="$2" must_exist="${3:-1}"
  local resolved_shadow resolved_input parent
  resolved_shadow="$(cd "$SHADOW_ROOT" && pwd -P)"
  if [[ "$must_exist" == "1" || -e "$input_path" || -L "$input_path" ]]; then
    if [[ -d "$input_path" ]]; then
      resolved_input="$(cd "$input_path" && pwd -P)"
    else
      resolved_input="$(cd "$(dirname "$input_path")" && pwd -P)/$(basename "$input_path")"
    fi
  else
    parent="$(dirname "$input_path")"; mkdir -p "$parent"
    resolved_input="$(cd "$parent" && pwd -P)/$(basename "$input_path")"
  fi
  case "$resolved_input" in "$resolved_shadow"/*) ;; *) echo "$label must resolve inside SHADOW_ROOT: $resolved_input" >&2; exit 2 ;; esac
}
require_inside_shadow "$FEATURE_MANIFEST" FEATURE_MANIFEST
require_inside_shadow "$CPA_MODEL_DIR" CPA_MODEL_DIR
require_inside_shadow "$PROJECT_FILE" PROJECT_FILE
require_inside_shadow "$CPA_TRAIN_RECEIPT" CPA_TRAIN_RECEIPT
require_inside_shadow "$MODEL_ACCEPTANCE_RECEIPT" MODEL_ACCEPTANCE_RECEIPT
require_inside_shadow "$MODEL_ACCEPTANCE_SHA256_FILE" MODEL_ACCEPTANCE_SHA256_FILE
require_inside_shadow "$PREDICTION_ROOT" PREDICTION_ROOT 0

acceptance_sha256="$(tr -d '[:space:]' < "$MODEL_ACCEPTANCE_SHA256_FILE")"
[[ "$acceptance_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid model-acceptance SHA sidecar" >&2; exit 2; }
observed_acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
observed_acceptance_sha256="${observed_acceptance_sha256%%[[:space:]]*}"
[[ "$observed_acceptance_sha256" == "$acceptance_sha256" ]] || {
  echo "Model-acceptance receipt SHA mismatch: expected=$acceptance_sha256 observed=$observed_acceptance_sha256" >&2
  exit 2
}

header="$(sed -n '1p' "$FEATURE_MANIFEST" | tr -d '\r')"
[[ "$header" == $'key\tfeature_path\treceipt_path' ]] || {
  echo "FEATURE_MANIFEST must have exact key/feature_path/receipt_path header: $FEATURE_MANIFEST" >&2
  exit 2
}
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FEATURE_MANIFEST" | tr -d '\r')"
IFS=$'\t' read -r field_key feature_shard feature_receipt extra <<< "$row"
[[ -z "${extra:-}" && "$field_key" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]] || {
  echo "Invalid feature manifest row for array task $SLURM_ARRAY_TASK_ID: $row" >&2
  exit 2
}
[[ -s "$feature_shard" && -s "$feature_receipt" ]] || {
  echo "Feature shard or receipt is unavailable: feature=$feature_shard receipt=$feature_receipt" >&2
  exit 2
}
require_inside_shadow "$feature_shard" feature_shard
require_inside_shadow "$feature_receipt" feature_receipt
well="${field_key%%_*}"
feature_name="${feature_shard##*/}"
case "$feature_name" in
  "${field_key}__original_broad_phenotype_features.tsv") shard_branch=original ;;
  "${field_key}__nucleated_broad_phenotype_features.tsv") shard_branch=nucleated ;;
  *) echo "Feature shard filename does not encode the frozen key/branch contract: $feature_shard" >&2; exit 2 ;;
esac
if [[ -n "$BRANCH" && "$BRANCH" != "$shard_branch" ]]; then
  echo "Feature shard branch differs from BRANCH: shard=$shard_branch requested=$BRANCH" >&2
  exit 2
fi
output_tsv="$PREDICTION_ROOT/shards/$well/${field_key}__${shard_branch}_broad_phenotype_predictions.tsv"
output_receipt="$PREDICTION_ROOT/receipts/$well/${field_key}__${shard_branch}.json"
mkdir -p "$(dirname "$output_tsv")" "$(dirname "$output_receipt")"

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"
args=(
  "$PREDICT_SCRIPT"
  --reference-root "$CPA_REFERENCE_ROOT"
  --dependency-lock "$DEPENDENCY_LOCK"
  --model-dir "$CPA_MODEL_DIR"
  --model-acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT"
  --model-acceptance-sha256 "$acceptance_sha256"
  --feature-shard "$feature_shard"
  --feature-receipt "$feature_receipt"
  --output-tsv "$output_tsv"
  --output-receipt "$output_receipt"
)
if [[ "${FORCE_PREDICT:-0}" == "1" ]]; then args+=(--force); elif [[ "${FORCE_PREDICT:-0}" != "0" ]]; then echo "FORCE_PREDICT must be 0 or 1" >&2; exit 2; fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "field_key=$field_key"
echo "branch=$shard_branch"
echo "feature_shard=$feature_shard"
echo "feature_receipt=$feature_receipt"
echo "prediction_tsv=$output_tsv"
echo "prediction_receipt=$output_receipt"
echo "cpa_reference_root=$CPA_REFERENCE_ROOT"
echo "cpa_reference_bind_mode=read_only"
echo "model_acceptance_receipt=$MODEL_ACCEPTANCE_RECEIPT"
echo "model_acceptance_sha256=$acceptance_sha256"
printf 'predict_arg=%s\n' "${args[@]}"

python -I "$CONTRACT_VALIDATOR" verify-model-acceptance \
  --shadow-root "$SHADOW_ROOT" \
  --project "$PROJECT_FILE" \
  --model-dir "$CPA_MODEL_DIR" \
  --train-receipt "$CPA_TRAIN_RECEIPT" \
  --feature-config "$FEATURE_CONFIG" \
  --classes-file "$CLASSES_FILE" \
  --submission-preflight "$SHADOW_ROOT/workflow_status/submission_preflight.tsv" \
  --acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT" \
  --expected-sha256 "$acceptance_sha256"
Rscript --version
Rscript "${args[@]}"
