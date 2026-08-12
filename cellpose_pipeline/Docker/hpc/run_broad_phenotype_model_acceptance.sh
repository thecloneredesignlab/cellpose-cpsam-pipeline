#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
HPC_CONTAINER_GPU=0
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT

: "${SHADOW_ROOT:?SHADOW_ROOT is required}"
: "${PROJECT_FILE:?PROJECT_FILE is required}"
: "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"
: "${CPA_TRAIN_RECEIPT:?CPA_TRAIN_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_RECEIPT:?MODEL_ACCEPTANCE_RECEIPT is required}"
: "${MODEL_ACCEPTANCE_SHA256_FILE:?MODEL_ACCEPTANCE_SHA256_FILE is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export HPC_PROJECT_ROOT
DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
FEATURE_CONFIG="${FEATURE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json}"
CLASSES_FILE="${CLASSES_FILE:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_classes_v1.tsv}"
PREFLIGHT_RECEIPT="$SHADOW_ROOT/workflow_status/submission_preflight.tsv"
TRAIN_VALIDATOR="${BROAD_PHENOTYPE_INPUT_CONTRACT_VALIDATOR:-$PROJECT_DIR/cellpose_pipeline/scripts/23_validate_broad_phenotype_production_contracts.py}"
ACCEPT_SCRIPT="${BROAD_PHENOTYPE_MODEL_ACCEPT_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/25_accept_broad_phenotype_model.R}"

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
    parent="$(dirname "$input_path")"
    mkdir -p "$parent"
    resolved_input="$(cd "$parent" && pwd -P)/$(basename "$input_path")"
  fi
  case "$resolved_input" in
    "$resolved_shadow"/*) ;;
    *) echo "$label must resolve inside SHADOW_ROOT: $resolved_input" >&2; exit 2 ;;
  esac
}

for required in "$PROJECT_FILE" "$CPA_MODEL_DIR" "$CPA_TRAIN_RECEIPT" "$PREFLIGHT_RECEIPT" "$CPA_REFERENCE_ROOT" "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$TRAIN_VALIDATOR" "$ACCEPT_SCRIPT"; do
  [[ -e "$required" ]] || { echo "Required model-acceptance input is unavailable: $required" >&2; exit 2; }
done
require_inside_shadow "$PROJECT_FILE" PROJECT_FILE
require_inside_shadow "$CPA_MODEL_DIR" CPA_MODEL_DIR
require_inside_shadow "$CPA_TRAIN_RECEIPT" CPA_TRAIN_RECEIPT
require_inside_shadow "$PREFLIGHT_RECEIPT" PREFLIGHT_RECEIPT
require_inside_shadow "$MODEL_ACCEPTANCE_RECEIPT" MODEL_ACCEPTANCE_RECEIPT 0
require_inside_shadow "$MODEL_ACCEPTANCE_SHA256_FILE" MODEL_ACCEPTANCE_SHA256_FILE 0

append_bind() {
  local bind="$1"
  case ",${HPC_CONTAINER_BINDS:-}," in
    *",$bind,"*) ;;
    *) HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:+$HPC_CONTAINER_BINDS,}$bind" ;;
  esac
}
append_bind "$SHADOW_ROOT:$SHADOW_ROOT"
append_bind "$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
export HPC_CONTAINER_BINDS

source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "shadow_root=$SHADOW_ROOT"
echo "project_file=$PROJECT_FILE"
echo "cpa_model_dir=$CPA_MODEL_DIR"
echo "cpa_train_receipt=$CPA_TRAIN_RECEIPT"
echo "model_acceptance_receipt=$MODEL_ACCEPTANCE_RECEIPT"

python -I "$TRAIN_VALIDATOR" verify-train-receipt \
  --shadow-root "$SHADOW_ROOT" \
  --project "$PROJECT_FILE" \
  --train-receipt "$CPA_TRAIN_RECEIPT"

Rscript "$ACCEPT_SCRIPT" \
  --shadow-root "$SHADOW_ROOT" \
  --project "$PROJECT_FILE" \
  --model-dir "$CPA_MODEL_DIR" \
  --train-receipt "$CPA_TRAIN_RECEIPT" \
  --feature-config "$FEATURE_CONFIG" \
  --classes-file "$CLASSES_FILE" \
  --submission-preflight "$PREFLIGHT_RECEIPT" \
  --reference-root "$CPA_REFERENCE_ROOT" \
  --dependency-lock "$DEPENDENCY_LOCK" \
  --output-receipt "$MODEL_ACCEPTANCE_RECEIPT"

acceptance_sha256="$(sha256sum "$MODEL_ACCEPTANCE_RECEIPT")"
acceptance_sha256="${acceptance_sha256%%[[:space:]]*}"
sha_tmp="${MODEL_ACCEPTANCE_SHA256_FILE}.tmp.$$"
printf '%s\n' "$acceptance_sha256" > "$sha_tmp"
if [[ -e "$MODEL_ACCEPTANCE_SHA256_FILE" ]]; then
  cmp -s "$sha_tmp" "$MODEL_ACCEPTANCE_SHA256_FILE" || {
    rm -f "$sha_tmp"
    echo "Existing model-acceptance SHA sidecar differs and was preserved" >&2
    exit 2
  }
  rm -f "$sha_tmp"
else
  mv "$sha_tmp" "$MODEL_ACCEPTANCE_SHA256_FILE"
fi

python -I "$TRAIN_VALIDATOR" verify-model-acceptance \
  --shadow-root "$SHADOW_ROOT" \
  --project "$PROJECT_FILE" \
  --model-dir "$CPA_MODEL_DIR" \
  --train-receipt "$CPA_TRAIN_RECEIPT" \
  --feature-config "$FEATURE_CONFIG" \
  --classes-file "$CLASSES_FILE" \
  --submission-preflight "$PREFLIGHT_RECEIPT" \
  --acceptance-receipt "$MODEL_ACCEPTANCE_RECEIPT" \
  --expected-sha256 "$acceptance_sha256"

echo "model_acceptance_sha256=$acceptance_sha256"
echo "model_acceptance_sha256_file=$MODEL_ACCEPTANCE_SHA256_FILE"
