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

: "${SHADOW_ROOT:?SHADOW_ROOT is required}"
: "${CPA_STAGE:?CPA_STAGE is required}"
PROJECT_FILE="${PROJECT_FILE:-$SHADOW_ROOT/projection_input/representative_umap/project.yml}"

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
DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
STAGE_SCRIPT="${BROAD_PHENOTYPE_CPA_STAGE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py}"
for required_path in "$PROJECT_FILE" "$CPA_REFERENCE_ROOT" "$DEPENDENCY_LOCK" "$STAGE_SCRIPT"; do
  [[ -e "$required_path" ]] || {
    echo "Required Cell Phenotype Annotator stage input is unavailable: $required_path" >&2
    exit 2
  }
done

case "$CPA_STAGE" in
  validate|umap|annotate|annotation-import|review-build|review-import|train|predict|report) ;;
  *) echo "Unsupported CPA_STAGE: $CPA_STAGE" >&2; exit 2 ;;
esac

require_shadow_input() {
  local input_path="$1"
  local label="$2"
  local resolved_input resolved_shadow
  if [[ -d "$input_path" ]]; then
    resolved_input="$(cd "$input_path" && pwd -P)"
  else
    resolved_input="$(cd "$(dirname "$input_path")" && pwd -P)/$(basename "$input_path")"
  fi
  resolved_shadow="$(cd "$SHADOW_ROOT" && pwd -P)"
  case "$resolved_input" in
    "$resolved_shadow"/*) ;;
    *) echo "$label must be staged inside SHADOW_ROOT for immutable SIF visibility: $resolved_input" >&2; exit 2 ;;
  esac
}

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"

args=(
  "$STAGE_SCRIPT"
  --stage "$CPA_STAGE"
  --project "$PROJECT_FILE"
  --shadow-root "$SHADOW_ROOT"
  --reference-root "$CPA_REFERENCE_ROOT"
  --dependency-lock "$DEPENDENCY_LOCK"
  --rscript Rscript
)
case "${CPA_STAGE_MODE:-}" in
  "") ;;
  check-config) args+=(--check-config) ;;
  dry-run) args+=(--dry-run) ;;
  overwrite) args+=(--overwrite) ;;
  *) echo "CPA_STAGE_MODE must be empty, check-config, dry-run, or overwrite" >&2; exit 2 ;;
esac
case "$CPA_STAGE" in
  validate)
    args+=(--validate-stage "${CPA_VALIDATE_STAGE:-all}")
    if [[ -n "${CPA_VALIDATE_OUTPUT_DIR:-}" ]]; then
      args+=(--output-dir "$CPA_VALIDATE_OUTPUT_DIR")
    fi
    ;;
  annotation-import)
    : "${CPA_SUBMISSION:?CPA_SUBMISSION is required for $CPA_STAGE}"
    [[ -f "$CPA_SUBMISSION" ]] || { echo "CPA_SUBMISSION is unavailable: $CPA_SUBMISSION" >&2; exit 2; }
    require_shadow_input "$CPA_SUBMISSION" CPA_SUBMISSION
    args+=(--submission "$CPA_SUBMISSION")
    ;;
  review-build)
    : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required for review-build}"
    [[ -d "$CPA_ANNOTATION_IMPORT_DIR" ]] || { echo "CPA_ANNOTATION_IMPORT_DIR is unavailable: $CPA_ANNOTATION_IMPORT_DIR" >&2; exit 2; }
    require_shadow_input "$CPA_ANNOTATION_IMPORT_DIR" CPA_ANNOTATION_IMPORT_DIR
    args+=(--annotation-import-dir "$CPA_ANNOTATION_IMPORT_DIR")
    ;;
  review-import)
    : "${CPA_SUBMISSION:?CPA_SUBMISSION is required for review-import}"
    : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required for review-import}"
    [[ -f "$CPA_SUBMISSION" ]] || { echo "CPA_SUBMISSION is unavailable: $CPA_SUBMISSION" >&2; exit 2; }
    [[ -d "$CPA_ANNOTATION_IMPORT_DIR" ]] || { echo "CPA_ANNOTATION_IMPORT_DIR is unavailable: $CPA_ANNOTATION_IMPORT_DIR" >&2; exit 2; }
    require_shadow_input "$CPA_SUBMISSION" CPA_SUBMISSION
    require_shadow_input "$CPA_ANNOTATION_IMPORT_DIR" CPA_ANNOTATION_IMPORT_DIR
    args+=(--submission "$CPA_SUBMISSION" --annotation-import-dir "$CPA_ANNOTATION_IMPORT_DIR")
    ;;
  train)
    : "${CPA_REVIEWED_LABELS:?CPA_REVIEWED_LABELS is required for train}"
    [[ -f "$CPA_REVIEWED_LABELS" ]] || { echo "CPA_REVIEWED_LABELS is unavailable: $CPA_REVIEWED_LABELS" >&2; exit 2; }
    require_shadow_input "$CPA_REVIEWED_LABELS" CPA_REVIEWED_LABELS
    args+=(--reviewed-labels "$CPA_REVIEWED_LABELS")
    ;;
  predict|report)
    : "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required for $CPA_STAGE}"
    [[ -d "$CPA_MODEL_DIR" ]] || { echo "CPA_MODEL_DIR is unavailable: $CPA_MODEL_DIR" >&2; exit 2; }
    require_shadow_input "$CPA_MODEL_DIR" CPA_MODEL_DIR
    args+=(--model-dir "$CPA_MODEL_DIR")
    ;;
esac

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=$sif_verification_source"
echo "hpc_container_gpu=0"
echo "shadow_root=$SHADOW_ROOT"
echo "project_file=$PROJECT_FILE"
echo "cpa_stage=$CPA_STAGE"
echo "cpa_reference_root=$CPA_REFERENCE_ROOT"
echo "cpa_reference_bind_mode=read_only"
printf 'cpa_stage_arg=%s\n' "${args[@]}"

python -I -c 'import sys; print("container_python=" + sys.executable)'
Rscript --version
python -I "${args[@]}"
