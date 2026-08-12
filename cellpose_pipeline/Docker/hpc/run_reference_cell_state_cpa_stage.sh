#!/usr/bin/env bash
set -euo pipefail

# Run one post-human Cell Phenotype Annotator stage for the independent
# reference_cell_state axis.  This worker deliberately shares only the pinned
# CPA runtime; it does not read or write the current viability classifier.

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"

HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_HPC_CONTAINER_IMAGE" ]] || {
  echo "HPC_CONTAINER_IMAGE must equal the frozen latest SIF path: $DEFAULT_HPC_CONTAINER_IMAGE" >&2
  exit 2
}
[[ "$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT" ]] || {
  echo "CPA_REFERENCE_ROOT must equal the frozen read-only checkout: $DEFAULT_CPA_REFERENCE_ROOT" >&2
  exit 2
}
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
case "${HPC_CONTAINER_NO_MOUNT+x}:${HPC_CONTAINER_NO_MOUNT:-}" in
  :|x:/share) ;;
  *) echo "HPC_CONTAINER_NO_MOUNT must be exactly /share for reference-cell-state work" >&2; exit 2 ;;
esac
HPC_CONTAINER_NO_MOUNT=/share
export HPC_CONTAINER_IMAGE CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT
export HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT is required}"
: "${CPA_STAGE:?CPA_STAGE is required}"
PROJECT_FILE="${PROJECT_FILE:-$REFERENCE_SHADOW_ROOT/projection_input/representative_umap/project.yml}"
BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_ROOT="$DATASET_ROOT/Nuclei"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"
case "/$SOURCE_SEGMENTATION_ROOT/" in
  */classification_*/*)
    echo "SOURCE_SEGMENTATION_ROOT must not be a current classification_* root" >&2
    exit 2
    ;;
esac

resolve_existing() {
  local path="$1"
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  realpath "$path"
}

require_inside_reference_shadow() {
  local input_path="$1" label="$2"
  local resolved_input resolved_shadow
  resolved_input="$(resolve_existing "$input_path")" || {
    echo "$label must be an absolute, existing, non-symlink path: $input_path" >&2
    exit 2
  }
  resolved_shadow="$(resolve_existing "$REFERENCE_SHADOW_ROOT")" || {
    echo "REFERENCE_SHADOW_ROOT must be an absolute, existing, non-symlink directory" >&2
    exit 2
  }
  case "$resolved_input" in
    "$resolved_shadow"/*) ;;
    *) echo "$label must resolve inside REFERENCE_SHADOW_ROOT: $resolved_input" >&2; exit 2 ;;
  esac
}

[[ -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
  echo "REFERENCE_SHADOW_ROOT is not a real directory: $REFERENCE_SHADOW_ROOT" >&2
  exit 2
}
for review_root in "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$review_root" == /* && -d "$review_root" && ! -L "$review_root" ]] || {
    echo "Required reference review root is not an absolute real directory: $review_root" >&2
    exit 2
  }
done
[[ -f "$PROJECT_FILE" && ! -L "$PROJECT_FILE" ]] || {
  echo "PROJECT_FILE is unavailable: $PROJECT_FILE" >&2
  exit 2
}
require_inside_reference_shadow "$PROJECT_FILE" PROJECT_FILE

case "$CPA_STAGE" in
  validate|annotation-import|review-build|review-import|train) ;;
  *)
    echo "Unsupported post-human CPA_STAGE: $CPA_STAGE" >&2
    echo "Allowed pre-freeze stages: validate, annotation-import, review-build, review-import, train" >&2
    exit 2
    ;;
esac

expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *)
    echo "HPC_CONTAINER_BINDS differs from the frozen reference-cell-state post-human allowlist" >&2
    exit 2
    ;;
esac
expected_forward_prefixes="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES"
case "${HPC_CONTAINER_FORWARD_PREFIXES+x}:${HPC_CONTAINER_FORWARD_PREFIXES:-}" in
  :) HPC_CONTAINER_FORWARD_PREFIXES="$expected_forward_prefixes" ;;
  "x:$expected_forward_prefixes") ;;
  *)
    echo "HPC_CONTAINER_FORWARD_PREFIXES differs from the frozen reference-cell-state allowlist" >&2
    exit 2
    ;;
esac
export HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$EXPECTED_HPC_CONTAINER_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT
DEPENDENCY_LOCK="${CPA_DEPENDENCY_LOCK:-$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv}"
STAGE_SCRIPT="${REFERENCE_CELL_STATE_CPA_STAGE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py}"
for required_path in "$CPA_REFERENCE_ROOT" "$DEPENDENCY_LOCK" "$STAGE_SCRIPT"; do
  [[ -e "$required_path" && ! -L "$required_path" ]] || {
    echo "Required reference-cell-state CPA input is unavailable: $required_path" >&2
    exit 2
  }
done

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$PROJECT_DIR"

# /share is first hidden wholesale, then only the five explicit allowlist binds
# are restored.  Verify from inside the container before any human labels are
# imported, reviewed, or used for model training.
hpc_apptainer_exec python -I - \
  "$DATASET_ROOT" "$REFERENCE_SHADOW_ROOT" "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT" <<'PY'
from pathlib import Path
import sys

dataset, reference, brightfield, nuclei, combined_masks = map(Path, sys.argv[1:])
for required in (reference, brightfield, nuclei, combined_masks):
    if not required.is_dir():
        raise SystemExit(f"required allowlisted path is not visible in SIF: {required}")
for forbidden in (dataset / "Dead", dataset / "Combined"):
    if forbidden.exists():
        raise SystemExit(f"forbidden image channel remained visible in SIF: {forbidden}")
results = dataset.parent / "results"
if results.exists():
    leaked = [path for path in results.iterdir() if path.name.startswith("classification_")]
    if leaked:
        raise SystemExit(f"current classification roots remained visible in SIF: {leaked[:3]}")
print("reference_posthuman_container_blinding=PASS")
PY

args=(
  "$STAGE_SCRIPT"
  --stage "$CPA_STAGE"
  --project "$PROJECT_FILE"
  --shadow-root "$REFERENCE_SHADOW_ROOT"
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
      require_inside_reference_shadow "$CPA_VALIDATE_OUTPUT_DIR" CPA_VALIDATE_OUTPUT_DIR
      args+=(--output-dir "$CPA_VALIDATE_OUTPUT_DIR")
    fi
    ;;
  annotation-import)
    : "${CPA_SUBMISSION:?CPA_SUBMISSION is required for annotation-import}"
    [[ -f "$CPA_SUBMISSION" ]] || { echo "CPA_SUBMISSION is unavailable: $CPA_SUBMISSION" >&2; exit 2; }
    require_inside_reference_shadow "$CPA_SUBMISSION" CPA_SUBMISSION
    args+=(--submission "$CPA_SUBMISSION")
    ;;
  review-build)
    : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required for review-build}"
    [[ -d "$CPA_ANNOTATION_IMPORT_DIR" ]] || { echo "CPA_ANNOTATION_IMPORT_DIR is unavailable" >&2; exit 2; }
    require_inside_reference_shadow "$CPA_ANNOTATION_IMPORT_DIR" CPA_ANNOTATION_IMPORT_DIR
    args+=(--annotation-import-dir "$CPA_ANNOTATION_IMPORT_DIR")
    ;;
  review-import)
    : "${CPA_SUBMISSION:?CPA_SUBMISSION is required for review-import}"
    : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required for review-import}"
    [[ -f "$CPA_SUBMISSION" && -d "$CPA_ANNOTATION_IMPORT_DIR" ]] || {
      echo "Review submission or annotation-import directory is unavailable" >&2
      exit 2
    }
    require_inside_reference_shadow "$CPA_SUBMISSION" CPA_SUBMISSION
    require_inside_reference_shadow "$CPA_ANNOTATION_IMPORT_DIR" CPA_ANNOTATION_IMPORT_DIR
    args+=(--submission "$CPA_SUBMISSION" --annotation-import-dir "$CPA_ANNOTATION_IMPORT_DIR")
    ;;
  train)
    : "${CPA_REVIEWED_LABELS:?CPA_REVIEWED_LABELS is required for train}"
    [[ -f "$CPA_REVIEWED_LABELS" ]] || { echo "CPA_REVIEWED_LABELS is unavailable" >&2; exit 2; }
    require_inside_reference_shadow "$CPA_REVIEWED_LABELS" CPA_REVIEWED_LABELS
    args+=(--reviewed-labels "$CPA_REVIEWED_LABELS")
    ;;
esac

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=frozen_identity_metadata"
echo "hpc_container_gpu=0"
echo "classifier_axis=reference_cell_state"
echo "current_classifier_access=forbidden"
echo "site_share_mount_disabled=1"
echo "review_inputs=brightfield,nuclei,combined_segmentation_mask"
echo "forbidden_visible_inputs=dead,combined_rgb,current_classification"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "project_file=$PROJECT_FILE"
echo "cpa_stage=$CPA_STAGE"
echo "cpa_reference_root=$CPA_REFERENCE_ROOT"
echo "cpa_reference_bind_mode=read_only"
printf 'cpa_stage_arg=%s\n' "${args[@]}"

python -I -c 'import sys; print("container_python=" + sys.executable)'
Rscript --version
python -I "${args[@]}"
