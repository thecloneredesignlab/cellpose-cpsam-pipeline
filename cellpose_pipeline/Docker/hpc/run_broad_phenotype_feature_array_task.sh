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
: "${FIELD_MANIFEST_DIR:?FIELD_MANIFEST_DIR must be the explicit Stage 06 manifest root}"
: "${FEATURE_ROOT:?FEATURE_ROOT must be the explicit broad-phenotype feature root}"
: "${TASK_LIST:?TASK_LIST must contain one Stage 06 field key per line}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

BRANCH="${BRANCH:-original}"
INCLUDE_NUCLEI_COMPARATOR="${INCLUDE_NUCLEI_COMPARATOR:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
case "$BRANCH" in
  original|nucleated) ;;
  *) echo "BRANCH must be original or nucleated: $BRANCH" >&2; exit 2 ;;
esac
for flag_name in INCLUDE_NUCLEI_COMPARATOR FORCE_FEATURES; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1: $flag_value" >&2
    exit 2
  fi
done

FIELD_KEY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST" | tr -d '[:space:]')"
if [[ ! "$FIELD_KEY" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]]; then
  echo "Invalid or missing field key for array index $SLURM_ARRAY_TASK_ID: $FIELD_KEY" >&2
  exit 2
fi
FIELD_RECORD="$FIELD_MANIFEST_DIR/records/${FIELD_KEY%%_*}/$FIELD_KEY.json"
[[ -s "$FIELD_RECORD" ]] || {
  echo "Stage 06 field record is missing: $FIELD_RECORD" >&2
  exit 2
}

FEATURE_SCRIPT="${BROAD_PHENOTYPE_FEATURE_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/16_extract_broad_phenotype_features.py}"
FEATURE_CONFIG="${FEATURE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json}"
[[ -f "$FEATURE_SCRIPT" ]] || {
  echo "Broad-phenotype feature extractor is unavailable: $FEATURE_SCRIPT" >&2
  exit 2
}
[[ -s "$FEATURE_CONFIG" ]] || {
  echo "Locked broad-phenotype feature configuration is unavailable: $FEATURE_CONFIG" >&2
  exit 2
}

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$FEATURE_ROOT"
cd "$PROJECT_DIR"

args=(
  "$FEATURE_SCRIPT"
  --field-record "$FIELD_RECORD"
  --out-dir "$FEATURE_ROOT"
  --config "$FEATURE_CONFIG"
  --branch "$BRANCH"
)
if [[ "$INCLUDE_NUCLEI_COMPARATOR" == "1" ]]; then
  args+=(--include-nuclei-comparator)
fi
if [[ "$FORCE_FEATURES" == "1" ]]; then
  args+=(--force)
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=$sif_verification_source"
echo "hpc_container_gpu=0"
echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "feature_root=$FEATURE_ROOT"
echo "field_key=$FIELD_KEY"
echo "field_record=$FIELD_RECORD"
echo "feature_config=$FEATURE_CONFIG"
echo "branch=$BRANCH"
printf 'feature_arg=%s\n' "${args[@]}"

python -I -c 'import sys; print("container_python=" + sys.executable)'
python -I "${args[@]}"
