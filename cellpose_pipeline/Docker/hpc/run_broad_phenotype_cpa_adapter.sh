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
: "${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT must be the explicit immutable current-classification root}"
: "${FEATURE_ROOT:?FEATURE_ROOT must be the explicit broad-phenotype feature root}"
: "${SHADOW_ROOT:?SHADOW_ROOT must be the isolated broad-phenotype output root}"
: "${TASK_LIST:?TASK_LIST must contain the exact feature field universe}"

FIELD_MANIFEST_FILE="${FIELD_MANIFEST_FILE:-$FIELD_MANIFEST_DIR/field_manifest.tsv}"
CLASSES_FILE="${CLASSES_FILE:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_classes_v1.tsv}"
FEATURE_CONFIG="${FEATURE_CONFIG:-$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json}"
ADAPTER_SCRIPT="${BROAD_PHENOTYPE_ADAPTER_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/17_build_broad_phenotype_cpa_inputs.py}"
PROJECTION_SCRIPT="${BROAD_PHENOTYPE_PROJECTION_SCRIPT:-$PROJECT_DIR/cellpose_pipeline/scripts/18_prepare_broad_phenotype_projection.py}"
INPUT_CONTRACT_VALIDATOR="${BROAD_PHENOTYPE_INPUT_CONTRACT_VALIDATOR:-$PROJECT_DIR/cellpose_pipeline/scripts/23_validate_broad_phenotype_production_contracts.py}"
BRANCH="${BRANCH:-original}"
INCLUDE_NUCLEI_COMPARATOR="${INCLUDE_NUCLEI_COMPARATOR:-1}"
PROJECT_ID="${BROAD_PHENOTYPE_PROJECT_ID:-broad_phenotype}"
PLATE_MAP="${PLATE_MAP:-$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv}"
SPLIT_SEED="${SPLIT_SEED:-20260812}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-5}"

case "$BRANCH" in
  original|nucleated) cpa_branch="$BRANCH" ;;
  *) echo "BRANCH must be original or nucleated: $BRANCH" >&2; exit 2 ;;
esac
case "$INCLUDE_NUCLEI_COMPARATOR" in 0|1) ;; *) echo "INCLUDE_NUCLEI_COMPARATOR must be 0 or 1" >&2; exit 2 ;; esac
for required_file in "$FIELD_MANIFEST_FILE" "$FIELD_MANIFEST_DIR/input_identity_manifest.tsv" "$CLASSES_FILE" "$FEATURE_CONFIG" "$PLATE_MAP" "$ADAPTER_SCRIPT" "$PROJECTION_SCRIPT" "$INPUT_CONTRACT_VALIDATOR" "$TASK_LIST"; do
  [[ -s "$required_file" ]] || {
    echo "Required CPA adapter input is missing or empty: $required_file" >&2
    exit 2
  }
done
[[ -d "$CLASSIFICATION_ROOT" ]] || {
  echo "Explicit current-classification root is unavailable: $CLASSIFICATION_ROOT" >&2
  exit 2
}

manifest_dir="$SHADOW_ROOT/workflow_status/feature_inventory"
feature_manifest="$manifest_dir/${BRANCH}_feature_manifest.tsv"
mkdir -p "$manifest_dir"
temporary_manifest="$manifest_dir/.${BRANCH}_feature_manifest.tsv.tmp.$$"
trap 'rm -f "$temporary_manifest"' EXIT
printf 'key\tfeature_path\treceipt_path\n' > "$temporary_manifest"
field_count=0
while IFS= read -r key || [[ -n "$key" ]]; do
  key="${key//[[:space:]]/}"
  [[ -n "$key" ]] || continue
  if [[ ! "$key" =~ ^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$ ]]; then
    echo "Invalid feature task key: $key" >&2
    exit 2
  fi
  well="${key%%_*}"
  feature_path="$FEATURE_ROOT/shards/$well/${key}__${BRANCH}_broad_phenotype_features.tsv"
  receipt_path="$FEATURE_ROOT/receipts/$well/${key}__${BRANCH}.json"
  [[ -s "$feature_path" && -s "$receipt_path" ]] || {
    echo "Feature shard or receipt is incomplete for $key: feature=$feature_path receipt=$receipt_path" >&2
    exit 2
  }
  printf '%s\t%s\t%s\n' "$key" "$feature_path" "$receipt_path" >> "$temporary_manifest"
  field_count=$((field_count + 1))
done < "$TASK_LIST"
[[ "$field_count" -gt 0 ]] || {
  echo "Feature task list is empty: $TASK_LIST" >&2
  exit 2
}
if [[ -e "$feature_manifest" ]]; then
  if cmp -s "$temporary_manifest" "$feature_manifest"; then
    rm -f "$temporary_manifest"
  else
    echo "Refusing to replace a non-identical frozen feature inventory: $feature_manifest" >&2
    exit 2
  fi
else
  mv "$temporary_manifest" "$feature_manifest"
fi
trap - EXIT

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$SHADOW_ROOT"
cd "$PROJECT_DIR"

args=(
  "$ADAPTER_SCRIPT"
  --field-manifest "$FIELD_MANIFEST_FILE"
  --feature-manifest "$feature_manifest"
  --classes-file "$CLASSES_FILE"
  --feature-config "$FEATURE_CONFIG"
  --shadow-root "$SHADOW_ROOT"
  --branch "$cpa_branch"
  --project-id "$PROJECT_ID"
  --split-seed "$SPLIT_SEED"
  --outer-folds "$OUTER_FOLDS"
  --inner-folds "$INNER_FOLDS"
)
if [[ -n "${HELDOUT_WELLS:-}" ]]; then
  args+=(--heldout-wells "$HELDOUT_WELLS")
else
  args+=(--plate-map "$PLATE_MAP")
fi
if [[ -n "${BROAD_PHENOTYPE_FEATURE_COLUMNS:-}" ]]; then
  args+=(--feature-columns "$BROAD_PHENOTYPE_FEATURE_COLUMNS")
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname -s)"
echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "sif_verification_source=$sif_verification_source"
echo "hpc_container_gpu=0"
echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "field_manifest_file=$FIELD_MANIFEST_FILE"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "feature_root=$FEATURE_ROOT"
echo "feature_manifest=$feature_manifest"
echo "feature_config=$FEATURE_CONFIG"
echo "plate_map=$PLATE_MAP"
echo "shadow_root=$SHADOW_ROOT"
echo "branch=$BRANCH"
echo "cpa_branch=$cpa_branch"
echo "field_count=$field_count"
printf 'adapter_arg=%s\n' "${args[@]}"

python -I -c 'import sys; print("container_python=" + sys.executable)'
alignment_root="$SHADOW_ROOT/workflow_status/feature_input_alignment"
alignment_receipt="$alignment_root/${BRANCH}.json"
alignment_audit="$alignment_root/${BRANCH}.tsv"
alignment_args=(
  "$INPUT_CONTRACT_VALIDATOR" feature-inputs
  --field-manifest "$FIELD_MANIFEST_FILE"
  --input-identity-manifest "$FIELD_MANIFEST_DIR/input_identity_manifest.tsv"
  --feature-manifest "$feature_manifest"
  --shadow-root "$SHADOW_ROOT"
  --branch "$BRANCH"
  --include-nuclei-comparator "$INCLUDE_NUCLEI_COMPARATOR"
  --output-receipt "$alignment_receipt"
  --output-audit "$alignment_audit"
)
printf 'feature_input_alignment_arg=%s\n' "${alignment_args[@]}"
python -I "${alignment_args[@]}"
python -I "${args[@]}"

projection_args=(
  "$PROJECTION_SCRIPT"
  --project "$SHADOW_ROOT/cpa/project.yml"
  --shadow-root "$SHADOW_ROOT"
  --well-split-freeze "$SHADOW_ROOT/workflow_status/adapter/well_split_freeze.tsv"
  --max-fields-per-well "${MAX_UMAP_FIELDS_PER_WELL:-20}"
  --max-cells-per-field "${MAX_UMAP_CELLS_PER_FIELD:-25}"
  --max-cells-per-well "${MAX_UMAP_CELLS_PER_WELL:-500}"
  --seed "${UMAP_SAMPLE_SEED:-20260812}"
)
printf 'projection_arg=%s\n' "${projection_args[@]}"
python -I "${projection_args[@]}"
