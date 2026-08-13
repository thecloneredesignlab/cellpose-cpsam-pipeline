#!/usr/bin/env bash
set -euo pipefail

required_host=hpctpa3pc0009
observed_host="$(hostname -s)"
[[ "$observed_host" == "$required_host" ]] || {
  echo "Expanded reference-cell-state calibration requires login-node ssh followed by ssh to $required_host; observed=$observed_host" >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "Expanded reference-cell-state calibration is direct and cannot run inside Slurm" >&2
  exit 2
}
for forbidden in SBATCH_NODELIST SBATCH_CONSTRAINT SBATCH_EXCLUDE CUDA_VISIBLE_DEVICES; do
  unset "$forbidden"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
HPC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd -P)}"
source "$HPC_ROOT/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_load_sif_identity

EXPERIMENT_ROOT="/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/results}"
DATASET_ROOT="${DATASET_ROOT:-$EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$RESULTS_ROOT/full_fusion_shape_strict_20260711_155940}"
BASE_V2_SHADOW_ROOT="${BASE_V2_SHADOW_ROOT:-$RESULTS_ROOT/reference_cell_state_shadow_v2_20260812_230516}"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$RESULTS_ROOT/broad_phenotype_shadow_20260812_075437}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
REFERENCE_SHADOW_ROOT="${REFERENCE_SHADOW_ROOT:-$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_${RUN_STAMP}_expanded39}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
REFERENCE_SNAPSHOT_ROOT="$CPA_REFERENCE_ROOT/reference/ltee-source"

case "$REFERENCE_SHADOW_ROOT" in
  "$RESULTS_ROOT"/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_*_expanded39) ;;
  *) echo "Expanded calibration root is outside the frozen test namespace: $REFERENCE_SHADOW_ROOT" >&2; exit 2 ;;
esac
[[ "$RESULTS_ROOT" == "$EXPERIMENT_ROOT/results" && -d "$RESULTS_ROOT/Tests_and_Parameters_calibration" ]] || {
  echo "Expanded calibration must write only under the experiment Tests_and_Parameters_calibration root" >&2
  exit 2
}
for root in "$BASE_V2_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$root" == /* && -d "$root" && ! -L "$root" ]] || {
    echo "Required frozen calibration input root is unavailable: $root" >&2
    exit 2
  }
done
if [[ -e "$REFERENCE_SHADOW_ROOT" || -L "$REFERENCE_SHADOW_ROOT" ]]; then
  [[ -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
    echo "Existing expanded calibration root is not a real directory: $REFERENCE_SHADOW_ROOT" >&2
    exit 2
  }
else
  mkdir "$REFERENCE_SHADOW_ROOT"
fi

BASE_PROJECT="$BASE_V2_SHADOW_ROOT/projection_input/representative_umap_v2/project.yml"
BROAD_FEATURES="$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/features.tsv"
EXPANDED_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_projection_expanded_v1.json"
EXPANDED_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/39_build_reference_cell_state_expanded_projection.R"
PROJECT_BUILDER="$PROJECT_DIR/cellpose_pipeline/scripts/40_prepare_reference_cell_state_expanded_annotation_project.py"
HISTORICAL_ADAPTER="$PROJECT_DIR/cellpose_pipeline/scripts/28_build_reference_cell_state_historical_projection.R"
CPA_STAGE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py"
MORPHOLOGY_WORKSPACE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/27_build_reference_morphology_workspace.py"
SEED1_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/33_build_reference_cell_state_seed1_review.R"
RENDER_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/37_render_reference_cell_state_exact_review.py"
DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
EXPANDED_OUTPUT="$REFERENCE_SHADOW_ROOT/workflow_status/expanded_projection_comparison"
PROJECT_FILE="$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/project.yml"
MORPHOLOGY_OUTPUT="$REFERENCE_SHADOW_ROOT/morphology_reference"
RECEIPT="$REFERENCE_SHADOW_ROOT/workflow_status/EXPANDED_CALIBRATION_COMPLETE.tsv"

for file in \
  "$BASE_PROJECT" "$BROAD_FEATURES" "$EXPANDED_CONFIG" "$EXPANDED_SCRIPT" \
  "$PROJECT_BUILDER" "$HISTORICAL_ADAPTER" "$CPA_STAGE_SCRIPT" \
  "$MORPHOLOGY_WORKSPACE_SCRIPT" "$SEED1_SCRIPT" "$RENDER_SCRIPT" \
  "$DEPENDENCY_LOCK" \
  "$REFERENCE_SNAPSHOT_ROOT/code/lib/Utils.R"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    echo "Required expanded calibration file is unavailable: $file" >&2
    exit 2
  }
done

reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"
BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_ROOT="$DATASET_ROOT/Nuclei"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"
for root in "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT"; do
  [[ -d "$root" && ! -L "$root" ]] || { echo "Allowed image evidence root is unavailable: $root" >&2; exit 2; }
done

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$BASE_V2_SHADOW_ROOT:$BASE_V2_SHADOW_ROOT:ro,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) echo "HPC_CONTAINER_BINDS differs from the expanded calibration allowlist" >&2; exit 2 ;;
esac
HPC_CONTAINER_RUNTIME_ROOT="$HPC_ROOT"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU
export HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT HPC_CONTAINER_BINDS
export HPC_CONTAINER_FORWARD_PREFIXES HPC_CONTAINER_RUNTIME_ROOT
source "$HPC_ROOT/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
reference_cell_state_v2_verify_container_rootfs_read_only
source "$HPC_ROOT/util/broad_phenotype_container_identity.sh"
mkdir -p "$REFERENCE_SHADOW_ROOT/workflow_status"
HPC_CONTAINER_IDENTITY_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/hpc_container_identity.json"
if [[ -e "$HPC_CONTAINER_IDENTITY_FILE" || -L "$HPC_CONTAINER_IDENTITY_FILE" ]]; then
  [[ -s "$HPC_CONTAINER_IDENTITY_FILE" && ! -L "$HPC_CONTAINER_IDENTITY_FILE" ]] || {
    echo "Existing expanded calibration SIF identity receipt is invalid" >&2
    exit 2
  }
else
  broad_phenotype_capture_container_identity \
    "$HPC_CONTAINER_IMAGE" \
    "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256" \
    "$HPC_CONTAINER_IDENTITY_FILE"
fi
HPC_CONTAINER_IDENTITY_FILE_SHA256="$(reference_cell_state_v2_sha256 "$HPC_CONTAINER_IDENTITY_FILE")"
export HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ -s "$RECEIPT" && ! -L "$RECEIPT" ]]; then
  started_utc="$(awk -F '\t' '$1=="start_utc" {print $2; exit}' "$RECEIPT")"
  ended_utc="$(awk -F '\t' '$1=="end_utc" {print $2; exit}' "$RECEIPT")"
  [[ -n "$started_utc" && -n "$ended_utc" ]] || { echo "Existing expanded calibration receipt lacks timestamps" >&2; exit 2; }
else
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ended_utc=""
fi

hpc_apptainer_exec python -I - "$DATASET_ROOT" "$REFERENCE_SHADOW_ROOT" "$BASE_V2_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT" <<'PY'
from pathlib import Path
import sys
dataset, output, base, broad = map(Path, sys.argv[1:])
for required in (output, base, broad, dataset / "Brightfield", dataset / "Nuclei"):
    if not required.exists():
        raise SystemExit(f"allowlisted expanded input is invisible: {required}")
for forbidden in (dataset / "Dead", dataset / "Combined"):
    if forbidden.exists():
        raise SystemExit(f"forbidden expanded channel remained visible: {forbidden}")
results = dataset.parent / "results"
if results.exists():
    leaked = [path for path in results.iterdir() if path.name.startswith("classification_")]
    if leaked:
        raise SystemExit(f"current classification root remained visible: {leaked[:3]}")
print("reference_v2_expanded_container_blinding=PASS")
PY

hpc_apptainer_exec Rscript "$EXPANDED_SCRIPT" \
  --check-config \
  --reference-snapshot-root "$REFERENCE_SNAPSHOT_ROOT" \
  --feature-config "$EXPANDED_CONFIG" \
  --historical-adapter-script "$HISTORICAL_ADAPTER"

hpc_apptainer_exec Rscript "$EXPANDED_SCRIPT" \
  --project "$BASE_PROJECT" \
  --broad-features "$BROAD_FEATURES" \
  --reference-snapshot-root "$REFERENCE_SNAPSHOT_ROOT" \
  --feature-config "$EXPANDED_CONFIG" \
  --historical-adapter-script "$HISTORICAL_ADAPTER" \
  --output-dir "$EXPANDED_OUTPUT" \
  --dbscan-cores 8 \
  --overwrite

hpc_apptainer_exec python -I "$PROJECT_BUILDER" \
  --base-project "$BASE_PROJECT" \
  --base-shadow-root "$BASE_V2_SHADOW_ROOT" \
  --expanded-projection "$EXPANDED_OUTPUT" \
  --reference-shadow-root "$REFERENCE_SHADOW_ROOT" \
  --expected-cell-count 32000 \
  --overwrite

for stage in validate umap annotate; do
  args=(
    "$CPA_STAGE_SCRIPT" --stage "$stage" --project "$PROJECT_FILE"
    --shadow-root "$REFERENCE_SHADOW_ROOT" --reference-root "$CPA_REFERENCE_ROOT"
    --dependency-lock "$DEPENDENCY_LOCK" --rscript Rscript
  )
  [[ "$stage" != validate ]] || args+=(--validate-stage umap)
  hpc_apptainer_exec python -I "${args[@]}"
done

mapfile -t annotation_manifests < <(
  find "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/runs" \
    -type f -name annotation_manifest.json -print | LC_ALL=C sort
)
[[ "${#annotation_manifests[@]}" -eq 1 ]] || {
  echo "Expected exactly one expanded annotation generation; observed=${#annotation_manifests[@]}" >&2
  exit 2
}
ANNOTATION_DIR="$(dirname "${annotation_manifests[0]}")"
hpc_apptainer_exec python -I "$MORPHOLOGY_WORKSPACE_SCRIPT" \
  --project "$PROJECT_FILE" \
  --shadow-root "$REFERENCE_SHADOW_ROOT" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output-dir "$MORPHOLOGY_OUTPUT" \
  --cluster-balance 0.2 \
  --minimum-cluster-representatives 2 \
  --overwrite

mapfile -t decision_values < <(
  hpc_apptainer_exec python -I - "$EXPANDED_OUTPUT/labelability_decision.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("schema_version") != "reference_cell_state_expanded_labelability_decision_v1" or value.get("status") != "COMPLETE":
    raise SystemExit("expanded labelability decision is incomplete")
print(value["computational_gate"])
print(value["overall_labelability"])
PY
)
[[ "${#decision_values[@]}" -eq 2 ]] || { echo "Could not parse expanded labelability decision" >&2; exit 2; }
computational_gate="${decision_values[0]}"
overall_labelability="${decision_values[1]}"
human_barrier="morphology_overlay_labelability_review_required"
[[ "$computational_gate" == PASS ]] || human_barrier="blind_500_cell_review_required_no_polygon"

if [[ "$computational_gate" == FAIL ]]; then
  seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
  seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
  hpc_apptainer_exec Rscript "$SEED1_SCRIPT" \
    --reference-root "$CPA_REFERENCE_ROOT" \
    --dependency-lock "$DEPENDENCY_LOCK" \
    --project "$PROJECT_FILE" \
    --expanded-labelability-decision "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/historical_projection/labelability_decision.json" \
    --output-dir "$seed1_selection"
  hpc_apptainer_exec python -I "$RENDER_SCRIPT" \
    --project "$PROJECT_FILE" \
    --review-set "$seed1_selection/seed1_review_set.tsv" \
    --review-manifest "$seed1_selection/seed1_review_manifest.json" \
    --output-dir "$seed1_render" \
    --padding 12
  for output in \
    "$seed1_selection/seed1_review_manifest.json" \
    "$seed1_selection/seed1_review_set.tsv" \
    "$seed1_render/exact_review_render_manifest.json" \
    "$seed1_render/exact_review.html" \
    "$seed1_render/crop_manifest.tsv"; do
    [[ -s "$output" && ! -L "$output" ]] || {
      echo "Expanded NO_GO blind-review evidence is unavailable: $output" >&2
      exit 2
    }
  done
  mapfile -t blind_review_stats < <(
    hpc_apptainer_exec python -I - "$seed1_selection/seed1_review_set.tsv" <<'PY'
import csv
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 500:
    raise SystemExit(f"Expanded blind review must contain exactly 500 rows; observed={len(rows)}")
if any(row.get("review_default_label") != "uncertain" for row in rows):
    raise SystemExit("Expanded blind review contains a non-uncertain default label")
if any(row.get("suggested_cell_state_label") for row in rows):
    raise SystemExit("Expanded blind review leaked a suggested cell-state label")
counts = Counter(row.get("source_id", "") for row in rows)
if "" in counts or max(counts.values()) > 8:
    raise SystemExit("Expanded blind review violates the maximum-eight-per-well contract")
print(len(rows))
print(max(counts.values()))
print(len(counts))
PY
  )
  [[ "${#blind_review_stats[@]}" -eq 3 ]] || {
    echo "Expanded blind-review statistics are incomplete" >&2
    exit 2
  }
  blind_row_count="${blind_review_stats[0]}"
  blind_max_per_well="${blind_review_stats[1]}"
  blind_well_count="${blind_review_stats[2]}"
  blind_receipt="$REFERENCE_SHADOW_ROOT/workflow_status/EXPANDED_BLIND_REVIEW_COMPLETE.tsv"
  blind_candidate="$REFERENCE_SHADOW_ROOT/workflow_status/.EXPANDED_BLIND_REVIEW_COMPLETE.tmp.$$"
  {
    printf 'property\tvalue\n'
    printf 'status\tHUMAN_REVIEW_REQUIRED\n'
    printf 'schema_version\treference_cell_state_expanded_blind_review_v1\n'
    printf 'selection_mode\tauthoritative_expanded_computational_nogo_all_unassigned\n'
    printf 'row_count\t%s\n' "$blind_row_count"
    printf 'observed_max_per_well\t%s\n' "$blind_max_per_well"
    printf 'well_count\t%s\n' "$blind_well_count"
    printf 'umap_or_cluster_labels_displayed\tfalse\n'
    printf 'current_classifier_displayed\tfalse\n'
    printf 'image_evidence\tbrightfield+nuclei_support+combined_mask_outline\n'
    printf 'seed1_review_manifest\t%s\n' "$seed1_selection/seed1_review_manifest.json"
    printf 'seed1_review_manifest_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_selection/seed1_review_manifest.json")"
    printf 'seed1_review_set_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_selection/seed1_review_set.tsv")"
    printf 'render_manifest_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_render/exact_review_render_manifest.json")"
    printf 'crop_manifest_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_render/crop_manifest.tsv")"
    printf 'human_workspace\t%s\n' "$seed1_render/exact_review.html"
    printf 'human_submission_expected_path\t%s\n' "$seed1_render/exact_review_submission.json"
  } > "$blind_candidate"
  if [[ -e "$blind_receipt" ]]; then
    cmp -s "$blind_candidate" "$blind_receipt" || {
      echo "Existing expanded blind-review receipt differs" >&2
      exit 2
    }
    rm -f "$blind_candidate"
  else
    mv "$blind_candidate" "$blind_receipt"
  fi
fi

sha256_value() { reference_cell_state_v2_sha256 "$1"; }
receipt_candidate="$REFERENCE_SHADOW_ROOT/workflow_status/.EXPANDED_CALIBRATION_COMPLETE.tmp.$$"
[[ -n "$ended_utc" ]] || ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  printf 'property\tvalue\n'
  printf 'status\tCOMPLETE\n'
  printf 'method_version\treference_ltee_historical_projection_expanded_comparison_v1\n'
  printf 'host\t%s\n' "$observed_host"
  printf 'execution_mode\tdirect_test_no_slurm_submission\n'
  printf 'gpu\t0\n'
  printf 'start_utc\t%s\n' "$started_utc"
  printf 'end_utc\t%s\n' "$ended_utc"
  printf 'reference_shadow_root\t%s\n' "$REFERENCE_SHADOW_ROOT"
  printf 'base_v2_shadow_root\t%s\n' "$BASE_V2_SHADOW_ROOT"
  printf 'parent_broad_shadow_root\t%s\n' "$PARENT_BROAD_SHADOW_ROOT"
  printf 'cell_count\t32000\n'
  printf 'profiles\tshape9,classifier12,expanded39\n'
  printf 'selected_annotation_profile\texpanded39\n'
  printf 'final_classifier_feature_profile\tclassifier12\n'
  printf 'expanded_features_allowed_in_final_classifier\tfalse\n'
  printf 'computational_labelability_gate\t%s\n' "$computational_gate"
  printf 'morphology_overlay_gate\tPENDING_HUMAN_REVIEW\n'
  printf 'overall_labelability\t%s\n' "$overall_labelability"
  printf 'human_barrier\t%s\n' "$human_barrier"
  printf 'expanded_projection_manifest\t%s\n' "$EXPANDED_OUTPUT/expanded_projection_manifest.json"
  printf 'expanded_projection_manifest_sha256\t%s\n' "$(sha256_value "$EXPANDED_OUTPUT/expanded_projection_manifest.json")"
  printf 'labelability_decision_sha256\t%s\n' "$(sha256_value "$EXPANDED_OUTPUT/labelability_decision.json")"
  printf 'project_sha256\t%s\n' "$(sha256_value "$PROJECT_FILE")"
  printf 'annotation_manifest\t%s\n' "${annotation_manifests[0]}"
  printf 'annotation_manifest_sha256\t%s\n' "$(sha256_value "${annotation_manifests[0]}")"
  printf 'morphology_overlay_manifest_sha256\t%s\n' "$(sha256_value "$MORPHOLOGY_OUTPUT/overlay_manifest.json")"
  printf 'morphology_workspace_sha256\t%s\n' "$(sha256_value "$MORPHOLOGY_OUTPUT/annotation_workspace.html")"
  printf 'sif_path\t%s\n' "$HPC_CONTAINER_IMAGE"
  printf 'sif_sha256\t%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"
  printf 'reference_snapshot_root\t%s\n' "$REFERENCE_SNAPSHOT_ROOT"
  printf 'dead_channel_read\tfalse\n'
  printf 'combined_rgb_read\tfalse\n'
  printf 'current_classification_read\tfalse\n'
  printf 'trajectory_read\tfalse\n'
  printf 'heldout_read\tfalse\n'
} > "$receipt_candidate"
if [[ -e "$RECEIPT" ]]; then
  cmp -s "$receipt_candidate" "$RECEIPT" || {
    echo "Existing expanded calibration receipt differs" >&2
    exit 2
  }
  rm -f "$receipt_candidate"
else
  mv "$receipt_candidate" "$RECEIPT"
fi

echo "expanded_calibration_complete=1"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "computational_labelability_gate=$computational_gate"
echo "morphology_workspace=$MORPHOLOGY_OUTPUT/annotation_workspace.html"
if [[ "$computational_gate" == FAIL ]]; then
  echo "blind_review_workspace=$REFERENCE_SHADOW_ROOT/human_review/seed1/render/exact_review.html"
  echo "blind_review_submission_expected_path=$REFERENCE_SHADOW_ROOT/human_review/seed1/render/exact_review_submission.json"
fi
echo "human_barrier=$human_barrier"
