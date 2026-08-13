#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${PARENT_BROAD_SHADOW_ROOT:?PARENT_BROAD_SHADOW_ROOT is required}"
: "${PARENT_PROJECT:?PARENT_PROJECT is required}"
: "${PARENT_PROJECTION_INPUT_MANIFEST:?PARENT_PROJECTION_INPUT_MANIFEST is required}"
: "${PARENT_UMAP_MANIFEST:?PARENT_UMAP_MANIFEST is required}"
: "${EXPECTED_PARENT_CELL_COUNT:?EXPECTED_PARENT_CELL_COUNT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_load_sif_identity

HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"

root_mode=formal
case "$REFERENCE_SHADOW_ROOT" in
  "$RESULTS_ROOT"/Tests_and_Parameters_calibration/*) root_mode=calibration ;;
esac
reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" "$root_mode" "$RESULTS_ROOT"
[[ "$EXPECTED_PARENT_CELL_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  reference_cell_state_v2_abort "EXPECTED_PARENT_CELL_COUNT must be positive"
  exit 2
}

BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_ROOT="$DATASET_ROOT/Nuclei"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"
for root in \
  "$PARENT_BROAD_SHADOW_ROOT" "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" \
  "$COMBINED_MASK_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$root" == /* && -d "$root" && ! -L "$root" ]] || {
    reference_cell_state_v2_abort "Required V2 Phase A root is unavailable: $root"
    exit 2
  }
done
for parent_file in \
  "$PARENT_PROJECT" "$PARENT_PROJECTION_INPUT_MANIFEST" "$PARENT_UMAP_MANIFEST"; do
  reference_cell_state_v2_require_inside "$parent_file" "$PARENT_BROAD_SHADOW_ROOT" parent_input
done

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the V2 Phase A allowlist"; exit 2 ;;
esac
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE
export HPC_CONTAINER_NO_MOUNT HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES
export CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT

HPC_CONTAINER_RUNTIME_ROOT="$SCRIPT_DIR"
export HPC_CONTAINER_RUNTIME_ROOT
source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
reference_cell_state_v2_verify_container_rootfs_read_only
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}"
HPC_PROJECT_ROOT="$PROJECT_DIR"
export PROJECT_DIR HPC_PROJECT_ROOT
DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
FEATURE_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_features_v2.json"
CLASSES_FILE="$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_classes_v2.tsv"
PLATE_MAP="$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv"
PROJECT_BUILDER="$PROJECT_DIR/cellpose_pipeline/scripts/26_prepare_reference_cell_state_project.py"
HISTORICAL_ADAPTER="$PROJECT_DIR/cellpose_pipeline/scripts/28_build_reference_cell_state_historical_projection.R"
CPA_STAGE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py"
MORPHOLOGY_WORKSPACE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/27_build_reference_morphology_workspace.py"
SEED1_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/33_build_reference_cell_state_seed1_review.R"
RENDER_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/37_render_reference_cell_state_exact_review.py"
REFERENCE_SNAPSHOT_ROOT="$CPA_REFERENCE_ROOT/reference/ltee-source"
PROJECT_FILE="$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/project.yml"
PHASE_ROOT="$REFERENCE_SHADOW_ROOT/workflow_status/reference_cell_state_phase_a_v2"
PHASE_RECEIPT="$PHASE_ROOT/PHASE_A_COMPLETE.tsv"
RUNTIME_RECEIPT="$REFERENCE_SHADOW_ROOT/workflow_status/reference_v2_runtime_packages.tsv"

for required in \
  "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$PLATE_MAP" "$PROJECT_BUILDER" \
  "$HISTORICAL_ADAPTER" "$CPA_STAGE_SCRIPT" "$MORPHOLOGY_WORKSPACE_SCRIPT" \
  "$SEED1_SCRIPT" "$RENDER_SCRIPT" \
  "$REFERENCE_SNAPSHOT_ROOT/code/lib/Utils.R"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    reference_cell_state_v2_abort "Required V2 Phase A file is unavailable: $required"
    exit 2
  }
done
if [[ -e "$PHASE_RECEIPT" || -L "$PHASE_RECEIPT" ]]; then
  reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT" >/dev/null
  echo "reference_cell_state_phase_a_v2_verified_reuse=1"
  exit 0
fi

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$PHASE_ROOT"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Fail before touching the project if the new SIF lacks any source-exact
# historical projection dependency or if the selected reference function
# hashes have drifted.
hpc_apptainer_exec Rscript "$HISTORICAL_ADAPTER" \
  --check-config \
  --reference-snapshot-root "$REFERENCE_SNAPSHOT_ROOT" \
  --feature-config "$FEATURE_CONFIG"
reference_cell_state_v2_test_maybe_fail phase_a_after_historical_preflight
runtime_receipt_candidate="$PHASE_ROOT/.reference_v2_runtime_packages.tmp.$$"
hpc_apptainer_exec Rscript - "$runtime_receipt_candidate" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
required <- c(
  "jsonlite", "digest", "magrittr", "uwot", "dbscan", "cluster", "glmnet",
  "dplyr", "tidyr", "purrr", "stringr", "ggplot2"
)
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing V2 R packages: ", paste(missing, collapse = ", "))
rows <- data.frame(package = c("R", required), version = c(
  paste(R.version$major, R.version$minor, sep = "."),
  vapply(required, function(x) as.character(utils::packageVersion(x)), character(1))
))
stopifnot(rows$version[rows$package == "R"] == "4.2.3")
stopifnot(rows$version[rows$package == "dbscan"] == "1.2.3")
stopifnot(rows$version[rows$package == "dplyr"] == "1.1.4")
stopifnot(rows$version[rows$package == "tidyr"] == "1.3.1")
stopifnot(rows$version[rows$package == "purrr"] == "1.2.0")
stopifnot(rows$version[rows$package == "stringr"] == "1.6.0")
stopifnot(rows$version[rows$package == "ggplot2"] == "4.0.0")
utils::write.table(rows, args[[1]], sep = "\t", quote = FALSE, row.names = FALSE)
RS
if [[ -e "$RUNTIME_RECEIPT" ]]; then
  [[ -f "$RUNTIME_RECEIPT" && ! -L "$RUNTIME_RECEIPT" ]] && \
     cmp -s "$runtime_receipt_candidate" "$RUNTIME_RECEIPT" || {
    reference_cell_state_v2_abort "V2 runtime package receipt differs on retry"
    exit 2
  }
  rm -f "$runtime_receipt_candidate"
else
  mv "$runtime_receipt_candidate" "$RUNTIME_RECEIPT"
fi

# Verify that /share was hidden and only the allowlisted input roots were
# restored. Dead, Combined RGB, and every current classification_* root must
# remain invisible before model freeze.
hpc_apptainer_exec python -I - \
  "$DATASET_ROOT" "$REFERENCE_SHADOW_ROOT" "$PARENT_BROAD_SHADOW_ROOT" <<'PY'
from pathlib import Path
import sys
dataset, reference, parent = map(Path, sys.argv[1:])
for required in (reference, parent, dataset / "Brightfield", dataset / "Nuclei"):
    if not required.exists():
        raise SystemExit(f"allowlisted V2 input is invisible: {required}")
for forbidden in (dataset / "Dead", dataset / "Combined"):
    if forbidden.exists():
        raise SystemExit(f"forbidden V2 channel remained visible: {forbidden}")
results = dataset.parent / "results"
if results.exists():
    leaked = [p for p in results.iterdir() if p.name.startswith("classification_")]
    if leaked:
        raise SystemExit(f"current classification root remained visible: {leaked[:3]}")
print("reference_v2_container_blinding=PASS")
PY

hpc_apptainer_exec python -I "$PROJECT_BUILDER" \
  --parent-project "$PARENT_PROJECT" \
  --parent-shadow-root "$PARENT_BROAD_SHADOW_ROOT" \
  --parent-projection-manifest "$PARENT_PROJECTION_INPUT_MANIFEST" \
  --parent-umap-manifest "$PARENT_UMAP_MANIFEST" \
  --reference-shadow-root "$REFERENCE_SHADOW_ROOT" \
  --method-version v2 \
  --classes-file "$CLASSES_FILE" \
  --feature-config "$FEATURE_CONFIG" \
  --plate-map "$PLATE_MAP" \
  --reference-snapshot-root "$REFERENCE_SNAPSHOT_ROOT" \
  --historical-projection-script "$HISTORICAL_ADAPTER" \
  --rscript Rscript \
  --dbscan-cores "$OMP_NUM_THREADS" \
  --expected-cell-count "$EXPECTED_PARENT_CELL_COUNT" \
  --seed 42
reference_cell_state_v2_test_maybe_fail phase_a_after_project_generation

for stage in validate umap annotate; do
  args=(
    "$CPA_STAGE_SCRIPT" --stage "$stage" --project "$PROJECT_FILE"
    --shadow-root "$REFERENCE_SHADOW_ROOT" --reference-root "$CPA_REFERENCE_ROOT"
    --dependency-lock "$DEPENDENCY_LOCK" --rscript Rscript
  )
  [[ "$stage" != validate ]] || args+=(--validate-stage umap)
  hpc_apptainer_exec python -I "${args[@]}"
done
reference_cell_state_v2_test_maybe_fail phase_a_after_cpa

mapfile -t annotation_manifests < <(
  find "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/runs" \
    -type f -name annotation_manifest.json -print | LC_ALL=C sort
)
[[ "${#annotation_manifests[@]}" -eq 1 ]] || {
  reference_cell_state_v2_abort "Expected exactly one V2 annotation generation; observed=${#annotation_manifests[@]}"
  exit 2
}
ANNOTATION_DIR="$(dirname "${annotation_manifests[0]}")"
hpc_apptainer_exec python -I "$MORPHOLOGY_WORKSPACE_SCRIPT" \
  --project "$PROJECT_FILE" \
  --shadow-root "$REFERENCE_SHADOW_ROOT" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output-dir "$REFERENCE_SHADOW_ROOT/morphology_reference" \
  --cluster-balance 0.2 \
  --minimum-cluster-representatives 2 \
  --overwrite
reference_cell_state_v2_test_maybe_fail phase_a_after_morphology

for output in \
  "$PROJECT_FILE" \
  "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/parent_import_manifest.json" \
  "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/historical_projection/historical_projection_manifest.json" \
  "$REFERENCE_SHADOW_ROOT/morphology_reference/annotation_workspace.html" \
  "$REFERENCE_SHADOW_ROOT/morphology_reference/overlay_manifest.json"; do
  [[ -s "$output" && ! -L "$output" ]] || {
    reference_cell_state_v2_abort "Required V2 Phase A output is unavailable: $output"
    exit 2
  }
done

plate_map_sha256="$(reference_cell_state_v2_sha256 "$PLATE_MAP")"
project_sha256="$(reference_cell_state_v2_sha256 "$PROJECT_FILE")"
historical_manifest="$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/historical_projection/historical_projection_manifest.json"
historical_sha256="$(reference_cell_state_v2_sha256 "$historical_manifest")"
annotation_sha256="$(reference_cell_state_v2_sha256 "${annotation_manifests[0]}")"
annotation_manifest="${annotation_manifests[0]}"
dependency_lock_sha256="$(reference_cell_state_v2_sha256 "$DEPENDENCY_LOCK")"
runtime_receipt_sha256="$(reference_cell_state_v2_sha256 "$RUNTIME_RECEIPT")"
morphology_overlay_manifest="$REFERENCE_SHADOW_ROOT/morphology_reference/overlay_manifest.json"
morphology_workspace="$REFERENCE_SHADOW_ROOT/morphology_reference/annotation_workspace.html"
morphology_overlay_sha256="$(reference_cell_state_v2_sha256 "$morphology_overlay_manifest")"
morphology_workspace_sha256="$(reference_cell_state_v2_sha256 "$morphology_workspace")"
one_cluster_fallback="$(hpc_apptainer_exec python -I - "$historical_manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("schema_version") != "reference_cell_state_historical_projection_v2" or payload.get("status") != "COMPLETE":
    raise SystemExit("historical projection manifest is not a complete V2 generation")
value = (payload.get("diagnostic_cluster") or {}).get("one_cluster_fallback")
if not isinstance(value, bool):
    raise SystemExit("historical projection manifest lacks boolean one_cluster_fallback")
print("true" if value else "false")
PY
)"
if [[ "$one_cluster_fallback" == true ]]; then
  method_parity_status=historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation
  review_sampling_contract=disclosed_seed1_all_unassigned_well_balanced_max500_and_seed2_probability_surrogate_not_pinned_reference_exact_sampling
  seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
  seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
  hpc_apptainer_exec Rscript "$SEED1_SCRIPT" \
    --reference-root "$CPA_REFERENCE_ROOT" \
    --dependency-lock "$DEPENDENCY_LOCK" \
    --project "$PROJECT_FILE" \
    --diagnostic-cluster-manifest "$historical_manifest" \
    --output-dir "$seed1_selection"
  reference_cell_state_v2_test_maybe_fail phase_a_after_seed1_selection
  hpc_apptainer_exec python -I "$RENDER_SCRIPT" \
    --project "$PROJECT_FILE" \
    --review-set "$seed1_selection/seed1_review_set.tsv" \
    --review-manifest "$seed1_selection/seed1_review_manifest.json" \
    --output-dir "$seed1_render" \
    --padding 12
  reference_cell_state_v2_test_maybe_fail phase_a_after_seed1_render
  [[ -s "$seed1_render/exact_review.html" && \
     -s "$seed1_render/exact_review_render_manifest.json" ]] || {
    reference_cell_state_v2_abort "One-cluster fallback exact Seed1 workspace is incomplete"
    exit 2
  }
  human_barrier=seed1_adapted_exact_review_submission_required
  barrier_workspace="$seed1_render/exact_review.html"
  barrier_submission="$seed1_render/exact_review_submission.json"
  seed1_selection_manifest="$seed1_selection/seed1_review_manifest.json"
  seed1_review_set="$seed1_selection/seed1_review_set.tsv"
  seed1_render_manifest="$seed1_render/exact_review_render_manifest.json"
  seed1_crop_manifest="$seed1_render/crop_render_status.tsv"
  for fallback_evidence in "$seed1_selection_manifest" "$seed1_review_set" \
    "$seed1_render_manifest" "$seed1_render/exact_review.html" "$seed1_crop_manifest"; do
    [[ -s "$fallback_evidence" && ! -L "$fallback_evidence" ]] || {
      reference_cell_state_v2_abort "One-cluster fallback evidence is unavailable: $fallback_evidence"
      exit 2
    }
  done
else
  method_parity_status=historical_core_parity_with_pinned_stable_polygon_sampling
  review_sampling_contract=pinned_reference_stable_polygon_seed1_seed2_sampling
  human_barrier=region_submission_required
  barrier_workspace="$REFERENCE_SHADOW_ROOT/morphology_reference/annotation_workspace.html"
  barrier_submission="$ANNOTATION_DIR/region_submission.json"
fi
finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
tmp="$PHASE_ROOT/.PHASE_A_COMPLETE.tsv.tmp.$$"
{
  printf 'property\tvalue\n'
  printf 'schema_version\treference_cell_state_phase_a_receipt_v2\n'
  printf 'status\tCOMPLETE\n'
  printf 'started_utc\t%s\n' "$started_utc"
  printf 'finished_utc\t%s\n' "$finished_utc"
  printf 'reference_shadow_root\t%s\n' "$REFERENCE_SHADOW_ROOT"
  printf 'project_file\t%s\n' "$PROJECT_FILE"
  printf 'project_sha256\t%s\n' "$project_sha256"
  printf 'plate_map\t%s\n' "$PLATE_MAP"
  printf 'plate_map_sha256\t%s\n' "$plate_map_sha256"
  printf 'identity_mapping\tcontext=SUM-159-NLS-{2N,4N};family=SUM-159-NLS;source_id=well;suffix=site+elapsed;condition=plate_condition\n'
  printf 'pixel_calibration_status\tunavailable_not_inferred\n'
  printf 'shape_unit_adapter\tpixel_units_to_historical_names_only\n'
  printf 'historical_projection_manifest\t%s\n' "$historical_manifest"
  printf 'historical_projection_manifest_sha256\t%s\n' "$historical_sha256"
  printf 'runtime_package_receipt\t%s\n' "$RUNTIME_RECEIPT"
  printf 'runtime_package_receipt_sha256\t%s\n' "$runtime_receipt_sha256"
  printf 'morphology_overlay_manifest\t%s\n' "$morphology_overlay_manifest"
  printf 'morphology_overlay_manifest_sha256\t%s\n' "$morphology_overlay_sha256"
  printf 'morphology_workspace\t%s\n' "$morphology_workspace"
  printf 'morphology_workspace_sha256\t%s\n' "$morphology_workspace_sha256"
  printf 'annotation_dir\t%s\n' "$ANNOTATION_DIR"
  printf 'annotation_manifest\t%s\n' "$annotation_manifest"
  printf 'annotation_manifest_sha256\t%s\n' "$annotation_sha256"
  printf 'hpc_container_image\t%s\n' "$HPC_CONTAINER_IMAGE"
  printf 'hpc_container_sha256\t%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"
  printf 'hpc_container_bytes\t%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES"
  printf 'filesystem_immutability_mode\t%s\n' "$REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE"
  printf 'runtime_rootfs_read_only\t%s\n' "$REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY"
  printf 'cpa_reference_root\t%s\n' "$CPA_REFERENCE_ROOT"
  printf 'cpa_reference_commit\t%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT"
  printf 'cpa_reference_tree\t%s\n' "$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE"
  printf 'cpa_reference_status\tclean\n'
  printf 'cpa_reference_mode\tpinned_read_only_source\n'
  printf 'dependency_lock\t%s\n' "$DEPENDENCY_LOCK"
  printf 'dependency_lock_sha256\t%s\n' "$dependency_lock_sha256"
  printf 'diagnostic_cluster_role\tmetadata_only_not_classifier_target\n'
  printf 'single_cluster_fallback_allowed\ttrue\n'
  printf 'one_cluster_fallback\t%s\n' "$one_cluster_fallback"
  printf 'method_parity_status\t%s\n' "$method_parity_status"
  printf 'review_sampling_contract\t%s\n' "$review_sampling_contract"
  printf 'human_barrier\t%s\n' "$human_barrier"
  printf 'human_workspace\t%s\n' "$barrier_workspace"
  printf 'human_submission_expected_path\t%s\n' "$barrier_submission"
  if [[ "$one_cluster_fallback" == true ]]; then
    printf 'seed1_selection_manifest\t%s\n' "$seed1_selection_manifest"
    printf 'seed1_selection_manifest_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_selection_manifest")"
    printf 'seed1_review_set\t%s\n' "$seed1_review_set"
    printf 'seed1_review_set_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_review_set")"
    printf 'seed1_render_manifest\t%s\n' "$seed1_render_manifest"
    printf 'seed1_render_manifest_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_render_manifest")"
    printf 'seed1_exact_review_html\t%s\n' "$seed1_render/exact_review.html"
    printf 'seed1_exact_review_html_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_render/exact_review.html")"
    printf 'seed1_crop_status\t%s\n' "$seed1_crop_manifest"
    printf 'seed1_crop_status_sha256\t%s\n' "$(reference_cell_state_v2_sha256 "$seed1_crop_manifest")"
  fi
  printf 'dead_channel_read_performed\tfalse\n'
  printf 'combined_rgb_read_performed\tfalse\n'
  printf 'current_classifier_read_performed\tfalse\n'
  printf 'legacy_no_go_enforcement\tnot_read\n'
  printf 'site_share_mount_disabled\ttrue\n'
} > "$tmp"
mv "$tmp" "$PHASE_RECEIPT"
reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT" >/dev/null

echo "reference_cell_state_phase_a_v2_complete=1"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "human_workspace=$barrier_workspace"
echo "human_submission_expected_path=$barrier_submission"
echo "one_cluster_fallback=$one_cluster_fallback"
echo "method_parity_status=$method_parity_status"
echo "human_barrier=$human_barrier"
