#!/usr/bin/env bash
set -euo pipefail

: "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${SOURCE_SEGMENTATION_ROOT:?SOURCE_SEGMENTATION_ROOT is required}"
: "${V2_ACTION:?V2_ACTION is required: post-region, post-fallback-seed1, post-seed1, post-seed2, or post-adjudication}"
case "$V2_ACTION" in post-region|post-fallback-seed1|post-seed1|post-seed2|post-adjudication) ;; *) echo "Unsupported V2_ACTION: $V2_ACTION" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_load_sif_identity
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"
mode=formal
case "$REFERENCE_SHADOW_ROOT" in "$RESULTS_ROOT"/Tests_and_Parameters_calibration/*) mode=calibration ;; esac
reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" "$mode" "$RESULTS_ROOT"
PHASE_A_AUDIT="$(reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT")"
METHOD_PARITY_STATUS="$(awk -F '\t' '$1=="method_parity_status"{print $2; n++} END{if(n!=1) exit 2}' <<< "$PHASE_A_AUDIT")"
PHASE_A_ONE_CLUSTER_FALLBACK="$(awk -F '\t' '$1=="one_cluster_fallback"{print $2; n++} END{if(n!=1) exit 2}' <<< "$PHASE_A_AUDIT")"
reference_cell_state_v2_require_posthuman_action_branch \
  "$V2_ACTION" "$PHASE_A_ONE_CLUSTER_FALLBACK"

PROJECT_FILE="$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/project.yml"
PARENT_IMPORT_MANIFEST="$(dirname "$PROJECT_FILE")/parent_import_manifest.json"
BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_ROOT="$DATASET_ROOT/Nuclei"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"
for input in "$PROJECT_FILE" "$PARENT_IMPORT_MANIFEST"; do
  reference_cell_state_v2_require_inside "$input" "$REFERENCE_SHADOW_ROOT" posthuman_input
done
for root in "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$root" == /* && -d "$root" && ! -L "$root" ]] || {
    reference_cell_state_v2_abort "Required V2 review-support root is unavailable: $root"
    exit 2
  }
done

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT:rw,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) reference_cell_state_v2_abort "HPC_CONTAINER_BINDS differs from the V2 exact-review allowlist"; exit 2 ;;
esac
export HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT
export HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES
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
CPA_STAGE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/20_run_cellphenotypeannotator_stage.py"
SEED1_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/33_build_reference_cell_state_seed1_review.R"
TRAIN_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/34_train_reference_cell_state_historical.R"
SEED2_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/35_build_reference_cell_state_seed2_review.R"
MERGE_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/36_merge_reference_cell_state_reviews.R"
RENDER_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/37_render_reference_cell_state_exact_review.py"
IMPORT_SCRIPT="$PROJECT_DIR/cellpose_pipeline/scripts/38_import_reference_cell_state_exact_review.R"
for required in "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CPA_STAGE_SCRIPT" "$SEED1_SCRIPT" \
  "$TRAIN_SCRIPT" "$SEED2_SCRIPT" "$MERGE_SCRIPT" "$RENDER_SCRIPT" "$IMPORT_SCRIPT"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    reference_cell_state_v2_abort "Required V2 post-human implementation is unavailable: $required"
    exit 2
  }
done

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"

assert_blinded() {
  hpc_apptainer_exec python -I - \
    "$DATASET_ROOT" "$REFERENCE_SHADOW_ROOT" "$BRIGHTFIELD_ROOT" "$NUCLEI_ROOT" "$COMBINED_MASK_ROOT" <<'PY'
from pathlib import Path
import sys
dataset, reference, bf, nuclei, masks = map(Path, sys.argv[1:])
for required in (reference, bf, nuclei, masks):
    if not required.exists(): raise SystemExit(f"required exact-review root invisible: {required}")
for forbidden in (dataset / "Dead", dataset / "Combined"):
    if forbidden.exists(): raise SystemExit(f"forbidden review channel visible: {forbidden}")
results = dataset.parent / "results"
if results.exists():
    leaked = [p for p in results.iterdir() if p.name.startswith("classification_")]
    if leaked: raise SystemExit(f"current classification root visible: {leaked[:3]}")
print("reference_v2_posthuman_blinding=PASS")
PY
}

locate_single() {
  local pattern="$1" label="$2"
  local -a paths=()
  while IFS= read -r path; do paths+=("$path"); done < <(
    find "$REFERENCE_SHADOW_ROOT" -type f -name "$pattern" -print | LC_ALL=C sort
  )
  [[ "${#paths[@]}" -eq 1 ]] || {
    reference_cell_state_v2_abort "Expected exactly one $label in V2 root; observed=${#paths[@]}"
    return 2
  }
  dirname "${paths[0]}"
}

seed1_annotation_import_dir() {
  local manifest="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection/seed1_review_manifest.json"
  reference_cell_state_v2_require_inside "$manifest" "$REFERENCE_SHADOW_ROOT" seed1_review_manifest
  hpc_apptainer_exec python -I - "$manifest" "$REFERENCE_SHADOW_ROOT" <<'PY'
import json, os, pathlib, sys
manifest, root = map(pathlib.Path, sys.argv[1:])
with manifest.open(encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("schema_version") != "reference_cell_state_seed1_review_v2":
    raise SystemExit("Seed1 review manifest schema mismatch")
value = payload.get("annotation_import_dir")
if not isinstance(value, str) or not value:
    raise SystemExit("Seed1 polygon review does not freeze annotation_import_dir")
resolved = pathlib.Path(os.path.realpath(value))
root_resolved = pathlib.Path(os.path.realpath(root))
if root_resolved != resolved and root_resolved not in resolved.parents:
    raise SystemExit("Seed1 annotation_import_dir escaped the V2 root")
if not (resolved / "annotation_import_manifest.json").is_file():
    raise SystemExit("Seed1 frozen annotation import is unavailable")
print(resolved)
PY
}

render_exact_review() {
  local seed="$1" selection_root="$2" render_root="$3"
  hpc_apptainer_exec python -I "$RENDER_SCRIPT" \
    --project "$PROJECT_FILE" \
    --review-set "$selection_root/seed${seed}_review_set.tsv" \
    --review-manifest "$selection_root/seed${seed}_review_manifest.json" \
    --output-dir "$render_root" \
    --padding 12
  for output in "$render_root/exact_review.html" "$render_root/exact_review_render_manifest.json"; do
    [[ -s "$output" && ! -L "$output" ]] || {
      reference_cell_state_v2_abort "Seed${seed} exact review renderer omitted: $output"
      return 2
    }
  done
}

finish_posthuman_stage() {
  local action="$1"; shift
  local receipt_root="$REFERENCE_SHADOW_ROOT/workflow_status/reference_cell_state_posthuman_v2"
  local receipt="$receipt_root/${action}_COMPLETE.tsv"
  local temporary="$receipt_root/.${action}_COMPLETE.tsv.tmp.$$"
  mkdir -p "$receipt_root"
  {
    printf 'property\tvalue\n'
    printf 'schema_version\treference_cell_state_posthuman_stage_receipt_v2\n'
    printf 'status\tCOMPLETE\n'
    printf 'action\t%s\n' "$action"
    printf 'method_parity_status\t%s\n' "$METHOD_PARITY_STATUS"
    local path index=0
    for path in "$@"; do
      [[ -s "$path" && ! -L "$path" ]] || {
        reference_cell_state_v2_abort "Post-human stage evidence is unavailable: $path"
        return 2
      }
      index=$((index + 1))
      printf 'evidence_%03d_path\t%s\n' "$index" "$path"
      printf 'evidence_%03d_sha256\t%s\n' "$index" "$(reference_cell_state_v2_sha256 "$path")"
    done
  } > "$temporary"
  if [[ -e "$receipt" ]]; then
    [[ -f "$receipt" && ! -L "$receipt" ]] && cmp -s "$temporary" "$receipt" || {
      reference_cell_state_v2_abort "Post-human stage receipt differs on verified retry"
      return 2
    }
    rm -f "$temporary"
  else
    mv "$temporary" "$receipt"
  fi
}

import_exact_review() {
  local seed="$1" selection_root="$2" render_root="$3" import_root="$4" submission="$5"
  reference_cell_state_v2_require_inside "$submission" "$REFERENCE_SHADOW_ROOT" "seed${seed}_submission"
  # Script 38 is both the atomic importer and the authoritative verified-reuse
  # gate.  Calling it again before merge re-hashes the exact selection,
  # renderer/crops, human submission, and immutable import outputs.
  hpc_apptainer_exec Rscript "$IMPORT_SCRIPT" \
    --reference-root "$CPA_REFERENCE_ROOT" \
    --dependency-lock "$DEPENDENCY_LOCK" \
    --review-set "$selection_root/seed${seed}_review_set.tsv" \
    --render-dir "$render_root" \
    --submission "$submission" \
    --output-dir "$import_root"
  [[ -s "$import_root/reviewed_labels.tsv" && -s "$import_root/review_import_manifest.json" ]] || {
    reference_cell_state_v2_abort "Seed${seed} exact review import is incomplete"
    return 2
  }
  reference_cell_state_v2_test_maybe_fail "posthuman_after_seed${seed}_import"
}

assert_blinded
case "$V2_ACTION" in
  post-region)
    : "${REGION_SUBMISSION:?REGION_SUBMISSION is required after the polygon human barrier}"
    reference_cell_state_v2_require_inside "$REGION_SUBMISSION" "$REFERENCE_SHADOW_ROOT" REGION_SUBMISSION
    hpc_apptainer_exec python -I "$CPA_STAGE_SCRIPT" \
      --stage annotation-import --project "$PROJECT_FILE" \
      --shadow-root "$REFERENCE_SHADOW_ROOT" --reference-root "$CPA_REFERENCE_ROOT" \
      --dependency-lock "$DEPENDENCY_LOCK" --rscript Rscript \
      --submission "$REGION_SUBMISSION"
    reference_cell_state_v2_test_maybe_fail posthuman_after_annotation_import
    ANNOTATION_IMPORT_DIR="$(locate_single annotation_import_manifest.json annotation-import-generation)"
    seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
    seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
    hpc_apptainer_exec Rscript "$SEED1_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --project "$PROJECT_FILE" --annotation-import-dir "$ANNOTATION_IMPORT_DIR" \
      --output-dir "$seed1_selection"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed1_selection
    render_exact_review 1 "$seed1_selection" "$seed1_render"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed1_render
    finish_posthuman_stage "$V2_ACTION" \
      "$seed1_selection/seed1_review_manifest.json" \
      "$seed1_render/exact_review_render_manifest.json" \
      "$seed1_render/exact_review.html"
    echo "reference_cell_state_v2_post_region_complete=1"
    echo "seed1_workspace=$seed1_render/exact_review.html"
    echo "seed1_submission_expected_path=$seed1_render/exact_review_submission.json"
    echo "human_barrier=seed1_exact_review_submission_required_pinned_stable_polygon_sampling"
    echo "method_parity_status=$METHOD_PARITY_STATUS"
    ;;
  post-seed1)
    seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
    seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
    SEED1_SUBMISSION="${SEED1_SUBMISSION:-$seed1_render/exact_review_submission.json}"
    seed1_import="$REFERENCE_SHADOW_ROOT/human_review/seed1/import"
    initial_model="$REFERENCE_SHADOW_ROOT/models/initial"
    initial_receipt="$REFERENCE_SHADOW_ROOT/workflow_status/training_v2/initial_train_receipt.json"
    seed2_selection="$REFERENCE_SHADOW_ROOT/human_review/seed2/selection"
    seed2_render="$REFERENCE_SHADOW_ROOT/human_review/seed2/render"
    ANNOTATION_IMPORT_DIR="$(seed1_annotation_import_dir)"
    import_exact_review 1 "$seed1_selection" "$seed1_render" "$seed1_import" "$SEED1_SUBMISSION"
    mkdir -p "$(dirname "$initial_receipt")"
    hpc_apptainer_exec Rscript "$TRAIN_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --feature-config "$FEATURE_CONFIG" --project "$PROJECT_FILE" \
      --reviewed-labels "$seed1_import/reviewed_labels.tsv" \
      --training-stage initial --output-dir "$initial_model" --output-receipt "$initial_receipt"
    reference_cell_state_v2_test_maybe_fail posthuman_after_initial_train
    hpc_apptainer_exec Rscript "$SEED2_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --project "$PROJECT_FILE" --annotation-import-dir "$ANNOTATION_IMPORT_DIR" \
      --initial-model-dir "$initial_model" \
      --seed1-reviewed-labels "$seed1_import/reviewed_labels.tsv" \
      --output-dir "$seed2_selection"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed2_selection
    render_exact_review 2 "$seed2_selection" "$seed2_render"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed2_render
    finish_posthuman_stage "$V2_ACTION" \
      "$seed1_import/review_import_manifest.json" "$initial_receipt" \
      "$seed2_selection/seed2_review_manifest.json" \
      "$seed2_render/exact_review_render_manifest.json"
    echo "reference_cell_state_v2_post_seed1_complete=1"
    echo "seed2_workspace=$seed2_render/exact_review.html"
    echo "seed2_submission_expected_path=$seed2_render/exact_review_submission.json"
    echo "human_barrier=seed2_exact_review_submission_required"
    echo "method_parity_status=$METHOD_PARITY_STATUS"
    ;;
  post-fallback-seed1)
    seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
    seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
    SEED1_SUBMISSION="${SEED1_SUBMISSION:-$seed1_render/exact_review_submission.json}"
    seed1_import="$REFERENCE_SHADOW_ROOT/human_review/seed1/import"
    initial_model="$REFERENCE_SHADOW_ROOT/models/initial"
    initial_receipt="$REFERENCE_SHADOW_ROOT/workflow_status/training_v2/initial_train_receipt.json"
    seed2_selection="$REFERENCE_SHADOW_ROOT/human_review/seed2/selection"
    seed2_render="$REFERENCE_SHADOW_ROOT/human_review/seed2/render"
    import_exact_review 1 "$seed1_selection" "$seed1_render" "$seed1_import" "$SEED1_SUBMISSION"
    mkdir -p "$(dirname "$initial_receipt")"
    hpc_apptainer_exec Rscript "$TRAIN_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --feature-config "$FEATURE_CONFIG" --project "$PROJECT_FILE" \
      --reviewed-labels "$seed1_import/reviewed_labels.tsv" \
      --training-stage initial --output-dir "$initial_model" --output-receipt "$initial_receipt"
    reference_cell_state_v2_test_maybe_fail posthuman_after_initial_train
    # Seed2's four targeted buckets use the all-unassigned Seed1 labels and
    # initial probabilities directly; polygon annotation-import is absent by
    # construction in the authoritative one-cluster fallback branch.
    hpc_apptainer_exec Rscript "$SEED2_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --project "$PROJECT_FILE" --diagnostic-cluster-manifest \
      "$REFERENCE_SHADOW_ROOT/projection_input/representative_umap_v2/historical_projection/historical_projection_manifest.json" \
      --initial-model-dir "$initial_model" \
      --seed1-reviewed-labels "$seed1_import/reviewed_labels.tsv" \
      --output-dir "$seed2_selection"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed2_selection
    render_exact_review 2 "$seed2_selection" "$seed2_render"
    reference_cell_state_v2_test_maybe_fail posthuman_after_seed2_render
    finish_posthuman_stage "$V2_ACTION" \
      "$seed1_import/review_import_manifest.json" "$initial_receipt" \
      "$seed2_selection/seed2_review_manifest.json" \
      "$seed2_render/exact_review_render_manifest.json"
    echo "reference_cell_state_v2_post_fallback_seed1_complete=1"
    echo "seed2_workspace=$seed2_render/exact_review.html"
    echo "seed2_submission_expected_path=$seed2_render/exact_review_submission.json"
    echo "human_barrier=seed2_exact_review_submission_required"
    echo "method_parity_status=$METHOD_PARITY_STATUS"
    ;;
  post-seed2|post-adjudication)
    seed1_selection="$REFERENCE_SHADOW_ROOT/human_review/seed1/selection"
    seed1_render="$REFERENCE_SHADOW_ROOT/human_review/seed1/render"
    SEED1_SUBMISSION="${SEED1_SUBMISSION:-$seed1_render/exact_review_submission.json}"
    seed1_import="$REFERENCE_SHADOW_ROOT/human_review/seed1/import"
    seed2_selection="$REFERENCE_SHADOW_ROOT/human_review/seed2/selection"
    seed2_render="$REFERENCE_SHADOW_ROOT/human_review/seed2/render"
    SEED2_SUBMISSION="${SEED2_SUBMISSION:-$seed2_render/exact_review_submission.json}"
    seed2_import="$REFERENCE_SHADOW_ROOT/human_review/seed2/import"
    merged_root="$REFERENCE_SHADOW_ROOT/human_review/merged"
    adjudication_barrier="$REFERENCE_SHADOW_ROOT/human_review/merged_adjudication_barrier"
    final_model="$REFERENCE_SHADOW_ROOT/models/final"
    final_receipt="$REFERENCE_SHADOW_ROOT/workflow_status/training_v2/final_train_receipt.json"
    import_exact_review 1 "$seed1_selection" "$seed1_render" "$seed1_import" "$SEED1_SUBMISSION"
    import_exact_review 2 "$seed2_selection" "$seed2_render" "$seed2_import" "$SEED2_SUBMISSION"
    merge_args=(
      "$MERGE_SCRIPT" --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK"
      --seed1-reviewed-labels "$seed1_import/reviewed_labels.tsv"
      --seed2-reviewed-labels "$seed2_import/reviewed_labels.tsv" --output-dir "$merged_root"
    )
    if [[ "$V2_ACTION" == post-adjudication ]]; then
      : "${REVIEW_ADJUDICATION:?REVIEW_ADJUDICATION is required for post-adjudication}"
      reference_cell_state_v2_require_inside "$REVIEW_ADJUDICATION" "$REFERENCE_SHADOW_ROOT" REVIEW_ADJUDICATION
      [[ -s "$adjudication_barrier/review_conflicts.tsv" && \
         -s "$adjudication_barrier/review_adjudication_barrier_manifest.json" ]] || {
        reference_cell_state_v2_abort "Adjudication requires the preserved prior HUMAN_ADJUDICATION_REQUIRED audit"
        exit 2
      }
      merge_args+=(--adjudication "$REVIEW_ADJUDICATION")
    elif [[ -n "${REVIEW_ADJUDICATION:-}" ]]; then
      reference_cell_state_v2_abort "REVIEW_ADJUDICATION is accepted only by post-adjudication"
      exit 2
    elif [[ -e "$adjudication_barrier" ]]; then
      reference_cell_state_v2_abort \
        "Existing immutable merge barrier requires a distinct post-adjudication stage"
      exit 2
    fi
    hpc_apptainer_exec Rscript "${merge_args[@]}"
    [[ -s "$merged_root/merged_reviewed_labels.tsv" ]] || {
      reference_cell_state_v2_abort "Review merge requires human adjudication before final training"
      exit 2
    }
    reference_cell_state_v2_test_maybe_fail posthuman_after_review_merge
    mkdir -p "$(dirname "$final_receipt")"
    hpc_apptainer_exec Rscript "$TRAIN_SCRIPT" \
      --reference-root "$CPA_REFERENCE_ROOT" --dependency-lock "$DEPENDENCY_LOCK" \
      --feature-config "$FEATURE_CONFIG" --project "$PROJECT_FILE" \
      --reviewed-labels "$merged_root/merged_reviewed_labels.tsv" \
      --training-stage final --output-dir "$final_model" --output-receipt "$final_receipt"
    reference_cell_state_v2_test_maybe_fail posthuman_after_final_train
    finish_posthuman_stage "$V2_ACTION" \
      "$seed1_import/review_import_manifest.json" \
      "$seed2_import/review_import_manifest.json" \
      "$merged_root/review_merge_manifest.json" "$final_receipt"
    echo "reference_cell_state_v2_post_seed2_complete=1"
    echo "cpa_model_dir=$final_model"
    echo "cpa_train_receipt=$final_receipt"
    echo "human_barrier=none_model_acceptance_ready"
    echo "method_parity_status=$METHOD_PARITY_STATUS"
    ;;
esac
