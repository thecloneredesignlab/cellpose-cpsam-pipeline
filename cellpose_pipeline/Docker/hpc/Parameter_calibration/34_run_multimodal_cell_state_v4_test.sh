#!/usr/bin/env bash
set -euo pipefail

required_host=hpctpa3pc0009
observed_host="$(hostname -s)"
[[ "$observed_host" == "$required_host" ]] || {
  echo "Multimodal cell-state V4 calibration requires login-node ssh followed by ssh to $required_host; observed=$observed_host" >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "Multimodal cell-state V4 calibration is direct and cannot run inside Slurm" >&2
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

EXPERIMENT_ROOT=/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/results}"
DATASET_ROOT="${DATASET_ROOT:-$EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$RESULTS_ROOT/full_fusion_shape_strict_20260711_155940}"
BASE_EXPANDED_ROOT="${BASE_EXPANDED_ROOT:-$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_20260813_024929_expanded39}"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$RESULTS_ROOT/broad_phenotype_shadow_20260812_075437}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
V4_SHADOW_ROOT="${V4_SHADOW_ROOT:-$RESULTS_ROOT/Tests_and_Parameters_calibration/multimodal_cell_state_v4_test_$RUN_STAMP}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
DIRECT_TEST_CONCURRENCY="${DIRECT_TEST_CONCURRENCY:-16}"
V4_ACTION="${V4_ACTION:-phase-a}"

[[ "$RESULTS_ROOT" == "$EXPERIMENT_ROOT/results" && -d "$RESULTS_ROOT/Tests_and_Parameters_calibration" ]] || {
  echo "V4 calibration must write only under the experiment Tests_and_Parameters_calibration root" >&2
  exit 2
}
calibration_parent="$RESULTS_ROOT/Tests_and_Parameters_calibration"
case "$(basename "$V4_SHADOW_ROOT")" in
  multimodal_cell_state_v4_test_*) ;;
  *) echo "V4 calibration root has an invalid generation name: $V4_SHADOW_ROOT" >&2; exit 2 ;;
esac
[[ "$(cd "$(dirname "$V4_SHADOW_ROOT")" && pwd -P)" == "$calibration_parent" ]] || {
  echo "V4 calibration root is outside the frozen test namespace: $V4_SHADOW_ROOT" >&2
  exit 2
}
[[ "$DIRECT_TEST_CONCURRENCY" =~ ^[0-9]+$ && "$DIRECT_TEST_CONCURRENCY" -ge 1 && "$DIRECT_TEST_CONCURRENCY" -le 32 ]] || {
  echo "DIRECT_TEST_CONCURRENCY must be an integer in [1,32]" >&2
  exit 2
}
case "$V4_ACTION" in
  phase-a|post-anchor|post-region|post-review) ;;
  *) echo "Unsupported V4_ACTION: $V4_ACTION" >&2; exit 2 ;;
esac
for root in "$BASE_EXPANDED_ROOT" "$PARENT_BROAD_SHADOW_ROOT" "$CPA_REFERENCE_ROOT"; do
  [[ "$root" == /* && -d "$root" && ! -L "$root" ]] || {
    echo "Required frozen V4 input root is unavailable: $root" >&2
    exit 2
  }
done
if [[ -e "$V4_SHADOW_ROOT" || -L "$V4_SHADOW_ROOT" ]]; then
  [[ -d "$V4_SHADOW_ROOT" && ! -L "$V4_SHADOW_ROOT" ]] || {
    echo "Existing V4 calibration root is not a real directory: $V4_SHADOW_ROOT" >&2
    exit 2
  }
  [[ "$(cd "$V4_SHADOW_ROOT" && pwd -P)" == "$calibration_parent/$(basename "$V4_SHADOW_ROOT")" ]] || {
    echo "Existing V4 calibration root resolves outside the frozen test namespace" >&2
    exit 2
  }
elif [[ "$V4_ACTION" != phase-a ]]; then
  echo "V4 post-human action requires an existing calibration root" >&2
  exit 2
fi

BASE_PROJECT="$BASE_EXPANDED_ROOT/projection_input/representative_umap_v2/project.yml"
EXPANDED_PROJECTION="$BASE_EXPANDED_ROOT/workflow_status/expanded_projection_comparison"
DRIVER="$PROJECT_DIR/cellpose_pipeline/scripts/54_run_multimodal_cell_state_v4_calibration.py"
POSTHUMAN_DRIVER="$PROJECT_DIR/cellpose_pipeline/scripts/59_run_multimodal_cell_state_v4_posthuman.py"
DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
for file in "$BASE_PROJECT" "$EXPANDED_PROJECTION/expanded_projection_manifest.json" "$DRIVER" "$POSTHUMAN_DRIVER" "$DEPENDENCY_LOCK"; do
  [[ -f "$file" && ! -L "$file" ]] || { echo "Required V4 calibration file is unavailable: $file" >&2; exit 2; }
done

reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"
BRIGHTFIELD_ROOT="$DATASET_ROOT/Brightfield"
NUCLEI_RAW_ROOT="$DATASET_ROOT/Nuclei"
DEAD_RAW_ROOT="$DATASET_ROOT/Dead"
COMBINED_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Combined/segmentations"
NUCLEI_MASK_ROOT="$SOURCE_SEGMENTATION_ROOT/Nuclei/segmentations"
NUCLEI_CORE_ROOT="$SOURCE_SEGMENTATION_ROOT/Nuclei/nucleus_core_seeds"
for root in "$BRIGHTFIELD_ROOT" "$NUCLEI_RAW_ROOT" "$DEAD_RAW_ROOT" "$COMBINED_MASK_ROOT" "$NUCLEI_MASK_ROOT" "$NUCLEI_CORE_ROOT"; do
  [[ -d "$root" && ! -L "$root" ]] || { echo "Allowed V4 evidence root is unavailable: $root" >&2; exit 2; }
done

submission=""
case "$V4_ACTION" in
  post-anchor) : "${ANCHOR_SUBMISSION:?ANCHOR_SUBMISSION is required}"; submission="$ANCHOR_SUBMISSION" ;;
  post-region) : "${REGION_SUBMISSION:?REGION_SUBMISSION is required}"; submission="$REGION_SUBMISSION" ;;
  post-review) : "${BROAD_REVIEW_SUBMISSION:?BROAD_REVIEW_SUBMISSION is required}"; submission="$BROAD_REVIEW_SUBMISSION" ;;
esac
if [[ -n "$submission" ]]; then
  case "$submission" in "$V4_SHADOW_ROOT"/*) ;; *) echo "V4 human submission must remain inside V4_SHADOW_ROOT" >&2; exit 2 ;; esac
  [[ -s "$submission" && ! -L "$submission" ]] || { echo "V4 human submission is unavailable: $submission" >&2; exit 2; }
fi

if [[ ! -d "$V4_SHADOW_ROOT" ]]; then
  mkdir "$V4_SHADOW_ROOT"
fi
mkdir -p "$V4_SHADOW_ROOT/workflow_status"

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
reference_cell_state_v2_require_clean_worker_environment
expected_binds="$V4_SHADOW_ROOT:$V4_SHADOW_ROOT:rw,$BASE_EXPANDED_ROOT:$BASE_EXPANDED_ROOT:ro,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_RAW_ROOT:$NUCLEI_RAW_ROOT:ro,$DEAD_RAW_ROOT:$DEAD_RAW_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$NUCLEI_MASK_ROOT:$NUCLEI_MASK_ROOT:ro,$NUCLEI_CORE_ROOT:$NUCLEI_CORE_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
case "${HPC_CONTAINER_BINDS+x}:${HPC_CONTAINER_BINDS:-}" in
  :) HPC_CONTAINER_BINDS="$expected_binds" ;;
  "x:$expected_binds") ;;
  *) echo "HPC_CONTAINER_BINDS differs from the V4 calibration allowlist" >&2; exit 2 ;;
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
HPC_CONTAINER_IDENTITY_FILE="$V4_SHADOW_ROOT/workflow_status/hpc_container_identity.json"
if [[ -e "$HPC_CONTAINER_IDENTITY_FILE" || -L "$HPC_CONTAINER_IDENTITY_FILE" ]]; then
  [[ -s "$HPC_CONTAINER_IDENTITY_FILE" && ! -L "$HPC_CONTAINER_IDENTITY_FILE" ]] || {
    echo "Existing V4 calibration SIF identity receipt is invalid" >&2
    exit 2
  }
else
  broad_phenotype_capture_container_identity \
    "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256" "$HPC_CONTAINER_IDENTITY_FILE"
fi
HPC_CONTAINER_IDENTITY_FILE_SHA256="$(reference_cell_state_v2_sha256 "$HPC_CONTAINER_IDENTITY_FILE")"
export HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256
broad_phenotype_worker_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256"

unset PYTHONPATH PYTHONHOME R_LIBS R_LIBS_USER R_ENVIRON_USER R_PROFILE_USER
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hpc_apptainer_exec python -I - "$DATASET_ROOT" "$SOURCE_SEGMENTATION_ROOT" "$V4_SHADOW_ROOT" <<'PY'
from pathlib import Path
import sys
dataset, segmentation, output = map(Path, sys.argv[1:])
allowed = (
    dataset / "Brightfield", dataset / "Nuclei", dataset / "Dead",
    segmentation / "Combined" / "segmentations",
    segmentation / "Nuclei" / "segmentations",
    segmentation / "Nuclei" / "nucleus_core_seeds",
    output,
)
for path in allowed:
    if not path.exists():
        raise SystemExit(f"allowlisted V4 path is invisible: {path}")
for forbidden in (dataset / "Combined", segmentation / "Dead"):
    if forbidden.exists():
        raise SystemExit(f"forbidden V4 channel remained visible: {forbidden}")
print("multimodal_cell_state_v4_container_blinding=PASS raw_dead_visible=1 existing_dead_segmentation_visible=0")
PY

case "$V4_ACTION" in
  phase-a)
    hpc_apptainer_exec python -I "$DRIVER" \
      --base-project "$BASE_PROJECT" \
      --base-shadow-root "$BASE_EXPANDED_ROOT" \
      --parent-broad-shadow-root "$PARENT_BROAD_SHADOW_ROOT" \
      --expanded-projection "$EXPANDED_PROJECTION" \
      --shadow-root "$V4_SHADOW_ROOT" \
      --project-root "$PROJECT_DIR" \
      --reference-root "$CPA_REFERENCE_ROOT" \
      --dependency-lock "$DEPENDENCY_LOCK" \
      --container-identity "$HPC_CONTAINER_IDENTITY_FILE" \
      --expected-container-sha256 "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256" \
      --concurrency "$DIRECT_TEST_CONCURRENCY" \
      --expected-cell-count 32000
    echo "multimodal_cell_state_v4_calibration_complete=1"
    echo "v4_shadow_root=$V4_SHADOW_ROOT"
    echo "annotation_workspace=$V4_SHADOW_ROOT/morphology_audit/annotation_workspace.html"
    echo "human_workspace=$V4_SHADOW_ROOT/human_review/anchor_seed1/render/exact_review.html"
    echo "human_submission_expected_path=$V4_SHADOW_ROOT/human_review/anchor_seed1/render/multimodal_v4_review_submission.json"
    echo "human_barrier=independent_three_channel_anchor_submission_required"
    ;;
  post-anchor|post-region|post-review)
    hpc_apptainer_exec python -I "$POSTHUMAN_DRIVER" \
      --action "$V4_ACTION" \
      --submission "$submission" \
      --shadow-root "$V4_SHADOW_ROOT" \
      --project-root "$PROJECT_DIR" \
      --reference-root "$CPA_REFERENCE_ROOT" \
      --dependency-lock "$DEPENDENCY_LOCK"
    ;;
esac
