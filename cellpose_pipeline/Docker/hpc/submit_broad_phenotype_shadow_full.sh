#!/usr/bin/env bash
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0

DEFAULT_HPC_CONTAINER_IMAGE="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_HPC_CONTAINER_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"
LEGACY_PATH_PREFIX="/share/lab_crd/lab_crd/"
LIVE_PATH_PREFIX="/share/lab_crd/"
DEFAULT_STALE_DATASET_ROOT="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages"
DEFAULT_STALE_SEGMENTATION_ROOT="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/20260626_SUM159_AC_Exp1_calibrated_dead_dual_branch_20260711_155940"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x /usr/bin/bash ]]; then
  SYSTEM_BASH_BIN=/usr/bin/bash
elif [[ -x /bin/bash ]]; then
  SYSTEM_BASH_BIN=/bin/bash
else
  echo "An absolute system Bash interpreter is required" >&2
  exit 2
fi
CONTROLLED_TOOL_PATH="$(dirname "$(command -v apptainer || true)")"
[[ "$CONTROLLED_TOOL_PATH" == /* ]] || CONTROLLED_TOOL_PATH=/usr/bin
for tool_name in sbatch sacctmgr; do
  tool_path="$(command -v "$tool_name" || true)"
  if [[ "$tool_path" == /* ]]; then
    tool_dir="$(dirname "$tool_path")"
    case ":$CONTROLLED_TOOL_PATH:" in
      *":$tool_dir:"*) ;;
      *) CONTROLLED_TOOL_PATH="$CONTROLLED_TOOL_PATH:$tool_dir" ;;
    esac
  fi
done
CONTROLLED_TOOL_PATH="$CONTROLLED_TOOL_PATH:/usr/bin:/bin"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SOURCE_PROJECT_DIR="$PROJECT_DIR"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_HPC_CONTAINER_IMAGE}"
HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXECUTION_MODE="${EXECUTION_MODE:-slurm}"
RESUME_STAGE="${RESUME_STAGE:-phase-a}"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"
EXPECTED_FIELDS="${EXPECTED_FIELDS:-27200}"
BRANCH="${BRANCH:-original}"
FEATURE_MAX_CONCURRENT="${FEATURE_MAX_CONCURRENT:-64}"
INCLUDE_NUCLEI_COMPARATOR="${INCLUDE_NUCLEI_COMPARATOR:-1}"
BROAD_PHENOTYPE_QOS="${BROAD_PHENOTYPE_QOS-xxlarge}"
MAX_UMAP_FIELDS_PER_WELL="${MAX_UMAP_FIELDS_PER_WELL:-20}"
SPLIT_SEED="${SPLIT_SEED:-20260812}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-5}"
BROAD_PHENOTYPE_PROJECT_ID="${BROAD_PHENOTYPE_PROJECT_ID:-broad_phenotype}"
MAX_UMAP_CELLS_PER_FIELD="${MAX_UMAP_CELLS_PER_FIELD:-25}"
MAX_UMAP_CELLS_PER_WELL="${MAX_UMAP_CELLS_PER_WELL:-500}"
UMAP_SAMPLE_SEED="${UMAP_SAMPLE_SEED:-20260812}"
FIXED_HPC_CONTAINER_FORWARD_PREFIXES="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES,REPORT_PLUGIN_ROOT"

for forbidden_runtime_override in HPC_CONTAINER_RUNTIME_ROOT HPC_CONTAINER_FORWARD_PREFIXES SOURCE_FIELD_MANIFEST_FILE; do
  [[ -z "${!forbidden_runtime_override+x}" ]] || {
    echo "$forbidden_runtime_override is forbidden; the submitter derives the runtime and manifest from frozen roots" >&2
    exit 2
  }
done

for forbidden_code_override in \
  BROAD_PHENOTYPE_FEATURE_SCRIPT \
  BROAD_PHENOTYPE_ADAPTER_SCRIPT \
  BROAD_PHENOTYPE_PROJECTION_SCRIPT \
  BROAD_PHENOTYPE_INPUT_CONTRACT_VALIDATOR \
  BROAD_PHENOTYPE_CPA_STAGE_SCRIPT \
  BROAD_PHENOTYPE_MODEL_ACCEPT_SCRIPT \
  BROAD_PHENOTYPE_SHARD_PREDICT_SCRIPT \
  BROAD_PHENOTYPE_FINALIZE_SCRIPT \
  BROAD_PHENOTYPE_SHARDED_FINALIZE_SCRIPT; do
  [[ -z "${!forbidden_code_override+x}" ]] || {
    echo "$forbidden_code_override is forbidden; the frozen code snapshot supplies every executable code path" >&2
    exit 2
  }
done
[[ -z "${HPC_CONTAINER_BINDS:-}" ]] || {
  echo "Ambient HPC_CONTAINER_BINDS is forbidden; the submitter constructs the complete bind allowlist" >&2
  exit 2
}
HPC_CONTAINER_BINDS=""
export HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_BINDS
for forbidden_algorithm_override in BROAD_PHENOTYPE_FEATURE_COLUMNS PROJECT_FILE CPA_VALIDATE_OUTPUT_DIR FORCE_FEATURES FORCE_PREDICT FORCE_FINALIZE CELLS_FILE MODEL_DIR; do
  [[ -z "${!forbidden_algorithm_override+x}" ]] || {
    echo "$forbidden_algorithm_override is forbidden at submission; the frozen workflow contract supplies it" >&2
    exit 2
  }
done
FORCE_FEATURES=0
FORCE_PREDICT=0
FORCE_FINALIZE=0

: "${RESULTS_ROOT:?RESULTS_ROOT must be the explicit experiment results directory}"
: "${DATASET_ROOT:?DATASET_ROOT must be the explicit raw-image dataset root}"
: "${FIELD_MANIFEST_DIR:?FIELD_MANIFEST_DIR must be the explicit Stage 06 manifest root}"
: "${CLASSIFICATION_ROOT:?CLASSIFICATION_ROOT must be the explicit immutable current-classification root}"
: "${LEGACY_NO_GO:?LEGACY_NO_GO is required for informational provenance only}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$(cd "$FIELD_MANIFEST_DIR/../.." && pwd -P)}"
STALE_DATASET_ROOT="${STALE_DATASET_ROOT:-$DEFAULT_STALE_DATASET_ROOT}"
STALE_SEGMENTATION_ROOT="${STALE_SEGMENTATION_ROOT:-$DEFAULT_STALE_SEGMENTATION_ROOT}"

for absolute_name in RESULTS_ROOT DATASET_ROOT FIELD_MANIFEST_DIR SOURCE_SEGMENTATION_ROOT CLASSIFICATION_ROOT LEGACY_NO_GO CPA_REFERENCE_ROOT HPC_CONTAINER_IMAGE PROJECT_DIR STALE_DATASET_ROOT STALE_SEGMENTATION_ROOT; do
  absolute_value="${!absolute_name}"
  [[ "$absolute_value" == /* ]] || {
    echo "$absolute_name must be absolute: $absolute_value" >&2
    exit 2
  }
done
[[ "$(basename "$RESULTS_ROOT")" == "results" ]] || {
  echo "RESULTS_ROOT must end in /results: $RESULTS_ROOT" >&2
  exit 2
}
[[ -d "$RESULTS_ROOT" && -d "$DATASET_ROOT" && -d "$FIELD_MANIFEST_DIR" && -d "$SOURCE_SEGMENTATION_ROOT" && -d "$CLASSIFICATION_ROOT" ]] || {
  echo "RESULTS_ROOT, DATASET_ROOT, FIELD_MANIFEST_DIR, SOURCE_SEGMENTATION_ROOT, and CLASSIFICATION_ROOT must exist" >&2
  exit 2
}
resolved_field_manifest_dir="$(cd "$FIELD_MANIFEST_DIR" && pwd -P)"
expected_field_manifest_dir="$(cd "$CLASSIFICATION_ROOT" && pwd -P)/workflow_status/postsegmentation_manifest"
[[ "$resolved_field_manifest_dir" == "$expected_field_manifest_dir" ]] || {
  echo "FIELD_MANIFEST_DIR must be CLASSIFICATION_ROOT/workflow_status/postsegmentation_manifest: expected=$expected_field_manifest_dir observed=$resolved_field_manifest_dir" >&2
  exit 2
}
[[ -f "$LEGACY_NO_GO" ]] || { echo "LEGACY_NO_GO is unavailable: $LEGACY_NO_GO" >&2; exit 2; }
[[ -r "$HPC_CONTAINER_IMAGE" ]] || { echo "HPC container is unavailable: $HPC_CONTAINER_IMAGE" >&2; exit 2; }
[[ -d "$CPA_REFERENCE_ROOT" ]] || { echo "Pinned Cell Phenotype Annotator checkout is unavailable: $CPA_REFERENCE_ROOT" >&2; exit 2; }

case "$EXECUTION_MODE" in
  slurm|direct_test) ;;
  *) echo "EXECUTION_MODE must be slurm or direct_test" >&2; exit 2 ;;
esac
case "$DRY_RUN_SUBMIT" in 0|1) ;; *) echo "DRY_RUN_SUBMIT must be 0 or 1" >&2; exit 2 ;; esac
case "$BRANCH" in original|nucleated) ;; *) echo "BRANCH must be original or nucleated" >&2; exit 2 ;; esac
case "$INCLUDE_NUCLEI_COMPARATOR" in 0|1) ;; *) echo "INCLUDE_NUCLEI_COMPARATOR must be 0 or 1" >&2; exit 2 ;; esac
[[ "$MAX_UMAP_FIELDS_PER_WELL" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_UMAP_FIELDS_PER_WELL must be a positive integer" >&2; exit 2; }
for positive_integer_name in SPLIT_SEED OUTER_FOLDS INNER_FOLDS MAX_UMAP_CELLS_PER_FIELD MAX_UMAP_CELLS_PER_WELL UMAP_SAMPLE_SEED; do
  positive_integer_value="${!positive_integer_name}"
  [[ "$positive_integer_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$positive_integer_name must be a positive integer: $positive_integer_value" >&2
    exit 2
  }
done
[[ "$BROAD_PHENOTYPE_PROJECT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "BROAD_PHENOTYPE_PROJECT_ID must be a filesystem-safe identifier: $BROAD_PHENOTYPE_PROJECT_ID" >&2
  exit 2
}

if [[ "$EXECUTION_MODE" == "slurm" ]]; then
  [[ "$EXPECTED_FIELDS" == "27200" ]] || {
    echo "Formal Slurm runs require the complete 27200-field universe; EXPECTED_FIELDS=$EXPECTED_FIELDS" >&2
    exit 2
  }
  [[ -z "${TASK_LIST_INPUT:-}" ]] || {
    echo "Formal Slurm runs forbid TASK_LIST_INPUT; the canonical Stage 06 manifest defines the universe" >&2
    exit 2
  }
  [[ -z "${HELDOUT_WELLS:-}" ]] || {
    echo "Formal Slurm runs forbid HELDOUT_WELLS; the frozen plate-map split is mandatory" >&2
    exit 2
  }
  case "${CPA_STAGE_MODE:-}" in
    ""|overwrite) ;;
    dry-run|check-config)
      echo "Formal Slurm runs forbid CPA_STAGE_MODE=${CPA_STAGE_MODE}" >&2
      exit 2
      ;;
    *) echo "CPA_STAGE_MODE must be empty or overwrite for a formal resume" >&2; exit 2 ;;
  esac
  if [[ "$RESUME_STAGE" == "phase-a" && -n "${CPA_STAGE_MODE:-}" ]]; then
    echo "Formal phase-A requires CPA_STAGE_MODE to be empty for real validate/UMAP/annotate execution" >&2
    exit 2
  fi
  [[ -z "${TEST_CPA_STAGE_MODE+x}" ]] || {
    echo "Formal Slurm runs forbid the TEST_CPA_STAGE_MODE override" >&2
    exit 2
  }
  EXECUTION_SCOPE="formal_full"
  SPLIT_MODE="plate_map_preregistered"
  FROZEN_HELDOUT_WELLS="plate_map_preregistered"
  SHADOW_ROOT="${SHADOW_ROOT:-$RESULTS_ROOT/broad_phenotype_shadow_$RUN_STAMP}"
  [[ "$(dirname "$SHADOW_ROOT")" == "$RESULTS_ROOT" && "$(basename "$SHADOW_ROOT")" == broad_phenotype_shadow_* ]] || {
    echo "Formal SHADOW_ROOT must be results/broad_phenotype_shadow_TIMESTAMP: $SHADOW_ROOT" >&2
    exit 2
  }
else
  : "${SHADOW_ROOT:?direct_test requires an explicit calibration SHADOW_ROOT}"
  : "${HELDOUT_WELLS:?direct_test requires one explicit calibration heldout well}"
  [[ "$HELDOUT_WELLS" =~ ^[A-H][0-9]+$ ]] || {
    echo "direct_test HELDOUT_WELLS must contain exactly one well: $HELDOUT_WELLS" >&2
    exit 2
  }
  EXECUTION_SCOPE="calibration_subset"
  SPLIT_MODE="explicit_heldout_wells"
  FROZEN_HELDOUT_WELLS="$HELDOUT_WELLS"
  calibration_root="$RESULTS_ROOT/Tests_and_Parameters_calibration"
  [[ "$SHADOW_ROOT" == "$calibration_root"/* && "$(basename "$SHADOW_ROOT")" == broad_phenotype_shadow_test_* ]] || {
    echo "Direct test SHADOW_ROOT must be under $calibration_root: $SHADOW_ROOT" >&2
    exit 2
  }
fi

CODE_SNAPSHOT_ROOT="$SHADOW_ROOT/workflow_status/code_snapshot"
CODE_SNAPSHOT_ARCHIVE="$SHADOW_ROOT/workflow_status/code_snapshot.tar"
CODE_SNAPSHOT_RECEIPT="$SHADOW_ROOT/workflow_status/code_snapshot_receipt.tsv"
if [[ "$RESUME_STAGE" != "phase-a" ]]; then
  [[ -d "$CODE_SNAPSHOT_ROOT" && -s "$CODE_SNAPSHOT_ARCHIVE" && -s "$CODE_SNAPSHOT_RECEIPT" ]] || {
    echo "Frozen code snapshot is unavailable for resume: $CODE_SNAPSHOT_ROOT" >&2
    exit 2
  }
  PROJECT_DIR="$CODE_SNAPSHOT_ROOT"
  snapshot_submitter="$CODE_SNAPSHOT_ROOT/cellpose_pipeline/Docker/hpc/submit_broad_phenotype_shadow_full.sh"
  [[ -f "$snapshot_submitter" ]] || {
    echo "Frozen code snapshot does not contain its submitter: $snapshot_submitter" >&2
    exit 2
  }
  current_submitter_realpath="$(cd "$SCRIPT_DIR" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
  snapshot_submitter_realpath="$(cd "$(dirname "$snapshot_submitter")" && pwd -P)/$(basename "$snapshot_submitter")"
  if [[ "$current_submitter_realpath" != "$snapshot_submitter_realpath" ]]; then
    [[ "$(head -n 1 "$CODE_SNAPSHOT_RECEIPT")" == $'property\tvalue' ]] || {
      echo "Code snapshot receipt header changed before resume re-exec" >&2
      exit 2
    }
    early_archive_sha="$(awk -F '\t' '$1=="archive_sha256" && $2 ~ /^[0-9a-f]{64}$/ {print $2; n++} END {if(n!=1) exit 2}' "$CODE_SNAPSHOT_RECEIPT")" || {
      echo "Code snapshot receipt must contain one valid archive_sha256 before resume re-exec" >&2
      exit 2
    }
    observed_early_archive_sha="$(sha256sum "$CODE_SNAPSHOT_ARCHIVE")"
    observed_early_archive_sha="${observed_early_archive_sha%%[[:space:]]*}"
    [[ "$observed_early_archive_sha" == "$early_archive_sha" ]] || {
      echo "Code snapshot archive hash differs before resume re-exec" >&2
      exit 2
    }
    tar -tf "$CODE_SNAPSHOT_ARCHIVE" | awk '
      /^\// {exit 2}
      {n=split($0, parts, "/"); for(i=1;i<=n;i++) if(parts[i]=="..") exit 3}
    ' || {
      echo "Code snapshot archive contains an unsafe path" >&2
      exit 2
    }
    early_snapshot_compare="$(mktemp -d "${TMPDIR:-/tmp}/broad-phenotype-resume-preexec.XXXXXX")"
    cleanup_early_snapshot_compare() {
      rm -rf -- "$early_snapshot_compare"
    }
    trap cleanup_early_snapshot_compare EXIT
    tar -xf "$CODE_SNAPSHOT_ARCHIVE" -C "$early_snapshot_compare" || {
      echo "Unable to extract code snapshot before resume re-exec" >&2
      exit 2
    }
    diff -qr "$early_snapshot_compare" "$CODE_SNAPSHOT_ROOT" >/dev/null || {
      echo "Extracted code snapshot differs from its frozen archive before resume re-exec" >&2
      exit 2
    }
    rm -rf -- "$early_snapshot_compare"
    trap - EXIT
    exec /usr/bin/env -i \
      PATH="$CONTROLLED_TOOL_PATH" \
      RESULTS_ROOT="$RESULTS_ROOT" \
      DATASET_ROOT="$DATASET_ROOT" \
      FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR" \
      SOURCE_SEGMENTATION_ROOT="$SOURCE_SEGMENTATION_ROOT" \
      CLASSIFICATION_ROOT="$CLASSIFICATION_ROOT" \
      LEGACY_NO_GO="$LEGACY_NO_GO" \
      CPA_REFERENCE_ROOT="$CPA_REFERENCE_ROOT" \
      HPC_CONTAINER_IMAGE="$HPC_CONTAINER_IMAGE" \
      PROJECT_DIR="$PROJECT_DIR" \
      SHADOW_ROOT="$SHADOW_ROOT" \
      RUN_STAMP="$RUN_STAMP" \
      EXECUTION_MODE="$EXECUTION_MODE" \
      RESUME_STAGE="$RESUME_STAGE" \
      DRY_RUN_SUBMIT="$DRY_RUN_SUBMIT" \
      EXPECTED_FIELDS="$EXPECTED_FIELDS" \
      BRANCH="$BRANCH" \
      FEATURE_MAX_CONCURRENT="$FEATURE_MAX_CONCURRENT" \
      INCLUDE_NUCLEI_COMPARATOR="$INCLUDE_NUCLEI_COMPARATOR" \
      BROAD_PHENOTYPE_QOS="$BROAD_PHENOTYPE_QOS" \
      MAX_UMAP_FIELDS_PER_WELL="$MAX_UMAP_FIELDS_PER_WELL" \
      SPLIT_SEED="$SPLIT_SEED" \
      OUTER_FOLDS="$OUTER_FOLDS" \
      INNER_FOLDS="$INNER_FOLDS" \
      BROAD_PHENOTYPE_PROJECT_ID="$BROAD_PHENOTYPE_PROJECT_ID" \
      MAX_UMAP_CELLS_PER_FIELD="$MAX_UMAP_CELLS_PER_FIELD" \
      MAX_UMAP_CELLS_PER_WELL="$MAX_UMAP_CELLS_PER_WELL" \
      UMAP_SAMPLE_SEED="$UMAP_SAMPLE_SEED" \
      STALE_DATASET_ROOT="$STALE_DATASET_ROOT" \
      STALE_SEGMENTATION_ROOT="$STALE_SEGMENTATION_ROOT" \
      CPA_STAGE_MODE="${CPA_STAGE_MODE:-}" \
      DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}" \
      RETRY_TASK_LIST="${RETRY_TASK_LIST:-}" \
      CPA_SUBMISSION="${CPA_SUBMISSION:-}" \
      CPA_ANNOTATION_IMPORT_DIR="${CPA_ANNOTATION_IMPORT_DIR:-}" \
      CPA_REVIEWED_LABELS="${CPA_REVIEWED_LABELS:-}" \
      CPA_MODEL_DIR="${CPA_MODEL_DIR:-}" \
      CPA_TRAIN_RECEIPT="${CPA_TRAIN_RECEIPT:-}" \
      PREDICTION_DIR="${PREDICTION_DIR:-}" \
      "$SYSTEM_BASH_BIN" "$snapshot_submitter" "$@"
  fi
fi

if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  submitter_project_root="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
  source_project_realpath="$(cd "$SOURCE_PROJECT_DIR" && pwd -P)"
  [[ "$source_project_realpath" == "$submitter_project_root" ]] || {
    echo "Phase-A PROJECT_DIR must be the checkout containing this submitter: submitter=$submitter_project_root project=$source_project_realpath" >&2
    exit 2
  }
  [[ ! -e "$SHADOW_ROOT" ]] || {
    echo "Refusing to replace an existing broad-phenotype shadow root: $SHADOW_ROOT" >&2
    exit 2
  }
else
  [[ -d "$SHADOW_ROOT" ]] || { echo "Resume SHADOW_ROOT is unavailable: $SHADOW_ROOT" >&2; exit 2; }
fi

append_bind() {
  local bind="$1"
  if [[ -n "${HPC_CONTAINER_BINDS:-}" ]]; then
    HPC_CONTAINER_BINDS="$HPC_CONTAINER_BINDS,$bind"
  else
    HPC_CONTAINER_BINDS="$bind"
  fi
}
append_bind "$RESULTS_ROOT:$RESULTS_ROOT:ro"
append_bind "$SHADOW_ROOT:$SHADOW_ROOT"
append_bind "$DATASET_ROOT:$DATASET_ROOT:ro"
append_bind "$FIELD_MANIFEST_DIR:$FIELD_MANIFEST_DIR:ro"
append_bind "$SOURCE_SEGMENTATION_ROOT:$SOURCE_SEGMENTATION_ROOT:ro"
append_bind "$CLASSIFICATION_ROOT:$CLASSIFICATION_ROOT:ro"
append_bind "$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
append_bind "$CODE_SNAPSHOT_ROOT:$CODE_SNAPSHOT_ROOT:ro"
export HPC_CONTAINER_BINDS

source "$SCRIPT_DIR/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare
source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"

for locked_override in CPA_DEPENDENCY_LOCK FEATURE_CONFIG CLASSES_FILE PLATE_MAP; do
  [[ -z "${!locked_override+x}" ]] || {
    echo "$locked_override overrides are forbidden; the tracked code snapshot supplies the locked input" >&2
    exit 2
  }
done
DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
FEATURE_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json"
CLASSES_FILE="$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_classes_v1.tsv"
PLATE_MAP="$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv"
for required_file in "$DEPENDENCY_LOCK" "$FEATURE_CONFIG" "$CLASSES_FILE" "$PLATE_MAP"; do
  [[ -s "$required_file" ]] || { echo "Locked broad-phenotype input is unavailable: $required_file" >&2; exit 2; }
done

lock_row="$(awk -F '\t' '$1=="cellphenotypeannotator" {print; n++} END {if(n!=1) exit 2}' "$DEPENDENCY_LOCK")" || {
  echo "Dependency lock must contain exactly one cellphenotypeannotator row" >&2
  exit 2
}
IFS=$'\t' read -r _ _ locked_commit locked_tree locked_license locked_license_sha locked_description_sha locked_mode <<< "$lock_row"
[[ "$locked_mode" == "private_read_only_source_checkout" ]] || { echo "Invalid dependency execution mode: $locked_mode" >&2; exit 2; }
observed_commit="$(git -C "$CPA_REFERENCE_ROOT" rev-parse HEAD)"
observed_tree="$(git -C "$CPA_REFERENCE_ROOT" rev-parse 'HEAD^{tree}')"
observed_status="$(git -C "$CPA_REFERENCE_ROOT" status --porcelain --untracked-files=all)"
observed_license_sha="$(sha256sum "$CPA_REFERENCE_ROOT/$locked_license")"; observed_license_sha="${observed_license_sha%%[[:space:]]*}"
observed_description_sha="$(sha256sum "$CPA_REFERENCE_ROOT/DESCRIPTION")"; observed_description_sha="${observed_description_sha%%[[:space:]]*}"
[[ "$observed_commit" == "$locked_commit" && "$observed_tree" == "$locked_tree" && -z "$observed_status" && "$observed_license_sha" == "$locked_license_sha" && "$observed_description_sha" == "$locked_description_sha" ]] || {
  echo "Pinned Cell Phenotype Annotator checkout does not match dependency lock" >&2
  exit 2
}

FEATURE_CPUS="${FEATURE_CPUS:-2}"; FEATURE_MEM="${FEATURE_MEM:-8G}"; FEATURE_TIME="${FEATURE_TIME:-02:00:00}"
PREFLIGHT_CPUS="${PREFLIGHT_CPUS:-2}"; PREFLIGHT_MEM="${PREFLIGHT_MEM:-8G}"; PREFLIGHT_TIME="${PREFLIGHT_TIME:-04:00:00}"
ADAPTER_CPUS="${ADAPTER_CPUS:-8}"; ADAPTER_MEM="${ADAPTER_MEM:-128G}"; ADAPTER_TIME="${ADAPTER_TIME:-12:00:00}"
VALIDATE_CPUS="${VALIDATE_CPUS:-4}"; VALIDATE_MEM="${VALIDATE_MEM:-128G}"; VALIDATE_TIME="${VALIDATE_TIME:-04:00:00}"
UMAP_CPUS="${UMAP_CPUS:-16}"; UMAP_MEM="${UMAP_MEM:-128G}"; UMAP_TIME="${UMAP_TIME:-12:00:00}"
ANNOTATE_CPUS="${ANNOTATE_CPUS:-4}"; ANNOTATE_MEM="${ANNOTATE_MEM:-128G}"; ANNOTATE_TIME="${ANNOTATE_TIME:-08:00:00}"
RESUME_CPUS="${RESUME_CPUS:-8}"; RESUME_MEM="${RESUME_MEM:-64G}"; RESUME_TIME="${RESUME_TIME:-08:00:00}"
REVIEW_CPUS="${REVIEW_CPUS:-8}"; REVIEW_MEM="${REVIEW_MEM:-256G}"; REVIEW_TIME="${REVIEW_TIME:-12:00:00}"
PREDICT_CPUS="${PREDICT_CPUS:-2}"; PREDICT_MEM="${PREDICT_MEM:-8G}"; PREDICT_TIME="${PREDICT_TIME:-02:00:00}"; PREDICT_MAX_CONCURRENT="${PREDICT_MAX_CONCURRENT:-64}"
MODEL_ACCEPTANCE_CPUS="${MODEL_ACCEPTANCE_CPUS:-4}"; MODEL_ACCEPTANCE_MEM="${MODEL_ACCEPTANCE_MEM:-16G}"; MODEL_ACCEPTANCE_TIME="${MODEL_ACCEPTANCE_TIME:-02:00:00}"

slurm_time_to_seconds() {
  local value="$1" days=0 clock part_count hours=0 minutes=0 seconds=0
  local -a time_parts
  value="${value//[[:space:]]/}"
  [[ -n "$value" ]] || return 2
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    clock="${value#*-}"
  else
    clock="$value"
  fi
  [[ "$days" =~ ^[0-9]+$ ]] || return 2
  IFS=: read -r -a time_parts <<< "$clock"
  part_count="${#time_parts[@]}"
  case "$part_count" in
    1) minutes="${time_parts[0]}" ;;
    2) minutes="${time_parts[0]}"; seconds="${time_parts[1]}" ;;
    3) hours="${time_parts[0]}"; minutes="${time_parts[1]}"; seconds="${time_parts[2]}" ;;
    *) return 2 ;;
  esac
  [[ "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]] || return 2
  days=$((10#$days)); hours=$((10#$hours)); minutes=$((10#$minutes)); seconds=$((10#$seconds))
  (( minutes < 60 && seconds < 60 )) || return 2
  printf '%s\n' "$((days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}

QOS_MAX_WALL="not_applicable"
validate_qos_walltime_limits() {
  [[ "$EXECUTION_MODE" == "slurm" && "$DRY_RUN_SUBMIT" == "0" && -n "$BROAD_PHENOTYPE_QOS" ]] || return 0
  local sacctmgr_bin
  sacctmgr_bin="$(command -v sacctmgr || true)"
  [[ "$sacctmgr_bin" == /* && -x "$sacctmgr_bin" ]] || {
    echo "sacctmgr is required to verify the Slurm QOS wall-time contract before submitting any job" >&2
    exit 2
  }
  local qos_row max_wall max_seconds stage pair requested requested_seconds
  qos_row="$(/usr/bin/env -u SLURM_CLUSTERS -u SLURM_CONF -u SLURM_CONF_SERVER \
    "$sacctmgr_bin" --noheader --parsable2 show qos "$BROAD_PHENOTYPE_QOS" format=Name,MaxWall 2>/dev/null \
    | awk -F '|' -v qos="$BROAD_PHENOTYPE_QOS" '$1==qos {print; n++} END {if(n!=1) exit 2}')" || {
    echo "Unable to resolve exactly one Slurm QOS named $BROAD_PHENOTYPE_QOS" >&2
    exit 2
  }
  max_wall="$(awk -F '|' '{print $2}' <<< "$qos_row")"
  if [[ -z "$max_wall" ]]; then
    QOS_MAX_WALL="unlimited"
    return 0
  fi
  max_seconds="$(slurm_time_to_seconds "$max_wall")" || {
    echo "Unsupported MaxWall value for QOS $BROAD_PHENOTYPE_QOS: $max_wall" >&2
    exit 2
  }
  QOS_MAX_WALL="$max_wall"
  for pair in \
    "preflight=$PREFLIGHT_TIME" "feature=$FEATURE_TIME" "adapter=$ADAPTER_TIME" \
    "validate=$VALIDATE_TIME" "umap=$UMAP_TIME" "annotate=$ANNOTATE_TIME" \
    "resume=$RESUME_TIME" "review=$REVIEW_TIME" "predict=$PREDICT_TIME" \
    "model_acceptance=$MODEL_ACCEPTANCE_TIME"; do
    stage="${pair%%=*}"
    requested="${pair#*=}"
    requested_seconds="$(slurm_time_to_seconds "$requested")" || {
      echo "Unsupported Slurm wall-time for $stage: $requested" >&2
      exit 2
    }
    (( requested_seconds <= max_seconds )) || {
      echo "Requested wall-time exceeds QOS MaxWall before any job was submitted: stage=$stage requested=$requested qos=$BROAD_PHENOTYPE_QOS max_wall=$max_wall" >&2
      exit 2
    }
  done
}
validate_qos_walltime_limits

SBATCH_BIN=""
if [[ "$EXECUTION_MODE" == "slurm" && "$DRY_RUN_SUBMIT" == "0" ]]; then
  SBATCH_BIN="$(command -v sbatch || true)"
  [[ "$SBATCH_BIN" == /* && -x "$SBATCH_BIN" ]] || {
    echo "An absolute executable sbatch is required before creating the broad-phenotype shadow root" >&2
    exit 2
  }
fi

SOURCE_FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR"
SOURCE_FIELD_MANIFEST_FILE="$SOURCE_FIELD_MANIFEST_DIR/field_manifest.tsv"
RESOLVED_FIELD_MANIFEST_DIR="$SHADOW_ROOT/workflow_status/resolved_stage06_manifest"
[[ -s "$SOURCE_FIELD_MANIFEST_FILE" ]] || { echo "Stage 06 field manifest is unavailable: $SOURCE_FIELD_MANIFEST_FILE" >&2; exit 2; }
TASK_LIST="$SHADOW_ROOT/workflow_status/field_tasks.txt"

configure_workers() {
  local worker_root="$1" worker
  FEATURE_WORKER="$worker_root/run_broad_phenotype_feature_array_task.sh"
  PREFLIGHT_WORKER="$worker_root/run_broad_phenotype_input_preflight.sh"
  PREDICT_WORKER="$worker_root/run_broad_phenotype_predict_array_task.sh"
  MODEL_ACCEPTANCE_WORKER="$worker_root/run_broad_phenotype_model_acceptance.sh"
  ADAPTER_WORKER="$worker_root/run_broad_phenotype_cpa_adapter.sh"
  STAGE_WORKER="$worker_root/run_broad_phenotype_cpa_stage.sh"
  FINALIZE_WORKER="$worker_root/run_broad_phenotype_finalize.sh"
  for worker in "$PREFLIGHT_WORKER" "$FEATURE_WORKER" "$PREDICT_WORKER" "$MODEL_ACCEPTANCE_WORKER" "$ADAPTER_WORKER" "$STAGE_WORKER" "$FINALIZE_WORKER"; do
    [[ -x "$worker" ]] || { echo "Required SIF-backed worker is unavailable: $worker" >&2; return 2; }
  done
}

submission_staging_dir=""
bootstrap_cleanup_armed=0
bootstrap_marker=""
cleanup_submission_staging() {
  if [[ -n "$submission_staging_dir" && -d "$submission_staging_dir" ]]; then
    rm -f -- "$submission_staging_dir/field_tasks.txt" "$submission_staging_dir/hpc_container_identity.json" "$submission_staging_dir/code_snapshot.tar"
    if [[ -d "$submission_staging_dir/snapshot_compare" ]]; then
      rm -rf -- "$submission_staging_dir/snapshot_compare"
    fi
    rmdir -- "$submission_staging_dir" 2>/dev/null || true
  fi
}
submission_exit_cleanup() {
  local status=$?
  set +e
  if [[ -n "${preflight_tmp:-}" ]]; then
    rm -f -- "$preflight_tmp"
  fi
  cleanup_submission_staging
  if [[ "$status" -ne 0 && "$bootstrap_cleanup_armed" == "1" && -n "$bootstrap_marker" && -f "$bootstrap_marker" ]]; then
    local resolved_parent expected_parent marker_value
    resolved_parent="$(cd "$(dirname "$SHADOW_ROOT")" 2>/dev/null && pwd -P)"
    if [[ "$EXECUTION_SCOPE" == "formal_full" ]]; then
      expected_parent="$(cd "$RESULTS_ROOT" && pwd -P)"
      [[ "$resolved_parent" == "$expected_parent" && "$(basename "$SHADOW_ROOT")" == broad_phenotype_shadow_* ]] || return "$status"
    else
      expected_parent="$(cd "$RESULTS_ROOT/Tests_and_Parameters_calibration" && pwd -P)"
      [[ "$resolved_parent" == "$expected_parent" && "$(basename "$SHADOW_ROOT")" == broad_phenotype_shadow_test_* ]] || return "$status"
    fi
    marker_value="$(<"$bootstrap_marker")"
    if [[ "$marker_value" == "pid=$$" ]]; then
      rm -rf -- "$SHADOW_ROOT"
    fi
  fi
  return "$status"
}
disarm_bootstrap_cleanup() {
  if [[ "$bootstrap_cleanup_armed" == "1" ]]; then
    rm -f -- "$bootstrap_marker"
    bootstrap_cleanup_armed=0
    bootstrap_marker=""
  fi
}
code_snapshot_receipt_value() {
  local property="$1" value
  value="$(awk -F '\t' -v property="$property" 'NR>1 && $1==property && $2!="" {print $2; n++} END {if(n!=1) exit 2}' "$CODE_SNAPSHOT_RECEIPT")" || {
    echo "Code snapshot receipt must contain exactly one nonblank $property" >&2
    return 2
  }
  printf '%s\n' "$value"
}
verify_code_snapshot() {
  local comparison_root="$1" expected_archive_sha observed_archive_sha expected_root
  [[ -d "$CODE_SNAPSHOT_ROOT" && -s "$CODE_SNAPSHOT_ARCHIVE" && -s "$CODE_SNAPSHOT_RECEIPT" ]] || {
    echo "Code snapshot generation is incomplete" >&2
    return 2
  }
  [[ "$(head -n 1 "$CODE_SNAPSHOT_RECEIPT")" == $'property\tvalue' ]] || {
    echo "Code snapshot receipt header changed" >&2
    return 2
  }
  [[ "$(code_snapshot_receipt_value schema_version)" == "broad_phenotype_code_snapshot_v1" ]] || {
    echo "Code snapshot receipt schema changed" >&2
    return 2
  }
  expected_root="$(code_snapshot_receipt_value snapshot_root)"
  [[ "$expected_root" == "$CODE_SNAPSHOT_ROOT" ]] || {
    echo "Code snapshot root differs from its receipt" >&2
    return 2
  }
  expected_archive_sha="$(code_snapshot_receipt_value archive_sha256)"
  observed_archive_sha="$(sha256sum "$CODE_SNAPSHOT_ARCHIVE")"; observed_archive_sha="${observed_archive_sha%%[[:space:]]*}"
  [[ "$expected_archive_sha" == "$observed_archive_sha" ]] || {
    echo "Code snapshot archive hash differs from its receipt" >&2
    return 2
  }
  [[ ! -e "$comparison_root" ]] || {
    echo "Code snapshot comparison directory already exists: $comparison_root" >&2
    return 2
  }
  mkdir "$comparison_root"
  if ! tar -xf "$CODE_SNAPSHOT_ARCHIVE" -C "$comparison_root"; then
    rm -rf -- "$comparison_root"
    echo "Unable to extract frozen code snapshot archive" >&2
    return 2
  fi
  if ! diff -qr "$comparison_root" "$CODE_SNAPSHOT_ROOT" >/dev/null; then
    rm -rf -- "$comparison_root"
    echo "Extracted code snapshot differs from its frozen archive" >&2
    return 2
  fi
  rm -rf -- "$comparison_root"
}
if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  submission_staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/broad-phenotype-submit.XXXXXX")"
  trap submission_exit_cleanup EXIT
  task_list_for_validation="$submission_staging_dir/field_tasks.txt"
  if [[ -n "${TASK_LIST_INPUT:-}" ]]; then
    [[ -s "$TASK_LIST_INPUT" ]] || { echo "TASK_LIST_INPUT is unavailable: $TASK_LIST_INPUT" >&2; exit 2; }
    cp "$TASK_LIST_INPUT" "$task_list_for_validation"
  else
    awk -F '\t' 'NR>1 && $1!="" {print $1}' "$SOURCE_FIELD_MANIFEST_FILE" > "$task_list_for_validation"
  fi
else
  task_list_for_validation="$TASK_LIST"
  [[ -s "$task_list_for_validation" ]] || { echo "Frozen phase-A task list is unavailable: $task_list_for_validation" >&2; exit 2; }
fi
N_TASKS="$(awk 'NF {n++} END {print n+0}' "$task_list_for_validation")"
[[ "$N_TASKS" -eq "$EXPECTED_FIELDS" ]] || {
  echo "Field universe mismatch: expected=$EXPECTED_FIELDS observed=$N_TASKS task_list=$task_list_for_validation" >&2
  exit 2
}
awk 'NF {if($0 !~ /^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$/) exit 2; if(seen[$0]++) exit 3}' "$task_list_for_validation" || {
  echo "Canonical field task list has an invalid or duplicate key: $task_list_for_validation" >&2
  exit 2
}

task_list_sha256="$(sha256sum "$task_list_for_validation")"; task_list_sha256="${task_list_sha256%%[[:space:]]*}"
source_manifest_sha256="$(sha256sum "$SOURCE_FIELD_MANIFEST_FILE")"; source_manifest_sha256="${source_manifest_sha256%%[[:space:]]*}"
feature_config_sha256="$(sha256sum "$FEATURE_CONFIG")"; feature_config_sha256="${feature_config_sha256%%[[:space:]]*}"
classes_sha256="$(sha256sum "$CLASSES_FILE")"; classes_sha256="${classes_sha256%%[[:space:]]*}"
plate_map_sha256="$(sha256sum "$PLATE_MAP")"; plate_map_sha256="${plate_map_sha256%%[[:space:]]*}"
dependency_lock_sha256="$(sha256sum "$DEPENDENCY_LOCK")"; dependency_lock_sha256="${dependency_lock_sha256%%[[:space:]]*}"
legacy_no_go_sha256="$(sha256sum "$LEGACY_NO_GO")"; legacy_no_go_sha256="${legacy_no_go_sha256%%[[:space:]]*}"
if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  project_git_sha="$(git -C "$SOURCE_PROJECT_DIR" rev-parse HEAD)"
  project_git_tree="$(git -C "$SOURCE_PROJECT_DIR" rev-parse "${project_git_sha}^{tree}")"
  project_git_status="$(git -C "$SOURCE_PROJECT_DIR" status --porcelain --untracked-files=all)"
  [[ -z "$project_git_status" ]] || {
    echo "SOURCE_PROJECT_DIR must be a clean immutable checkout before broad-phenotype submission" >&2
    exit 2
  }
  snapshot_archive_staging="$submission_staging_dir/code_snapshot.tar"
  git -C "$SOURCE_PROJECT_DIR" archive --format=tar --output "$snapshot_archive_staging" "$project_git_sha"
  [[ "$(git -C "$SOURCE_PROJECT_DIR" rev-parse HEAD)" == "$project_git_sha" \
     && -z "$(git -C "$SOURCE_PROJECT_DIR" status --porcelain --untracked-files=all)" ]] || {
    echo "SOURCE_PROJECT_DIR changed while the code snapshot was being created" >&2
    exit 2
  }
  code_snapshot_archive_sha256="$(sha256sum "$snapshot_archive_staging")"; code_snapshot_archive_sha256="${code_snapshot_archive_sha256%%[[:space:]]*}"
else
  resume_snapshot_comparison="${TMPDIR:-/tmp}/broad-phenotype-resume-snapshot.$$"
  verify_code_snapshot "$resume_snapshot_comparison"
  project_git_sha="$(code_snapshot_receipt_value project_git_sha)"
  project_git_tree="$(code_snapshot_receipt_value project_git_tree)"
  project_git_status="clean_snapshot"
  code_snapshot_archive_sha256="$(code_snapshot_receipt_value archive_sha256)"
  SOURCE_PROJECT_DIR="$(code_snapshot_receipt_value source_project_dir)"
  snapshot_submitter="$CODE_SNAPSHOT_ROOT/cellpose_pipeline/Docker/hpc/submit_broad_phenotype_shadow_full.sh"
  [[ -f "$snapshot_submitter" ]] || {
    echo "Frozen code snapshot does not contain its submitter: $snapshot_submitter" >&2
    exit 2
  }
  cmp -s "$SCRIPT_DIR/submit_broad_phenotype_shadow_full.sh" "$snapshot_submitter" || {
    echo "Resume must be orchestrated by the exact submitter frozen in the code snapshot: $snapshot_submitter" >&2
    exit 2
  }
fi

HPC_CONTAINER_IDENTITY_FILE="$SHADOW_ROOT/workflow_status/hpc_container_identity.json"
if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  identity_capture_file="$submission_staging_dir/hpc_container_identity.json"
  broad_phenotype_capture_container_identity \
    "$HPC_CONTAINER_IMAGE" \
    "$EXPECTED_HPC_CONTAINER_SHA256" \
    "$identity_capture_file"
else
  identity_capture_file="$HPC_CONTAINER_IDENTITY_FILE"
  [[ -s "$HPC_CONTAINER_IDENTITY_FILE" ]] || {
    echo "Frozen HPC container identity file is unavailable: $HPC_CONTAINER_IDENTITY_FILE" >&2
    exit 2
  }
fi
HPC_CONTAINER_IDENTITY_FILE_SHA256="$(sha256sum "$identity_capture_file")"
HPC_CONTAINER_IDENTITY_FILE_SHA256="${HPC_CONTAINER_IDENTITY_FILE_SHA256%%[[:space:]]*}"
broad_phenotype_verify_container_identity \
  "$HPC_CONTAINER_IMAGE" \
  "$EXPECTED_HPC_CONTAINER_SHA256" \
  "$identity_capture_file" \
  "$HPC_CONTAINER_IDENTITY_FILE_SHA256"
observed_sif_sha256="$BROAD_PHENOTYPE_CONTAINER_SHA256"
HPC_CONTAINER_VERIFIED_SHA256="$observed_sif_sha256"

if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  mkdir "$SHADOW_ROOT"
  bootstrap_marker="$SHADOW_ROOT/.broad_phenotype_bootstrap_owned.$$"
  printf 'pid=%s\n' "$$" > "$bootstrap_marker"
  bootstrap_cleanup_armed=1
  mkdir -p "$SHADOW_ROOT/workflow_status" "$SHADOW_ROOT/logs"
  mv "$task_list_for_validation" "$TASK_LIST"
  mv "$identity_capture_file" "$HPC_CONTAINER_IDENTITY_FILE"
  mv "$snapshot_archive_staging" "$CODE_SNAPSHOT_ARCHIVE"
  mkdir "$CODE_SNAPSHOT_ROOT"
  tar -xf "$CODE_SNAPSHOT_ARCHIVE" -C "$CODE_SNAPSHOT_ROOT"
  code_snapshot_receipt_tmp="$SHADOW_ROOT/workflow_status/.code_snapshot_receipt.tmp.$$"
  sed $'s/\\\\t/\t/g' > "$code_snapshot_receipt_tmp" <<EOF
property\tvalue
schema_version\tbroad_phenotype_code_snapshot_v1
source_project_dir\t$SOURCE_PROJECT_DIR
project_git_sha\t$project_git_sha
project_git_tree\t$project_git_tree
snapshot_root\t$CODE_SNAPSHOT_ROOT
archive_path\t$CODE_SNAPSHOT_ARCHIVE
archive_sha256\t$code_snapshot_archive_sha256
EOF
  mv "$code_snapshot_receipt_tmp" "$CODE_SNAPSHOT_RECEIPT"
  verify_code_snapshot "$submission_staging_dir/snapshot_compare"
  PROJECT_DIR="$CODE_SNAPSHOT_ROOT"
  DEPENDENCY_LOCK="$PROJECT_DIR/cellpose_pipeline/configs/cellphenotypeannotator_dependency.lock.tsv"
  FEATURE_CONFIG="$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_features_v1.json"
  CLASSES_FILE="$PROJECT_DIR/cellpose_pipeline/configs/broad_phenotype_classes_v1.tsv"
  PLATE_MAP="$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv"
  [[ "$(sha256sum "$DEPENDENCY_LOCK" | awk '{print $1}')" == "$dependency_lock_sha256" \
     && "$(sha256sum "$FEATURE_CONFIG" | awk '{print $1}')" == "$feature_config_sha256" \
     && "$(sha256sum "$CLASSES_FILE" | awk '{print $1}')" == "$classes_sha256" \
     && "$(sha256sum "$PLATE_MAP" | awk '{print $1}')" == "$plate_map_sha256" ]] || {
    echo "Tracked locked inputs differ inside the code snapshot" >&2
    exit 2
  }
  rmdir -- "$submission_staging_dir"
  submission_staging_dir=""
  broad_phenotype_verify_container_identity \
    "$HPC_CONTAINER_IMAGE" \
    "$EXPECTED_HPC_CONTAINER_SHA256" \
    "$HPC_CONTAINER_IDENTITY_FILE" \
    "$HPC_CONTAINER_IDENTITY_FILE_SHA256"
else
  mkdir -p "$SHADOW_ROOT/workflow_status" "$SHADOW_ROOT/logs"
fi

configure_workers "$PROJECT_DIR/cellpose_pipeline/Docker/hpc"
HPC_CONTAINER_RUNTIME_ROOT="$PROJECT_DIR/cellpose_pipeline/Docker/hpc"
HPC_CONTAINER_FORWARD_PREFIXES="$FIXED_HPC_CONTAINER_FORWARD_PREFIXES"

export PROJECT_DIR HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_CONTAINER_VERIFIED_SHA256
export HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_RUNTIME_ROOT HPC_CONTAINER_FORWARD_PREFIXES
export HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256
export CODE_SNAPSHOT_ROOT CODE_SNAPSHOT_ARCHIVE CODE_SNAPSHOT_RECEIPT
export CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT DEPENDENCY_LOCK FEATURE_CONFIG CLASSES_FILE PLATE_MAP
export SHADOW_ROOT DATASET_ROOT FIELD_MANIFEST_DIR SOURCE_SEGMENTATION_ROOT CLASSIFICATION_ROOT LEGACY_NO_GO BRANCH INCLUDE_NUCLEI_COMPARATOR MAX_UMAP_FIELDS_PER_WELL
export SPLIT_SEED OUTER_FOLDS INNER_FOLDS BROAD_PHENOTYPE_PROJECT_ID MAX_UMAP_CELLS_PER_FIELD MAX_UMAP_CELLS_PER_WELL UMAP_SAMPLE_SEED
export FORCE_FEATURES FORCE_PREDICT FORCE_FINALIZE LEGACY_PATH_PREFIX LIVE_PATH_PREFIX
export STALE_DATASET_ROOT STALE_SEGMENTATION_ROOT
HPC_PROJECT_ROOT="$PROJECT_DIR"
export HPC_PROJECT_ROOT
if [[ "$RESUME_STAGE" != "phase-a" ]]; then
  base_preflight="$SHADOW_ROOT/workflow_status/submission_preflight.tsv"
  [[ -s "$base_preflight" ]] || { echo "Frozen phase-A submission preflight is unavailable: $base_preflight" >&2; exit 2; }
  frozen_value() {
    local property="$1"
    local value
    value="$(awk -F '\t' -v property="$property" '$1==property && $2!="" {print $2; n++} END {if(n!=1) exit 2}' "$base_preflight")" || {
      echo "Frozen phase-A preflight must contain exactly one nonblank $property" >&2
      exit 2
    }
    printf '%s\n' "$value"
  }
  compare_frozen() {
    local property="$1" observed="$2" expected
    expected="$(frozen_value "$property")"
    [[ "$observed" == "$expected" ]] || {
      echo "Resume provenance drift for $property: frozen=$expected observed=$observed" >&2
      exit 2
    }
  }
  compare_frozen canonical_task_list_sha256 "$task_list_sha256"
  compare_frozen source_field_manifest_sha256 "$source_manifest_sha256"
  compare_frozen source_segmentation_root "$SOURCE_SEGMENTATION_ROOT"
  compare_frozen results_root "$RESULTS_ROOT"
  compare_frozen dataset_root "$DATASET_ROOT"
  compare_frozen source_field_manifest_dir "$SOURCE_FIELD_MANIFEST_DIR"
  compare_frozen resolved_field_manifest_dir "$RESOLVED_FIELD_MANIFEST_DIR"
  compare_frozen classification_root "$CLASSIFICATION_ROOT"
  compare_frozen source_field_manifest_file "$SOURCE_FIELD_MANIFEST_FILE"
  compare_frozen stale_dataset_root "$STALE_DATASET_ROOT"
  compare_frozen stale_segmentation_root "$STALE_SEGMENTATION_ROOT"
  compare_frozen feature_config_sha256 "$feature_config_sha256"
  compare_frozen classes_sha256 "$classes_sha256"
  compare_frozen plate_map_sha256 "$plate_map_sha256"
  compare_frozen dependency_lock_sha256 "$dependency_lock_sha256"
  compare_frozen project_git_sha "$project_git_sha"
  compare_frozen project_git_tree "$project_git_tree"
  compare_frozen project_dir "$PROJECT_DIR"
  compare_frozen source_project_dir "$SOURCE_PROJECT_DIR"
  compare_frozen code_snapshot_root "$CODE_SNAPSHOT_ROOT"
  compare_frozen code_snapshot_archive_sha256 "$code_snapshot_archive_sha256"
  compare_frozen hpc_container_sha256 "$observed_sif_sha256"
  compare_frozen hpc_container_identity_file "$HPC_CONTAINER_IDENTITY_FILE"
  compare_frozen hpc_container_identity_file_sha256 "$HPC_CONTAINER_IDENTITY_FILE_SHA256"
  compare_frozen cpa_reference_commit "$observed_commit"
  compare_frozen cpa_reference_tree "$observed_tree"
  compare_frozen legacy_no_go_sha256 "$legacy_no_go_sha256"
  compare_frozen branch "$BRANCH"
  compare_frozen include_nuclei_comparator "$INCLUDE_NUCLEI_COMPARATOR"
  compare_frozen max_umap_fields_per_well "$MAX_UMAP_FIELDS_PER_WELL"
  compare_frozen split_seed "$SPLIT_SEED"
  compare_frozen outer_folds "$OUTER_FOLDS"
  compare_frozen inner_folds "$INNER_FOLDS"
  compare_frozen broad_phenotype_project_id "$BROAD_PHENOTYPE_PROJECT_ID"
  compare_frozen max_umap_cells_per_field "$MAX_UMAP_CELLS_PER_FIELD"
  compare_frozen max_umap_cells_per_well "$MAX_UMAP_CELLS_PER_WELL"
  compare_frozen umap_sample_seed "$UMAP_SAMPLE_SEED"
  compare_frozen force_features "$FORCE_FEATURES"
  compare_frozen force_predict "$FORCE_PREDICT"
  compare_frozen force_finalize "$FORCE_FINALIZE"
  compare_frozen execution_scope "$EXECUTION_SCOPE"
  compare_frozen split_mode "$SPLIT_MODE"
  compare_frozen heldout_wells "$FROZEN_HELDOUT_WELLS"
  compare_frozen broad_phenotype_qos "${BROAD_PHENOTYPE_QOS:-none}"
  compare_frozen qos_max_wall "$QOS_MAX_WALL"
fi
export TASK_LIST

export RESOLVED_FIELD_MANIFEST_DIR SOURCE_FIELD_MANIFEST_FILE
if [[ "$RESUME_STAGE" != "phase-a" && "$RESUME_STAGE" != "input-preflight" && ( ! -s "$RESOLVED_FIELD_MANIFEST_DIR/field_manifest.tsv" || ! -s "$RESOLVED_FIELD_MANIFEST_DIR/path_resolution_manifest.tsv" ) ]]; then
  echo "Resolved Stage 06 manifest is incomplete: $RESOLVED_FIELD_MANIFEST_DIR" >&2
  exit 2
fi
FIELD_MANIFEST_DIR="$RESOLVED_FIELD_MANIFEST_DIR"
FIELD_MANIFEST_FILE="$RESOLVED_FIELD_MANIFEST_DIR/field_manifest.tsv"
FEATURE_ROOT="$SHADOW_ROOT/features/$BRANCH"
PROJECT_FILE="${PROJECT_FILE:-$SHADOW_ROOT/projection_input/representative_umap/project.yml}"
export SOURCE_FIELD_MANIFEST_DIR FIELD_MANIFEST_DIR FIELD_MANIFEST_FILE FEATURE_ROOT PROJECT_FILE

if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  preflight_receipt="$SHADOW_ROOT/workflow_status/submission_preflight.tsv"
  [[ ! -e "$preflight_receipt" ]] || { echo "Phase-A preflight receipt already exists: $preflight_receipt" >&2; exit 2; }
else
  preflight_receipt="$SHADOW_ROOT/workflow_status/resume_preflight_${RESUME_STAGE}_${RUN_STAMP}.tsv"
  [[ ! -e "$preflight_receipt" ]] || { echo "Resume preflight receipt already exists: $preflight_receipt" >&2; exit 2; }
fi
preflight_tmp="${preflight_receipt}.tmp.$$"
sed $'s/\\\\t/\t/g' > "$preflight_tmp" <<EOF
property\tvalue
execution_mode\t$EXECUTION_MODE
execution_scope\t$EXECUTION_SCOPE
split_mode\t$SPLIT_MODE
heldout_wells\t$FROZEN_HELDOUT_WELLS
cpa_stage_mode\t${CPA_STAGE_MODE:-}
resume_stage\t$RESUME_STAGE
broad_phenotype_qos\t${BROAD_PHENOTYPE_QOS:-none}
qos_max_wall\t$QOS_MAX_WALL
project_dir\t$PROJECT_DIR
source_project_dir\t$SOURCE_PROJECT_DIR
code_snapshot_root\t$CODE_SNAPSHOT_ROOT
code_snapshot_archive\t$CODE_SNAPSHOT_ARCHIVE
code_snapshot_archive_sha256\t$code_snapshot_archive_sha256
results_root\t$RESULTS_ROOT
dataset_root\t$DATASET_ROOT
source_segmentation_root\t$SOURCE_SEGMENTATION_ROOT
stale_dataset_root\t$STALE_DATASET_ROOT
stale_segmentation_root\t$STALE_SEGMENTATION_ROOT
shadow_root\t$SHADOW_ROOT
source_field_manifest_dir\t$SOURCE_FIELD_MANIFEST_DIR
resolved_field_manifest_dir\t$FIELD_MANIFEST_DIR
classification_root\t$CLASSIFICATION_ROOT
task_count\t$N_TASKS
canonical_task_list\t$TASK_LIST
canonical_task_list_sha256\t$task_list_sha256
source_field_manifest_file\t$SOURCE_FIELD_MANIFEST_FILE
source_field_manifest_sha256\t$source_manifest_sha256
feature_config_sha256\t$feature_config_sha256
classes_sha256\t$classes_sha256
plate_map\t$PLATE_MAP
plate_map_sha256\t$plate_map_sha256
dependency_lock_sha256\t$dependency_lock_sha256
project_git_sha\t$project_git_sha
project_git_tree\t$project_git_tree
project_git_status\tclean
branch\t$BRANCH
include_nuclei_comparator\t$INCLUDE_NUCLEI_COMPARATOR
max_umap_fields_per_well\t$MAX_UMAP_FIELDS_PER_WELL
split_seed\t$SPLIT_SEED
outer_folds\t$OUTER_FOLDS
inner_folds\t$INNER_FOLDS
broad_phenotype_project_id\t$BROAD_PHENOTYPE_PROJECT_ID
max_umap_cells_per_field\t$MAX_UMAP_CELLS_PER_FIELD
max_umap_cells_per_well\t$MAX_UMAP_CELLS_PER_WELL
umap_sample_seed\t$UMAP_SAMPLE_SEED
force_features\t$FORCE_FEATURES
force_predict\t$FORCE_PREDICT
force_finalize\t$FORCE_FINALIZE
hpc_container_image\t$HPC_CONTAINER_IMAGE
hpc_container_sha256\t$observed_sif_sha256
hpc_container_identity_file\t$HPC_CONTAINER_IDENTITY_FILE
hpc_container_identity_file_sha256\t$HPC_CONTAINER_IDENTITY_FILE_SHA256
hpc_container_gpu\t0
cpa_reference_root\t$CPA_REFERENCE_ROOT
cpa_reference_commit\t$observed_commit
cpa_reference_tree\t$observed_tree
cpa_reference_bind_mode\tread_only
legacy_no_go\t$LEGACY_NO_GO
legacy_no_go_sha256\t$legacy_no_go_sha256
legacy_no_go_enforcement\tinformational_only
path_resolution_ambiguity_count\t$([[ "$RESUME_STAGE" == "phase-a" || "$RESUME_STAGE" == "input-preflight" ]] && echo pending_j0 || echo 0)
EOF
mv "$preflight_tmp" "$preflight_receipt"
preflight_tmp=""

echo "runtime=apptainer_sif"
echo "hpc_container_image=$HPC_CONTAINER_IMAGE"
echo "hpc_container_sha256=$observed_sif_sha256"
echo "hpc_container_identity_file=$HPC_CONTAINER_IDENTITY_FILE"
echo "hpc_container_identity_file_sha256=$HPC_CONTAINER_IDENTITY_FILE_SHA256"
echo "hpc_container_gpu=0"
echo "shadow_root=$SHADOW_ROOT"
echo "source_field_manifest_dir=$SOURCE_FIELD_MANIFEST_DIR"
echo "dataset_root=$DATASET_ROOT"
echo "source_segmentation_root=$SOURCE_SEGMENTATION_ROOT"
echo "resolved_field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "legacy_no_go_enforcement=informational_only"
echo "cpa_reference_root=$CPA_REFERENCE_ROOT"
echo "cpa_reference_bind_mode=read_only"
echo "task_count=$N_TASKS"

direct_concurrency="${DIRECT_TEST_CONCURRENCY:-4}"
if [[ "$EXECUTION_MODE" == "direct_test" ]]; then
  [[ "$direct_concurrency" =~ ^[1-9][0-9]*$ && "$direct_concurrency" -le 32 ]] || {
    echo "DIRECT_TEST_CONCURRENCY must be an integer in [1, 32]: $direct_concurrency" >&2
    exit 2
  }
elif [[ -n "${DEPENDENCY_JOB_ID:-}" ]]; then
  [[ "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "DEPENDENCY_JOB_ID must be numeric" >&2; exit 2; }
fi

run_direct_phase_a() {
  "$PREFLIGHT_WORKER"
  local feature_cpus="${FEATURE_CPUS:-2}"
  echo "direct_test_concurrency=$direct_concurrency"
  echo "direct_test_feature_cpus_per_worker=$feature_cpus"
  local -a indices=()
  local index
  for ((index=1; index<=N_TASKS; index++)); do
    indices+=("$index")
  done
  printf '%s\n' "${indices[@]}" | xargs -n 1 -P "$direct_concurrency" bash -c '
    feature_cpus="$1"
    feature_worker="$2"
    array_index="$3"
    SLURM_ARRAY_TASK_ID="$array_index" SLURM_CPUS_PER_TASK="$feature_cpus" "$feature_worker"
  ' _ "$feature_cpus" "$FEATURE_WORKER"
  SLURM_CPUS_PER_TASK="${ADAPTER_CPUS:-8}" "$ADAPTER_WORKER"
  local stage
  for stage in validate umap annotate; do
    if [[ "$stage" == "validate" ]]; then
      CPA_STAGE="$stage" CPA_VALIDATE_STAGE=umap CPA_STAGE_MODE="" "$STAGE_WORKER"
    else
      CPA_STAGE="$stage" CPA_STAGE_MODE="" "$STAGE_WORKER"
    fi
  done
  echo "phase_a_complete=1"
  echo "human_barrier=annotation_region_submission_required"
}

if [[ "$EXECUTION_MODE" == "direct_test" ]]; then
  [[ "$RESUME_STAGE" == "phase-a" ]] || { echo "direct_test supports phase-a only" >&2; exit 2; }
  disarm_bootstrap_cleanup
  run_direct_phase_a
  exit 0
fi

cpu_sbatch() {
  /usr/bin/env -i PATH=/usr/bin:/bin "$SBATCH_BIN" --parsable "$@"
}
submit_job() {
  if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
    printf 'DRYRUN_%s\n' "$1"
    return 0
  fi
  local label="$1"; shift
  local output
  output="$(cpu_sbatch "$@")" || { echo "Slurm submission failed for $label" >&2; return 1; }
  output="${output%%;*}"
  [[ "$output" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID for $label: $output" >&2; return 1; }
  printf '%s\n' "$output"
}
worker_wrap_command() {
  local worker="$1"
  case "$worker" in
    "$CODE_SNAPSHOT_ROOT"/*) ;;
    *) echo "Slurm worker must reside in the frozen code snapshot: $worker" >&2; return 2 ;;
  esac
  [[ -x "$worker" ]] || { echo "Slurm worker is not executable: $worker" >&2; return 2; }
  printf 'export PATH=/usr/bin:/bin; exec %q %q' "$SYSTEM_BASH_BIN" "$worker"
}
slurm_export_spec() {
  local extra_names="${1:-}"
  local common_names="PROJECT_DIR HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES HPC_CONTAINER_RUNTIME_ROOT HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256 HPC_PROJECT_ROOT CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT DEPENDENCY_LOCK FEATURE_CONFIG CLASSES_FILE PLATE_MAP SHADOW_ROOT DATASET_ROOT SOURCE_SEGMENTATION_ROOT CLASSIFICATION_ROOT LEGACY_NO_GO BRANCH INCLUDE_NUCLEI_COMPARATOR MAX_UMAP_FIELDS_PER_WELL SPLIT_SEED OUTER_FOLDS INNER_FOLDS BROAD_PHENOTYPE_PROJECT_ID MAX_UMAP_CELLS_PER_FIELD MAX_UMAP_CELLS_PER_WELL UMAP_SAMPLE_SEED STALE_DATASET_ROOT STALE_SEGMENTATION_ROOT LEGACY_PATH_PREFIX LIVE_PATH_PREFIX TASK_LIST SOURCE_FIELD_MANIFEST_FILE SOURCE_FIELD_MANIFEST_DIR FIELD_MANIFEST_DIR FIELD_MANIFEST_FILE FEATURE_ROOT PROJECT_FILE FORCE_FEATURES FORCE_PREDICT FORCE_FINALIZE"
  local name value spec=""
  for name in $common_names $extra_names; do
    [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || {
      echo "Invalid Slurm export variable name: $name" >&2
      return 2
    }
    [[ "$name" != "ALL" && "$name" != "NONE" ]] || {
      echo "Slurm export specification must not contain ALL or NONE" >&2
      return 2
    }
    [[ -n "${!name+x}" ]] || {
      echo "Slurm export variable is not defined: $name" >&2
      return 2
    }
    value="${!name-}"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
      echo "Slurm export value contains a forbidden newline: $name" >&2
      return 2
    }
    spec="${spec:+$spec,}${name}"
  done
  printf '%s\n' "$spec"
}
dependency_args=()
if [[ -n "${DEPENDENCY_JOB_ID:-}" ]]; then
  dependency_args+=(--dependency "afterok:$DEPENDENCY_JOB_ID")
fi
base_args=(--chdir "$PROJECT_DIR" --output "$SHADOW_ROOT/logs/%x.%A_%a.out" --error "$SHADOW_ROOT/logs/%x.%A_%a.err")
if [[ -n "$BROAD_PHENOTYPE_QOS" ]]; then
  base_args+=(--qos "$BROAD_PHENOTYPE_QOS")
fi

if [[ "$RESUME_STAGE" == "phase-a" ]]; then
  phase_a_dag="$SHADOW_ROOT/workflow_status/slurm_dag.tsv"
  [[ ! -e "$phase_a_dag" ]] || { echo "Immutable phase-A DAG receipt already exists: $phase_a_dag" >&2; exit 2; }
  preflight_export="$(slurm_export_spec "RESOLVED_FIELD_MANIFEST_DIR")"
  feature_export="$(slurm_export_spec)"
  adapter_export="$(slurm_export_spec)"
  preflight_wrap="$(worker_wrap_command "$PREFLIGHT_WORKER")"
  feature_wrap="$(worker_wrap_command "$FEATURE_WORKER")"
  adapter_wrap="$(worker_wrap_command "$ADAPTER_WORKER")"
  stage_wrap="$(worker_wrap_command "$STAGE_WORKER")"
  CPA_STAGE=validate CPA_VALIDATE_STAGE=umap CPA_STAGE_MODE=; export CPA_STAGE CPA_VALIDATE_STAGE CPA_STAGE_MODE
  validate_export="$(slurm_export_spec "CPA_STAGE CPA_VALIDATE_STAGE CPA_STAGE_MODE")"
  umap_export="$(slurm_export_spec "CPA_STAGE CPA_STAGE_MODE")"
  annotate_export="$umap_export"
  printf 'stage\tjob_id\tdependency\n' > "$phase_a_dag"
  disarm_bootstrap_cleanup
  preflight_job="$(submit_job input-preflight --job-name bp_input_preflight --cpus-per-task "$PREFLIGHT_CPUS" --mem "$PREFLIGHT_MEM" --time "$PREFLIGHT_TIME" "${base_args[@]}" "${dependency_args[@]}" --export="$preflight_export" --wrap="$preflight_wrap")"
  printf 'input_preflight\t%s\t%s\n' "$preflight_job" "${DEPENDENCY_JOB_ID:-none}" >> "$phase_a_dag"
  feature_job="$(submit_job feature --job-name bp_feature --cpus-per-task "$FEATURE_CPUS" --mem "$FEATURE_MEM" --time "$FEATURE_TIME" --array "1-${N_TASKS}%${FEATURE_MAX_CONCURRENT}" "${base_args[@]}" --dependency "afterok:$preflight_job" --export="$feature_export" --wrap="$feature_wrap")"
  printf 'feature\t%s\t%s\n' "$feature_job" "$preflight_job" >> "$phase_a_dag"
  adapter_job="$(submit_job adapter --job-name bp_adapter --cpus-per-task "$ADAPTER_CPUS" --mem "$ADAPTER_MEM" --time "$ADAPTER_TIME" "${base_args[@]}" --dependency "afterok:$feature_job" --export="$adapter_export" --wrap="$adapter_wrap")"
  printf 'adapter_projection\t%s\t%s\n' "$adapter_job" "$feature_job" >> "$phase_a_dag"
  CPA_STAGE=validate CPA_VALIDATE_STAGE=umap CPA_STAGE_MODE=; export CPA_STAGE CPA_VALIDATE_STAGE CPA_STAGE_MODE
  validate_job="$(submit_job validate --job-name bp_validate --cpus-per-task "$VALIDATE_CPUS" --mem "$VALIDATE_MEM" --time "$VALIDATE_TIME" "${base_args[@]}" --dependency "afterok:$adapter_job" --export="$validate_export" --wrap="$stage_wrap")"
  printf 'validate\t%s\t%s\n' "$validate_job" "$adapter_job" >> "$phase_a_dag"
  CPA_STAGE=umap CPA_VALIDATE_STAGE= CPA_STAGE_MODE=; export CPA_STAGE CPA_STAGE_MODE
  umap_job="$(submit_job umap --job-name bp_umap --cpus-per-task "$UMAP_CPUS" --mem "$UMAP_MEM" --time "$UMAP_TIME" "${base_args[@]}" --dependency "afterok:$validate_job" --export="$umap_export" --wrap="$stage_wrap")"
  printf 'umap\t%s\t%s\n' "$umap_job" "$validate_job" >> "$phase_a_dag"
  CPA_STAGE=annotate CPA_STAGE_MODE=; export CPA_STAGE CPA_STAGE_MODE
  annotate_job="$(submit_job annotate --job-name bp_annotate --cpus-per-task "$ANNOTATE_CPUS" --mem "$ANNOTATE_MEM" --time "$ANNOTATE_TIME" "${base_args[@]}" --dependency "afterok:$umap_job" --export="$annotate_export" --wrap="$stage_wrap")"
  printf 'annotate\t%s\t%s\n' "$annotate_job" "$umap_job" >> "$phase_a_dag"
  printf 'human_annotation_barrier\tnot_submitted\tannotate\n' >> "$phase_a_dag"
  echo "input_preflight_job_id=$preflight_job"
  echo "feature_job_id=$feature_job"
  echo "adapter_job_id=$adapter_job"
  echo "validate_job_id=$validate_job"
  echo "umap_job_id=$umap_job"
  echo "annotate_job_id=$annotate_job"
  echo "human_barrier=annotation_region_submission_required"
  exit 0
fi

if [[ "$RESUME_STAGE" == "predict-sharded" ]]; then
  : "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required for predict-sharded}"
  : "${CPA_TRAIN_RECEIPT:?CPA_TRAIN_RECEIPT is required for predict-sharded}"
  canonical_feature_manifest="$SHADOW_ROOT/workflow_status/feature_inventory/${BRANCH}_feature_manifest.tsv"
  canonical_prediction_root="$SHADOW_ROOT/predictions/sharded"
  [[ -z "${FEATURE_MANIFEST+x}" || "$FEATURE_MANIFEST" == "$canonical_feature_manifest" ]] || {
    echo "FEATURE_MANIFEST must be the canonical frozen inventory: $canonical_feature_manifest" >&2
    exit 2
  }
  [[ -z "${PREDICTION_ROOT+x}" || "$PREDICTION_ROOT" == "$canonical_prediction_root" ]] || {
    echo "PREDICTION_ROOT must be the canonical sharded output root: $canonical_prediction_root" >&2
    exit 2
  }
  FEATURE_MANIFEST="$canonical_feature_manifest"
  PREDICTION_ROOT="$canonical_prediction_root"
  [[ -s "$FEATURE_MANIFEST" && -d "$CPA_MODEL_DIR" && -s "$CPA_TRAIN_RECEIPT" && -s "$PROJECT_FILE" ]] || { echo "Sharded prediction feature manifest, model, train receipt, or project is unavailable" >&2; exit 2; }
  require_shadow_path() {
    local input_path="$1" label="$2" create_directory="${3:-0}"
    local resolved_shadow resolved_input
    resolved_shadow="$(cd "$SHADOW_ROOT" && pwd -P)"
    if [[ "$create_directory" == "1" ]]; then mkdir -p "$input_path"; fi
    if [[ -d "$input_path" ]]; then
      resolved_input="$(cd "$input_path" && pwd -P)"
    else
      resolved_input="$(cd "$(dirname "$input_path")" && pwd -P)/$(basename "$input_path")"
    fi
    case "$resolved_input" in "$resolved_shadow"/*) ;; *) echo "$label must resolve inside SHADOW_ROOT: $resolved_input" >&2; exit 2 ;; esac
  }
  require_shadow_path "$FEATURE_MANIFEST" FEATURE_MANIFEST
  require_shadow_path "$CPA_MODEL_DIR" CPA_MODEL_DIR
  require_shadow_path "$CPA_TRAIN_RECEIPT" CPA_TRAIN_RECEIPT
  require_shadow_path "$PROJECT_FILE" PROJECT_FILE
  require_shadow_path "$PREDICTION_ROOT" PREDICTION_ROOT 1
  acceptance_root="$SHADOW_ROOT/workflow_status/model_acceptance"
  mkdir -p "$acceptance_root"
  MODEL_ACCEPTANCE_RECEIPT="$acceptance_root/model_acceptance.json"
  MODEL_ACCEPTANCE_SHA256_FILE="$acceptance_root/model_acceptance.sha256"
  prediction_tasks="$(awk 'NR>1 && NF {n++} END {print n+0}' "$FEATURE_MANIFEST")"
  [[ "$prediction_tasks" -eq "$EXPECTED_FIELDS" ]] || { echo "Sharded prediction universe mismatch: expected=$EXPECTED_FIELDS observed=$prediction_tasks" >&2; exit 2; }
  export FEATURE_MANIFEST PREDICTION_ROOT CPA_MODEL_DIR CPA_TRAIN_RECEIPT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE
  prediction_submission="$SHADOW_ROOT/workflow_status/resume_submission_predict-sharded_${RUN_STAMP}.tsv"
  [[ ! -e "$prediction_submission" ]] || { echo "Immutable resume submission already exists: $prediction_submission" >&2; exit 2; }
  prediction_export="$(slurm_export_spec "FEATURE_MANIFEST PREDICTION_ROOT CPA_MODEL_DIR CPA_TRAIN_RECEIPT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE")"
  acceptance_job="$(submit_job model-acceptance --job-name bp_model_acceptance --cpus-per-task "$MODEL_ACCEPTANCE_CPUS" --mem "$MODEL_ACCEPTANCE_MEM" --time "$MODEL_ACCEPTANCE_TIME" "${base_args[@]}" "${dependency_args[@]}" --export="$prediction_export" --wrap="$(worker_wrap_command "$MODEL_ACCEPTANCE_WORKER")")"
  prediction_job="$(submit_job predict-sharded --job-name bp_predict_sharded --cpus-per-task "$PREDICT_CPUS" --mem "$PREDICT_MEM" --time "$PREDICT_TIME" --array "1-${prediction_tasks}%${PREDICT_MAX_CONCURRENT}" "${base_args[@]}" --dependency "afterok:$acceptance_job" --export="$prediction_export" --wrap="$(worker_wrap_command "$PREDICT_WORKER")")"
  printf 'stage\tjob_id\tdependency\nmodel-acceptance\t%s\t%s\npredict-sharded\t%s\t%s\n' "$acceptance_job" "${DEPENDENCY_JOB_ID:-none}" "$prediction_job" "$acceptance_job" > "$prediction_submission"
  echo "model_acceptance_job_id=$acceptance_job"
  echo "prediction_array_job_id=$prediction_job"
  echo "prediction_root=$PREDICTION_ROOT"
  exit 0
fi

if [[ "$RESUME_STAGE" == "input-preflight" ]]; then
  [[ ! -e "$RESOLVED_FIELD_MANIFEST_DIR" ]] || { echo "Resolved Stage 06 manifest already exists: $RESOLVED_FIELD_MANIFEST_DIR" >&2; exit 2; }
  resume_submission="$SHADOW_ROOT/workflow_status/resume_submission_input-preflight_${RUN_STAMP}.tsv"
  [[ ! -e "$resume_submission" ]] || { echo "Immutable resume submission already exists: $resume_submission" >&2; exit 2; }
  resume_job="$(submit_job input-preflight --job-name bp_input_preflight --cpus-per-task "$PREFLIGHT_CPUS" --mem "$PREFLIGHT_MEM" --time "$PREFLIGHT_TIME" "${base_args[@]}" "${dependency_args[@]}" --export="$(slurm_export_spec "RESOLVED_FIELD_MANIFEST_DIR")" --wrap="$(worker_wrap_command "$PREFLIGHT_WORKER")")"
  echo -e "stage\tjob_id\tdependency\ninput-preflight\t$resume_job\t${DEPENDENCY_JOB_ID:-none}" > "$resume_submission"
  echo "resume_stage=input-preflight"
  echo "resume_job_id=$resume_job"
  exit 0
fi

if [[ "$RESUME_STAGE" == "feature-retry" ]]; then
  : "${RETRY_TASK_LIST:?RETRY_TASK_LIST is required for feature-retry}"
  [[ "$RETRY_TASK_LIST" == /* && -s "$RETRY_TASK_LIST" ]] || { echo "RETRY_TASK_LIST must be an absolute nonempty file" >&2; exit 2; }
  awk 'NF != 1 || $0 != $1 || $0 !~ /^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$/ {exit 2} END {if(NR==0) exit 3}' "$RETRY_TASK_LIST" || {
    echo "RETRY_TASK_LIST must contain one exact valid field key per nonblank line" >&2
    exit 2
  }
  retry_task_count="$(sort -u "$RETRY_TASK_LIST" | awk 'END {print NR}')"
  raw_retry_count="$(awk 'END {print NR}' "$RETRY_TASK_LIST")"
  [[ "$retry_task_count" -eq "$raw_retry_count" ]] || { echo "RETRY_TASK_LIST contains duplicate keys" >&2; exit 2; }
  while IFS= read -r retry_key; do grep -Fxq "$retry_key" "$TASK_LIST" || { echo "Retry key is outside canonical field universe: $retry_key" >&2; exit 2; }; done < "$RETRY_TASK_LIST"
  retry_sha="$(sha256sum "$RETRY_TASK_LIST")"; retry_sha="${retry_sha%%[[:space:]]*}"
  retry_freeze_dir="$SHADOW_ROOT/workflow_status/retry_task_lists"
  retry_frozen_list="$retry_freeze_dir/${retry_sha}.txt"
  mkdir -p "$retry_freeze_dir"
  if [[ -e "$retry_frozen_list" ]]; then
    cmp -s "$RETRY_TASK_LIST" "$retry_frozen_list" || { echo "Frozen retry-list hash collision: $retry_frozen_list" >&2; exit 2; }
  else
    retry_tmp="$retry_freeze_dir/.${retry_sha}.tmp.$$"
    cp "$RETRY_TASK_LIST" "$retry_tmp"
    mv "$retry_tmp" "$retry_frozen_list"
  fi
  export TASK_LIST="$retry_frozen_list"
  resume_submission="$SHADOW_ROOT/workflow_status/resume_submission_feature-retry_${RUN_STAMP}_${retry_sha:0:12}.tsv"
  [[ ! -e "$resume_submission" ]] || { echo "Immutable resume submission already exists: $resume_submission" >&2; exit 2; }
  resume_job="$(submit_job feature-retry --job-name bp_feature_retry --cpus-per-task "$FEATURE_CPUS" --mem "$FEATURE_MEM" --time "$FEATURE_TIME" --array "1-${retry_task_count}%${FEATURE_MAX_CONCURRENT}" "${base_args[@]}" "${dependency_args[@]}" --export="$(slurm_export_spec)" --wrap="$(worker_wrap_command "$FEATURE_WORKER")")"
  echo -e "stage\tjob_id\tdependency\tretry_task_list\tfrozen_retry_task_list\tretry_task_list_sha256\nfeature-retry\t$resume_job\t${DEPENDENCY_JOB_ID:-none}\t$RETRY_TASK_LIST\t$retry_frozen_list\t$retry_sha" > "$resume_submission"
  echo "resume_stage=feature-retry"
  echo "resume_job_id=$resume_job"
  exit 0
fi
resume_cpus="$RESUME_CPUS"; resume_mem="$RESUME_MEM"; resume_time="$RESUME_TIME"
case "$RESUME_STAGE" in
  adapter) resume_cpus="$ADAPTER_CPUS"; resume_mem="$ADAPTER_MEM"; resume_time="$ADAPTER_TIME" ;;
  validate) resume_cpus="$VALIDATE_CPUS"; resume_mem="$VALIDATE_MEM"; resume_time="$VALIDATE_TIME" ;;
  umap) resume_cpus="$UMAP_CPUS"; resume_mem="$UMAP_MEM"; resume_time="$UMAP_TIME" ;;
  annotate) resume_cpus="$ANNOTATE_CPUS"; resume_mem="$ANNOTATE_MEM"; resume_time="$ANNOTATE_TIME" ;;
  review-build|review-import) resume_cpus="$REVIEW_CPUS"; resume_mem="$REVIEW_MEM"; resume_time="$REVIEW_TIME" ;;
esac

resume_extra_names=""
if [[ "$RESUME_STAGE" == "adapter" ]]; then
  resume_worker="$ADAPTER_WORKER"
elif [[ "$RESUME_STAGE" == "validate" ]]; then
  CPA_STAGE=validate; CPA_VALIDATE_STAGE=umap; CPA_STAGE_MODE="${CPA_STAGE_MODE:-}"; export CPA_STAGE CPA_VALIDATE_STAGE CPA_STAGE_MODE
  resume_worker="$STAGE_WORKER"; resume_extra_names="CPA_STAGE CPA_VALIDATE_STAGE CPA_STAGE_MODE"
elif [[ "$RESUME_STAGE" == "umap" || "$RESUME_STAGE" == "annotate" ]]; then
  CPA_STAGE="$RESUME_STAGE"; CPA_STAGE_MODE="${CPA_STAGE_MODE:-}"; export CPA_STAGE CPA_STAGE_MODE
  resume_worker="$STAGE_WORKER"; resume_extra_names="CPA_STAGE CPA_STAGE_MODE"
else

case "$RESUME_STAGE" in
  annotation-import) : "${CPA_SUBMISSION:?CPA_SUBMISSION is required}"; export CPA_SUBMISSION; resume_extra_names="CPA_SUBMISSION" ;;
  review-build) : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required}"; export CPA_ANNOTATION_IMPORT_DIR; resume_extra_names="CPA_ANNOTATION_IMPORT_DIR" ;;
  review-import)
    : "${CPA_SUBMISSION:?CPA_SUBMISSION is required}"
    : "${CPA_ANNOTATION_IMPORT_DIR:?CPA_ANNOTATION_IMPORT_DIR is required}"
    export CPA_SUBMISSION CPA_ANNOTATION_IMPORT_DIR
    resume_extra_names="CPA_SUBMISSION CPA_ANNOTATION_IMPORT_DIR"
    ;;
  train) : "${CPA_REVIEWED_LABELS:?CPA_REVIEWED_LABELS is required}"; export CPA_REVIEWED_LABELS; resume_extra_names="CPA_REVIEWED_LABELS" ;;
  predict|report) : "${CPA_MODEL_DIR:?CPA_MODEL_DIR is required}"; export CPA_MODEL_DIR; resume_extra_names="CPA_MODEL_DIR" ;;
  finalize)
    : "${PREDICTION_DIR:?PREDICTION_DIR is required}"
    FINALIZE_MODE=canonical
    export PREDICTION_DIR FINALIZE_MODE
    resume_extra_names="PREDICTION_DIR FINALIZE_MODE"
    ;;
  finalize-sharded)
    canonical_feature_manifest="$SHADOW_ROOT/workflow_status/feature_inventory/${BRANCH}_feature_manifest.tsv"
    canonical_prediction_root="$SHADOW_ROOT/predictions/sharded"
    [[ -z "${FEATURE_MANIFEST+x}" || "$FEATURE_MANIFEST" == "$canonical_feature_manifest" ]] || { echo "FEATURE_MANIFEST is not canonical" >&2; exit 2; }
    [[ -z "${PREDICTION_ROOT+x}" || "$PREDICTION_ROOT" == "$canonical_prediction_root" ]] || { echo "PREDICTION_ROOT is not canonical" >&2; exit 2; }
    FEATURE_MANIFEST="$canonical_feature_manifest"
    PREDICTION_ROOT="$canonical_prediction_root"
    MODEL_ACCEPTANCE_RECEIPT="${MODEL_ACCEPTANCE_RECEIPT:-$SHADOW_ROOT/workflow_status/model_acceptance/model_acceptance.json}"
    MODEL_ACCEPTANCE_SHA256_FILE="${MODEL_ACCEPTANCE_SHA256_FILE:-$SHADOW_ROOT/workflow_status/model_acceptance/model_acceptance.sha256}"
    [[ -s "$FEATURE_MANIFEST" && -d "$PREDICTION_ROOT" && -s "$MODEL_ACCEPTANCE_RECEIPT" && -s "$MODEL_ACCEPTANCE_SHA256_FILE" ]] || { echo "Sharded finalize inputs are unavailable" >&2; exit 2; }
    FINALIZE_MODE=sharded
    export FEATURE_MANIFEST PREDICTION_ROOT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE FINALIZE_MODE
    resume_extra_names="FEATURE_MANIFEST PREDICTION_ROOT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE FINALIZE_MODE"
    ;;
  *) echo "Unsupported RESUME_STAGE: $RESUME_STAGE" >&2; exit 2 ;;
esac
if [[ "$RESUME_STAGE" == "finalize" || "$RESUME_STAGE" == "finalize-sharded" ]]; then
  resume_worker="$FINALIZE_WORKER"
else
  CPA_STAGE="$RESUME_STAGE"; CPA_STAGE_MODE="${CPA_STAGE_MODE:-}"; export CPA_STAGE CPA_STAGE_MODE
  resume_worker="$STAGE_WORKER"
  resume_extra_names="$resume_extra_names CPA_STAGE CPA_STAGE_MODE"
fi
fi
resume_export="$(slurm_export_spec "$resume_extra_names")"
resume_submission="$SHADOW_ROOT/workflow_status/resume_submission_${RESUME_STAGE}_${RUN_STAMP}.tsv"
[[ ! -e "$resume_submission" ]] || { echo "Immutable resume submission already exists: $resume_submission" >&2; exit 2; }
resume_job="$(submit_job "$RESUME_STAGE" --job-name "bp_${RESUME_STAGE//-/_}" --cpus-per-task "$resume_cpus" --mem "$resume_mem" --time "$resume_time" "${base_args[@]}" "${dependency_args[@]}" --export="$resume_export" --wrap="$(worker_wrap_command "$resume_worker")")"
echo -e "stage\tjob_id\tdependency\n$RESUME_STAGE\t$resume_job\t${DEPENDENCY_JOB_ID:-none}" > "$resume_submission"
echo "resume_stage=$RESUME_STAGE"
echo "resume_job_id=$resume_job"
