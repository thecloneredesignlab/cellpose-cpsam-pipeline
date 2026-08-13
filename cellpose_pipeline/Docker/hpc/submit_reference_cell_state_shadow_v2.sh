#!/usr/bin/env bash
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -x /usr/bin/bash ]]; then SYSTEM_BASH_BIN=/usr/bin/bash
elif [[ -x /bin/bash ]]; then SYSTEM_BASH_BIN=/bin/bash
else echo "An absolute system Bash interpreter is required" >&2; exit 2
fi

# The public caller cannot assert frozen execution.  A child can acquire this
# in-process state only by consuming the one-use token created after the outer
# process verifies and extracts the frozen archive.
EARLY_FROZEN_REEXEC_VERIFIED=0
if [[ -n "${REFERENCE_CELL_STATE_V2_FROZEN_REEXEC+x}" || \
      -n "${REFERENCE_CELL_STATE_V2_FROZEN_TOKEN+x}" || \
      -n "${REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE+x}" ]]; then
  [[ "${REFERENCE_CELL_STATE_V2_FROZEN_REEXEC:-}" == 1 && \
     "${REFERENCE_CELL_STATE_V2_FROZEN_TOKEN:-}" =~ ^[0-9a-f]{64}$ && \
     "${REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE:-}" == /* && \
     -f "$REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE" && \
     ! -L "$REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE" && \
     "${PROJECT_DIR:-}" == /* ]] || {
    echo "Ambient/incomplete frozen-reexec assertion is forbidden" >&2; exit 2;
  }
  frozen_token_file="$REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE"
  frozen_token_parent="$(cd "$(dirname "$frozen_token_file")" && pwd -P)"
  frozen_project="$(cd "$PROJECT_DIR" && pwd -P)"
  frozen_script="$(realpath "${BASH_SOURCE[0]}")"
  frozen_token_parent_mode="$(python3 -I -c 'import os,stat,sys; print(format(stat.S_IMODE(os.stat(sys.argv[1]).st_mode), "o"))' "$frozen_token_parent")" || exit 2
  [[ "$(basename "$frozen_token_parent")" == reference-cell-state-v2-resume.* && \
     "$frozen_token_parent_mode" == 700 && \
     "$frozen_project" == "$frozen_token_parent/project" && \
     "$SCRIPT_DIR" == "$frozen_project/cellpose_pipeline/Docker/hpc" && \
     "$frozen_script" == "$SCRIPT_DIR/submit_reference_cell_state_shadow_v2.sh" ]] || {
    echo "Frozen-reexec token is not bound to this extracted submitter" >&2; exit 2;
  }
  token_value() {
    local key="$1"
    awk -F '\t' -v key="$key" '$1==key{print $2; n++} END{if(n!=1) exit 2}' "$frozen_token_file"
  }
  [[ "$(sed -n '1p' "$frozen_token_file")" == $'property\tvalue' && \
     "$(token_value schema_version)" == reference_cell_state_v2_frozen_reexec_token_v1 && \
     "$(token_value nonce)" == "$REFERENCE_CELL_STATE_V2_FROZEN_TOKEN" && \
     "$(token_value issuer_pid)" == "$PPID" && \
     "$(token_value project_root)" == "$frozen_project" && \
     "$(token_value submitter)" == "$frozen_script" && \
     "$(token_value shadow_root)" == "$(realpath "${REFERENCE_SHADOW_ROOT:?}")" && \
     "$(token_value archive)" == "$(realpath "${REFERENCE_SHADOW_ROOT:?}/workflow_status/code_snapshot.tar")" ]] || {
    echo "Frozen-reexec token identity mismatch" >&2; exit 2;
  }
  token_archive_sha="$(token_value archive_sha256)"
  observed_token_archive_sha="$(sha256sum "$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot.tar")"
  observed_token_archive_sha="${observed_token_archive_sha%%[[:space:]]*}"
  [[ "$token_archive_sha" =~ ^[0-9a-f]{64}$ && "$observed_token_archive_sha" == "$token_archive_sha" ]] || {
    echo "Frozen-reexec archive changed after extraction" >&2; exit 2;
  }
  rm -f -- "$frozen_token_file"
  unset -f token_value
  unset REFERENCE_CELL_STATE_V2_FROZEN_REEXEC REFERENCE_CELL_STATE_V2_FROZEN_TOKEN REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE
  EARLY_FROZEN_REEXEC_VERIFIED=1
fi

# A continuation must not source even a helper from the mutable/live checkout.
# Do the minimum archive/receipt validation inline, extract to a private local
# directory, compare it with the recorded shared snapshot, then re-exec the
# submitter from that verified archive under an allowlisted environment.
EARLY_V2_STAGE="${V2_STAGE:-phase-a}"
EARLY_EXISTING_V2_ROOT=0
if [[ -n "${REFERENCE_SHADOW_ROOT:-}" && -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]]; then
  EARLY_EXISTING_V2_ROOT=1
fi
if [[ ( "$EARLY_V2_STAGE" != phase-a || "$EARLY_EXISTING_V2_ROOT" == 1 ) && \
      "$EARLY_FROZEN_REEXEC_VERIFIED" != 1 ]]; then
  : "${REFERENCE_SHADOW_ROOT:?A V2 continuation requires REFERENCE_SHADOW_ROOT before frozen re-exec}"
  EARLY_DEFAULT_RESULTS_ROOT="/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results"
  early_mode="${EXECUTION_MODE:-slurm}"
  case "$early_mode" in slurm|direct_test) ;; *) echo "Invalid continuation execution mode" >&2; exit 2 ;; esac
  early_results_input="${RESULTS_ROOT:-$EARLY_DEFAULT_RESULTS_ROOT}"
  [[ "$early_results_input" == /* && -d "$early_results_input" && ! -L "$early_results_input" && \
     -d "$EARLY_DEFAULT_RESULTS_ROOT" && ! -L "$EARLY_DEFAULT_RESULTS_ROOT" ]] || {
    echo "V2 continuation results namespace is unavailable" >&2; exit 2;
  }
  early_results="$(cd "$early_results_input" && pwd -P)"
  early_default_results="$(cd "$EARLY_DEFAULT_RESULTS_ROOT" && pwd -P)"
  [[ "$early_results" == "$early_default_results" ]] || {
    echo "V2 continuation RESULTS_ROOT differs from the frozen experiment namespace" >&2; exit 2;
  }
  [[ "$REFERENCE_SHADOW_ROOT" == /* && -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
    echo "V2 continuation root must be an absolute real directory" >&2; exit 2;
  }
  early_shadow="$(cd "$REFERENCE_SHADOW_ROOT" && pwd -P)"
  early_shadow_parent="$(dirname "$early_shadow")"
  early_shadow_name="$(basename "$early_shadow")"
  if [[ "$early_mode" == slurm ]]; then
    [[ "$early_shadow_parent" == "$early_results" && "$early_shadow_name" == reference_cell_state_shadow_v2_* && \
       "$early_shadow_name" != reference_cell_state_shadow_v2_test_* ]] || {
      echo "Formal V2 continuation root escaped or differs from its exact results namespace" >&2; exit 2;
    }
  else
    early_calibration="$early_results/Tests_and_Parameters_calibration"
    [[ -d "$early_calibration" && ! -L "$early_calibration" ]] || { echo "Calibration namespace is unavailable" >&2; exit 2; }
    early_calibration="$(cd "$early_calibration" && pwd -P)"
    [[ "$early_shadow_parent" == "$early_calibration" && "$early_shadow_name" == reference_cell_state_shadow_v2_test_* ]] || {
      echo "Calibration V2 continuation root escaped or differs from its exact test namespace" >&2; exit 2;
    }
  fi
  REFERENCE_SHADOW_ROOT="$early_shadow"
  RESULTS_ROOT="$early_results"
  resume_workflow="$REFERENCE_SHADOW_ROOT/workflow_status"
  [[ -d "$resume_workflow" && ! -L "$resume_workflow" && \
     "$(realpath "$resume_workflow")" == "$REFERENCE_SHADOW_ROOT/workflow_status" ]] || {
    echo "Frozen V2 workflow_status must be a real in-root directory" >&2; exit 2;
  }
  resume_archive="$resume_workflow/code_snapshot.tar"
  resume_receipt="$resume_workflow/code_snapshot_receipt.tsv"
  resume_shared="$resume_workflow/code_snapshot"
  [[ -f "$resume_archive" && ! -L "$resume_archive" && "$(realpath "$resume_archive")" == "$resume_archive" && \
     -f "$resume_receipt" && ! -L "$resume_receipt" && "$(realpath "$resume_receipt")" == "$resume_receipt" && \
     -d "$resume_shared" && ! -L "$resume_shared" && "$(realpath "$resume_shared")" == "$resume_shared" ]] || {
    echo "Frozen V2 resume inputs are unavailable, symlinked, or escaped their exact paths" >&2; exit 2;
  }
  [[ "$(sed -n '1p' "$resume_receipt")" == $'property\tvalue' ]] || { echo "Frozen V2 resume receipt header changed" >&2; exit 2; }
  resume_schema="$(awk -F '\t' '$1=="schema_version"{print $2; n++} END{if(n!=1) exit 2}' "$resume_receipt")" || exit 2
  resume_sha="$(awk -F '\t' '$1=="archive_sha256"{print $2; n++} END{if(n!=1) exit 2}' "$resume_receipt")" || exit 2
  resume_root="$(awk -F '\t' '$1=="snapshot_root"{print $2; n++} END{if(n!=1) exit 2}' "$resume_receipt")" || exit 2
  observed_resume_sha="$(sha256sum "$resume_archive")"; observed_resume_sha="${observed_resume_sha%%[[:space:]]*}"
  [[ "$resume_schema" == reference_cell_state_v2_code_snapshot_v1 && \
     "$resume_sha" =~ ^[0-9a-f]{64}$ && "$resume_sha" == "$observed_resume_sha" && \
     "$resume_root" == "$resume_shared" ]] || { echo "Frozen V2 resume receipt/path/archive identity mismatch" >&2; exit 2; }
  tar -tf "$resume_archive" | awk '/^\// {exit 2} {n=split($0,p,"/"); for(i=1;i<=n;i++) if(p[i]=="..") exit 3}' || {
    echo "Frozen V2 resume archive contains an unsafe path" >&2; exit 2;
  }
  if tar -tvf "$resume_archive" | awk 'substr($1,1,1)!="-" && substr($1,1,1)!="d"{found=1} END{exit(found?0:1)}'; then
    echo "Frozen V2 resume archive contains a link or special member" >&2
    exit 2
  fi
  early_tmp="$(mktemp -d /tmp/reference-cell-state-v2-resume.XXXXXX)"
  early_tmp="$(cd "$early_tmp" && pwd -P)"
  early_cleanup(){ [[ ! -d "$early_tmp/project" ]] || chmod -R u+w "$early_tmp/project" 2>/dev/null || true; rm -rf -- "$early_tmp"; }
  trap early_cleanup EXIT
  trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
  mkdir "$early_tmp/project"
  tar -xf "$resume_archive" -C "$early_tmp/project"
  extracted_submitter="$early_tmp/project/cellpose_pipeline/Docker/hpc/submit_reference_cell_state_shadow_v2.sh"
  [[ -f "$extracted_submitter" && ! -L "$extracted_submitter" && \
     "$(realpath "$extracted_submitter")" == "$extracted_submitter" ]] || {
    echo "Frozen archive did not yield an exact regular submitter" >&2; exit 2;
  }
  chmod -R a-w "$early_tmp/project" 2>/dev/null || true
  diff -qr "$early_tmp/project" "$resume_shared" >/dev/null || { echo "Shared V2 snapshot differs from its verified archive" >&2; exit 2; }
  umask 077
  frozen_nonce="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
  [[ "$frozen_nonce" =~ ^[0-9a-f]{64}$ ]] || { echo "Could not generate frozen-reexec nonce" >&2; exit 2; }
  frozen_token_file="$early_tmp/frozen-reexec-token.tsv"
  cat > "$frozen_token_file" <<EOF
property	value
schema_version	reference_cell_state_v2_frozen_reexec_token_v1
nonce	$frozen_nonce
issuer_pid	$$
project_root	$(realpath "$early_tmp/project")
submitter	$(realpath "$early_tmp/project/cellpose_pipeline/Docker/hpc/submit_reference_cell_state_shadow_v2.sh")
shadow_root	$REFERENCE_SHADOW_ROOT
archive	$(realpath "$resume_archive")
archive_sha256	$resume_sha
EOF
  resume_tool_path=/usr/bin:/bin
  for tool_name in apptainer sbatch sacct flock; do
    tool_path="$(command -v "$tool_name" || true)"
    [[ "$tool_path" == /* ]] || continue
    case ":$resume_tool_path:" in *":$(dirname "$tool_path"):"*) ;; *) resume_tool_path="$(dirname "$tool_path"):$resume_tool_path" ;; esac
  done
  resume_names=(EXECUTION_MODE V2_STAGE DRY_RUN_SUBMIT RUN_STAMP EXPERIMENT_ROOT RESULTS_ROOT
    PARENT_BROAD_SHADOW_ROOT DATASET_ROOT SOURCE_SEGMENTATION_ROOT HPC_CONTAINER_IMAGE
    CPA_REFERENCE_ROOT EXPECTED_PARENT_CELL_COUNT DEPENDENCY_JOB_ID REFERENCE_QOS REFERENCE_SHADOW_ROOT)
  for optional in REGION_SUBMISSION SEED1_SUBMISSION SEED2_SUBMISSION REVIEW_ADJUDICATION \
    CURRENT_CLASSIFICATION_ROOT CURRENT_PREDICTIONS CURRENT_MANIFEST COMPARISON_ROOT; do
    [[ -z "${!optional+x}" ]] || resume_names+=("$optional")
  done
  clean_resume=(/usr/bin/env -i "PATH=$resume_tool_path" \
    "PROJECT_DIR=$early_tmp/project" "REFERENCE_CELL_STATE_V2_FROZEN_REEXEC=1" \
    "REFERENCE_CELL_STATE_V2_FROZEN_TOKEN=$frozen_nonce" \
    "REFERENCE_CELL_STATE_V2_FROZEN_TOKEN_FILE=$frozen_token_file")
  for name in "${resume_names[@]}"; do [[ -z "${!name+x}" ]] || clean_resume+=("$name=${!name}"); done
  set +e
  "${clean_resume[@]}" "$early_tmp/project/cellpose_pipeline/Docker/hpc/submit_reference_cell_state_shadow_v2.sh"
  resume_status=$?
  set -e
  early_cleanup
  trap - EXIT
  exit "$resume_status"
fi
source "$SCRIPT_DIR/util/reference_cell_state_v2_contract.sh"
reference_cell_state_v2_reject_ambient_runtime_overrides
reference_cell_state_v2_load_sif_identity

DEFAULT_EXPERIMENT_ROOT="$REFERENCE_CELL_STATE_V2_DEFAULT_EXPERIMENT_ROOT"
DEFAULT_PARENT_BROAD_ROOT="$DEFAULT_EXPERIMENT_ROOT/results/broad_phenotype_shadow_20260812_075437"
EXECUTION_MODE="${EXECUTION_MODE:-slurm}"
V2_STAGE="${V2_STAGE:-phase-a}"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DEFAULT_EXPERIMENT_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/results}"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$DEFAULT_PARENT_BROAD_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$EXPERIMENT_ROOT/results/full_fusion_shape_strict_20260711_155940}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$REFERENCE_CELL_STATE_V2_DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-$REFERENCE_CELL_STATE_V2_DEFAULT_CPA_ROOT}"
EXPECTED_PARENT_CELL_COUNT="${EXPECTED_PARENT_CELL_COUNT:-32000}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
REFERENCE_QOS="${REFERENCE_QOS:-xxlarge}"

case "$EXECUTION_MODE" in slurm|direct_test) ;; *) echo "EXECUTION_MODE must be slurm or direct_test" >&2; exit 2 ;; esac
case "$V2_STAGE" in phase-a|post-region|post-fallback-seed1|post-seed1|post-seed2|post-adjudication|predict|compare) ;;
  *) echo "V2_STAGE must be phase-a, post-region, post-fallback-seed1, post-seed1, post-seed2, post-adjudication, predict, or compare" >&2; exit 2 ;;
esac
case "$DRY_RUN_SUBMIT" in 0|1) ;; *) echo "DRY_RUN_SUBMIT must be 0 or 1" >&2; exit 2 ;; esac
[[ "$EXPECTED_PARENT_CELL_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_PARENT_CELL_COUNT must be positive" >&2; exit 2; }
[[ -z "$DEPENDENCY_JOB_ID" || "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "DEPENDENCY_JOB_ID must be empty or numeric" >&2; exit 2; }
if [[ "$EXECUTION_MODE" == direct_test ]]; then
  [[ "$(hostname -s)" == hpctpa3pc0009 ]] || {
    echo "V2 direct calibration is restricted to hpctpa3pc0009" >&2
    exit 2
  }
  [[ -z "${SLURM_JOB_ID:-}" ]] || {
    echo "V2 direct calibration must not execute inside a Slurm allocation" >&2
    exit 2
  }
fi

SOURCE_PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}"
SOURCE_PROJECT_DIR="$(cd "$SOURCE_PROJECT_DIR" && pwd -P)"
[[ "$SOURCE_PROJECT_DIR" == "$(cd "$SCRIPT_DIR/../../.." && pwd -P)" ]] || {
  echo "PROJECT_DIR must be the checkout containing this submitter" >&2
  exit 2
}

real_dir() {
  local path="$1" label="$2"
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] || {
    echo "$label must be an absolute existing real directory: $path" >&2
    return 2
  }
  realpath "$path"
}
RESULTS_ROOT="$(real_dir "$RESULTS_ROOT" RESULTS_ROOT)"
PARENT_BROAD_SHADOW_ROOT="$(real_dir "$PARENT_BROAD_SHADOW_ROOT" PARENT_BROAD_SHADOW_ROOT)"
DATASET_ROOT="$(real_dir "$DATASET_ROOT" DATASET_ROOT)"
SOURCE_SEGMENTATION_ROOT="$(real_dir "$SOURCE_SEGMENTATION_ROOT" SOURCE_SEGMENTATION_ROOT)"
CPA_REFERENCE_ROOT="$(real_dir "$CPA_REFERENCE_ROOT" CPA_REFERENCE_ROOT)"
reference_cell_state_v2_require_runtime_identity "$HPC_CONTAINER_IMAGE" "$CPA_REFERENCE_ROOT"

if [[ "$EXECUTION_MODE" == slurm ]]; then
  [[ "$RESULTS_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/results" && \
     "$PARENT_BROAD_SHADOW_ROOT" == "$DEFAULT_PARENT_BROAD_ROOT" && \
     "$DATASET_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages" && \
     "$SOURCE_SEGMENTATION_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/results/full_fusion_shape_strict_20260711_155940" && \
     "$EXPECTED_PARENT_CELL_COUNT" == 32000 && "$REFERENCE_QOS" == xxlarge ]] || {
    echo "Formal V2 execution differs from the frozen SUM159 roots/resources" >&2
    exit 2
  }
else
  case "$RESULTS_ROOT" in "$DEFAULT_EXPERIMENT_ROOT/results") ;; *) echo "Calibration RESULTS_ROOT must remain the frozen experiment results root" >&2; exit 2 ;; esac
fi

phase_a_bootstrap_new=0
if [[ "$V2_STAGE" == phase-a ]]; then
  if [[ -z "${REFERENCE_SHADOW_ROOT:-}" ]]; then
    if [[ "$EXECUTION_MODE" == slurm ]]; then
      REFERENCE_SHADOW_ROOT="$RESULTS_ROOT/reference_cell_state_shadow_v2_$RUN_STAMP"
    else
      REFERENCE_SHADOW_ROOT="$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_v2_test_$RUN_STAMP"
    fi
  fi
  if [[ ! -e "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]]; then
    [[ "$REFERENCE_SHADOW_ROOT" == /* ]] || { echo "V2 Phase A requires an absolute root" >&2; exit 2; }
    phase_a_bootstrap_new=1
    if [[ "$EXECUTION_MODE" == slurm ]]; then
      [[ "$(dirname "$REFERENCE_SHADOW_ROOT")" == "$RESULTS_ROOT" && \
         "$(basename "$REFERENCE_SHADOW_ROOT")" == reference_cell_state_shadow_v2_* && \
         "$(basename "$REFERENCE_SHADOW_ROOT")" != reference_cell_state_shadow_v2_test_* ]] || {
        echo "Formal V2 root must be a new direct child of the exact RESULTS_ROOT" >&2; exit 2;
      }
    else
      calibration_parent="$RESULTS_ROOT/Tests_and_Parameters_calibration"
      [[ -d "$calibration_parent" && ! -L "$calibration_parent" && \
         "$(dirname "$REFERENCE_SHADOW_ROOT")" == "$calibration_parent" && \
         "$(basename "$REFERENCE_SHADOW_ROOT")" == reference_cell_state_shadow_v2_test_* ]] || {
        echo "V2 calibration root must be a new direct child of Tests_and_Parameters_calibration" >&2; exit 2;
      }
    fi
  else
    [[ "$EARLY_FROZEN_REEXEC_VERIFIED" == 1 ]] || {
      echo "Existing Phase A roots may resume only through the verified frozen archive" >&2
      exit 2
    }
    root_mode=formal
    [[ "$EXECUTION_MODE" == slurm ]] || root_mode=calibration
    reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" "$root_mode" "$RESULTS_ROOT"
  fi
else
  : "${REFERENCE_SHADOW_ROOT:?REFERENCE_SHADOW_ROOT is required for a staged V2 continuation}"
  root_mode=formal
  [[ "$EXECUTION_MODE" == slurm ]] || root_mode=calibration
  reference_cell_state_v2_require_shadow_root "$REFERENCE_SHADOW_ROOT" "$root_mode" "$RESULTS_ROOT"
fi

PARENT_PROJECT="$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/project.yml"
PARENT_PROJECTION_INPUT_MANIFEST="$PARENT_BROAD_SHADOW_ROOT/projection_input/projection_input_manifest.json"
parent_umap_manifests=()
while IFS= read -r path; do parent_umap_manifests+=("$path"); done < <(
  find "$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/runs" -type f -name umap_manifest.json -print | LC_ALL=C sort
)
[[ -s "$PARENT_PROJECT" && -s "$PARENT_PROJECTION_INPUT_MANIFEST" && "${#parent_umap_manifests[@]}" == 1 ]] || {
  echo "Frozen parent development project/manifests are incomplete or ambiguous" >&2
  exit 2
}
PARENT_UMAP_MANIFEST="${parent_umap_manifests[0]}"
FEATURE_MANIFEST="$PARENT_BROAD_SHADOW_ROOT/workflow_status/feature_inventory/original_feature_manifest.tsv"
CELLS_FILE="$PARENT_BROAD_SHADOW_ROOT/cpa/cells.tsv"

bootstrap_owned=0
if [[ "$V2_STAGE" == phase-a && "$phase_a_bootstrap_new" == 1 ]]; then
  [[ -z "$(git -C "$SOURCE_PROJECT_DIR" status --porcelain --untracked-files=all)" ]] || {
    echo "V2 Phase A requires a clean committed checkout" >&2
    exit 2
  }
  project_git_sha="$(git -C "$SOURCE_PROJECT_DIR" rev-parse HEAD)"
  project_git_tree="$(git -C "$SOURCE_PROJECT_DIR" rev-parse "${project_git_sha}^{tree}")"
  cleanup_bootstrap() {
    local status=$?
    if [[ "$bootstrap_owned" == 1 && -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]]; then
      local marker="$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_v2_bootstrap_owner"
      if [[ -f "$marker" && "$(cat "$marker")" == "$$" ]]; then
        local snapshot="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot"
        [[ ! -d "$snapshot" ]] || chmod -R u+w "$snapshot" 2>/dev/null || true
        rm -rf -- "$REFERENCE_SHADOW_ROOT"
      fi
    fi
    exit "$status"
  }
  trap cleanup_bootstrap EXIT
  mkdir "$REFERENCE_SHADOW_ROOT"
  mkdir -p "$REFERENCE_SHADOW_ROOT/workflow_status" "$REFERENCE_SHADOW_ROOT/logs"
  printf '%s\n' "$$" > "$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_v2_bootstrap_owner"
  bootstrap_owned=1
  CODE_SNAPSHOT_ARCHIVE="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot.tar"
  CODE_SNAPSHOT_ROOT="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot"
  git -C "$SOURCE_PROJECT_DIR" archive --format=tar "$project_git_sha" > "$REFERENCE_SHADOW_ROOT/workflow_status/.code_snapshot.tar.tmp.$$"
  [[ "$(git -C "$SOURCE_PROJECT_DIR" rev-parse HEAD)" == "$project_git_sha" && \
     -z "$(git -C "$SOURCE_PROJECT_DIR" status --porcelain --untracked-files=all)" ]] || {
    echo "Project HEAD or cleanliness changed while freezing the V2 archive" >&2
    exit 2
  }
  mv "$REFERENCE_SHADOW_ROOT/workflow_status/.code_snapshot.tar.tmp.$$" "$CODE_SNAPSHOT_ARCHIVE"
  code_snapshot_archive_sha256="$(reference_cell_state_v2_sha256 "$CODE_SNAPSHOT_ARCHIVE")"
  mkdir "$CODE_SNAPSHOT_ROOT"
  tar -xf "$CODE_SNAPSHOT_ARCHIVE" -C "$CODE_SNAPSHOT_ROOT"
  chmod -R a-w "$CODE_SNAPSHOT_ROOT" 2>/dev/null || true
  if find "$CODE_SNAPSHOT_ROOT" \( -perm -0200 -o -perm -0020 -o -perm -0002 \) -print -quit | grep -q .; then
    snapshot_host_permission_mode=shared_filesystem_mode_bits_unavailable
  else
    snapshot_host_permission_mode=posix_mode_bits_read_only
  fi
  cat > "$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot_receipt.tsv" <<EOF
property	value
schema_version	reference_cell_state_v2_code_snapshot_v1
project_git_sha	$project_git_sha
project_git_tree	$project_git_tree
archive_path	$CODE_SNAPSHOT_ARCHIVE
archive_sha256	$code_snapshot_archive_sha256
snapshot_root	$CODE_SNAPSHOT_ROOT
host_permission_mode	$snapshot_host_permission_mode
EOF
  HPC_CONTAINER_IDENTITY_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/hpc_container_identity.json"
  source "$SCRIPT_DIR/util/broad_phenotype_container_identity.sh"
  broad_phenotype_capture_container_identity "$HPC_CONTAINER_IMAGE" "$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256" "$HPC_CONTAINER_IDENTITY_FILE"
  HPC_CONTAINER_IDENTITY_FILE_SHA256="$(reference_cell_state_v2_sha256 "$HPC_CONTAINER_IDENTITY_FILE")"
else
  CODE_SNAPSHOT_ARCHIVE="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot.tar"
  CODE_SNAPSHOT_ROOT="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot"
  CODE_SNAPSHOT_RECEIPT="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot_receipt.tsv"
  HPC_CONTAINER_IDENTITY_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/hpc_container_identity.json"
  for input in "$CODE_SNAPSHOT_ARCHIVE" "$CODE_SNAPSHOT_ROOT" "$CODE_SNAPSHOT_RECEIPT" "$HPC_CONTAINER_IDENTITY_FILE"; do
    [[ -e "$input" && ! -L "$input" ]] || { echo "Frozen V2 execution input is unavailable: $input" >&2; exit 2; }
  done
  code_snapshot_archive_sha256="$(awk -F '\t' '$1=="archive_sha256"{print $2}' "$CODE_SNAPSHOT_RECEIPT")"
  project_git_sha="$(awk -F '\t' '$1=="project_git_sha"{print $2}' "$CODE_SNAPSHOT_RECEIPT")"
  project_git_tree="$(awk -F '\t' '$1=="project_git_tree"{print $2}' "$CODE_SNAPSHOT_RECEIPT")"
  snapshot_host_permission_mode="$(awk -F '\t' '$1=="host_permission_mode"{print $2}' "$CODE_SNAPSHOT_RECEIPT")"
  [[ "$code_snapshot_archive_sha256" =~ ^[0-9a-f]{64}$ && \
     "$(reference_cell_state_v2_sha256 "$CODE_SNAPSHOT_ARCHIVE")" == "$code_snapshot_archive_sha256" ]] || {
    echo "Frozen V2 code snapshot archive changed" >&2
    exit 2
  }
  HPC_CONTAINER_IDENTITY_FILE_SHA256="$(reference_cell_state_v2_sha256 "$HPC_CONTAINER_IDENTITY_FILE")"
fi

PROJECT_DIR="$CODE_SNAPSHOT_ROOT"
for frozen in \
  "$PROJECT_DIR/cellpose_pipeline/Docker/hpc/submit_reference_cell_state_shadow_v2.sh" \
  "$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_features_v2.json" \
  "$PROJECT_DIR/cellpose_pipeline/configs/reference_cell_state_classes_v2.tsv"; do
  [[ -f "$frozen" && ! -L "$frozen" ]] || { echo "Frozen V2 snapshot lacks $frozen" >&2; exit 2; }
done

ambient_sbatch_names=()
while IFS= read -r name; do ambient_sbatch_names+=("$name"); done < <(compgen -e | LC_ALL=C sort | awk '/^SBATCH_/')
submission_preflight="$REFERENCE_SHADOW_ROOT/workflow_status/submission_preflight_v2.tsv"
if [[ "$V2_STAGE" == phase-a && "$phase_a_bootstrap_new" == 1 ]]; then
  cat > "$submission_preflight" <<EOF
property	value
schema_version	reference_cell_state_submission_preflight_v2
status	COMPLETE
execution_mode	$EXECUTION_MODE
reference_shadow_root	$REFERENCE_SHADOW_ROOT
parent_broad_shadow_root	$PARENT_BROAD_SHADOW_ROOT
parent_project	$PARENT_PROJECT
parent_projection_manifest	$PARENT_PROJECTION_INPUT_MANIFEST
parent_umap_manifest	$PARENT_UMAP_MANIFEST
expected_parent_cell_count	$EXPECTED_PARENT_CELL_COUNT
dataset_root	$DATASET_ROOT
source_segmentation_root	$SOURCE_SEGMENTATION_ROOT
project_git_sha	$project_git_sha
project_git_tree	$project_git_tree
code_snapshot_archive_sha256	$code_snapshot_archive_sha256
code_snapshot_host_permission_mode	$snapshot_host_permission_mode
hpc_container_image	$HPC_CONTAINER_IMAGE
hpc_container_sha256	$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_SHA256
hpc_container_bytes	$REFERENCE_CELL_STATE_V2_EXPECTED_SIF_BYTES
filesystem_immutability_mode	$REFERENCE_CELL_STATE_V2_FILESYSTEM_IMMUTABILITY_MODE
runtime_rootfs_read_only	$REFERENCE_CELL_STATE_V2_RUNTIME_ROOTFS_READ_ONLY
cpa_reference_root	$CPA_REFERENCE_ROOT
cpa_reference_commit	$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_COMMIT
cpa_reference_tree	$REFERENCE_CELL_STATE_V2_EXPECTED_CPA_TREE
classes_config	reference_cell_state_classes_v2.tsv
projection_profile	historical_promoted_shape_9
classifier_profile	promoted_shape_plus_rfs_boundary_12
pixel_calibration_status	unavailable_not_inferred
shape_unit_adapter	pixel_units_to_historical_names_only
diagnostic_cluster_role	metadata_only
dead_channel_bound	false
combined_rgb_bound	false
current_classifier_bound	false
legacy_no_go_enforcement	not_read
ambient_sbatch_variables_removed	${#ambient_sbatch_names[@]}
formal_submit_environment	/usr/bin/env_-i
EOF
fi

SBATCH_BIN="$(command -v sbatch || true)"
APPTAINER_BIN="$(command -v apptainer || true)"
SACCT_BIN="$(command -v sacct || true)"
FLOCK_BIN="$(command -v flock || true)"
CONTROLLED_TOOL_PATH="/usr/bin:/bin"
for tool in "$SBATCH_BIN" "$APPTAINER_BIN" "$SACCT_BIN" "$FLOCK_BIN"; do
  [[ "$tool" == /* ]] || continue
  case ":$CONTROLLED_TOOL_PATH:" in *":$(dirname "$tool"):"*) ;; *) CONTROLLED_TOOL_PATH="$(dirname "$tool"):$CONTROLLED_TOOL_PATH" ;; esac
done
if [[ "$EXECUTION_MODE" == slurm && "$DRY_RUN_SUBMIT" == 0 ]]; then
  [[ "$SBATCH_BIN" == /* && -x "$SBATCH_BIN" ]] || { echo "sbatch is unavailable" >&2; exit 127; }
  [[ "$SACCT_BIN" == /* && -x "$SACCT_BIN" ]] || { echo "sacct is unavailable; fail-closed retry state cannot be established" >&2; exit 127; }
fi
[[ "$FLOCK_BIN" == /* && -x "$FLOCK_BIN" ]] || { echo "flock is unavailable; crash-safe submitter serialization cannot be established" >&2; exit 127; }

worker_wrap_command() {
  local relative="$1" program
  program='set -euo pipefail
export PATH="$1"; archive="$2"; expected="$3"; canonical="$4"; relative="$5"; shell_bin="$6"
tmp_base="${SLURM_TMPDIR:-/tmp}"
[[ "$tmp_base" == /* && -d "$tmp_base" ]] || { echo "SLURM_TMPDIR is invalid" >&2; exit 2; }
tmp_parent="$(mktemp -d "$tmp_base/reference-cell-state-v2-code.XXXXXX")"
cleanup(){ [[ ! -d "${local_project:-}" ]] || chmod -R u+w "$local_project" 2>/dev/null || true; rm -rf -- "$tmp_parent"; }
trap cleanup EXIT
trap "exit 129" HUP
trap "exit 130" INT
trap "exit 143" TERM
local_archive="$tmp_parent/code_snapshot.tar"; local_project="$tmp_parent/project"
cp -- "$archive" "$local_archive"
observed="$(sha256sum "$local_archive")"; observed="${observed%%[[:space:]]*}"
[[ "$observed" == "$expected" ]] || { echo "Node-local archive SHA-256 mismatch" >&2; exit 2; }
tar -tf "$local_archive" | awk '\''/^\// {exit 2} {n=split($0,p,"/"); for(i=1;i<=n;i++) if(p[i]=="..") exit 3}'\'' || { echo "Unsafe archive member" >&2; exit 2; }
mkdir "$local_project"; tar -xf "$local_archive" -C "$local_project"; chmod -R a-w "$local_project"
local_worker="$local_project/$relative"
[[ -x "$local_worker" && ! -L "$local_worker" ]] || { echo "Node-local V2 worker unavailable" >&2; exit 2; }
export HPC_PROJECT_ROOT_SOURCE="$local_project" PROJECT_DIR="$canonical" HPC_PROJECT_ROOT="$canonical"
export HPC_CONTAINER_RUNTIME_ROOT="$local_project/cellpose_pipeline/Docker/hpc"
exec "$shell_bin" "$local_worker"'
  printf 'exec %q --noprofile --norc -c %q reference-cell-state-v2-node-local %q %q %q %q %q %q' \
    "$SYSTEM_BASH_BIN" "$program" "$CONTROLLED_TOOL_PATH" "$CODE_SNAPSHOT_ARCHIVE" \
    "$code_snapshot_archive_sha256" "$CODE_SNAPSHOT_ROOT" "$relative" "$SYSTEM_BASH_BIN"
}

BASE_JOB_ENV=(
  HPC_CONTAINER_IMAGE CPA_REFERENCE_ROOT REFERENCE_SHADOW_ROOT RESULTS_ROOT
  DATASET_ROOT SOURCE_SEGMENTATION_ROOT PARENT_BROAD_SHADOW_ROOT
  HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256
)
record_initial_job_ledger() {
  local job_name="$1" job_id="$2" dependency="$3" array_spec="$4"
  reference_cell_state_v2_attempt_ledger_record \
    "$ATTEMPT_LEDGER" "$job_name" "$job_name" "$job_id" "$dependency" "$array_spec" INITIAL none || {
      echo "Attempt ledger rejected initial job_name=$job_name" >&2
      return 2
    }
  reference_cell_state_v2_ledger_record \
    "$SUBMISSION_LEDGER" "$job_name" "$job_id" "$dependency" "$array_spec" || {
      echo "Submission ledger already contains or rejected job_name=$job_name" >&2
      return 2
    }
}
record_retry_job_ledger() {
  local job_name="$1" job_id="$2" dependency="$3" array_spec="$4"
  local trigger_state="$5" prior_job_id="$6"
  reference_cell_state_v2_attempt_ledger_record \
    "$ATTEMPT_LEDGER" "$job_name" "$job_name" "$job_id" "$dependency" "$array_spec" \
    "$trigger_state" "$prior_job_id" || {
      echo "Attempt ledger rejected retry job_name=$job_name prior_job_id=$prior_job_id" >&2
      return 2
  }
}
reuse_submitted_job() {
  local job_name="$1" dependency="${2:-none}" array_spec="${3:-none}"
  reference_cell_state_v2_ledger_reuse \
    "$SUBMISSION_LEDGER" "$job_name" "$dependency" "$array_spec"
}
submit_clean_job() {
  local relative="$1" job_name="$2" cpus="$3" memory="$4" walltime="$5" dependency="$6" array_spec="$7"
  local trigger_state="$8" prior_job_id="$9"
  shift 9
  local -a extra_names=("$@") clean_env=(/usr/bin/env -i "PATH=$CONTROLLED_TOOL_PATH") args=()
  local name value
  for name in "${BASE_JOB_ENV[@]}" "${extra_names[@]}"; do
    [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ && -n "${!name+x}" ]] || { echo "Undefined clean job environment: $name" >&2; return 2; }
    value="${!name}"
    clean_env+=("$name=$value")
  done
  args=(--parsable --job-name "$job_name" --qos "$REFERENCE_QOS" --cpus-per-task "$cpus" --mem "$memory" --time "$walltime"
    --output "$REFERENCE_SHADOW_ROOT/logs/%A_%a.$job_name.out" --error "$REFERENCE_SHADOW_ROOT/logs/%A_%a.$job_name.err"
    --chdir "$REFERENCE_SHADOW_ROOT" --wrap "$(worker_wrap_command "$relative")")
  [[ -z "$dependency" ]] || args+=(--dependency "afterok:$dependency")
  [[ -z "$array_spec" ]] || args+=(--array "$array_spec")
  if [[ "$DRY_RUN_SUBMIT" == 1 ]]; then
    job_id="DRYRUN_$job_name"
  else
    job_id="$("${clean_env[@]}" "$SBATCH_BIN" "${args[@]}")"
    job_id="${job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job id: $job_id" >&2; return 2; }
  fi
  if [[ "$trigger_state" == INITIAL ]]; then
    record_initial_job_ledger "$job_name" "$job_id" "$dependency" "$array_spec"
  else
    record_retry_job_ledger "$job_name" "$job_id" "$dependency" "$array_spec" "$trigger_state" "$prior_job_id"
  fi
  printf '%s\n' "$job_id"
}

reconcile_submission_ledgers() {
  local submitted_utc job_name job_id dependency array_spec extra latest
  while IFS=$'\t' read -r submitted_utc job_name job_id dependency array_spec extra; do
    [[ "$submitted_utc" == submitted_utc ]] && continue
    [[ -z "${extra:-}" ]] || return 2
    if ! latest="$(reference_cell_state_v2_attempt_ledger_latest "$ATTEMPT_LEDGER" "$job_name")"; then
      reference_cell_state_v2_attempt_ledger_record \
        "$ATTEMPT_LEDGER" "$job_name" "$job_name" "$job_id" "$dependency" "$array_spec" INITIAL none || return 2
    else
      awk -F '\t' -v node="$job_name" -v id="$job_id" -v dependency="$dependency" -v array="$array_spec" \
        '$2==node && $3==1 && $5==id && $6==dependency && $7==array{ok++} END{exit(ok==1?0:2)}' \
        "$ATTEMPT_LEDGER" || return 2
    fi
  done < "$SUBMISSION_LEDGER"
  while IFS=$'\t' read -r submitted_utc logical_node attempt job_name job_id dependency array_spec trigger prior extra; do
    [[ "$submitted_utc" == submitted_utc || "$attempt" != 1 ]] && continue
    [[ -z "${extra:-}" ]] || return 2
    if reference_cell_state_v2_ledger_reuse "$SUBMISSION_LEDGER" "$job_name" "$dependency" "$array_spec" >/dev/null; then
      :
    else
      local reuse_status=$?
      [[ "$reuse_status" == 1 ]] || return 2
      reference_cell_state_v2_ledger_record \
        "$SUBMISSION_LEDGER" "$job_name" "$job_id" "$dependency" "$array_spec" || return 2
    fi
  done < "$ATTEMPT_LEDGER"
}

query_slurm_job_state() {
  local job_id="$1" array_spec="${2:-none}" output_file state query_status
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 2
  output_file="$(mktemp /tmp/reference-cell-state-v2-sacct.XXXXXX)" || return 2
  if ! /usr/bin/env -i "PATH=$CONTROLLED_TOOL_PATH" LC_ALL=C \
    "$SACCT_BIN" --noheader --parsable2 --allocations --array --jobs "$job_id" \
      --format=JobID,JobIDRaw,State > "$output_file"; then
    rm -f -- "$output_file"
    echo "sacct failed while resolving V2 retry state for job $job_id" >&2
    return 2
  fi
  set +e
  state="$(reference_cell_state_v2_aggregate_sacct_state "$job_id" "$array_spec" "$output_file")"
  query_status=$?
  set -e
  rm -f -- "$output_file"
  if [[ "$query_status" != 0 ]]; then
    echo "sacct allocation state/coverage is not authoritative for V2 job $job_id" >&2
    return 2
  fi
  printf '%s\n' "$state"
}

audit_completed_logical_node() {
  local logical_node="$1"
  case "$logical_node" in
    reference_cell_state_v2_phase_a)
      reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT" >/dev/null
      ;;
    reference_cell_state_v2_post-*)
      reference_cell_state_v2_audit_posthuman_stage \
        "$REFERENCE_SHADOW_ROOT" "${logical_node#reference_cell_state_v2_}" >/dev/null
      ;;
    reference_cell_state_v2_accept)
      reference_cell_state_v2_audit_model_acceptance \
        "$MODEL_ACCEPTANCE_RECEIPT" "$MODEL_ACCEPTANCE_SHA256_FILE" >/dev/null
      ;;
    reference_cell_state_v2_predict)
      reference_cell_state_v2_audit_prediction_shards "$PREDICTION_ROOT" "$task_count" >/dev/null
      ;;
    reference_cell_state_v2_finalize)
      reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT" >/dev/null
      ;;
    reference_cell_state_v2_compare)
      reference_cell_state_v2_audit_comparison "$COMPARISON_ROOT" >/dev/null
      ;;
    *)
      echo "No completed-product audit exists for logical node $logical_node" >&2
      return 2
      ;;
  esac
}

FORMAL_RESOLVED_JOB_ID=
FORMAL_RESOLUTION_ACTION=
FORMAL_RETRY_TRIGGER=INITIAL
FORMAL_RETRY_PRIOR=none
resolve_formal_job() {
  local logical_node="$1" expected_dependency="${2:-none}" expected_array="${3:-none}"
  local allow_terminal_dependency_rebind="${4:-0}" latest state
  [[ "$allow_terminal_dependency_rebind" == 0 || "$allow_terminal_dependency_rebind" == 1 ]] || return 2
  FORMAL_RESOLVED_JOB_ID=
  FORMAL_RESOLUTION_ACTION=
  FORMAL_RETRY_TRIGGER=INITIAL
  FORMAL_RETRY_PRIOR=none
  if ! latest="$(reference_cell_state_v2_attempt_ledger_latest "$ATTEMPT_LEDGER" "$logical_node")"; then
    FORMAL_RESOLUTION_ACTION=submit_initial
    return 1
  fi
  local submitted_utc observed_node attempt observed_name job_id dependency array_spec trigger prior extra
  IFS=$'\t' read -r submitted_utc observed_node attempt observed_name job_id dependency array_spec trigger prior extra <<< "$latest"
  [[ -z "${extra:-}" && "$observed_node" == "$logical_node" && "$observed_name" == "$logical_node" && \
     "$attempt" =~ ^[1-9][0-9]*$ && "$job_id" =~ ^[0-9]+$ ]] || {
    echo "Latest V2 attempt record is malformed for $logical_node" >&2
    return 2
  }
  state="$(query_slurm_job_state "$job_id" "$array_spec")" || return 2
  if [[ "$array_spec" != "$expected_array" ]]; then
    echo "Latest V2 attempt differs from the immutable array contract: $logical_node job=$job_id" >&2
    return 2
  fi
  if [[ "$dependency" != "$expected_dependency" ]]; then
    case "$state" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|SIGNALING|STAGE_OUT)
        echo "Latest V2 attempt is active but its frozen dependency/array differs; wait for a terminal state: $logical_node job=$job_id" >&2
        return 2
        ;;
      COMPLETED)
        echo "Latest V2 attempt completed under a different dependency/array contract: $logical_node job=$job_id" >&2
        return 2
        ;;
      BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
        [[ "$allow_terminal_dependency_rebind" == 1 ]] || {
          echo "Caller changed the immutable external dependency for $logical_node" >&2
          return 2
        }
        # Prediction DAG replacements bind the latest internal upstream job.
        ;;
      *)
        echo "Cannot resolve dependency drift under unclassified state: $logical_node job=$job_id state=$state" >&2
        return 2
        ;;
    esac
  fi
  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|SIGNALING|STAGE_OUT)
      FORMAL_RESOLVED_JOB_ID="$job_id"
      FORMAL_RESOLUTION_ACTION=reuse_active
      echo "reference_cell_state_v2_slurm_attempt_active=$logical_node:$job_id:$state"
      return 0
      ;;
    COMPLETED)
      audit_completed_logical_node "$logical_node" || {
        echo "Slurm reports COMPLETED but the V2 product is missing, incomplete, or changed: $logical_node job=$job_id" >&2
        return 2
      }
      FORMAL_RESOLVED_JOB_ID="$job_id"
      FORMAL_RESOLUTION_ACTION=reuse_completed
      echo "reference_cell_state_v2_slurm_attempt_verified_complete=$logical_node:$job_id"
      return 0
      ;;
    BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
      FORMAL_RESOLUTION_ACTION=submit_retry
      FORMAL_RETRY_TRIGGER="$state"
      FORMAL_RETRY_PRIOR="$job_id"
      echo "reference_cell_state_v2_slurm_attempt_retry=$logical_node:$job_id:$state"
      return 1
      ;;
    *)
      echo "Slurm state is not classified for fail-closed V2 recovery: $logical_node job=$job_id state=$state" >&2
      return 2
      ;;
  esac
}

run_direct_worker() {
  local relative="$1" cpus="$2"
  shift 2
  local name wrap
  local -a clean_env=(/usr/bin/env -i "PATH=$CONTROLLED_TOOL_PATH" "SLURM_CPUS_PER_TASK=$cpus")
  for name in "${BASE_JOB_ENV[@]}" "$@"; do
    [[ -n "${!name+x}" ]] || { echo "Undefined direct-test worker environment: $name" >&2; return 2; }
    clean_env+=("$name=${!name}")
  done
  wrap="$(worker_wrap_command "$relative")"
  "${clean_env[@]}" "$SYSTEM_BASH_BIN" --noprofile --norc -c "$wrap"
}

mkdir -p "$REFERENCE_SHADOW_ROOT/logs"
stage_summary="$REFERENCE_SHADOW_ROOT/workflow_status/submission_${V2_STAGE}_v2.tsv"
SUBMISSION_LEDGER="$REFERENCE_SHADOW_ROOT/workflow_status/submission_${V2_STAGE}_jobs_v2.tsv"
ATTEMPT_LEDGER="$REFERENCE_SHADOW_ROOT/workflow_status/submission_${V2_STAGE}_attempts_v2.tsv"
stage_summary_exists=0
if [[ -e "$stage_summary" || -L "$stage_summary" ]]; then
  [[ -f "$stage_summary" && ! -L "$stage_summary" && \
     "$(sed -n '1p' "$stage_summary")" == $'property\tvalue' && \
     "$(awk -F '\t' '$1=="stage"{print $2; n++} END{if(n!=1) exit 2}' "$stage_summary")" == "$V2_STAGE" ]] || {
    echo "Existing V2 stage summary is invalid" >&2
    exit 2
  }
  completed_ledger="$REFERENCE_SHADOW_ROOT/workflow_status/submission_${V2_STAGE}_jobs_v2.tsv"
  [[ -f "$completed_ledger" && ! -L "$completed_ledger" && \
     "$(sed -n '1p' "$completed_ledger")" == $'submitted_utc\tjob_name\tjob_id\tdependency_job_id\tarray_spec' ]] || {
    echo "Completed V2 stage lacks its durable submission ledger" >&2
    exit 2
  }
  stage_summary_exists=1
fi
reference_cell_state_v2_ledger_init_or_verify "$SUBMISSION_LEDGER" || {
  echo "Existing partial V2 submission ledger is malformed or duplicates a job" >&2
  exit 2
}
reference_cell_state_v2_attempt_ledger_init_or_verify "$ATTEMPT_LEDGER" || {
  echo "Existing V2 attempt ledger is malformed or violates append-only attempt order" >&2
  exit 2
}
# From this point the archive, receipts, and durable ledgers are sufficient for
# recovery. Never delete a root after the public orchestrator starts resolving
# or submitting jobs.
if [[ "$bootstrap_owned" == 1 ]]; then
  bootstrap_owned=0
  rm -f "$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_v2_bootstrap_owner"
  trap - EXIT
fi
ATTEMPT_LOCK_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/submission_${V2_STAGE}_orchestrator.lock"
[[ ! -L "$ATTEMPT_LOCK_FILE" && ( ! -e "$ATTEMPT_LOCK_FILE" || -f "$ATTEMPT_LOCK_FILE" ) ]] || {
  echo "V2 stage orchestrator lock path is unsafe" >&2; exit 2;
}
exec 9>>"$ATTEMPT_LOCK_FILE"
"$FLOCK_BIN" -x 9 || { echo "Could not acquire the V2 stage orchestrator lock" >&2; exit 2; }
# Refresh all summary state while holding the kernel-released lock. This closes
# the pre-lock race where another submitter publishes the immutable summary.
stage_summary_exists=0
if [[ -e "$stage_summary" || -L "$stage_summary" ]]; then
  [[ -f "$stage_summary" && ! -L "$stage_summary" && \
     "$(sed -n '1p' "$stage_summary")" == $'property\tvalue' && \
     "$(awk -F '\t' '$1=="stage"{print $2; n++} END{if(n!=1) exit 2}' "$stage_summary")" == "$V2_STAGE" ]] || {
    echo "Existing V2 stage summary changed while acquiring its lock" >&2; exit 2;
  }
  stage_summary_exists=1
fi
reference_cell_state_v2_attempt_ledger_init_or_verify "$ATTEMPT_LEDGER" || {
  echo "V2 attempt ledger changed while acquiring its lock" >&2
  exit 2
}
reconcile_submission_ledgers || {
  echo "V2 initial and attempt submission ledgers disagree" >&2
  exit 2
}
if [[ "$stage_summary_exists" == 1 ]]; then
  while IFS= read -r recorded_job_id; do
    [[ -z "$recorded_job_id" ]] || awk -F '\t' -v id="$recorded_job_id" \
      'NR>1 && $5==id{found++} END{exit(found==1?0:2)}' "$ATTEMPT_LEDGER" || {
        echo "Stage summary job id is absent or duplicated in its append-only attempt ledger: $recorded_job_id" >&2
        exit 2
      }
  done < <(awk -F '\t' '$1=="job_id" || $1 ~ /_job_id$/{print $2}' "$stage_summary")
  if [[ "$EXECUTION_MODE" == direct_test ]]; then
    case "$V2_STAGE" in
      phase-a)
        reference_cell_state_v2_audit_phase_a "$REFERENCE_SHADOW_ROOT" >/dev/null
        echo "reference_cell_state_phase_a_v2_verified_reuse=1"
        ;;
      post-*) reference_cell_state_v2_audit_posthuman_stage "$REFERENCE_SHADOW_ROOT" "$V2_STAGE" >/dev/null ;;
      predict) reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT" >/dev/null ;;
      *) echo "Unsupported direct-test completed-stage audit: $V2_STAGE" >&2; exit 2 ;;
    esac
    echo "reference_cell_state_v2_submission_verified_reuse=1"
    echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
    echo "submission_summary=$stage_summary"
    exit 0
  fi
fi
METHOD_PARITY_STATUS=pending_phase_a_worker
if [[ "$V2_STAGE" != phase-a ]]; then
  METHOD_PARITY_STATUS="$(reference_cell_state_v2_method_parity_status "$REFERENCE_SHADOW_ROOT")"
fi

case "$V2_STAGE" in
  phase-a)
    PARENT_PROJECT="$PARENT_PROJECT"
    PARENT_PROJECTION_INPUT_MANIFEST="$PARENT_PROJECTION_INPUT_MANIFEST"
    PARENT_UMAP_MANIFEST="$PARENT_UMAP_MANIFEST"
    export PARENT_PROJECT PARENT_PROJECTION_INPUT_MANIFEST PARENT_UMAP_MANIFEST EXPECTED_PARENT_CELL_COUNT
    phase_dependency="${DEPENDENCY_JOB_ID:-none}"
    if [[ "$EXECUTION_MODE" == direct_test ]] && job_id="$(reuse_submitted_job reference_cell_state_v2_phase_a "$phase_dependency" none)"; then
      :
    elif [[ "$EXECUTION_MODE" == direct_test && $? -eq 2 ]]; then
      exit 2
    elif [[ "$EXECUTION_MODE" == direct_test ]]; then
      run_direct_worker cellpose_pipeline/Docker/hpc/run_reference_cell_state_phase_a_v2.sh 8 \
        PARENT_PROJECT PARENT_PROJECTION_INPUT_MANIFEST PARENT_UMAP_MANIFEST EXPECTED_PARENT_CELL_COUNT
      job_id=direct_test
      record_initial_job_ledger reference_cell_state_v2_phase_a direct_test "$phase_dependency" none
    else
      if resolve_formal_job reference_cell_state_v2_phase_a "$phase_dependency" none; then
        job_id="$FORMAL_RESOLVED_JOB_ID"
      else
        resolution_status=$?
        [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
        job_id="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_phase_a_v2.sh reference_cell_state_v2_phase_a 8 128G 12:00:00 "$DEPENDENCY_JOB_ID" "" \
          "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" \
          PARENT_PROJECT PARENT_PROJECTION_INPUT_MANIFEST PARENT_UMAP_MANIFEST EXPECTED_PARENT_CELL_COUNT)"
      fi
    fi
    reference_cell_state_v2_test_maybe_fail submission_after_phase_a_ledger
    if [[ "$stage_summary_exists" == 0 ]]; then
      cat > "${stage_summary}.tmp.$$" <<EOF
property	value
schema_version	reference_cell_state_v2_submission_summary
stage	phase-a
job_id	$job_id
reference_shadow_root	$REFERENCE_SHADOW_ROOT
next_barrier	determined_by_historical_optimizer
method_parity_status	$METHOD_PARITY_STATUS
EOF
      mv "${stage_summary}.tmp.$$" "$stage_summary"
    fi
    ;;
  post-region|post-fallback-seed1|post-seed1|post-seed2|post-adjudication)
    V2_ACTION="$V2_STAGE"
    extra=(V2_ACTION)
    if [[ "$V2_STAGE" == post-region ]]; then
      : "${REGION_SUBMISSION:?REGION_SUBMISSION is required for post-region}"
      extra+=(REGION_SUBMISSION)
    elif [[ ( "$V2_STAGE" == post-seed1 || "$V2_STAGE" == post-fallback-seed1 ) && -n "${SEED1_SUBMISSION:-}" ]]; then
      extra+=(SEED1_SUBMISSION)
    elif [[ "$V2_STAGE" == post-seed2 ]]; then
      [[ -z "${SEED2_SUBMISSION:-}" ]] || extra+=(SEED2_SUBMISSION)
      [[ -z "${REVIEW_ADJUDICATION:-}" ]] || {
        echo "Use V2_STAGE=post-adjudication when REVIEW_ADJUDICATION is supplied" >&2
        exit 2
      }
    elif [[ "$V2_STAGE" == post-adjudication ]]; then
      : "${REVIEW_ADJUDICATION:?REVIEW_ADJUDICATION is required for post-adjudication}"
      extra+=(REVIEW_ADJUDICATION)
    fi
    export V2_ACTION REGION_SUBMISSION SEED1_SUBMISSION SEED2_SUBMISSION REVIEW_ADJUDICATION
    post_job_name="reference_cell_state_v2_${V2_STAGE}"
    post_dependency="${DEPENDENCY_JOB_ID:-none}"
    if [[ "$EXECUTION_MODE" == direct_test ]] && job_id="$(reuse_submitted_job "$post_job_name" "$post_dependency" none)"; then
      :
    elif [[ "$EXECUTION_MODE" == direct_test && $? -eq 2 ]]; then
      exit 2
    elif [[ "$EXECUTION_MODE" == direct_test ]]; then
      run_direct_worker cellpose_pipeline/Docker/hpc/run_reference_cell_state_cpa_stage_v2.sh 4 "${extra[@]}"
      job_id=direct_test
      record_initial_job_ledger "$post_job_name" direct_test "$post_dependency" none
    else
      if resolve_formal_job "$post_job_name" "$post_dependency" none; then
        job_id="$FORMAL_RESOLVED_JOB_ID"
      else
        resolution_status=$?
        [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
        job_id="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_cpa_stage_v2.sh "$post_job_name" 4 64G 12:00:00 "$DEPENDENCY_JOB_ID" "" \
          "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" "${extra[@]}")"
      fi
    fi
    reference_cell_state_v2_test_maybe_fail "submission_after_${V2_STAGE}_ledger"
    if [[ "$stage_summary_exists" == 0 ]]; then
      cat > "${stage_summary}.tmp.$$" <<EOF
property	value
schema_version	reference_cell_state_v2_submission_summary
stage	$V2_STAGE
job_id	$job_id
reference_shadow_root	$REFERENCE_SHADOW_ROOT
method_parity_status	$METHOD_PARITY_STATUS
EOF
      mv "${stage_summary}.tmp.$$" "$stage_summary"
    fi
    ;;
  predict)
    [[ -s "$FEATURE_MANIFEST" && -s "$CELLS_FILE" ]] || { echo "Frozen original feature inventory/cells table is unavailable" >&2; exit 2; }
    task_count="$(( $(wc -l < "$FEATURE_MANIFEST") - 1 ))"
    [[ "$task_count" =~ ^[1-9][0-9]*$ ]] || { echo "Feature inventory is empty" >&2; exit 2; }
    [[ "$EXECUTION_MODE" != slurm || "$task_count" == 27200 ]] || { echo "Formal V2 prediction requires exactly 27200 field shards" >&2; exit 2; }
    CPA_MODEL_DIR="$REFERENCE_SHADOW_ROOT/models/final"
    CPA_TRAIN_RECEIPT="$REFERENCE_SHADOW_ROOT/workflow_status/training_v2/final_train_receipt.json"
    MODEL_ACCEPTANCE_RECEIPT="$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance_v2/model_acceptance.json"
    MODEL_ACCEPTANCE_SHA256_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/model_acceptance_v2/model_acceptance.sha256"
    PREDICTION_ROOT="$REFERENCE_SHADOW_ROOT/prediction_shards_v2"
    export FEATURE_MANIFEST CELLS_FILE CPA_MODEL_DIR CPA_TRAIN_RECEIPT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE PREDICTION_ROOT
    if [[ "$EXECUTION_MODE" == direct_test ]]; then
      if accept_job="$(reuse_submitted_job reference_cell_state_v2_accept none none)"; then :
      elif [[ $? -eq 2 ]]; then exit 2
      else
        run_direct_worker cellpose_pipeline/Docker/hpc/run_reference_cell_state_model_acceptance_v2.sh 4 CPA_MODEL_DIR CPA_TRAIN_RECEIPT
        accept_job=direct_test
        record_initial_job_ledger reference_cell_state_v2_accept "$accept_job" none none
      fi
      if predict_job="$(reuse_submitted_job reference_cell_state_v2_predict "$accept_job" "1-$task_count")"; then :
      elif [[ $? -eq 2 ]]; then exit 2
      else
        for ((task=1; task<=task_count; task++)); do
          SLURM_ARRAY_TASK_ID="$task" run_direct_worker cellpose_pipeline/Docker/hpc/run_reference_cell_state_predict_array_task_v2.sh 1 \
            FEATURE_MANIFEST PREDICTION_ROOT CPA_MODEL_DIR MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE SLURM_ARRAY_TASK_ID
        done
        predict_job=direct_test
        record_initial_job_ledger reference_cell_state_v2_predict "$predict_job" "$accept_job" "1-$task_count"
      fi
      if finalize_job="$(reuse_submitted_job reference_cell_state_v2_finalize "$predict_job" none)"; then :
      elif [[ $? -eq 2 ]]; then exit 2
      else
        run_direct_worker cellpose_pipeline/Docker/hpc/run_reference_cell_state_finalize_v2.sh 1 \
          FEATURE_MANIFEST PREDICTION_ROOT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE CELLS_FILE
        finalize_job=direct_test
        record_initial_job_ledger reference_cell_state_v2_finalize "$finalize_job" "$predict_job" none
      fi
    else
      accept_dependency="${DEPENDENCY_JOB_ID:-none}"
      if resolve_formal_job reference_cell_state_v2_accept "$accept_dependency" none; then
        accept_job="$FORMAL_RESOLVED_JOB_ID"
      else
        resolution_status=$?
        [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
        accept_job="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_model_acceptance_v2.sh reference_cell_state_v2_accept 4 32G 04:00:00 "$DEPENDENCY_JOB_ID" "" \
          "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" CPA_MODEL_DIR CPA_TRAIN_RECEIPT)"
      fi
      if resolve_formal_job reference_cell_state_v2_predict "$accept_job" "1-$task_count%64" 1; then
        predict_job="$FORMAL_RESOLVED_JOB_ID"
      else
        resolution_status=$?
        [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
        predict_job="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_predict_array_task_v2.sh reference_cell_state_v2_predict 1 8G 12:00:00 "$accept_job" "1-$task_count%64" \
          "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" FEATURE_MANIFEST PREDICTION_ROOT CPA_MODEL_DIR MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE)"
      fi
      if resolve_formal_job reference_cell_state_v2_finalize "$predict_job" none 1; then
        finalize_job="$FORMAL_RESOLVED_JOB_ID"
      else
        resolution_status=$?
        [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
        finalize_job="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_finalize_v2.sh reference_cell_state_v2_finalize 1 32G 12:00:00 "$predict_job" "" \
          "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" FEATURE_MANIFEST PREDICTION_ROOT MODEL_ACCEPTANCE_RECEIPT MODEL_ACCEPTANCE_SHA256_FILE CELLS_FILE)"
      fi
    fi
    if [[ "$stage_summary_exists" == 0 ]]; then
      cat > "${stage_summary}.tmp.$$" <<EOF
property	value
schema_version	reference_cell_state_v2_prediction_submission_summary
stage	predict
model_acceptance_job_id	$accept_job
prediction_array_job_id	$predict_job
finalize_job_id	$finalize_job
feature_task_count	$task_count
comparison_submitted	false
method_parity_status	$METHOD_PARITY_STATUS
EOF
      mv "${stage_summary}.tmp.$$" "$stage_summary"
    fi
    ;;
  compare)
    [[ "$EXECUTION_MODE" == slurm ]] || { echo "Optional classifier comparison is formal-only and post-freeze" >&2; exit 2; }
    reference_cell_state_v2_audit_final_predictions "$REFERENCE_SHADOW_ROOT" >/dev/null || {
      echo "Optional comparison requires a freshly audited final V2 SHADOW_ONLY technical GO generation" >&2
      exit 2
    }
    : "${CURRENT_CLASSIFICATION_ROOT:?CURRENT_CLASSIFICATION_ROOT is required for optional comparison}"
    if [[ -n "${CURRENT_PREDICTIONS:-}" && -z "${CURRENT_MANIFEST:-}" ]]; then :
    elif [[ -n "${CURRENT_MANIFEST:-}" && -z "${CURRENT_PREDICTIONS:-}" ]]; then :
    else echo "Set exactly one CURRENT_PREDICTIONS or CURRENT_MANIFEST" >&2; exit 2; fi
    CURRENT_CLASSIFICATION_ROOT="$(real_dir "$CURRENT_CLASSIFICATION_ROOT" CURRENT_CLASSIFICATION_ROOT)"
    if [[ -n "${CURRENT_PREDICTIONS:-}" ]]; then
      current_input_kind=predictions current_input_path="$CURRENT_PREDICTIONS"
    else
      current_input_kind=manifest current_input_path="$CURRENT_MANIFEST"
    fi
    reference_cell_state_v2_require_inside "$current_input_path" "$CURRENT_CLASSIFICATION_ROOT" current_comparison_input
    current_input_path="$(realpath "$current_input_path")"
    current_input_sha256="$(reference_cell_state_v2_sha256 "$current_input_path")"
    reference_predictions_path="$REFERENCE_SHADOW_ROOT/predictions/reference_cell_state_predictions.tsv"
    reference_cell_state_v2_require_inside "$reference_predictions_path" "$REFERENCE_SHADOW_ROOT" reference_comparison_input
    reference_predictions_sha256="$(reference_cell_state_v2_sha256 "$reference_predictions_path")"
    comparison_invocation="$REFERENCE_SHADOW_ROOT/workflow_status/submission_compare_invocation_v2.tsv"
    if [[ -e "$comparison_invocation" || -L "$comparison_invocation" ]]; then
      [[ -f "$comparison_invocation" && ! -L "$comparison_invocation" && \
         "$(sed -n '1p' "$comparison_invocation")" == $'property\tvalue' ]] || {
        echo "Existing comparison invocation identity is malformed" >&2; exit 2;
      }
      invocation_value() {
        awk -F '\t' -v key="$1" '$1==key{print $2; n++} END{if(n!=1) exit 2}' "$comparison_invocation"
      }
      recorded_comparison_root="$(invocation_value comparison_root)" || exit 2
      [[ "$(invocation_value schema_version)" == reference_cell_state_v2_comparison_invocation_v1 && \
         "$(invocation_value current_classification_root)" == "$CURRENT_CLASSIFICATION_ROOT" && \
         "$(invocation_value current_input_kind)" == "$current_input_kind" && \
         "$(invocation_value current_input_path)" == "$current_input_path" && \
         "$(invocation_value current_input_sha256)" == "$current_input_sha256" && \
         "$(invocation_value reference_predictions)" == "$reference_predictions_path" && \
         "$(invocation_value reference_predictions_sha256)" == "$reference_predictions_sha256" ]] || {
        echo "Comparison continuation differs from its immutable current/reference input identity" >&2; exit 2;
      }
      if [[ -n "${COMPARISON_ROOT:-}" && "$COMPARISON_ROOT" != "$recorded_comparison_root" ]]; then
        echo "Requested comparison root differs from the immutable invocation root" >&2; exit 2
      fi
      COMPARISON_ROOT="$recorded_comparison_root"
      unset -f invocation_value
    else
      COMPARISON_ROOT="${COMPARISON_ROOT:-$RESULTS_ROOT/reference_cell_state_comparison_v2_$RUN_STAMP}"
      [[ "$COMPARISON_ROOT" == /* && "$(dirname "$COMPARISON_ROOT")" == "$RESULTS_ROOT" && \
         "$(basename "$COMPARISON_ROOT")" == reference_cell_state_comparison_v2_* && \
         ! -L "$COMPARISON_ROOT" ]] || {
        echo "Comparison root must be a direct reference_cell_state_comparison_v2_* child of RESULTS_ROOT" >&2; exit 2;
      }
      cat > "${comparison_invocation}.tmp.$$" <<EOF
property	value
schema_version	reference_cell_state_v2_comparison_invocation_v1
current_classification_root	$CURRENT_CLASSIFICATION_ROOT
current_input_kind	$current_input_kind
current_input_path	$current_input_path
current_input_sha256	$current_input_sha256
reference_predictions	$reference_predictions_path
reference_predictions_sha256	$reference_predictions_sha256
comparison_root	$COMPARISON_ROOT
EOF
      mv "${comparison_invocation}.tmp.$$" "$comparison_invocation"
    fi
    if [[ "$stage_summary_exists" == 1 && \
          "$(awk -F '\t' '$1=="comparison_root"{print $2; n++} END{if(n!=1) exit 2}' "$stage_summary")" != "$COMPARISON_ROOT" ]]; then
      echo "Existing comparison summary differs from the immutable invocation root" >&2; exit 2
    fi
    if [[ "$current_input_kind" == predictions ]]; then
      CURRENT_PREDICTIONS="$current_input_path"
    else
      CURRENT_MANIFEST="$current_input_path"
    fi
    export CURRENT_CLASSIFICATION_ROOT CURRENT_PREDICTIONS CURRENT_MANIFEST COMPARISON_ROOT
    compare_extra=(CURRENT_CLASSIFICATION_ROOT COMPARISON_ROOT)
    [[ -z "${CURRENT_PREDICTIONS:-}" ]] || compare_extra+=(CURRENT_PREDICTIONS)
    [[ -z "${CURRENT_MANIFEST:-}" ]] || compare_extra+=(CURRENT_MANIFEST)
    compare_dependency="${DEPENDENCY_JOB_ID:-none}"
    if resolve_formal_job reference_cell_state_v2_compare "$compare_dependency" none; then
      job_id="$FORMAL_RESOLVED_JOB_ID"
    else
      resolution_status=$?
      [[ "$resolution_status" == 1 ]] || exit "$resolution_status"
      job_id="$(submit_clean_job cellpose_pipeline/Docker/hpc/run_reference_cell_state_compare_v2.sh reference_cell_state_v2_compare 1 16G 04:00:00 "$DEPENDENCY_JOB_ID" "" \
        "$FORMAL_RETRY_TRIGGER" "$FORMAL_RETRY_PRIOR" "${compare_extra[@]}")"
    fi
    if [[ "$stage_summary_exists" == 0 ]]; then
      cat > "${stage_summary}.tmp.$$" <<EOF
property	value
schema_version	reference_cell_state_v2_comparison_submission_summary
stage	compare
job_id	$job_id
comparison_root	$COMPARISON_ROOT
interpretation	descriptive_cross_classification_not_accuracy
method_parity_status	$METHOD_PARITY_STATUS
EOF
      mv "${stage_summary}.tmp.$$" "$stage_summary"
    fi
    ;;
esac

echo "reference_cell_state_v2_stage=$V2_STAGE"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "submission_summary=$stage_summary"
echo "legacy_no_go_enforcement=not_read"
