#!/usr/bin/env bash
set -euo pipefail

DEFAULT_EXPERIMENT_ROOT="/share/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
DEFAULT_PARENT_BROAD_ROOT="$DEFAULT_EXPERIMENT_ROOT/results/broad_phenotype_shadow_20260812_075437"
DEFAULT_SIF="/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif"
EXPECTED_SIF_SHA256="a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427"
DEFAULT_CPA_REFERENCE_ROOT="/share/lab_crd/taoli/Dependencies/cell-phenotype-annotator/7d1e23077efb85d503ca5333df76acab1ae3640c"
EXPECTED_CPA_COMMIT="7d1e23077efb85d503ca5333df76acab1ae3640c"
EXPECTED_CPA_TREE="7cd003ae4be29a5851ba305d92249cd9f8ff888e"

EXECUTION_MODE="${EXECUTION_MODE:-slurm}"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DEFAULT_EXPERIMENT_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/results}"
PARENT_BROAD_SHADOW_ROOT="${PARENT_BROAD_SHADOW_ROOT:-$DEFAULT_PARENT_BROAD_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_SEGMENTATION_ROOT="${SOURCE_SEGMENTATION_ROOT:-$EXPERIMENT_ROOT/results/full_fusion_shape_strict_20260711_155940}"
HPC_CONTAINER_IMAGE="${HPC_CONTAINER_IMAGE:-$DEFAULT_SIF}"
CPA_REFERENCE_ROOT="${CPA_REFERENCE_ROOT:-${CELL_PHENOTYPE_ANNOTATOR_ROOT:-$DEFAULT_CPA_REFERENCE_ROOT}}"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
EXPECTED_PARENT_CELL_COUNT="${EXPECTED_PARENT_CELL_COUNT:-32000}"
REFERENCE_CELL_STATE_SEED="${REFERENCE_CELL_STATE_SEED:-20260812}"
REFERENCE_MAX_REPRESENTATIVES="${REFERENCE_MAX_REPRESENTATIVES:-300}"
REFERENCE_QOS="${REFERENCE_QOS:-xxlarge}"
REFERENCE_CPUS="${REFERENCE_CPUS:-8}"
REFERENCE_MEM="${REFERENCE_MEM:-128G}"
REFERENCE_TIME="${REFERENCE_TIME:-12:00:00}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"

case "$EXECUTION_MODE" in
  slurm|direct_test) ;;
  *) echo "EXECUTION_MODE must be slurm or direct_test" >&2; exit 2 ;;
esac
case "$DRY_RUN_SUBMIT" in 0|1) ;; *) echo "DRY_RUN_SUBMIT must be 0 or 1" >&2; exit 2 ;; esac
for integer_value in "$EXPECTED_PARENT_CELL_COUNT" "$REFERENCE_CELL_STATE_SEED" "$REFERENCE_MAX_REPRESENTATIVES" "$REFERENCE_CPUS"; do
  [[ "$integer_value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive integer: $integer_value" >&2; exit 2; }
done
[[ -z "$DEPENDENCY_JOB_ID" || "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "DEPENDENCY_JOB_ID must be empty or numeric" >&2
  exit 2
}

for forbidden_name in \
  HPC_CONTAINER_BINDS HPC_CONTAINER_RUNTIME_ROOT HPC_CONTAINER_FORWARD_PREFIXES \
  HPC_PROJECT_ROOT HPC_PROJECT_ROOT_SOURCE HPC_CONTAINER_NO_MOUNT CPA_DEPENDENCY_LOCK \
  REFERENCE_CELL_STATE_PROJECT_BUILDER REFERENCE_MORPHOLOGY_WORKSPACE_SCRIPT; do
  [[ -z "${!forbidden_name+x}" ]] || {
    echo "Ambient override is forbidden for reference-cell-state submission: $forbidden_name" >&2
    exit 2
  }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEPLOYMENT_PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$DEPLOYMENT_PROJECT_DIR}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd -P)"
[[ "$PROJECT_DIR" == "$DEPLOYMENT_PROJECT_DIR" ]] || {
  echo "PROJECT_DIR must be the checkout containing this submitter" >&2
  exit 2
}

resolve_real_directory() {
  local path="$1" label="$2" resolved
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] || {
    echo "$label must be an absolute existing real directory: $path" >&2
    return 2
  }
  resolved="$(cd "$path" && pwd -P)"
  printf '%s\n' "$resolved"
}
RESULTS_ROOT="$(resolve_real_directory "$RESULTS_ROOT" RESULTS_ROOT)"
PARENT_BROAD_SHADOW_ROOT="$(resolve_real_directory "$PARENT_BROAD_SHADOW_ROOT" PARENT_BROAD_SHADOW_ROOT)"
DATASET_ROOT="$(resolve_real_directory "$DATASET_ROOT" DATASET_ROOT)"
SOURCE_SEGMENTATION_ROOT="$(resolve_real_directory "$SOURCE_SEGMENTATION_ROOT" SOURCE_SEGMENTATION_ROOT)"
BRIGHTFIELD_ROOT="$(resolve_real_directory "$DATASET_ROOT/Brightfield" BRIGHTFIELD_ROOT)"
NUCLEI_ROOT="$(resolve_real_directory "$DATASET_ROOT/Nuclei" NUCLEI_ROOT)"
COMBINED_MASK_ROOT="$(resolve_real_directory "$SOURCE_SEGMENTATION_ROOT/Combined/segmentations" COMBINED_MASK_ROOT)"
CPA_REFERENCE_ROOT="$(resolve_real_directory "$CPA_REFERENCE_ROOT" CPA_REFERENCE_ROOT)"
CELL_PHENOTYPE_ANNOTATOR_ROOT="$CPA_REFERENCE_ROOT"
[[ "$HPC_CONTAINER_IMAGE" == /* && -r "$HPC_CONTAINER_IMAGE" && ! -L "$HPC_CONTAINER_IMAGE" ]] || {
  echo "Latest SIF is unavailable or symlinked: $HPC_CONTAINER_IMAGE" >&2
  exit 2
}
[[ "$HPC_CONTAINER_IMAGE" == "$DEFAULT_SIF" && "$CPA_REFERENCE_ROOT" == "$DEFAULT_CPA_REFERENCE_ROOT" ]] || {
  echo "Reference-cell-state execution requires the pinned latest SIF and reference checkout paths" >&2
  exit 2
}

if [[ -z "${REFERENCE_SHADOW_ROOT:-}" ]]; then
  if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    REFERENCE_SHADOW_ROOT="$RESULTS_ROOT/reference_cell_state_shadow_$RUN_STAMP"
  else
    REFERENCE_SHADOW_ROOT="$RESULTS_ROOT/Tests_and_Parameters_calibration/reference_cell_state_shadow_test_$RUN_STAMP"
  fi
fi
[[ "$REFERENCE_SHADOW_ROOT" == /* && ! -e "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]] || {
  echo "REFERENCE_SHADOW_ROOT must be a new absolute path: $REFERENCE_SHADOW_ROOT" >&2
  exit 2
}
if [[ "$EXECUTION_MODE" == "slurm" ]]; then
  [[ "$RESULTS_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/results" && \
     "$DATASET_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/20260626_SUM159_AC_Exp1_SeparateImages" && \
     "$SOURCE_SEGMENTATION_ROOT" == "$DEFAULT_EXPERIMENT_ROOT/results/full_fusion_shape_strict_20260711_155940" ]] || {
    echo "Formal reference-cell-state input and result roots must match the frozen SUM159 experiment contract" >&2
    exit 2
  }
  [[ "$PARENT_BROAD_SHADOW_ROOT" == "$DEFAULT_PARENT_BROAD_ROOT" ]] || {
    echo "Formal reference-cell-state analysis is pinned to the completed parent broad root: $DEFAULT_PARENT_BROAD_ROOT" >&2
    exit 2
  }
  [[ "$EXPECTED_PARENT_CELL_COUNT" == "32000" ]] || {
    echo "Formal reference-cell-state analysis requires exactly 32000 development cells" >&2
    exit 2
  }
  [[ "$REFERENCE_CELL_STATE_SEED" == "20260812" && "$REFERENCE_MAX_REPRESENTATIVES" == "300" ]] || {
    echo "Formal reference-cell-state seed and representative cap are frozen at 20260812 and 300" >&2
    exit 2
  }
  [[ "$REFERENCE_QOS" == "xxlarge" && "$REFERENCE_CPUS" == "8" && \
     "$REFERENCE_MEM" == "128G" && "$REFERENCE_TIME" == "12:00:00" ]] || {
    echo "Formal Phase A resources are frozen at xxlarge/8 CPU/128G/12:00:00" >&2
    exit 2
  }
  case "$REFERENCE_SHADOW_ROOT" in
    "$RESULTS_ROOT"/reference_cell_state_shadow_*) ;;
    *) echo "Formal output must be a reference_cell_state_shadow_* child of RESULTS_ROOT" >&2; exit 2 ;;
  esac
else
  calibration_parent="$RESULTS_ROOT/Tests_and_Parameters_calibration"
  [[ -d "$calibration_parent" && ! -L "$calibration_parent" ]] || {
    echo "Calibration parent is unavailable: $calibration_parent" >&2
    exit 2
  }
  case "$REFERENCE_SHADOW_ROOT" in
    "$calibration_parent"/reference_cell_state_shadow_test_*) ;;
    *) echo "Direct tests must write under Tests_and_Parameters_calibration/reference_cell_state_shadow_test_*" >&2; exit 2 ;;
  esac
fi

PARENT_PROJECT="$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/project.yml"
PARENT_PROJECTION_INPUT_MANIFEST="$PARENT_BROAD_SHADOW_ROOT/projection_input/projection_input_manifest.json"
PARENT_IMAGES="$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/images.tsv"
[[ -s "$PARENT_PROJECT" && ! -L "$PARENT_PROJECT" ]] || {
  echo "Parent representative project is unavailable: $PARENT_PROJECT" >&2
  exit 2
}
[[ -s "$PARENT_PROJECTION_INPUT_MANIFEST" && ! -L "$PARENT_PROJECTION_INPUT_MANIFEST" ]] || {
  echo "Parent projection-input manifest is unavailable: $PARENT_PROJECTION_INPUT_MANIFEST" >&2
  exit 2
}
[[ -s "$PARENT_IMAGES" && ! -L "$PARENT_IMAGES" ]] || {
  echo "Parent review images table is unavailable: $PARENT_IMAGES" >&2
  exit 2
}
parent_umap_manifests=()
while IFS= read -r parent_umap_manifest; do
  parent_umap_manifests+=("$parent_umap_manifest")
done < <(
  find "$PARENT_BROAD_SHADOW_ROOT/projection_input/representative_umap/runs" \
    -type f -name umap_manifest.json -print | LC_ALL=C sort
)
[[ "${#parent_umap_manifests[@]}" -eq 1 ]] || {
  echo "Expected exactly one frozen parent UMAP manifest; observed=${#parent_umap_manifests[@]}" >&2
  exit 2
}
PARENT_UMAP_MANIFEST="${parent_umap_manifests[0]}"

export GIT_OPTIONAL_LOCKS=0
project_git_status="$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
[[ -z "$project_git_status" ]] || {
  echo "Reference-cell-state submission requires a clean project checkout" >&2
  printf '%s\n' "$project_git_status" >&2
  exit 2
}
project_git_sha="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
project_git_tree="$(git -C "$PROJECT_DIR" rev-parse "${project_git_sha}^{tree}")"
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$project_git_sha" ]] || {
  echo "Project HEAD changed during submission preflight" >&2
  exit 2
}
cpa_commit="$(git -C "$CPA_REFERENCE_ROOT" rev-parse HEAD)"
cpa_tree="$(git -C "$CPA_REFERENCE_ROOT" rev-parse "${cpa_commit}^{tree}")"
cpa_status="$(git -C "$CPA_REFERENCE_ROOT" status --porcelain --untracked-files=all)"
[[ "$cpa_commit" == "$EXPECTED_CPA_COMMIT" && "$cpa_tree" == "$EXPECTED_CPA_TREE" && -z "$cpa_status" ]] || {
  echo "Pinned Cell Phenotype Annotator checkout identity mismatch" >&2
  exit 2
}

if [[ "$EXECUTION_MODE" == "slurm" && "$DRY_RUN_SUBMIT" == "0" ]]; then
  SBATCH_BIN="$(command -v sbatch || true)"
  [[ "$SBATCH_BIN" == /* && -x "$SBATCH_BIN" ]] || { echo "sbatch is unavailable" >&2; exit 127; }
else
  SBATCH_BIN="${SBATCH_BIN:-/usr/bin/false}"
fi
SYSTEM_BASH_BIN="$(command -v bash)"
[[ "$SYSTEM_BASH_BIN" == /* && -x "$SYSTEM_BASH_BIN" ]] || { echo "bash is unavailable" >&2; exit 127; }
SHA256SUM_BIN="$(command -v sha256sum || true)"
[[ "$SHA256SUM_BIN" == /* && -x "$SHA256SUM_BIN" ]] || { echo "sha256sum is unavailable" >&2; exit 127; }
EXPECTED_PARENT_PROJECT_SHA256="$($SHA256SUM_BIN "$PARENT_PROJECT")"
EXPECTED_PARENT_PROJECT_SHA256="${EXPECTED_PARENT_PROJECT_SHA256%%[[:space:]]*}"
EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256="$($SHA256SUM_BIN "$PARENT_PROJECTION_INPUT_MANIFEST")"
EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256="${EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256%%[[:space:]]*}"
EXPECTED_PARENT_UMAP_MANIFEST_SHA256="$($SHA256SUM_BIN "$PARENT_UMAP_MANIFEST")"
EXPECTED_PARENT_UMAP_MANIFEST_SHA256="${EXPECTED_PARENT_UMAP_MANIFEST_SHA256%%[[:space:]]*}"
EXPECTED_PARENT_IMAGES_SHA256="$($SHA256SUM_BIN "$PARENT_IMAGES")"
EXPECTED_PARENT_IMAGES_SHA256="${EXPECTED_PARENT_IMAGES_SHA256%%[[:space:]]*}"
for frozen_sha in \
  "$EXPECTED_PARENT_PROJECT_SHA256" \
  "$EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256" \
  "$EXPECTED_PARENT_UMAP_MANIFEST_SHA256" \
  "$EXPECTED_PARENT_IMAGES_SHA256"; do
  [[ "$frozen_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid parent input SHA-256 freeze" >&2; exit 2; }
done

bootstrap_cleanup_armed=0
cleanup_bootstrap() {
  local status=$?
  if [[ "$bootstrap_cleanup_armed" == "1" && -d "$REFERENCE_SHADOW_ROOT" && ! -L "$REFERENCE_SHADOW_ROOT" ]]; then
    marker="$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_bootstrap_owner"
    if [[ -f "$marker" && "$(cat "$marker")" == "$$" ]]; then
      snapshot="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot"
      [[ ! -d "$snapshot" ]] || chmod -R u+w "$snapshot" 2>/dev/null || true
      rm -rf -- "$REFERENCE_SHADOW_ROOT"
    fi
  fi
  exit "$status"
}
trap cleanup_bootstrap EXIT

mkdir "$REFERENCE_SHADOW_ROOT"
mkdir -p "$REFERENCE_SHADOW_ROOT/workflow_status" "$REFERENCE_SHADOW_ROOT/logs"
printf '%s\n' "$$" > "$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_bootstrap_owner"
bootstrap_cleanup_armed=1

CODE_SNAPSHOT_ARCHIVE="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot.tar"
CODE_SNAPSHOT_ROOT="$REFERENCE_SHADOW_ROOT/workflow_status/code_snapshot"
snapshot_staging="$REFERENCE_SHADOW_ROOT/workflow_status/.code_snapshot.tar.tmp.$$"
git -C "$PROJECT_DIR" archive --format=tar "$project_git_sha" > "$snapshot_staging"
mv "$snapshot_staging" "$CODE_SNAPSHOT_ARCHIVE"
code_snapshot_archive_sha256="$($SHA256SUM_BIN "$CODE_SNAPSHOT_ARCHIVE")"
code_snapshot_archive_sha256="${code_snapshot_archive_sha256%%[[:space:]]*}"
mkdir "$CODE_SNAPSHOT_ROOT"
tar -xf "$CODE_SNAPSHOT_ARCHIVE" -C "$CODE_SNAPSHOT_ROOT"
chmod -R a-w "$CODE_SNAPSHOT_ROOT" 2>/dev/null || true
PROJECT_DIR="$CODE_SNAPSHOT_ROOT"

HPC_CONTAINER_IDENTITY_FILE="$REFERENCE_SHADOW_ROOT/workflow_status/hpc_container_identity.json"
source "$CODE_SNAPSHOT_ROOT/cellpose_pipeline/Docker/hpc/util/broad_phenotype_container_identity.sh"
broad_phenotype_capture_container_identity \
  "$HPC_CONTAINER_IMAGE" \
  "$EXPECTED_SIF_SHA256" \
  "$HPC_CONTAINER_IDENTITY_FILE"
HPC_CONTAINER_IDENTITY_FILE_SHA256="$($SHA256SUM_BIN "$HPC_CONTAINER_IDENTITY_FILE")"
HPC_CONTAINER_IDENTITY_FILE_SHA256="${HPC_CONTAINER_IDENTITY_FILE_SHA256%%[[:space:]]*}"

HPC_CONTAINER_GPU=0
HPC_PROJECT_ROOT_BIND_MODE=ro
HPC_CONTAINER_NO_MOUNT=/share
HPC_PROJECT_ROOT="$CODE_SNAPSHOT_ROOT"
HPC_CONTAINER_RUNTIME_ROOT="$CODE_SNAPSHOT_ROOT/cellpose_pipeline/Docker/hpc"
HPC_CONTAINER_FORWARD_PREFIXES="PYTHONNOUSERSITE,KMP_DUPLICATE_LIB_OK,MPLCONFIGDIR,CUDA_VISIBLE_DEVICES"
HPC_CONTAINER_BINDS="$REFERENCE_SHADOW_ROOT:$REFERENCE_SHADOW_ROOT,$PARENT_BROAD_SHADOW_ROOT:$PARENT_BROAD_SHADOW_ROOT:ro,$BRIGHTFIELD_ROOT:$BRIGHTFIELD_ROOT:ro,$NUCLEI_ROOT:$NUCLEI_ROOT:ro,$COMBINED_MASK_ROOT:$COMBINED_MASK_ROOT:ro,$CPA_REFERENCE_ROOT:$CPA_REFERENCE_ROOT:ro"
WORKER="$CODE_SNAPSHOT_ROOT/cellpose_pipeline/Docker/hpc/run_reference_cell_state_phase_a.sh"
[[ -x "$WORKER" ]] || { echo "Frozen snapshot lacks Phase A worker: $WORKER" >&2; exit 2; }

preflight="$REFERENCE_SHADOW_ROOT/workflow_status/submission_preflight.tsv"
{
  printf 'property\tvalue\n'
  printf 'schema_version\treference_cell_state_submission_preflight_v1\n'
  printf 'execution_mode\t%s\n' "$EXECUTION_MODE"
  printf 'reference_shadow_root\t%s\n' "$REFERENCE_SHADOW_ROOT"
  printf 'parent_broad_shadow_root\t%s\n' "$PARENT_BROAD_SHADOW_ROOT"
  printf 'parent_project\t%s\n' "$PARENT_PROJECT"
  printf 'parent_project_sha256\t%s\n' "$EXPECTED_PARENT_PROJECT_SHA256"
  printf 'parent_projection_input_manifest\t%s\n' "$PARENT_PROJECTION_INPUT_MANIFEST"
  printf 'parent_projection_input_manifest_sha256\t%s\n' "$EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256"
  printf 'parent_umap_manifest\t%s\n' "$PARENT_UMAP_MANIFEST"
  printf 'parent_umap_manifest_sha256\t%s\n' "$EXPECTED_PARENT_UMAP_MANIFEST_SHA256"
  printf 'parent_images\t%s\n' "$PARENT_IMAGES"
  printf 'parent_images_sha256\t%s\n' "$EXPECTED_PARENT_IMAGES_SHA256"
  printf 'expected_parent_cell_count\t%s\n' "$EXPECTED_PARENT_CELL_COUNT"
  printf 'dataset_root\t%s\n' "$DATASET_ROOT"
  printf 'source_segmentation_root\t%s\n' "$SOURCE_SEGMENTATION_ROOT"
  printf 'project_git_sha\t%s\n' "$project_git_sha"
  printf 'project_git_tree\t%s\n' "$project_git_tree"
  printf 'code_snapshot_archive_sha256\t%s\n' "$code_snapshot_archive_sha256"
  printf 'hpc_container_image\t%s\n' "$HPC_CONTAINER_IMAGE"
  printf 'hpc_container_sha256\t%s\n' "$EXPECTED_SIF_SHA256"
  printf 'cpa_reference_root\t%s\n' "$CPA_REFERENCE_ROOT"
  printf 'cpa_reference_commit\t%s\n' "$cpa_commit"
  printf 'cpa_reference_tree\t%s\n' "$cpa_tree"
  printf 'reference_cell_state_seed\t%s\n' "$REFERENCE_CELL_STATE_SEED"
  printf 'reference_max_representatives\t%s\n' "$REFERENCE_MAX_REPRESENTATIVES"
  printf 'class_ontology\tlive_cell,dead_cell,multinucleated_cell\n'
  printf 'decision_priority\tmultinucleated_cell>dead_cell>live_cell\n'
  printf 'training_feature_profile\thistorical_promoted_shape_9\n'
  printf 'current_classification_root_bound\tfalse\n'
  printf 'dead_channel_bound\tfalse\n'
  printf 'site_share_mount_disabled\ttrue\n'
} > "$preflight"

export PROJECT_DIR HPC_CONTAINER_IMAGE HPC_CONTAINER_GPU HPC_PROJECT_ROOT_BIND_MODE HPC_CONTAINER_NO_MOUNT
export HPC_CONTAINER_BINDS HPC_CONTAINER_FORWARD_PREFIXES HPC_CONTAINER_RUNTIME_ROOT
export HPC_CONTAINER_IDENTITY_FILE HPC_CONTAINER_IDENTITY_FILE_SHA256 HPC_PROJECT_ROOT
export REFERENCE_SHADOW_ROOT PARENT_BROAD_SHADOW_ROOT EXPECTED_PARENT_CELL_COUNT
export PARENT_PROJECT PARENT_PROJECTION_INPUT_MANIFEST PARENT_UMAP_MANIFEST
export PARENT_IMAGES EXPECTED_PARENT_PROJECT_SHA256 EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256 EXPECTED_PARENT_UMAP_MANIFEST_SHA256 EXPECTED_PARENT_IMAGES_SHA256
export DATASET_ROOT SOURCE_SEGMENTATION_ROOT CPA_REFERENCE_ROOT CELL_PHENOTYPE_ANNOTATOR_ROOT
export REFERENCE_CELL_STATE_SEED REFERENCE_MAX_REPRESENTATIVES

run_direct() {
  bootstrap_cleanup_armed=0
  rm -f "$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_bootstrap_owner"
  SLURM_CPUS_PER_TASK="$REFERENCE_CPUS" "$WORKER"
}

worker_wrap_command() {
  local worker_relative="cellpose_pipeline/Docker/hpc/run_reference_cell_state_phase_a.sh"
  local program
  program='set -euo pipefail
export PATH=/usr/bin:/bin:/sbin
archive="$1"; expected="$2"; canonical="$3"; relative="$4"; shell_bin="$5"
tmp_base="${SLURM_TMPDIR:-/tmp}"
[[ "$tmp_base" == /* && -d "$tmp_base" ]] || { echo "SLURM_TMPDIR is invalid" >&2; exit 2; }
tmp_parent="$(mktemp -d "$tmp_base/reference-cell-state-code.XXXXXX")"
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
mkdir "$local_project"; tar -xf "$local_archive" -C "$local_project"
local_worker="$local_project/$relative"
[[ -x "$local_worker" && ! -L "$local_worker" ]] || { echo "Node-local worker unavailable" >&2; exit 2; }
chmod -R a-w "$local_project"
if find "$local_project" \( -perm -0200 -o -perm -0020 -o -perm -0002 \) -print -quit | grep -q .; then
  echo "Node-local code snapshot still contains a writable path" >&2
  exit 2
fi
export HPC_PROJECT_ROOT_SOURCE="$local_project"
export PROJECT_DIR="$canonical"
export HPC_PROJECT_ROOT="$canonical"
export HPC_CONTAINER_RUNTIME_ROOT="$local_project/cellpose_pipeline/Docker/hpc"
"$shell_bin" "$local_worker"'
  printf 'exec %q --noprofile --norc -c %q reference-cell-state-node-local %q %q %q %q %q' \
    "$SYSTEM_BASH_BIN" "$program" "$CODE_SNAPSHOT_ARCHIVE" \
    "$code_snapshot_archive_sha256" "$CODE_SNAPSHOT_ROOT" "$worker_relative" "$SYSTEM_BASH_BIN"
}

slurm_exports="PROJECT_DIR,HPC_CONTAINER_IMAGE,HPC_CONTAINER_GPU,HPC_PROJECT_ROOT_BIND_MODE,HPC_CONTAINER_NO_MOUNT,HPC_CONTAINER_BINDS,HPC_CONTAINER_FORWARD_PREFIXES,HPC_CONTAINER_RUNTIME_ROOT,HPC_CONTAINER_IDENTITY_FILE,HPC_CONTAINER_IDENTITY_FILE_SHA256,HPC_PROJECT_ROOT,REFERENCE_SHADOW_ROOT,PARENT_BROAD_SHADOW_ROOT,EXPECTED_PARENT_CELL_COUNT,PARENT_PROJECT,PARENT_PROJECTION_INPUT_MANIFEST,PARENT_UMAP_MANIFEST,PARENT_IMAGES,EXPECTED_PARENT_PROJECT_SHA256,EXPECTED_PARENT_PROJECTION_MANIFEST_SHA256,EXPECTED_PARENT_UMAP_MANIFEST_SHA256,EXPECTED_PARENT_IMAGES_SHA256,DATASET_ROOT,SOURCE_SEGMENTATION_ROOT,CPA_REFERENCE_ROOT,CELL_PHENOTYPE_ANNOTATOR_ROOT,REFERENCE_CELL_STATE_SEED,REFERENCE_MAX_REPRESENTATIVES"
cpu_sbatch() {
  local name old_ifs="$IFS"
  local -a clean_env=(/usr/bin/env -i PATH=/usr/bin:/bin:/sbin)
  IFS=,
  for name in $slurm_exports; do
    [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ && -n "${!name+x}" ]] || { IFS="$old_ifs"; echo "Undefined Slurm export: $name" >&2; return 2; }
    clean_env+=("$name=${!name}")
  done
  IFS="$old_ifs"
  "${clean_env[@]}" "$SBATCH_BIN" --parsable "$@"
}

if [[ "$EXECUTION_MODE" == "direct_test" ]]; then
  run_direct
  trap - EXIT
  exit 0
fi

wrap="$(worker_wrap_command)"
dependency_args=()
[[ -z "$DEPENDENCY_JOB_ID" ]] || dependency_args+=(--dependency "afterok:$DEPENDENCY_JOB_ID")
submit_args=(
  --job-name reference_cell_state_phase_a
  --qos "$REFERENCE_QOS"
  --cpus-per-task "$REFERENCE_CPUS"
  --mem "$REFERENCE_MEM"
  --time "$REFERENCE_TIME"
  --output "$REFERENCE_SHADOW_ROOT/logs/%j.reference_cell_state_phase_a.out"
  --error "$REFERENCE_SHADOW_ROOT/logs/%j.reference_cell_state_phase_a.err"
  --chdir "$REFERENCE_SHADOW_ROOT"
  --export "$slurm_exports"
  --wrap "$wrap"
  "${dependency_args[@]}"
)

if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
  job_id="DRYRUN_REFERENCE_CELL_STATE_PHASE_A"
else
  job_id="$(cpu_sbatch "${submit_args[@]}")"
  job_id="${job_id%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job id: $job_id" >&2; exit 2; }
fi

bootstrap_cleanup_armed=0
rm -f "$REFERENCE_SHADOW_ROOT/workflow_status/.reference_cell_state_bootstrap_owner"
trap - EXIT
{
  printf 'property\tvalue\n'
  printf 'schema_version\treference_cell_state_submission_summary_v1\n'
  printf 'phase_a_job_id\t%s\n' "$job_id"
  printf 'reference_shadow_root\t%s\n' "$REFERENCE_SHADOW_ROOT"
  printf 'human_barrier\tregion_submission_required_after_phase_a\n'
} > "$REFERENCE_SHADOW_ROOT/SUBMISSION_SUMMARY.tsv"

echo "reference_cell_state_phase_a_job_id=$job_id"
echo "reference_shadow_root=$REFERENCE_SHADOW_ROOT"
echo "human_barrier=region_submission_required_after_phase_a"
