#!/bin/bash
set -euo pipefail

# Re-run only the full-time-course classification stage against immutable
# segmentation outputs. The calibrated d0 method is implemented as the default
# behavior of scripts/08_fuse_multichannel_classification.py and therefore
# applies to every selected field without passing a d0-only option here.

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$BASE/results/full_fusion_shape_strict_20260711_155940}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/classification_$STAMP}"

SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/hpc"
MANIFEST_WORKER="${MANIFEST_WORKER:-$SCRIPT_DIR/run_postsegmentation_manifest.sh}"
CLASSIFICATION_WORKER="${CLASSIFICATION_WORKER:-$SCRIPT_DIR/run_multichannel_classification_fusion_array_task.sh}"
MERGE_WORKER="${MERGE_WORKER:-$SCRIPT_DIR/run_multichannel_classification_fusion_merge.sh}"
PLOT_WORKER="${PLOT_WORKER:-$SCRIPT_DIR/analysisi/05_run_well_count_timecourse_plots.sh}"

FIELD_MANIFEST_DIR="$OUT_ROOT/workflow_status/postsegmentation_manifest"
TASK_DIR="$OUT_ROOT/workflow_status/task_lists"
TASK_LIST_FUSION="$TASK_DIR/classification_keys.tsv"
LOG_DIR="$OUT_ROOT/logs"
ORIGINAL_OUT_DIR="$OUT_ROOT/classification_fusion"
NUCLEATED_OUT_DIR="$OUT_ROOT/classification_fusion_nucleated_only"
NUCLEATED_BRANCH_ROOT="$SOURCE_RUN_ROOT/nucleated_only"
SUBMISSION_SUMMARY="$OUT_ROOT/SUBMISSION_SUMMARY.txt"
ORIGINAL_PLOT_DIR="$OUT_ROOT/analysis/well_count_timecourses/fusion"
NUCLEATED_PLOT_DIR="$OUT_ROOT/analysis/well_count_timecourses/fusion-nucleated-only"

EXPECTED_FIELDS="${EXPECTED_FIELDS:-27200}"
JOB_QOS="${JOB_QOS:-xxlarge}"
JOB_PARTITION="${JOB_PARTITION:-}"
JOB_ACCOUNT="${JOB_ACCOUNT:-}"
SBATCH_JOB_PREFIX="${SBATCH_JOB_PREFIX:-cpsam_classification_full}"
CLASSIFICATION_TIME="${CLASSIFICATION_TIME:-12:00:00}"
CLASSIFICATION_CPUS="${CLASSIFICATION_CPUS:-1}"
CLASSIFICATION_MEM="${CLASSIFICATION_MEM:-4G}"
MERGE_TIME="${MERGE_TIME:-12:00:00}"
MERGE_CPUS="${MERGE_CPUS:-1}"
MERGE_MEM="${MERGE_MEM:-8G}"
MANIFEST_TIME="${MANIFEST_TIME:-12:00:00}"
MANIFEST_CPUS="${MANIFEST_CPUS:-1}"
MANIFEST_MEM="${MANIFEST_MEM:-4G}"
PLOT_TIME="${PLOT_TIME:-02:00:00}"
PLOT_CPUS="${PLOT_CPUS:-1}"
PLOT_MEM="${PLOT_MEM:-8G}"
EXPECTED_TIMEPOINTS="${EXPECTED_TIMEPOINTS:-85}"
EXPECTED_SITES="${EXPECTED_SITES:-4}"
PLOT_DPI="${PLOT_DPI:-200}"
FORCE_FUSION="${FORCE_FUSION:-1}"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"

required_directories=(
  "$PROJECT_DIR"
  "$INPUT_ROOT/Combined"
  "$INPUT_ROOT/Brightfield"
  "$INPUT_ROOT/Dead"
  "$INPUT_ROOT/Nuclei"
  "$SOURCE_RUN_ROOT/Combined/segmentations"
  "$SOURCE_RUN_ROOT/Brightfield/segmentations"
  "$SOURCE_RUN_ROOT/Dead/segmentations"
  "$SOURCE_RUN_ROOT/Nuclei/segmentations"
  "$SOURCE_RUN_ROOT/Nuclei/nucleus_core_seeds"
  "$NUCLEATED_BRANCH_ROOT/Combined/segmentations"
  "$NUCLEATED_BRANCH_ROOT/Brightfield/segmentations"
)
for required in "${required_directories[@]}"; do
  if [[ ! -d "$required" ]]; then
    echo "Required classification source directory is missing: $required" >&2
    exit 2
  fi
done
for worker in "$MANIFEST_WORKER" "$CLASSIFICATION_WORKER" "$MERGE_WORKER" "$PLOT_WORKER"; do
  if [[ ! -x "$worker" ]]; then
    echo "Required production worker is missing or not executable: $worker" >&2
    exit 2
  fi
done
if [[ ! "$EXPECTED_FIELDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_FIELDS must be a positive integer: $EXPECTED_FIELDS" >&2
  exit 2
fi
if [[ ! "$EXPECTED_TIMEPOINTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_TIMEPOINTS must be a positive integer: $EXPECTED_TIMEPOINTS" >&2
  exit 2
fi
if [[ ! "$EXPECTED_SITES" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_SITES must be a positive integer: $EXPECTED_SITES" >&2
  exit 2
fi
if [[ "$DRY_RUN_SUBMIT" != "0" && "$DRY_RUN_SUBMIT" != "1" ]]; then
  echo "DRY_RUN_SUBMIT must be 0 or 1: $DRY_RUN_SUBMIT" >&2
  exit 2
fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to reuse an existing classification output root: $OUT_ROOT" >&2
  exit 2
fi

mkdir -p "$TASK_DIR" "$LOG_DIR" "$ORIGINAL_OUT_DIR" "$NUCLEATED_OUT_DIR"
temporary_task_list="$TASK_LIST_FUSION.tmp.$$"
find "$INPUT_ROOT/Combined" -maxdepth 1 -type f \( \
    -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
  \) -printf '%f\n' | LC_ALL=C sort | awk '
  {
    name = $0
    sub(/\.[^.]*$/, "", name)
    key = name
    sub(/^SUM159_AC_/, "", key)
    sub(/^Exp1_/, "", key)
    sub(/^BF_/, "", key)
    sub(/^Dead_/, "", key)
    if (key ~ /^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$/) {
      print key
    } else {
      print "Could not extract classification key from Combined image: " $0 > "/dev/stderr"
      bad = 1
    }
  }
  END { exit bad ? 1 : 0 }
' > "$temporary_task_list"

raw_task_count="$(awk 'NF {n++} END {print n+0}' "$temporary_task_list")"
LC_ALL=C sort -u "$temporary_task_list" > "$TASK_LIST_FUSION"
rm -f "$temporary_task_list"
N_TASKS="$(awk 'NF {n++} END {print n+0}' "$TASK_LIST_FUSION")"
if [[ "$raw_task_count" -ne "$N_TASKS" ]]; then
  echo "Duplicate classification keys detected: raw=$raw_task_count unique=$N_TASKS" >&2
  exit 2
fi
if [[ "$N_TASKS" -ne "$EXPECTED_FIELDS" ]]; then
  echo "Expected $EXPECTED_FIELDS classification fields but found $N_TASKS" >&2
  exit 2
fi
ARRAY_SPEC="1-$N_TASKS"

COMMON_SBATCH_ARGS=()
if [[ -n "$JOB_QOS" ]]; then
  COMMON_SBATCH_ARGS+=(--qos "$JOB_QOS")
fi
if [[ -n "$JOB_PARTITION" ]]; then
  COMMON_SBATCH_ARGS+=(--partition "$JOB_PARTITION")
fi
if [[ -n "$JOB_ACCOUNT" ]]; then
  COMMON_SBATCH_ARGS+=(--account "$JOB_ACCOUNT")
fi

MANIFEST_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_manifest"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$MANIFEST_TIME"
  --cpus-per-task "$MANIFEST_CPUS"
  --mem "$MANIFEST_MEM"
)
CLASSIFICATION_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --array "$ARRAY_SPEC"
  --output "$LOG_DIR/%x_%A_%a.out"
  --error "$LOG_DIR/%x_%A_%a.err"
  --time "$CLASSIFICATION_TIME"
  --cpus-per-task "$CLASSIFICATION_CPUS"
  --mem "$CLASSIFICATION_MEM"
)
MERGE_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$MERGE_TIME"
  --cpus-per-task "$MERGE_CPUS"
  --mem "$MERGE_MEM"
)
PLOT_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_well_counts"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$PLOT_TIME"
  --cpus-per-task "$PLOT_CPUS"
  --mem "$PLOT_MEM"
)

write_submission_summary() {
  {
    echo "classification_only=1"
    echo "prepared_at=$(date --iso-8601=seconds)"
    echo "submission_mode=$([[ "$DRY_RUN_SUBMIT" == "1" ]] && echo dry_run || echo submitted)"
    echo "project_dir=$PROJECT_DIR"
    echo "project_git_sha=$(git -c safe.directory="$PROJECT_DIR" -C "$PROJECT_DIR" rev-parse HEAD)"
    echo "input_root=$INPUT_ROOT"
    echo "source_run_root=$SOURCE_RUN_ROOT"
    echo "output_root=$OUT_ROOT"
    echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
    echo "task_list=$TASK_LIST_FUSION"
    echo "task_count=$N_TASKS"
    echo "array_spec=$ARRAY_SPEC"
    echo "original_output=$ORIGINAL_OUT_DIR"
    echo "nucleated_output=$NUCLEATED_OUT_DIR"
    echo "original_well_count_plot=$ORIGINAL_PLOT_DIR/well_live_dead_counts_over_time.png"
    echo "nucleated_well_count_plot=$NUCLEATED_PLOT_DIR/well_live_dead_counts_over_time.png"
    echo "classification_method=cellpose_pipeline/scripts/08_fuse_multichannel_classification.py"
    echo "classification_timepoint=all"
    echo "classification_cpus_per_task=$CLASSIFICATION_CPUS"
    echo "classification_mem_per_task=$CLASSIFICATION_MEM"
    echo "classification_time=$CLASSIFICATION_TIME"
    echo "classification_qos=$JOB_QOS"
    echo "classification_partition=${JOB_PARTITION:-cluster_default}"
    echo "classification_account=${JOB_ACCOUNT:-cluster_default}"
    echo "classification_gpu=none"
    echo "merge_cpus_per_task=$MERGE_CPUS"
    echo "merge_mem_per_task=$MERGE_MEM"
    echo "merge_time=$MERGE_TIME"
    echo "plot_cpus_per_task=$PLOT_CPUS"
    echo "plot_mem_per_task=$PLOT_MEM"
    echo "plot_time=$PLOT_TIME"
    echo "plot_expected_timepoints=$EXPECTED_TIMEPOINTS"
    echo "plot_expected_sites=$EXPECTED_SITES"
    echo "manifest_job_id=${MANIFEST_JOB_ID:-not_submitted}"
    echo "original_array_job_id=${ORIGINAL_JOB_ID:-not_submitted}"
    echo "original_merge_job_id=${ORIGINAL_MERGE_JOB_ID:-not_submitted}"
    echo "nucleated_array_job_id=${NUCLEATED_JOB_ID:-not_submitted}"
    echo "nucleated_merge_job_id=${NUCLEATED_MERGE_JOB_ID:-not_submitted}"
    echo "well_count_plot_job_id=${PLOT_JOB_ID:-not_submitted}"
  } > "$SUBMISSION_SUMMARY"
}

echo "classification_only=1"
echo "input_root=$INPUT_ROOT"
echo "source_run_root=$SOURCE_RUN_ROOT"
echo "output_root=$OUT_ROOT"
echo "task_count=$N_TASKS"
echo "array_spec=$ARRAY_SPEC"
echo "original_output=$ORIGINAL_OUT_DIR"
echo "nucleated_output=$NUCLEATED_OUT_DIR"
echo "original_well_count_plot=$ORIGINAL_PLOT_DIR/well_live_dead_counts_over_time.png"
echo "nucleated_well_count_plot=$NUCLEATED_PLOT_DIR/well_live_dead_counts_over_time.png"
printf 'manifest_sbatch_arg=%s\n' "${MANIFEST_SBATCH_ARGS[@]}"
printf 'classification_sbatch_arg=%s\n' "${CLASSIFICATION_SBATCH_ARGS[@]}"
printf 'merge_sbatch_arg=%s\n' "${MERGE_SBATCH_ARGS[@]}"
printf 'plot_sbatch_arg=%s\n' "${PLOT_SBATCH_ARGS[@]}"

if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
  write_submission_summary
  echo "dry_run_submit=1"
  echo "dependency_graph=manifest -> {original_array,nucleated_array} -> {original_merge,nucleated_merge} -> well_count_plots"
  exit 0
fi

cpu_sbatch() {
  env \
    -u SBATCH_QOS \
    -u SBATCH_PARTITION \
    -u SBATCH_ACCOUNT \
    -u SBATCH_GRES \
    -u SBATCH_TRES_PER_NODE \
    -u SBATCH_GPUS \
    -u SBATCH_GPUS_PER_NODE \
    -u SBATCH_GPUS_PER_TASK \
    sbatch --parsable "$@"
}

submit_job() {
  local output
  output="$(cpu_sbatch "$@")"
  printf '%s\n' "${output%%;*}"
}

MANIFEST_JOB_ID="$(submit_job "${MANIFEST_SBATCH_ARGS[@]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$SOURCE_RUN_ROOT",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",NUCLEATED_BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",FORCE_FIELD_MANIFEST=1 \
  "$MANIFEST_WORKER")"

ORIGINAL_JOB_ID="$(submit_job "${CLASSIFICATION_SBATCH_ARGS[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_original" \
  --dependency "afterok:$MANIFEST_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$SOURCE_RUN_ROOT",RUN_NAME=.,OUT_DIR="$ORIGINAL_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original,FORCE_FUSION="$FORCE_FUSION",CONTINUE_ON_ERROR=0 \
  "$CLASSIFICATION_WORKER")"

NUCLEATED_JOB_ID="$(submit_job "${CLASSIFICATION_SBATCH_ARGS[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_nucleated" \
  --dependency "afterok:$MANIFEST_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$SOURCE_RUN_ROOT",RUN_NAME=.,OUT_DIR="$NUCLEATED_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated,COMBINED_RUN="$NUCLEATED_BRANCH_ROOT/Combined",BRIGHTFIELD_RUN="$NUCLEATED_BRANCH_ROOT/Brightfield",DEAD_RUN="$SOURCE_RUN_ROOT/Dead",NUCLEI_RUN="$SOURCE_RUN_ROOT/Nuclei",FORCE_FUSION="$FORCE_FUSION",CONTINUE_ON_ERROR=0 \
  "$CLASSIFICATION_WORKER")"

ORIGINAL_MERGE_JOB_ID="$(submit_job "${MERGE_SBATCH_ARGS[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_original_merge" \
  --dependency "afterok:$ORIGINAL_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$SOURCE_RUN_ROOT",OUT_DIR="$ORIGINAL_OUT_DIR",FORCE_FUSION=1 \
  "$MERGE_WORKER")"

NUCLEATED_MERGE_JOB_ID="$(submit_job "${MERGE_SBATCH_ARGS[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_nucleated_merge" \
  --dependency "afterok:$NUCLEATED_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$SOURCE_RUN_ROOT",OUT_DIR="$NUCLEATED_OUT_DIR",FORCE_FUSION=1 \
  "$MERGE_WORKER")"

PLOT_JOB_ID="$(submit_job "${PLOT_SBATCH_ARGS[@]}" \
  --dependency "afterok:$ORIGINAL_MERGE_JOB_ID:$NUCLEATED_MERGE_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$OUT_ROOT",EXPECTED_TIMEPOINTS="$EXPECTED_TIMEPOINTS",EXPECTED_SITES="$EXPECTED_SITES",PLOT_DPI="$PLOT_DPI" \
  "$PLOT_WORKER")"

write_submission_summary

echo "manifest_job_id=$MANIFEST_JOB_ID"
echo "original_array_job_id=$ORIGINAL_JOB_ID"
echo "original_merge_job_id=$ORIGINAL_MERGE_JOB_ID"
echo "nucleated_array_job_id=$NUCLEATED_JOB_ID"
echo "nucleated_merge_job_id=$NUCLEATED_MERGE_JOB_ID"
echo "well_count_plot_job_id=$PLOT_JOB_ID"
echo "submission_summary=$SUBMISSION_SUMMARY"
