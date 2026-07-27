#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

# Re-run only full-time-course classification against immutable segmentation
# outputs. Per-field d0-calibrated classification is followed by the frozen
# late-field-collapse and object-level multi-signal death-rescue stage.

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
hpc_container_ignore_host_runtime "${PYTHON:-}"
PYTHON=python
INPUT_ROOT="${INPUT_ROOT:-$BASE/20260626_SUM159_AC_Exp1_SeparateImages}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$BASE/results/full_fusion_shape_strict_20260711_155940}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/classification_$STAMP}"

SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/Docker/hpc"
MANIFEST_WORKER="${MANIFEST_WORKER:-$SCRIPT_DIR/run_postsegmentation_manifest.sh}"
CLASSIFICATION_WORKER="${CLASSIFICATION_WORKER:-$SCRIPT_DIR/run_multichannel_classification_fusion_array_task.sh}"
MERGE_WORKER="${MERGE_WORKER:-$SCRIPT_DIR/run_multichannel_classification_fusion_merge.sh}"
LATE_DEATH_PREPARE_WORKER="${LATE_DEATH_PREPARE_WORKER:-$SCRIPT_DIR/run_late_dead_trajectory_prepare.sh}"
LATE_DEATH_WELL_WORKER="${LATE_DEATH_WELL_WORKER:-$SCRIPT_DIR/run_late_dead_trajectory_well_array_task.sh}"
LATE_DEATH_FINALIZE_WORKER="${LATE_DEATH_FINALIZE_WORKER:-$SCRIPT_DIR/run_late_dead_trajectory_finalize.sh}"
PLOT_WORKER="${PLOT_WORKER:-$SCRIPT_DIR/analysisi/05_run_well_count_timecourse_plots.sh}"
DOSE_RESPONSE_WORKER="${DOSE_RESPONSE_WORKER:-$SCRIPT_DIR/analysisi/06_run_dose_response_analysis.sh}"
REPORT_WORKER="${REPORT_WORKER:-$SCRIPT_DIR/analysisi/07_run_full_classification_report.sh}"
CALIBRATION_REPORT_WORKER="${CALIBRATION_REPORT_WORKER:-$SCRIPT_DIR/Parameter_calibration/27_generate_late_dead_d0_d5_report.sh}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$BASE/results/Tests_and_Parameters_calibration/death_classification_consensus_optimization_20260725_213646}"
CALIBRATION_CONFIGURATION_DIR="${CALIBRATION_CONFIGURATION_DIR:-$CALIBRATION_ROOT/optimization}"
CALIBRATION_GO_NO_GO="${CALIBRATION_GO_NO_GO:-$CALIBRATION_CONFIGURATION_DIR/FULL_CLASSIFICATION_GO_NO_GO.json}"
SEGMENTATION_FREEZE_MANIFEST="${SEGMENTATION_FREEZE_MANIFEST:-$PROJECT_DIR/cellpose_pipeline/configs/segmentation_freeze_v3_20260725.json}"
SEGMENTATION_FREEZE_VERIFIER="${SEGMENTATION_FREEZE_VERIFIER:-$PROJECT_DIR/cellpose_pipeline/scripts/15_verify_segmentation_freeze.py}"

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
DOSE_RESPONSE_DIR="$OUT_ROOT/analysis/dose_response"
FULL_REPORT_DIR="$OUT_ROOT/analysis/reports"
SEGMENTATION_FREEZE_RECEIPT="$OUT_ROOT/workflow_status/segmentation_freeze_verification.json"

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
LATE_DEATH_PREPARE_TIME="${LATE_DEATH_PREPARE_TIME:-12:00:00}"
LATE_DEATH_PREPARE_CPUS="${LATE_DEATH_PREPARE_CPUS:-32}"
LATE_DEATH_PREPARE_MEM="${LATE_DEATH_PREPARE_MEM:-256G}"
LATE_DEATH_WELL_TIME="${LATE_DEATH_WELL_TIME:-04:00:00}"
LATE_DEATH_WELL_CPUS="${LATE_DEATH_WELL_CPUS:-1}"
LATE_DEATH_WELL_MEM="${LATE_DEATH_WELL_MEM:-48G}"
LATE_DEATH_FINALIZE_TIME="${LATE_DEATH_FINALIZE_TIME:-06:00:00}"
LATE_DEATH_FINALIZE_CPUS="${LATE_DEATH_FINALIZE_CPUS:-1}"
LATE_DEATH_FINALIZE_MEM="${LATE_DEATH_FINALIZE_MEM:-32G}"
MANIFEST_TIME="${MANIFEST_TIME:-12:00:00}"
MANIFEST_CPUS="${MANIFEST_CPUS:-1}"
MANIFEST_MEM="${MANIFEST_MEM:-4G}"
PLOT_TIME="${PLOT_TIME:-02:00:00}"
PLOT_CPUS="${PLOT_CPUS:-1}"
PLOT_MEM="${PLOT_MEM:-8G}"
DOSE_RESPONSE_TIME="${DOSE_RESPONSE_TIME:-04:00:00}"
DOSE_RESPONSE_CPUS="${DOSE_RESPONSE_CPUS:-1}"
DOSE_RESPONSE_MEM="${DOSE_RESPONSE_MEM:-16G}"
REPORT_TIME="${REPORT_TIME:-06:00:00}"
REPORT_CPUS="${REPORT_CPUS:-4}"
REPORT_MEM="${REPORT_MEM:-32G}"
CALIBRATION_REPORT_TIME="${CALIBRATION_REPORT_TIME:-04:00:00}"
CALIBRATION_REPORT_CPUS="${CALIBRATION_REPORT_CPUS:-4}"
CALIBRATION_REPORT_MEM="${CALIBRATION_REPORT_MEM:-32G}"
EXPECTED_TIMEPOINTS="${EXPECTED_TIMEPOINTS:-85}"
EXPECTED_SITES="${EXPECTED_SITES:-4}"
EXPECTED_WELLS="${EXPECTED_WELLS:-80}"
EXPECTED_DOSE_RESPONSE_FILES="${EXPECTED_DOSE_RESPONSE_FILES:-123}"
PLOT_DPI="${PLOT_DPI:-200}"
DOSE_DPI="${DOSE_DPI:-220}"
FORCE_FUSION="${FORCE_FUSION:-1}"
FORCE_LATE_DEATH="${FORCE_LATE_DEATH:-0}"
FORCE_DOSE_RESPONSE="${FORCE_DOSE_RESPONSE:-0}"
FORCE_REPORT="${FORCE_REPORT:-0}"
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
  "$CALIBRATION_CONFIGURATION_DIR"
)
for required in "${required_directories[@]}"; do
  if [[ ! -d "$required" ]]; then
    echo "Required classification source directory is missing: $required" >&2
    exit 2
  fi
done
for worker in \
  "$MANIFEST_WORKER" \
  "$CLASSIFICATION_WORKER" \
  "$MERGE_WORKER" \
  "$LATE_DEATH_PREPARE_WORKER" \
  "$LATE_DEATH_WELL_WORKER" \
  "$LATE_DEATH_FINALIZE_WORKER" \
  "$PLOT_WORKER" \
  "$DOSE_RESPONSE_WORKER" \
  "$REPORT_WORKER" \
  "$CALIBRATION_REPORT_WORKER"; do
  if [[ ! -x "$worker" ]]; then
    echo "Required production worker is missing or not executable: $worker" >&2
    exit 2
  fi
done
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Container Python runtime is unavailable: $PYTHON" >&2
  exit 2
}
for required_file in \
  "$CALIBRATION_GO_NO_GO" \
  "$SEGMENTATION_FREEZE_MANIFEST" \
  "$SEGMENTATION_FREEZE_VERIFIER"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required classification validation input is missing: $required_file" >&2
    exit 2
  fi
done
calibration_decision="$(
  "$PYTHON" - "$CALIBRATION_GO_NO_GO" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision", "MISSING"))
PY
)"
if [[ "$calibration_decision" != "GO" ]]; then
  echo "Full classification is blocked by calibration decision: $calibration_decision" >&2
  exit 2
fi
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
if [[ ! "$EXPECTED_WELLS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_WELLS must be a positive integer: $EXPECTED_WELLS" >&2
  exit 2
fi
if [[ ! "$EXPECTED_DOSE_RESPONSE_FILES" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_DOSE_RESPONSE_FILES must be a positive integer: $EXPECTED_DOSE_RESPONSE_FILES" >&2
  exit 2
fi
for flag_name in FORCE_FUSION FORCE_LATE_DEATH FORCE_DOSE_RESPONSE FORCE_REPORT DRY_RUN_SUBMIT; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1: $flag_value" >&2
    exit 2
  fi
done
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to reuse an existing classification output root: $OUT_ROOT" >&2
  exit 2
fi

mkdir -p "$TASK_DIR" "$LOG_DIR" "$ORIGINAL_OUT_DIR" "$NUCLEATED_OUT_DIR"
"$PYTHON" -I "$SEGMENTATION_FREEZE_VERIFIER" \
  --repo-root "$PROJECT_DIR" \
  --manifest "$SEGMENTATION_FREEZE_MANIFEST" \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --output "$SEGMENTATION_FREEZE_RECEIPT"
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
N_WELLS="$(
  awk -F_ 'NF {print $1}' "$TASK_LIST_FUSION" |
    LC_ALL=C sort -u |
    awk 'NF {n++} END {print n+0}'
)"
if [[ "$N_WELLS" -ne "$EXPECTED_WELLS" ]]; then
  echo "Expected $EXPECTED_WELLS wells but found $N_WELLS" >&2
  exit 2
fi
WELL_ARRAY_SPEC="1-$N_WELLS"

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
LATE_DEATH_PREPARE_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_late_death_prepare"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$LATE_DEATH_PREPARE_TIME"
  --cpus-per-task "$LATE_DEATH_PREPARE_CPUS"
  --mem "$LATE_DEATH_PREPARE_MEM"
)
LATE_DEATH_WELL_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_late_death_well"
  --array "$WELL_ARRAY_SPEC"
  --output "$LOG_DIR/%x_%A_%a.out"
  --error "$LOG_DIR/%x_%A_%a.err"
  --time "$LATE_DEATH_WELL_TIME"
  --cpus-per-task "$LATE_DEATH_WELL_CPUS"
  --mem "$LATE_DEATH_WELL_MEM"
)
LATE_DEATH_FINALIZE_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_late_death_finalize"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$LATE_DEATH_FINALIZE_TIME"
  --cpus-per-task "$LATE_DEATH_FINALIZE_CPUS"
  --mem "$LATE_DEATH_FINALIZE_MEM"
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
DOSE_RESPONSE_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_dose_response"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$DOSE_RESPONSE_TIME"
  --cpus-per-task "$DOSE_RESPONSE_CPUS"
  --mem "$DOSE_RESPONSE_MEM"
)
REPORT_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_report"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$REPORT_TIME"
  --cpus-per-task "$REPORT_CPUS"
  --mem "$REPORT_MEM"
)
CALIBRATION_REPORT_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name "${SBATCH_JOB_PREFIX}_d0_d5_report"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$CALIBRATION_REPORT_TIME"
  --cpus-per-task "$CALIBRATION_REPORT_CPUS"
  --mem "$CALIBRATION_REPORT_MEM"
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
    echo "well_count=$N_WELLS"
    echo "well_array_spec=$WELL_ARRAY_SPEC"
    echo "original_output=$ORIGINAL_OUT_DIR"
    echo "nucleated_output=$NUCLEATED_OUT_DIR"
    echo "consensus_summary=$OUT_ROOT/classification_consensus/summaries/cell_count_summary.csv"
    echo "consensus_well_count_plot=$OUT_ROOT/analysis/well_count_timecourses/fusion-consensus/well_live_dead_counts_over_time.png"
    echo "original_well_count_plot=$ORIGINAL_PLOT_DIR/well_live_dead_counts_over_time.png"
    echo "nucleated_well_count_plot=$NUCLEATED_PLOT_DIR/well_live_dead_counts_over_time.png"
    echo "dose_response_root=$DOSE_RESPONSE_DIR"
    echo "full_classification_report=$FULL_REPORT_DIR/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html"
    echo "d0_d5_calibration_report=$CALIBRATION_ROOT/report_final/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.html"
    echo "calibration_root=$CALIBRATION_ROOT"
    echo "calibration_go_no_go=$CALIBRATION_GO_NO_GO"
    echo "segmentation_freeze_manifest=$SEGMENTATION_FREEZE_MANIFEST"
    echo "segmentation_freeze_receipt=$SEGMENTATION_FREEZE_RECEIPT"
    echo "classification_method=cellpose_pipeline/scripts/08_fuse_multichannel_classification.py+cellpose_pipeline/scripts/14_apply_late_dead_trajectory_refinement.py"
    echo "late_death_method_version=death_classification_consensus_v2_20260725"
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
    echo "late_death_prepare_cpus_per_task=$LATE_DEATH_PREPARE_CPUS"
    echo "late_death_prepare_mem_per_task=$LATE_DEATH_PREPARE_MEM"
    echo "late_death_prepare_time=$LATE_DEATH_PREPARE_TIME"
    echo "late_death_well_cpus_per_task=$LATE_DEATH_WELL_CPUS"
    echo "late_death_well_mem_per_task=$LATE_DEATH_WELL_MEM"
    echo "late_death_well_time=$LATE_DEATH_WELL_TIME"
    echo "late_death_finalize_cpus_per_task=$LATE_DEATH_FINALIZE_CPUS"
    echo "late_death_finalize_mem_per_task=$LATE_DEATH_FINALIZE_MEM"
    echo "late_death_finalize_time=$LATE_DEATH_FINALIZE_TIME"
    echo "plot_cpus_per_task=$PLOT_CPUS"
    echo "plot_mem_per_task=$PLOT_MEM"
    echo "plot_time=$PLOT_TIME"
    echo "plot_expected_timepoints=$EXPECTED_TIMEPOINTS"
    echo "plot_expected_sites=$EXPECTED_SITES"
    echo "dose_response_cpus_per_task=$DOSE_RESPONSE_CPUS"
    echo "dose_response_mem_per_task=$DOSE_RESPONSE_MEM"
    echo "dose_response_time=$DOSE_RESPONSE_TIME"
    echo "dose_response_expected_files=$EXPECTED_DOSE_RESPONSE_FILES"
    echo "report_cpus_per_task=$REPORT_CPUS"
    echo "report_mem_per_task=$REPORT_MEM"
    echo "report_time=$REPORT_TIME"
    echo "calibration_report_cpus_per_task=$CALIBRATION_REPORT_CPUS"
    echo "calibration_report_mem_per_task=$CALIBRATION_REPORT_MEM"
    echo "calibration_report_time=$CALIBRATION_REPORT_TIME"
    echo "manifest_job_id=${MANIFEST_JOB_ID:-not_submitted}"
    echo "original_array_job_id=${ORIGINAL_JOB_ID:-not_submitted}"
    echo "original_merge_job_id=${ORIGINAL_MERGE_JOB_ID:-not_submitted}"
    echo "nucleated_array_job_id=${NUCLEATED_JOB_ID:-not_submitted}"
    echo "nucleated_merge_job_id=${NUCLEATED_MERGE_JOB_ID:-not_submitted}"
    echo "late_death_prepare_job_id=${LATE_DEATH_PREPARE_JOB_ID:-not_submitted}"
    echo "late_death_well_array_job_id=${LATE_DEATH_WELL_JOB_ID:-not_submitted}"
    echo "late_death_finalize_job_id=${LATE_DEATH_FINALIZE_JOB_ID:-not_submitted}"
    echo "well_count_plot_job_id=${PLOT_JOB_ID:-not_submitted}"
    echo "dose_response_job_id=${DOSE_RESPONSE_JOB_ID:-not_submitted}"
    echo "d0_d5_calibration_report_job_id=${CALIBRATION_REPORT_JOB_ID:-not_submitted}"
    echo "full_classification_report_job_id=${REPORT_JOB_ID:-not_submitted}"
  } > "$SUBMISSION_SUMMARY"
}

echo "classification_only=1"
echo "input_root=$INPUT_ROOT"
echo "source_run_root=$SOURCE_RUN_ROOT"
echo "output_root=$OUT_ROOT"
echo "task_count=$N_TASKS"
echo "array_spec=$ARRAY_SPEC"
echo "well_count=$N_WELLS"
echo "well_array_spec=$WELL_ARRAY_SPEC"
echo "original_output=$ORIGINAL_OUT_DIR"
echo "nucleated_output=$NUCLEATED_OUT_DIR"
echo "consensus_summary=$OUT_ROOT/classification_consensus/summaries/cell_count_summary.csv"
echo "consensus_well_count_plot=$OUT_ROOT/analysis/well_count_timecourses/fusion-consensus/well_live_dead_counts_over_time.png"
echo "original_well_count_plot=$ORIGINAL_PLOT_DIR/well_live_dead_counts_over_time.png"
echo "nucleated_well_count_plot=$NUCLEATED_PLOT_DIR/well_live_dead_counts_over_time.png"
echo "dose_response_root=$DOSE_RESPONSE_DIR"
echo "full_classification_report=$FULL_REPORT_DIR/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html"
echo "d0_d5_calibration_report=$CALIBRATION_ROOT/report_final/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.html"
printf 'manifest_sbatch_arg=%s\n' "${MANIFEST_SBATCH_ARGS[@]}"
printf 'classification_sbatch_arg=%s\n' "${CLASSIFICATION_SBATCH_ARGS[@]}"
printf 'merge_sbatch_arg=%s\n' "${MERGE_SBATCH_ARGS[@]}"
printf 'late_death_prepare_sbatch_arg=%s\n' "${LATE_DEATH_PREPARE_SBATCH_ARGS[@]}"
printf 'late_death_well_sbatch_arg=%s\n' "${LATE_DEATH_WELL_SBATCH_ARGS[@]}"
printf 'late_death_finalize_sbatch_arg=%s\n' "${LATE_DEATH_FINALIZE_SBATCH_ARGS[@]}"
printf 'plot_sbatch_arg=%s\n' "${PLOT_SBATCH_ARGS[@]}"
printf 'dose_response_sbatch_arg=%s\n' "${DOSE_RESPONSE_SBATCH_ARGS[@]}"
printf 'calibration_report_sbatch_arg=%s\n' "${CALIBRATION_REPORT_SBATCH_ARGS[@]}"
printf 'report_sbatch_arg=%s\n' "${REPORT_SBATCH_ARGS[@]}"

if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
  write_submission_summary
  echo "dry_run_submit=1"
  echo "dependency_graph={d0_d5_calibration_report,manifest -> {original_array,nucleated_array} -> {original_merge,nucleated_merge} -> late_death_prepare -> late_death_well_array[1-$N_WELLS] -> late_death_finalize -> well_count_plots -> dose_response} -> full_classification_report"
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
  if ! output="$(cpu_sbatch "$@")"; then
    echo "Slurm submission failed; verify that sbatch is available in PATH" >&2
    return 1
  fi
  output="${output%%;*}"
  if [[ ! "$output" =~ ^[0-9]+$ ]]; then
    echo "Slurm submission did not return a numeric job ID: ${output:-<empty>}" >&2
    return 1
  fi
  printf '%s\n' "$output"
}

CALIBRATION_REPORT_JOB_ID="$(submit_job "${CALIBRATION_REPORT_SBATCH_ARGS[@]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CALIBRATION_ROOT="$CALIBRATION_ROOT",FORCE_REPORT=0 \
  "$CALIBRATION_REPORT_WORKER")"

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

LATE_DEATH_PREPARE_JOB_ID="$(submit_job "${LATE_DEATH_PREPARE_SBATCH_ARGS[@]}" \
  --dependency "afterok:$ORIGINAL_MERGE_JOB_ID:$NUCLEATED_MERGE_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CLASSIFICATION_ROOT="$OUT_ROOT",PLATE_MAP="$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv",EXPECTED_FIELDS_PER_BRANCH="$EXPECTED_FIELDS",EXPECTED_WELLS="$N_WELLS",WORKERS="$LATE_DEATH_PREPARE_CPUS",FORCE_LATE_DEATH="$FORCE_LATE_DEATH",CALIBRATION_GO_NO_GO="$CALIBRATION_GO_NO_GO",SEGMENTATION_FREEZE_RECEIPT="$SEGMENTATION_FREEZE_RECEIPT" \
  "$LATE_DEATH_PREPARE_WORKER")"

LATE_DEATH_WELL_JOB_ID="$(submit_job "${LATE_DEATH_WELL_SBATCH_ARGS[@]}" \
  --dependency "afterok:$LATE_DEATH_PREPARE_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CLASSIFICATION_ROOT="$OUT_ROOT",EXPECTED_FIELDS_PER_BRANCH="$EXPECTED_FIELDS",EXPECTED_WELLS="$N_WELLS",FORCE_LATE_DEATH="$FORCE_LATE_DEATH",CALIBRATION_GO_NO_GO="$CALIBRATION_GO_NO_GO",SEGMENTATION_FREEZE_RECEIPT="$SEGMENTATION_FREEZE_RECEIPT" \
  "$LATE_DEATH_WELL_WORKER")"

LATE_DEATH_FINALIZE_JOB_ID="$(submit_job "${LATE_DEATH_FINALIZE_SBATCH_ARGS[@]}" \
  --dependency "afterany:$LATE_DEATH_WELL_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CLASSIFICATION_ROOT="$OUT_ROOT",EXPECTED_FIELDS_PER_BRANCH="$EXPECTED_FIELDS",EXPECTED_WELLS="$N_WELLS",CALIBRATION_GO_NO_GO="$CALIBRATION_GO_NO_GO",SEGMENTATION_FREEZE_RECEIPT="$SEGMENTATION_FREEZE_RECEIPT" \
  "$LATE_DEATH_FINALIZE_WORKER")"

PLOT_JOB_ID="$(submit_job "${PLOT_SBATCH_ARGS[@]}" \
  --dependency "afterok:$LATE_DEATH_FINALIZE_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$OUT_ROOT",EXPECTED_TIMEPOINTS="$EXPECTED_TIMEPOINTS",EXPECTED_SITES="$EXPECTED_SITES",PLOT_DPI="$PLOT_DPI" \
  "$PLOT_WORKER")"

DOSE_RESPONSE_JOB_ID="$(submit_job "${DOSE_RESPONSE_SBATCH_ARGS[@]}" \
  --dependency "afterok:$PLOT_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$OUT_ROOT",DOSE_DPI="$DOSE_DPI",EXPECTED_FILES="$EXPECTED_DOSE_RESPONSE_FILES",FORCE_DOSE_RESPONSE="$FORCE_DOSE_RESPONSE" \
  "$DOSE_RESPONSE_WORKER")"

REPORT_JOB_ID="$(submit_job "${REPORT_SBATCH_ARGS[@]}" \
  --dependency "afterok:$DOSE_RESPONSE_JOB_ID:$CALIBRATION_REPORT_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$OUT_ROOT",CALIBRATION_ROOT="$CALIBRATION_ROOT",EXPECTED_FIELDS_PER_BRANCH="$EXPECTED_FIELDS",EXPECTED_TIMEPOINTS="$EXPECTED_TIMEPOINTS",EXPECTED_DOSE_RESPONSE_FILES="$EXPECTED_DOSE_RESPONSE_FILES",FORCE_REPORT="$FORCE_REPORT" \
  "$REPORT_WORKER")"

write_submission_summary

echo "manifest_job_id=$MANIFEST_JOB_ID"
echo "original_array_job_id=$ORIGINAL_JOB_ID"
echo "original_merge_job_id=$ORIGINAL_MERGE_JOB_ID"
echo "nucleated_array_job_id=$NUCLEATED_JOB_ID"
echo "nucleated_merge_job_id=$NUCLEATED_MERGE_JOB_ID"
echo "late_death_prepare_job_id=$LATE_DEATH_PREPARE_JOB_ID"
echo "late_death_well_array_job_id=$LATE_DEATH_WELL_JOB_ID"
echo "late_death_finalize_job_id=$LATE_DEATH_FINALIZE_JOB_ID"
echo "well_count_plot_job_id=$PLOT_JOB_ID"
echo "dose_response_job_id=$DOSE_RESPONSE_JOB_ID"
echo "d0_d5_calibration_report_job_id=$CALIBRATION_REPORT_JOB_ID"
echo "full_classification_report_job_id=$REPORT_JOB_ID"
echo "submission_summary=$SUBMISSION_SUMMARY"
