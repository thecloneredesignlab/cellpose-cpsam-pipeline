#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

# Backfill the d0+Day-5 calibration report, full-cohort dose response, and
# full-cohort report for the completed 2026-07-23 classification run.

BASE="${BASE:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide}"
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RESULT_ROOT="${RESULT_ROOT:-$BASE/results/classification_20260723_101944}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$BASE/results/Tests_and_Parameters_calibration/late_dead_trajectory_optimization_20260723_024151}"
SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/Docker/hpc"
CALIBRATION_REPORT_WORKER="${CALIBRATION_REPORT_WORKER:-$SCRIPT_DIR/Parameter_calibration/27_generate_late_dead_d0_d5_report.sh}"
DOSE_RESPONSE_WORKER="${DOSE_RESPONSE_WORKER:-$SCRIPT_DIR/analysisi/06_run_dose_response_analysis.sh}"
FULL_REPORT_WORKER="${FULL_REPORT_WORKER:-$SCRIPT_DIR/analysisi/07_run_full_classification_report.sh}"
LOG_DIR="${LOG_DIR:-$RESULT_ROOT/logs}"
SUMMARY="${SUMMARY:-$RESULT_ROOT/REPORT_BACKFILL_SUBMISSION_SUMMARY.txt}"

JOB_QOS="${JOB_QOS:-xxlarge}"
JOB_PARTITION="${JOB_PARTITION:-}"
JOB_ACCOUNT="${JOB_ACCOUNT:-}"
CALIBRATION_REPORT_TIME="${CALIBRATION_REPORT_TIME:-04:00:00}"
CALIBRATION_REPORT_CPUS="${CALIBRATION_REPORT_CPUS:-4}"
CALIBRATION_REPORT_MEM="${CALIBRATION_REPORT_MEM:-32G}"
DOSE_RESPONSE_TIME="${DOSE_RESPONSE_TIME:-04:00:00}"
DOSE_RESPONSE_CPUS="${DOSE_RESPONSE_CPUS:-1}"
DOSE_RESPONSE_MEM="${DOSE_RESPONSE_MEM:-16G}"
FULL_REPORT_TIME="${FULL_REPORT_TIME:-06:00:00}"
FULL_REPORT_CPUS="${FULL_REPORT_CPUS:-4}"
FULL_REPORT_MEM="${FULL_REPORT_MEM:-32G}"
EXPECTED_FIELDS_PER_BRANCH="${EXPECTED_FIELDS_PER_BRANCH:-27200}"
EXPECTED_TIMEPOINTS="${EXPECTED_TIMEPOINTS:-85}"
EXPECTED_DOSE_RESPONSE_FILES="${EXPECTED_DOSE_RESPONSE_FILES:-82}"
DOSE_DPI="${DOSE_DPI:-220}"
FORCE_CALIBRATION_REPORT="${FORCE_CALIBRATION_REPORT:-0}"
FORCE_DOSE_RESPONSE="${FORCE_DOSE_RESPONSE:-0}"
FORCE_FULL_REPORT="${FORCE_FULL_REPORT:-0}"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"

for required in \
  "$PROJECT_DIR" \
  "$RESULT_ROOT" \
  "$RESULT_ROOT/workflow_status/late_death_trajectory/_SUCCESS" \
  "$RESULT_ROOT/analysis/well_count_timecourses/fusion/well_time_cell_state_counts.csv" \
  "$RESULT_ROOT/analysis/well_count_timecourses/fusion-nucleated-only/well_time_cell_state_counts.csv" \
  "$CALIBRATION_ROOT/_SUCCESS_V3" \
  "$CALIBRATION_ROOT/optimization_v3/best_configuration.json"; do
  [[ -e "$required" ]] || {
    echo "Required report-backfill input is missing: $required" >&2
    exit 2
  }
done
for worker in \
  "$CALIBRATION_REPORT_WORKER" \
  "$DOSE_RESPONSE_WORKER" \
  "$FULL_REPORT_WORKER"; do
  [[ -x "$worker" ]] || {
    echo "Required report-backfill worker is missing or not executable: $worker" >&2
    exit 2
  }
done
for integer_name in \
  CALIBRATION_REPORT_CPUS \
  DOSE_RESPONSE_CPUS \
  FULL_REPORT_CPUS \
  EXPECTED_FIELDS_PER_BRANCH \
  EXPECTED_TIMEPOINTS \
  EXPECTED_DOSE_RESPONSE_FILES; do
  integer_value="${!integer_name}"
  [[ "$integer_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$integer_name must be a positive integer: $integer_value" >&2
    exit 2
  }
done
for flag_name in \
  FORCE_CALIBRATION_REPORT \
  FORCE_DOSE_RESPONSE \
  FORCE_FULL_REPORT \
  DRY_RUN_SUBMIT; do
  flag_value="${!flag_name}"
  [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
    echo "$flag_name must be 0 or 1: $flag_value" >&2
    exit 2
  }
done

mkdir -p "$LOG_DIR"

COMMON_SBATCH_ARGS=()
[[ -n "$JOB_QOS" ]] && COMMON_SBATCH_ARGS+=(--qos "$JOB_QOS")
[[ -n "$JOB_PARTITION" ]] && COMMON_SBATCH_ARGS+=(--partition "$JOB_PARTITION")
[[ -n "$JOB_ACCOUNT" ]] && COMMON_SBATCH_ARGS+=(--account "$JOB_ACCOUNT")

CALIBRATION_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name cpsam_d0_d5_report
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$CALIBRATION_REPORT_TIME"
  --cpus-per-task "$CALIBRATION_REPORT_CPUS"
  --mem "$CALIBRATION_REPORT_MEM"
)
DOSE_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name cpsam_dose_response
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$DOSE_RESPONSE_TIME"
  --cpus-per-task "$DOSE_RESPONSE_CPUS"
  --mem "$DOSE_RESPONSE_MEM"
)
FULL_REPORT_SBATCH_ARGS=(
  "${COMMON_SBATCH_ARGS[@]}"
  --job-name cpsam_full_classification_report
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "$FULL_REPORT_TIME"
  --cpus-per-task "$FULL_REPORT_CPUS"
  --mem "$FULL_REPORT_MEM"
)

write_summary() {
  {
    echo "prepared_at=$(date --iso-8601=seconds)"
    echo "submission_mode=$([[ "$DRY_RUN_SUBMIT" == "1" ]] && echo dry_run || echo submitted)"
    echo "project_dir=$PROJECT_DIR"
    echo "project_git_sha=$(git -c safe.directory="$PROJECT_DIR" -C "$PROJECT_DIR" rev-parse HEAD)"
    echo "classification_root=$RESULT_ROOT"
    echo "calibration_root=$CALIBRATION_ROOT"
    echo "dose_response_root=$RESULT_ROOT/analysis/dose_response"
    echo "calibration_report=$CALIBRATION_ROOT/report_final/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.html"
    echo "full_report=$RESULT_ROOT/analysis/reports/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html"
    echo "qos=$JOB_QOS"
    echo "calibration_report_resources=${CALIBRATION_REPORT_CPUS}cpu,$CALIBRATION_REPORT_MEM,$CALIBRATION_REPORT_TIME"
    echo "dose_response_resources=${DOSE_RESPONSE_CPUS}cpu,$DOSE_RESPONSE_MEM,$DOSE_RESPONSE_TIME"
    echo "full_report_resources=${FULL_REPORT_CPUS}cpu,$FULL_REPORT_MEM,$FULL_REPORT_TIME"
    echo "calibration_report_job_id=${CALIBRATION_REPORT_JOB_ID:-not_submitted}"
    echo "dose_response_job_id=${DOSE_RESPONSE_JOB_ID:-not_submitted}"
    echo "full_report_job_id=${FULL_REPORT_JOB_ID:-not_submitted}"
  } > "$SUMMARY"
}

echo "classification_root=$RESULT_ROOT"
echo "calibration_root=$CALIBRATION_ROOT"
printf 'calibration_report_sbatch_arg=%s\n' "${CALIBRATION_SBATCH_ARGS[@]}"
printf 'dose_response_sbatch_arg=%s\n' "${DOSE_SBATCH_ARGS[@]}"
printf 'full_report_sbatch_arg=%s\n' "${FULL_REPORT_SBATCH_ARGS[@]}"

if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
  write_summary
  echo "dry_run_submit=1"
  echo "dependency_graph={d0_d5_report,dose_response} -> full_classification_report"
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

CALIBRATION_REPORT_JOB_ID="$(submit_job "${CALIBRATION_SBATCH_ARGS[@]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CALIBRATION_ROOT="$CALIBRATION_ROOT",FORCE_REPORT="$FORCE_CALIBRATION_REPORT" \
  "$CALIBRATION_REPORT_WORKER")"

DOSE_RESPONSE_JOB_ID="$(submit_job "${DOSE_SBATCH_ARGS[@]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$RESULT_ROOT",DOSE_DPI="$DOSE_DPI",EXPECTED_FILES="$EXPECTED_DOSE_RESPONSE_FILES",FORCE_DOSE_RESPONSE="$FORCE_DOSE_RESPONSE" \
  "$DOSE_RESPONSE_WORKER")"

FULL_REPORT_JOB_ID="$(submit_job "${FULL_REPORT_SBATCH_ARGS[@]}" \
  --dependency "afterok:$CALIBRATION_REPORT_JOB_ID:$DOSE_RESPONSE_JOB_ID" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$RESULT_ROOT",CALIBRATION_ROOT="$CALIBRATION_ROOT",EXPECTED_FIELDS_PER_BRANCH="$EXPECTED_FIELDS_PER_BRANCH",EXPECTED_TIMEPOINTS="$EXPECTED_TIMEPOINTS",EXPECTED_DOSE_RESPONSE_FILES="$EXPECTED_DOSE_RESPONSE_FILES",FORCE_REPORT="$FORCE_FULL_REPORT" \
  "$FULL_REPORT_WORKER")"

write_summary

echo "calibration_report_job_id=$CALIBRATION_REPORT_JOB_ID"
echo "dose_response_job_id=$DOSE_RESPONSE_JOB_ID"
echo "full_report_job_id=$FULL_REPORT_JOB_ID"
echo "submission_summary=$SUMMARY"
