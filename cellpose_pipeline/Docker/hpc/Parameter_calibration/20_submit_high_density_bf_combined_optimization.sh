#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/high_density_bf_combined_optimization_hpc_20260711}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

SCREEN_WORKER="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/21_run_high_density_bf_combined_screen.sh"
SCORE_WORKER="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/22_run_high_density_bf_combined_score.sh"
QC_WORKER="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/23_run_high_density_bf_combined_qc.sh"

COMMON_EXPORT="ALL,PROJECT_DIR=$PROJECT_DIR,RESULTS_ROOT=$RESULTS_ROOT,OUT_ROOT=$OUT_ROOT"
DRY_RUN_SUBMIT="${DRY_RUN_SUBMIT:-0}"

submit_sbatch() {
  if [[ "$DRY_RUN_SUBMIT" == "1" ]]; then
    printf 'dry_run_sbatch' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf 'Submitted batch job DRYRUN\n'
    return 0
  fi
  sbatch "$@"
}

SCREEN_SBATCH_ARGS=(
  --job-name "${JOB_PREFIX:-hd_bf_combined}_screen"
  --qos "${SCREEN_QOS:-small}"
  --time "${SCREEN_TIME:-12:00:00}"
  --cpus-per-task "${SCREEN_CPUS:-4}"
  --mem "${SCREEN_MEM:-32G}"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --export "$COMMON_EXPORT"
)
SCREEN_GRES_VALUE="${SCREEN_GRES-gpu:a30:1}"
if [[ -n "$SCREEN_GRES_VALUE" ]]; then
  SCREEN_SBATCH_ARGS+=(--gres "$SCREEN_GRES_VALUE")
fi

SCREEN_OUTPUT="$(
  submit_sbatch "${SCREEN_SBATCH_ARGS[@]}" "$SCREEN_WORKER"
)"
echo "$SCREEN_OUTPUT"
SCREEN_JOB_ID="$(printf '%s\n' "$SCREEN_OUTPUT" | awk '{print $4}')"

SCORE_OUTPUT="$(
  submit_sbatch \
    --job-name "${JOB_PREFIX:-hd_bf_combined}_score" \
    --dependency "afterok:$SCREEN_JOB_ID" \
    --qos "${SCORE_QOS:-xxlarge}" \
    --time "${SCORE_TIME:-03:00:00}" \
    --cpus-per-task "${SCORE_CPUS:-4}" \
    --mem "${SCORE_MEM:-32G}" \
    --output "$LOG_DIR/%x_%j.out" \
    --error "$LOG_DIR/%x_%j.err" \
    --export "$COMMON_EXPORT" \
    "$SCORE_WORKER"
)"
echo "$SCORE_OUTPUT"
SCORE_JOB_ID="$(printf '%s\n' "$SCORE_OUTPUT" | awk '{print $4}')"

QC_OUTPUT="$(
  submit_sbatch \
    --job-name "${JOB_PREFIX:-hd_bf_combined}_qc" \
    --dependency "afterok:$SCORE_JOB_ID" \
    --qos "${QC_QOS:-xxlarge}" \
    --time "${QC_TIME:-02:00:00}" \
    --cpus-per-task "${QC_CPUS:-4}" \
    --mem "${QC_MEM:-16G}" \
    --output "$LOG_DIR/%x_%j.out" \
    --error "$LOG_DIR/%x_%j.err" \
    --export "$COMMON_EXPORT" \
    "$QC_WORKER"
)"
echo "$QC_OUTPUT"
QC_JOB_ID="$(printf '%s\n' "$QC_OUTPUT" | awk '{print $4}')"

echo "out_root=$OUT_ROOT"
echo "screen_job_id=$SCREEN_JOB_ID"
echo "score_job_id=$SCORE_JOB_ID"
echo "qc_job_id=$QC_JOB_ID"
