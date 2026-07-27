#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

BASE="/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
INPUT_ROOT="${INPUT_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
RUN_ROOT="${RUN_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/classification_fusion}"

WORKER="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/analysisi/02_run_multichannel_classification_fusion.sh"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/cellpose_pipeline/hpc/logs}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory does not exist: $PROJECT_DIR" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input root does not exist: $INPUT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$WORKER" ]]; then
  echo "Worker script is missing or not executable: $WORKER" >&2
  exit 2
fi

SBATCH_ARGS=(
  --job-name "${SBATCH_JOB_NAME:-cpsam_fusion}"
  --output "$LOG_DIR/%x_%j.out"
  --error "$LOG_DIR/%x_%j.err"
  --time "${SBATCH_TIME:-4:00:00}"
  --cpus-per-task "${SBATCH_CPUS:-4}"
  --mem "${SBATCH_MEM:-24G}"
)

if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
fi
SBATCH_QOS_VALUE="${SBATCH_QOS:-small}"
if [[ -n "$SBATCH_QOS_VALUE" ]]; then
  SBATCH_ARGS+=(--qos "$SBATCH_QOS_VALUE")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi
if [[ -n "${SBATCH_GRES:-}" ]]; then
  SBATCH_ARGS+=(--gres "$SBATCH_GRES")
fi
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  SBATCH_ARGS+=(--dependency "afterok:$AFTEROK_JOB_ID")
fi

EXPORTS=(
  "PROJECT_DIR=$PROJECT_DIR"
  "INPUT_ROOT=$INPUT_ROOT"
  "RUN_ROOT=$RUN_ROOT"
  "OUT_DIR=$OUT_DIR"
  "FORCE_FUSION=${FORCE_FUSION:-0}"
  "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}"
)
for optional_name in COMBINED_RUN BRIGHTFIELD_RUN DEAD_RUN NUCLEI_RUN FUSION_LIMIT FUSION_KEY; do
  optional_value="${!optional_name:-}"
  if [[ -n "$optional_value" ]]; then
    EXPORTS+=("$optional_name=$optional_value")
  fi
done

if [[ "${DRY_RUN_SUBMIT:-0}" == "1" ]]; then
  echo "dry_run_submit=1"
  echo "project_dir=$PROJECT_DIR"
  echo "input_root=$INPUT_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "out_dir=$OUT_DIR"
  echo "worker=$WORKER"
  echo "afterok_job_id=${AFTEROK_JOB_ID:-none}"
  printf 'export=%s\n' "${EXPORTS[@]}"
  printf 'sbatch_arg=%s\n' "${SBATCH_ARGS[@]}"
  exit 0
fi

IFS=,
SUBMIT_OUTPUT="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --export="ALL,${EXPORTS[*]}" \
    "$WORKER"
)"
unset IFS
echo "$SUBMIT_OUTPUT"
JOB_ID="$(printf '%s\n' "$SUBMIT_OUTPUT" | awk '{print $4}')"
echo "fusion_job_id=$JOB_ID"
echo "input_root=$INPUT_ROOT"
echo "run_root=$RUN_ROOT"
echo "out_dir=$OUT_DIR"
echo "logs=$LOG_DIR/${SBATCH_JOB_NAME:-cpsam_fusion}_${JOB_ID}.out"
