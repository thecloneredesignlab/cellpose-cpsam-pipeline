#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 INPUT_ROOT [RUN_NAME] [OUT_ROOT]" >&2
  exit 2
fi

INPUT_ROOT=$1
RUN_NAME=${2:-separate_images_cpsam_$(date +%Y%m%d_%H%M%S)}
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OUT_ROOT=${3:-${OUT_ROOT:-"$PROJECT_DIR/cellpose_pipeline/workflow_runs"}}
PYTHON_BIN=${PYTHON_BIN:-python}
ENABLE_HIGH_DENSITY_PROFILES="${ENABLE_HIGH_DENSITY_PROFILES:-1}"
HIGH_DENSITY_CALLS_CSV="${HIGH_DENSITY_CALLS_CSV:-}"
AUTO_GENERATE_DENSITY_CALLS="${AUTO_GENERATE_DENSITY_CALLS:-1}"
DENSITY_CALLS_NAME="${DENSITY_CALLS_NAME:-density_calls.csv}"

RUN_DIR="$OUT_ROOT/$RUN_NAME"
MANIFEST_DIR="$RUN_DIR/manifests"
LOG_DIR="$RUN_DIR/logs"
MANIFEST="$MANIFEST_DIR/${RUN_NAME}_images.txt"
DENSITY_WORKER="$PROJECT_DIR/cellpose_pipeline/hpc/Parameter_calibration/03_run_density_prepass.sh"

mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
: > "$MANIFEST"

if [[ ! -f "$DENSITY_WORKER" ]]; then
  echo "Density prepass script is missing: $DENSITY_WORKER" >&2
  exit 2
fi

for folder in Brightfield Dead Nuclei; do
  if [[ -d "$INPUT_ROOT/$folder" ]]; then
    find "$INPUT_ROOT/$folder" -maxdepth 1 -type f \( \
      -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
    \) | sort >> "$MANIFEST"
  else
    echo "missing_supported_folder=$INPUT_ROOT/$folder"
  fi
done

if [[ -d "$INPUT_ROOT/Dead_Uncalibrated" ]]; then
  echo "skipping_folder=$INPUT_ROOT/Dead_Uncalibrated"
fi

N_IMAGES=$(wc -l < "$MANIFEST" | tr -d '[:space:]')
if [[ "$N_IMAGES" == "0" ]]; then
  echo "No supported images found under $INPUT_ROOT" >&2
  exit 1
fi

GPU_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--use-gpu)
fi

HIGH_DENSITY_ARGS=()
if [[ "${ENABLE_HIGH_DENSITY_PROFILES:-1}" == "0" ]]; then
  HIGH_DENSITY_ARGS+=(--no-enable-high-density-profiles)
else
  HIGH_DENSITY_ARGS+=(--enable-high-density-profiles)
  if [[ -n "${HIGH_DENSITY_CALLS_CSV:-}" ]]; then
    HIGH_DENSITY_ARGS+=(--high-density-calls-csv "$HIGH_DENSITY_CALLS_CSV")
  fi
fi

PREFLIGHT_ARGS=(
  "$PROJECT_DIR/cellpose_pipeline/scripts/01_segment_images.py"
  --dir "$INPUT_ROOT"
  --recursive
  --profile-mode auto
  --skip-unknown-profiles
  --preflight-only
  "${HIGH_DENSITY_ARGS[@]}"
  "${GPU_ARGS[@]}"
)
if [[ "${PREFLIGHT_CHECK_MODELS:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--check-models)
fi

"$PYTHON_BIN" "${PREFLIGHT_ARGS[@]}"

SBATCH_ARGS=(
  --job-name "$RUN_NAME"
  --array "1-$N_IMAGES"
  --output "$LOG_DIR/%x_%A_%a.out"
  --error "$LOG_DIR/%x_%A_%a.err"
  --time "${SBATCH_TIME:-12:00:00}"
  --cpus-per-task "${SBATCH_CPUS:-4}"
  --mem "${SBATCH_MEM:-24G}"
)

if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
fi
if [[ -n "${SBATCH_QOS:-}" ]]; then
  SBATCH_ARGS+=(--qos "$SBATCH_QOS")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi
SBATCH_GRES_VALUE=${SBATCH_GRES-gpu:1}
if [[ -n "$SBATCH_GRES_VALUE" ]]; then
  SBATCH_ARGS+=(--gres "$SBATCH_GRES_VALUE")
fi

DENSITY_JOB_ID=""
if [[ "$ENABLE_HIGH_DENSITY_PROFILES" != "0" && -z "$HIGH_DENSITY_CALLS_CSV" && "$AUTO_GENERATE_DENSITY_CALLS" != "0" ]]; then
  HIGH_DENSITY_CALLS_CSV="$RUN_DIR/qc/$DENSITY_CALLS_NAME"
  DENSITY_SBATCH_ARGS=(
    --job-name "${RUN_NAME}_density"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${DENSITY_SBATCH_TIME:-${SBATCH_TIME:-12:00:00}}"
    --cpus-per-task "${DENSITY_SBATCH_CPUS:-${SBATCH_CPUS:-4}}"
    --mem "${DENSITY_SBATCH_MEM:-${SBATCH_MEM:-24G}}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    DENSITY_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  if [[ -n "${SBATCH_QOS:-}" ]]; then
    DENSITY_SBATCH_ARGS+=(--qos "$SBATCH_QOS")
  fi
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    DENSITY_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
  if [[ -n "$SBATCH_GRES_VALUE" ]]; then
    DENSITY_SBATCH_ARGS+=(--gres "$SBATCH_GRES_VALUE")
  fi
  if [[ "${DRY_RUN_SUBMIT:-0}" != "1" ]]; then
    DENSITY_SUBMIT_OUTPUT="$(
      sbatch "${DENSITY_SBATCH_ARGS[@]}" \
        --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",OUT_ROOT="$OUT_ROOT",RUN_NAME="$RUN_NAME",USE_GPU="${USE_GPU:-1}",ENABLE_HIGH_DENSITY_PROFILES="$ENABLE_HIGH_DENSITY_PROFILES",DENSITY_CALLS_NAME="$DENSITY_CALLS_NAME" \
        "$DENSITY_WORKER" "$INPUT_ROOT" "$RUN_NAME" "$OUT_ROOT" "$PROJECT_DIR"
    )"
    echo "$DENSITY_SUBMIT_OUTPUT"
    DENSITY_JOB_ID="$(printf '%s\n' "$DENSITY_SUBMIT_OUTPUT" | awk '{print $4}')"
    SBATCH_ARGS+=(--dependency "afterok:$DENSITY_JOB_ID")
  fi
fi

export ENABLE_HIGH_DENSITY_PROFILES
export HIGH_DENSITY_CALLS_CSV
export DENSITY_CALLS_NAME

echo "manifest=$MANIFEST"
echo "n_images=$N_IMAGES"
echo "run_dir=$RUN_DIR"
echo "enable_high_density_profiles=$ENABLE_HIGH_DENSITY_PROFILES"
echo "auto_generate_density_calls=$AUTO_GENERATE_DENSITY_CALLS"
echo "density_calls_name=$DENSITY_CALLS_NAME"
echo "high_density_calls_csv=${HIGH_DENSITY_CALLS_CSV:-unset}"
echo "density_prepass_job_id=${DENSITY_JOB_ID:-none}"
if [[ "${DRY_RUN_SUBMIT:-0}" == "1" ]]; then
  printf 'sbatch_arg=%s\n' "${SBATCH_ARGS[@]}"
  exit 0
fi
sbatch "${SBATCH_ARGS[@]}" \
  "$PROJECT_DIR/cellpose_pipeline/hpc/Parameter_calibration/02_run_separate_images_array_task.sh" \
  "$MANIFEST" "$RUN_NAME" "$OUT_ROOT" "$PROJECT_DIR"
