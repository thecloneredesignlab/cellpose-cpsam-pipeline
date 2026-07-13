#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUT_ROOT RUN_NAME PROJECT_DIR" >&2
  exit 2
fi

OUT_ROOT=$1
RUN_NAME=$2
PROJECT_DIR=$3
DENSITY_CALLS_NAME="${DENSITY_CALLS_NAME:-density_calls.csv}"
NUCLEI_COUNT_THRESHOLD="${NUCLEI_COUNT_THRESHOLD:-4000}"
NUCLEI_MASK_FRACTION_THRESHOLD="${NUCLEI_MASK_FRACTION_THRESHOLD:-0.22}"
NUCLEI_MEDIAN_NN_THRESHOLD="${NUCLEI_MEDIAN_NN_THRESHOLD:-16.0}"
ENABLE_NUCLEI_MASK_FRACTION_TRIGGER="${ENABLE_NUCLEI_MASK_FRACTION_TRIGGER:-0}"

if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
  RUN_DIR="$OUT_ROOT"
else
  RUN_DIR="$OUT_ROOT/$RUN_NAME"
fi
DENSITY_CALLS_CSV="$RUN_DIR/qc/$DENSITY_CALLS_NAME"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "run_name=$RUN_NAME"
echo "out_root=$OUT_ROOT"
echo "run_dir=$RUN_DIR"
echo "density_calls_csv=$DENSITY_CALLS_CSV"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "python_bin=$PYTHON_BIN"
echo "enable_nuclei_mask_fraction_trigger=$ENABLE_NUCLEI_MASK_FRACTION_TRIGGER"

MASK_FRACTION_ARGS=(--no-enable-mask-fraction-trigger)
if [[ "$ENABLE_NUCLEI_MASK_FRACTION_TRIGGER" == "1" ]]; then
  MASK_FRACTION_ARGS=(--enable-mask-fraction-trigger)
fi

"$PYTHON_BIN" -I cellpose_pipeline/scripts/02_call_high_density_from_nuclei_masks.py \
  --run-dir "$RUN_DIR" \
  --out-csv "$DENSITY_CALLS_CSV" \
  --nuclei-count-threshold "$NUCLEI_COUNT_THRESHOLD" \
  --nuclei-mask-fraction-threshold "$NUCLEI_MASK_FRACTION_THRESHOLD" \
  --nuclei-median-nn-threshold "$NUCLEI_MEDIAN_NN_THRESHOLD" \
  "${MASK_FRACTION_ARGS[@]}"

if [[ ! -s "$DENSITY_CALLS_CSV" ]]; then
  echo "Density calls CSV was not created or is empty: $DENSITY_CALLS_CSV" >&2
  exit 3
fi

echo "density_table_complete=1"
