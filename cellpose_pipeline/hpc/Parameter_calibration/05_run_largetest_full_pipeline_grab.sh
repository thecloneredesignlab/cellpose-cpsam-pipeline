#!/bin/bash
set -Eeuo pipefail

BASE="${BASE:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide}"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/largetest_full_fusion_shape_strict_grab_$(date +%Y%m%d_%H%M%S)}"
EXPECTED_PER_PROFILE="${EXPECTED_PER_PROFILE:-20}"
EXPECTED_HOST="${EXPECTED_HOST:-hpctpa3pc0061}"
PYTHON_EXPECTED="${PYTHON_EXPECTED:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/scripts"
HPC_DIR="$PROJECT_DIR/cellpose_pipeline/hpc"
STATUS_DIR="$OUT_ROOT/status"
LOG_DIR="$OUT_ROOT/logs"
PROVENANCE_DIR="$OUT_ROOT/provenance"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
TIMINGS="$STATUS_DIR/stage_timings.tsv"
SHAPE_ROOT="$OUT_ROOT/shape_strict"
TASK_LIST_SHAPE="$STATUS_DIR/shape_keys.txt"

if [[ -d "$OUT_ROOT" ]] && [[ -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "${RESUME:-0}" != "1" ]]; then
  echo "Output directory is not empty; set RESUME=1 only for an intentional resume: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$PROVENANCE_DIR" "$OUT_ROOT/qc"
if [[ "${RESUME:-0}" == "1" ]]; then
  rm -f "$STATUS_DIR/FAILED"
fi

exec > >(tee -a "$PIPELINE_LOG") 2>&1

CURRENT_STAGE="initialization"
on_error() {
  local code=$?
  trap - ERR
  printf 'stage=%s\nexit_code=%s\ntime=%s\n' "$CURRENT_STAGE" "$code" "$(date -Iseconds)" > "$STATUS_DIR/FAILED"
  echo "pipeline_failed stage=$CURRENT_STAGE exit_code=$code" >&2
  exit "$code"
}
trap on_error ERR

if [[ ! -s "$TIMINGS" ]]; then
  printf 'stage\tstart_epoch\tend_epoch\telapsed_sec\n' > "$TIMINGS"
fi

run_stage() {
  local stage=$1
  shift
  CURRENT_STAGE="$stage"
  local start_epoch end_epoch
  start_epoch="$(date +%s)"
  echo "stage_start=$stage time=$(date -Iseconds)"
  "$@"
  end_epoch="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "$stage" "$start_epoch" "$end_epoch" "$((end_epoch - start_epoch))" >> "$TIMINGS"
  printf 'stage=%s\ncompleted=%s\n' "$stage" "$(date -Iseconds)" > "$STATUS_DIR/${stage}.done"
  echo "stage_complete=$stage elapsed_sec=$((end_epoch - start_epoch))"
}

count_supported_images() {
  find "$1" -maxdepth 1 -type f \( \
    -iname "*.tif" -o -iname "*.tiff" -o -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \
  \) | wc -l | tr -d '[:space:]'
}

require_file_count() {
  local directory=$1
  local pattern=$2
  local expected=$3
  local label=$4
  local observed
  observed="$(find "$directory" -maxdepth 1 -type f -name "$pattern" | wc -l | tr -d '[:space:]')"
  echo "artifact_count label=$label observed=$observed expected=$expected directory=$directory"
  if [[ "$observed" -ne "$expected" ]]; then
    echo "Unexpected artifact count for $label: $observed != $expected" >&2
    return 1
  fi
}

stage_environment() {
  local host_short
  host_short="$(hostname -s)"
  echo "host=$host_short"
  if [[ "$host_short" != "$EXPECTED_HOST" ]]; then
    echo "Expected grab node $EXPECTED_HOST but running on $host_short" >&2
    return 1
  fi
  if [[ ! -d "$PROJECT_DIR" || ! -d "$INPUT_ROOT" ]]; then
    echo "Missing project or input root" >&2
    return 1
  fi
  for profile in Brightfield Combined Dead Nuclei; do
    local count
    count="$(count_supported_images "$INPUT_ROOT/$profile")"
    echo "raw_count profile=$profile count=$count"
    if [[ "$count" -ne "$EXPECTED_PER_PROFILE" ]]; then
      echo "Expected $EXPECTED_PER_PROFILE images in $profile, found $count" >&2
      return 1
    fi
  done
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$PROVENANCE_DIR/gpu.csv"
  "$PYTHON_BIN" -I "$SCRIPT_DIR/01_segment_images.py" \
    --dir "$INPUT_ROOT" \
    --recursive \
    --run-name . \
    --out-root "$OUT_ROOT" \
    --profile-mode auto \
    --skip-unknown-profiles \
    --enable-high-density-profiles \
    --preflight-only \
    --check-models \
    --use-gpu
}

segment_folder() {
  local profile=$1
  shift
  "$PYTHON_BIN" -I "$SCRIPT_DIR/01_segment_images.py" \
    --dir "$INPUT_ROOT/$profile" \
    --run-name . \
    --out-root "$OUT_ROOT/$profile" \
    --summary-name segmentation_summary.csv \
    --profile-mode auto \
    --skip-unknown-profiles \
    --flat-profile-output \
    --segmentation-only \
    --use-gpu \
    --gpu-device 0 \
    "$@"
  require_file_count "$OUT_ROOT/$profile/segmentations" "*_cp_masks.tif" "$EXPECTED_PER_PROFILE" "$profile masks"
  require_file_count "$OUT_ROOT/$profile/metadata" "*_segmentation_metadata.json" "$EXPECTED_PER_PROFILE" "$profile metadata"
  require_file_count "$OUT_ROOT/$profile/qc/segmentation_overlays" "*_segmentation_overlay.png" "$EXPECTED_PER_PROFILE" "$profile overlays"
}

stage_nuclei() {
  segment_folder Nuclei --no-enable-high-density-profiles --no-auto-generate-density-calls
  require_file_count "$OUT_ROOT/Nuclei/nucleus_core_seeds" "*_core_masks.tif" "$EXPECTED_PER_PROFILE" "Nuclei core masks"
}

stage_density() {
  "$PYTHON_BIN" -I "$SCRIPT_DIR/02_call_high_density_from_nuclei_masks.py" \
    --run-dir "$OUT_ROOT" \
    --out-csv "$OUT_ROOT/qc/density_calls.csv" \
    --nuclei-count-threshold 4000 \
    --nuclei-mask-fraction-threshold 0.22 \
    --nuclei-median-nn-threshold 16.0 \
    --no-enable-mask-fraction-trigger
  local rows
  rows="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$OUT_ROOT/qc/density_calls.csv")"
  echo "density_rows=$rows"
  if [[ "$rows" -ne "$EXPECTED_PER_PROFILE" ]]; then
    echo "Expected $EXPECTED_PER_PROFILE density rows, found $rows" >&2
    return 1
  fi
}

stage_cell_profile() {
  local profile=$1
  segment_folder "$profile" \
    --enable-high-density-profiles \
    --high-density-calls-csv "$OUT_ROOT/qc/density_calls.csv" \
    --no-auto-generate-density-calls
}

stage_fusion() {
  "$PYTHON_BIN" -I "$SCRIPT_DIR/08_fuse_multichannel_classification.py" \
    --run-root "$OUT_ROOT" \
    --input-root "$INPUT_ROOT" \
    --out-dir "$OUT_ROOT/classification_fusion" \
    --force
  local rows
  rows="$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$OUT_ROOT/classification_fusion/summaries/cell_count_summary.csv")"
  echo "fusion_summary_rows=$rows"
  if [[ "$rows" -ne "$EXPECTED_PER_PROFILE" ]]; then
    echo "Expected $EXPECTED_PER_PROFILE fusion rows, found $rows" >&2
    return 1
  fi
  if [[ -s "$OUT_ROOT/classification_fusion/failures.csv" ]]; then
    echo "Fusion failures were reported" >&2
    return 1
  fi
}

build_shape_keys() {
  local unsorted="$TASK_LIST_SHAPE.unsorted"
  find "$INPUT_ROOT/Combined" -maxdepth 1 -type f \( \
      -iname "*.tif" -o -iname "*.tiff" -o -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \
    \) | sort | awk '
      {
        name = $0
        sub(/^.*\//, "", name)
        sub(/\.[^.]*$/, "", name)
        key = name
        sub(/^SUM159_AC_/, "", key)
        sub(/^Exp1_/, "", key)
        sub(/^BF_/, "", key)
        sub(/^Dead_/, "", key)
        if (key ~ /^[A-H][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m$/) {
          print key
        } else {
          print "Could not extract shape key from: " $0 > "/dev/stderr"
          bad = 1
        }
      }
      END { exit bad ? 1 : 0 }
    ' > "$unsorted"
  sort -u "$unsorted" > "$TASK_LIST_SHAPE"
  rm -f "$unsorted"
  local count
  count="$(wc -l < "$TASK_LIST_SHAPE" | tr -d '[:space:]')"
  echo "shape_key_count=$count"
  if [[ "$count" -ne "$EXPECTED_PER_PROFILE" ]]; then
    echo "Expected $EXPECTED_PER_PROFILE shape keys, found $count" >&2
    return 1
  fi
}

stage_shape_strict() {
  build_shape_keys
  export PROJECT_DIR INPUT_ROOT TASK_LIST_SHAPE SHAPE_ROOT
  export RUN_ROOT="$OUT_ROOT"
  export SLURM_CPUS_PER_TASK="${SHAPE_CPUS:-2}"
  local task_id
  for task_id in $(seq 1 "$EXPECTED_PER_PROFILE"); do
    echo "shape_direct_task=$task_id/$EXPECTED_PER_PROFILE"
    SLURM_ARRAY_TASK_ID="$task_id" bash -l "$HPC_DIR/run_shape_strict_array_task.sh"
  done
  SHAPE_MAX_QC_FIELDS="${SHAPE_MAX_QC_FIELDS:-24}" \
  SHAPE_MAX_CROPS_PER_FIELD="${SHAPE_MAX_CROPS_PER_FIELD:-8}" \
  SHAPE_QC_CROP_SIZE="${SHAPE_QC_CROP_SIZE:-224}" \
    bash -l "$HPC_DIR/run_shape_strict_finalize.sh"
}

stage_audit() {
  "$PYTHON_BIN" -I "$SCRIPT_DIR/analysisi/06_audit_largetest_full_pipeline.py" \
    --input-root "$INPUT_ROOT" \
    --run-root "$OUT_ROOT" \
    --expected-per-profile "$EXPECTED_PER_PROFILE" \
    --pipeline-log "$PIPELINE_LOG"
}

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cellpose_largetest_grab_${USER}_$$"
export OMP_NUM_THREADS="${GRAB_CPUS:-2}"
export MKL_NUM_THREADS="${GRAB_CPUS:-2}"
export OPENBLAS_NUM_THREADS="${GRAB_CPUS:-2}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
if [[ "$PYTHON_BIN" != "$PYTHON_EXPECTED" ]]; then
  echo "Expected Python $PYTHON_EXPECTED, activated $PYTHON_BIN" >&2
  exit 2
fi

echo "pipeline=large_test_full_fusion_shape_strict_grab"
echo "project_dir=$PROJECT_DIR"
echo "input_root=$INPUT_ROOT"
echo "out_root=$OUT_ROOT"
echo "python_bin=$PYTHON_BIN"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
printf 'run_root=%s\nstarted=%s\n' "$OUT_ROOT" "$(date -Iseconds)" > "$OUT_ROOT/run_manifest.txt"

sha256sum \
  "$SCRIPT_DIR/01_segment_images.py" \
  "$SCRIPT_DIR/02_call_high_density_from_nuclei_masks.py" \
  "$SCRIPT_DIR/08_fuse_multichannel_classification.py" \
  "$SCRIPT_DIR/09_apply_shape_aware_nucleus_splits.py" \
  "$SCRIPT_DIR/11_render_shape_aware_nucleus_split_qc.py" \
  "$SCRIPT_DIR/10_merge_shape_strict_shards.py" \
  "$SCRIPT_DIR/12_finalize_shape_strict_qc.py" \
  "$SCRIPT_DIR/analysisi/06_audit_largetest_full_pipeline.py" \
  "$HPC_DIR/Parameter_calibration/05_run_largetest_full_pipeline_grab.sh" \
  > "$PROVENANCE_DIR/code_sha256.txt"
"$PYTHON_BIN" -m pip show cellpose torch numpy scipy scikit-image > "$PROVENANCE_DIR/python_packages.txt"

run_stage environment_preflight stage_environment
run_stage nuclei_gpu stage_nuclei
run_stage density_table stage_density
run_stage brightfield_gpu stage_cell_profile Brightfield
run_stage combined_gpu stage_cell_profile Combined
run_stage dead_gpu stage_cell_profile Dead
run_stage fusion_merge stage_fusion
run_stage shape_strict stage_shape_strict
run_stage final_audit stage_audit

printf 'completed=%s\nrun_root=%s\n' "$(date -Iseconds)" "$OUT_ROOT" > "$OUT_ROOT/_SUCCESS"
echo "pipeline_complete=1"
echo "final_run_root=$OUT_ROOT"
echo "final_audit=$OUT_ROOT/final_audit.json"
