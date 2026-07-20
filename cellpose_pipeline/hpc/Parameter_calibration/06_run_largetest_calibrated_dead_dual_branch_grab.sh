#!/bin/bash
set -Eeuo pipefail

BASE="${BASE:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide}"
PROJECT_DIR="${PROJECT_DIR:-$BASE/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-$BASE/SUM159_AC_Exp1_SeparateImages_largetest}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/test}"
OUT_ROOT="${OUT_ROOT:-$RESULTS_ROOT/largetest_calibrated_dead_dual_branch_$(date +%Y%m%d_%H%M%S)}"
FULL_CALIBRATION_ROOT="${FULL_CALIBRATION_ROOT:-$RESULTS_ROOT/dead_combined_blue_full_calibration_v2_20260711/final}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$FULL_CALIBRATION_ROOT/dead_combined_blue_calibration.json}"
CALIBRATION_MAP="${CALIBRATION_MAP:-$FULL_CALIBRATION_ROOT/dead_combined_blue_image_calibration_map.csv}"
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
TASK_LIST="$STATUS_DIR/keys.txt"
NUCLEATED_ROOT="$OUT_ROOT/nucleated_only"

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" && "${RESUME:-0}" != "1" ]]; then
  echo "Output root is not empty: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$PROVENANCE_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

CURRENT_STAGE=initialization
on_error() {
  local code=$?
  trap - ERR
  printf 'stage=%s\nexit_code=%s\ntime=%s\n' "$CURRENT_STAGE" "$code" "$(date -Iseconds)" > "$STATUS_DIR/FAILED"
  exit "$code"
}
trap on_error ERR
printf 'stage\tstart_epoch\tend_epoch\telapsed_sec\n' > "$TIMINGS"

run_stage() {
  local stage=$1
  shift
  CURRENT_STAGE="$stage"
  local start end
  start="$(date +%s)"
  echo "stage_start=$stage"
  "$@"
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "$stage" "$start" "$end" "$((end-start))" >> "$TIMINGS"
  printf 'completed=%s\n' "$(date -Iseconds)" > "$STATUS_DIR/$stage.done"
}

require_count() {
  local dir=$1 pattern=$2 expected=$3 label=$4
  local observed
  observed="$(find "$dir" -maxdepth 1 -type f -name "$pattern" | wc -l | tr -d '[:space:]')"
  echo "artifact_count label=$label observed=$observed expected=$expected"
  [[ "$observed" -eq "$expected" ]]
}

build_keys() {
  find "$INPUT_ROOT/Combined" -maxdepth 1 -type f -name '*.tif' | sort | sed -E 's#^.*/##; s/\.tif$//; s/^SUM159_AC_//' > "$TASK_LIST"
  require_count "$INPUT_ROOT/Combined" '*.tif' "$EXPECTED_PER_PROFILE" raw_combined
  [[ "$(wc -l < "$TASK_LIST" | tr -d '[:space:]')" -eq "$EXPECTED_PER_PROFILE" ]]
}

stage_environment() {
  [[ "$(hostname -s)" == "$EXPECTED_HOST" ]]
  [[ -d "$PROJECT_DIR" && -d "$INPUT_ROOT" && -f "$CALIBRATION_JSON" && -f "$CALIBRATION_MAP" ]]
  for profile in Brightfield Combined Dead Nuclei; do
    require_count "$INPUT_ROOT/$profile" '*.tif' "$EXPECTED_PER_PROFILE" "raw_$profile"
  done
  build_keys
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$PROVENANCE_DIR/gpu.csv"
  "$PYTHON_BIN" -I "$SCRIPT_DIR/01_segment_images.py" \
    --dir "$INPUT_ROOT" --recursive --run-name . --out-root "$OUT_ROOT" \
    --profile-mode auto --skip-unknown-profiles --preflight-only --check-models --use-gpu
  cp "$CALIBRATION_JSON" "$PROVENANCE_DIR/dead_combined_blue_calibration.json"
  cp "$CALIBRATION_MAP" "$PROVENANCE_DIR/dead_combined_blue_image_calibration_map.csv"
}

segment_profile() {
  local profile=$1
  shift
  "$PYTHON_BIN" -I "$SCRIPT_DIR/01_segment_images.py" \
    --dir "$INPUT_ROOT/$profile" --run-name . --out-root "$OUT_ROOT/$profile" \
    --summary-name segmentation_summary.csv --profile-mode auto --skip-unknown-profiles \
    --flat-profile-output --segmentation-only --use-gpu --gpu-device 0 "$@"
  require_count "$OUT_ROOT/$profile/segmentations" '*_cp_masks.tif' "$EXPECTED_PER_PROFILE" "$profile"
}

stage_nuclei() {
  segment_profile Nuclei --no-enable-high-density-profiles --no-auto-generate-density-calls
  require_count "$OUT_ROOT/Nuclei/nucleus_core_seeds" '*_core_masks.tif' "$EXPECTED_PER_PROFILE" nuclei_core
}

stage_density() {
  "$PYTHON_BIN" -I "$SCRIPT_DIR/02_call_high_density_from_nuclei_masks.py" \
    --run-dir "$OUT_ROOT" --out-csv "$OUT_ROOT/qc/density_calls.csv" \
    --nuclei-count-threshold 4000 --nuclei-mask-fraction-threshold 0.22 \
    --nuclei-median-nn-threshold 16.0 --no-enable-mask-fraction-trigger
}

stage_cells() {
  segment_profile Brightfield --enable-high-density-profiles --high-density-calls-csv "$OUT_ROOT/qc/density_calls.csv" --no-auto-generate-density-calls
  segment_profile Combined --enable-high-density-profiles --high-density-calls-csv "$OUT_ROOT/qc/density_calls.csv" --no-auto-generate-density-calls
}

stage_dead_consensus() {
  local dead_image key combined_image
  while IFS= read -r dead_image; do
    key="$(basename "$dead_image")"
    key="${key%.tif}"
    key="${key#SUM159_AC_}"
    key="${key#Exp1_}"
    key="${key#Dead_}"
    combined_image="$INPUT_ROOT/Combined/SUM159_AC_${key}.tif"
    "$PYTHON_BIN" -I "$SCRIPT_DIR/04_segment_dead_with_combined_blue_consensus.py" \
      --dead-image "$dead_image" --combined-image "$combined_image" --out-root "$OUT_ROOT/Dead" \
      --calibration-json "$CALIBRATION_JSON" --calibration-map "$CALIBRATION_MAP" \
      --dead-calibration-mode bounded-background --dead-model cpsam_v2 --dead-diameter 22 --dead-cellprob-threshold -2.75 \
      --blue-model cpsam --blue-transform blue_excess_mean --blue-diameter 22 --blue-cellprob-threshold -3.0 \
      --use-gpu --force
  done < <(find "$INPUT_ROOT/Dead" -maxdepth 1 -type f -name '*.tif' | sort)
  "$PYTHON_BIN" -I "$SCRIPT_DIR/05_merge_dead_consensus_outputs.py" --dead-run "$OUT_ROOT/Dead" --expected-keys "$TASK_LIST"
  "$PYTHON_BIN" -I "$SCRIPT_DIR/Parameter_calibration/24_render_dead_consensus_qc.py" --input-root "$INPUT_ROOT" --dead-run "$OUT_ROOT/Dead" --out-dir "$OUT_ROOT/Dead/qc"
  require_count "$OUT_ROOT/Dead/segmentations" '*_cp_masks.tif' "$EXPECTED_PER_PROFILE" dead_consensus
}

run_fusion() {
  local out_dir=$1 combined_run=$2 brightfield_run=$3
  "$PYTHON_BIN" -I "$SCRIPT_DIR/08_fuse_multichannel_classification.py" \
    --run-root "$OUT_ROOT" --combined-run "$combined_run" --brightfield-run "$brightfield_run" \
    --dead-run "$OUT_ROOT/Dead" --nuclei-run "$OUT_ROOT/Nuclei" --input-root "$INPUT_ROOT" \
    --out-dir "$out_dir" --force
  [[ ! -s "$out_dir/failures.csv" ]]
}

stage_fusion_all() {
  run_fusion "$OUT_ROOT/classification_fusion" "$OUT_ROOT/Combined" "$OUT_ROOT/Brightfield"
}

stage_nucleated_branch() {
  "$PYTHON_BIN" -I "$SCRIPT_DIR/07_build_nucleated_cell_branch.py" \
    --run-root "$OUT_ROOT" --input-root "$INPUT_ROOT" --out-root "$NUCLEATED_ROOT" --render-qc --force
  run_fusion "$OUT_ROOT/classification_fusion_nucleated_only" "$NUCLEATED_ROOT/Combined" "$NUCLEATED_ROOT/Brightfield"
}

run_shape_branch() {
  local cell_root=$1 fusion_root=$2 shape_root=$3
  export PROJECT_DIR INPUT_ROOT TASK_LIST_SHAPE="$TASK_LIST" RUN_ROOT="$OUT_ROOT"
  export CELL_RUN_ROOT="$cell_root" FUSION_ROOT="$fusion_root" SHAPE_ROOT="$shape_root"
  export SLURM_CPUS_PER_TASK="${SHAPE_CPUS:-2}"
  local task_id
  for task_id in $(seq 1 "$EXPECTED_PER_PROFILE"); do
    SLURM_ARRAY_TASK_ID="$task_id" FORCE_SHAPE_STRICT=1 bash -l "$HPC_DIR/run_shape_strict_array_task.sh"
  done
  bash -l "$HPC_DIR/run_shape_strict_finalize.sh"
}

stage_shape() {
  run_shape_branch "$OUT_ROOT" "$OUT_ROOT/classification_fusion" "$OUT_ROOT/shape_strict"
  run_shape_branch "$NUCLEATED_ROOT" "$OUT_ROOT/classification_fusion_nucleated_only" "$OUT_ROOT/shape_strict_nucleated_only"
}

stage_audit() {
  "$PYTHON_BIN" -I - "$OUT_ROOT" "$EXPECTED_PER_PROFILE" <<'PY'
import csv, json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
checks = {}
for profile in ("Brightfield", "Combined", "Dead", "Nuclei"):
    checks[f"{profile}_masks"] = len(list((root / profile / "segmentations").glob("*_cp_masks.tif"))) == expected
for branch in ("classification_fusion", "classification_fusion_nucleated_only"):
    rows = list(csv.DictReader((root / branch / "summaries" / "cell_count_summary.csv").open()))
    checks[f"{branch}_rows"] = len(rows) == expected
for branch in ("shape_strict", "shape_strict_nucleated_only"):
    payload = json.loads((root / branch / "shape_strict_summary.json").read_text())
    checks[f"{branch}_fields"] = int(payload["n_fields"]) == expected
checks["dead_consensus_success"] = (root / "Dead" / "_SUCCESS").is_file()
checks["nucleated_branch_success"] = (root / "nucleated_only" / "_SUCCESS").is_file()
(root / "final_audit.json").write_text(json.dumps({"checks": checks, "all_passed": all(checks.values())}, indent=2, sort_keys=True) + "\n")
if not all(checks.values()):
    raise SystemExit(checks)
print("final_audit_passed=1")
PY
}

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_dual_branch_${USER}_$$"
export OMP_NUM_THREADS="${GRAB_CPUS:-2}" MKL_NUM_THREADS="${GRAB_CPUS:-2}" OPENBLAS_NUM_THREADS="${GRAB_CPUS:-2}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
[[ "$PYTHON_BIN" == "$PYTHON_EXPECTED" ]]

echo "pipeline=largetest_calibrated_dead_dual_branch"
echo "out_root=$OUT_ROOT"
sha256sum "$SCRIPT_DIR/01_segment_images.py" "$SCRIPT_DIR/03_calibrate_dead_combined_blue.py" \
  "$SCRIPT_DIR/07_build_nucleated_cell_branch.py" "$SCRIPT_DIR/04_segment_dead_with_combined_blue_consensus.py" \
  "$SCRIPT_DIR/05_merge_dead_consensus_outputs.py" "$HPC_DIR/Parameter_calibration/06_run_largetest_calibrated_dead_dual_branch_grab.sh" \
  > "$PROVENANCE_DIR/code_sha256.txt"

run_stage environment_preflight stage_environment
run_stage nuclei_gpu stage_nuclei
run_stage density_table stage_density
run_stage bf_combined_gpu stage_cells
run_stage dead_consensus_gpu stage_dead_consensus
run_stage classification_all stage_fusion_all
run_stage classification_nucleated_only stage_nucleated_branch
run_stage shape_strict_dual_branch stage_shape
run_stage final_audit stage_audit
printf 'completed=%s\nrun_root=%s\n' "$(date -Iseconds)" "$OUT_ROOT" > "$OUT_ROOT/_SUCCESS"
echo "pipeline_complete=1"
echo "final_run_root=$OUT_ROOT"
