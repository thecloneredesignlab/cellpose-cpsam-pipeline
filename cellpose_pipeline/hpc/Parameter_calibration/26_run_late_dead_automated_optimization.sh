#!/bin/bash
set -euo pipefail

BASE=/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide
PROJECT=/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:-$BASE/results/classification_20260721_102004}"
D0_AUDIT_ROOT="${D0_AUDIT_ROOT:-$BASE/results/dead_d0_classification_audit}"
PLATE_MAP="${PLATE_MAP:-$PROJECT/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv}"
CALIBRATION_PARENT="${CALIBRATION_PARENT:-$BASE/results/Tests_and_Parameters_calibration}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$CALIBRATION_PARENT/late_dead_trajectory_optimization_$RUN_STAMP}"
CALIBRATION_CODE_ROOT="${CALIBRATION_CODE_ROOT:-$PROJECT/cellpose_pipeline/scripts/Parameter_calibration}"
PRODUCTION_CODE_ROOT="${PRODUCTION_CODE_ROOT:-$PROJECT/cellpose_pipeline/scripts}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"
TRAJECTORY_WELLS="${TRAJECTORY_WELLS:-A9,B9,C2,D2,E2,F2,E9,F9,G9,H9}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-48}"
SEED="${SEED:-260723}"
TARGET_E9_DEAD_FRACTION="${TARGET_E9_DEAD_FRACTION:-0.95}"
TARGET_E9_HOLDOUT_DEAD_FRACTION="${TARGET_E9_HOLDOUT_DEAD_FRACTION:-0.90}"
MAX_CONTROL_FPR="${MAX_CONTROL_FPR:-0.01}"

if [[ "$(hostname -s)" != "hpctpa3pc0009" ]]; then
  echo "This calibration must run on hpctpa3pc0009; current host: $(hostname -s)" >&2
  exit 1
fi
for required in \
  "$CLASSIFICATION_ROOT" \
  "$D0_AUDIT_ROOT" \
  "$PLATE_MAP" \
  "$CALIBRATION_CODE_ROOT"; do
  [[ -e "$required" ]] || {
    echo "Required input is missing: $required" >&2
    exit 1
  }
done
for script in \
  27_build_late_dead_calibration_dataset.py \
  28_optimize_late_dead_rescue.py \
  29_render_late_dead_rescue_qc.py; do
  [[ -f "$CALIBRATION_CODE_ROOT/$script" ]] || {
    echo "Required calibration script is missing: $CALIBRATION_CODE_ROOT/$script" >&2
    exit 1
  }
done
for script in \
  "$PRODUCTION_CODE_ROOT/13_build_late_dead_trajectory_dataset.py" \
  "$PRODUCTION_CODE_ROOT/_shared/late_dead_trajectory_model.py"; do
  [[ -f "$script" ]] || {
    echo "Required shared production module is missing: $script" >&2
    exit 1
  }
done
[[ -x "$PYTHON_BIN" ]] || {
  echo "Python is unavailable: $PYTHON_BIN" >&2
  exit 1
}
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to reuse an existing calibration output: $OUT_ROOT" >&2
  exit 1
fi

mkdir -p \
  "$OUT_ROOT/logs" \
  "$OUT_ROOT/code_snapshot/Parameter_calibration" \
  "$OUT_ROOT/code_snapshot/_shared"
cp "$CALIBRATION_CODE_ROOT/27_build_late_dead_calibration_dataset.py" \
  "$OUT_ROOT/code_snapshot/Parameter_calibration/"
cp "$CALIBRATION_CODE_ROOT/28_optimize_late_dead_rescue.py" \
  "$OUT_ROOT/code_snapshot/Parameter_calibration/"
cp "$CALIBRATION_CODE_ROOT/29_render_late_dead_rescue_qc.py" \
  "$OUT_ROOT/code_snapshot/Parameter_calibration/"
cp "$PRODUCTION_CODE_ROOT/13_build_late_dead_trajectory_dataset.py" \
  "$OUT_ROOT/code_snapshot/"
cp "$PRODUCTION_CODE_ROOT/_shared/late_dead_trajectory_model.py" \
  "$OUT_ROOT/code_snapshot/_shared/"
CODE_SNAPSHOT="$OUT_ROOT/code_snapshot"
CALIBRATION_SNAPSHOT="$CODE_SNAPSHOT/Parameter_calibration"
LOG="$OUT_ROOT/logs/late_dead_trajectory_optimization.log"
exec > >(tee -a "$LOG") 2>&1

finish() {
  rc=$?
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_code=$rc"
  if [[ $rc -eq 0 ]]; then
    touch "$OUT_ROOT/_SUCCESS"
  else
    touch "$OUT_ROOT/_FAILED"
  fi
}
trap finish EXIT

export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$OUT_ROOT/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "python_bin=$PYTHON_BIN"
echo "git_sha=$(git -c safe.directory="$PROJECT" -C "$PROJECT" rev-parse HEAD)"
echo "classification_root=$CLASSIFICATION_ROOT"
echo "d0_audit_root=$D0_AUDIT_ROOT"
echo "plate_map=$PLATE_MAP"
echo "output_root=$OUT_ROOT"
echo "trajectory_wells=$TRAJECTORY_WELLS"
echo "parallel_workers=$PARALLEL_WORKERS"
echo "seed=$SEED"
echo "target_e9_dead_fraction=$TARGET_E9_DEAD_FRACTION"
echo "target_e9_holdout_dead_fraction=$TARGET_E9_HOLDOUT_DEAD_FRACTION"
echo "max_control_fpr=$MAX_CONTROL_FPR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "step=01_build_full_trajectory_dataset"
"$PYTHON_BIN" -I "$CALIBRATION_SNAPSHOT/27_build_late_dead_calibration_dataset.py" \
  --classification-root "$CLASSIFICATION_ROOT" \
  --d0-audit-root "$D0_AUDIT_ROOT" \
  --plate-map "$PLATE_MAP" \
  --out-dir "$OUT_ROOT" \
  --trajectory-wells "$TRAJECTORY_WELLS" \
  --trajectory-min-hours 2 \
  --trajectory-max-hours 168 \
  --d0-fields 320 \
  --workers "$PARALLEL_WORKERS" \
  --seed "$SEED"

echo "step=02_optimize_field_and_object_states"
"$PYTHON_BIN" -I "$CALIBRATION_SNAPSHOT/28_optimize_late_dead_rescue.py" \
  --dataset-root "$OUT_ROOT" \
  --out-dir "$OUT_ROOT/optimization" \
  --seed "$SEED" \
  --target-e9-dead-fraction "$TARGET_E9_DEAD_FRACTION" \
  --target-holdout-dead-fraction "$TARGET_E9_HOLDOUT_DEAD_FRACTION" \
  --max-control-false-positive-rate "$MAX_CONTROL_FPR" \
  --late-min-hours 72

echo "step=03_render_exhaustive_qc_and_html"
"$PYTHON_BIN" -I "$CALIBRATION_SNAPSHOT/29_render_late_dead_rescue_qc.py" \
  --optimization-root "$OUT_ROOT/optimization" \
  --out-dir "$OUT_ROOT/report" \
  --qc-wells E9,F9 \
  --key-hours 0,24,48,72,96,120,144,168 \
  --render-all-e9 \
  --render-all-d0 \
  --workers 12

"$PYTHON_BIN" - "$OUT_ROOT/optimization/best_configuration.json" "$OUT_ROOT/RUN_SUMMARY.txt" <<'PY'
import json
import sys

configuration = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = configuration["anchor_metrics"]
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write("metric_semantics=operational_anchors_not_biological_ground_truth\n")
    handle.write(f"method={configuration['method']}\n")
    handle.write(f"production_integration={configuration['production_integration']}\n")
    handle.write(
        f"e9_day5_baseline_dead_fraction={metrics['development_e9_original_day5_baseline_dead_fraction']}\n"
    )
    handle.write(
        f"e9_day5_final_dead_fraction={metrics['development_e9_original_day5_dead_fraction']}\n"
    )
    handle.write(
        f"e9_holdout_day5_min_dead_fraction={metrics['holdout_e9_day5_min_dead_fraction']}\n"
    )
    handle.write(
        f"e9_holdout_day5_median_dead_fraction={metrics['holdout_e9_day5_median_dead_fraction']}\n"
    )
    handle.write(
        f"f9_day5_median_dead_fraction={metrics['f9_day5_median_dead_fraction']}\n"
    )
    handle.write(
        f"blue_supported_anchor_preservation={metrics['blue_supported_anchor_preservation']}\n"
    )
    handle.write(
        f"temporal_remnant_anchor_recall={metrics['temporal_remnant_anchor_recall']}\n"
    )
    handle.write(
        f"temporal_remnant_anchor_count={metrics['temporal_remnant_anchor_count']}\n"
    )
    handle.write(
        "untreated_live_anchor_counterfactual_fpr="
        f"{metrics['untreated_live_anchor_counterfactual_fpr']}\n"
    )
    handle.write(f"changed_d0_object_count={metrics['changed_d0_object_count']}\n")
PY

echo "best_configuration=$OUT_ROOT/optimization/best_configuration.json"
echo "field_summary=$OUT_ROOT/optimization/late_death_field_summary.csv"
echo "html_report=$OUT_ROOT/report/LATE_DEATH_TRAJECTORY_REFINEMENT_REPORT.html"
echo "qc_inventory=$OUT_ROOT/report/qc_inventory.csv"
echo "run_summary=$OUT_ROOT/RUN_SUMMARY.txt"
echo "late_dead_trajectory_optimization_complete=1"
