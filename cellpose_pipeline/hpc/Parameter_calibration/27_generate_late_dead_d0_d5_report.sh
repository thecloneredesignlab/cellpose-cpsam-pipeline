#!/bin/bash
set -euo pipefail

BASE="${BASE:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide}"
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$BASE/results/Tests_and_Parameters_calibration/death_classification_consensus_optimization_20260725_213646}"
OUTPUT_DIR="${OUTPUT_DIR:-$CALIBRATION_ROOT/report_final}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"
NODE_ROOT="${NODE_ROOT:-/home/4482173/.local/opt/node-v24.18.0-linux-x64}"
REPORT_PLUGIN_ROOT="${REPORT_PLUGIN_ROOT:-/home/4482173/.local/share/data-analytics/0.2.8}"
FORCE_REPORT="${FORCE_REPORT:-0}"

for required in \
  "$CALIBRATION_ROOT/_SUCCESS" \
  "$CALIBRATION_ROOT/RUN_SUMMARY.txt" \
  "$CALIBRATION_ROOT/feature_cache/dataset_summary.json" \
  "$CALIBRATION_ROOT/optimization/best_configuration.json" \
  "$CALIBRATION_ROOT/optimization/anchor_metrics.json" \
  "$CALIBRATION_ROOT/optimization/FULL_CLASSIFICATION_GO_NO_GO.json" \
  "$CALIBRATION_ROOT/optimization/field_parameter_trials.csv" \
  "$CALIBRATION_ROOT/optimization/object_parameter_trials.csv" \
  "$CALIBRATION_ROOT/optimization/late_death_field_summary.csv" \
  "$CALIBRATION_ROOT/report/qc_inventory.csv" \
  "$PROJECT_DIR/cellpose_pipeline/report/generate_late_dead_d0_d5_calibration_report.py"; do
  [[ -e "$required" ]] || {
    echo "Required d0+d5 calibration report input is missing: $required" >&2
    exit 2
  }
done
[[ -x "$PYTHON_BIN" ]] || {
  echo "CellPose Python environment is unavailable: $PYTHON_BIN" >&2
  exit 2
}
[[ -x "$NODE_ROOT/bin/node" && -x "$NODE_ROOT/bin/npm" ]] || {
  echo "Node.js report runtime is unavailable: $NODE_ROOT" >&2
  exit 2
}
[[ -f "$REPORT_PLUGIN_ROOT/package.json" ]] || {
  echo "Data Analytics report environment is unavailable: $REPORT_PLUGIN_ROOT" >&2
  exit 2
}

export PATH="$NODE_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_d0_d5_report_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "calibration_root=$CALIBRATION_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "python_bin=$PYTHON_BIN"
echo "node_root=$NODE_ROOT"
echo "report_plugin_root=$REPORT_PLUGIN_ROOT"

REPORT_ARGS=(
  cellpose_pipeline/report/generate_late_dead_d0_d5_calibration_report.py
  --calibration-root "$CALIBRATION_ROOT"
  --output-dir "$OUTPUT_DIR"
  --plugin-root "$REPORT_PLUGIN_ROOT"
)
if [[ "$FORCE_REPORT" == "1" ]]; then
  REPORT_ARGS+=(--force)
fi
"$PYTHON_BIN" -I "${REPORT_ARGS[@]}"

HTML="$OUTPUT_DIR/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.html"
ARTIFACT="$OUTPUT_DIR/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.artifact.json"
RECEIPT="$OUTPUT_DIR/DEAD_CLASSIFICATION_D0_D5_CALIBRATION_REPORT.build.json"
test -s "$HTML"
test -s "$ARTIFACT"
test -s "$RECEIPT"
"$PYTHON_BIN" - "$RECEIPT" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["report_mode"] == "d0_d5_calibration"
assert receipt["completed_shards"] == 7360
assert receipt["failed_shards"] == 0
assert receipt["changed_d0_object_count"] == 0
assert receipt["operational_decision"] == "GO"
assert receipt["biological_accuracy_claimed"] is False
assert receipt["qc_grid_columns"] == 2
assert receipt["qc_grid_rows"] == 2
assert receipt["d0_qc_panels_per_case"] == 4
assert receipt["d5_qc_panels_per_case"] == 4
assert receipt["qc_composite_font_scale"] == 2
assert receipt["qc_composite_title_font_scale"] == 1
assert receipt["qc_composite_label_font_scale"] == 2
assert receipt["late_death_evidence_panel_embedded"] is False
assert receipt["classification_boundary_width"] == 2
assert receipt["html_enhancement"]["carousel_groups"] == 3
assert receipt["html_enhancement"]["carousel_slides"] == 8
assert sorted(receipt["html_enhancement"]["carousel_group_sizes"].values()) == [2, 3, 3]
assert receipt["html_enhancement"]["carousel_image_height_cap_pixels"] == 1800
assert receipt["html_enhancement"]["carousel_viewport_fit"] is True
assert receipt["html_enhancement"]["standalone_image_viewport_fit"] is True
assert receipt["html_enhancement"]["high_resolution_image_frames"] >= 8
assert receipt["html_enhancement"]["high_resolution_render_scale"] == 6
PY

echo "d0_d5_calibration_report_complete=1"
echo "html_report=$HTML"
echo "artifact_json=$ARTIFACT"
echo "build_receipt=$RECEIPT"
