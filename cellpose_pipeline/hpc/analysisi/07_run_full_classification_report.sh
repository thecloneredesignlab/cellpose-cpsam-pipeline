#!/bin/bash
set -euo pipefail

BASE="${BASE:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide}"
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
PREVIOUS_CLASSIFICATION_ROOT="${PREVIOUS_CLASSIFICATION_ROOT:-$BASE/results/classification_20260721_102004}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$BASE/results/Tests_and_Parameters_calibration/death_classification_consensus_optimization_20260725_213646}"
PLATE_MAP="${PLATE_MAP:-$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv}"
WORKFLOW_PDF="${WORKFLOW_PDF:-$PROJECT_DIR/docs/death_classification_workflow.pdf}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/analysis/reports}"
PYTHON_BIN="${PYTHON_BIN:-/home/4482173/.conda/envs/cellpose_cpsam/bin/python}"
NODE_ROOT="${NODE_ROOT:-/home/4482173/.local/opt/node-v24.18.0-linux-x64}"
REPORT_PLUGIN_ROOT="${REPORT_PLUGIN_ROOT:-/home/4482173/.local/share/data-analytics/0.2.8}"
EXPECTED_FIELDS_PER_BRANCH="${EXPECTED_FIELDS_PER_BRANCH:-27200}"
EXPECTED_TIMEPOINTS="${EXPECTED_TIMEPOINTS:-85}"
EXPECTED_DOSE_RESPONSE_FILES="${EXPECTED_DOSE_RESPONSE_FILES:-123}"
FORCE_REPORT="${FORCE_REPORT:-0}"

for required in \
  "$RESULT_ROOT/workflow_status/late_death_trajectory/_SUCCESS" \
  "$RESULT_ROOT/workflow_status/dose_response/_SUCCESS" \
  "$RESULT_ROOT/SUBMISSION_SUMMARY.txt" \
  "$RESULT_ROOT/late_death_refinement/production_configuration.json" \
  "$RESULT_ROOT/late_death_refinement/refinement_summary.csv" \
  "$RESULT_ROOT/late_death_refinement/FULL_CLASSIFICATION_GO_NO_GO.json" \
  "$RESULT_ROOT/classification_consensus/summaries/cell_count_summary.csv" \
  "$PREVIOUS_CLASSIFICATION_ROOT/classification_fusion/summaries/cell_count_summary.csv" \
  "$CALIBRATION_ROOT/optimization/best_configuration.json" \
  "$CALIBRATION_ROOT/optimization/FULL_CLASSIFICATION_GO_NO_GO.json" \
  "$PLATE_MAP" \
  "$WORKFLOW_PDF" \
  "$PROJECT_DIR/cellpose_pipeline/report/generate_full_classification_report.py"; do
  [[ -e "$required" ]] || {
    echo "Required full-classification report input is missing: $required" >&2
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_full_report_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR" "$RESULT_ROOT/workflow_status/full_classification_report"

cd "$PROJECT_DIR"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "result_root=$RESULT_ROOT"
echo "previous_classification_root=$PREVIOUS_CLASSIFICATION_ROOT"
echo "calibration_root=$CALIBRATION_ROOT"
echo "workflow_pdf=$WORKFLOW_PDF"
echo "output_dir=$OUTPUT_DIR"
echo "python_bin=$PYTHON_BIN"
echo "node_root=$NODE_ROOT"
echo "report_plugin_root=$REPORT_PLUGIN_ROOT"

REPORT_ARGS=(
  cellpose_pipeline/report/generate_full_classification_report.py
  --classification-root "$RESULT_ROOT"
  --previous-classification-root "$PREVIOUS_CLASSIFICATION_ROOT"
  --calibration-root "$CALIBRATION_ROOT"
  --plate-map "$PLATE_MAP"
  --workflow-pdf "$WORKFLOW_PDF"
  --output-dir "$OUTPUT_DIR"
  --plugin-root "$REPORT_PLUGIN_ROOT"
  --expected-fields-per-branch "$EXPECTED_FIELDS_PER_BRANCH"
  --expected-timepoints "$EXPECTED_TIMEPOINTS"
  --expected-dose-response-files "$EXPECTED_DOSE_RESPONSE_FILES"
)
if [[ "$FORCE_REPORT" == "1" ]]; then
  REPORT_ARGS+=(--force)
fi
"$PYTHON_BIN" -I "${REPORT_ARGS[@]}"

HTML="$OUTPUT_DIR/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html"
ARTIFACT="$OUTPUT_DIR/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.artifact.json"
RECEIPT="$OUTPUT_DIR/DEAD_CLASSIFICATION_FULL_COHORT_REPORT.build.json"
test -s "$HTML"
test -s "$ARTIFACT"
test -s "$RECEIPT"
"$PYTHON_BIN" - "$RECEIPT" "$EXPECTED_FIELDS_PER_BRANCH" "$EXPECTED_DOSE_RESPONSE_FILES" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected_fields = int(sys.argv[2])
expected_dose_files = int(sys.argv[3])
assert receipt["report_mode"] == "full_cohort_classification"
assert receipt["fields_per_branch"] == expected_fields
assert receipt["refinement_rows"] == 2 * expected_fields
assert receipt["refinement_failures"] == 0
assert receipt["dose_response_files"] == expected_dose_files
assert receipt["qc_sample_count"] == 36
assert receipt["qc_condition_count"] == 6
assert receipt["qc_ploidy_levels"] == ["2N", "4N"]
assert receipt["qc_time_levels_hours"] == [0.0, 72.0, 120.0]
assert receipt["previous_classification_root"].endswith("/classification_20260721_102004")
assert receipt["qc_panels_per_sample"] == 10
assert receipt["qc_grid_columns"] == 4
assert receipt["qc_grid_rows"] == 3
assert receipt["qc_composite_font_scale"] == 2
assert receipt["qc_composite_title_font_scale"] == 1
assert receipt["qc_composite_label_font_scale"] == 2
assert receipt["workflow_figure_embedded"] is True
assert receipt["workflow_pdf"].endswith("/docs/death_classification_workflow.pdf")
assert receipt["classification_boundary_width"] == 2
assert len(receipt["d0_uncertainty_cases"]) == 2
assert receipt["dose_response_embedded_panel_count"] == 18
assert receipt["well_count_plot_embedded"] is True
assert receipt["well_count_plot_pdf"].endswith("/well_live_dead_counts_over_time.pdf")
assert receipt["html_enhancement"]["carousel_groups"] == 16
assert receipt["html_enhancement"]["carousel_slides"] == 56
assert sorted(receipt["html_enhancement"]["carousel_group_sizes"].values()) == [2] + [3] * 12 + [6] * 3
assert receipt["html_enhancement"]["carousel_image_height_cap_pixels"] == 1800
assert receipt["html_enhancement"]["carousel_viewport_fit"] is True
assert receipt["html_enhancement"]["standalone_image_viewport_fit"] is True
assert receipt["html_enhancement"]["high_resolution_image_frames"] >= 57
assert receipt["html_enhancement"]["high_resolution_render_scale"] == 6
PY

touch "$RESULT_ROOT/workflow_status/full_classification_report/_SUCCESS"
echo "full_classification_report_complete=1"
echo "html_report=$HTML"
echo "artifact_json=$ARTIFACT"
echo "build_receipt=$RECEIPT"
