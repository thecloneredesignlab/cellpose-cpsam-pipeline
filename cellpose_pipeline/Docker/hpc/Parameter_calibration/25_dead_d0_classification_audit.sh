#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

BASE=/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide
PROJECT=/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3
HPC_PROJECT_ROOT="${PROJECT}"
export HPC_PROJECT_ROOT
RAW="$BASE/20260626_SUM159_AC_Exp1_SeparateImages"
SOURCE_RESULTS="$BASE/results/full_fusion_shape_strict_20260711_155940"
DEFAULT_OUT="$BASE/results/dead_d0_classification_audit"
OUT="${1:-$DEFAULT_OUT}"
HISTORICAL_REFERENCE_ROOT="${HISTORICAL_REFERENCE_ROOT:-$OUT/historical_reference}"
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python
hpc_container_ignore_host_runtime "${NODE_ROOT:-}"
unset NODE_ROOT
REPORT_PLUGIN_ROOT="${REPORT_PLUGIN_ROOT:?Set REPORT_PLUGIN_ROOT to the deployed Data Analytics plugin directory}"
WORKER="$PROJECT/cellpose_pipeline/Docker/hpc/Parameter_calibration/25_dead_d0_classification_audit.sh"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-16}"
TASKS="$OUT/d0_keys.txt"
LOG="$OUT/logs/d0_full_rerun_parallel.log"

if [[ "${1:-}" == "__field" ]]; then
  [[ $# -eq 5 ]] || { echo "Internal field usage error" >&2; exit 2; }
  KEY="$2"
  BRANCH="$3"
  CLASSIFICATION_OUT="$4"
  RUN_OUT="$5"
  FIELD_RECORD="$RUN_OUT/field_manifest/records/${KEY%%_*}/$KEY.json"
  [[ -f "$FIELD_RECORD" ]] || { echo "Missing field record: $FIELD_RECORD" >&2; exit 1; }
  export PYTHONNOUSERSITE=1 KMP_DUPLICATE_LIB_OK=TRUE
  unset PYTHONPATH PYTHONHOME
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export MPLCONFIGDIR="$RUN_OUT/.mplconfig/$BRANCH/$KEY"
  mkdir -p "$MPLCONFIGDIR"
  cd "$PROJECT"
  exec "$PYTHON_BIN" -I cellpose_pipeline/scripts/08_fuse_multichannel_classification.py \
    --field-record "$FIELD_RECORD" \
    --key "$KEY" \
    --timepoint d0 \
    --cell-mask-branch "$BRANCH" \
    --out-dir "$CLASSIFICATION_OUT" \
    --per-key-only
fi

if [[ "$(hostname -s)" != "hpctpa3pc0009" ]]; then
  echo "This workflow must run on hpctpa3pc0009; current host: $(hostname -s)" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python environment is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$WORKER" ]]; then
  echo "Unified workflow script is unavailable or not executable: $WORKER" >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Container Node.js report runtime is unavailable" >&2
  exit 1
fi
if [[ ! -f "$REPORT_PLUGIN_ROOT/package.json" ]]; then
  echo "Data Analytics report plugin is unavailable: $REPORT_PLUGIN_ROOT" >&2
  exit 1
fi
for required_reference in \
  "$HISTORICAL_REFERENCE_ROOT/context_aware_all_d0/predictions" \
  "$HISTORICAL_REFERENCE_ROOT/context_aware_all_d0/qc/label_overlays" \
  "$HISTORICAL_REFERENCE_ROOT/strong_direct_all_d0/predictions" \
  "$HISTORICAL_REFERENCE_ROOT/strong_direct_all_d0/qc/label_overlays" \
  "$HISTORICAL_REFERENCE_ROOT/automated_reference_audit" \
  "$HISTORICAL_REFERENCE_ROOT/automated_reference_audit_final" \
  "$HISTORICAL_REFERENCE_ROOT/qc"; do
  [[ -d "$required_reference" ]] || { echo "Missing frozen historical reference directory: $required_reference" >&2; exit 1; }
done
for required_reference in \
  "$HISTORICAL_REFERENCE_ROOT/d0_branch_before_after_metrics.csv" \
  "$HISTORICAL_REFERENCE_ROOT/debugging_stage_provenance.json" \
  "$HISTORICAL_REFERENCE_ROOT/FROZEN_REFERENCE_SHA256.txt"; do
  [[ -f "$required_reference" ]] || { echo "Missing frozen historical reference file: $required_reference" >&2; exit 1; }
done
cmp -s \
  "$PROJECT/cellpose_pipeline/report/dead_classification_debugging_stages.json" \
  "$HISTORICAL_REFERENCE_ROOT/debugging_stage_provenance.json" || {
    echo "Frozen reference provenance does not match the checked-out report provenance" >&2
    exit 1
  }
(
  cd "$HISTORICAL_REFERENCE_ROOT"
  sha256sum -c FROZEN_REFERENCE_SHA256.txt
)

mkdir -p "$OUT/logs"
rm -f "$OUT/_SUCCESS"
if [[ -f "$OUT/_FAILED" ]]; then
  mv "$OUT/_FAILED" "$OUT/_FAILED.$(date +%H%M%S)"
fi
exec > >(tee -a "$LOG") 2>&1

finish() {
  rc=$?
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_code=$rc"
  if [[ $rc -eq 0 ]]; then
    touch "$OUT/_SUCCESS"
  else
    touch "$OUT/_FAILED"
  fi
}
trap finish EXIT

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "parallel_workers=$PARALLEL_WORKERS"
echo "git_sha=$(git -c safe.directory="$PROJECT" -C "$PROJECT" rev-parse HEAD)"
echo "raw_root=$RAW"
echo "source_result_root=$SOURCE_RESULTS"
echo "output_root=$OUT"
echo "historical_reference_root=$HISTORICAL_REFERENCE_ROOT"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
HPC_CONTAINER_BINDS="${HPC_CONTAINER_BINDS:+${HPC_CONTAINER_BINDS},}${REPORT_PLUGIN_ROOT}"
export HPC_CONTAINER_BINDS REPORT_PLUGIN_ROOT
export MPLCONFIGDIR="$OUT/.mplconfig/main"
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT"

echo "step=01_build_d0_manifest"
"$PYTHON_BIN" -I cellpose_pipeline/scripts/06_build_postsegmentation_field_manifest.py \
  --input-root "$RAW" \
  --run-root "$SOURCE_RESULTS" \
  --timepoint d0 \
  --out-dir "$OUT/field_manifest"

awk -F '\t' 'NR > 1 {print $1}' "$OUT/field_manifest/field_manifest.tsv" > "$TASKS"
task_count=$(awk 'END {print NR}' "$TASKS")
echo "task_count=$task_count"
[[ "$task_count" -eq 320 ]]

echo "step=02_classify_original_branch_parallel"
mkdir -p "$OUT/classification_original"
xargs -P "$PARALLEL_WORKERS" -I {} \
  "$WORKER" __field "{}" original "$OUT/classification_original" "$OUT" < "$TASKS"
"$PYTHON_BIN" -I cellpose_pipeline/scripts/08_fuse_multichannel_classification.py \
  --out-dir "$OUT/classification_original" \
  --timepoint d0 \
  --merge-summaries-only

echo "step=03_classify_nucleated_only_branch_parallel"
mkdir -p "$OUT/classification_nucleated_only"
xargs -P "$PARALLEL_WORKERS" -I {} \
  "$WORKER" __field "{}" nucleated "$OUT/classification_nucleated_only" "$OUT" < "$TASKS"
"$PYTHON_BIN" -I cellpose_pipeline/scripts/08_fuse_multichannel_classification.py \
  --out-dir "$OUT/classification_nucleated_only" \
  --timepoint d0 \
  --merge-summaries-only

echo "step=04_audit_both_branches"
"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/25_audit_dead_classification_without_ground_truth.py \
  --branch "original=$OUT/classification_original" \
  --branch "nucleated_only=$OUT/classification_nucleated_only" \
  --timepoint d0 \
  --out-dir "$OUT/automated_audit" \
  --annotation-out-dir "$OUT/annotations"

echo "step=05_stress_test_both_branches"
"$PYTHON_BIN" -I cellpose_pipeline/scripts/Parameter_calibration/26_stress_test_dead_object_detection.py \
  --field-record-root "$OUT/field_manifest/records" \
  --branch "original=$OUT/classification_original" \
  --branch "nucleated_only=$OUT/classification_nucleated_only" \
  --timepoint d0 \
  --out-dir "$OUT/detector_stress"

echo "step=06_validate_outputs"
for branch in classification_original classification_nucleated_only; do
  feature_count=$(find "$OUT/$branch/features" -maxdepth 1 -type f -name '*00d00h00m_per_cell_fusion_features.csv' | wc -l)
  prediction_count=$(find "$OUT/$branch/predictions" -maxdepth 1 -type f -name '*00d00h00m_per_cell_predictions.csv' | wc -l)
  object_count=$(find "$OUT/$branch/dead_objects" -maxdepth 1 -type f -name '*00d00h00m_dead_object_features.csv' | wc -l)
  summary_count=$(find "$OUT/$branch/summaries" -maxdepth 1 -type f -name '*00d00h00m_summary.csv' | wc -l)
  cell_qc_count=$(find "$OUT/$branch/qc/label_overlays" -maxdepth 1 -type f -name '*00d00h00m_state_overlay.png' | wc -l)
  object_qc_count=$(find "$OUT/$branch/qc/dead_object_overlays" -maxdepth 1 -type f -name '*00d00h00m_dead_object_overlay.png' | wc -l)
  overlap_qc_count=$(find "$OUT/$branch/qc/overlap_state_overlays" -maxdepth 1 -type f -name '*00d00h00m_overlap_state_overlay.png' | wc -l)
  echo "$branch features=$feature_count predictions=$prediction_count dead_objects=$object_count summaries=$summary_count cell_qc=$cell_qc_count object_qc=$object_qc_count overlap_qc=$overlap_qc_count"
  [[ "$feature_count" -eq 320 ]]
  [[ "$prediction_count" -eq 320 ]]
  [[ "$object_count" -eq 320 ]]
  [[ "$summary_count" -eq 320 ]]
  [[ "$cell_qc_count" -eq 320 ]]
  [[ "$object_qc_count" -eq 320 ]]
  [[ "$overlap_qc_count" -eq 320 ]]
done

manifest_count=$(find "$OUT/field_manifest/records" -type f -name '*00d00h00m.json' | wc -l)
audit_field_count=$(awk 'END {print NR - 1}' "$OUT/automated_audit/per_field_metrics.csv")
audit_qc_count=$(awk 'END {print NR - 1}' "$OUT/automated_audit/qc_inventory.csv")
stress_field_count=$(awk 'END {print NR - 1}' "$OUT/detector_stress/detection_stress_per_field.csv")
annotation_count=$(awk 'END {print NR - 1}' "$OUT/annotations/final_multilevel_annotations.csv")
echo "manifest_count=$manifest_count"
echo "audit_field_rows=$audit_field_count"
echo "audit_qc_rows=$audit_qc_count"
echo "stress_field_rows=$stress_field_count"
echo "annotation_rows=$annotation_count"
[[ "$manifest_count" -eq 320 ]]
[[ "$audit_field_count" -eq 640 ]]
[[ "$audit_qc_count" -eq 640 ]]
[[ "$stress_field_count" -eq 640 ]]

echo "step=07_generate_html_report"
"$PYTHON_BIN" -I cellpose_pipeline/report/generate_dead_classification_improvement_report.py \
  --audit-root "$OUT" \
  --report-mode historical-comparison \
  --historical-reference-root "$HISTORICAL_REFERENCE_ROOT" \
  --input-root "$RAW" \
  --result-root "$SOURCE_RESULTS" \
  --timepoint d0 \
  --artifact-json "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.artifact.json" \
  --output-html "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.html" \
  --build-receipt "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.build.json" \
  --plugin-root "$REPORT_PLUGIN_ROOT" \
  --force
test -s "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.html"
test -s "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.artifact.json"
test -s "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.build.json"
"$PYTHON_BIN" - "$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.build.json" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["report_mode"] == "historical_comparison", receipt.get("report_mode")
assert receipt["dataset_rows"]["debugging_stages"] == 7, receipt["dataset_rows"]
assert receipt["html_enhancement"]["high_resolution_image_frames"] == 7, receipt["html_enhancement"]
assert len(receipt["expanded_live_override_fields"]) == 6
assert len(receipt["expanded_death_miss_fields"]) == 6
PY

{
  echo "git_sha=$(git -c safe.directory="$PROJECT" -C "$PROJECT" rev-parse HEAD)"
  echo "timepoint=00d00h00m"
  echo "selected_fields=320"
  echo "parallel_workers=$PARALLEL_WORKERS"
  echo "raw_root=$RAW"
  echo "source_result_root=$SOURCE_RESULTS"
  echo "classification_original=$OUT/classification_original"
  echo "classification_nucleated_only=$OUT/classification_nucleated_only"
  echo "automated_audit=$OUT/automated_audit"
  echo "annotations=$OUT/annotations"
  echo "detector_stress=$OUT/detector_stress"
  echo "report_mode=historical_comparison"
  echo "historical_reference_root=$HISTORICAL_REFERENCE_ROOT"
  echo "historical_report_input_map=$OUT/report_inputs/historical_comparison/REPORT_INPUT_MAP.json"
  echo "debugging_stage_provenance=$HISTORICAL_REFERENCE_ROOT/debugging_stage_provenance.json"
  echo "html_report=$OUT/DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.html"
  echo "annotation_rows=$annotation_count"
} > "$OUT/RUN_SUMMARY.txt"

echo "d0_full_rerun_complete=1"
