#!/bin/bash
set -euo pipefail

# Submit only the CPU post-segmentation analysis chain.  Existing Nuclei,
# Brightfield, Combined, and Dead segmentation outputs are treated as immutable
# prerequisites and are never resubmitted by this script.
PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
RUN_ROOT="${RUN_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940}"
TASK_LIST_FUSION="${TASK_LIST_FUSION:-$PROJECT_DIR/cellpose_pipeline/hpc/task_lists/cpsam_separate_20260711_160239_2944836_fusion_keys.tsv}"

SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/hpc"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$RUN_ROOT/workflow_status/postsegmentation_manifest}"
NUCLEATED_BRANCH_ROOT="${NUCLEATED_BRANCH_ROOT:-$RUN_ROOT/nucleated_only}"
FUSION_OUT_DIR="${FUSION_OUT_DIR:-$RUN_ROOT/classification_fusion}"
NUCLEATED_FUSION_OUT_DIR="${NUCLEATED_FUSION_OUT_DIR:-$RUN_ROOT/classification_fusion_nucleated_only}"
SHAPE_ROOT="${SHAPE_ROOT:-$RUN_ROOT/shape_strict}"
NUCLEATED_SHAPE_ROOT="${NUCLEATED_SHAPE_ROOT:-$RUN_ROOT/shape_strict_nucleated_only}"
POSTSEG_ARRAY_MAX_CONCURRENT="${POSTSEG_ARRAY_MAX_CONCURRENT:-256}"

SBATCH_PARTITION="${SBATCH_PARTITION:-red}"
SBATCH_QOS="${SBATCH_QOS:-xxlarge}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-tao.li2}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_JOB_PREFIX="${SBATCH_JOB_PREFIX:-cpsam_postseg_rerun}"
FORCE_FIELD_MANIFEST="${FORCE_FIELD_MANIFEST:-1}"
FORCE_FUSION="${FORCE_FUSION:-1}"
FORCE_NUCLEATED_BRANCH="${FORCE_NUCLEATED_BRANCH:-1}"
FORCE_SHAPE_STRICT="${FORCE_SHAPE_STRICT:-1}"

for required in "$PROJECT_DIR" "$INPUT_ROOT" "$RUN_ROOT"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -s "$TASK_LIST_FUSION" ]]; then
  echo "Task list is missing or empty: $TASK_LIST_FUSION" >&2
  exit 2
fi
for profile in Combined Brightfield Dead Nuclei; do
  if [[ ! -d "$RUN_ROOT/$profile/segmentations" ]]; then
    echo "Upstream segmentation is incomplete: $RUN_ROOT/$profile/segmentations" >&2
    exit 2
  fi
done
if [[ ! -d "$RUN_ROOT/Nuclei/nucleus_core_seeds" ]]; then
  echo "Upstream nucleus-core masks are missing: $RUN_ROOT/Nuclei/nucleus_core_seeds" >&2
  exit 2
fi

N_TASKS="$(awk 'NF {n++} END {print n+0}' "$TASK_LIST_FUSION")"
if [[ "$N_TASKS" -le 0 ]]; then
  echo "No field keys in $TASK_LIST_FUSION" >&2
  exit 2
fi
if [[ ! "$POSTSEG_ARRAY_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "POSTSEG_ARRAY_MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi
ARRAY_SPEC="1-$N_TASKS%$POSTSEG_ARRAY_MAX_CONCURRENT"
mkdir -p "$LOG_DIR" "$RUN_ROOT/workflow_status"

cpu_sbatch() {
  env \
    -u SBATCH_GRES \
    -u SBATCH_TRES_PER_NODE \
    -u SBATCH_GPUS \
    -u SBATCH_GPUS_PER_NODE \
    -u SBATCH_GPUS_PER_TASK \
    sbatch --parsable "$@"
}

submit_job() {
  local output
  output="$(cpu_sbatch "$@")"
  printf '%s\n' "${output%%;*}"
}

COMMON=(
  --partition "$SBATCH_PARTITION"
  --qos "$SBATCH_QOS"
  --account "$SBATCH_ACCOUNT"
  --time "$SBATCH_TIME"
  --cpus-per-task 1
)

MANIFEST_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_manifest" \
  --mem 4G \
  --output "$LOG_DIR/%x_%j.out" \
  --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",NUCLEATED_BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",FORCE_FIELD_MANIFEST="$FORCE_FIELD_MANIFEST" \
  "$SCRIPT_DIR/run_postsegmentation_manifest.sh")"

FUSION_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_fusion" \
  --array "$ARRAY_SPEC" --mem 4G \
  --dependency "afterok:$MANIFEST_JOB_ID" \
  --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",RUN_NAME=.,OUT_DIR="$FUSION_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original,FORCE_FUSION="$FORCE_FUSION",CONTINUE_ON_ERROR=0 \
  "$SCRIPT_DIR/run_multichannel_classification_fusion_array_task.sh")"

NUCLEATED_BRANCH_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_nucleated" \
  --array "$ARRAY_SPEC" --mem 8G \
  --dependency "afterok:$MANIFEST_JOB_ID" \
  --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",TASK_LIST_BRANCH="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",RENDER_NUCLEATED_QC=0,FORCE_NUCLEATED_BRANCH="$FORCE_NUCLEATED_BRANCH" \
  "$SCRIPT_DIR/run_nucleated_branch_array_task.sh")"

FUSION_MERGE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_fusion_merge" --mem 4G \
  --dependency "afterok:$FUSION_JOB_ID" \
  --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",OUT_DIR="$FUSION_OUT_DIR",FORCE_FUSION=1 \
  "$SCRIPT_DIR/run_multichannel_classification_fusion_merge.sh")"

NUCLEATED_BRANCH_MERGE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_nucleated_merge" --mem 8G \
  --dependency "afterok:$NUCLEATED_BRANCH_JOB_ID" \
  --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",TASK_LIST_BRANCH="$TASK_LIST_FUSION" \
  "$SCRIPT_DIR/run_nucleated_branch_merge.sh")"

NUCLEATED_FUSION_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_fusion_nucleated" \
  --array "$ARRAY_SPEC" --mem 4G \
  --dependency "afterok:$NUCLEATED_BRANCH_MERGE_JOB_ID" \
  --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",RUN_NAME=.,OUT_DIR="$NUCLEATED_FUSION_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated,COMBINED_RUN="$NUCLEATED_BRANCH_ROOT/Combined",BRIGHTFIELD_RUN="$NUCLEATED_BRANCH_ROOT/Brightfield",DEAD_RUN="$RUN_ROOT/Dead",NUCLEI_RUN="$RUN_ROOT/Nuclei",FORCE_FUSION="$FORCE_FUSION",CONTINUE_ON_ERROR=0 \
  "$SCRIPT_DIR/run_multichannel_classification_fusion_array_task.sh")"

NUCLEATED_FUSION_MERGE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_fusion_nuc_merge" --mem 4G \
  --dependency "afterok:$NUCLEATED_FUSION_JOB_ID" \
  --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",OUT_DIR="$NUCLEATED_FUSION_OUT_DIR",FORCE_FUSION=1 \
  "$SCRIPT_DIR/run_multichannel_classification_fusion_merge.sh")"

SHAPE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_shape_strict" \
  --array "$ARRAY_SPEC" --mem 8G \
  --dependency "afterok:$FUSION_MERGE_JOB_ID" \
  --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",SHAPE_ROOT="$SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original,FORCE_SHAPE_STRICT="$FORCE_SHAPE_STRICT" \
  "$SCRIPT_DIR/run_shape_strict_array_task.sh")"

SHAPE_FINALIZE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_shape_finalize" --mem 16G \
  --dependency "afterok:$SHAPE_JOB_ID" \
  --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",SHAPE_ROOT="$SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original \
  "$SCRIPT_DIR/run_shape_strict_finalize.sh")"

NUCLEATED_SHAPE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_shape_nucleated" \
  --array "$ARRAY_SPEC" --mem 8G \
  --dependency "afterok:$NUCLEATED_FUSION_MERGE_JOB_ID" \
  --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",CELL_RUN_ROOT="$NUCLEATED_BRANCH_ROOT",FUSION_ROOT="$NUCLEATED_FUSION_OUT_DIR",SHAPE_ROOT="$NUCLEATED_SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated,FORCE_SHAPE_STRICT="$FORCE_SHAPE_STRICT" \
  "$SCRIPT_DIR/run_shape_strict_array_task.sh")"

NUCLEATED_SHAPE_FINALIZE_JOB_ID="$(submit_job "${COMMON[@]}" \
  --job-name "${SBATCH_JOB_PREFIX}_shape_nuc_finalize" --mem 16G \
  --dependency "afterok:$NUCLEATED_SHAPE_JOB_ID" \
  --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$RUN_ROOT",CELL_RUN_ROOT="$NUCLEATED_BRANCH_ROOT",FUSION_ROOT="$NUCLEATED_FUSION_OUT_DIR",SHAPE_ROOT="$NUCLEATED_SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated \
  "$SCRIPT_DIR/run_shape_strict_finalize.sh")"

echo "postsegmentation_only=1"
echo "run_root=$RUN_ROOT"
echo "task_list_fusion=$TASK_LIST_FUSION"
echo "n_tasks=$N_TASKS"
echo "array_spec=$ARRAY_SPEC"
echo "field_manifest_job_id=$MANIFEST_JOB_ID"
echo "fusion_array_job_id=$FUSION_JOB_ID"
echo "fusion_merge_job_id=$FUSION_MERGE_JOB_ID"
echo "nucleated_branch_array_job_id=$NUCLEATED_BRANCH_JOB_ID"
echo "nucleated_branch_merge_job_id=$NUCLEATED_BRANCH_MERGE_JOB_ID"
echo "nucleated_fusion_array_job_id=$NUCLEATED_FUSION_JOB_ID"
echo "nucleated_fusion_merge_job_id=$NUCLEATED_FUSION_MERGE_JOB_ID"
echo "shape_array_job_id=$SHAPE_JOB_ID"
echo "shape_finalize_job_id=$SHAPE_FINALIZE_JOB_ID"
echo "nucleated_shape_array_job_id=$NUCLEATED_SHAPE_JOB_ID"
echo "nucleated_shape_finalize_job_id=$NUCLEATED_SHAPE_FINALIZE_JOB_ID"
