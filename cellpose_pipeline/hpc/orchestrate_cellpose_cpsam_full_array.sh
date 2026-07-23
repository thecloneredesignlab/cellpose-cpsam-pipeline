#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
INPUT_ROOT="${INPUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages}"
OUT_ROOT="${OUT_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-.}"
if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
  WORKFLOW_RUN_DIR="$OUT_ROOT"
else
  WORKFLOW_RUN_DIR="$OUT_ROOT/$RUN_NAME"
fi
EXPECTED_SUBDIRS="${EXPECTED_SUBDIRS:-4}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-}"
SKIP_SUBDIRS="${SKIP_SUBDIRS:-Dead_Uncalibrated}"
ENABLE_HIGH_DENSITY_PROFILES="${ENABLE_HIGH_DENSITY_PROFILES:-1}"
HIGH_DENSITY_CALLS_CSV="${HIGH_DENSITY_CALLS_CSV:-}"
AUTO_GENERATE_DENSITY_CALLS="${AUTO_GENERATE_DENSITY_CALLS:-1}"
DENSITY_CALLS_NAME="${DENSITY_CALLS_NAME:-density_calls.csv}"
ENABLE_DEAD_CALIBRATION="${ENABLE_DEAD_CALIBRATION:-1}"
ENABLE_DEAD_CONSENSUS="${ENABLE_DEAD_CONSENSUS:-1}"
CALIBRATION_SHARD_COUNT="${CALIBRATION_SHARD_COUNT:-32}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$WORKFLOW_RUN_DIR/dead_preprocess_calibration}"
DEAD_CALIBRATION_JSON="${DEAD_CALIBRATION_JSON:-$CALIBRATION_ROOT/final/dead_combined_blue_calibration.json}"
DEAD_CALIBRATION_MAP="${DEAD_CALIBRATION_MAP:-$CALIBRATION_ROOT/final/dead_combined_blue_image_calibration_map.csv}"
ENABLE_FUSION_CLASSIFICATION="${ENABLE_FUSION_CLASSIFICATION:-1}"
ENABLE_NUCLEATED_BRANCH="${ENABLE_NUCLEATED_BRANCH:-1}"
ENABLE_LATE_DEATH_REFINEMENT="${ENABLE_LATE_DEATH_REFINEMENT:-$ENABLE_NUCLEATED_BRANCH}"
if [[ -z "${FUSION_OUT_DIR:-}" ]]; then
  if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
    FUSION_OUT_DIR="$OUT_ROOT/classification_fusion"
  else
    FUSION_OUT_DIR="$OUT_ROOT/$RUN_NAME/classification_fusion"
  fi
fi
FUSION_KEY_SOURCE_DIR="${FUSION_KEY_SOURCE_DIR:-$INPUT_ROOT/Combined}"
FUSION_KEY_LIST="${FUSION_KEY_LIST:-}"
NUCLEATED_BRANCH_ROOT="${NUCLEATED_BRANCH_ROOT:-$WORKFLOW_RUN_DIR/nucleated_only}"
NUCLEATED_FUSION_OUT_DIR="${NUCLEATED_FUSION_OUT_DIR:-$WORKFLOW_RUN_DIR/classification_fusion_nucleated_only}"
NUCLEATED_SHAPE_ROOT="${NUCLEATED_SHAPE_ROOT:-$WORKFLOW_RUN_DIR/shape_strict_nucleated_only}"
FIELD_MANIFEST_DIR="${FIELD_MANIFEST_DIR:-$WORKFLOW_RUN_DIR/workflow_status/postsegmentation_manifest}"
POSTSEG_ARRAY_MAX_CONCURRENT="${POSTSEG_ARRAY_MAX_CONCURRENT:-256}"
ENABLE_SHAPE_STRICT="${ENABLE_SHAPE_STRICT:-1}"
if [[ -z "${SHAPE_ROOT:-}" ]]; then
  if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
    SHAPE_ROOT="$OUT_ROOT/shape_strict"
  else
    SHAPE_ROOT="$OUT_ROOT/$RUN_NAME/shape_strict"
  fi
fi

SCRIPT_DIR="$PROJECT_DIR/cellpose_pipeline/hpc"
WORKER="$SCRIPT_DIR/run_cellpose_cpsam_array_task.sh"
DENSITY_TABLE_WORKER="$SCRIPT_DIR/run_density_table_from_masks.sh"
DEAD_CALIBRATION_ARRAY_WORKER="$SCRIPT_DIR/run_dead_calibration_array_task.sh"
DEAD_CALIBRATION_MERGE_WORKER="$SCRIPT_DIR/run_dead_calibration_merge.sh"
DEAD_CONSENSUS_MERGE_WORKER="$SCRIPT_DIR/run_dead_consensus_merge.sh"
FUSION_ARRAY_WORKER="$SCRIPT_DIR/run_multichannel_classification_fusion_array_task.sh"
FUSION_MERGE_WORKER="$SCRIPT_DIR/run_multichannel_classification_fusion_merge.sh"
LATE_DEATH_WORKER="$SCRIPT_DIR/run_late_dead_trajectory_refinement.sh"
SHAPE_ARRAY_WORKER="$SCRIPT_DIR/run_shape_strict_array_task.sh"
SHAPE_FINALIZE_WORKER="$SCRIPT_DIR/run_shape_strict_finalize.sh"
NUCLEATED_BRANCH_ARRAY_WORKER="$SCRIPT_DIR/run_nucleated_branch_array_task.sh"
NUCLEATED_BRANCH_MERGE_WORKER="$SCRIPT_DIR/run_nucleated_branch_merge.sh"
FIELD_MANIFEST_WORKER="$SCRIPT_DIR/run_postsegmentation_manifest.sh"
TASK_DIR="$SCRIPT_DIR/task_lists"
LOG_DIR="$SCRIPT_DIR/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
TASK_LIST_ALL="$TASK_DIR/cpsam_separate_${RUN_STAMP}_all.tsv"
TASK_LIST_NUCLEI="$TASK_DIR/cpsam_separate_${RUN_STAMP}_nuclei.tsv"
TASK_LIST_MAIN="$TASK_DIR/cpsam_separate_${RUN_STAMP}_main.tsv"
TASK_LIST_FUSION="$TASK_DIR/cpsam_separate_${RUN_STAMP}_fusion_keys.tsv"
DUPLICATE_REPORT="$TASK_DIR/cpsam_separate_${RUN_STAMP}_duplicate_stems.tsv"
SUBDIR_COUNT_REPORT="$TASK_DIR/cpsam_separate_${RUN_STAMP}_subdir_counts.tsv"

mkdir -p "$TASK_DIR" "$LOG_DIR" "$OUT_ROOT"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory does not exist: $PROJECT_DIR" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input root directory does not exist: $INPUT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$WORKER" ]]; then
  echo "Worker script is missing or not executable: $WORKER" >&2
  exit 2
fi
if [[ ! -x "$DENSITY_TABLE_WORKER" ]]; then
  echo "Density table script is missing or not executable: $DENSITY_TABLE_WORKER" >&2
  exit 2
fi
if [[ "$ENABLE_DEAD_CALIBRATION" != "0" ]]; then
  if [[ ! -x "$DEAD_CALIBRATION_ARRAY_WORKER" || ! -x "$DEAD_CALIBRATION_MERGE_WORKER" ]]; then
    echo "Dead calibration workers are missing or not executable." >&2
    exit 2
  fi
fi
if [[ "$ENABLE_DEAD_CONSENSUS" != "0" && ! -x "$DEAD_CONSENSUS_MERGE_WORKER" ]]; then
  echo "Dead consensus merge worker is missing or not executable: $DEAD_CONSENSUS_MERGE_WORKER" >&2
  exit 2
fi
if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
  if [[ ! -x "$NUCLEATED_BRANCH_ARRAY_WORKER" || ! -x "$NUCLEATED_BRANCH_MERGE_WORKER" ]]; then
    echo "Nucleated-only branch workers are missing or not executable." >&2
    exit 2
  fi
  if [[ "$ENABLE_FUSION_CLASSIFICATION" == "0" ]]; then
    echo "Nucleated-only branch requires fusion classification." >&2
    exit 2
  fi
fi
if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" && ! -x "$FIELD_MANIFEST_WORKER" ]]; then
  echo "Post-segmentation manifest worker is missing or not executable: $FIELD_MANIFEST_WORKER" >&2
  exit 2
fi
if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" ]]; then
  if [[ ! -x "$FUSION_ARRAY_WORKER" ]]; then
    echo "Fusion array worker script is missing or not executable: $FUSION_ARRAY_WORKER" >&2
    exit 2
  fi
  if [[ ! -x "$FUSION_MERGE_WORKER" ]]; then
    echo "Fusion merge worker script is missing or not executable: $FUSION_MERGE_WORKER" >&2
    exit 2
  fi
fi
if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
  if [[ "$ENABLE_FUSION_CLASSIFICATION" == "0" || "$ENABLE_NUCLEATED_BRANCH" == "0" ]]; then
    echo "Late-death refinement requires both original and nucleated-only fusion classifications." >&2
    exit 2
  fi
  if [[ ! -x "$LATE_DEATH_WORKER" ]]; then
    echo "Late-death refinement worker is missing or not executable: $LATE_DEATH_WORKER" >&2
    exit 2
  fi
fi
if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
  if [[ "$ENABLE_FUSION_CLASSIFICATION" == "0" ]]; then
    echo "shape_strict requires fusion classification and its merged results." >&2
    exit 2
  fi
  if [[ "$RUN_NAME" != "." && -n "$RUN_NAME" ]]; then
    echo "shape_strict production integration currently requires RUN_NAME=." >&2
    exit 2
  fi
  if [[ ! -x "$SHAPE_ARRAY_WORKER" ]]; then
    echo "shape_strict array worker is missing or not executable: $SHAPE_ARRAY_WORKER" >&2
    exit 2
  fi
  if [[ ! -x "$SHAPE_FINALIZE_WORKER" ]]; then
    echo "shape_strict finalize worker is missing or not executable: $SHAPE_FINALIZE_WORKER" >&2
    exit 2
  fi
fi

AUTO_DENSITY_PIPELINE=0
if [[ "$ENABLE_HIGH_DENSITY_PROFILES" != "0" && -z "$HIGH_DENSITY_CALLS_CSV" && "$AUTO_GENERATE_DENSITY_CALLS" != "0" ]]; then
  AUTO_DENSITY_PIPELINE=1
fi

mapfile -t ALL_INPUT_SUBDIRS < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
INPUT_SUBDIRS=()
for input_subdir in "${ALL_INPUT_SUBDIRS[@]}"; do
  group_name="$(basename "$input_subdir")"
  skip=false
  for skip_name in $SKIP_SUBDIRS; do
    if [[ "$group_name" == "$skip_name" ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == true ]]; then
    continue
  fi
  INPUT_SUBDIRS+=("$input_subdir")
done

N_SUBDIRS="${#INPUT_SUBDIRS[@]}"
if [[ "$N_SUBDIRS" -eq 0 ]]; then
  echo "No processable child directories found in $INPUT_ROOT" >&2
  exit 2
fi
if [[ -n "$EXPECTED_SUBDIRS" && "$N_SUBDIRS" -ne "$EXPECTED_SUBDIRS" ]]; then
  echo "Expected $EXPECTED_SUBDIRS processable child directories but found $N_SUBDIRS in $INPUT_ROOT" >&2
  echo "Skipped child directory names: $SKIP_SUBDIRS" >&2
  printf '%s\n' "${INPUT_SUBDIRS[@]}" >&2
  exit 2
fi

: > "$TASK_LIST_ALL"
: > "$TASK_LIST_NUCLEI"
: > "$TASK_LIST_MAIN"
: > "$TASK_LIST_FUSION"
: > "$SUBDIR_COUNT_REPORT"
N_DEAD_TASKS=0
N_COMBINED_TASKS=0
for input_subdir in "${INPUT_SUBDIRS[@]}"; do
  group_name="$(basename "$input_subdir")"
  if [[ -z "$group_name" || "$group_name" == "." || "$group_name" == ".." || "$group_name" == *"/"* ]]; then
    echo "Invalid child directory name for output grouping: $group_name" >&2
    exit 2
  fi
  mkdir -p "$OUT_ROOT/$group_name"
  GROUP_PATHS="$TASK_DIR/cpsam_separate_${RUN_STAMP}_${group_name}_paths.tmp"
  find "$input_subdir" -maxdepth 1 -type f \( \
      -iname "*.tif" -o -iname "*.tiff" -o -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \
    \) | sort > "$GROUP_PATHS"
  subdir_count="$(wc -l < "$GROUP_PATHS" | tr -d '[:space:]')"
  if [[ "$group_name" == "Dead" ]]; then
    N_DEAD_TASKS="$subdir_count"
  elif [[ "$group_name" == "Combined" ]]; then
    N_COMBINED_TASKS="$subdir_count"
  fi
  printf '%s\t%s\t%s\n' "$group_name" "$input_subdir" "$subdir_count" >> "$SUBDIR_COUNT_REPORT"
  if [[ "$subdir_count" -eq 0 ]]; then
    echo "No supported images found in child directory: $input_subdir" >&2
    echo "Subdirectory count report retained at: $SUBDIR_COUNT_REPORT" >&2
    rm -f "$GROUP_PATHS"
    exit 2
  fi
  awk -v group="$group_name" 'BEGIN { OFS = "\t" } { print $0, group }' "$GROUP_PATHS" >> "$TASK_LIST_ALL"
  if [[ "$AUTO_DENSITY_PIPELINE" == "1" && "$group_name" == "Nuclei" ]]; then
    awk -v group="$group_name" 'BEGIN { OFS = "\t" } { print $0, group }' "$GROUP_PATHS" >> "$TASK_LIST_NUCLEI"
  else
    awk -v group="$group_name" 'BEGIN { OFS = "\t" } { print $0, group }' "$GROUP_PATHS" >> "$TASK_LIST_MAIN"
  fi
  rm -f "$GROUP_PATHS"
done

if [[ "$ENABLE_DEAD_CONSENSUS" != "0" || "$ENABLE_DEAD_CALIBRATION" != "0" ]]; then
  if [[ "$N_DEAD_TASKS" -eq 0 || "$N_DEAD_TASKS" -ne "$N_COMBINED_TASKS" ]]; then
    echo "Dead/Combined paired processing requires equal nonzero counts; Dead=$N_DEAD_TASKS Combined=$N_COMBINED_TASKS" >&2
    exit 2
  fi
fi

N_TOTAL_TASKS="$(wc -l < "$TASK_LIST_ALL" | tr -d '[:space:]')"
N_NUCLEI_TASKS="$(wc -l < "$TASK_LIST_NUCLEI" | tr -d '[:space:]')"
N_MAIN_TASKS="$(wc -l < "$TASK_LIST_MAIN" | tr -d '[:space:]')"
if [[ "$N_TOTAL_TASKS" -eq 0 ]]; then
  echo "No supported images found under child directories of $INPUT_ROOT" >&2
  exit 2
fi
if [[ "$AUTO_DENSITY_PIPELINE" == "1" && "$N_NUCLEI_TASKS" -eq 0 ]]; then
  echo "Auto density calls require Nuclei images, but no Nuclei tasks were found." >&2
  echo "All task list retained at: $TASK_LIST_ALL" >&2
  exit 2
fi
if [[ "$N_MAIN_TASKS" -eq 0 ]]; then
  echo "No main segmentation tasks found after task-list construction." >&2
  echo "All task list retained at: $TASK_LIST_ALL" >&2
  exit 2
fi
if [[ -n "$EXPECTED_IMAGES" && "$N_TOTAL_TASKS" -ne "$EXPECTED_IMAGES" ]]; then
  echo "Expected $EXPECTED_IMAGES images but found $N_TOTAL_TASKS under $INPUT_ROOT" >&2
  echo "All task list retained at: $TASK_LIST_ALL" >&2
  exit 2
fi

awk '
  BEGIN { FS = "\t" }
  {
    path = $1
    group = $2
    name = path
    sub(/^.*\//, "", name)
    sub(/\.[^.]*$/, "", name)
    key = group "\t" name
    count[key]++
  }
  END {
    for (key in count) {
      if (count[key] > 1) {
        print key "\t" count[key]
        dup = 1
      }
    }
    exit dup ? 1 : 0
  }
' "$TASK_LIST_ALL" > "$DUPLICATE_REPORT" || {
  echo "Duplicate image stems detected. See: $DUPLICATE_REPORT" >&2
  exit 2
}
rm -f "$DUPLICATE_REPORT"

N_FUSION_TASKS=0
if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" ]]; then
  if [[ -n "$FUSION_KEY_LIST" ]]; then
    if [[ ! -f "$FUSION_KEY_LIST" ]]; then
      echo "Fusion key list does not exist: $FUSION_KEY_LIST" >&2
      exit 2
    fi
    awk 'NF {print $1}' "$FUSION_KEY_LIST" | sort -u > "$TASK_LIST_FUSION"
  else
    if [[ ! -d "$FUSION_KEY_SOURCE_DIR" ]]; then
      echo "Fusion classification requires Combined keys, but key source dir does not exist: $FUSION_KEY_SOURCE_DIR" >&2
      echo "Set ENABLE_FUSION_CLASSIFICATION=0 to skip fusion or FUSION_KEY_LIST=/path/to/keys.txt to override." >&2
      exit 2
    fi
    FUSION_UNSORTED="$TASK_LIST_FUSION.unsorted"
    find "$FUSION_KEY_SOURCE_DIR" -maxdepth 1 -type f \( \
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
            print "Could not extract fusion key from Combined image: " $0 > "/dev/stderr"
            bad = 1
          }
        }
        END { exit bad ? 1 : 0 }
      ' > "$FUSION_UNSORTED"
    sort -u "$FUSION_UNSORTED" > "$TASK_LIST_FUSION"
    rm -f "$FUSION_UNSORTED"
  fi
  N_FUSION_TASKS="$(wc -l < "$TASK_LIST_FUSION" | tr -d '[:space:]')"
  if [[ "$N_FUSION_TASKS" -eq 0 ]]; then
    echo "Fusion classification is enabled, but no keys were written to $TASK_LIST_FUSION" >&2
    exit 2
  fi
fi
N_SHAPE_TASKS=0
if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
  N_SHAPE_TASKS="$N_FUSION_TASKS"
fi
POSTSEG_ARRAY_SPEC="1-$N_FUSION_TASKS"
if [[ "$POSTSEG_ARRAY_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  POSTSEG_ARRAY_SPEC+="%$POSTSEG_ARRAY_MAX_CONCURRENT"
elif [[ "$POSTSEG_ARRAY_MAX_CONCURRENT" != "0" ]]; then
  echo "POSTSEG_ARRAY_MAX_CONCURRENT must be 0 or a positive integer: $POSTSEG_ARRAY_MAX_CONCURRENT" >&2
  exit 2
fi

SBATCH_ARGS=(
  --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}"
  --array "1-$N_MAIN_TASKS"
  --output "$LOG_DIR/%x_%A_%a.out"
  --error "$LOG_DIR/%x_%A_%a.err"
  --time "${SBATCH_TIME:-12:00:00}"
  --cpus-per-task "${SBATCH_CPUS:-2}"
  --mem "${SBATCH_MEM:-8G}"
)

if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
fi
SBATCH_QOS_VALUE="${SBATCH_QOS:-xxlarge}"
if [[ -n "$SBATCH_QOS_VALUE" ]]; then
  SBATCH_ARGS+=(--qos "$SBATCH_QOS_VALUE")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi
SBATCH_GRES_VALUE="${SBATCH_GRES:-gpu:a30:1}"
# SBATCH_* environment variables are interpreted by every later sbatch call.
# Keep the captured GPU request only in the explicit GPU argument arrays so
# density, fusion, merge, and shape_strict cannot inherit a GPU implicitly.
unset SBATCH_GRES
unset SBATCH_TRES_PER_NODE
if [[ -n "$SBATCH_GRES_VALUE" ]]; then
  SBATCH_ARGS+=(--gres "$SBATCH_GRES_VALUE")
fi
CPU_ONLY_SBATCH_ENV=(
  env
  -u SBATCH_GRES
  -u SBATCH_TRES_PER_NODE
  -u SBATCH_GPUS
  -u SBATCH_GPUS_PER_NODE
  -u SBATCH_GPUS_PER_TASK
)
FUSION_SBATCH_QOS_VALUE="${FUSION_SBATCH_QOS:-$SBATCH_QOS_VALUE}"
FUSION_MERGE_SBATCH_QOS_VALUE="${FUSION_MERGE_SBATCH_QOS:-$FUSION_SBATCH_QOS_VALUE}"
LATE_DEATH_SBATCH_QOS_VALUE="${LATE_DEATH_SBATCH_QOS:-$FUSION_MERGE_SBATCH_QOS_VALUE}"
DENSITY_TABLE_SBATCH_QOS_VALUE="${DENSITY_TABLE_SBATCH_QOS:-$SBATCH_QOS_VALUE}"
SHAPE_SBATCH_QOS_VALUE="${SHAPE_SBATCH_QOS:-$FUSION_SBATCH_QOS_VALUE}"
SHAPE_FINALIZE_SBATCH_QOS_VALUE="${SHAPE_FINALIZE_SBATCH_QOS:-$SHAPE_SBATCH_QOS_VALUE}"
CALIBRATION_SBATCH_QOS_VALUE="${CALIBRATION_SBATCH_QOS:-xxlarge}"
CALIBRATION_MERGE_SBATCH_QOS_VALUE="${CALIBRATION_MERGE_SBATCH_QOS:-xxlarge}"
DEAD_CONSENSUS_MERGE_SBATCH_QOS_VALUE="${DEAD_CONSENSUS_MERGE_SBATCH_QOS:-xxlarge}"
NUCLEATED_BRANCH_SBATCH_QOS_VALUE="${NUCLEATED_BRANCH_SBATCH_QOS:-xxlarge}"
NUCLEATED_BRANCH_MERGE_SBATCH_QOS_VALUE="${NUCLEATED_BRANCH_MERGE_SBATCH_QOS:-xxlarge}"
FIELD_MANIFEST_SBATCH_QOS_VALUE="${FIELD_MANIFEST_SBATCH_QOS:-xxlarge}"

FUSION_SBATCH_ARGS=()
FUSION_MERGE_SBATCH_ARGS=()
LATE_DEATH_SBATCH_ARGS=()
SHAPE_SBATCH_ARGS=()
SHAPE_FINALIZE_SBATCH_ARGS=()
CALIBRATION_SBATCH_ARGS=()
CALIBRATION_MERGE_SBATCH_ARGS=()
DEAD_CONSENSUS_MERGE_SBATCH_ARGS=()
NUCLEATED_BRANCH_SBATCH_ARGS=()
NUCLEATED_BRANCH_MERGE_SBATCH_ARGS=()
FIELD_MANIFEST_SBATCH_ARGS=()
if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" ]]; then
  FUSION_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_fusion"
    --array "$POSTSEG_ARRAY_SPEC"
    --output "$LOG_DIR/%x_%A_%a.out"
    --error "$LOG_DIR/%x_%A_%a.err"
    --time "${FUSION_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${FUSION_SBATCH_CPUS:-1}"
    --mem "${FUSION_SBATCH_MEM:-4G}"
  )
  FIELD_MANIFEST_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_postseg_manifest"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${FIELD_MANIFEST_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${FIELD_MANIFEST_SBATCH_CPUS:-1}"
    --mem "${FIELD_MANIFEST_SBATCH_MEM:-4G}"
  )
  FUSION_MERGE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_fusion_merge"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${FUSION_MERGE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${FUSION_MERGE_SBATCH_CPUS:-1}"
    --mem "${FUSION_MERGE_SBATCH_MEM:-4G}"
  )
  if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
    LATE_DEATH_SBATCH_ARGS=(
      --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_late_death"
      --output "$LOG_DIR/%x_%j.out"
      --error "$LOG_DIR/%x_%j.err"
      --time "${LATE_DEATH_SBATCH_TIME:-12:00:00}"
      --cpus-per-task "${LATE_DEATH_SBATCH_CPUS:-32}"
      --mem "${LATE_DEATH_SBATCH_MEM:-256G}"
    )
  fi
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    FUSION_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    FUSION_MERGE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    FIELD_MANIFEST_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
      LATE_DEATH_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    fi
  fi
  if [[ -n "$FUSION_SBATCH_QOS_VALUE" ]]; then
    FUSION_SBATCH_ARGS+=(--qos "$FUSION_SBATCH_QOS_VALUE")
  fi
  if [[ -n "$FUSION_MERGE_SBATCH_QOS_VALUE" ]]; then
    FUSION_MERGE_SBATCH_ARGS+=(--qos "$FUSION_MERGE_SBATCH_QOS_VALUE")
  fi
  if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" && -n "$LATE_DEATH_SBATCH_QOS_VALUE" ]]; then
    LATE_DEATH_SBATCH_ARGS+=(--qos "$LATE_DEATH_SBATCH_QOS_VALUE")
  fi
  if [[ -n "$FIELD_MANIFEST_SBATCH_QOS_VALUE" ]]; then
    FIELD_MANIFEST_SBATCH_ARGS+=(--qos "$FIELD_MANIFEST_SBATCH_QOS_VALUE")
  fi
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    FUSION_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    FUSION_MERGE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    FIELD_MANIFEST_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
      LATE_DEATH_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    fi
  fi
fi
if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
  SHAPE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_shape_strict"
    --array "$POSTSEG_ARRAY_SPEC"
    --output "$LOG_DIR/%x_%A_%a.out"
    --error "$LOG_DIR/%x_%A_%a.err"
    --time "${SHAPE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${SHAPE_SBATCH_CPUS:-1}"
    --mem "${SHAPE_SBATCH_MEM:-8G}"
  )
  SHAPE_FINALIZE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_shape_finalize"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${SHAPE_FINALIZE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${SHAPE_FINALIZE_SBATCH_CPUS:-1}"
    --mem "${SHAPE_FINALIZE_SBATCH_MEM:-16G}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    SHAPE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    SHAPE_FINALIZE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  if [[ -n "$SHAPE_SBATCH_QOS_VALUE" ]]; then
    SHAPE_SBATCH_ARGS+=(--qos "$SHAPE_SBATCH_QOS_VALUE")
  fi
  if [[ -n "$SHAPE_FINALIZE_SBATCH_QOS_VALUE" ]]; then
    SHAPE_FINALIZE_SBATCH_ARGS+=(--qos "$SHAPE_FINALIZE_SBATCH_QOS_VALUE")
  fi
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    SHAPE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    SHAPE_FINALIZE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
fi

if [[ "$ENABLE_DEAD_CALIBRATION" != "0" ]]; then
  CALIBRATION_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_dead_cal"
    --array "1-$CALIBRATION_SHARD_COUNT"
    --output "$LOG_DIR/%x_%A_%a.out"
    --error "$LOG_DIR/%x_%A_%a.err"
    --time "${CALIBRATION_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${CALIBRATION_SBATCH_CPUS:-2}"
    --mem "${CALIBRATION_SBATCH_MEM:-16G}"
  )
  CALIBRATION_MERGE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_dead_cal_merge"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${CALIBRATION_MERGE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${CALIBRATION_MERGE_SBATCH_CPUS:-2}"
    --mem "${CALIBRATION_MERGE_SBATCH_MEM:-32G}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    CALIBRATION_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    CALIBRATION_MERGE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  [[ -n "$CALIBRATION_SBATCH_QOS_VALUE" ]] && CALIBRATION_SBATCH_ARGS+=(--qos "$CALIBRATION_SBATCH_QOS_VALUE")
  [[ -n "$CALIBRATION_MERGE_SBATCH_QOS_VALUE" ]] && CALIBRATION_MERGE_SBATCH_ARGS+=(--qos "$CALIBRATION_MERGE_SBATCH_QOS_VALUE")
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    CALIBRATION_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    CALIBRATION_MERGE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
fi

if [[ "$ENABLE_DEAD_CONSENSUS" != "0" ]]; then
  DEAD_CONSENSUS_MERGE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_dead_merge"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${DEAD_CONSENSUS_MERGE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${DEAD_CONSENSUS_MERGE_SBATCH_CPUS:-1}"
    --mem "${DEAD_CONSENSUS_MERGE_SBATCH_MEM:-8G}"
  )
  [[ -n "${SBATCH_PARTITION:-}" ]] && DEAD_CONSENSUS_MERGE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  [[ -n "$DEAD_CONSENSUS_MERGE_SBATCH_QOS_VALUE" ]] && DEAD_CONSENSUS_MERGE_SBATCH_ARGS+=(--qos "$DEAD_CONSENSUS_MERGE_SBATCH_QOS_VALUE")
  [[ -n "${SBATCH_ACCOUNT:-}" ]] && DEAD_CONSENSUS_MERGE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi

if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
  NUCLEATED_BRANCH_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_nucleated"
    --array "$POSTSEG_ARRAY_SPEC"
    --output "$LOG_DIR/%x_%A_%a.out"
    --error "$LOG_DIR/%x_%A_%a.err"
    --time "${NUCLEATED_BRANCH_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${NUCLEATED_BRANCH_SBATCH_CPUS:-1}"
    --mem "${NUCLEATED_BRANCH_SBATCH_MEM:-8G}"
  )
  NUCLEATED_BRANCH_MERGE_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_nucleated_merge"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${NUCLEATED_BRANCH_MERGE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${NUCLEATED_BRANCH_MERGE_SBATCH_CPUS:-1}"
    --mem "${NUCLEATED_BRANCH_MERGE_SBATCH_MEM:-8G}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    NUCLEATED_BRANCH_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
    NUCLEATED_BRANCH_MERGE_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  [[ -n "$NUCLEATED_BRANCH_SBATCH_QOS_VALUE" ]] && NUCLEATED_BRANCH_SBATCH_ARGS+=(--qos "$NUCLEATED_BRANCH_SBATCH_QOS_VALUE")
  [[ -n "$NUCLEATED_BRANCH_MERGE_SBATCH_QOS_VALUE" ]] && NUCLEATED_BRANCH_MERGE_SBATCH_ARGS+=(--qos "$NUCLEATED_BRANCH_MERGE_SBATCH_QOS_VALUE")
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    NUCLEATED_BRANCH_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
    NUCLEATED_BRANCH_MERGE_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
fi

if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
  WORKFLOW_RUN_DIR="$OUT_ROOT"
else
  WORKFLOW_RUN_DIR="$OUT_ROOT/$RUN_NAME"
fi
AUTO_DENSITY_CALLS_CSV="$WORKFLOW_RUN_DIR/qc/$DENSITY_CALLS_NAME"
NUCLEI_JOB_ID=""
DENSITY_TABLE_JOB_ID=""
CALIBRATION_JOB_ID=""
CALIBRATION_MERGE_JOB_ID=""
MAIN_DEP_IDS=()

if [[ "$AUTO_DENSITY_PIPELINE" == "1" ]]; then
  HIGH_DENSITY_CALLS_CSV="$AUTO_DENSITY_CALLS_CSV"
  NUCLEI_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_nuclei"
    --array "1-$N_NUCLEI_TASKS"
    --output "$LOG_DIR/%x_%A_%a.out"
    --error "$LOG_DIR/%x_%A_%a.err"
    --time "${NUCLEI_SBATCH_TIME:-${SBATCH_TIME:-12:00:00}}"
    --cpus-per-task "${NUCLEI_SBATCH_CPUS:-${SBATCH_CPUS:-2}}"
    --mem "${NUCLEI_SBATCH_MEM:-${SBATCH_MEM:-8G}}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    NUCLEI_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  if [[ -n "$SBATCH_QOS_VALUE" ]]; then
    NUCLEI_SBATCH_ARGS+=(--qos "$SBATCH_QOS_VALUE")
  fi
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    NUCLEI_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
  if [[ -n "$SBATCH_GRES_VALUE" ]]; then
    NUCLEI_SBATCH_ARGS+=(--gres "$SBATCH_GRES_VALUE")
  fi
  DENSITY_SBATCH_ARGS=(
    --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_density_table"
    --output "$LOG_DIR/%x_%j.out"
    --error "$LOG_DIR/%x_%j.err"
    --time "${DENSITY_TABLE_SBATCH_TIME:-12:00:00}"
    --cpus-per-task "${DENSITY_TABLE_SBATCH_CPUS:-1}"
    --mem "${DENSITY_TABLE_SBATCH_MEM:-4G}"
  )
  if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    DENSITY_SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
  fi
  if [[ -n "$DENSITY_TABLE_SBATCH_QOS_VALUE" ]]; then
    DENSITY_SBATCH_ARGS+=(--qos "$DENSITY_TABLE_SBATCH_QOS_VALUE")
  fi
  if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    DENSITY_SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
  fi
fi

if [[ "${DRY_RUN_SUBMIT:-0}" == "1" ]]; then
  echo "dry_run_submit=1"
  echo "task_list_all=$TASK_LIST_ALL"
  echo "task_list_nuclei=$TASK_LIST_NUCLEI"
  echo "task_list_main=$TASK_LIST_MAIN"
  echo "task_list_fusion=$TASK_LIST_FUSION"
  echo "subdir_count_report=$SUBDIR_COUNT_REPORT"
  echo "input_root=$INPUT_ROOT"
  echo "skipped_subdirs=$SKIP_SUBDIRS"
  echo "n_subdirs=$N_SUBDIRS"
  printf 'subdir=%s\n' "${INPUT_SUBDIRS[@]}"
  echo "auto_density_pipeline=$AUTO_DENSITY_PIPELINE"
  echo "n_total_tasks=$N_TOTAL_TASKS"
  echo "n_nuclei_tasks=$N_NUCLEI_TASKS"
  echo "n_main_tasks=$N_MAIN_TASKS"
  echo "n_fusion_tasks=$N_FUSION_TASKS"
  echo "n_shape_tasks=$N_SHAPE_TASKS"
  echo "out_root=$OUT_ROOT"
  echo "run_name=$RUN_NAME"
  echo "enable_high_density_profiles=$ENABLE_HIGH_DENSITY_PROFILES"
  echo "auto_generate_density_calls=$AUTO_GENERATE_DENSITY_CALLS"
  echo "density_calls_name=$DENSITY_CALLS_NAME"
  echo "high_density_calls_csv=${HIGH_DENSITY_CALLS_CSV:-unset}"
  echo "density_table_worker=$DENSITY_TABLE_WORKER"
  echo "enable_dead_calibration=$ENABLE_DEAD_CALIBRATION"
  echo "enable_dead_consensus=$ENABLE_DEAD_CONSENSUS"
  echo "calibration_root=$CALIBRATION_ROOT"
  echo "calibration_shard_count=$CALIBRATION_SHARD_COUNT"
  echo "dead_calibration_json=$DEAD_CALIBRATION_JSON"
  echo "dead_calibration_map=$DEAD_CALIBRATION_MAP"
  echo "n_dead_tasks=$N_DEAD_TASKS"
  echo "enable_nucleated_branch=$ENABLE_NUCLEATED_BRANCH"
  echo "nucleated_branch_root=$NUCLEATED_BRANCH_ROOT"
  echo "nucleated_fusion_out_dir=$NUCLEATED_FUSION_OUT_DIR"
  echo "nucleated_shape_root=$NUCLEATED_SHAPE_ROOT"
  echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
  echo "postseg_array_max_concurrent=$POSTSEG_ARRAY_MAX_CONCURRENT"
  echo "enable_fusion_classification=$ENABLE_FUSION_CLASSIFICATION"
  echo "enable_late_death_refinement=$ENABLE_LATE_DEATH_REFINEMENT"
  echo "fusion_key_source_dir=$FUSION_KEY_SOURCE_DIR"
  echo "fusion_key_list=${FUSION_KEY_LIST:-generated}"
  echo "fusion_out_dir=$FUSION_OUT_DIR"
  echo "fusion_sbatch_qos=$FUSION_SBATCH_QOS_VALUE"
  echo "fusion_merge_sbatch_qos=$FUSION_MERGE_SBATCH_QOS_VALUE"
  echo "late_death_sbatch_qos=$LATE_DEATH_SBATCH_QOS_VALUE"
  echo "fusion_array_worker=$FUSION_ARRAY_WORKER"
  echo "fusion_merge_worker=$FUSION_MERGE_WORKER"
  echo "late_death_worker=$LATE_DEATH_WORKER"
  echo "enable_shape_strict=$ENABLE_SHAPE_STRICT"
  echo "shape_root=$SHAPE_ROOT"
  echo "shape_sbatch_qos=$SHAPE_SBATCH_QOS_VALUE"
  echo "shape_finalize_sbatch_qos=$SHAPE_FINALIZE_SBATCH_QOS_VALUE"
  echo "shape_array_worker=$SHAPE_ARRAY_WORKER"
  echo "shape_finalize_worker=$SHAPE_FINALIZE_WORKER"
  echo "nuclei_array_job_id=${NUCLEI_JOB_ID:-planned}"
  echo "density_table_job_id=${DENSITY_TABLE_JOB_ID:-planned}"
  echo "calibration_array_job_id=${CALIBRATION_JOB_ID:-planned}"
  echo "calibration_merge_job_id=${CALIBRATION_MERGE_JOB_ID:-planned}"
  echo "worker=$WORKER"
  if [[ "$AUTO_DENSITY_PIPELINE" == "1" ]]; then
    printf 'nuclei_sbatch_arg=%s\n' "${NUCLEI_SBATCH_ARGS[@]}"
    printf 'density_sbatch_arg=%s\n' "${DENSITY_SBATCH_ARGS[@]}"
    echo "main_dependency_density=afterok:<density_table_job_id>"
  fi
  if [[ "$ENABLE_DEAD_CALIBRATION" != "0" ]]; then
    printf 'calibration_sbatch_arg=%s\n' "${CALIBRATION_SBATCH_ARGS[@]}"
    printf 'calibration_merge_sbatch_arg=%s\n' "${CALIBRATION_MERGE_SBATCH_ARGS[@]}"
    echo "main_dependency_calibration=afterok:<calibration_merge_job_id>"
  fi
  if [[ "$ENABLE_DEAD_CONSENSUS" != "0" ]]; then
    printf 'dead_consensus_merge_sbatch_arg=%s\n' "${DEAD_CONSENSUS_MERGE_SBATCH_ARGS[@]}"
  fi
  if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
    printf 'nucleated_branch_sbatch_arg=%s\n' "${NUCLEATED_BRANCH_SBATCH_ARGS[@]}"
    printf 'nucleated_branch_merge_sbatch_arg=%s\n' "${NUCLEATED_BRANCH_MERGE_SBATCH_ARGS[@]}"
  fi
  printf 'sbatch_arg=%s\n' "${SBATCH_ARGS[@]}"
  if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" ]]; then
    if [[ "$ENABLE_DEAD_CONSENSUS" != "0" ]]; then
      echo "dead_consensus_merge_dependency=afterok:<main_array_job_id>"
      echo "field_manifest_dependency=afterok:<dead_consensus_merge_job_id>"
    else
      echo "field_manifest_dependency=afterok:<main_array_job_id>"
    fi
    echo "fusion_dependency=afterok:<field_manifest_job_id>"
    echo "fusion_merge_dependency=afterok:<fusion_array_job_id>"
    printf 'fusion_sbatch_arg=%s\n' "${FUSION_SBATCH_ARGS[@]}"
    printf 'fusion_merge_sbatch_arg=%s\n' "${FUSION_MERGE_SBATCH_ARGS[@]}"
    printf 'field_manifest_sbatch_arg=%s\n' "${FIELD_MANIFEST_SBATCH_ARGS[@]}"
    if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
      echo "late_death_dependency=afterok:<fusion_merge_job_id>:<nucleated_fusion_merge_job_id>"
      printf 'late_death_sbatch_arg=%s\n' "${LATE_DEATH_SBATCH_ARGS[@]}"
    fi
  fi
  if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
    echo "nucleated_branch_dependency=afterok:<field_manifest_job_id>"
    echo "nucleated_branch_merge_dependency=afterok:<nucleated_branch_array_job_id>"
    echo "nucleated_fusion_dependency=afterok:<nucleated_branch_merge_job_id>"
    echo "nucleated_fusion_merge_dependency=afterok:<nucleated_fusion_array_job_id>"
  fi
  if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
    if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
      echo "shape_dependency=afterok:<late_death_refinement_job_id>"
    else
      echo "shape_dependency=afterok:<fusion_merge_job_id>"
    fi
    echo "shape_finalize_dependency=afterok:<shape_array_job_id>"
    if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
      if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
        echo "nucleated_shape_dependency=afterok:<late_death_refinement_job_id>"
      else
        echo "nucleated_shape_dependency=afterok:<nucleated_fusion_merge_job_id>"
      fi
      echo "nucleated_shape_finalize_dependency=afterok:<nucleated_shape_array_job_id>"
    fi
    printf 'shape_sbatch_arg=%s\n' "${SHAPE_SBATCH_ARGS[@]}"
    printf 'shape_finalize_sbatch_arg=%s\n' "${SHAPE_FINALIZE_SBATCH_ARGS[@]}"
  fi
  exit 0
fi

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellpose_cpsam
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

"$PYTHON_BIN" -I - <<'PY'
from importlib import metadata
from pathlib import Path
from cellpose import models

version = metadata.version("cellpose")
print("cellpose_version", version)
if version != "4.2.1.1":
    raise SystemExit(f"Expected cellpose==4.2.1.1, found {version}")

model_names = list(getattr(models, "MODEL_NAMES", []))
print("cellpose_model_names", model_names)
for model_name in ("cpsam", "cpsam_v2"):
    if model_name not in model_names:
        raise SystemExit(f"{model_name} is not available in cellpose.models.MODEL_NAMES")
    model_path = Path.home() / ".cellpose" / "models" / model_name
    print(f"{model_name}_cache_path", model_path)
    print(f"{model_name}_cache_exists", model_path.exists())
    if model_path.exists():
        print(f"{model_name}_cache_size", model_path.stat().st_size)
PY

cd "$PROJECT_DIR"
PREFLIGHT_HIGH_DENSITY_ARGS=()
if [[ "$ENABLE_HIGH_DENSITY_PROFILES" == "0" ]]; then
  PREFLIGHT_HIGH_DENSITY_ARGS+=(--no-enable-high-density-profiles)
else
  PREFLIGHT_HIGH_DENSITY_ARGS+=(--enable-high-density-profiles)
  if [[ -n "$HIGH_DENSITY_CALLS_CSV" && "$AUTO_DENSITY_PIPELINE" != "1" ]]; then
    PREFLIGHT_HIGH_DENSITY_ARGS+=(--high-density-calls-csv "$HIGH_DENSITY_CALLS_CSV")
  fi
fi
"$PYTHON_BIN" -I cellpose_pipeline/scripts/01_segment_images.py \
  --dir "$INPUT_ROOT" \
  --recursive \
  --run-name "$RUN_NAME" \
  --out-root "$OUT_ROOT" \
  --profile-mode auto \
  --skip-unknown-profiles \
  "${PREFLIGHT_HIGH_DENSITY_ARGS[@]}" \
  --preflight-only

if [[ "$ENABLE_DEAD_CALIBRATION" != "0" ]]; then
  CALIBRATION_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${CALIBRATION_SBATCH_ARGS[@]}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",CALIBRATION_ROOT="$CALIBRATION_ROOT",CALIBRATION_SHARD_COUNT="$CALIBRATION_SHARD_COUNT" \
      "$DEAD_CALIBRATION_ARRAY_WORKER"
  )"
  echo "$CALIBRATION_SUBMIT_OUTPUT"
  CALIBRATION_JOB_ID="$(printf '%s\n' "$CALIBRATION_SUBMIT_OUTPUT" | awk '{print $4}')"
  CALIBRATION_MERGE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${CALIBRATION_MERGE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$CALIBRATION_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CALIBRATION_ROOT="$CALIBRATION_ROOT",CALIBRATION_SHARD_COUNT="$CALIBRATION_SHARD_COUNT",CALIBRATION_EXPECTED_PAIRS="$N_DEAD_TASKS" \
      "$DEAD_CALIBRATION_MERGE_WORKER"
  )"
  echo "$CALIBRATION_MERGE_SUBMIT_OUTPUT"
  CALIBRATION_MERGE_JOB_ID="$(printf '%s\n' "$CALIBRATION_MERGE_SUBMIT_OUTPUT" | awk '{print $4}')"
  MAIN_DEP_IDS+=("$CALIBRATION_MERGE_JOB_ID")
elif [[ "$ENABLE_DEAD_CONSENSUS" != "0" ]]; then
  if [[ ! -f "$DEAD_CALIBRATION_JSON" || ! -f "$DEAD_CALIBRATION_MAP" ]]; then
    echo "Dead consensus requires existing calibration JSON/map when calibration stage is disabled." >&2
    exit 2
  fi
fi

if [[ "$AUTO_DENSITY_PIPELINE" == "1" ]]; then
  NUCLEI_SUBMIT_OUTPUT="$(
    sbatch "${NUCLEI_SBATCH_ARGS[@]}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",OUT_ROOT="$OUT_ROOT",TASK_LIST="$TASK_LIST_NUCLEI",RUN_NAME="$RUN_NAME",USE_GPU="${USE_GPU:-1}",SEGMENTATION_ONLY=1,FORCE_CLASSIFICATION=0,ENABLE_HIGH_DENSITY_PROFILES=0,HIGH_DENSITY_CALLS_CSV="" \
      "$WORKER"
  )"
  echo "$NUCLEI_SUBMIT_OUTPUT"
  NUCLEI_JOB_ID="$(printf '%s\n' "$NUCLEI_SUBMIT_OUTPUT" | awk '{print $4}')"
  DENSITY_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${DENSITY_SBATCH_ARGS[@]}" \
      --dependency "afterok:$NUCLEI_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",OUT_ROOT="$OUT_ROOT",RUN_NAME="$RUN_NAME",DENSITY_CALLS_NAME="$DENSITY_CALLS_NAME" \
      "$DENSITY_TABLE_WORKER" "$OUT_ROOT" "$RUN_NAME" "$PROJECT_DIR"
  )"
  echo "$DENSITY_SUBMIT_OUTPUT"
  DENSITY_TABLE_JOB_ID="$(printf '%s\n' "$DENSITY_SUBMIT_OUTPUT" | awk '{print $4}')"
  MAIN_DEP_IDS+=("$DENSITY_TABLE_JOB_ID")
fi

if [[ "${#MAIN_DEP_IDS[@]}" -gt 0 ]]; then
  MAIN_DEPENDENCY="afterok:$(IFS=:; echo "${MAIN_DEP_IDS[*]}")"
  SBATCH_ARGS+=(--dependency "$MAIN_DEPENDENCY")
fi

SUBMIT_OUTPUT="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",OUT_ROOT="$OUT_ROOT",TASK_LIST="$TASK_LIST_MAIN",RUN_NAME="$RUN_NAME",USE_GPU="${USE_GPU:-1}",SEGMENTATION_ONLY="${SEGMENTATION_ONLY:-0}",FORCE_CLASSIFICATION="${FORCE_CLASSIFICATION:-1}",ENABLE_HIGH_DENSITY_PROFILES="$ENABLE_HIGH_DENSITY_PROFILES",HIGH_DENSITY_CALLS_CSV="$HIGH_DENSITY_CALLS_CSV",ENABLE_DEAD_CONSENSUS="$ENABLE_DEAD_CONSENSUS",DEAD_CALIBRATION_JSON="$DEAD_CALIBRATION_JSON",DEAD_CALIBRATION_MAP="$DEAD_CALIBRATION_MAP",DEAD_CALIBRATION_MODE="${DEAD_CALIBRATION_MODE:-bounded-background}",DEAD_MODEL="${DEAD_MODEL:-cpsam_v2}",DEAD_DIAMETER="${DEAD_DIAMETER:-22}",DEAD_CELLPROB_THRESHOLD="${DEAD_CELLPROB_THRESHOLD:--2.75}",DEAD_BLUE_MODEL="${DEAD_BLUE_MODEL:-cpsam}",DEAD_BLUE_TRANSFORM="${DEAD_BLUE_TRANSFORM:-blue_excess_mean}",DEAD_BLUE_DIAMETER="${DEAD_BLUE_DIAMETER:-22}",DEAD_BLUE_CELLPROB_THRESHOLD="${DEAD_BLUE_CELLPROB_THRESHOLD:--3.0}" \
    "$WORKER"
)"
echo "$SUBMIT_OUTPUT"

JOB_ID="$(printf '%s\n' "$SUBMIT_OUTPUT" | awk '{print $4}')"
DEAD_CONSENSUS_MERGE_JOB_ID=""
POST_SEGMENTATION_JOB_ID="$JOB_ID"
if [[ "$ENABLE_DEAD_CONSENSUS" != "0" ]]; then
  DEAD_CONSENSUS_MERGE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${DEAD_CONSENSUS_MERGE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT",RUN_NAME="$RUN_NAME",EXPECTED_KEYS="$TASK_LIST_FUSION" \
      "$DEAD_CONSENSUS_MERGE_WORKER"
  )"
  echo "$DEAD_CONSENSUS_MERGE_SUBMIT_OUTPUT"
  DEAD_CONSENSUS_MERGE_JOB_ID="$(printf '%s\n' "$DEAD_CONSENSUS_MERGE_SUBMIT_OUTPUT" | awk '{print $4}')"
  POST_SEGMENTATION_JOB_ID="$DEAD_CONSENSUS_MERGE_JOB_ID"
fi
FIELD_MANIFEST_JOB_ID=""
FUSION_JOB_ID=""
FUSION_MERGE_JOB_ID=""
NUCLEATED_BRANCH_JOB_ID=""
NUCLEATED_BRANCH_MERGE_JOB_ID=""
NUCLEATED_FUSION_JOB_ID=""
NUCLEATED_FUSION_MERGE_JOB_ID=""
LATE_DEATH_JOB_ID=""
SHAPE_JOB_ID=""
SHAPE_FINALIZE_JOB_ID=""
NUCLEATED_SHAPE_JOB_ID=""
NUCLEATED_SHAPE_FINALIZE_JOB_ID=""
if [[ "$ENABLE_FUSION_CLASSIFICATION" != "0" ]]; then
  FIELD_MANIFEST_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${FIELD_MANIFEST_SBATCH_ARGS[@]}" \
      --dependency "afterok:$POST_SEGMENTATION_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",NUCLEATED_BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",FORCE_FIELD_MANIFEST=1 \
      "$FIELD_MANIFEST_WORKER"
  )"
  echo "$FIELD_MANIFEST_SUBMIT_OUTPUT"
  FIELD_MANIFEST_JOB_ID="$(printf '%s\n' "$FIELD_MANIFEST_SUBMIT_OUTPUT" | awk '{print $4}')"

  FUSION_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${FUSION_SBATCH_ARGS[@]}" \
      --dependency "afterok:$FIELD_MANIFEST_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$OUT_ROOT",RUN_NAME="$RUN_NAME",OUT_DIR="$FUSION_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original,FORCE_FUSION="${FORCE_FUSION:-0}",CONTINUE_ON_ERROR="${FUSION_CONTINUE_ON_ERROR:-0}" \
      "$FUSION_ARRAY_WORKER"
  )"
  echo "$FUSION_SUBMIT_OUTPUT"
  FUSION_JOB_ID="$(printf '%s\n' "$FUSION_SUBMIT_OUTPUT" | awk '{print $4}')"

  FUSION_MERGE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${FUSION_MERGE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$FUSION_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT",OUT_DIR="$FUSION_OUT_DIR",FORCE_FUSION=1 \
      "$FUSION_MERGE_WORKER"
  )"
  echo "$FUSION_MERGE_SUBMIT_OUTPUT"
  FUSION_MERGE_JOB_ID="$(printf '%s\n' "$FUSION_MERGE_SUBMIT_OUTPUT" | awk '{print $4}')"
fi

if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
  NUCLEATED_BRANCH_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${NUCLEATED_BRANCH_SBATCH_ARGS[@]}" \
      --dependency "afterok:$FIELD_MANIFEST_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",TASK_LIST_BRANCH="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",RENDER_NUCLEATED_QC="${RENDER_NUCLEATED_QC:-0}",FORCE_NUCLEATED_BRANCH="${FORCE_NUCLEATED_BRANCH:-0}" \
      "$NUCLEATED_BRANCH_ARRAY_WORKER"
  )"
  echo "$NUCLEATED_BRANCH_SUBMIT_OUTPUT"
  NUCLEATED_BRANCH_JOB_ID="$(printf '%s\n' "$NUCLEATED_BRANCH_SUBMIT_OUTPUT" | awk '{print $4}')"
  NUCLEATED_BRANCH_MERGE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${NUCLEATED_BRANCH_MERGE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$NUCLEATED_BRANCH_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",BRANCH_ROOT="$NUCLEATED_BRANCH_ROOT",TASK_LIST_BRANCH="$TASK_LIST_FUSION" \
      "$NUCLEATED_BRANCH_MERGE_WORKER"
  )"
  echo "$NUCLEATED_BRANCH_MERGE_SUBMIT_OUTPUT"
  NUCLEATED_BRANCH_MERGE_JOB_ID="$(printf '%s\n' "$NUCLEATED_BRANCH_MERGE_SUBMIT_OUTPUT" | awk '{print $4}')"

  NUCLEATED_FUSION_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${FUSION_SBATCH_ARGS[@]}" \
      --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_fusion_nucleated" \
      --dependency "afterok:$NUCLEATED_BRANCH_MERGE_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",RUN_NAME=.,OUT_DIR="$NUCLEATED_FUSION_OUT_DIR",TASK_LIST_FUSION="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated,COMBINED_RUN="$NUCLEATED_BRANCH_ROOT/Combined",BRIGHTFIELD_RUN="$NUCLEATED_BRANCH_ROOT/Brightfield",DEAD_RUN="$WORKFLOW_RUN_DIR/Dead",NUCLEI_RUN="$WORKFLOW_RUN_DIR/Nuclei",FORCE_FUSION="${FORCE_FUSION:-0}",CONTINUE_ON_ERROR="${FUSION_CONTINUE_ON_ERROR:-0}" \
      "$FUSION_ARRAY_WORKER"
  )"
  echo "$NUCLEATED_FUSION_SUBMIT_OUTPUT"
  NUCLEATED_FUSION_JOB_ID="$(printf '%s\n' "$NUCLEATED_FUSION_SUBMIT_OUTPUT" | awk '{print $4}')"
  NUCLEATED_FUSION_MERGE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${FUSION_MERGE_SBATCH_ARGS[@]}" \
      --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_fusion_nuc_merge" \
      --dependency "afterok:$NUCLEATED_FUSION_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$WORKFLOW_RUN_DIR",OUT_DIR="$NUCLEATED_FUSION_OUT_DIR",FORCE_FUSION=1 \
      "$FUSION_MERGE_WORKER"
  )"
  echo "$NUCLEATED_FUSION_MERGE_SUBMIT_OUTPUT"
  NUCLEATED_FUSION_MERGE_JOB_ID="$(printf '%s\n' "$NUCLEATED_FUSION_MERGE_SUBMIT_OUTPUT" | awk '{print $4}')"
fi
if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
  LATE_DEATH_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${LATE_DEATH_SBATCH_ARGS[@]}" \
      --dependency "afterok:$FUSION_MERGE_JOB_ID:$NUCLEATED_FUSION_MERGE_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CLASSIFICATION_ROOT="$WORKFLOW_RUN_DIR",PLATE_MAP="$PROJECT_DIR/cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv",EXPECTED_FIELDS_PER_BRANCH="$N_FUSION_TASKS",WORKERS="${LATE_DEATH_WORKERS:-${LATE_DEATH_SBATCH_CPUS:-32}}",FORCE_LATE_DEATH="${FORCE_LATE_DEATH:-0}" \
      "$LATE_DEATH_WORKER"
  )"
  echo "$LATE_DEATH_SUBMIT_OUTPUT"
  LATE_DEATH_JOB_ID="$(printf '%s\n' "$LATE_DEATH_SUBMIT_OUTPUT" | awk '{print $4}')"
fi
if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
  ORIGINAL_SHAPE_DEPENDENCY="$FUSION_MERGE_JOB_ID"
  NUCLEATED_SHAPE_DEPENDENCY="$NUCLEATED_FUSION_MERGE_JOB_ID"
  if [[ "$ENABLE_LATE_DEATH_REFINEMENT" != "0" ]]; then
    ORIGINAL_SHAPE_DEPENDENCY="$LATE_DEATH_JOB_ID"
    NUCLEATED_SHAPE_DEPENDENCY="$LATE_DEATH_JOB_ID"
  fi
  SHAPE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${SHAPE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$ORIGINAL_SHAPE_DEPENDENCY" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$OUT_ROOT",SHAPE_ROOT="$SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original,FORCE_SHAPE_STRICT="${FORCE_SHAPE_STRICT:-0}" \
      "$SHAPE_ARRAY_WORKER"
  )"
  echo "$SHAPE_SUBMIT_OUTPUT"
  SHAPE_JOB_ID="$(printf '%s\n' "$SHAPE_SUBMIT_OUTPUT" | awk '{print $4}')"

  SHAPE_FINALIZE_SUBMIT_OUTPUT="$(
    "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${SHAPE_FINALIZE_SBATCH_ARGS[@]}" \
      --dependency "afterok:$SHAPE_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$OUT_ROOT",SHAPE_ROOT="$SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=original \
      "$SHAPE_FINALIZE_WORKER"
  )"
  echo "$SHAPE_FINALIZE_SUBMIT_OUTPUT"
  SHAPE_FINALIZE_JOB_ID="$(printf '%s\n' "$SHAPE_FINALIZE_SUBMIT_OUTPUT" | awk '{print $4}')"

  if [[ "$ENABLE_NUCLEATED_BRANCH" != "0" ]]; then
    NUCLEATED_SHAPE_SUBMIT_OUTPUT="$(
      "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${SHAPE_SBATCH_ARGS[@]}" \
        --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_shape_nucleated" \
        --dependency "afterok:$NUCLEATED_SHAPE_DEPENDENCY" \
        --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",CELL_RUN_ROOT="$NUCLEATED_BRANCH_ROOT",FUSION_ROOT="$NUCLEATED_FUSION_OUT_DIR",SHAPE_ROOT="$NUCLEATED_SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated,FORCE_SHAPE_STRICT="${FORCE_SHAPE_STRICT:-0}" \
        "$SHAPE_ARRAY_WORKER"
    )"
    echo "$NUCLEATED_SHAPE_SUBMIT_OUTPUT"
    NUCLEATED_SHAPE_JOB_ID="$(printf '%s\n' "$NUCLEATED_SHAPE_SUBMIT_OUTPUT" | awk '{print $4}')"
    NUCLEATED_SHAPE_FINALIZE_SUBMIT_OUTPUT="$(
      "${CPU_ONLY_SBATCH_ENV[@]}" sbatch "${SHAPE_FINALIZE_SBATCH_ARGS[@]}" \
        --job-name "${SBATCH_JOB_NAME:-cpsam_full_fusion}_shape_nuc_finalize" \
        --dependency "afterok:$NUCLEATED_SHAPE_JOB_ID" \
        --export=ALL,PROJECT_DIR="$PROJECT_DIR",INPUT_ROOT="$INPUT_ROOT",RUN_ROOT="$WORKFLOW_RUN_DIR",CELL_RUN_ROOT="$NUCLEATED_BRANCH_ROOT",FUSION_ROOT="$NUCLEATED_FUSION_OUT_DIR",SHAPE_ROOT="$NUCLEATED_SHAPE_ROOT",TASK_LIST_SHAPE="$TASK_LIST_FUSION",FIELD_MANIFEST_DIR="$FIELD_MANIFEST_DIR",CELL_MASK_BRANCH=nucleated \
        "$SHAPE_FINALIZE_WORKER"
    )"
    echo "$NUCLEATED_SHAPE_FINALIZE_SUBMIT_OUTPUT"
    NUCLEATED_SHAPE_FINALIZE_JOB_ID="$(printf '%s\n' "$NUCLEATED_SHAPE_FINALIZE_SUBMIT_OUTPUT" | awk '{print $4}')"
  fi
fi
echo "task_list_all=$TASK_LIST_ALL"
echo "task_list_nuclei=$TASK_LIST_NUCLEI"
echo "task_list_main=$TASK_LIST_MAIN"
echo "task_list_fusion=$TASK_LIST_FUSION"
echo "subdir_count_report=$SUBDIR_COUNT_REPORT"
echo "input_root=$INPUT_ROOT"
echo "skipped_subdirs=$SKIP_SUBDIRS"
echo "n_subdirs=$N_SUBDIRS"
printf 'subdir=%s\n' "${INPUT_SUBDIRS[@]}"
echo "auto_density_pipeline=$AUTO_DENSITY_PIPELINE"
echo "n_total_tasks=$N_TOTAL_TASKS"
echo "n_nuclei_tasks=$N_NUCLEI_TASKS"
echo "n_main_tasks=$N_MAIN_TASKS"
echo "n_fusion_tasks=$N_FUSION_TASKS"
echo "n_shape_tasks=$N_SHAPE_TASKS"
echo "out_root=$OUT_ROOT"
echo "run_name=$RUN_NAME"
echo "field_manifest_dir=$FIELD_MANIFEST_DIR"
echo "postseg_array_max_concurrent=$POSTSEG_ARRAY_MAX_CONCURRENT"
echo "enable_high_density_profiles=$ENABLE_HIGH_DENSITY_PROFILES"
echo "auto_generate_density_calls=$AUTO_GENERATE_DENSITY_CALLS"
echo "density_calls_name=$DENSITY_CALLS_NAME"
echo "high_density_calls_csv=${HIGH_DENSITY_CALLS_CSV:-unset}"
echo "nuclei_array_job_id=${NUCLEI_JOB_ID:-none}"
echo "density_table_job_id=${DENSITY_TABLE_JOB_ID:-none}"
echo "calibration_array_job_id=${CALIBRATION_JOB_ID:-none}"
echo "calibration_merge_job_id=${CALIBRATION_MERGE_JOB_ID:-none}"
echo "main_array_job_id=$JOB_ID"
echo "dead_consensus_merge_job_id=${DEAD_CONSENSUS_MERGE_JOB_ID:-none}"
echo "field_manifest_job_id=${FIELD_MANIFEST_JOB_ID:-none}"
echo "fusion_array_job_id=${FUSION_JOB_ID:-none}"
echo "fusion_merge_job_id=${FUSION_MERGE_JOB_ID:-none}"
echo "nucleated_branch_array_job_id=${NUCLEATED_BRANCH_JOB_ID:-none}"
echo "nucleated_branch_merge_job_id=${NUCLEATED_BRANCH_MERGE_JOB_ID:-none}"
echo "nucleated_fusion_array_job_id=${NUCLEATED_FUSION_JOB_ID:-none}"
echo "nucleated_fusion_merge_job_id=${NUCLEATED_FUSION_MERGE_JOB_ID:-none}"
echo "late_death_refinement_job_id=${LATE_DEATH_JOB_ID:-none}"
echo "shape_array_job_id=${SHAPE_JOB_ID:-none}"
echo "shape_finalize_job_id=${SHAPE_FINALIZE_JOB_ID:-none}"
echo "nucleated_shape_array_job_id=${NUCLEATED_SHAPE_JOB_ID:-none}"
echo "nucleated_shape_finalize_job_id=${NUCLEATED_SHAPE_FINALIZE_JOB_ID:-none}"
echo "fusion_out_dir=$FUSION_OUT_DIR"
echo "nucleated_fusion_out_dir=$NUCLEATED_FUSION_OUT_DIR"
echo "shape_root=$SHAPE_ROOT"
echo "nucleated_shape_root=$NUCLEATED_SHAPE_ROOT"
if [[ "$RUN_NAME" == "." || -z "$RUN_NAME" ]]; then
  echo "expected_run_dirs=$OUT_ROOT/<subdir>"
else
  echo "expected_run_dirs=$OUT_ROOT/<subdir>/$RUN_NAME"
fi
echo "expected_fusion_outputs=$FUSION_OUT_DIR"
if [[ "$ENABLE_SHAPE_STRICT" != "0" ]]; then
  echo "expected_shape_outputs=$SHAPE_ROOT"
fi
echo "logs=$LOG_DIR/${SBATCH_JOB_NAME:-cpsam_full_fusion}_${JOB_ID}_<task>.out"
