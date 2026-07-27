#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/util/hpc_container_apptainer_runtime.sh"
hpc_container_prepare

PROJECT_DIR="${PROJECT_DIR:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3}"
HPC_PROJECT_ROOT="${PROJECT_DIR}"
export HPC_PROJECT_ROOT
RUN_ROOT="${RUN_ROOT:-/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/classification_fusion}"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
hpc_container_ignore_host_runtime "${PYTHON_BIN:-}"
PYTHON_BIN=python

export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_fusion_merge_mpl_${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

cd "$PROJECT_DIR"

ARGS=(
  cellpose_pipeline/scripts/08_fuse_multichannel_classification.py
  --out-dir "$OUT_DIR"
  --merge-summaries-only
)

if [[ "${FORCE_FUSION:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "host=$(hostname)"
echo "project_dir=$PROJECT_DIR"
echo "run_root=$RUN_ROOT"
echo "out_dir=$OUT_DIR"
echo "runtime=apptainer_sif"
echo "python_bin=$PYTHON_BIN"
printf 'fusion_merge_arg=%s\n' "${ARGS[@]}"

"$PYTHON_BIN" -I "${ARGS[@]}"
