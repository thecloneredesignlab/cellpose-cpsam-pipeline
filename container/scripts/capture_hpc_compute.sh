#!/usr/bin/env bash

set -euo pipefail

: "${CAPTURE_STAGE:?CAPTURE_STAGE is required}"
: "${HPC_REPO_PATH:?HPC_REPO_PATH is required}"

input_dir="${CAPTURE_STAGE}/input"
raw_dir="${CAPTURE_STAGE}/${CAPTURE_OUTPUT_NAME:-raw-compute}"
skill_scripts="${CAPTURE_STAGE}/skill-scripts"

mkdir -p "${raw_dir}/system" "${raw_dir}/ecosystems/cellpose"
nvidia-smi >"${raw_dir}/system/nvidia-smi.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
  --format=csv,noheader >"${raw_dir}/system/nvidia-gpus.csv"

# This child process owns the one clean module/Conda initialization used by
# the generic capture. Do not pre-activate the environment in this wrapper.
bash "${skill_scripts}/capture_hpc_environment.sh" \
  "${raw_dir}" \
  "${input_dir}/module-init.sh" \
  "${HPC_REPO_PATH}" \
  python,conda \
  "${input_dir}/commands.tsv"

# The generic capture ran in a child process, so this parent is still clean.
# Activate once here for the repository-local Cellpose artifact adapter.
source "${input_dir}/module-init.sh"
python "${input_dir}/capture_cellpose_artifacts.py" \
  "${input_dir}/cellpose-models.requested.tsv" \
  "${raw_dir}/ecosystems/cellpose" \
  --load-models --gpu

find "${raw_dir}" -type f ! -name SHA256SUMS -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum >"${raw_dir}/SHA256SUMS"
