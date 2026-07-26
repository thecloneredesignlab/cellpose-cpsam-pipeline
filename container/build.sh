#!/usr/bin/env bash

set -euo pipefail

image_tag="${IMAGE_TAG:-cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models}"
conda_context="${CONDA_PACKAGES_CONTEXT:?Set CONDA_PACKAGES_CONTEXT to a directory containing packages/}"
pip_context="${PIP_WHEELS_CONTEXT:?Set PIP_WHEELS_CONTEXT to a directory containing wheels/}"
model_context="${CELLPOSE_MODELS_CONTEXT:?Set CELLPOSE_MODELS_CONTEXT to a directory containing models/}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/scripts/verify_conda_package_context.py" \
  "${script_dir}/locks/conda-explicit-linux-64.lock.txt" \
  "${script_dir}/locks/conda-package-archives.sha256" \
  "${conda_context}/packages"
python3 "${script_dir}/scripts/verify_model_context.py" \
  "${script_dir}/locks/cellpose-models.lock.tsv" \
  "${model_context}/models"

docker buildx build \
  --platform linux/amd64 \
  --build-context "conda_packages=${conda_context}" \
  --build-context "pip_wheels=${pip_context}" \
  --build-context "models=${model_context}" \
  --load \
  --tag "${image_tag}" \
  "${script_dir}"
