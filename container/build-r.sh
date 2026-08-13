#!/usr/bin/env bash

set -euo pipefail

image_tag="${IMAGE_TAG:-zafiro/cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models-reference-v2-parity}"
r_artifacts_context="${R_ARTIFACTS_CONTEXT:?Set R_ARTIFACTS_CONTEXT to the prepared offline R artifact directory}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_revision="${SOURCE_REVISION:-$(git -C "${script_dir}/.." rev-parse HEAD)}"
if [[ -n "$(git -C "${script_dir}/.." status --porcelain --untracked-files=normal)" ]]; then
  default_source_tree_state="dirty"
else
  default_source_tree_state="clean"
fi
source_tree_state="${SOURCE_TREE_STATE:-${default_source_tree_state}}"
build_date="${BUILD_DATE:-$(date -u +'%Y-%m-%dT%H:%M:%SZ')}"

python3 "${script_dir}/scripts/verify_r_artifact_context.py" \
  "${r_artifacts_context}" \
  "${script_dir}/locks"

docker buildx build \
  --platform linux/amd64 \
  --build-arg "BUILD_DATE=${build_date}" \
  --build-arg "SOURCE_REVISION=${source_revision}" \
  --build-arg "SOURCE_TREE_STATE=${source_tree_state}" \
  --build-context "r_artifacts=${r_artifacts_context}" \
  --file "${script_dir}/Dockerfile.r" \
  --load \
  --tag "${image_tag}" \
  "${script_dir}"
