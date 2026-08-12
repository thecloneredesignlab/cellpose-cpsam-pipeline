#!/usr/bin/env bash

set -euo pipefail

r_artifacts_context="${R_ARTIFACTS_CONTEXT:?Set R_ARTIFACTS_CONTEXT to the destination directory}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/scripts/download_locked_r_artifacts.py" \
  "${script_dir}/locks" \
  "${r_artifacts_context}"
python3 "${script_dir}/scripts/verify_r_artifact_context.py" \
  "${r_artifacts_context}" \
  "${script_dir}/locks"
