#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TARGET="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/33_run_multimodal_cell_state_v3_test.sh"
[[ -x "$TARGET" ]] || {
  echo "SIF-backed multimodal cell-state V3 calibration is unavailable: $TARGET" >&2
  exit 2
}
export PROJECT_DIR
exec "$TARGET" "$@"
