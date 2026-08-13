#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TARGET="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/31_run_reference_cell_state_shadow_v2_test.sh"
[[ -x "$TARGET" ]] || {
  echo "SIF-backed reference-cell-state V2 calibration is unavailable: $TARGET" >&2
  exit 2
}
export PROJECT_DIR
exec "$TARGET" "$@"
