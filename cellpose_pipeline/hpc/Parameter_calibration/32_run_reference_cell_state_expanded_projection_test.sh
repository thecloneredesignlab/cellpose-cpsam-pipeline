#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TARGET="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/32_run_reference_cell_state_expanded_projection_test.sh"
[[ -x "$TARGET" ]] || {
  echo "SIF-backed expanded reference-cell-state calibration is unavailable: $TARGET" >&2
  exit 2
}
export PROJECT_DIR
exec "$TARGET" "$@"
