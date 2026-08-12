#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TARGET="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/Parameter_calibration/29_run_broad_phenotype_shadow_test.sh"
[[ -x "$TARGET" ]] || {
  echo "SIF-backed broad-phenotype test entry is unavailable: $TARGET" >&2
  exit 2
}
export PROJECT_DIR
exec "$TARGET" "$@"
