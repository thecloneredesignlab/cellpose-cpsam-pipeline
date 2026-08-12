#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TARGET="$PROJECT_DIR/cellpose_pipeline/Docker/hpc/submit_broad_phenotype_shadow_full.sh"
[[ -x "$TARGET" ]] || {
  echo "SIF-backed broad-phenotype submitter is unavailable: $TARGET" >&2
  exit 2
}
export PROJECT_DIR
exec "$TARGET" "$@"
