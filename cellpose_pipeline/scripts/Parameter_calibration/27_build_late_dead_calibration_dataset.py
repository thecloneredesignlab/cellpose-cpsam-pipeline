#!/usr/bin/env python3
"""Build the late-death calibration dataset with the shared production builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "13_build_late_dead_trajectory_dataset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "late_dead_trajectory_dataset_local",
    BUILDER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load late-death dataset builder: {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


if __name__ == "__main__":
    raise SystemExit(BUILDER.main())
