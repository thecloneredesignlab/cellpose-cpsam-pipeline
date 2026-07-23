#!/usr/bin/env python3
"""Optimize the shared density-aware late-death trajectory model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "_shared"
    / "late_dead_trajectory_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "late_dead_trajectory_model_local",
    MODEL_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load late-death trajectory model: {MODEL_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


if __name__ == "__main__":
    raise SystemExit(MODEL.main())
