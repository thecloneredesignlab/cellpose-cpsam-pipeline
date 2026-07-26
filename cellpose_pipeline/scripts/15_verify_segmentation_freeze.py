#!/usr/bin/env python3
"""Verify that classification reuses the frozen v3 segmentation implementation.

This verifier is read-only.  It validates the recorded source-code hashes and
the exact immutable segmentation result root before a classification-only
Slurm workflow can be submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RESULT_DIRECTORIES = (
    "Combined/segmentations",
    "Brightfield/segmentations",
    "Dead/segmentations",
    "Nuclei/segmentations",
    "Nuclei/nucleus_core_seeds",
    "nucleated_only/Combined/segmentations",
    "nucleated_only/Brightfield/segmentations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    source_run_root = args.source_run_root.resolve()
    payload = json.loads(manifest_path.read_text())
    expected_root = Path(payload["frozen_result_root"])
    if source_run_root != expected_root:
        raise ValueError(
            "Classification source does not match the frozen segmentation root: "
            f"expected={expected_root}, observed={source_run_root}"
        )
    failures = []
    for relative, expected_hash in payload["sha256"].items():
        path = repo_root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_hash != expected_hash:
            failures.append(
                {
                    "path": relative,
                    "error": "sha256_mismatch",
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
    missing_directories = [
        relative
        for relative in REQUIRED_RESULT_DIRECTORIES
        if not (source_run_root / relative).is_dir()
    ]
    if failures or missing_directories:
        raise RuntimeError(
            "Segmentation freeze verification failed: "
            f"code_failures={failures}, "
            f"missing_result_directories={missing_directories}"
        )
    receipt = {
        "schema_version": 1,
        "verified": True,
        "classification_only": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "baseline_git_sha": payload["baseline_git_sha"],
        "manifest": str(manifest_path),
        "source_run_root": str(source_run_root),
        "verified_file_count": len(payload["sha256"]),
        "verified_result_directories": list(REQUIRED_RESULT_DIRECTORIES),
        "policy": payload["policy"],
    }
    receipt["receipt_path"] = str(args.output.resolve())
    write_json_atomic(args.output, receipt)
    print(f"segmentation_freeze_verified=1")
    print(f"verified_file_count={receipt['verified_file_count']}")
    print(f"source_run_root={source_run_root}")
    print(f"receipt={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
