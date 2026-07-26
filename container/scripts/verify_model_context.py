#!/usr/bin/env python3
"""Verify that a model-only build context matches the public model lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_lock", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    with args.model_lock.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {row["model"]: row for row in rows}
    observed = {path.name: path for path in args.model_dir.iterdir() if path.is_file()}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise SystemExit(f"Model context mismatch: missing={missing}, extra={extra}")

    for name, row in expected.items():
        path = observed[name]
        if path.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"Model size mismatch: {name}")
        actual = sha256(path)
        if actual != row["sha256"]:
            raise SystemExit(
                f"Model SHA-256 mismatch for {name}: expected={row['sha256']} actual={actual}"
            )

    print(f"Verified {len(expected)} Cellpose model artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
