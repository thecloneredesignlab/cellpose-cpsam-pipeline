#!/usr/bin/env python3
"""Verify an offline Conda package context against an explicit lock."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("explicit_lock", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--write-sha256", type=Path)
    args = parser.parse_args()

    expected: dict[str, str] = {}
    for raw in args.explicit_lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "@EXPLICIT":
            continue
        url, separator, expected_md5 = line.partition("#")
        if not separator or len(expected_md5) != 32:
            raise SystemExit(f"Explicit entry lacks a single MD5 fragment: {line}")
        name = Path(urlparse(url).path).name
        if name in expected:
            raise SystemExit(f"Duplicate package filename in explicit lock: {name}")
        expected[name] = expected_md5

    observed = {
        path.name: path
        for path in args.package_dir.iterdir()
        if path.is_file() and (path.name.endswith(".conda") or path.name.endswith(".tar.bz2"))
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise SystemExit(f"Offline package mismatch: missing={missing}, extra={extra}")

    sha256_lines: list[str] = []
    total_bytes = 0
    for name in sorted(expected):
        path = observed[name]
        actual_md5 = digest(path, "md5")
        if actual_md5 != expected[name]:
            raise SystemExit(
                f"MD5 mismatch for {name}: expected={expected[name]} actual={actual_md5}"
            )
        total_bytes += path.stat().st_size
        sha256_lines.append(f"{digest(path, 'sha256')}  {name}")

    if args.write_sha256:
        args.write_sha256.parent.mkdir(parents=True, exist_ok=True)
        args.write_sha256.write_text("\n".join(sha256_lines) + "\n", encoding="utf-8")

    print(
        f"Verified {len(expected)} explicit Conda archives "
        f"({total_bytes} bytes) with MD5 and SHA-256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
