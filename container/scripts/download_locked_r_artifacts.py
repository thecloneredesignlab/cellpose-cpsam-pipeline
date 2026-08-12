#!/usr/bin/env python3
"""Download the committed R artifact locks without dependency re-resolution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


LOCK_LAYOUT = (
    ("r-runtime.lock.tsv", "runtime"),
    ("r-system-debs.lock.tsv", "system"),
    ("r-packages.lock.tsv", "packages"),
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(row: dict[str, str], destination: Path) -> None:
    if destination.exists():
        if (
            destination.stat().st_size != int(row["bytes"])
            or sha256(destination) != row["sha256"]
        ):
            raise SystemExit(f"Existing artifact does not match lock: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise SystemExit(f"Remove stale partial download before retrying: {partial}")
    subprocess.run(
        [
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--silent",
            "--show-error",
            "--output",
            str(partial),
            row["url"],
        ],
        check=True,
    )
    if partial.stat().st_size != int(row["bytes"]):
        raise SystemExit(f"Downloaded byte-size mismatch: {row['filename']}")
    actual_sha256 = sha256(partial)
    if actual_sha256 != row["sha256"]:
        raise SystemExit(
            f"Downloaded SHA-256 mismatch for {row['filename']}: "
            f"expected={row['sha256']} actual={actual_sha256}"
        )
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    count = 0
    total_bytes = 0
    for lock_name, destination_name in LOCK_LAYOUT:
        with (args.lock_dir / lock_name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            destination = args.output_dir / destination_name / row["filename"]
            download(row, destination)
            count += 1
            total_bytes += destination.stat().st_size
    print(f"Prepared {count} locked R artifacts ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
