#!/usr/bin/env python3
"""Verify the complete offline R artifact context against committed locks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

from prepare_r_package_context import archive_description


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_lock(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"Artifact lock is empty: {path}")
    filenames = [row["filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise SystemExit(f"Artifact lock has duplicate filenames: {path}")
    return rows


def deb_field(path: Path, field: str) -> str:
    completed = subprocess.run(
        ["dpkg-deb", "--field", str(path), field],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def installed_deb_version(package: str, architecture: str) -> str | None:
    query = package if architecture == "all" else f"{package}:{architecture}"
    completed = subprocess.run(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${Version}", query],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    status, separator, version = completed.stdout.partition("\t")
    if not separator or not status.startswith("ii"):
        return None
    return version.strip()


def verify_group(
    artifact_dir: Path,
    lock_path: Path,
    suffix: str,
    validate_deb: bool,
    validate_r_package: bool,
    reject_installed_debs: bool = False,
) -> tuple[int, int]:
    rows = read_lock(lock_path)
    expected = {row["filename"]: row for row in rows}
    observed = {
        path.name: path
        for path in artifact_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"Offline R artifact mismatch for {artifact_dir}: "
            f"missing={missing} extra={extra}"
        )

    total_bytes = 0
    for filename, row in expected.items():
        path = observed[filename]
        actual_bytes = path.stat().st_size
        if actual_bytes != int(row["bytes"]):
            raise SystemExit(
                f"Byte-size mismatch for {filename}: "
                f"expected={row['bytes']} actual={actual_bytes}"
            )
        actual_sha256 = sha256(path)
        if actual_sha256 != row["sha256"]:
            raise SystemExit(
                f"SHA-256 mismatch for {filename}: "
                f"expected={row['sha256']} actual={actual_sha256}"
            )
        if validate_deb:
            for field, lock_field in (
                ("Package", "package"),
                ("Version", "version"),
                ("Architecture", "architecture"),
            ):
                actual = deb_field(path, field)
                if actual != row[lock_field]:
                    raise SystemExit(
                        f"Deb metadata mismatch for {filename} {field}: "
                        f"expected={row[lock_field]} actual={actual}"
                    )
            if reject_installed_debs:
                installed = installed_deb_version(row["package"], row["architecture"])
                if installed is not None:
                    raise SystemExit(
                        "Locked R system dependency would overwrite the immutable "
                        f"base image package {row['package']}:{row['architecture']} "
                        f"version {installed}"
                    )
        if validate_r_package:
            description = archive_description(path, row["package"])
            if description.get("Version") != row["version"]:
                raise SystemExit(f"R package version mismatch for {filename}")
            if description.get("Built", "") != row["built"]:
                raise SystemExit(f"R package build metadata mismatch for {filename}")
        total_bytes += actual_bytes
    return len(rows), total_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("lock_dir", type=Path)
    parser.add_argument(
        "--reject-installed-system-packages",
        action="store_true",
        help="fail if any locked system deb is already installed in the base image",
    )
    args = parser.parse_args()

    validate_deb = shutil.which("dpkg-deb") is not None
    runtime_count, runtime_bytes = verify_group(
        args.artifact_dir / "runtime",
        args.lock_dir / "r-runtime.lock.tsv",
        ".deb",
        validate_deb,
        False,
        False,
    )
    system_count, system_bytes = verify_group(
        args.artifact_dir / "system",
        args.lock_dir / "r-system-debs.lock.tsv",
        ".deb",
        validate_deb,
        False,
        args.reject_installed_system_packages,
    )
    package_count, package_bytes = verify_group(
        args.artifact_dir / "packages",
        args.lock_dir / "r-packages.lock.tsv",
        ".tar.gz",
        False,
        True,
        False,
    )
    print(
        f"Verified {runtime_count} R runtime deb, {system_count} system debs, "
        f"and {package_count} R binary packages "
        f"({runtime_bytes + system_bytes + package_bytes} bytes) with SHA-256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
