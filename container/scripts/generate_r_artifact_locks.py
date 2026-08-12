#!/usr/bin/env python3
"""Generate deterministic lock tables for an external R artifact context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shlex
import subprocess
from pathlib import Path

from prepare_r_package_context import (
    PPM_BASE,
    R_DEB_URL,
    archive_description,
    installation_order,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def deb_field(path: Path, field: str) -> str:
    completed = subprocess.run(
        ["dpkg-deb", "--field", str(path), field],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_system_urls(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("'"):
            continue
        fields = shlex.split(line)
        if len(fields) < 3:
            raise SystemExit(f"Malformed apt URI line: {line}")
        url, filename = fields[:2]
        if not url.startswith(("http://", "https://")):
            continue
        url = url.replace("http://", "https://", 1)
        if re.match(
            r"^https://snapshot\.debian\.org/archive/"
            r"(?:debian|debian-security)/\d{8}T\d{6}Z/",
            url,
        ) is None:
            raise SystemExit(
                "System deb URL is not pinned to a timestamped Debian snapshot: "
                f"{url}"
            )
        if filename in result:
            raise SystemExit(f"Duplicate apt filename: {filename}")
        result[filename] = url
    return result


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("system_apt_uri_output", type=Path)
    parser.add_argument("lock_dir", type=Path)
    args = parser.parse_args()

    runtime_files = sorted((args.artifact_dir / "runtime").glob("*.deb"))
    if [path.name for path in runtime_files] != ["r-4.2.3_1_amd64.deb"]:
        raise SystemExit(f"Unexpected R runtime artifacts: {runtime_files}")
    runtime_path = runtime_files[0]
    runtime_rows = [
        {
            "package": deb_field(runtime_path, "Package"),
            "version": deb_field(runtime_path, "Version"),
            "architecture": deb_field(runtime_path, "Architecture"),
            "filename": runtime_path.name,
            "bytes": runtime_path.stat().st_size,
            "sha256": sha256(runtime_path),
            "url": R_DEB_URL,
        }
    ]

    package_paths = sorted((args.artifact_dir / "packages").glob("*.tar.gz"))
    records = {}
    paths_by_package: dict[str, Path] = {}
    for package_path in package_paths:
        package = package_path.name.split("_", 1)[0]
        record = archive_description(package_path, package)
        records[package] = record
        paths_by_package[package] = package_path
    order = installation_order(records)
    package_rows: list[dict[str, object]] = []
    for position, package in enumerate(order, start=1):
        path = paths_by_package[package]
        record = records[package]
        package_rows.append(
            {
                "install_order": position,
                "package": package,
                "version": record["Version"],
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "built": record.get("Built", ""),
                "url": f"{PPM_BASE}/{path.name}",
            }
        )

    system_urls = read_system_urls(args.system_apt_uri_output)
    system_paths = sorted((args.artifact_dir / "system").glob("*.deb"))
    observed_system = {path.name for path in system_paths}
    if observed_system != set(system_urls):
        raise SystemExit(
            "System deb URI closure differs from downloaded closure: "
            f"missing={sorted(set(system_urls) - observed_system)} "
            f"extra={sorted(observed_system - set(system_urls))}"
        )
    system_rows: list[dict[str, object]] = []
    for path in system_paths:
        system_rows.append(
            {
                "package": deb_field(path, "Package"),
                "version": deb_field(path, "Version"),
                "architecture": deb_field(path, "Architecture"),
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "url": system_urls[path.name],
            }
        )
    system_rows.sort(key=lambda row: (str(row["package"]), str(row["architecture"])))

    common_fields = [
        "package",
        "version",
        "architecture",
        "filename",
        "bytes",
        "sha256",
        "url",
    ]
    write_tsv(args.lock_dir / "r-runtime.lock.tsv", common_fields, runtime_rows)
    write_tsv(args.lock_dir / "r-system-debs.lock.tsv", common_fields, system_rows)
    write_tsv(
        args.lock_dir / "r-packages.lock.tsv",
        ["install_order", "package", "version", "filename", "bytes", "sha256", "built", "url"],
        package_rows,
    )
    print(
        f"Wrote locks for {len(runtime_rows)} R runtime deb, "
        f"{len(system_rows)} system debs, and {len(package_rows)} R packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
