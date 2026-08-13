#!/usr/bin/env python3
"""Download the pinned Posit R runtime and binary R-package closure.

This prepares the runtime/ and packages/ portions of the external
R_ARTIFACTS_CONTEXT. Debian dependency archives are captured separately from
the exact base image so apt resolves against the same installed package set.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import tarfile
import tempfile
from pathlib import Path


SNAPSHOT = "2025-11-11"
PPM_BASE = (
    "https://packagemanager.posit.co/cran/"
    f"{SNAPSHOT}/bin/linux/trixie-x86_64/4.2/src/contrib"
)
R_DEB_URL = "https://cdn.posit.co/r/debian-13/pkgs/r-4.2.3_1_amd64.deb"
R_DEB_SHA256 = "d2c1316527b21211d2802f123c8b5dca7265f6c69df8ce39c8b0aacc79260952"

EXPECTED_PACKAGE_VERSIONS = {
    "BH": "1.87.0-1",
    "FNN": "1.1.4.1",
    "R6": "2.6.1",
    "RColorBrewer": "1.1-3",
    "RSpectra": "0.16-2",
    "Rcpp": "1.1.0",
    "RcppAnnoy": "0.0.22",
    "RcppEigen": "0.3.4.0.2",
    "RcppProgress": "0.4.2",
    "S7": "0.2.0",
    "brio": "1.1.5",
    "callr": "3.7.6",
    "cli": "3.6.5",
    "cpp11": "0.5.2",
    "crayon": "1.5.3",
    "dbscan": "1.2.3",
    "desc": "1.4.3",
    "diffobj": "0.3.6",
    "digest": "0.6.37",
    "dplyr": "1.1.4",
    "dqrng": "0.4.1",
    "evaluate": "1.0.5",
    "farver": "2.1.2",
    "foreach": "1.5.2",
    "fs": "1.6.6",
    "generics": "0.1.4",
    "ggplot2": "4.0.0",
    "glmnet": "4.1-10",
    "glue": "1.8.0",
    "gtable": "0.3.6",
    "irlba": "2.3.5.1",
    "isoband": "0.2.7",
    "iterators": "1.0.14",
    "jpeg": "0.1-11",
    "jsonlite": "2.0.0",
    "labeling": "0.4.3",
    "lifecycle": "1.0.4",
    "magrittr": "2.0.4",
    "pillar": "1.11.1",
    "pkgbuild": "1.4.8",
    "pkgconfig": "2.0.3",
    "pkgload": "1.4.1",
    "png": "0.1-8",
    "praise": "1.0.0",
    "processx": "3.8.6",
    "ps": "1.9.1",
    "purrr": "1.2.0",
    "rlang": "1.1.6",
    "rprojroot": "2.1.1",
    "scales": "1.4.0",
    "shape": "1.4.6.1",
    "sitmo": "2.0.2",
    "stringi": "1.8.7",
    "stringr": "1.6.0",
    "testthat": "3.2.3",
    "tibble": "3.3.0",
    "tidyr": "1.3.1",
    "tidyselect": "1.2.1",
    "tiff": "0.1-12",
    "utf8": "1.2.6",
    "uwot": "0.2.4",
    "vctrs": "0.6.5",
    "viridisLite": "0.4.2",
    "waldo": "0.6.2",
    "withr": "3.0.2",
    "yaml": "2.3.10",
}

ROOT_PACKAGES = (
    "digest",
    "jpeg",
    "jsonlite",
    "png",
    "tiff",
    "yaml",
    "glmnet",
    "pkgload",
    "testthat",
    "uwot",
    "dbscan",
    "dplyr",
    "tidyr",
    "purrr",
    "stringr",
    "ggplot2",
)

BASE_PACKAGES = {
    "R",
    "base",
    "compiler",
    "datasets",
    "graphics",
    "grDevices",
    "grid",
    "methods",
    "parallel",
    "splines",
    "stats",
    "stats4",
    "tcltk",
    "tools",
    "utils",
    "boot",
    "class",
    "cluster",
    "codetools",
    "foreign",
    "KernSmooth",
    "lattice",
    "MASS",
    "Matrix",
    "mgcv",
    "nlme",
    "nnet",
    "rpart",
    "spatial",
    "survival",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    partial = destination.with_name(destination.name + ".partial")
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
            url,
        ],
        check=True,
    )
    partial.replace(destination)


def parse_dcf(text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for block in text.strip().split("\n\n"):
        record: dict[str, str] = {}
        key = ""
        for line in block.splitlines():
            if line.startswith((" ", "\t")) and key:
                record[key] += " " + line.strip()
            elif ": " in line:
                key, value = line.split(": ", 1)
                record[key] = value
        if "Package" in record:
            records[record["Package"]] = record
    return records


def dependencies(record: dict[str, str]) -> list[str]:
    result: list[str] = []
    for field in ("Depends", "Imports", "LinkingTo"):
        for specification in record.get(field, "").split(","):
            package = specification.split("(", 1)[0].strip()
            if package and package not in BASE_PACKAGES and package not in result:
                result.append(package)
    return result


def installation_order(records: dict[str, dict[str, str]]) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()
    active: set[str] = set()

    def visit(package: str) -> None:
        if package in visited:
            return
        if package in active:
            raise SystemExit(f"Dependency cycle at {package}")
        if package not in records:
            raise SystemExit(f"Package is absent from pinned PPM index: {package}")
        active.add(package)
        for dependency in dependencies(records[package]):
            visit(dependency)
        active.remove(package)
        visited.add(package)
        ordered.append(package)

    for root in ROOT_PACKAGES:
        visit(root)
    return ordered


def archive_description(path: Path, package: str) -> dict[str, str]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(f"{package}/DESCRIPTION")
        handle = archive.extractfile(member)
        if handle is None:
            raise SystemExit(f"DESCRIPTION cannot be read from {path}")
        records = parse_dcf(handle.read().decode("utf-8"))
    if package not in records:
        raise SystemExit(f"Unexpected DESCRIPTION package in {path}")
    return records[package]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    runtime_dir = args.output_dir / "runtime"
    package_dir = args.output_dir / "packages"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    r_deb = runtime_dir / "r-4.2.3_1_amd64.deb"
    download(R_DEB_URL, r_deb)
    actual_r_sha256 = sha256(r_deb)
    if actual_r_sha256 != R_DEB_SHA256:
        raise SystemExit(
            f"R deb SHA-256 mismatch: expected={R_DEB_SHA256} "
            f"actual={actual_r_sha256}"
        )

    with tempfile.TemporaryDirectory(prefix="cellpose-r-index-") as temporary:
        index_path = Path(temporary) / "PACKAGES.gz"
        download(f"{PPM_BASE}/PACKAGES.gz", index_path)
        records = parse_dcf(gzip.decompress(index_path.read_bytes()).decode("utf-8"))

    order = installation_order(records)
    if set(order) != set(EXPECTED_PACKAGE_VERSIONS):
        missing = sorted(set(EXPECTED_PACKAGE_VERSIONS) - set(order))
        extra = sorted(set(order) - set(EXPECTED_PACKAGE_VERSIONS))
        raise SystemExit(f"R package closure changed: missing={missing} extra={extra}")

    total_bytes = 0
    for position, package in enumerate(order, start=1):
        version = records[package]["Version"]
        expected = EXPECTED_PACKAGE_VERSIONS[package]
        if version != expected:
            raise SystemExit(
                f"Version drift for {package}: expected={expected} actual={version}"
            )
        filename = f"{package}_{version}.tar.gz"
        path = package_dir / filename
        download(f"{PPM_BASE}/{filename}", path)
        description = archive_description(path, package)
        if description.get("Version") != version:
            raise SystemExit(f"Archive version mismatch for {package}")
        built = description.get("Built", "")
        # Posit publishes both compiled and pure-R Linux binary tarballs in
        # this repository.  Every archive must advertise the pinned R 4.2
        # binary ABI; only compiled packages carry an architecture token.
        if not built.startswith("R 4.2."):
            raise SystemExit(f"Unexpected binary build metadata for {package}: {built}")
        total_bytes += path.stat().st_size
        print(
            f"{position:02d}\t{package}\t{version}\t{path.name}\t"
            f"{path.stat().st_size}\t{sha256(path)}\t{built}"
        )

    observed = {path.name for path in package_dir.glob("*.tar.gz")}
    expected_files = {
        f"{package}_{version}.tar.gz"
        for package, version in EXPECTED_PACKAGE_VERSIONS.items()
    }
    if observed != expected_files:
        raise SystemExit(
            "R package directory is not exact: "
            f"missing={sorted(expected_files - observed)} "
            f"extra={sorted(observed - expected_files)}"
        )

    print(
        f"Prepared R 4.2.3 plus {len(order)} binary R packages "
        f"({total_bytes} package bytes) from PPM snapshot {SNAPSHOT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
