from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = PROJECT_ROOT / "cellpose_pipeline"
DEFAULT_RAW_DIR = PROJECT_ROOT / "20260619_SUM159_Doxorubicin_Cyclophosphamide" / "20260626_SUM159_AC_Exp_1"

IMAGE_SUFFIXES = {".tif", ".tiff"}
MOVIE_SUFFIXES = {".avi", ".mp4", ".wmv"}

FILENAME_RE = re.compile(
    r"SUM159_AC_+?(?P<well>[A-H]\d{1,2})_(?P<site>\d+)"
    r"(?:_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedName:
    well: str | None
    site: str | None
    day: int | None
    hour: int | None
    minute: int | None
    elapsed_hours: float | None


def parse_image_name(path: Path) -> ParsedName:
    match = FILENAME_RE.search(path.stem)
    if not match:
        return ParsedName(None, None, None, None, None, None)

    day = _int_or_none(match.group("day"))
    hour = _int_or_none(match.group("hour"))
    minute = _int_or_none(match.group("minute"))
    elapsed = None
    if day is not None and hour is not None and minute is not None:
        elapsed = day * 24 + hour + minute / 60

    return ParsedName(
        well=match.group("well").upper(),
        site=match.group("site"),
        day=day,
        hour=hour,
        minute=minute,
        elapsed_hours=elapsed,
    )


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value is not None else None


def iter_files(root: Path, suffixes: set[str]):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in suffixes:
                yield path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_project(path: Path) -> str:
    abs_path = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def annotation_mask_for(image_path: Path, mask_filter: str) -> Path:
    if mask_filter.startswith("_"):
        return image_path.with_name(f"{image_path.stem}{mask_filter}")
    return image_path.with_suffix(mask_filter)


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_file:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)
