"""Shared helpers for selecting exact Incucyte time points from field keys."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


KEY_RE = re.compile(r"([A-H]\d+_\d+_(\d{2}d\d{2}h\d{2}m))")
TIMEPOINT_RE = re.compile(r"^\d{2}d\d{2}h\d{2}m$")
D0_TIMEPOINT = "00d00h00m"
D0_ALIASES = {"0", "d0", "day0", "day_0", D0_TIMEPOINT}


def normalize_timepoint(value: str) -> str:
    """Return the canonical ``DDdHHhMMm`` selector, accepting common d0 aliases."""

    normalized = value.strip().lower().lstrip("_")
    if normalized in D0_ALIASES:
        return D0_TIMEPOINT
    if TIMEPOINT_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"Invalid timepoint {value!r}; use DDdHHhMMm (for example, {D0_TIMEPOINT}) or d0"
        )
    return normalized


def extract_key_and_timepoint(value: str | Path) -> tuple[str, str]:
    """Extract the normalized field key and exact timepoint from a path or name."""

    match = KEY_RE.search(Path(value).name)
    if match is None:
        raise ValueError(f"Cannot extract field key/timepoint from {value}")
    return match.group(1), match.group(2)


def key_matches_timepoint(key: str, timepoint: str | None) -> bool:
    """Return whether a normalized key belongs to the requested exact timepoint."""

    if timepoint is None:
        return True
    _key, observed = extract_key_and_timepoint(key)
    return observed == normalize_timepoint(timepoint)


def select_keys(keys: Iterable[str], timepoint: str | None) -> list[str]:
    """Sort and select keys, preserving all keys when no selector is supplied."""

    return sorted(key for key in keys if key_matches_timepoint(key, timepoint))
