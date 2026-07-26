#!/usr/bin/env python3
"""Create an identity-safe, checksum-covered public HPC snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

SENSITIVE_KEY = re.compile(
    r"(?i)(token|password|passwd|secret|credential|authorization|cookie|"
    r"private.?key|client.?cert|proxy)"
)
SSH_TARGET = re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\b")
GPU_UUID = re.compile(r"\bGPU-[0-9A-Fa-f-]{16,}\b")
KEY_VALUE = re.compile(
    r"^(?P<prefix>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)(?P<sep>\s*[=\t]\s*)(?P<value>.*)$"
)
IDENTITY_KEYS = {
    "hostname": "<HPC_HOST>",
    "slurm_job_id": "<SLURM_JOB_ID>",
    "slurm_node_list": "<SLURM_NODE>",
    "slurm_job_name": "<SLURM_JOB>",
    "repo_root": "<HPC_REPO_PATH>",
}


def parse_redaction(value: str) -> tuple[str, str]:
    if "=" not in value:
        return value, "<REDACTED>"
    source, replacement = value.split("=", 1)
    if not source:
        raise argparse.ArgumentTypeError("Redaction source must not be empty")
    if not replacement:
        replacement = "<REDACTED>"
    return source, replacement


def capture_metadata(raw: Path) -> dict[str, str]:
    path = raw / "capture_metadata.tsv"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key] = value
    return values


def sanitize_string(value: str, replacements: list[tuple[str, str]]) -> str:
    for source, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if source:
            value = value.replace(source, replacement)
    value = SSH_TARGET.sub("<SSH_TARGET>", value)
    value = GPU_UUID.sub("<GPU_UUID>", value)
    return value


def sanitize_json(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<REDACTED>"
                if SENSITIVE_KEY.search(str(key))
                else sanitize_json(item, replacements)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json(item, replacements) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, replacements)
    return value


def sanitize_text(text: str, replacements: list[tuple[str, str]]) -> str:
    text = sanitize_string(text, replacements)
    output: list[str] = []
    for line in text.splitlines():
        match = KEY_VALUE.match(line)
        if not match:
            output.append(line)
            continue
        key = match.group("key")
        key_lower = key.lower()
        if key_lower in IDENTITY_KEYS:
            output.append(
                f"{match.group('prefix')}{key}{match.group('sep')}{IDENTITY_KEYS[key_lower]}"
            )
        elif key.upper() == key and SENSITIVE_KEY.search(key):
            output.append(f"{match.group('prefix')}{key}{match.group('sep')}<REDACTED>")
        else:
            output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_snapshot", type=Path)
    parser.add_argument("public_snapshot", type=Path)
    parser.add_argument(
        "--redact",
        action="append",
        default=[],
        type=parse_redaction,
        metavar="VALUE=PLACEHOLDER",
    )
    args = parser.parse_args()

    raw = args.raw_snapshot.resolve(strict=True)
    public = args.public_snapshot.resolve()
    if raw == public or raw in public.parents:
        raise SystemExit("Public snapshot must not equal or contain the raw snapshot")
    if public.exists() and any(public.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {public}")
    public.mkdir(parents=True, exist_ok=True)

    replacements = list(args.redact)
    metadata = capture_metadata(raw)
    for key, placeholder in IDENTITY_KEYS.items():
        value = metadata.get(key, "")
        if value and value != "NA":
            replacements.append((value, placeholder))

    copied = 0
    for source in sorted(path for path in raw.rglob("*") if path.is_file()):
        if source.name == "SHA256SUMS":
            continue
        payload = source.read_bytes()
        if b"\x00" in payload:
            raise SystemExit(f"Binary file requires explicit private review: {source}")
        relative = source.relative_to(raw)
        target = public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = payload.decode("utf-8", errors="replace")
        if source.suffix.lower() == ".json":
            parsed = json.loads(text)
            sanitized = json.dumps(
                sanitize_json(parsed, replacements),
                indent=2,
                sort_keys=True,
            ) + "\n"
        else:
            sanitized = sanitize_text(text, replacements)
        target.write_text(sanitized, encoding="utf-8")
        shutil.copymode(source, target)
        copied += 1

    checksum_lines = []
    for path in sorted(item for item in public.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(public).as_posix()}")
    (public / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"Sanitized {copied} text files into {public}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
