#!/usr/bin/env python3
"""Import one exact V4 three-channel review as immutable human evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "multimodal_cell_state_v4_review_import_v1"
SUBMISSION_SCHEMA = "multimodal_cell_state_v4_review_submission_v1"
RENDER_SCHEMA = "multimodal_cell_state_v4_review_render_v1"
PRIMARY = {"non_dead", "dead", "uncertain_or_unreviewable"}
STAGES = {
    "not_applicable_non_dead",
    "dead_marker_positive",
    "dead_marker_weak_or_transition",
    "dead_marker_negative_nucleus_lost_like",
    "death_stage_indeterminate",
}
NUCLEAR = {
    "single_nucleus",
    "multinucleated",
    "fragmented_nuclei",
    "nucleus_absent",
    "nuclear_state_uncertain",
}
CONFIDENCE = {"high", "medium", "low"}
OUTPUT_FIELDS = [
    "cell_id",
    "primary_death_label",
    "death_stage_label",
    "nuclear_state_label",
    "label_confidence",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "training_eligible",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--review-set", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, rows = list(reader.fieldnames or ()), list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Invalid TSV header: {path}")
    return fields, rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def confine(path: Path, root: Path, label: str, *, directory: bool = False) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    candidate = Path(os.path.realpath(lexical))
    root = Path(os.path.realpath(root.expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the V4 shadow root: {candidate}") from error
    if lexical.is_symlink():
        raise ValueError(f"{label} is a symlink: {lexical}")
    if directory:
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
    elif not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def stable_id_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def review_token(render_generation_id: str, cell_id: str) -> str:
    return hashlib.sha256(f"{render_generation_id}|{cell_id}".encode()).hexdigest()[:24]


def no_controls(value: str, field: str) -> str:
    if any(ord(character) < 32 and character not in "\t" for character in value):
        raise ValueError(f"Submission {field} contains control characters")
    return value


def require_timestamp(value: str) -> str:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid reviewed_at timestamp: {text}") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    return text


def artifact_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 review import contains a symlink: {path}")
        if path.is_file() and path.name not in excluded:
            output[path.relative_to(root).as_posix()] = sha256(path)
    return output


def verify_render(render: Path, review_ids: list[str]) -> dict[str, Any]:
    manifest_path = render / "exact_review_render_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != RENDER_SCHEMA
        or manifest.get("status") != "HUMAN_REVIEW_REQUIRED"
        or manifest.get("all_crops_available") is not True
        or manifest.get("exact_selection_preserved") is not True
        or manifest.get("secondary_sampling_performed") is not False
    ):
        raise ValueError("V4 render manifest is incomplete")
    declared = manifest.get("artifact_file_sha256")
    expected = {
        "exact_review_html": "exact_review.html",
        "crop_render_status": "crop_render_status.tsv",
        "crop_manifest": "crop_manifest.tsv",
    }
    if not isinstance(declared, dict) or set(declared) != set(expected):
        raise ValueError("V4 render artifact declaration differs")
    for role, name in expected.items():
        path = render / name
        if not path.is_file() or path.is_symlink() or sha256(path) != declared[role]:
            raise ValueError(f"V4 render artifact hash differs: {role}")
    fields, crops = read_tsv(render / "crop_manifest.tsv")
    if fields != ["morphology_umap_row_key", "channel", "relative_path", "sha256"]:
        raise ValueError("V4 crop manifest schema differs")
    pairs = [(row["morphology_umap_row_key"], row["channel"]) for row in crops]
    expected_pairs = [
        (cell_id, channel)
        for cell_id in review_ids
        for channel in ("brightfield", "dead", "nuclei")
    ]
    if sorted(pairs) != sorted(expected_pairs) or len(pairs) != len(set(pairs)):
        raise ValueError("V4 crop manifest does not exactly cover the review set")
    for row in crops:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("crops",):
            raise ValueError("Unsafe V4 crop path")
        crop = render / relative
        if not crop.is_file() or crop.is_symlink() or sha256(crop) != row["sha256"]:
            raise ValueError(f"V4 crop hash differs: {relative}")
    if sha256(render / "crop_manifest.tsv") != manifest.get("crop_aggregate_sha256"):
        raise ValueError("V4 crop aggregate hash differs")
    return manifest


def verify_existing(output: Path, inputs: dict[str, Any], implementation: str) -> None:
    manifest_path = output / "review_import_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("inputs") != inputs
        or manifest.get("implementation_sha256") != implementation
    ):
        raise ValueError("Existing V4 review import identity differs")
    observed = artifact_hashes(output, {"review_import_manifest.json"})
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 review import artifact set/hash differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project = args.project.expanduser().resolve(strict=True)
    if project.parent.parent.name != "projection_input":
        raise ValueError("V4 project must be under shadow/projection_input/<project>")
    shadow = project.parent.parent.parent
    if shadow.is_symlink() or not shadow.is_dir():
        raise ValueError("V4 shadow root is invalid")
    review_set = confine(args.review_set, shadow, "review set")
    review_manifest = confine(args.review_manifest, shadow, "review manifest")
    render = confine(args.render_dir, shadow, "render directory", directory=True)
    submission_path = confine(args.submission, shadow, "review submission")
    output = Path(os.path.realpath(args.output_dir.expanduser()))
    try:
        output.relative_to(shadow)
    except ValueError as error:
        raise ValueError("Review import output escapes the V4 shadow root") from error
    fields, review_rows = read_tsv(review_set)
    if "morphology_umap_row_key" not in fields or not review_rows:
        raise ValueError("V4 review set lacks stable IDs")
    review_ids = [row["morphology_umap_row_key"] for row in review_rows]
    if any(not value for value in review_ids) or len(review_ids) != len(set(review_ids)):
        raise ValueError("V4 review set has blank or duplicate stable IDs")
    selection_manifest = load_json(review_manifest)
    if (
        int(selection_manifest.get("row_count", -1)) != len(review_ids)
        or selection_manifest.get("stable_id_sha256") != stable_id_hash(review_ids)
    ):
        raise ValueError("V4 review selection identity differs")
    render_manifest = verify_render(render, review_ids)
    submission = load_json(submission_path)
    if (
        submission.get("schema_version") != SUBMISSION_SCHEMA
        or submission.get("identity") != render_manifest.get("identity")
    ):
        raise ValueError("V4 submission was not exported from this exact render")
    rows = submission.get("rows")
    if not isinstance(rows, list) or len(rows) != len(review_ids):
        raise ValueError("V4 submission row count differs from the review set")
    render_generation_id = str(
        render_manifest.get("identity", {}).get("render_generation_id", "")
    )
    expected_tokens = [
        review_token(render_generation_id, cell_id) for cell_id in review_ids
    ]
    if [row.get("review_token") for row in rows] != expected_tokens:
        raise ValueError("V4 submission must preserve the exact opaque review-token order")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"V4 submission row {index} is not an object")
        primary = str(row.get("primary_death_label", "")).strip()
        stage = str(row.get("death_stage_label", "")).strip()
        nuclear = str(row.get("nuclear_state_label", "")).strip()
        confidence = str(row.get("label_confidence", "")).strip()
        reviewer = no_controls(str(row.get("reviewer", "")).strip(), "reviewer")
        reviewed_at = require_timestamp(str(row.get("reviewed_at", "")))
        notes = no_controls(str(row.get("review_notes", "")).strip(), "review_notes")
        if primary not in PRIMARY or stage not in STAGES or nuclear not in NUCLEAR:
            raise ValueError(f"V4 submission row {index} has an unsupported label")
        if confidence not in CONFIDENCE or not reviewer:
            raise ValueError(f"V4 submission row {index} lacks human evidence")
        if primary == "dead" and stage == "not_applicable_non_dead":
            raise ValueError("Dead cells require a death-stage descriptor")
        if primary == "non_dead" and stage != "not_applicable_non_dead":
            raise ValueError("Non-dead cells require not_applicable_non_dead")
        if primary == "uncertain_or_unreviewable" and stage != "death_stage_indeterminate":
            raise ValueError("Uncertain cells require death_stage_indeterminate")
        if (primary == "uncertain_or_unreviewable" or confidence == "low") and not notes:
            raise ValueError("Uncertain or low-confidence reviews require notes")
        eligible = primary in {"dead", "non_dead"} and confidence in {"high", "medium"}
        normalized.append(
            {
                "cell_id": review_ids[index - 1],
                "primary_death_label": primary,
                "death_stage_label": stage,
                "nuclear_state_label": nuclear,
                "label_confidence": confidence,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "review_notes": notes,
                "training_eligible": "true" if eligible else "false",
            }
        )
    implementation = sha256(Path(__file__).resolve())
    inputs = {
        "project": sha256(project),
        "review_set": sha256(review_set),
        "review_manifest": sha256(review_manifest),
        "render_manifest": sha256(render / "exact_review_render_manifest.json"),
        "crop_manifest": sha256(render / "crop_manifest.tsv"),
        "submission": sha256(submission_path),
    }
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ValueError("Existing V4 review import is not a real directory")
        verify_existing(output, inputs, implementation)
        print(f"multimodal_cell_state_v4_review_import={output} verified_reuse=1")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        write_tsv(staging / "reviewed_labels.tsv", normalized)
        shutil.copyfile(submission_path, staging / "review_submission.json")
        artifacts = artifact_hashes(staging)
        counts = Counter(row["primary_death_label"] for row in normalized)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "inputs": inputs,
            "implementation_sha256": implementation,
            "row_count": len(normalized),
            "stable_id_sha256": stable_id_hash(review_ids),
            "primary_label_counts": dict(counts),
            "training_eligible_rows": sum(
                row["training_eligible"] == "true" for row in normalized
            ),
            "training_policy": "manual_three_channel_high_or_medium_confidence_dead_or_non_dead_only",
            "polygon_or_umap_label_authoritative": False,
            "output_file_sha256": artifacts,
        }
        (staging / "review_import_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v4_review_import={output}")
    print("review_import_status=COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import sys

        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
