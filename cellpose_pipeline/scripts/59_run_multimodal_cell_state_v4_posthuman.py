#!/usr/bin/env python3
"""Advance V4 across one explicitly completed human barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "multimodal_cell_state_v4_posthuman_receipt_v1"
ACTIONS = ("post-anchor", "post-region", "post-review")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a plain file: {path}")
    return value


def directory(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_dir() or path.is_symlink():
        raise ValueError(f"{label} is not a real directory: {path}")
    return value


def inside(path: Path, root: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if value == root or root not in value.parents or path.is_symlink():
        raise ValueError(f"{label} escapes the V4 shadow root: {value}")
    return value


def run(command: list[str], label: str) -> str:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{result.stdout}")
    return result.stdout


def one(root: Path, name: str, label: str) -> Path:
    values = sorted(path for path in root.rglob(name) if path.is_file() and not path.is_symlink())
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {label}; observed={len(values)}")
    return values[0]


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Existing V4 posthuman receipt differs: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)


def cpa(
    script: Path, stage: str, project: Path, shadow: Path, reference: Path,
    lock: Path, *, submission: Path | None = None,
) -> None:
    command = [
        sys.executable, "-I", str(script), "--stage", stage, "--project", str(project),
        "--shadow-root", str(shadow), "--reference-root", str(reference),
        "--dependency-lock", str(lock), "--rscript", "Rscript",
    ]
    if stage == "validate":
        command.extend(("--validate-stage", "umap"))
    if submission is not None:
        command.extend(("--submission", str(submission)))
    run(command, f"V4 CPA {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shadow = directory(args.shadow_root, "V4 shadow root")
    project_root = directory(args.project_root, "project root")
    reference = directory(args.reference_root, "CPA reference root")
    lock = file(args.dependency_lock, "dependency lock")
    submission = inside(file(args.submission, "human submission"), shadow, "human submission")
    scripts = project_root / "cellpose_pipeline/scripts"
    paths = {
        "cpa": file(scripts / "20_run_cellphenotypeannotator_stage.py", "CPA runner"),
        "workspace": file(scripts / "52_render_multimodal_cell_state_v4_workspace.py", "V4 workspace renderer"),
        "review": file(scripts / "55_render_multimodal_cell_state_v4_review.py", "V4 review renderer"),
        "select_umap": file(scripts / "56_select_multimodal_cell_state_v4_death_resolution.py", "V4 death-UMAP selector"),
        "select_review": file(scripts / "57_build_multimodal_cell_state_v4_broad_region_review.py", "V4 broad-review selector"),
        "import_review": file(scripts / "58_import_multimodal_cell_state_v4_review.py", "V4 review importer"),
    }
    balanced = inside(
        file(shadow / "projection_input/multimodal_cell_state_v4/project.yml", "balanced V4 project"),
        shadow, "balanced V4 project",
    )
    projection = inside(
        directory(shadow / "workflow_status/multimodal_v4_projection", "V4 projection"),
        shadow, "V4 projection",
    )
    evidence: list[Path] = [submission]
    if args.action == "post-anchor":
        selection = shadow / "human_review/anchor_seed1/selection"
        render = shadow / "human_review/anchor_seed1/render"
        imported = shadow / "human_review/anchor_seed1/import"
        run([
            sys.executable, "-I", str(paths["import_review"]), "--project", str(balanced),
            "--review-set", str(selection / "anchor_review_set.tsv"),
            "--review-manifest", str(selection / "anchor_review_manifest.json"),
            "--render-dir", str(render), "--submission", str(submission),
            "--output-dir", str(imported),
        ], "V4 anchor review import")
        death_dir = shadow / "projection_input/multimodal_cell_state_v4_death_resolution"
        run([
            sys.executable, "-I", str(paths["select_umap"]),
            "--balanced-project", str(balanced), "--projection-root", str(projection),
            "--anchor-import-dir", str(imported), "--shadow-root", str(shadow),
            "--output-dir", str(death_dir), "--overwrite",
        ], "V4 death-resolution UMAP selection")
        death_project = death_dir / "project.yml"
        for stage in ("validate", "umap", "annotate"):
            cpa(paths["cpa"], stage, death_project, shadow, reference, lock)
        annotation_html = one(death_dir / "runs", "annotation.html", "death-resolution annotation HTML")
        workspace = shadow / "death_resolution_morphology_audit"
        run([
            sys.executable, "-I", str(paths["workspace"]), "--project", str(death_project),
            "--shadow-root", str(shadow), "--output-dir", str(workspace),
            "--annotation-html", str(annotation_html), "--overwrite",
        ], "V4 death-resolution morphology workspace")
        expected = annotation_html.parent / "region_submission.json"
        barrier = "broad_dead_region_submission_required"
        evidence.extend((imported / "review_import_manifest.json", death_dir / "death_resolution_manifest.json", workspace / "morphology_audit_manifest.json"))
        workspace_path = workspace / "annotation_workspace.html"
    elif args.action == "post-region":
        death_dir = shadow / "projection_input/multimodal_cell_state_v4_death_resolution"
        death_project = file(death_dir / "project.yml", "death-resolution project")
        cpa(paths["cpa"], "annotation-import", death_project, shadow, reference, lock, submission=submission)
        annotation_import = one(death_dir / "runs", "annotation_import_manifest.json", "death-region annotation import").parent
        selection = shadow / "human_review/broad_region/selection"
        run([
            sys.executable, "-I", str(paths["select_review"]), "--project", str(death_project),
            "--annotation-import-dir", str(annotation_import), "--shadow-root", str(shadow),
            "--output-dir", str(selection), "--max-total", "500", "--max-per-well", "8",
            "--overwrite",
        ], "V4 broad-region review selection")
        render = shadow / "human_review/broad_region/render"
        run([
            sys.executable, "-I", str(paths["review"]), "--project", str(death_project),
            "--review-set", str(selection / "broad_region_review_set.tsv"),
            "--review-manifest", str(selection / "broad_region_review_manifest.json"),
            "--output-dir", str(render), "--padding", "12",
        ], "V4 broad-region three-channel review render")
        expected = render / "multimodal_v4_review_submission.json"
        barrier = "broad_region_three_channel_review_submission_required"
        evidence.extend((annotation_import / "annotation_import_manifest.json", selection / "broad_region_review_manifest.json", render / "exact_review_render_manifest.json"))
        workspace_path = render / "exact_review.html"
    else:
        death_project = file(
            shadow / "projection_input/multimodal_cell_state_v4_death_resolution/project.yml",
            "death-resolution project",
        )
        selection = shadow / "human_review/broad_region/selection"
        render = shadow / "human_review/broad_region/render"
        imported = shadow / "human_review/broad_region/import"
        run([
            sys.executable, "-I", str(paths["import_review"]), "--project", str(death_project),
            "--review-set", str(selection / "broad_region_review_set.tsv"),
            "--review-manifest", str(selection / "broad_region_review_manifest.json"),
            "--render-dir", str(render), "--submission", str(submission),
            "--output-dir", str(imported),
        ], "V4 broad-region review import")
        expected = imported / "reviewed_labels.tsv"
        barrier = "model_training_requires_separate_technical_acceptance"
        evidence.append(imported / "review_import_manifest.json")
        workspace_path = expected
    for path in evidence:
        file(path, "V4 posthuman evidence")
    receipt_path = shadow / "workflow_status" / f"MULTIMODAL_CELL_STATE_V4_{args.action.upper().replace('-', '_')}_COMPLETE.json"
    existing_completed = ""
    if receipt_path.is_file() and not receipt_path.is_symlink():
        existing_completed = str(
            json.loads(receipt_path.read_text(encoding="utf-8")).get(
                "completed_utc", ""
            )
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "action": args.action,
        "completed_utc": existing_completed or datetime.now(timezone.utc).isoformat(),
        "human_barrier": barrier,
        "workspace": str(workspace_path),
        "expected_next_input": str(expected),
        "input_submission": {"path": str(submission), "sha256": sha256(submission)},
        "evidence": [{"path": str(path), "sha256": sha256(path)} for path in evidence[1:]],
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    write_receipt(receipt_path, receipt)
    print(f"v4_posthuman_action={args.action}")
    print(f"workspace={workspace_path}")
    print(f"expected_next_input={expected}")
    print(f"human_barrier={barrier}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
