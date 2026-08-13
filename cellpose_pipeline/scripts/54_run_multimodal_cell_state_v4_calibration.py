#!/usr/bin/env python3
"""Run the resumable direct-node V4 calibration to its anchor-review barrier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "multimodal_cell_state_v4_calibration_receipt_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--base-shadow-root", type=Path, required=True)
    parser.add_argument("--parent-broad-shadow-root", type=Path, required=True)
    parser.add_argument("--expanded-projection", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--container-identity", type=Path, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--expected-cell-count", type=int, default=32000)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a plain file: {path}")
    return value


def require_dir(path: Path, label: str) -> Path:
    value = path.expanduser().resolve(strict=True)
    if not value.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a real directory: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def run(command: list[str], label: str) -> str:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit {result.returncode}:\n{result.stdout}"
        )
    return result.stdout


def resolve_project_asset(project_path: Path, value: Any, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Base project lacks {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    return require_file(path, label)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    candidate = path.parent / f".{path.name}.tmp.{os.getpid()}"
    candidate.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if path.exists():
        if path.read_bytes() != candidate.read_bytes():
            candidate.unlink()
            raise ValueError(f"Existing calibration receipt differs: {path}")
        candidate.unlink()
    else:
        os.rename(candidate, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.expected_cell_count < 1:
        raise ValueError("Invalid V4 calibration concurrency/cell count")
    project_root = require_dir(args.project_root, "project root")
    base_root = require_dir(args.base_shadow_root, "base shadow root")
    broad_root = require_dir(args.parent_broad_shadow_root, "parent broad shadow root")
    expanded = require_dir(args.expanded_projection, "expanded projection audit")
    shadow = require_dir(args.shadow_root, "V4 shadow root")
    reference_root = require_dir(args.reference_root, "CPA reference root")
    base_project = require_file(args.base_project, "base project")
    dependency_lock = require_file(args.dependency_lock, "dependency lock")
    container_identity = require_file(args.container_identity, "container identity")
    if any(
        root == shadow or root in shadow.parents or shadow in root.parents
        for root in (base_root, broad_root)
    ):
        raise ValueError("V4 shadow root must be independent from frozen parent roots")

    scripts = project_root / "cellpose_pipeline" / "scripts"
    config = require_file(
        project_root
        / "cellpose_pipeline"
        / "configs"
        / "multimodal_cell_state_v4.json",
        "V4 config",
    )
    extractor = require_file(
        scripts / "48_extract_multimodal_cell_state_v4_features.py", "field extractor"
    )
    helper = require_file(
        scripts / "_shared" / "multimodal_cell_state_v4_features.py", "feature helper"
    )
    merger = require_file(
        scripts / "49_merge_multimodal_cell_state_v4_features.py", "feature merger"
    )
    projector = require_file(
        scripts / "50_build_multimodal_cell_state_v4_projection.R", "projection builder"
    )
    project_builder = require_file(
        scripts / "51_prepare_multimodal_cell_state_v4_project.py", "project builder"
    )
    audit_renderer = require_file(
        scripts / "52_render_multimodal_cell_state_v4_workspace.py", "workspace renderer"
    )
    blind_builder = require_file(
        scripts / "53_build_multimodal_cell_state_v4_anchor_review.py",
        "anchor-review builder",
    )
    exact_renderer = require_file(
        scripts / "55_render_multimodal_cell_state_v4_review.py",
        "three-channel review renderer",
    )
    cpa_stage = require_file(
        scripts / "20_run_cellphenotypeannotator_stage.py", "CPA stage runner"
    )
    implementation = require_file(Path(__file__), "calibration driver")

    base = load_json(base_project)
    cells_path = resolve_project_asset(
        base_project, base.get("cells_file"), "base cells"
    )
    fields, cells = read_tsv(cells_path)
    required = {"cell_id", "key", "well", "mask_label", "split"}
    if not required.issubset(fields) or len(cells) != args.expected_cell_count:
        raise ValueError("Base development cell contract differs")
    if any(row["split"] != "development" for row in cells):
        raise ValueError("Heldout cells entered V4 calibration")
    field_keys = sorted({(row["well"], row["key"]) for row in cells})
    feature_root = shadow / "workflow_status" / "multimodal_v4_field_features"
    feature_root.mkdir(parents=True, exist_ok=True)

    def extract_field(item: tuple[str, str]) -> tuple[str, str]:
        well, key = item
        record = (
            broad_root
            / "workflow_status"
            / "resolved_stage06_manifest"
            / "records"
            / well
            / f"{key}.json"
        )
        broad = (
            broad_root
            / "features"
            / "original"
            / "shards"
            / well
            / f"{key}__original_broad_phenotype_features.tsv"
        )
        output = feature_root / well / key
        command = [
            sys.executable,
            "-I",
            str(extractor),
            "--field-record",
            str(record),
            "--broad-feature-tsv",
            str(broad),
            "--selected-cells",
            str(cells_path),
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--overwrite",
        ]
        return key, run(command, f"V4 field extraction {key}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(extract_field, item): item for item in field_keys}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"v4_field_features={completed}/{len(futures)}", flush=True)

    merge_root = shadow / "workflow_status" / "multimodal_v4_feature_merge"
    run(
        [
            sys.executable,
            "-I",
            str(merger),
            "--project",
            str(base_project),
            "--feature-root",
            str(feature_root),
            "--feature-config",
            str(config),
            "--field-extractor",
            str(extractor),
            "--feature-helper",
            str(helper),
            "--output-dir",
            str(merge_root),
            "--expected-cell-count",
            str(args.expected_cell_count),
            "--overwrite",
        ],
        "V4 feature merge",
    )

    projection_root = shadow / "workflow_status" / "multimodal_v4_projection"
    run(
        [
            "Rscript",
            str(projector),
            "--cells",
            str(cells_path),
            "--features",
            str(merge_root / "features.tsv"),
            "--expanded-projection",
            str(expanded),
            "--config",
            str(config),
            "--output-dir",
            str(projection_root),
            "--overwrite",
        ],
        "V4 projection",
    )

    project_dir = shadow / "projection_input" / "multimodal_cell_state_v4"
    run(
        [
            sys.executable,
            "-I",
            str(project_builder),
            "--base-project",
            str(base_project),
            "--base-shadow-root",
            str(base_root),
            "--feature-merge",
            str(merge_root),
            "--projection",
            str(projection_root),
            "--shadow-root",
            str(shadow),
            "--output-dir",
            str(project_dir),
            "--expected-cell-count",
            str(args.expected_cell_count),
            "--overwrite",
        ],
        "V4 CPA project",
    )
    project_file = project_dir / "project.yml"
    for stage in ("validate", "umap", "annotate"):
        command = [
            sys.executable,
            "-I",
            str(cpa_stage),
            "--stage",
            stage,
            "--project",
            str(project_file),
            "--shadow-root",
            str(shadow),
            "--reference-root",
            str(reference_root),
            "--dependency-lock",
            str(dependency_lock),
            "--rscript",
            "Rscript",
        ]
        if stage == "validate":
            command.extend(("--validate-stage", "umap"))
        run(command, f"V4 CPA {stage}")

    annotation_htmls = sorted((project_dir / "runs").rglob("annotation.html"))
    if len(annotation_htmls) != 1:
        raise ValueError(
            f"V4 CPA must yield exactly one annotation generation, observed={len(annotation_htmls)}"
        )
    annotation_html = require_file(annotation_htmls[0], "V4 annotation HTML")

    audit_root = shadow / "morphology_audit"
    run(
        [
            sys.executable,
            "-I",
            str(audit_renderer),
            "--project",
            str(project_file),
            "--shadow-root",
            str(shadow),
            "--output-dir",
            str(audit_root),
            "--annotation-html",
            str(annotation_html),
            "--overwrite",
        ],
        "V4 morphology audit",
    )

    anchor_root = shadow / "human_review" / "anchor_seed1" / "selection"
    run(
        [
            sys.executable,
            "-I",
            str(blind_builder),
            "--project",
            str(project_file),
            "--shadow-root",
            str(shadow),
            "--output-dir",
            str(anchor_root),
            "--max-total",
            "300",
            "--overwrite",
        ],
        "V4 independent anchor-review selection",
    )
    render_root = shadow / "human_review" / "anchor_seed1" / "render"
    run(
        [
            sys.executable,
            "-I",
            str(exact_renderer),
            "--project",
            str(project_file),
            "--review-set",
            str(anchor_root / "anchor_review_set.tsv"),
            "--review-manifest",
            str(anchor_root / "anchor_review_manifest.json"),
            "--output-dir",
            str(render_root),
            "--padding",
            "12",
        ],
        "V4 three-channel anchor-review render",
    )

    decision = load_json(projection_root / "calibration_decision.json")
    anchor_manifest = load_json(anchor_root / "anchor_review_manifest.json")
    render_manifest = load_json(render_root / "exact_review_render_manifest.json")
    if decision.get("representation_gate") != "GO_TO_INDEPENDENT_ANCHOR_REVIEW":
        raise ValueError("V4 representation did not pass block-equalization gate")
    if (
        anchor_manifest.get("row_count") != 300
        or render_manifest.get("all_crops_available") is not True
    ):
        raise ValueError("V4 independent anchor-review barrier is incomplete")
    identity_value = load_json(container_identity)
    if identity_value.get("expected_sha256") != args.expected_container_sha256:
        raise ValueError("Container identity receipt differs from expected SIF SHA")
    receipt_path = (
        shadow
        / "workflow_status"
        / "MULTIMODAL_CELL_STATE_V4_CALIBRATION_COMPLETE.json"
    )
    completed_utc = (
        str(load_json(receipt_path).get("completed_utc", ""))
        if receipt_path.is_file() and not receipt_path.is_symlink()
        else datetime.now(timezone.utc).isoformat()
    )
    if not completed_utc:
        raise ValueError("Existing V4 calibration receipt lacks completed_utc")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "HUMAN_REVIEW_REQUIRED",
        "method_version": "sum159_multimodal_cell_state_v4",
        "completed_utc": completed_utc,
        "host": os.uname().nodename.split(".")[0],
        "execution_mode": "direct_test_no_slurm_no_gpu",
        "cell_count": args.expected_cell_count,
        "field_count": len(field_keys),
        "representation_gate": decision["representation_gate"],
        "primary_profile": decision["primary_profile"],
        "selected_run_id": decision["selected_run_id"],
        "balanced_dead_cell_island_required": False,
        "death_resolution_selection_status": "PENDING_INDEPENDENT_ANCHOR_SUBMISSION",
        "anchor_review_rows": anchor_manifest["row_count"],
        "anchor_review_umap_polygon_or_current_labels_displayed": False,
        "human_barrier": "independent_three_channel_anchor_submission_required",
        "human_workspace": str(render_root / "exact_review.html"),
        "annotation_workspace": str(audit_root / "annotation_workspace.html"),
        "human_submission_expected_path": str(
            render_root / "multimodal_v4_review_submission.json"
        ),
        "inputs": {
            "base_project": {
                "path": str(base_project),
                "sha256": sha256_file(base_project),
            },
            "base_cells": {"path": str(cells_path), "sha256": sha256_file(cells_path)},
            "expanded_projection": {
                "path": str(expanded),
                "sha256": sha256_file(expanded / "expanded_projection_manifest.json"),
            },
            "container_identity": {
                "path": str(container_identity),
                "sha256": sha256_file(container_identity),
            },
            "implementation": {
                "path": str(implementation),
                "sha256": sha256_file(implementation),
            },
        },
        "artifacts": {
            "feature_merge_manifest": {
                "path": str(merge_root / "feature_merge_manifest.json"),
                "sha256": sha256_file(merge_root / "feature_merge_manifest.json"),
            },
            "projection_manifest": {
                "path": str(projection_root / "projection_manifest.json"),
                "sha256": sha256_file(projection_root / "projection_manifest.json"),
            },
            "project_manifest": {
                "path": str(project_dir / "parent_import_manifest.json"),
                "sha256": sha256_file(project_dir / "parent_import_manifest.json"),
            },
            "morphology_audit_manifest": {
                "path": str(audit_root / "morphology_audit_manifest.json"),
                "sha256": sha256_file(audit_root / "morphology_audit_manifest.json"),
            },
            "anchor_review_manifest": {
                "path": str(anchor_root / "anchor_review_manifest.json"),
                "sha256": sha256_file(anchor_root / "anchor_review_manifest.json"),
            },
            "exact_review_render_manifest": {
                "path": str(render_root / "exact_review_render_manifest.json"),
                "sha256": sha256_file(
                    render_root / "exact_review_render_manifest.json"
                ),
            },
        },
        "blinding": {
            "dead_raw": "read_as_independent_measurement",
            "existing_dead_segmentation": "not_read",
            "combined_rgb": "not_read",
            "current_classification": "not_read",
            "trajectory": "not_read",
            "heldout": "not_read",
        },
    }
    write_json_atomic(receipt_path, receipt)
    print("multimodal_cell_state_v4_calibration_complete=1")
    print(f"shadow_root={shadow}")
    print(f"human_workspace={receipt['human_workspace']}")
    print(f"human_submission_expected_path={receipt['human_submission_expected_path']}")
    print(f"human_barrier={receipt['human_barrier']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
