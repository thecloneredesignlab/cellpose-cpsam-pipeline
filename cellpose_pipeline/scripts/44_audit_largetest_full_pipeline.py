#!/usr/bin/env python3
"""Audit a complete direct-grab large-test production run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
PROFILES = ("Brightfield", "Combined", "Dead", "Nuclei")
FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "Segmentation failures:",
    "shape_strict production stage failed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit direct-grab large-test pipeline outputs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-per-profile", type=int, default=20)
    parser.add_argument("--pipeline-log", type=Path)
    return parser.parse_args()


def image_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())


def glob_count(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob(pattern) if path.is_file())


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    run_root = args.run_root.resolve()
    expected = int(args.expected_per_profile)
    checks: dict[str, bool] = {}
    inventory: dict[str, Any] = {"raw": {}, "segmentation": {}}

    for profile in PROFILES:
        raw_count = image_count(input_root / profile)
        mask_count = glob_count(run_root / profile / "segmentations", "*_cp_masks.tif")
        metadata_count = glob_count(
            run_root / profile / "metadata", "*_segmentation_metadata.json"
        )
        overlay_count = glob_count(
            run_root / profile / "qc" / "segmentation_overlays",
            "*_segmentation_overlay.png",
        )
        inventory["raw"][profile] = raw_count
        inventory["segmentation"][profile] = {
            "masks": mask_count,
            "metadata": metadata_count,
            "overlays": overlay_count,
        }
        checks[f"raw_{profile}_count"] = raw_count == expected
        checks[f"mask_{profile}_count"] = mask_count == expected
        checks[f"metadata_{profile}_count"] = metadata_count == expected
        checks[f"overlay_{profile}_count"] = overlay_count == expected

    core_count = glob_count(run_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif")
    inventory["nucleus_core_masks"] = core_count
    checks["nucleus_core_count"] = core_count == expected

    density_path = run_root / "qc" / "density_calls.csv"
    density_rows = csv_rows(density_path)
    high_density_keys = sorted(
        row.get("key", "") for row in density_rows if truthy(row.get("high_density", ""))
    )
    inventory["density"] = {
        "path": str(density_path),
        "rows": len(density_rows),
        "high_density_count": len(high_density_keys),
        "high_density_keys": high_density_keys,
    }
    checks["density_row_count"] = len(density_rows) == expected
    checks["density_keys_unique"] = len({row.get("key", "") for row in density_rows}) == expected

    fusion_root = run_root / "classification_fusion"
    fusion_summary_path = fusion_root / "summaries" / "cell_count_summary.csv"
    fusion_rows = csv_rows(fusion_summary_path)
    per_field_summaries = [
        path
        for path in (fusion_root / "summaries").glob("*_summary.csv")
        if path.name != "cell_count_summary.csv"
    ] if (fusion_root / "summaries").is_dir() else []
    fusion_failure_rows = csv_rows(fusion_root / "failures.csv")
    per_field_failure_count = glob_count(fusion_root / "failures", "*_failure.csv")
    inventory["fusion"] = {
        "summary": str(fusion_summary_path),
        "summary_rows": len(fusion_rows),
        "per_field_summaries": len(per_field_summaries),
        "aggregate_failure_rows": len(fusion_failure_rows),
        "per_field_failure_files": per_field_failure_count,
    }
    checks["fusion_summary_row_count"] = len(fusion_rows) == expected
    checks["fusion_per_field_summary_count"] = len(per_field_summaries) == expected
    checks["fusion_keys_unique"] = len({row.get("key", "") for row in fusion_rows}) == expected
    checks["fusion_no_failures"] = not fusion_failure_rows and per_field_failure_count == 0

    shape_root = run_root / "shape_strict"
    shape_summary_path = shape_root / "shape_strict_summary.json"
    shape_summary: dict[str, Any] = {}
    if shape_summary_path.is_file():
        shape_summary = json.loads(shape_summary_path.read_text())
    shape_field_rows = csv_rows(shape_root / "field_summary.csv")
    shape_event_rows = csv_rows(shape_root / "split_events.csv")
    shape_qc_rows = csv_rows(shape_root / "qc" / "qc_manifest.csv")
    shape_mask_count = glob_count(shape_root / "Nuclei" / "segmentations", "*_cp_masks.tif")
    shape_core_count = glob_count(
        shape_root / "Nuclei" / "nucleus_core_seeds", "*_core_masks.tif"
    )
    shape_validation = shape_summary.get("validation", {})
    inventory["shape_strict"] = {
        "summary": str(shape_summary_path),
        "fields": len(shape_field_rows),
        "split_events": len(shape_event_rows),
        "qc_rows": len(shape_qc_rows),
        "extent_masks": shape_mask_count,
        "core_masks": shape_core_count,
        "summary_payload": shape_summary,
    }
    checks["shape_success_marker"] = (shape_root / "_SUCCESS").is_file()
    checks["shape_summary_exists"] = bool(shape_summary)
    checks["shape_field_count"] = len(shape_field_rows) == expected
    checks["shape_extent_count"] = shape_mask_count == expected
    checks["shape_core_count"] = shape_core_count == expected
    checks["shape_event_count_matches_summary"] = len(shape_event_rows) == int(
        shape_summary.get("n_split_parents", -1)
    )
    checks["shape_qc_count_matches_summary"] = len(shape_qc_rows) == int(
        shape_summary.get("n_qc_fields", -1)
    )
    checks["shape_internal_validation"] = bool(shape_validation) and all(
        bool(value) for value in shape_validation.values()
    )

    pipeline_log = args.pipeline_log.resolve() if args.pipeline_log else run_root / "logs" / "pipeline.log"
    log_text = pipeline_log.read_text(errors="replace") if pipeline_log.is_file() else ""
    fatal_log_hits = [pattern for pattern in FATAL_LOG_PATTERNS if pattern in log_text]
    inventory["pipeline_log"] = {
        "path": str(pipeline_log),
        "exists": pipeline_log.is_file(),
        "fatal_patterns": fatal_log_hits,
    }
    checks["pipeline_log_exists"] = pipeline_log.is_file()
    checks["pipeline_log_no_fatal_patterns"] = not fatal_log_hits
    checks["stage_timings_exists"] = (run_root / "status" / "stage_timings.tsv").is_file()

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "input_root": str(input_root),
        "run_root": str(run_root),
        "expected_per_profile": expected,
        "inventory": inventory,
        "checks": checks,
        "all_checks_pass": not failed_checks,
        "failed_checks": failed_checks,
    }
    write_json(run_root / "final_audit.json", payload)
    lines = [
        "# Large-test direct-grab final audit",
        "",
        f"- run root: `{run_root}`",
        f"- expected fields per profile: {expected}",
        f"- high-density fields: {len(high_density_keys)}",
        f"- fusion fields: {len(fusion_rows)}",
        f"- shape_strict splits: {len(shape_event_rows)}",
        f"- all checks pass: {not failed_checks}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items())
    if failed_checks:
        lines.extend(("", "## Failed checks", ""))
        lines.extend(f"- `{name}`" for name in failed_checks)
    (run_root / "final_audit.md").write_text("\n".join(lines) + "\n")
    print(f"final_audit={run_root / 'final_audit.json'}", flush=True)
    print(f"all_checks_pass={not failed_checks}", flush=True)
    if failed_checks:
        print(f"failed_checks={','.join(failed_checks)}", flush=True)
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
