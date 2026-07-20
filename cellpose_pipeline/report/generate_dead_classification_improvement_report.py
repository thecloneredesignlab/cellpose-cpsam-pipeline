#!/usr/bin/env python3
"""Build the self-contained SUM159 d0 dead-classification improvement report.

The report is derived from the English comparison Markdown, cohort-wide audit
CSVs, and the frozen QC overlays generated during iterative classifier tuning.
Scientific inputs are read-only.  The final output is a portable HTML report
packaged with the same canonical artifact builder, navigation, and high-
resolution lightbox used by ``generate_results_analysis_report.py``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_main_report_module() -> Any:
    path = Path(__file__).with_name("generate_results_analysis_report.py")
    spec = importlib.util.spec_from_file_location("_sum159_main_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the main report generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAIN_REPORT = load_main_report_module()
if MAIN_REPORT._IMAGE_IMPORT_ERROR is not None:
    raise RuntimeError(
        "Report generation requires numpy, tifffile, and Pillow; use the configured CellPose environment"
    ) from MAIN_REPORT._IMAGE_IMPORT_ERROR

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def load_current_run_report_module() -> Any:
    path = Path(__file__).with_name("dead_classification_current_run_report.py")
    spec = importlib.util.spec_from_file_location("_dead_classification_current_run_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load current-run report support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CURRENT_RUN_REPORT = load_current_run_report_module()


REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEPOINT_HELPER_PATH = REPO_ROOT / "cellpose_pipeline" / "scripts" / "_shared" / "timepoint_selection.py"
TIMEPOINT_HELPER_SPEC = importlib.util.spec_from_file_location(
    "timepoint_selection_local",
    TIMEPOINT_HELPER_PATH,
)
if TIMEPOINT_HELPER_SPEC is None or TIMEPOINT_HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load timepoint selection helpers: {TIMEPOINT_HELPER_PATH}")
TIMEPOINT_HELPER = importlib.util.module_from_spec(TIMEPOINT_HELPER_SPEC)
sys.modules[TIMEPOINT_HELPER_SPEC.name] = TIMEPOINT_HELPER
TIMEPOINT_HELPER_SPEC.loader.exec_module(TIMEPOINT_HELPER)
extract_key_and_timepoint = TIMEPOINT_HELPER.extract_key_and_timepoint
normalize_timepoint = TIMEPOINT_HELPER.normalize_timepoint

DEFAULT_AUDIT_ROOT = REPO_ROOT / "results" / "dead_d0_classification_audit"
DEFAULT_METHODS_MD = Path(__file__).with_name("DEAD_CLASSIFICATION_IMPROVEMENT_COMPARISON.md")
DEFAULT_ARTIFACT_JSON = DEFAULT_AUDIT_ROOT / "DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.artifact.json"
DEFAULT_OUTPUT_HTML = DEFAULT_AUDIT_ROOT / "DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.html"
DEFAULT_BUILD_RECEIPT = DEFAULT_AUDIT_ROOT / "DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.build.json"

MAX_ARTIFACT_BYTES = MAIN_REPORT.MAX_ARTIFACT_BYTES
SOURCE_PANEL_SIZE = (1408, 1040)
CROP_SIZE = 260
PANEL_BODY_SIZE = 500
PANEL_HEADER_HEIGHT = 58
PANEL_WIDTH = PANEL_BODY_SIZE
PANEL_HEIGHT = PANEL_BODY_SIZE + PANEL_HEADER_HEIGHT
FIGURE_MARGIN = 36
FIGURE_GAP = 18
FIGURE_TITLE_HEIGHT = 76
FIGURE_NOTE_HEIGHT = 82

BG = (18, 20, 24)
PANEL_BG = (29, 32, 38)
TEXT = (245, 247, 250)
MUTED = (184, 191, 202)
YELLOW = (255, 220, 55)
BLUE = (70, 145, 245)
ORANGE = (232, 139, 57)

PIE_CATEGORY_COLORS = {
    "same_cell": (72, 132, 214),
    "cell_associated_dead_cell": (72, 132, 214),
    "overlapping_live_dead_multi_nucleus": (232, 139, 57),
    "supplemental_overlapping_live_dead_multi_nucleus": (232, 139, 57),
    "live_with_death_signal": (190, 104, 178),
    "supplemental_live_with_death_signal": (190, 104, 178),
    "dead_only": (218, 181, 70),
    "supplemental_dead_only": (218, 181, 70),
    "adjacent_or_overlapping_dead_uncertain": (135, 145, 160),
    "supplemental_adjacent_or_overlapping_dead_uncertain": (135, 145, 160),
    "merged_multiple_objects": (116, 154, 104),
    "supplemental_merged_multiple_objects": (116, 154, 104),
}

SENTINELS = (
    {
        "token": "e2",
        "key": "E2_1_00d00h00m",
        "combined_id": 32,
        "dead_id": 2,
        "short_title": "E2: live-cell false-positive protection",
    },
    {
        "token": "f5",
        "key": "F5_1_00d00h00m",
        "combined_id": 83,
        "dead_id": 6,
        "short_title": "F5: multi-nucleus live/death overlap",
    },
    {
        "token": "h9",
        "key": "H9_4_00d00h00m",
        "combined_id": 113,
        "dead_id": 18,
        "short_title": "H9: one-nucleus live cell with death signal",
    },
)

EXPANDED_CASES_PER_GROUP = 6
LIVE_DEAD_OVERLAP_RELATIONS = {
    "adjacent_dead",
    "overlapping_live_dead_multi_nucleus",
    "live_with_death_signal",
    "adjacent_or_overlapping_dead_uncertain",
}
GALLERY_PANEL_WIDTH = 230
GALLERY_PANEL_HEIGHT = 257
GALLERY_PANEL_GAP = 8
GALLERY_CASE_GAP = 16
GALLERY_CASE_HEADER = 43
GALLERY_CASE_FOOTER = 38
GALLERY_COLUMNS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument(
        "--report-mode",
        choices=("auto", "current-run", "historical-comparison"),
        default="auto",
        help=(
            "Select the report contract explicitly. 'auto' preserves legacy layout detection; "
            "'historical-comparison' can combine current HPC outputs with a frozen historical reference."
        ),
    )
    parser.add_argument(
        "--historical-reference-root",
        type=Path,
        help=(
            "Frozen context-aware/strong-direct reference bundle used by historical-comparison mode "
            "when --audit-root is a current-run HPC result."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Complete raw-image root containing Brightfield, Combined, Dead, and Nuclei directories.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        help="Complete pipeline result root; paired with --input-root for source-inventory validation.",
    )
    parser.add_argument(
        "--timepoint",
        default="d0",
        help="Exact source timepoint to validate when complete roots are supplied (default: d0).",
    )
    parser.add_argument("--methods-md", type=Path, default=DEFAULT_METHODS_MD)
    parser.add_argument("--artifact-json", type=Path, default=DEFAULT_ARTIFACT_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--build-receipt", type=Path, default=DEFAULT_BUILD_RECEIPT)
    parser.add_argument(
        "--plugin-root",
        type=Path,
        help="Data Analytics plugin root. If omitted, the installed plugin cache is searched.",
    )
    parser.add_argument(
        "--debug-figure-dir",
        type=Path,
        help="Optional directory for review copies of the high-resolution QC figures.",
    )
    parser.add_argument(
        "--high-resolution-images-json",
        type=Path,
        help="Optional temporary high-resolution image sidecar; the final HTML remains self-contained.",
    )
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    return MAIN_REPORT.require_file(path)


def require_dir(path: Path) -> Path:
    return MAIN_REPORT.require_dir(path)


HISTORICAL_CURRENT_LINKS = {
    "object_aware_all_d0_v4": "classification_original",
    "object_aware_all_d0_v4_nucleated_only": "classification_nucleated_only",
    "object_aware_automated_reference_audit_v4": "automated_audit",
    "dead_object_detection_stress_v2": "detector_stress",
    "final_annotation_summary.csv": "annotations/final_annotation_summary.csv",
    "final_multilevel_annotations.csv": "annotations/final_multilevel_annotations.csv",
}
HISTORICAL_FROZEN_LINKS = {
    "context_aware_all_d0": "context_aware_all_d0",
    "strong_direct_all_d0": "strong_direct_all_d0",
    "automated_reference_audit": "automated_reference_audit",
    "automated_reference_audit_final": "automated_reference_audit_final",
    "qc": "qc",
    "d0_branch_before_after_metrics.csv": "d0_branch_before_after_metrics.csv",
    "debugging_stage_provenance.json": "debugging_stage_provenance.json",
}


def ensure_relative_symlink(link: Path, target: Path) -> None:
    target = require_file(target) if target.is_file() else require_dir(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        existing = (link.parent / os.readlink(link)).resolve()
        if existing == target:
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Historical report input path exists and is not a symlink: {link}")
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())


def prepare_historical_comparison_root(current_root: Path, reference_root: Path) -> tuple[Path, Path]:
    current_root = require_dir(current_root)
    reference_root = require_dir(reference_root)
    report_root = current_root / "report_inputs" / "historical_comparison"
    report_root.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for name, relative in HISTORICAL_CURRENT_LINKS.items():
        target = current_root / relative
        ensure_relative_symlink(report_root / name, target)
        links[name] = str(target)
    for name, relative in HISTORICAL_FROZEN_LINKS.items():
        target = reference_root / relative
        ensure_relative_symlink(report_root / name, target)
        links[name] = str(target)
    input_map = report_root / "REPORT_INPUT_MAP.json"
    MAIN_REPORT.atomic_write_json(
        input_map,
        {
            "report_mode": "historical-comparison",
            "current_run_root": str(current_root),
            "historical_reference_root": str(reference_root),
            "links": links,
        },
    )
    return report_root, input_map


def validate_source_inventory(
    input_root: Path,
    result_root: Path,
    timepoint: str,
) -> dict[str, Any]:
    """Validate one selected timepoint within complete raw-image and result roots."""

    input_root = require_dir(input_root)
    result_root = require_dir(result_root)
    sources = {
        "raw_brightfield": (input_root / "Brightfield", None),
        "raw_combined": (input_root / "Combined", None),
        "raw_dead": (input_root / "Dead", None),
        "raw_nuclei": (input_root / "Nuclei", None),
        "mask_brightfield": (result_root / "Brightfield" / "segmentations", "_cp_masks.tif"),
        "mask_combined": (result_root / "Combined" / "segmentations", "_cp_masks.tif"),
        "mask_dead": (result_root / "Dead" / "segmentations", "_cp_masks.tif"),
        "mask_nuclei_extent": (result_root / "Nuclei" / "segmentations", "_cp_masks.tif"),
        "mask_nuclei_core": (result_root / "Nuclei" / "nucleus_core_seeds", "_core_masks.tif"),
        "mask_nucleated_brightfield": (
            result_root / "nucleated_only" / "Brightfield" / "segmentations",
            "_cp_masks.tif",
        ),
        "mask_nucleated_combined": (
            result_root / "nucleated_only" / "Combined" / "segmentations",
            "_cp_masks.tif",
        ),
    }
    selected: dict[str, set[str]] = {}
    for label, (directory, suffix) in sources.items():
        require_dir(directory)
        keys: set[str] = set()
        for path in directory.iterdir():
            if not path.is_file() or (suffix is not None and not path.name.endswith(suffix)):
                continue
            try:
                key, observed = extract_key_and_timepoint(path)
            except ValueError:
                continue
            if observed != timepoint:
                continue
            if key in keys:
                raise RuntimeError(f"Duplicate {label} key at {timepoint}: {key}")
            keys.add(key)
        if not keys:
            raise RuntimeError(f"No {timepoint} files found for {label} under {directory}")
        selected[label] = keys

    expected = selected["raw_combined"]
    mismatches: list[str] = []
    for label, keys in selected.items():
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            mismatches.append(f"{label}: missing={missing[:5]} extra={extra[:5]}")
    if mismatches:
        raise RuntimeError("Selected source key mismatch: " + "; ".join(mismatches))
    return {
        "input_root": str(input_root),
        "result_root": str(result_root),
        "timepoint": timepoint,
        "selected_field_count": len(expected),
        "source_counts": {label: len(keys) for label, keys in selected.items()},
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def integer(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing integer field {field}")
    return int(float(value))


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing numeric field {field}")
    return float(value)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def one_row(rows: list[dict[str, str]], field: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(field) == value]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row with {field}={value!r}, found {len(matches)}")
    return matches[0]


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file matching {directory / pattern}, found {len(matches)}")
    return require_file(matches[0])


def discover_plugin_root() -> Path:
    base = Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    candidates = sorted(
        (path for path in base.glob("*") if (path / "package.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No installed Data Analytics plugin was found; pass --plugin-root explicitly"
        )
    return candidates[0]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_image(path: Path) -> Image.Image:
    with Image.open(require_file(path)) as image:
        return image.convert("RGB")


def crop_panel(
    image: Image.Image,
    panel_column: int,
    panel_row: int,
    center_x: float,
    center_y: float,
    label: str,
    source_origin_x: int | None = None,
    source_origin_y: int | None = None,
) -> Image.Image:
    panel_width, panel_height = SOURCE_PANEL_SIZE
    panel_left = panel_column * panel_width if source_origin_x is None else source_origin_x
    panel_top = panel_row * panel_height if source_origin_y is None else source_origin_y
    expected_width = panel_left + panel_width
    expected_height = panel_top + panel_height
    if image.width < expected_width or image.height < expected_height:
        raise ValueError(
            f"Overlay is smaller than its declared panel grid: {image.size}, needs {expected_width}x{expected_height}"
        )

    left = int(round(panel_left + center_x - CROP_SIZE / 2))
    top = int(round(panel_top + center_y - CROP_SIZE / 2))
    left = max(panel_left, min(left, panel_left + panel_width - CROP_SIZE))
    top = max(panel_top, min(top, panel_top + panel_height - CROP_SIZE))
    body = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    body = body.resize((PANEL_BODY_SIZE, PANEL_BODY_SIZE), Image.Resampling.LANCZOS)

    target_x = (panel_left + center_x - left) * PANEL_BODY_SIZE / CROP_SIZE
    target_y = (panel_top + center_y - top) * PANEL_BODY_SIZE / CROP_SIZE
    draw = ImageDraw.Draw(body)
    radius = 25
    draw.ellipse(
        (target_x - radius, target_y - radius, target_x + radius, target_y + radius),
        outline=YELLOW,
        width=5,
    )
    draw.line((target_x - 34, target_y, target_x - 18, target_y), fill=YELLOW, width=4)
    draw.line((target_x + 18, target_y, target_x + 34, target_y), fill=YELLOW, width=4)
    draw.line((target_x, target_y - 34, target_x, target_y - 18), fill=YELLOW, width=4)
    draw.line((target_x, target_y + 18, target_x, target_y + 34), fill=YELLOW, width=4)

    tile = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), PANEL_BG)
    tile.paste(body, (0, PANEL_HEADER_HEIGHT))
    header = ImageDraw.Draw(tile)
    header.text((18, 15), label, fill=TEXT, font=font(25, bold=True))
    return tile


def stage_overlay(audit_root: Path, stage: str, key: str) -> Image.Image:
    path = find_one(audit_root / stage / "qc" / "label_overlays", f"*{key}*_state_overlay.png")
    return read_image(path)


def dead_object_overlay(audit_root: Path, key: str) -> Image.Image:
    path = find_one(
        audit_root / "object_aware_all_d0_v4" / "qc" / "dead_object_overlays",
        f"*{key}*_dead_object_overlay.png",
    )
    return read_image(path)


def overlap_state_overlay(audit_root: Path, key: str) -> Image.Image:
    path = find_one(
        audit_root / "object_aware_all_d0_v4" / "qc" / "overlap_state_overlays",
        f"*{key}*_overlap_state_overlay.png",
    )
    return read_image(path)


def dead_object_row(audit_root: Path, key: str, dead_id: int) -> dict[str, str]:
    path = find_one(
        audit_root / "object_aware_all_d0_v4" / "dead_objects",
        f"*{key}*_dead_object_features.csv",
    )
    return one_row(read_rows(path), "dead_mask_id", str(dead_id))


def prediction_row(audit_root: Path, stage: str, key: str, combined_id: int) -> dict[str, str]:
    path = find_one(audit_root / stage / "predictions", f"*{key}*_per_cell_predictions.csv")
    return one_row(read_rows(path), "mask_id", str(combined_id))


def e2_original_context_overlay(audit_root: Path) -> Image.Image:
    return read_image(require_file(audit_root / "qc" / "E2_1_00d00h00m_before_after_overlay.png"))


def compose_sentinel_figure(audit_root: Path, sentinel: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    key = sentinel["key"]
    combined_id = int(sentinel["combined_id"])
    dead_id = int(sentinel["dead_id"])
    object_row = dead_object_row(audit_root, key, dead_id)
    center_x = number(object_row, "centroid_x")
    center_y = number(object_row, "centroid_y")

    object_overlay = dead_object_overlay(audit_root, key)
    overlap_overlay = overlap_state_overlay(audit_root, key)
    final_overlay = stage_overlay(audit_root, "object_aware_all_d0_v4", key)
    context_overlay = stage_overlay(audit_root, "context_aware_all_d0", key)
    strong_overlay = stage_overlay(audit_root, "strong_direct_all_d0", key)

    if sentinel["token"] == "e2":
        baseline_context = e2_original_context_overlay(audit_root)
        tiles = [
            crop_panel(object_overlay, 0, 0, center_x, center_y, "A  Combined RGB"),
            crop_panel(
                baseline_context,
                1,
                0,
                center_x,
                center_y,
                "B  Original classification",
                source_origin_y=25,
            ),
            crop_panel(
                baseline_context,
                1,
                1,
                center_x,
                center_y,
                "C  Context-aware repair",
                source_origin_y=1090,
            ),
            crop_panel(object_overlay, 1, 0, center_x, center_y, "D  Raw Dead channel"),
            crop_panel(final_overlay, 1, 0, center_x, center_y, "E  Corrected cell state"),
            crop_panel(object_overlay, 2, 0, center_x, center_y, "F  Final object relation"),
        ]
    else:
        tiles = [
            crop_panel(overlap_overlay, 0, 0, center_x, center_y, "A  Combined RGB"),
            crop_panel(strong_overlay, 1, 0, center_x, center_y, "B  Previous threshold-only"),
            crop_panel(context_overlay, 1, 0, center_x, center_y, "C  Context-aware intermediate"),
            crop_panel(overlap_overlay, 2, 0, center_x, center_y, "D  Raw Dead + confirmed mask"),
            crop_panel(overlap_overlay, 1, 0, center_x, center_y, "E  Nuclei evidence"),
            crop_panel(overlap_overlay, 4, 0, center_x, center_y, "F  Final overlapping masks"),
        ]

    width = FIGURE_MARGIN * 2 + PANEL_WIDTH * 3 + FIGURE_GAP * 2
    height = FIGURE_TITLE_HEIGHT + PANEL_HEIGHT * 2 + FIGURE_GAP + FIGURE_NOTE_HEIGHT
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (FIGURE_MARGIN, 20),
        f"{sentinel['short_title']}  |  cell {combined_id}, Dead mask {dead_id}",
        fill=TEXT,
        font=font(31, bold=True),
    )
    for index, tile in enumerate(tiles):
        row = index // 3
        column = index % 3
        x = FIGURE_MARGIN + column * (PANEL_WIDTH + FIGURE_GAP)
        y = FIGURE_TITLE_HEIGHT + row * (PANEL_HEIGHT + FIGURE_GAP)
        sheet.paste(tile, (x, y))

    relation = object_row["association_relation"]
    confirmed = truthy(object_row["confirmed_dead_object"])
    evidence_note = (
        f"Yellow ring: Dead mask {dead_id}.  P90 delta {number(object_row, 'p90_delta'):.2f}; "
        f"SNR {number(object_row, 'snr'):.2f}; Combined overlap {number(object_row, 'combined_overlap_fraction'):.3f}."
    )
    relation_note = (
        f"Nuclei in cell: {integer(object_row, 'combined_nuclei_count')}; final relation: {relation}; "
        f"confirmed Dead object: {'yes' if confirmed else 'no'}."
    )
    draw.text(
        (FIGURE_MARGIN, height - FIGURE_NOTE_HEIGHT + 9),
        evidence_note,
        fill=MUTED,
        font=font(20),
    )
    draw.text(
        (FIGURE_MARGIN, height - FIGURE_NOTE_HEIGHT + 40),
        relation_note,
        fill=MUTED,
        font=font(20),
    )

    stages = {
        "context": prediction_row(audit_root, "context_aware_all_d0", key, combined_id)["state"],
        "strong_direct": prediction_row(audit_root, "strong_direct_all_d0", key, combined_id)["state"],
        "final": prediction_row(audit_root, "object_aware_all_d0_v4", key, combined_id)["state"],
    }
    evidence = {
        "sample": key,
        "cell_id": combined_id,
        "dead_mask_id": dead_id,
        "context_aware_cell_state": stages["context"],
        "strong_direct_cell_state": stages["strong_direct"],
        "final_cell_state": stages["final"],
        "final_object_relation": relation,
        "confirmed_dead_object": confirmed,
        "dead_p90_delta": number(object_row, "p90_delta"),
        "dead_snr": number(object_row, "snr"),
        "combined_overlap_fraction": number(object_row, "combined_overlap_fraction"),
        "combined_nuclei_count": integer(object_row, "combined_nuclei_count"),
        "dead_object_nuclei_inside_count": integer(object_row, "dead_object_nuclei_inside_count"),
        "dead_nucleus_overlap_fraction": number(object_row, "dead_nucleus_overlap_fraction"),
        "nucleus_supported_relation": object_row["nucleus_supported_relation"],
    }
    return sheet, evidence


def render_sentinel_figures(audit_root: Path) -> tuple[dict[str, Image.Image], list[dict[str, Any]]]:
    figures: dict[str, Image.Image] = {}
    evidence: list[dict[str, Any]] = []
    for sentinel in SENTINELS:
        image, row = compose_sentinel_figure(audit_root, sentinel)
        figures[f"sentinel_{sentinel['token']}"] = image
        evidence.append(row)
    expected = {
        "E2_1_00d00h00m": ("live", "live", "live", "adjacent_candidate", False),
        "F5_1_00d00h00m": ("live", "dead", "live", "overlapping_live_dead_multi_nucleus", True),
        "H9_4_00d00h00m": ("live", "dead", "live", "live_with_death_signal", True),
    }
    for row in evidence:
        observed = (
            row["context_aware_cell_state"],
            row["strong_direct_cell_state"],
            row["final_cell_state"],
            row["final_object_relation"],
            row["confirmed_dead_object"],
        )
        if observed != expected[row["sample"]]:
            raise RuntimeError(f"Sentinel regression for {row['sample']}: {observed} != {expected[row['sample']]}")
    return figures, evidence


def csv_lookup(directory: Path, pattern: str, id_field: str) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    paths = sorted(directory.glob(pattern))
    if len(paths) != 320:
        raise RuntimeError(f"Expected 320 files matching {directory / pattern}, found {len(paths)}")
    for path in paths:
        for row in read_rows(path):
            identity = (row["key"], integer(row, id_field))
            if identity in lookup:
                raise RuntimeError(f"Duplicate object identity {identity} while reading {directory}")
            lookup[identity] = row
    return lookup


def case_evidence_row(
    group: str,
    early_stage: str,
    audit_row: dict[str, str],
    object_row: dict[str, str],
    final_row: dict[str, str],
) -> dict[str, Any]:
    return {
        "case_group": group,
        "sample": audit_row["key"],
        "well": audit_row["key"].split("_", 1)[0],
        "cell_id": integer(audit_row, "combined_mask_id"),
        "dead_mask_id": integer(audit_row, "dead_mask_id"),
        "early_stage": early_stage,
        "early_cell_state": audit_row["final_state"],
        "final_cell_state": final_row["state"],
        "final_object_relation": object_row["association_relation"],
        "confirmed_dead_object": truthy(object_row["confirmed_dead_object"]),
        "dead_p90_delta": number(audit_row, "dead_p90_delta"),
        "dead_snr": number(audit_row, "dead_snr"),
        "combined_overlap_fraction": number(audit_row, "dead_combined_overlap_fraction"),
        "combined_nuclei_count": integer(object_row, "combined_nuclei_count"),
        "dead_object_nuclei_inside_count": integer(object_row, "dead_object_nuclei_inside_count"),
        "dead_nucleus_overlap_fraction": number(object_row, "dead_nucleus_overlap_fraction"),
        "nucleus_supported_relation": object_row["nucleus_supported_relation"],
    }


def select_distinct_well_cases(
    candidates: list[dict[str, Any]],
    count: int,
    excluded_wells: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_wells = set(excluded_wells)
    for candidate in sorted(
        candidates,
        key=lambda row: (
            -float(row["dead_snr"]),
            -float(row["dead_p90_delta"]),
            str(row["sample"]),
            int(row["cell_id"]),
        ),
    ):
        if candidate["well"] in used_wells:
            continue
        selected.append(candidate)
        used_wells.add(candidate["well"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} distinct-well cases were available; expected {count}")
    return selected


def select_expanded_cases(audit_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_objects = csv_lookup(
        audit_root / "object_aware_all_d0_v4" / "dead_objects",
        "*_dead_object_features.csv",
        "dead_mask_id",
    )
    final_cells = csv_lookup(
        audit_root / "object_aware_all_d0_v4" / "predictions",
        "*_per_cell_predictions.csv",
        "mask_id",
    )
    sentinel_wells = {str(sentinel["key"]).split("_", 1)[0] for sentinel in SENTINELS}

    live_candidates: list[dict[str, Any]] = []
    for audit_row in read_rows(
        audit_root / "automated_reference_audit_final" / "automated_reference_per_cell.csv"
    ):
        if audit_row["branch"] != "final_original" or not truthy(audit_row["possible_live_false_positive"]):
            continue
        identity = (audit_row["key"], integer(audit_row, "dead_mask_id"))
        cell_identity = (audit_row["key"], integer(audit_row, "combined_mask_id"))
        object_row = final_objects.get(identity)
        final_row = final_cells.get(cell_identity)
        if object_row is None or final_row is None:
            continue
        if (
            audit_row["rgb_state"] != "live"
            or audit_row["final_state"] != "dead"
            or final_row["state"] != "live"
            or object_row["association_relation"] not in LIVE_DEAD_OVERLAP_RELATIONS
            or not truthy(object_row["confirmed_dead_object"])
        ):
            continue
        live_candidates.append(
            case_evidence_row(
                "Live-override proxy",
                "Previous threshold-only",
                audit_row,
                object_row,
                final_row,
            )
        )
    live_cases = select_distinct_well_cases(
        live_candidates,
        EXPANDED_CASES_PER_GROUP,
        sentinel_wells,
    )

    miss_candidates: list[dict[str, Any]] = []
    for audit_row in read_rows(
        audit_root / "automated_reference_audit" / "automated_reference_per_cell.csv"
    ):
        if (
            audit_row["branch"] != "previous_original"
            or not truthy(audit_row["automated_reference_dead"])
            or truthy(audit_row["automated_reference_detected"])
        ):
            continue
        identity = (audit_row["key"], integer(audit_row, "dead_mask_id"))
        cell_identity = (audit_row["key"], integer(audit_row, "combined_mask_id"))
        object_row = final_objects.get(identity)
        final_row = final_cells.get(cell_identity)
        if object_row is None or final_row is None:
            continue
        final_state = final_row["state"]
        final_relation = object_row["association_relation"]
        relation_is_consistent = (
            (final_relation == "same_cell" and final_state == "dead")
            or (final_relation in LIVE_DEAD_OVERLAP_RELATIONS and final_state == "live")
        )
        if (
            audit_row["final_state"] == "dead"
            or not relation_is_consistent
            or not truthy(object_row["confirmed_dead_object"])
        ):
            continue
        miss_candidates.append(
            case_evidence_row(
                "Context-aware death miss",
                "Context-aware",
                audit_row,
                object_row,
                final_row,
            )
        )
    expanded_wells = sentinel_wells | {str(case["well"]) for case in live_cases}
    miss_cases = select_distinct_well_cases(
        miss_candidates,
        EXPANDED_CASES_PER_GROUP,
        expanded_wells,
    )
    return live_cases, miss_cases


def gallery_panel(
    image: Image.Image,
    panel_column: int,
    center_x: float,
    center_y: float,
    label: str,
) -> Image.Image:
    panel = crop_panel(image, panel_column, 0, center_x, center_y, label)
    return panel.resize((GALLERY_PANEL_WIDTH, GALLERY_PANEL_HEIGHT), Image.Resampling.LANCZOS)


def compose_case_gallery(
    audit_root: Path,
    cases: list[dict[str, Any]],
    gallery_kind: str,
) -> Image.Image:
    if len(cases) != EXPANDED_CASES_PER_GROUP:
        raise ValueError(f"Expected {EXPANDED_CASES_PER_GROUP} gallery cases, found {len(cases)}")
    panel_row_width = GALLERY_PANEL_WIDTH * 4 + GALLERY_PANEL_GAP * 3
    case_width = panel_row_width + 24
    case_height = GALLERY_CASE_HEADER + GALLERY_PANEL_HEIGHT + GALLERY_CASE_FOOTER
    rows = math.ceil(len(cases) / GALLERY_COLUMNS)
    title_height = 78
    note_height = 58
    width = FIGURE_MARGIN * 2 + case_width * GALLERY_COLUMNS + GALLERY_CASE_GAP
    height = title_height + rows * case_height + (rows - 1) * GALLERY_CASE_GAP + note_height
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    if gallery_kind == "live_override":
        title = "Expanded live-cell correction gallery: previous threshold-only versus corrected result"
        footer = (
            "Each previous threshold-only result changed an RGB-live cell to dead. The corrected output restores the live state "
            "and renders the retained confirmed-death mask above the live-cell layer."
        )
    elif gallery_kind == "death_miss":
        title = "Expanded death-object recovery: final cell-level attribution"
        footer = (
            "Each context-aware result missed a strong-death reference. Final output retains both mask layers and uses cell-scale "
            "coverage plus nucleus evidence to describe same-cell death or live-with-death-signal overlap."
        )
    else:
        raise ValueError(f"Unknown gallery kind: {gallery_kind}")
    draw.text((FIGURE_MARGIN, 21), title, fill=TEXT, font=font(31, bold=True))

    for index, case in enumerate(cases):
        key = str(case["sample"])
        cell_id = int(case["cell_id"])
        dead_id = int(case["dead_mask_id"])
        object_row = dead_object_row(audit_root, key, dead_id)
        center_x = number(object_row, "centroid_x")
        center_y = number(object_row, "centroid_y")
        object_overlay = dead_object_overlay(audit_root, key)
        final_overlap = overlap_state_overlay(audit_root, key)
        if gallery_kind == "live_override":
            early_overlay = stage_overlay(audit_root, "strong_direct_all_d0", key)
            panels = [
                gallery_panel(object_overlay, 0, center_x, center_y, "Combined RGB"),
                gallery_panel(object_overlay, 1, center_x, center_y, "Raw Dead channel"),
                gallery_panel(early_overlay, 1, center_x, center_y, "Previous result: dead"),
                gallery_panel(final_overlap, 4, center_x, center_y, "Final: live + death mask"),
            ]
        else:
            early_overlay = stage_overlay(audit_root, "context_aware_all_d0", key)
            final_state = str(case["final_cell_state"])
            panels = [
                gallery_panel(object_overlay, 0, center_x, center_y, "Combined RGB"),
                gallery_panel(object_overlay, 1, center_x, center_y, "Raw Dead"),
                gallery_panel(early_overlay, 1, center_x, center_y, "Previous result: live"),
                gallery_panel(
                    final_overlap,
                    4,
                    center_x,
                    center_y,
                    f"Final: {final_state} + death mask",
                ),
            ]

        column = index % GALLERY_COLUMNS
        row = index // GALLERY_COLUMNS
        case_x = FIGURE_MARGIN + column * (case_width + GALLERY_CASE_GAP)
        case_y = title_height + row * (case_height + GALLERY_CASE_GAP)
        draw.rounded_rectangle(
            (case_x, case_y, case_x + case_width, case_y + case_height),
            radius=12,
            fill=PANEL_BG,
            outline=(55, 60, 68),
            width=2,
        )
        draw.text(
            (case_x + 12, case_y + 8),
            f"Case {index + 1}: {key}  |  cell {cell_id}, Dead mask {dead_id}",
            fill=TEXT,
            font=font(21, bold=True),
        )
        panel_y = case_y + GALLERY_CASE_HEADER
        for panel_index, panel in enumerate(panels):
            panel_x = case_x + 12 + panel_index * (GALLERY_PANEL_WIDTH + GALLERY_PANEL_GAP)
            sheet.paste(panel, (panel_x, panel_y))
        draw.text(
            (case_x + 12, panel_y + GALLERY_PANEL_HEIGHT + 7),
            (
                f"P90 delta {float(case['dead_p90_delta']):.1f}  |  SNR {float(case['dead_snr']):.1f}  |  "
                f"Overlap {float(case['combined_overlap_fraction']):.3f}  |  Nuclei {int(case['combined_nuclei_count'])}  |  "
                f"{case['final_object_relation']}"
            ),
            fill=MUTED,
            font=font(17),
        )
    draw.text(
        (FIGURE_MARGIN, height - note_height + 15),
        footer,
        fill=MUTED,
        font=font(19),
    )
    return sheet


def compose_shared_legend_pies(
    audit_root: Path,
    metric_level: str,
    title: str,
    subtitle: str,
    category_order: list[str],
    category_labels: dict[str, str],
) -> Image.Image:
    summary_rows = read_rows(audit_root / "final_annotation_summary.csv")
    branch_labels = {"original": "Original", "nucleated_only": "Nucleated-only"}
    branch_rows: dict[str, list[dict[str, str]]] = {}
    for branch, display in branch_labels.items():
        indexed = {
            row["subclassification"]: row
            for row in summary_rows
            if row["analysis_branch"] == branch and row["metric_level"] == metric_level
        }
        branch_rows[display] = [indexed[key] for key in category_order if key in indexed]

    width, height = 2100, 1120
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    title_font = font(42, bold=True)
    subtitle_font = font(25)
    branch_font = font(34, bold=True)
    percent_font = font(24, bold=True)
    center_font = font(30, bold=True)
    legend_font = font(25)
    legend_note_font = font(21)

    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 30), title, font=title_font, fill=TEXT)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(
        ((width - (subtitle_box[2] - subtitle_box[0])) / 2, 90),
        subtitle,
        font=subtitle_font,
        fill=MUTED,
    )

    centers = {"Original": (560, 470), "Nucleated-only": (1540, 470)}
    radius = 260
    for branch, rows in branch_rows.items():
        cx, cy = centers[branch]
        branch_box = draw.textbbox((0, 0), branch, font=branch_font)
        draw.text(
            (cx - (branch_box[2] - branch_box[0]) / 2, 142),
            branch,
            font=branch_font,
            fill=TEXT,
        )
        total = sum(integer(row, "count") for row in rows)
        if not total:
            raise RuntimeError(f"No pie-chart observations for {metric_level}/{branch}")
        start = -90.0
        outside_labels: list[dict[str, Any]] = []
        for row in rows:
            key = row["subclassification"]
            count = integer(row, "count")
            if count <= 0:
                continue
            share = count / total
            end = start + 360.0 * share
            color = PIE_CATEGORY_COLORS[key]
            draw.pieslice(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                start=start,
                end=end,
                fill=color,
                outline=BG,
                width=4,
            )
            mid = math.radians((start + end) / 2.0)
            label = f"{share:.1%}"
            if share >= 0.035:
                lx = cx + math.cos(mid) * radius * 0.70
                ly = cy + math.sin(mid) * radius * 0.70
                box = draw.textbbox((0, 0), label, font=percent_font)
                tw, th = box[2] - box[0], box[3] - box[1]
                draw.rounded_rectangle(
                    (lx - tw / 2 - 7, ly - th / 2 - 5, lx + tw / 2 + 7, ly + th / 2 + 5),
                    radius=8,
                    fill=(24, 27, 32),
                    outline=(225, 230, 236),
                    width=1,
                )
                draw.text((lx - tw / 2, ly - th / 2 - 1), label, font=percent_font, fill=TEXT)
            else:
                side = 1 if math.cos(mid) >= 0 else -1
                outside_labels.append(
                    {
                        "side": side,
                        "anchor": (cx + math.cos(mid) * radius * 0.94, cy + math.sin(mid) * radius * 0.94),
                        "x": cx + side * radius * 1.20,
                        "y": cy + math.sin(mid) * radius * 1.06,
                        "label": label,
                        "color": color,
                    }
                )
            start = end

        for side in (-1, 1):
            labels = sorted(
                (item for item in outside_labels if item["side"] == side),
                key=lambda item: item["y"],
            )
            previous_y = -10_000.0
            for item in labels:
                item["y"] = max(float(item["y"]), previous_y + 38.0)
                previous_y = float(item["y"])
                ax, ay = item["anchor"]
                elbow_x = cx + side * radius * 1.04
                draw.line((ax, ay, elbow_x, item["y"], item["x"], item["y"]), fill=item["color"], width=3)
                box = draw.textbbox((0, 0), item["label"], font=percent_font)
                tw, th = box[2] - box[0], box[3] - box[1]
                tx = item["x"] + 8 if side > 0 else item["x"] - tw - 8
                draw.text((tx, item["y"] - th / 2), item["label"], font=percent_font, fill=TEXT)

        inner = 112
        draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=BG, outline=(72, 78, 88), width=3)
        count_text = f"n={total:,}"
        count_box = draw.textbbox((0, 0), count_text, font=center_font)
        draw.text(
            (cx - (count_box[2] - count_box[0]) / 2, cy - (count_box[3] - count_box[1]) / 2 - 2),
            count_text,
            font=center_font,
            fill=TEXT,
        )

    legend_y = 792
    draw.line((90, legend_y - 36, width - 90, legend_y - 36), fill=(56, 61, 70), width=2)
    legend_columns = 3
    column_width = (width - 180) / legend_columns
    row_height = 112
    for index, key in enumerate(category_order):
        row_index, col_index = divmod(index, legend_columns)
        x = 100 + col_index * column_width
        y = legend_y + row_index * row_height
        color = PIE_CATEGORY_COLORS[key]
        draw.rounded_rectangle((x, y + 3, x + 30, y + 33), radius=5, fill=color)
        label = category_labels[key]
        draw.multiline_text((x + 44, y), label, font=legend_font, fill=TEXT, spacing=3)
    note = "Shared legend and color mapping across both subplots; exact counts and ratios are reported in the adjacent table."
    note_box = draw.textbbox((0, 0), note, font=legend_note_font)
    draw.text(
        ((width - (note_box[2] - note_box[0])) / 2, height - 48),
        note,
        font=legend_note_font,
        fill=MUTED,
    )
    return sheet


def render_report_figures(
    audit_root: Path,
    live_cases: list[dict[str, Any]],
    miss_cases: list[dict[str, Any]],
) -> tuple[dict[str, Image.Image], list[dict[str, Any]]]:
    figures, sentinel_rows = render_sentinel_figures(audit_root)
    figures["live_override_gallery"] = compose_case_gallery(audit_root, live_cases, "live_override")
    figures["death_miss_gallery"] = compose_case_gallery(audit_root, miss_cases, "death_miss")
    figures["death_relation_pies"] = compose_shared_legend_pies(
        audit_root,
        "confirmed_death_object_relation",
        "Confirmed Death-object relation share",
        "Two branch-specific subplots use one shared relation legend; each subplot sums to 100%.",
        [
            "same_cell",
            "overlapping_live_dead_multi_nucleus",
            "live_with_death_signal",
            "dead_only",
            "adjacent_or_overlapping_dead_uncertain",
            "merged_multiple_objects",
        ],
        {
            "same_cell": "Same cell",
            "overlapping_live_dead_multi_nucleus": "Overlapping live/dead\n(multiple nuclei)",
            "live_with_death_signal": "Live with death signal",
            "dead_only": "Dead-only object",
            "adjacent_or_overlapping_dead_uncertain": "Spatially uncertain",
            "merged_multiple_objects": "Additional merged object",
        },
    )
    figures["death_event_pies"] = compose_shared_legend_pies(
        audit_root,
        "object_aware_death_event_composition",
        "Final object-aware death-count share",
        "Dead cells and supplemental Death-object relations partition the complete total in each branch.",
        [
            "cell_associated_dead_cell",
            "supplemental_overlapping_live_dead_multi_nucleus",
            "supplemental_live_with_death_signal",
            "supplemental_dead_only",
            "supplemental_adjacent_or_overlapping_dead_uncertain",
            "supplemental_merged_multiple_objects",
        ],
        {
            "cell_associated_dead_cell": "Final dead cell",
            "supplemental_overlapping_live_dead_multi_nucleus": "Overlapping live/dead\n(multiple nuclei)",
            "supplemental_live_with_death_signal": "Live with death signal",
            "supplemental_dead_only": "Dead-only object",
            "supplemental_adjacent_or_overlapping_dead_uncertain": "Spatially uncertain",
            "supplemental_merged_multiple_objects": "Additional merged object",
        },
    )
    return figures, sentinel_rows


def aggregate_summaries(path: Path) -> dict[str, int]:
    rows = read_rows(path)
    if len(rows) != 320:
        raise RuntimeError(f"Expected 320 d0 summary rows in {path}, found {len(rows)}")
    fields = [
        "dead_cell_count",
        "supplemental_dead_object_count",
        "object_aware_dead_count",
        "confirmed_dead_object_count",
        "segmented_dead_object_count",
    ]
    return {field: sum(integer(row, field) for row in rows) for field in fields}


def csv_shape(path: Path) -> tuple[int, int]:
    require_file(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return sum(1 for _row in reader), len(header)


def annotation_metric(
    rows: list[dict[str, str]],
    branch: str,
    metric_level: str,
    classification: str,
    subclassification: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["analysis_branch"] == branch
        and row["metric_level"] == metric_level
        and row["classification"] == classification
        and row["subclassification"] == subclassification
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one final annotation metric for "
            f"{branch}/{metric_level}/{classification}/{subclassification}, found {len(matches)}"
        )
    return matches[0]


def debugging_stage_rows(path: Path) -> list[dict[str, Any]]:
    path = require_file(path)
    payload = MAIN_REPORT.read_json(path)
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RuntimeError(f"Debugging-stage provenance has no stages: {path}")
    template_fields = ("field_runner_template", "merge_template", "audit_template")
    missing_templates = [field for field in template_fields if not payload.get(field)]
    if missing_templates:
        raise RuntimeError(
            f"Debugging-stage provenance is missing command templates {missing_templates}: {path}"
        )
    command_templates = " | ".join(
        (
            f"Field: {payload['field_runner_template']}",
            f"Merge: {payload['merge_template']}",
            f"Audit: {payload['audit_template']}",
        )
    )
    required = (
        "order",
        "stage",
        "objective",
        "key_parameters",
        "run_method",
        "output_artifacts",
        "code_provenance",
        "reproducibility",
    )
    required_set = set(required)
    normalized: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    for row in stages:
        if not isinstance(row, dict) or not required_set.issubset(row):
            missing = sorted(required_set - set(row if isinstance(row, dict) else {}))
            raise RuntimeError(f"Invalid debugging-stage provenance row; missing {missing}: {path}")
        order = int(row["order"])
        if order in seen_orders:
            raise RuntimeError(f"Duplicate debugging-stage order {order}: {path}")
        seen_orders.add(order)
        normalized_row = {key: row[key] for key in required}
        normalized_row["command_templates"] = command_templates
        normalized.append(normalized_row)
    return sorted(normalized, key=lambda row: int(row["order"]))


def build_datasets(
    audit_root: Path,
    debugging_stage_provenance: Path,
    sentinel_rows: list[dict[str, Any]],
    live_cases: list[dict[str, Any]],
    miss_cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    before_after = read_rows(audit_root / "d0_branch_before_after_metrics.csv")
    context_audit = read_rows(audit_root / "automated_reference_audit" / "automated_reference_metrics.csv")
    strong_audit = read_rows(audit_root / "automated_reference_audit_final" / "automated_reference_metrics.csv")
    final_audit = read_rows(
        audit_root / "object_aware_automated_reference_audit_v4" / "automated_reference_metrics.csv"
    )
    stress = read_rows(audit_root / "dead_object_detection_stress_v2" / "detection_stress_metrics.csv")
    strong_cells = read_rows(
        audit_root / "automated_reference_audit_final" / "automated_reference_per_cell.csv"
    )
    final_annotation_summary = read_rows(audit_root / "final_annotation_summary.csv")
    master_annotation_path = audit_root / "final_multilevel_annotations.csv"
    master_annotation_rows, master_annotation_columns = csv_shape(master_annotation_path)

    branch_labels = {"original": "Original", "nucleated_only": "Nucleated-only"}
    stage_rows: list[dict[str, Any]] = []
    for branch, label in branch_labels.items():
        baseline = next(row for row in before_after if row["branch"] == branch and row["version"] == "baseline")
        context = next(row for row in before_after if row["branch"] == branch and row["version"] == "after")
        strong_key = "final_original" if branch == "original" else "final_nucleated"
        final_key = branch
        strong = one_row(strong_audit, "branch", strong_key)
        final = one_row(final_audit, "branch", final_key)
        for stage, source in (
            ("Original result", baseline),
            ("Context-aware", context),
            ("Previous threshold-only", strong),
            ("Final object-aware", final),
        ):
            if "rgb_live_to_dead_rate" in source:
                denominator = integer(source, "rgb_live_cells")
                numerator = integer(source, "rgb_live_classified_dead")
                rate = number(source, "rgb_live_to_dead_rate")
            else:
                denominator = integer(source, "conservative_rgb_live_reference_count")
                numerator = integer(source, "possible_live_false_positive_count")
                rate = number(source, "possible_live_false_positive_upper_bound")
            stage_rows.append(
                {
                    "branch": label,
                    "stage": stage,
                    "rgb_live_cells": denominator,
                    "possible_live_overrides": numerator,
                    "live_false_positive_proxy": rate,
                }
            )

    reference_rows: list[dict[str, Any]] = []
    for branch, label in branch_labels.items():
        context_key = "previous_original" if branch == "original" else "previous_nucleated"
        strong_key = "final_original" if branch == "original" else "final_nucleated"
        for stage, source, unit in (
            ("Context-aware", one_row(context_audit, "branch", context_key), "Cell-associated reference"),
            ("Previous threshold-only", one_row(context_audit, "branch", strong_key), "Cell-associated reference"),
            ("Final object-aware", one_row(final_audit, "branch", branch), "Complete strong-object reference"),
        ):
            if stage == "Final object-aware":
                count_field = "automated_reference_dead_object_count"
                detected_field = "automated_reference_dead_object_detected"
                recall_field = "automated_reference_dead_object_recall"
            else:
                count_field = "automated_reference_death_count"
                detected_field = "automated_reference_death_detected"
                recall_field = "automated_reference_death_recall"
            reference_rows.append(
                {
                    "branch": label,
                    "stage": stage,
                    "reference_unit": unit,
                    "reference_count": integer(source, count_field),
                    "detected": integer(source, detected_field),
                    "operational_recall": number(source, recall_field),
                }
            )

    final_summary_paths = {
        "Original": audit_root / "object_aware_all_d0_v4" / "summaries" / "cell_count_summary.csv",
        "Nucleated-only": audit_root
        / "object_aware_all_d0_v4_nucleated_only"
        / "summaries"
        / "cell_count_summary.csv",
    }
    final_summaries = {label: aggregate_summaries(path) for label, path in final_summary_paths.items()}
    original_dead_calls = {
        "Original": integer(next(row for row in before_after if row["branch"] == "original" and row["version"] == "baseline"), "dead_cells"),
        "Nucleated-only": integer(next(row for row in before_after if row["branch"] == "nucleated_only" and row["version"] == "baseline"), "dead_cells"),
    }
    composition_rows: list[dict[str, Any]] = []
    composition_table: list[dict[str, Any]] = []
    for label, summary in final_summaries.items():
        total = summary["object_aware_dead_count"]
        for component, field in (
            ("Cell-associated dead", "dead_cell_count"),
            ("Supplemental Dead objects", "supplemental_dead_object_count"),
        ):
            composition_rows.append(
                {
                    "branch": label,
                    "component": component,
                    "count": summary[field],
                    "object_aware_total": total,
                }
            )
        composition_table.append(
            {
                "branch": label,
                "original_dead_calls": original_dead_calls[label],
                "final_dead_cell_count": summary["dead_cell_count"],
                "supplemental_dead_objects": summary["supplemental_dead_object_count"],
                "object_aware_dead_count": total,
                "confirmed_dead_objects": summary["confirmed_dead_object_count"],
                "change_from_original": total / original_dead_calls[label] - 1.0,
            }
        )

    represented: dict[str, set[tuple[str, int]]] = {"Original": set(), "Nucleated-only": set()}
    per_cell_branch = {"final_original": "Original", "final_nucleated": "Nucleated-only"}
    for row in strong_cells:
        dead_id = integer(row, "dead_mask_id")
        if dead_id > 0:
            represented[per_cell_branch[row["branch"]]].add((row["key"], dead_id))
    source_objects = integer(one_row(final_audit, "branch", "original"), "segmented_dead_object_count")
    representation_rows: list[dict[str, Any]] = []
    for label in ("Original", "Nucleated-only"):
        old_count = len(represented[label])
        for structure, count in (("Old per-cell structure", old_count), ("Final object ledger", source_objects)):
            representation_rows.append(
                {
                    "branch": label,
                    "structure": structure,
                    "represented_objects": count,
                    "source_objects": source_objects,
                    "missing_objects": source_objects - count,
                    "representation_rate": count / source_objects,
                }
            )

    stress_original = one_row(stress, "branch", "original")
    stress_rows = [
        {
            "test": "Strong-object recovery",
            "series": "Observed",
            "rate": number(stress_original, "strong_reference_recall"),
            "objects": integer(stress_original, "strong_reference_recovered"),
            "denominator": integer(stress_original, "strong_reference_count"),
        },
        {
            "test": "Strong-object recovery",
            "series": "Required",
            "rate": 0.9995,
            "objects": None,
            "denominator": None,
        },
        {
            "test": "Synthetic relocation",
            "series": "Observed",
            "rate": number(stress_original, "synthetic_relocation_recall"),
            "objects": integer(stress_original, "synthetic_relocation_recovered"),
            "denominator": integer(stress_original, "strong_reference_count"),
        },
        {
            "test": "Synthetic relocation",
            "series": "Required",
            "rate": 0.9995,
            "objects": None,
            "denominator": None,
        },
    ]

    final_original = one_row(final_audit, "branch", "original")
    final_nucleated = one_row(final_audit, "branch", "nucleated_only")
    if source_objects != 8289:
        raise RuntimeError(f"Expected the frozen 8,289-object ledger, found {source_objects}")
    if integer(final_original, "automated_reference_dead_object_detected") != 1189:
        raise RuntimeError("The final original branch no longer retains all 1,189 strong-object references")
    if integer(final_nucleated, "automated_reference_dead_object_detected") != 1189:
        raise RuntimeError("The final nucleated-only branch no longer retains all 1,189 strong-object references")

    overview = [
        {
            "fields_per_branch": 320,
            "analysis_branches": 2,
            "source_dead_objects": source_objects,
            "ledger_dead_objects": source_objects,
            "strong_reference_objects": 1189,
            "strong_reference_detected": 1189,
            "rgb_live_overrides": integer(final_original, "possible_live_false_positive_count")
            + integer(final_nucleated, "possible_live_false_positive_count"),
            "cell_qc_per_branch": integer(final_original, "qc_complete_count"),
            "dead_object_qc_per_branch": 320,
            "overlap_qc_per_branch": 320,
            "multilevel_annotation_rows": master_annotation_rows,
            "multilevel_annotation_columns": master_annotation_columns,
        }
    ]

    annotation_branch_labels = {"original": "Original", "nucleated_only": "Nucleated-only"}
    final_cell_state_rows: list[dict[str, Any]] = []
    final_cell_detail_rows: list[dict[str, Any]] = []
    cell_uncertainty_rows: list[dict[str, Any]] = []
    confirmed_death_relation_rows: list[dict[str, Any]] = []
    death_uncertainty_rows: list[dict[str, Any]] = []
    object_aware_event_detail_rows: list[dict[str, Any]] = []
    for branch, label in annotation_branch_labels.items():
        for state in ("live", "dead", "transitional", "uncertain", "artifact"):
            state_metric = annotation_metric(
                final_annotation_summary,
                branch,
                "cell_state",
                state,
                "all",
            )
            uncertainty_metric = annotation_metric(
                final_annotation_summary,
                branch,
                "cell_uncertainty_by_state",
                state,
                "uncertain",
            )
            confidence_counts: dict[str, int] = {}
            for confidence in ("high", "medium", "low"):
                matches = [
                    row
                    for row in final_annotation_summary
                    if row["analysis_branch"] == branch
                    and row["metric_level"] == "cell_confidence_by_state"
                    and row["classification"] == state
                    and row["subclassification"] == confidence
                ]
                confidence_counts[confidence] = integer(matches[0], "count") if matches else 0
            final_cell_state_rows.append(
                {
                    "branch": label,
                    "state": state,
                    "count": integer(state_metric, "count"),
                    "all_cells": integer(state_metric, "denominator"),
                    "share_of_all_cells": number(state_metric, "percentage"),
                    "high_confidence": confidence_counts["high"],
                    "medium_confidence": confidence_counts["medium"],
                    "low_confidence": confidence_counts["low"],
                    "non_high_confidence": integer(uncertainty_metric, "count"),
                    "non_high_or_relation_uncertain_rate": number(
                        uncertainty_metric,
                        "percentage",
                    ),
                }
            )
            if state in {"live", "dead"}:
                cell_uncertainty_rows.append(
                    {
                        "branch": label,
                        "final_state": state,
                        "uncertain_count": integer(uncertainty_metric, "count"),
                        "state_count": integer(uncertainty_metric, "denominator"),
                        "uncertainty_rate": number(uncertainty_metric, "percentage"),
                    }
                )

        for row in final_annotation_summary:
            if (
                row["analysis_branch"] == branch
                and row["metric_level"] == "cell_detailed_classification"
            ):
                final_cell_detail_rows.append(
                    {
                        "branch": label,
                        "final_state": row["classification"],
                        "detailed_annotation": row["subclassification"],
                        "count": integer(row, "count"),
                        "state_denominator": integer(row, "denominator"),
                        "within_state_share": number(row, "percentage"),
                    }
                )
            if (
                row["analysis_branch"] == branch
                and row["metric_level"] == "confirmed_death_object_relation"
            ):
                confirmed_death_relation_rows.append(
                    {
                        "branch": label,
                        "relation": row["subclassification"],
                        "count": integer(row, "count"),
                        "confirmed_death_objects": integer(row, "denominator"),
                        "share_of_confirmed_death": number(row, "percentage"),
                    }
                )
            if (
                row["analysis_branch"] == branch
                and row["metric_level"] == "object_aware_death_event_composition"
            ):
                component_labels = {
                    "cell_associated_dead_cell": "Final dead cells",
                    "supplemental_overlapping_live_dead_multi_nucleus": (
                        "Overlapping live/dead, multiple nuclei"
                    ),
                    "supplemental_live_with_death_signal": "Live with death signal",
                    "supplemental_dead_only": "Dead-only objects",
                    "supplemental_adjacent_or_overlapping_dead_uncertain": (
                        "Spatially uncertain confirmed objects"
                    ),
                    "supplemental_merged_multiple_objects": "Additional merged objects",
                }
                component = component_labels.get(
                    row["subclassification"],
                    row["subclassification"],
                )
                object_aware_event_detail_rows.append(
                    {
                        "branch": label,
                        "component": component,
                        "count": integer(row, "count"),
                        "object_aware_total": integer(row, "denominator"),
                        "share_of_object_aware_total": number(row, "percentage"),
                    }
                )

        confirmed_uncertainty = annotation_metric(
            final_annotation_summary,
            branch,
            "death_object_uncertainty_by_status",
            "dead",
            "uncertain",
        )
        candidate_uncertainty = annotation_metric(
            final_annotation_summary,
            branch,
            "death_object_uncertainty_by_status",
            "candidate",
            "uncertain",
        )
        death_uncertainty_rows.extend(
            [
                {
                    "branch": label,
                    "annotation_unit": "Confirmed Death object",
                    "uncertainty_definition": "Confirmed signal with uncertain spatial attribution",
                    "count": integer(confirmed_uncertainty, "count"),
                    "denominator": integer(confirmed_uncertainty, "denominator"),
                    "percentage": number(confirmed_uncertainty, "percentage"),
                },
                {
                    "branch": label,
                    "annotation_unit": "Death candidate",
                    "uncertainty_definition": "Retained signal not confirmed as a final Death object",
                    "count": integer(candidate_uncertainty, "count"),
                    "denominator": integer(candidate_uncertainty, "denominator"),
                    "percentage": number(candidate_uncertainty, "percentage"),
                },
            ]
        )

    annotation_level_definitions = [
        {
            "level": "Image",
            "unit": "One zero-time-point field",
            "annotation": "Image identity, well, site, time point, and branch.",
            "role": "Defines the acquisition context for every cell and Death object.",
        },
        {
            "level": "Cell",
            "unit": "One Combined mask",
            "annotation": "RGB state, morphology, BF/nucleus support, final cell state, reason, and confidence.",
            "role": "Determines whether the segmented Combined object is live, dead, transitional, uncertain, or artifact.",
        },
        {
            "level": "Death object",
            "unit": "One upstream Dead mask",
            "annotation": "Signal intensity, background-normalized evidence, confirmation status, and supplemental count status.",
            "role": "Determines whether an independently segmented Dead-channel object is confirmed, candidate, or rejected.",
        },
        {
            "level": "Cell–Death relation",
            "unit": "One Death object linked to zero or one Combined cell",
            "annotation": "Spatial overlap, distance, cell-scale support, and association relation.",
            "role": "Prevents object evidence from automatically rewriting the state of a nearby cell.",
        },
        {
            "level": "Nucleus-supported relation",
            "unit": "One linked Death object",
            "annotation": "Nuclei in the Combined cell, nuclei inside/near the Death mask, distance, and overlap fraction.",
            "role": "Distinguishes probable multi-cell overlap from a one-nucleus live cell carrying death signal.",
        },
        {
            "level": "Final multilevel annotation",
            "unit": "One cell row or one Death-object row",
            "annotation": "Final classification, detailed classification, confidence, uncertainty, and cross-level record link.",
            "role": "Provides one non-duplicating ledger for cohort-level analysis and audit.",
        },
    ]

    classification_definitions = [
        {
            "annotation_unit": "Cell",
            "final_classification": "live",
            "definition": "Countable Combined cell without an accepted cell-level death override; a supplemental confirmed Death object may occupy the same pixels.",
            "counting_rule": "Count once as a live cell; any confirmed supplemental Death object is counted separately.",
        },
        {
            "annotation_unit": "Cell",
            "final_classification": "dead",
            "definition": "Combined cell with a matched same-cell Dead object whose RGB/Dead-channel and cell-scale support pass the final override rule.",
            "counting_rule": "Count once in dead_cell_count.",
        },
        {
            "annotation_unit": "Cell",
            "final_classification": "transitional",
            "definition": "Reserved final state for mixed live/death appearance when the cell-level decision remains transitional.",
            "counting_rule": "Reported separately; zero final d0 cells in both current branches.",
        },
        {
            "annotation_unit": "Cell",
            "final_classification": "uncertain",
            "definition": "Reserved final state when a cell cannot be assigned live or dead under the decision rules.",
            "counting_rule": "Reported separately; zero final d0 cells in both current branches.",
        },
        {
            "annotation_unit": "Cell",
            "final_classification": "artifact",
            "definition": "Combined mask excluded by size or unsupported artifact criteria.",
            "counting_rule": "Excluded from countable-cell denominators.",
        },
        {
            "annotation_unit": "Death object",
            "final_classification": "dead",
            "definition": "Retained Dead signal confirmed by strong direct evidence or local dual-channel RGB support.",
            "counting_rule": "Counted once as a confirmed Death object; relation determines same-cell versus supplemental use.",
        },
        {
            "annotation_unit": "Death object",
            "final_classification": "candidate",
            "definition": "Retained Dead-channel signal that does not satisfy final confirmation criteria.",
            "counting_rule": "Audited but excluded from final death counts.",
        },
        {
            "annotation_unit": "Death object",
            "final_classification": "rejected",
            "definition": "Segmented Dead mask that fails the retained-signal gate.",
            "counting_rule": "Preserved in the ledger for audit but excluded from final death counts.",
        },
    ]

    output_contract = [
        {
            "output": "object_aware_dead_count",
            "grain": "One row per image",
            "purpose": "Primary downstream d0 death count: cell-associated deaths plus supplemental confirmed objects.",
        },
        {
            "output": "dead_cell_count",
            "grain": "One row per image",
            "purpose": "Count of Combined cells whose final cell-level state is dead.",
        },
        {
            "output": "supplemental_dead_object_count",
            "grain": "One row per image",
            "purpose": "Confirmed dead-only, adjacent-dead, and additional merged-neighbor objects.",
        },
        {
            "output": "dead_objects/<image>_dead_object_features.csv",
            "grain": "One row per upstream Dead mask",
            "purpose": "Complete auditable ledger of all 8,289 objects and their association relation.",
        },
        {
            "output": "qc/dead_object_overlays/<image>_dead_object_overlay.png",
            "grain": "One image per field",
            "purpose": "Combined RGB, raw Dead signal, and relation-colored object QC.",
        },
        {
            "output": "overlap_masks/<image>_cell_dead_overlap_masks.tif",
            "grain": "Two label channels per field",
            "purpose": "Channel 0 stores cell instances and channel 1 stores confirmed Dead objects, allowing pixel overlap.",
        },
        {
            "output": "final_multilevel_annotations.csv",
            "grain": "One row per cell or Death object across both branches",
            "purpose": "Complete 140-column cross-level annotation ledger with stable record IDs and cell–Death links.",
        },
        {
            "output": "final_annotation_summary.csv",
            "grain": "One row per metric, branch, class, and subclass",
            "purpose": "Validated counts, denominators, and percentages used directly by report tables and charts.",
        },
        {
            "output": "qc/overlap_state_overlays/<image>_overlap_state_overlay.png",
            "grain": "One image per field",
            "purpose": "Nucleus-aware QC with the confirmed-death layer rendered above the cell-state layer.",
        },
    ]

    return {
        "overview": overview,
        "live_fp_stage": stage_rows,
        "reference_coverage": reference_rows,
        "object_representation": representation_rows,
        "final_composition": composition_rows,
        "final_composition_table": composition_table,
        "detector_stress": stress_rows,
        "sentinel_evidence": sentinel_rows,
        "expanded_case_evidence": live_cases + miss_cases,
        "annotation_level_definitions": annotation_level_definitions,
        "classification_definitions": classification_definitions,
        "final_cell_states": final_cell_state_rows,
        "final_cell_details": final_cell_detail_rows,
        "cell_uncertainty": cell_uncertainty_rows,
        "confirmed_death_relations": confirmed_death_relation_rows,
        "death_uncertainty": death_uncertainty_rows,
        "object_aware_event_details": object_aware_event_detail_rows,
        "output_contract": output_contract,
        "debugging_stages": debugging_stage_rows(debugging_stage_provenance),
    }


def sql_text(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def make_sources(
    expanded_cases: list[dict[str, Any]],
    debugging_stage_provenance: Path,
) -> list[dict[str, Any]]:
    base = "results/dead_d0_classification_audit"
    require_file(debugging_stage_provenance)
    provenance_path = f"{base}/debugging_stage_provenance.json"
    if len(expanded_cases) != 12:
        raise RuntimeError(f"Expected 12 expanded comparison cases, found {len(expanded_cases)}")
    selected_case_values = ",".join(
        "("
        + ",".join(
            (
                sql_text(row["case_group"]),
                sql_text(row["sample"]),
                str(int(row["cell_id"])),
                str(int(row["dead_mask_id"])),
                sql_text(row["early_stage"]),
            )
        )
        + ")"
        for row in expanded_cases
    )
    return [
        {
            "id": "methods_md",
            "label": "Dead-classification improvement comparison",
            "path": "cellpose_pipeline/report/DEAD_CLASSIFICATION_IMPROVEMENT_COMPARISON.md",
            "query": {
                "engine": "duckdb",
                "description": "Materialize the downstream output contract documented by the comparison Markdown.",
                "sql": (
                    "SELECT * FROM (VALUES "
                    "('object_aware_dead_count','One row per image'),"
                    "('dead_cell_count','One row per image'),"
                    "('supplemental_dead_object_count','One row per image'),"
                    "('dead_objects/<image>_dead_object_features.csv','One row per upstream Dead mask'),"
                    "('qc/dead_object_overlays/<image>_dead_object_overlay.png','One image per field')) "
                    "AS output_contract(output, grain)"
                ),
                "tables_used": ["DEAD_CLASSIFICATION_IMPROVEMENT_COMPARISON.md"],
            },
        },
        {
            "id": "stage_provenance",
            "label": "Frozen debugging-stage parameters and run provenance",
            "path": provenance_path,
            "query": {
                "engine": "duckdb",
                "description": "Read the ordered historical classifier stages, parameters, run templates, and code-provenance limits.",
                "sql": (
                    f"SELECT stage.* FROM read_json_auto({sql_text(provenance_path)}, "
                    "maximum_object_size=10485760) root, UNNEST(root.stages) AS t(stage) "
                    "ORDER BY stage.\"order\""
                ),
                "tables_used": ["debugging_stage_provenance.json"],
            },
        },
        {
            "id": "before_after",
            "label": "d0 baseline and context-aware branch metrics",
            "path": f"{base}/d0_branch_before_after_metrics.csv",
            "query": {
                "engine": "duckdb",
                "description": "Compare the same RGB-live denominator before and after context-aware gating.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/d0_branch_before_after_metrics.csv', header=true) ORDER BY branch, version",
                "tables_used": ["d0_branch_before_after_metrics.csv"],
                "metric_definitions": {
                    "live_false_positive_proxy": "RGB-live cells classified as dead divided by all RGB-live cells."
                },
            },
        },
        {
            "id": "iteration_comparison",
            "label": "Four-stage d0 live-cell override comparison",
            "path": f"{base}/four-stage-live-override-comparison",
            "query": {
                "engine": "duckdb",
                "description": "Combine like-for-like RGB-live override numerators and denominators across all four classifier stages.",
                "sql": (
                    f"WITH early AS (SELECT branch, version, rgb_live_cells, rgb_live_classified_dead possible_live_overrides, rgb_live_to_dead_rate live_false_positive_proxy FROM read_csv_auto('{base}/d0_branch_before_after_metrics.csv', header=true)), "
                    f"strong AS (SELECT branch, conservative_rgb_live_reference_count rgb_live_cells, possible_live_false_positive_count possible_live_overrides, possible_live_false_positive_upper_bound live_false_positive_proxy FROM read_csv_auto('{base}/automated_reference_audit_final/automated_reference_metrics.csv', header=true)), "
                    f"final AS (SELECT branch, conservative_rgb_live_reference_count rgb_live_cells, possible_live_false_positive_count possible_live_overrides, possible_live_false_positive_upper_bound live_false_positive_proxy FROM read_csv_auto('{base}/object_aware_automated_reference_audit_v4/automated_reference_metrics.csv', header=true)) "
                    "SELECT * FROM early UNION ALL SELECT branch, 'strong_direct', rgb_live_cells, possible_live_overrides, live_false_positive_proxy FROM strong UNION ALL SELECT branch, 'object_aware', rgb_live_cells, possible_live_overrides, live_false_positive_proxy FROM final"
                ),
                "tables_used": [
                    "d0_branch_before_after_metrics.csv",
                    "automated_reference_audit_final/automated_reference_metrics.csv",
                    "object_aware_automated_reference_audit_v4/automated_reference_metrics.csv",
                ],
                "metric_definitions": {
                    "live_false_positive_proxy": "RGB-live cells classified as dead divided by all RGB-live cells."
                },
            },
        },
        {
            "id": "stage_audit",
            "label": "Context-aware and previous threshold-only automated reference audit",
            "path": f"{base}/automated_reference_audit",
            "query": {
                "engine": "duckdb",
                "description": "Read cell-associated automated-reference recovery and conservative live-cell override proxies.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/automated_reference_audit/automated_reference_metrics.csv', header=true)",
                "tables_used": ["automated_reference_audit/automated_reference_metrics.csv"],
            },
        },
        {
            "id": "reference_comparison",
            "label": "Cell-associated and object-level automated reference comparison",
            "path": f"{base}/automated-reference-comparison",
            "query": {
                "engine": "duckdb",
                "description": "Combine the earlier cell-associated references with the final complete strong-object reference.",
                "sql": (
                    f"SELECT branch, automated_reference_death_count reference_count, automated_reference_death_detected detected, automated_reference_death_recall operational_recall, 'Cell-associated reference' reference_unit FROM read_csv_auto('{base}/automated_reference_audit/automated_reference_metrics.csv', header=true) "
                    f"UNION ALL SELECT branch, automated_reference_dead_object_count, automated_reference_dead_object_detected, automated_reference_dead_object_recall, 'Complete strong-object reference' FROM read_csv_auto('{base}/object_aware_automated_reference_audit_v4/automated_reference_metrics.csv', header=true)"
                ),
                "tables_used": [
                    "automated_reference_audit/automated_reference_metrics.csv",
                    "object_aware_automated_reference_audit_v4/automated_reference_metrics.csv",
                ],
            },
        },
        {
            "id": "strong_audit",
            "label": "Previous threshold-only output audit",
            "path": f"{base}/automated_reference_audit_final",
            "query": {
                "engine": "duckdb",
                "description": "Read the previous threshold-only stage's reference recovery and reconstruct object representation in the old per-cell structure.",
                "sql": (
                    f"WITH old_structure AS (SELECT branch, key, dead_mask_id FROM read_csv_auto('{base}/automated_reference_audit_final/automated_reference_per_cell.csv', header=true) WHERE dead_mask_id > 0), "
                    f"source_objects AS (SELECT max(segmented_dead_object_count) source_objects FROM read_csv_auto('{base}/object_aware_automated_reference_audit_v4/automated_reference_metrics.csv', header=true)) "
                    "SELECT branch, count(DISTINCT (key, dead_mask_id)) represented_objects, source_objects FROM old_structure CROSS JOIN source_objects GROUP BY branch, source_objects"
                ),
                "tables_used": [
                    "automated_reference_audit_final/automated_reference_metrics.csv",
                    "automated_reference_audit_final/automated_reference_per_cell.csv",
                    "object_aware_automated_reference_audit_v4/automated_reference_metrics.csv",
                ],
            },
        },
        {
            "id": "object_audit",
            "label": "Final object-aware automated reference audit",
            "path": f"{base}/object_aware_automated_reference_audit_v4",
            "query": {
                "engine": "duckdb",
                "description": "Read final object-ledger coverage, strong-object recovery, live-cell override proxy, and QC completeness.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/object_aware_automated_reference_audit_v4/automated_reference_metrics.csv', header=true)",
                "tables_used": [
                    "object_aware_automated_reference_audit_v4/automated_reference_metrics.csv",
                    "object_aware_automated_reference_audit_v4/per_field_metrics.csv",
                ],
            },
        },
        {
            "id": "final_annotations",
            "label": "Final multilevel annotation ledger and classification summaries",
            "path": f"{base}/final_multilevel_annotations.csv",
            "query": {
                "engine": "duckdb",
                "description": "Read final cell states, Death-object relations, confidence tiers, uncertainty flags, and their explicit denominators.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/final_annotation_summary.csv', header=true) ORDER BY analysis_branch, metric_level, classification, subclassification",
                "tables_used": [
                    "final_multilevel_annotations.csv",
                    "final_annotation_summary.csv",
                ],
                "metric_definitions": {
                    "cell_uncertainty": "Final cell state is explicitly uncertain, classification confidence is medium/low, or a confirmed linked Death object has uncertain spatial attribution.",
                    "confirmed_death_relation_share": "Confirmed Death objects in one relation divided by all confirmed Death objects in the same branch.",
                },
            },
        },
        {
            "id": "final_summaries",
            "label": "Final object-aware per-field summaries",
            "path": f"{base}/object_aware_all_d0_v4/summaries",
            "query": {
                "engine": "duckdb",
                "description": "Aggregate final cell-associated and supplemental Dead counts across 320 fields per branch.",
                "sql": (
                    f"SELECT 'Original' branch, sum(dead_cell_count) dead_cell_count, sum(supplemental_dead_object_count) supplemental_dead_object_count, sum(object_aware_dead_count) object_aware_dead_count FROM read_csv_auto('{base}/object_aware_all_d0_v4/summaries/cell_count_summary.csv', header=true) "
                    f"UNION ALL SELECT 'Nucleated-only', sum(dead_cell_count), sum(supplemental_dead_object_count), sum(object_aware_dead_count) FROM read_csv_auto('{base}/object_aware_all_d0_v4_nucleated_only/summaries/cell_count_summary.csv', header=true)"
                ),
                "tables_used": [
                    "object_aware_all_d0_v4/summaries/cell_count_summary.csv",
                    "object_aware_all_d0_v4_nucleated_only/summaries/cell_count_summary.csv",
                ],
            },
        },
        {
            "id": "stress_test",
            "label": "Independent raw-signal detector stress test",
            "path": f"{base}/dead_object_detection_stress_v2/detection_stress_metrics.csv",
            "query": {
                "engine": "duckdb",
                "description": "Read global-seed recovery, synthetic relocation recovery, and shifted-seed negative-control hits.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/dead_object_detection_stress_v2/detection_stress_metrics.csv', header=true)",
                "tables_used": ["dead_object_detection_stress_v2/detection_stress_metrics.csv"],
            },
        },
        {
            "id": "sentinel_qc",
            "label": "E2, F5, and H9 iterative QC overlays and object ledgers",
            "path": f"{base}/object_aware_all_d0_v4/qc",
            "query": {
                "engine": "duckdb",
                "description": "Select the three fixed sentinel cells and Dead masks from each classifier stage and final object ledger.",
                "sql": (
                    f"WITH targets(key, cell_id, dead_mask_id) AS (VALUES ('E2_1_00d00h00m',32,2),('F5_1_00d00h00m',83,6),('H9_4_00d00h00m',113,18)), "
                    f"objects AS (SELECT * FROM read_csv_auto('{base}/object_aware_all_d0_v4/dead_objects/*_dead_object_features.csv', header=true)), "
                    f"final_cells AS (SELECT * FROM read_csv_auto('{base}/object_aware_all_d0_v4/predictions/*_per_cell_predictions.csv', header=true)) "
                    "SELECT targets.key, targets.cell_id, targets.dead_mask_id, final_cells.state final_cell_state, objects.association_relation final_object_relation, objects.combined_nuclei_count, objects.dead_object_nuclei_inside_count, objects.dead_nucleus_overlap_fraction, objects.p90_delta, objects.snr FROM targets JOIN final_cells ON targets.key=final_cells.key AND targets.cell_id=final_cells.mask_id JOIN objects ON targets.key=objects.key AND targets.dead_mask_id=objects.dead_mask_id"
                ),
                "tables_used": [
                    "object_aware_all_d0_v4/dead_objects/*_dead_object_features.csv",
                    "context_aware_all_d0/predictions/*_per_cell_predictions.csv",
                    "strong_direct_all_d0/predictions/*_per_cell_predictions.csv",
                    "object_aware_all_d0_v4/predictions/*_per_cell_predictions.csv",
                    "object_aware_all_d0_v4/qc/label_overlays/*.png",
                    "object_aware_all_d0_v4/qc/dead_object_overlays/*.png",
                    "object_aware_all_d0_v4/qc/overlap_state_overlays/*.png",
                    "object_aware_all_d0_v4/overlap_masks/*.tif",
                ],
                "filters": {"sentinel_fields": ["E2_1_00d00h00m", "F5_1_00d00h00m", "H9_4_00d00h00m"]},
            },
        },
        {
            "id": "expanded_qc",
            "label": "Deterministic expanded live-override and death-miss QC cohort",
            "path": f"{base}/expanded-comparison-qc",
            "query": {
                "engine": "duckdb",
                "description": "Reproduce the exact six distinct-well live-override and six distinct-well context-aware death-miss cases shown in the expanded galleries.",
                "sql": (
                    "WITH selected(case_group,key,cell_id,dead_mask_id,early_stage) AS (VALUES "
                    + selected_case_values
                    + "), "
                    + f"strong_live AS (SELECT key, combined_mask_id cell_id, dead_mask_id, final_state early_cell_state, dead_p90_delta, dead_snr, dead_combined_overlap_fraction FROM read_csv_auto('{base}/automated_reference_audit_final/automated_reference_per_cell.csv', header=true) WHERE branch='final_original' AND possible_live_false_positive=true), "
                    + f"context_miss AS (SELECT key, combined_mask_id cell_id, dead_mask_id, final_state early_cell_state, dead_p90_delta, dead_snr, dead_combined_overlap_fraction FROM read_csv_auto('{base}/automated_reference_audit/automated_reference_per_cell.csv', header=true) WHERE branch='previous_original' AND automated_reference_dead=true AND automated_reference_detected=false), "
                    + "early AS (SELECT 'Previous threshold-only' early_stage, * FROM strong_live UNION ALL SELECT 'Context-aware', * FROM context_miss), "
                    + f"objects AS (SELECT key, dead_mask_id, association_relation final_object_relation, confirmed_dead_object, combined_nuclei_count, dead_object_nuclei_inside_count, dead_nucleus_overlap_fraction FROM read_csv_auto('{base}/object_aware_all_d0_v4/dead_objects/*_dead_object_features.csv', header=true)), "
                    + f"final_cells AS (SELECT key, mask_id cell_id, state final_cell_state FROM read_csv_auto('{base}/object_aware_all_d0_v4/predictions/*_per_cell_predictions.csv', header=true)) "
                    + "SELECT selected.case_group, selected.key sample, selected.cell_id, selected.dead_mask_id, selected.early_stage, early.early_cell_state, final_cells.final_cell_state, objects.final_object_relation, objects.confirmed_dead_object, objects.combined_nuclei_count, objects.dead_object_nuclei_inside_count, objects.dead_nucleus_overlap_fraction, early.dead_p90_delta, early.dead_snr, early.dead_combined_overlap_fraction combined_overlap_fraction FROM selected JOIN early USING(key,cell_id,dead_mask_id,early_stage) JOIN objects USING(key,dead_mask_id) JOIN final_cells USING(key,cell_id) ORDER BY selected.case_group, selected.key, selected.cell_id"
                ),
                "tables_used": [
                    "automated_reference_audit_final/automated_reference_per_cell.csv",
                    "automated_reference_audit/automated_reference_per_cell.csv",
                    "object_aware_all_d0_v4/dead_objects/*_dead_object_features.csv",
                    "context_aware_all_d0/qc/label_overlays/*.png",
                    "strong_direct_all_d0/qc/label_overlays/*.png",
                    "object_aware_all_d0_v4/qc/label_overlays/*.png",
                    "object_aware_all_d0_v4/qc/dead_object_overlays/*.png",
                    "object_aware_all_d0_v4/qc/overlap_state_overlays/*.png",
                    "object_aware_all_d0_v4/overlap_masks/*.tif",
                ],
                "filters": {
                    "cases_per_group": 6,
                    "distinct_wells": True,
                    "sort": "dead_snr descending, then dead_p90_delta descending",
                    "excluded_sentinel_wells": ["E2", "F5", "H9"],
                },
            },
        },
    ]


def build_manifest(
    generated_at: str,
    image_uris: dict[str, str],
    image_sizes: dict[str, tuple[int, int]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "SUM159 d0 Dead-Classification Improvement Report"
    cards = [
        {
            "id": "master_annotation_card",
            "description": "One non-duplicating cell/Death-object ledger across both final branches.",
            "dataset": "overview",
            "sourceId": "final_annotations",
            "metrics": [
                {"label": "Multilevel annotation rows", "field": "multilevel_annotation_rows", "format": "number"},
                {"label": "Annotation columns", "field": "multilevel_annotation_columns", "format": "number"},
            ],
        },
        {
            "id": "ledger_card",
            "description": "Every upstream Dead segmentation object is preserved as an auditable row in each branch.",
            "dataset": "overview",
            "sourceId": "object_audit",
            "metrics": [
                {"label": "Dead objects in final ledger", "field": "ledger_dead_objects", "format": "number"},
                {"label": "Source Dead objects", "field": "source_dead_objects", "format": "number"},
            ],
        },
        {
            "id": "reference_card",
            "description": "Operational recovery among existing masks meeting the fixed strong-object definition.",
            "dataset": "overview",
            "sourceId": "object_audit",
            "metrics": [
                {"label": "Strong objects detected", "field": "strong_reference_detected", "format": "number"},
                {"label": "Strong-object denominator", "field": "strong_reference_objects", "format": "number"},
            ],
        },
        {
            "id": "live_override_card",
            "description": "RGB-live cells forcibly changed to dead by Dead-channel evidence across both final branches.",
            "dataset": "overview",
            "sourceId": "object_audit",
            "metrics": [{"label": "Final RGB-live overrides", "field": "rgb_live_overrides", "format": "number"}],
        },
        {
            "id": "qc_card",
            "description": "Validated cell-state and Dead-object QC fields produced independently for each branch.",
            "dataset": "overview",
            "sourceId": "object_audit",
            "metrics": [
                {"label": "QC fields per output type and branch", "field": "cell_qc_per_branch", "format": "number"},
                {"label": "Analysis branches", "field": "analysis_branches", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "live_fp_chart",
            "title": "RGB-live-to-dead proxy across classifier stages",
            "subtitle": "All 320 zero-time-point fields; the same RGB-live denominator is used within each branch.",
            "type": "bar",
            "dataset": "live_fp_stage",
            "sourceId": "iteration_comparison",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "stage", "type": "ordinal", "label": "Classifier stage"},
                "y": {"field": "live_false_positive_proxy", "type": "quantitative", "label": "Live FP proxy", "format": "percent"},
                "color": {"field": "branch", "type": "nominal", "label": "Branch"},
                "tooltip": [
                    {"field": "possible_live_overrides", "type": "quantitative", "label": "RGB-live cells changed to dead", "format": "number"},
                    {"field": "rgb_live_cells", "type": "quantitative", "label": "RGB-live denominator", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "reference_chart",
            "title": "Automated death-reference recovery",
            "subtitle": "Context-aware and previous threshold-only stages use cell-associated references; final object-aware uses all strong Dead-mask objects.",
            "type": "bar",
            "dataset": "reference_coverage",
            "sourceId": "reference_comparison",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "stage", "type": "ordinal", "label": "Classifier stage"},
                "y": {"field": "operational_recall", "type": "quantitative", "label": "Operational recall", "format": "percent"},
                "color": {"field": "branch", "type": "nominal", "label": "Branch"},
                "tooltip": [
                    {"field": "reference_unit", "type": "nominal", "label": "Reference unit"},
                    {"field": "detected", "type": "quantitative", "label": "Detected", "format": "number"},
                    {"field": "reference_count", "type": "quantitative", "label": "Reference count", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "representation_chart",
            "title": "Representation of upstream Dead objects",
            "subtitle": "Coverage of the same 8,289 upstream Dead masks before and after the object-ledger redesign.",
            "type": "bar",
            "dataset": "object_representation",
            "sourceId": "strong_audit",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "structure", "type": "ordinal", "label": "Output structure"},
                "y": {"field": "representation_rate", "type": "quantitative", "label": "Object representation", "format": "percent"},
                "color": {"field": "branch", "type": "nominal", "label": "Branch"},
                "tooltip": [
                    {"field": "represented_objects", "type": "quantitative", "label": "Represented objects", "format": "number"},
                    {"field": "missing_objects", "type": "quantitative", "label": "Missing from structure", "format": "number"},
                    {"field": "source_objects", "type": "quantitative", "label": "Source objects", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "composition_chart",
            "title": "Final object-aware death-count composition",
            "subtitle": "Final dead cells and each independently confirmed supplemental relation across 320 fields.",
            "type": "bar",
            "dataset": "object_aware_event_details",
            "sourceId": "final_annotations",
            "valueFormat": "number",
            "options": {"grouping": "stacked"},
            "encodings": {
                "x": {"field": "branch", "type": "nominal", "label": "Branch"},
                "y": {"field": "count", "type": "quantitative", "label": "Final death count", "format": "number"},
                "color": {"field": "component", "type": "nominal", "label": "Count component"},
                "tooltip": [
                    {"field": "object_aware_total", "type": "quantitative", "label": "Object-aware total", "format": "number"}
                ],
            },
            "layout": "full",
        },
        {
            "id": "cell_state_chart",
            "title": "Final cell-state composition",
            "subtitle": "Share of all Combined-cell annotations across 320 d0 fields in each branch.",
            "type": "bar",
            "dataset": "final_cell_states",
            "sourceId": "final_annotations",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "state", "type": "nominal", "label": "Final cell classification"},
                "y": {"field": "share_of_all_cells", "type": "quantitative", "label": "Share of all cells", "format": "percent"},
                "color": {"field": "branch", "type": "nominal", "label": "Branch"},
                "tooltip": [
                    {"field": "count", "type": "quantitative", "label": "Cells", "format": "number"},
                    {"field": "all_cells", "type": "quantitative", "label": "All Combined cells", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "death_relation_chart",
            "title": "Confirmed Death-object relation composition",
            "subtitle": "Each branch contains 1,416 confirmed Death objects; colors show the final cell–Death relation.",
            "type": "bar",
            "dataset": "confirmed_death_relations",
            "sourceId": "final_annotations",
            "valueFormat": "number",
            "options": {"grouping": "stacked"},
            "encodings": {
                "x": {"field": "branch", "type": "nominal", "label": "Branch"},
                "y": {"field": "count", "type": "quantitative", "label": "Confirmed Death objects", "format": "number"},
                "color": {"field": "relation", "type": "nominal", "label": "Final relation"},
                "tooltip": [
                    {"field": "share_of_confirmed_death", "type": "quantitative", "label": "Share of confirmed Death objects", "format": "percent"},
                    {"field": "confirmed_death_objects", "type": "quantitative", "label": "Confirmed Death denominator", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "cell_uncertainty_chart",
            "title": "Non-high-confidence or relation-uncertain cell annotations",
            "subtitle": "Live and dead cells are shown separately; the denominator is the corresponding final-state count in each branch.",
            "type": "bar",
            "dataset": "cell_uncertainty",
            "sourceId": "final_annotations",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "final_state", "type": "nominal", "label": "Final cell state"},
                "y": {"field": "uncertainty_rate", "type": "quantitative", "label": "Uncertainty rate", "format": "percent"},
                "color": {"field": "branch", "type": "nominal", "label": "Branch"},
                "tooltip": [
                    {"field": "uncertain_count", "type": "quantitative", "label": "Uncertain annotations", "format": "number"},
                    {"field": "state_count", "type": "quantitative", "label": "State denominator", "format": "number"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "stress_chart",
            "title": "Independent global-seed detector stress test",
            "subtitle": "Observed recovery is compared with the 99.95% operational detection requirement.",
            "type": "bar",
            "dataset": "detector_stress",
            "sourceId": "stress_test",
            "valueFormat": "percent",
            "options": {"grouping": "grouped"},
            "encodings": {
                "x": {"field": "test", "type": "nominal", "label": "Stress-test scenario"},
                "y": {"field": "rate", "type": "quantitative", "label": "Recovery rate", "format": "percent"},
                "color": {"field": "series", "type": "nominal", "label": "Observed versus requirement"},
                "tooltip": [
                    {"field": "objects", "type": "quantitative", "label": "Recovered objects", "format": "number"},
                    {"field": "denominator", "type": "quantitative", "label": "Reference objects", "format": "number"},
                ],
            },
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "debugging_stage_table",
            "title": "Classifier debugging stages, parameters, and execution provenance",
            "subtitle": "Ordered d0 development history; frozen-reference stages are explicitly distinguished from computationally reproducible stages.",
            "dataset": "debugging_stages",
            "sourceId": "stage_provenance",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "Round", "format": "number"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "objective", "label": "Purpose", "type": "text"},
                {"field": "key_parameters", "label": "Key parameters and rules", "type": "text"},
                {"field": "run_method", "label": "Run method", "type": "text"},
                {"field": "command_templates", "label": "Exact command templates", "type": "text"},
                {"field": "output_artifacts", "label": "Frozen or current outputs", "type": "text"},
                {"field": "code_provenance", "label": "Code provenance", "type": "text"},
                {"field": "reproducibility", "label": "Reproducibility status", "type": "text"},
            ],
        },
        {
            "id": "annotation_level_table",
            "title": "Annotation levels and analytical units",
            "subtitle": "Each level has a distinct unit and role; cell and Death-object rows are not interchangeable.",
            "dataset": "annotation_level_definitions",
            "sourceId": "final_annotations",
            "columns": [
                {"field": "level", "label": "Annotation level", "type": "text"},
                {"field": "unit", "label": "Unit", "type": "text"},
                {"field": "annotation", "label": "Stored annotation", "type": "text"},
                {"field": "role", "label": "Role in classification", "type": "text"},
            ],
        },
        {
            "id": "classification_definition_table",
            "title": "Final classification definitions",
            "subtitle": "Cell classifications and Death-object classifications use different denominators and counting rules.",
            "dataset": "classification_definitions",
            "sourceId": "final_annotations",
            "columns": [
                {"field": "annotation_unit", "label": "Unit", "type": "text"},
                {"field": "final_classification", "label": "Final classification", "type": "text"},
                {"field": "definition", "label": "Operational definition", "type": "text"},
                {"field": "counting_rule", "label": "Counting rule", "type": "text"},
            ],
        },
        {
            "id": "final_cell_state_table",
            "title": "Final cell classifications and confidence",
            "subtitle": "Counts use one row per Combined cell; uncertainty is defined as medium/low confidence or uncertain Death attribution.",
            "dataset": "final_cell_states",
            "sourceId": "final_annotations",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "state", "label": "Final cell state", "type": "text"},
                {"field": "count", "label": "Cells", "format": "number"},
                {"field": "all_cells", "label": "All cells", "format": "number"},
                {"field": "share_of_all_cells", "label": "Share of all cells", "format": "percent"},
                {"field": "high_confidence", "label": "High confidence", "format": "number"},
                {"field": "medium_confidence", "label": "Medium confidence", "format": "number"},
                {"field": "low_confidence", "label": "Low confidence", "format": "number"},
                {"field": "non_high_confidence", "label": "Uncertain annotations", "format": "number"},
                {"field": "non_high_or_relation_uncertain_rate", "label": "Uncertainty rate", "format": "percent"},
            ],
        },
        {
            "id": "final_cell_detail_table",
            "title": "Detailed annotations within each final cell state",
            "subtitle": "Live and dead states are decomposed using linked confirmed Death objects without duplicating cell rows.",
            "dataset": "final_cell_details",
            "sourceId": "final_annotations",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "final_state", "label": "Final state", "type": "text"},
                {"field": "detailed_annotation", "label": "Detailed annotation", "type": "text"},
                {"field": "count", "label": "Cells", "format": "number"},
                {"field": "state_denominator", "label": "State denominator", "format": "number"},
                {"field": "within_state_share", "label": "Within-state share", "format": "percent"},
            ],
        },
        {
            "id": "death_relation_table",
            "title": "Final relation of confirmed Death objects",
            "subtitle": "The denominator is all 1,416 confirmed Death objects in the corresponding branch.",
            "dataset": "confirmed_death_relations",
            "sourceId": "final_annotations",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "relation", "label": "Final relation", "type": "text"},
                {"field": "count", "label": "Confirmed Death objects", "format": "number"},
                {"field": "confirmed_death_objects", "label": "Denominator", "format": "number"},
                {"field": "share_of_confirmed_death", "label": "Share", "format": "percent"},
            ],
        },
        {
            "id": "death_uncertainty_table",
            "title": "Death-object uncertainty inventory",
            "subtitle": "Confirmed-but-spatially-uncertain objects are separated from unconfirmed retained candidates.",
            "dataset": "death_uncertainty",
            "sourceId": "final_annotations",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "annotation_unit", "label": "Annotation unit", "type": "text"},
                {"field": "uncertainty_definition", "label": "Uncertainty definition", "type": "text"},
                {"field": "count", "label": "Uncertain", "format": "number"},
                {"field": "denominator", "label": "Denominator", "format": "number"},
                {"field": "percentage", "label": "Rate", "format": "percent"},
            ],
        },
        {
            "id": "stage_table",
            "title": "Like-for-like live-cell override comparison",
            "subtitle": "Exact numerator, denominator, and proxy rate for each classifier stage and branch.",
            "dataset": "live_fp_stage",
            "sourceId": "iteration_comparison",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "rgb_live_cells", "label": "RGB-live cells", "format": "number"},
                {"field": "possible_live_overrides", "label": "Changed to dead", "format": "number"},
                {"field": "live_false_positive_proxy", "label": "Live FP proxy", "format": "percent"},
            ],
        },
        {
            "id": "reference_table",
            "title": "Automated reference recovery by stage",
            "subtitle": "Reference units are shown explicitly because the final object-aware denominator is broader.",
            "dataset": "reference_coverage",
            "sourceId": "reference_comparison",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "reference_unit", "label": "Reference unit", "type": "text"},
                {"field": "reference_count", "label": "Reference count", "format": "number"},
                {"field": "detected", "label": "Detected", "format": "number"},
                {"field": "operational_recall", "label": "Operational recall", "format": "percent"},
            ],
        },
        {
            "id": "sentinel_table",
            "title": "Sentinel-field regression results",
            "subtitle": "Cell state and object relation for the three failure mechanisms used during tuning.",
            "dataset": "sentinel_evidence",
            "sourceId": "sentinel_qc",
            "defaultSort": {"field": "sample", "direction": "asc"},
            "columns": [
                {"field": "sample", "label": "Field", "type": "text"},
                {"field": "cell_id", "label": "Cell", "format": "number"},
                {"field": "dead_mask_id", "label": "Dead mask", "format": "number"},
                {"field": "context_aware_cell_state", "label": "Context-aware", "type": "text"},
                {"field": "strong_direct_cell_state", "label": "Previous threshold-only", "type": "text"},
                {"field": "final_cell_state", "label": "Final cell", "type": "text"},
                {"field": "final_object_relation", "label": "Final relation", "type": "text"},
                {"field": "combined_nuclei_count", "label": "Nuclei in cell", "format": "number"},
                {"field": "dead_p90_delta", "label": "P90 delta", "format": "number"},
                {"field": "dead_snr", "label": "SNR", "format": "number"},
            ],
        },
        {
            "id": "expanded_case_table",
            "title": "Expanded comparison-case evidence",
            "subtitle": "Six distinct-well cases per mechanism, selected deterministically by Dead SNR after fixed eligibility filters.",
            "dataset": "expanded_case_evidence",
            "sourceId": "expanded_qc",
            "defaultSort": {"field": "case_group", "direction": "asc"},
            "columns": [
                {"field": "case_group", "label": "Case group", "type": "text"},
                {"field": "sample", "label": "Field", "type": "text"},
                {"field": "cell_id", "label": "Cell", "format": "number"},
                {"field": "dead_mask_id", "label": "Dead mask", "format": "number"},
                {"field": "early_stage", "label": "Compared stage", "type": "text"},
                {"field": "early_cell_state", "label": "Earlier state", "type": "text"},
                {"field": "final_cell_state", "label": "Final cell state", "type": "text"},
                {"field": "final_object_relation", "label": "Final relation", "type": "text"},
                {"field": "combined_nuclei_count", "label": "Nuclei in cell", "format": "number"},
                {"field": "dead_object_nuclei_inside_count", "label": "Nuclei in Dead mask", "format": "number"},
                {"field": "dead_p90_delta", "label": "P90 delta", "format": "number"},
                {"field": "dead_snr", "label": "SNR", "format": "number"},
                {"field": "combined_overlap_fraction", "label": "Combined overlap", "format": "percent"},
            ],
        },
        {
            "id": "composition_table",
            "title": "Final death-count components and original-call comparison",
            "subtitle": "Cohort totals across all 320 fields in each branch.",
            "dataset": "final_composition_table",
            "sourceId": "final_summaries",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "original_dead_calls", "label": "Original calls", "format": "number"},
                {"field": "final_dead_cell_count", "label": "Final dead cells", "format": "number"},
                {"field": "supplemental_dead_objects", "label": "Supplemental objects", "format": "number"},
                {"field": "object_aware_dead_count", "label": "Object-aware total", "format": "number"},
                {"field": "change_from_original", "label": "Change from original", "format": "percent", "movement": True},
            ],
        },
        {
            "id": "object_aware_event_detail_table",
            "title": "Detailed composition of the final object-aware death total",
            "subtitle": "Final dead cells and supplemental confirmed Death-object relations; denominator is the object-aware total in each branch.",
            "dataset": "object_aware_event_details",
            "sourceId": "final_annotations",
            "defaultSort": {"field": "branch", "direction": "asc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "component", "label": "Final death component", "type": "text"},
                {"field": "count", "label": "Events", "format": "number"},
                {"field": "object_aware_total", "label": "Object-aware total", "format": "number"},
                {"field": "share_of_object_aware_total", "label": "Share of total", "format": "percent"},
            ],
        },
        {
            "id": "output_table",
            "title": "Final output contract",
            "subtitle": "Outputs that should be used for downstream d0 death analysis and object-level audit.",
            "dataset": "output_contract",
            "sourceId": "methods_md",
            "defaultSort": {"field": "output", "direction": "asc"},
            "columns": [
                {"field": "output", "label": "Output", "type": "text"},
                {"field": "grain", "label": "Grain", "type": "text"},
                {"field": "purpose", "label": "Use", "type": "text"},
            ],
        },
    ]

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## The object-aware design resolves the original cell/object conflict\n\n"
                "The original classifier treated a spatially matched Dead object as proof that the entire Combined cell was dead. "
                "That decision produced many E2-like live-cell overrides. The first context-aware repair protected RGB-live cells but "
                "then rejected compact or partially overlapping strong objects such as F5 and H9. The final workflow separates the "
                "cell state from the Dead-object state: an RGB-live cell can remain live while a nearby confirmed object is retained "
                "and counted independently. Across both 320-field branches, the final audit records every upstream Dead mask, retains "
                "every strong-object reference, and performs no Dead-channel override of an RGB-live cell. These are operational "
                "consistency results, not biological sensitivity or specificity."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "master_annotation_card",
                "ledger_card",
                "reference_card",
                "live_override_card",
                "qc_card",
            ],
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope and operational metric definitions\n\n"
                "The comparison covers all **320 `00d00h00m` fields** in both the original Combined-cell branch and the "
                "nucleated-only branch. Combined, Brightfield, Nuclei, shape-strict, and upstream Dead segmentation masks were "
                "frozen; only Dead association and classification were changed.\n\n"
                "**Live false-positive proxy** is the number of RGB-live cells whose final state is dead divided by all RGB-live "
                "cells. It is a conservative override proxy, not a manually verified false-positive rate. **Strong-object "
                "operational recall** is the fraction of existing Dead masks with `keep_signal = True`, `P90 delta >= 40`, and "
                "`SNR >= 100` that remain confirmed in the final output. It cannot measure true deaths that were never segmented."
            ),
        },
        {
            "id": "classification_process",
            "type": "markdown",
            "body": (
                "## Classification proceeds from signal evidence to two linked final labels\n\n"
                "1. **Segmented inputs remain frozen.** The workflow reads the Combined-cell, Dead, Nuclei, Brightfield, and "
                "RGB-derived evidence already produced for each field.\n"
                "2. **Dead-signal screening** calculates object intensity relative to local background. `keep_signal` identifies "
                "a retained candidate; failed objects remain in the ledger as `rejected`.\n"
                "3. **Death-object confirmation** labels a retained object `dead` when it has strong direct Dead-channel evidence "
                "or local dual-channel RGB support. Other retained objects remain `candidate`.\n"
                "4. **Cell–Death association** uses overlap, distance, RGB context, and cell-scale coverage to decide whether a "
                "confirmed object belongs to the same cell or should remain supplemental.\n"
                "5. **Nucleus-supported refinement** uses the number and location of nuclei to distinguish probable multi-cell "
                "overlap from a one-nucleus live cell carrying death signal.\n"
                "6. **Final cell classification** assigns the Combined cell independently as `live`, `dead`, `transitional`, "
                "`uncertain`, or `artifact`. A live cell and a confirmed Death object may therefore validly overlap.\n"
                "7. **Final multilevel output** writes one cell row and one row for every Death object, connected by stable record "
                "IDs. This preserves both biological units without double-counting cells."
            ),
        },
        {
            "id": "annotation_levels_intro",
            "type": "markdown",
            "body": (
                "## Six annotation levels preserve evidence without collapsing biological units\n\n"
                "The image context, Combined cell, independently segmented Death object, spatial association, nucleus-supported "
                "interpretation, and final multilevel classification are stored separately. The table below defines the unit and "
                "decision role of each layer."
            ),
        },
        {"id": "annotation_level_table_block", "type": "table", "tableId": "annotation_level_table", "layout": "full"},
        {
            "id": "final_classification_intro",
            "type": "markdown",
            "body": (
                "## Final classification is defined separately for cells and Death objects\n\n"
                "For a **cell**, the final label describes the Combined mask. For a **Death object**, the final label describes the "
                "independently segmented Dead-channel mask. `live + confirmed Death` is therefore an intentional two-layer result, "
                "not a contradictory cell label. The counting rules below define exactly which unit contributes to each total."
            ),
        },
        {"id": "classification_definition_table_block", "type": "table", "tableId": "classification_definition_table", "layout": "full"},
        {
            "id": "original_failure",
            "type": "markdown",
            "body": (
                "## Unconditional spatial association caused the original live-cell errors\n\n"
                "The original signal gate was deliberately permissive, but any retained object assigned to a Combined cell could "
                "overwrite that cell's RGB state. Weak blue/Dead signal near a red live cell, or a small shrunken death object beside "
                "a larger attached cell, therefore transferred an object-level observation to the wrong biological unit. The "
                "failure was structural rather than a single bad intensity threshold."
            ),
        },
        {
            "id": "e2_figure_heading",
            "type": "markdown",
            "body": (
                "### Figure 1. E2 shows why RGB-live protection is necessary\n\n"
                "Cell 32 is RGB-live. The weak blue/Dead object is close enough to be spatially associated but is not a confirmed "
                "death object. The original result changed the cell to dead; context-aware and final object-aware logic preserve "
                "the live state. The final ledger keeps mask 2 as an `adjacent_candidate`, so the evidence is auditable without "
                "inflating the death count. The original and corrected cell classifications are vertically aligned in the middle "
                "column. Yellow rings mark the same Dead-mask centroid in every panel."
            ),
        },
        {
            "id": "sentinel_e2_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["sentinel_e2"],
                "E2 cell 32 and Dead mask 2 across original, context-aware, and final object-aware classifications",
                "Figure 1. E2 live-cell false-positive mechanism and final object-aware resolution.",
                *image_sizes["sentinel_e2"],
            ),
            "layout": "full",
        },
        {
            "id": "live_override_gallery_heading",
            "type": "markdown",
            "body": (
                "### Figure 2. Six additional cases compare the previous threshold-only result with the corrected result\n\n"
                "The **previous threshold-only stage** was the intermediate approach that allowed sufficiently strong Dead-channel "
                "evidence to change the state of an entire matched cell. These cases are selected from its 334 RGB-live-to-dead "
                "override proxies using a fixed rule: "
                "confirmed final overlapping death object, final cell restored to live, one case per well, sentinel wells excluded, "
                "then highest Dead SNR. The last two panels in every case are the direct comparison: **Previous result: dead** "
                "followed by **Final: live + death mask**. The corrected workflow preserves both layers in the same pixels. These are "
                "operational proxy cases, not manually annotated false positives."
            ),
        },
        {
            "id": "live_override_gallery_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["live_override_gallery"],
                "Six distinct-well RGB-live override proxy cases comparing the previous threshold-only and corrected object-aware results",
                "Figure 2. Previous threshold-only classification and corrected classification are adjacent in the last two panels of each case.",
                *image_sizes["live_override_gallery"],
            ),
            "layout": "full",
        },
        {
            "id": "first_fix_tradeoff",
            "type": "markdown",
            "body": (
                "## Cell-level overlap protection reduced false positives but created death misses\n\n"
                "The first revision required stronger Dead intensity, supportive RGB context, and substantial coverage of the "
                "Combined cell before a live cell could change state. This reduced the aggregate live override proxy from about "
                "5.7%-6.0% to about 0.014%, but compact death objects occupying only a small fraction of a much larger red cell "
                "were rejected. F5 and H9 are the two sentinel mechanisms that exposed this recall loss."
            ),
        },
        {
            "id": "f5_figure_heading",
            "type": "markdown",
            "body": (
                "### Figure 3. F5 supports overlapping live and death objects with multiple nuclei\n\n"
                "Dead mask 6 has very strong local signal but covers only a small fraction of cell 83. Context-aware gating kept "
                "the cell live but lost the death object. The previous threshold-only recovery changed the entire cell to dead. "
                "The Combined region contains four nucleus cores, including three whose centroids fall inside Dead mask 6. This "
                "supports probable live/dead object overlap. The final panel therefore preserves the live-cell layer and draws the "
                "confirmed death layer above it instead of forcing a single state."
            ),
        },
        {
            "id": "sentinel_f5_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["sentinel_f5"],
                "F5 cell 83 and Dead mask 6 across the previous threshold-only, context-aware intermediate, and corrected object-aware classifications",
                "Figure 3. F5 overlapping live and confirmed-death masks supported by multiple nuclei.",
                *image_sizes["sentinel_f5"],
            ),
            "layout": "full",
        },
        {
            "id": "h9_figure_heading",
            "type": "markdown",
            "body": (
                "### Figure 4. H9 has one nucleus and is retained as live with death signal\n\n"
                "Dead mask 18 overlaps a red Combined-cell region but has strong local Dead-channel evidence. A cell-level overlap "
                "rule rejected it, whereas the previous threshold-only recovery changed cell 113 to dead. The final local object "
                "analysis finds one nucleus core in the Combined cell and one nucleus centroid inside the Dead mask, not two nuclei. "
                "Following the stated rule, cell 113 remains live at this stage and mask 18 is labeled `live_with_death_signal`. "
                "The final overlap panel displays the confirmed death mask above the live-cell mask, consistent with a cell that "
                "may be progressing through death rather than two demonstrated cells."
            ),
        },
        {
            "id": "sentinel_h9_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["sentinel_h9"],
                "H9 cell 113 and Dead mask 18 across the previous threshold-only, context-aware intermediate, and corrected object-aware classifications",
                "Figure 4. H9 one-nucleus live cell with an overlapping confirmed-death signal layer.",
                *image_sizes["sentinel_h9"],
            ),
            "layout": "full",
        },
        {
            "id": "uncertain_cell_attribution_fix",
            "type": "markdown",
            "body": (
                "## Cell-scale support prevents strong adjacent objects from rewriting RGB-uncertain cells\n\n"
                "The first object-aware result protected cells already classified as RGB-live, but it still treated every retained "
                "Dead object assigned to an RGB-uncertain cell as `same_cell`. Figure 5 case 3 exposed that remaining shortcut: "
                "Dead mask 7 covered only 6.47% of a 1,761-pixel C6 Combined cell, so the strong object was real but its transfer to "
                "the entire cell was not supported. The corrected rule requires at least 45% Combined-mask coverage, or at least "
                "30% coverage when the Combined cell is compact (area at most 500 pixels). Across 320 fields, 203 RGB-uncertain "
                "cell records changed from dead to live in each branch, while all 1,416 confirmed Dead objects remained retained. "
                "The v4 output now writes separate cell and confirmed-death instance masks plus a two-channel TIFF, so the two "
                "labels can occupy the same pixels. Nucleus evidence then distinguishes multi-nucleus probable overlap from a "
                "single-nucleus live cell carrying death signal. These remain operational interpretations without manual ground truth."
            ),
        },
        {
            "id": "death_miss_gallery_heading",
            "type": "markdown",
            "body": (
                "### Figure 5. Six context-aware death-object misses show the final attribution decision\n\n"
                "These cases are selected from context-aware automated strong-reference misses using a fixed rule: final confirmed "
                "object, a consistent `same_cell`/dead or overlapping-death/live result, one case per well, all detailed-sentinel and "
                "expanded live-gallery wells excluded, then highest Dead SNR. The last panel states both the corrected cell state "
                "and visibly overlays the confirmed death mask. Cases 1 (H8), 3 (C6), and 5 (D8) each contain one nucleus and are "
                "reported as `live_with_death_signal`; case 6 (F2) retains `same_cell` and dead because the compact cell has "
                "sufficient object coverage. This gallery evaluates retention and attribution of existing strong Dead masks; it does not measure "
                "unsegmented biological deaths."
            ),
        },
        {
            "id": "death_miss_gallery_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["death_miss_gallery"],
                "Six distinct-well strong-reference misses comparing context-aware rejection with final overlapping cell and death masks",
                "Figure 5. Recovered Dead objects displayed above the retained cell-state masks.",
                *image_sizes["death_miss_gallery"],
            ),
            "layout": "full",
        },
        {"id": "sentinel_table_block", "type": "table", "tableId": "sentinel_table", "layout": "full"},
        {"id": "expanded_case_table_block", "type": "table", "tableId": "expanded_case_table", "layout": "full"},
        {
            "id": "iteration_results",
            "type": "markdown",
            "body": (
                "## Iterative classifier performance exposes the threshold tradeoff\n\n"
                "Context-aware gating nearly removed live-cell overrides but missed strong F5/H9-type evidence. The previous "
                "threshold-only recovery stage (internally named `strong-direct`) restored the cell-associated automated "
                "references, but the live override proxy rose to 0.426% and "
                "0.457%, with unstable per-field performance. The final object-aware design removes the tradeoff by preventing "
                "object evidence from automatically rewriting an RGB-live cell."
            ),
        },
        {"id": "live_fp_chart_block", "type": "chart", "chartId": "live_fp_chart", "layout": "full"},
        {"id": "stage_table_block", "type": "table", "tableId": "stage_table", "layout": "full"},
        {
            "id": "reference_result",
            "type": "markdown",
            "body": (
                "## Strong-reference recovery returns to 100% under the appropriate output unit\n\n"
                "Context-aware recovery was 67.11% in the original branch and 61.53% in the nucleated-only branch. The previous threshold-only stage "
                "reached 100% for cell-associated references, while the final object-aware audit reaches 100% for the broader set "
                "of strong objects from all Dead masks. The bars must therefore be read with the reference-unit labels in the "
                "table; the final denominator includes objects that the old cell association could not represent."
            ),
        },
        {"id": "reference_chart_block", "type": "chart", "chartId": "reference_chart", "layout": "full"},
        {"id": "reference_table_block", "type": "table", "tableId": "reference_table", "layout": "full"},
        {
            "id": "ledger_result",
            "type": "markdown",
            "body": (
                "## A complete object ledger removes losses caused by one-object-per-cell association\n\n"
                "The old per-cell structure represented 6,953 of 8,289 objects in the original branch and 5,984 in the "
                "nucleated-only branch. Unassigned masks and additional objects sharing one Combined-cell match were absent. The "
                "final output writes one row for every upstream mask, including rejected and unassigned candidates, yielding "
                "8,289/8,289 rows with no per-field count mismatch in either branch."
            ),
        },
        {"id": "representation_chart_block", "type": "chart", "chartId": "representation_chart", "layout": "full"},
        {
            "id": "final_cell_classification_result",
            "type": "markdown",
            "body": (
                "## Final cell classifications remain dominated by live cells, with uncertainty reported inside each state\n\n"
                "The original branch contains 88,724 live, 1,164 dead, and 427 artifact cells; the nucleated-only branch contains "
                "80,561 live, 935 dead, and 10 artifact cells. Neither branch has a final `transitional` or `uncertain` cell state. "
                "That does not mean every live/dead assignment is high-confidence: medium/low-confidence decisions and uncertain "
                "Death attribution are quantified separately below."
            ),
        },
        {"id": "cell_state_chart_block", "type": "chart", "chartId": "cell_state_chart", "layout": "full"},
        {"id": "final_cell_state_table_block", "type": "table", "tableId": "final_cell_state_table", "layout": "full"},
        {
            "id": "final_cell_detail_result",
            "type": "markdown",
            "body": (
                "### Detailed cell annotations distinguish ordinary live cells from overlapping Death evidence\n\n"
                "In the original branch, 327 live cells are annotated `live_overlapping_dead_multi_nucleus`, 260 are "
                "`live_with_death_signal`, and 4 have an uncertain Death relation. In the nucleated-only branch, the corresponding "
                "counts are 332, 277, and 0. Dead cells are also decomposed into those linked to a confirmed same-cell Death object "
                "and those supported by the cell-level classifier without a linked object meeting the stricter confirmation definition."
            ),
        },
        {"id": "final_cell_detail_table_block", "type": "table", "tableId": "final_cell_detail_table", "layout": "full"},
        {
            "id": "confirmed_death_relation_result",
            "type": "markdown",
            "body": (
                "## Confirmed Death objects are decomposed by their final relation to cells\n\n"
                "Each branch contains 1,416 confirmed Death objects. In the original branch, 758 are `same_cell`, 341 are "
                "`overlapping_live_dead_multi_nucleus`, 269 are `live_with_death_signal`, 34 are `dead_only`, 7 have uncertain "
                "adjacent/overlapping attribution, and 7 are additional merged objects. The nucleated-only branch contains 715, "
                "346, 287, 59, 0, and 9 objects in those respective categories. These are object counts; they must not be read as "
                "mutually exclusive cell counts."
            ),
        },
        {"id": "death_relation_chart_block", "type": "chart", "chartId": "death_relation_chart", "layout": "full"},
        {
            "id": "death_relation_ratio_intro",
            "type": "markdown",
            "body": (
                "### Relation ratios within confirmed Death objects\n\n"
                "The following pies use **1,416 confirmed Death objects per branch** as the denominator. Each object contributes to "
                "exactly one final relation slice, so the percentages sum to 100% within each branch."
            ),
        },
        {
            "id": "death_relation_pies_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["death_relation_pies"],
                "Original and nucleated-only confirmed Death-object relation pies shown as two subplots with one shared legend",
                "Confirmed Death-object relation ratios. Both branch subplots use a shared legend and consistent colors.",
                *image_sizes["death_relation_pies"],
            ),
            "layout": "full",
        },
        {"id": "death_relation_table_block", "type": "table", "tableId": "death_relation_table", "layout": "full"},
        {
            "id": "uncertainty_result",
            "type": "markdown",
            "body": (
                "## Uncertainty is retained within live, dead, and Death-object annotations\n\n"
                "Cell uncertainty is operationally defined as medium/low classification confidence or uncertain linked-Death "
                "attribution. The original branch contains 10,234 uncertain live annotations (11.53% of live) and 344 uncertain "
                "dead annotations (29.55% of dead); the nucleated-only branch contains 3,973 (4.93%) and 189 (20.21%), respectively. "
                "Separately, 7/1,416 confirmed Death objects in the original branch have uncertain spatial attribution and none do "
                "in the nucleated-only branch. All 6,107 retained-but-unconfirmed candidates per branch remain explicitly marked "
                "as candidates rather than final Death objects."
            ),
        },
        {"id": "cell_uncertainty_chart_block", "type": "chart", "chartId": "cell_uncertainty_chart", "layout": "full"},
        {"id": "death_uncertainty_table_block", "type": "table", "tableId": "death_uncertainty_table", "layout": "full"},
        {
            "id": "count_result",
            "type": "markdown",
            "body": (
                "## Final counts distinguish cell-associated death from supplemental objects\n\n"
                "The primary downstream count is now `object_aware_dead_count`, the sum of `dead_cell_count` and confirmed "
                "supplemental objects. This produces 1,822 events in the original branch and 1,636 in the nucleated-only branch. "
                "The large reduction from the original 6,936 and 5,980 cell calls mainly reflects removal of unconditional "
                "Dead-channel overrides, but without manual labels the entire difference cannot be called corrected error."
            ),
        },
        {"id": "composition_chart_block", "type": "chart", "chartId": "composition_chart", "layout": "full"},
        {
            "id": "object_aware_death_ratio_intro",
            "type": "markdown",
            "body": (
                "### Ratios within the final object-aware death total\n\n"
                "The pies below use the complete final death total in each branch as the denominator: **1,822 events** in the "
                "Original branch and **1,636 events** in the Nucleated-only branch. Final dead cells and each supplemental "
                "Death-object relation are mutually exclusive count components within these totals."
            ),
        },
        {
            "id": "death_event_pies_image",
            "type": "html",
            "body": MAIN_REPORT.image_block_body(
                image_uris["death_event_pies"],
                "Original and nucleated-only final object-aware death-count pies shown as two subplots with one shared legend",
                "Final object-aware death-count ratios. Both branch subplots use a shared legend and consistent colors.",
                *image_sizes["death_event_pies"],
            ),
            "layout": "full",
        },
        {
            "id": "object_aware_event_detail_table_block",
            "type": "table",
            "tableId": "object_aware_event_detail_table",
            "layout": "full",
        },
        {"id": "composition_table_block", "type": "table", "tableId": "composition_table", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Final object-aware method and output semantics\n\n"
                "Every Dead mask is retained independently and assigned a relation: `same_cell`, `overlapping_live_dead_multi_nucleus`, "
                "`live_with_death_signal`, `adjacent_or_overlapping_dead_uncertain`, "
                "`adjacent_candidate`, `dead_only`, `merged_multiple_objects`, or an auditable rejected/unassigned state. "
                "Confirmation uses strong direct Dead-channel evidence or local RGB evidence measured inside the Dead mask rather "
                "than the RGB state of the entire Combined cell. RGB-live cells are protected from forced Dead-channel override; "
                "confirmed neighboring objects contribute through `supplemental_dead_object_count`. For RGB-uncertain cells, a "
                "Dead object changes the cell state only when it covers at least 45% of the Combined mask, or when the cell is "
                "compact (area at most 500 pixels) and coverage is at least 30%. Otherwise the cell remains live and the confirmed "
                "death object is preserved in a separate mask layer. Two or more nuclei support probable live/dead overlap; one "
                "nucleus is reported conservatively as a live cell with death signal."
            ),
        },
        {
            "id": "debugging_stage_provenance_text",
            "type": "markdown",
            "body": (
                "## Frozen intermediate stages make the development comparison reproducible\n\n"
                "The context-aware, strong-direct, and early object-aware results were created during iterative calibration before "
                "each working-tree state was committed separately. Their saved prediction tables and QC overlays are therefore "
                "treated as immutable historical references rather than silently regenerated with the final classifier. The table "
                "records the objective, thresholds, run template, outputs, and code-provenance status for every stage. Final v4 "
                "classification remains computationally reproducible from the recorded git revision and the complete d0 source trees."
            ),
            "sourceId": "stage_provenance",
        },
        {"id": "debugging_stage_table_block", "type": "table", "tableId": "debugging_stage_table", "layout": "full"},
        {"id": "output_table_block", "type": "table", "tableId": "output_table", "layout": "full"},
        {
            "id": "stress_test_result",
            "type": "markdown",
            "body": (
                "## The segmentation-external global detector was not production-ready\n\n"
                "An independent raw-signal seed detector recovered only 655/1,189 strong objects (55.09%) and only 293/1,189 "
                "synthetically relocated objects (24.64%). Although the shifted-seed live-cell hit rate stayed near 0.1%, recovery "
                "was far below the 99.95% operational requirement. The detector therefore remains a calibration stress test and "
                "was not allowed to create or modify production Dead masks."
            ),
        },
        {"id": "stress_chart_block", "type": "chart", "chartId": "stress_chart", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## What the current audit establishes—and what it cannot establish\n\n"
                "- **Established:** every existing upstream Dead mask is represented in the final ledger; every object meeting the "
                "fixed strong-reference definition is retained; no RGB-live cell is forcibly changed to dead by Dead-channel "
                "evidence; and both output types have complete 320/320 QC in each branch.\n"
                "- **Not established:** biological death sensitivity, biological live-cell specificity, accuracy of the upstream "
                "Dead segmentation, or correctness of every object-aware count.\n"
                "- **Reason:** no manual ground truth or independent death assay is available. A true dead cell with no upstream "
                "Dead mask is absent from the reference denominator, and the RGB-live rule can itself be wrong."
            ),
        },
        {
            "id": "recommended_use",
            "type": "markdown",
            "body": (
                "## Recommended downstream use and regression checks\n\n"
                "1. Use `object_aware_dead_count` as the primary d0 death summary.\n"
                "2. Retain `dead_cell_count` and `supplemental_dead_object_count` so every total remains decomposable.\n"
                "3. Use `final_multilevel_annotations.csv` for cohort-wide cell/Death analysis; use the per-image source tables only "
                "when tracing an individual field.\n"
                "4. Use `final_annotation_summary.csv` for reported classification totals because every row carries its explicit "
                "denominator and percentage.\n"
                "5. After any classifier change, require both branches to preserve the 8,289-object ledger, detect all 1,189 strong "
                "references, keep the per-field live override proxy below 0.5%, and produce complete 320/320 cell and object QC.\n"
                "6. Do not promote segmentation-external candidates until an independent detector passes both the high-recall and "
                "shifted-channel negative-control requirements."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "A limited expert review, positive-control wells, or an independent death marker would be required to estimate true "
                "sensitivity and specificity. Until such evidence exists, future work should focus on preserving the frozen "
                "segmentation and output invariants, auditing difficult object relations, and treating all reported accuracy values "
                "as operational proxies."
            ),
        },
    ]

    return {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Technical comparison of the original and iteratively improved SUM159 d0 Dead-classification results.",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]} for source in sources],
        "blocks": blocks,
    }


def build_artifact(
    datasets: dict[str, list[dict[str, Any]]],
    figures: dict[str, Image.Image],
    debugging_stage_provenance: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    sources = make_sources(
        datasets["expanded_case_evidence"],
        debugging_stage_provenance,
    )
    attempts = [(0.42, 74), (0.38, 68), (0.34, 62), (0.30, 56), (0.27, 50)]
    last_size = 0
    for scale, quality in attempts:
        image_uris: dict[str, str] = {}
        image_sizes: dict[str, tuple[int, int]] = {}
        image_bytes: dict[str, int] = {}
        image_sha256: dict[str, str] = {}
        for key, figure in figures.items():
            canonical = figure.resize(
                (max(1, round(figure.width * scale)), max(1, round(figure.height * scale))),
                Image.Resampling.LANCZOS,
            )
            uri, encoded_bytes, digest = MAIN_REPORT.encode_sheet(canonical, quality)
            image_uris[key] = uri
            image_sizes[key] = canonical.size
            image_bytes[key] = encoded_bytes
            image_sha256[key] = digest
        artifact = {
            "surface": "report",
            "manifest": build_manifest(generated_at, image_uris, image_sizes, sources),
            "snapshot": {
                "version": 1,
                "generatedAt": generated_at,
                "status": "ready",
                "datasets": datasets,
            },
            "sources": sources,
            "package_info": {
                "originUrl": "artifact://dead-d0-classification-audit/improvement-report",
                "controls": {"edit": False, "refresh": False, "share": False},
            },
        }
        last_size = len(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if last_size <= MAX_ARTIFACT_BYTES:
            return artifact, {
                "generated_at": generated_at,
                "artifact_bytes": last_size,
                "canonical_image_scale": scale,
                "canonical_image_quality": quality,
                "canonical_image_bytes": image_bytes,
                "canonical_image_sha256": image_sha256,
                "canonical_image_sizes": {key: list(size) for key, size in image_sizes.items()},
            }
    raise RuntimeError(f"Artifact remains too large: {last_size:,} > {MAX_ARTIFACT_BYTES:,} bytes")


def write_debug_figures(directory: Path, figures: dict[str, Image.Image]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for key, figure in figures.items():
        figure.save(directory / f"{key}.png", format="PNG", dpi=(600, 600))


def chart_map() -> list[dict[str, Any]]:
    return [
        {
            "section": "Iterative classifier performance",
            "question": "How did the same RGB-live override proxy change across the four classifier stages?",
            "family": "Comparison and ranking",
            "type": "grouped bar",
            "fields": ["stage", "branch", "live_false_positive_proxy"],
            "claim": "Object-aware separation removes the threshold tradeoff and ends forced RGB-live overrides.",
        },
        {
            "section": "Automated reference recovery",
            "question": "Which stages retain the available automated death references?",
            "family": "Comparison and benchmark",
            "type": "grouped bar plus exact table",
            "fields": ["stage", "branch", "operational_recall", "reference_unit"],
            "claim": "Final recovery is complete for the broader object-level reference, with denominator differences disclosed.",
        },
        {
            "section": "Object-ledger completeness",
            "question": "How much of the fixed 8,289-object source is represented by each output structure?",
            "family": "Progression",
            "type": "grouped bar",
            "fields": ["structure", "branch", "representation_rate"],
            "claim": "The complete ledger eliminates unassigned and same-cell-collision losses.",
        },
        {
            "section": "Final count composition",
            "question": "What contributes to the final object-aware death total?",
            "family": "Composition",
            "type": "stacked bar",
            "fields": ["branch", "component", "count"],
            "claim": "The final total remains decomposable into cell-associated and supplemental objects.",
        },
        {
            "section": "Final count composition ratios",
            "question": "What percentage of the final object-aware death total comes from dead cells and each supplemental relation?",
            "family": "Composition",
            "type": "two-subplot donut figure with one shared legend",
            "fields": ["component", "share_of_object_aware_total", "count", "object_aware_total"],
            "claim": "The two branch subplots use one shared legend and consistent colors while partitioning each full final death total into mutually exclusive count components.",
        },
        {
            "section": "Final cell classifications",
            "question": "How are all Combined cells distributed across final states?",
            "family": "Composition",
            "type": "grouped percentage bar plus exact table",
            "fields": ["branch", "state", "count", "share_of_all_cells"],
            "claim": "Final states are reported once per cell, including zero transitional/uncertain categories.",
        },
        {
            "section": "Confirmed Death-object relations",
            "question": "What final relation explains each confirmed Death object?",
            "family": "Composition",
            "type": "stacked bar plus exact table",
            "fields": ["branch", "relation", "count", "share_of_confirmed_death"],
            "claim": "Confirmed Death objects separate same-cell death from multi-nucleus overlap and live-with-death signal.",
        },
        {
            "section": "Confirmed Death-object relation ratios",
            "question": "What percentage of confirmed Death objects belongs to each mutually exclusive final relation in each branch?",
            "family": "Composition",
            "type": "two-subplot donut figure with one shared legend",
            "fields": ["relation", "share_of_confirmed_death", "count", "confirmed_death_objects"],
            "claim": "The shared legend and branch-specific 1,416-object denominators make the two relation mixes directly comparable as percentages.",
        },
        {
            "section": "Classification uncertainty",
            "question": "How many final live and dead cell annotations are not high-confidence or have uncertain Death attribution?",
            "family": "Uncertainty and comparison",
            "type": "grouped percentage bar",
            "fields": ["branch", "final_state", "uncertain_count", "uncertainty_rate"],
            "claim": "Uncertainty remains visible inside both live and dead final states instead of being hidden by the final label.",
        },
        {
            "section": "Independent detector stress test",
            "question": "Did the segmentation-external detector satisfy the 99.95% recovery requirement?",
            "family": "Uncertainty and benchmark",
            "type": "grouped benchmark bar",
            "fields": ["test", "series", "rate"],
            "claim": "Observed recovery was far below the production requirement, so the detector was not deployed.",
        },
    ]


def main() -> int:
    args = parse_args()
    artifact_path = args.artifact_json.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    plugin_root = require_dir((args.plugin_root or discover_plugin_root()).expanduser().resolve())

    if args.package_only:
        high_resolution = None
        if args.high_resolution_images_json is not None:
            high_resolution = MAIN_REPORT.read_json(args.high_resolution_images_json.expanduser().resolve())
        enhancement = MAIN_REPORT.package_html(
            artifact_path,
            output_html,
            plugin_root,
            args.force,
            high_resolution,
        )
        print(json.dumps({"artifact_json": str(artifact_path), "output_html": str(output_html), "html_enhancement": enhancement}, indent=2))
        return 0

    requested_audit_root = require_dir(args.audit_root.expanduser().resolve())
    audit_root = requested_audit_root
    historical_reference_root: Path | None = None
    report_input_map: Path | None = None
    if (args.input_root is None) != (args.result_root is None):
        raise ValueError("--input-root and --result-root must be supplied together")
    selected_timepoint = normalize_timepoint(args.timepoint)
    source_inventory = None
    if args.input_root is not None and args.result_root is not None:
        source_inventory = validate_source_inventory(
            args.input_root.expanduser().resolve(),
            args.result_root.expanduser().resolve(),
            selected_timepoint,
        )
    current_layout = CURRENT_RUN_REPORT.is_current_run_layout(requested_audit_root)
    if args.report_mode == "current-run" and not current_layout:
        raise RuntimeError(f"Explicit current-run mode requires the current HPC directory layout: {requested_audit_root}")
    use_current_report = args.report_mode == "current-run" or (
        args.report_mode == "auto" and current_layout
    )
    if use_current_report:
        receipt = CURRENT_RUN_REPORT.build_current_report(
            args,
            MAIN_REPORT,
            plugin_root,
            source_inventory,
        )
        print(json.dumps(receipt, indent=2))
        return 0
    if args.report_mode == "historical-comparison" and current_layout:
        historical_reference_root = require_dir(
            (args.historical_reference_root or requested_audit_root / "historical_reference")
            .expanduser()
            .resolve()
        )
        audit_root, report_input_map = prepare_historical_comparison_root(
            requested_audit_root,
            historical_reference_root,
        )
    methods_md = require_file(args.methods_md.expanduser().resolve())
    if "Technical Summary" not in methods_md.read_text():
        raise RuntimeError(f"The methods comparison Markdown does not contain the expected technical summary: {methods_md}")
    for path in (artifact_path, output_html, args.build_receipt.expanduser().resolve()):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {path}")

    debugging_stage_provenance = audit_root / "debugging_stage_provenance.json"
    if not debugging_stage_provenance.is_file():
        debugging_stage_provenance = Path(__file__).resolve().with_name(
            "dead_classification_debugging_stages.json"
        )
    debugging_stage_provenance = require_file(debugging_stage_provenance)

    live_cases, miss_cases = select_expanded_cases(audit_root)
    figures, sentinel_rows = render_report_figures(audit_root, live_cases, miss_cases)
    datasets = build_datasets(
        audit_root,
        debugging_stage_provenance,
        sentinel_rows,
        live_cases,
        miss_cases,
    )
    artifact, receipt = build_artifact(datasets, figures, debugging_stage_provenance)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    MAIN_REPORT.atomic_write_json(artifact_path, artifact)

    high_resolution = MAIN_REPORT.build_high_resolution_image_payload(figures)
    if args.high_resolution_images_json is not None:
        sidecar = args.high_resolution_images_json.expanduser().resolve()
        if sidecar.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {sidecar}")
        MAIN_REPORT.atomic_write_json(sidecar, high_resolution)
    if args.debug_figure_dir is not None:
        write_debug_figures(args.debug_figure_dir.expanduser().resolve(), figures)

    enhancement = MAIN_REPORT.package_html(
        artifact_path,
        output_html,
        plugin_root,
        args.force,
        high_resolution,
    )
    receipt.update(
        {
            "report_mode": "historical_comparison",
            "report": str(output_html),
            "artifact_json": str(artifact_path),
            "methods_markdown": str(methods_md),
            "audit_root": str(audit_root),
            "requested_audit_root": str(requested_audit_root),
            "historical_reference_root": (
                str(historical_reference_root) if historical_reference_root is not None else None
            ),
            "report_input_map": str(report_input_map) if report_input_map is not None else None,
            "debugging_stage_provenance": str(
                debugging_stage_provenance
            ),
            "source_inventory": source_inventory,
            "selected_timepoint": selected_timepoint,
            "plugin_root": str(plugin_root),
            "dataset_rows": {key: len(rows) for key, rows in datasets.items()},
            "sentinel_fields": [row["sample"] for row in sentinel_rows],
            "expanded_live_override_fields": [row["sample"] for row in live_cases],
            "expanded_death_miss_fields": [row["sample"] for row in miss_cases],
            "high_resolution_image_bytes": sum(
                int(image["encoded_bytes"]) for image in high_resolution["images"].values()
            ),
            "high_resolution_image_sizes": {
                key: [image["width"], image["height"]]
                for key, image in high_resolution["images"].items()
            },
            "html_enhancement": enhancement,
            "chart_map": chart_map(),
        }
    )
    receipt_path = args.build_receipt.expanduser().resolve()
    MAIN_REPORT.atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        os._exit(1)
