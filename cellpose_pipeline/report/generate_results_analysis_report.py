#!/usr/bin/env python3
"""Build the full-cohort SUM159 analysis report with embedded QC figures.

The scientific outputs are read-only inputs. The script selects one fixed set
of three high-density and three low-density fields, renders one segmentation
composite and one classification/shape_strict composite per field, embeds the
local production-workflow diagram and repository methods, and packages one
self-contained HTML file with the Data Analytics portable report builder.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

_IMAGE_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw, ImageFont, features as pil_features
except ModuleNotFoundError as exc:
    # Packaging an existing artifact does not require the image-analysis stack.
    # The normal generation path checks this sentinel before reading any TIFF.
    _IMAGE_IMPORT_ERROR = exc


KEY_RE = re.compile(r"([A-H]\d+_\d+_\d{2}d\d{2}h\d{2}m)")
MAX_ARTIFACT_BYTES = 2_850_000
QC_RENDER_SCALE = 6
HIGH_RES_PRINT_PPI = 600
HIGH_RES_WEBP_QUALITY = 96
HIGH_RES_JPEG_QUALITY = 98
CAROUSEL_IMAGE_HEIGHT_CAP = 1800

BG = (18, 20, 24)
PANEL_BG = (29, 32, 38)
TEXT = (245, 247, 250)
MUTED = (184, 191, 202)
CYAN = (35, 220, 235)
MAGENTA = (245, 72, 205)
YELLOW = (255, 220, 55)
GREEN = (45, 205, 110)
RED = (235, 70, 72)
BLUE = (70, 145, 245)
GRAY = (145, 150, 160)
WHITE = (250, 250, 250)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW_PDF = REPO_ROOT / "docs" / "recommended_full_production_workflow.pdf"
DEFAULT_METHODS_README = REPO_ROOT / "README.md"
DEFAULT_REPORT_RUN_NAME = "full_fusion_shape_strict_20260711_155940"
DEFAULT_LOCAL_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_LOCAL_REPORT_RUN_ROOT = DEFAULT_LOCAL_RESULTS_ROOT / DEFAULT_REPORT_RUN_NAME / "report_inputs_6samples"
DEFAULT_HPC_PROJECT_DIR = Path(
    "/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/"
    "SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3"
)
DEFAULT_HPC_RESULT_ROOT = Path(
    "/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/"
    "SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/"
    "20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940"
)


@dataclass(frozen=True)
class Sample:
    label: str
    key: str
    density_class: str
    selection_reason: str
    manifest: dict[str, str]
    density: dict[str, str]
    fusion_all: dict[str, str]
    fusion_nucleated: dict[str, str]
    dead: dict[str, str]
    shape_all: dict[str, str]
    shape_nucleated: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_LOCAL_REPORT_RUN_ROOT,
        help=(
            "Compact local report snapshot root. Defaults to the v3 six-sample snapshot under "
            f"{DEFAULT_LOCAL_RESULTS_ROOT}."
        ),
    )
    parser.add_argument("--artifact-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument(
        "--workflow-pdf",
        type=Path,
        default=DEFAULT_WORKFLOW_PDF,
        help="One-page production-workflow PDF embedded in the Methods chapter.",
    )
    parser.add_argument(
        "--methods-readme",
        type=Path,
        default=DEFAULT_METHODS_README,
        help="Detailed Markdown methods source integrated into the report.",
    )
    parser.add_argument(
        "--plugin-root",
        type=Path,
        help="Data Analytics plugin root containing package.json and report:deliver.",
    )
    parser.add_argument(
        "--sample-keys",
        nargs=6,
        metavar=("HD1", "HD2", "HD3", "LD1", "LD2", "LD3"),
        help="Optional fixed keys. First three must be high density; last three low density.",
    )
    parser.add_argument(
        "--debug-qc-dir",
        type=Path,
        help="Optional QA-only directory for decoded contact sheets. HTML never depends on it.",
    )
    parser.add_argument(
        "--high-resolution-images-json",
        type=Path,
        help=(
            "Optional temporary sidecar carrying 6x high-resolution report images between rendering "
            "and local HTML packaging. Images are embedded into the final HTML."
        ),
    )
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with require_file(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with require_file(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def index_rows(path: Path, key: str = "key", delimiter: str = ",") -> dict[str, dict[str, str]]:
    rows = read_rows(path, delimiter=delimiter)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise ValueError(f"Missing {key!r} in {path}")
        if value in result:
            raise ValueError(f"Duplicate {key}={value!r} in {path}")
        result[value] = row
    return result


def float_value(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(row: dict[str, str], field: str, default: int = 0) -> int:
    return int(round(float_value(row, field, float(default))))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def extract_key(value: str | Path) -> str:
    match = KEY_RE.search(str(value))
    if match is None:
        raise ValueError(f"Cannot extract field key from {value}")
    return match.group(1)


def safe_source_path(run_root: Path, relative: str) -> str:
    return f"results/{report_snapshot_name(run_root)}/{relative.strip('/')}"


def report_snapshot_name(run_root: Path) -> str:
    if run_root.name == "report_inputs_6samples" and run_root.parent.name:
        return run_root.parent.name
    return run_root.name


def portable_hpc_path(path: Path) -> str:
    return f"hpc:{path}"


def resolve_report_path(run_root: Path, path_text: str) -> str:
    """Resolve stale production paths against a compact local report snapshot."""
    original = Path(path_text)
    if original.is_file():
        return str(original)

    marker = f"/{run_root.name}/"
    normalized = str(original)
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = run_root / relative
        if candidate.is_file():
            return str(candidate)

    channel = original.parent.name
    if channel in {"Brightfield", "Combined", "Dead", "Nuclei"}:
        candidate = run_root / "raw_input" / channel / original.name
        if candidate.is_file():
            return str(candidate)

    for anchor in (
        "shape_strict_nucleated_only",
        "shape_strict",
        "classification_fusion_nucleated_only",
        "classification_fusion",
        "nucleated_only",
        "Brightfield",
        "Combined",
        "Dead",
        "Nuclei",
        "workflow_status",
    ):
        anchor_marker = f"/{anchor}/"
        if anchor_marker not in normalized:
            continue
        relative = anchor + "/" + normalized.rsplit(anchor_marker, 1)[1]
        candidate = run_root / relative
        if candidate.is_file():
            return str(candidate)
    return path_text


def resolve_row_paths(run_root: Path, rows: dict[str, dict[str, str]], fields: list[str]) -> None:
    for row in rows.values():
        for field in fields:
            value = row.get(field, "")
            if value:
                row[field] = resolve_report_path(run_root, value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def qc_px(value: int | float) -> int:
    return max(1, int(round(float(value) * QC_RENDER_SCALE)))


def qc_size(size: tuple[int, int]) -> tuple[int, int]:
    return qc_px(size[0]), qc_px(size[1])


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=qc_px(size))
    return ImageFont.load_default()


@lru_cache(maxsize=128)
def read_tiff_cached(path_text: str) -> np.ndarray:
    return np.asarray(tifffile.imread(path_text))


def read_tiff(path: str | Path) -> np.ndarray:
    return read_tiff_cached(str(require_file(Path(path))))


def squeeze_2d(array: np.ndarray) -> np.ndarray:
    result = np.squeeze(array)
    if result.ndim != 2:
        raise ValueError(f"Expected 2D image, found shape={array.shape}")
    return result


def boundaries(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def normalize_scalar(raw: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    data = np.asarray(raw, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        norm = np.zeros(data.shape, dtype=np.float32)
    else:
        lo, hi = np.percentile(finite, [low, high])
        norm = np.clip((data - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.repeat((norm[..., None] * 255.0).astype(np.uint8), 3, axis=2)


def normalize_nuclei(raw: np.ndarray) -> np.ndarray:
    gray = normalize_scalar(raw)[..., 0].astype(np.float32) / 255.0
    rgb = np.stack((gray, gray * 0.22, gray * 0.22), axis=2)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def normalize_rgb(raw: np.ndarray) -> np.ndarray:
    data = np.asarray(raw)
    if data.ndim == 2:
        return normalize_scalar(data)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError(f"Expected RGB image, found shape={data.shape}")
    rgb = data[..., :3].astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    lo, hi = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
    return np.clip((rgb - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)


def paint_edge(rgb: np.ndarray, edge: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = np.asarray(rgb).copy()
    result[edge] = np.asarray(color, dtype=np.uint8)
    return result


def alpha_fill(rgb: np.ndarray, region: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = np.asarray(rgb).copy().astype(np.float32)
    if np.any(region):
        result[region] = (1.0 - alpha) * result[region] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def fit_image(array: np.ndarray | Image.Image, size: tuple[int, int]) -> Image.Image:
    image = array if isinstance(array, Image.Image) else Image.fromarray(np.asarray(array, dtype=np.uint8))
    target_w, target_h = qc_size(size)
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (target_w, target_h), BG)
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas


def label_image(image: Image.Image, label: str, height: int = 22) -> Image.Image:
    height = qc_px(height)
    canvas = Image.new("RGB", (image.width, image.height + height), PANEL_BG)
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((qc_px(6), qc_px(3)), label, fill=TEXT, font=font(13, bold=True))
    return canvas


def count_labels(mask: np.ndarray) -> int:
    labels = np.unique(mask)
    return int(np.sum(labels > 0))


def sample_header(sample: Sample) -> str:
    nuclei = int_value(sample.density, "nuclei_count")
    nn = float_value(sample.density, "nuclei_median_nn_px", float("nan"))
    nn_text = "NA" if not math.isfinite(nn) else f"{nn:.1f}px"
    return f"{sample.label} | {sample.key} | nuclei {nuclei:,} | median NN {nn_text}"


def make_tile(body: Image.Image, sample: Sample, detail: str, width: int = 430) -> Image.Image:
    width = qc_px(width)
    header_h = qc_px(54)
    canvas = Image.new("RGB", (width, body.height + header_h), PANEL_BG)
    canvas.paste(body, ((width - body.width) // 2, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((qc_px(8), qc_px(6)), sample_header(sample), fill=TEXT, font=font(14, bold=True))
    draw.text((qc_px(8), qc_px(30)), detail, fill=MUTED, font=font(12))
    return canvas


def contact_sheet(
    samples: list[Sample],
    render_tile: Callable[[Sample], Image.Image],
    legend: str,
) -> Image.Image:
    if len(samples) != 6:
        raise ValueError(f"Exactly six QC samples are required, found {len(samples)}")
    tiles = [render_tile(sample) for sample in samples]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    gap = qc_px(10)
    legend_h = qc_px(34)
    sheet = Image.new("RGB", (3 * tile_w + 4 * gap, legend_h + 2 * tile_h + 3 * gap), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, qc_px(8)), legend, fill=TEXT, font=font(14, bold=True))
    for index, tile in enumerate(tiles):
        x = gap + (index % 3) * (tile_w + gap)
        y = legend_h + gap + (index // 3) * (tile_h + gap)
        sheet.paste(tile, (x, y))
    return sheet


def load_result_tables(run_root: Path) -> dict[str, Any]:
    manifest_dir = run_root / "workflow_status" / "postsegmentation_manifest"
    manifest_rows = index_rows(manifest_dir / "field_manifest.tsv", delimiter="\t")
    shape_all = index_rows(run_root / "shape_strict" / "field_summary.csv")
    shape_nucleated = index_rows(run_root / "shape_strict_nucleated_only" / "field_summary.csv")
    resolve_row_paths(
        run_root,
        manifest_rows,
        [
            "combined_raw",
            "combined_mask",
            "brightfield_raw",
            "brightfield_mask",
            "dead_raw",
            "dead_mask",
            "nuclei_raw",
            "nuclei_extent_mask",
            "nuclei_core_mask",
            "nucleated_combined_mask",
            "nucleated_brightfield_mask",
        ],
    )
    resolve_row_paths(run_root, shape_all, ["extent_mask", "core_mask"])
    resolve_row_paths(run_root, shape_nucleated, ["extent_mask", "core_mask"])
    return {
        "manifest_rows": manifest_rows,
        "manifest_summary": read_json(manifest_dir / "manifest_summary.json"),
        "density": index_rows(run_root / "qc" / "density_calls.csv"),
        "dead_fields": index_rows(run_root / "Dead" / "dead_consensus_field_summary.csv"),
        "dead_summary": read_json(run_root / "Dead" / "dead_consensus_summary.json"),
        "fusion_all_rows": read_rows(run_root / "classification_fusion" / "summaries" / "cell_count_summary.csv"),
        "fusion_all": index_rows(run_root / "classification_fusion" / "summaries" / "cell_count_summary.csv"),
        "fusion_nucleated_rows": read_rows(run_root / "classification_fusion_nucleated_only" / "summaries" / "cell_count_summary.csv"),
        "fusion_nucleated": index_rows(run_root / "classification_fusion_nucleated_only" / "summaries" / "cell_count_summary.csv"),
        "nucleated_summary": read_json(run_root / "nucleated_only" / "branch_summary.json"),
        "shape_all": shape_all,
        "shape_all_summary": read_json(run_root / "shape_strict" / "shape_strict_summary.json"),
        "shape_nucleated": shape_nucleated,
        "shape_nucleated_summary": read_json(run_root / "shape_strict_nucleated_only" / "shape_strict_summary.json"),
        "calibration": read_json(run_root / "dead_preprocess_calibration" / "final" / "dead_combined_blue_calibration.json"),
    }


def candidate_keys(tables: dict[str, Any]) -> list[str]:
    sets = [
        set(tables["manifest_rows"]),
        set(tables["density"]),
        set(tables["dead_fields"]),
        set(tables["fusion_all"]),
        set(tables["fusion_nucleated"]),
        set(tables["shape_all"]),
        set(tables["shape_nucleated"]),
    ]
    keys = sorted(set.intersection(*sets))
    if not keys:
        raise RuntimeError("No common field keys across required report inputs")
    return keys


def density_score(row: dict[str, str]) -> float:
    count = max(float_value(row, "nuclei_count", 0.0), 0.0)
    nn = max(float_value(row, "nuclei_median_nn_px", 999.0), 1e-3)
    return math.log1p(count) - 0.35 * math.log(nn)


def select_quantile_keys(
    keys: list[str],
    tables: dict[str, Any],
    high_density: bool,
) -> list[tuple[str, str]]:
    density = tables["density"]
    shape = tables["shape_all"]
    fusion = tables["fusion_all"]
    candidates = [key for key in keys if truthy(density[key].get("high_density")) == high_density]
    if len(candidates) < 3:
        raise RuntimeError(f"Need three {'high' if high_density else 'low'}-density fields, found {len(candidates)}")
    ordered = sorted(candidates, key=lambda key: (density_score(density[key]), key))
    targets = (0.15, 0.50, 0.85)
    selected: list[str] = []
    used_wells: set[str] = set()
    reasons: dict[str, str] = {}
    for target in targets:
        index = int(round(target * (len(ordered) - 1)))
        ranked = sorted(
            range(len(ordered)),
            key=lambda idx: (abs(idx - index), ordered[idx]),
        )
        choice = next(
            (
                ordered[idx]
                for idx in ranked
                if ordered[idx] not in selected and fusion[ordered[idx]].get("well", "") not in used_wells
            ),
            next(ordered[idx] for idx in ranked if ordered[idx] not in selected),
        )
        selected.append(choice)
        used_wells.add(fusion[choice].get("well", ""))
        reasons[choice] = f"density-score quantile {target:.2f}"

    if not any(int_value(shape[key], "split_events") > 0 for key in selected):
        split_candidates = [key for key in ordered if int_value(shape[key], "split_events") > 0 and key not in selected]
        if split_candidates:
            median_index = (len(ordered) - 1) / 2.0
            index_by_key = {key: idx for idx, key in enumerate(ordered)}
            replacement = min(split_candidates, key=lambda key: (abs(index_by_key[key] - median_index), key))
            replaced = selected[1]
            selected[1] = replacement
            reasons.pop(replaced, None)
            reasons[replacement] = "median-density field with at least one accepted shape_strict split"

    selected = sorted(selected, key=lambda key: (density_score(density[key]), key))
    return [(key, reasons[key]) for key in selected]


def validate_selected_key(key: str, tables: dict[str, Any], run_root: Path) -> None:
    row = tables["manifest_rows"][key]
    required_fields = [
        "combined_raw",
        "combined_mask",
        "brightfield_raw",
        "brightfield_mask",
        "dead_raw",
        "dead_mask",
        "nuclei_raw",
        "nuclei_extent_mask",
        "nuclei_core_mask",
        "nucleated_combined_mask",
        "nucleated_brightfield_mask",
    ]
    for field in required_fields:
        require_file(Path(row[field]))
    require_file(Path(tables["shape_all"][key]["extent_mask"]))
    require_file(Path(tables["shape_all"][key]["core_mask"]))
    require_file(Path(tables["shape_nucleated"][key]["extent_mask"]))
    require_file(Path(tables["shape_nucleated"][key]["core_mask"]))
    combined_stem = Path(row["combined_raw"]).stem
    require_file(run_root / "classification_fusion" / "predictions" / f"{combined_stem}_per_cell_predictions.csv")
    require_file(run_root / "classification_fusion_nucleated_only" / "predictions" / f"{combined_stem}_per_cell_predictions.csv")
    require_file(run_root / "Dead" / "intermediate" / "dead_primary" / f"{key}_cp_masks.tif")
    require_file(run_root / "Dead" / "intermediate" / "combined_blue" / f"{key}_cp_masks.tif")
    require_file(run_root / "Dead" / "object_provenance" / f"{key}.csv")


def build_samples(
    tables: dict[str, Any],
    run_root: Path,
    fixed_keys: list[str] | None,
) -> list[Sample]:
    keys = candidate_keys(tables)
    if fixed_keys:
        if len(set(fixed_keys)) != 6:
            raise ValueError("--sample-keys must contain six unique keys")
        missing = [key for key in fixed_keys if key not in keys]
        if missing:
            raise KeyError(f"Requested QC keys are not complete across all inputs: {missing}")
        chosen = [(key, "explicit --sample-keys selection") for key in fixed_keys]
    else:
        chosen = select_quantile_keys(keys, tables, True) + select_quantile_keys(keys, tables, False)

    samples: list[Sample] = []
    for index, (key, reason) in enumerate(chosen):
        expected_high = index < 3
        actual_high = truthy(tables["density"][key].get("high_density"))
        if actual_high != expected_high:
            raise ValueError(
                f"QC order must be three high-density then three low-density fields; key={key} high_density={actual_high}"
            )
        validate_selected_key(key, tables, run_root)
        samples.append(
            Sample(
                label=("HD" if actual_high else "LD") + f"-{index + 1 if actual_high else index - 2}",
                key=key,
                density_class="High density" if actual_high else "Low density",
                selection_reason=reason,
                manifest=tables["manifest_rows"][key],
                density=tables["density"][key],
                fusion_all=tables["fusion_all"][key],
                fusion_nucleated=tables["fusion_nucleated"][key],
                dead=tables["dead_fields"][key],
                shape_all=tables["shape_all"][key],
                shape_nucleated=tables["shape_nucleated"][key],
            )
        )
    return samples


def render_nuclei_tile(sample: Sample) -> Image.Image:
    raw = squeeze_2d(read_tiff(sample.manifest["nuclei_raw"]))
    extent = squeeze_2d(read_tiff(sample.manifest["nuclei_extent_mask"]))
    core = squeeze_2d(read_tiff(sample.manifest["nuclei_core_mask"]))
    overlay = normalize_nuclei(raw)
    overlay = alpha_fill(overlay, core > 0, CYAN, 0.22)
    overlay = paint_edge(overlay, boundaries(extent), MAGENTA)
    overlay = paint_edge(overlay, boundaries(core), CYAN)
    body = fit_image(overlay, (430, 318))
    detail = f"extent {count_labels(extent):,} | core {count_labels(core):,} | magenta extent, cyan core"
    return make_tile(body, sample, detail)


def render_cell_segmentation_tile(sample: Sample, profile: str) -> Image.Image:
    if profile == "Brightfield":
        raw = squeeze_2d(read_tiff(sample.manifest["brightfield_raw"]))
        mask = squeeze_2d(read_tiff(sample.manifest["brightfield_mask"]))
        overlay = normalize_scalar(raw)
    elif profile == "Combined":
        raw = read_tiff(sample.manifest["combined_raw"])
        mask = squeeze_2d(read_tiff(sample.manifest["combined_mask"]))
        overlay = normalize_rgb(raw)
    else:
        raise ValueError(profile)
    overlay = paint_edge(overlay, boundaries(mask), GREEN)
    body = fit_image(overlay, (430, 318))
    density_profile = "high-density profile" if sample.density_class.startswith("High") else "baseline profile"
    detail = f"{count_labels(mask):,} objects | {density_profile} | green instance boundaries"
    return make_tile(body, sample, detail)


def provenance_map(path: Path) -> dict[int, str]:
    rows = read_rows(path)
    return {int(row["final_id"]): row["source"] for row in rows}


def small_panel(array: np.ndarray, label: str, size: tuple[int, int] = (204, 142)) -> Image.Image:
    return label_image(fit_image(array, size), label, height=21)


def render_dead_tile(sample: Sample, run_root: Path) -> Image.Image:
    dead_raw = squeeze_2d(read_tiff(sample.manifest["dead_raw"]))
    combined_raw = read_tiff(sample.manifest["combined_raw"])
    primary = squeeze_2d(read_tiff(run_root / "Dead" / "intermediate" / "dead_primary" / f"{sample.key}_cp_masks.tif"))
    blue_mask = squeeze_2d(read_tiff(run_root / "Dead" / "intermediate" / "combined_blue" / f"{sample.key}_cp_masks.tif"))
    final_mask = squeeze_2d(read_tiff(sample.manifest["dead_mask"]))

    dead_panel = normalize_scalar(dead_raw)
    dead_panel = paint_edge(dead_panel, boundaries(primary), RED)

    combined_float = np.asarray(combined_raw[..., :3], dtype=np.float32)
    red_channel, green_channel, blue_channel = (
        combined_float[..., 0],
        combined_float[..., 1],
        combined_float[..., 2],
    )
    blue_excess = np.clip(blue_channel - 0.5 * (red_channel + green_channel), 0.0, None)
    blue_panel = normalize_scalar(blue_excess, 0.0, 99.8)
    blue_panel = paint_edge(blue_panel, boundaries(blue_mask), CYAN)

    combined_panel = normalize_rgb(combined_raw)
    consensus = combined_panel.copy()
    source_colors = {
        "dual_channel": GREEN,
        "dead_primary_only": RED,
        "combined_blue_rescue": BLUE,
    }
    final_edges = boundaries(final_mask)
    for label_id, source in provenance_map(run_root / "Dead" / "object_provenance" / f"{sample.key}.csv").items():
        consensus[final_edges & (final_mask == label_id)] = source_colors.get(source, WHITE)

    panels = [
        small_panel(dead_panel, "Dead raw + primary"),
        small_panel(blue_panel, "Blue excess + candidate"),
        small_panel(combined_panel, "Combined RGB"),
        small_panel(consensus, "Final consensus provenance"),
    ]
    panel_gap = qc_px(6)
    body = Image.new("RGB", (qc_px(430), 2 * panels[0].height + panel_gap), BG)
    for index, panel in enumerate(panels):
        body.paste(panel, ((index % 2) * qc_px(215), (index // 2) * (panel.height + panel_gap)))
    detail = (
        f"primary {int_value(sample.dead, 'dead_primary_objects'):,} | final {int_value(sample.dead, 'final_objects'):,} | "
        f"dual {int_value(sample.dead, 'matched_objects'):,} | rescue {int_value(sample.dead, 'combined_blue_rescued_objects'):,}"
    )
    return make_tile(body, sample, detail)


def prediction_path(run_root: Path, branch: str, sample: Sample) -> Path:
    combined_stem = Path(sample.manifest["combined_raw"]).stem
    return run_root / branch / "predictions" / f"{combined_stem}_per_cell_predictions.csv"


def render_classification_tile(sample: Sample, run_root: Path, nucleated: bool) -> Image.Image:
    branch = "classification_fusion_nucleated_only" if nucleated else "classification_fusion"
    mask_path = sample.manifest["nucleated_combined_mask" if nucleated else "combined_mask"]
    mask = squeeze_2d(read_tiff(mask_path)).astype(np.int64, copy=False)
    core = squeeze_2d(read_tiff(sample.manifest["nuclei_core_mask"]))
    raw = normalize_rgb(read_tiff(sample.manifest["combined_raw"]))
    predictions = read_rows(prediction_path(run_root, branch, sample))
    state_by_id = {int(row["mask_id"]): row["state"] for row in predictions}
    state_colors = {"live": GREEN, "dead": RED, "artifact": GRAY}
    max_label = int(mask.max()) if mask.size else 0
    lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    active = np.zeros(max_label + 1, dtype=bool)
    for label_id, state in state_by_id.items():
        if 0 < label_id <= max_label:
            lut[label_id] = state_colors.get(state, GRAY)
            active[label_id] = True
    overlay = raw.astype(np.float32)
    active_pixels = active[mask]
    colors = lut[mask]
    overlay[active_pixels] = 0.66 * overlay[active_pixels] + 0.34 * colors[active_pixels]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    state_mask = np.zeros(mask.shape, dtype=np.uint8)
    for code, state in enumerate(("live", "dead", "artifact"), start=1):
        ids = [label_id for label_id, value in state_by_id.items() if value == state]
        if ids:
            state_mask[np.isin(mask, np.asarray(ids, dtype=mask.dtype))] = code
    state_edge = boundaries(state_mask)
    for code, state in enumerate(("live", "dead", "artifact"), start=1):
        overlay[state_edge & (state_mask == code)] = state_colors[state]
    overlay[boundaries(core)] = np.asarray(YELLOW, dtype=np.uint8)
    body = fit_image(overlay, (430, 318))
    counts = {state: sum(value == state for value in state_by_id.values()) for state in ("live", "dead", "artifact")}
    detail = (
        f"live {counts['live']:,} | dead {counts['dead']:,} | artifact {counts['artifact']:,} | "
        "yellow nucleus core"
    )
    return make_tile(body, sample, detail)


def render_shape_tile(sample: Sample, nucleated: bool) -> Image.Image:
    raw = normalize_nuclei(squeeze_2d(read_tiff(sample.manifest["nuclei_raw"])))
    baseline = squeeze_2d(read_tiff(sample.manifest["nuclei_extent_mask"]))
    baseline_core = squeeze_2d(read_tiff(sample.manifest["nuclei_core_mask"]))
    shape_row = sample.shape_nucleated if nucleated else sample.shape_all
    strict = squeeze_2d(read_tiff(shape_row["extent_mask"]))
    strict_core = squeeze_2d(read_tiff(shape_row["core_mask"]))

    changed = (strict != baseline) & (baseline > 0)
    view = (slice(None), slice(None))
    view_label = "full field"
    if np.any(changed):
        y, x = np.where(changed)
        height, width = baseline.shape
        center_y = int(round((int(y.min()) + int(y.max())) / 2.0))
        center_x = int(round((int(x.min()) + int(x.max())) / 2.0))
        changed_span = max(int(y.max() - y.min() + 1), int(x.max() - x.min() + 1))
        crop_size = min(max(320, changed_span + 160), height, width)
        y0 = min(max(center_y - crop_size // 2, 0), height - crop_size)
        x0 = min(max(center_x - crop_size // 2, 0), width - crop_size)
        view = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
        view_label = "split zoom"

    before = paint_edge(raw, boundaries(baseline), MAGENTA)
    before = paint_edge(before, boundaries(baseline_core), CYAN)
    after = paint_edge(raw, boundaries(strict), MAGENTA)
    after = paint_edge(after, boundaries(strict_core), CYAN)
    if np.any(changed):
        after[boundaries(changed.astype(np.uint8))] = np.asarray(YELLOW, dtype=np.uint8)

    before = before[view]
    after = after[view]

    panels = [
        label_image(fit_image(before, (207, 235)), f"Baseline | {view_label}"),
        label_image(fit_image(after, (207, 235)), f"Strict | {view_label}"),
    ]
    body = Image.new("RGB", (qc_px(430), panels[0].height), BG)
    body.paste(panels[0], (0, 0))
    body.paste(panels[1], (qc_px(223), 0))
    detail = (
        f"nuclei {int_value(shape_row, 'baseline_nuclei'):,}→{int_value(shape_row, 'shape_strict_nuclei'):,} | "
        f"accepted parents {int_value(shape_row, 'split_events'):,} | {view_label} | yellow change"
    )
    return make_tile(body, sample, detail)


def filter_overlay(raw: np.ndarray, original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    retained = filtered > 0
    removed = (original > 0) & ~retained
    overlay = alpha_fill(raw, retained, CYAN, 0.24)
    overlay = alpha_fill(overlay, removed, MAGENTA, 0.58)
    overlay = paint_edge(overlay, boundaries(filtered), CYAN)
    removed_labels = np.where(removed, original, 0)
    overlay = paint_edge(overlay, boundaries(removed_labels), MAGENTA)
    return overlay


def render_nucleated_filter_tile(sample: Sample) -> Image.Image:
    combined_raw = normalize_rgb(read_tiff(sample.manifest["combined_raw"]))
    combined_original = squeeze_2d(read_tiff(sample.manifest["combined_mask"]))
    combined_filtered = squeeze_2d(read_tiff(sample.manifest["nucleated_combined_mask"]))
    combined = filter_overlay(combined_raw, combined_original, combined_filtered)

    bf_raw = normalize_scalar(squeeze_2d(read_tiff(sample.manifest["brightfield_raw"])))
    bf_original = squeeze_2d(read_tiff(sample.manifest["brightfield_mask"]))
    bf_filtered = squeeze_2d(read_tiff(sample.manifest["nucleated_brightfield_mask"]))
    bf = filter_overlay(bf_raw, bf_original, bf_filtered)

    panels = [
        label_image(fit_image(combined, (207, 235)), "Combined retained/removed"),
        label_image(fit_image(bf, (207, 235)), "Brightfield retained/removed"),
    ]
    body = Image.new("RGB", (qc_px(430), panels[0].height), BG)
    body.paste(panels[0], (0, 0))
    body.paste(panels[1], (qc_px(223), 0))
    detail = (
        f"Combined {count_labels(combined_original):,}→{count_labels(combined_filtered):,} | "
        f"BF {count_labels(bf_original):,}→{count_labels(bf_filtered):,} | cyan retained, magenta removed"
    )
    return make_tile(body, sample, detail)


def tile_body(tile: Image.Image, header_height: int = 54) -> Image.Image:
    return tile.crop((0, qc_px(header_height), tile.width, tile.height))


def panel_card(
    content: Image.Image | np.ndarray,
    panel_label: str,
    title: str,
    detail: str,
    size: tuple[int, int] = (326, 230),
) -> Image.Image:
    body = fit_image(content, size)
    header_h = qc_px(30)
    detail_h = qc_px(27)
    canvas = Image.new("RGB", (body.width, header_h + body.height + detail_h), PANEL_BG)
    canvas.paste(body, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((qc_px(7), qc_px(5)), f"({panel_label}) {title}", fill=TEXT, font=font(13, bold=True))
    draw.text((qc_px(7), header_h + body.height + qc_px(5)), detail, fill=MUTED, font=font(10))
    return canvas


def legend_strip(
    width: int,
    items: list[tuple[tuple[int, int, int], str]],
    footer: str,
) -> Image.Image:
    gap = qc_px(10)
    line_h = qc_px(22)
    swatch = qc_px(11)
    text_font = font(11)
    provisional = Image.new("RGB", (width, qc_px(130)), BG)
    draw = ImageDraw.Draw(provisional)
    x = gap
    y = qc_px(7)
    for color, label in items:
        text_width = draw.textbbox((0, 0), label, font=text_font)[2]
        item_width = swatch + qc_px(5) + text_width + qc_px(18)
        if x + item_width > width - gap and x > gap:
            x = gap
            y += line_h
        draw.rounded_rectangle((x, y + qc_px(2), x + swatch, y + qc_px(13)), radius=qc_px(2), fill=color)
        draw.text((x + swatch + qc_px(5), y), label, fill=TEXT, font=text_font)
        x += item_width
    y += line_h
    draw.text((gap, y), footer, fill=MUTED, font=font(10))
    return provisional.crop((0, 0, width, min(provisional.height, y + qc_px(24))))


def compose_sample_figure(
    sample: Sample,
    title: str,
    panels: list[Image.Image],
    columns: int,
    legend_items: list[tuple[tuple[int, int, int], str]],
    footer: str,
) -> Image.Image:
    gap = qc_px(10)
    panel_w = max(panel.width for panel in panels)
    panel_h = max(panel.height for panel in panels)
    rows = int(math.ceil(len(panels) / columns))
    width = columns * panel_w + (columns + 1) * gap
    header_h = qc_px(58)
    legend = legend_strip(width, legend_items, footer)
    height = header_h + rows * panel_h + (rows + 1) * gap + legend.height
    figure = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(figure)
    draw.text((gap, qc_px(8)), title, fill=TEXT, font=font(18, bold=True))
    draw.text((gap, qc_px(34)), sample_header(sample), fill=MUTED, font=font(12))
    for index, panel in enumerate(panels):
        x = gap + (index % columns) * (panel_w + gap)
        y = header_h + gap + (index // columns) * (panel_h + gap)
        figure.paste(panel, (x, y))
    figure.paste(legend, (0, height - legend.height))
    return figure


def dead_component_images(sample: Sample, run_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dead_raw = squeeze_2d(read_tiff(sample.manifest["dead_raw"]))
    combined_raw = read_tiff(sample.manifest["combined_raw"])
    primary = squeeze_2d(read_tiff(run_root / "Dead" / "intermediate" / "dead_primary" / f"{sample.key}_cp_masks.tif"))
    blue_mask = squeeze_2d(read_tiff(run_root / "Dead" / "intermediate" / "combined_blue" / f"{sample.key}_cp_masks.tif"))
    final_mask = squeeze_2d(read_tiff(sample.manifest["dead_mask"]))

    dead_panel = paint_edge(normalize_scalar(dead_raw), boundaries(primary), RED)
    combined_float = np.asarray(combined_raw[..., :3], dtype=np.float32)
    blue_excess = np.clip(
        combined_float[..., 2] - 0.5 * (combined_float[..., 0] + combined_float[..., 1]),
        0.0,
        None,
    )
    blue_panel = paint_edge(normalize_scalar(blue_excess, 0.0, 99.8), boundaries(blue_mask), CYAN)

    consensus = normalize_rgb(combined_raw)
    source_colors = {
        "dual_channel": GREEN,
        "dead_primary_only": RED,
        "combined_blue_rescue": BLUE,
    }
    final_edges = boundaries(final_mask)
    for label_id, source in provenance_map(run_root / "Dead" / "object_provenance" / f"{sample.key}.csv").items():
        consensus[final_edges & (final_mask == label_id)] = source_colors.get(source, WHITE)
    return dead_panel, blue_panel, consensus


def render_segmentation_sample_figure(sample: Sample, run_root: Path) -> Image.Image:
    nuclei = tile_body(render_nuclei_tile(sample))
    brightfield = tile_body(render_cell_segmentation_tile(sample, "Brightfield"))
    combined = tile_body(render_cell_segmentation_tile(sample, "Combined"))
    dead_primary, combined_blue, dead_consensus = dead_component_images(sample, run_root)
    nucleated_filter = tile_body(render_nucleated_filter_tile(sample))
    density_profile = "high-density profile" if sample.density_class.startswith("High") else "baseline profile"
    panels = [
        panel_card(
            nuclei,
            "A",
            "Nuclei extent and core",
            f"extent {count_labels(read_tiff(sample.manifest['nuclei_extent_mask'])):,}; core {count_labels(read_tiff(sample.manifest['nuclei_core_mask'])):,}",
        ),
        panel_card(
            brightfield,
            "B",
            "Brightfield segmentation",
            f"{count_labels(read_tiff(sample.manifest['brightfield_mask'])):,} objects; {density_profile}",
        ),
        panel_card(
            combined,
            "C",
            "Combined segmentation",
            f"{count_labels(read_tiff(sample.manifest['combined_mask'])):,} objects; {density_profile}",
        ),
        panel_card(
            dead_primary,
            "D",
            "Dead-primary candidates",
            f"{int_value(sample.dead, 'dead_primary_objects'):,} candidates",
        ),
        panel_card(
            combined_blue,
            "E",
            "Combined-blue candidates",
            f"{int_value(sample.dead, 'combined_blue_objects'):,} candidates",
        ),
        panel_card(
            dead_consensus,
            "F",
            "Final Dead consensus",
            f"{int_value(sample.dead, 'final_objects'):,} objects; {int_value(sample.dead, 'combined_blue_rescued_objects'):,} rescued",
        ),
        panel_card(
            nucleated_filter,
            "G",
            "Nucleated-only filter",
            f"Combined {count_labels(read_tiff(sample.manifest['combined_mask'])):,}->{count_labels(read_tiff(sample.manifest['nucleated_combined_mask'])):,}",
        ),
    ]
    return compose_sample_figure(
        sample,
        f"Segmentation QC composite - {sample.label}",
        panels,
        columns=4,
        legend_items=[
            (MAGENTA, "nucleus extent / removed cell mask"),
            (CYAN, "nucleus core / blue candidate / retained cell mask"),
            (GREEN, "cell boundary / dual-channel Dead"),
            (RED, "Dead-primary-only evidence"),
            (BLUE, "Combined-blue rescue"),
        ],
        footer="Panels A-G use the same field of view. Compare boundaries against raw structure, then trace Dead provenance and filter effects.",
    )


def render_analysis_sample_figure(sample: Sample, run_root: Path) -> Image.Image:
    classification_all = tile_body(render_classification_tile(sample, run_root, False))
    shape_all = tile_body(render_shape_tile(sample, False))
    classification_nucleated = tile_body(render_classification_tile(sample, run_root, True))
    shape_nucleated = tile_body(render_shape_tile(sample, True))
    panels = [
        panel_card(
            classification_all,
            "A",
            "All-cell classification",
            f"live {int_value(sample.fusion_all, 'live_cell_count'):,}; dead {int_value(sample.fusion_all, 'dead_cell_count'):,}; artifact {int_value(sample.fusion_all, 'artifact_count'):,}",
            size=(500, 300),
        ),
        panel_card(
            shape_all,
            "B",
            "All-cell shape_strict",
            f"nuclei {int_value(sample.shape_all, 'baseline_nuclei'):,}->{int_value(sample.shape_all, 'shape_strict_nuclei'):,}; split parents {int_value(sample.shape_all, 'split_events'):,}",
            size=(500, 300),
        ),
        panel_card(
            classification_nucleated,
            "C",
            "Nucleated-only classification",
            f"live {int_value(sample.fusion_nucleated, 'live_cell_count'):,}; dead {int_value(sample.fusion_nucleated, 'dead_cell_count'):,}; artifact {int_value(sample.fusion_nucleated, 'artifact_count'):,}",
            size=(500, 300),
        ),
        panel_card(
            shape_nucleated,
            "D",
            "Nucleated-only shape_strict",
            f"nuclei {int_value(sample.shape_nucleated, 'baseline_nuclei'):,}->{int_value(sample.shape_nucleated, 'shape_strict_nuclei'):,}; split parents {int_value(sample.shape_nucleated, 'split_events'):,}",
            size=(500, 300),
        ),
    ]
    return compose_sample_figure(
        sample,
        f"Classification and shape_strict composite - {sample.label}",
        panels,
        columns=2,
        legend_items=[
            (GREEN, "live cell"),
            (RED, "dead cell"),
            (GRAY, "artifact or other unresolved state"),
            (YELLOW, "nucleus-core boundary / accepted split change"),
            (MAGENTA, "nucleus-extent boundary"),
            (CYAN, "nucleus-core boundary"),
        ],
        footer="Read A versus C to assess the nucleus-support filter; read B versus D to assess whether cell context changes accepted strict splits.",
    )


def render_workflow_pdf(pdf_path: Path) -> Image.Image:
    require_file(pdf_path)
    with tempfile.TemporaryDirectory(prefix="cpsam_report_workflow_") as directory:
        prefix = Path(directory) / "workflow"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", "-singlefile", str(pdf_path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with Image.open(prefix.with_suffix(".png")) as rendered:
            return rendered.convert("RGB").copy()


def render_qc_sheets(
    samples: list[Sample],
    run_root: Path,
    workflow_pdf: Path,
) -> dict[str, Image.Image]:
    if len(samples) != 6:
        raise ValueError(f"Exactly six QC samples are required, found {len(samples)}")
    figures: dict[str, Image.Image] = {"workflow": render_workflow_pdf(workflow_pdf)}
    for sample in samples:
        token = sample.label.lower().replace("-", "")
        figures[f"segmentation_{token}"] = render_segmentation_sample_figure(sample, run_root)
    for sample in samples:
        token = sample.label.lower().replace("-", "")
        figures[f"analysis_{token}"] = render_analysis_sample_figure(sample, run_root)
    return figures


def encode_sheet(image: Image.Image, quality: int, high_resolution: bool = False) -> tuple[str, int, str]:
    buffer = io.BytesIO()
    exif = Image.Exif()
    if high_resolution:
        exif[282] = float(HIGH_RES_PRINT_PPI)
        exif[283] = float(HIGH_RES_PRINT_PPI)
        exif[296] = 2
    if pil_features.check("webp"):
        image.save(
            buffer,
            format="WEBP",
            quality=quality,
            method=6,
            exact=high_resolution,
            exif=exif if high_resolution else b"",
        )
        mime = "image/webp"
    else:
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=0 if high_resolution else 2,
            optimize=True,
            progressive=True,
            dpi=(HIGH_RES_PRINT_PPI, HIGH_RES_PRINT_PPI) if high_resolution else (72, 72),
            exif=exif if high_resolution else b"",
        )
        mime = "image/jpeg"
    raw = buffer.getvalue()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw), hashlib.sha256(raw).hexdigest()


def build_high_resolution_image_payload(sheets: dict[str, Image.Image]) -> dict[str, Any]:
    images: dict[str, dict[str, Any]] = {}
    quality = HIGH_RES_WEBP_QUALITY if pil_features.check("webp") else HIGH_RES_JPEG_QUALITY
    for key, sheet in sheets.items():
        data_uri, encoded_bytes, digest = encode_sheet(sheet, quality, high_resolution=True)
        images[f"{key}_image"] = {
            "data_uri": data_uri,
            "width": sheet.width,
            "height": sheet.height,
            "encoded_bytes": encoded_bytes,
            "sha256": digest,
        }
    return {
        "version": 1,
        "render_scale": QC_RENDER_SCALE,
        "print_ppi": HIGH_RES_PRINT_PPI,
        "quality": quality,
        "encoding": "webp" if pil_features.check("webp") else "jpeg-4:4:4",
        "images": images,
    }


def image_block_body(data_uri: str, alt: str, caption: str, width: int, height: int) -> str:
    return (
        '<style>html,body{margin:0!important;padding:0!important;overflow:hidden!important}'
        'figure{margin:0!important;padding:0!important;overflow:hidden!important}'
        'img{display:block!important;width:100%!important;max-width:100%!important;height:auto!important}'
        'figcaption{display:block;margin:6px 0 0;padding:0;font:12px/18px system-ui,sans-serif}</style>'
        '<figure style="margin:0">'
        f'<img src="{data_uri}" alt="{html.escape(alt, quote=True)}" '
        f'width="{int(width)}" height="{int(height)}">'
        f'<figcaption>{html.escape(caption)}</figcaption>'
        "</figure>"
    )


def aggregate_fusion(rows: list[dict[str, str]]) -> dict[str, int | float]:
    fields = [
        "total_masks",
        "total_cell_count",
        "live_cell_count",
        "dead_cell_count",
        "artifact_count",
        "dead_channel_override_count",
        "bf_supported_count",
        "nuclei_supported_count",
        "nuclei_inside_total",
    ]
    result: dict[str, int | float] = {field: sum(int_value(row, field) for row in rows) for field in fields}
    result["n_fields"] = len(rows)
    total_cells = int(result["total_cell_count"])
    result["dead_fraction"] = int(result["dead_cell_count"]) / total_cells if total_cells else 0.0
    return result


def make_sources(run_root: Path) -> list[dict[str, Any]]:
    base = f"results/{report_snapshot_name(run_root)}"
    return [
        {
            "id": "manifest_summary",
            "label": "Validated full-cohort field manifest",
            "path": f"{base}/workflow_status/postsegmentation_manifest/manifest_summary.json",
            "query": {
                "engine": "duckdb",
                "description": "Read the validated field count and join it to the documented top-level result inventory.",
                "sql": (
                    f"WITH manifest AS (SELECT n_fields FROM read_json_auto('{base}/workflow_status/postsegmentation_manifest/manifest_summary.json')), "
                    "directories(directory) AS (VALUES ('Nuclei'),('qc'),('Brightfield'),('Combined'),"
                    "('dead_preprocess_calibration'),('Dead'),('workflow_status'),('classification_fusion'),"
                    "('nucleated_only'),('classification_fusion_nucleated_only'),('shape_strict'),"
                    "('shape_strict_nucleated_only')) SELECT manifest.n_fields, directories.directory FROM manifest CROSS JOIN directories"
                ),
                "tables_used": ["manifest_summary.json", "validated HPC result inventory"],
            },
        },
        {
            "id": "profile_metadata",
            "label": "Per-field segmentation metadata and production configuration",
            "path": f"{base}/profile-metadata",
            "query": {
                "engine": "duckdb",
                "description": "Extract the distinct model, threshold, size, and preprocessing profiles used by segmentation.",
                "sql": (
                    "SELECT DISTINCT profile.folder AS profile, profile.model AS model, profile.diameter AS diameter, "
                    "profile.cellprob_threshold AS cellprob, profile.min_size AS min_size, profile.preprocess AS preprocess, "
                    f"profile.input_transform AS input_transform FROM read_json_auto('{base}/*/metadata/*.json') ORDER BY profile, diameter DESC"
                ),
                "tables_used": ["Brightfield/metadata/*.json", "Combined/metadata/*.json", "Nuclei/metadata/*.json"],
            },
        },
        {
            "id": "density_calls",
            "label": "Nucleus-core density calls",
            "path": f"{base}/qc/density_calls.csv",
            "query": {
                "engine": "duckdb",
                "description": "Count high- and low-density fields from the production core-mask density table.",
                "sql": (
                    f"SELECT count(*) AS n_fields, count(*) FILTER (WHERE high_density=true) AS n_high_density, "
                    f"count(*) FILTER (WHERE high_density=false) AS n_low_density FROM read_csv_auto('{base}/qc/density_calls.csv', header=true)"
                ),
                "tables_used": ["qc/density_calls.csv"],
                "metric_definitions": {
                    "high_density": "True when nucleus-core count is at least 4,000 or median nearest-neighbor distance is at most 16 pixels."
                },
            },
        },
        {
            "id": "dead_calibration",
            "label": "Full-cohort Dead and Combined-blue calibration",
            "path": f"{base}/dead_preprocess_calibration/final/dead_combined_blue_calibration.json",
            "query": {
                "engine": "duckdb",
                "description": "Read cohort size, shard count, recommended intensity scales, and calibration validation flags.",
                "sql": (
                    f"SELECT n_pairs, n_shards, dead.recommended_fixed_low AS dead_low, dead.recommended_fixed_high AS dead_high, "
                    f"combined_blue_excess_mean.recommended_fixed_high AS blue_high, validation FROM read_json_auto('{base}/dead_preprocess_calibration/final/dead_combined_blue_calibration.json')"
                ),
                "tables_used": ["dead_combined_blue_calibration.json"],
            },
        },
        {
            "id": "dead_consensus",
            "label": "Dead dual-channel consensus summary",
            "path": f"{base}/Dead/dead_consensus_summary.json",
            "query": {
                "engine": "duckdb",
                "description": "Expand final Dead objects into dual-channel, Dead-primary-only, and Combined-blue-rescue provenance.",
                "sql": (
                    f"WITH s AS (SELECT * FROM read_json_auto('{base}/Dead/dead_consensus_summary.json')) "
                    "SELECT source, object_count, object_count::DOUBLE/final_objects AS fraction FROM s, LATERAL "
                    "(VALUES ('Dual channel', provenance_counts.dual_channel), ('Dead primary only', provenance_counts.dead_primary_only), "
                    "('Combined-blue rescue', provenance_counts.combined_blue_rescue)) AS p(source, object_count)"
                ),
                "tables_used": ["Dead/dead_consensus_summary.json"],
            },
        },
        {
            "id": "nucleated_summary",
            "label": "Nucleated-only mask-filter summary",
            "path": f"{base}/nucleated_only/branch_summary.json",
            "query": {
                "engine": "duckdb",
                "description": "Read Brightfield and Combined object counts before and after nucleus-centroid filtering.",
                "sql": (
                    "SELECT n_fields, allow_extent_fallback, profiles.Brightfield.original_cells AS bf_original, "
                    "profiles.Brightfield.retained_cells AS bf_retained, profiles.Combined.original_cells AS combined_original, "
                    f"profiles.Combined.retained_cells AS combined_retained FROM read_json_auto('{base}/nucleated_only/branch_summary.json')"
                ),
                "tables_used": ["nucleated_only/branch_summary.json"],
            },
        },
        {
            "id": "fusion_comparison",
            "label": "All-cell and nucleated-only fusion summaries",
            "path": f"{base}/fusion-branch-summaries",
            "query": {
                "engine": "duckdb",
                "description": "Aggregate per-field live, dead, artifact, and total counts for both classification branches.",
                "sql": (
                    f"WITH rows AS (SELECT 'All-cell' AS branch,* FROM read_csv_auto('{base}/classification_fusion/summaries/cell_count_summary.csv',header=true) "
                    f"UNION ALL SELECT 'Nucleated-only' AS branch,* FROM read_csv_auto('{base}/classification_fusion_nucleated_only/summaries/cell_count_summary.csv',header=true)) "
                    "SELECT branch,count(*) AS fields,sum(total_masks) AS total_masks,sum(total_cell_count) AS total_cells,"
                    "sum(live_cell_count) AS live_cells,sum(dead_cell_count) AS dead_cells,sum(artifact_count) AS artifacts,"
                    "sum(dead_cell_count)::DOUBLE/sum(total_cell_count) AS dead_fraction FROM rows GROUP BY branch ORDER BY branch"
                ),
                "tables_used": [
                    "classification_fusion/summaries/cell_count_summary.csv",
                    "classification_fusion_nucleated_only/summaries/cell_count_summary.csv",
                ],
            },
        },
        {
            "id": "shape_comparison",
            "label": "All-cell and nucleated-only shape_strict summaries",
            "path": f"{base}/shape-strict-summaries",
            "query": {
                "engine": "duckdb",
                "description": "Combine strict split counts and validation totals from both cell-mask support branches.",
                "sql": (
                    f"SELECT 'All-cell' AS branch,n_baseline_nuclei,n_shape_strict_nuclei,n_split_parents,n_fields_with_splits,count_increase_fraction FROM read_json_auto('{base}/shape_strict/shape_strict_summary.json') "
                    f"UNION ALL SELECT 'Nucleated-only',n_baseline_nuclei,n_shape_strict_nuclei,n_split_parents,n_fields_with_splits,count_increase_fraction FROM read_json_auto('{base}/shape_strict_nucleated_only/shape_strict_summary.json')"
                ),
                "tables_used": ["shape_strict/shape_strict_summary.json", "shape_strict_nucleated_only/shape_strict_summary.json"],
            },
        },
        {
            "id": "qc_selection",
            "label": "Deterministic six-field QC cohort",
            "path": f"{base}/embedded-report-qc-selection",
            "query": {
                "engine": "duckdb",
                "description": "Select three high-density and three low-density complete fields at fixed density-score quantiles, preferring distinct wells and at least one accepted strict split per stratum when available.",
                "sql": (
                    f"WITH density AS (SELECT * FROM read_csv_auto('{base}/qc/density_calls.csv',header=true)), "
                    f"fusion AS (SELECT * FROM read_csv_auto('{base}/classification_fusion/summaries/cell_count_summary.csv',header=true)), "
                    f"shape AS (SELECT * FROM read_csv_auto('{base}/shape_strict/field_summary.csv',header=true)) "
                    "SELECT density.key,density.high_density,density.nuclei_count,density.nuclei_median_nn_px,fusion.dead_fraction,shape.split_events "
                    "FROM density JOIN fusion USING(key) JOIN shape USING(key)"
                ),
                "tables_used": ["qc/density_calls.csv", "classification_fusion/summaries/cell_count_summary.csv", "shape_strict/field_summary.csv"],
                "filters": {"sample_count": 6, "high_density": 3, "low_density": 3, "same_keys_for_every_qc_panel": True},
            },
        },
        {
            "id": "pipeline_code",
            "label": "Production workflow scripts",
            "path": portable_hpc_path(DEFAULT_HPC_PROJECT_DIR),
            "query": {
                "engine": "duckdb",
                "description": "Reconstruct the ordered afterok workflow stages implemented by the production submission script.",
                "sql": (
                    "SELECT * FROM (VALUES (1,'Environment preflight'),(2,'Nuclei segmentation'),(3,'Dead calibration'),"
                    "(4,'Density table'),(5,'BF/Combined/Dead segmentation'),(6,'Field manifest'),"
                    "(7,'Two classification branches'),(8,'Two shape_strict branches')) AS workflow(stage_order,stage) ORDER BY stage_order"
                ),
                "tables_used": [
                    "cellpose_pipeline/hpc/submit_full_fusion_production.sh",
                    "cellpose_pipeline/hpc/orchestrate_cellpose_cpsam_full_array.sh",
                    "cellpose_pipeline/scripts/01_segment_images.py",
                    "cellpose_pipeline/scripts/02_call_high_density_from_nuclei_masks.py",
                    "cellpose_pipeline/scripts/03_calibrate_dead_combined_blue.py",
                    "cellpose_pipeline/scripts/04_segment_dead_with_combined_blue_consensus.py",
                    "cellpose_pipeline/scripts/05_merge_dead_consensus_outputs.py",
                    "cellpose_pipeline/scripts/06_build_postsegmentation_field_manifest.py",
                    "cellpose_pipeline/scripts/07_build_nucleated_cell_branch.py",
                    "cellpose_pipeline/scripts/08_fuse_multichannel_classification.py",
                    "cellpose_pipeline/scripts/09_apply_shape_aware_nucleus_splits.py",
                    "cellpose_pipeline/scripts/10_merge_shape_strict_shards.py",
                ],
            },
        },
        {
            "id": "result_inventory",
            "label": "Validated result-directory and file inventory",
            "path": portable_hpc_path(DEFAULT_HPC_RESULT_ROOT / "workflow_status" / "postsegmentation_manifest"),
            "query": {
                "engine": "duckdb",
                "description": "Document the authoritative result directories and key files represented in this report.",
                "sql": (
                    "SELECT * FROM (VALUES "
                    "(1,'Nuclei/','Extent and core masks'),(2,'qc/','Density calls'),"
                    "(3,'Brightfield/','Brightfield cell masks'),(4,'Combined/','Combined cell masks'),"
                    "(5,'dead_preprocess_calibration/','Full-cohort Dead calibration'),"
                    "(6,'Dead/','Dual-channel Dead consensus'),(7,'workflow_status/','Validated field manifest'),"
                    "(8,'classification_fusion/','All-cell fusion'),(9,'nucleated_only/','Nucleus-supported mask branch'),"
                    "(10,'classification_fusion_nucleated_only/','Nucleated-only fusion'),"
                    "(11,'shape_strict/','All-cell strict nucleus splitting'),"
                    "(12,'shape_strict_nucleated_only/','Nucleated-only strict nucleus splitting')) "
                    "AS inventory(display_order,directory,result_type) ORDER BY display_order"
                ),
                "tables_used": [
                    "workflow_status/postsegmentation_manifest/manifest_summary.json",
                    "workflow_status/postsegmentation_manifest/field_manifest.tsv",
                    "validated result-directory inventory",
                ],
            },
        },
    ]


def build_datasets(run_root: Path, tables: dict[str, Any], samples: list[Sample]) -> dict[str, list[dict[str, Any]]]:
    manifest_summary = tables["manifest_summary"]
    dead_summary = tables["dead_summary"]
    shape_all = tables["shape_all_summary"]
    shape_nucleated = tables["shape_nucleated_summary"]
    fusion_all = aggregate_fusion(tables["fusion_all_rows"])
    fusion_nucleated = aggregate_fusion(tables["fusion_nucleated_rows"])
    density_rows = list(tables["density"].values())
    n_high = sum(truthy(row.get("high_density")) for row in density_rows)
    n_fields = int(manifest_summary["n_fields"])

    directory_map = [
        (1, "Nuclei/", "Extent masks, intensity-supported core masks, metadata, and segmentation QC", "cpsam_v2 high-recall extent plus within-extent core extraction", "segmentations/, nucleus_core_seeds/, metadata/"),
        (2, "qc/", "Per-field nucleus density metrics and high-density calls", "Core-mask count and nearest-neighbor thresholds", "density_calls.csv"),
        (3, "Brightfield/", "Brightfield cell-instance masks and metadata", "Baseline cpsam or calibrated high-density CLAHE profile", "segmentations/, metadata/"),
        (4, "Combined/", "Combined cell-instance masks used as fusion anchors", "RGB luma baseline or calibrated high-density background subtraction", "segmentations/, metadata/"),
        (5, "dead_preprocess_calibration/", "Full-cohort Dead and Combined-blue intensity calibration", "Thirty-two-shard histogram and per-image quantile scan", "final/dead_combined_blue_calibration.json"),
        (6, "Dead/", "Final dual-channel Dead masks and object provenance", "Dead primary plus conservative Combined-blue rescue", "segmentations/, dead_consensus_summary.json"),
        (7, "workflow_status/", "Validated path manifest for every field and analysis branch", "Exact-key-set and file-presence validation", "postsegmentation_manifest/field_manifest.tsv"),
        (8, "classification_fusion/", "All-cell per-object features, predictions, and counts", "Combined anchor plus RGB, BF, nucleus-core, and Dead evidence", "predictions/, summaries/cell_count_summary.csv"),
        (9, "nucleated_only/", "BF and Combined masks after removal of objects lacking nucleus-centroid support", "Core centroid with extent-centroid fallback", "filter_decisions.csv, branch_summary.json"),
        (10, "classification_fusion_nucleated_only/", "Nucleated-only per-object features, predictions, and counts", "Same fusion logic on filtered cell masks", "predictions/, summaries/cell_count_summary.csv"),
        (11, "shape_strict/", "All-cell-supported strict nucleus split layer and QC", "Shape gate, stable two-peak evidence, within-parent watershed", "split_events.csv, shape_strict_summary.json, qc/"),
        (12, "shape_strict_nucleated_only/", "Nucleated-only-supported strict nucleus split layer and QC", "Same strict split logic using filtered cell support", "split_events.csv, shape_strict_summary.json, qc/"),
    ]

    pipeline_steps = [
        (1, "Environment preflight", "Raw image root, Cellpose environment, model cache", "Validate Cellpose 4.2.1.1, cpsam/cpsam_v2, inputs, and task lists", "Submission-ready state", "None"),
        (2, "Nuclei segmentation", "Nuclei raw images", "cpsam_v2 extent plus intensity-supported core", "Nuclei/", "afterok preflight"),
        (3, "Dead calibration", "Paired Dead and Combined raw images", "Thirty-two-shard cohort scan and merge", "dead_preprocess_calibration/final/", "Parallel with Nuclei"),
        (4, "Density table", "Nuclei core masks", "Count or nearest-neighbor threshold", "qc/density_calls.csv", "afterok Nuclei"),
        (5, "BF/Combined/Dead segmentation", "Raw images, density calls, Dead calibration", "Density-specific cell profiles and dual-channel Dead consensus", "Brightfield/, Combined/, Dead/", "afterok density and calibration"),
        (6, "Field manifest", "All base segmentation outputs", "Validate exact key sets and record canonical paths", "workflow_status/postsegmentation_manifest/", "afterok Dead merge"),
        (7, "Two fusion branches", "Four channels and field manifest", "All-cell fusion; filtered-mask branch followed by independent fusion", "classification_fusion*/", "afterok manifest/branch merge"),
        (8, "Two shape_strict branches", "Nuclei raw/extent/core, cell masks, fusion", "Stable two-peak strict split and QC", "shape_strict*/", "afterok corresponding fusion merge"),
    ]

    profile_params = [
        (1, "Nuclei", "cpsam_v2", 24, -2.75, 5, "Per-image percentile 0.5–99.9; flow 0", "Extent plus local intensity core"),
        (2, "Brightfield, non-high-density", "cpsam", 25, -1.75, 20, "Per-image percentile 1–99; flow 0", "Cell-instance mask"),
        (3, "Brightfield, high-density", "cpsam", 22, -2.25, 10, "CLAHE clip 1.4, tile 16; flow 0", "Crowding-adjusted cell mask"),
        (4, "Combined, non-high-density", "cpsam", 25, -1.75, 20, "RGB luma plus percentile 1–99; flow 0", "Fusion anchor mask"),
        (5, "Combined, high-density", "cpsam", 22, -2.25, 10, "RGB luma plus background sigma 12; flow 0", "Crowding-adjusted fusion anchor"),
        (6, "Dead primary", "cpsam_v2", 22, -2.75, 12, "Full-cohort bounded background plus raw-signal filtering", "Dead primary candidates"),
        (7, "Combined blue-excess", "cpsam", 22, -3.0, 10, "max(B-(R+G)/2,0) with cohort-fixed scaling", "Dead rescue candidates"),
    ]

    qc_rows: list[dict[str, Any]] = []
    for sample in samples:
        all_cells = int_value(sample.fusion_all, "total_cell_count")
        nucleated_cells = int_value(sample.fusion_nucleated, "total_cell_count")
        qc_rows.append(
            {
                "display_order": len(qc_rows) + 1,
                "sample": sample.label,
                "key": sample.key,
                "density_class": sample.density_class,
                "nuclei_count": int_value(sample.density, "nuclei_count"),
                "median_nn_px": float_value(sample.density, "nuclei_median_nn_px"),
                "all_cell_count": all_cells,
                "nucleated_cell_count": nucleated_cells,
                "retained_fraction": nucleated_cells / all_cells if all_cells else 0.0,
                "all_cell_dead_fraction": float_value(sample.fusion_all, "dead_fraction"),
                "dead_final_objects": int_value(sample.dead, "final_objects"),
                "shape_splits_all": int_value(sample.shape_all, "split_events"),
                "shape_splits_nucleated": int_value(sample.shape_nucleated, "split_events"),
                "selection_reason": sample.selection_reason,
            }
        )

    key_files = [
        (1, "qc/density_calls.csv", "One row per field", "Density metrics, trigger reasons, and final high-density call"),
        (2, "dead_preprocess_calibration/final/dead_combined_blue_calibration.json", "Full cohort", "Dead/Combined-blue intensity scales and validation"),
        (3, "Dead/dead_consensus_summary.json", "Full cohort", "Primary, blue, matched, rescue, and final Dead totals"),
        (4, "Dead/dead_consensus_object_provenance.csv", "One row per final Dead object", "Dual-channel, primary-only, or blue-rescue source"),
        (5, "workflow_status/postsegmentation_manifest/field_manifest.tsv", "One row per field", "Canonical raw and mask paths for both cell branches"),
        (6, "classification_fusion/predictions/*_per_cell_predictions.csv", "One row per all-cell Combined object", "Final state, reason, confidence, and channel matches"),
        (7, "classification_fusion/summaries/cell_count_summary.csv", "One row per field", "All-cell total/live/dead/artifact counts"),
        (8, "nucleated_only/filter_decisions.csv", "One row per BF/Combined object", "Retention decision and nucleus-centroid support"),
        (9, "classification_fusion_nucleated_only/summaries/cell_count_summary.csv", "One row per field", "Nucleated-only total/live/dead/artifact counts"),
        (10, "shape_strict/split_events.csv", "One row per accepted parent split", "Peak, valley, stability, shape, and cell-support evidence"),
        (11, "shape_strict/shape_strict_summary.json", "Full cohort", "All-cell-supported strict split validation"),
        (12, "shape_strict_nucleated_only/shape_strict_summary.json", "Full cohort", "Nucleated-only-supported strict split validation"),
    ]

    return {
        "overview": [
            {
                "n_fields": n_fields,
                "n_high_density": n_high,
                "high_density_fraction": n_high / n_fields,
                "n_dead_final": int(dead_summary["final_objects"]),
                "n_cells_all": int(fusion_all["total_cell_count"]),
                "n_cells_nucleated": int(fusion_nucleated["total_cell_count"]),
                "n_shape_split_parents": int(shape_all["n_split_parents"]),
            }
        ],
        "directory_map": [
            {"order": order, "directory": directory, "result": result, "method": method, "authoritative": authoritative}
            for order, directory, result, method, authoritative in directory_map
        ],
        "pipeline_steps": [
            {"order": order, "stage": stage, "input": input_value, "method": method, "output": output, "dependency": dependency}
            for order, stage, input_value, method, output, dependency in pipeline_steps
        ],
        "profile_params": [
            {"order": order, "profile": profile, "model": model, "diameter": diameter, "cellprob": cellprob, "min_size": min_size, "preprocess": preprocess, "output": output}
            for order, profile, model, diameter, cellprob, min_size, preprocess, output in profile_params
        ],
        "dead_provenance": [
            {"source": "Dual channel", "object_count": int(dead_summary["provenance_counts"]["dual_channel"]), "fraction": int(dead_summary["provenance_counts"]["dual_channel"]) / int(dead_summary["final_objects"]), "definition": "Dead primary and Combined-blue matched"},
            {"source": "Dead primary only", "object_count": int(dead_summary["provenance_counts"]["dead_primary_only"]), "fraction": int(dead_summary["provenance_counts"]["dead_primary_only"]) / int(dead_summary["final_objects"]), "definition": "Strict raw-signal-supported Dead object without a blue match"},
            {"source": "Combined-blue rescue", "object_count": int(dead_summary["provenance_counts"]["combined_blue_rescue"]), "fraction": int(dead_summary["provenance_counts"]["combined_blue_rescue"]) / int(dead_summary["final_objects"]), "definition": "Blue-only candidate passing relaxed Dead raw-signal evidence"},
        ],
        "fusion_branch": [
            {"branch_order": 1, "branch": "All-cell", "fields": int(fusion_all["n_fields"]), "total_masks": int(fusion_all["total_masks"]), "total_cells": int(fusion_all["total_cell_count"]), "live_cells": int(fusion_all["live_cell_count"]), "dead_cells": int(fusion_all["dead_cell_count"]), "artifacts": int(fusion_all["artifact_count"]), "dead_fraction": float(fusion_all["dead_fraction"])},
            {"branch_order": 2, "branch": "Nucleated-only", "fields": int(fusion_nucleated["n_fields"]), "total_masks": int(fusion_nucleated["total_masks"]), "total_cells": int(fusion_nucleated["total_cell_count"]), "live_cells": int(fusion_nucleated["live_cell_count"]), "dead_cells": int(fusion_nucleated["dead_cell_count"]), "artifacts": int(fusion_nucleated["artifact_count"]), "dead_fraction": float(fusion_nucleated["dead_fraction"])},
        ],
        "shape_branch": [
            {"branch_order": 1, "branch": "All-cell shape_strict", "baseline_nuclei": int(shape_all["n_baseline_nuclei"]), "strict_nuclei": int(shape_all["n_shape_strict_nuclei"]), "split_parents": int(shape_all["n_split_parents"]), "fields_with_splits": int(shape_all["n_fields_with_splits"]), "increase_fraction": float(shape_all["count_increase_fraction"])},
            {"branch_order": 2, "branch": "Nucleated-only shape_strict", "baseline_nuclei": int(shape_nucleated["n_baseline_nuclei"]), "strict_nuclei": int(shape_nucleated["n_shape_strict_nuclei"]), "split_parents": int(shape_nucleated["n_split_parents"]), "fields_with_splits": int(shape_nucleated["n_fields_with_splits"]), "increase_fraction": float(shape_nucleated["count_increase_fraction"])},
        ],
        "qc_samples": qc_rows,
        "key_files": [
            {"order": order, "path": path, "grain": grain, "meaning": meaning}
            for order, path, grain, meaning in key_files
        ],
        "visual_map": [
            {"order": 1, "section": "All-cell", "visual": "Nuclei extent/core", "claim": "Core masks provide exposure-resistant nucleus support"},
            {"order": 2, "section": "All-cell", "visual": "Brightfield segmentation", "claim": "Density-specific profiles preserve cell separation"},
            {"order": 3, "section": "All-cell", "visual": "Combined segmentation", "claim": "Combined masks define the fusion anchor population"},
            {"order": 4, "section": "All-cell", "visual": "Dead consensus", "claim": "Dead and Combined-blue evidence are reconciled conservatively"},
            {"order": 5, "section": "All-cell", "visual": "Fusion classification", "claim": "Per-cell states are assigned on all Combined objects"},
            {"order": 6, "section": "All-cell", "visual": "shape_strict", "claim": "Only evidence-supported merged nuclei are split"},
            {"order": 7, "section": "Nucleated-only", "visual": "Mask filtering", "claim": "Objects lacking nucleus-centroid support are removed"},
            {"order": 8, "section": "Nucleated-only", "visual": "Fusion classification", "claim": "Classification is recomputed on retained cell objects"},
            {"order": 9, "section": "Nucleated-only", "visual": "shape_strict", "claim": "Strict splitting is reevaluated with filtered cell support"},
        ],
    }


def build_manifest(
    run_root: Path,
    generated_at: str,
    image_uris: dict[str, str],
    image_sizes: dict[str, tuple[int, int]],
    samples: list[Sample],
    methods_markdown: str,
) -> dict[str, Any]:
    title = "SUM159 v3 Full-Cohort Segmentation and Classification Analysis Guide"

    cards = [
        {
            "id": "field_count",
            "description": "Fields validated in the canonical post-segmentation manifest.",
            "dataset": "overview",
            "sourceId": "manifest_summary",
            "metrics": [{"label": "Validated fields", "field": "n_fields", "format": "compact"}],
        },
        {
            "id": "high_density_count",
            "description": "Fields routed through the calibrated high-density cell-segmentation profiles.",
            "dataset": "overview",
            "sourceId": "density_calls",
            "metrics": [
                {"label": "High-density fields", "field": "n_high_density", "format": "compact"},
                {"label": "Share of fields", "field": "high_density_fraction", "format": "percent"},
            ],
        },
        {
            "id": "dead_object_count",
            "description": "Final Dead objects after primary-channel filtering and conservative Combined-blue rescue.",
            "dataset": "overview",
            "sourceId": "dead_consensus",
            "metrics": [{"label": "Final Dead objects", "field": "n_dead_final", "format": "compact"}],
        },
        {
            "id": "all_cell_count",
            "description": "Objects classified as live or dead in the all-cell fusion branch.",
            "dataset": "overview",
            "sourceId": "fusion_comparison",
            "metrics": [{"label": "All-cell classified cells", "field": "n_cells_all", "format": "compact"}],
        },
        {
            "id": "nucleated_cell_count",
            "description": "Objects classified as live or dead after nucleus-support filtering.",
            "dataset": "overview",
            "sourceId": "fusion_comparison",
            "metrics": [{"label": "Nucleated-only cells", "field": "n_cells_nucleated", "format": "compact"}],
        },
        {
            "id": "shape_split_count",
            "description": "Merged nucleus parents accepted for strict splitting in the all-cell-supported branch.",
            "dataset": "overview",
            "sourceId": "shape_comparison",
            "metrics": [{"label": "Accepted shape splits", "field": "n_shape_split_parents", "format": "compact"}],
        },
    ]

    charts = [
        {
            "id": "dead_provenance_chart",
            "title": "Final Dead-object provenance",
            "subtitle": "Object counts after dual-channel consensus and raw-signal gates.",
            "type": "bar",
            "dataset": "dead_provenance",
            "sourceId": "dead_consensus",
            "valueFormat": "compact",
            "encodings": {
                "x": {"field": "source", "type": "nominal", "label": "Evidence source"},
                "y": {"field": "object_count", "type": "quantitative", "label": "Final objects", "format": "compact"},
                "tooltip": [
                    {"field": "fraction", "type": "quantitative", "label": "Share", "format": "percent"},
                    {"field": "definition", "type": "nominal", "label": "Definition"},
                ],
            },
            "layout": "full",
        }
    ]

    tables = [
        {
            "id": "qc_sample_table",
            "title": "Fixed six-field QC cohort",
            "subtitle": "The same three high-density and three low-density fields appear in every QC panel.",
            "dataset": "qc_samples",
            "sourceId": "qc_selection",
            "columns": [
                {"field": "sample", "label": "Sample", "type": "text"},
                {"field": "key", "label": "Field key", "type": "text"},
                {"field": "density_class", "label": "Density", "type": "text"},
                {"field": "nuclei_count", "label": "Nuclei", "format": "number"},
                {"field": "median_nn_px", "label": "Median NN, px", "format": "number"},
                {"field": "all_cell_count", "label": "All-cell count", "format": "number"},
                {"field": "nucleated_cell_count", "label": "Nucleated-only", "format": "number"},
                {"field": "retained_fraction", "label": "Retained", "format": "percent"},
                {"field": "shape_splits_all", "label": "Strict splits", "format": "number"},
            ],
        },
        {
            "id": "fusion_branch_table",
            "title": "Classification branch totals",
            "subtitle": "Counts are aggregated from the per-field classification summaries.",
            "dataset": "fusion_branch",
            "sourceId": "fusion_comparison",
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "fields", "label": "Fields", "format": "number"},
                {"field": "total_masks", "label": "Combined masks", "format": "compact"},
                {"field": "total_cells", "label": "Classified cells", "format": "compact"},
                {"field": "live_cells", "label": "Live", "format": "compact"},
                {"field": "dead_cells", "label": "Dead", "format": "compact"},
                {"field": "artifacts", "label": "Artifacts", "format": "compact"},
                {"field": "dead_fraction", "label": "Dead fraction", "format": "percent"},
            ],
        },
        {
            "id": "shape_branch_table",
            "title": "shape_strict branch totals",
            "subtitle": "Strict masks differ from the baseline only where all split gates pass.",
            "dataset": "shape_branch",
            "sourceId": "shape_comparison",
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "baseline_nuclei", "label": "Baseline nuclei", "format": "compact"},
                {"field": "strict_nuclei", "label": "Strict nuclei", "format": "compact"},
                {"field": "split_parents", "label": "Split parents", "format": "number"},
                {"field": "fields_with_splits", "label": "Fields with splits", "format": "number"},
                {"field": "increase_fraction", "label": "Count increase", "format": "percent"},
            ],
        },
        {
            "id": "directory_map_table",
            "title": "Result directory map",
            "subtitle": "Each directory contains one defined stage or downstream branch.",
            "dataset": "directory_map",
            "sourceId": "result_inventory",
            "columns": [
                {"field": "directory", "label": "Directory", "type": "text"},
                {"field": "result", "label": "Result", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "authoritative", "label": "Authoritative outputs", "type": "text"},
            ],
        },
        {
            "id": "pipeline_table",
            "title": "Production workflow and dependencies",
            "subtitle": "The order reflects the production afterok dependency chain.",
            "dataset": "pipeline_steps",
            "sourceId": "pipeline_code",
            "columns": [
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "input", "label": "Input", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "output", "label": "Output", "type": "text"},
                {"field": "dependency", "label": "Dependency", "type": "text"},
            ],
        },
        {
            "id": "profile_table",
            "title": "Segmentation profiles and parameters",
            "subtitle": "Production profiles used for nuclei, cells, and dual-channel Dead segmentation.",
            "dataset": "profile_params",
            "sourceId": "profile_metadata",
            "columns": [
                {"field": "profile", "label": "Profile", "type": "text"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "diameter", "label": "Diameter", "format": "number"},
                {"field": "cellprob", "label": "Cellprob", "format": "number"},
                {"field": "min_size", "label": "Min size", "format": "number"},
                {"field": "preprocess", "label": "Preprocessing", "type": "text"},
                {"field": "output", "label": "Output role", "type": "text"},
            ],
        },
        {
            "id": "key_files_table",
            "title": "Key machine-readable outputs",
            "subtitle": "Start with these files for field-level or object-level downstream analysis.",
            "dataset": "key_files",
            "sourceId": "result_inventory",
            "columns": [
                {"field": "path", "label": "Relative path", "type": "text"},
                {"field": "grain", "label": "Row grain", "type": "text"},
                {"field": "meaning", "label": "Meaning", "type": "text"},
            ],
        },
    ]

    if "workflow" in image_uris:
        return build_grouped_report_manifest(
            run_root,
            generated_at,
            image_uris,
            image_sizes,
            samples,
            methods_markdown,
            cards,
            charts,
            tables,
        )

    image_specs = {
        "nuclei": (
            "Nuclei segmentation QC",
            "The exposure-aware nucleus workflow keeps a high-recall extent mask while deriving an intensity-supported core inside each extent. Magenta outlines show extents; cyan outlines and fills show cores used for density and nucleus support.",
            "Nuclei extent and core overlays for the fixed six-field QC cohort",
            "Nuclei extent/core QC for the same six fields.",
            "profile_metadata",
        ),
        "brightfield": (
            "Brightfield segmentation QC",
            "Green instance boundaries show the density-specific Brightfield result. High-density fields use the calibrated crowding profile; low-density fields use the baseline profile.",
            "Brightfield segmentation overlays for the fixed six-field QC cohort",
            "Brightfield instance-mask QC for the same six fields.",
            "profile_metadata",
        ),
        "combined": (
            "Combined segmentation QC",
            "The Combined segmentation is the anchor population for per-cell fusion. Green boundaries expose both separation in crowded regions and possible residual merged objects.",
            "Combined segmentation overlays for the fixed six-field QC cohort",
            "Combined instance-mask QC for the same six fields.",
            "profile_metadata",
        ),
        "dead": (
            "Dead dual-channel consensus QC",
            "Each field shows Dead-primary candidates, Combined blue-excess candidates, the Combined RGB image, and final provenance. Green denotes matched dual-channel evidence, red Dead-primary-only evidence, and blue conservative Combined-blue rescue.",
            "Dead primary, Combined-blue, and consensus overlays for the fixed six-field QC cohort",
            "Dead dual-channel consensus QC for the same six fields.",
            "dead_consensus",
        ),
        "classification_all": (
            "All-cell fusion classification QC",
            "The all-cell branch classifies every Combined anchor as live, dead, or artifact after integrating RGB, Brightfield, nucleus-core, and Dead evidence. Yellow nucleus-core boundaries make nucleus support directly visible.",
            "All-cell classification overlays for the fixed six-field QC cohort",
            "All-cell classification QC for the same six fields.",
            "fusion_comparison",
        ),
        "shape_all": (
            "All-cell shape_strict QC",
            "Baseline and strict nucleus masks are shown side by side. Yellow marks a changed split boundary; unchanged fields demonstrate the conservative default when stable two-peak and shape evidence are absent.",
            "All-cell shape-strict overlays for the fixed six-field QC cohort",
            "All-cell-supported shape_strict QC for the same six fields.",
            "shape_comparison",
        ),
        "nucleated_filter": (
            "Nucleated-only mask-filter QC",
            "The second branch removes Brightfield and Combined objects lacking nucleus-centroid support before classification. Cyan objects are retained and magenta objects are excluded from this branch only.",
            "Nucleated-only mask filtering overlays for the fixed six-field QC cohort",
            "Nucleus-support filtering QC for the same six fields.",
            "nucleated_summary",
        ),
        "classification_nucleated": (
            "Nucleated-only fusion classification QC",
            "Classification is recomputed after filtering rather than copied from the all-cell branch. The same color mapping is retained so differences can be compared field by field.",
            "Nucleated-only classification overlays for the fixed six-field QC cohort",
            "Nucleated-only classification QC for the same six fields.",
            "fusion_comparison",
        ),
        "shape_nucleated": (
            "Nucleated-only shape_strict QC",
            "Strict nucleus splitting is reevaluated with filtered cell-mask support. The paired baseline/strict view distinguishes accepted corrections from nuclei intentionally left unchanged.",
            "Nucleated-only shape-strict overlays for the fixed six-field QC cohort",
            "Nucleated-only-supported shape_strict QC for the same six fields.",
            "shape_comparison",
        ),
    }

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical Summary\n\n"
                "This report documents the completed full-cohort segmentation, density routing, dual-channel Dead consensus, "
                "two classification branches, and two shape_strict branches. The all-cell result is presented first; the "
                "nucleated-only sensitivity branch follows. All QC panels use one fixed cohort of six fields so visual "
                "differences across stages are attributable to processing rather than sample substitution."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "field_count",
                "high_density_count",
                "dead_object_count",
                "all_cell_count",
                "nucleated_cell_count",
                "shape_split_count",
            ],
        },
        {
            "id": "qc_selection_heading",
            "type": "markdown",
            "sourceId": "qc_selection",
            "body": (
                "## QC Sample Selection\n\n"
                "Three high-density and three low-density complete fields were selected deterministically across the "
                "density-score range, with distinct wells preferred. These exact keys are reused for every segmentation, "
                "classification, filtering, and shape_strict panel below."
            ),
        },
        {"id": "qc_sample_table_block", "type": "table", "tableId": "qc_sample_table", "layout": "full"},
        {
            "id": "all_cell_heading",
            "type": "markdown",
            "body": (
                "## All-cell Branch\n\n"
                "This primary branch retains all Combined segmentation objects, uses Brightfield and nucleus evidence as "
                "supporting features, and reports live, dead, and artifact states before strict nucleus-shape correction."
            ),
        },
    ]

    for key in ("nuclei", "brightfield", "combined", "dead", "classification_all", "shape_all"):
        heading, paragraph, alt, caption, source_id = image_specs[key]
        blocks.extend(
            [
                {
                    "id": f"{key}_explanation",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": f"### {heading}\n\n{paragraph}",
                },
                {
                    "id": f"{key}_image",
                    "type": "html",
                    "body": image_block_body(image_uris[key], alt, caption, *image_sizes[key]),
                    "layout": "full",
                },
            ]
        )
        if key == "dead":
            blocks.extend(
                [
                    {
                        "id": "dead_provenance_explanation",
                        "type": "markdown",
                        "sourceId": "dead_consensus",
                        "body": (
                            "The provenance distribution below quantifies how much of the final Dead result is supported "
                            "by both channels, by the Dead primary alone, or by a Combined-blue rescue that passed raw-signal evidence."
                        ),
                    },
                    {"id": "dead_provenance_chart_block", "type": "chart", "chartId": "dead_provenance_chart", "layout": "full"},
                ]
            )

    blocks.extend(
        [
            {
                "id": "nucleated_heading",
                "type": "markdown",
                "body": (
                    "## Nucleated-only Branch\n\n"
                    "This sensitivity branch removes cell masks without nucleus-centroid support, reruns classification on "
                    "the retained objects, and then reevaluates shape_strict using the filtered cell context. It does not "
                    "replace the all-cell result; both branches remain independently available."
                ),
            },
        ]
    )
    for key in ("nucleated_filter", "classification_nucleated", "shape_nucleated"):
        heading, paragraph, alt, caption, source_id = image_specs[key]
        blocks.extend(
            [
                {
                    "id": f"{key}_explanation",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": f"### {heading}\n\n{paragraph}",
                },
                {
                    "id": f"{key}_image",
                    "type": "html",
                    "body": image_block_body(image_uris[key], alt, caption, *image_sizes[key]),
                    "layout": "full",
                },
            ]
        )

    blocks.extend(
        [
            {
                "id": "branch_comparison_heading",
                "type": "markdown",
                "body": (
                    "## Branch Comparison\n\n"
                    "Use the all-cell branch when maximum inclusion is required. Use the nucleated-only branch as a "
                    "fragment-resistant sensitivity analysis. Differences between branches reflect the nucleus-support "
                    "filter and the resulting independent reclassification, not a change to the underlying raw images."
                ),
            },
            {"id": "fusion_branch_table_block", "type": "table", "tableId": "fusion_branch_table", "layout": "full"},
            {"id": "shape_branch_table_block", "type": "table", "tableId": "shape_branch_table", "layout": "full"},
            {
                "id": "scope_heading",
                "type": "markdown",
                "body": (
                    "## Scope and Result Layout\n\n"
                    f"The report describes the immutable result snapshot named `{run_root.name}`. Scientific outputs are "
                    "read-only inputs to this document; report generation does not rerun segmentation, classification, "
                    "filtering, fusion, or shape_strict."
                ),
            },
            {"id": "directory_map_block", "type": "table", "tableId": "directory_map_table", "layout": "full"},
            {
                "id": "methodology_heading",
                "type": "markdown",
                "body": (
                    "## Methodology and Workflow\n\n"
                    "Nuclei segmentation and full-cohort Dead calibration are upstream foundations. Nucleus cores drive "
                    "density routing; density and calibration then parameterize Brightfield, Combined, and Dead processing. "
                    "A validated field manifest binds all channels before the two classification and shape_strict branches."
                ),
            },
            {"id": "pipeline_table_block", "type": "table", "tableId": "pipeline_table", "layout": "full"},
            {"id": "profile_table_block", "type": "table", "tableId": "profile_table", "layout": "full"},
            {
                "id": "key_files_heading",
                "type": "markdown",
                "body": (
                    "## Key Machine-readable Outputs\n\n"
                    "Use field summaries for cohort-level analysis and prediction or provenance tables when object-level "
                    "auditability is required. Mask TIFF files remain the spatial source of truth."
                ),
            },
            {"id": "key_files_table_block", "type": "table", "tableId": "key_files_table", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and Robustness\n\n"
                    "- No manually annotated ground truth is available, so the report supports internal consistency and "
                    "cross-channel calibration rather than an absolute segmentation-accuracy estimate.\n"
                    "- Red-fluorescent nuclei can appear enlarged and have diffuse borders after longer exposure; extent "
                    "masks therefore should not be interpreted as exact physical nuclear boundaries.\n"
                    "- The six embedded fields are a stratified QC cohort, not a substitute for full-cohort distributional checks.\n"
                    "- Combined-blue rescue is intentionally conservative; weak true Dead objects may remain unrescued when "
                    "raw Dead evidence is insufficient.\n"
                    "- The nucleated-only branch can exclude genuinely anucleate or severely damaged cells and must be "
                    "reported alongside, not silently substituted for, the all-cell branch."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended Use\n\n"
                    "1. Treat the all-cell classification summaries as the primary inclusive output.\n"
                    "2. Repeat key biological comparisons with the nucleated-only summaries as a sensitivity analysis.\n"
                    "3. Review the same six-field QC sequence when code, parameters, or calibration inputs change.\n"
                    "4. For outlying wells, trace from the field summary to per-object predictions, provenance, and masks.\n"
                    "5. Keep baseline nuclei and shape_strict nuclei as separate analysis layers so accepted splits remain auditable."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further Questions\n\n"
                    "- Are treatment-level conclusions stable between the all-cell and nucleated-only branches?\n"
                    "- Do Dead-primary-only and blue-rescue fractions vary systematically by plate position or treatment?\n"
                    "- Are high-density wells enriched for accepted shape_strict splits or residual merged cell masks?\n"
                    "- Which fields are simultaneous outliers for nucleus density, artifact fraction, and branch-retention fraction?"
                ),
            },
        ]
    )

    return {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Methods, directory structure, result interpretation, and embedded QC for the completed SUM159 full-cohort workflow.",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": [
            {"id": source["id"], "label": source["label"], "path": source["path"]}
            for source in make_sources(run_root)
        ],
        "blocks": blocks,
    }


def prepare_methods_markdown(markdown: str) -> str:
    """Nest the repository README below the report's Methods chapter."""
    lines = markdown.splitlines()
    if lines and re.fullmatch(r"#\s+.+", lines[0].strip()):
        lines = lines[1:]
    nested: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            nested.append(line)
            continue
        if not in_fence:
            match = re.match(r"^(#{1,4})(\s+.*)$", line)
            if match:
                line = "#" * min(6, len(match.group(1)) + 2) + match.group(2)
        nested.append(line)
    return "\n".join(nested).strip()


def segmentation_figure_notes(sample: Sample) -> str:
    return (
        "**Panel guide and legend.** (A) The nucleus image uses magenta for the high-recall extent boundary and cyan "
        "for the intensity-supported core. (B-C) Green boundaries mark Brightfield and Combined cell instances; this "
        f"field used the **{'high-density' if sample.density_class.startswith('High') else 'baseline'}** cell profile. "
        "(D) Red boundaries are Dead-primary candidates. (E) Cyan boundaries are Combined blue-excess candidates. "
        "(F) Final Dead-object boundaries are green when both channels match, red for Dead-primary-only support, and "
        "blue for a Combined-blue rescue that passed raw-Dead evidence. (G) Cyan masks are retained in the "
        "nucleated-only branch and magenta masks are removed because nucleus-centroid support is absent.\n\n"
        "**How to read it.** Inspect A first to judge diffuse or overexposed nuclear edges; compare B and C for merged "
        "or fragmented cell boundaries; then compare D-F to see which signal created each final Dead object. Finish at "
        "G to determine whether the nucleus-supported branch removes plausible debris without discarding clearly "
        "nucleated cells. Every panel is the same field and is geometrically aligned."
    )


def analysis_figure_notes(sample: Sample) -> str:
    return (
        "**Panel guide and legend.** (A) All-cell fusion classification: green is live, red is dead, gray is artifact "
        "or another unresolved non-live/dead state, and yellow outlines the nucleus core. (B) All-cell shape_strict: "
        "baseline and strict nucleus masks are paired; magenta outlines the extent, cyan outlines the core, and yellow "
        "marks an accepted changed split boundary. (C-D) repeat A-B after removing cell masks without nuclear evidence.\n\n"
        "**How to read it.** Compare A with C to identify classification or count changes caused by the nuclear-evidence "
        "filter, not by a new raw image. Compare B with D to see whether the changed cell context alters strict split "
        "acceptance. A full-field negative-control view means no proposed split passed every stability, shape, foreground, "
        "and cell-support gate; it is not a missing result."
    )


def build_grouped_report_manifest(
    run_root: Path,
    generated_at: str,
    image_uris: dict[str, str],
    image_sizes: dict[str, tuple[int, int]],
    samples: list[Sample],
    methods_markdown: str,
    cards: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "SUM159 v3 Full-Cohort Segmentation and Classification Analysis Guide"
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical Summary\n\n"
                "This self-contained report documents the completed cohort-wide calibration, four-channel segmentation, "
                "density-aware routing, Dead/Combined-blue consensus, all-cell fusion classification, nucleated-only "
                "sensitivity branch, and the two corresponding shape_strict analyses. The same fixed cohort of six fields "
                "is used throughout: three high-density fields first, followed by three low-density fields."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "field_count",
                "high_density_count",
                "dead_object_count",
                "all_cell_count",
                "nucleated_cell_count",
                "shape_split_count",
            ],
        },
        {
            "id": "qc_selection_heading",
            "type": "markdown",
            "sourceId": "qc_selection",
            "body": (
                "## Fixed QC Cohort\n\n"
                "The six reviewed fields are locked in the order HD-1, HD-2, HD-3, LD-1, LD-2, LD-3. Reusing identical "
                "fields across all figures makes stage-to-stage and branch-to-branch differences interpretable."
            ),
        },
        {"id": "qc_sample_table_block", "type": "table", "tableId": "qc_sample_table", "layout": "full"},
        {
            "id": "results_summary_heading",
            "type": "markdown",
            "body": (
                "## Cohort-level Results\n\n"
                "The all-cell branch is the inclusive analysis. The nucleated-only branch independently filters cell masks "
                "and reruns classification, providing a sensitivity analysis for debris and non-cell masks. shape_strict "
                "remains a separate conservative nucleus-splitting layer and never overwrites baseline nucleus masks."
            ),
        },
        {"id": "fusion_branch_table_block", "type": "table", "tableId": "fusion_branch_table", "layout": "full"},
        {"id": "shape_branch_table_block", "type": "table", "tableId": "shape_branch_table", "layout": "full"},
        {
            "id": "result_layout_heading",
            "type": "markdown",
            "body": (
                "## Result Directory Structure\n\n"
                f"The immutable production snapshot is `{report_snapshot_name(run_root)}`. Each directory below represents one defined "
                "workflow stage or downstream analysis branch; mask TIFF files and per-object tables remain the spatial "
                "and tabular sources of truth."
            ),
        },
        {"id": "directory_map_block", "type": "table", "tableId": "directory_map_table", "layout": "full"},
        {
            "id": "methods_heading",
            "type": "markdown",
            "body": (
                "## Methods\n\n"
                "The workflow begins with environment and input validation. Nuclei segmentation and full-cohort Dead/"
                "Combined-blue calibration run independently; their outputs converge before density-aware Brightfield, "
                "Combined, and final Dead segmentation. A validated field manifest then supports the all-cell and "
                "nucleus-supported branches. Processing-stage descriptions in this chapter are text only; result images "
                "are collected later in the Figures chapter."
            ),
        },
        {
            "id": "workflow_diagram_heading",
            "type": "markdown",
            "body": (
                "### Method Diagram 1. Recommended production workflow\n\n"
                "The diagram shows the production dependency graph. Blue denotes segmentation work, green denotes "
                "calibration or validated outputs, orange denotes calibration and nucleus-supported filtering, purple "
                "denotes classification/shape analysis, and dark blue marks the `afterok` convergence gate."
            ),
        },
        {
            "id": "workflow_image",
            "type": "html",
            "body": image_block_body(
                image_uris["workflow"],
                "Recommended full production workflow dependency diagram",
                "Method Diagram 1. Recommended production workflow and dependency structure.",
                *image_sizes["workflow"],
            ),
            "layout": "full",
        },
        {"id": "pipeline_table_block", "type": "table", "tableId": "pipeline_table", "layout": "full"},
        {"id": "profile_table_block", "type": "table", "tableId": "profile_table", "layout": "full"},
        {
            "id": "detailed_methods",
            "type": "markdown",
            "body": "### Detailed Pipeline, Execution, and Calibration Reference\n\n" + methods_markdown,
        },
        {
            "id": "key_files_heading",
            "type": "markdown",
            "body": (
                "## Key Machine-readable Outputs\n\n"
                "Field summaries support cohort-level analysis; prediction and provenance tables support object-level "
                "auditing. The field manifest provides canonical channel and mask relationships."
            ),
        },
        {"id": "key_files_table_block", "type": "table", "tableId": "key_files_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and Robustness\n\n"
                "- No manually annotated ground truth is available; the evidence supports internal consistency and "
                "cross-channel calibration rather than an absolute accuracy estimate.\n"
                "- Longer red-fluorescence exposure can enlarge apparent nuclei and blur their edges. Extent masks are "
                "therefore high-recall operational boundaries, while cores provide more conservative nuclear evidence.\n"
                "- The six figures are a density-stratified QC cohort and do not replace full-cohort distribution checks.\n"
                "- Combined-blue rescue is conservative, so weak true Dead objects can remain unrescued.\n"
                "- The nucleated-only branch can exclude truly anucleate or severely damaged cells and must remain a "
                "reported sensitivity branch rather than silently replacing the all-cell result."
            ),
        },
        {
            "id": "figures_heading",
            "type": "markdown",
            "body": (
                "## Figures\n\n"
                "All result images are consolidated here. Figures 1-6 show segmentation QC for the six fixed fields. "
                "Figures 7-12 then pair all-cell and nucleated-only classification with their corresponding shape_strict "
                "views, preserving the same sample order. Figure 13 summarizes final Dead-object provenance across the "
                "full cohort."
            ),
        },
        {
            "id": "segmentation_figures_heading",
            "type": "markdown",
            "body": "### Segmentation QC Figures\n\nEach figure consolidates every major segmentation output for one field.",
        },
    ]

    for index, sample in enumerate(samples, start=1):
        token = sample.label.lower().replace("-", "")
        key = f"segmentation_{token}"
        figure_title = f"Figure {index}. {sample.label} ({sample.key}) segmentation QC"
        blocks.extend(
            [
                {
                    "id": f"{key}_heading",
                    "type": "markdown",
                    "body": f"### {figure_title}\n\n{segmentation_figure_notes(sample)}",
                },
                {
                    "id": f"{key}_image",
                    "type": "html",
                    "body": image_block_body(
                        image_uris[key],
                        f"{figure_title}, panels A through G",
                        figure_title,
                        *image_sizes[key],
                    ),
                    "layout": "full",
                },
            ]
        )

    blocks.append(
        {
            "id": "analysis_figures_heading",
            "type": "markdown",
            "body": (
                "### Classification and shape_strict Figures\n\n"
                "Within each figure, the all-cell branch appears first (A-B), followed by the nucleated-only branch (C-D)."
            ),
        }
    )
    for offset, sample in enumerate(samples, start=7):
        token = sample.label.lower().replace("-", "")
        key = f"analysis_{token}"
        figure_title = f"Figure {offset}. {sample.label} ({sample.key}) classification and shape_strict QC"
        blocks.extend(
            [
                {
                    "id": f"{key}_heading",
                    "type": "markdown",
                    "body": f"### {figure_title}\n\n{analysis_figure_notes(sample)}",
                },
                {
                    "id": f"{key}_image",
                    "type": "html",
                    "body": image_block_body(
                        image_uris[key],
                        f"{figure_title}, panels A through D",
                        figure_title,
                        *image_sizes[key],
                    ),
                    "layout": "full",
                },
            ]
        )

    blocks.extend(
        [
            {
                "id": "dead_provenance_figure_heading",
                "type": "markdown",
                "body": (
                    "### Figure 13. Full-cohort final Dead-object provenance\n\n"
                    "This cohort-level bar chart counts final Dead objects by evidence source. **Dual channel** means a "
                    "Dead-primary object matched a Combined-blue candidate; **Dead primary only** means the object passed "
                    "strict raw-Dead signal filtering without a blue match; **Combined-blue rescue** means a blue-only "
                    "candidate passed the relaxed raw-Dead evidence gate. Bar height is the object count, and the tooltip "
                    "reports the fraction of all final Dead objects. Read this chart after Figures 1-12 as a cohort-wide "
                    "context for the per-field provenance colors in panel F."
                ),
            },
            {
                "id": "dead_provenance_chart_block",
                "type": "chart",
                "chartId": "dead_provenance_chart",
                "layout": "full",
            },
        ]
    )

    return {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Detailed methods, directory structure, cohort summaries, and twelve per-sample embedded QC figures for "
            "the completed SUM159 full-cohort workflow."
        ),
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": [
            {"id": source["id"], "label": source["label"], "path": source["path"]}
            for source in make_sources(run_root)
        ],
        "blocks": blocks,
    }


def build_artifact(
    run_root: Path,
    datasets: dict[str, list[dict[str, Any]]],
    sheets: dict[str, Image.Image],
    samples: list[Sample],
    methods_markdown: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    base_scale = 1.0 / QC_RENDER_SCALE
    attempts = [
        (base_scale, 72),
        (base_scale, 64),
        (base_scale, 56),
        (base_scale * 0.92, 52),
        (base_scale * 0.85, 48),
        (base_scale * 0.78, 45),
        (base_scale * 0.72, 42),
    ]
    last_size = 0
    for scale, quality in attempts:
        uris: dict[str, str] = {}
        image_bytes: dict[str, int] = {}
        image_sha256: dict[str, str] = {}
        image_sizes: dict[str, tuple[int, int]] = {}
        for key, sheet in sheets.items():
            encoded_sheet = sheet
            if scale < 1.0:
                encoded_sheet = sheet.resize(
                    (max(1, int(round(sheet.width * scale))), max(1, int(round(sheet.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
            uris[key], image_bytes[key], image_sha256[key] = encode_sheet(encoded_sheet, quality)
            image_sizes[key] = encoded_sheet.size

        sources = make_sources(run_root)
        artifact = {
            "surface": "report",
            "manifest": build_manifest(
                run_root,
                generated_at,
                uris,
                image_sizes,
                samples,
                methods_markdown,
            ),
            "snapshot": {
                "version": 1,
                "generatedAt": generated_at,
                "status": "ready",
                "datasets": datasets,
            },
            "sources": sources,
            "package_info": {
                "originUrl": f"artifact://{report_snapshot_name(run_root)}/results-analysis-guide",
                "controls": {"edit": False, "refresh": False, "share": False},
            },
        }
        last_size = len(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if last_size <= MAX_ARTIFACT_BYTES:
            return artifact, {
                "artifact_bytes": last_size,
                "image_bytes": image_bytes,
                "image_sha256": image_sha256,
                "image_total_bytes": sum(image_bytes.values()),
                "image_sizes": {key: list(size) for key, size in image_sizes.items()},
                "scale": scale,
                "quality": quality,
                "source_render_scale": QC_RENDER_SCALE,
                "generated_at": generated_at,
            }
    raise RuntimeError(
        f"Embedded QC artifact remains too large after bounded compression: {last_size:,} > {MAX_ARTIFACT_BYTES:,} bytes"
    )


def write_debug_sheets(directory: Path, sheets: dict[str, Image.Image]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[282] = float(HIGH_RES_PRINT_PPI)
    exif[283] = float(HIGH_RES_PRINT_PPI)
    exif[296] = 2
    for key, sheet in sheets.items():
        if pil_features.check("webp"):
            sheet.save(
                directory / f"{key}.webp",
                format="WEBP",
                quality=HIGH_RES_WEBP_QUALITY,
                method=6,
                exact=True,
                exif=exif,
            )
        else:
            sheet.save(
                directory / f"{key}.jpg",
                format="JPEG",
                quality=HIGH_RES_JPEG_QUALITY,
                subsampling=0,
                optimize=True,
                progressive=True,
                dpi=(HIGH_RES_PRINT_PPI, HIGH_RES_PRINT_PPI),
                exif=exif,
            )


def report_navigation_entries(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current_parent: str | None = None
    for block in artifact.get("manifest", {}).get("blocks", []):
        if block.get("type") != "markdown":
            continue
        block_id = str(block.get("id", ""))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", block_id):
            continue
        match = re.search(r"^(#{2,3})\s+(.+?)\s*$", str(block.get("body", "")), flags=re.MULTILINE)
        if match is None:
            continue
        level = len(match.group(1))
        label = match.group(2)
        label = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", label)
        label = re.sub(r"[`*]", "", label).strip()
        if level == 2:
            current_parent = block_id
            entries.append({"id": block_id, "label": label, "level": 2, "parent": None})
        elif current_parent is not None:
            entries.append({"id": block_id, "label": label, "level": 3, "parent": current_parent})
    if not entries:
        raise RuntimeError("No level-2/3 report headings were available for the left navigation")
    return entries


def report_image_frames(artifact: dict[str, Any]) -> list[dict[str, int | str]]:
    frames: list[dict[str, int | str]] = []
    for block in artifact.get("manifest", {}).get("blocks", []):
        if block.get("type") != "html":
            continue
        block_id = str(block.get("id", ""))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", block_id):
            continue
        size = re.search(r'width="(\d+)"\s+height="(\d+)"', str(block.get("body", "")))
        if size is None:
            continue
        frames.append({"id": block_id, "width": int(size.group(1)), "height": int(size.group(2))})
    return frames


def report_carousel_groups(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = artifact.get("manifest", {}).get("blocks", [])
    blocks_by_id = {
        str(block.get("id", "")): block
        for block in blocks
        if str(block.get("id", ""))
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_blocks: set[str] = set()
    for block in blocks:
        if block.get("type") != "html" or "carouselGroup" not in block:
            continue
        image_id = str(block.get("id", ""))
        group_id = str(block.get("carouselGroup", ""))
        text_id = str(block.get("carouselTextId", ""))
        label = str(block.get("carouselLabel", "")).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", group_id):
            raise ValueError(f"Invalid report carousel group id: {group_id!r}")
        if image_id in seen_blocks or text_id in seen_blocks:
            raise ValueError(f"Report carousel block is reused: {image_id} / {text_id}")
        if text_id not in blocks_by_id or blocks_by_id[text_id].get("type") != "markdown":
            raise ValueError(
                f"Report carousel image {image_id} references missing markdown block {text_id}"
            )
        try:
            index = int(block.get("carouselIndex"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Report carousel image {image_id} has an invalid index"
            ) from error
        if index < 0 or not label:
            raise ValueError(
                f"Report carousel image {image_id} requires a nonnegative index and label"
            )
        grouped.setdefault(group_id, []).append(
            {
                "index": index,
                "label": label,
                "textId": text_id,
                "imageId": image_id,
            }
        )
        seen_blocks.update((image_id, text_id))

    result: list[dict[str, Any]] = []
    for group_id, slides in grouped.items():
        slides.sort(key=lambda slide: slide["index"])
        indices = [slide["index"] for slide in slides]
        if len(slides) < 2 or indices != list(range(len(slides))):
            raise ValueError(
                f"Report carousel {group_id} must contain contiguous indices from zero: {indices}"
            )
        result.append({"id": group_id, "slides": slides})
    return result


def validate_high_resolution_image_payload(
    payload: dict[str, Any] | None,
    image_frames: list[dict[str, int | str]],
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if payload.get("version") != 1 or not isinstance(payload.get("images"), dict):
        raise ValueError("High-resolution image payload must use version=1 and contain an images object")
    if int(payload.get("print_ppi", 0)) != HIGH_RES_PRINT_PPI:
        raise ValueError(f"High-resolution image payload must carry {HIGH_RES_PRINT_PPI} PPI metadata")
    images = payload["images"]
    expected = {str(frame["id"]): frame for frame in image_frames}
    missing = sorted(set(expected) - set(images))
    extra = sorted(set(images) - set(expected))
    if missing or extra:
        raise ValueError(f"High-resolution image ids do not match report frames: missing={missing}, extra={extra}")
    for image_id, frame in expected.items():
        image = images[image_id]
        if not isinstance(image, dict):
            raise TypeError(f"High-resolution image entry must be an object: {image_id}")
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        data_uri = str(image.get("data_uri", ""))
        if width < int(frame["width"]) * 1.8 or height < int(frame["height"]) * 1.8:
            raise ValueError(
                f"High-resolution image is not at least 1.8x the canonical frame: {image_id} "
                f"{width}x{height} versus {frame['width']}x{frame['height']}"
            )
        source_ratio = int(frame["width"]) / int(frame["height"])
        high_ratio = width / height
        if abs(source_ratio - high_ratio) > 0.002:
            raise ValueError(f"High-resolution image aspect ratio changed: {image_id}")
        if not re.fullmatch(r"data:image/(?:webp|jpeg);base64,[A-Za-z0-9+/=]+", data_uri):
            raise ValueError(f"Unsupported high-resolution image data URI: {image_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(image.get("sha256", ""))):
            raise ValueError(f"Missing high-resolution image SHA256: {image_id}")
    return payload


def high_resolution_payload_script(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return ""
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    return (
        '<script id="report-high-resolution-images" type="application/json" '
        'data-report-image-quality-version="1">'
        f"{encoded}</script>"
    )


def navigation_style(image_frames: list[dict[str, int | str]]) -> str:
    frame_rules: list[str] = []
    for frame in image_frames:
        block_id = frame["id"]
        width = int(frame["width"])
        height = int(frame["height"]) + 32
        frame_rules.append(
            f'#{block_id} iframe.report-html-frame,'
            f'.portable-block[data-artifact-block-id="{block_id}"] .portable-custom-html iframe{{'
            f'height:auto!important;min-height:0!important;aspect-ratio:{width}/{height};'
            'overflow:hidden!important;scrollbar-width:none!important}'
        )
    return """
<style id="report-navigation-style" data-report-navigation-version="1">
#report-toc{position:fixed;z-index:90;top:60px;left:12px;display:flex;flex-direction:column;width:232px;max-height:calc(100vh - 76px);padding:12px 10px;border:1px solid var(--portable-border,#d9d9d9);border-radius:14px;background:var(--portable-canvas,Canvas);color:var(--portable-ink,CanvasText);box-shadow:0 10px 28px rgba(0,0,0,.08)}
#report-toc-header{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:0 6px 8px;border-bottom:1px solid var(--portable-border,#d9d9d9)}
#report-toc-title{font-size:12px;font-weight:700;line-height:18px;letter-spacing:.04em;text-transform:uppercase}
#report-toc-list{min-height:0;padding:8px 0 2px;overflow-y:auto;overscroll-behavior:contain;scrollbar-width:thin}
.report-toc-group+.report-toc-group{margin-top:2px}
.report-toc-row{display:flex;align-items:center;gap:2px}
.report-toc-link{display:block;min-width:0;flex:1;padding:7px 8px;border-radius:8px;color:var(--portable-muted,#5d5d5d);font-size:12px;line-height:16px;text-decoration:none;overflow-wrap:anywhere}
.report-toc-link:hover{background:var(--portable-surface-subtle,#f5f5f5);color:var(--portable-ink,CanvasText)}
.report-toc-link.is-active{background:var(--portable-surface-subtle,#f1f1f1);color:var(--portable-accent,#0b57d0);font-weight:650}
.report-toc-toggle{flex:0 0 26px;width:26px;height:26px;padding:0;border:0;border-radius:7px;background:transparent;color:var(--portable-muted,#5d5d5d);font-size:18px;line-height:26px;cursor:pointer;transition:transform .16s ease}
.report-toc-toggle:hover{background:var(--portable-surface-subtle,#f5f5f5)}
.report-toc-group.is-expanded>.report-toc-row .report-toc-toggle{transform:rotate(90deg)}
.report-toc-children{max-height:0;margin-left:8px;padding-left:8px;overflow:hidden;border-left:1px solid var(--portable-border,#d9d9d9);opacity:.35;transition:max-height .18s ease,opacity .18s ease}
.report-toc-group.is-expanded>.report-toc-children{max-height:520px;opacity:1}
.report-toc-child{padding-top:5px;padding-bottom:5px;font-size:11px;line-height:15px}
#report-toc-close,#report-toc-mobile-toggle{font:600 12px/18px system-ui,sans-serif;cursor:pointer}
#report-toc-close{display:none;width:28px;height:28px;padding:0;border:0;border-radius:8px;background:transparent;color:inherit;font-size:20px;line-height:28px}
#report-toc-mobile-toggle{position:fixed;z-index:92;top:58px;left:12px;display:none;padding:7px 11px;border:1px solid var(--portable-border,#d9d9d9);border-radius:999px;background:var(--portable-canvas,Canvas);color:var(--portable-ink,CanvasText);box-shadow:0 6px 18px rgba(0,0,0,.1)}
#report-toc-backdrop{position:fixed;z-index:88;inset:0;display:none;background:rgba(0,0,0,.26)}
.report-html-frame,.portable-custom-html iframe{overflow:hidden!important}
.portable-custom-html{overflow:visible!important}
.report-toc-link:focus-visible,.report-toc-toggle:focus-visible,#report-toc-close:focus-visible,#report-toc-mobile-toggle:focus-visible{outline:2px solid var(--portable-accent,#0b57d0);outline-offset:2px}
#report-image-lightbox[hidden]{display:none!important}
#report-image-lightbox{position:fixed;z-index:180;inset:0;display:flex;flex-direction:column;background:rgba(8,10,14,.94);color:#fff}
#report-lightbox-toolbar{display:flex;align-items:center;gap:8px;min-height:54px;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.18);background:rgba(15,17,22,.96)}
#report-lightbox-title{min-width:0;flex:1;overflow:hidden;color:#f8fafc;font:500 13px/18px system-ui,sans-serif;text-overflow:ellipsis;white-space:nowrap}
#report-lightbox-hint{color:#cbd5e1;font:500 12px/18px system-ui,sans-serif;white-space:nowrap}
.report-lightbox-button{height:34px;min-width:34px;padding:0 10px;border:1px solid rgba(255,255,255,.28);border-radius:8px;background:rgba(255,255,255,.08);color:#fff;font:600 12px/18px system-ui,sans-serif;cursor:pointer}
.report-lightbox-button:hover{background:rgba(255,255,255,.16)}
.report-lightbox-button:focus-visible{outline:2px solid #93c5fd;outline-offset:2px}
#report-lightbox-viewport{position:relative;flex:1;min-height:0;overflow:hidden;touch-action:none;cursor:zoom-out}
#report-lightbox-viewport.is-dragging{cursor:grabbing}
#report-lightbox-image{position:absolute;top:50%;left:50%;max-width:none;max-height:none;transform-origin:center center;cursor:grab;user-select:none;-webkit-user-drag:none;will-change:transform}
#report-lightbox-viewport.is-dragging #report-lightbox-image{cursor:grabbing}
body.report-lightbox-open{overflow:hidden!important}
#data-analytics-portable-fallback .portable-block-stack{width:100%!important;max-width:none!important;margin-right:0!important;margin-left:0!important}
.report-image-interaction-host{position:relative!important}
.report-image-interaction-surface{position:absolute;z-index:3;inset:0;display:block;min-width:0;min-height:0;margin:0;padding:0;border:0;background:transparent;cursor:zoom-in;touch-action:pan-y}
.report-image-interaction-surface:focus-visible{outline:2px solid var(--portable-accent,#0b57d0);outline-offset:-4px}
.report-carousel{--report-carousel-image-height-cap:1800px;position:relative;grid-column:1/-1!important;width:100%!important;max-width:none!important;max-height:calc(100dvh - 64px);box-sizing:border-box;margin:10px 0 28px;padding:0 50px 14px;border:1px solid var(--portable-border,#d9d9d9);border-radius:16px;background:var(--portable-canvas,Canvas);box-shadow:0 8px 24px rgba(0,0,0,.06)}
.report-carousel-viewport{width:100%;min-height:0;overflow:hidden;overflow:clip;touch-action:pan-y;overscroll-behavior-x:contain}
.report-carousel-track{display:flex;width:100%;align-items:stretch;transform:translate3d(0,0,0);transition:transform .32s cubic-bezier(.22,.61,.36,1);will-change:transform}
.report-carousel.is-dragging .report-carousel-track{transition:none}
.report-carousel-slide{display:flex;flex:0 0 100%;min-width:0;min-height:0;flex-direction:column;align-self:stretch;box-sizing:border-box;padding:0 4px}
.report-carousel-slide>*{width:100%!important;max-width:100%!important;box-sizing:border-box}
.report-carousel-arrow{position:absolute;z-index:4;top:50%;display:grid;width:38px;height:38px;margin-top:-31px;padding:0;place-items:center;border:1px solid rgba(255,255,255,.32);border-radius:999px;background:rgba(23,28,38,.78);color:#fff;font:600 27px/1 system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.22);cursor:pointer;transition:opacity .16s ease,transform .16s ease,background .16s ease}
.report-carousel-arrow:hover:not(:disabled){background:rgba(23,28,38,.94);transform:scale(1.06)}
.report-carousel-arrow:disabled{opacity:.24;cursor:default}
.report-carousel-arrow:focus-visible,.report-carousel-dot:focus-visible,.report-carousel-viewport:focus-visible{outline:2px solid var(--portable-accent,#0b57d0);outline-offset:3px}
.report-carousel-prev{left:7px}.report-carousel-next{right:7px}
.report-carousel-progress{display:flex;min-height:28px;align-items:center;justify-content:center;gap:9px;padding:10px 0 0}
.report-carousel-dot{width:11px;height:11px;padding:0;border:1.5px solid var(--portable-muted,#777);border-radius:999px;background:transparent;cursor:pointer;transition:width .16s ease,background .16s ease,border-color .16s ease}
.report-carousel-dot[aria-current="true"]{width:25px;border-color:var(--portable-accent,#0b57d0);background:var(--portable-accent,#0b57d0)}
.report-carousel-status{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.report-carousel iframe{display:block;max-width:100%!important;max-height:var(--report-carousel-image-height-cap)!important;margin-right:auto!important;margin-left:auto!important}
@media screen and (min-width:1280px){
  #data-analytics-portable-fallback .portable-block-stack{width:calc(100% - 244px)!important;margin-left:244px!important}
}
	@media screen and (max-width:1279px){
	  #report-toc{top:12px;left:12px;max-height:calc(100vh - 24px);transform:translateX(calc(-100% - 24px));transition:transform .2s ease}
  #report-toc-close,#report-toc-mobile-toggle{display:block}
  body.report-toc-open{overflow:hidden}
  body.report-toc-open #report-toc{transform:translateX(0)}
  body.report-toc-open #report-toc-backdrop{display:block}
  #report-lightbox-hint{display:none}
  .report-carousel{padding-right:42px;padding-left:42px}
  .report-carousel-arrow{width:34px;height:34px;font-size:24px}
}
@media print{
  #report-toc,#report-toc-mobile-toggle,#report-toc-backdrop,.report-carousel-arrow,.report-carousel-progress{display:none!important}
  .report-carousel{padding:0;border:0;box-shadow:none}
  .report-carousel-viewport{overflow:visible}
  .report-carousel-track{display:block!important;transform:none!important}
  .report-carousel-slide{display:block!important;width:100%!important;break-inside:avoid;page-break-inside:avoid}
  .report-carousel iframe{width:100%!important;height:auto!important;max-height:none!important}
}
	""" + "\n".join(frame_rules) + "\n</style>"


def lightbox_markup() -> str:
    return (
        '<div id="report-image-lightbox" hidden role="dialog" aria-modal="true" '
        'aria-labelledby="report-lightbox-title">'
        '<div id="report-lightbox-toolbar">'
        '<div id="report-lightbox-title">High-resolution QC image</div>'
        '<div id="report-lightbox-hint">Drag or zoom inside the image; click outside to return</div>'
        '<button class="report-lightbox-button" id="report-lightbox-zoom-out" type="button" '
        'aria-label="Zoom out">-</button>'
        '<button class="report-lightbox-button" id="report-lightbox-zoom-in" type="button" '
        'aria-label="Zoom in">+</button>'
        '<button class="report-lightbox-button" id="report-lightbox-fit" type="button" '
        'aria-label="Fit image to screen">Fit</button>'
        '<button class="report-lightbox-button" id="report-lightbox-actual" type="button" '
        'aria-label="Show image at original pixel size">1:1</button>'
        '<button class="report-lightbox-button" id="report-lightbox-close" type="button" '
        'aria-label="Close high-resolution image viewer">Close</button>'
        "</div>"
        '<div id="report-lightbox-viewport">'
        '<img id="report-lightbox-image" alt="">'
        "</div>"
        "</div>"
    )


def navigation_script(
    entries: list[dict[str, Any]],
    image_frames: list[dict[str, int | str]],
    carousel_groups: list[dict[str, Any]],
) -> str:
    entries_json = json.dumps(entries, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    image_ids_json = json.dumps([frame["id"] for frame in image_frames], separators=(",", ":"))
    carousels_json = json.dumps(
        carousel_groups,
        ensure_ascii=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return """
<script id="report-navigation-script" data-report-navigation-version="1">
(()=>{
const entries=__ENTRIES__;
const imageIds=__IMAGE_IDS__;
const carouselGroups=__CAROUSELS__;
const toc=document.getElementById("report-toc");
const list=document.getElementById("report-toc-list");
const mobileToggle=document.getElementById("report-toc-mobile-toggle");
const closeButton=document.getElementById("report-toc-close");
const backdrop=document.getElementById("report-toc-backdrop");
const lightbox=document.getElementById("report-image-lightbox");
const lightboxTitle=document.getElementById("report-lightbox-title");
const lightboxViewport=document.getElementById("report-lightbox-viewport");
const lightboxImage=document.getElementById("report-lightbox-image");
const lightboxClose=document.getElementById("report-lightbox-close");
const lightboxZoomIn=document.getElementById("report-lightbox-zoom-in");
const lightboxZoomOut=document.getElementById("report-lightbox-zoom-out");
const lightboxFit=document.getElementById("report-lightbox-fit");
const lightboxActual=document.getElementById("report-lightbox-actual");
const highResolutionNode=document.getElementById("report-high-resolution-images");
let highResolutionImages={};
if(highResolutionNode){try{highResolutionImages=JSON.parse(highResolutionNode.textContent).images||{}}catch(error){console.error("Unable to read embedded high-resolution QC images",error)}}
const links=new Map();
const groups=new Map();
const toggles=new Map();
const childMap=new Map();
const carouselStates=new WeakMap();
const CAROUSEL_IMAGE_HEIGHT_CAP=1800;
const CAROUSEL_VIEWPORT_TOP=56;
const CAROUSEL_VIEWPORT_BOTTOM=16;
for(const entry of entries){if(entry.level===3){if(!childMap.has(entry.parent))childMap.set(entry.parent,[]);childMap.get(entry.parent).push(entry)}}
function targetFor(id){
  const reader=document.getElementById("data-analytics-portable-reader");
  const enhanced=document.getElementById(id);
  if(reader&&reader.getAttribute("aria-hidden")!=="true"&&getComputedStyle(reader).display!=="none"&&enhanced&&reader.contains(enhanced))return enhanced;
  const fallback=document.getElementById("data-analytics-portable-fallback");
  if(fallback&&getComputedStyle(fallback).display!=="none")return fallback.querySelector(`[data-artifact-block-id="${id}"]`);
  return enhanced||fallback?.querySelector(`[data-artifact-block-id="${id}"]`)||null;
}
function blockForSurface(root,surface,id){
  if(surface==="reader")return root.querySelector(`#${id}`);
  return root.querySelector(`.portable-block[data-artifact-block-id="${id}"]`);
}
function visibleViewportHeight(){
  return Math.max(320,Math.floor(window.visualViewport?.height||window.innerHeight||800));
}
function numericStyle(style,name){
  const value=Number.parseFloat(style.getPropertyValue(name));return Number.isFinite(value)?value:0;
}
function frameAspectRatio(frame){
  const declared=getComputedStyle(frame).aspectRatio||"";
  const parts=declared.split("/").map(value=>Number.parseFloat(value.trim()));
  if(parts.length===2&&parts.every(value=>Number.isFinite(value)&&value>0))return parts[0]/parts[1];
  const rect=frame.getBoundingClientRect();
  if(rect.width>0&&rect.height>0)return rect.width/rect.height;
  const id=frame.dataset.reportImageId||frame.closest(".portable-block")?.dataset.artifactBlockId;
  const image=id?highResolutionImages[id]:null;
  if(image?.width>0&&image?.height>0)return image.width/image.height;
  return 1;
}
function fitCarouselToViewport(state){
  if(!state?.wrapper?.isConnected)return;
  const viewportHeight=visibleViewportHeight();
  const wrapperStyle=getComputedStyle(state.wrapper);
  const progressHeight=state.progress.getBoundingClientRect().height;
  const wrapperChrome=
    numericStyle(wrapperStyle,"padding-top")+numericStyle(wrapperStyle,"padding-bottom")+
    numericStyle(wrapperStyle,"border-top-width")+numericStyle(wrapperStyle,"border-bottom-width")+
    progressHeight+12;
  const availableWidth=Math.max(160,state.viewport.clientWidth-8);
  let maximumFittedHeight=0;
  state.slideNodes.forEach(slide=>{
    const frame=slide.querySelector("iframe");if(!frame)return;
    const textBlock=slide.children[0];
    const textHeight=textBlock?.getBoundingClientRect().height||0;
    const availableHeight=Math.max(
      120,
      viewportHeight-CAROUSEL_VIEWPORT_TOP-CAROUSEL_VIEWPORT_BOTTOM-wrapperChrome-textHeight,
    );
    const heightCap=Math.min(CAROUSEL_IMAGE_HEIGHT_CAP,availableHeight);
    const ratio=Math.max(.1,frameAspectRatio(frame));
    const fittedWidth=Math.max(1,Math.min(availableWidth,heightCap*ratio));
    const fittedHeight=Math.max(1,fittedWidth/ratio);
    frame.style.setProperty("width",`${Math.floor(fittedWidth)}px`,"important");
    frame.style.setProperty("height",`${Math.floor(fittedHeight)}px`,"important");
    frame.style.setProperty("max-height",`${CAROUSEL_IMAGE_HEIGHT_CAP}px`,"important");
    maximumFittedHeight=Math.max(maximumFittedHeight,fittedHeight);
  });
  state.wrapper.dataset.carouselImageHeightCap=String(CAROUSEL_IMAGE_HEIGHT_CAP);
  state.wrapper.dataset.carouselFittedImageHeight=String(Math.round(maximumFittedHeight));
}
function fitStandaloneImageFrame(frame){
  if(!frame?.isConnected||frame.closest(".report-carousel"))return;
  const host=frame.parentElement;
  const availableWidth=Math.max(
    160,
    host?.clientWidth||frame.getBoundingClientRect().width||160,
  );
  const heightCap=Math.min(
    CAROUSEL_IMAGE_HEIGHT_CAP,
    Math.max(
      120,
      visibleViewportHeight()-CAROUSEL_VIEWPORT_TOP-CAROUSEL_VIEWPORT_BOTTOM-140,
    ),
  );
  const ratio=Math.max(.1,frameAspectRatio(frame));
  const fittedWidth=Math.max(1,Math.min(availableWidth,heightCap*ratio));
  frame.style.setProperty("width",`${Math.floor(fittedWidth)}px`,"important");
  frame.style.setProperty("height",`${Math.floor(fittedWidth/ratio)}px`,"important");
  frame.style.setProperty("max-height",`${CAROUSEL_IMAGE_HEIGHT_CAP}px`,"important");
  frame.dataset.reportViewportFitted="true";
}
function fitAllCarousels(keepVisible=false){
  let visibleState=null;let visibleArea=0;
  const viewportHeight=visibleViewportHeight();
  document.querySelectorAll(".report-carousel").forEach(wrapper=>{
    const state=carouselStates.get(wrapper);if(!state)return;
    fitCarouselToViewport(state);
    if(keepVisible){
      const rect=wrapper.getBoundingClientRect();
      const area=Math.max(
        0,
        Math.min(rect.bottom,viewportHeight-CAROUSEL_VIEWPORT_BOTTOM)-
        Math.max(rect.top,CAROUSEL_VIEWPORT_TOP),
      );
      if(area>visibleArea){visibleArea=area;visibleState=state}
    }
  });
  document.querySelectorAll("iframe[data-report-image-id]").forEach(
    frame=>fitStandaloneImageFrame(frame),
  );
  if(visibleState&&visibleArea>0)requestAnimationFrame(()=>keepCarouselInViewport(visibleState));
}
function keepCarouselInViewport(state){
  const rect=state.wrapper.getBoundingClientRect();
  const viewportHeight=visibleViewportHeight();
  if(rect.height>viewportHeight-CAROUSEL_VIEWPORT_TOP-CAROUSEL_VIEWPORT_BOTTOM)return;
  if(rect.top<CAROUSEL_VIEWPORT_TOP||rect.bottom>viewportHeight-CAROUSEL_VIEWPORT_BOTTOM){
    window.scrollTo({
      top:Math.max(0,window.scrollY+rect.top-CAROUSEL_VIEWPORT_TOP),
      behavior:"smooth",
    });
  }
}
function updateCarousel(state,index,animate=true){
  const last=state.slides.length-1;
  state.index=Math.max(0,Math.min(last,index));
  state.viewport.scrollLeft=0;
  state.track.style.transition=animate?"":"none";
  state.track.style.transform=`translate3d(${-100*state.index}%,0,0)`;
  if(!animate)requestAnimationFrame(()=>{state.track.style.transition=""});
  state.previous.disabled=state.index===0;
  state.next.disabled=state.index===last;
  state.dots.forEach((dot,dotIndex)=>{
    const current=dotIndex===state.index;
    dot.setAttribute("aria-current",String(current));
    dot.setAttribute("aria-label",`${current?"Current: ":"Show "}${state.slides[dotIndex].label}, slide ${dotIndex+1} of ${state.slides.length}`);
  });
  state.slideNodes.forEach((slide,slideIndex)=>{
    const current=slideIndex===state.index;
    slide.setAttribute("aria-hidden",String(!current));
    if("inert" in slide)slide.inert=!current;
  });
  state.status.textContent=`${state.slides[state.index].label}, slide ${state.index+1} of ${state.slides.length}`;
  fitCarouselToViewport(state);
  requestAnimationFrame(()=>{
    state.viewport.scrollLeft=0;fitCarouselToViewport(state);
    if(animate)keepCarouselInViewport(state);
  });
}
function beginCarouselDrag(state,event,captureTarget){
  if(document.body.classList.contains("report-lightbox-open")||event.button!==0)return;
  state.pointerId=event.pointerId;state.startX=event.clientX;state.startY=event.clientY;
  state.lastX=event.clientX;state.dragging=false;state.horizontal=false;state.captureTarget=captureTarget;
  try{captureTarget.setPointerCapture(event.pointerId)}catch(error){}
}
function moveCarouselDrag(state,event){
  if(event.pointerId!==state.pointerId)return;
  const dx=event.clientX-state.startX;const dy=event.clientY-state.startY;state.lastX=event.clientX;
  if(!state.horizontal&&Math.hypot(dx,dy)>8){
    if(Math.abs(dx)<=Math.abs(dy)){state.pointerId=null;return}
    state.horizontal=true;state.dragging=true;state.wrapper.classList.add("is-dragging");
  }
  if(!state.horizontal)return;
  event.preventDefault();
  const width=Math.max(1,state.viewport.getBoundingClientRect().width);
  const boundedDx=(state.index===0&&dx>0)||(state.index===state.slides.length-1&&dx<0)?dx*.28:dx;
  state.track.style.transform=`translate3d(calc(${-100*state.index}% + ${boundedDx}px),0,0)`;
}
function endCarouselDrag(state,event){
  if(event.pointerId!==state.pointerId)return false;
  const dx=event.clientX-state.startX;
  const wasDragging=state.dragging;
  if(state.captureTarget?.hasPointerCapture?.(event.pointerId))state.captureTarget.releasePointerCapture(event.pointerId);
  state.pointerId=null;state.captureTarget=null;state.dragging=false;state.horizontal=false;state.wrapper.classList.remove("is-dragging");
  if(wasDragging){
    const width=Math.max(1,state.viewport.getBoundingClientRect().width);
    const threshold=Math.min(120,Math.max(45,width*.10));
    const target=Math.abs(dx)>=threshold?state.index+(dx<0?1:-1):state.index;
    updateCarousel(state,target,true);
    state.suppressClick=true;setTimeout(()=>{state.suppressClick=false},180);
  }else updateCarousel(state,state.index,true);
  return wasDragging;
}
function bindCarouselPointerSurface(state,target){
  target.addEventListener("pointerdown",event=>{
    if(event.target.closest?.("button"))return;
    beginCarouselDrag(state,event,target);
  });
  target.addEventListener("pointermove",event=>moveCarouselDrag(state,event));
  target.addEventListener("pointerup",event=>endCarouselDrag(state,event));
  target.addEventListener("pointercancel",event=>endCarouselDrag(state,event));
}
function buildCarousel(root,surface,group){
  if(root.querySelector(`.report-carousel[data-carousel-group="${group.id}"]`))return;
  const nodes=group.slides.map(slide=>({
    meta:slide,
    text:blockForSurface(root,surface,slide.textId),
    image:blockForSurface(root,surface,slide.imageId),
  }));
  if(nodes.some(item=>!item.text||!item.image))return;
  const parent=nodes[0].text.parentNode;
  if(!parent||nodes.some(item=>item.text.parentNode!==parent||item.image.parentNode!==parent))return;
  const wrapper=document.createElement("section");wrapper.className="report-carousel";
  wrapper.dataset.carouselGroup=group.id;wrapper.dataset.carouselSurface=surface;
  wrapper.setAttribute("role","region");wrapper.setAttribute("aria-roledescription","carousel");
  wrapper.setAttribute("aria-label",`Figure carousel: ${group.slides.map(slide=>slide.label).join(", ")}`);
  const viewport=document.createElement("div");viewport.className="report-carousel-viewport";
  viewport.tabIndex=0;
  const track=document.createElement("div");track.className="report-carousel-track";
  const slideNodes=[];
  parent.insertBefore(wrapper,nodes[0].text);
  for(const item of nodes){
    const slide=document.createElement("div");slide.className="report-carousel-slide";
    slide.dataset.carouselIndex=String(item.meta.index);
    slide.setAttribute("role","group");slide.setAttribute("aria-roledescription","slide");
    slide.setAttribute("aria-label",`${item.meta.label}, slide ${item.meta.index+1} of ${nodes.length}`);
    slide.append(item.text,item.image);track.appendChild(slide);slideNodes.push(slide);
  }
  viewport.appendChild(track);
  const previous=document.createElement("button");previous.type="button";
  previous.className="report-carousel-arrow report-carousel-prev";previous.textContent="‹";
  previous.setAttribute("aria-label","Show previous figure");
  const next=document.createElement("button");next.type="button";
  next.className="report-carousel-arrow report-carousel-next";next.textContent="›";
  next.setAttribute("aria-label","Show next figure");
  const progress=document.createElement("div");progress.className="report-carousel-progress";
  const dots=group.slides.map((slide,index)=>{
    const dot=document.createElement("button");dot.type="button";dot.className="report-carousel-dot";
    dot.title=slide.label;dot.addEventListener("click",()=>updateCarousel(state,index,true));
    progress.appendChild(dot);return dot;
  });
  const status=document.createElement("div");status.className="report-carousel-status";
  status.setAttribute("aria-live","polite");progress.appendChild(status);
  const state={wrapper,viewport,track,previous,next,progress,status,dots,slides:group.slides,slideNodes,index:0,pointerId:null,startX:0,startY:0,lastX:0,dragging:false,horizontal:false,captureTarget:null,suppressClick:false};
  carouselStates.set(wrapper,state);
  previous.addEventListener("click",()=>updateCarousel(state,state.index-1,true));
  next.addEventListener("click",()=>updateCarousel(state,state.index+1,true));
  viewport.addEventListener("keydown",event=>{
    if(document.body.classList.contains("report-lightbox-open"))return;
    if(event.key==="ArrowLeft"){event.preventDefault();updateCarousel(state,state.index-1,true)}
    else if(event.key==="ArrowRight"){event.preventDefault();updateCarousel(state,state.index+1,true)}
    else if(event.key==="Home"){event.preventDefault();updateCarousel(state,0,true)}
    else if(event.key==="End"){event.preventDefault();updateCarousel(state,state.slides.length-1,true)}
  });
  bindCarouselPointerSurface(state,viewport);
  wrapper.append(viewport,previous,next,progress);
  updateCarousel(state,0,false);
}
function configureCarousels(){
  const reader=document.getElementById("data-analytics-portable-reader");
  const fallback=document.getElementById("data-analytics-portable-fallback");
  if(carouselGroups.length&&fallback){
    fallback.classList.remove("portable-enhanced-hidden");
    fallback.style.display="block";fallback.removeAttribute("aria-hidden");
    if(reader){reader.style.display="none";reader.setAttribute("aria-hidden","true")}
  }
  for(const group of carouselGroups){
    if(reader)buildCarousel(reader,"reader",group);
    if(fallback)buildCarousel(fallback,"fallback",group);
  }
  requestAnimationFrame(()=>fitAllCarousels(false));
}
function carouselStateForFrame(frame){
  const wrapper=frame.closest(".report-carousel");
  return wrapper?carouselStates.get(wrapper):null;
}
function closeDrawer(){document.body.classList.remove("report-toc-open");mobileToggle?.setAttribute("aria-expanded","false")}
function scrollToEntry(entry){const target=targetFor(entry.id);if(!target)return;const top=target.getBoundingClientRect().top+window.scrollY-68;window.scrollTo({top:Math.max(0,top),behavior:"smooth"});closeDrawer()}
function makeLink(entry,child=false){const link=document.createElement("a");link.className=`report-toc-link${child?" report-toc-child":""}`;link.href=`#${entry.id}`;link.textContent=entry.label;link.dataset.targetId=entry.id;link.addEventListener("click",event=>{event.preventDefault();scrollToEntry(entry)});links.set(entry.id,link);return link}
for(const entry of entries.filter(item=>item.level===2)){
  const group=document.createElement("div");group.className="report-toc-group";group.dataset.groupId=entry.id;
  const row=document.createElement("div");row.className="report-toc-row";row.appendChild(makeLink(entry));
  const children=childMap.get(entry.id)||[];
  if(children.length){
    const toggle=document.createElement("button");toggle.className="report-toc-toggle";toggle.type="button";toggle.textContent="›";toggle.setAttribute("aria-label",`Toggle ${entry.label} subsections`);toggle.setAttribute("aria-expanded","false");toggle.addEventListener("click",()=>{const expanded=group.classList.toggle("is-expanded");toggle.setAttribute("aria-expanded",String(expanded))});row.appendChild(toggle);toggles.set(entry.id,toggle);
    const childBox=document.createElement("div");childBox.className="report-toc-children";for(const child of children)childBox.appendChild(makeLink(child,true));group.append(row,childBox);
  }else group.appendChild(row);
  list.appendChild(group);groups.set(entry.id,group);
}
function setActive(entry){
  for(const [id,link] of links){const active=id===entry.id;link.classList.toggle("is-active",active);if(active)link.setAttribute("aria-current","location");else link.removeAttribute("aria-current")}
  const activeGroup=entry.level===3?entry.parent:entry.id;
  for(const [id,group] of groups){const expanded=id===activeGroup&&childMap.has(id);group.classList.toggle("is-expanded",expanded);toggles.get(id)?.setAttribute("aria-expanded",String(expanded))}
  links.get(entry.id)?.scrollIntoView({block:"nearest"});
}
let updateQueued=false;
let activeLightboxId=null;
let lightboxScale=1;
let lightboxFitScale=1;
let lightboxPanX=0;
let lightboxPanY=0;
let lightboxDragging=false;
let lightboxDragStartX=0;
let lightboxDragStartY=0;
let lightboxDragPanX=0;
let lightboxDragPanY=0;
let lightboxPointerId=null;
let lightboxBackgroundPress=false;
let lightboxPointerMoved=false;
function updateActive(){
  updateQueued=false;const positioned=entries.map(entry=>({entry,target:targetFor(entry.id)})).filter(item=>item.target).map(item=>({...item,top:item.target.getBoundingClientRect().top+window.scrollY}));if(!positioned.length)return;
  const marker=window.scrollY+104;let active=positioned[0].entry;for(const item of positioned){if(item.top<=marker)active=item.entry;else break}setActive(active)
}
function queueUpdate(){if(updateQueued)return;updateQueued=true;requestAnimationFrame(updateActive)}
function updateLightboxTransform(){
  lightboxImage.style.transform=`translate(calc(-50% + ${lightboxPanX}px),calc(-50% + ${lightboxPanY}px)) scale(${lightboxScale})`;
}
function resetLightbox(mode="fit"){
  const image=highResolutionImages[activeLightboxId];if(!image||!lightboxViewport)return;
  const rect=lightboxViewport.getBoundingClientRect();
  const paddedWidth=Math.max(120,rect.width-32);
  const paddedHeight=Math.max(120,rect.height-32);
  lightboxFitScale=Math.min(paddedWidth/image.width,paddedHeight/image.height,1);
  lightboxScale=mode==="actual"?1:lightboxFitScale;
  lightboxPanX=0;lightboxPanY=0;updateLightboxTransform();
}
function zoomLightbox(factor,clientX=null,clientY=null){
  const image=highResolutionImages[activeLightboxId];if(!image||!lightboxViewport)return;
  const previous=lightboxScale;
  const minimum=Math.max(lightboxFitScale*.35,.025);
  const next=Math.min(12,Math.max(minimum,previous*factor));
  if(Math.abs(next-previous)<.0001)return;
  const rect=lightboxViewport.getBoundingClientRect();
  const originX=clientX===null?rect.left+rect.width/2:clientX;
  const originY=clientY===null?rect.top+rect.height/2:clientY;
  const dx=originX-rect.left-rect.width/2-lightboxPanX;
  const dy=originY-rect.top-rect.height/2-lightboxPanY;
  const ratio=next/previous;
  lightboxPanX+=dx*(1-ratio);
  lightboxPanY+=dy*(1-ratio);
  lightboxScale=next;updateLightboxTransform();
}
function closeLightbox(){
  if(!lightbox||lightbox.hidden)return;
  lightbox.hidden=true;document.body.classList.remove("report-lightbox-open");
  activeLightboxId=null;lightboxDragging=false;lightboxBackgroundPress=false;lightboxPointerMoved=false;lightboxPointerId=null;lightboxViewport?.classList.remove("is-dragging");
}
function openLightbox(id,label){
  const image=highResolutionImages[id];if(!image||!lightbox||!lightboxImage)return;
  activeLightboxId=id;
  lightboxTitle.textContent=label||id;
  lightboxImage.alt=label||id;
  lightboxImage.onload=()=>resetLightbox("fit");
  lightboxImage.src=image.data_uri;
  lightbox.hidden=false;document.body.classList.add("report-lightbox-open");closeDrawer();
  requestAnimationFrame(()=>{resetLightbox("fit");lightboxClose?.focus({preventScroll:true})});
}
function upgradeImageFrame(frame,id){
  const image=highResolutionImages[id];if(!image||frame.dataset.reportHighResolutionSha===image.sha256)return;
  const source=frame.getAttribute("srcdoc")||"";if(!source)return;
  let upgraded=source.replace(/(<img\\b[^>]*\\bsrc=["'])data:image\\/(?:webp|jpeg);base64,[A-Za-z0-9+/=]+(["'][^>]*>)/i,`$1${image.data_uri}$2`);
  upgraded=upgraded.replace(/width=["']\\d+["']\\s+height=["']\\d+["']/i,`width="${image.width}" height="${image.height}"`);
  if(upgraded===source)return;frame.dataset.reportHighResolutionSha=image.sha256;frame.setAttribute("srcdoc",upgraded);
}
function bindImageInteractionSurface(frame,id,carouselState){
  const host=frame.parentElement;if(!host)return;
  host.classList.add("report-image-interaction-host");
  let surface=host.querySelector(`.report-image-interaction-surface[data-report-image-id="${id}"]`);
  if(surface)return;
  const slide=frame.closest(".report-carousel-slide");
  const label=slide?.getAttribute("aria-label")||`High-resolution report image ${id}`;
  surface=document.createElement("button");surface.type="button";
  surface.className="report-image-interaction-surface";
  surface.dataset.reportImageId=id;
  surface.setAttribute("role","button");surface.setAttribute("tabindex","0");
  surface.setAttribute("aria-label",`Open high-resolution image: ${label}`);
  surface.setAttribute("title","Click to inspect the embedded high-resolution image");
  if(carouselState){
    surface.addEventListener("pointerdown",event=>beginCarouselDrag(carouselState,event,surface));
    surface.addEventListener("pointermove",event=>moveCarouselDrag(carouselState,event));
    surface.addEventListener("pointerup",event=>{
      if(event.pointerId!==carouselState.pointerId)return;
      const wasDragging=endCarouselDrag(carouselState,event);
      if(!wasDragging){
        carouselState.suppressClick=true;
        openLightbox(id,label);
        setTimeout(()=>{carouselState.suppressClick=false},220);
      }
    });
    surface.addEventListener("pointercancel",event=>endCarouselDrag(carouselState,event));
  }
  surface.addEventListener("click",event=>{event.preventDefault();if(!carouselState?.suppressClick)openLightbox(id,label)});
  surface.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();openLightbox(id,label)}});
  host.appendChild(surface);
}
function bindImageFrame(frame,id){
  const image=highResolutionImages[id];if(!image)return;
  const carouselState=carouselStateForFrame(frame);
  if(frame.hasAttribute("sandbox")){bindImageInteractionSurface(frame,id,carouselState);return}
  let doc;try{doc=frame.contentDocument}catch(error){doc=null}
  if(!doc){bindImageInteractionSurface(frame,id,carouselState);return}
  const img=doc.querySelector("img");if(!img)return;
  if(carouselState&&img.dataset.reportCarouselSwipeBound!=="1"){
    img.dataset.reportCarouselSwipeBound="1";img.style.touchAction="pan-y";
    img.addEventListener("pointerdown",event=>beginCarouselDrag(carouselState,event,img));
    img.addEventListener("pointermove",event=>moveCarouselDrag(carouselState,event));
    img.addEventListener("pointerup",event=>endCarouselDrag(carouselState,event));
    img.addEventListener("pointercancel",event=>endCarouselDrag(carouselState,event));
  }
  if(img.dataset.reportLightboxBound==="1")return;
  const caption=doc.querySelector("figcaption")?.textContent?.trim()||img.getAttribute("alt")||id;
  img.dataset.reportLightboxBound="1";
  img.style.cursor="zoom-in";
  img.setAttribute("role","button");
  img.setAttribute("tabindex","0");
  img.setAttribute("title","Click to inspect the embedded high-resolution image");
  img.addEventListener("click",event=>{event.preventDefault();if(!carouselStateForFrame(frame)?.suppressClick)openLightbox(id,caption)});
  img.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();openLightbox(id,caption)}});
}
function prepareImageFrame(frame,id){
  if(!highResolutionImages[id])return;
  frame.dataset.reportImageId=id;
  frame.setAttribute("title","Click the image to inspect high-resolution details");
  if(frame.hasAttribute("sandbox"))bindImageInteractionSurface(frame,id,carouselStateForFrame(frame));
  if(frame.dataset.reportLightboxLoadBound!=="1"){
    frame.dataset.reportLightboxLoadBound="1";
    frame.addEventListener("load",()=>bindImageFrame(frame,frame.dataset.reportImageId));
  }
  requestAnimationFrame(()=>bindImageFrame(frame,id));
  setTimeout(()=>bindImageFrame(frame,id),150);
}
function configureImageFrames(){
  configureCarousels();
  for(const id of imageIds){const candidates=[];const enhanced=document.getElementById(id)?.querySelector("iframe");const fallback=document.querySelector(`.portable-block[data-artifact-block-id="${id}"] iframe`);if(enhanced)candidates.push(enhanced);if(fallback)candidates.push(fallback);for(const frame of candidates){frame.setAttribute("scrolling","no");frame.style.overflow="hidden";prepareImageFrame(frame,id);upgradeImageFrame(frame,id);prepareImageFrame(frame,id);fitStandaloneImageFrame(frame)}}
  queueUpdate();
}
mobileToggle?.addEventListener("click",()=>{const open=document.body.classList.toggle("report-toc-open");mobileToggle.setAttribute("aria-expanded",String(open))});
closeButton?.addEventListener("click",closeDrawer);backdrop?.addEventListener("click",closeDrawer);
lightboxClose?.addEventListener("click",closeLightbox);
lightboxZoomIn?.addEventListener("click",()=>zoomLightbox(1.25));
lightboxZoomOut?.addEventListener("click",()=>zoomLightbox(.8));
lightboxFit?.addEventListener("click",()=>resetLightbox("fit"));
lightboxActual?.addEventListener("click",()=>resetLightbox("actual"));
lightboxViewport?.addEventListener("wheel",event=>{if(!activeLightboxId)return;event.preventDefault();zoomLightbox(event.deltaY<0?1.18:1/1.18,event.clientX,event.clientY)},{passive:false});
lightboxViewport?.addEventListener("pointerdown",event=>{
  if(!activeLightboxId||event.button!==0)return;
  lightboxPointerId=event.pointerId;lightboxPointerMoved=false;lightboxDragStartX=event.clientX;lightboxDragStartY=event.clientY;
  if(event.target===lightboxImage){event.preventDefault();lightboxDragging=true;lightboxBackgroundPress=false;lightboxDragPanX=lightboxPanX;lightboxDragPanY=lightboxPanY;lightboxViewport.setPointerCapture(event.pointerId);lightboxViewport.classList.add("is-dragging")}
  else if(event.target===lightboxViewport){lightboxDragging=false;lightboxBackgroundPress=true;lightboxViewport.setPointerCapture(event.pointerId)}
});
lightboxViewport?.addEventListener("pointermove",event=>{
  if(event.pointerId!==lightboxPointerId)return;
  const dx=event.clientX-lightboxDragStartX;const dy=event.clientY-lightboxDragStartY;if(Math.hypot(dx,dy)>4)lightboxPointerMoved=true;
  if(!lightboxDragging)return;lightboxPanX=lightboxDragPanX+dx;lightboxPanY=lightboxDragPanY+dy;updateLightboxTransform()
});
lightboxViewport?.addEventListener("pointerup",event=>{
  if(event.pointerId!==lightboxPointerId)return;
  const closeFromBackground=lightboxBackgroundPress&&!lightboxPointerMoved;
  lightboxDragging=false;lightboxBackgroundPress=false;lightboxPointerMoved=false;lightboxPointerId=null;
  if(lightboxViewport.hasPointerCapture(event.pointerId))lightboxViewport.releasePointerCapture(event.pointerId);
  lightboxViewport.classList.remove("is-dragging");
  if(closeFromBackground)closeLightbox()
});
lightboxViewport?.addEventListener("pointercancel",event=>{
  if(event.pointerId!==lightboxPointerId)return;
  lightboxDragging=false;lightboxBackgroundPress=false;lightboxPointerMoved=false;lightboxPointerId=null;lightboxViewport.classList.remove("is-dragging")
});
document.addEventListener("keydown",event=>{if(!lightbox||lightbox.hidden)return;if(event.key==="Escape"){event.preventDefault();closeLightbox()}else if(event.key==="+"||event.key==="="){event.preventDefault();zoomLightbox(1.25)}else if(event.key==="-"||event.key==="_"){event.preventDefault();zoomLightbox(.8)}else if(event.key==="0"){event.preventDefault();resetLightbox("fit")}else if(event.key==="1"){event.preventDefault();resetLightbox("actual")}});
window.addEventListener("scroll",queueUpdate,{passive:true});window.addEventListener("resize",()=>{closeDrawer();configureImageFrames();fitAllCarousels(true)},{passive:true});window.addEventListener("hashchange",queueUpdate);
window.visualViewport?.addEventListener("resize",()=>fitAllCarousels(true),{passive:true});
configureImageFrames();setTimeout(configureImageFrames,250);setTimeout(configureImageFrames,1000);
})();
</script>
""".replace("__ENTRIES__", entries_json).replace("__IMAGE_IDS__", image_ids_json).replace("__CAROUSELS__", carousels_json)


def enhance_packaged_html(
    artifact_json: Path,
    output_html: Path,
    high_resolution_images: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = read_json(artifact_json)
    entries = report_navigation_entries(artifact)
    image_frames = report_image_frames(artifact)
    carousel_groups = report_carousel_groups(artifact)
    if not image_frames:
        raise RuntimeError("No dimensioned embedded image blocks were found for no-scroll rendering")
    high_resolution_images = validate_high_resolution_image_payload(high_resolution_images, image_frames)

    document = require_file(output_html).read_text()
    if "data-report-navigation-version=" in document:
        raise RuntimeError("The packaged HTML already contains a report-navigation enhancement")
    if "</head>" not in document or "</body>" not in document:
        raise RuntimeError("The packaged report is missing required HTML closing tags")

    navigation = (
        '<button id="report-toc-mobile-toggle" type="button" aria-controls="report-toc" aria-expanded="false">Contents</button>'
        '<div id="report-toc-backdrop" aria-hidden="true"></div>'
        '<nav id="report-toc" aria-label="Report navigation">'
        '<div id="report-toc-header"><span id="report-toc-title">On this page</span>'
        '<button id="report-toc-close" type="button" aria-label="Close report navigation">×</button></div>'
        '<div id="report-toc-list"></div></nav>'
    )
    document = document.replace("</head>", navigation_style(image_frames) + "\n</head>", 1)
    payload_script = high_resolution_payload_script(high_resolution_images)
    document = document.replace(
        "</body>",
        navigation
        + lightbox_markup()
        + payload_script
        + navigation_script(entries, image_frames, carousel_groups)
        + "\n</body>",
        1,
    )
    temporary = output_html.with_name(f".{output_html.name}.enhanced.{os.getpid()}")
    temporary.write_text(document)
    os.replace(temporary, output_html)
    high_resolution_entries = (high_resolution_images or {}).get("images", {})
    return {
        "navigation_entries": len(entries),
        "navigation_groups": sum(entry["level"] == 2 for entry in entries),
        "navigation_children": sum(entry["level"] == 3 for entry in entries),
        "carousel_groups": len(carousel_groups),
        "carousel_slides": sum(
            len(group["slides"]) for group in carousel_groups
        ),
        "carousel_group_sizes": {
            group["id"]: len(group["slides"]) for group in carousel_groups
        },
        "carousel_image_height_cap_pixels": CAROUSEL_IMAGE_HEIGHT_CAP,
        "carousel_viewport_fit": True,
        "standalone_image_viewport_fit": True,
        "no_scroll_image_frames": len(image_frames),
        "high_resolution_image_frames": len(high_resolution_entries),
        "high_resolution_image_bytes": sum(
            int(image.get("encoded_bytes", 0)) for image in high_resolution_entries.values()
        ),
        "high_resolution_render_scale": (high_resolution_images or {}).get("render_scale"),
        "high_resolution_print_ppi": (high_resolution_images or {}).get("print_ppi"),
	        "high_resolution_quality": (high_resolution_images or {}).get("quality"),
	        "high_resolution_lightbox": bool(high_resolution_entries),
	    }


def package_html(
    artifact_json: Path,
    output_html: Path,
    plugin_root: Path,
    force: bool,
    high_resolution_images: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_file(artifact_json)
    require_dir(plugin_root)
    require_file(plugin_root / "package.json")
    if output_html.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {output_html}")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npm",
            "run",
            "report:deliver",
            "--",
            "--input",
            str(artifact_json.resolve()),
            "--output",
            str(output_html.resolve()),
        ],
        cwd=plugin_root,
        check=True,
    )
    return enhance_packaged_html(artifact_json, output_html, high_resolution_images)


def main() -> int:
    args = parse_args()
    artifact_json = args.artifact_json.expanduser()

    if args.package_only:
        if args.output_html is None or args.plugin_root is None:
            raise ValueError("--package-only requires --output-html and --plugin-root")
        high_resolution_images = None
        if args.high_resolution_images_json is not None:
            high_resolution_images = read_json(args.high_resolution_images_json.expanduser())
        enhancement = package_html(
            artifact_json,
            args.output_html.expanduser(),
            args.plugin_root.expanduser(),
            args.force,
            high_resolution_images,
        )
        print(
            json.dumps(
                {
                    "artifact_json": str(artifact_json),
                    "output_html": str(args.output_html),
                    "html_enhancement": enhancement,
                },
                indent=2,
            )
        )
        return 0

    if _IMAGE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Report generation requires numpy, tifffile, and Pillow; use the configured cellpose_cpsam environment"
        ) from _IMAGE_IMPORT_ERROR
    run_root = require_dir(args.run_root.expanduser().resolve())
    if artifact_json.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {artifact_json}")

    tables = load_result_tables(run_root)
    samples = build_samples(tables, run_root, args.sample_keys)
    workflow_pdf = require_file(args.workflow_pdf.expanduser().resolve())
    methods_readme = require_file(args.methods_readme.expanduser().resolve())
    methods_markdown = prepare_methods_markdown(methods_readme.read_text())
    sheets = render_qc_sheets(samples, run_root, workflow_pdf)
    datasets = build_datasets(run_root, tables, samples)
    artifact, receipt = build_artifact(run_root, datasets, sheets, samples, methods_markdown)
    atomic_write_json(artifact_json, artifact)

    high_resolution_images: dict[str, Any] | None = None
    if args.high_resolution_images_json is not None or args.output_html is not None:
        high_resolution_images = build_high_resolution_image_payload(sheets)
    if args.high_resolution_images_json is not None:
        high_resolution_path = args.high_resolution_images_json.expanduser()
        if high_resolution_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {high_resolution_path}")
        atomic_write_json(high_resolution_path, high_resolution_images)

    if args.debug_qc_dir is not None:
        write_debug_sheets(args.debug_qc_dir.expanduser(), sheets)

    output_html: Path | None = None
    html_enhancement: dict[str, Any] | None = None
    if args.output_html is not None:
        if args.plugin_root is None:
            raise ValueError("--output-html requires --plugin-root")
        output_html = args.output_html.expanduser()
        html_enhancement = package_html(
            artifact_json,
            output_html,
            args.plugin_root.expanduser(),
            args.force,
            high_resolution_images,
        )

    receipt.update(
        {
            "run_root": str(run_root),
            "workflow_pdf": str(workflow_pdf),
            "methods_readme": str(methods_readme),
            "artifact_json": str(artifact_json),
            "high_resolution_images_json": (
                str(args.high_resolution_images_json.expanduser())
                if args.high_resolution_images_json is not None
                else None
            ),
            "output_html": str(output_html) if output_html else None,
            "html_enhancement": html_enhancement,
            "qc_samples": [
                {"label": sample.label, "key": sample.key, "density_class": sample.density_class}
                for sample in samples
            ],
            "embedded_report_images": len(sheets),
            "embedded_result_figures": len(sheets) - 1,
            "high_resolution_image_bytes": (
                sum(int(image["encoded_bytes"]) for image in high_resolution_images["images"].values())
                if high_resolution_images is not None
                else 0
            ),
            "high_resolution_render_scale": (
                high_resolution_images.get("render_scale") if high_resolution_images is not None else None
            ),
            "high_resolution_print_ppi": (
                high_resolution_images.get("print_ppi") if high_resolution_images is not None else None
            ),
            "high_resolution_quality": (
                high_resolution_images.get("quality") if high_resolution_images is not None else None
            ),
        }
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
