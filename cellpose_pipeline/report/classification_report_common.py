#!/usr/bin/env python3
"""Shared helpers for SUM159 classification calibration and production reports."""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _load_main_report_module() -> Any:
    path = Path(__file__).with_name("generate_results_analysis_report.py")
    spec = importlib.util.spec_from_file_location("_classification_main_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the main report generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAIN_REPORT = _load_main_report_module()
if MAIN_REPORT._IMAGE_IMPORT_ERROR is not None:
    raise RuntimeError(
        "Report generation requires numpy, tifffile, and Pillow; "
        "use the configured CellPose environment"
    ) from MAIN_REPORT._IMAGE_IMPORT_ERROR

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


REPORT_IMAGE_RENDER_SCALE = 6


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(require_file(path).read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with require_file(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in require_file(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    text = str(value).strip()
    if text == "":
        return default
    number = float(text)
    if not math.isfinite(number):
        return default
    return number


def as_int(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    if text == "":
        return default
    return int(round(float(text)))


def display_branch(value: str) -> str:
    return {
        "consensus": "Consensus (authoritative)",
        "fusion-consensus": "Consensus (authoritative)",
        "original": "Original",
        "fusion": "Original",
        "nucleated_only": "Nucleated-only",
        "fusion-nucleated-only": "Nucleated-only",
    }.get(value, value)


def generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def discover_plugin_root() -> Path:
    bases = (
        Path.home()
        / ".codex"
        / "plugins"
        / "cache"
        / "openai-curated-remote"
        / "data-analytics",
        Path.home() / ".local" / "share" / "data-analytics",
    )
    candidates: list[Path] = []
    for base in bases:
        if not base.is_dir():
            continue
        candidates.extend(path for path in base.glob("*") if (path / "package.json").is_file())
    if not candidates:
        raise FileNotFoundError(
            "No installed Data Analytics report environment was found; "
            "pass --plugin-root explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
        Path(
            "/usr/share/fonts/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
        Path(
            "/System/Library/Fonts/Supplemental/"
            + ("Arial Bold.ttf" if bold else "Arial.ttf")
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_image(path: Path) -> Image.Image:
    with Image.open(require_file(path)) as source:
        return source.convert("RGB")


def _wrap_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    label_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=label_font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_fitted_line(
    canvas: Image.Image,
    text: str,
    text_font: ImageFont.ImageFont,
    *,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, int, int],
) -> None:
    measurement = ImageDraw.Draw(Image.new("L", (1, 1)))
    left, top, right, bottom = measurement.textbbox((0, 0), text, font=text_font)
    width = max(1, right - left)
    height = max(1, bottom - top)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text(
        (-left, -top),
        text,
        fill=255,
        font=text_font,
    )
    target_width = min(width, max_width)
    if target_width != width:
        mask = mask.resize((target_width, height), Image.Resampling.LANCZOS)
    canvas.paste(
        fill,
        (x, y, x + target_width, y + height),
        mask,
    )


def labeled_grid(
    panels: Iterable[tuple[Image.Image, str]],
    title: str,
    columns: int = 2,
    panel_width: int = 1500,
    title_font_size: int = 52,
    label_font_size: int = 44,
) -> Image.Image:
    prepared: list[tuple[Image.Image, str]] = []
    for image, label in panels:
        height = max(1, round(image.height * panel_width / image.width))
        prepared.append(
            (image.resize((panel_width, height), Image.Resampling.LANCZOS), label)
        )
    if not prepared:
        raise ValueError("At least one panel is required")
    columns = max(1, min(columns, len(prepared)))
    rows = math.ceil(len(prepared) / columns)
    gap = 22
    label_font = font(label_font_size, True)
    measurement = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped_labels = [
        _wrap_label(measurement, label, label_font, panel_width - 40)
        for _image, label in prepared
    ]
    line_spacing = max(6, label_font_size // 8)
    line_height = label_font_size + line_spacing
    title_height = max(96, title_font_size + 48)
    label_height = max(
        78,
        max(len(lines) for lines in wrapped_labels) * line_height + 34,
    )
    panel_body_height = max(image.height for image, _ in prepared)
    cell_height = label_height + panel_body_height
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns - 1) * gap,
            title_height + rows * cell_height + (rows - 1) * gap,
        ),
        (18, 20, 24),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(title_font_size, True)
    _draw_fitted_line(
        canvas,
        title,
        title_font,
        x=28,
        y=max(16, (title_height - title_font_size) // 2 - 2),
        max_width=canvas.width - 56,
        fill=(245, 247, 250),
    )
    for index, ((image, _label), label_lines) in enumerate(
        zip(prepared, wrapped_labels, strict=True)
    ):
        column = index % columns
        row = index // columns
        x = column * (panel_width + gap)
        y = title_height + row * (cell_height + gap)
        draw.rectangle(
            (x, y, x + panel_width, y + label_height),
            fill=(35, 39, 47),
        )
        label_block_height = len(label_lines) * line_height - line_spacing
        label_y = y + max(10, (label_height - label_block_height) // 2 - 2)
        for line_index, line in enumerate(label_lines):
            _draw_fitted_line(
                canvas,
                line,
                label_font,
                x=x + 20,
                y=label_y + line_index * line_height,
                max_width=panel_width - 40,
                fill=(245, 247, 250),
            )
        canvas.paste(image, (x, y + label_height))
    return canvas


def build_scaled_high_resolution_payload(
    figures: dict[str, Image.Image],
    image_data: dict[str, tuple[str, int, int]],
    render_scale: int = REPORT_IMAGE_RENDER_SCALE,
) -> dict[str, Any]:
    if render_scale <= 1:
        raise ValueError("render_scale must be greater than one")
    images: dict[str, dict[str, Any]] = {}
    quality = (
        MAIN_REPORT.HIGH_RES_WEBP_QUALITY
        if MAIN_REPORT.pil_features.check("webp")
        else MAIN_REPORT.HIGH_RES_JPEG_QUALITY
    )
    for key, figure in figures.items():
        _uri, canonical_width, canonical_height = image_data[key]
        target_size = (
            int(canonical_width) * render_scale,
            int(canonical_height) * render_scale,
        )
        high_resolution = figure.resize(target_size, Image.Resampling.LANCZOS)
        data_uri, encoded_bytes, digest = MAIN_REPORT.encode_sheet(
            high_resolution,
            quality,
            high_resolution=True,
        )
        images[f"{key}_image"] = {
            "data_uri": data_uri,
            "width": high_resolution.width,
            "height": high_resolution.height,
            "encoded_bytes": encoded_bytes,
            "sha256": digest,
        }
    return {
        "version": 1,
        "render_scale": render_scale,
        "print_ppi": MAIN_REPORT.HIGH_RES_PRINT_PPI,
        "quality": quality,
        "encoding": (
            "webp"
            if MAIN_REPORT.pil_features.check("webp")
            else "jpeg-4:4:4"
        ),
        "images": images,
    }


def encode_figures(
    figures: dict[str, Image.Image],
    canonical_max_width: int = 700,
    canonical_quality: int = 78,
) -> tuple[dict[str, tuple[str, int, int]], dict[str, Any]]:
    attempts = (
        (canonical_max_width, canonical_quality),
        (650, 74),
        (600, 70),
        (550, 66),
        (500, 62),
        (450, 56),
        (400, 50),
    )
    image_data: dict[str, tuple[str, int, int]] = {}
    for max_width, quality in attempts:
        candidate: dict[str, tuple[str, int, int]] = {}
        encoded_total = 0
        for key, figure in figures.items():
            target_width = min(max_width, max(1, figure.width))
            scale = min(1.0, target_width / max(1, figure.width))
            canonical = figure.resize(
                (
                    max(1, round(figure.width * scale)),
                    max(1, round(figure.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            uri, encoded_bytes, _digest = MAIN_REPORT.encode_sheet(
                canonical,
                quality,
            )
            candidate[key] = (uri, canonical.width, canonical.height)
            encoded_total += encoded_bytes
        # Base64 expands the bytes by about one third. This leaves more than
        # 500 KB for the manifest and bounded datasets below the 2.85 MB
        # portable-artifact limit used by the established report generator.
        if math.ceil(encoded_total * 4 / 3) <= 2_300_000:
            image_data = candidate
            break
    if not image_data:
        raise RuntimeError("Unable to bound canonical report images below 2.3 MB")
    high_resolution = build_scaled_high_resolution_payload(figures, image_data)
    return image_data, high_resolution


def image_body(
    image_data: tuple[str, int, int],
    alt: str,
    caption: str,
) -> str:
    uri, width, height = image_data
    return MAIN_REPORT.image_block_body(uri, alt, caption, width, height)


def logical_source(
    source_id: str,
    label: str,
    root_label: str,
    relative_path: str,
    description: str,
) -> dict[str, Any]:
    logical_path = f"{root_label}/{relative_path}"
    is_csv = relative_path.endswith((".csv", ".csv.gz", ".tsv"))
    return {
        "id": source_id,
        "label": label,
        "path": logical_path,
        "query": {
            "engine": "duckdb",
            "description": description,
            "sql": (
                f"SELECT * FROM read_csv_auto('{logical_path}', header=true)"
                if is_csv
                else "SELECT 'Source is a saved report input or image inventory' AS source_contract"
            ),
            "tables_used": [relative_path],
        },
    }


def package_report(
    *,
    artifact: dict[str, Any],
    artifact_path: Path,
    output_html: Path,
    build_receipt: Path,
    plugin_root: Path,
    force: bool,
    high_resolution_images: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    artifact_path = artifact_path.expanduser().resolve()
    output_html = output_html.expanduser().resolve()
    build_receipt = build_receipt.expanduser().resolve()
    for path in (artifact_path, output_html, build_receipt):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    artifact_bytes = len(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if artifact_bytes > MAIN_REPORT.MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            "Portable report artifact exceeds the bounded payload limit: "
            f"{artifact_bytes:,} > {MAIN_REPORT.MAX_ARTIFACT_BYTES:,} bytes"
        )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    MAIN_REPORT.atomic_write_json(artifact_path, artifact)
    enhancement = MAIN_REPORT.package_html(
        artifact_path,
        output_html,
        require_dir(plugin_root),
        force,
        high_resolution_images,
    )
    receipt.update(
        {
            "artifact_json": str(artifact_path),
            "report": str(output_html),
            "plugin_root": str(plugin_root),
            "html_enhancement": enhancement,
            "artifact_bytes": artifact_path.stat().st_size,
            "report_bytes": output_html.stat().st_size,
        }
    )
    MAIN_REPORT.atomic_write_json(build_receipt, receipt)
    return receipt


def artifact_payload(
    *,
    title: str,
    description: str,
    manifest: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    sources: list[dict[str, Any]],
    origin: str,
    timestamp: str,
) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest.update(
        {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": description,
            "generatedAt": timestamp,
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in sources
            ],
        }
    )
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {
            "originUrl": origin,
            "controls": {"edit": False, "refresh": False, "share": False},
        },
    }


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def atomic_write_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)
