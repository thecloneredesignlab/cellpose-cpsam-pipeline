#!/usr/bin/env python3
"""Render a cluster-colored, image-linked audit atlas for V4 UMAP geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
RENDER_HELPER_PATH = SCRIPT_DIR / "27_build_reference_morphology_workspace.py"
SPEC = importlib.util.spec_from_file_location(
    "v4_morphology_render_helper", RENDER_HELPER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load morphology rendering helpers: {RENDER_HELPER_PATH}")
RENDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDER
SPEC.loader.exec_module(RENDER)

SCHEMA_VERSION = "multimodal_cell_state_v4_morphology_audit_v1"
PROJECT_IDS = {
    "multimodal_cell_state_v4_development",
    "multimodal_cell_state_v4_death_resolution",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-html", type=Path, required=True)
    parser.add_argument("--max-representatives", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--mask-padding", type=int, default=6)
    parser.add_argument("--overlay-width", type=int, default=2400)
    parser.add_argument("--overlay-height", type=int, default=1800)
    parser.add_argument("--overlay-tile-px", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


REPRESENTATIVE_FIELDS = [
    "selection_rank", "selection_round", "cell_id", "image_id", "mask_label",
    "well", "context_column", "context_value", "cluster", "Dim1", "Dim2",
    "crop_x0_px", "crop_y0_px", "crop_x1_px", "crop_y1_px",
    "brightfield_crop", "dead_crop", "nuclei_crop",
    "brightfield_crop_sha256", "dead_crop_sha256", "nuclei_crop_sha256",
    "brightfield_source_sha256", "dead_source_sha256", "nuclei_source_sha256",
    "combined_mask_source_sha256",
]


def load_v4_images(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    fields, rows = read_tsv(path)
    required = {
        "image_id", "channel_id", "image_path", "page_index", "channel_index",
        "mask_path", "mask_channel_index", "width", "height",
    }
    if not required.issubset(fields):
        raise ValueError(f"V4 images.tsv lacks fields: {sorted(required-set(fields))}")
    output: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        image_id, channel = row["image_id"], row["channel_id"].lower()
        if channel not in {"brightfield", "dead", "nuclei"}:
            raise ValueError(f"Unexpected V4 image channel: {channel}")
        if not image_id or channel in output[image_id]:
            raise ValueError(f"Duplicate/incomplete V4 channel: {image_id}/{channel}")
        output[image_id][channel] = row
    incomplete = [key for key, value in output.items() if set(value) != {"brightfield", "dead", "nuclei"}]
    if incomplete:
        raise ValueError(f"V4 images lack three-channel pairs: {incomplete[:10]}")
    return dict(output)


def fit_display_windows(
    image_rows: dict[str, dict[str, dict[str, str]]],
) -> dict[str, tuple[float, float]]:
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    for image_id in sorted(image_rows):
        for channel in ("brightfield", "dead", "nuclei"):
            plane, _ = RENDER.load_channel(
                image_rows[image_id][channel], f"{channel} display calibration"
            )
            finite = np.asarray(plane[np.isfinite(plane)], dtype=np.float64)
            if not finite.size:
                raise ValueError(f"V4 display source has no finite pixels: {image_id}/{channel}")
            stride = max(1, finite.size // 4096)
            samples[channel].append(finite[::stride][:4096])
    windows: dict[str, tuple[float, float]] = {}
    quantiles = {"brightfield": (0.01, 0.99), "dead": (0.01, 0.998), "nuclei": (0.01, 0.998)}
    for channel, chunks in samples.items():
        values = np.concatenate(chunks)
        low, high = np.quantile(values, quantiles[channel])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"V4 global display window is invalid: {channel}")
        windows[channel] = (float(low), float(high))
    return windows


def scale_fixed(array: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    value = np.clip((np.asarray(array, dtype=float) - low) / (high - low), 0, 1)
    value[~np.isfinite(value)] = 0
    return np.rint(value * 255).astype(np.uint8)


def fluorescence_crop(
    scaled: np.ndarray, selected_mask: np.ndarray,
    bounds: tuple[int, int, int, int], color: tuple[int, int, int],
) -> Image.Image:
    x0, y0, x1, y1 = bounds
    value = scaled[y0:y1, x0:x1]
    obj = selected_mask[y0:y1, x0:x1]
    boundary = RENDER.mask_boundary(obj)
    rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
    # Promote before applying the display color. Multiplying uint8 values by
    # components such as 255 otherwise wraps modulo 256 and collapses nearly
    # every fluorescence intensity to 0 or 1.
    normalized = value.astype(np.float32) / 255.0
    for index, component in enumerate(color):
        rgba[..., index] = np.rint(normalized * component).astype(np.uint8)
    rgba[boundary, :3] = 255
    rgba[..., 3] = np.where(obj, 232, 0).astype(np.uint8)
    rgba[boundary, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def render_three_channel_crops(
    representatives: list[dict[str, Any]],
    image_rows: dict[str, dict[str, dict[str, str]]],
    output: Path, padding: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[float, float]]]:
    windows = fit_display_windows(image_rows)
    by_image: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for rank, row in enumerate(representatives, start=1):
        by_image[str(row["image_id"])].append((rank, row))
    rendered: list[dict[str, Any]] = []
    sources: dict[tuple[str, Path], str] = {}
    for image_id in sorted(by_image):
        rows = image_rows[image_id]
        planes: dict[str, np.ndarray] = {}
        paths: dict[str, Path] = {}
        for channel in ("brightfield", "dead", "nuclei"):
            planes[channel], paths[channel] = RENDER.load_channel(rows[channel], channel)
            sources[(f"{channel}_raw", paths[channel])] = sha256_file(paths[channel])
        mask, mask_path = RENDER.load_mask(rows["brightfield"])
        sources[("combined_mask", mask_path)] = sha256_file(mask_path)
        if any(Path(rows[channel]["mask_path"]).expanduser().resolve() != mask_path for channel in ("dead", "nuclei")):
            raise ValueError(f"V4 channel mask identity differs: {image_id}")
        scaled = {channel: scale_fixed(planes[channel], windows[channel]) for channel in planes}
        for rank, row in by_image[image_id]:
            label = int(row["mask_label"])
            selected_mask = mask == label
            yy, xx = np.nonzero(selected_mask)
            if not xx.size:
                raise ValueError(f"V4 selected mask is absent: {row['cell_id']}")
            bounds = (
                max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
                min(mask.shape[1], int(xx.max()) + padding + 1),
                min(mask.shape[0], int(yy.max()) + padding + 1),
            )
            token = hashlib.sha256(row["cell_id"].encode()).hexdigest()[:12]
            base = f"{rank:03d}_{token}"
            relative = {
                channel: Path("crops") / f"{base}__{channel}.png"
                for channel in ("brightfield", "dead", "nuclei")
            }
            RENDER.save_png(
                RENDER.rgba_crop(scaled["brightfield"], selected_mask, bounds, "brightfield"),
                output / relative["brightfield"],
            )
            RENDER.save_png(fluorescence_crop(scaled["dead"], selected_mask, bounds, (255, 45, 141)), output / relative["dead"])
            RENDER.save_png(fluorescence_crop(scaled["nuclei"], selected_mask, bounds, (0, 255, 255)), output / relative["nuclei"])
            rendered.append({
                "selection_rank": rank, "selection_round": row["_selection_round"],
                "cell_id": row["cell_id"], "image_id": image_id,
                "mask_label": label, "well": row["well"],
                "context_column": row["_context_column"], "context_value": row["_context_value"],
                "cluster": row.get("cluster", ""), "Dim1": format(float(row["Dim1"]), ".17g"),
                "Dim2": format(float(row["Dim2"]), ".17g"),
                "crop_x0_px": bounds[0], "crop_y0_px": bounds[1], "crop_x1_px": bounds[2], "crop_y1_px": bounds[3],
                **{f"{channel}_crop": relative[channel].as_posix() for channel in relative},
                **{f"{channel}_crop_sha256": sha256_file(output / relative[channel]) for channel in relative},
                **{f"{channel}_source_sha256": sha256_file(paths[channel]) for channel in paths},
                "combined_mask_source_sha256": sha256_file(mask_path),
            })
    rendered.sort(key=lambda row: int(row["selection_rank"]))
    source_rows = [
        {"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
        for (role, path), digest in sorted(sources.items(), key=lambda value: (value[0][0], str(value[0][1])))
    ]
    return rendered, source_rows, windows


def resolve_asset(project_path: Path, project: dict[str, Any], field: str) -> Path:
    text = str(project.get(field, ""))
    if not text:
        raise ValueError(f"V4 project lacks {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_path.parent / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"V4 project asset is unavailable or a symlink: {path}")
    return path


def deterministic_farthest(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count >= len(rows):
        return sorted(rows, key=lambda row: row["cell_id"])
    ordered = sorted(rows, key=lambda row: row["cell_id"])
    coordinates = np.asarray(
        [[float(row["Dim1"]), float(row["Dim2"])] for row in ordered]
    )
    digest_values = [
        hashlib.sha256(f"{seed}|{row['cell_id']}".encode()).hexdigest()
        for row in ordered
    ]
    first = min(range(len(ordered)), key=lambda index: digest_values[index])
    selected = [first]
    distances = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    distances[first] = -1
    while len(selected) < count:
        maximum = float(np.max(distances))
        candidates = np.flatnonzero(np.isclose(distances, maximum, rtol=0, atol=1e-15))
        chosen = min(candidates, key=lambda index: digest_values[int(index)])
        selected.append(int(chosen))
        candidate_distance = np.sum((coordinates - coordinates[chosen]) ** 2, axis=1)
        distances = np.minimum(distances, candidate_distance)
        distances[selected] = -1
    return [ordered[index] for index in selected]


def allocate_quotas(
    groups: dict[tuple[str, str], list[dict[str, Any]]], total: int
) -> dict[tuple[str, str], int]:
    keys = sorted(groups)
    if total < len(keys):
        raise ValueError(
            "Representative budget is smaller than context-cluster coverage"
        )
    total_rows = sum(len(groups[key]) for key in keys)
    quotas = {key: 1 for key in keys}
    remaining = total - len(keys)
    ideals = {key: remaining * len(groups[key]) / total_rows for key in keys}
    for key in keys:
        add = min(len(groups[key]) - 1, int(math.floor(ideals[key])))
        quotas[key] += add
    while sum(quotas.values()) < total:
        eligible = [key for key in keys if quotas[key] < len(groups[key])]
        if not eligible:
            break
        key = max(
            eligible,
            key=lambda item: (
                ideals[item] - math.floor(ideals[item]),
                len(groups[item]) - quotas[item],
                tuple(reversed(item)),
            ),
        )
        quotas[key] += 1
        ideals[key] = math.floor(ideals[key])
    return quotas


def directory_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V4 audit contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                result[relative] = sha256_file(path)
    return result


def verify_existing(
    output: Path,
    expected_inputs: dict[str, Any],
    implementation_sha: str,
    helper_sha: str,
) -> None:
    manifest_path = output / "morphology_audit_manifest.json"
    identity_path = output / "morphology_audit_generation_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise ValueError("Existing V4 morphology audit is partial")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("inputs") != expected_inputs
    ):
        raise ValueError("Existing V4 morphology audit identity differs")
    observed = directory_hashes(
        output,
        {"morphology_audit_manifest.json", "morphology_audit_generation_identity.json"},
    )
    if manifest.get("output_file_sha256") != observed:
        raise ValueError("Existing V4 morphology audit artifact set/hash differs")
    expected = {
        "schema_version": "multimodal_cell_state_v4_morphology_audit_generation_identity_v1",
        "status": "COMPLETE",
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": implementation_sha,
        "render_helper_sha256": helper_sha,
        "output_file_sha256": observed,
    }
    if load_json(identity_path) != expected:
        raise ValueError("Existing V4 morphology audit generation identity differs")


def render_channel_overlay(
    all_rows: list[dict[str, Any]], representatives: list[dict[str, Any]],
    output: Path, channel: str, width: int, height: int, tile_px: int,
) -> dict[str, float]:
    margin = 70
    locations, bounds = RENDER.coordinate_transform(all_rows, width, height, margin)
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((margin, margin, width - margin, height - margin), fill="white", outline=(85, 94, 104), width=2)
    for row in all_rows:
        draw.point(locations[row["cell_id"]], fill=RENDER.cluster_color(row.get("cluster", "0")))
    dimensions = [max(int(row["crop_x1_px"]) - int(row["crop_x0_px"]), int(row["crop_y1_px"]) - int(row["crop_y0_px"])) for row in representatives]
    reference = float(np.median(dimensions))
    scale = tile_px / reference
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    field = f"{channel}_crop"
    for row in representatives:
        with Image.open(output / row[field]) as source:
            crop = source.convert("RGBA")
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), resampling)
        px, py = locations[row["cell_id"]]
        canvas.paste(crop, (px - crop.width // 2, py - crop.height // 2), crop)
    title = {"brightfield": "Brightfield morphology", "dead": "Dead fluorescence", "nuclei": "Nuclei fluorescence"}[channel]
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), f"V4 UMAP: {title} reference", fill=(24, 32, 42))
    draw.text((margin, 39), "Same representative cells, coordinates, crop bounds and fixed development-level display window", fill=(83, 92, 103))
    RENDER.save_png(canvas, output / f"{channel}_morphology_overlay.png")
    return {**bounds, "reference_source_crop_max_dimension_px": reference, "source_pixel_to_overlay_pixel_scale": scale}


def build_atlas(rendered: list[dict[str, Any]], identity: dict[str, Any]) -> str:
    cards = []
    for row in rendered:
        panels = "".join(
            f'<figure><img loading="lazy" src="{quote(row[f"{channel}_crop"], safe="/")}"><figcaption>{title}</figcaption></figure>'
            for channel, title in (("brightfield", "Brightfield"), ("dead", "Dead fluorescence"), ("nuclei", "Nuclei fluorescence"))
        )
        cards.append(
            f'<article><header><b>#{row["selection_rank"]}</b> {html.escape(str(row["well"]))}</header><div class="channels">{panels}</div><code>{html.escape(str(row["cell_id"]))}</code><small>cluster={html.escape(str(row["cluster"]))}; UMAP=({row["Dim1"]},{row["Dim2"]})</small></article>'
        )
    identity_text = html.escape(json.dumps(identity, sort_keys=True))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>V4 three-channel morphology atlas</title><style>body{{font:14px system-ui;background:#eef1f4;margin:0}}header.top{{position:sticky;top:0;background:white;padding:12px;z-index:2}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:10px;padding:10px}}article{{background:white;padding:8px;border:1px solid #ccd3da}}.channels{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}}figure{{margin:0;background:#111;text-align:center}}img{{width:100%;height:160px;object-fit:contain}}figcaption{{background:#f4f6f8;padding:4px;font-size:11px}}code,small{{display:block;overflow-wrap:anywhere;font-size:10px;margin-top:5px}}</style></head><body><header class="top"><b>V4 three-channel morphology atlas</b><div>{identity_text}</div></header><main>{''.join(cards)}</main></body></html>'''


def build_workspace(annotation_html: Path, output: Path, identity: dict[str, Any]) -> str:
    annotation = quote(os.path.relpath(annotation_html, output).replace(os.sep, "/"), safe="/.")
    identity_json = json.dumps(identity, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>V4 death-region annotation workspace</title><style>*{{box-sizing:border-box}}html,body{{height:100%;margin:0;font:13px system-ui;background:#e9edf1}}header{{height:52px;background:white;padding:9px 14px;border-bottom:1px solid #ccd3da}}main{{height:calc(100% - 52px);display:grid;grid-template-columns:minmax(620px,1.4fr) minmax(420px,1fr);gap:8px;padding:8px}}section{{background:white;border:1px solid #ccd3da;overflow:hidden}}iframe{{width:100%;height:100%;border:0}}.refs{{overflow:auto;padding:6px}}.refs img{{width:100%;display:block;margin-bottom:8px}}nav a{{margin-right:10px}}</style></head><body><header><b>V4 broad death-region annotation</b> — polygons are provisional; reviewed three-channel labels are authoritative.<nav><a href="morphology_atlas.html">Three-channel atlas</a><a href="brightfield_morphology_overlay.png">Brightfield</a><a href="dead_morphology_overlay.png">Dead</a><a href="nuclei_morphology_overlay.png">Nuclei</a></nav></header><main><section><iframe src="{annotation}"></iframe></section><section class="refs"><img src="brightfield_morphology_overlay.png"><img src="dead_morphology_overlay.png"><img src="nuclei_morphology_overlay.png"></section></main><script type="application/json" id="identity">{identity_json}</script></body></html>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_representatives < 1 or args.mask_padding < 0:
        raise ValueError("Invalid V4 morphology audit representative/padding option")
    if min(args.overlay_width, args.overlay_height) < 400 or args.overlay_tile_px < 4:
        raise ValueError("Invalid V4 morphology audit canvas/tile size")
    implementation = Path(__file__).resolve()
    implementation_sha = sha256_file(implementation)
    helper_sha = sha256_file(RENDER_HELPER_PATH.resolve())
    shadow = args.shadow_root.expanduser().resolve()
    if not shadow.is_dir() or shadow.is_symlink():
        raise ValueError(f"V4 shadow root is unavailable: {shadow}")
    project_path = args.project.expanduser().resolve()
    if (
        not project_path.is_file()
        or project_path.is_symlink()
        or shadow not in project_path.parents
    ):
        raise ValueError("V4 project must be a plain file inside its shadow root")
    output = args.output_dir.expanduser().resolve()
    annotation_html = args.annotation_html.expanduser().resolve()
    if not annotation_html.is_file() or annotation_html.is_symlink() or shadow not in annotation_html.parents:
        raise ValueError("V4 annotation HTML must be an immutable file inside the shadow root")
    if (
        output == shadow
        or shadow not in output.parents
        or output in project_path.parents
    ):
        raise ValueError(
            "V4 morphology audit output must be separate inside its shadow root"
        )
    project = load_json(project_path)
    if (
        project.get("schema_version") != "cell_phenotype_annotator_project_v1"
        or project.get("project_id") not in PROJECT_IDS
    ):
        raise ValueError("Unsupported V4 project identity")
    cells_path = resolve_asset(project_path, project, "cells_file")
    images_path = resolve_asset(project_path, project, "images_file")
    projection = project.get("projection")
    if not isinstance(projection, dict) or projection.get("mode") != "existing_umap":
        raise ValueError("V4 project does not declare an existing UMAP")
    coordinate_path = project_path.parent / str(projection.get("coordinate_file", ""))
    coordinate_path = coordinate_path.resolve()
    if not coordinate_path.is_file() or coordinate_path.is_symlink():
        raise ValueError("V4 coordinate file is unavailable")
    clusters_path = project_path.parent / "projection" / "diagnostic_clusters.tsv"
    manifest_path = project_path.parent / "projection" / "projection_manifest.json"
    for path in (clusters_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"V4 projection audit input is unavailable: {path}")
    projection_manifest = load_json(manifest_path)
    if (
        projection_manifest.get("schema_version")
        != "multimodal_cell_state_v4_projection_v1"
        or projection_manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("V4 projection manifest is incomplete")
    declared = projection_manifest.get("output_file_sha256", {})
    if declared.get("umap.tsv") != sha256_file(coordinate_path) or declared.get(
        "diagnostic_clusters.tsv"
    ) != sha256_file(clusters_path):
        raise ValueError("V4 project projection hashes differ from authority")
    cell_fields, cells = read_tsv(cells_path)
    _, coordinates = read_tsv(coordinate_path)
    _, clusters = read_tsv(clusters_path)
    required_cells = {
        "cell_id",
        "image_id",
        "mask_label",
        "well",
        "context_key",
        "split",
    }
    if not required_cells.issubset(cell_fields):
        raise ValueError("V4 cells lack morphology audit identity")
    cell_ids = [row["cell_id"] for row in cells]
    if [row["cell_id"] for row in coordinates] != cell_ids or [
        row["cell_id"] for row in clusters
    ] != cell_ids:
        raise ValueError("V4 morphology audit row universe differs")
    if any(row["split"] != "development" for row in cells):
        raise ValueError("Heldout cells entered V4 morphology audit")
    joined: list[dict[str, Any]] = []
    for cell, coordinate, cluster in zip(cells, coordinates, clusters, strict=True):
        joined.append(
            {
                **cell,
                "Dim1": float(coordinate["Dim1"]),
                "Dim2": float(coordinate["Dim2"]),
                "cluster": cluster["diagnostic_cluster"],
            }
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(row["context_key"], row["cluster"])].append(row)
    quotas = allocate_quotas(groups, min(args.max_representatives, len(joined)))
    selected: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(groups)):
        for row in deterministic_farthest(
            groups[key], quotas[key], args.seed + group_index
        ):
            selected.append(
                {
                    **row,
                    "_selection_round": "context_cluster_farthest_point",
                    "_context_column": "context_key",
                    "_context_value": row["context_key"],
                }
            )
    selected.sort(
        key=lambda row: (row["context_key"], int(row["cluster"]), row["cell_id"])
    )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    image_rows = load_v4_images(images_path)
    inputs = {
        "project": {"path": str(project_path), "sha256": sha256_file(project_path)},
        "cells": {"path": str(cells_path), "sha256": sha256_file(cells_path)},
        "images": {"path": str(images_path), "sha256": sha256_file(images_path)},
        "annotation_html": {"path": str(annotation_html), "sha256": sha256_file(annotation_html)},
        "coordinates": {
            "path": str(coordinate_path),
            "sha256": sha256_file(coordinate_path),
        },
        "diagnostic_clusters": {
            "path": str(clusters_path),
            "sha256": sha256_file(clusters_path),
        },
        "projection_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "implementation": {"path": str(implementation), "sha256": implementation_sha},
        "render_helper": {
            "path": str(RENDER_HELPER_PATH.resolve()),
            "sha256": helper_sha,
        },
    }
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError("Existing V4 morphology audit must be a real directory")
        if not args.overwrite:
            raise FileExistsError(output)
        verify_existing(output, inputs, implementation_sha, helper_sha)
        print(f"multimodal_cell_state_v4_morphology_audit={output}")
        print("generation_status=verified_reuse")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        rendered, source_assets, windows = render_three_channel_crops(
            selected, image_rows, staging, args.mask_padding
        )
        write_tsv(staging / "representative_cells.tsv", REPRESENTATIVE_FIELDS, rendered)
        bounds = {
            channel: render_channel_overlay(joined, rendered, staging, channel, args.overlay_width, args.overlay_height, args.overlay_tile_px)
            for channel in ("brightfield", "dead", "nuclei")
        }
        identity = {
            "project_id": project["project_id"],
            "selected_run_id": projection_manifest["selected_run_id"],
            "row_count": len(joined),
            "representative_count": len(rendered),
            "diagnostic_cluster_role": "navigation_only_not_cell_state_label",
        }
        (staging / "morphology_atlas.html").write_text(build_atlas(rendered, identity), encoding="utf-8")
        (staging / "annotation_workspace.html").write_text(build_workspace(annotation_html, staging, identity), encoding="utf-8")
        outputs = directory_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "method_version": "sum159_multimodal_cell_state_v4",
            "inputs": inputs,
            "selection": {
                "seed": args.seed,
                "maximum_representatives": args.max_representatives,
                "selected_count": len(rendered),
                "strata": "context_key_by_diagnostic_cluster",
                "within_stratum": "deterministic_umap_farthest_point_real_cells",
                "diagnostic_cluster_role": "navigation_only_not_cell_state_label",
            },
            "render": {
                "channels": ["brightfield", "dead", "nuclei"],
                "shared_representative_identity": True,
                "display_window_policy": "development_locked_channel_specific_global_quantiles_no_per_cell_autocontrast",
                "display_windows_raw_units": {channel: {"low": value[0], "high": value[1]} for channel, value in windows.items()},
                "combined_mask_role": "localization_and_outline_only",
                "overlay_bounds": bounds,
                "source_assets": source_assets,
            },
            "heldout_read": False,
            "forbidden_inputs_read": [],
            "training_label_authority": "none_visual_audit_only",
            "output_file_sha256": outputs,
        }
        manifest_file = staging / "morphology_audit_manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation = {
            "schema_version": "multimodal_cell_state_v4_morphology_audit_generation_identity_v1",
            "status": "COMPLETE",
            "manifest_sha256": sha256_file(manifest_file),
            "implementation_sha256": implementation_sha,
            "render_helper_sha256": helper_sha,
            "output_file_sha256": outputs,
        }
        (staging / "morphology_audit_generation_identity.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(
                f"V4 morphology audit output appeared during staging: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"multimodal_cell_state_v4_morphology_audit={output}")
    print(f"representatives={len(selected)}")
    print("training_label_authority=none")
    print("generation_status=created")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
