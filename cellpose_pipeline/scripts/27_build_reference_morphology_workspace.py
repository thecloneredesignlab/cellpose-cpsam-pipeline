#!/usr/bin/env python3
"""Build a reference-style morphology overlay beside an immutable CPA editor.

The historical LTEE workflow placed real cell-shape cutouts back at their UMAP
coordinates and displayed that image next to the polygon editor.  This sidecar
recreates that behaviour without changing the Cell Phenotype Annotator (CPA)
generation.  Selection and the overlay use only UMAP coordinates plus
cell/well identity.  Brightfield is the displayed morphology channel, the
Combined object mask is only the localization anchor, and Nuclei is emitted as
separate human-review support.  Dead, Combined RGB, and current classifier
outputs are forbidden inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import numpy as np
import tifffile
from PIL import Image, ImageDraw


SCHEMA_VERSION = "reference_morphology_workspace_v1"
SCHEMA_VERSION_V2 = "reference_morphology_workspace_v2"
PROJECT_SCHEMA_VERSION = "cell_phenotype_annotator_project_v1"
ALLOWED_CHANNELS = {"brightfield", "nuclei"}
FORBIDDEN_COLUMN_RE = re.compile(
    r"(^|_)(?:class|label|dead|rgb|state|final_state|classification|confidence|trajectory|"
    r"prediction|probability|review|annotation|outcome|response)(?:_|$)",
    re.IGNORECASE,
)
REPRESENTATIVE_FIELDS = [
    "selection_rank",
    "selection_round",
    "cell_id",
    "image_id",
    "mask_label",
    "well",
    "context_column",
    "context_value",
    "Dim1",
    "Dim2",
    "crop_x0_px",
    "crop_y0_px",
    "crop_x1_px",
    "crop_y1_px",
    "brightfield_crop",
    "nuclei_crop",
    "brightfield_crop_sha256",
    "nuclei_crop_sha256",
    "brightfield_source_sha256",
    "nuclei_source_sha256",
    "combined_mask_source_sha256",
]
REPRESENTATIVE_FIELDS_V2 = [
    *REPRESENTATIVE_FIELDS[:8],
    "cluster",
    *REPRESENTATIVE_FIELDS[8:],
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Reference CPA project.yml, or its containing directory.",
    )
    annotation = parser.add_mutually_exclusive_group(required=True)
    annotation.add_argument(
        "--annotation-dir",
        type=Path,
        help="Completed immutable CPA annotation generation directory.",
    )
    annotation.add_argument(
        "--annotation-payload",
        type=Path,
        help="annotation_payload.json; its manifest and HTML must be siblings.",
    )
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-representatives", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--context-column",
        help="Optional metadata column balanced before well; defaults to well.",
    )
    parser.add_argument("--mask-padding", type=int, default=6)
    parser.add_argument("--overlay-width", type=int, default=2400)
    parser.add_argument("--overlay-height", type=int, default=1800)
    parser.add_argument("--cluster-balance", type=float, default=0.2)
    parser.add_argument("--minimum-cluster-representatives", type=int, default=2)
    parser.add_argument(
        "--overlay-tile-px",
        type=int,
        default=40,
        help=(
            "Displayed size of the median source crop. One global scale is then "
            "used for every cutout so relative cell size is preserved."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Verify and reuse a byte-identical generation; never replace conflicts.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be inside shadow root {root}: {resolved}") from error
    return resolved


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise RuntimeError("Non-JSON YAML projects require PyYAML") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one mapping: {path}")
    return value


def resolve_project_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project.parent / path
    return path.resolve()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        fields = [str(field) for field in reader.fieldnames]
        if any(not field for field in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"TSV has blank or duplicate fields: {path}")
        return fields, [
            {field: str(row.get(field, "") or "") for field in fields}
            for row in reader
        ]


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def json_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def strict_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer: {value!r}") from error
    if str(parsed) != value.strip() and value.strip() not in {f"+{parsed}"}:
        raise ValueError(f"{field} must use canonical integer text: {value!r}")
    return parsed


def finite_float(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite: {value!r}")
    return parsed


def forbidden_metadata_column(column: str) -> bool:
    # mask_label is a segmentation identity/localization key, not a response.
    return column != "mask_label" and bool(FORBIDDEN_COLUMN_RE.search(column))


def verify_annotation_generation(
    annotation_dir: Path,
    payload_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    html_path = annotation_dir / "annotation.html"
    manifest_path = annotation_dir / "annotation_manifest.json"
    for path in (payload_path, html_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = load_mapping(payload_path)
    manifest = load_mapping(manifest_path)
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict):
        raise ValueError("CPA annotation manifest lacks artifact_file_sha256")
    expected = {
        "annotation_html": html_path,
        "annotation_payload": payload_path,
    }
    if set(declared) != set(expected):
        raise ValueError(
            "CPA annotation manifest artifact set changed: "
            f"expected={sorted(expected)} observed={sorted(declared)}"
        )
    for role, path in expected.items():
        if declared[role] != sha256_file(path):
            raise ValueError(f"CPA annotation artifact hash mismatch: {role}")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("CPA annotation payload lacks identity")
    for field in (
        "project_id",
        "run_id",
        "projection_id",
        "annotation_id",
        "class_config_sha256",
        "row_universe_sha256",
        "coordinate_sha256",
    ):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"CPA annotation identity lacks {field}")
        if field in manifest and str(manifest[field]) != identity[field]:
            raise ValueError(f"CPA annotation payload/manifest identity mismatch: {field}")
    return payload, manifest, html_path


def annotation_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points = payload.get("points")
    if not isinstance(points, dict):
        raise ValueError("CPA annotation payload lacks points")
    metadata = points.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("CPA annotation payload lacks point metadata")
    forbidden = sorted(column for column in metadata if forbidden_metadata_column(column))
    if forbidden:
        raise ValueError(f"Forbidden current-classification metadata entered annotation: {forbidden}")
    required = ("cell_id", "Dim1", "Dim2")
    arrays: dict[str, list[Any]] = {}
    for field in required:
        value = points.get(field)
        if not isinstance(value, list):
            raise ValueError(f"CPA annotation points.{field} must be an array")
        arrays[field] = value
    length = len(arrays["cell_id"])
    if length < 1 or any(len(value) != length for value in arrays.values()):
        raise ValueError("CPA annotation point arrays have inconsistent lengths")
    for column, value in metadata.items():
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"CPA annotation metadata.{column} has wrong length")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(length):
        cell_id = str(arrays["cell_id"][index])
        if not cell_id or cell_id in seen:
            raise ValueError(f"CPA annotation has blank or duplicate cell_id: {cell_id!r}")
        seen.add(cell_id)
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "Dim1": finite_float(arrays["Dim1"][index], f"Dim1[{index}]"),
            "Dim2": finite_float(arrays["Dim2"][index], f"Dim2[{index}]"),
        }
        row.update({column: values[index] for column, values in metadata.items()})
        rows.append(row)
    return rows


def load_cells(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    fields, rows = read_tsv(path)
    required = {"cell_id", "image_id", "mask_label", "well"}
    if not required.issubset(fields):
        raise ValueError(f"cells.tsv lacks fields: {sorted(required-set(fields))}")
    forbidden = sorted(field for field in fields if forbidden_metadata_column(field))
    if forbidden:
        raise ValueError(f"Forbidden current-classification columns entered cells.tsv: {forbidden}")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        cell_id = row["cell_id"]
        if not cell_id or cell_id in output:
            raise ValueError(f"cells.tsv has blank or duplicate cell_id: {cell_id!r}")
        if not row["image_id"] or not row["well"]:
            raise ValueError(f"cells.tsv has incomplete image/well identity: {cell_id}")
        label = strict_int(row["mask_label"], f"mask_label for {cell_id}")
        if label <= 0:
            raise ValueError(f"mask_label must be positive: {cell_id}")
        output[cell_id] = row
    return fields, output


def join_points_to_cells(
    points: list[dict[str, Any]],
    cells: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    point_ids = {str(row["cell_id"]) for row in points}
    if point_ids != set(cells):
        raise ValueError(
            "CPA annotation/cells.tsv universe mismatch: "
            f"annotation_only={len(point_ids-set(cells))} cells_only={len(set(cells)-point_ids)}"
        )
    joined: list[dict[str, Any]] = []
    for point in points:
        cell = cells[str(point["cell_id"])]
        for field in ("image_id", "mask_label", "well"):
            if field in point and str(point[field]) != cell[field]:
                raise ValueError(f"CPA annotation/cells.tsv {field} mismatch: {point['cell_id']}")
        joined.append({**cell, **point})
    return joined


def balanced_quotas(
    capacities: dict[str, int],
    total: int,
    seed: int,
    scope: str,
) -> dict[str, int]:
    quotas = {key: 0 for key in capacities}
    remaining = min(total, sum(capacities.values()))
    order = sorted(capacities, key=lambda key: (stable_hash(seed, scope, key), key))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise AssertionError("Balanced quota allocation stalled")
    return quotas


def spatial_real_cells(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    scope: str,
) -> list[dict[str, Any]]:
    """Deterministic farthest-point design using only real UMAP observations."""

    if count <= 0:
        return []
    if count >= len(rows):
        return sorted(rows, key=lambda row: stable_hash(seed, scope, row["cell_id"]))
    coords = np.asarray([[row["Dim1"], row["Dim2"]] for row in rows], dtype=np.float64)
    tie_rank = np.asarray(
        [stable_hash(seed, scope, row["cell_id"]) for row in rows], dtype=object
    )
    center = np.mean(coords, axis=0)
    center_distance = np.sum((coords - center) ** 2, axis=1)
    minimum = float(np.min(center_distance))
    first_candidates = np.flatnonzero(center_distance == minimum)
    first = min(first_candidates.tolist(), key=lambda index: str(tie_rank[index]))
    selected = [first]
    minimum_distance = np.sum((coords - coords[first]) ** 2, axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < count:
        maximum = float(np.max(minimum_distance))
        candidates = np.flatnonzero(minimum_distance == maximum)
        chosen = min(candidates.tolist(), key=lambda index: str(tie_rank[index]))
        selected.append(chosen)
        distance = np.sum((coords - coords[chosen]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return [rows[index] for index in selected]


def load_reference_v2_representatives(
    project_path: Path,
    shadow_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    projection_root = project_path.parent / "historical_projection"
    representatives_path = require_inside(
        projection_root / "historical_representatives.tsv",
        shadow_root,
        "Historical representative list",
    )
    manifest_path = require_inside(
        projection_root / "historical_projection_manifest.json",
        shadow_root,
        "Historical projection manifest",
    )
    if not representatives_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "V2 rendering requires the frozen historical representative generation"
        )
    manifest = load_mapping(manifest_path)
    if (
        manifest.get("schema_version")
        != "reference_cell_state_historical_projection_v2"
        or manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("Historical projection manifest is not complete V2")
    selection = manifest.get("representative_selection")
    expected_selection = {
        "role": "authoritative_rendering_cell_list",
        "outer_function_name": "get_all_cell_lines_overlay_representatives",
        "inner_function_name": "get_spatially_uniform_representatives",
        "seed": 1,
        "total_n": 300,
        "balance": 0.2,
        "minimum_cluster_representatives": 2,
        "output_file": "historical_representatives.tsv",
    }
    if not isinstance(selection, dict) or any(
        selection.get(key) != value for key, value in expected_selection.items()
    ):
        raise ValueError("Historical representative selection contract drifted")
    representatives_sha256 = sha256_file(representatives_path)
    declared_outputs = manifest.get("output_file_sha256")
    if (
        not isinstance(declared_outputs, dict)
        or declared_outputs.get("historical_representatives.tsv")
        != representatives_sha256
        or selection.get("output_sha256") != representatives_sha256
    ):
        raise ValueError("Historical representative list hash disagrees with manifest")
    fields, frozen = read_tsv(representatives_path)
    expected_fields = [
        "selection_rank",
        "cell_id",
        "context_key",
        "source_id",
        "cluster",
        "Dim1",
        "Dim2",
    ]
    if fields != expected_fields:
        raise ValueError("Historical representative list schema drifted")
    by_id = {str(row["cell_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise AssertionError("Joined V2 cell universe contains duplicates")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, frozen_row in enumerate(frozen, start=1):
        if strict_int(frozen_row["selection_rank"], "selection_rank") != rank:
            raise ValueError("Historical representative ranks are not contiguous")
        cell_id = frozen_row["cell_id"]
        if cell_id in seen or cell_id not in by_id:
            raise ValueError(f"Invalid historical representative cell_id: {cell_id}")
        seen.add(cell_id)
        source = by_id[cell_id]
        for field in ("context_key", "source_id", "cluster"):
            if str(source.get(field, "")) != frozen_row[field]:
                raise ValueError(
                    f"Historical representative {field} disagrees for {cell_id}"
                )
        for field in ("Dim1", "Dim2"):
            if not math.isclose(
                finite_float(source[field], field),
                finite_float(frozen_row[field], field),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Historical representative {field} disagrees for {cell_id}"
                )
        selected.append(
            {
                **source,
                "_selection_round": rank,
                "_context_value": str(source["context_key"]),
            }
        )
    if len(selected) != int(selection.get("selected_count", -1)):
        raise ValueError("Historical representative count disagrees with manifest")
    available_counts: dict[tuple[str, str], int] = defaultdict(int)
    selected_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        available_counts[(str(row["context_key"]), str(row["cluster"]))] += 1
    for row in selected:
        selected_counts[(str(row["context_key"]), str(row["cluster"]))] += 1
    audit = [
        {
            "scope": f"context:{context}",
            "cluster": cluster,
            "available_n": available_counts[(context, cluster)],
            "selected_n": selected_counts[(context, cluster)],
            "selection_authority": "historical_representatives.tsv",
        }
        for context, cluster in sorted(
            available_counts, key=lambda value: (value[0], int(value[1]))
        )
    ]
    provenance = {
        "path": str(representatives_path),
        "sha256": representatives_sha256,
        "historical_projection_manifest": str(manifest_path),
        "historical_projection_manifest_sha256": sha256_file(manifest_path),
    }
    return selected, audit, provenance


def choose_representatives(
    rows: list[dict[str, Any]],
    max_representatives: int,
    seed: int,
    context_column: str,
) -> list[dict[str, Any]]:
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        context = str(row.get(context_column, ""))
        if not context:
            raise ValueError(f"Blank {context_column} for cell_id={row['cell_id']}")
        row["_context_value"] = context
        by_context[context].append(row)
    total = min(max_representatives, len(rows))
    context_quotas = balanced_quotas(
        {key: len(value) for key, value in by_context.items()}, total, seed, "context"
    )
    selected: list[dict[str, Any]] = []
    for context in sorted(by_context):
        by_well: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_context[context]:
            by_well[str(row["well"])].append(row)
        well_quotas = balanced_quotas(
            {key: len(value) for key, value in by_well.items()},
            context_quotas[context],
            seed,
            f"context:{context}:well",
        )
        for well in sorted(by_well):
            chosen = spatial_real_cells(
                by_well[well],
                well_quotas[well],
                seed,
                f"context:{context}:well:{well}",
            )
            for selection_round, row in enumerate(chosen, start=1):
                selected.append({**row, "_selection_round": selection_round})
    selected.sort(
        key=lambda row: (
            int(row["_selection_round"]),
            stable_hash(seed, "rank", row["_context_value"], row["well"]),
            stable_hash(seed, "rank-cell", row["cell_id"]),
        )
    )
    if len(selected) != total or len({row["cell_id"] for row in selected}) != total:
        raise AssertionError("Representative selection cardinality invariant failed")
    return selected


def load_images(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    fields, rows = read_tsv(path)
    required = {
        "image_id",
        "channel_id",
        "image_path",
        "mask_path",
        "width",
        "height",
    }
    if not required.issubset(fields):
        raise ValueError(f"images.tsv lacks fields: {sorted(required-set(fields))}")
    channels = {row["channel_id"].strip().lower() for row in rows}
    if channels != ALLOWED_CHANNELS:
        raise ValueError(
            "Reference morphology images.tsv must contain only Brightfield and Nuclei; "
            f"observed={sorted(channels)}"
        )
    output: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        image_id = row["image_id"]
        channel = row["channel_id"].strip().lower()
        if not image_id or channel in output[image_id]:
            raise ValueError(f"Duplicate/incomplete image channel row: {image_id}/{channel}")
        output[image_id][channel] = row
    incomplete = sorted(
        image_id for image_id, values in output.items() if set(values) != ALLOWED_CHANNELS
    )
    if incomplete:
        raise ValueError(f"Images lack BF/Nuclei pairs: {incomplete[:10]}")
    return dict(output)


def parse_one_based_index(value: str, field: str) -> int | None:
    if not value.strip():
        return None
    parsed = strict_int(value, field)
    if parsed < 1:
        raise ValueError(f"{field} must be one-based")
    return parsed - 1


def read_plane(
    path: Path,
    *,
    page_index: int | None,
    channel_index: int | None,
    expected_height: int,
    expected_width: int,
    role: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        if page_index is None:
            array = tifffile.imread(path)
        else:
            with tifffile.TiffFile(path) as image:
                if page_index >= len(image.pages):
                    raise ValueError(f"{role} page index is out of range: {path}")
                array = image.pages[page_index].asarray()
    else:
        if page_index is not None:
            raise ValueError(f"{role} page_index requires TIFF: {path}")
        array = np.asarray(Image.open(path))
    array = np.asarray(array)
    if array.ndim == 2:
        if channel_index not in {None, 0}:
            raise ValueError(f"{role} channel index is invalid for scalar image: {path}")
    elif array.ndim == 3 and channel_index is not None:
        if array.shape[:2] == (expected_height, expected_width):
            if channel_index >= array.shape[2]:
                raise ValueError(f"{role} channel index is out of range: {path}")
            array = array[:, :, channel_index]
        elif array.shape[1:] == (expected_height, expected_width):
            if channel_index >= array.shape[0]:
                raise ValueError(f"{role} channel index is out of range: {path}")
            array = array[channel_index, :, :]
        else:
            raise ValueError(f"{role} channel axis cannot be resolved: {path} shape={array.shape}")
    else:
        # Refuse implicit RGB/component reduction: Combined RGB must never be
        # able to masquerade as the morphology Brightfield input.
        raise ValueError(f"{role} must resolve explicitly to one scalar plane: {path} shape={array.shape}")
    if array.shape != (expected_height, expected_width):
        raise ValueError(
            f"{role} dimensions disagree with images.tsv: {path} "
            f"expected={(expected_height, expected_width)} observed={array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{role} is not numeric: {path}")
    return array


def load_channel(row: dict[str, str], role: str) -> tuple[np.ndarray, Path]:
    width = strict_int(row["width"], f"{role}.width")
    height = strict_int(row["height"], f"{role}.height")
    path = Path(row["image_path"]).expanduser().resolve()
    return (
        read_plane(
            path,
            page_index=parse_one_based_index(row.get("page_index", ""), f"{role}.page_index"),
            channel_index=parse_one_based_index(
                row.get("channel_index", ""), f"{role}.channel_index"
            ),
            expected_height=height,
            expected_width=width,
            role=role,
        ),
        path,
    )


def load_mask(row: dict[str, str]) -> tuple[np.ndarray, Path]:
    width = strict_int(row["width"], "mask.width")
    height = strict_int(row["height"], "mask.height")
    path = Path(row["mask_path"]).expanduser().resolve()
    return (
        read_plane(
            path,
            page_index=None,
            channel_index=parse_one_based_index(
                row.get("mask_channel_index", ""), "mask.mask_channel_index"
            ),
            expected_height=height,
            expected_width=width,
            role="Combined object mask",
        ),
        path,
    )


def normalize_plane(array: np.ndarray, low: float, high: float) -> np.ndarray:
    finite = np.asarray(array[np.isfinite(array)], dtype=np.float64)
    if not finite.size:
        raise ValueError("Displayed raw channel contains no finite pixels")
    lower, upper = np.percentile(finite, [low, high])
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        raise ValueError("Displayed raw channel percentiles are not finite")
    if upper <= lower:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((np.asarray(array, dtype=np.float64) - lower) / (upper - lower), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return np.rint(scaled * 255).astype(np.uint8)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def rgba_crop(
    scaled: np.ndarray,
    selected_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    channel: str,
) -> Image.Image:
    x0, y0, x1, y1 = bounds
    values = scaled[y0:y1, x0:x1]
    object_mask = selected_mask[y0:y1, x0:x1]
    boundary = mask_boundary(object_mask)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    if channel == "brightfield":
        rgba[..., :3] = values[..., None]
        rgba[boundary, :3] = np.asarray([255, 184, 32], dtype=np.uint8)
    elif channel == "nuclei":
        rgba[..., 1] = values
        rgba[..., 2] = values
        rgba[boundary, :3] = np.asarray([255, 255, 255], dtype=np.uint8)
    else:
        raise AssertionError(channel)
    rgba[..., 3] = np.where(object_mask, 232, 0).astype(np.uint8)
    rgba[boundary, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def display_percentiles(row: dict[str, str], role: str) -> tuple[float, float]:
    low = finite_float(row.get("display_percentile_low", "1") or "1", f"{role}.low")
    high = finite_float(row.get("display_percentile_high", "99") or "99", f"{role}.high")
    if not 0 <= low < high <= 100:
        raise ValueError(f"Invalid {role} display percentiles: {low}, {high}")
    return low, high


def render_crops(
    representatives: list[dict[str, Any]],
    image_rows: dict[str, dict[str, dict[str, str]]],
    output: Path,
    padding: int,
    include_cluster: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for rank, row in enumerate(representatives, start=1):
        by_image[str(row["image_id"])].append((rank, row))
    missing = sorted(set(by_image) - set(image_rows))
    if missing:
        raise ValueError(f"Selected cells lack images.tsv rows: {missing[:10]}")
    rendered: list[dict[str, Any]] = []
    source_hashes: dict[Path, str] = {}
    source_roles: dict[tuple[str, Path], None] = {}
    for image_id in sorted(by_image):
        rows = image_rows[image_id]
        brightfield, bf_path = load_channel(rows["brightfield"], "Brightfield")
        nuclei, nuclei_path = load_channel(rows["nuclei"], "Nuclei")
        mask, mask_path = load_mask(rows["brightfield"])
        second_mask = Path(rows["nuclei"]["mask_path"]).expanduser().resolve()
        if second_mask != mask_path:
            raise ValueError(f"BF/Nuclei mask identity differs for image_id={image_id}")
        for role, path in (
            ("brightfield_raw", bf_path),
            ("nuclei_raw", nuclei_path),
            ("combined_mask", mask_path),
        ):
            source_roles[(role, path)] = None
            source_hashes.setdefault(path, sha256_file(path))
        bf_low, bf_high = display_percentiles(rows["brightfield"], "Brightfield")
        nu_low, nu_high = display_percentiles(rows["nuclei"], "Nuclei")
        bf_scaled = normalize_plane(brightfield, bf_low, bf_high)
        nuclei_scaled = normalize_plane(nuclei, nu_low, nu_high)
        for rank, row in by_image[image_id]:
            label = strict_int(str(row["mask_label"]), f"mask_label for {row['cell_id']}")
            selected_mask = mask == label
            yy, xx = np.nonzero(selected_mask)
            if not len(xx):
                raise ValueError(
                    f"Selected mask label is absent: image_id={image_id} label={label}"
                )
            x0 = max(0, int(np.min(xx)) - padding)
            x1 = min(mask.shape[1], int(np.max(xx)) + padding + 1)
            y0 = max(0, int(np.min(yy)) - padding)
            y1 = min(mask.shape[0], int(np.max(yy)) + padding + 1)
            bounds = (x0, y0, x1, y1)
            token = stable_hash(row["cell_id"])[:12]
            base = f"{rank:03d}_{token}"
            bf_relative = Path("crops") / f"{base}__brightfield.png"
            nuclei_relative = Path("crops") / f"{base}__nuclei_support.png"
            save_png(
                rgba_crop(bf_scaled, selected_mask, bounds, "brightfield"),
                output / bf_relative,
            )
            save_png(
                rgba_crop(nuclei_scaled, selected_mask, bounds, "nuclei"),
                output / nuclei_relative,
            )
            rendered.append(
                {
                    "selection_rank": rank,
                    "selection_round": row["_selection_round"],
                    "cell_id": row["cell_id"],
                    "image_id": image_id,
                    "mask_label": label,
                    "well": row["well"],
                    "context_column": row["_context_column"],
                    "context_value": row["_context_value"],
                    **({"cluster": row.get("cluster", "")} if include_cluster else {}),
                    "Dim1": format(float(row["Dim1"]), ".17g"),
                    "Dim2": format(float(row["Dim2"]), ".17g"),
                    "crop_x0_px": x0,
                    "crop_y0_px": y0,
                    "crop_x1_px": x1,
                    "crop_y1_px": y1,
                    "brightfield_crop": bf_relative.as_posix(),
                    "nuclei_crop": nuclei_relative.as_posix(),
                    "brightfield_crop_sha256": sha256_file(output / bf_relative),
                    "nuclei_crop_sha256": sha256_file(output / nuclei_relative),
                    "brightfield_source_sha256": source_hashes[bf_path],
                    "nuclei_source_sha256": source_hashes[nuclei_path],
                    "combined_mask_source_sha256": source_hashes[mask_path],
                }
            )
    rendered.sort(key=lambda row: int(row["selection_rank"]))
    assets = [
        {
            "role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": source_hashes[path],
        }
        for role, path in sorted(source_roles, key=lambda item: (item[0], str(item[1])))
    ]
    return rendered, assets


def coordinate_transform(
    rows: list[dict[str, Any]], width: int, height: int, margin: int
) -> tuple[dict[str, tuple[int, int]], dict[str, float]]:
    x = np.asarray([float(row["Dim1"]) for row in rows])
    y = np.asarray([float(row["Dim2"]) for row in rows])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        px = margin + round((float(row["Dim1"]) - x_min) / (x_max - x_min) * usable_width)
        py = height - margin - round(
            (float(row["Dim2"]) - y_min) / (y_max - y_min) * usable_height
        )
        result[str(row["cell_id"])] = (px, py)
    return result, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def cluster_color(value: object) -> tuple[int, int, int]:
    cluster = str(value)
    if cluster == "0":
        return (170, 176, 184)
    palette = (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    )
    try:
        index = int(cluster)
    except ValueError:
        index = int(stable_hash("cluster-color", cluster)[:8], 16)
    return palette[(index - 1) % len(palette)]


def render_overlay(
    all_rows: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    workspace: Path,
    width: int,
    height: int,
    tile_px: int,
    cluster_coloring: bool = False,
) -> dict[str, float]:
    margin = 70
    locations, bounds = coordinate_transform(all_rows, width, height, margin)
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((margin, margin, width - margin, height - margin), fill=(255, 255, 255), outline=(85, 94, 104), width=2)
    for row in all_rows:
        px, py = locations[str(row["cell_id"])]
        draw.point(
            (px, py),
            fill=(
                cluster_color(row.get("cluster", "0"))
                if cluster_coloring
                else (205, 211, 217)
            ),
        )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    source_max_dimensions = [
        max(
            int(row["crop_x1_px"]) - int(row["crop_x0_px"]),
            int(row["crop_y1_px"]) - int(row["crop_y0_px"]),
        )
        for row in representatives
    ]
    reference_source_pixels = float(np.median(source_max_dimensions))
    source_to_overlay_scale = tile_px / reference_source_pixels
    for row in representatives:
        with Image.open(workspace / str(row["brightfield_crop"])) as source_crop:
            crop = source_crop.convert("RGBA")
        size = (
            max(1, round(crop.width * source_to_overlay_scale)),
            max(1, round(crop.height * source_to_overlay_scale)),
        )
        crop = crop.resize(size, resampling)
        px, py = locations[str(row["cell_id"])]
        canvas.paste(crop, (px - crop.width // 2, py - crop.height // 2), crop)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Reference morphology UMAP: Brightfield objects", fill=(24, 32, 42))
    draw.text(
        (margin, 39),
        (
            "Points = diagnostic cluster metadata; gold = Combined-mask boundary; Nuclei excluded"
            if cluster_coloring
            else "Constant source-pixel scale; gold = Combined-mask boundary; Nuclei excluded"
        ),
        fill=(83, 92, 103),
    )
    save_png(canvas, workspace / "umap_morphology_overlay.png")
    bounds["reference_source_crop_max_dimension_px"] = reference_source_pixels
    bounds["source_pixel_to_overlay_pixel_scale"] = source_to_overlay_scale
    return bounds


def atlas_html(
    representatives: list[dict[str, Any]],
    identity: dict[str, Any],
    include_cluster: bool = False,
) -> str:
    contexts = sorted({str(row["context_value"]) for row in representatives})
    wells = sorted({str(row["well"]) for row in representatives})
    cards = []
    for row in representatives:
        cluster_text = (
            f'; cluster={html.escape(str(row.get("cluster", "")))}'
            if include_cluster
            else ""
        )
        cards.append(
            f'''<article class="card" data-context="{html.escape(str(row["context_value"]), quote=True)}" data-well="{html.escape(str(row["well"]), quote=True)}" data-search="{html.escape((str(row["cell_id"])+" "+str(row["image_id"])).lower(), quote=True)}">
<header><b>#{row["selection_rank"]}</b><span>{html.escape(str(row["well"]))}</span></header>
<div class="pair"><figure><img loading="lazy" src="{quote(str(row["brightfield_crop"]), safe='/')}" alt="Brightfield cell crop"><figcaption>Brightfield + mask boundary</figcaption></figure><figure><img loading="lazy" src="{quote(str(row["nuclei_crop"]), safe='/')}" alt="Nuclei support crop"><figcaption>Nuclei (review support only)</figcaption></figure></div>
<details><summary>Identity</summary><code>{html.escape(str(row["cell_id"]))}</code><br><small>image={html.escape(str(row["image_id"]))}; label={row["mask_label"]}{cluster_text}; UMAP=({row["Dim1"]}, {row["Dim2"]})</small></details>
</article>'''
        )
    context_options = "".join(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
        for value in contexts
    )
    well_options = "".join(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
        for value in wells
    )
    identity_text = " / ".join(
        html.escape(str(identity.get(key, "")))
        for key in ("project_id", "run_id", "projection_id", "annotation_id")
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference morphology atlas</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eef1f4;color:#17212b;font:14px system-ui,sans-serif}}header.top{{position:sticky;top:0;z-index:2;background:white;border-bottom:1px solid #ccd3da;padding:12px 18px}}h1{{font-size:18px;margin:0 0 5px}}.id{{color:#65717d;font-size:11px;overflow-wrap:anywhere}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}select,input,a{{border:1px solid #bdc7d0;border-radius:5px;background:white;color:#17212b;padding:7px}}a{{text-decoration:none}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px;padding:10px}}.card{{background:white;border:1px solid #d6dce2;border-radius:7px;padding:9px;min-width:0}}.card>header{{display:flex;justify-content:space-between;margin-bottom:6px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}figure{{margin:0;background:#111820;border-radius:5px;overflow:hidden;text-align:center}}img{{width:100%;height:150px;object-fit:contain;image-rendering:auto}}figcaption{{background:#f5f7f9;color:#495460;font-size:11px;padding:5px}}details{{margin-top:7px}}code{{font-size:10px;overflow-wrap:anywhere}}.hidden{{display:none}}</style></head>
<body><header class="top"><h1>Reference morphology atlas</h1><div class="id">{identity_text}</div><div class="controls"><input id="search" placeholder="cell_id or image_id"><select id="context"><option value="">All contexts</option>{context_options}</select><select id="well"><option value="">All wells</option>{well_options}</select><a href="umap_morphology_overlay.png">Open full overlay</a></div></header>
<main id="cards">{''.join(cards)}</main>
<script>"use strict";const q=document.getElementById("search"),c=document.getElementById("context"),w=document.getElementById("well"),cards=[...document.querySelectorAll(".card")];function filter(){{const text=q.value.trim().toLowerCase();for(const card of cards)card.classList.toggle("hidden",!!((text&&!card.dataset.search.includes(text))||(c.value&&card.dataset.context!==c.value)||(w.value&&card.dataset.well!==w.value)))}}q.addEventListener("input",filter);c.addEventListener("change",filter);w.addEventListener("change",filter);</script></body></html>'''


def workspace_html(
    annotation_html: Path,
    output_dir: Path,
    identity: dict[str, Any],
) -> str:
    annotation_relative = os.path.relpath(annotation_html, output_dir).replace(os.sep, "/")
    source = quote(annotation_relative, safe="/.")
    identity_json = json_script(identity)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference cell-state annotation workspace</title>
<style>*{{box-sizing:border-box}}html,body{{height:100%;margin:0;background:#e9edf1;font:13px system-ui,sans-serif;color:#17212b}}header{{height:48px;display:flex;align-items:center;gap:14px;padding:0 14px;background:white;border-bottom:1px solid #cbd3da}}header b{{font-size:15px}}header span{{color:#687480}}header a{{margin-left:auto}}main{{height:calc(100% - 48px);display:grid;grid-template-columns:minmax(600px,1.6fr) minmax(360px,.8fr);gap:8px;padding:8px}}section{{background:white;border:1px solid #cbd3da;border-radius:6px;overflow:hidden;display:grid;grid-template-rows:38px 1fr}}h2{{font-size:13px;margin:0;padding:10px;border-bottom:1px solid #d7dde3}}iframe{{width:100%;height:100%;border:0}}.reference{{overflow:auto;background:#111820;text-align:center}}.reference img{{max-width:100%;height:auto;display:block;margin:auto}}.links{{position:absolute;right:18px;top:59px;background:rgba(255,255,255,.94);padding:7px;border-radius:5px}}a{{color:#175fa8}}@media(max-width:1000px){{main{{grid-template-columns:1fr;grid-template-rows:minmax(650px,1fr) auto;height:auto}}.reference img{{max-height:none}}}}</style></head>
<body><header><b>Reference cell-state annotation</b><span>Polygon labels are provisional; image review remains authoritative.</span><a href="morphology_atlas.html" target="_blank">Open atlas</a></header><main><section><h2>Immutable CPA polygon editor</h2><iframe src="{source}" title="CPA annotation editor"></iframe></section><section><h2>Brightfield morphology reference</h2><div class="reference"><img src="umap_morphology_overlay.png" alt="Brightfield morphology UMAP overlay"></div></section></main><script type="application/json" id="workspace-identity">{identity_json}</script></body></html>'''


def directory_hashes(path: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {
        candidate.relative_to(path).as_posix(): sha256_file(candidate)
        for candidate in sorted(path.rglob("*"), key=lambda value: str(value))
        if candidate.is_file() and candidate.relative_to(path).as_posix() not in excluded
    }


def install_generation(staging: Path, output: Path, overwrite: bool) -> str:
    if not output.exists():
        os.replace(staging, output)
        return "created"
    if not overwrite:
        raise FileExistsError(
            f"Output exists; use --overwrite to verify byte-identical reuse: {output}"
        )
    if not output.is_dir() or directory_hashes(output) != directory_hashes(staging):
        raise FileExistsError(f"Existing morphology workspace conflicts and was preserved: {output}")
    shutil.rmtree(staging)
    return "verified_reuse"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = sha256_file(implementation_path)
    if args.max_representatives < 1:
        raise ValueError("--max-representatives must be positive")
    if args.mask_padding < 0:
        raise ValueError("--mask-padding must be nonnegative")
    if min(args.overlay_width, args.overlay_height) < 400 or args.overlay_tile_px < 4:
        raise ValueError("Overlay dimensions/tile size are too small")
    if not 0 <= args.cluster_balance <= 1:
        raise ValueError("--cluster-balance must be between zero and one")
    if args.minimum_cluster_representatives < 1:
        raise ValueError("--minimum-cluster-representatives must be positive")

    shadow_root = args.shadow_root.expanduser().resolve()
    if not shadow_root.is_dir():
        raise FileNotFoundError(shadow_root)
    project_argument = require_inside(args.project, shadow_root, "Reference project")
    project_path = project_argument / "project.yml" if project_argument.is_dir() else project_argument
    if not project_path.is_file():
        raise FileNotFoundError(project_path)
    project = load_mapping(project_path)
    if project.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported CPA project schema: {project.get('schema_version')}")
    for field in ("project_id", "cells_file", "images_file"):
        if not isinstance(project.get(field), str) or not project[field]:
            raise ValueError(f"Reference project lacks {field}")
    v2 = project.get("project_id") == "reference_cell_state_development_v2"
    if args.seed is None:
        args.seed = 1 if v2 else 20260812
    if v2 and args.seed != 1:
        raise ValueError("V2 reference morphology sampling seed is frozen at 1")
    if v2 and args.max_representatives != 300:
        raise ValueError("V2 reference morphology total_n is frozen at 300")
    if v2 and not math.isclose(args.cluster_balance, 0.2, abs_tol=1e-12):
        raise ValueError("V2 reference morphology cluster balance is frozen at 0.2")
    if v2 and args.minimum_cluster_representatives != 2:
        raise ValueError("V2 reference morphology minimum cluster representatives is frozen at 2")
    cells_path = require_inside(
        resolve_project_path(project_path, project["cells_file"]), shadow_root, "cells.tsv"
    )
    images_path = require_inside(
        resolve_project_path(project_path, project["images_file"]), shadow_root, "images.tsv"
    )
    if args.annotation_dir:
        annotation_dir = require_inside(args.annotation_dir, shadow_root, "Annotation directory")
        payload_path = annotation_dir / "annotation_payload.json"
    else:
        assert args.annotation_payload is not None
        payload_path = require_inside(args.annotation_payload, shadow_root, "Annotation payload")
        annotation_dir = payload_path.parent
    output = require_inside(args.output_dir, shadow_root, "Morphology workspace output")
    if (
        output == annotation_dir
        or annotation_dir in output.parents
        or output in annotation_dir.parents
    ):
        raise ValueError("Morphology sidecar and immutable annotation generation must not overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists; use --overwrite to verify byte-identical reuse: {output}"
        )

    payload, annotation_manifest, annotation_html_path = verify_annotation_generation(
        annotation_dir, payload_path
    )
    identity = dict(payload["identity"])
    if identity["project_id"] != project["project_id"]:
        raise ValueError("Reference project_id disagrees with CPA annotation identity")
    points = annotation_points(payload)
    fields, cells = load_cells(cells_path)
    joined = join_points_to_cells(points, cells)
    if int(annotation_manifest.get("row_count", -1)) != len(joined):
        raise ValueError("CPA annotation manifest row_count disagrees with verified universe")
    context_column = args.context_column or ("context_key" if v2 else "well")
    if context_column not in fields and not all(context_column in row for row in joined):
        raise ValueError(f"Unknown context column: {context_column}")
    for row in joined:
        row["_context_column"] = context_column
    cluster_audit: list[dict[str, Any]] = []
    historical_representative_input: dict[str, Any] | None = None
    if v2:
        if context_column != "context_key":
            raise ValueError("V2 morphology context is frozen at context_key, not wells")
        if {str(row.get("context_key", "")) for row in joined} != {
            "SUM-159-NLS-2N",
            "SUM-159-NLS-4N",
        }:
            raise ValueError("V2 morphology requires the two frozen ploidy contexts")
        if any("cluster" not in row for row in joined):
            raise ValueError("V2 morphology requires diagnostic cluster metadata")
        (
            representatives,
            cluster_audit,
            historical_representative_input,
        ) = load_reference_v2_representatives(
            project_path,
            shadow_root,
            joined,
        )
    else:
        representatives = choose_representatives(
            joined, args.max_representatives, args.seed, context_column
        )
    images = load_images(images_path)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        rendered, source_assets = render_crops(
            representatives,
            images,
            staging,
            args.mask_padding,
            include_cluster=v2,
        )
        write_tsv(
            staging / "representative_cells.tsv",
            REPRESENTATIVE_FIELDS_V2 if v2 else REPRESENTATIVE_FIELDS,
            rendered,
        )
        if v2:
            write_tsv(
                staging / "cluster_selection_audit.tsv",
                [
                    "scope",
                    "cluster",
                    "available_n",
                    "selected_n",
                    "selection_authority",
                ],
                cluster_audit,
            )
        bounds = render_overlay(
            joined,
            rendered,
            staging,
            args.overlay_width,
            args.overlay_height,
            args.overlay_tile_px,
            cluster_coloring=v2,
        )
        (staging / "morphology_atlas.html").write_text(
            atlas_html(rendered, identity, include_cluster=v2), encoding="utf-8"
        )
        (staging / "annotation_workspace.html").write_text(
            workspace_html(annotation_html_path, output, identity), encoding="utf-8"
        )
        output_hashes = directory_hashes(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION_V2 if v2 else SCHEMA_VERSION,
            "status": "COMPLETE",
            "identity": identity,
            "inputs": {
                "project": {
                    "path": str(project_path),
                    "sha256": sha256_file(project_path),
                },
                "cells": {
                    "path": str(cells_path),
                    "sha256": sha256_file(cells_path),
                },
                "images": {
                    "path": str(images_path),
                    "sha256": sha256_file(images_path),
                },
                "annotation_manifest": {
                    "path": str(annotation_dir / "annotation_manifest.json"),
                    "sha256": sha256_file(annotation_dir / "annotation_manifest.json"),
                },
                "annotation_html": {
                    "path": str(annotation_html_path),
                    "sha256": sha256_file(annotation_html_path),
                },
                "annotation_payload": {
                    "path": str(payload_path),
                    "sha256": sha256_file(payload_path),
                },
                **(
                    {
                        "implementation": {
                            "path": str(implementation_path),
                            "sha256": implementation_sha256,
                        },
                        "historical_representatives": historical_representative_input,
                    }
                    if v2 and historical_representative_input
                    else {}
                ),
                "source_assets": source_assets,
            },
            "selection": {
                "policy": (
                    "render exact frozen IDs selected by pinned historical R AST: context-equal allocation; "
                    "diagnostic-cluster convex-hull-area/count quotas; cluster-internal R k-means centers "
                    "mapped to nearest real cells"
                    if v2
                    else "hierarchical balanced context/well quotas; deterministic UMAP farthest-point real-cell design"
                ),
                "inputs": [
                    "cell_id",
                    "well",
                    context_column,
                    *( ["cluster"] if v2 else [] ),
                    "Dim1",
                    "Dim2",
                ],
                "seed": args.seed,
                "context_column": context_column,
                "max_representatives": args.max_representatives,
                **(
                    {
                        "cluster_balance": args.cluster_balance,
                        "minimum_cluster_representatives": args.minimum_cluster_representatives,
                        "context_count_expected": 2,
                        "selection_authority": "historical_projection/historical_representatives.tsv",
                        "selection_implementation": "pinned_reference_R_AST",
                        "outer_function_name": "get_all_cell_lines_overlay_representatives",
                        "inner_function_name": "get_spatially_uniform_representatives",
                        "diagnostic_cluster_role": "sampling_and_overlay_metadata_only",
                    }
                    if v2
                    else {}
                ),
                "selected_count": len(rendered),
                "well_count": len({row["well"] for row in rendered}),
                "context_count": len({row["context_value"] for row in rendered}),
            },
            "render": {
                "brightfield_role": "primary morphology overlay",
                "nuclei_role": "human review support only; excluded from selection and UMAP overlay",
                "combined_mask_role": "localization and visible boundary only",
                "mask_outside_alpha": 0,
                "mask_padding_px": args.mask_padding,
                "overlay_width_px": args.overlay_width,
                "overlay_height_px": args.overlay_height,
                "overlay_median_crop_target_px": args.overlay_tile_px,
                "overlay_size_policy": "one global source-pixel scale; relative cell size preserved",
                "background_point_color": (
                    "diagnostic_cluster_palette_noise_zero_grey"
                    if v2
                    else "neutral_grey"
                ),
                "coordinate_bounds": bounds,
            },
            "blinding_contract": {
                "dead_channel": "not_read",
                "combined_rgb": "not_read",
                "current_classification": "not_read",
                "provisional_annotation_labels": "not_read",
                "forbidden_input_count": 0,
            },
            "output_artifact_sha256": output_hashes,
        }
        if v2 and sha256_file(implementation_path) != implementation_sha256:
            raise RuntimeError("Morphology workspace implementation changed during execution")
        (staging / "overlay_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = install_generation(staging, output, args.overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"morphology_workspace={output}")
    print(f"representatives={len(rendered)}")
    print(f"annotation_workspace={output / 'annotation_workspace.html'}")
    print(f"generation_status={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
