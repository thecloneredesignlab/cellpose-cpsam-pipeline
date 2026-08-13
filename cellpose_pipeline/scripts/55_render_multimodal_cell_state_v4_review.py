#!/usr/bin/env python3
"""Render an exact V4 three-channel review selection without resampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile
from PIL import Image


SCHEMA_VERSION = "multimodal_cell_state_v4_review_render_v1"
SUBMISSION_SCHEMA = "multimodal_cell_state_v4_review_submission_v1"
PRIMARY_LABELS = ("non_dead", "dead", "uncertain_or_unreviewable")
DEATH_STAGE_LABELS = (
    "not_applicable_non_dead", "dead_marker_positive",
    "dead_marker_weak_or_transition", "dead_marker_negative_nucleus_lost_like",
    "death_stage_indeterminate",
)
NUCLEAR_LABELS = (
    "single_nucleus", "multinucleated", "fragmented_nuclei",
    "nucleus_absent", "nuclear_state_uncertain",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--review-set", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=12)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def render_generation_id(
    base_identity: dict[str, Any], crop_manifest_sha256: str,
    renderer_implementation_sha256: str, padding: int,
) -> str:
    payload = {
        "base_identity": base_identity,
        "crop_manifest_sha256": crop_manifest_sha256,
        "renderer_implementation_sha256": renderer_implementation_sha256,
        "padding": padding,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "exact_review_render_" + hashlib.sha256(encoded).hexdigest()[:24]


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def reject_symlinks_and_confine(
    path: Path, root: Path, label: str, *, must_exist: bool,
) -> Path:
    candidate = lexical_absolute(path)
    root = lexical_absolute(root)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside reference shadow root: {candidate}") from error
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} contains a symlink path component: {current}")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    if not must_exist and os.path.lexists(candidate) and candidate.is_symlink():
        raise ValueError(f"{label} is a symlink: {candidate}")
    return candidate


def safe_crop_path(output: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or value.parts[0] != "crops" or ".." in value.parts:
        raise ValueError(f"Unsafe crop path in manifest: {relative}")
    return reject_symlinks_and_confine(output / value, output, "crop artifact", must_exist=True)


def review_manifest_id_hash(manifest: dict[str, Any], ids: list[str]) -> str | None:
    value = manifest.get("stable_id_sha256")
    if isinstance(value, str) and value:
        return value
    # Seed2 historically binds the exact set through its output review_set hash.
    return stable_id_sha(ids)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields or not rows or len(fields) != len(set(fields)):
        raise ValueError(f"Invalid TSV: {path}")
    return fields, rows


def write_table(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_render_generation(
    output: Path,
    expected_base_identity: dict[str, Any],
    expected_inputs: dict[str, dict[str, str]],
    expected_ids: list[str],
    renderer_implementation_sha256: str,
    padding: int,
) -> None:
    manifest_path = reject_symlinks_and_confine(
        output / "exact_review_render_manifest.json", output,
        "existing render manifest", must_exist=True,
    )
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "HUMAN_REVIEW_REQUIRED"
        or manifest.get("inputs") != expected_inputs
        or manifest.get("exact_selection_preserved") is not True
        or manifest.get("secondary_sampling_performed") is not False
        or manifest.get("all_crops_available") is not True
    ):
        raise ValueError("Existing exact-review render identity conflicts and was preserved")
    artifact_names = {
        "exact_review_html": "exact_review.html",
        "crop_render_status": "crop_render_status.tsv",
        "crop_manifest": "crop_manifest.tsv",
    }
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict) or set(declared) != set(artifact_names):
        raise ValueError("Existing exact-review artifact hash set is incomplete")
    for role, name in artifact_names.items():
        artifact = reject_symlinks_and_confine(
            output / name, output, f"existing {role}", must_exist=True,
        )
        if sha256(artifact) != declared[role]:
            raise ValueError(f"Existing exact-review artifact hash mismatch: {role}")
    fields, crops = read_table(output / "crop_manifest.tsv")
    expected_fields = ["morphology_umap_row_key", "channel", "relative_path", "sha256"]
    if fields != expected_fields or len(crops) != 3 * len(expected_ids):
        raise ValueError("Existing crop manifest schema or row count changed")
    pairs = {(row["morphology_umap_row_key"], row["channel"]) for row in crops}
    expected_pairs = {(stable_id, channel) for stable_id in expected_ids for channel in ("brightfield", "dead", "nuclei")}
    if pairs != expected_pairs or len(pairs) != len(crops):
        raise ValueError("Existing crop manifest does not cover every stable ID/channel exactly once")
    for row in crops:
        crop = safe_crop_path(output, row["relative_path"])
        if sha256(crop) != row["sha256"]:
            raise ValueError(f"Existing crop hash mismatch: {row['relative_path']}")
    if sha256(output / "crop_manifest.tsv") != manifest.get("crop_aggregate_sha256"):
        raise ValueError("Existing crop aggregate SHA mismatch")
    crop_sha = sha256(output / "crop_manifest.tsv")
    expected_identity = dict(expected_base_identity)
    expected_identity.update({
        "crop_manifest_sha256": crop_sha,
        "renderer_implementation_sha256": renderer_implementation_sha256,
        "render_padding": padding,
        "render_generation_id": render_generation_id(
            expected_base_identity, crop_sha, renderer_implementation_sha256, padding,
        ),
    })
    if manifest.get("identity") != expected_identity:
        raise ValueError("Existing exact-review render identity conflicts and was preserved")


def load_project(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Project must be a mapping")
    return value


def index_table(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        value = (row.get(key) or "").strip()
        if not value or value in out:
            raise ValueError(f"Blank or duplicate {key} in {label}")
        out[value] = row
    return out


def read_plane(path: Path, page: str, channel: str, height: int, width: int) -> np.ndarray:
    page_index = int(page) - 1 if page.strip() else None
    channel_index = int(channel) - 1 if channel.strip() else None
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path, key=page_index) if page_index is not None else tifffile.imread(path)
    else:
        array = np.asarray(Image.open(path))
    array = np.asarray(array)
    if array.ndim == 3:
        if channel_index is None:
            raise ValueError(f"Explicit channel_index required: {path}")
        if array.shape[:2] == (height, width):
            array = array[:, :, channel_index]
        elif array.shape[1:] == (height, width):
            array = array[channel_index]
        else:
            raise ValueError(f"Unresolved channel axis: {path}")
    elif array.ndim != 2 or channel_index not in {None, 0}:
        raise ValueError(f"Image must resolve to scalar plane: {path}")
    if array.shape != (height, width):
        raise ValueError(f"Image shape mismatch: {path}")
    return array


def normalize(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
        raise ValueError("Invalid display percentiles")
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((np.asarray(array, dtype=float) - lo) / (hi - lo), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return np.rint(scaled * 255).astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    inside = mask.copy()
    inside[1:] &= mask[:-1]
    inside[:-1] &= mask[1:]
    inside[:, 1:] &= mask[:, :-1]
    inside[:, :-1] &= mask[:, 1:]
    return mask & ~inside


def crop_image(values: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int], channel: str) -> Image.Image:
    x0, y0, x1, y1 = bounds
    values = values[y0:y1, x0:x1]
    object_mask = mask[y0:y1, x0:x1]
    edge = boundary(object_mask)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    if channel == "brightfield":
        rgb[:] = values[..., None]
        rgb[edge] = (255, 184, 32)
    elif channel == "nuclei":
        rgb[..., 1] = values
        rgb[..., 2] = values
        rgb[edge] = (255, 255, 255)
    elif channel == "dead":
        rgb[..., 0] = values
        rgb[..., 2] = np.rint(values * 0.55).astype(np.uint8)
        rgb[edge] = (255, 255, 255)
    else:
        raise AssertionError(channel)
    return Image.fromarray(rgb, "RGB")


def render_assets(
    selected: list[dict[str, str]],
    cells: dict[str, dict[str, str]],
    images: list[dict[str, str]],
    output: Path,
    padding: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, tuple[float, float]]]:
    image_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in images:
        key = ((row.get("image_id") or ""), (row.get("channel_id") or "").lower())
        if not all(key) or key in image_index:
            raise ValueError("Duplicate/incomplete images.tsv channel row")
        image_index[key] = row
    channels = {channel for _, channel in image_index}
    if channels != {"brightfield", "dead", "nuclei"}:
        raise ValueError(f"V4 review requires exactly three channels: {sorted(channels)}")
    window_samples: dict[str, list[np.ndarray]] = {channel: [] for channel in channels}
    for (image_id, channel), image_row in sorted(image_index.items()):
        width, height = int(image_row["width"]), int(image_row["height"])
        plane = read_plane(Path(image_row["image_path"]), image_row.get("page_index", ""), image_row.get("channel_index", ""), height, width)
        finite = np.asarray(plane[np.isfinite(plane)], dtype=float)
        stride = max(1, finite.size // 4096)
        window_samples[channel].append(finite[::stride][:4096])
    quantiles = {"brightfield": (0.01, 0.99), "dead": (0.01, 0.998), "nuclei": (0.01, 0.998)}
    windows = {channel: tuple(map(float, np.quantile(np.concatenate(values), quantiles[channel]))) for channel, values in window_samples.items()}
    if any(not math.isfinite(low) or not math.isfinite(high) or high <= low for low, high in windows.values()):
        raise ValueError("V4 global review display window is invalid")
    rendered: list[dict[str, Any]] = []
    status: list[dict[str, str]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(selected, 1):
        stable_id = row["morphology_umap_row_key"]
        cell = cells.get(stable_id)
        if cell is None:
            raise ValueError(f"Selected stable ID absent from cells.tsv: {stable_id}")
        image_id = cell["image_id"]
        bf_row = image_index.get((image_id, "brightfield"))
        dead_row = image_index.get((image_id, "dead"))
        nuclei_row = image_index.get((image_id, "nuclei"))
        if bf_row is None or dead_row is None or nuclei_row is None:
            raise ValueError(f"Missing BF/Dead/Nuclei rows: {image_id}")
        if image_id not in cache:
            width, height = int(bf_row["width"]), int(bf_row["height"])
            bf = read_plane(Path(bf_row["image_path"]), bf_row.get("page_index", ""), bf_row.get("channel_index", ""), height, width)
            dead = read_plane(Path(dead_row["image_path"]), dead_row.get("page_index", ""), dead_row.get("channel_index", ""), height, width)
            nuclei = read_plane(Path(nuclei_row["image_path"]), nuclei_row.get("page_index", ""), nuclei_row.get("channel_index", ""), height, width)
            mask_path = Path(bf_row["mask_path"])
            if any(mask_path.resolve() != Path(item["mask_path"]).resolve() for item in (dead_row, nuclei_row)):
                raise ValueError(f"BF/Dead/Nuclei mask identity mismatch: {image_id}")
            mask = read_plane(mask_path, "", bf_row.get("mask_channel_index", ""), height, width)
            cache[image_id] = (bf, dead, nuclei, mask)
        bf, dead, nuclei, mask = cache[image_id]
        label = int(cell["mask_label"])
        selected_mask = mask == label
        yy, xx = np.nonzero(selected_mask)
        if not xx.size:
            raise ValueError(f"Combined mask label absent: {stable_id}")
        bounds = (
            max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
            min(mask.shape[1], int(xx.max()) + padding + 1),
            min(mask.shape[0], int(yy.max()) + padding + 1),
        )
        bf_scaled = normalize(bf, *windows["brightfield"])
        dead_scaled = normalize(dead, *windows["dead"])
        nuclei_scaled = normalize(nuclei, *windows["nuclei"])
        token = hashlib.sha256(stable_id.encode()).hexdigest()[:12]
        bf_name = f"{rank:03d}_{token}__brightfield.png"
        dead_name = f"{rank:03d}_{token}__dead.png"
        nuclei_name = f"{rank:03d}_{token}__nuclei.png"
        crop_image(bf_scaled, selected_mask, bounds, "brightfield").save(crops / bf_name)
        crop_image(dead_scaled, selected_mask, bounds, "dead").save(crops / dead_name)
        crop_image(nuclei_scaled, selected_mask, bounds, "nuclei").save(crops / nuclei_name)
        rendered.append({
            "stable_id": stable_id,
            "brightfield": f"crops/{bf_name}",
            "dead": f"crops/{dead_name}",
            "nuclei": f"crops/{nuclei_name}",
            "brightfield_sha256": sha256(crops / bf_name),
            "dead_sha256": sha256(crops / dead_name),
            "nuclei_sha256": sha256(crops / nuclei_name),
        })
        status.append({
            "morphology_umap_row_key": stable_id, "image_id": image_id,
            "mask_label": str(label), "brightfield_status": "ok", "dead_status": "ok", "nuclei_status": "ok",
            "combined_mask_status": "ok",
        })
    return rendered, status, windows


def build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>V4 three-channel death review</title><style>
body{{font-family:Arial;margin:16px}}header{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #aaa;z-index:2}}#guide{{background:#f5f7fa;border:1px solid #ccd3dc;padding:10px;margin:8px 0}}#tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:12px}}article{{border:1px solid #bbb;padding:8px}}article.reviewed{{border:2px solid #16875b}}.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:#222}}img{{width:100%;height:180px;object-fit:contain}}label{{display:block;margin:5px 0}}select,textarea,input{{width:100%;box-sizing:border-box}}input[type=checkbox]{{width:auto}}code{{font-size:9px;overflow-wrap:anywhere}}button{{margin-right:8px}}#status{{font-weight:bold}}
</style></head><body><header><h1>V4 three-channel death review</h1><p id="summary"></p><label>Reviewer<input id="reviewer"></label><button id="export">Export authoritative submission</button><span id="status"></span></header><main id="tiles"></main>
<script type="application/json" id="data">{payload_json}</script><script>
"use strict";const p=JSON.parse(document.getElementById("data").textContent);const $=x=>document.getElementById(x);const storageKey=`multimodal-cell-state-v4:${{p.identity.render_generation_id}}`;const empty=()=>p.rows.map(x=>({{...x,primary_label:x.review_default_label||"uncertain_or_unreviewable",death_stage:"death_stage_indeterminate",nuclear_state:"nuclear_state_uncertain",confidence:"high",notes:"",reviewed:false}}));let state=empty();
function save(){{try{{localStorage.setItem(storageKey,JSON.stringify({{reviewer:$("reviewer").value,state}}))}}catch(e){{$('status').textContent='Browser draft storage unavailable; export before closing'}}}}
function restore(){{try{{const saved=JSON.parse(localStorage.getItem(storageKey)||"null");if(!saved||!Array.isArray(saved.state)||saved.state.length!==p.rows.length)return;if(!saved.state.every((r,i)=>r.review_token===p.rows[i].review_token))return;state=saved.state;$("reviewer").value=saved.reviewer||""}}catch(e){{state=empty()}}}}
function updateSummary(){{const reviewed=state.filter(r=>r.reviewed).length;const dead=state.filter(r=>r.primary_label==="dead").length;const nondead=state.filter(r=>r.primary_label==="non_dead").length;$("summary").textContent=`Reviewed ${{reviewed}}/${{state.length}} | dead ${{dead}} | non-dead ${{nondead}} | uncertain ${{state.length-dead-nondead}}`;}}
function render(){{const header=document.querySelector("header");const guide=document.createElement("div");guide.id="guide";guide.innerHTML='<b>Primary question: is this cell dead, regardless of death stage?</b><br>Inspect the same Combined-mask object in Brightfield, Dead fluorescence, and Nuclei fluorescence. Dead-negative plus nucleus-loss is not automatically live; use BF disruption and all three channels. Record a death stage only as a secondary descriptor. UMAP, polygon, condition, time, and the current classifier are hidden. High/medium reviewed labels may become training evidence.';header.insertBefore(guide,$("reviewer"));const clear=document.createElement("button");clear.textContent="Clear saved draft";clear.onclick=()=>{{if(!confirm("Clear all saved labels and review progress?"))return;localStorage.removeItem(storageKey);state=empty();$("reviewer").value="";document.getElementById("tiles").replaceChildren();renderTiles();save()}};header.insertBefore(clear,$("status"));renderTiles();}}
function opts(values,current){{return values.map(x=>`<option value="${{x}}" ${{x===current?"selected":""}}>${{x}}</option>`).join("")}}
function renderTiles(){{const root=$("tiles");root.replaceChildren();state.forEach((r,index)=>{{const a=document.createElement("article");if(r.reviewed)a.classList.add("reviewed");a.innerHTML=`<div class="images"><img src="${{r.brightfield}}" alt="Brightfield"><img src="${{r.dead}}" alt="Dead fluorescence"><img src="${{r.nuclei}}" alt="Nuclei fluorescence"></div><p>Three-channel blinded review</p><code>Review item #${{index+1}}</code><label>Primary death label<select class="primary">${{opts(p.primary_labels,r.primary_label)}}</select></label><label>Death-stage descriptor<select class="stage">${{opts(p.death_stage_labels,r.death_stage)}}</select></label><label>Nuclear state<select class="nuclear">${{opts(p.nuclear_labels,r.nuclear_state)}}</select></label><label>Confidence<select class="confidence"><option ${{r.confidence==="high"?"selected":""}}>high</option><option ${{r.confidence==="medium"?"selected":""}}>medium</option><option ${{r.confidence==="low"?"selected":""}}>low</option></select></label><label>Notes<textarea class="notes"></textarea></label><label><input type="checkbox" class="reviewed" ${{r.reviewed?"checked":""}}> I inspected this cell</label>`;a.querySelector(".notes").value=r.notes||"";a.querySelector(".primary").onchange=e=>{{r.primary_label=e.target.value;if(r.primary_label==="non_dead")r.death_stage="not_applicable_non_dead";else if(r.primary_label==="uncertain_or_unreviewable")r.death_stage="death_stage_indeterminate";save();renderTiles()}};a.querySelector(".stage").onchange=e=>{{r.death_stage=e.target.value;save()}};a.querySelector(".nuclear").onchange=e=>{{r.nuclear_state=e.target.value;save()}};a.querySelector(".confidence").onchange=e=>{{r.confidence=e.target.value;save()}};a.querySelector(".notes").oninput=e=>{{r.notes=e.target.value;save()}};a.querySelector(".reviewed").onchange=e=>{{r.reviewed=e.target.checked;a.classList.toggle("reviewed",r.reviewed);save();updateSummary()}};root.appendChild(a)}});updateSummary()}}
function download(value){{const blob=new Blob([JSON.stringify(value,null,2)+"\\n"],{{type:"application/json"}});const u=URL.createObjectURL(blob);const a=document.createElement("a");a.href=u;a.download="multimodal_v4_review_submission.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),0)}}
$("reviewer").oninput=save;$("export").onclick=()=>{{const reviewer=$("reviewer").value.trim();if(!reviewer){{$("status").textContent="Reviewer required";return}}const remaining=state.filter(r=>!r.reviewed).length;if(remaining){{$("status").textContent=`${{remaining}} cells still require explicit inspection`;return}}const invalid=state.filter(r=>(r.primary_label==="dead"&&r.death_stage==="not_applicable_non_dead")||(r.primary_label==="non_dead"&&r.death_stage!=="not_applicable_non_dead")||(r.primary_label==="uncertain_or_unreviewable"&&r.death_stage!=="death_stage_indeterminate"));if(invalid.length){{$("status").textContent="Death-stage descriptor must agree with the primary label";return}}const missingNotes=state.filter(r=>(r.primary_label==="uncertain_or_unreviewable"||r.confidence==="low")&&!String(r.notes||"").trim());if(missingNotes.length){{$("status").textContent="Uncertain or low-confidence reviews require notes";return}}const stamp=new Date().toISOString();download({{schema_version:p.submission_schema_version,identity:p.identity,creation_metadata:{{created_at:stamp,client_version:"multimodal-cell-state-v4-review"}},rows:state.map(r=>({{review_token:r.review_token,primary_death_label:r.primary_label,death_stage_label:r.death_stage,nuclear_state_label:r.nuclear_state,label_confidence:r.confidence,reviewer:reviewer,reviewed_at:stamp,review_notes:r.notes}}))}});$("status").textContent="Submission downloaded";save()}};restore();render();
</script></body></html>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_candidate = lexical_absolute(args.project)
    if project_candidate.parent.parent.name != "projection_input":
        raise ValueError("Project must be under V4_SHADOW_ROOT/projection_input/<project>/project.yml")
    shadow_root = project_candidate.parent.parent.parent
    project_path = reject_symlinks_and_confine(
        project_candidate, shadow_root, "project", must_exist=True,
    )
    review_path = reject_symlinks_and_confine(
        args.review_set, shadow_root, "review set", must_exist=True,
    )
    manifest_path = reject_symlinks_and_confine(
        args.review_manifest, shadow_root, "review manifest", must_exist=True,
    )
    output = reject_symlinks_and_confine(
        args.output_dir, shadow_root, "render output", must_exist=False,
    )
    if args.padding < 0:
        raise ValueError("--padding must be nonnegative")
    project = load_project(project_path)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") not in {
        "multimodal_cell_state_v4_anchor_review_v1",
        "multimodal_cell_state_v4_broad_region_review_v1",
    }:
        raise ValueError("Review manifest is not an accepted V4 exact preselection")
    fields, selected = read_table(review_path)
    required = {"morphology_umap_row_key", "review_default_label", "review_sampling_bucket", "context_key", "source_id"}
    if not required.issubset(fields):
        raise ValueError(f"Review set lacks fields: {sorted(required-set(fields))}")
    ids = [row["morphology_umap_row_key"] for row in selected]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Review set stable IDs are blank or duplicated")
    if int(manifest.get("row_count", -1)) != len(selected):
        raise ValueError("Review manifest row_count mismatch")
    if review_manifest_id_hash(manifest, ids) != stable_id_sha(ids):
        raise ValueError("Review manifest stable-ID universe mismatch")
    project_dir = project_path.parent
    cells_path = reject_symlinks_and_confine(
        project_dir / project["cells_file"], shadow_root, "project cells", must_exist=True,
    )
    images_path = reject_symlinks_and_confine(
        project_dir / project["images_file"], shadow_root, "project images", must_exist=True,
    )
    _, cell_rows = read_table(cells_path)
    _, image_rows = read_table(images_path)
    cells = index_table(cell_rows, "cell_id", "cells.tsv")
    review_id = "exact_review_" + hashlib.sha256(
        (sha256(review_path) + sha256(manifest_path) + stable_id_sha(ids)).encode()
    ).hexdigest()[:20]
    base_identity = {
        "review_id": review_id,
        "project_sha256": sha256(project_path),
        "review_set_sha256": sha256(review_path),
        "review_manifest_sha256": sha256(manifest_path),
        "stable_id_sha256": stable_id_sha(ids),
        "row_count": len(ids),
    }
    renderer_implementation_sha256 = sha256(Path(__file__).resolve())
    inputs = {
        "project": {"path": str(project_path), "sha256": sha256(project_path)},
        "cells": {"path": str(cells_path), "sha256": sha256(cells_path)},
        "images": {"path": str(images_path), "sha256": sha256(images_path)},
        "review_set": {"path": str(review_path), "sha256": sha256(review_path)},
        "review_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    }
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Render output exists but is not a directory: {output}")
        verify_render_generation(
            output, base_identity, inputs, ids, renderer_implementation_sha256,
            args.padding,
        )
        print(f"reference_cell_state_exact_review_verified_reuse=1 rows={len(selected)} html={output/'exact_review.html'}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rendered, status, windows = render_assets(selected, cells, image_rows, staging, args.padding)
        assets = {row["stable_id"]: row for row in rendered}
        payload_rows = []
        for row in selected:
            asset = assets[row["morphology_umap_row_key"]]
            payload_rows.append({
                "stable_id": row["morphology_umap_row_key"],
                "review_default_label": row["review_default_label"],
                "brightfield": asset["brightfield"], "dead": asset["dead"],
                "nuclei": asset["nuclei"],
            })
        write_table(staging / "crop_render_status.tsv", list(status[0]), status)
        crop_rows: list[dict[str, str]] = []
        for row in rendered:
            crop_rows.extend((
                {
                    "morphology_umap_row_key": row["stable_id"], "channel": "brightfield",
                    "relative_path": row["brightfield"], "sha256": row["brightfield_sha256"],
                },
                {
                    "morphology_umap_row_key": row["stable_id"], "channel": "dead",
                    "relative_path": row["dead"], "sha256": row["dead_sha256"],
                },
                {
                    "morphology_umap_row_key": row["stable_id"], "channel": "nuclei",
                    "relative_path": row["nuclei"], "sha256": row["nuclei_sha256"],
                },
            ))
        write_table(
            staging / "crop_manifest.tsv",
            ["morphology_umap_row_key", "channel", "relative_path", "sha256"],
            crop_rows,
        )
        crop_manifest_sha256 = sha256(staging / "crop_manifest.tsv")
        identity = dict(base_identity)
        identity.update({
            "crop_manifest_sha256": crop_manifest_sha256,
            "renderer_implementation_sha256": renderer_implementation_sha256,
            "render_padding": args.padding,
            "render_generation_id": render_generation_id(
                base_identity, crop_manifest_sha256,
                renderer_implementation_sha256, args.padding,
            ),
        })
        payload_rows = [
            {
                **{name: value for name, value in row.items() if name != "stable_id"},
                "review_token": hashlib.sha256(
                    f"{identity['render_generation_id']}|{row['stable_id']}".encode()
                ).hexdigest()[:24],
            }
            for row in payload_rows
        ]
        payload = {
            "submission_schema_version": SUBMISSION_SCHEMA,
            "identity": identity,
            "primary_labels": list(PRIMARY_LABELS),
            "death_stage_labels": list(DEATH_STAGE_LABELS),
            "nuclear_labels": list(NUCLEAR_LABELS),
            "rows": payload_rows,
        }
        (staging / "exact_review.html").write_text(build_html(payload), encoding="utf-8")
        render_manifest = {
            "schema_version": SCHEMA_VERSION, "status": "HUMAN_REVIEW_REQUIRED",
            "submission_schema_version": SUBMISSION_SCHEMA, "identity": identity,
            "reference_shadow_root": str(shadow_root), "inputs": inputs,
            "review_set": str(review_path), "review_manifest": str(manifest_path),
            "exact_selection_preserved": True, "secondary_sampling_performed": False,
            "channels": ["brightfield", "dead", "nuclei"], "localization_anchor": "combined_mask",
            "display_window_policy": "development_locked_channel_specific_global_quantiles_no_per_cell_autocontrast",
            "display_windows_raw_units": {channel: {"low": value[0], "high": value[1]} for channel, value in windows.items()},
            "all_crops_available": True,
            "reviewer_hidden_fields": [
                "stable_cell_id", "sampling_bucket", "context_key", "source_id",
                "UMAP_coordinates", "polygon_assignment", "condition", "time",
            ],
            "artifact_file_sha256": {
                "exact_review_html": sha256(staging / "exact_review.html"),
                "crop_render_status": sha256(staging / "crop_render_status.tsv"),
                "crop_manifest": sha256(staging / "crop_manifest.tsv"),
            },
            "crop_aggregate_sha256": crop_manifest_sha256,
            "renderer_implementation": str(Path(__file__).resolve()),
            "renderer_implementation_sha256": renderer_implementation_sha256,
            "render_padding": args.padding,
        }
        (staging / "exact_review_render_manifest.json").write_text(json.dumps(render_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if os.path.lexists(output):
            raise FileExistsError(f"Render output appeared during staging: {output}")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"reference_cell_state_exact_review_rendered=1 rows={len(selected)} html={output/'exact_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
