#!/usr/bin/env python3
"""Build the current-run d0 classification audit report from the production layout."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CURRENT_DIRS = (
    "classification_original",
    "classification_nucleated_only",
    "automated_audit",
    "annotations",
    "detector_stress",
)
SENTINEL_KEYS = ("E2_1_00d00h00m", "F5_1_00d00h00m", "H9_4_00d00h00m")


def is_current_run_layout(root: Path) -> bool:
    return all((root / name).is_dir() for name in CURRENT_DIRS)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def display_branch(value: str) -> str:
    return {"original": "Original", "nucleated_only": "Nucleated-only"}.get(value, value)


def selected(rows_: list[dict[str, str]], level: str) -> list[dict[str, str]]:
    return [row for row in rows_ if row["metric_level"] == level]


def current_datasets(root: Path) -> dict[str, list[dict[str, Any]]]:
    audit = rows(root / "automated_audit" / "automated_reference_metrics.csv")
    stress = rows(root / "detector_stress" / "detection_stress_metrics.csv")
    annotations = rows(root / "annotations" / "final_annotation_summary.csv")

    cell_states = [
        {
            "branch": display_branch(row["analysis_branch"]),
            "state": row["classification"],
            "count": int(row["count"]),
            "all_cells": int(row["denominator"]),
            "share_of_all_cells": float(row["percentage"]),
        }
        for row in selected(annotations, "cell_state")
    ]
    uncertainty = [
        {
            "branch": display_branch(row["analysis_branch"]),
            "final_state": row["classification"],
            "uncertain_count": int(row["count"]),
            "state_count": int(row["denominator"]),
            "uncertainty_rate": float(row["percentage"]),
        }
        for row in selected(annotations, "cell_uncertainty_by_state")
        if row["classification"] in {"live", "dead"}
    ]
    relations = [
        {
            "branch": display_branch(row["analysis_branch"]),
            "relation": row["subclassification"].replace("_", " "),
            "count": int(row["count"]),
            "confirmed_death_objects": int(row["denominator"]),
            "share_of_confirmed_death": float(row["percentage"]),
        }
        for row in selected(annotations, "confirmed_death_object_relation")
    ]
    event_composition = [
        {
            "branch": display_branch(row["analysis_branch"]),
            "component": row["subclassification"].replace("_", " "),
            "count": int(row["count"]),
            "object_aware_total": int(row["denominator"]),
            "share": float(row["percentage"]),
        }
        for row in selected(annotations, "object_aware_death_event_composition")
    ]
    audit_rows = [
        {
            "branch": display_branch(row["branch"]),
            "all_cells": int(row["total_combined_cells"]),
            "strong_objects": int(row["automated_reference_dead_object_count"]),
            "strong_objects_detected": int(row["automated_reference_dead_object_detected"]),
            "operational_recall": float(row["automated_reference_dead_object_recall"]),
            "rgb_live_reference": int(row["conservative_rgb_live_reference_count"]),
            "possible_live_false_positives": int(row["possible_live_false_positive_count"]),
            "live_false_positive_upper_bound": float(row["possible_live_false_positive_upper_bound"]),
            "qc_complete": int(row["qc_complete_count"]),
            "qc_incomplete": int(row["qc_incomplete_count"]),
        }
        for row in audit
    ]
    stress_rows: list[dict[str, Any]] = []
    for row in stress:
        branch = display_branch(row["branch"])
        denominator = int(row["strong_reference_count"])
        stress_rows.extend(
            [
                {
                    "branch": branch,
                    "test": "Strong-object global seed",
                    "rate": float(row["strong_reference_recall"]),
                    "recovered": int(row["strong_reference_recovered"]),
                    "denominator": denominator,
                },
                {
                    "branch": branch,
                    "test": "Synthetic relocation",
                    "rate": float(row["synthetic_relocation_recall"]),
                    "recovered": int(row["synthetic_relocation_recovered"]),
                    "denominator": denominator,
                },
            ]
        )
    master = root / "annotations" / "final_multilevel_annotations.csv"
    with master.open(newline="") as handle:
        reader = csv.reader(handle)
        columns = len(next(reader, []))
        master_rows = sum(1 for _ in reader)
    first = audit_rows[0]
    overview = [{
        "selected_fields": 320,
        "analysis_branches": 2,
        "annotation_rows": master_rows,
        "annotation_columns": columns,
        "segmented_dead_objects": int(audit[0]["segmented_dead_object_count"]),
        "strong_reference_objects": first["strong_objects"],
        "possible_live_false_positives": sum(row["possible_live_false_positives"] for row in audit_rows),
        "qc_fields": sum(row["qc_complete"] for row in audit_rows),
    }]
    return {
        "overview": overview,
        "cell_states": cell_states,
        "cell_uncertainty": uncertainty,
        "death_relations": relations,
        "event_composition": event_composition,
        "audit_metrics": audit_rows,
        "stress_metrics": stress_rows,
    }


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def find_qc(root: Path, branch: str, subdir: str, key: str, suffix: str) -> Path:
    matches = sorted((root / branch / "qc" / subdir).glob(f"*{key}*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one QC image for {branch}/{subdir}/{key}, found {len(matches)}")
    return matches[0]


def qc_figure(root: Path, key: str) -> Image.Image:
    specifications = (
        ("classification_original", "label_overlays", "_state_overlay.png", "Original branch: final cell states"),
        ("classification_original", "dead_object_overlays", "_dead_object_overlay.png", "Original branch: Death objects"),
        ("classification_original", "overlap_state_overlays", "_overlap_state_overlay.png", "Original branch: overlapping masks"),
        ("classification_nucleated_only", "overlap_state_overlays", "_overlap_state_overlay.png", "Nucleated-only branch: overlapping masks"),
    )
    panels: list[tuple[Image.Image, str]] = []
    target_width = 1500
    for branch, subdir, suffix, label in specifications:
        with Image.open(find_qc(root, branch, subdir, key, suffix)) as source:
            image = source.convert("RGB")
        height = max(1, round(image.height * target_width / image.width))
        panels.append((image.resize((target_width, height), Image.Resampling.LANCZOS), label))
    header = 58
    gap = 22
    cell_height = max(image.height for image, _ in panels) + header
    canvas = Image.new("RGB", (target_width * 2 + gap, cell_height * 2 + gap + 78), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), f"{key}: current-run final classification QC", fill=(245, 247, 250), font=font(34, True))
    for index, (image, label) in enumerate(panels):
        x = (index % 2) * (target_width + gap)
        y = 78 + (index // 2) * (cell_height + gap)
        draw.rectangle((x, y, x + target_width, y + header), fill=(35, 39, 47))
        draw.text((x + 18, y + 14), label, fill=(245, 247, 250), font=font(24, True))
        canvas.paste(image, (x, y + header))
    return canvas


def current_sources(root: Path) -> list[dict[str, Any]]:
    base = f"results/{root.name}"
    definitions = (
        ("summaries", "Final classification summaries", "classification_original/summaries/cell_count_summary.csv"),
        ("audit", "Automated no-ground-truth audit", "automated_audit/automated_reference_metrics.csv"),
        ("annotations", "Final multilevel annotation summary", "annotations/final_annotation_summary.csv"),
        ("stress", "Independent detector stress test", "detector_stress/detection_stress_metrics.csv"),
        ("qc", "Per-field final classification QC", "classification_original/qc/label_overlays/*.png"),
    )
    return [
        {
            "id": source_id,
            "label": label,
            "path": f"{base}/{relative}",
            "query": {
                "engine": "duckdb",
                "description": f"Read {label.lower()} for the 320-field d0 run.",
                "sql": f"SELECT * FROM read_csv_auto('{base}/{relative}', header=true)" if not relative.endswith("*.png") else "SELECT 'QC images are selected by exact d0 field key' AS selection_rule",
                "tables_used": [relative],
            },
        }
        for source_id, label, relative in definitions
    ]


def current_manifest(generated_at: str, datasets: dict[str, list[dict[str, Any]]], image_data: dict[str, tuple[str, int, int]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    charts = [
        {
            "id": "cell_state_chart", "title": "Final cell-state composition", "subtitle": "All 320 d0 fields; denominator is all Combined masks in each branch.",
            "type": "bar", "dataset": "cell_states", "sourceId": "annotations", "valueFormat": "percent", "options": {"grouping": "grouped"},
            "encodings": {"x": {"field": "state", "type": "nominal", "label": "Final state"}, "y": {"field": "share_of_all_cells", "type": "quantitative", "label": "Share", "format": "percent"}, "color": {"field": "branch", "type": "nominal", "label": "Branch"}, "tooltip": [{"field": "count", "type": "quantitative", "label": "Cells", "format": "number"}]},
        },
        {
            "id": "death_relation_chart", "title": "Confirmed Death-object relations", "subtitle": "Mutually exclusive final relations among confirmed Death objects.",
            "type": "bar", "dataset": "death_relations", "sourceId": "annotations", "valueFormat": "percent", "options": {"grouping": "stacked"},
            "encodings": {"x": {"field": "branch", "type": "nominal", "label": "Branch"}, "y": {"field": "share_of_confirmed_death", "type": "quantitative", "label": "Share", "format": "percent"}, "color": {"field": "relation", "type": "nominal", "label": "Relation"}, "tooltip": [{"field": "count", "type": "quantitative", "label": "Objects", "format": "number"}]},
        },
        {
            "id": "event_chart", "title": "Final death-event composition", "subtitle": "Cell-associated death and supplemental confirmed objects sum to the final event total.",
            "type": "bar", "dataset": "event_composition", "sourceId": "annotations", "valueFormat": "percent", "options": {"grouping": "stacked"},
            "encodings": {"x": {"field": "branch", "type": "nominal", "label": "Branch"}, "y": {"field": "share", "type": "quantitative", "label": "Share", "format": "percent"}, "color": {"field": "component", "type": "nominal", "label": "Component"}, "tooltip": [{"field": "count", "type": "quantitative", "label": "Events", "format": "number"}]},
        },
        {
            "id": "uncertainty_chart", "title": "Uncertainty retained within final cell states", "subtitle": "Denominator is the corresponding live or dead state in each branch.",
            "type": "bar", "dataset": "cell_uncertainty", "sourceId": "annotations", "valueFormat": "percent", "options": {"grouping": "grouped"},
            "encodings": {"x": {"field": "final_state", "type": "nominal", "label": "Cell state"}, "y": {"field": "uncertainty_rate", "type": "quantitative", "label": "Uncertainty rate", "format": "percent"}, "color": {"field": "branch", "type": "nominal", "label": "Branch"}, "tooltip": [{"field": "uncertain_count", "type": "quantitative", "label": "Uncertain", "format": "number"}]},
        },
        {
            "id": "stress_chart", "title": "Independent detector stress test", "subtitle": "Diagnostic detector only; these rates are not biological sensitivity.",
            "type": "bar", "dataset": "stress_metrics", "sourceId": "stress", "valueFormat": "percent", "options": {"grouping": "grouped"},
            "encodings": {"x": {"field": "test", "type": "nominal", "label": "Test"}, "y": {"field": "rate", "type": "quantitative", "label": "Recovery", "format": "percent"}, "color": {"field": "branch", "type": "nominal", "label": "Branch"}, "tooltip": [{"field": "recovered", "type": "quantitative", "label": "Recovered", "format": "number"}, {"field": "denominator", "type": "quantitative", "label": "Denominator", "format": "number"}]},
        },
    ]
    tables = [
        {"id": "audit_table", "title": "Automated operational audit", "subtitle": "Proxy metrics without manual ground truth.", "dataset": "audit_metrics", "sourceId": "audit", "columns": [
            {"field": "branch", "label": "Branch", "type": "text"}, {"field": "all_cells", "label": "All cells", "format": "number"},
            {"field": "strong_objects", "label": "Strong objects", "format": "number"}, {"field": "strong_objects_detected", "label": "Detected", "format": "number"},
            {"field": "operational_recall", "label": "Operational recall", "format": "percent"}, {"field": "possible_live_false_positives", "label": "Possible live FP", "format": "number"},
            {"field": "live_false_positive_upper_bound", "label": "Live FP upper bound", "format": "percent"}, {"field": "qc_complete", "label": "QC complete", "format": "number"},
        ]},
        {"id": "relation_table", "title": "Confirmed Death-object relation details", "subtitle": "Exact counts and branch-specific denominators.", "dataset": "death_relations", "sourceId": "annotations", "columns": [
            {"field": "branch", "label": "Branch", "type": "text"}, {"field": "relation", "label": "Relation", "type": "text"},
            {"field": "count", "label": "Objects", "format": "number"}, {"field": "confirmed_death_objects", "label": "Denominator", "format": "number"},
            {"field": "share_of_confirmed_death", "label": "Share", "format": "percent"},
        ]},
    ]
    cards = [
        {"id": "run_card", "description": "Validated d0 scope and final annotation ledger.", "dataset": "overview", "sourceId": "annotations", "metrics": [{"label": "Fields", "field": "selected_fields", "format": "number"}, {"label": "Annotation rows", "field": "annotation_rows", "format": "number"}]},
        {"id": "object_card", "description": "Upstream Dead objects retained in the auditable object ledger.", "dataset": "overview", "sourceId": "audit", "metrics": [{"label": "Segmented Dead objects", "field": "segmented_dead_objects", "format": "number"}, {"label": "Strong references", "field": "strong_reference_objects", "format": "number"}]},
        {"id": "qc_card", "description": "Branch-aware field QC inventory.", "dataset": "overview", "sourceId": "qc", "metrics": [{"label": "Complete branch-field QC", "field": "qc_fields", "format": "number"}, {"label": "Live FP proxies", "field": "possible_live_false_positives", "format": "number"}]},
    ]
    blocks: list[dict[str, Any]] = [
        {"id": "technical_summary", "type": "markdown", "body": "## Technical summary\n\nThe complete 320-field d0 cohort was classified in both the original and nucleated-only branches. Final cell states, independent Death-object relations, uncertainty annotations, automated operational proxies, detector stress tests, and per-field QC are reported from this run. The proxy audit is not manual biological ground truth."},
        {"id": "cards", "type": "metric-strip", "cardIds": ["run_card", "object_card", "qc_card"]},
        {"id": "classification_definition", "type": "markdown", "body": "## Classification keeps cell state and Death-object evidence as separate layers\n\nThe cell label belongs to the Combined-cell mask. The Death-object label belongs to an independently segmented Dead-channel mask. A live cell can therefore overlap a confirmed Death object without being forcibly relabeled dead. Nucleus multiplicity and cell-scale overlap distinguish same-cell death, multi-nucleus live/dead overlap, live-with-death-signal, dead-only, and merged-object relations."},
        {"id": "cell_state_text", "type": "markdown", "body": "## Final cell states are reported for both analysis branches\n\nThe chart uses all Combined masks as the denominator and keeps artifact, live, dead, transitional, and uncertain states explicit."},
        {"id": "cell_state_block", "type": "chart", "chartId": "cell_state_chart", "layout": "full"},
        {"id": "relation_text", "type": "markdown", "body": "## Confirmed Death objects are decomposed by their final relation to cells\n\nEach confirmed object contributes to one mutually exclusive relation. This separates same-cell death from overlapping or supplemental evidence without duplicating cell rows."},
        {"id": "relation_block", "type": "chart", "chartId": "death_relation_chart", "layout": "full"},
        {"id": "relation_table_block", "type": "table", "tableId": "relation_table", "layout": "full"},
        {"id": "event_text", "type": "markdown", "body": "## Final counts distinguish cell-associated death from supplemental objects\n\nThe object-aware event total combines dead-cell calls with confirmed supplemental objects associated with live overlap, dead-only regions, or merged-object evidence."},
        {"id": "event_block", "type": "chart", "chartId": "event_chart", "layout": "full"},
        {"id": "uncertainty_text", "type": "markdown", "body": "## Uncertainty remains visible within live and dead annotations\n\nMedium/low confidence and uncertain spatial attribution are counted inside each final state rather than hidden by the final label."},
        {"id": "uncertainty_block", "type": "chart", "chartId": "uncertainty_chart", "layout": "full"},
        {"id": "audit_text", "type": "markdown", "body": "## Automated audit measures operational consistency without manual ground truth\n\nStrong-object recovery and possible live-cell false positives are operational proxy metrics. Values of 100% or 0% do not establish biological sensitivity or specificity."},
        {"id": "audit_table_block", "type": "table", "tableId": "audit_table", "layout": "full"},
    ]
    for index, key in enumerate(SENTINEL_KEYS, 1):
        uri, width, height = image_data[key]
        blocks.extend([
            {"id": f"qc_{index}_text", "type": "markdown", "body": f"### QC example {index}: {key}\n\nThe same field is shown with final cell-state, Death-object, overlap-layer, and nucleated-only views for direct visual audit."},
            {"id": f"qc_{index}_image", "type": "html", "body": _image_body(uri, key, width, height), "layout": "full"},
        ])
    blocks.extend([
        {"id": "stress_text", "type": "markdown", "body": "## The independent global detector remains diagnostic only\n\nIts recovery rates quantify a deliberately separate stress test. Because recovery is incomplete, it is not connected to production classification and must not be interpreted as biological sensitivity."},
        {"id": "stress_block", "type": "chart", "chartId": "stress_chart", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": "## Limitations and next steps\n\nNo manual ground-truth table is available. All accuracy-like values are therefore operational proxies tied to existing masks, RGB context, nucleus evidence, and fixed thresholds. Review the complete per-field QC inventory when investigating individual calls, and use the multilevel annotation ledger for downstream cohort analysis."},
    ])
    return {"version": 1, "surface": "report", "title": "SUM159 d0 Dead-Classification Audit", "description": "Current-run technical audit of final d0 classification and Death-object attribution.", "generatedAt": generated_at, "cards": cards, "charts": charts, "tables": tables, "sources": [{"id": s["id"], "label": s["label"], "path": s["path"]} for s in sources], "blocks": blocks}


def _image_body(uri: str, key: str, width: int, height: int) -> str:
    import html
    return f'<style>html,body{{margin:0;padding:0;overflow:hidden}}figure{{margin:0}}img{{display:block;width:100%;height:auto}}figcaption{{font:12px/18px system-ui,sans-serif;margin-top:6px}}</style><figure><img src="{uri}" alt="{html.escape(key)} final classification QC" width="{width}" height="{height}"><figcaption>{html.escape(key)}: current-run final classification QC across both branches.</figcaption></figure>'


def build_current_report(args: Any, main_report: Any, plugin_root: Path, source_inventory: dict[str, Any] | None) -> dict[str, Any]:
    root = args.audit_root.expanduser().resolve()
    datasets = current_datasets(root)
    figures = {key: qc_figure(root, key) for key in SENTINEL_KEYS}
    sources = current_sources(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    image_data: dict[str, tuple[str, int, int]] = {}
    high_resolution_figures: dict[str, Image.Image] = {}
    for key, figure in figures.items():
        canonical = figure.resize((max(1, round(figure.width * 0.42)), max(1, round(figure.height * 0.42))), Image.Resampling.LANCZOS)
        uri, _bytes, _digest = main_report.encode_sheet(canonical, 74)
        image_data[key] = (uri, canonical.width, canonical.height)
        high_resolution_figures[f"qc_{SENTINEL_KEYS.index(key) + 1}"] = figure
    artifact = {"surface": "report", "manifest": current_manifest(generated_at, datasets, image_data, sources), "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": datasets}, "sources": sources, "package_info": {"originUrl": "artifact://dead-d0-classification-audit/current-run", "controls": {"edit": False, "refresh": False, "share": False}}}
    artifact_path = args.artifact_json.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    receipt_path = args.build_receipt.expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    main_report.atomic_write_json(artifact_path, artifact)
    high_resolution = main_report.build_high_resolution_image_payload(high_resolution_figures)
    enhancement = main_report.package_html(artifact_path, output_html, plugin_root, args.force, high_resolution)
    receipt = {"report_mode": "current_run", "report": str(output_html), "artifact_json": str(artifact_path), "audit_root": str(root), "source_inventory": source_inventory, "dataset_rows": {key: len(value) for key, value in datasets.items()}, "sentinel_fields": list(SENTINEL_KEYS), "html_enhancement": enhancement}
    main_report.atomic_write_json(receipt_path, receipt)
    return receipt
