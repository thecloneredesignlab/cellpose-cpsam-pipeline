from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cellpose_pipeline/scripts"
SELECT_UMAP = SCRIPTS / "56_select_multimodal_cell_state_v4_death_resolution.py"
SELECT_REVIEW = SCRIPTS / "57_build_multimodal_cell_state_v4_broad_region_review.py"
RENDER = SCRIPTS / "55_render_multimodal_cell_state_v4_review.py"
IMPORT = SCRIPTS / "58_import_multimodal_cell_state_v4_review.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV4WorkflowTests(unittest.TestCase):
    def run_ok(self, command: list[str]) -> str:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def build_fixture(self, root: Path) -> tuple[Path, Path, Path, list[dict[str, object]]]:
        shadow = root / "multimodal_cell_state_v4_test_fixture"
        project_dir = shadow / "projection_input" / "balanced"
        (project_dir / "projection").mkdir(parents=True)
        mask = np.arange(1, 9, dtype=np.uint16).reshape(2, 4)
        bf = (np.arange(8, dtype=np.uint16).reshape(2, 4) + 10) * 10
        dead = (np.asarray([5, 7, 10, 12, 100, 120, 140, 160], dtype=np.uint16).reshape(2, 4))
        nuclei = (np.asarray([120, 100, 80, 60, 20, 15, 10, 5], dtype=np.uint16).reshape(2, 4))
        for name, value in (("mask.tif", mask), ("bf.tif", bf), ("dead.tif", dead), ("nuclei.tif", nuclei)):
            tifffile.imwrite(project_dir / name, value)
        cells: list[dict[str, object]] = []
        coordinates: list[dict[str, object]] = []
        clusters: list[dict[str, object]] = []
        for index in range(8):
            well = f"W{index+1:02d}"
            cell_id = f"original|{well}_1_00d00h00m|{index+1}"
            cells.append({
                "cell_id": cell_id, "image_id": "image1", "mask_label": index + 1,
                "key": f"{well}_1_00d00h00m", "well": well, "site": 1,
                "elapsed_hours": 0, "context_key": "2N" if index < 4 else "4N",
                "source_id": well, "split": "development",
                "diagnostic_cluster": 1 if index < 4 else 2,
                "nuclei_measurement_status": "present_supported" if index % 2 else "absent_supported",
                "dead_measurement_status": "dead_signal_present_supported" if index >= 4 else "dead_signal_absent_supported",
            })
            coordinates.append({
                "cell_id": cell_id,
                "Dim1": 0.1 + 0.05 * index if index < 4 else 0.7 + 0.05 * (index - 4),
                "Dim2": 0.1 + 0.2 * (index % 4),
            })
            clusters.append({
                "cell_id": cell_id, "diagnostic_cluster": 1 if index < 4 else 2,
                "cluster_role": "navigation_only_not_cell_state_label",
            })
        write_tsv(project_dir / "cells.tsv", list(cells[0]), cells)
        write_tsv(project_dir / "features.tsv", ["cell_id", "feature1"], [
            {"cell_id": row["cell_id"], "feature1": index + 1} for index, row in enumerate(cells)
        ])
        image_fields = [
            "image_id", "channel_id", "image_path", "mask_path", "width", "height",
            "page_index", "channel_index", "mask_channel_index",
            "display_percentile_low", "display_percentile_high",
        ]
        image_rows = []
        for channel, filename in (("brightfield", "bf.tif"), ("dead", "dead.tif"), ("nuclei", "nuclei.tif")):
            image_rows.append({
                "image_id": "image1", "channel_id": channel,
                "image_path": str(project_dir / filename), "mask_path": str(project_dir / "mask.tif"),
                "width": 4, "height": 2, "page_index": 1, "channel_index": 1,
                "mask_channel_index": 1, "display_percentile_low": 1,
                "display_percentile_high": 99.8,
            })
        write_tsv(project_dir / "images.tsv", image_fields, image_rows)
        write_tsv(project_dir / "classes.tsv", ["class_id", "display_name", "color"], [
            {"class_id": "non_dead", "display_name": "Non-dead", "color": "#16875b"},
            {"class_id": "dead", "display_name": "Dead", "color": "#d43d51"},
        ])
        write_tsv(project_dir / "projection/umap.tsv", list(coordinates[0]), coordinates)
        write_tsv(project_dir / "projection/diagnostic_clusters.tsv", list(clusters[0]), clusters)
        project = {
            "schema_version": "cell_phenotype_annotator_project_v1",
            "project_id": "multimodal_cell_state_v4_development",
            "cells_file": "cells.tsv", "features_file": "features.tsv",
            "images_file": "images.tsv", "classes_file": "classes.tsv", "runs_dir": "runs",
            "projection": {"mode": "existing_umap", "coordinate_file": "projection/umap.tsv", "allow_subset": False},
            "annotation": {"title": "fixture", "direct_class_limit": 8, "point_radius": 1.5, "boundary_tolerance": 1e-10},
            "classifier": {
                "feature_columns": ["feature1"],
                "group_column": "source_id",
                "allow_ungrouped": False,
            },
        }
        (project_dir / "project.yml").write_text(json.dumps(project) + "\n")
        projection = shadow / "workflow_status/projection"
        profiles = [
            "v4_balanced_four_block", "v4_death_resolution_dead35_nuclei25",
            "v4_death_resolution_dead30_bf30",
        ]
        metric_rows = []
        for profile_index, profile in enumerate(profiles):
            run = projection / "profiles" / profile / "grid" / "nn15_md0p1_seed20260813"
            shifted = [dict(row, Dim1=float(row["Dim1"]) + profile_index * 0.001) for row in coordinates]
            write_tsv(run / "umap.tsv", list(shifted[0]), shifted)
            metric_rows.append({
                "profile": profile, "run_id": run.name, "n_neighbors": 15,
                "min_dist": 0.1, "seed": 20260813,
                "input_umap_neighbor_overlap": 0.9,
                "mean_seed_neighbor_jaccard": 0.9 - profile_index * 0.01,
                "selection_score": 0.9,
            })
        write_tsv(projection / "umap_grid_metrics.tsv", list(metric_rows[0]), metric_rows)
        (projection / "projection_manifest.json").write_text(json.dumps({
            "schema_version": "multimodal_cell_state_v4_projection_v1", "status": "COMPLETE",
            "selected_feature_columns": ["feature1"],
        }) + "\n")
        anchor_import = shadow / "human_review/anchor_seed1/import"
        anchor_rows = []
        for index, cell in enumerate(cells):
            primary = "non_dead" if index < 4 else "dead"
            anchor_rows.append({
                "cell_id": cell["cell_id"], "primary_death_label": primary,
                "death_stage_label": "not_applicable_non_dead" if primary == "non_dead" else "dead_marker_positive",
                "nuclear_state_label": "single_nucleus", "label_confidence": "high",
                "reviewer": "fixture-reviewer", "reviewed_at": "2026-08-13T12:00:00-04:00",
                "review_notes": "", "training_eligible": "true",
            })
        write_tsv(anchor_import / "reviewed_labels.tsv", list(anchor_rows[0]), anchor_rows)
        (anchor_import / "review_import_manifest.json").write_text(json.dumps({
            "schema_version": "multimodal_cell_state_v4_review_import_v1", "status": "COMPLETE",
            "output_file_sha256": {"reviewed_labels.tsv": sha(anchor_import / "reviewed_labels.tsv")},
        }) + "\n")
        return shadow, project_dir / "project.yml", projection, cells

    def test_anchor_selects_death_umap_then_region_review_imports_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow, balanced_project, projection, cells = self.build_fixture(root)
            death_project = shadow / "projection_input/multimodal_cell_state_v4_death_resolution"
            created = self.run_ok([
                sys.executable, str(SELECT_UMAP), "--balanced-project", str(balanced_project),
                "--projection-root", str(projection), "--anchor-import-dir",
                str(shadow / "human_review/anchor_seed1/import"), "--shadow-root", str(shadow),
                "--output-dir", str(death_project), "--minimum-per-class", "1", "--neighbors", "1",
            ])
            self.assertIn("selected_profile=", created)
            self.assertTrue((death_project / "project.yml").is_file())
            self.assertFalse((death_project / "runs").exists())
            project_value = json.loads((death_project / "project.yml").read_text())
            self.assertEqual(project_value["project_id"], "multimodal_cell_state_v4_death_resolution")
            self.assertEqual(project_value["classifier"]["group_column"], "source_id")
            self.assertFalse(project_value["classifier"]["allow_ungrouped"])

            annotation = shadow / "annotation_import"
            annotation.mkdir()
            polygon = [
                {"x": 0.55, "y": -0.1}, {"x": 1.0, "y": -0.1},
                {"x": 1.0, "y": 1.0}, {"x": 0.55, "y": 1.0},
            ]
            region_payload = {"regions": [{
                "region_id": "dead-region", "assignment_state": "class_assigned",
                "class_id": "dead", "polygon": polygon,
            }]}
            (annotation / "regions.json").write_text(json.dumps(region_payload) + "\n")
            (annotation / "region_submission.json").write_text(json.dumps(region_payload) + "\n")
            label_rows = []
            for index, cell in enumerate(cells):
                inside = index >= 4
                label_rows.append({
                    "cell_id": cell["cell_id"],
                    "assignment_state": "class_assigned" if inside else "unreviewed",
                    "class_id": "dead" if inside else "", "region_id": "dead-region" if inside else "",
                })
            write_tsv(annotation / "provisional_labels.tsv", list(label_rows[0]), label_rows)
            artifacts = {
                "region_submission": sha(annotation / "region_submission.json"),
                "regions": sha(annotation / "regions.json"),
                "provisional_labels": sha(annotation / "provisional_labels.tsv"),
            }
            (annotation / "annotation_import_manifest.json").write_text(json.dumps({
                "schema_version": "cell_phenotype_annotator_annotation_import_v1",
                "artifact_file_sha256": artifacts,
            }) + "\n")
            selection = shadow / "human_review/broad_region/selection"
            self.run_ok([
                sys.executable, str(SELECT_REVIEW), "--project", str(death_project / "project.yml"),
                "--annotation-import-dir", str(annotation), "--shadow-root", str(shadow),
                "--output-dir", str(selection), "--max-total", "4", "--max-per-well", "1",
                "--minimum-per-boundary-side", "1",
            ])
            selected = read_tsv(selection / "broad_region_review_set.tsv")
            self.assertEqual(len(selected), 4)
            self.assertEqual({row["review_default_label"] for row in selected}, {"uncertain_or_unreviewable"})
            self.assertEqual(len({row["review_sampling_bucket"] for row in selected}), 4)
            render = shadow / "human_review/broad_region/render"
            self.run_ok([
                sys.executable, str(RENDER), "--project", str(death_project / "project.yml"),
                "--review-set", str(selection / "broad_region_review_set.tsv"),
                "--review-manifest", str(selection / "broad_region_review_manifest.json"),
                "--output-dir", str(render), "--padding", "1",
            ])
            html = (render / "exact_review.html").read_text()
            self.assertIn("Brightfield", html); self.assertIn("Dead fluorescence", html); self.assertIn("Nuclei", html)
            self.assertNotIn("death_region_boundary_inside |", html)
            self.assertNotIn(selected[0]["morphology_umap_row_key"], html)
            render_manifest = json.loads((render / "exact_review_render_manifest.json").read_text())
            submission_rows = []
            render_generation_id = render_manifest["identity"]["render_generation_id"]
            for index, row in enumerate(selected):
                primary = "dead" if index % 2 else "non_dead"
                submission_rows.append({
                    "review_token": hashlib.sha256(
                        f"{render_generation_id}|{row['morphology_umap_row_key']}".encode()
                    ).hexdigest()[:24],
                    "primary_death_label": primary,
                    "death_stage_label": "dead_marker_positive" if primary == "dead" else "not_applicable_non_dead",
                    "nuclear_state_label": "single_nucleus", "label_confidence": "high",
                    "reviewer": "fixture-reviewer", "reviewed_at": "2026-08-13T12:30:00-04:00",
                    "review_notes": "",
                })
            submission = render / "multimodal_v4_review_submission.json"
            submission.write_text(json.dumps({
                "schema_version": "multimodal_cell_state_v4_review_submission_v1",
                "identity": render_manifest["identity"], "creation_metadata": {
                    "created_at": "2026-08-13T12:30:00-04:00", "client_version": "fixture",
                }, "rows": submission_rows,
            }) + "\n")
            imported = shadow / "human_review/broad_region/import"
            self.run_ok([
                sys.executable, str(IMPORT), "--project", str(death_project / "project.yml"),
                "--review-set", str(selection / "broad_region_review_set.tsv"),
                "--review-manifest", str(selection / "broad_region_review_manifest.json"),
                "--render-dir", str(render), "--submission", str(submission),
                "--output-dir", str(imported),
            ])
            imported_rows = read_tsv(imported / "reviewed_labels.tsv")
            self.assertEqual(len(imported_rows), 4)
            self.assertTrue(all(row["training_eligible"] == "true" for row in imported_rows))
            reused = self.run_ok([
                sys.executable, str(IMPORT), "--project", str(death_project / "project.yml"),
                "--review-set", str(selection / "broad_region_review_set.tsv"),
                "--review-manifest", str(selection / "broad_region_review_manifest.json"),
                "--render-dir", str(render), "--submission", str(submission),
                "--output-dir", str(imported),
            ])
            self.assertIn("verified_reuse=1", reused)


if __name__ == "__main__":
    unittest.main()
