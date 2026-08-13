from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "cellpose_pipeline" / "scripts"
CONFIG = ROOT / "cellpose_pipeline" / "configs" / "multimodal_cell_state_v3.json"
EXTRACTOR = SCRIPTS / "41_extract_multimodal_cell_state_v3_features.py"
HELPER = SCRIPTS / "_shared" / "multimodal_cell_state_v3_features.py"
MERGER = SCRIPTS / "42_merge_multimodal_cell_state_v3_features.py"
PROJECT_BUILDER = SCRIPTS / "44_prepare_multimodal_cell_state_v3_project.py"
AUDIT = SCRIPTS / "45_render_multimodal_cell_state_v3_audit.py"
BLIND = SCRIPTS / "46_build_multimodal_cell_state_v3_blind_review.py"
RENDER = SCRIPTS / "37_render_reference_cell_state_exact_review.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV3WorkflowTests(unittest.TestCase):
    def run_ok(self, command: list[str]) -> str:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_merge_project_audit_and_blind_review_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            shadow = root / "multimodal_cell_state_v3_test_fixture"
            base_project_dir = base / "projection_input" / "base"
            base_project_dir.mkdir(parents=True)
            (shadow / "workflow_status").mkdir(parents=True)
            mask = np.arange(1, 9, dtype=np.uint16).reshape(2, 4)
            brightfield = np.arange(8, dtype=np.uint16).reshape(2, 4) + 10
            nuclei = np.arange(8, dtype=np.uint16).reshape(2, 4) + 30
            for name, array in (
                ("mask.tif", mask),
                ("bf.tif", brightfield),
                ("nuclei.tif", nuclei),
            ):
                tifffile.imwrite(base_project_dir / name, array)
            cells: list[dict[str, str]] = []
            for index in range(8):
                well = f"W{index // 2 + 1:02d}"
                key = f"{well}_1_00d00h{index:02d}m"
                cells.append(
                    {
                        "cell_id": f"original|{key}|{index+1}",
                        "image_id": "image1",
                        "mask_label": str(index + 1),
                        "key": key,
                        "well": well,
                        "site": "1",
                        "elapsed_hours": str(index),
                        "context_key": "2N" if index % 2 == 0 else "4N",
                        "source_id": well,
                        "split": "development",
                    }
                )
            write_tsv(base_project_dir / "cells.tsv", list(cells[0]), cells)
            image_fields = [
                "image_id",
                "channel_id",
                "image_path",
                "mask_path",
                "width",
                "height",
                "page_index",
                "channel_index",
                "mask_channel_index",
                "display_percentile_low",
                "display_percentile_high",
            ]
            image_rows = []
            for channel, filename in (
                ("brightfield", "bf.tif"),
                ("nuclei", "nuclei.tif"),
            ):
                image_rows.append(
                    {
                        "image_id": "image1",
                        "channel_id": channel,
                        "image_path": str(base_project_dir / filename),
                        "mask_path": str(base_project_dir / "mask.tif"),
                        "width": "4",
                        "height": "2",
                        "page_index": "1",
                        "channel_index": "1",
                        "mask_channel_index": "1",
                        "display_percentile_low": "1",
                        "display_percentile_high": "99",
                    }
                )
            write_tsv(base_project_dir / "images.tsv", image_fields, image_rows)
            class_rows = [
                {"class_id": "live_cell", "display_name": "Live", "color": "#16875b"},
                {"class_id": "dead_cell", "display_name": "Dead", "color": "#d43d51"},
                {
                    "class_id": "multinucleated_cell",
                    "display_name": "Multi",
                    "color": "#7257c8",
                },
            ]
            write_tsv(base_project_dir / "classes.tsv", list(class_rows[0]), class_rows)
            base_project = {
                "schema_version": "cell_phenotype_annotator_project_v1",
                "project_id": "base",
                "cells_file": "cells.tsv",
                "images_file": "images.tsv",
                "classes_file": "classes.tsv",
            }
            (base_project_dir / "project.yml").write_text(
                json.dumps(base_project) + "\n", encoding="utf-8"
            )

            feature_root = shadow / "workflow_status" / "fields"
            feature_fields = [
                "cell_id",
                "key",
                "well",
                "source_id",
                "mask_label",
                "branch",
                "area_px2",
                "bf_object_median",
                "nucleus_absent",
                "nuclei_measurement_status",
            ]
            for index, cell in enumerate(cells):
                generation = feature_root / cell["well"] / cell["key"]
                generation.mkdir(parents=True)
                row = {
                    **{
                        name: cell[name]
                        for name in (
                            "cell_id",
                            "key",
                            "well",
                            "source_id",
                            "mask_label",
                        )
                    },
                    "branch": "original",
                    "area_px2": str(10 + index),
                    "bf_object_median": str(20 + index),
                    "nucleus_absent": str(index % 2),
                    "nuclei_measurement_status": (
                        "absent_supported" if index % 2 else "present_supported"
                    ),
                }
                write_tsv(generation / "features.tsv", feature_fields, [row])
                receipt = {
                    "schema_version": "multimodal_cell_state_v3_feature_generation_v1",
                    "status": "COMPLETE",
                    "key": cell["key"],
                    "well": cell["well"],
                    "row_count": 1,
                    "inputs": {
                        "config": {"sha256": sha(CONFIG)},
                        "implementation": {"sha256": sha(EXTRACTOR)},
                        "feature_helper": {"sha256": sha(HELPER)},
                    },
                    "output_file_sha256": {
                        "features.tsv": sha(generation / "features.tsv")
                    },
                }
                (generation / "feature_receipt.json").write_text(
                    json.dumps(receipt) + "\n", encoding="utf-8"
                )

            merge = shadow / "workflow_status" / "merge"
            merge_command = [
                sys.executable,
                str(MERGER),
                "--project",
                str(base_project_dir / "project.yml"),
                "--feature-root",
                str(feature_root),
                "--feature-config",
                str(CONFIG),
                "--field-extractor",
                str(EXTRACTOR),
                "--feature-helper",
                str(HELPER),
                "--output-dir",
                str(merge),
                "--expected-cell-count",
                "8",
                "--overwrite",
            ]
            self.assertIn("generation_status=created", self.run_ok(merge_command))
            self.assertIn(
                "generation_status=verified_reuse", self.run_ok(merge_command)
            )

            projection = shadow / "workflow_status" / "projection"
            projection.mkdir()
            umap_rows = [
                {
                    "cell_id": cell["cell_id"],
                    "Dim1": str(index % 4),
                    "Dim2": str(index // 4),
                }
                for index, cell in enumerate(cells)
            ]
            cluster_rows = [
                {
                    "cell_id": cell["cell_id"],
                    "diagnostic_cluster": str(index % 2),
                    "cluster_role": "navigation_only_not_cell_state_label",
                }
                for index, cell in enumerate(cells)
            ]
            write_tsv(projection / "umap.tsv", list(umap_rows[0]), umap_rows)
            write_tsv(
                projection / "diagnostic_clusters.tsv",
                list(cluster_rows[0]),
                cluster_rows,
            )
            decision = {
                "schema_version": "multimodal_cell_state_v3_calibration_decision_v1",
                "status": "COMPLETE",
                "representation_gate": "GO_TO_BLIND_LABEL_MAPPING",
                "dead_cell_island_required": False,
            }
            (projection / "calibration_decision.json").write_text(
                json.dumps(decision) + "\n", encoding="utf-8"
            )
            projection_outputs = {
                name: sha(projection / name)
                for name in (
                    "umap.tsv",
                    "diagnostic_clusters.tsv",
                    "calibration_decision.json",
                )
            }
            projection_manifest = {
                "schema_version": "multimodal_cell_state_v3_projection_v1",
                "status": "COMPLETE",
                "selected_run_id": "fixture",
                "selected_feature_columns": [
                    "area_px2",
                    "bf_object_median",
                    "nucleus_absent",
                ],
                "output_file_sha256": projection_outputs,
            }
            (projection / "projection_manifest.json").write_text(
                json.dumps(projection_manifest) + "\n", encoding="utf-8"
            )

            project_dir = shadow / "projection_input" / "multimodal_cell_state_v3"
            project_command = [
                sys.executable,
                str(PROJECT_BUILDER),
                "--base-project",
                str(base_project_dir / "project.yml"),
                "--base-shadow-root",
                str(base),
                "--feature-merge",
                str(merge),
                "--projection",
                str(projection),
                "--shadow-root",
                str(shadow),
                "--output-dir",
                str(project_dir),
                "--expected-cell-count",
                "8",
                "--overwrite",
            ]
            self.assertIn("generation_status=created", self.run_ok(project_command))
            self.assertIn(
                "generation_status=verified_reuse", self.run_ok(project_command)
            )

            audit = shadow / "morphology_audit"
            audit_command = [
                sys.executable,
                str(AUDIT),
                "--project",
                str(project_dir / "project.yml"),
                "--shadow-root",
                str(shadow),
                "--output-dir",
                str(audit),
                "--max-representatives",
                "4",
                "--overlay-width",
                "400",
                "--overlay-height",
                "400",
                "--overlay-tile-px",
                "10",
                "--overwrite",
            ]
            self.assertIn("generation_status=created", self.run_ok(audit_command))
            self.assertIn(
                "generation_status=verified_reuse", self.run_ok(audit_command)
            )

            blind = shadow / "human_review" / "blind_seed1" / "selection"
            blind_command = [
                sys.executable,
                str(BLIND),
                "--project",
                str(project_dir / "project.yml"),
                "--shadow-root",
                str(shadow),
                "--output-dir",
                str(blind),
                "--max-total",
                "4",
                "--max-per-well",
                "1",
                "--overwrite",
            ]
            self.assertIn("generation_status=created", self.run_ok(blind_command))
            selected = read_tsv(blind / "blind_review_set.tsv")
            self.assertEqual(len(selected), 4)
            self.assertEqual(
                max(Counter(row["source_id"] for row in selected).values()), 1
            )
            self.assertTrue(
                all(row["review_default_label"] == "uncertain" for row in selected)
            )
            self.assertIn(
                "generation_status=verified_reuse", self.run_ok(blind_command)
            )

            render = shadow / "human_review" / "blind_seed1" / "render"
            render_command = [
                sys.executable,
                str(RENDER),
                "--project",
                str(project_dir / "project.yml"),
                "--review-set",
                str(blind / "blind_review_set.tsv"),
                "--review-manifest",
                str(blind / "blind_review_manifest.json"),
                "--output-dir",
                str(render),
                "--padding",
                "1",
            ]
            self.assertIn("rows=4", self.run_ok(render_command))
            self.assertIn("verified_reuse=1", self.run_ok(render_command))
            manifest = json.loads(
                (render / "exact_review_render_manifest.json").read_text()
            )
            self.assertTrue(manifest["all_crops_available"])
            self.assertEqual(manifest["identity"]["row_count"], 4)

            with (blind / "blind_review_set.tsv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            rejected = subprocess.run(
                blind_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact set/hash differs", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
