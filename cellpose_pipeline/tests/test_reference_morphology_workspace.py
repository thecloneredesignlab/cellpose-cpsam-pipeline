from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cellpose_pipeline" / "scripts" / "27_build_reference_morphology_workspace.py"
SPEC = importlib.util.spec_from_file_location("reference_morphology_workspace_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WORKSPACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKSPACE)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class ReferenceMorphologyWorkspaceTests(unittest.TestCase):
    def make_fixture(self, root: Path, v2: bool = False) -> dict[str, Path]:
        shadow = root / "shadow"
        project_root = shadow / "reference_project"
        source = shadow / "source"
        annotation = project_root / "runs" / "reference" / "run_1" / "projection" / "projection_1" / "annotation" / "annotation_1"
        project_root.mkdir(parents=True)
        annotation.mkdir(parents=True)

        cell_rows: list[dict[str, object]] = []
        image_rows: list[dict[str, object]] = []
        cell_ids: list[str] = []
        dim1: list[float] = []
        dim2: list[float] = []
        metadata: dict[str, list[object]] = {
            "image_id": [],
            "mask_label": [],
            "well": [],
            "split": [],
        }
        if v2:
            metadata["context_key"] = []
            metadata["source_id"] = []
            metadata["cluster"] = []
        fixture_wells = ("A01", "B01", "C01", "D01") if v2 else (
            "A01",
            "B01",
            "C01",
        )
        for well_index, well in enumerate(fixture_wells):
            image_id = f"{well}_1_00d00h00m"
            field = source / image_id
            field.mkdir(parents=True)
            yy, xx = np.mgrid[0:48, 0:48]
            brightfield = (xx * 23 + yy * 7 + well_index * 101).astype(np.uint16)
            nuclei = np.zeros((48, 48), dtype=np.uint16)
            mask = np.zeros((48, 48), dtype=np.uint16)
            boxes = ((4, 4), (27, 4), (4, 27), (27, 27))
            for label, (x0, y0) in enumerate(boxes, start=1):
                mask[y0 : y0 + 12, x0 : x0 + 12] = label
                nuclei[y0 + 3 : y0 + 6, x0 + 3 : x0 + 6] = 500 + label
                cell_id = f"original|{image_id}|{label}"
                cell_ids.append(cell_id)
                dim1.append(float(well_index * 10 + (label % 2) * 2))
                dim2.append(float((label // 3) * 4 + well_index))
                metadata["image_id"].append(image_id)
                metadata["mask_label"].append(str(label))
                metadata["well"].append(well)
                metadata["split"].append("development")
                context_key = (
                    "SUM-159-NLS-2N" if well_index < 2 else "SUM-159-NLS-4N"
                )
                cluster = str(1 + ((label - 1) % 2))
                if v2:
                    metadata["context_key"].append(context_key)
                    metadata["source_id"].append(well)
                    metadata["cluster"].append(cluster)
                cell_rows.append(
                    {
                        "cell_id": cell_id,
                        "image_id": image_id,
                        "mask_label": label,
                        "well": well,
                        "split": "development",
                        **(
                            {
                                "context_key": context_key,
                                "source_id": well,
                                "cluster": cluster,
                            }
                            if v2
                            else {}
                        ),
                    }
                )
            bf_path = field / "brightfield.tif"
            nuclei_path = field / "nuclei.tif"
            mask_path = field / "combined_mask.tif"
            # Mirror Incucyte exports: the full-resolution plane is the first
            # IFD and a one-pixel auxiliary image follows it.
            with tifffile.TiffWriter(bf_path) as writer:
                writer.write(brightfield)
                writer.write(np.zeros((1, 1), dtype=brightfield.dtype))
            with tifffile.TiffWriter(nuclei_path) as writer:
                writer.write(nuclei)
                writer.write(np.zeros((1, 1), dtype=nuclei.dtype))
            tifffile.imwrite(mask_path, mask)
            for channel, image_path, color in (
                ("brightfield", bf_path, "#ffffff"),
                ("nuclei", nuclei_path, "#00ffff"),
            ):
                image_rows.append(
                    {
                        "image_id": image_id,
                        "channel_id": channel,
                        "image_path": image_path,
                        "channel_index": "",
                        "page_index": 1,
                        "display_name": channel.title(),
                        "display_color": color,
                        "display_percentile_low": 1,
                        "display_percentile_high": 99,
                        "mask_path": mask_path,
                        "mask_channel_index": 1,
                        "width": 48,
                        "height": 48,
                        "alpha_policy": "reject",
                    }
                )

        cells = project_root / "cells.tsv"
        images = project_root / "images.tsv"
        write_tsv(
            cells,
            [
                "cell_id",
                "image_id",
                "mask_label",
                "well",
                "split",
                *(["context_key", "source_id", "cluster"] if v2 else []),
            ],
            cell_rows,
        )
        image_fields = [
            "image_id",
            "channel_id",
            "image_path",
            "channel_index",
            "page_index",
            "display_name",
            "display_color",
            "display_percentile_low",
            "display_percentile_high",
            "mask_path",
            "mask_channel_index",
            "width",
            "height",
            "alpha_policy",
        ]
        write_tsv(images, image_fields, image_rows)
        project = project_root / "project.yml"
        project.write_text(
            json.dumps(
                {
                    "schema_version": WORKSPACE.PROJECT_SCHEMA_VERSION,
                    "project_id": (
                        "reference_cell_state_development_v2"
                        if v2
                        else "reference_development"
                    ),
                    "cells_file": "cells.tsv",
                    "images_file": "images.tsv",
                }
            )
            + "\n"
        )
        if v2:
            historical_root = project_root / "historical_projection"
            historical_root.mkdir()
            historical_representatives = historical_root / "historical_representatives.tsv"
            frozen_rows = [
                {
                    "selection_rank": index,
                    "cell_id": cell_id,
                    "context_key": cell_rows[index - 1]["context_key"],
                    "source_id": cell_rows[index - 1]["source_id"],
                    "cluster": cell_rows[index - 1]["cluster"],
                    "Dim1": dim1[index - 1],
                    "Dim2": dim2[index - 1],
                }
                for index, cell_id in enumerate(cell_ids, start=1)
            ]
            write_tsv(
                historical_representatives,
                [
                    "selection_rank",
                    "cell_id",
                    "context_key",
                    "source_id",
                    "cluster",
                    "Dim1",
                    "Dim2",
                ],
                frozen_rows,
            )
            representative_sha256 = WORKSPACE.sha256_file(historical_representatives)
            (historical_root / "historical_projection_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "reference_cell_state_historical_projection_v2",
                        "status": "COMPLETE",
                        "representative_selection": {
                            "role": "authoritative_rendering_cell_list",
                            "outer_function_name": "get_all_cell_lines_overlay_representatives",
                            "inner_function_name": "get_spatially_uniform_representatives",
                            "seed": 1,
                            "total_n": 300,
                            "balance": 0.2,
                            "minimum_cluster_representatives": 2,
                            "selected_count": len(frozen_rows),
                            "output_file": "historical_representatives.tsv",
                            "output_sha256": representative_sha256,
                        },
                        "output_file_sha256": {
                            "historical_representatives.tsv": representative_sha256
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        identity = {
            "project_id": (
                "reference_cell_state_development_v2"
                if v2
                else "reference_development"
            ),
            "run_id": "run_1",
            "projection_id": "projection_1",
            "annotation_id": "annotation_1",
            "class_config_sha256": "1" * 64,
            "row_universe_sha256": "2" * 64,
            "coordinate_sha256": "3" * 64,
        }
        payload = annotation / "annotation_payload.json"
        payload.write_text(
            json.dumps(
                {
                    "schema_version": "cell_phenotype_annotator_annotation_v1",
                    "identity": identity,
                    "points": {
                        "cell_id": cell_ids,
                        "Dim1": dim1,
                        "Dim2": dim2,
                        "metadata": metadata,
                    },
                    "classes": [],
                },
                sort_keys=True,
            )
            + "\n"
        )
        annotation_html = annotation / "annotation.html"
        annotation_html.write_text("<!doctype html><title>immutable CPA editor</title>\n")
        manifest = {
            "schema_version": "cell_phenotype_annotator_annotation_v1",
            **identity,
            "row_count": len(cell_ids),
            "artifact_file_sha256": {
                "annotation_html": WORKSPACE.sha256_file(annotation_html),
                "annotation_payload": WORKSPACE.sha256_file(payload),
            },
        }
        (annotation / "annotation_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n"
        )
        return {
            "shadow": shadow,
            "project": project,
            "annotation": annotation,
            "images": images,
        }

    def argv(self, fixture: dict[str, Path], output: Path) -> list[str]:
        return [
            "--project",
            str(fixture["project"]),
            "--shadow-root",
            str(fixture["shadow"]),
            "--annotation-dir",
            str(fixture["annotation"]),
            "--output-dir",
            str(output),
            "--max-representatives",
            "6",
            "--seed",
            "20260812",
            "--overlay-width",
            "600",
            "--overlay-height",
            "450",
            "--overlay-tile-px",
            "20",
        ]

    def test_reference_style_workspace_is_balanced_blinded_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            output = fixture["shadow"] / "morphology_workspace"
            self.assertEqual(WORKSPACE.main(self.argv(fixture, output)), 0)
            required = {
                "representative_cells.tsv",
                "crops",
                "umap_morphology_overlay.png",
                "morphology_atlas.html",
                "overlay_manifest.json",
                "annotation_workspace.html",
            }
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            rows = read_tsv(output / "representative_cells.tsv")
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {well: sum(row["well"] == well for row in rows) for well in ("A01", "B01", "C01")},
                {"A01": 2, "B01": 2, "C01": 2},
            )
            self.assertEqual(len({row["cell_id"] for row in rows}), 6)
            first_crop = Image.open(output / rows[0]["brightfield_crop"]).convert("RGBA")
            self.assertEqual(first_crop.getpixel((0, 0))[3], 0)
            self.assertGreater(int(np.max(np.asarray(first_crop)[..., 3])), 0)
            with Image.open(output / "umap_morphology_overlay.png") as overlay:
                self.assertEqual(overlay.mode, "RGB")

            manifest = json.loads((output / "overlay_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["selection"]["selected_count"], 6)
            self.assertEqual(manifest["selection"]["well_count"], 3)
            self.assertEqual(
                manifest["blinding_contract"],
                {
                    "combined_rgb": "not_read",
                    "current_classification": "not_read",
                    "dead_channel": "not_read",
                    "forbidden_input_count": 0,
                    "provisional_annotation_labels": "not_read",
                },
            )
            self.assertEqual(
                {asset["role"] for asset in manifest["inputs"]["source_assets"]},
                {"brightfield_raw", "nuclei_raw", "combined_mask"},
            )
            for relative, expected in manifest["output_artifact_sha256"].items():
                self.assertEqual(WORKSPACE.sha256_file(output / relative), expected)
            workspace_html = (output / "annotation_workspace.html").read_text()
            self.assertIn("Immutable CPA polygon editor", workspace_html)
            self.assertIn("umap_morphology_overlay.png", workspace_html)
            first_hashes = WORKSPACE.directory_hashes(output)
            self.assertEqual(WORKSPACE.main([*self.argv(fixture, output), "--overwrite"]), 0)
            self.assertEqual(first_hashes, WORKSPACE.directory_hashes(output))

    def test_annotation_hash_and_forbidden_channel_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            payload = fixture["annotation"] / "annotation_payload.json"
            payload.write_text(payload.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                WORKSPACE.main(self.argv(fixture, fixture["shadow"] / "bad_hash"))

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            fields, rows = WORKSPACE.read_tsv(fixture["images"])
            forbidden = dict(rows[0])
            forbidden["channel_id"] = "dead"
            write_tsv(fixture["images"], fields, [*rows, forbidden])
            with self.assertRaisesRegex(ValueError, "only Brightfield and Nuclei"):
                WORKSPACE.main(self.argv(fixture, fixture["shadow"] / "bad_channel"))

    def test_current_state_metadata_is_rejected_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            cells = fixture["project"].parent / "cells.tsv"
            fields, rows = WORKSPACE.read_tsv(cells)
            for row in rows:
                row["final_state"] = "live"
            write_tsv(cells, [*fields, "final_state"], rows)
            with self.assertRaisesRegex(ValueError, "Forbidden current-classification"):
                WORKSPACE.main(self.argv(fixture, fixture["shadow"] / "state_leak"))

    def test_cap_smaller_than_well_count_does_not_overselect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            output = fixture["shadow"] / "two_representatives"
            argv = self.argv(fixture, output)
            argv[argv.index("--max-representatives") + 1] = "2"
            self.assertEqual(WORKSPACE.main(argv), 0)
            rows = read_tsv(output / "representative_cells.tsv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["well"] for row in rows}), 2)

    def test_v2_uses_context_cluster_quota_kmeans_and_cluster_coloring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary), v2=True)
            output = fixture["shadow"] / "morphology_workspace_v2"
            argv = self.argv(fixture, output)
            argv[argv.index("--seed") + 1] = "1"
            argv[argv.index("--max-representatives") + 1] = "300"
            argv.extend(
                [
                    "--cluster-balance",
                    "0.2",
                    "--minimum-cluster-representatives",
                    "2",
                ]
            )
            self.assertEqual(WORKSPACE.main(argv), 0)
            rows = read_tsv(output / "representative_cells.tsv")
            self.assertEqual(len(rows), 16)
            self.assertEqual({row["cluster"] for row in rows}, {"1", "2"})
            self.assertTrue((output / "cluster_selection_audit.tsv").is_file())
            audit = read_tsv(output / "cluster_selection_audit.tsv")
            self.assertEqual(
                {row["scope"] for row in audit},
                {"context:SUM-159-NLS-2N", "context:SUM-159-NLS-4N"},
            )
            manifest = json.loads((output / "overlay_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], WORKSPACE.SCHEMA_VERSION_V2)
            self.assertEqual(manifest["selection"]["seed"], 1)
            self.assertEqual(manifest["selection"]["context_count_expected"], 2)
            self.assertEqual(manifest["selection"]["cluster_balance"], 0.2)
            self.assertEqual(
                manifest["selection"]["selection_implementation"],
                "pinned_reference_R_AST",
            )
            self.assertNotIn("kmeans_rng_adaptation", manifest["selection"])
            self.assertEqual(
                manifest["inputs"]["implementation"]["sha256"],
                WORKSPACE.sha256_file(WORKSPACE.Path(WORKSPACE.__file__).resolve()),
            )
            self.assertEqual(
                manifest["render"]["background_point_color"],
                "diagnostic_cluster_palette_noise_zero_grey",
            )
            first_hashes = WORKSPACE.directory_hashes(output)
            self.assertEqual(WORKSPACE.main([*argv, "--overwrite"]), 0)
            self.assertEqual(first_hashes, WORKSPACE.directory_hashes(output))


if __name__ == "__main__":
    unittest.main()
