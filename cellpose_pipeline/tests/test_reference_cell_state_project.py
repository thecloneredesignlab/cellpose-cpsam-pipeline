from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cellpose_pipeline" / "scripts" / "26_prepare_reference_cell_state_project.py"
SPEC = importlib.util.spec_from_file_location("reference_cell_state_project_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


class ReferenceCellStateProjectTests(unittest.TestCase):
    def make_parent(self, root: Path, count: int = 4) -> dict[str, Path]:
        parent = root / "broad_parent"
        project_dir = parent / "projection_input" / "representative_umap"
        project_dir.mkdir(parents=True)
        (project_dir / "runs").mkdir()
        cells: list[dict[str, object]] = []
        features: list[dict[str, object]] = []
        for index in range(1, count + 1):
            well = "A01" if index <= count // 2 else "A02"
            key = f"{well}_1_0d0h0m"
            cell_id = f"original|{key}|{index}"
            cells.append(
                {
                    "cell_id": cell_id,
                    "image_id": key,
                    "mask_label": index,
                    "branch": "original",
                    "key": key,
                    "well": well,
                    "site": 1,
                    "day": 0,
                    "hour": 0,
                    "minute": 0,
                    "elapsed_hours": 0,
                    "split": "development",
                }
            )
            row: dict[str, object] = {"cell_id": cell_id, "bf_object_mean": index / 10}
            row.update(
                {
                    feature: index + feature_index / 10
                    for feature_index, feature in enumerate(PREPARE.REFERENCE_FEATURES, 1)
                }
            )
            features.append(row)
        write_tsv(project_dir / "cells.tsv", list(cells[0]), cells)
        write_tsv(
            project_dir / "features.tsv",
            ["cell_id", "bf_object_mean", *PREPARE.REFERENCE_FEATURES],
            features,
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
        image_rows: list[dict[str, object]] = []
        for image_id in sorted({str(row["image_id"]) for row in cells}):
            for channel_id in ("brightfield", "nuclei", "dead"):
                image_rows.append(
                    {
                        "image_id": image_id,
                        "channel_id": channel_id,
                        "image_path": f"/read_only/{image_id}_{channel_id}.tif",
                        "channel_index": "",
                        "page_index": "",
                        "display_name": channel_id.title(),
                        "display_color": "#ffffff",
                        "display_percentile_low": 1,
                        "display_percentile_high": 99,
                        "mask_path": f"/read_only/{image_id}_mask.tif",
                        "mask_channel_index": 1,
                        "width": 32,
                        "height": 32,
                        "alpha_policy": "reject",
                    }
                )
        write_tsv(project_dir / "images.tsv", image_fields, image_rows)
        write_tsv(
            project_dir / "classes.tsv",
            ["class_id", "display_name", "color", "trainable", "role", "order"],
            [
                {
                    "class_id": "old_broad_class",
                    "display_name": "Old",
                    "color": "#000000",
                    "trainable": "true",
                    "role": "phenotype",
                    "order": 1,
                }
            ],
        )
        project = {
            "schema_version": PREPARE.PROJECT_SCHEMA_VERSION,
            "project_id": "broad_phenotype_development",
            "classes_file": "classes.tsv",
            "cells_file": "cells.tsv",
            "features_file": "features.tsv",
            "images_file": "images.tsv",
            "runs_dir": "runs",
            "projection": {
                "mode": "compute_umap",
                "feature_columns": ["bf_object_mean", *PREPARE.REFERENCE_FEATURES],
                "transform": "robust",
            },
            "classifier": {
                "feature_columns": ["bf_object_mean", *PREPARE.REFERENCE_FEATURES]
            },
        }
        project_path = project_dir / "project.yml"
        project_path.write_text(json.dumps(project, indent=2) + "\n")
        umap_dir = (
            project_dir
            / "runs"
            / "broad_phenotype_development"
            / "run_fixture"
            / "projection"
            / "umap_fixture"
        )
        umap_dir.mkdir(parents=True)
        input_manifest = umap_dir / "input_manifest.tsv"
        write_tsv(
            input_manifest,
            ["input_role", "normalized_path", "raw_sha256"],
            [
                {
                    "input_role": "cells",
                    "normalized_path": str((project_dir / "cells.tsv").resolve()),
                    "raw_sha256": PREPARE.sha256_file(project_dir / "cells.tsv"),
                },
                {
                    "input_role": "features",
                    "normalized_path": str((project_dir / "features.tsv").resolve()),
                    "raw_sha256": PREPARE.sha256_file(project_dir / "features.tsv"),
                },
                {
                    "input_role": "project_config",
                    "normalized_path": str(project_path.resolve()),
                    "raw_sha256": PREPARE.sha256_file(project_path),
                },
            ],
        )
        umap_artifacts: dict[str, Path] = {}
        for role, filename in PREPARE.UMAP_ARTIFACTS.items():
            artifact = umap_dir / filename
            if role == "input_manifest":
                pass
            elif role == "umap":
                write_tsv(
                    artifact,
                    ["cell_id", "Dim1", "Dim2"],
                    [
                        {"cell_id": row["cell_id"], "Dim1": index, "Dim2": -index}
                        for index, row in enumerate(reversed(cells), start=1)
                    ],
                )
            else:
                artifact.write_text(f"fixture {role}\n")
            umap_artifacts[role] = artifact
        umap_manifest = umap_dir / "umap_manifest.json"
        umap_manifest.write_text(
            json.dumps(
                {
                    "schema_version": PREPARE.PARENT_UMAP_SCHEMA_VERSION,
                    "project_id": "broad_phenotype_development",
                    "run_id": "run_fixture",
                    "projection_id": "umap_fixture",
                    "projection_mode": "compute_umap",
                    "configured_cell_count": count,
                    "row_count": count,
                    "cell_universe_sha256": "fixture_parent_cell_universe",
                    "artifact_file_sha256": {
                        role: PREPARE.sha256_file(path)
                        for role, path in umap_artifacts.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        selection = project_dir / "selection.tsv"
        write_tsv(
            selection,
            ["cell_id"],
            [{"cell_id": row["cell_id"]} for row in cells],
        )
        split = parent / "projection_input" / "split_manifest.tsv"
        write_tsv(
            split,
            ["well", "split"],
            [
                {"well": "A01", "split": "development"},
                {"well": "A02", "split": "development"},
            ],
        )
        manifest_path = parent / "projection_input" / "projection_input_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": PREPARE.PARENT_MANIFEST_SCHEMA_VERSION,
                    "representative_project": str(project_path.resolve()),
                    "representative_project_sha256": PREPARE.sha256_file(project_path),
                    "representative_development_cell_count": count,
                    "representative_heldout_cell_count": 0,
                    "heldout_exclusion_verified": True,
                    "selection": str(selection.resolve()),
                    "selection_sha256": PREPARE.sha256_file(selection),
                    "split_manifest": str(split.resolve()),
                    "split_manifest_sha256": PREPARE.sha256_file(split),
                },
                indent=2,
            )
            + "\n"
        )
        return {
            "parent": parent,
            "project": project_path,
            "manifest": manifest_path,
            "cells": project_dir / "cells.tsv",
            "features": project_dir / "features.tsv",
            "umap_manifest": umap_manifest,
            "umap_input_manifest": input_manifest,
        }

    def refresh_umap_input_hash(self, fixture: dict[str, Path], role: str, path: Path) -> None:
        fields, rows = read_tsv(fixture["umap_input_manifest"])
        for row in rows:
            if row["input_role"] == role:
                row["raw_sha256"] = PREPARE.sha256_file(path)
        write_tsv(fixture["umap_input_manifest"], fields, rows)
        manifest = json.loads(fixture["umap_manifest"].read_text())
        manifest["artifact_file_sha256"]["input_manifest"] = PREPARE.sha256_file(
            fixture["umap_input_manifest"]
        )
        fixture["umap_manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

    def argv(self, fixture: dict[str, Path], reference: Path, count: int = 4) -> list[str]:
        return [
            "--parent-project",
            str(fixture["project"]),
            "--parent-shadow-root",
            str(fixture["parent"]),
            "--reference-shadow-root",
            str(reference),
            "--expected-cell-count",
            str(count),
        ]

    def test_builds_independent_exact_nine_feature_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            parent_hashes = {
                name: PREPARE.sha256_file(path)
                for name, path in fixture.items()
                if name not in {"parent"}
            }
            reference = root / "reference"
            self.assertEqual(PREPARE.main(self.argv(fixture, reference)), 0)
            output = reference / "projection_input" / "representative_umap"

            self.assertEqual(
                (output / "cells.tsv").read_bytes(), fixture["cells"].read_bytes()
            )
            feature_fields, feature_rows = read_tsv(output / "features.tsv")
            self.assertEqual(feature_fields, ["cell_id", *PREPARE.REFERENCE_FEATURES])
            self.assertEqual(len(feature_rows), 4)
            image_fields, image_rows = read_tsv(output / "images.tsv")
            self.assertIn("mask_path", image_fields)
            self.assertEqual(
                {row["channel_id"] for row in image_rows}, {"brightfield", "nuclei"}
            )
            class_fields, classes = read_tsv(output / "classes.tsv")
            self.assertIn("trainable", class_fields)
            self.assertEqual(
                [row["class_id"] for row in classes],
                ["multinucleated_cell", "dead_cell", "live_cell"],
            )
            self.assertNotIn("unassigned", [row["class_id"] for row in classes])

            project = json.loads((output / "project.yml").read_text())
            self.assertEqual(project["project_id"], PREPARE.PROJECT_ID)
            self.assertEqual(
                project["projection"]["feature_columns"], list(PREPARE.REFERENCE_FEATURES)
            )
            self.assertEqual(project["projection"]["transform"], "robust")
            self.assertNotIn("coordinate_file", project["projection"])
            self.assertEqual(project["classifier"]["feature_columns"], list(PREPARE.REFERENCE_FEATURES))
            self.assertEqual(project["classifier"]["alpha"], 1.0)
            self.assertEqual(project["classifier"]["lambda_rule"], "lambda.1se")
            self.assertEqual(project["classifier"]["minimum_confidence"], 0.8)
            self.assertEqual(project["review"]["quotas"]["default_per_class"], 50)
            self.assertEqual(project["review"]["quotas"]["unassigned"], 100)
            self.assertEqual(project["review"]["max_per_group"], 8)

            receipt = json.loads((output / "parent_import_manifest.json").read_text())
            self.assertEqual(receipt["row_lock"]["cell_count"], 4)
            self.assertTrue(receipt["row_lock"]["cells_byte_identical_to_parent"])
            self.assertFalse(
                receipt["reference_contract"]["parent_umap_coordinates_imported"]
            )
            self.assertEqual(receipt["reference_contract"]["nuclei_policy"], "review_display_only")
            self.assertEqual(
                receipt["reference_contract"]["unassigned_state"],
                "built_in_nontraining_review_state",
            )
            self.assertEqual(
                parent_hashes,
                {
                    name: PREPARE.sha256_file(path)
                    for name, path in fixture.items()
                    if name not in {"parent"}
                },
            )

    def test_rejects_parent_manifest_hash_drift_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            fixture["project"].write_text(fixture["project"].read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash disagrees"):
                PREPARE.main(self.argv(fixture, root / "reference"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            output = root / "reference" / "projection_input" / "representative_umap"
            output.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                PREPARE.main(self.argv(fixture, root / "reference"))

    def test_rejects_row_identity_drift_and_heldout_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            fields, rows = read_tsv(fixture["features"])
            rows[1]["cell_id"] = "original|A01_1_0d0h0m|999"
            write_tsv(fixture["features"], fields, rows)
            self.refresh_umap_input_hash(fixture, "features", fixture["features"])
            with self.assertRaisesRegex(ValueError, "lockstep identity"):
                PREPARE.main(self.argv(fixture, root / "reference"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            fields, rows = read_tsv(fixture["cells"])
            rows[0]["split"] = "heldout"
            write_tsv(fixture["cells"], fields, rows)
            self.refresh_umap_input_hash(fixture, "cells", fixture["cells"])
            with self.assertRaisesRegex(ValueError, "Heldout"):
                PREPARE.main(self.argv(fixture, root / "reference"))

    def test_rejects_symlinked_parent_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_parent(root)
            real_features = fixture["features"].with_name("features.real.tsv")
            fixture["features"].rename(real_features)
            fixture["features"].symlink_to(real_features)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                PREPARE.main(self.argv(fixture, root / "reference"))


if __name__ == "__main__":
    unittest.main()
