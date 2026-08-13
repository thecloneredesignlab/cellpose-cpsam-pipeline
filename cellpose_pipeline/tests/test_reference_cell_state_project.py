from __future__ import annotations

import csv
import contextlib
import importlib.util
import io
import json
import subprocess
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
    def make_parent(
        self, root: Path, count: int = 4, wells: list[str] | None = None
    ) -> dict[str, Path]:
        parent = root / "broad_parent"
        project_dir = parent / "projection_input" / "representative_umap"
        project_dir.mkdir(parents=True)
        (project_dir / "runs").mkdir()
        cells: list[dict[str, object]] = []
        features: list[dict[str, object]] = []
        for index in range(1, count + 1):
            well = wells[index - 1] if wells is not None else (
                "A01" if index <= count // 2 else "A02"
            )
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
            row: dict[str, object] = {
                "cell_id": cell_id,
                "bf_object_mean": index / 10,
                "bf_boundary_mean": index + 0.01,
                "bf_interior_mean": index + 0.02,
                "bf_interior_minus_boundary_mean": 0.01,
            }
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
            [
                "cell_id",
                "bf_object_mean",
                *PREPARE.REFERENCE_FEATURES,
                "bf_boundary_mean",
                "bf_interior_mean",
                "bf_interior_minus_boundary_mean",
            ],
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
                "feature_columns": [
                    "bf_object_mean",
                    *PREPARE.REFERENCE_FEATURES,
                    "bf_boundary_mean",
                    "bf_interior_mean",
                    "bf_interior_minus_boundary_mean",
                ],
                "transform": "robust",
            },
            "classifier": {
                "feature_columns": [
                    "bf_object_mean",
                    *PREPARE.REFERENCE_FEATURES,
                    "bf_boundary_mean",
                    "bf_interior_mean",
                    "bf_interior_minus_boundary_mean",
                ]
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
        selected_wells = sorted({str(row["well"]) for row in cells})
        write_tsv(
            split,
            ["well", "split"],
            [{"well": well, "split": "development"} for well in selected_wells],
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
            "images": project_dir / "images.tsv",
            "umap_manifest": umap_manifest,
            "umap_input_manifest": input_manifest,
        }

    def add_v2_split_provenance(
        self, fixture: dict[str, Path], development_wells: set[str]
    ) -> None:
        plate_map = PREPARE.default_plate_map_path()
        with plate_map.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        adapter_root = fixture["parent"] / "workflow_status" / "adapter"
        well_split = adapter_root / "well_split_freeze.tsv"
        split_rows = []
        for row in rows:
            split_rows.append(
                {
                    **row,
                    "split": "development" if row["well"] in development_wells else "heldout",
                    "split_strategy": "condition_balanced_paired_replicate_v1",
                    "split_seed": "20260812",
                    "assignment_sha256": "fixture",
                }
            )
        write_tsv(well_split, list(PREPARE.WELL_SPLIT_FIELDS), split_rows)
        condition_split = adapter_root / "condition_split_freeze.tsv"
        write_tsv(condition_split, ["fixture"], [{"fixture": "frozen"}])
        adapter_manifest = adapter_root / "adapter_manifest.json"
        adapter_manifest.write_text('{"schema_version":"broad_phenotype_cpa_adapter_v1"}\n')
        manifest = json.loads(fixture["manifest"].read_text())
        for name, path in (
            ("well_split_freeze", well_split),
            ("condition_split_freeze", condition_split),
            ("adapter_manifest", adapter_manifest),
        ):
            manifest[name] = str(path.resolve())
            manifest[f"{name}_sha256"] = PREPARE.sha256_file(path)
        fixture["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

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

    def make_v2_case(
        self, root: Path
    ) -> tuple[dict[str, Path], Path, list[str], Path]:
        with PREPARE.default_plate_map_path().open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            plate_rows = list(csv.DictReader(handle))
        development = {
            row["well"]
            for index, row in enumerate(plate_rows, start=1)
            if index % 5 != 0
        }
        self.assertEqual(len(development), 64)
        fixture = self.make_parent(root, count=64, wells=sorted(development))
        self.add_v2_split_provenance(fixture, development)
        reference = root / "reference"
        argv = [
            *self.argv(fixture, reference, count=64),
            "--method-version",
            "v2",
            "--reference-snapshot-root",
            str(
                ROOT.parent
                / "cell-phenotype-annotator"
                / "reference"
                / "ltee-source"
            ),
            "--plate-map",
            str(PREPARE.default_plate_map_path()),
        ]
        output = reference / "projection_input" / "representative_umap_v2"
        return fixture, reference, argv, output

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

    def test_v2_builds_existing_historical_projection_and_identity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, _, argv, output = self.make_v2_case(root)
            development_fields, development_rows = read_tsv(fixture["cells"])
            self.assertIn("well", development_fields)
            development = {row["well"] for row in development_rows}
            self.assertEqual(PREPARE.main(argv), 0)
            project = json.loads((output / "project.yml").read_text())
            self.assertEqual(project["project_id"], PREPARE.PROJECT_ID_V2)
            self.assertEqual(
                project["projection"],
                {
                    "mode": "existing_umap",
                    "coordinate_file": "historical_projection/umap.tsv",
                    "allow_subset": False,
                },
            )
            self.assertEqual(
                project["classifier"]["feature_columns"],
                list(PREPARE.CLASSIFIER_FEATURES_V2),
            )
            self.assertEqual(project["classifier"]["group_column"], "source_id")
            self.assertNotIn("minimum_confidence", project["classifier"])
            _, classes = read_tsv(output / "classes.tsv")
            self.assertEqual(
                [row["class_id"] for row in classes],
                ["live_cell", "dead_cell", "multinucleated_cell"],
            )
            self.assertEqual(
                [row["color"] for row in classes],
                ["#16875b", "#d43d51", "#7257c8"],
            )
            self.assertTrue(
                all("priority" not in row["description"].casefold() for row in classes)
            )
            config_check = subprocess.run(
                [
                    "Rscript",
                    "-e",
                    (
                        "pkgload::load_all(commandArgs(TRUE)[1], quiet=TRUE);"
                        "cellphenotypeannotator::read_project_config(commandArgs(TRUE)[2]);"
                        'cat("project_config=PASS\\n")'
                    ),
                    str(ROOT.parent / "cell-phenotype-annotator"),
                    str(output / "project.yml"),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(config_check.returncode, 0, config_check.stdout)
            feature_fields, feature_rows = read_tsv(output / "features.tsv")
            self.assertEqual(
                feature_fields, ["cell_id", *PREPARE.CLASSIFIER_FEATURES_V2]
            )
            self.assertEqual(len(feature_rows), 64)
            cell_fields, cells = read_tsv(output / "cells.tsv")
            self.assertTrue(
                set((*PREPARE.HISTORICAL_IDENTITY_FIELDS, "cluster")).issubset(cell_fields)
            )
            self.assertEqual(
                {row["context_key"] for row in cells},
                {"SUM-159-NLS-2N", "SUM-159-NLS-4N"},
            )
            self.assertEqual({row["source_id"] for row in cells}, development)
            self.assertTrue(all(row["segmentation_object_id"] == row["cell_id"] for row in cells))
            projection_manifest = json.loads(
                (
                    output
                    / "historical_projection"
                    / "historical_projection_manifest.json"
                ).read_text()
            )
            self.assertEqual(projection_manifest["status"], "COMPLETE")
            self.assertEqual(projection_manifest["projection"]["seed"], 42)
            self.assertEqual(
                projection_manifest["diagnostic_cluster"]["exact_grid_count"], 588
            )
            self.assertEqual(
                projection_manifest["representative_selection"]["outer_function_name"],
                "get_all_cell_lines_overlay_representatives",
            )
            self.assertTrue(
                (output / "historical_projection" / "historical_representatives.tsv").is_file()
            )
            receipt = json.loads((output / "parent_import_manifest.json").read_text())
            self.assertEqual(receipt["schema_version"], PREPARE.SCHEMA_VERSION_V2)
            self.assertEqual(receipt["identity_mapping"]["well_count"], 64)
            self.assertEqual(receipt["identity_mapping"]["context_count"], 2)
            self.assertEqual(
                receipt["identity_mapping"]["plate_map"]["sha256"],
                PREPARE.PLATE_MAP_SHA256,
            )
            self.assertEqual(
                receipt["reference_contract"]["class_ids_order"],
                ["live_cell", "dead_cell", "multinucleated_cell"],
            )
            self.assertEqual(
                receipt["reference_contract"]["class_order_semantics"],
                "historical_annotation_source_order",
            )
            self.assertEqual(
                receipt["reference_contract"]["unit_adaptation"][
                    "physical_unit_claim"
                ],
                "not_asserted",
            )
            self.assertIn(
                "historical_log1p_precedes_zscore",
                receipt["reference_contract"]["unit_adaptation"][
                    "unrecoverable_production_artifact_difference"
                ],
            )
            self.assertNotIn(
                "class_ids_priority_order", receipt["reference_contract"]
            )
            self.assertEqual(
                receipt["implementation"]["project_builder"]["sha256"],
                PREPARE.sha256_file(SCRIPT),
            )
            self.assertEqual(
                receipt["implementation"]["historical_projection_adapter"][
                    "sha256"
                ],
                PREPARE.sha256_file(
                    ROOT
                    / "cellpose_pipeline"
                    / "scripts"
                    / "28_build_reference_cell_state_historical_projection.R"
                ),
            )
            self.assertFalse(receipt["row_lock"]["cells_byte_identical_to_parent"])

    def test_v2_complete_generation_is_verified_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reference, argv, output = self.make_v2_case(root)
            orphan = (
                reference
                / "projection_input"
                / ".representative_umap_v2.tmp.interrupted"
            )
            orphan.mkdir(parents=True)
            (orphan / "partial.txt").write_text("interrupted\n")
            first_stdout = io.StringIO()
            with contextlib.redirect_stdout(first_stdout):
                self.assertEqual(PREPARE.main(argv), 0)
            self.assertIn("generation_status=created", first_stdout.getvalue())
            identity_path = output / PREPARE.GENERATION_IDENTITY_FILENAME_V2
            self.assertTrue(identity_path.is_file())
            identity = json.loads(identity_path.read_text())
            self.assertEqual(identity["status"], "COMPLETE")
            self.assertEqual(identity["method_version"], "v2")
            self.assertRegex(identity["generation_id"], r"^[0-9a-f]{64}$")
            before = {
                path.relative_to(output).as_posix(): PREPARE.sha256_file(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            second_stdout = io.StringIO()
            with contextlib.redirect_stdout(second_stdout):
                self.assertEqual(PREPARE.main(argv), 0)
            self.assertIn("generation_status=verified_reuse", second_stdout.getvalue())
            self.assertEqual(
                before,
                {
                    path.relative_to(output).as_posix(): PREPARE.sha256_file(path)
                    for path in output.rglob("*")
                    if path.is_file()
                },
            )
            self.assertTrue((orphan / "partial.txt").is_file())

    def test_v2_reuse_rejects_project_artifact_and_manifest_drift(self) -> None:
        mutations = {
            "project": lambda output: (output / "project.yml").write_text(
                (output / "project.yml").read_text() + "\n"
            ),
            "artifact": lambda output: (
                output / "historical_projection" / "umap.tsv"
            ).write_text(
                (output / "historical_projection" / "umap.tsv").read_text()
                + "\n"
            ),
            "manifest": lambda output: (
                output / PREPARE.IMPORT_MANIFEST_FILENAME
            ).write_text(
                (output / PREPARE.IMPORT_MANIFEST_FILENAME).read_text() + "\n"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, argv, output = self.make_v2_case(root)
                self.assertEqual(PREPARE.main(argv), 0)
                mutate(output)
                with self.assertRaisesRegex(
                    (ValueError, FileExistsError), "drift|conflict|identity"
                ):
                    PREPARE.main(argv)

    def test_v2_reuse_rejects_input_and_implementation_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, _, argv, _ = self.make_v2_case(root)
            self.assertEqual(PREPARE.main(argv), 0)
            fixture["images"].write_text(fixture["images"].read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "parent input identity drifted"):
                PREPARE.main(argv)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv, _ = self.make_v2_case(root)
            adapter_copy = root / "historical_projection_adapter.R"
            adapter_copy.write_bytes(
                (
                    ROOT
                    / "cellpose_pipeline"
                    / "scripts"
                    / "28_build_reference_cell_state_historical_projection.R"
                ).read_bytes()
            )
            argv.extend(
                ["--historical-projection-script", str(adapter_copy)]
            )
            self.assertEqual(PREPARE.main(argv), 0)
            adapter_copy.write_bytes(adapter_copy.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                ValueError, "generation identity|implementation"
            ):
                PREPARE.main(argv)

    def test_v2_reuse_rejects_partial_final_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv, output = self.make_v2_case(root)
            output.mkdir(parents=True)
            (output / "project.yml").write_text("{}\n")
            with self.assertRaisesRegex(FileExistsError, "partial"):
                PREPARE.main(argv)


if __name__ == "__main__":
    unittest.main()
