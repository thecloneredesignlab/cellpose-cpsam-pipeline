from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "39_build_reference_cell_state_expanded_projection.R"
)
ADAPTER = (
    ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "28_build_reference_cell_state_historical_projection.R"
)
PROJECT_BUILDER = (
    ROOT
    / "cellpose_pipeline"
    / "scripts"
    / "40_prepare_reference_cell_state_expanded_annotation_project.py"
)
CONFIG = (
    ROOT
    / "cellpose_pipeline"
    / "configs"
    / "reference_cell_state_projection_expanded_v1.json"
)
REFERENCE = ROOT.parent / "cell-phenotype-annotator" / "reference" / "ltee-source"


def run_r(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["Rscript", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode:
        raise AssertionError(result.stdout)
    return result


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


class ExpandedProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("Rscript") is None:
            raise unittest.SkipTest("Rscript is unavailable")
        if not (REFERENCE / "code" / "lib" / "Utils.R").is_file():
            raise unittest.SkipTest("Pinned local reference snapshot is unavailable")
        package_check = run_r(
            [
                "-e",
                (
                    'needed<-c("jsonlite","digest","magrittr","dplyr","stringr",'
                    '"ggplot2","tidyr","purrr","uwot","dbscan","cluster");'
                    "quit(status=ifelse(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)),0,2))"
                ),
            ],
            check=False,
        )
        if package_check.returncode:
            raise unittest.SkipTest(
                "Expanded historical R dependency closure is unavailable"
            )

    def make_fixture(self, root: Path, row_count: int = 400) -> tuple[Path, Path, Path]:
        config = json.loads(CONFIG.read_text())
        config["cell_universe"]["expected_cell_count"] = row_count
        config_path = root / "expanded_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        expanded = config["profiles"]["expanded39"]["feature_columns"]
        excluded = [row["feature"] for row in config["excluded_feature_columns"]]
        cells_path = root / "cells.tsv"
        base_features_path = root / "features.tsv"
        broad_path = root / "broad_features.tsv"
        with cells_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cell_id",
                    "context_key",
                    "source_id",
                    "well",
                    "split",
                    "cluster",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(row_count):
                writer.writerow(
                    {
                        "cell_id": f"original|A2_1_0d0h0m|{index + 1}",
                        "context_key": (
                            "SUM-159-NLS-2N"
                            if index < row_count // 2
                            else "SUM-159-NLS-4N"
                        ),
                        "source_id": f"{chr(65 + (index % 8))}{2 + (index % 10):02d}",
                        "well": f"{chr(65 + (index % 8))}{2 + (index % 10):02d}",
                        "split": "development",
                        "cluster": "1",
                    }
                )
        with broad_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cell_id", *expanded, *excluded],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(row_count):
                cluster_shift = 0.0 if index < row_count // 2 else 1.4
                row: dict[str, object] = {
                    "cell_id": f"original|A2_1_0d0h0m|{index + 1}"
                }
                for feature_index, feature in enumerate(expanded):
                    value = (
                        1.0
                        + feature_index * 0.03
                        + cluster_shift * (1 + (feature_index % 3))
                        + (index % 17) * 0.007
                        + math.sin((index + 1) * (feature_index + 2)) * 0.002
                    )
                    if feature == "bf_interior_minus_boundary_mean":
                        value = -0.2 + cluster_shift * 0.12 + (index % 13) * 0.002
                    row[feature] = f"{value:.12f}"
                row[excluded[0]] = "7797337.9"
                row[excluded[1]] = "18039216"
                writer.writerow(row)
        classifier12 = config["profiles"]["classifier12"]["feature_columns"]
        broad_fields, broad_rows = read_tsv(broad_path)
        with base_features_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cell_id", *classifier12],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in broad_rows:
                writer.writerow(
                    {field: row[field] for field in ["cell_id", *classifier12]}
                )
        with (root / "images.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "channel_id", "image_path", "mask_path"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for channel in ("brightfield", "nuclei"):
                writer.writerow(
                    {
                        "image_id": "A2_1_0d0h0m",
                        "channel_id": channel,
                        "image_path": f"/{channel}.tif",
                        "mask_path": "/combined.tif",
                    }
                )
        with (root / "classes.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["class_id", "display_name", "color"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for class_id, display, color in (
                ("live_cell", "Live cell", "#16875b"),
                ("dead_cell", "Dead cell", "#d43d51"),
                ("multinucleated_cell", "Multinucleated cell", "#7257c8"),
            ):
                writer.writerow(
                    {"class_id": class_id, "display_name": display, "color": color}
                )
        project = root / "project.yml"
        project.write_text(
            json.dumps(
                {
                    "schema_version": "cell_phenotype_annotator_project_v1",
                    "project_id": "reference_cell_state_development_v2",
                    "cells_file": "cells.tsv",
                    "features_file": "features.tsv",
                    "images_file": "images.tsv",
                    "classes_file": "classes.tsv",
                    "runs_dir": "runs",
                    "annotation": {"title": "Reference cell-state annotation"},
                    "classifier": {"feature_columns": classifier12},
                }
            )
            + "\n"
        )
        return project, broad_path, config_path

    def args(
        self,
        project: Path,
        broad: Path,
        config: Path,
        output: Path,
        *,
        overwrite: bool = False,
    ) -> list[str]:
        values = [
            str(SCRIPT),
            "--project",
            str(project),
            "--broad-features",
            str(broad),
            "--reference-snapshot-root",
            str(REFERENCE),
            "--feature-config",
            str(config),
            "--historical-adapter-script",
            str(ADAPTER),
            "--output-dir",
            str(output),
            "--dbscan-cores",
            "1",
        ]
        if overwrite:
            values.append("--overwrite")
        return values

    def test_config_is_explicit_and_excludes_unstable_ratios(self) -> None:
        config = json.loads(CONFIG.read_text())
        self.assertEqual(
            list(config["profiles"]), ["shape9", "classifier12", "expanded39"]
        )
        self.assertEqual(len(config["profiles"]["shape9"]["feature_columns"]), 9)
        self.assertEqual(len(config["profiles"]["classifier12"]["feature_columns"]), 12)
        self.assertEqual(len(config["profiles"]["expanded39"]["feature_columns"]), 39)
        self.assertEqual(
            [row["feature"] for row in config["excluded_feature_columns"]],
            [
                "bf_object_robust_z_global_background",
                "bf_object_iqr_over_background_iqr",
            ],
        )
        self.assertFalse(
            config["classifier_contract"]["expanded_features_allowed_in_classifier"]
        )

    def test_check_config_uses_pinned_historical_loader(self) -> None:
        result = run_r(
            [
                str(SCRIPT),
                "--check-config",
                "--reference-snapshot-root",
                str(REFERENCE),
                "--feature-config",
                str(CONFIG),
                "--historical-adapter-script",
                str(ADAPTER),
            ]
        )
        self.assertIn("expanded_projection_config=PASS", result.stdout)
        self.assertIn("profile_count=3", result.stdout)
        self.assertIn("selected_annotation_profile=expanded39", result.stdout)

    def test_three_profile_generation_atomic_reuse_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, broad, config = self.make_fixture(root)
            output = root / "expanded_projection"
            created = run_r(self.args(project, broad, config, output))
            self.assertIn("generation_status=created", created.stdout)
            summary_fields, summary = read_tsv(output / "profile_summary.tsv")
            self.assertEqual(summary_fields[0], "profile")
            self.assertEqual(
                [row["profile"] for row in summary],
                ["shape9", "classifier12", "expanded39"],
            )
            self.assertEqual(
                [row["feature_count"] for row in summary], ["9", "12", "39"]
            )
            for profile in ("shape9", "classifier12", "expanded39"):
                profile_root = output / "profiles" / profile
                self.assertTrue((profile_root / "umap.tsv").is_file())
                self.assertEqual(len(read_tsv(profile_root / "umap.tsv")[1]), 400)
                self.assertEqual(
                    len(read_tsv(profile_root / "dbscan_scan.tsv")[1]), 588
                )
                self.assertEqual(
                    len(read_tsv(profile_root / "bootstrap_stability.tsv")[1]), 20
                )
            _, expanded_transform = read_tsv(
                output / "profiles" / "expanded39" / "feature_transform_manifest.tsv"
            )
            self.assertEqual(len(expanded_transform), 39)
            transformed = {row["pipeline_feature"] for row in expanded_transform}
            self.assertNotIn("bf_object_robust_z_global_background", transformed)
            self.assertNotIn("bf_object_iqr_over_background_iqr", transformed)
            manifest = json.loads(
                (output / "expanded_projection_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["cell_universe"]["row_count"], 400)
            self.assertFalse(manifest["cell_universe"]["heldout_read"])
            self.assertFalse(
                manifest["classifier_boundary"][
                    "expanded39_allowed_in_final_classifier"
                ]
            )
            self.assertEqual(
                manifest["inputs"]["implementation"]["sha256"],
                hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            )
            decision = json.loads((output / "labelability_decision.json").read_text())
            self.assertEqual(
                decision["morphology_overlay_gate"], "PENDING_HUMAN_REVIEW"
            )
            reused = run_r(self.args(project, broad, config, output, overwrite=True))
            self.assertIn("generation_status=verified_reuse", reused.stdout)
            with (output / "profiles" / "expanded39" / "umap.tsv").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("tamper\n")
            rejected = run_r(
                self.args(project, broad, config, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact set/hash differs", rejected.stdout)

    def test_expanded_annotation_project_keeps_classifier12_and_replaces_only_geometry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            expanded_shadow = root / "expanded_shadow"
            base.mkdir()
            expanded_shadow.mkdir()
            project, broad, config = self.make_fixture(base)
            projection = expanded_shadow / "workflow_status" / "expanded_projection"
            run_r(self.args(project, broad, config, projection))
            output = expanded_shadow / "projection_input" / "representative_umap_v2"
            command = [
                str(PROJECT_BUILDER),
                "--base-project",
                str(project),
                "--base-shadow-root",
                str(base),
                "--expanded-projection",
                str(projection),
                "--reference-shadow-root",
                str(expanded_shadow),
                "--output-dir",
                str(output),
                "--expected-cell-count",
                "400",
            ]
            created = subprocess.run(
                [str(Path(shutil.which("python3") or "python3")), *command],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(created.returncode, 0, created.stdout)
            project_value = json.loads((output / "project.yml").read_text())
            self.assertEqual(
                project_value["project_id"], "reference_cell_state_development_v2"
            )
            self.assertEqual(
                project_value["classifier"]["feature_columns"],
                json.loads(config.read_text())["profiles"]["classifier12"][
                    "feature_columns"
                ],
            )
            self.assertEqual(
                project_value["projection"]["coordinate_file"],
                "historical_projection/umap.tsv",
            )
            _, base_cells = read_tsv(base / "cells.tsv")
            _, expanded_cells = read_tsv(output / "cells.tsv")
            self.assertEqual(
                [row["cell_id"] for row in base_cells],
                [row["cell_id"] for row in expanded_cells],
            )
            self.assertEqual(
                [row["cluster"] for row in expanded_cells],
                [
                    row["cluster"]
                    for row in read_tsv(projection / "diagnostic_clusters.tsv")[1]
                ],
            )
            historical = json.loads(
                (
                    output
                    / "historical_projection"
                    / "historical_projection_manifest.json"
                ).read_text()
            )
            self.assertEqual(
                historical["schema_version"],
                "reference_cell_state_historical_projection_expanded_v1",
            )
            self.assertFalse(
                historical["expanded_features_allowed_in_final_classifier"]
            )
            reused = subprocess.run(
                [
                    str(Path(shutil.which("python3") or "python3")),
                    *command,
                    "--overwrite",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(reused.returncode, 0, reused.stdout)
            self.assertIn("generation_status=verified_reuse", reused.stdout)

    def test_cell_order_and_nonfinite_selected_features_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, broad, config = self.make_fixture(root)
            fields, rows = read_tsv(broad)
            rows[0], rows[1] = rows[1], rows[0]
            with broad.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            rejected = run_r(
                self.args(project, broad, config, root / "order"), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("exact lockstep", rejected.stdout)

            rows[0], rows[1] = rows[1], rows[0]
            rows[0]["bf_object_mean"] = "NaN"
            with broad.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            rejected = run_r(
                self.args(project, broad, config, root / "nonfinite"), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("nonfinite", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
