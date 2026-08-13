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
    / "28_build_reference_cell_state_historical_projection.R"
)
CONFIG = (
    ROOT
    / "cellpose_pipeline"
    / "configs"
    / "reference_cell_state_features_v2.json"
)
REFERENCE = ROOT.parent / "cell-phenotype-annotator" / "reference" / "ltee-source"
PROJECTION_FEATURES = [
    "area_px2",
    "perimeter_px",
    "roundness",
    "aspect_ratio",
    "extent",
    "solidity",
    "equivalent_diameter_px",
    "major_axis_px",
    "minor_axis_px",
]
CLASSIFIER_FEATURES = [
    *PROJECTION_FEATURES,
    "bf_boundary_mean",
    "bf_interior_mean",
    "bf_interior_minus_boundary_mean",
]


def run_r(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
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


class HistoricalProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("Rscript") is None:
            raise unittest.SkipTest("Rscript is unavailable")
        if not (REFERENCE / "code" / "lib" / "Utils.R").is_file():
            raise unittest.SkipTest("Pinned local reference snapshot is unavailable")

    def make_project(self, root: Path, row_count: int = 40) -> Path:
        features = root / "features.tsv"
        cells = root / "cells.tsv"
        cell_rows: list[dict[str, object]] = []
        with features.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cell_id", *CLASSIFIER_FEATURES],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(row_count):
                context = (
                    "SUM-159-NLS-2N"
                    if index < row_count // 2
                    else "SUM-159-NLS-4N"
                )
                row: dict[str, object] = {
                    "cell_id": f"original|A2_1_0d0h0m|{index + 1}"
                }
                row.update(
                    {
                        feature: 1
                        + index * 0.11
                        + feature_index * 0.07
                        + math.sin((index + 1) * (feature_index + 1)) * 0.01
                        for feature_index, feature in enumerate(CLASSIFIER_FEATURES)
                    }
                )
                writer.writerow(row)
                cell_rows.append(
                    {
                        "cell_id": row["cell_id"],
                        "context_key": context,
                        "source_id": "A02" if index < row_count // 2 else "B02",
                    }
                )
        with cells.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cell_id", "context_key", "source_id"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(cell_rows)
        project = root / "project.yml"
        project.write_text(
            json.dumps(
                {
                    "schema_version": "cell_phenotype_annotator_project_v1",
                    "project_id": "reference_cell_state_development_v2",
                    "cells_file": "cells.tsv",
                    "features_file": "features.tsv",
                    "projection": {
                        "mode": "existing_umap",
                        "coordinate_file": "historical_projection/umap.tsv",
                        "allow_subset": False,
                    },
                }
            )
            + "\n"
        )
        return project

    def projection_args(
        self, project: Path, output: Path, *, overwrite: bool = False
    ) -> list[str]:
        arguments = [
            str(SCRIPT),
            "--project",
            str(project),
            "--reference-snapshot-root",
            str(REFERENCE),
            "--output-dir",
            str(output),
            "--feature-config",
            str(CONFIG),
            "--dbscan-cores",
            "1",
        ]
        if overwrite:
            arguments.append("--overwrite")
        return arguments

    def test_feature_config_has_no_duplicate_json_keys(self) -> None:
        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key}")
                value[key] = item
            return value

        parsed = json.loads(CONFIG.read_text(), object_pairs_hook=reject_duplicates)
        self.assertEqual(
            parsed["historical_projection"]["authoritative_dbscan"]["max_clusters"],
            10,
        )

    def test_selective_ast_check_config_and_source_hash_guard(self) -> None:
        result = run_r(
            [
                str(SCRIPT),
                "--check-config",
                "--reference-snapshot-root",
                str(REFERENCE),
                "--feature-config",
                str(CONFIG),
            ]
        )
        self.assertIn("historical_projection_config=PASS", result.stdout)
        self.assertIn("selected_function_count=6", result.stdout)
        self.assertIn(
            "required_packages=jsonlite,digest,magrittr,dplyr,stringr,ggplot2,tidyr,purrr,uwot,dbscan,cluster",
            result.stdout,
        )

        with tempfile.TemporaryDirectory() as temporary:
            altered_root = Path(temporary)
            altered = altered_root / "code" / "lib" / "Utils.R"
            altered.parent.mkdir(parents=True)
            altered.write_bytes((REFERENCE / "code" / "lib" / "Utils.R").read_bytes() + b"\n")
            rejected = run_r(
                [
                    str(SCRIPT),
                    "--check-config",
                    "--reference-snapshot-root",
                    str(altered_root),
                    "--feature-config",
                    str(CONFIG),
                ],
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Utils.R SHA-256 mismatch", rejected.stdout)

        with tempfile.TemporaryDirectory() as temporary:
            altered_config = Path(temporary) / "features.json"
            value = json.loads(CONFIG.read_text())
            value["historical_projection"]["selective_ast_loader"][
                "function_sha256"
            ]["plot_projection_by_passage"] = "0" * 64
            altered_config.write_text(json.dumps(value) + "\n")
            rejected = run_r(
                [
                    str(SCRIPT),
                    "--check-config",
                    "--reference-snapshot-root",
                    str(REFERENCE),
                    "--feature-config",
                    str(altered_config),
                ],
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("function hashes differ", rejected.stdout)

    def test_projection_outputs_exact_grid_and_reference_preprocessing_parity(self) -> None:
        package_check = run_r(
            [
                "-e",
                (
                    'needed<-c("dplyr","stringr","ggplot2");'
                    "quit(status=ifelse(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)),0,2))"
                ),
            ],
            check=False,
        )
        if package_check.returncode:
            self.skipTest("Real dplyr/stringr/ggplot2 parity closure is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root)
            output = root / "historical_projection"
            run_r(self.projection_args(project, output))
            required = {
                "umap.tsv",
                "diagnostic_clusters.tsv",
                "fixed_dbscan_clusters.tsv",
                "dbscan_scan.tsv",
                "feature_transform_manifest.tsv",
                "pca_variance.tsv",
                "preprocessed_projection_features.tsv",
                "historical_representatives.tsv",
                "historical_projection_manifest.json",
                "historical_projection_generation_identity.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, required)
            _, scan = read_tsv(output / "dbscan_scan.tsv")
            self.assertEqual(len(scan), 588)
            manifest = json.loads(
                (output / "historical_projection_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse(manifest["selective_source"]["full_source_evaluated"])
            self.assertEqual(manifest["projection"]["seed"], 42)
            self.assertEqual(manifest["projection"]["n_neighbors"], 15)
            self.assertEqual(manifest["projection"]["knn_k"], 5)
            self.assertTrue(manifest["diagnostic_cluster"]["one_cluster_fallback"])
            self.assertEqual(manifest["diagnostic_cluster"]["n_clusters"], 1)
            self.assertEqual(
                manifest["authority"]["fixed_dbscan"],
                "audit_only_not_authoritative",
            )
            self.assertEqual(
                manifest["features"]["pixel_calibration_status"],
                "unavailable_not_recoverable",
            )
            self.assertEqual(
                manifest["features"]["historical_name_mapping_role"],
                "case_sensitive_schema_dispatch_only_not_physical_unit_claim",
            )
            self.assertIn(
                "historical_log1p_precedes_zscore",
                manifest["features"]["unrecoverable_production_artifact_difference"],
            )
            _, diagnostic_rows = read_tsv(output / "diagnostic_clusters.tsv")
            self.assertEqual({row["cluster"] for row in diagnostic_rows}, {"1"})
            self.assertEqual(
                manifest["inputs"]["implementation"]["sha256"],
                hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["representative_selection"]["output_sha256"],
                hashlib.sha256(
                    (output / "historical_representatives.tsv").read_bytes()
                ).hexdigest(),
            )

            reference_processed = root / "reference_processed.tsv"
            parity_script = root / "parity.R"
            parity_script.write_text(
                r'''args <- commandArgs(trailingOnly=TRUE)
library(dplyr); library(stringr); library(ggplot2); library(tidyr); library(purrr)
expressions <- as.list(parse(args[[1]], keep.source=TRUE))
targets <- c("impute_missing_values_knn", "morphology_projection_metadata_columns", "plot_projection_by_passage",
             "get_spatially_uniform_representatives", "get_all_cell_lines_overlay_representatives")
env <- new.env(parent=globalenv())
for (expr in expressions) {
  if (is.call(expr) && length(expr) >= 3L && as.character(expr[[1L]]) %in% c("<-", "=") &&
      is.symbol(expr[[2L]]) && as.character(expr[[2L]]) %in% targets &&
      is.call(expr[[3L]]) && identical(as.character(expr[[3L]][[1L]]), "function")) eval(expr, envir=env)
}
features <- read.delim(args[[2]], check.names=FALSE)
current <- c("area_px2","perimeter_px","roundness","aspect_ratio","extent","solidity","equivalent_diameter_px","major_axis_px","minor_axis_px")
historical_names <- c("Area.µm.2","perimeter.µm","roundness","aspect_ratio","extent","solidity","equi_diameter","Major_Axis","Minor_Axis")
historical <- as.data.frame(lapply(current, function(name) features[[name]]), check.names=FALSE)
names(historical) <- historical_names
historical$source_id <- "A0"
result <- env$plot_projection_by_passage(list(reference_v2=historical), method="pca", seed=42L,
  n_neighbors=15L, n_components=2L, n_pcs=10L, cluster=FALSE, feature_columns=historical_names)
write.table(data.frame(cell_id=features$cell_id, as.matrix(result$umapinputafterPreprocess), check.names=FALSE),
  args[[3]], sep="\t", quote=FALSE, row.names=FALSE)
scaled <- historical
scaled$`Area.µm.2` <- scaled$`Area.µm.2` * 0.4225
scaled$Major_Axis <- scaled$Major_Axis * 0.65
scaled$Minor_Axis <- scaled$Minor_Axis * 0.65
scaled_result <- env$plot_projection_by_passage(list(reference_v2=scaled), method="pca", seed=42L,
  n_neighbors=15L, n_components=2L, n_pcs=10L, cluster=FALSE, feature_columns=historical_names)
if (!isTRUE(all.equal(as.matrix(result$umapinputafterPreprocess),
                      as.matrix(scaled_result$umapinputafterPreprocess), tolerance=1e-12))) {
  stop("constant positive scaling of the three historical linear size columns changed z-scores")
}
umap <- read.delim(args[[4]], check.names=FALSE)
diagnostic <- read.delim(args[[5]], check.names=FALSE)
cells <- read.delim(args[[6]], check.names=FALSE)
selection_input <- merge(umap, diagnostic[,c("cell_id","cluster")], by="cell_id", sort=FALSE)
selection_input <- merge(selection_input, cells, by="cell_id", sort=FALSE)
selection_input <- selection_input[match(umap$cell_id, selection_input$cell_id),,drop=FALSE]
set.seed(1L)
representatives <- env$get_all_cell_lines_overlay_representatives(
  selection_input, total_n=300L, balance=.2
)
write.table(data.frame(selection_rank=seq_len(nrow(representatives)),cell_id=representatives$cell_id),
  args[[7]], sep="\t", quote=FALSE, row.names=FALSE)
'''
            )
            reference_representatives = root / "reference_representatives.tsv"
            run_r(
                [
                    str(parity_script),
                    str(REFERENCE / "code" / "lib" / "Utils.R"),
                    str(root / "features.tsv"),
                    str(reference_processed),
                    str(output / "umap.tsv"),
                    str(output / "diagnostic_clusters.tsv"),
                    str(root / "cells.tsv"),
                    str(reference_representatives),
                ]
            )
            fields_a, rows_a = read_tsv(
                output / "preprocessed_projection_features.tsv"
            )
            fields_b, rows_b = read_tsv(reference_processed)
            self.assertEqual(fields_a, fields_b)
            self.assertEqual(
                [row["cell_id"] for row in rows_a],
                [row["cell_id"] for row in rows_b],
            )
            for row_a, row_b in zip(rows_a, rows_b):
                for feature in fields_a[1:]:
                    self.assertAlmostEqual(
                        float(row_a[feature]), float(row_b[feature]), places=12
                    )
            _, selected_a = read_tsv(output / "historical_representatives.tsv")
            _, selected_b = read_tsv(reference_representatives)
            self.assertEqual(
                [row["cell_id"] for row in selected_a],
                [row["cell_id"] for row in selected_b],
            )

    def test_verified_reuse_rejects_all_identity_and_partial_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root)
            output = root / "historical_projection"
            run_r(self.projection_args(project, output))
            before = {
                path.name: path.read_bytes()
                for path in output.iterdir()
                if path.is_file()
            }
            reused = run_r(
                self.projection_args(project, output, overwrite=True)
            )
            self.assertIn("generation_status=verified_reuse", reused.stdout)
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in output.iterdir()
                    if path.is_file()
                },
            )

            artifact = output / "umap.tsv"
            artifact.write_bytes(before["umap.tsv"] + b"\n")
            rejected = run_r(
                self.projection_args(project, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("output hash conflict", rejected.stdout)
            artifact.write_bytes(before["umap.tsv"])

            manifest_path = output / "historical_projection_manifest.json"
            manifest_path.write_bytes(
                before["historical_projection_manifest.json"] + b"\n"
            )
            rejected = run_r(
                self.projection_args(project, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("identity or manifest conflicts", rejected.stdout)
            manifest_path.write_bytes(before["historical_projection_manifest.json"])

            project_before = project.read_bytes()
            project.write_bytes(project_before + b"\n")
            rejected = run_r(
                self.projection_args(project, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("input path/hash conflicts: project", rejected.stdout)
            project.write_bytes(project_before)

            identity_path = output / "historical_projection_generation_identity.json"
            adapter_copy = root / "historical_projection_adapter.R"
            adapter_copy.write_bytes(SCRIPT.read_bytes())
            alternate_output = root / "historical_projection_alternate"
            alternate_arguments = self.projection_args(project, alternate_output)
            alternate_arguments[0] = str(adapter_copy)
            run_r(alternate_arguments)
            adapter_copy.write_bytes(adapter_copy.read_bytes() + b"\n")
            alternate_arguments.append("--overwrite")
            rejected = run_r(
                alternate_arguments, check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("input path/hash conflicts: implementation", rejected.stdout)

            extra = output / "partial.tmp"
            extra.write_text("partial\n")
            rejected = run_r(
                self.projection_args(project, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact set conflicts", rejected.stdout)
            extra.unlink()

            identity_path.unlink()
            rejected = run_r(
                self.projection_args(project, output, overwrite=True), check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("output is partial", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
