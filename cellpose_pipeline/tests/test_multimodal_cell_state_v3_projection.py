from __future__ import annotations

import csv
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
    / "43_build_multimodal_cell_state_v3_projection.R"
)
CONFIG = ROOT / "cellpose_pipeline" / "configs" / "multimodal_cell_state_v3.json"


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV3ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("Rscript") is None:
            raise unittest.SkipTest("Rscript is unavailable")
        check = run_r(
            [
                "-e",
                'needed<-c("jsonlite","digest","uwot","dbscan");quit(status=ifelse(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)),0,2))',
            ],
            check=False,
        )
        if check.returncode:
            raise unittest.SkipTest("V3 R dependencies are unavailable")

    def make_fixture(
        self, root: Path, row_count: int = 240
    ) -> tuple[Path, Path, Path, Path]:
        config = json.loads(CONFIG.read_text())
        config["cell_universe"]["expected_development_cells"] = row_count
        config["cell_universe"]["expected_development_wells"] = 8
        config["projection_grid"]["n_neighbors"] = [15]
        config["projection_grid"]["min_dist"] = [0.1]
        config["projection_grid"]["seeds"] = [20260813, 20260814]
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        cells_path = root / "cells.tsv"
        with cells_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "cell_id",
                "well",
                "site",
                "elapsed_hours",
                "context_key",
                "source_id",
                "split",
            ]
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for index in range(row_count):
                well = f"{chr(65 + index % 8)}{2 + index % 4}"
                writer.writerow(
                    {
                        "cell_id": f"original|A10_1_01d02h00m|{index+1}",
                        "well": well,
                        "site": 1 + index % 3,
                        "elapsed_hours": 24 + index % 12,
                        "context_key": (
                            "SUM-159-NLS-2N"
                            if index < row_count // 2
                            else "SUM-159-NLS-4N"
                        ),
                        "source_id": well,
                        "split": "development",
                    }
                )
        features_path = root / "features.tsv"
        shape = config["feature_blocks"]["shape"]["candidate_columns"]
        bf = config["feature_blocks"]["brightfield"]["candidate_columns"]
        nuclei = config["feature_blocks"]["nuclei"]["candidate_columns"]
        qc = config["nuclei_zero_aware"]["raw_signal_qc_columns"]
        fields = ["cell_id", *shape, *bf, *nuclei, *qc, "nuclei_measurement_status"]
        with features_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for index in range(row_count):
                absent = index % 7 == 0
                missed = index % 31 == 0
                present = not absent and not missed
                row: dict[str, object] = {
                    "cell_id": f"original|A10_1_01d02h00m|{index+1}"
                }
                row.update(
                    area_px2=500 + (index % 61) * 27,
                    roundness=0.3 + (index % 23) / 50,
                    aspect_ratio=1.0 + (index % 17) / 10,
                    extent=0.35 + (index % 19) / 40,
                    solidity=0.6 + (index % 13) / 40,
                )
                for feature_index, name in enumerate(bf):
                    value = 0.02 + (index % (17 + feature_index)) * (
                        0.003 + feature_index * 0.0002
                    )
                    if name in (
                        "bf_object_mean_minus_ring_background",
                        "bf_interior_minus_boundary_mean",
                        "bf_glcm_correlation",
                    ):
                        value -= 0.08
                    row[name] = (
                        value + math.sin((index + 1) * (feature_index + 1)) * 0.002
                    )
                row["nucleus_absent"] = 1 if absent else 0
                row["nuclei_count_excess"] = math.log1p(index % 4 if present else 0)
                for feature_index, name in enumerate(nuclei[2:]):
                    row[name] = (
                        ""
                        if not present
                        else 0.05 + (index % (11 + feature_index)) * 0.02
                    )
                row["nuclei_cell_to_ring_contrast"] = (
                    5 + (index % 9) * 0.2 if missed else (index % 13) * 0.1
                )
                row["nuclei_signal_positive_fraction"] = (
                    0.3 if missed else 0.02 + (index % 7) * 0.01
                )
                row["nuclei_measurement_status"] = (
                    "possible_nuclei_mask_miss"
                    if missed
                    else "absent_supported" if absent else "present_supported"
                )
                writer.writerow(row)

        expanded = root / "expanded"
        (expanded / "profiles" / "expanded39").mkdir(parents=True)
        (expanded / "expanded_projection_manifest.json").write_text(
            json.dumps(
                {"status": "COMPLETE", "selected_annotation_profile": "expanded39"}
            )
            + "\n"
        )
        (expanded / "profile_summary.tsv").write_text(
            "profile\tfeature_count\nexpanded39\t39\n"
        )
        with (expanded / "profiles" / "expanded39" / "umap.tsv").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write("cell_id\tDim1\tDim2\n")
            for index in range(row_count):
                handle.write(
                    f"original|A10_1_01d02h00m|{index+1}\t{index/10}\t{math.sin(index/7)}\n"
                )
        return cells_path, features_path, config_path, expanded

    def args(
        self,
        cells: Path,
        features: Path,
        config: Path,
        expanded: Path,
        output: Path,
        *,
        overwrite: bool = False,
    ) -> list[str]:
        values = [
            str(SCRIPT),
            "--cells",
            str(cells),
            "--features",
            str(features),
            "--expanded-projection",
            str(expanded),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
        if overwrite:
            values.append("--overwrite")
        return values

    def test_config_check(self) -> None:
        result = run_r([str(SCRIPT), "--check-config", "--config", str(CONFIG)])
        self.assertIn("multimodal_cell_state_v3_config=PASS", result.stdout)
        self.assertIn("block_equal_target=0.3333333333333333", result.stdout)

    def test_projection_equal_blocks_ablation_atomic_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells, features, config, expanded = self.make_fixture(root)
            output = root / "projection"
            created = run_r(self.args(cells, features, config, expanded, output))
            self.assertIn("generation_status=created", created.stdout)
            self.assertIn("block_equalization=PASS", created.stdout)
            decision = json.loads((output / "calibration_decision.json").read_text())
            self.assertEqual(
                decision["representation_gate"], "GO_TO_BLIND_LABEL_MAPPING"
            )
            self.assertFalse(decision["dead_cell_island_required"])
            blocks = [
                row
                for row in read_tsv(output / "block_distance_contribution.tsv")
                if row["profile"] == "v3_pruned_block_equal"
            ]
            self.assertEqual(len(blocks), 3)
            for row in blocks:
                self.assertAlmostEqual(
                    float(row["observed_fraction"]), 1 / 3, places=12
                )
            retention = read_tsv(output / "feature_retention_manifest.tsv")
            self.assertNotIn(
                "nuclei_overlap_area_px2", {row["feature"] for row in retention}
            )
            self.assertTrue((output / "umap.tsv").is_file())
            self.assertEqual(len(read_tsv(output / "umap.tsv")), 240)
            manifest = json.loads((output / "projection_manifest.json").read_text())
            self.assertEqual(manifest["primary_profile"], "v3_pruned_block_equal")
            self.assertFalse(manifest["heldout_read"])
            reused = run_r(
                self.args(cells, features, config, expanded, output, overwrite=True)
            )
            self.assertIn("generation_status=verified_reuse", reused.stdout)
            with (output / "umap.tsv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            rejected = run_r(
                self.args(cells, features, config, expanded, output, overwrite=True),
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact set/hash differs", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
