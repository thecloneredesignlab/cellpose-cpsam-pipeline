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
SCRIPT = ROOT / "cellpose_pipeline/scripts/50_build_multimodal_cell_state_v4_projection.R"
CONFIG = ROOT / "cellpose_pipeline/configs/multimodal_cell_state_v4.json"


def run_r(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["Rscript", *arguments], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode:
        raise AssertionError(result.stdout)
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MultimodalCellStateV4ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("Rscript") is None:
            raise unittest.SkipTest("Rscript unavailable")
        check = run_r(
            ["-e", 'needed<-c("jsonlite","digest","uwot","dbscan");quit(status=ifelse(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)),0,2))'],
            check=False,
        )
        if check.returncode:
            raise unittest.SkipTest("V4 R dependencies unavailable")

    def make_fixture(self, root: Path, count: int = 240) -> tuple[Path, Path, Path, Path]:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["cell_universe"]["expected_development_cells"] = count
        config["cell_universe"]["expected_development_wells"] = 8
        config["projection_grid"]["n_neighbors"] = [15]
        config["projection_grid"]["min_dist"] = [0.1]
        config["projection_grid"]["seeds"] = [20260813, 20260814]
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        cells_path = root / "cells.tsv"
        cell_fields = ["cell_id", "well", "site", "elapsed_hours", "context_key", "source_id", "split"]
        with cells_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cell_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for index in range(count):
                well = f"W{index % 8 + 1:02d}"
                writer.writerow({
                    "cell_id": f"original|A10_1_01d02h00m|{index+1}", "well": well,
                    "site": str(1 + index % 3), "elapsed_hours": str(24 + index % 12),
                    "context_key": "2N" if index < count // 2 else "4N",
                    "source_id": well, "split": "development",
                })
        blocks = config["feature_blocks"]
        shape = blocks["shape"]["candidate_columns"]
        bf = blocks["brightfield"]["candidate_columns"]
        nuclei = blocks["nuclei"]["candidate_columns"]
        dead = blocks["dead"]["candidate_columns"]
        fields = [
            "cell_id", *shape, *bf, *nuclei,
            "nuclei_cell_to_ring_contrast", "nuclei_signal_positive_fraction",
            "nuclei_measurement_status", *dead, "dead_signal_saturation_fraction",
            "dead_measurement_status",
        ]
        features_path = root / "features.tsv"
        with features_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for index in range(count):
                nucleus_absent = index % 9 == 0
                nucleus_miss = index % 37 == 0
                nucleus_present = not nucleus_absent and not nucleus_miss
                dead_absent = index % 6 == 0
                dead_saturated = index % 41 == 0
                dead_uncertain = index % 43 == 0
                dead_present = not dead_absent and not dead_saturated and not dead_uncertain
                dead_status = (
                    "dead_signal_saturated" if dead_saturated else
                    "dead_signal_background_uncertain" if dead_uncertain else
                    "dead_signal_absent_supported" if dead_absent else
                    "dead_signal_present_supported"
                )
                row: dict[str, object] = {"cell_id": f"original|A10_1_01d02h00m|{index+1}"}
                row.update(
                    area_px2=350 + (index % 73) * 19,
                    roundness=0.25 + (index % 29) / 50,
                    aspect_ratio=1 + (index % 19) / 8,
                    extent=0.35 + (index % 17) / 30,
                    solidity=0.55 + (index % 13) / 30,
                )
                for offset, name in enumerate(bf):
                    value = 0.03 + (index % (11 + offset)) * (0.004 + offset / 10000)
                    if name in {"bf_object_mean_minus_ring_background", "bf_interior_minus_boundary_mean", "bf_glcm_correlation"}:
                        value -= 0.1
                    row[name] = value + math.sin((index + 1) * (offset + 1)) * 0.003
                row["nucleus_absent"] = 1 if nucleus_absent else 0
                row["nuclei_count_excess"] = math.log1p(index % 4 if nucleus_present else 0)
                for offset, name in enumerate(nuclei[2:]):
                    row[name] = "" if not nucleus_present else 0.03 + (index % (9 + offset)) * 0.025
                row["nuclei_cell_to_ring_contrast"] = 5 if nucleus_miss else 0.5 + index % 7
                row["nuclei_signal_positive_fraction"] = 0.3 if nucleus_miss else 0.02 + (index % 8) * 0.02
                row["nuclei_measurement_status"] = (
                    "possible_nuclei_mask_miss" if nucleus_miss else
                    "absent_supported" if nucleus_absent else "present_supported"
                )
                row["dead_signal_absent"] = 1 if dead_status == "dead_signal_absent_supported" else 0
                for offset, name in enumerate(dead[1:6]):
                    row[name] = (
                        "" if dead_status == "dead_signal_background_uncertain" else
                        4.0 + offset * 0.2 if dead_status == "dead_signal_saturated" else
                        0.04 + (index % (13 + offset)) * 0.03
                    )
                for offset, name in enumerate(dead[6:]):
                    row[name] = "" if not dead_present else 0.05 + (index % (7 + offset)) * 0.04
                row["dead_signal_saturation_fraction"] = 0.2 if dead_saturated else 0.001 * (index % 10)
                row["dead_measurement_status"] = dead_status
                writer.writerow(row)
        expanded = root / "expanded"
        (expanded / "profiles/expanded39").mkdir(parents=True)
        (expanded / "expanded_projection_manifest.json").write_text(
            json.dumps({"status": "COMPLETE", "selected_annotation_profile": "expanded39"}) + "\n"
        )
        (expanded / "profile_summary.tsv").write_text("profile\tfeature_count\nexpanded39\t39\n")
        with (expanded / "profiles/expanded39/umap.tsv").open("w", encoding="utf-8") as handle:
            handle.write("cell_id\tDim1\tDim2\n")
            for index in range(count):
                handle.write(f"original|A10_1_01d02h00m|{index+1}\t{index/10}\t{math.sin(index/7)}\n")
        return cells_path, features_path, config_path, expanded

    def test_config_check(self) -> None:
        result = run_r([str(SCRIPT), "--check-config", "--config", str(CONFIG)])
        self.assertIn("multimodal_cell_state_v4_config=PASS", result.stdout)
        self.assertIn("block_equal_target=0.25", result.stdout)

    def test_four_block_projection_grid_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells, features, config, expanded = self.make_fixture(root)
            output = root / "projection"
            command = [str(SCRIPT), "--cells", str(cells), "--features", str(features),
                       "--expanded-projection", str(expanded), "--config", str(config),
                       "--output-dir", str(output)]
            created = run_r(command)
            self.assertIn("block_equalization=PASS", created.stdout)
            decision = json.loads((output / "calibration_decision.json").read_text())
            self.assertEqual(decision["representation_gate"], "GO_TO_INDEPENDENT_ANCHOR_REVIEW")
            rows = [row for row in read_tsv(output / "block_distance_contribution.tsv") if row["profile"] == "v4_balanced_four_block"]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(float(row["observed_fraction"]), 0.25, places=10)
            metrics = read_tsv(output / "umap_grid_metrics.tsv")
            death_metrics = [row for row in metrics if row["profile"] in {
                "v4_balanced_four_block", "v4_death_resolution_dead35_nuclei25",
                "v4_death_resolution_dead30_bf30",
            }]
            self.assertEqual(len(death_metrics), 6)
            manifest = json.loads((output / "projection_manifest.json").read_text())
            self.assertFalse(manifest["heldout_read"])
            self.assertIn("dead_signal_absent", manifest["selected_feature_columns"])
            primary = read_tsv(
                output / "profiles/v4_balanced_four_block/processed_features.tsv"
            )
            self.assertNotEqual(
                float(primary[0]["dead_cell_to_ring_median_contrast"]), 0.0,
                "saturated Dead signal must retain censored-high continuous evidence",
            )
            reused = run_r([*command, "--overwrite"])
            self.assertIn("generation_status=verified_reuse", reused.stdout)


if __name__ == "__main__":
    unittest.main()
