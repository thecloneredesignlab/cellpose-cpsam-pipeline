from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cellpose_pipeline" / "scripts" / "18_prepare_broad_phenotype_projection.py"
SPEC = importlib.util.spec_from_file_location("broad_phenotype_projection_sampling_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROJECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROJECTION)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class BroadPhenotypeProjectionSamplingTests(unittest.TestCase):
    def test_field_cap_bounds_source_image_universe_before_cell_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells_path = root / "cells.tsv"
            features_path = root / "features.tsv"
            cells: list[dict[str, object]] = []
            features: list[dict[str, object]] = []
            for well, split, field_count in (
                ("A01", "development", 5),
                ("B01", "heldout", 1),
            ):
                for field_index in range(1, field_count + 1):
                    key = f"{well}_{field_index}_0d0h0m"
                    for label in range(1, 5):
                        cell_id = f"original|{key}|{label}"
                        cells.append(
                            {
                                "cell_id": cell_id,
                                "image_id": key,
                                "key": key,
                                "well": well,
                                "branch": "original",
                                "mask_label": label,
                                "split": split,
                            }
                        )
                        features.append({"cell_id": cell_id, "feature_a": label + field_index})
            write_tsv(
                cells_path,
                ["cell_id", "image_id", "key", "well", "branch", "mask_label", "split"],
                cells,
            )
            write_tsv(features_path, ["cell_id", "feature_a"], features)

            first = PROJECTION.stream_representative_selection(
                cells_path,
                features_path,
                ["feature_a"],
                {"A01": "development", "B01": "heldout"},
                20260812,
                2,
                2,
                10,
            )[-1]
            second = PROJECTION.stream_representative_selection(
                cells_path,
                features_path,
                ["feature_a"],
                {"A01": "development", "B01": "heldout"},
                20260812,
                2,
                2,
                10,
            )[-1]

            self.assertEqual(first, second)
            self.assertEqual({row["well"] for row in first}, {"A01"})
            self.assertLessEqual(len({row["image_id"] for row in first}), 2)
            self.assertLessEqual(len(first), 4)
            self.assertTrue(all(1 <= row["selected_field_rank"] <= 2 for row in first))
            self.assertTrue(all(1 <= row["field_rank"] <= 2 for row in first))


if __name__ == "__main__":
    unittest.main()
