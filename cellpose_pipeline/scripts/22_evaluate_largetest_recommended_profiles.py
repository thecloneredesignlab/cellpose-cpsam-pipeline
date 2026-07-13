#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


CHANNELS = ("Brightfield", "Dead", "Nuclei")
PREFIXES = {
    "Brightfield": "SUM159_AC_Exp1_BF_",
    "Dead": "SUM159_AC_Exp1_Dead_",
    "Nuclei": "SUM159_AC_Exp1_",
}
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class MaskStats:
    count: int
    mask_fraction: float
    area_median: float
    area_p10: float
    area_p90: float
    small_fraction: float
    large_fraction: float
    centroids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommended largetest Cellpose profiles across matched triplets.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Workflow run dir containing segmentations/<channel>.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bf-nuc-distance", type=float, default=35.0)
    parser.add_argument("--dead-nuc-distance", type=float, default=35.0)
    return parser.parse_args()


def key_for(channel: str, path: Path) -> str:
    prefix = PREFIXES[channel]
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"{path.name} does not start with expected prefix {prefix}")
    return stem[len(prefix) :]


def collect_images(input_root: Path, channel: str) -> dict[str, Path]:
    folder = input_root / channel
    return {
        key_for(channel, path): path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def mask_path_for(run_dir: Path, channel: str, image_path: Path) -> Path:
    return run_dir / "segmentations" / channel / f"{image_path.stem}_cp_masks.tif"


def normalize_image(image: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(data, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    labels = mask.astype(np.int64, copy=False)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def mask_stats(mask_path: Path) -> MaskStats:
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if labels.size == 0:
        return MaskStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros((0, 2), dtype=np.float64))

    ys, xs = np.nonzero(mask)
    labs = mask[ys, xs].astype(np.int64, copy=False)
    max_label = int(mask.max())
    areas_by_label = np.bincount(labs, minlength=max_label + 1).astype(np.float64)
    label_areas = areas_by_label[labels]
    sum_y = np.bincount(labs, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(labs, weights=xs, minlength=max_label + 1)
    centroids = np.column_stack((sum_y[labels] / label_areas, sum_x[labels] / label_areas))
    return MaskStats(
        count=int(labels.size),
        mask_fraction=float((mask > 0).sum() / mask.size),
        area_median=float(np.median(label_areas)),
        area_p10=float(np.percentile(label_areas, 10)),
        area_p90=float(np.percentile(label_areas, 90)),
        small_fraction=float((label_areas < 35).mean()),
        large_fraction=float((label_areas > 10000).mean()),
        centroids=centroids,
    )


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0:
        return np.zeros(0, dtype=np.float64)
    if target.size == 0:
        return np.full(source.shape[0], np.inf, dtype=np.float64)
    chunk = 512
    out = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk):
        block = source[start : start + chunk]
        diff = block[:, None, :] - target[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        out[start : start + chunk] = np.sqrt(np.min(dist2, axis=1))
    return out


def match_metrics(source: MaskStats, target: MaskStats, threshold: float) -> dict[str, float]:
    distances = nearest_distances(source.centroids, target.centroids)
    if distances.size == 0:
        return {"match_rate": 0.0, "median_distance": 0.0, "p90_distance": 0.0}
    finite = distances[np.isfinite(distances)]
    return {
        "match_rate": float((distances <= threshold).sum() / distances.size),
        "median_distance": float(np.median(finite)) if finite.size else math.inf,
        "p90_distance": float(np.percentile(finite, 90)) if finite.size else math.inf,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def density_groups(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["nuclei_count"]), row["key"]))
    n = len(ordered)
    for index, row in enumerate(ordered):
        if index < n / 3:
            group = "low"
        elif index < 2 * n / 3:
            group = "medium"
        else:
            group = "high"
        row["density_group"] = group


def flag_row(row: dict) -> list[str]:
    flags: list[str] = []
    bf_ratio = float(row["brightfield_nuclei_count_ratio"])
    dead_ratio = float(row["dead_nuclei_count_ratio"])
    nuc_small = float(row["nuclei_small_fraction"])
    bf_to_nuc = float(row["brightfield_to_nuclei_match_rate"])
    nuc_to_bf = float(row["nuclei_to_brightfield_match_rate"])
    dead_to_nuc = float(row["dead_to_nuclei_match_rate"])

    if bf_ratio < 0.75:
        flags.append("brightfield_possible_undersegmentation")
    if bf_ratio > 1.35:
        flags.append("brightfield_possible_oversegmentation")
    if bf_to_nuc < 0.70:
        flags.append("brightfield_low_nuclei_spatial_match")
    if nuc_to_bf < 0.70:
        flags.append("nuclei_low_brightfield_spatial_match")
    if nuc_small > 0.20:
        flags.append("nuclei_many_small_objects")
    if dead_ratio > 1.20:
        flags.append("dead_possible_noise_or_oversegmentation")
    if int(row["dead_count"]) > 0 and dead_to_nuc < 0.65:
        flags.append("dead_low_nuclei_spatial_match")
    return flags


def make_channel_overlay(raw_path: Path, mask_path: Path, color: tuple[int, int, int]) -> Image.Image:
    raw = tifffile.imread(raw_path)
    if raw.ndim > 2:
        raw = np.squeeze(raw)
    norm = normalize_image(raw)
    rgb = np.repeat((norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask_pixels = mask > 0
    edge = boundary_mask(mask)
    overlay = rgb.copy()
    overlay[mask_pixels] = (0.65 * overlay[mask_pixels] + 0.35 * np.array(color)).astype(np.uint8)
    overlay[edge] = np.array(color, dtype=np.uint8)
    return Image.fromarray(overlay)


def make_triplet_panel(
    key: str,
    raw_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    row: dict,
    out_path: Path,
) -> None:
    colors = {
        "Brightfield": (255, 40, 40),
        "Dead": (255, 0, 220),
        "Nuclei": (30, 255, 50),
    }
    overlays = [make_channel_overlay(raw_paths[channel], mask_paths[channel], colors[channel]) for channel in CHANNELS]
    thumb_w, thumb_h = 420, 310
    header_h = 72
    panel = Image.new("RGB", (thumb_w * 3, thumb_h + header_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    title = (
        f"{key} | density={row['density_group']} | counts BF/Dead/Nuc="
        f"{row['brightfield_count']}/{row['dead_count']}/{row['nuclei_count']} | "
        f"BF/Nuc={float(row['brightfield_nuclei_count_ratio']):.2f}"
    )
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    flag_text = str(row.get("flags", ""))
    draw.text((8, 30), flag_text[:190], fill=(160, 0, 0), font=font)
    for index, (channel, image) in enumerate(zip(CHANNELS, overlays)):
        image = image.convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = index * thumb_w
        draw.text((x + 8, 52), channel, fill=(0, 0, 0), font=font)
        panel.paste(image, (x, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_contact_sheet(panel_paths: list[Path], out_path: Path) -> None:
    if not panel_paths:
        return
    thumb_w, thumb_h = 630, 191
    columns = 1
    sheet = Image.new("RGB", (thumb_w * columns, thumb_h * len(panel_paths)), "white")
    y = 0
    for path in panel_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (0, y))
        y += thumb_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()

    raw_by_channel = {channel: collect_images(input_root, channel) for channel in CHANNELS}
    keys = sorted(set(raw_by_channel["Brightfield"]) & set(raw_by_channel["Dead"]) & set(raw_by_channel["Nuclei"]))
    if not keys:
        raise SystemExit("No matched triplets found")

    rows: list[dict] = []
    flagged_rows: list[dict] = []
    panel_paths: list[Path] = []

    for key in keys:
        raw_paths = {channel: raw_by_channel[channel][key] for channel in CHANNELS}
        mask_paths = {channel: mask_path_for(run_dir, channel, raw_paths[channel]) for channel in CHANNELS}
        missing = [str(path) for path in mask_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing masks for {}: {}".format(key, missing))

        stats = {channel: mask_stats(mask_paths[channel]) for channel in CHANNELS}
        bf_nuc = match_metrics(stats["Brightfield"], stats["Nuclei"], args.bf_nuc_distance)
        nuc_bf = match_metrics(stats["Nuclei"], stats["Brightfield"], args.bf_nuc_distance)
        dead_nuc = match_metrics(stats["Dead"], stats["Nuclei"], args.dead_nuc_distance)
        dead_bf = match_metrics(stats["Dead"], stats["Brightfield"], args.dead_nuc_distance)

        nuc_count = max(stats["Nuclei"].count, 1)
        row = {
            "key": key,
            "brightfield_file": raw_paths["Brightfield"].name,
            "dead_file": raw_paths["Dead"].name,
            "nuclei_file": raw_paths["Nuclei"].name,
            "brightfield_count": stats["Brightfield"].count,
            "dead_count": stats["Dead"].count,
            "nuclei_count": stats["Nuclei"].count,
            "brightfield_nuclei_count_ratio": stats["Brightfield"].count / nuc_count,
            "dead_nuclei_count_ratio": stats["Dead"].count / nuc_count,
            "brightfield_mask_fraction": stats["Brightfield"].mask_fraction,
            "dead_mask_fraction": stats["Dead"].mask_fraction,
            "nuclei_mask_fraction": stats["Nuclei"].mask_fraction,
            "brightfield_area_median": stats["Brightfield"].area_median,
            "dead_area_median": stats["Dead"].area_median,
            "nuclei_area_median": stats["Nuclei"].area_median,
            "brightfield_area_p10": stats["Brightfield"].area_p10,
            "dead_area_p10": stats["Dead"].area_p10,
            "nuclei_area_p10": stats["Nuclei"].area_p10,
            "brightfield_area_p90": stats["Brightfield"].area_p90,
            "dead_area_p90": stats["Dead"].area_p90,
            "nuclei_area_p90": stats["Nuclei"].area_p90,
            "brightfield_small_fraction": stats["Brightfield"].small_fraction,
            "dead_small_fraction": stats["Dead"].small_fraction,
            "nuclei_small_fraction": stats["Nuclei"].small_fraction,
            "brightfield_large_fraction": stats["Brightfield"].large_fraction,
            "dead_large_fraction": stats["Dead"].large_fraction,
            "nuclei_large_fraction": stats["Nuclei"].large_fraction,
            "brightfield_to_nuclei_match_rate": bf_nuc["match_rate"],
            "nuclei_to_brightfield_match_rate": nuc_bf["match_rate"],
            "dead_to_nuclei_match_rate": dead_nuc["match_rate"],
            "dead_to_brightfield_match_rate": dead_bf["match_rate"],
            "brightfield_to_nuclei_median_distance": bf_nuc["median_distance"],
            "nuclei_to_brightfield_median_distance": nuc_bf["median_distance"],
            "dead_to_nuclei_median_distance": dead_nuc["median_distance"],
            "dead_to_brightfield_median_distance": dead_bf["median_distance"],
            "brightfield_mask_path": str(mask_paths["Brightfield"]),
            "dead_mask_path": str(mask_paths["Dead"]),
            "nuclei_mask_path": str(mask_paths["Nuclei"]),
        }
        rows.append(row)

    density_groups(rows)
    for row in rows:
        flags = flag_row(row)
        row["flags"] = ";".join(flags)
        if flags:
            flagged_rows.append(row.copy())
        key = str(row["key"])
        raw_paths = {channel: raw_by_channel[channel][key] for channel in CHANNELS}
        mask_paths = {channel: mask_path_for(run_dir, channel, raw_paths[channel]) for channel in CHANNELS}
        panel_path = out_dir / "qc_panels" / f"{key}.png"
        make_triplet_panel(key, raw_paths, mask_paths, row, panel_path)
        panel_paths.append(panel_path)

    write_csv(out_dir / "largetest_triplet_metrics.csv", rows)
    write_csv(out_dir / "largetest_flagged_cases.csv", flagged_rows)

    density_rows: list[dict] = []
    for group in ("low", "medium", "high"):
        subset = [row for row in rows if row["density_group"] == group]
        density_rows.append(
            {
                "density_group": group,
                "n_triplets": len(subset),
                "nuclei_count_median": median([float(row["nuclei_count"]) for row in subset]),
                "brightfield_count_median": median([float(row["brightfield_count"]) for row in subset]),
                "dead_count_median": median([float(row["dead_count"]) for row in subset]),
                "brightfield_nuclei_ratio_mean": mean([float(row["brightfield_nuclei_count_ratio"]) for row in subset]),
                "dead_nuclei_ratio_mean": mean([float(row["dead_nuclei_count_ratio"]) for row in subset]),
                "brightfield_to_nuclei_match_mean": mean([float(row["brightfield_to_nuclei_match_rate"]) for row in subset]),
                "nuclei_to_brightfield_match_mean": mean([float(row["nuclei_to_brightfield_match_rate"]) for row in subset]),
                "dead_to_nuclei_match_mean": mean([float(row["dead_to_nuclei_match_rate"]) for row in subset]),
                "flagged_triplets": sum(1 for row in subset if row["flags"]),
            }
        )
    write_csv(out_dir / "largetest_density_group_summary.csv", density_rows)
    make_contact_sheet(panel_paths, out_dir / "qc_panels_contact_sheet.png")

    total = len(rows)
    report = [
        "# Largetest Recommended Profile Evaluation",
        "",
        f"Input root: `{input_root}`",
        f"Workflow run dir: `{run_dir}`",
        f"Triplets evaluated: {total}",
        "",
        "## Overall Metrics",
        "",
        f"- Median nuclei count: {median([float(row['nuclei_count']) for row in rows]):.1f}",
        f"- Mean Brightfield/Nuclei count ratio: {mean([float(row['brightfield_nuclei_count_ratio']) for row in rows]):.3f}",
        f"- Mean Dead/Nuclei count ratio: {mean([float(row['dead_nuclei_count_ratio']) for row in rows]):.3f}",
        f"- Mean Brightfield->Nuclei match rate: {mean([float(row['brightfield_to_nuclei_match_rate']) for row in rows]):.3f}",
        f"- Mean Nuclei->Brightfield match rate: {mean([float(row['nuclei_to_brightfield_match_rate']) for row in rows]):.3f}",
        f"- Mean Dead->Nuclei match rate: {mean([float(row['dead_to_nuclei_match_rate']) for row in rows]):.3f}",
        f"- Flagged triplets: {len(flagged_rows)}/{total}",
        "",
        "## Density Groups",
        "",
    ]
    for row in density_rows:
        report.append(
            f"- {row['density_group']}: n={row['n_triplets']}, "
            f"median nuclei={float(row['nuclei_count_median']):.1f}, "
            f"BF/Nuc={float(row['brightfield_nuclei_ratio_mean']):.3f}, "
            f"BF->Nuc match={float(row['brightfield_to_nuclei_match_mean']):.3f}, "
            f"flags={row['flagged_triplets']}"
        )
    report.extend(
        [
            "",
            "## Output Files",
            "",
            "- `largetest_triplet_metrics.csv`",
            "- `largetest_density_group_summary.csv`",
            "- `largetest_flagged_cases.csv`",
            "- `qc_panels/`",
            "- `qc_panels_contact_sheet.png`",
            "",
            "## Notes",
            "",
            "These are automated cross-channel consistency metrics and QC panels. They do not replace manual visual review, but they identify density-dependent failures and cases worth inspecting before full HPC production.",
        ]
    )
    (out_dir / "largetest_evaluation_report.md").write_text("\n".join(report) + "\n")

    print(f"triplets={total}")
    print(f"metrics={out_dir / 'largetest_triplet_metrics.csv'}")
    print(f"density_summary={out_dir / 'largetest_density_group_summary.csv'}")
    print(f"flagged={out_dir / 'largetest_flagged_cases.csv'}")
    print(f"report={out_dir / 'largetest_evaluation_report.md'}")


if __name__ == "__main__":
    main()
