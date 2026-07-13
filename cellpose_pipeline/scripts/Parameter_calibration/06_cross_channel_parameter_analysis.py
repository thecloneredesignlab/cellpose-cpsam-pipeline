#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


TIME_KEY_RE = re.compile(r"(H\d+_\d+_\d+d\d+h\d+m)")
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class MaskRecord:
    folder: str
    config: str
    image: str
    key: str
    mask_path: Path


@dataclass
class MaskStats:
    count: int
    labels: np.ndarray
    centroids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Cellpose parameter sets with cross-channel constraints.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bf-nuc-threshold", type=float, default=35.0)
    parser.add_argument("--dead-threshold", type=float, default=35.0)
    parser.add_argument("--current-bf", default="bf_precision_d25_cp-2_flow0_min20")
    parser.add_argument("--current-dead", default="dead_precision_d30_cp-2p25_flow0_min10")
    parser.add_argument("--current-nuc", default="nuc_baseline_d24_cp-2p5_flow0_min5")
    return parser.parse_args()


def extract_key(name: str) -> str:
    match = TIME_KEY_RE.search(name)
    if not match:
        raise ValueError(f"Cannot extract time key from {name}")
    return match.group(1)


def iter_images(input_root: Path, folder: str) -> dict[str, Path]:
    root = input_root / folder
    return {
        extract_key(path.name): path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def read_summary(path: Path) -> list[MaskRecord]:
    records: list[MaskRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                MaskRecord(
                    folder=row["folder"],
                    config=row["config"],
                    image=row["image"],
                    key=extract_key(row["image"]),
                    mask_path=Path(row["mask_path"]),
                )
            )
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mask_stats(path: Path, cache: dict[Path, MaskStats]) -> MaskStats:
    if path in cache:
        return cache[path]
    mask = tifffile.imread(path)
    labels = np.unique(mask)
    labels = labels[labels > 0].astype(np.int64, copy=False)
    if labels.size == 0:
        stats = MaskStats(0, labels, np.zeros((0, 2), dtype=np.float64))
        cache[path] = stats
        return stats

    ys, xs = np.nonzero(mask)
    labs = mask[ys, xs].astype(np.int64, copy=False)
    max_label = int(mask.max())
    areas = np.bincount(labs, minlength=max_label + 1).astype(np.float64)
    sum_y = np.bincount(labs, weights=ys, minlength=max_label + 1)
    sum_x = np.bincount(labs, weights=xs, minlength=max_label + 1)
    label_ids = labels.astype(np.int64, copy=False)
    centroids = np.column_stack((sum_y[label_ids] / areas[label_ids], sum_x[label_ids] / areas[label_ids]))
    stats = MaskStats(int(labels.size), label_ids, centroids)
    cache[path] = stats
    return stats


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0:
        return np.zeros(0, dtype=np.float64)
    if target.size == 0:
        return np.full(source.shape[0], np.inf, dtype=np.float64)
    diff = source[:, None, :] - target[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.min(dist2, axis=1))


def match_metrics(source: MaskStats, target: MaskStats, threshold: float) -> dict[str, float | int]:
    distances = nearest_distances(source.centroids, target.centroids)
    if distances.size == 0:
        return {
            "matched": 0,
            "total": 0,
            "match_rate": 0.0,
            "median_nearest_distance": 0.0,
        }
    finite = distances[np.isfinite(distances)]
    matched = int((distances <= threshold).sum())
    median_distance = float(np.median(finite)) if finite.size else math.inf
    return {
        "matched": matched,
        "total": int(distances.size),
        "match_rate": float(matched / distances.size),
        "median_nearest_distance": median_distance,
    }


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    data = image.astype(np.float32, copy=False)
    low, high = np.percentile(data, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    labels = mask.astype(np.int64, copy=False)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return edge & (labels > 0)


def draw_crosses(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], size: int = 6) -> None:
    for y, x in points:
        draw.line((x - size, y, x + size, y), fill=color, width=2)
        draw.line((x, y - size, x, y + size), fill=color, width=2)


def make_combo_overlay(
    raw_paths: dict[str, Path],
    bf_record: MaskRecord,
    dead_record: MaskRecord,
    nuc_record: MaskRecord,
    stats_cache: dict[Path, MaskStats],
    out_path: Path,
    bf_nuc_threshold: float,
    dead_threshold: float,
) -> None:
    bf_raw = tifffile.imread(raw_paths["Brightfield"])
    dead_raw = tifffile.imread(raw_paths["Dead"])
    nuc_raw = tifffile.imread(raw_paths["Nuclei"])
    bf_base = normalize_for_display(bf_raw)
    dead_norm = normalize_for_display(dead_raw)
    nuc_norm = normalize_for_display(nuc_raw)
    rgb = np.repeat((bf_base * 190).astype(np.uint8)[:, :, None], 3, axis=2)
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], (nuc_norm * 180).astype(np.uint8))
    rgb[:, :, 0] = np.maximum(rgb[:, :, 0], (dead_norm * 220).astype(np.uint8))
    rgb[:, :, 2] = np.maximum(rgb[:, :, 2], (dead_norm * 140).astype(np.uint8))

    bf_mask = tifffile.imread(bf_record.mask_path)
    dead_mask = tifffile.imread(dead_record.mask_path)
    nuc_mask = tifffile.imread(nuc_record.mask_path)
    rgb[boundary_mask(bf_mask)] = np.array([255, 40, 40], dtype=np.uint8)
    rgb[boundary_mask(nuc_mask)] = np.array([40, 255, 40], dtype=np.uint8)
    rgb[boundary_mask(dead_mask)] = np.array([255, 40, 255], dtype=np.uint8)

    bf_stats = mask_stats(bf_record.mask_path, stats_cache)
    dead_stats = mask_stats(dead_record.mask_path, stats_cache)
    nuc_stats = mask_stats(nuc_record.mask_path, stats_cache)
    bf_to_nuc_dist = nearest_distances(bf_stats.centroids, nuc_stats.centroids)
    nuc_to_bf_dist = nearest_distances(nuc_stats.centroids, bf_stats.centroids)
    dead_to_nuc_dist = nearest_distances(dead_stats.centroids, nuc_stats.centroids)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if bf_to_nuc_dist.size:
        draw_crosses(draw, bf_stats.centroids[bf_to_nuc_dist > bf_nuc_threshold], (255, 0, 0))
    if nuc_to_bf_dist.size:
        draw_crosses(draw, nuc_stats.centroids[nuc_to_bf_dist > bf_nuc_threshold], (0, 255, 0))
    if dead_to_nuc_dist.size:
        draw_crosses(draw, dead_stats.centroids[dead_to_nuc_dist > dead_threshold], (255, 0, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_contact_sheet(paths: list[Path], out_path: Path, title: str, columns: int = 2) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 480, 354
    label_h = 30
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h) + 34), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for idx, path in enumerate(paths):
        row, col = divmod(idx, columns)
        x = col * thumb_w
        y = 34 + row * (thumb_h + label_h)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 4), path.stem[:70], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    raw_by_folder = {
        "Brightfield": iter_images(args.input_root, "Brightfield"),
        "Dead": iter_images(args.input_root, "Dead"),
        "Nuclei": iter_images(args.input_root, "Nuclei"),
    }
    keys = sorted(set(raw_by_folder["Brightfield"]) & set(raw_by_folder["Dead"]) & set(raw_by_folder["Nuclei"]))
    if not keys:
        raise SystemExit("No matched triplets found")

    records: list[MaskRecord] = []
    for summary_csv in args.summary_csv:
        records.extend(read_summary(summary_csv))

    by_folder_config_key: dict[tuple[str, str, str], MaskRecord] = {}
    configs_by_folder: dict[str, set[str]] = {"Brightfield": set(), "Dead": set(), "Nuclei": set()}
    for record in records:
        if record.folder not in configs_by_folder:
            continue
        if record.key not in keys:
            continue
        by_folder_config_key[(record.folder, record.config, record.key)] = record
        configs_by_folder[record.folder].add(record.config)

    pair_rows = []
    for key in keys:
        pair_rows.append(
            {
                "key": key,
                "brightfield_image": raw_by_folder["Brightfield"][key].name,
                "dead_image": raw_by_folder["Dead"][key].name,
                "nuclei_image": raw_by_folder["Nuclei"][key].name,
            }
        )
    write_csv(out_root / "matched_triplets.csv", pair_rows)

    stats_cache: dict[Path, MaskStats] = {}
    metric_rows: list[dict] = []
    for bf_config in sorted(configs_by_folder["Brightfield"]):
        for dead_config in sorted(configs_by_folder["Dead"]):
            for nuc_config in sorted(configs_by_folder["Nuclei"]):
                for key in keys:
                    bf_record = by_folder_config_key.get(("Brightfield", bf_config, key))
                    dead_record = by_folder_config_key.get(("Dead", dead_config, key))
                    nuc_record = by_folder_config_key.get(("Nuclei", nuc_config, key))
                    if not (bf_record and dead_record and nuc_record):
                        continue
                    bf_stats = mask_stats(bf_record.mask_path, stats_cache)
                    dead_stats = mask_stats(dead_record.mask_path, stats_cache)
                    nuc_stats = mask_stats(nuc_record.mask_path, stats_cache)
                    bf_to_nuc = match_metrics(bf_stats, nuc_stats, args.bf_nuc_threshold)
                    nuc_to_bf = match_metrics(nuc_stats, bf_stats, args.bf_nuc_threshold)
                    dead_to_nuc = match_metrics(dead_stats, nuc_stats, args.dead_threshold)
                    dead_to_bf = match_metrics(dead_stats, bf_stats, args.dead_threshold)
                    count_diff = abs(bf_stats.count - nuc_stats.count)
                    denom = max(bf_stats.count, nuc_stats.count, 1)
                    dead_excess = max(0, dead_stats.count - nuc_stats.count)
                    row = {
                        "key": key,
                        "bf_config": bf_config,
                        "dead_config": dead_config,
                        "nuc_config": nuc_config,
                        "bf_count": bf_stats.count,
                        "dead_count": dead_stats.count,
                        "nuc_count": nuc_stats.count,
                        "bf_nuc_count_abs_diff": count_diff,
                        "bf_nuc_count_diff_fraction": count_diff / denom,
                        "dead_excess_over_nuc": dead_excess / max(nuc_stats.count, 1),
                        "bf_to_nuc_match_rate": bf_to_nuc["match_rate"],
                        "nuc_to_bf_match_rate": nuc_to_bf["match_rate"],
                        "dead_to_nuc_match_rate": dead_to_nuc["match_rate"],
                        "dead_to_bf_match_rate": dead_to_bf["match_rate"],
                        "bf_to_nuc_median_distance": bf_to_nuc["median_nearest_distance"],
                        "nuc_to_bf_median_distance": nuc_to_bf["median_nearest_distance"],
                        "dead_to_nuc_median_distance": dead_to_nuc["median_nearest_distance"],
                        "dead_to_bf_median_distance": dead_to_bf["median_nearest_distance"],
                    }
                    row["cross_channel_score"] = (
                        0.30 * row["bf_nuc_count_diff_fraction"]
                        + 0.20 * (1.0 - row["bf_to_nuc_match_rate"])
                        + 0.20 * (1.0 - row["nuc_to_bf_match_rate"])
                        + 0.20 * (1.0 - row["dead_to_nuc_match_rate"])
                        + 0.10 * row["dead_excess_over_nuc"]
                    )
                    metric_rows.append(row)

    write_csv(out_root / "triplet_metrics.csv", metric_rows)

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in metric_rows:
        grouped.setdefault((row["bf_config"], row["dead_config"], row["nuc_config"]), []).append(row)

    summary_rows: list[dict] = []
    for (bf_config, dead_config, nuc_config), rows in grouped.items():
        summary_rows.append(
            {
                "bf_config": bf_config,
                "dead_config": dead_config,
                "nuc_config": nuc_config,
                "n_images": len(rows),
                "score_mean": mean([float(row["cross_channel_score"]) for row in rows]),
                "bf_count_mean": mean([float(row["bf_count"]) for row in rows]),
                "dead_count_mean": mean([float(row["dead_count"]) for row in rows]),
                "nuc_count_mean": mean([float(row["nuc_count"]) for row in rows]),
                "bf_nuc_abs_diff_mean": mean([float(row["bf_nuc_count_abs_diff"]) for row in rows]),
                "bf_nuc_diff_fraction_mean": mean([float(row["bf_nuc_count_diff_fraction"]) for row in rows]),
                "bf_to_nuc_match_mean": mean([float(row["bf_to_nuc_match_rate"]) for row in rows]),
                "nuc_to_bf_match_mean": mean([float(row["nuc_to_bf_match_rate"]) for row in rows]),
                "dead_to_nuc_match_mean": mean([float(row["dead_to_nuc_match_rate"]) for row in rows]),
                "dead_to_bf_match_mean": mean([float(row["dead_to_bf_match_rate"]) for row in rows]),
                "dead_excess_over_nuc_mean": mean([float(row["dead_excess_over_nuc"]) for row in rows]),
            }
        )
    summary_rows.sort(key=lambda row: float(row["score_mean"]))
    for idx, row in enumerate(summary_rows, 1):
        row["rank"] = idx
    write_csv(out_root / "combo_summary.csv", summary_rows)

    selected = summary_rows[:1]
    current = [
        row
        for row in summary_rows
        if row["bf_config"] == args.current_bf
        and row["dead_config"] == args.current_dead
        and row["nuc_config"] == args.current_nuc
    ]
    selected.extend(current[:1])

    for row in selected:
        combo_label = (
            "rank01"
            if int(row["rank"]) == 1
            else "current"
        )
        overlay_paths = []
        for key in keys:
            bf_record = by_folder_config_key[("Brightfield", row["bf_config"], key)]
            dead_record = by_folder_config_key[("Dead", row["dead_config"], key)]
            nuc_record = by_folder_config_key[("Nuclei", row["nuc_config"], key)]
            raw_paths = {
                "Brightfield": raw_by_folder["Brightfield"][key],
                "Dead": raw_by_folder["Dead"][key],
                "Nuclei": raw_by_folder["Nuclei"][key],
            }
            out_path = out_root / "overlays" / combo_label / f"{key}.png"
            make_combo_overlay(
                raw_paths,
                bf_record,
                dead_record,
                nuc_record,
                stats_cache,
                out_path,
                args.bf_nuc_threshold,
                args.dead_threshold,
            )
            overlay_paths.append(out_path)
        write_contact_sheet(
            overlay_paths,
            out_root / "overlays" / f"{combo_label}_contact_sheet.png",
            f"{combo_label}: {row['bf_config']} | {row['dead_config']} | {row['nuc_config']}",
        )

    top_lines = ["# Top cross-channel combinations", ""]
    for row in summary_rows[:20]:
        top_lines.append(
            f"{row['rank']}. score={float(row['score_mean']):.4f}; "
            f"BF={row['bf_config']}; Dead={row['dead_config']}; Nuclei={row['nuc_config']}; "
            f"counts={float(row['bf_count_mean']):.1f}/{float(row['dead_count_mean']):.1f}/{float(row['nuc_count_mean']):.1f}; "
            f"BF-Nuc diff={float(row['bf_nuc_abs_diff_mean']):.1f}; "
            f"match BF->Nuc={float(row['bf_to_nuc_match_mean']):.3f}, "
            f"Nuc->BF={float(row['nuc_to_bf_match_mean']):.3f}, "
            f"Dead->Nuc={float(row['dead_to_nuc_match_mean']):.3f}"
        )
    top_lines.append("")
    if current:
        row = current[0]
        top_lines.append(
            f"Current combo rank={row['rank']}, score={float(row['score_mean']):.4f}; "
            f"BF={row['bf_config']}; Dead={row['dead_config']}; Nuclei={row['nuc_config']}"
        )
    (out_root / "top_combinations.md").write_text("\n".join(top_lines) + "\n")

    print(f"matched_triplets={out_root / 'matched_triplets.csv'}")
    print(f"triplet_metrics={out_root / 'triplet_metrics.csv'}")
    print(f"combo_summary={out_root / 'combo_summary.csv'}")
    print(f"top_combinations={out_root / 'top_combinations.md'}")


if __name__ == "__main__":
    main()
