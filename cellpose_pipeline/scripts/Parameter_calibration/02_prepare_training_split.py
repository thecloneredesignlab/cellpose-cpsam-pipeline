#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "_shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from common import IMAGE_SUFFIXES, annotation_mask_for, ensure_dir, link_or_copy, parse_image_name, relative_to_project, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Split annotated Cellpose images into train/test folders.")
    parser.add_argument("--annotation-dir", type=Path, default=Path("cellpose_pipeline/annotation_images/raw"))
    parser.add_argument("--train-dir", type=Path, default=Path("cellpose_pipeline/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("cellpose_pipeline/test"))
    parser.add_argument("--manifest", type=Path, default=Path("cellpose_pipeline/manifests/training_split.csv"))
    parser.add_argument("--mask-filter", default="_seg.npy")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    args = parser.parse_args()

    annotation_dir = args.annotation_dir if args.annotation_dir.is_absolute() else Path.cwd() / args.annotation_dir
    pairs = []
    for image_path in annotation_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = annotation_mask_for(image_path, args.mask_filter)
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise SystemExit(
            f"No annotated pairs found in {annotation_dir}. Expected masks ending with {args.mask_filter} next to each TIFF."
        )

    random.Random(args.seed).shuffle(pairs)
    n_test = max(1, round(len(pairs) * args.test_frac)) if len(pairs) >= 5 else 0
    test_pairs = set(pairs[:n_test])

    rows = []
    for image_path, mask_path in pairs:
        split = "test" if (image_path, mask_path) in test_pairs else "train"
        split_dir = ensure_dir(args.test_dir if split == "test" else args.train_dir)
        dst_image = split_dir / image_path.name
        dst_mask = split_dir / mask_path.name
        link_or_copy(image_path, dst_image, args.copy)
        link_or_copy(mask_path, dst_mask, args.copy)
        parsed = parse_image_name(image_path)
        rows.append(
            {
                "split": split,
                "image_path": relative_to_project(dst_image),
                "mask_path": relative_to_project(dst_mask),
                "source_image": relative_to_project(image_path),
                "well": parsed.well or "",
                "site": parsed.site or "",
                "elapsed_hours": f"{parsed.elapsed_hours:.3f}" if parsed.elapsed_hours is not None else "",
            }
        )

    write_csv(args.manifest, rows, ["split", "image_path", "mask_path", "source_image", "well", "site", "elapsed_hours"])
    print(f"Prepared {sum(r['split'] == 'train' for r in rows)} train and {sum(r['split'] == 'test' for r in rows)} test pairs.")
    print(f"Split manifest: {args.manifest}")


if __name__ == "__main__":
    main()
