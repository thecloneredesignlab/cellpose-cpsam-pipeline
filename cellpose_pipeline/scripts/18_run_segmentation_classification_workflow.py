#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def iter_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def expected_mask_path(segmentation_dir: Path, image_path: Path) -> Path:
    return segmentation_dir / f"{image_path.stem}_cp_masks.tif"


def run_cellpose(
    image_path: Path,
    mask_path: Path,
    segmentation_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    if args.skip_existing and mask_path.exists():
        print(f"Using existing segmentation: {mask_path}")
        return

    cmd = [
        "cellpose",
        "--image_path",
        str(image_path),
        "--pretrained_model",
        args.pretrained_model,
        "--channel_axis",
        str(args.channel_axis),
        "--diameter",
        str(args.diameter),
        "--flow_threshold",
        str(args.flow_threshold),
        "--cellprob_threshold",
        str(args.cellprob_threshold),
        "--min_size",
        str(args.min_size),
        "--savedir",
        str(segmentation_dir),
        "--save_tif",
        "--no_npy",
    ]
    if args.use_gpu:
        cmd.extend(["--use_gpu", "--gpu_device", args.gpu_device])

    run_command(cmd, env, args.dry_run)


def run_classifier(
    image_path: Path,
    mask_path: Path,
    classification_dir: Path,
    image_id: str,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "cellpose_pipeline/scripts/17_classify_cell_states.py",
        "--raw-image",
        str(image_path),
        "--mask-image",
        str(mask_path),
        "--image-id",
        image_id,
        "--out-dir",
        str(classification_dir),
    ]
    run_command(cmd, env, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cellpose segmentation and live/dead/transitional classification as one workflow."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image-path", type=Path, action="append", help="One image to process. Can be passed repeatedly.")
    inputs.add_argument("--dir", type=Path, help="Directory of images to process.")
    parser.add_argument(
        "--mask-image",
        type=Path,
        help="Existing mask for a single --image-path. Skips segmentation and runs classification only.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively collect images when --dir is used.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed.")
    parser.add_argument("--run-name", default="segmentation_classification", help="Name under workflow_runs.")
    parser.add_argument("--out-root", type=Path, default=Path("cellpose_pipeline/workflow_runs"))
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument("--channel-axis", type=int, default=2)
    parser.add_argument("--diameter", type=float, default=30)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-device", default="mps")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true", help="Log failures and continue with the next image.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_image and (not args.image_path or len(args.image_path) != 1):
        raise SystemExit("--mask-image is only supported with exactly one --image-path")

    if args.dir:
        images = iter_images(args.dir, args.recursive)
    else:
        images = [path for path in args.image_path]

    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit("No images selected.")

    run_dir = args.out_root / args.run_name
    segmentation_dir = run_dir / "segmentations"
    classification_dir = run_dir / "classification"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLCONFIGDIR", "cellpose_pipeline/tmp/matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print(f"run_dir={run_dir}", flush=True)
    print(f"n_images={len(images)}", flush=True)
    print(f"cellpose={shutil.which('cellpose') or 'cellpose'}", flush=True)

    failures = []
    for index, image_path in enumerate(images, start=1):
        image_path = image_path.resolve()
        image_id = image_path.stem
        mask_path = args.mask_image.resolve() if args.mask_image else expected_mask_path(segmentation_dir, image_path)

        print(f"[{index}/{len(images)}] {image_path}", flush=True)
        try:
            if args.mask_image:
                print(f"Using supplied mask: {mask_path}", flush=True)
            else:
                run_cellpose(image_path, mask_path, segmentation_dir, args, env)

            if not args.dry_run and not mask_path.exists():
                raise FileNotFoundError(f"Expected mask not found after segmentation: {mask_path}")

            run_classifier(image_path, mask_path, classification_dir, image_id, env, args.dry_run)
        except Exception as exc:
            failures.append((str(image_path), repr(exc)))
            print(f"FAILED [{index}/{len(images)}] {image_path}: {exc!r}", flush=True)
            if not args.continue_on_error:
                raise

    if failures:
        failure_path = run_dir / "failures.tsv"
        with failure_path.open("w") as handle:
            handle.write("image_path\terror\n")
            for image_path, error in failures:
                handle.write(f"{image_path}\t{error}\n")
        print(f"Workflow complete with {len(failures)} failures: {failure_path}", flush=True)
    else:
        print(f"Workflow complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
