#!/usr/bin/env python3
"""Verify the reconstructed Cellpose runtime, ABI surface, and model files."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path


EXPECTED_VERSIONS = {
    "cellpose": "4.2.1.1",
    "matplotlib": "3.10.9",
    "numpy": "1.26.4",
    "opencv-python": "5.0.0",
    "pandas": "2.3.3",
    "pillow": "12.0.0",
    "torch": "2.12.1",
    "scikit-image": "0.25.2",
    "scipy": "1.15.2",
    "tifffile": "2025.5.10",
    "torchvision": "0.27.1",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_versions() -> None:
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise SystemExit(
                f"Version mismatch for {distribution}: expected={expected} actual={actual}"
            )


def verify_compiled_stack() -> None:
    import cv2
    import numpy as np
    import pandas as pd
    import scipy.ndimage
    import skimage.segmentation
    import tifffile
    import torch
    from PIL import Image

    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    blurred = cv2.GaussianBlur(matrix, (3, 3), 0)
    if blurred.shape != matrix.shape:
        raise SystemExit("OpenCV ABI smoke test failed")
    labels, count = scipy.ndimage.label(matrix > 7)
    if labels.shape != matrix.shape or count != 1:
        raise SystemExit("SciPy ABI smoke test failed")
    boundaries = skimage.segmentation.find_boundaries(labels)
    if boundaries.shape != matrix.shape:
        raise SystemExit("scikit-image ABI smoke test failed")
    if torch.mm(torch.eye(2), torch.eye(2)).shape != (2, 2):
        raise SystemExit("PyTorch ABI smoke test failed")
    if pd.DataFrame({"value": [1]}).shape != (1, 1):
        raise SystemExit("pandas smoke test failed")
    if Image.fromarray(matrix.astype(np.uint8)).size != (4, 4):
        raise SystemExit("Pillow smoke test failed")
    if tifffile.__version__ != EXPECTED_VERSIONS["tifffile"]:
        raise SystemExit("tifffile import smoke test failed")


def verify_models(load_models: bool) -> None:
    from cellpose import models

    lock_path = Path("/opt/hpc-environment/locks/cellpose-models.lock.tsv")
    model_dir = Path(os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", ""))
    if model_dir.resolve() != Path(models.MODEL_DIR).resolve():
        raise SystemExit(
            f"Cellpose model directory mismatch: env={model_dir} runtime={models.MODEL_DIR}"
        )
    with lock_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    available = set(map(str, getattr(models, "MODEL_NAMES", [])))
    for row in rows:
        name = row["model"]
        if name not in available:
            raise SystemExit(f"Cellpose does not expose required model: {name}")
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"Model file is missing: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"Model size mismatch: {name}")
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"Model SHA-256 mismatch: {name}")
        if load_models:
            before = (path.stat().st_size, path.stat().st_mtime_ns, sha256(path))
            model = models.CellposeModel(gpu=False, pretrained_model=name)
            del model
            gc.collect()
            after = (path.stat().st_size, path.stat().st_mtime_ns, sha256(path))
            if before != after:
                raise SystemExit(f"Model changed during offline load: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()

    verify_versions()
    verify_compiled_stack()
    if not args.skip_models:
        verify_models(load_models=not args.skip_model_load)

    import torch

    print(f"python_runtime=OK")
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print("cellpose_environment_verification=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
