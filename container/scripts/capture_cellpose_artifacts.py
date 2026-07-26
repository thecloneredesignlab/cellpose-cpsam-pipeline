#!/usr/bin/env python3
"""Capture exact Cellpose runtime and pretrained-model artifact metadata."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def requested_models(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [row["model"] for row in rows if row.get("required", "TRUE") == "TRUE"]


def load_model(models: Any, name: str, use_gpu: bool) -> dict[str, str]:
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=name)
    record = {
        "load_status": "loaded",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "device": str(getattr(model, "device", "")),
    }
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requested_models_tsv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cellpose
    from cellpose import models

    cellpose_version = importlib.metadata.version("cellpose")
    model_dir = Path(models.MODEL_DIR).resolve()
    model_url = str(getattr(models, "_MODEL_URL", ""))
    available = list(map(str, getattr(models, "MODEL_NAMES", [])))

    runtime = {
        "cellpose_version": cellpose_version,
        "cellpose_package": str(Path(cellpose.__file__).resolve()),
        "models_module": str(Path(models.__file__).resolve()),
        "model_dir": str(model_dir),
        "model_dir_environment": os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", ""),
        "model_url": model_url,
        "available_model_names": available,
        "cellpose_model_init_parameters": sorted(
            inspect.signature(models.CellposeModel).parameters
        ),
    }

    try:
        import torch

        runtime.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_build": str(torch.version.cuda),
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "torch_device_count": torch.cuda.device_count(),
                "torch_device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        runtime["torch_import"] = "missing"

    rows: list[dict[str, str | int]] = []
    for name in requested_models(args.requested_models_tsv):
        if name not in available:
            raise SystemExit(f"Required model is not exposed by Cellpose: {name}")
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"Required model file is missing: {path}")
        before = (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        load_record = {"load_status": "not_requested", "model_class": "", "device": ""}
        if args.load_models:
            load_record = load_model(models, name, args.gpu)
        after = (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        if before != after:
            raise SystemExit(f"Model artifact changed while it was being audited: {path}")
        rows.append(
            {
                "model": name,
                "path": str(path),
                "source_url": model_url + name,
                "bytes": before[0],
                "sha256": before[2],
                **load_record,
            }
        )

    (args.output_dir / "cellpose-runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cellpose-models.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "model",
            "path",
            "source_url",
            "bytes",
            "sha256",
            "load_status",
            "model_class",
            "device",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Captured Cellpose {cellpose_version} and {len(rows)} required model artifacts "
        f"into {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
