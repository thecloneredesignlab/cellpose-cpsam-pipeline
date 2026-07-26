# HPC Cellpose container

This directory reconstructs the repository's captured RED HPC runtime as a
`linux/amd64` Docker image. It does not copy an HPC environment directory.
Instead, it installs 294 digest-verified Conda archives, 15 hash-pinned
pip-origin wheels, and two hash-pinned Cellpose model files.

## Built image

- Tag: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
- Repository revision: `9c75b9b3f6d313d926759e768cd381686cc93bc6`
- Python: 3.10.20
- Cellpose: 4.2.1.1
- PyTorch: 2.12.1
- CUDA user-space build: 13.0
- Models: `cpsam` and `cpsam_v2`
- Model directory: `/opt/cellpose/models`

The image contains the runtime and models, but not the repository source.
Bind-mount the desired checkout at `/workspace`.

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$PWD",dst=/workspace \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  python cellpose_pipeline/scripts/segment_cellpose_images.py --help
```

## Rebuild

Prepare three external contexts. Their required layout is:

```text
conda-context/packages/<294 captured package archives>
pip-context/wheels/<15 captured wheels>
model-context/models/cpsam
model-context/models/cpsam_v2
```

The exact Conda URLs and MD5 values are in
`locks/conda-explicit-linux-64.lock.txt`; SHA-256 values are in
`locks/conda-package-archives.sha256`. The pip wheel hashes are in
`locks/pip-origin-requirements.lock.txt`. Model URLs, revision, sizes, hashes,
and license are in `locks/cellpose-models.lock.tsv`.

Then run:

```bash
export CONDA_PACKAGES_CONTEXT=/absolute/path/to/conda-context
export PIP_WHEELS_CONTEXT=/absolute/path/to/pip-context
export CELLPOSE_MODELS_CONTEXT=/absolute/path/to/model-context
bash container/build.sh
```

`build.sh` validates the Conda archives and model files before invoking
Buildx. The Docker build installs both Python package sets offline.

## Verification

Run the full offline environment and model-load check:

```bash
docker run --rm --network none --platform linux/amd64 \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  python /opt/hpc-environment/scripts/verify_environment.py
```

Run repository tests against a read-only checkout:

```bash
docker run --rm --network none --platform linux/amd64 \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  python -m unittest discover -s cellpose_pipeline/tests -v
```

The sanitized HPC evidence is under `evidence/hpc-compute.public`. Known
reconstruction differences and the unresolved external npm plugin input are
documented in `evidence/environment-gaps.md`.
