# HPC Cellpose and R container

This directory reconstructs the repository's captured RED HPC runtime as a
`linux/amd64` Docker image. It does not copy an HPC environment directory.
Instead, it installs 294 digest-verified Conda archives, 15 hash-pinned
pip-origin wheels, and two hash-pinned Cellpose model files.

The R extension adds Posit's Debian 13 build of R 4.2.3 plus the complete
binary dependency closure needed by `cell-phenotype-annotator`. It is built
from an immutable published Cellpose base manifest, does not alter the
captured Python/Conda environment, and refuses to overwrite a system package
already present in that base.

## Built image

- Tag: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
- Repository revision: `9c75b9b3f6d313d926759e768cd381686cc93bc6`
- Python: 3.10.20
- Cellpose: 4.2.1.1
- PyTorch: 2.12.1
- CUDA user-space build: 13.0
- Models: `cpsam` and `cpsam_v2`
- Model directory: `/opt/cellpose/models`
- R: 4.2.3, Posit Debian 13 amd64 build 1
- R home: `/opt/R/4.2.3/lib/R`
- R packages: 43 exact Posit Package Manager binaries
- Classifier packages: `glmnet` 4.1-10 and `uwot` 0.2.4
- R package snapshot: 2025-11-11, trixie-x86_64/R 4.2
- Debian package snapshots: 2025-10-20 base-compatible packages plus
  2026-08-11 security builds already represented in the base image

The image contains the runtime and models, but not the repository source.
Bind-mount the desired checkout at `/workspace`.

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$PWD",dst=/workspace \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  python cellpose_pipeline/scripts/01_segment_images.py --help
```

The reference repository source and trained models are not copied into the
image. Only its declared R runtime dependencies are present.

## Build the R extension

`Dockerfile.r` extends the immutable `linux/amd64` Cellpose manifest
`sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76`.
The build requires an external R artifact context with this layout:

```text
r-artifacts/
  runtime/r-4.2.3_1_amd64.deb
  system/<122 locked Debian archives>
  packages/<43 locked R binary archives>
```

Prepare and verify it from the committed URL, byte-size, and SHA-256 locks:

```bash
export R_ARTIFACTS_CONTEXT=/absolute/path/to/r-artifacts
bash container/prepare-r-artifacts.sh
```

Then build the requested local tag:

```bash
export R_ARTIFACTS_CONTEXT=/absolute/path/to/r-artifacts
bash container/build-r.sh
```

For a candidate tag that does not replace the local deployment alias:

```bash
IMAGE_TAG=cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models-r4.2.3 \
  R_ARTIFACTS_CONTEXT=/absolute/path/to/r-artifacts \
  bash container/build-r.sh
```

The installation layer has networking disabled. It verifies the complete
artifact context before installing the local Debian archives and R binaries.
The base manifest is fixed in `Dockerfile.r` rather than being a build-time
override, and the in-image verifier rejects any system deb that would replace
an installed base package. System archive URLs point to timestamped Debian
snapshots.
The R site library has no write permission bits, and user R profiles and user
libraries are disabled to prevent an HPC home directory from changing package
resolution. The inherited default container user is root; use a non-root UID
or a read-only container filesystem when runtime-enforced immutability is
required.

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

Run the full offline R environment check:

```bash
docker run --rm --network none --platform linux/amd64 \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  Rscript /opt/hpc-environment/scripts/verify_r_environment.R
```

Run the full offline Python environment and model-load check:

```bash
docker run --rm --network none --platform linux/amd64 \
  cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models \
  python /opt/hpc-environment/scripts/verify_environment.py
```

The R verifier checks all 43 exact package versions, PNG/JPEG/TIFF I/O,
JSON/YAML/digest round trips, grouped multinomial `glmnet`, deterministic
single-threaded `uwot`, and every installed R shared-library dependency.

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
