# Container verification

Verification date: 2026-07-26

## Original Python-only image

- Tag: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
- Image ID: `sha256:94737bfe98665118428f0aa8a678ebf94e12fd57437a90b3130f5079a6bb4a68`
- Platform: `linux/amd64`
- Docker inspect size: 8,937,823,825 bytes
- Repository revision: `9c75b9b3f6d313d926759e768cd381686cc93bc6`
- Model revision: `7c61431b5fbb078f3296754bd15d9f51b320f837`

## Checks completed

- The public build-context identity and credential scan passed.
- All 294 explicit Conda archives passed the captured MD5 checks and the
  independent SHA-256 lock.
- Both Cellpose model files passed byte-size and SHA-256 checks.
- Both models were loaded with Cellpose on the captured NVIDIA A30 HPC
  runtime.
- Both models were loaded again from the final image with
  `--network none`; no first-run model download was possible.
- Python 3.10.20, Cellpose 4.2.1.1, PyTorch 2.12.1, CUDA build 13.0, numeric
  imports, OpenCV, TIFF, image processing, and Cellpose imports passed.
- `python`, `micromamba`, `pdftoppm`, and `npm` are available from a login
  shell.
- The repository's 50 unit tests passed in the final image with the checkout
  mounted read-only and container networking disabled.
- The image history credential scan passed.
- The exported root filesystem forbidden credential-path scan passed.

The local Apple Silicon host cannot expose an NVIDIA device to Docker.
GPU-side model loading was therefore captured and validated on the target
HPC A30 runtime, while the final image was validated locally through the same
Cellpose model constructors in offline CPU mode.

## R 4.2.3 extension verification

Verification date: 2026-08-11

- Candidate tag: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models-r4.2.3`
- Final local alias: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
- Original-image backup tag:
  `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models-python-only-sha94737bfe`
- OCI index and local image ID:
  `sha256:110a2a73ca936dcb2008e87e19a80f6fa11cd9bdb2188bbab8a14664ffdc0a3a`
- Linux/amd64 manifest:
  `sha256:a30c58038c5f3450f471a04fb75940b06d1334b331c0ce3abf1f540ac9911765`
- Platform: `linux/amd64`
- Docker inspect size: 9,205,620,559 bytes
- Immutable base manifest: `sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76`
- OCI source and image URL:
  `https://github.com/thecloneredesignlab/cellpose-cpsam-pipeline`
- R runtime: Posit R 4.2.3, Debian 13 amd64 build 1
- R package source: Posit Package Manager snapshot 2025-11-11,
  trixie-x86_64/R 4.2 binaries
- Debian package sources: timestamped 2025-10-20 and 2026-08-11
  snapshots

Checks completed:

- One R runtime deb, 122 Debian dependency archives, and 43 R binary package
  archives (221,272,178 bytes total) passed byte-size and SHA-256 validation.
- The build-time base guard confirmed that none of the 122 system packages
  replaced a package already installed in the immutable base image.
- A complete base-versus-derived `dpkg-query` comparison found no changed or
  removed base package version; `libc6` and `libc-bin` both remained 2.41-12.
- The R installation layer completed with Docker networking disabled and
  `dpkg --audit` reported no pending package state.
- All 43 exact R package versions loaded from the read-only site library.
- The site library has no write permission bits and was not writable as a
  non-root UID. The inherited default user remains root, so strict runtime
  immutability additionally requires a non-root UID or read-only filesystem.
- PNG, JPEG, and TIFF read/write smoke tests passed.
- JSON, YAML, and SHA-256 digest round trips passed.
- `glmnet` 4.1-10 completed a grouped multinomial fit and finite probability
  prediction whose row sums were within `1e-12` of one.
- `uwot` 0.2.4 returned bit-identical coordinates across two runs using the
  reference workflow's deterministic, single-threaded settings.
- Every installed R shared library passed `ldd` without an unresolved library.
- The original Python ABI check and offline CPU loading of `cpsam` and
  `cpsam_v2` passed after R was installed.
- All 71 tests in this repository passed offline with the checkout mounted
  read-only.
- The reference repository was mounted read-only; `R CMD build` and
  `R CMD check --no-manual`, including its `testthat` suite, completed with
  `Status: OK` under R 4.2.3.
- Docker Hub tag `zafiro/cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
  was published and remotely resolved to OCI index
  `sha256:110a2a73ca936dcb2008e87e19a80f6fa11cd9bdb2188bbab8a14664ffdc0a3a`.

The Apple Silicon host cannot expose an NVIDIA device to Docker. The published
OCI `linux/amd64` manifest was therefore converted to a SIF on RED HPC and
validated on an NVIDIA A30 in Slurm job `19940886`. Both `cpsam` and `cpsam_v2`
loaded on `cuda:0`; CUDA availability and the complete Cellpose verifier passed.
The resulting 6,881,107,968-byte SIF has SHA-256
`a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427`.
