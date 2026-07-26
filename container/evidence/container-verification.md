# Container verification

Verification date: 2026-07-26

## Final local image

- Tag: `cellpose-cpsam-pipeline:hpc-cellpose-4.2.1.1-models`
- Image ID: `sha256:94737bfe98665118428f0aa8a678ebf94e12fd57437a90b3130f5079a6bb4a68`
- Platform: `linux/amd64`
- Uncompressed size: 8,937,823,825 bytes
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
