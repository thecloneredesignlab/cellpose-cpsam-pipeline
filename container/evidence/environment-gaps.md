# Environment reconstruction differences

- The HPC workload uses the `Anaconda3/2024.02-1` module. The container uses
  the digest-pinned Micromamba 2.3.3 base to install the captured explicit
  Linux package artifacts. The package artifacts are reproduced; the module
  wrapper itself is not copied.
- The HPC operating system exposes glibc 2.28. The container base is Debian 13.
  This is an ABI-compatible reconstruction target, not a byte-identical copy
  of the HPC operating system.
- NVIDIA driver 580.105.08 is supplied by the HPC host. The image contains the
  captured CUDA user-space stack and does not package a host kernel driver.
- `npm` is not available in the selected HPC module environment. One
  repository report-delivery path calls an external npm plugin root that is
  not present in this repository. The container installs npm, but that
  external plugin remains an explicitly unresolved input.
- The image contains both required Cellpose model artifacts and sets
  `CELLPOSE_LOCAL_MODELS_PATH=/opt/cellpose/models`; runtime model loading does
  not depend on a user home cache or a first-run download.
