# SIF runtime provenance

The SIF-backed submission layer requires an absolute image path through
`HPC_CONTAINER_IMAGE`. The path is deliberately not committed because it is a
deployment detail.

The verified runtime artifact has:

- SIF SHA-256:
  `a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427`
- OCI index:
  `sha256:110a2a73ca936dcb2008e87e19a80f6fa11cd9bdb2188bbab8a14664ffdc0a3a`
- OCI `linux/amd64` source manifest:
  `zafiro/cellpose-cpsam-pipeline@sha256:a30c58038c5f3450f471a04fb75940b06d1334b331c0ce3abf1f540ac9911765`
- immutable Cellpose base manifest:
  `sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76`
- architecture: `linux/amd64`
- Cellpose: `4.2.1.1`
- bundled models: `cpsam`, `cpsam_v2`
- R: Posit `4.2.3`
- locked R packages: `43`, including `glmnet` `4.1-10` and `uwot` `0.2.4`
- Apptainer: `1.4.2-1.el8`
- NVIDIA A30 validation: Slurm job `19940886`, completed `0:0` on
  `2026-08-12`; both bundled models loaded on `cuda:0`

Treat the checksum as the immutable identity and the supplied filesystem path
as mutable deployment configuration.
