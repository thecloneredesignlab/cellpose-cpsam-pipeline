# SIF runtime provenance

The SIF-backed submission layer requires an absolute image path through
`HPC_CONTAINER_IMAGE`. The path is deliberately not committed because it is a
deployment detail.

The verified runtime artifact has:

- SIF SHA-256:
  `5a77773585d0dc75567dc94c9afb0df406ca1f64d1df449b92ba7880d9476cfe`
- OCI source:
  `zafiro/cellpose-cpsam-pipeline@sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76`
- architecture: `linux/amd64`
- Cellpose: `4.2.1.1`
- bundled models: `cpsam`, `cpsam_v2`

Treat the checksum as the immutable identity and the supplied filesystem path
as mutable deployment configuration.
