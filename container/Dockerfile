ARG BASE_IMAGE=mambaorg/micromamba@sha256:800e7ade3ffe29c9a9ac2026163131495f8197c3852e572c5835beb4e8a33cd6

FROM ${BASE_IMAGE} AS environment-builder

USER root
COPY locks/ /opt/hpc-environment/locks/
COPY scripts/verify_environment.py /opt/hpc-environment/scripts/verify_environment.py
COPY --from=conda_packages packages/ /opt/conda/pkgs/
COPY --from=pip_wheels wheels/ /opt/offline/pip-wheels/

RUN cd /opt/conda/pkgs \
    && sha256sum -c /opt/hpc-environment/locks/conda-package-archives.sha256 \
    && micromamba create --yes --name cellpose_cpsam --offline \
       --file /opt/hpc-environment/locks/conda-explicit-linux-64.lock.txt \
    && /opt/conda/envs/cellpose_cpsam/bin/python -m pip install \
       --no-deps --no-index --require-hashes \
       --find-links=/opt/offline/pip-wheels \
       --requirement /opt/hpc-environment/locks/pip-origin-requirements.lock.txt \
    && /opt/conda/envs/cellpose_cpsam/bin/python \
       /opt/hpc-environment/scripts/verify_environment.py \
       --skip-models \
    && micromamba clean --all --yes

FROM ${BASE_IMAGE} AS runtime

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
       aria2 \
       bzip2 \
       ca-certificates \
       curl \
       git \
       gzip \
       nodejs \
       npm \
       poppler-utils \
       rsync \
       tar \
       unzip \
       wget \
       xz-utils \
       zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=environment-builder /opt/conda/envs/cellpose_cpsam /opt/conda/envs/cellpose_cpsam
COPY --from=models models/cpsam /opt/cellpose/models/cpsam
COPY --from=models models/cpsam_v2 /opt/cellpose/models/cpsam_v2
COPY locks/ /opt/hpc-environment/locks/
COPY scripts/ /opt/hpc-environment/scripts/

ENV CELLPOSE_LOCAL_MODELS_PATH=/opt/cellpose/models \
    ENV_NAME=cellpose_cpsam \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH=/opt/conda/envs/cellpose_cpsam/bin:${PATH} \
    PYTHONNOUSERSITE=1

RUN chmod 0444 /opt/cellpose/models/cpsam /opt/cellpose/models/cpsam_v2 \
    && ln -s /opt/conda/envs/cellpose_cpsam/bin/python /usr/local/bin/python \
    && cd /opt/cellpose/models \
    && printf '%s  %s\n' \
       e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2 cpsam \
       0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667 cpsam_v2 \
       | sha256sum -c - \
    && /opt/conda/envs/cellpose_cpsam/bin/python \
       /opt/hpc-environment/scripts/verify_environment.py \
       --skip-model-load

LABEL org.opencontainers.image.title="cellpose-cpsam-pipeline HPC runtime" \
      org.opencontainers.image.description="Reconstructed Cellpose 4.2.1.1 CUDA runtime with cpsam and cpsam_v2" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause" \
      org.opencontainers.image.revision="9c75b9b3f6d313d926759e768cd381686cc93bc6" \
      org.opencontainers.image.source="https://github.com/thecloneredesignlab/cellpose-cpsam-pipeline" \
      org.opencontainers.image.vendor="The Clone Redesign Lab" \
      org.opencontainers.image.cellpose-model-source="https://huggingface.co/mouseland/cellpose-sam" \
      org.opencontainers.image.cellpose-model-revision="7c61431b5fbb078f3296754bd15d9f51b320f837"

WORKDIR /workspace
