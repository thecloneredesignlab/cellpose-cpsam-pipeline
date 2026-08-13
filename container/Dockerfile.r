# syntax=docker/dockerfile:1.7

FROM zafiro/cellpose-cpsam-pipeline@sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76

ARG BUILD_DATE
ARG SOURCE_REVISION
ARG SOURCE_TREE_STATE

USER root

COPY locks/r-runtime.lock.tsv /opt/hpc-environment/locks/r-runtime.lock.tsv
COPY locks/r-system-debs.lock.tsv /opt/hpc-environment/locks/r-system-debs.lock.tsv
COPY locks/r-packages.lock.tsv /opt/hpc-environment/locks/r-packages.lock.tsv
COPY locks/runtime-summary.lock.tsv /opt/hpc-environment/locks/runtime-summary.lock.tsv
COPY locks/SHA256SUMS /opt/hpc-environment/locks/SHA256SUMS
COPY scripts/prepare_r_package_context.py /opt/hpc-environment/scripts/prepare_r_package_context.py
COPY scripts/verify_r_artifact_context.py /opt/hpc-environment/scripts/verify_r_artifact_context.py
COPY scripts/install_r_packages.R /opt/hpc-environment/scripts/install_r_packages.R
COPY scripts/verify_r_environment.R /opt/hpc-environment/scripts/verify_r_environment.R

ENV R_HOME=/opt/R/4.2.3/lib/R \
    R_LIBS_SITE=/opt/R/4.2.3/lib/R/site-library \
    R_LIBS_USER=/opt/R/4.2.3/lib/R/site-library \
    R_ENVIRON_USER=/dev/null \
    R_PROFILE_USER=/dev/null

RUN --network=none \
    --mount=from=r_artifacts,source=.,target=/opt/r-artifacts,readonly \
    python /opt/hpc-environment/scripts/verify_r_artifact_context.py \
      /opt/r-artifacts /opt/hpc-environment/locks \
      --reject-installed-system-packages \
    && DEBIAN_FRONTEND=noninteractive dpkg --unpack \
       /opt/r-artifacts/system/*.deb \
       /opt/r-artifacts/runtime/*.deb \
    && DEBIAN_FRONTEND=noninteractive dpkg --configure --pending \
    && dpkg --audit \
    && ln -s /opt/R/4.2.3/bin/R /usr/local/bin/R \
    && ln -s /opt/R/4.2.3/bin/Rscript /usr/local/bin/Rscript \
    && Rscript /opt/hpc-environment/scripts/install_r_packages.R \
       /opt/hpc-environment/locks/r-packages.lock.tsv \
       /opt/r-artifacts/packages \
    && chmod -R a-w /opt/R/4.2.3/lib/R/site-library \
    && Rscript /opt/hpc-environment/scripts/verify_r_environment.R \
    && python /opt/hpc-environment/scripts/verify_environment.py \
       --skip-model-load

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="hpc-cellpose-4.2.1.1-models-reference-v2-parity" \
      org.opencontainers.image.source-tree-state="${SOURCE_TREE_STATE}" \
      org.opencontainers.image.url="https://github.com/thecloneredesignlab/cellpose-cpsam-pipeline" \
      org.opencontainers.image.description="Cellpose 4.2.1.1 CUDA runtime with cpsam, cpsam_v2, Posit R 4.2.3, and the locked reference-V2 R closure" \
      org.opencontainers.image.licenses="NOASSERTION" \
      org.opencontainers.image.base.digest="sha256:5f398bb194993d163507494eb7b3c630aab66eb0b64455ae18cc2c80952d8f76" \
      org.opencontainers.image.base.revision="9c75b9b3f6d313d926759e768cd381686cc93bc6" \
      org.opencontainers.image.base.licenses="Apache-2.0 AND BSD-3-Clause" \
      org.opencontainers.image.r.version="4.2.3" \
      org.opencontainers.image.r.distribution="Posit Debian 13 amd64 build 1" \
      org.opencontainers.image.r.package-snapshot="2025-11-11" \
      org.opencontainers.image.r.debian-snapshots="2025-10-20,2026-08-11" \
      org.opencontainers.image.r.package-count="66" \
      org.opencontainers.image.r.dbscan.version="1.2.3" \
      org.opencontainers.image.r.runtime-sha256="d2c1316527b21211d2802f123c8b5dca7265f6c69df8ce39c8b0aacc79260952"

WORKDIR /workspace
