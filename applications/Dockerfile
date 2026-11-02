# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu
ARG MAMBAFORGE_VERSION=23.11.0-0
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/envs/psichic_fp/bin:/opt/conda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/conda/envs/psichic_fp/lib:${LD_LIBRARY_PATH} \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends bzip2 ca-certificates curl git tini \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh" -o /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p /opt/conda \
    && rm /tmp/mambaforge.sh \
    && /opt/conda/bin/mamba clean -afy
COPY environment_gpu.yml /tmp/environment_gpu.yml
RUN /opt/conda/bin/mamba env create -f /tmp/environment_gpu.yml \
    && /opt/conda/bin/mamba clean -afy
COPY psichic-ligand/ /tmp/psichic-ligand/
RUN python -m pip install --no-build-isolation --no-deps /tmp/psichic-ligand \
    && python -m psichic_ligand verify \
    && rm -rf /tmp/psichic-ligand
COPY psichic_plus_screening/ /opt/min-psichic_plus/

FROM gpu AS workflows
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY LICENSE /opt/psichic-workflows/LICENSE
COPY psichic_plus_screening/ /opt/psichic-workflows/psichic_plus_screening/
COPY applications/ /opt/psichic-workflows/applications/
ENV PYTHONPATH=/opt/psichic-workflows \
    PSICHIC_SCREENING_DIR=/opt/min-psichic_plus \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/psichic-workflows
RUN python -m pip install --no-cache-dir -c applications/generator/constraints.txt \
    -r applications/generator/requirements.txt \
    'boto3>=1.34,<2' 'pyarrow>=15,<24' 'ipython>=8,<10' 'jinja2>=3.1,<4' \
    && python -m applications.generator.install \
    && python -m pip install --no-deps --no-build-isolation applications/screener \
    && python -m pip check \
    && python -m psichic_ligand verify
RUN git clone https://github.com/QuantumAI4Bio/AdaptiveFlow-VS.git /opt/AdaptiveFlow-VS \
    && git -C /opt/AdaptiveFlow-VS checkout --detach df74e005e456e94654fa93a40d748de399450b81 \
    && python -m psichic_adaptiveflow.install /opt/AdaptiveFlow-VS
RUN git config --global --add safe.directory /upstream
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "applications.generator", "--help"]
