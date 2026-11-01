# syntax=docker/dockerfile:1
ARG PSICHIC_BASE_IMAGE=cpu
FROM mambaorg/micromamba:2.0.5 AS cpu
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*
COPY environment_cpu.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/psichic_fp/bin:$PATH \
    LD_LIBRARY_PATH=/opt/conda/envs/psichic_fp/lib \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
SHELL ["micromamba", "run", "-n", "psichic_fp", "/bin/bash", "-c"]
RUN --mount=type=bind,source=psichic-ligand,target=/tmp/psichic-ligand,rw \
    python -m pip install --no-build-isolation --no-deps /tmp/psichic-ligand \
    && python -m psichic_ligand verify
COPY psichic_plus_screening/ /opt/min-psichic_plus/

FROM ${PSICHIC_BASE_IMAGE} AS workflows
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY psichic_plus_synthemol/requirements.txt psichic_plus_synthemol/constraints.txt /tmp/synthemol/
COPY psichic_plus_adaptive/requirements.txt /tmp/adaptive-requirements.txt
RUN python -m pip install --no-cache-dir -c /tmp/synthemol/constraints.txt \
    -r /tmp/synthemol/requirements.txt -r /tmp/adaptive-requirements.txt \
    && git clone https://github.com/swansonk14/SyntheMol.git /tmp/synthemol-source \
    && git -C /tmp/synthemol-source checkout --detach 60302339148339355fc3e568ce200ed06207f99c \
    && rm -rf /tmp/synthemol-source/synthemol/resources \
    && python -m pip install --no-cache-dir --no-deps /tmp/synthemol-source \
    && rm -rf /tmp/synthemol-source \
    && python -m pip check \
    && python -m psichic_ligand verify
RUN git config --global --add safe.directory /upstream
COPY LICENSE /opt/psichic-workflows/LICENSE
COPY psichic_plus_adaptive/ /opt/psichic-workflows/psichic_plus_adaptive/
COPY psichic_plus_synthemol/ /opt/psichic-workflows/psichic_plus_synthemol/
ENV PYTHONPATH=/opt/psichic-workflows \
    PSICHIC_SCREENING_DIR=/opt/min-psichic_plus \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/psichic-workflows
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "psichic_plus_synthemol", "--help"]
