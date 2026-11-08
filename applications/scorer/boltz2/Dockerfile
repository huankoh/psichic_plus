# syntax=docker/dockerfile:1
ARG PSICHIC_IMAGE=psichic-workflows:latest
FROM ${PSICHIC_IMAGE}
ARG SOURCE_REVISION=unknown
ENV PSICHIC_WORKFLOW_BUILD_REVISION=${SOURCE_REVISION}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY applications/scorer/ /opt/psichic-workflows/applications/scorer/
RUN python -m pip install --no-cache-dir -r applications/scorer/boltz2/requirements.txt \
    && python -m pip check \
    && python -c "import importlib.util; assert importlib.util.find_spec('boltz') is None"
RUN /opt/conda/bin/mamba create -y --prefix /opt/boltz --channel conda-forge python=3.12.8 pip=25.0.1 \
    && /opt/conda/bin/mamba clean -afy
RUN env -u PYTHONPATH -u LD_LIBRARY_PATH /opt/boltz/bin/python -m pip install --no-cache-dir \
    -r applications/scorer/boltz2/requirements-boltz.lock \
    && env -u PYTHONPATH -u LD_LIBRARY_PATH /opt/boltz/bin/python -m pip check \
    && env -u PYTHONPATH -u LD_LIBRARY_PATH /opt/boltz/bin/python -c "import importlib.metadata as m; assert m.version('boltz') == '2.2.1'" \
    && env -u PYTHONPATH -u LD_LIBRARY_PATH /opt/boltz/bin/boltz predict --help \
    && python -m applications.scorer.boltz2 --help
ENV BOLTZ_PYTHON=/opt/boltz/bin/python \
    BOLTZ_CACHE=/cache/boltz \
    TORCH_HOME=/cache/torch
CMD ["python", "-m", "applications.scorer.boltz2", "--help"]
