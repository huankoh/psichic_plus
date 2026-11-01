# syntax=docker/dockerfile:1
ARG PSICHIC_BASE_IMAGE
FROM ${PSICHIC_BASE_IMAGE}

COPY psichic_plus_adaptive/requirements.txt /tmp/adaptive-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/adaptive-requirements.txt
RUN git config --global --add safe.directory /upstream
COPY LICENSE /opt/psichic-adaptive/LICENSE
COPY psichic_plus_adaptive/ /opt/psichic-adaptive/psichic_plus_adaptive/
ENV PYTHONPATH=/opt/psichic-adaptive \
    PSICHIC_SCREENING_DIR=/opt/min-psichic_plus
WORKDIR /opt/psichic-adaptive
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "psichic_plus_adaptive"]
CMD ["--help"]
