# docker/airflow/Dockerfile
FROM apache/airflow:slim-3.2.1-python3.11

USER root
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
# 1. Pre-create the directory structure so it exists
RUN mkdir -p /app/artifacts && chown -R airflow:root /app/artifacts && chmod -R g+rwX /app/artifacts

# Install Airflow + ML deps as airflow user
USER airflow

COPY docker/airflow/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy what airflow needs (no api, no bentoml serving)
COPY --chown=airflow:root config/             /app/config/
COPY --chown=airflow:root src/__init__.py     /app/src/__init__.py
COPY --chown=airflow:root src/common/         /app/src/common/
COPY --chown=airflow:root src/observability/  /app/src/observability/
COPY --chown=airflow:root src/data/           /app/src/data/
COPY --chown=airflow:root src/drift/          /app/src/drift/
COPY --chown=airflow:root src/training/       /app/src/training/
COPY --chown=airflow:root src/evaluation/     /app/src/evaluation/
COPY --chown=airflow:root src/deployment/     /app/src/deployment/
COPY --chown=airflow:root src/registry/       /app/src/registry/
COPY --chown=airflow:root src/decision/       /app/src/decision/
COPY --chown=airflow:root src/orchestration/  /app/src/orchestration/
COPY --chown=airflow:root monitoring/         /app/monitoring/

# CRITICAL: src/orchestration is the dags folder
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/src/orchestration
ENV PYTHONPATH=/app:/app/artifacts
