# docker/bentoml/Dockerfile
FROM ml_platform_base AS bentoml

# BentoML + ML serving deps
RUN pip install --no-cache-dir \
    bentoml==1.4.39 \
    evidently==0.7.21 \
    great-expectations==1.16.1

# Copy what bentoml needs (no api, no airflow)
COPY config/             config/
COPY src/__init__.py     src/__init__.py
COPY src/common/         src/common/
COPY src/observability/  src/observability/
COPY src/data/           src/data/
COPY src/drift/          src/drift/
COPY src/training/       src/training/
COPY src/evaluation/     src/evaluation/
COPY src/deployment/     src/deployment/
COPY src/registry/       src/registry/
COPY src/decision/       src/decision/
COPY src/orchestration/  src/orchestration/
COPY src/serving/        src/serving/

RUN mkdir -p \
    artifacts/processors \
    artifacts/reference \
    artifacts/models \
    artifacts/model_registry \
    logs

EXPOSE 3000
CMD ["bash", "-c", "\
    python -m src.orchestration.pipeline customer_churn_model && \
    bentoml serve src.serving.bentoml_service:MLPlatformService --port 3000 --host 0.0.0.0"]