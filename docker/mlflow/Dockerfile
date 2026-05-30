# docker/mlflow/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir mlflow==3.12.0

RUN mkdir -p /mlflow /mlflow-artifacts

EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--default-artifact-root", "/mlflow-artifacts", \
     "--backend-store-uri", "sqlite:///mlflow/mlflow.db"]