# docker/mlflow/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir mlflow==3.12.0

RUN mkdir -p tmp/mlflow tmp/mlflow-artifacts

EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "*:5000", \
     "--default-artifact-root", "/tmp/mlflow-artifacts", \
     "--backend-store-uri", "sqlite:///tmp/mlflow/mlflow.db"]