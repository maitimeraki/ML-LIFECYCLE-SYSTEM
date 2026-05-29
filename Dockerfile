# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create directories
RUN mkdir -p \
    artifacts/processors \
    artifacts/reference \
    artifacts/models \
    artifacts/model_registry \
    reports/drift \
    logs \
    mlflow-artifacts

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 3000