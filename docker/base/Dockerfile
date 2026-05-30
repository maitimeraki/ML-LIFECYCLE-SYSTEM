# docker/base/Dockerfile
# Shared base image with common ML dependencies
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/base/requirements.txt /tmp/base-requirements.txt
RUN pip install --no-cache-dir -r /tmp/base-requirements.txt

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1