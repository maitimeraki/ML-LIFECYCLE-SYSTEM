# docker/fastapi/Dockerfile
FROM ml_platform_base AS fastapi

# FastAPI-specific deps
RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn[standard]==0.47.0 \
    httpx==0.28.1 \
    python-multipart==0.0.29 \
    pyjwt==2.13.0

# Copy ONLY what fastapi needs
COPY config/                 config/
COPY monitoring/             monitoring/
COPY src/__init__.py         src/__init__.py
COPY src/api/                src/api/
COPY src/common/             src/common/
COPY src/observability/      src/observability/
COPY src/registry/           src/registry/
COPY src/data/validation.py  src/data/validation.py
COPY src/data/__init__.py    src/data/__init__.py
COPY src/serving/predictor.py src/serving/predictor.py
COPY src/serving/__init__.py  src/serving/__init__.py

RUN mkdir -p artifacts/model_registry artifacts/processors logs

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]