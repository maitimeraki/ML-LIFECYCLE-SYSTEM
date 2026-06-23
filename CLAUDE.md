# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Common Development Commands

- `make help` – Show available make targets.
- `make install` – Install all Python dependencies (editable) with dev and tuning extras.
- `make test` – Run the full test suite with coverage (`pytest tests/ -v --cov=src`).
- `make lint` – Run static analysis (`ruff check src/ tests/`) and type checking (`mypy src/`).
- `make run` – Start the FastAPI API server locally (`uvicorn` on port 8000 with auto-reload).
- `make pipeline` – Execute the demo ML lifecycle pipeline (`scripts/run_pipeline.py`).
- `make docker-up` – Build and start all services (MLflow, BentoML, FastAPI, Prometheus, Grafana).
- `make docker-down` – Stop and remove the Docker composition.

**Running a single test**
```bash
pytest tests/unit/test_api.py -vv
```

## High-Level Architecture

```
Production Traffic → Observability (Prometheus / OTel) → Drift Detection →
Decision Engine (Champion/Challenger) → Data Loading & Preprocessing →
Training & HPO → Evaluation & Validation → Model Registry → Serving (BentoML / FastAPI)
```

- **API (`src/api/`)** – FastAPI application (`app.py`) with route handlers for model and pipeline operations. Includes middleware (auth, logging) and client wrappers for MLflow and BentoML.
- **Observability (`src/observability/`)** – Prometheus metrics exporter and OpenTelemetry tracing. Config files for Prometheus and Grafana live in the `monitoring/` directory at the repo root.
- **Drift Detection (`src/drift/`)** – Statistical drift detection (`detect.py`) and reporting (`report.py`).
- **Decision Engine (`src/decision/`)** – Champion/challenger model selection policies and deployment strategy logic.
- **Data Layer (`src/data/`)** – Data loading and preprocessing utilities used during training and inference.
- **Training (`src/training/`)** – Core training loop (`trainer.py`) and hyperparameter optimization (`hyperparameter_tuner.py`).
- **Evaluation (`src/evaluation/`)** – Model validation metrics and pre-deployment gating.
- **Model Registry (`src/registry/`)** – Model version tracking, artifact storage, and MLflow integration.
- **Serving (`src/serving/`)** – BentoML and FastAPI service definitions for model inference.
- **Deployment (`src/deployment/`)** – Docker and Kubernetes deployment managers.
- **Orchestration (`src/orchestration/`)** – Pipeline orchestrator (`pipeline.py`) and scheduler.
- **Common (`src/common/`)** – Shared utilities, constants, and cross-domain helpers.

## Project Layout Highlights

- `src/` – Core library, split by domain (api, data, training, deployment, etc.).
- `tests/unit/` – pytest unit tests mirroring the package structure.
- `scripts/` – Executable entry points: `run_pipeline.py`, `train_model.py`, `evaluate_model.py`, `deploy_service.py`, `preprocess_data.py`, `monitor_logs.py`.
- `docker/` – Per-service Dockerfiles (Airflow, base, BentoML, FastAPI, MLflow).
- `monitoring/` – Prometheus scrape config and Grafana dashboards.
- `config/` – YAML configuration for pipelines and services.
- `artifacts/` – Generated model artifacts, logs, and registry snapshots.

## Service Ports (Docker Compose)

| Service    | Port  | Description           |
|------------|-------|-----------------------|
| MLflow     | 5000  | Experiment tracking   |
| BentoML    | 3000  | Model serving         |
| FastAPI    | 8000  | API gateway           |
| Prometheus | 9090  | Metrics scraping      |
| Grafana    | 3001  | Dashboards            |

## Guidelines for Future Claude Instances

- Prefer Make targets for routine actions; they encapsulate environment setup.
- When extending the pipeline, add new modules under the appropriate `src/` domain package. Register new pipeline stages in `src/orchestration/pipeline.py`.
- Keep Prometheus metric registration in `src/observability/prometheus_exporter.py`; expose traces via `src/observability/otel_tracing.py`.
- Follow the immutable data principles defined in the global coding-style rules (no in-place mutation).
- Environment configuration lives in `.env` (see `.env.example` for the full list: `MLFLOW_TRACKING_URI`, `DRIFT_PSI_THRESHOLD`, `DEPLOYMENT_STRATEGY`, etc.).
