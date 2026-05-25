# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

- `make help` – Show available make targets.
- `make install` – Install all Python dependencies in editable mode.
- `make test` – Run the full test suite with coverage reports.
- `make test-unit` – Run unit tests only.
- `make test-integration` – Run integration tests.
- `make lint` – Run static analysis (ruff) and type checking (mypy).
- `make format` – Apply code formatting (black) and auto‑fix lint issues.
- `make serve` – Start the FastAPI model‑serving API locally (`uvicorn` on port 8000).
- `make pipeline` – Execute the complete ML lifecycle pipeline (`scripts/run_full_lifecycle.py`).
- `make docker-build` – Build Docker images defined in `docker-compose.yml`.
- `make docker-up` – Spin up all services (Docker Compose) in detached mode.
- `make docker-down` – Stop and remove the Docker composition.
- `make clean` – Remove generated artifacts, logs, and Python caches.

**Running a single test**
```bash
pytest tests/unit/test_drift_detection.py::TestDriftDetection::test_basic_drift -vv
```

## High‑Level Architecture

```
Production Traffic → Monitoring → Decision Engine → Data Validation → Training → Validation Gates → Model Registry → Canary Deployment → Production
```

- **Monitoring (`src/monitoring`)** – Tracks model performance, drift, and alerts. Exposes Prometheus metrics.
- **Decision Engine (`src/decision_engine`)** – Determines whether a model should be retrained based on drift, performance, and business rules.
- **Data Layer (`src/data`)** – Schema registry, statistical validation, and drift detection utilities used throughout the pipeline.
- **Training (`src/training`)** – Core training logic (`trainer.py`) and hyper‑parameter tuning (`hyperparameter_tuner.py`).
- **Evaluation (`src/evaluation`)** – Validates trained models (`model_validator.py`) before promotion.
- **Model Registry (`src/registry`)** – Stores model artifacts, version metadata, and champion/challenger selection.
- **Deployment (`src/deployment`)** – Canary deployment orchestrator and FastAPI serving (`model_server.py`).
- **Orchestration (`src/orchestration`)** – `pipeline_orchestrator.py` glues the steps together; `scripts/run_full_lifecycle.py` provides a CLI entry point.
- **Utilities (`src/utils`)** – Logging, metrics, and custom exception handling used across the stack.

## Project Layout Highlights

- `src/` – Core library code, grouped by functional domain (data, training, evaluation, deployment, etc.).
- `tests/` – Unit and integration tests mirroring the package structure.
- `scripts/` – Convenience scripts for running the full lifecycle.
- `Dockerfile` / `docker-compose.yml` – Container definitions for reproducible environments.
- `Makefile` – Consolidates common developer tasks.

## Guidelines for Future Claude Instances

- Prefer using the Make targets for routine actions; they encapsulate environment setup.
- When extending the pipeline, add new modules under the appropriate domain package and expose a clear entry point in `src/orchestration/pipeline_orchestrator.py`.
- Keep metrics registration in `src/utils/metrics.py` and expose them via the `/metrics` endpoint in `model_server.py`.
- Follow the immutable data principles defined in the global coding‑style rules.
