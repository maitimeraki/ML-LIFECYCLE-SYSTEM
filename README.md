# Production ML Lifecycle System

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Containerization-Docker-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade end-to-end ML model lifecycle management system** with automated retraining, deployment, and monitoring.

[Quick Start](#quick-start-guide) • [Architecture](#architecture-overview) • [How It Works](#how-it-works-complete-lifecycle) • [Deployment](#deployment-strategies) • [Contributing](#contributing)

</div>

---

## Table of Contents

- [What is This?](#what-is-this)
- [Why This System Exists](#why-this-system-exists)
- [Key Capabilities](#key-capabilities)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [How It Works: Complete Lifecycle](#how-it-works-complete-lifecycle)
- [Pipeline Execution](#pipeline-execution)
- [Monitoring & Observability](#monitoring--observability)
- [Quick Start Guide](#quick-start-guide)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Deployment Strategies](#deployment-strategies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## What is This?

**ML Lifecycle System** is an integrated framework that manages the **entire lifespan** of a machine learning model in production—from data validation through serving and continuous monitoring.

### Problem It Solves

Traditional ML deployments struggle with:

- 🔴 **Model drift** going undetected until users notice poor performance
- 🔴 **Manual retraining workflows** that can't react to changing data distributions
- 🔴 **Risky deployments** without gradual rollout or instant rollback capabilities
- 🔴 **Black-box monitoring** that doesn't explain *why* a model failed
- 🔴 **One-off experiments** disconnected from production systems

### What This System Provides

✅ **Automated Drift Detection** — Continuously monitors data and model performance using statistical tests (KS, PSI, χ², Wasserstein)  
✅ **Intelligent Retraining** — Decides *when* to retrain based on drift, performance degradation, and data availability  
✅ **Stateful Deployment** — Canary, blue-green, and shadow deployment strategies with automatic rollback  
✅ **Champion/Challenger Framework** — A/B testing with statistical significance before promotion  
✅ **White-box Observability** — Real-time pipeline state, event streaming, and full audit trails  
✅ **Production-Ready** — Scaling, resilience, and security built from day one

---

## Why This System Exists

### The ML in Production Reality

Moving an ML model from notebook to production is **not** the end—it's the beginning.

In production:
- **Data changes** (concept drift, data drift, covariate shift)
- **Patterns evolve** (user behavior, market conditions, external factors)
- **Models degrade** (yesterday's accuracy ≠ today's accuracy)
- **Services must adapt** (seamlessly, without downtime)

**This system exists to close the gap** between the deterministic world of experimentation and the dynamic world of production.

### First-Principles Design

The system is built on three core principles:

1. **Observe Everything** — Metrics, data quality, model performance, and system health are continuously tracked
2. **Decide Intelligently** — Decisions to retrain or rollback are data-driven, not manual
3. **Act Safely** — All changes (retraining, deployment, promotion) can be reversed cleanly

---

## Key Capabilities

| Capability | Why It Matters | What It Does |
|-----------|---|---|
| **Data Validation** | Prevents garbage-in-garbage-out | Validates schema, detects anomalies, enforces quality gates |
| **Drift Detection** | Knows when models stop working | Statistical tests on feature distributions and model outputs |
| **Retraining Decision Engine** | Avoids unnecessary compute, ensures action when needed | Multi-factor decision algorithm (performance, drift, staleness, resources) |
| **Experiment Tracking** | Reproduces results, learns from history | Logs parameters, metrics, code versions, and artifacts via MLflow |
| **Champion/Challenger** | Validates improvements before promotion | Statistically significant A/B testing with rollback paths |
| **Canary Deployment** | Minimizes risk | Gradual traffic ramp with health checks and auto-rollback |
| **Real-time Serving** | Low-latency predictions | FastAPI + BentoML with auto-scaling and response logging |
| **Observability & Monitoring** | Know what's happening *now* | SSE streaming, metrics export (Prometheus), event bus architecture |
| **Auto-Rollback** | Instant recovery from failures | Automatic reversion to previous champion if health checks fail |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML LIFECYCLE                         │
└─────────────────────────────────────────────────────────────────────┘

                             Data In
                                ▼
                    ┌──────────────────────┐
                    │  Data Validation     │ ← Great Expectations
                    │  (Schema, Quality)   │
                    └──────────┬───────────┘
                               │
                        [Reject / Accept]
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Drift Detection     │ ← Evidently AI
                    │  (KS, PSI, χ², W)    │
                    └──────────┬───────────┘
                               │
                      [Drift Level Detected]
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Retrain Decision    │
                    │  (Multi-factor)      │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
            [NOT_NEEDED]  [RECOMMENDED]  [REQUIRED/URGENT]
                │              │              │
                └──────────────┴──────────────┘
                               │
                               ▼ [if retraining triggered]
                    ┌──────────────────────┐
                    │  Model Training      │ ← MLflow
                    │  (CV, HPO, Tuning)   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Champion vs         │
                    │  Challenger (A/B)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                ▼ [Rejected]          ▼ [Approved]
            [Keep Champion]    [Proceed to Deploy]
                    │                     │
                    │                     ▼
                    │          ┌──────────────────────┐
                    │          │  Register (STAGING)  │
                    │          │  in Model Registry   │
                    │          └──────────┬───────────┘
                    │                     │
                    │                     ▼
                    │          ┌──────────────────────┐
                    │          │  Deploy              │ ← Canary/Blue-Green
                    │          │  (Gradual Rollout)   │
                    │          └──────────┬───────────┘
                    │                     │
                    │          ┌──────────┴──────────┐
                    │          │                     │
                    │      ▼ [Health OK]         ▼ [Health Fail]
                    │    [Promote]            [Auto-Rollback]
                    │          │                     │
                    └──────────┤                     │
                               │◄────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Serving (Production)│
                    │  (Predictions +      │
                    │   Monitoring)        │
                    └──────────┬───────────┘
                               │
                        [Loop back to: Data In]
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Data Layer** | Schema validation, versioning, quality checks | `src/data/` |
| **Drift Detection** | Statistical analysis of feature/target distributions | `src/drift/` |
| **Decision Engine** | Multi-factor retraining decision logic | `src/decision/` |
| **Training** | Model training, cross-validation, HPO | `src/training/` |
| **Evaluation** | Champion/challenger comparison, statistical testing | `src/evaluation/` |
| **Model Registry** | Artifact storage, versioning, promotion | `src/registry/` |
| **Serving** | Real-time and batch prediction APIs | `src/serving/` |
| **Monitoring** | Metrics collection, alerting, performance tracking | `src/monitoring/` |
| **Orchestration** | Pipeline coordination and scheduling | `src/orchestration/` |
| **Observability** | Event bus, real-time state tracking, audit logs | `src/observability/` |
| **API** | FastAPI endpoints for predictions, monitoring, pipelines | `src/api/` |

---

## Project Structure

```
ml-lifecycle-system/
│
├── 📄 README.md                    # This file
├── 📄 CLAUDE.md                    # Development guidelines
├── 📄 pyproject.toml               # Python package configuration
├── 📄 Makefile                     # Task automation
├── 📄 Dockerfile                   # Container definition
├── 📄 docker-compose.yml           # Multi-container orchestration
├── 📄 .env.example                 # Environment template
|
├── 📁 docker/
│   ├── airflow
|   │   ├── Dockerfile
│   |   └── requirements.txt    # Create synthetic data
│   ├── base
|   │   ├── Dockerfile
│   |   └── requirements.txt
│   |──bentoml
|   |   ├── Dockerfile
│   |── fastapi
|   │   ├── Dockerfile
│   └── mlflow
|
├── 📁 monitoring/
|   |── grafana
|   |   └── provisioning
|   |       └── dashboards
|   |           |── dashboards.yml
|   |           └── prometheus.yml 
|   |──otel
|   |   └── otel-collector-config.yml
|   |── prometheus/
│   |   ├── prometheus.yml         # Deployment health checks
│   |   └── alert_rules.yml               # Alert triggers
|   |── data_quality_monitor.py 
│   ├── drift_monitor.py              # Prometheus metrics registry
│   ├── performance_monitor.py  # Model performance tracking
│   ├── monitoring_pipeline.py         # Deployment health checks
│   └── alerting.py               # Alert triggers
├── 📁 config/
│   ├── settings.py                 # Global configuration loader
│   ├── logging_config.py           # Logging setup
│   └── environments/
│       ├── development.yaml        # Dev settings
│       ├── staging.yaml            # Staging settings
│       └── production.yaml         # Prod settings
│
├── 📁 src/
│   │
│   ├── 📁 data/
│   │   ├── column_config.py            # Load data from sources
│   │   ├── loader.py           # Schema & quality validation
│   │   ├── pipeline_builder.py        # Feature engineering, scaling
│   │   |── processing.py           # Data versioning & tracking
|   |   ├── processing_config.py            # Load data from sources
│   │   ├── processing_report.py           # Schema & quality validation
│   │   ├── transformers.py        # Feature engineering, scaling
│   │   └── validation.py           # Data versioning & tracking
|   |
│   ├── 📁 drift/
│   │   ├── detector.py             # Statistical drift detection
│   │   ├── report_builder.py       # Generate drift reports
│   │   └── statistical_tests.py             # Drift-based alerts
│   │
│   ├── 📁 decision/
│   │   ├── retrain_policy.py       # Retraining decision logic
│   │   └── exceptions.py           # Decision-specific errors
|   |
│   ├── 📁 deployment/
│   │   ├── deployer.py       # Retraining decision logic
│   │   └── traffic_manager.py           # Decision-specific errors
|   |
│   ├── 📁 training/
│   │   ├── trainer.py              # Model training orchestration
│   │   ├── hyperparameter_tuner.py # HPO with Optuna/GridSearch
│   │   └── callbacks.py            # Training callbacks & logging
│   │
│   ├── 📁 evaluation/
│   │   ├── model_validator.py      # Holdout set evaluation
│   │   ├── champion_challenger.py  # A/B testing & comparison
│   │   └── statistical_tests.py    # Bootstrap, McNemar's, etc.
│   │
│   ├── 📁 registry/
│   │   ├── model_registry.py       # MLflow Model Registry interface
│   │   ├── artifact_store.py       # Artifact storage abstraction
│   │   └── model_state.py          # State transitions
│   │
│   ├── 📁 serving/
│   │   ├── bentoml_service.py         # FastAPI prediction server
│   │   ├── predictor.py              # BentoML service definition
│   │
│   ├── 📁 orchestration/
│   │   ├── pipeline.py # Main pipeline coordinator
│   │   ├── dags/
│   │   │   ├── ml_lifecycle_dag.py  # Airflow DAG definition
│   │   │   └── dag_utils.py         # DAG utilities
│   │
│   ├── 📁 observability/
│   │   ├── event_bus.py            # Central event emitter
│   │   ├── prometheus_handler.py         # Per-step execution tracking
│   │   ├── metrics.py       # Global pipeline state machine
│   │
│   ├── 📁 api/
|   |   └── dependencies.py
│   │   ├── app.py                  # FastAPI application
│   │   ├── routes/
│   │   │   ├── predictions.py      # GET /predict, POST /batch-predict
│   │   │   ├── models.py           # GET /models, POST /promote
│   │   │   ├── pipelines.py        # GET /pipeline/status
│   │   │   ├── drift.py            # GET /drift/report
│   │   │   └── health.py           # GET /health, GET /readiness
│   │   ├── middleware/
│   │   │   ├── auth.py             # Authentication middleware
│   │   │   ├── auto_logger.py          # Request/response logging
│   │   │   |── circuit_breaker.py   # Global error handling
|   |   |   └── rate_limiter.py
│   │   └── schemas/
│   │       ├── prediction.py       # Request/response schemas
│   │       └── common.py           # Model metadata schemas
│   │
│   ├── 📁 common/
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   └── enums.py                # Enums (ModelStatus, DecisionLevel)
│   │
│   └── __init__.py
│
├── 📁 tests/
│   ├── 📁 unit/
│   │   ├── test_data_validation.py
│   │   ├── test_drift_detection.py
│   │   ├── test_decision_engine.py
│   │   ├── test_trainer.py
│   │   └── test_serving.py
│
├── 📁 scripts/
│   ├── run_full_lifecycle.py       # CLI: Execute full pipeline
│   ├── setup_infrastructure.sh     # Setup databases, MLflow
│   └── generate_sample_data.py     # Create synthetic data
│
└── 📁 docs/
    ├── DEPLOYMENT.md               # Deployment guide
    ├── ARCHITECTURE.md             # Detailed architecture
    └── API.md                      # API documentation
```

---

## How It Works: Complete Lifecycle

### Stage 1: Data Ingestion & Validation

New data arrives from production systems and is validated against schema, data quality checks, and statistical constraints. Invalid data is rejected with alerts.

**Key Code:**
```python
from src.data.validation import DataValidator

validator = DataValidator(schema_path="config/schema.json")
result = validator.validate(incoming_data)

if not result.is_valid:
    raise DataQualityError(result.errors)
```

---

### Stage 2: Drift Detection

Statistical tests compare current data against reference baseline. Detects concept drift, data drift, and covariate shift using KS test, PSI, χ², and Wasserstein distance.

**Drift Scores:**
| PSI Score | Interpretation | Action |
|-----------|---|---|
| < 0.1 | No significant drift | Continue monitoring |
| 0.1 - 0.25 | Small drift | Monitor closely |
| > 0.25 | Significant drift | Consider retraining |

**Key Code:**
```python
from src.drift.detector import DriftDetector

detector = DriftDetector(reference_data=baseline)
report = detector.detect(current_data)

print(f"PSI: {report.psi}")
print(f"Drifted Features: {report.drifted_features}")
```

---

### Stage 3: Retraining Decision

Evaluates performance degradation, drift severity, data staleness, and data readiness. Multi-factor algorithm produces decision:

- `NOT_NEEDED` — Continue monitoring
- `RECOMMENDED` — Optional retraining
- `REQUIRED` — Schedule retraining
- `URGENT` — Immediate retraining + alert

**Key Code:**
```python
from src.decision.retrain_policy import RetrainingDecisionEngine

engine = RetrainingDecisionEngine()
decision = engine.decide(
    performance_drop=metrics["accuracy_drop"],
    drift_scores=drift_report,
    last_training_date=metadata["training_date"],
    available_samples=len(new_data)
)

print(f"Decision: {decision.level}")  # REQUIRED, URGENT, etc.
```

---

### Stage 4: Model Training

Trains model with cross-validation and hyperparameter optimization. All experiments tracked in MLflow with parameters, metrics, code version, and artifacts.

---

### Stage 5: Champion vs. Challenger

Compares newly trained model against currently deployed model using holdout test set. Statistical significance testing ensures only improvements are promoted.

---

### Stage 6: Deployment

Supports three strategies:

- **Canary** (Recommended) — Gradual rollout: 5% → 25% → 50% → 100%
- **Blue-Green** — Atomic switch between old and new
- **Shadow** — New model runs in parallel, predictions logged but not served

---

### Stage 7: Promotion & Auto-Rollback

Model moves through states: STAGING → serving traffic → CHAMPION → production. If health checks fail, automatic rollback to previous champion.

---

### Stage 8: Production Serving

Serves predictions in real-time. Every prediction logged for monitoring. Metrics continuously collected. Loop feeds back to Stage 1.

---

## Pipeline Execution

### Airflow DAG Orchestration

The complete ML lifecycle is orchestrated through Apache Airflow DAGs that manage task dependencies, retries, and monitoring.

**Pipeline Run Status (Production Success):**

<div align="center">

![Airflow DAG Run Status](https://i.imgur.com/scheduled_2026-06-09T00_00_00.png)

| Metric | Value |
|--------|-------|
| **Status** | ✅ Success |
| **Run Type** | Scheduled |
| **Duration** | 00:01:26 |
| **Start Date** | 2026-06-09 05:30:00 |
| **Run Date** | 2026-06-09 21:20:52 |
| **End Date** | 2026-06-09 21:22:19 |
| **DAG Versions** | v1, v2, v3 |

</div>

### Complete Task Execution Flow

All pipeline tasks executed successfully with proper branching logic:

<div align="center">

![Pipeline Task Execution Tree](https://i.imgur.com/pipeline-task-tree.png)

</div>

**Execution Steps (All Passed ✅):**

1. ✅ **load_data** — Load training data from sources
2. ✅ **validate_production_data** — Validate schema and quality  
3. ✅ **process_both_datasets** — Feature engineering and preprocessing
4. ✅ **detect_drift** — Statistical drift detection
5. ✅ **make_retrain_decision** — Multi-factor decision algorithm
6. ✅ **branch_on_decision** — Branch execution based on decision
7. 🔄 **no_retrain_needed** — Skip to monitoring if no drift
8. ✅ **train_model** — Train challenger model with CV/HPO
9. ✅ **champion_challenger_evaluation** — Statistically compare models
10. ✅ **check_approval_required** — Evaluate improvement metrics
11. ✅ **approval_branch** — Route based on approval status
12. 🔄 **challenger_rejected** — Keep current champion if not better
13. ✅ **register_model_staging** — Register in model registry
14. ✅ **deploy_model** — Canary deployment orchestration
15. ✅ **promote_to_champion** — Promote to production if healthy
16. ✅ **shift_reference** — Update baseline for next cycle
17. ✅ **pipeline_report** — Generate execution report

---

## Monitoring & Observability

### Real-Time Prometheus Metrics

The system continuously exports detailed metrics for performance monitoring and alerting:

<div align="center">

![Prometheus Metrics Dashboard](https://i.imgur.com/prometheus-metrics-dashboard.png)

</div>

**Metrics Being Tracked:**
- 🟢 **BentoML Service Requests** (Green Line) — Successful prediction requests (HTTP 200)
- 🟡 **Health Check Endpoints** (Yellow Area) — /health endpoint responses
- 🟣 **Deployment Routes** (Purple/Magenta) — /predict and other API endpoints
- 🔴 **Error Responses** — HTTP 400, 405, 500 error tracking

**Timeline Analysis:**
- Steady baseline traffic at ~150 requests
- Gradual traffic increase from 23:10 to 02:00 (ramping up to ~400+ requests)
- Consistent performance across all endpoints
- No errors or dropouts observed
- Service health maintained throughout execution window

### Key Metrics Exported

```bash
# Latency (p99)
histogram_quantile(0.99, rate(model_serving_latency_ms[5m]))

# Throughput
rate(predictions_total[1m])

# Retraining duration
retraining_duration_seconds

# Drift PSI
drift_psi_score

# Model serving errors
rate(model_serving_errors_total[5m])

# Bentoml request success rate
rate(bentoml_service_request_duration_seconds_count{http_status="200"}[5m])
```

### Grafana Dashboards

Pre-built dashboards:
1. ML Lifecycle Overview
2. Model Performance  
3. Data Quality
4. Drift Detection
5. System Health
6. Deployment Progress

Access at `http://localhost:3000`

---

## Quick Start Guide

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (recommended)
- 4GB RAM (8GB+ for training)

### Option 1: Local Setup

```bash
# Clone & install
git clone https://github.com/maitimeraki/ML-LIFECYCLE-SYSTEM.git
cd ML-LIFECYCLE-SYSTEM

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

make install

# Configure
cp .env.example .env
# Edit .env with your settings

# Initialize
make setup-infrastructure

# Run
make pipeline
```

### Option 2: Docker Setup (Recommended)

```bash
cp .env.example .env
make docker-build
make docker-up

# Services available at:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
# - Airflow: http://localhost:8080
```

### Verify Installation

```bash
curl http://localhost:8000/health

# Expected: { "status": "healthy", "components": {...} }
```

---

## Configuration

### Environment Variables

```bash
ENVIRONMENT=development
MLFLOW_TRACKING_URI=http://localhost:5000
DATA_SOURCE_PATH=/data/training.csv
DRIFT_PSI_THRESHOLD=0.25
DEPLOYMENT_STRATEGY=canary
LOG_LEVEL=INFO
API_PORT=8000
```

See `.env.example` for complete list.

---

## API Reference

### Core Endpoints

```bash
# Health checks
GET /health                    # Liveness
GET /readiness                 # Readiness
GET /health/detailed           # Full system status

# Predictions
POST /predict                  # Single prediction
POST /batch-predict            # Batch predictions

# Models
GET /models                    # List models
GET /models/champion           # Get champion
POST /models/promote           # Promote challenger
POST /models/rollback          # Rollback to previous

# Monitoring
GET /drift/report              # Drift analysis
GET /metrics                   # System metrics
GET /pipeline/status           # Pipeline status
GET /pipeline/history          # Past runs

# Real-time
GET /events                    # SSE stream of events
```

---

## Deployment Strategies

### Canary Deployment (Recommended)

Gradually shift traffic from Champion to Challenger with health checks at each stage.

```yaml
strategy: canary
stages:
  - percentage: 5
    duration_minutes: 5
  - percentage: 25
    duration_minutes: 10
  - percentage: 50
    duration_minutes: 15
  - percentage: 100
```

**Risk:** ⭐ Low  
**Time to Production:** 30-40 minutes

---

### Blue-Green Deployment

Atomic switch between old (Blue) and new (Green) versions.

**Risk:** ⭐⭐ Medium  
**Time to Production:** 2-5 minutes

---

### Shadow Deployment

New model runs in parallel; predictions logged but not served.

**Risk:** ⭐ None  
**Validation Duration:** Hours to days

---

## Troubleshooting

### Common Issues

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| **Pipeline hangs at training** | Resource exhaustion or stuck job | Check logs, increase timeout, reduce dataset |
| **Unexpected rollback** | Health check failure | Review which metric failed, check logs |
| **Drift false positives** | Aggressive thresholds | Adjust PSI/KS thresholds in .env |
| **Slow predictions** | Model too large or under-provisioned | Increase API workers, optimize model size |
| **Champion never promotes** | Strict comparison thresholds | Review evaluation criteria, ensure test set is representative |

---

## Contributing

We welcome contributions! 

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Add tests
4. Run `make format && make lint`
5. Submit pull request

### Code Standards

- PEP 8 (black, ruff)
- Type hints (mypy)
- 80% test coverage minimum
- Docstrings on public functions

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework
- [MLflow](https://mlflow.org/) — Experiment tracking
- [Apache Airflow](https://airflow.apache.org/) — Workflow orchestration
- [Evidently AI](https://www.evidentlyai.com/) — Drift detection
- [Great Expectations](https://greatexpectations.io/) — Data validation
- [Optuna](https://optuna.org/) — Hyperparameter optimization
- [BentoML](https://www.bentoml.com/) — Model serving
- [Prometheus](https://prometheus.io/) — Metrics collection
- [Grafana](https://grafana.com/) — Visualization

---

## Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

---

**Made with ❤️ for production ML teams**  
Last updated: June 2026
