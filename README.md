# Production ML Lifecycle

## Overview

This document outlines the end-to-end production machine learning lifecycle, from data ingestion to model serving and continuous monitoring. The workflow ensures model reliability, performance, and automatic retraining decisions based on data drift and model metrics.

---

## Project Structure
ml_lifecycle_platform/
│
├── README.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── logging_config.py
│   └── environments/
│       ├── development.yaml
│       ├── staging.yaml
│       └── production.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── validation.py              # ← Great Expectations
│   │   ├── preprocessing.py
│   │   └── versioning.py
│   │
│   ├── drift/
│   │   ├── __init__.py
│   │   ├── detector.py                # ← Evidently
│   │   ├── report_builder.py          # ← Evidently Reports
│   │   └── alerting.py
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   └── retrain_policy.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                 # ← MLflow
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── validator.py               # ← MLflow
│   │   └── champion_challenger.py
│   │
│   ├── registry/
│   │   ├── __init__.py
│   │   └── model_registry.py          # ← MLflow Registry
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── service.py                 # ← BentoML
│   │   ├── bentofile.yaml
│   │   └── runners.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # ← Prometheus
│   │   └── performance_monitor.py
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── pipeline.py                # ← Airflow DAG
│   │   └── dags/
│   │       ├── __init__.py
│   │       ├── ml_lifecycle_dag.py    # ← Airflow DAG definition
│   │       └── dag_utils.py
│   │
│   ├── observability/                 # NEW - White-box tracking
│   │   ├── __init__.py
│   │   ├── event_bus.py               # Central event emitter
│   │   ├── step_tracker.py            # Per-step tracking
│   │   ├── pipeline_state.py          # Global pipeline state
│   │   └── formatters.py              # Standardized output formats
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py
│   │   │   ├── models.py
│   │   │   ├── pipelines.py
│   │   │   ├── drift.py
│   │   │   ├── observability.py       # NEW - SSE/WebSocket
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── pipeline_events.py     # Frontend-ready schemas
│   │
│   └── common/
│       ├── __init__.py
│       ├── exceptions.py
│       ├── enums.py
│       └── utils.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
│
└── scripts/
    ├── run_pipeline.py
    └── setup_infrastructure.sh

## Lifecycle Stages

### 1. New Data (Real-World)

- Incoming data from production systems, user interactions, or external sources
- Raw, unlabeled data streams requiring validation before processing

### 2. Validate (Schema + Quality)

**Checks performed:**
- Schema validation (data types, required fields, constraints)
- Data quality checks (missing values, outliers, duplicates)
- Statistical validation (distributions, ranges)

**Outcome:**
- ✅ **Valid** → Proceed to drift detection
- ❌ **Invalid** → Reject data, log error, trigger alert

---

### 3. Drift Detection

**Statistical tests used:**
| Test | Use Case |
|------|----------|
| **KS Test (Kolmogorov-Smirnov)** | Continuous feature distribution comparison |
| **PSI (Population Stability Index)** | Overall feature stability scoring |
| **Chi-Square (χ²)** | Categorical feature drift detection |
| **Wasserstein Distance** | Earth mover's distance for distribution shifts |

**Output:** Drift scores per feature and overall drift severity

---

### 4. Retrain Decision Engine

**Decision factors:**
- Model performance degradation
- Drift severity (feature-wise & overall)
- Data staleness (time since last training)
- Volume of new validated data available
- Resource availability (compute, storage)

**Decision outputs:**
| Status | Action |
|--------|--------|
| **NOT_NEEDED** | Continue monitoring, no action required |
| **RECOMMENDED** | Schedule retraining, optional |
| **REQUIRED** | Initiate retraining pipeline |
| **URGENT** | Immediate retraining + alert on-call |

---

### 5. Model Training

**Components:**
- **Cross-Validation (CV)** : K-fold or time-series split validation
- **Hyperparameter Optimization (HPO)** : Grid search, random search, or Bayesian optimization
- **Experiment Tracking** : MLflow, Weights & Biases, or Neptune
  - Log parameters, metrics, artifacts, and code version

**Output:** Trained model artifacts + training metadata

---

### 6. Champion vs. Challenger

Compare currently deployed model (Champion) against newly trained model (Challenger).

**Evaluation criteria:**
- Holdout test set performance (accuracy, F1, AUC, etc.)
- Statistical significance testing (bootstrap, McNemar's test)
- Business metric impact simulation
- Inference latency / resource comparison

**Decision:**
| Result | Action |
|--------|--------|
| **Challenger REJECTED** | Keep Champion, log results, discard Challenger |
| **Challenger APPROVED** | Proceed to deployment |

---

### 7. Deployment Strategies

| Strategy | Description | Risk Level |
|----------|-------------|------------|
| **Canary** | Gradual rollout to small % of traffic, monitor, ramp up | Low |
| **Blue-Green** | Switch traffic entirely from old (blue) to new (green) version | Medium |
| **Shadow** | New model runs in parallel, predictions logged but not served | Minimal |

**Promotion criteria:**
- Health checks pass (latency, error rate, prediction distribution)
- No regression on key business metrics
- Manual approval (if configured)

---

### 8. Serving (Production)

**Features:**
- Real-time or batch prediction serving
- Low-latency inference with auto-scaling
- Response logging for future retraining loops
- Continuous metric collection (latency, throughput, error rate)

**Monitoring loop:**
- Prediction distribution tracking
- Feature importance monitoring
- Input/output schema validation
- Performance degradation alerts

---

## Auto-Rollback Mechanism

If health check **FAILS** after deployment:
1. Automatically revert to previous model version
2. Log rollback event and metrics
3. Trigger alert for manual investigation
4. Pause further retraining attempts until issue resolved

If health **OK**:
1. Fully promote new model
2. Retire or archive previous champion
3. Continue monitoring and loop back to **Step 1** (New Data)

---

## Key Metrics to Track

| Category | Metrics |
|----------|---------|
| **Data Quality** | Missing rate, schema violations, duplicate ratio |
| **Drift** | PSI, KS statistic, χ² p-value, Wasserstein distance |
| **Model Performance** | Accuracy, precision, recall, F1, AUC-ROC, log loss |
| **Business Impact** | Conversion rate, revenue lift, error reduction |
| **System Health** | p99 latency, throughput, error rate, CPU/memory usage |

---

## Tools Recommendation

| Component | Recommended Tools |
|-----------|-------------------|
| Drift Detection | Evidently AI, Alibi Detect, DeepChecks |
| Experiment Tracking | MLflow, Weights & Biases, Neptune |
| HPO | Optuna, Hyperopt, Ray Tune |
| Deployment | KServe, Seldon Core, BentoML, TensorFlow Serving |
| Monitoring | Prometheus + Grafana, Datadog, New Relic |

---

## Quick Reference Flow
                    ┌──────────────────────────────────────────────────┐
                    │           PRODUCTION ML LIFECYCLE                │
                    └──────────────────────────────────────────────────┘

    ┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
    │  New Data    │────▶│  Validate   │────▶│  Drift Detection │
    │  (Real-World)│     │  (Schema +  │     │  (KS, PSI, χ²,  │
    │              │     │   Quality)  │     │   Wasserstein)   │
    └─────────────┘     └──────┬──────┘     └────────┬─────────┘
                               │                      │
                        ❌ Reject if                   │
                        invalid                       ▼
                                            ┌──────────────────┐
                                            │  Retrain Decision│
                                            │  Engine          │
                        ┌───────────────────│  (Performance +  │
                        │                   │   Drift + Stale  │
                        │                   │   + Data Ready)  │
                        │                   └────────┬─────────┘
                        │                            │
                    NOT_NEEDED              RECOMMENDED/REQUIRED/URGENT
                    (Continue                        │
                     monitoring)                     ▼
                                            ┌──────────────────┐
                                            │  Model Training  │
                                            │  (CV, HPO,       │
                                            │   Experiment     │
                                            │   Tracking)      │
                                            └────────┬─────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Champion vs     │
                                            │  Challenger      │
                                            │  (Statistical    │
                                            │   Significance)  │
                                            └────────┬─────────┘
                                                     │
                                        ┌────────────┴───────────┐
                                        │                        │
                                  Challenger             Challenger
                                  REJECTED               APPROVED
                                  (Keep Champion)             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │  Deployment      │
                                                    │  (Canary /       │
                                                    │   Blue-Green /   │
                                                    │   Shadow)        │
                                                    └────────┬─────────┘
                                                             │
                                                    ┌────────┴─────────┐
                                                    │                  │
                                               Health OK          Health FAIL
                                               (Promote)          (Auto-Rollback)
                                                    │
                                                    ▼
                                            ┌──────────────────┐
                                            │  SERVING         │
                                            │  (Predictions +  │
                                            │   Monitoring +   │
                                            │   Loop Back) ◀───┘
                                            └──────────────────┘



## Model State Transitions & Rollback Strategy
champion_challenger → APPROVED
        │
        ▼
register(STAGING)        ← Task 9
        │                  Model exists in registry
        │                  NOT serving traffic
        │                  Rollback path: clear
        │
        ▼
deploy(canary/blue-green) ← Task 10  [retries=3]
        │                  Model IS serving traffic
        │                  Registry still shows STAGING
        │                  If fail: stays STAGING, champion still serves
        │
        ▼
promote(CHAMPION)         ← Task 11
        │                  Registry and production NOW in sync
        │                  Old champion → ARCHIVED
        │                  Rollback: registry.rollback_to_previous()
        │
        ▼
shift_reference()         ← Task 12
                           Production data saved as next reference
                           Cycle complete

### Strategy 9 – Register (STAGING)

| Property | Value |
|----------|-------|
| Action | Register model in registry |
| Status | STAGING |
| Serving Traffic | No |
| Rollback Path | Clear registry entry |

**Note:** Model exists in registry but is NOT serving traffic.

---

### Strategy 10 – Deploy (Canary / Blue-Green)

| Property | Value |
|----------|-------|
| Action | Deploy with canary/blue-green strategy |
| Retries | 3 attempts |
| Status | STAGING (registry unchanged) |
| Serving Traffic | Yes |
| If Fail | Stays STAGING, old champion still serves |

**Note:** Model IS serving traffic. Registry still shows STAGING.

---

### Strategy 11 – Promote (CHAMPION)

| Property | Value |
|----------|-------|
| Action | Promote model to champion |
| Status | CHAMPION |
| Old Champion | Moves to ARCHIVED |
| Rollback | registry.rollback_to_previous() |

**Note:** Registry and production are now in sync.

---

### Strategy 12 – Shift Reference

| Property | Value |
|----------|-------|
| Action | Save production data as next reference |
| Status | Cycle complete |
| If Fail | Next cycle uses old reference |

**Note:** Production data saved for future drift detection.

---

## Rollback States

| Failure At | Registry Status | Who Serves? | Recovery |
|------------|----------------|-------------|----------|
| After Task 9 fails | STAGING | Old champion | Retry Task 9 |
| After Task 10 fails | STAGING | Old champion | Retry Task 10 |
| After Task 11 fails | STAGING | New model (serving) | Retry Task 11 |
| After Task 12 fails | CHAMPION | New champion | Retry Task 12 |

## Key Guarantees

- ✅ Every failure is clean and recoverable
- ✅ Every state is auditable
- ✅ Old champion always available until promotion succeeds
- ✅ No deadlock states