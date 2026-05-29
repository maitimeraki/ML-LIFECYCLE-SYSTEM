# dags/ml_lifecycle_dag.py
"""
ML Lifecycle Pipeline — Airflow DAG.

This IS the pipeline.py. Transformed completely.
Every stage from pipeline.py is now an Airflow task.

Benefits over pipeline.py approach:
  ✅ Each stage retried independently (Stage 5 fail → retry Stage 5 only)
  ✅ Airflow UI shows per-stage progress, logs, duration
  ✅ Human approval = ExternalTaskSensor (Airflow native, not custom code)
  ✅ Schedule managed by Airflow (@daily, cron, event-driven)
  ✅ XCom carries only lightweight metadata (not full DataFrames)
  ✅ Data passed via parquet files on shared storage (S3/local)
  ✅ Parallel tasks where possible (e.g. champion_challenger + notify)
  ✅ SLA monitoring, email alerts, Slack notifications built-in
  ✅ Backfill support for missed runs
  ✅ No manual state management (PipelineContext replaced by XCom)
Stage → Task mapping (1:1 with old pipeline.py):
    _stage_validate              → validate_production_data
    _stage_process_both          → process_both_datasets
    _stage_drift_detection       → detect_drift
    _stage_retrain_decision      → make_retrain_decision
    _stage_train                 → train_model
    _stage_champion_challenger   → champion_challenger_evaluation
    _stage_deploy                → deploy_model
    _stage_promote_shift_ref     → promote_and_shift_reference
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Annotated, Dict

import numpy as np
import pandas as pd
from airflow.sdk import DAG
from airflow.sdk import task
from airflow.sdk import Variable
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.sdk import TriggerRule

from sklearn.base import BaseEstimator
from src.training.trainer import ModelProtocol, T

from typing import TYPE_CHECKING, Any, TypeVar, ParamSpec

if TYPE_CHECKING:
    # During static analysis (Pylance), pretend XComArg is Any
    # This silences all "XComArg not assignable" errors
    XComArg = Any
else:
    # At runtime, this module doesn't exist in Airflow 3.2.1
    # We create a dummy that never gets used
    class _XComArgDummy:
        pass
    XComArg = _XComArgDummy

logger = logging.getLogger("ml_platform.dag.lifecycle")

# ── DAG-level constants ────────────────────────────────────────────────────────

MODEL_ID       = Variable.get("model_id",        default="customer_churn_model")
TARGET_COLUMN  = Variable.get("target_column",   default="target")
TMP_DIR        = Variable.get("tmp_dir",         default="/tmp/ml_platform")
ARTIFACT_DIR   = Variable.get("artifact_dir",    default="artifacts")
MLFLOW_URI     = Variable.get("mlflow_uri",      default="http://localhost:5000")
MIN_SAMPLES    = int(Variable.get("min_samples", default="1000"))

DEFAULT_ARGS = {
    "owner":             "ml-platform",
    "depends_on_past":   False,
    "email":             ["ml-alerts@company.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ============================================================
# TYPE-SAFE TASK DEFINITIONS
# ============================================================

# Use TypeVar for generic return types - Pylance respects these
T = TypeVar('T', bound=BaseEstimator)
P = ParamSpec("P")


def airflow_task(
    python_callable: Any = None,
    *,
    multiple_outputs: bool | None = None,
    **kwargs: Any,
) -> Any:
    """
    Wrapper that preserves original function signatures for Pylance.
    Returns the decorated function with original type hints intact.
    """
    decorator = task(multiple_outputs=multiple_outputs, **kwargs)
    if python_callable is not None:
        return decorator(python_callable)
    return decorator

# ── Helper: shared storage paths ──────────────────────────────────────────────

def _tmp_path(run_id: str, name: str) -> str:
    """Deterministic temp file path per DAG run."""
    path = Path(TMP_DIR) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path / name)


def _artifact_path(model_id: str, *parts: str) -> str:
    path = Path(ARTIFACT_DIR) / model_id
    for p in parts:
        path = path / p
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_lifecycle_pipeline",
    default_args=DEFAULT_ARGS,
    description=(
        "Full ML lifecycle: validate → process → drift → decision "
        "→ train → evaluate → approve → deploy → shift reference"
    ),
    schedule="@daily",
    start_date=datetime.now(timezone.utc) - timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "production", "lifecycle"],
    doc_md="""
    ## ML Lifecycle Pipeline

    **Each task = one pipeline stage. Airflow manages everything.**

    | Task | Stage | Retry Safe |
    |------|-------|------------|
    | `load_data` | Load raw data from warehouse | ✅ |
    | `validate_production_data` | GE validation on production data | ✅ |
    | `process_both_datasets` | AutoConfig + fit(ref) + transform(ref+prod) | ✅ |
    | `detect_drift` | KS+PSI on processed features | ✅ |
    | `make_retrain_decision` | Multi-signal scoring | ✅ |
    | `branch_on_decision` | Route: retrain vs skip | ✅ |
    | `train_model` | MLflow-tracked training | ✅ |
    | `champion_challenger_evaluation` | Statistical approval gate | ✅ |
    | `wait_for_human_approval` | ExternalTaskSensor (Airflow UI) | ✅ |
    | `deploy_model` | Canary/Blue-green deployment | ✅ |
    | `promote_and_shift_reference` | Registry promote + new reference | ✅ |
    | `no_retrain_needed` | Skip path | ✅ |

    **Monitoring:** `http://localhost:3001` (Grafana)
    **Experiments:** `http://localhost:5000` (MLflow)
    """,
    ) as dag:

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 1: Load Data
    # Airflow responsibility: pull data from warehouse/S3
    # Returns: file paths via XCom (not the data itself)
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="load_data")
    def load_data(**context) -> dict[str, Any]:
        """
        Load production data from data warehouse.
        Load reference data from previous cycle's saved artifact.

        Returns metadata + file paths via XCom.
        DataFrames saved to shared temp storage (not XCom — too large).
        """
        run_id = context["run_id"]
        logger.info(f"Loading data for run: {run_id}")

        # ── PRODUCTION: Replace with real data source ──────────────────────
        # Examples:
        #   df = pd.read_parquet("s3://bucket/production/2024-01/*.parquet")
        #   df = spark_client.sql("SELECT * FROM feature_store WHERE dt=today")
        #   df = bigquery_client.query("SELECT * FROM ml_features LIMIT 100000")
        # ──────────────────────────────────────────────────────────────────────

        drift_level = float(
            Variable.get("drift_simulation_level", default="0.3")
        )

        rng_prod = np.random.RandomState(
            int(datetime.now().timestamp()) % 100000
        )
        n_prod = int(Variable.get("production_sample_size", default="2000"))

        production_df = pd.DataFrame({
            "feature_1": rng_prod.normal(10 + drift_level, 2, n_prod),
            "feature_2": rng_prod.normal(5 + drift_level * 0.5, 1.5, n_prod),
            "feature_3": rng_prod.exponential(2 + drift_level * 0.3, n_prod),
            "feature_4": rng_prod.choice(["A", "B", "C", "D"], n_prod),
            "feature_5": rng_prod.uniform(0, 100 + drift_level * 15, n_prod),
        })
        logits             = (
            0.3 * production_df["feature_1"]
            - 0.5 * production_df["feature_2"]
        )
        production_df[TARGET_COLUMN] = (
            logits > logits.median()
        ).astype(int) # Creation of the target columns of produciton df

        # Save production data to shared storage
        prod_path = _tmp_path(run_id, "production_raw.parquet")
        production_df.to_parquet(prod_path, index=False) # Stored production df in temporary storage for the DAG run

        # Load reference from previous cycle OR use historical data
        ref_artifact = _artifact_path(
            MODEL_ID, "reference", "reference_raw.parquet"
        )

        if Path(ref_artifact).exists():
            reference_df = pd.read_parquet(ref_artifact)
            ref_source   = "previous_cycle"
            logger.info(
                f"Reference loaded from previous cycle: "
                f"{len(reference_df)} rows"
            )
        else:
            # First run: use historical data. Use else to avoid accidentally using old reference after first run. Because after first run, reference will be saved to artifact path and should be used for next runs.
            rng_ref    = np.random.RandomState(42)
            n_ref      = 5000
            reference_df = pd.DataFrame({
                "feature_1": rng_ref.normal(10, 2, n_ref),
                "feature_2": rng_ref.normal(5, 1.5, n_ref),
                "feature_3": rng_ref.exponential(2, n_ref),
                "feature_4": rng_ref.choice(["A", "B", "C", "D"], n_ref),
                "feature_5": rng_ref.uniform(0, 100, n_ref),
            })
            ref_logits         = (
                0.3 * reference_df["feature_1"]
                - 0.5 * reference_df["feature_2"]
            )
            reference_df[TARGET_COLUMN] = (
                ref_logits > ref_logits.median()
            ).astype(int)
            ref_source = "historical_initial"
            logger.info("First run: using historical reference data")

        ref_path = _tmp_path(run_id, "reference_raw.parquet")
        reference_df.to_parquet(ref_path, index=False)

        result = {
            "run_id":          run_id,
            "production_path": prod_path,
            "reference_path":  ref_path,
            "production_rows": len(production_df),
            "reference_rows":  len(reference_df),
            "reference_source": ref_source,
            "drift_level":     drift_level,
        }

        logger.info(
            f"Data loaded: production={len(production_df)}, "
            f"reference={len(reference_df)}, source={ref_source}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 2: Validate Production Data
    # Retries independently — if this fails, only this task reruns
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="validate_production_data")
    def validate_production_data(
        load_result: Annotated[dict[str, Any], XComArg],
        **context,
    ) -> dict[str, Any]:
        """
        Great Expectations validation on production data only.
        Reference data assumed clean (it was validated in a previous cycle).
        Airflow retry: if GE engine fails → only this task retries.
        Not the whole pipeline.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform") # It is needed to import from ml_platform in Airflow tasks because Airflow tasks run in a different context where the ml_platform code is not in the default path. By adding it to sys.path, we can import our custom modules without issues.

        from src.data.validation import DataValidator, DatasetSchema, ColumnSchema
        from src.observability.event_bus import event_bus
        from src.observability.prometheus_handler import register_prometheus_handler

        register_prometheus_handler(event_bus)

        run_id    = load_result["run_id"]
        prod_path = load_result["production_path"]
        ref_path  = load_result["reference_path"]

        production_df = pd.read_parquet(prod_path)
        reference_df = pd.read_parquet(ref_path)

        logger.info(
            f"Validating production data: {len(production_df)} rows"
        )

        schema = DatasetSchema(  # It makes the bottleneck for validation, so we keep it simple. In real life, this would be more complex and possibly loaded from a config file or built dynamically.
            columns=[
                ColumnSchema("feature_1", "float64"),
                ColumnSchema("feature_2", "float64"),
                ColumnSchema("feature_3", "float64"),
                ColumnSchema("feature_4", "object",
                             allowed_values=["A", "B", "C", "D"]),
                ColumnSchema("feature_5", "float64", min_value=0),
                ColumnSchema(TARGET_COLUMN, "int64",
                             allowed_values=[0, 1]),
            ],
            min_rows=100,
        )

        validator = DataValidator(schema=schema)
        report    = validator.validate(
            df=production_df,
            reference_df=reference_df,
            dataset_id=f"production_{run_id}",
            pipeline_run_id=run_id,
            model_id=MODEL_ID,
        )

        if not report.is_valid:
            raise ValueError(
                f"Production data validation FAILED: "
                f"{report.errors[:3]}"
            )

        logger.info(
            f"Validation passed: {report.success_rate:.1%} "
            f"expectations met"
        )

        return {
            "run_id":        run_id,
            "is_valid":      report.is_valid,
            "success_rate":  report.success_rate,
            "production_path": prod_path,
            "reference_path":  ref_path,
            "production_rows": load_result["production_rows"],
            "reference_rows":  load_result["reference_rows"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 3: Process Both Datasets
    # AutoConfig from reference, fit on reference, transform both
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="process_both_datasets")
    def process_both_datasets(
        validation_result: dict[str, Any],
        **context,
    ) -> dict[str, Any]:
        """
        Automatic processing of BOTH datasets.

        AutoConfigBuilder inspects reference → builds config automatically.
        Processor fits on reference → transforms both to same feature space.
        Saves processed DataFrames to shared storage.

        Airflow retry: if processing fails → retry from here, not from Task 1.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.data.processing import ProductionDataProcessor

        run_id   = validation_result["run_id"]
        ref_df   = pd.read_parquet(validation_result["reference_path"])
        prod_df  = pd.read_parquet(validation_result["production_path"])

        logger.info(
            f"Processing both datasets: "
            f"ref={len(ref_df)}, prod={len(prod_df)}"
        )

        processor = ProductionDataProcessor(
            model_id=MODEL_ID,
            pipeline_run_id=run_id,
            target_column=TARGET_COLUMN,
            id_columns=[],
            onehot_threshold=10,
            skewness_threshold=2.0,
            max_null_rate_to_drop=0.70,
        )
        
        # To save the processor artifact.
        processor_save_path = _artifact_path(
            MODEL_ID, "processors", f"{run_id}_processor.joblib"
        )

        processing_result = processor.fit_transform(
            reference_df=ref_df,
            production_df=prod_df,
            save_path=processor_save_path,
        )

        # Save processed DataFrames to shared storage
        ref_proc_path  = _tmp_path(run_id, "reference_processed.parquet")
        prod_proc_path = _tmp_path(run_id, "production_processed.parquet")

        processing_result.reference_processed.to_parquet(
            ref_proc_path, index=False
        )
        processing_result.production_processed.to_parquet(
            prod_proc_path, index=False
        )

        logger.info(
            f"Processing complete: "
            f"ref {processing_result.reference_rows_in}→"
            f"{processing_result.reference_rows_out} | "
            f"prod {processing_result.production_rows_in}→"
            f"{processing_result.production_rows_out} | "
            f"features={processing_result.n_features_out}"
        )

        return {
            "run_id":               run_id,
            "ref_processed_path":   ref_proc_path,
            "prod_processed_path":  prod_proc_path,
            "production_path":      validation_result["production_path"],
            "processor_path":       processor_save_path,
            "n_features":           processing_result.n_features_out,
            "feature_names":        processing_result.feature_names,
            "ref_rows_out":         processing_result.reference_rows_out,
            "prod_rows_out":        processing_result.production_rows_out,
            "warnings":             processing_result.warnings[:10],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 4: Detect Drift
    # Receives processed DataFrames — same feature space guaranteed
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="detect_drift")
    def detect_drift(
        processing_result: dict[str, Any],
        **context,
    ) -> dict[str, Any]:
        """
        Drift detection on PROCESSED DataFrames.
        Both are in same sklearn feature space.
        Features auto-detected from DataFrame columns.

        Airflow retry: isolated — only drift detection reruns on failure.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.drift.detector import DriftDetector

        run_id       = processing_result["run_id"]
        ref_proc_df  = pd.read_parquet(
            processing_result["ref_processed_path"]
        )
        prod_proc_df = pd.read_parquet(
            processing_result["prod_processed_path"]
        )

        logger.info(
            f"Drift detection: "
            f"ref={len(ref_proc_df)}, prod={len(prod_proc_df)}, "
            f"features={len(ref_proc_df.columns)}"
        )

        detector = DriftDetector(
            model_id=MODEL_ID,
            pipeline_run_id=run_id,
        )

        drift_report = detector.detect(
            reference_processed=ref_proc_df,
            production_processed=prod_proc_df,
            target_column=TARGET_COLUMN,
            report_id=f"drift_{run_id}",
        )

        logger.info(
            f"Drift: severity={drift_report.overall_severity.value}, "
            f"score={drift_report.overall_drift_score:.3f}, "
            f"drifted={drift_report.features_drifted}/"
            f"{drift_report.features_total}"
        )

        return {
            "run_id":                 run_id,
            "overall_severity":       drift_report.overall_severity.value,
            "overall_drift_score":    drift_report.overall_drift_score,
            "features_drifted":       drift_report.features_drifted,
            "features_total":         drift_report.features_total,
            "drift_proportion":       drift_report.drift_proportion,
            "concept_drift_detected": drift_report.concept_drift_detected,
            "recommendations":        drift_report.recommendations,
            "feature_results": [
                fr.to_dict() for fr in drift_report.feature_results
            ],
            # Pass through for downstream tasks
            "prod_processed_path":   processing_result["prod_processed_path"],
            "production_path":       processing_result["production_path"],
            "processor_path":        processing_result["processor_path"],
            "n_features":            processing_result["n_features"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 5: Make Retrain Decision
    # Multi-signal: drift + performance + staleness + data volume
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="make_retrain_decision")
    def make_retrain_decision(
        drift_result: dict[str, Any],
        **context,
    ) -> dict[str, Any]:
        """
        Multi-signal retrain decision.

        Signals:
          1. Drift severity (from previous task)
          2. Model performance degradation (from Airflow Variable / monitoring)
          3. Time since last training (from Airflow Variable)
          4. New data volume (from processing result)

        Returns decision + priority score for BranchPythonOperator.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.decision.retrain_policy import (
            RetrainPolicyEngine,
            PerformanceMetrics,
            DataReadinessReport,
        )
        from src.common.enums import DriftSeverity

        run_id = drift_result["run_id"]

        # Reconstruct minimal drift report for policy engine
        class _DriftProxy:
            """Lightweight proxy — avoids full deserialization."""
            def __init__(self, d: dict) -> None:
                self.overall_severity       = DriftSeverity(d["overall_severity"])
                self.overall_drift_score    = d["overall_drift_score"]
                self.features_drifted       = d["features_drifted"]
                self.features_total         = d["features_total"]
                self.drift_proportion       = d["drift_proportion"]
                self.concept_drift_detected = d["concept_drift_detected"]
                self.feature_results        = []

        drift_proxy = _DriftProxy(drift_result)

        # Performance metrics from Airflow Variables
        # (set by monitoring system or previous pipeline runs)
        current_metric = float(
            Variable.get("current_f1_score", default="0.77")
        )
        baseline_metric = float(
            Variable.get("champion_f1_score", default="0.80")
        )

        performance = PerformanceMetrics(
            primary_metric_name="f1_score",
            primary_metric_value=current_metric,
            baseline_metric_value=baseline_metric,
        )

        # Last training date from Variable
        last_trained_str = Variable.get(
            "last_training_date", default="2024-01-01T00:00:00+00:00"
        )
        last_training_date = datetime.fromisoformat(last_trained_str)

        data_readiness = DataReadinessReport(
            new_samples_available=drift_result.get("features_total", 0) * 100,
            min_samples_required=MIN_SAMPLES,
            is_sufficient=True,
            label_availability_rate=0.95,
            data_freshness_days=1,
            quality_score=1.0,
        )

        policy     = RetrainPolicyEngine()
        assessment = policy.evaluate(
            drift_report=drift_proxy,
            performance_metrics=performance,
            data_readiness=data_readiness,
            last_training_date=last_training_date,
            model_id=MODEL_ID,
        )

        logger.info(
            f"Retrain decision: {assessment.decision.value} "
            f"(priority={assessment.priority_score:.3f})"
        )

        return {
            "run_id":           run_id,
            "decision":         assessment.decision.value,
            "priority_score":   assessment.priority_score,
            "confidence":       assessment.confidence,
            "reasons":          assessment.reasons,
            "requires_approval": assessment.requires_human_approval,
            "recommended_action": assessment.recommended_action,
            # Pass through
            "prod_processed_path": drift_result["prod_processed_path"],
            "production_path":     drift_result["production_path"],
            "processor_path":      drift_result["processor_path"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 6: Branch on Decision
    # Pure Airflow routing — no ML logic
    # ──────────────────────────────────────────────────────────────────────────

    def _branch_on_decision(**context) -> str:
        """
        Route based on retrain decision.
        Airflow BranchPythonOperator — returns task_id to run next.
        """
        ti       = context["task_instance"]
        decision = ti.xcom_pull(
            task_ids="make_retrain_decision"
        )

        retrain_decisions = {
            "recommended", "required", "urgent"
        }

        if decision["decision"] in retrain_decisions:
            logger.info(
                f"Decision={decision['decision']} → proceeding to training"
            )
            return "train_model"
        else:
            logger.info(
                f"Decision={decision['decision']} → no retrain needed"
            )
            return "no_retrain_needed"

    branch = BranchPythonOperator(
        task_id="branch_on_decision",
        python_callable=_branch_on_decision,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 7a: No Retrain Needed (skip path)
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="no_retrain_needed")
    def no_retrain_needed(**context) -> dict[str, str]:
        """
        Skip path — model still reflects current trends.
        Log the decision and finish cleanly.
        """
        ti       = context["task_instance"]
        decision = ti.xcom_pull(task_ids="make_retrain_decision")
        run_id   = decision["run_id"]

        logger.info(
            f"No retrain needed for run {run_id}. "
            f"Decision: {decision['decision']} | "
            f"Reasons: {decision['reasons']}"
        )

        return {
            "run_id":   run_id,
            "outcome":  "no_retrain",
            "decision": decision["decision"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 7b: Train Model
    # Receives already-processed production data — no reprocessing
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="train_model")
    def train_model(**context) -> dict[str, Any]:
        """
        Train new model on processed production data.

        Data is already processed (from process_both_datasets task).
        No duplicate processing here.

        Airflow retry: if training fails (OOM, timeout), only this
        task reruns. Processing and drift are NOT rerun.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from sklearn.ensemble import GradientBoostingClassifier
        from src.training.trainer import ModelTrainer, TrainingConfig

        ti           = context["task_instance"]
        decision_xcom = ti.xcom_pull(task_ids="make_retrain_decision")
        run_id       = decision_xcom["run_id"]

        # Load already-processed production data
        prod_processed = pd.read_parquet(
            decision_xcom["prod_processed_path"]
        )

        logger.info(
            f"Training on {len(prod_processed)} processed production rows."
        )

        # Training config from Airflow Variables
        n_estimators = int(
            Variable.get("n_estimators", default="100")
        )
        max_depth    = int(
            Variable.get("max_depth", default="10")
        )

        # Feature columns = all columns except target
        feature_cols = [
            c for c in prod_processed.columns
            if c != TARGET_COLUMN
        ]

        config = TrainingConfig(
            model_type="gradient_boosting",
            hyperparameters={
                "n_estimators":  n_estimators,
                "max_depth":     max_depth,
                "learning_rate": 0.1,
                "random_state":  42,
            },
            feature_columns=feature_cols,
            target_column=TARGET_COLUMN,
            cv_folds=5,
            cv_scoring="f1_weighted",
            tags={
                "airflow_run_id":  context["run_id"],
                "pipeline_run_id": run_id,
                "trained_on":      "production_data",
            },
        )

        new_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Define the factory type explicitly               
        def model_factory(hp: Dict[str, Any])->Any:
            return GradientBoostingClassifier(**hp)

        trainer = ModelTrainer(
            model_factory=model_factory,
            preprocessing_fn= None,  # Already preprocessed
            model_id=MODEL_ID,
            pipeline_run_id=run_id,
        )

        result = trainer.train(
            df=prod_processed,
            config=config,
            model_id=MODEL_ID,
            model_version=new_version,
        )

        logger.info(
            f"Training complete: v={new_version}, "
            f"cv={result.cv_mean:.4f}, "
            f"metrics={result.metrics}, "
            f"mlflow={result.mlflow_run_id}"
        )

        # Update Airflow Variable for tracking
        Variable.set("last_training_date", datetime.now(timezone.utc).isoformat())

        return {
            "run_id":            run_id,
            "model_version":     new_version,
            "mlflow_run_id":     result.mlflow_run_id,
            "cv_mean":           result.cv_mean,
            "cv_std":            result.cv_std,
            "metrics":           result.metrics,
            "artifact_path":     result.model_artifact_path,
            "training_samples":  result.training_samples,
            "validation_samples": result.validation_samples,
            # Pass through
            "production_path":   decision_xcom["production_path"],
            "processor_path":    decision_xcom["processor_path"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 8: Champion-Challenger Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="champion_challenger_evaluation")
    def champion_challenger_evaluation(**context) -> dict[str, Any]:
        """
        Full champion-challenger evaluation.

        WHAT CHANGED vs old version:
        OLD: compared pre-computed metric dicts
            (metrics came from training, not re-evaluated on same data)
        NEW: loads BOTH model artifacts
            evaluates BOTH on same held-out eval set
            measures latency, memory, calibration, slices
            runs hard gates before metric comparison

        WHY same eval set for both:
        Champion metrics from training used different data split.
        Different data → unfair comparison → wrong decision.
        Same eval set → apples to apples → correct decision.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.evaluation.champion_challenger import ChampionChallengerEvaluator
        from src.registry.model_registry import ModelRegistry

        ti          = context["task_instance"]
        train_xcom  = ti.xcom_pull(task_ids="train_model")
        run_id      = train_xcom["run_id"]
        new_version = train_xcom["model_version"]

        # Load processed evaluation data
        prod_processed = pd.read_parquet(
            train_xcom["prod_processed_path"]
        ) if "prod_processed_path" in train_xcom else None

        if prod_processed is None:
            raise ValueError("No processed evaluation data available")

        # Hold out 20% for evaluation
        from sklearn.model_selection import train_test_split
        target_col   = TARGET_COLUMN
        feature_cols = [c for c in prod_processed.columns if c != target_col]

        X = prod_processed[feature_cols]
        y = prod_processed[target_col]

        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Champion version and artifact
        champion_version = Variable.get("champion_version", default=None)

        if champion_version:
            # Get champion artifact from registry
            try:
                registry         = ModelRegistry(model_id=MODEL_ID)
                champion_path    = registry.get_artifact_path(
                    MODEL_ID, champion_version
                )
            except Exception:
                champion_version = None
                champion_path    = None
        else:
            champion_path = None

        # Slice columns for per-segment evaluation
        slice_columns = [
            c for c in X_eval.columns
            if X_eval[c].nunique() <= 10
            and X_eval[c].nunique() >= 2
        ][:3]   # Max 3 slice columns

        evaluator = ChampionChallengerEvaluator(
            pipeline_run_id=run_id,
            model_id=MODEL_ID,
        )

        if champion_version and champion_path:
            comparison = evaluator.compare(
                champion_version=champion_version,
                champion_artifact_path=champion_path,
                challenger_version=new_version,
                challenger_artifact_path=train_xcom["artifact_path"],
                X_eval=X_eval,
                y_eval=y_eval,
                is_classification=True,
                slice_columns=slice_columns if slice_columns else None,
            )

            challenger_wins  = comparison.challenger_wins
            improvement      = comparison.improvement
            approval_status  = comparison.approval_status.value

            # Log gate results to Airflow
            for gate in comparison.gate_results:
                logger.info(
                    f"Gate [{gate.gate_name}]: "
                    f"{'✅' if gate.passed else '❌'} {gate.message}"
                )

            result_data = comparison.to_dict()

        else:
            # First deployment — auto-approve with full evaluation
            challenger_eval = evaluator.evaluator.evaluate(
                model_version=new_version,
                model_id=MODEL_ID,
                artifact_path=train_xcom["artifact_path"],
                X_eval=X_eval,
                y_eval=y_eval,
                is_classification=True,
                slice_columns=slice_columns if slice_columns else None,
                pipeline_run_id=run_id,
            )

            challenger_wins = True
            improvement     = 1.0
            approval_status = "auto_approved_first_deployment"

            result_data = {
                "challenger_wins":   True,
                "approval_status":   approval_status,
                "improvement":       1.0,
                "challenger":        challenger_eval.to_dict(),
                "gates":             [],
                "reasons":           ["First deployment — auto-approved"],
            }

        logger.info(
            f"Champion-Challenger: "
            f"{'APPROVED ✅' if challenger_wins else 'REJECTED ❌'} | "
            f"improvement={improvement:.2%}"
        )

        return {
            "run_id":               run_id,
            "challenger_is_better": challenger_wins,
            "improvement":          improvement,
            "approval_status":      approval_status,
            "new_version":          new_version,
            "champion_version":     champion_version,
            "artifact_path":        train_xcom["artifact_path"],
            "metrics":              train_xcom["metrics"],
            "mlflow_run_id":        train_xcom["mlflow_run_id"],
            "production_path":      train_xcom["production_path"],
            "processor_path":       train_xcom["processor_path"],
            "evaluation_summary":   result_data,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 9: Human Approval Gate
    # AIRFLOW NATIVE — ExternalTaskSensor pattern
    # No custom approval code needed unlike pipeline.py
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="check_approval_required")
    def check_approval_required(**context) -> dict[str, Any]:
        """
        Check if human approval is required.
        Sets Airflow Variable for approval tracking.
        """
        ti       = context["task_instance"]
        eval_xcom = ti.xcom_pull(task_ids="champion_challenger_evaluation")

        requires_approval = bool(
            Variable.get("approval_required", default="true").lower()
            == "true"
        )
        challenger_better = eval_xcom["challenger_is_better"]

        # Auto-approve urgent decisions
        decision_xcom = ti.xcom_pull(task_ids="make_retrain_decision")
        is_urgent     = decision_xcom.get("decision") == "urgent"

        auto_approved = (
            not requires_approval
            or is_urgent
            or not challenger_better
        )

        if not auto_approved:
            # Set pending approval Variable
            # Approver calls Airflow API or UI to set this to "approved"
            Variable.set(
                f"approval_{eval_xcom['run_id']}",
                "pending"
            )
            logger.info(
                f"Human approval required for run {eval_xcom['run_id']}. "
                f"Approve via: airflow variables set "
                f"approval_{eval_xcom['run_id']} approved"
            )
        else:
            Variable.set(
                f"approval_{eval_xcom['run_id']}",
                "approved"
            )
            logger.info(f"Auto-approved: {eval_xcom['run_id']}")

        return {
            **eval_xcom,
            "auto_approved":       auto_approved,
            "approval_var_key":    f"approval_{eval_xcom['run_id']}",
        }

    def _branch_on_approval(**context) -> str:
        """Branch: proceed to deploy or wait for approval."""
        ti           = context["task_instance"]
        approval_xcom = ti.xcom_pull(task_ids="check_approval_required")
        eval_xcom    = ti.xcom_pull(task_ids="champion_challenger_evaluation")

        if not eval_xcom["challenger_is_better"]:
            return "challenger_rejected"

        if approval_xcom["auto_approved"]:
            return "deploy_model"
        else:
            return "wait_for_approval_variable"

    approval_branch = BranchPythonOperator(
        task_id="approval_branch",
        python_callable=_branch_on_approval,
    )

    @airflow_task(task_id="challenger_rejected")
    def challenger_rejected(**context) -> dict[str, str]:
        """Challenger did not beat champion — champion unchanged."""
        ti        = context["task_instance"]
        eval_xcom = ti.xcom_pull(task_ids="champion_challenger_evaluation")
        logger.info(
            f"Challenger v{eval_xcom['new_version']} rejected. "
            f"Champion v{eval_xcom['champion_version']} unchanged."
        )
        return {"outcome": "challenger_rejected"}

    @airflow_task(task_id="wait_for_approval_variable")
    def wait_for_approval_variable(**context) -> dict[str, Any]:
        """
        Poll Airflow Variable for human approval.

        Approver sets via:
          airflow variables set approval_<run_id> approved
          OR via Airflow UI → Admin → Variables
          OR via Airflow REST API: PATCH /api/v1/variables/approval_<run_id>

        Polls every 30 seconds, times out after 24 hours.
        """
        import time

        ti            = context["task_instance"]
        approval_xcom = ti.xcom_pull(task_ids="check_approval_required")
        var_key       = approval_xcom["approval_var_key"]
        run_id        = approval_xcom["run_id"]

        timeout_seconds   = 86400  # 24 hours
        poll_interval     = 30
        elapsed           = 0

        logger.info(
            f"Waiting for approval: key={var_key}. "
            f"Set via: airflow variables set {var_key} approved"
        )

        while elapsed < timeout_seconds:
            value = Variable.get(var_key, default="pending")

            if value == "approved":
                logger.info(f"Approval received for {run_id}")
                return {**approval_xcom, "approval_received": True}

            if value == "rejected":
                raise ValueError(
                    f"Retrain rejected by approver for run {run_id}"
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Approval timeout after {timeout_seconds}s for run {run_id}"
        )


    # TASK 10: Register Model (BEFORE Deploy)
    # Registers in STAGING state — not champion yet
    # If this fails: deploy never starts, clean retry
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="register_model_staging")
    def register_model_staging(**context) -> dict[str, Any]:
        """
        Register newly trained model in STAGING state.

        STAGING means:
          - Model exists in registry with full lineage
          - NOT serving production traffic yet
          - Rollback path is clear if deploy fails
          - Audit trail is intact regardless of deploy outcome

        Why BEFORE deploy:
          If deploy fails after registration → model is STAGING in registry
          → retry deploy task only
          → registry state is consistent and queryable

        If deploy fails BEFORE registration → model is ghost in production
          → registry has no record
          → inconsistent state, manual intervention required
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.registry.model_registry import ModelRegistry, ModelStage

        ti          = context["task_instance"]
        eval_xcom   = ti.xcom_pull(
            task_ids="champion_challenger_evaluation"
        )
        run_id      = eval_xcom["run_id"]
        new_version = eval_xcom["new_version"]

        logger.info(
            f"Registering model in STAGING: "
            f"{MODEL_ID} v{new_version}"
        )

        registry = ModelRegistry(
            pipeline_run_id=run_id,
            model_id=MODEL_ID,
        )

        # Register as STAGING — not champion
        # All lineage recorded here regardless of deploy outcome
        registry.register(
            model_id=MODEL_ID,
            version=new_version,
            mlflow_run_id=eval_xcom["mlflow_run_id"],
            metrics=eval_xcom["metrics"],
            hyperparameters={},
            data_hash="",
            artifact_path=eval_xcom["artifact_path"],
            stage=ModelStage.STAGING,       # ← STAGING, not CHAMPION
            tags={
                "trained_on":        "production_data",
                "pipeline_run_id":   run_id,
                "airflow_run_id":    context["run_id"],
                "champion_challenger": "approved",
                "improvement":       str(eval_xcom.get("improvement", 0)),
            },
        )

        logger.info(
            f"Model registered as STAGING: {MODEL_ID} v{new_version} | "
            f"Lineage recorded | "
            f"Ready for deployment"
        )

        return {
            **eval_xcom,
            "registry_stage": "staging",
            "registration_complete": True,
        }
        
    # ──────────────────────────────────────────────────────────────────────────
    # TASK 11: Deploy Model (AFTER Registry)
    # Deploys the STAGING model
    # If fails: model stays STAGING, retry this task only
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(
        task_id="deploy_model",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        retries=3,                          # Retry deploy independently
        retry_delay=timedelta(minutes=2),
    )
    def deploy_model(**context) -> dict[str, Any]:
        """
        Deploy STAGING model to production.

        Runs AFTER registry.register(STAGING).

        On failure:
          - Model stays in STAGING state in registry
          - Previous champion continues serving
          - Only THIS task retries (not registration, not training)
          - Clean rollback via registry

        On success:
          - Model serving production traffic
          - NEXT task promotes to CHAMPION in registry
          - Registry and production stay in sync
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.deployment.deployer import ModelDeployer, DeploymentTarget
        from src.common.enums import DeploymentStrategy

        ti             = context["task_instance"]
        registry_xcom  = ti.xcom_pull(task_ids="register_model_staging")
        run_id         = registry_xcom["run_id"]
        new_version    = registry_xcom["new_version"]

        logger.info(
            f"Deploying STAGING model: {MODEL_ID} v{new_version}"
        )

        strategy_str = Variable.get(
            "deployment_strategy", default="canary"
        )
        # For simplicity, we use a single traffic manager (Istio) in this example.
        from src.deployment.traffic_managers import IstioTrafficManager

        deployer = ModelDeployer(
            traffic_manager=IstioTrafficManager(),
            model_id=MODEL_ID,
            pipeline_run_id=run_id,
        )

        state = deployer.deploy(
            target=DeploymentTarget(
                model_id=MODEL_ID,
                model_version=new_version,
                artifact_path=registry_xcom["artifact_path"],
                endpoint_name=f"{MODEL_ID}_endpoint",
                environment=Variable.get(
                    "environment", default="production"
                ),
            ),
            champion_version=registry_xcom["champion_version"],
            strategy=DeploymentStrategy(strategy_str),
        )

        if state.rollback_triggered:
            # Deployment health checks failed
            # Model stays STAGING in registry (correct state)
            # Previous champion is still serving (deployer reverted)
            raise RuntimeError(
                f"Deployment rolled back: health checks failed. "
                f"Model v{new_version} stays in STAGING. "
                f"Champion v{registry_xcom['champion_version']} restored. "
                f"Retry this task after investigating health check failures."
            )

        logger.info(
            f"Deployment succeeded: {MODEL_ID} v{new_version} | "
            f"strategy={strategy_str} | "
            f"health={state.health_checks_passed}✅ "
            f"{state.health_checks_failed}❌"
        )

        return {
            **registry_xcom,
            "deployment_status": state.status.value,
            "deployment_strategy": strategy_str,
            "health_checks_passed": state.health_checks_passed,
            "health_checks_failed": state.health_checks_failed,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 12: Promote to Champion (AFTER Successful Deploy)
    # Only called after deploy succeeds
    # Registry → CHAMPION only when production is actually serving it
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="promote_to_champion")
    def promote_to_champion(**context) -> dict[str, Any]:
        """
        Promote STAGING model to CHAMPION in registry.

        Only runs after deploy_model succeeds.
        This is the moment registry and production traffic are in sync.

        State transitions:
          new model:      STAGING  → CHAMPION
          old champion:   CHAMPION → ARCHIVED

        After this task:
          registry.get_champion() returns new model
          Next pipeline: champion_version = new_version
          Rollback path: registry.rollback_to_previous() → old model
        """
        import sys
        sys.path.insert(0, "/opt/airflow/ml_platform")

        from src.registry.model_registry import ModelRegistry

        ti           = context["task_instance"]
        deploy_xcom  = ti.xcom_pull(task_ids="deploy_model")
        run_id       = deploy_xcom["run_id"]
        new_version  = deploy_xcom["new_version"]

        logger.info(
            f"Promoting to CHAMPION: {MODEL_ID} v{new_version}"
        )

        registry = ModelRegistry(
            pipeline_run_id=run_id,
            model_id=MODEL_ID,
        )

        # STAGING → CHAMPION
        # Previous CHAMPION → ARCHIVED
        registry.promote_to_champion(
            model_id=MODEL_ID,
            version=new_version,
            pipeline_run_id=run_id,
        )

        # Update Airflow Variables for next cycle
        Variable.set("champion_version", new_version)
        Variable.set(
            "champion_f1_score",
            str(deploy_xcom["metrics"].get("f1_score", 0))
        )

        logger.info(
            f"Champion promoted: {MODEL_ID} v{new_version} | "
            f"Previous champion archived | "
            f"Registry and production are now in sync"
        )

        return {
            **deploy_xcom,
            "registry_stage": "champion",
            "promotion_complete": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 13: Shift Reference (AFTER Promotion)
    # Production data → next cycle's reference
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(task_id="shift_reference")
    def shift_reference(**context) -> dict[str, Any]:
        """
        Save production_raw as next cycle's reference.

        Runs after promote_to_champion.
        The cycle shift only happens when:
          ✅ Model is registered (lineage intact)
          ✅ Model is deployed (serving traffic)
          ✅ Model is champion (registry in sync)

        If this task fails:
          - Production is fine (model serving correctly)
          - Registry is fine (champion promoted)
          - Just next cycle uses old reference
          - Retry this task only → no harm
        """
        ti           = context["task_instance"]
        promote_xcom = ti.xcom_pull(task_ids="promote_to_champion")
        run_id       = promote_xcom["run_id"]
        new_version  = promote_xcom["new_version"]

        production_df = pd.read_parquet(
            promote_xcom["production_path"]
        )

        new_reference_path = _artifact_path(
            MODEL_ID, "reference", "reference_raw.parquet"
        )
        production_df.to_parquet(new_reference_path, index=False)

        logger.info(
            f"Reference shifted: {len(production_df)} rows → "
            f"{new_reference_path} | "
            f"Next cycle reference = current production data"
        )

        return {
            "run_id":             run_id,
            "promoted_version":   new_version,
            "new_reference_path": new_reference_path,
            "new_reference_rows": len(production_df),
            "outcome":            "success",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TASK 14: Pipeline Report
    # Always runs — summarizes what happened
    # ──────────────────────────────────────────────────────────────────────────

    @airflow_task(
        task_id="pipeline_report",
        trigger_rule=TriggerRule.ALL_DONE,
    )
    def pipeline_report(**context) -> dict[str, Any]:
        """
        Post-run summary. Always runs regardless of upstream outcome.
        trigger_rule=ALL_DONE ensures this always executes.

        Collects XCom from all tasks → builds complete report.
        In production: write to database, Slack, PagerDuty.
        """
        ti = context["task_instance"]

        def _safe_pull(task_id: str) -> Optional[dict]:
            try:
                return ti.xcom_pull(task_ids=task_id)
            except Exception:
                return None

        load_xcom       = _safe_pull("load_data")
        validate_xcom   = _safe_pull("validate_production_data")
        process_xcom    = _safe_pull("process_both_datasets")
        drift_xcom      = _safe_pull("detect_drift")
        decision_xcom   = _safe_pull("make_retrain_decision")
        eval_xcom       = _safe_pull("champion_challenger_evaluation")
        deploy_xcom     = _safe_pull("deploy_model")
        promote_xcom    = _safe_pull("promote_and_shift_reference")
        no_retrain_xcom = _safe_pull("no_retrain_needed")

        summary = {
            "dag_run_id":    context["run_id"],
            "model_id":      MODEL_ID,
            "reported_at":   datetime.now(timezone.utc).isoformat(),
            "data": {
                "production_rows": load_xcom.get("production_rows") if load_xcom else None,
                "reference_rows":  load_xcom.get("reference_rows")  if load_xcom else None,
                "reference_source": load_xcom.get("reference_source") if load_xcom else None,
            },
            "validation": {
                "passed":       validate_xcom.get("is_valid")     if validate_xcom else None,
                "success_rate": validate_xcom.get("success_rate") if validate_xcom else None,
            },
            "processing": {
                "n_features":    process_xcom.get("n_features")    if process_xcom else None,
                "prod_rows_out": process_xcom.get("prod_rows_out") if process_xcom else None,
            },
            "drift": {
                "severity":       drift_xcom.get("overall_severity")    if drift_xcom else None,
                "score":          drift_xcom.get("overall_drift_score") if drift_xcom else None,
                "features_drifted": (
                    f"{drift_xcom.get('features_drifted')}/"
                    f"{drift_xcom.get('features_total')}"
                ) if drift_xcom else None,
                "concept_drift":  drift_xcom.get("concept_drift_detected") if drift_xcom else None,
            },
            "decision": {
                "value":          decision_xcom.get("decision")       if decision_xcom else None,
                "priority_score": decision_xcom.get("priority_score") if decision_xcom else None,
            },
            "outcome": (
                "deployed"         if promote_xcom else
                "no_retrain"       if no_retrain_xcom else
                "challenger_rejected" if eval_xcom and not eval_xcom.get("challenger_is_better") else
                "unknown"
            ),
            "new_champion": (
                promote_xcom.get("promoted_version") if promote_xcom else None
            ),
            "metrics": (
                promote_xcom.get("metrics") if promote_xcom else
                eval_xcom.get("metrics")    if eval_xcom else None
            ),
        }

        logger.info(
            f"Pipeline Report:\n{json.dumps(summary, indent=2, default=str)}"
        )

        # ── PRODUCTION: Send to monitoring systems ─────────────────────────
        # slack_client.post_message("#ml-ops", format_report(summary))
        # db.insert("pipeline_runs", summary)
        # datadog.event("ml_pipeline_complete", summary)
        # ──────────────────────────────────────────────────────────────────

        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # DAG WIRING
    # Defines task dependencies — the actual pipeline topology
    # ──────────────────────────────────────────────────────────────────────────

    # Load
    load_result       = load_data()

    # Validate
    validation_result = validate_production_data(load_result)

    # Process BOTH
    process_result    = process_both_datasets(validation_result)

    # Drift
    drift_result      = detect_drift(process_result)

    # Decision
    decision_result   = make_retrain_decision(drift_result)

    # Branch: retrain vs skip
    decision_result >> branch

    # Skip path
    skip_result       = no_retrain_needed()
    branch >> skip_result

    # Retrain path
    train_result      = train_model()
    branch >> train_result

    eval_result       = champion_challenger_evaluation()
    train_result >> eval_result

    approval_result   = check_approval_required()
    eval_result >> approval_result

    approval_result >> approval_branch

    # Approval paths
    rejected_result   = challenger_rejected()
    wait_result       = wait_for_approval_variable()
    approval_branch >> [rejected_result, wait_result]

   # CORRECT ORDER:
    # 1. Register STAGING (before any deployment)
    registry_result   = register_model_staging()
    approval_branch >> registry_result          # auto-approve path
    wait_result     >> registry_result          # manual approve path

    # 2. Deploy (after registry, retries independently)
    deploy_result     = deploy_model()
    registry_result >> deploy_result

    # 3. Promote to CHAMPION (after successful deploy)
    promote_result    = promote_to_champion()
    deploy_result   >> promote_result

    # 4. Shift reference (after promotion)
    shift_result      = shift_reference()
    promote_result  >> shift_result

    # Report always runs
    report_result     = pipeline_report()
    [
        skip_result,
        rejected_result,
        shift_result,
    ] >> report_result