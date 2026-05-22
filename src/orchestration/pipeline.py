# src/orchestration/pipeline.py
"""
ML Lifecycle Pipeline — modified for automatic processing flow.

STAGE ORDER (MODIFIED):
────────────────────────
1. VALIDATE          production_data (raw)
2. PROCESS BOTH      fit on reference → transform ref + prod automatically
3. DRIFT DETECTION   ref_processed vs prod_processed (same feature space)
4. RETRAIN DECISION  multi-signal
5. TRAIN             on prod_processed (already processed — no reprocessing)
6. CHAMPION-CHALLENGER
7. DEPLOY
8. PROMOTE + SAVE    prod_raw as new reference for next cycle

KEY CHANGES:
  - No manual processing config
  - No separate processing stage for production only
  - AutoConfigBuilder inspects reference_df automatically
  - ProcessingResult carries both processed DataFrames
  - DriftDetector receives processed DataFrames directly
  - Training receives prod_processed (no duplicate processing)
  - Reference updated to production_raw after success
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from sklearn.base import BaseEstimator

from config.settings import get_settings
from src.common.enums import (
    DeploymentStrategy,
    PipelineStatus,
    RetrainDecision,
)
from src.common.exceptions import MLPlatformError
from src.data.validation import DataValidator, DatasetSchema, ValidationReport
from src.data.processing import ProcessingResult, ProductionDataProcessor
from src.decision.retrain_policy import (
    DataReadinessReport,
    PerformanceMetrics,
    RetrainAssessment,
    RetrainPolicyEngine,
)
from src.deployment.deployer import DeploymentState, DeploymentTarget, ModelDeployer
from src.drift.detector import DriftDetector, DriftReport
from src.evaluation.champion_challenger import (
    ChampionChallengerEvaluator,
    ComparisonResult,
    ModelPerformanceSnapshot,
)
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import PIPELINE_RUNS_TOTAL, PIPELINE_STAGE_DURATION
from src.registry.model_registry import ModelRegistry, ModelStage
from src.training.trainer import ModelTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger("ml_platform.orchestration.pipeline")


@dataclass
class PipelineContext:
    """
    Carries state across all pipeline stages.

    Processing result carries BOTH processed DataFrames.
    Drift detector receives them directly.
    Training receives prod_processed — no duplicate processing.
    """
    pipeline_run_id: str
    model_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None

    # Stage results
    validation_report: Optional[ValidationReport] = None
    processing_result: Optional[ProcessingResult] = None
    drift_report: Optional[DriftReport] = None
    retrain_assessment: Optional[RetrainAssessment] = None
    training_result: Optional[TrainingResult] = None
    comparison_result: Optional[ComparisonResult] = None
    deployment_state: Optional[DeploymentState] = None

    # Tracking
    stages_completed: list[str] = field(default_factory=list)
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    current_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        proc = self.processing_result
        return {
            "pipeline_run_id":     self.pipeline_run_id,
            "model_id":            self.model_id,
            "status":              self.status.value,
            "started_at":          self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at else None
            ),
            "stages_completed":    self.stages_completed,
            "stage_durations_ms":  self.stage_durations_ms,
            "current_stage":       self.current_stage,
            "errors":              self.errors,
            "warnings":            self.warnings,
            # Processing summary
            "processing": {
                "reference_rows_in":   proc.reference_rows_in   if proc else None,
                "reference_rows_out":  proc.reference_rows_out  if proc else None,
                "production_rows_in":  proc.production_rows_in  if proc else None,
                "production_rows_out": proc.production_rows_out if proc else None,
                "n_features":          proc.n_features_out      if proc else None,
                "processor_path":      proc.processor_path      if proc else None,
            },
            # Drift
            "drift_severity":      (
                self.drift_report.overall_severity.value
                if self.drift_report else None
            ),
            "drift_score":         (
                self.drift_report.overall_drift_score
                if self.drift_report else None
            ),
            "features_drifted": (
                f"{self.drift_report.features_drifted}/"
                f"{self.drift_report.features_total}"
                if self.drift_report else None
            ),
            # Decision
            "retrain_decision":     (
                self.retrain_assessment.decision.value
                if self.retrain_assessment else None
            ),
            "retrain_priority":     (
                self.retrain_assessment.priority_score
                if self.retrain_assessment else None
            ),
            # Training
            "new_model_version":    (
                self.training_result.model_version
                if self.training_result else None
            ),
            "challenger_approved":  (
                self.comparison_result.challenger_is_better
                if self.comparison_result else None
            ),
            "deployment_status":    (
                self.deployment_state.status.value
                if self.deployment_state else None
            ),
        }


class _StageTimer:
    def __init__(self, ctx: PipelineContext, stage: str) -> None:
        self._ctx   = ctx
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start             = time.perf_counter()
        self._ctx.current_stage = self._stage
        return self

    def __exit__(self, *_: Any) -> None:
        ms = (time.perf_counter() - self._start) * 1000
        self._ctx.stage_durations_ms[self._stage] = round(ms, 1)
        PIPELINE_STAGE_DURATION.labels(
            model_id=self._ctx.model_id, stage=self._stage
        ).observe(ms / 1000)


class MLLifecyclePipeline:
    """
    Continuous ML lifecycle pipeline.

    AUTOMATIC processing — no manual config.
    Both reference and production go through same processor.
    Drift detected on processed features.
    Training uses already-processed production data.

    CYCLE 1:
      reference  = historical data (1 year ago)
      production = current real user data
      → process both
      → drift detected
      → retrain on processed production data
      → save production_raw as new reference

    CYCLE 2:
      reference  = last cycle's production_raw (auto-loaded)
      production = new current data
      → process both (new processor fitted on new reference)
      → drift detected
      → retrain or not
      → update reference again

    And so on forever.
    """

    def __init__(
        self,
        model_id: str,
        dataset_schema: DatasetSchema,
        training_config: TrainingConfig,
        model_factory: Callable[[dict[str, Any]], BaseEstimator],
        prometheus_url: str = "http://localhost:9090",
    ) -> None:
        self.model_id        = model_id
        self.settings        = get_settings()
        self.training_config = training_config

        self.data_validator      = DataValidator(schema=dataset_schema)
        self.retrain_policy      = RetrainPolicyEngine()
        self.champion_challenger = ChampionChallengerEvaluator()
        self._model_factory      = model_factory
        self.prometheus_url      = prometheus_url

        # Paths for reference data persistence
        self._reference_path = (
            f"artifacts/reference/{model_id}/reference_raw.parquet"
        )
        self._processor_path = (
            f"artifacts/processors/{model_id}/processor.joblib"
        )

    def run(
        self,
        production_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        champion_version: Optional[str] = None,
        champion_metrics: Optional[dict[str, float]] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        last_training_date: Optional[datetime] = None,
        force_retrain: bool = False,
        deployment_strategy: Optional[DeploymentStrategy] = None,
    ) -> PipelineContext:
        """
        Run complete ML lifecycle pipeline.

        Parameters
        ----------
        production_data:
            Raw data from real production traffic.
        reference_data:
            Raw data the current model was trained on.
            If None: auto-loaded from previous cycle's saved reference.
        """
        pipeline_run_id = str(uuid.uuid4())
        ctx = PipelineContext(
            pipeline_run_id=pipeline_run_id,
            model_id=self.model_id,
            status=PipelineStatus.RUNNING,
        )

        logger.info(
            f"Pipeline started: run_id={pipeline_run_id}, "
            f"model={self.model_id}, "
            f"production={len(production_data)} rows"
        )

        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id,
            event_type=EventType.PIPELINE_STARTED,
            step_name="pipeline",
            title=f"🚀 Pipeline: {self.model_id}",
            message=(
                f"production={len(production_data)} rows | "
                f"champion={champion_version or 'none'}"
            ),
            model_id=self.model_id,
            data={
                "production_rows":  len(production_data),
                "champion_version": champion_version,
                "force_retrain":    force_retrain,
            },
        ))

        try:
            # Load reference data (auto from previous cycle if not provided)
            ref_data = self._load_reference(reference_data, pipeline_run_id)

            # ─────────────────────────────────────────────────────────────
            # STAGE 1: Validate production data
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_validate(
                ctx, production_data, pipeline_run_id
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 2: Process BOTH datasets automatically
            # ─────────────────────────────────────────────────────────────
            # Fit on reference → transform both
            # Returns ProcessingResult with ref_processed + prod_processed
            ctx, processing_result = self._stage_process_both(
                ctx,
                reference_data=ref_data,
                production_data=production_data,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 3: Drift Detection on processed data
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_drift_detection(
                ctx,
                ref_processed=processing_result.reference_processed,
                prod_processed=processing_result.production_processed,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 4: Retrain Decision
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_retrain_decision(
                ctx,
                production_data=production_data,
                performance_metrics=performance_metrics,
                last_training_date=last_training_date,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # Check retrain gate
            should_retrain = force_retrain or (
                ctx.retrain_assessment is not None
                and ctx.retrain_assessment.decision in (
                    RetrainDecision.RECOMMENDED,
                    RetrainDecision.REQUIRED,
                    RetrainDecision.URGENT,
                )
            )

            if not should_retrain:
                logger.info(
                    "No retrain required. "
                    "Model still reflects current production trends."
                )
                ctx.status = PipelineStatus.SUCCEEDED
                ctx.current_stage = "complete"
                return self._finalize(ctx)

            # Approval gate
            if (
                ctx.retrain_assessment is not None
                and ctx.retrain_assessment.requires_human_approval
                and not force_retrain
            ):
                return self._handle_approval_gate(ctx, pipeline_run_id)

            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Train on prod_processed (already processed — no redo)
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_train(
                ctx,
                training_data=processing_result.production_processed,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 6: Champion-Challenger
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_champion_challenger(
                ctx,
                champion_version=champion_version,
                champion_metrics=champion_metrics,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            if (
                ctx.comparison_result is not None
                and not ctx.comparison_result.challenger_is_better
            ):
                logger.info(
                    "New model did not beat champion. "
                    "Champion unchanged."
                )
                ctx.status = PipelineStatus.SUCCEEDED
                ctx.current_stage = "complete"
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 7: Deploy
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_deploy(
                ctx,
                champion_version=champion_version,
                deployment_strategy=deployment_strategy,
                pipeline_run_id=pipeline_run_id,
            )
            if ctx.status == PipelineStatus.FAILED:
                return self._finalize(ctx)

            # ─────────────────────────────────────────────────────────────
            # STAGE 8: Promote + Save production_raw as new reference
            # THE KEY CYCLE SHIFT: production becomes next cycle's reference
            # ─────────────────────────────────────────────────────────────
            ctx = self._stage_promote_and_shift_reference(
                ctx,
                production_raw=production_data,   # RAW production saved as reference
                champion_version=champion_version,
                pipeline_run_id=pipeline_run_id,
            )

            ctx.status = PipelineStatus.SUCCEEDED
            ctx.current_stage = "complete"

        except MLPlatformError as exc:
            ctx.status = PipelineStatus.FAILED
            ctx.errors.append(f"[{ctx.current_stage}] {exc.message}")
            logger.error(f"Pipeline failed: {exc.message}", exc_info=True)

        except Exception as exc:
            ctx.status = PipelineStatus.FAILED
            ctx.errors.append(f"[{ctx.current_stage}] {str(exc)}")
            logger.error(f"Unexpected error: {exc}", exc_info=True)

        return self._finalize(ctx)

    # ─────────────────────────────────────────────────────────────────────
    # Stage implementations
    # ─────────────────────────────────────────────────────────────────────

    def _stage_validate(
        self,
        ctx: PipelineContext,
        production_data: pd.DataFrame,
        pipeline_run_id: str,
    ) -> PipelineContext:
        """Stage 1: Validate RAW production data structure."""
        with _StageTimer(ctx, "validation"):
            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.PIPELINE_STAGE_STARTED,
                step_name="validation",
                title="📋 Stage 1/8: Validating Production Data",
                message=f"{len(production_data)} rows | {len(production_data.columns)} cols",
                model_id=self.model_id,
            ))

            try:
                report = self.data_validator.validate(
                    df=production_data,
                    dataset_id=f"production_{pipeline_run_id}",
                    pipeline_run_id=pipeline_run_id,
                    model_id=self.model_id,
                )
                ctx.validation_report = report
                ctx.stages_completed.append("validation")

                if not report.is_valid:
                    ctx.status = PipelineStatus.FAILED
                    ctx.errors.append(
                        f"Validation failed ({report.success_rate:.1%}): "
                        f"{report.errors[:3]}"
                    )
                else:
                    logger.info(
                        f"Validation passed: {report.success_rate:.1%}"
                    )

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"validation: {exc}")

        return ctx

    def _stage_process_both(
        self,
        ctx: PipelineContext,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        pipeline_run_id: str,
    ) -> tuple[PipelineContext, ProcessingResult]:
        """
        Stage 2: Automatic processing of BOTH datasets.

        AutoConfigBuilder inspects reference_data → builds config.
        Processor fits on reference_data → transforms both.
        Both DataFrames now in same feature space.
        """
        empty_result = ProcessingResult(
            reference_processed=pd.DataFrame(),
            production_processed=pd.DataFrame(),
            reference_report=None,   # type: ignore
            production_report=None,  # type: ignore
        )

        with _StageTimer(ctx, "data_processing"):
            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.PIPELINE_STAGE_STARTED,
                step_name="data_processing",
                title="⚙️ Stage 2/8: Processing Both Datasets (Auto)",
                message=(
                    f"Auto-config from reference ({len(reference_data)} rows) | "
                    f"Fit on reference → transform both"
                ),
                model_id=self.model_id,
            ))

            try:
                processor = ProductionDataProcessor(
                    model_id=self.model_id,
                    pipeline_run_id=pipeline_run_id,
                    target_column=self.training_config.target_column,
                    id_columns=[],
                    onehot_threshold=10,
                    skewness_threshold=2.0,
                    max_null_rate_to_drop=0.70,
                )

                save_path = (
                    f"artifacts/processors/{self.model_id}/"
                    f"{pipeline_run_id}_processor.joblib"
                )

                # Single call — handles everything automatically
                processing_result = processor.fit_transform(
                    reference_df=reference_data,
                    production_df=production_data,
                    save_path=save_path,
                )

                ctx.processing_result = processing_result
                ctx.stages_completed.append("data_processing")
                ctx.warnings.extend(processing_result.warnings[:5])

                logger.info(
                    f"Processing complete: "
                    f"ref {processing_result.reference_rows_in}→"
                    f"{processing_result.reference_rows_out} | "
                    f"prod {processing_result.production_rows_in}→"
                    f"{processing_result.production_rows_out} | "
                    f"features={processing_result.n_features_out}"
                )

                event_bus.emit(make_event(
                    pipeline_run_id=pipeline_run_id,
                    event_type=EventType.PIPELINE_STAGE_COMPLETED,
                    step_name="data_processing",
                    title=(
                        f"✅ Processing Complete | "
                        f"ref: {processing_result.reference_rows_in}→"
                        f"{processing_result.reference_rows_out} | "
                        f"prod: {processing_result.production_rows_in}→"
                        f"{processing_result.production_rows_out}"
                    ),
                    message=(
                        f"Features: {processing_result.n_features_out} | "
                        f"Processor: {save_path}"
                    ),
                    model_id=self.model_id,
                    data=processing_result.to_dict(),
                ))

                return ctx, processing_result

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"data_processing: {exc}")
                logger.error(f"Processing failed: {exc}", exc_info=True)
                return ctx, empty_result

    def _stage_drift_detection(
        self,
        ctx: PipelineContext,
        ref_processed: pd.DataFrame,
        prod_processed: pd.DataFrame,
        pipeline_run_id: str,
    ) -> PipelineContext:
        """
        Stage 3: Drift detection on PROCESSED DataFrames.

        Both are in same feature space → KS/PSI comparison valid.
        Features auto-detected from DataFrame columns.
        No manual feature lists.
        """
        with _StageTimer(ctx, "drift_detection"):
            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.PIPELINE_STAGE_STARTED,
                step_name="drift_detection",
                title="🔍 Stage 3/8: Drift Detection (Processed Features)",
                message=(
                    f"ref_processed={len(ref_processed)} | "
                    f"prod_processed={len(prod_processed)} | "
                    f"features={len(ref_processed.columns)}"
                ),
                model_id=self.model_id,
            ))

            try:
                detector = DriftDetector(
                    model_id=self.model_id,
                    pipeline_run_id=pipeline_run_id,
                )

                # Pass processed DataFrames directly
                # No feature lists needed — detector auto-detects from columns
                drift_report = detector.detect(
                    reference_processed=ref_processed,
                    production_processed=prod_processed,
                    target_column=self.training_config.target_column,
                    report_id=f"drift_{pipeline_run_id}",
                    reference_dataset_id="reference_processed",
                    current_dataset_id="production_processed",
                )

                ctx.drift_report = drift_report
                ctx.stages_completed.append("drift_detection")

                logger.info(
                    f"Drift: severity={drift_report.overall_severity.value}, "
                    f"score={drift_report.overall_drift_score:.3f}, "
                    f"drifted={drift_report.features_drifted}/"
                    f"{drift_report.features_total}"
                )

            except Exception as exc:
                # Non-fatal: drift failure is a warning
                ctx.warnings.append(f"drift_detection: {exc}")
                ctx.stages_completed.append("drift_detection")
                logger.warning(f"Drift detection failed: {exc}", exc_info=True)

        return ctx

    def _stage_retrain_decision(
        self,
        ctx: PipelineContext,
        production_data: pd.DataFrame,
        performance_metrics: Optional[PerformanceMetrics],
        last_training_date: Optional[datetime],
        pipeline_run_id: str,
    ) -> PipelineContext:
        """Stage 4: Multi-signal retrain decision."""
        with _StageTimer(ctx, "retrain_decision"):
            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.PIPELINE_STAGE_STARTED,
                step_name="retrain_decision",
                title="🤔 Stage 4/8: Retrain Decision",
                message="Evaluating drift + performance + staleness + data volume",
                model_id=self.model_id,
            ))

            try:
                label_col  = self.training_config.target_column
                label_rate = (
                    1.0 - production_data[label_col].isnull().mean()
                    if label_col in production_data.columns
                    else 0.0
                )

                data_readiness = DataReadinessReport(
                    new_samples_available=len(production_data),
                    min_samples_required=self.settings.retrain.min_new_samples,
                    is_sufficient=(
                        len(production_data) >= self.settings.retrain.min_new_samples
                    ),
                    label_availability_rate=float(label_rate),
                    data_freshness_days=0,
                    quality_score=(
                        ctx.validation_report.success_rate
                        if ctx.validation_report else 1.0
                    ),
                )

                assessment = self.retrain_policy.evaluate(
                    drift_report=ctx.drift_report,
                    performance_metrics=performance_metrics,
                    data_readiness=data_readiness,
                    last_training_date=last_training_date,
                    model_id=self.model_id,
                )
                ctx.retrain_assessment = assessment
                ctx.stages_completed.append("retrain_decision")

                logger.info(
                    f"Retrain: {assessment.decision.value} "
                    f"(priority={assessment.priority_score:.3f})"
                )

                event_bus.emit(make_event(
                    pipeline_run_id=pipeline_run_id,
                    event_type=EventType.RETRAIN_DECISION_MADE,
                    step_name="retrain_decision",
                    title=f"🤔 {assessment.decision.value.upper()}",
                    message=" | ".join(assessment.reasons[:3]),
                    model_id=self.model_id,
                    data={**assessment.to_dict(), "model_id": self.model_id},
                ))

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"retrain_decision: {exc}")

        return ctx

    def _stage_train(
        self,
        ctx: PipelineContext,
        training_data: pd.DataFrame,
        pipeline_run_id: str,
    ) -> PipelineContext:
        """
        Stage 5: Train on prod_processed.
        Data already processed — no duplicate processing here.
        """
        with _StageTimer(ctx, "training"):
            new_version = datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S"
            )

            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.PIPELINE_STAGE_STARTED,
                step_name="training",
                title=f"🏋️ Stage 5/8: Training v{new_version}",
                message=(
                    f"Training on {len(training_data)} processed production rows"
                ),
                model_id=self.model_id,
                model_version=new_version,
            ))

            try:
                trainer = ModelTrainer(
                    model_factory=self._model_factory,
                    model_id=self.model_id,
                    pipeline_run_id=pipeline_run_id,
                )

                result = trainer.train(
                    df=training_data,
                    config=self.training_config,
                    model_id=self.model_id,
                    model_version=new_version,
                )
                ctx.training_result = result
                ctx.stages_completed.append("training")

                logger.info(
                    f"Training complete: v={new_version}, "
                    f"cv={result.cv_mean:.4f}, "
                    f"metrics={result.metrics}"
                )

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"training: {exc}")

        return ctx

    def _stage_champion_challenger(
        self,
        ctx: PipelineContext,
        champion_version: Optional[str],
        champion_metrics: Optional[dict[str, float]],
        pipeline_run_id: str,
    ) -> PipelineContext:
        """Stage 6: Compare new model vs current champion."""
        with _StageTimer(ctx, "evaluation"):
            assert ctx.training_result is not None
            new_version = ctx.training_result.model_version

            try:
                if champion_version and champion_metrics:
                    primary = self._resolve_primary_metric(
                        self.training_config.cv_scoring
                    )

                    comparison = self.champion_challenger.compare(
                        champion=ModelPerformanceSnapshot(
                            model_id=self.model_id,
                            model_version=champion_version,
                            metrics=champion_metrics,
                            primary_metric_name=primary,
                            primary_metric_value=champion_metrics.get(
                                primary, 0.0
                            ),
                            evaluation_dataset_id="champion_eval",
                            evaluation_samples=0,
                        ),
                        challenger=ModelPerformanceSnapshot(
                            model_id=self.model_id,
                            model_version=new_version,
                            metrics=ctx.training_result.metrics,
                            primary_metric_name=primary,
                            primary_metric_value=ctx.training_result.metrics.get(
                                primary, 0.0
                            ),
                            evaluation_dataset_id=f"production_{pipeline_run_id}",
                            evaluation_samples=ctx.training_result.validation_samples,
                        ),
                    )
                    ctx.comparison_result = comparison
                    logger.info(
                        f"Champion-Challenger: "
                        f"better={comparison.challenger_is_better}, "
                        f"improvement={comparison.improvement:.4f}"
                    )

                ctx.stages_completed.append("evaluation")

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"evaluation: {exc}")

        return ctx

    def _stage_deploy(
        self,
        ctx: PipelineContext,
        champion_version: Optional[str],
        deployment_strategy: Optional[DeploymentStrategy],
        pipeline_run_id: str,
    ) -> PipelineContext:
        """Stage 7: Deploy new model."""
        with _StageTimer(ctx, "deployment"):
            assert ctx.training_result is not None
            new_version = ctx.training_result.model_version

            try:
                deployer = ModelDeployer(
                    prometheus_url=self.prometheus_url,
                    model_id=self.model_id,
                    pipeline_run_id=pipeline_run_id,
                )

                state = deployer.deploy(
                    target=DeploymentTarget(
                        model_id=self.model_id,
                        model_version=new_version,
                        artifact_path=ctx.training_result.model_artifact_path,
                        endpoint_name=f"{self.model_id}_endpoint",
                        environment=self.settings.environment.value,
                    ),
                    champion_version=champion_version,
                    strategy=deployment_strategy or DeploymentStrategy(
                        self.settings.deployment.strategy
                    ),
                )

                ctx.deployment_state = state
                ctx.stages_completed.append("deployment")

            except Exception as exc:
                ctx.status = PipelineStatus.FAILED
                ctx.errors.append(f"deployment: {exc}")

        return ctx

    def _stage_promote_and_shift_reference(
        self,
        ctx: PipelineContext,
        production_raw: pd.DataFrame,
        champion_version: Optional[str],
        pipeline_run_id: str,
    ) -> PipelineContext:
        """
        Stage 8: Promote new model + shift reference.

        THE CYCLE KEY:
        ─────────────
        production_raw → saved as new reference (RAW, not processed)

        Next cycle:
          reference = this production_raw
          New processor will fit on this → new feature space
          Drift compared in that new space

        WHY save RAW (not processed)?
          Next cycle's processor will AUTO-CONFIG from this data.
          If we saved processed data, config would be wrong
          (fitted on previous reference, not on this data).
          RAW data → new AutoConfigBuilder → correct new processor.
        """
        with _StageTimer(ctx, "promote_and_shift_reference"):
            assert ctx.training_result is not None
            new_version = ctx.training_result.model_version

            try:
                # Register + promote in model registry
                registry = ModelRegistry(
                    pipeline_run_id=pipeline_run_id,
                    model_id=self.model_id,
                )
                registry.register(
                    model_id=self.model_id,
                    version=new_version,
                    mlflow_run_id=ctx.training_result.mlflow_run_id,
                    metrics=ctx.training_result.metrics,
                    hyperparameters=self.training_config.hyperparameters,
                    data_hash=ctx.training_result.data_hash,
                    artifact_path=ctx.training_result.model_artifact_path,
                    stage=ModelStage.STAGING,
                    tags={
                        "trained_on":      "production_data",
                        "pipeline_run_id": pipeline_run_id,
                        "production_rows": str(len(production_raw)),
                    },
                )

                if (
                    ctx.deployment_state is not None
                    and ctx.deployment_state.status == PipelineStatus.SUCCEEDED
                ):
                    registry.promote_to_champion(
                        model_id=self.model_id,
                        version=new_version,
                        pipeline_run_id=pipeline_run_id,
                    )

                # ── SAVE RAW PRODUCTION AS NEW REFERENCE ───────────────────
                # This is the cycle shift
                # Next run: reference = this production data
                Path(self._reference_path).parent.mkdir(
                    parents=True, exist_ok=True
                )
                production_raw.to_parquet(
                    self._reference_path, index=False
                )

                logger.info(
                    f"✅ Reference shifted: "
                    f"{self._reference_path} "
                    f"({len(production_raw)} rows) | "
                    f"Next cycle: reference = current production"
                )

                ctx.stages_completed.append("promote_and_shift_reference")

                event_bus.emit(make_event(
                    pipeline_run_id=pipeline_run_id,
                    event_type=EventType.PIPELINE_STAGE_COMPLETED,
                    step_name="promote_and_shift_reference",
                    title=(
                        f"👑 Champion: v{new_version} | "
                        f"Reference → current production"
                    ),
                    message=(
                        f"New reference: {len(production_raw)} rows saved. "
                        f"Next cycle compares against this data."
                    ),
                    model_id=self.model_id,
                    model_version=new_version,
                    data={
                        "new_champion":          new_version,
                        "previous_champion":     champion_version,
                        "new_reference_rows":    len(production_raw),
                        "new_reference_path":    self._reference_path,
                        "metrics":               ctx.training_result.metrics,
                    },
                ))

            except Exception as exc:
                ctx.warnings.append(f"promote_and_shift_reference: {exc}")
                logger.warning(f"Promotion failed (non-fatal): {exc}")

        return ctx

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_reference(
        self,
        reference_data: Optional[pd.DataFrame],
        pipeline_run_id: str,
    ) -> pd.DataFrame:
        """
        Load reference data.
        Priority:
          1. Explicitly passed (first run / override)
          2. Auto-loaded from previous cycle's saved reference
        """
        if reference_data is not None:
            logger.info(
                f"Using provided reference: {len(reference_data)} rows"
            )
            return reference_data

        if Path(self._reference_path).exists():
            ref = pd.read_parquet(self._reference_path)
            logger.info(
                f"Auto-loaded reference from previous cycle: "
                f"{len(ref)} rows ({self._reference_path})"
            )
            return ref

        raise ValueError(
            f"No reference data. "
            f"On first run, pass reference_data explicitly. "
            f"Path checked: {self._reference_path}"
        )

    def _handle_approval_gate(
        self,
        ctx: PipelineContext,
        pipeline_run_id: str,
    ) -> PipelineContext:
        ctx.status        = PipelineStatus.AWAITING_APPROVAL
        ctx.current_stage = "awaiting_approval"

        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id,
            event_type=EventType.PIPELINE_AWAITING_APPROVAL,
            step_name="awaiting_approval",
            title="⏸️ Awaiting Approval",
            message=(
                f"Decision: "
                f"{ctx.retrain_assessment.decision.value if ctx.retrain_assessment else '?'}"
            ),
            model_id=self.model_id,
            data={
                "approval_endpoint": (
                    f"/api/v1/pipelines/{pipeline_run_id}/approve"
                ),
            },
        ))
        return ctx

    def _finalize(self, ctx: PipelineContext) -> PipelineContext:
        ctx.completed_at = datetime.now(timezone.utc)
        duration_s       = (ctx.completed_at - ctx.started_at).total_seconds()

        PIPELINE_RUNS_TOTAL.labels(
            model_id=self.model_id, status=ctx.status.value
        ).inc()

        event_bus.emit(make_event(
            pipeline_run_id=ctx.pipeline_run_id,
            event_type=(
                EventType.PIPELINE_COMPLETED
                if ctx.status != PipelineStatus.FAILED
                else EventType.PIPELINE_FAILED
            ),
            step_name="pipeline",
            title=(
                f"{'✅' if ctx.status == PipelineStatus.SUCCEEDED else '❌'} "
                f"Pipeline {ctx.status.value.title()}"
            ),
            message=(
                f"Duration: {duration_s:.1f}s | "
                f"Stages: {ctx.stages_completed}"
            ),
            model_id=self.model_id,
            duration_ms=duration_s * 1000,
            data=ctx.to_dict(),
        ))

        logger.info(
            f"Pipeline {ctx.pipeline_run_id}: {ctx.status.value} "
            f"in {duration_s:.1f}s | stages={ctx.stages_completed}"
        )

        return ctx

    @staticmethod
    def _resolve_primary_metric(cv_scoring: str) -> str:
        return {
            "f1_weighted": "f1_score",
            "f1_macro":    "f1_score",
            "f1":          "f1_score",
            "accuracy":    "accuracy",
            "roc_auc":     "roc_auc",
            "neg_mean_squared_error": "rmse",
            "r2":          "r2",
        }.get(cv_scoring, "f1_score")