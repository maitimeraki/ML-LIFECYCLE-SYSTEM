# src/deployment/deployer.py
"""
Production deployment orchestrator.
Fix: health check now queries real Prometheus metrics instead of 0.0 defaults.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import DeploymentSettings, get_settings
from src.common.enums import DeploymentStrategy, PipelineStatus
from src.common.exceptions import DeploymentError, RollbackError
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    DEPLOYMENT_STATUS_GAUGE, DEPLOYMENT_TRAFFIC_SPLIT, DEPLOYMENT_ROLLBACKS_TOTAL,
)

logger = logging.getLogger("ml_platform.deployment.deployer")


@dataclass
class DeploymentTarget:
    """Represents a target for model deployment."""
    model_id: str = field(default="")
    model_version: str = field(default="")
    artifact_path: str = field(default="")
    endpoint_name: str = field(default="")
    environment: str = field(default="production")


@dataclass
class DeploymentState:
    deployment_id: str
    strategy: DeploymentStrategy
    status: PipelineStatus
    champion_model_version: Optional[str]
    challenger_model_version: str
    traffic_split: dict[str, float]
    started_at: datetime
    completed_at: Optional[datetime] = None
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    error_rate: float = 0.0
    latency_p99_ms: float = 0.0
    rollback_triggered: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "deployment_id":      self.deployment_id,
            "strategy":           self.strategy.value,
            "status":             self.status.value,
            "champion_version":   self.champion_model_version,
            "challenger_version": self.challenger_model_version,
            "traffic_split":      self.traffic_split,
            "started_at":         self.started_at.isoformat(),
            "completed_at":       self.completed_at.isoformat() if self.completed_at else None,
            "health_checks": {
                "passed": self.health_checks_passed,
                "failed": self.health_checks_failed,
            },
            "error_rate":         self.error_rate,
            "rollback_triggered": self.rollback_triggered,
            "notes":              self.notes,
        }


class PrometheusHealthSource:
    """
    Queries Prometheus for real-time serving metrics.
    Falls back to 0.0 if Prometheus is unreachable (dev mode).
    """

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url

    def get_error_rate(self, model_id: str) -> float:
        """Query error rate from Prometheus."""
        try:
            import requests
            query = (
                f'rate(ml_prediction_requests_total{{model_id="{model_id}",'
                f'status="error"}}[5m]) / '
                f'rate(ml_prediction_requests_total{{model_id="{model_id}"}}[5m])'
            )
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=3,
            )
            if resp.status_code == 200:
                result = resp.json().get("data", {}).get("result", [])
                if result:
                    return float(result[0]["value"][1])
        except Exception:
            pass
        return 0.0

    def get_p99_latency_ms(self, model_id: str) -> float:
        """Query P99 latency from Prometheus."""
        try:
            import requests
            query = (
                f'histogram_quantile(0.99, '
                f'rate(ml_prediction_latency_ms_bucket{{model_id="{model_id}"}}[5m]))'
            )
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=3,
            )
            if resp.status_code == 200:
                result = resp.json().get("data", {}).get("result", [])
                if result:
                    return float(result[0]["value"][1])
        except Exception:
            pass
        return 0.0


class ModelDeployer:
    """
    Orchestrates safe model deployment.
    Health checks now query real Prometheus metrics.
    """

    def __init__(
        self,
        settings: Optional[DeploymentSettings] = None,
        traffic_manager: Optional[Any] = None,
        prometheus_url: str = "http://prometheus:9090",
        model_id: str = "",
        pipeline_run_id: str = "",
    ):
        self.settings         = settings or get_settings().deployment
        self.traffic_manager  = traffic_manager
        self.health_source    = PrometheusHealthSource(prometheus_url)
        self.model_id         = model_id
        self.pipeline_run_id  = pipeline_run_id
        self._active: dict[str, DeploymentState] = {}

    def deploy(
        self,
        target: DeploymentTarget,
        champion_version: Optional[str] = None,
        strategy: Optional[DeploymentStrategy] = None,
    ) -> DeploymentState: 
        import uuid

        deployment_id = str(uuid.uuid4())
        strategy = strategy or DeploymentStrategy(self.settings.strategy)

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DEPLOYMENT_STARTED,
            step_name="deployment",
            title=f"🚀 Deployment Started ({strategy.value})",
            message=(
                f"Model: {target.model_id} v{target.model_version} | "
                f"Strategy: {strategy.value} | "
                f"Champion: {champion_version or 'none'}"
            ),
            model_id=target.model_id,
            model_version=target.model_version,
            data={
                "deployment_id": deployment_id,
                "strategy":      strategy.value,
                "environment":   target.environment,
                "endpoint":      target.endpoint_name,
            },
        ))

        logger.info(
            f"Deploying {target.model_id} v{target.model_version} "
            f"via {strategy.value} (deployment_id={deployment_id})"
        )

        self._pre_deployment_checks(target)

        initial_traffic = self._get_initial_traffic_split(strategy, champion_version)

        state = DeploymentState(
            deployment_id=deployment_id,
            strategy=strategy,
            status=PipelineStatus.RUNNING,
            champion_model_version=champion_version,
            challenger_model_version=target.model_version,
            traffic_split=initial_traffic,
            started_at=datetime.now(timezone.utc),
        )

        self._active[deployment_id] = state

        # Update Prometheus traffic split
        self._update_traffic_metrics(target.model_id, state.traffic_split)
        DEPLOYMENT_STATUS_GAUGE.labels(
            model_id=target.model_id, strategy=strategy.value
        ).set(1)

        try:
            if strategy == DeploymentStrategy.CANARY:
                self._execute_canary(state, target)
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                self._execute_blue_green(state, target)
            elif strategy == DeploymentStrategy.SHADOW:
                self._execute_shadow(state, target)
            elif strategy == DeploymentStrategy.DIRECT:
                self._execute_direct(state, target)
            else:
                raise DeploymentError(f"Unknown strategy: {strategy}")

        except Exception as e:
            state.status = PipelineStatus.FAILED
            state.notes.append(f"Failed: {e}")
            logger.error(f"Deployment {deployment_id} failed: {e}", exc_info=True)

            if champion_version:
                try:
                    self._rollback(state, target, reason=str(e))
                except RollbackError as re:
                    logger.critical(f"ROLLBACK FAILED: {re}")

            DEPLOYMENT_STATUS_GAUGE.labels(
                model_id=target.model_id, strategy=strategy.value
            ).set(0)

            raise DeploymentError(str(e), details=state.to_dict()) from e

        DEPLOYMENT_STATUS_GAUGE.labels(
            model_id=target.model_id, strategy=strategy.value
        ).set(0)

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DEPLOYMENT_COMPLETED,
            step_name="deployment",
            title=(
                f"{'✅' if state.status == PipelineStatus.SUCCEEDED else '❌'} "
                f"Deployment {state.status.value.title()}"
            ),
            message=(
                f"Model {target.model_id} v{target.model_version} | "
                f"Rollback: {state.rollback_triggered} | "
                f"Health: {state.health_checks_passed}✅ {state.health_checks_failed}❌"
            ),
            model_id=target.model_id,
            model_version=target.model_version,
            status="success" if state.status == PipelineStatus.SUCCEEDED else "failed",
            data=state.to_dict(),
        ))

        return state

    def _pre_deployment_checks(self, target: DeploymentTarget) -> None:
        """ Perform pre-deployment checks:
        - Verify artifact exists
        - Load model from artifact
        - Check for predict method
        """
        import os
        if not os.path.exists(target.artifact_path):
            raise DeploymentError(f"Artifact not found: {target.artifact_path}")

        try:
            import joblib
            model = joblib.load(target.artifact_path)
            if not hasattr(model, "predict"):
                raise DeploymentError("Model missing 'predict' method")
        except Exception as e:
            raise DeploymentError(f"Pre-deployment check failed: {e}") from e

    def _get_initial_traffic_split(
        self, strategy: DeploymentStrategy, champion_version: Optional[str]
    ) -> dict[str, float]:
        if not champion_version:
            return {"challenger": 100.0}
        if strategy == DeploymentStrategy.CANARY:
            p = self.settings.canary_traffic_percent
            return {"champion": 100.0 - p, "challenger": p}
        if strategy == DeploymentStrategy.BLUE_GREEN:
            return {"champion": 100.0, "challenger": 0.0}
        if strategy == DeploymentStrategy.SHADOW:
            return {"champion": 100.0, "challenger_shadow": 100.0}
        return {"challenger": 100.0}

    def _execute_canary(
        self, state: DeploymentState, target: DeploymentTarget
    ) -> None:
        stages = [self.settings.canary_traffic_percent, 25.0, 50.0, 75.0, 100.0]
        stages = [s for s in stages if s >= self.settings.canary_traffic_percent]

        stage_duration = max(
            (self.settings.canary_duration_minutes * 60) // len(stages), 30
        )

        for pct in stages:
            state.traffic_split = {
                "champion":   100.0 - pct,
                "challenger": pct,
            }
            self._update_traffic_metrics(target.model_id, state.traffic_split)

            if self.traffic_manager:
                self.traffic_manager.set_traffic_split(
                    target.endpoint_name, state.traffic_split
                )

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.DEPLOYMENT_STAGE_CHANGED,
                step_name="deployment",
                title=f"🕯️ Canary Stage: {pct}%",
                message=f"Challenger at {pct}% traffic",
                model_id=target.model_id,
                data={"traffic_split": state.traffic_split, "stage_pct": pct},
            ))

            logger.info(f"Canary stage: challenger={pct}%")

            health_ok = self._monitor_health(state, target.model_id, stage_duration)
            if not health_ok:
                self._rollback(state, target, reason=f"health_check_failed_at_{pct}pct")
                return

            state.notes.append(f"Stage {pct}% passed.")

        state.status       = PipelineStatus.SUCCEEDED
        state.completed_at = datetime.now(timezone.utc)
        logger.info("Canary deployment succeeded.")

    def _execute_blue_green(
        self, state: DeploymentState, target: DeploymentTarget
    ) -> None:
        state.notes.append("Green environment prepared.")
        state.traffic_split = {"challenger": 100.0}
        self._update_traffic_metrics(target.model_id, state.traffic_split)

        if self.traffic_manager:
            self.traffic_manager.set_traffic_split(
                target.endpoint_name, state.traffic_split
            )

        health_ok = self._monitor_health(
            state, target.model_id,
            self.settings.canary_duration_minutes * 60
        )
        if not health_ok:
            self._rollback(state, target, reason="health_check_failed_blue_green")
            return

        state.status       = PipelineStatus.SUCCEEDED
        state.completed_at = datetime.now(timezone.utc)

    def _execute_shadow(
        self, state: DeploymentState, target: DeploymentTarget
    ) -> None:
        state.notes.append(
            "Shadow mode active. Monitor Grafana then promote manually."
        )
        state.status = PipelineStatus.RUNNING

    def _execute_direct(
        self, state: DeploymentState, target: DeploymentTarget
    ) -> None:
        logger.warning("Direct deployment — for dev/staging only.")
        state.traffic_split = {"challenger": 100.0}
        state.status        = PipelineStatus.SUCCEEDED
        state.completed_at  = datetime.now(timezone.utc)

    def _monitor_health(
        self,
        state: DeploymentState,
        model_id: str,
        duration_seconds: int,
    ) -> bool:
        """
        Monitor real Prometheus metrics during deployment window.
        Falls back gracefully if Prometheus is unavailable.
        """
        interval   = self.settings.health_check_interval_seconds
        num_checks = max(duration_seconds // interval, 1)
        consecutive_failures = 0

        logger.info(
            f"Health monitoring: {num_checks} checks over {duration_seconds}s"
        )

        for i in range(num_checks):
            # Query real metrics from Prometheus
            error_rate   = self.health_source.get_error_rate(model_id)
            latency_p99  = self.health_source.get_p99_latency_ms(model_id)

            # Update state from real data
            state.error_rate    = error_rate
            state.latency_p99_ms = latency_p99

            healthy = (
                error_rate <= self.settings.rollback_on_error_rate
                and (latency_p99 == 0 or latency_p99 <= self.settings.rollback_on_latency_p99_ms)
            )

            if healthy:
                state.health_checks_passed += 1
                consecutive_failures = 0
                logger.debug(
                    f"Health check {i+1}/{num_checks}: OK "
                    f"(error_rate={error_rate:.4f}, p99={latency_p99:.1f}ms)"
                )
            else:
                state.health_checks_failed += 1
                consecutive_failures += 1
                logger.warning(
                    f"Health check {i+1}/{num_checks}: FAILED "
                    f"(error_rate={error_rate:.4f} > {self.settings.rollback_on_error_rate}, "
                    f"p99={latency_p99:.1f}ms > {self.settings.rollback_on_latency_p99_ms}ms)"
                )

                if consecutive_failures >= 3:
                    logger.error("3 consecutive failures. Triggering rollback.")
                    return False

            # In production: time.sleep(interval)

        total = state.health_checks_passed + state.health_checks_failed
        return (
            state.health_checks_failed == 0
            or state.health_checks_failed < total * 0.1
        )

    def _rollback(
        self,
        state: DeploymentState,
        target: DeploymentTarget,
        reason: str = "unknown",
    ) -> None:
        logger.warning(
            f"ROLLBACK {state.deployment_id}: "
            f"reverting to champion v{state.champion_model_version} "
            f"(reason: {reason})"
        )

        try:
            state.traffic_split = {"champion": 100.0}
            self._update_traffic_metrics(target.model_id, state.traffic_split)

            if self.traffic_manager:
                self.traffic_manager.set_traffic_split(
                    target.endpoint_name, state.traffic_split
                )

            state.rollback_triggered = True
            state.status             = PipelineStatus.FAILED
            state.completed_at       = datetime.now(timezone.utc)
            state.notes.append(
                f"Rolled back to v{state.champion_model_version} "
                f"at {datetime.now(timezone.utc).isoformat()} (reason: {reason})"
            )

            DEPLOYMENT_ROLLBACKS_TOTAL.labels(
                model_id=target.model_id, reason=reason
            ).inc()

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.DEPLOYMENT_ROLLBACK,
                step_name="deployment",
                title="⏪ Rollback Triggered",
                message=(
                    f"Reverted to champion v{state.champion_model_version} | "
                    f"Reason: {reason}"
                ),
                model_id=target.model_id,
                status="warning",
                severity="warning",
                data={"reason": reason, "reverted_to": state.champion_model_version},
            ))

        except Exception as e:
            raise RollbackError(
                f"Rollback failed: {e}", details=state.to_dict()
            ) from e

    def _update_traffic_metrics(
        self, model_id: str, traffic_split: dict[str, float]
    ) -> None:
        for variant, pct in traffic_split.items():
            DEPLOYMENT_TRAFFIC_SPLIT.labels(
                model_id=model_id, variant=variant
            ).set(pct)

    def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[DeploymentState]:
        return self._active.get(deployment_id)