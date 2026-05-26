# config/settings.py
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    model_config = {"env_prefix": "DB_", "extra": "ignore"}

    host: str = "localhost"
    port: int = 5432
    name: str = "ml_platform"
    user: str = "ml_user"
    password: str = "changeme"

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class MLflowSettings(BaseSettings):
    model_config = {"env_prefix": "MLFLOW_", "extra": "ignore"}

    tracking_uri: str = "http://localhost:5000"
    artifact_root: str = "./mlflow-artifacts"
    experiment_prefix: str = "ml_platform"


class DriftSettings(BaseSettings):
    model_config = {"env_prefix": "DRIFT_", "extra": "ignore"}

    psi_threshold: float = 0.2
    ks_p_value_threshold: float = 0.05
    feature_drift_proportion_threshold: float = 0.5
    monitoring_window_days: int = 7
    min_samples_for_drift: int = 500


class RetrainSettings(BaseSettings):
    model_config = {"env_prefix": "RETRAIN_", "extra": "ignore"}

    performance_degradation_threshold: float = 0.05
    min_new_samples: int = 1000
    max_staleness_days: int = 90
    approval_required: bool = True
    auto_retrain_enabled: bool = False


class DeploymentSettings(BaseSettings):
    """ Defines deployment strategies and thresholds for canary releases, including rollback conditions based on error rates and latency."""
    model_config = {"env_prefix": "DEPLOY_", "extra": "ignore"}

    strategy: str = "canary"
    canary_traffic_percent: float = 10.0
    canary_duration_minutes: int = 60
    rollback_on_error_rate: float = 0.01
    rollback_on_latency_p99_ms: float = 500.0
    health_check_interval_seconds: int = 30
    min_champion_improvement: float = 0.01


class PrometheusSettings(BaseSettings):
    model_config = {"env_prefix": "PROMETHEUS_", "extra": "ignore"}

    enabled: bool = True
    port: int = 9090
    metrics_path: str = "/metrics"


class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    environment: Environment = Environment.DEVELOPMENT
    project_name: str = "ML Lifecycle Platform"
    version: str = "1.0.0"
    debug: bool = False

    # Sub-configurations — instantiated with no env-prefix conflicts
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    retrain: RetrainSettings = Field(default_factory=RetrainSettings)
    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)

    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    registry_dir: Path = Path("artifacts/model_registry")

    @field_validator("data_dir", "model_dir", "log_dir", "registry_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings