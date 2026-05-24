# src/registry/model_registry.py
"""
Production model registry backed by MLflow Model Registry.
Falls back to file-based registry if MLflow is unavailable.
Tracks champion/challenger lineage, stage transitions, and rollback history.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import get_settings
from src.common.exceptions import ModelRegistryError, ModelNotFoundError
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import CHAMPION_METRIC, MODEL_INFO

logger = logging.getLogger("ml_platform.registry.model_registry")


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    CHAMPION    = "champion"       # Production
    CHALLENGER  = "challenger"     # Under evaluation
    ARCHIVED    = "archived"
    REJECTED    = "rejected"


@dataclass
class ModelVersion:
    model_id:       str
    version:        str
    stage:          ModelStage
    created_at:     str
    updated_at:     str
    metrics:        dict[str, float]
    hyperparameters: dict[str, Any]
    data_hash:      str
    mlflow_run_id:  str = ""
    parent_version: Optional[str] = None
    deployment_id:  Optional[str] = None
    tags:           dict[str, str] = field(default_factory=dict)
    description:    str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


class ModelRegistry:
    """
    MLflow-backed model registry with file-system fallback.

    Responsibilities:
    - Register new model versions with full metadata
    - Promote champion (archives previous champion)
    - Rollback to previous champion
    - Query lineage: what data trained this model?
    - Emit events for every state transition
    """

    def __init__(
        self,
        pipeline_run_id: str = "",
        model_id: str = "",
    ):
        self.settings         = get_settings()
        self.pipeline_run_id  = pipeline_run_id
        self.model_id_default = model_id

        # MLflow client
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        self.mlflow_client = MlflowClient()

        # File-based registry (always maintained as backup)-> this steps are so so crucial.
        self._registry_path  = self.settings.registry_dir
        self._models_dir     = self._registry_path / "models"
        self._metadata_file  = self._registry_path / "registry.json"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the model registry from the metadata file."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                for vid, vdata in data.items():
                    vdata["stage"] = ModelStage(vdata["stage"])
                    self._registry[vid] = ModelVersion(**vdata)
            except Exception as e:
                logger.warning(f"Could not load registry file: {e}")

    def _save_registry(self) -> None:
        data = {}
        for vid, v in self._registry.items():
            d = v.to_dict()
            data[vid] = d
        with open(self._metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(
        self,
        model_id: str,
        version: str,
        mlflow_run_id: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        data_hash: str,
        artifact_path: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        tags: Optional[dict[str, str]] = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a new model version."""
        now = datetime.now(timezone.utc).isoformat()

        # Copy artifact to registry
        src = Path(artifact_path)
        if src.exists():
            dest = self._models_dir / f"{model_id}__{version}.joblib"
            shutil.copy2(str(src), str(dest))

        # Determine parent (previous champion)
        champion = self.get_champion(model_id)
        parent_version = champion.version if champion else None

        mv = ModelVersion(
            model_id=model_id,
            version=version,
            stage=stage,
            created_at=now,
            updated_at=now,
            metrics=metrics,
            hyperparameters=hyperparameters,
            data_hash=data_hash,
            mlflow_run_id=mlflow_run_id,
            parent_version=parent_version,
            tags=tags or {},
            description=description,
        )

        registry_key = f"{model_id}__{version}"
        self._registry[registry_key] = mv
        self._save_registry()

        # Register in MLflow Model Registry
        try:
            self.mlflow_client.create_registered_model(model_id)
        except Exception:
            pass  # Already exists

        try:
            if mlflow_run_id:
                self.mlflow_client.create_model_version(
                    name=model_id,
                    source=f"runs:/{mlflow_run_id}/model",
                    run_id=mlflow_run_id,
                    tags={
                        "stage":        stage.value,
                        "data_hash":    data_hash,
                        "parent":       parent_version or "",
                        **(tags or {}),
                    },
                )
        except Exception as e:
            logger.warning(f"MLflow model version creation failed: {e}")

        logger.info(
            f"Registered {model_id} v{version} at stage={stage.value} "
            f"(mlflow_run={mlflow_run_id})"
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.PIPELINE_STAGE_COMPLETED,
            step_name="registry",
            title=f"📦 Model Registered: {model_id} v{version}",
            message=f"Stage: {stage.value} | Parent: {parent_version}",
            model_id=model_id,
            model_version=version,
            data={
                "stage":          stage.value,
                "parent_version": parent_version,
                "mlflow_run_id":  mlflow_run_id,
                "metrics":        metrics,
            },
        ))

        return mv

    def promote_to_champion(
        self, model_id: str, version: str, pipeline_run_id: str = ""
    ) -> ModelVersion:
        """Promote a version to champion, archiving the current champion."""
        key = f"{model_id}__{version}"
        if key not in self._registry:
            raise ModelNotFoundError(
                f"Version '{version}' of '{model_id}' not in registry",
                details={"model_id": model_id, "version": version},
            )

        # Archive current champion
        current = self.get_champion(model_id)
        if current:
            current.stage      = ModelStage.ARCHIVED
            current.updated_at = datetime.now(timezone.utc).isoformat()
            # MLflow stage transition for previous champion
            self._try_mlflow_stage_transition(
                model_id, current.mlflow_run_id, "Archived"
            )
            logger.info(f"Archived previous champion: {model_id} v{current.version}")

        # Promote new champion
        new_champion            = self._registry[key]
        new_champion.stage      = ModelStage.CHAMPION
        new_champion.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_registry()

        # MLflow stage transition for new champion
        self._try_mlflow_stage_transition(
            model_id, new_champion.mlflow_run_id, "Production"
        )

        # Update Prometheus
        for metric, value in new_champion.metrics.items():
            CHAMPION_METRIC.labels(model_id=model_id, metric=metric).set(value)

        MODEL_INFO.labels(model_id=model_id).info({
            "version":        version,
            "mlflow_run_id":  new_champion.mlflow_run_id,
            "data_hash":      new_champion.data_hash,
        })

        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id or self.pipeline_run_id,
            event_type=EventType.DEPLOYMENT_COMPLETED,
            step_name="registry",
            title=f"👑 New Champion: {model_id} v{version}",
            message=(
                f"Previous champion: {current.version if current else 'none'} → "
                f"New champion: {version}"
            ),
            model_id=model_id,
            model_version=version,
            data={
                "new_champion":      version,
                "previous_champion": current.version if current else None,
                "metrics":           new_champion.metrics,
            },
        ))

        logger.info(f"Promoted {model_id} v{version} to champion.")
        return new_champion

    def rollback_to_previous(self, model_id: str) -> ModelVersion:
        """Rollback to most recently archived champion."""
        archived = [
            v for v in self._registry.values()
            if v.model_id == model_id and v.stage == ModelStage.ARCHIVED
        ]
        archived.sort(key=lambda x: x.updated_at, reverse=True)

        if not archived:
            raise ModelRegistryError(
                f"No archived versions for '{model_id}'",
                details={"model_id": model_id},
            )

        # Reject current champion
        current = self.get_champion(model_id)
        if current:
            current.stage      = ModelStage.REJECTED
            current.updated_at = datetime.now(timezone.utc).isoformat()

        # Restore previous
        previous            = archived[0]
        previous.stage      = ModelStage.CHAMPION
        previous.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_registry()

        logger.warning(
            f"Rolled back {model_id}: "
            f"{current.version if current else '?'} → {previous.version}"
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DEPLOYMENT_ROLLBACK,
            step_name="registry",
            title=f"⏪ Registry Rollback: {model_id}",
            message=f"Restored v{previous.version}",
            model_id=model_id,
            model_version=previous.version,
            data={
                "rolled_back_from": current.version if current else None,
                "rolled_back_to":   previous.version,
            },
        ))

        return previous
    

    def get_champion(self, model_id: str) -> Optional[ModelVersion]:
        """Get the current champion for a model."""
        for v in self._registry.values():
            if v.model_id == model_id and v.stage == ModelStage.CHAMPION:
                return v
        return None

    def get_version(self, model_id: str, version: str) -> ModelVersion:
        key = f"{model_id}__{version}"
        if key not in self._registry:
            raise ModelNotFoundError(
                f"Not found: {model_id} v{version}",
                details={"model_id": model_id, "version": version},
            )
        return self._registry[key]

    def list_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None,
    ) -> list[ModelVersion]:
        versions = [
            v for v in self._registry.values()
            if v.model_id == model_id
            and (stage is None or v.stage == stage)
        ]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_artifact_path(self, model_id: str, version: str) -> str:
        path = self._models_dir / f"{model_id}__{version}.joblib"
        if not path.exists():
            raise ModelNotFoundError(
                f"Artifact missing for {model_id} v{version}",
                details={"expected_path": str(path)},
            )
        return str(path)

    def _try_mlflow_stage_transition(
        self, model_id: str, mlflow_run_id: str, stage: str
    ) -> None:
        """ Attempt to transition MLflow model version stage, with error handling."""
        if not mlflow_run_id:
            return
        try:
            versions = self.mlflow_client.search_model_versions(
                f"name='{model_id}' and run_id='{mlflow_run_id}'"
            )
            for v in versions:
                # Only transition if not already in desired stage
                self.mlflow_client.transition_model_version_stage(
                    name=model_id,
                    version=v.version,
                    stage=stage,
                    archive_existing_versions=(stage == "Production"),
                )
        except Exception as e:
            logger.warning(f"MLflow stage transition failed: {e}")