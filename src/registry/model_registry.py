# src/registry/model_registry.py
"""
Production model registry backed by MLflow Model Registry.
File-system fallback for offline/air-gapped environments.
Handles stage transitions, champion discovery, and rollback.
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
    STAGING = "staging"
    CHAMPION = "production"  # MLflow uses "Production", we map internally
    CHALLENGER = "challenger"
    ARCHIVED = "archived"
    REJECTED = "rejected"


@dataclass
class ModelVersion:
    model_id: str
    version: str
    stage: ModelStage
    created_at: str
    updated_at: str
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    data_hash: str
    mlflow_run_id: str = ""
    parent_version: Optional[str] = None
    deployment_id: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


VERSION_TAG = "tag_version"  # To store our own versioning in MLflow tags
class ModelRegistry:
    """
    MLflow-first model registry with local file fallback.
    
    Priority:
    1. Query MLflow Model Registry (source of truth)
    2. Fall back to local registry.json if MLflow unavailable
    3. Cache MLflow state locally for resilience
    """

    def __init__(
        self,
        pipeline_run_id: str = "",
        model_id: str = "",
    ):
        self.settings = get_settings()
        self.pipeline_run_id = pipeline_run_id
        self.model_id = model_id or getattr(self.settings, 'model_id', 'customer_churn_model')

        # MLflow client
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        self.mlflow_client = MlflowClient()

        # Local file cache (resilience + fast lookups)
        self._registry_path = self.settings.registry_dir
        self._models_dir = self.settings.models_dir
        self._metadata_file = self._registry_path / "registry.json"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory registry (loaded from file, updated from MLflow)
        self._registry: dict[str, ModelVersion] = {}
        self._load_registry()
        
        # MLflow availability flag
        self._mlflow_available = self._check_mlflow_connection()

    # ─────────────────────────────────────────────────────────────────────────
    # MLflow connectivity
    # ─────────────────────────────────────────────────────────────────────────

    def _check_mlflow_connection(self) -> bool:
        """Test MLflow connectivity without failing."""
        try:
            self.mlflow_client.search_registered_models()
            logger.info("MLflow registry connected")
            return True
        except Exception as e:
            logger.warning(f"MLflow unavailable, using file fallback: {e}")
            return False

    def _refresh_mlflow_state(self, model_id: str) -> None:
        """Sync MLflow state to local cache."""
        if not self._mlflow_available:
            return
            
        try:
            versions = self.mlflow_client.search_model_versions(
                f"name='{model_id}'"
            )
            logger.debug(f"Found {len(versions)} versions in MLflow for {model_id} of data has {versions}")
            
            for mv in versions:
                # Extract OUR custom version from tags
                tag_version = mv.tags.get(VERSION_TAG, mv.version)
                # Map MLflow stage to our enum
                stage_map = {
                    "Production": ModelStage.CHAMPION,
                    "Staging": ModelStage.STAGING,
                    "Archived": ModelStage.ARCHIVED,
                    "None": ModelStage.DEVELOPMENT,
                }
                stage = stage_map.get(str(mv.current_stage), ModelStage.DEVELOPMENT)
                
                # Build ModelVersion from MLflow data
                key = f"{model_id}__{tag_version}"
                self._registry[key] = ModelVersion(
                    model_id=model_id,
                    version=mv.version,
                    stage=stage,
                    created_at=str(mv.creation_timestamp),
                    updated_at=str(mv.last_updated_timestamp),
                    metrics={},  # Would need run lookup
                    hyperparameters={},
                    data_hash=mv.tags.get("data_hash", ""),
                    mlflow_run_id=str(mv.run_id),
                    parent_version=mv.tags.get("parent", None),
                    tags=dict(mv.tags),
                    description=mv.description or "",
                )
            
            self._save_registry()
            logger.debug(f"Synced {len(versions)} versions from MLflow for {model_id}")
            
        except Exception as e:
            logger.warning(f"MLflow sync failed: {e}")
            self._mlflow_available = False

    # ─────────────────────────────────────────────────────────────────────────
    # File-based persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        """Load from local file cache."""
        if not self._metadata_file.exists():
            return
            
        try:
            with open(self._metadata_file) as f:
                data = json.load(f)
            for vid, vdata in data.items():
                vdata["stage"] = ModelStage(vdata["stage"])
                self._registry[vid] = ModelVersion(**vdata)
            logger.info(f"Loaded {len(self._registry)} versions from local cache")
        except Exception as e:
            logger.warning(f"Could not load registry file: {e}")

    def _save_registry(self) -> None:
        """Persist to local file cache."""
        data = {vid: v.to_dict() for vid, v in self._registry.items()}
        with open(self._metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────────────────────
    # Champion discovery (MLflow-first)
    # ─────────────────────────────────────────────────────────────────────────

    def get_champion(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get current champion. MLflow-first, fallback to local cache.
        """
        # Try MLflow first
        if self._mlflow_available:
            try:
                # # Search MLflow versions by our custom_version tag
                # filter_string = (
                #     f"name = '{model_id}' and "
                #     f"tags.{VERSION_TAG} = '{tag_version}'"
                # )
                # Method 1: By stage
                versions = self.mlflow_client.get_latest_versions(
                    model_id, stages=["Production"]
                )
                logger.debug(f"MLflow champion lookup found of {(versions)} versions for {model_id}")
                if versions:
                    mv = versions[0]
                    return ModelVersion(
                        model_id=model_id,
                        version=mv.version,
                        stage=ModelStage.CHAMPION,
                        created_at=str(mv.creation_timestamp),
                        updated_at=str(mv.last_updated_timestamp),
                        metrics={},
                        hyperparameters={},
                        data_hash=mv.tags.get("data_hash", ""),
                        mlflow_run_id=str(mv.run_id),
                        parent_version=mv.tags.get("parent", None),
                        tags=dict(mv.tags),
                        description=mv.description or "",
                    )
                
                # Method 2: By alias (MLflow 2.x+)
                try:
                    mv = self.mlflow_client.get_model_version_by_alias(
                        model_id, "champion"
                    )
                    return ModelVersion(
                        model_id=model_id,
                        version=mv.version,
                        stage=ModelStage.CHAMPION,
                        created_at=str(mv.creation_timestamp),
                        updated_at=str(mv.last_updated_timestamp),
                        metrics={},
                        hyperparameters={},
                        data_hash=str(mv.tags.get("data_hash", "")),
                        mlflow_run_id=str(mv.run_id),
                        parent_version=mv.tags.get("parent", None),
                        tags=dict(mv.tags),
                        description=mv.description or "",
                    )
                except Exception:
                    pass
                    
            except Exception as e:
                logger.warning(f"MLflow champion lookup failed: {e}")
                self._mlflow_available = False

        # Fallback: local cache
        self._refresh_mlflow_state(model_id)  # Try once more
        for v in self._registry.values():
            if v.model_id == model_id and v.stage == ModelStage.CHAMPION:
                return v
        
        return None

    def get_champion_artifact_path(self, model_id: str, tag_version:str) -> tuple[str, str]:
        """
        Get champion artifact path and version. 
        Returns (path, version) or raises ModelNotFoundError.
        """
        champion = self.get_champion(model_id)
        if champion is None:
            raise ModelNotFoundError(
                f"No champion found for '{model_id}'",
                details={
                    "model_id": model_id,
                    "mlflow_available": self._mlflow_available,
                    "local_versions": [
                        f"{v.model_id}__{v.version}:{v.stage.value}"
                        for v in self._registry.values()
                    ],
                },
            )
        
        # Try MLflow artifact download first
        if self._mlflow_available and champion.mlflow_run_id:
            try:
                local_path = self.mlflow_client.download_artifacts(
                    champion.mlflow_run_id, "models"
                )
                return local_path, champion.version
            except Exception as e:
                logger.warning(f"MLflow artifact download failed: {e}")

        # Fallback: local file path
        local_path = self._models_dir / f"{model_id}__{tag_version}.joblib"
        if local_path.exists():
            return str(local_path), tag_version
        
        raise ModelNotFoundError(
            f"Artifact not found for champion {model_id} v{tag_version}",
            details={
                "mlflow_run_id": champion.mlflow_run_id,
                "local_path": str(local_path),
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────

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
        """Register new model version. Sync to MLflow and local cache."""
        now = datetime.now(timezone.utc).isoformat()

        # Copy artifact locally (always)
        src = Path(artifact_path)
        if src.exists():
            dest = self._models_dir / f"{model_id}__{version}.joblib"
            shutil.copy2(str(src), str(dest))

        # Determine parent
        champion = self.get_champion(model_id)
        parent_version = champion.version if champion else None

        mv = ModelVersion(
            model_id=model_id,
            version=champion.version if champion else version,
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

        # Save locally
        registry_key = f"{model_id}__{version}"
        self._registry[registry_key] = mv
        self._save_registry()

        # Sync to MLflow (best effort)
        if self._mlflow_available:
            try:
                self._sync_to_mlflow(mv)
            except Exception as e:
                logger.warning(f"MLflow sync failed, keeping local: {e}")
                self._mlflow_available = False

        logger.info(f"Registered {model_id} v{champion.version if champion else version} at stage={stage.value}")
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

    def _sync_to_mlflow(self, mv: ModelVersion) -> None:
        """Push local state to MLflow Model Registry."""
        try:
            self.mlflow_client.create_registered_model(mv.model_id)
        except Exception:
            pass  # Already exists

        mlflow_stage = "Staging" if mv.stage == ModelStage.CHALLENGER else (
            "Production" if mv.stage == ModelStage.CHAMPION else
            "Archived" if mv.stage == ModelStage.ARCHIVED else
            "None"
        )

        self.mlflow_client.create_model_version(
            name=mv.model_id,
            source=f"runs:/{mv.mlflow_run_id}/model",
            run_id=mv.mlflow_run_id,
            tags={
                "stage": mv.stage.value,
                "data_hash": mv.data_hash,
                "parent": mv.parent_version or "",
                **mv.tags,
            },
        )
        
        if mlflow_stage != "None":
            # Need to transition after creation
            versions = self.mlflow_client.search_model_versions(
                f"name='{mv.model_id}' and run_id='{mv.mlflow_run_id}'"
            )
            for v in versions:
                self.mlflow_client.transition_model_version_stage(
                    name=mv.model_id,
                    version=v.version,
                    stage=mlflow_stage,
                    archive_existing_versions=(mlflow_stage == "Production"),
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Promotion
    # ─────────────────────────────────────────────────────────────────────────

    def promote_to_champion(
        self, model_id: str, version: str, pipeline_run_id: str = ""
    ) -> ModelVersion:
        """Promote version to champion with atomic rollback support."""
        # Validate exists
        key = f"{model_id}__{version}"
        if key not in self._registry:
            raise ModelNotFoundError(
                f"Version '{version}' of '{model_id}' not found",
                details={"available": list(self._registry.keys())},
            )

        # Archive current champion
        current = self.get_champion(model_id)
        if current:
            current.stage = ModelStage.ARCHIVED
            current.updated_at = datetime.now(timezone.utc).isoformat()
            self._try_mlflow_stage_transition(
                model_id, current.mlflow_run_id, "Archived"
            )

        # Promote new
        new_champion = self._registry[key]
        new_champion.stage = ModelStage.CHAMPION
        new_champion.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_registry()

        # Sync to MLflow
        self._try_mlflow_stage_transition(
            model_id, new_champion.mlflow_run_id, "Production"
        )
        
        # Set alias for easy loading
        if self._mlflow_available:
            try:
                self.mlflow_client.set_registered_model_alias(
                    name=model_id,
                    alias="champion",
                    version=new_champion.version,
                )
            except Exception as e:
                logger.warning(f"Could not set champion alias: {e}")

        # Metrics
        for metric, value in new_champion.metrics.items():
            CHAMPION_METRIC.labels(model_id=model_id, metric=metric).set(value)

        MODEL_INFO.labels(model_id=model_id).info({
            "version": new_champion.version,
            "mlflow_run_id": new_champion.mlflow_run_id,
            "data_hash": new_champion.data_hash,
        })

        logger.info(f"Promoted {model_id} v{new_champion.version} to champion")
        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id or self.pipeline_run_id,
            event_type=EventType.DEPLOYMENT_COMPLETED,
            step_name="registry",
            title=f"👑 New Champion: {model_id} v{new_champion.version}",
            message=(
                f"Previous champion: {current.version if current else 'none'} → "
                f"New champion: {new_champion.version}"
            ),
            model_id=model_id,
            model_version=new_champion.version,
            data={
                "new_champion":      new_champion.version,
                "previous_champion": current.version if current else None,
                "metrics":           new_champion.metrics,
            },
        ))
        return new_champion

    def _try_mlflow_stage_transition(
        self, model_id: str, mlflow_run_id: str, stage: str
    ) -> None:
        """Best-effort MLflow stage transition."""
        if not self._mlflow_available or not mlflow_run_id:
            return
            
        try:
            versions = self.mlflow_client.search_model_versions(
                f"name='{model_id}' and run_id='{mlflow_run_id}'"
            )
            for v in versions:
                self.mlflow_client.transition_model_version_stage(
                    name=model_id,
                    version=v.version,
                    stage=stage,
                    archive_existing_versions=(stage == "Production"),
                )
        except Exception as e:
            logger.warning(f"MLflow stage transition failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Rollback
    # ─────────────────────────────────────────────────────────────────────────

    def rollback_to_previous(self, model_id: str) -> ModelVersion:
        """Rollback to most recently archived champion."""
        archived = sorted(
            [
                v for v in self._registry.values()
                if v.model_id == model_id and v.stage == ModelStage.ARCHIVED
            ],
            key=lambda x: x.updated_at,
            reverse=True,
        )

        if not archived:
            raise ModelRegistryError(
                f"No archived versions for '{model_id}'",
                details={"model_id": model_id},
            )

        # Reject current
        current = self.get_champion(model_id)
        if current:
            current.stage = ModelStage.REJECTED
            current.updated_at = datetime.now(timezone.utc).isoformat()

        # Restore previous
        previous = archived[0]
        previous.stage = ModelStage.CHAMPION
        previous.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_registry()

        # Sync to MLflow
        self._try_mlflow_stage_transition(
            model_id, previous.mlflow_run_id, "Production"
        )

        logger.warning(f"Rolled back {model_id}: restored v{previous.version}")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def get_version(self, model_id: str, version: str) -> ModelVersion:
        key = f"{model_id}__{version}"
        if key not in self._registry:
            raise ModelNotFoundError(
                f"Not found: {model_id} v{version}",
                details={"available": list(self._registry.keys())},
            )
        return self._registry[key]

    def list_versions(
        self, model_id: str, stage: Optional[ModelStage] = None
    ) -> list[ModelVersion]:
        versions = [
            v for v in self._registry.values()
            if v.model_id == model_id
            and (stage is None or v.stage == stage)
        ]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_artifact_path(self, model_id: str, version: str) -> str:
        """Local artifact path (fallback)."""
        path = self._models_dir / f"{model_id}__{version}.joblib"
        if not path.exists():
            raise ModelNotFoundError(
                f"Artifact missing: {model_id} v{version}",
                details={"expected": str(path)},
            )
        return str(path)