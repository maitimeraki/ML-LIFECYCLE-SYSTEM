# src/api/routes/models.py
"""
Model management endpoints.
Registry queries + champion hot-swap.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import BentoMLDep, RateLimiterDep
from src.api.middleware.auth import AuthContext, require_admin_scope, require_read_scope
from src.api.schemas.common import ErrorResponse, SuccessResponse
from src.common.exceptions import ModelNotFoundError, ModelRegistryError
from src.registry.model_registry import ModelRegistry, ModelStage
from config.settings import get_settings

logger = logging.getLogger("ml_platform.api.models")
router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "/{model_id}/champion",
    summary="Get current champion",
)
async def get_champion(
    model_id: str,
    auth:     AuthContext = Depends(require_read_scope),
) -> dict:
    """Return current champion metadata from registry."""
    try:
        registry = ModelRegistry(model_id=model_id)
        champion = registry.get_champion(model_id)

        if not champion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"No champion for '{model_id}'"},
            )

        settings = get_settings()
        return {
            **champion.to_dict(),
            "mlflow_url": (
                f"{settings.mlflow.tracking_uri}"
                f"/#/runs/{champion.mlflow_run_id}"
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(exc)},
        )


@router.get(
    "/{model_id}/versions",
    summary="List model versions",
)
async def list_versions(
    model_id: str,
    stage:    Optional[str] = Query(None),
    auth:     AuthContext   = Depends(require_read_scope),
) -> dict:
    """List all registered versions, optionally filtered by stage."""
    try:
        registry   = ModelRegistry(model_id=model_id)
        stage_enum = ModelStage(stage) if stage else None
        versions   = registry.list_versions(model_id, stage=stage_enum)

        return {
            "model_id": model_id,
            "count":    len(versions),
            "versions": [v.to_dict() for v in versions],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Invalid stage: {exc}"},
        )


@router.post(
    "/{model_id}/reload",
    response_model=SuccessResponse,
    summary="Hot-swap champion (admin)",
    description=(
        "Trigger champion model reload on BentoML service.\n\n"
        "Call this after Airflow promotes a new champion.\n"
        "**Requires admin scope.**\n"
    ),
)
async def reload_champion(
    model_id:     str,
    auth:         AuthContext = Depends(require_admin_scope),
    rate_limiter: RateLimiterDep = None,
    bentoml:      BentoMLDep     = None,
) -> SuccessResponse:
    """Hot-swap champion on BentoML without restarting the service."""
    rate_limiter.check(client_id=auth.client_id, scope="admin")

    try:
        result = await bentoml.reload_champion()
        return SuccessResponse(
            message=f"Champion reloaded: {result.get('new_version')}",
            data=result,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": str(exc)},
        )


@router.post(
    "/{model_id}/rollback",
    response_model=SuccessResponse,
    summary="Rollback to previous champion (admin)",
)
async def rollback(
    model_id:     str,
    auth:         AuthContext = Depends(require_admin_scope),
    rate_limiter: RateLimiterDep = None,
    bentoml:      BentoMLDep     = None,
) -> SuccessResponse:
    """Rollback registry to previous champion + reload BentoML."""
    rate_limiter.check(client_id=auth.client_id, scope="admin")

    try:
        registry = ModelRegistry(model_id=model_id)
        previous = registry.rollback_to_previous(model_id)

        # Reload BentoML to serve rolled-back champion
        await bentoml.reload_champion()

        return SuccessResponse(
            message=f"Rolled back to v{previous.version}",
            data={"rolled_back_to": previous.version},
        )

    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.message},
        )