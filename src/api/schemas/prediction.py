# src/api/schemas/prediction.py
"""
Request and response schemas for prediction endpoints.
Pydantic v2 — strict validation, clear error messages.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    """
    Single prediction request from API consumer.
    Validated BEFORE reaching BentoML.
    """
    features: dict[str, Any] = Field(
        ...,
        description="Feature key-value pairs",
        min_length=1,
        examples=[{
            "feature_1": 10.2,
            "feature_2": 4.8,
            "feature_3": 2.1,
            "feature_5": 55.0,
        }],
    )
    model_version: Optional[str] = Field(
        None,
        description="Pin specific version. Omit for champion.",
        pattern=r"^\d{8}_\d{6}$",  # e.g. 20240315_142100
    )
    return_probabilities: bool = Field(
        False,
        description="Return per-class probabilities",
    )
    request_id: Optional[str] = Field(
        None,
        description="Caller idempotency key",
        max_length=128,
    )

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features dict cannot be empty")
        # Reject features with all None values
        if all(val is None for val in v.values()):
            raise ValueError("all feature values are None")
        return v

    @field_validator("features")
    @classmethod
    def no_inf_values(cls, v: dict) -> dict:
        import math
        for key, val in v.items():
            if isinstance(val, float) and (
                math.isinf(val) or math.isnan(val)
            ):
                raise ValueError(
                    f"Feature '{key}' has invalid value: {val}"
                )
        return v


class PredictionResponse(BaseModel):
    request_id:    str
    prediction:    Any
    probabilities: Optional[dict[str, float]] = None
    model_id:      str
    model_version: str
    latency_ms:    float
    cached:        bool
    timestamp:     str


class BatchPredictionRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of feature dicts",
    )
    return_probabilities: bool = False

    @field_validator("instances")
    @classmethod
    def instances_valid(cls, v: list) -> list:
        for i, inst in enumerate(v):
            if not inst:
                raise ValueError(f"Instance {i} is empty")
        return v


class BatchPredictionResponse(BaseModel):
    predictions:           list[Any]
    probabilities:         Optional[list[dict[str, float]]] = None
    model_id:              str
    model_version:         str
    total_latency_ms:      float
    per_sample_latency_ms: float
    batch_size:            int


class FeedbackRequest(BaseModel):
    """
    Ground truth feedback for a previous prediction.
    Used by PerformanceMonitor for delayed label matching.
    """
    request_id:   str = Field(..., description="Original prediction request_id")
    ground_truth: Any = Field(..., description="Actual label/value")
    feedback_at:  Optional[datetime] = None

    @field_validator("request_id")
    @classmethod
    def request_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("request_id cannot be empty")
        return v.strip()