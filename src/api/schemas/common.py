# src/api/schemas/common.py
"""Common response schemas shared across routes."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error:      str
    detail:     Optional[str]    = None
    request_id: Optional[str]    = None
    code:       Optional[str]    = None


class HealthResponse(BaseModel):
    status:        str
    version:       str
    environment:   str
    timestamp:     str
    model_loaded:  bool
    model_version: Optional[str]


class SuccessResponse(BaseModel):
    message:    str
    data:       Optional[dict[str, Any]] = None