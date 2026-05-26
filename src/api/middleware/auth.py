# src/api/middleware/auth.py
"""
Authentication middleware.

Supports two schemes:
  1. API Key  → X-API-Key header  (service-to-service)
  2. JWT      → Bearer token      (user-facing)

API keys stored in environment / Vault.
JWT verified with public key (RS256).

Scopes:
  predict       → POST /predict, POST /predict/batch
  admin         → model reload, pipeline trigger
  read          → GET endpoints only
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

logger = logging.getLogger("ml_platform.api.auth")

# ── API Key ────────────────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for service-to-service auth",
)

# In production: load from Vault / AWS Secrets Manager
_VALID_API_KEYS: dict[str, dict] = {
    key: {"name": name, "scopes": scopes}
    for key, name, scopes in [
        (
            os.getenv("API_KEY_PREDICT", "dev-predict-key"),
            "prediction-service",
            ["predict", "read"],
        ),
        (
            os.getenv("API_KEY_ADMIN", "dev-admin-key"),
            "admin-service",
            ["predict", "admin", "read"],
        ),
        (
            os.getenv("API_KEY_READONLY", "dev-read-key"),
            "monitoring-service",
            ["read"],
        ),
    ]
}

# ── JWT ────────────────────────────────────────────────────────────────────────

BEARER_SCHEME = HTTPBearer(auto_error=False)


# ── Dependency ─────────────────────────────────────────────────────────────────

class AuthContext:
    """Resolved authentication context, attached to request state."""

    def __init__(
        self,
        client_id: str,
        scopes:    list[str],
        auth_type: str,
    ) -> None:
        self.client_id = client_id
        self.scopes    = scopes
        self.auth_type = auth_type

    def require_scope(self, scope: str) -> None:
        if scope not in self.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error":    "Insufficient scope",
                    "required": scope,
                    "granted":  self.scopes,
                },
            )


async def get_auth_context(
    api_key: Optional[str] = Security(API_KEY_HEADER),
    bearer:  Optional[HTTPAuthorizationCredentials] = Security(BEARER_SCHEME),
) -> AuthContext:
    """
    FastAPI dependency — resolves auth from API key OR JWT.
    Raises 401 if neither is provided or valid.
    """
    # ── Try API Key first ──────────────────────────────────────────────────
    if api_key:
        key_info = _VALID_API_KEYS.get(api_key)
        if key_info:
            return AuthContext(
                client_id=key_info["name"],
                scopes=key_info["scopes"],
                auth_type="api_key",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid API key"},
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Try JWT ────────────────────────────────────────────────────────────
    if bearer:
        try:
            ctx = _verify_jwt(bearer.credentials)
            return ctx
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": f"Invalid JWT: {exc}"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── No auth provided ───────────────────────────────────────────────────
    # Allow in development only
    env = os.getenv("ENVIRONMENT", "development")
    if env == "development":
        return AuthContext(
            client_id="dev-anonymous",
            scopes=["predict", "read"],
            auth_type="development",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "Authentication required"},
        headers={"WWW-Authenticate": "ApiKey, Bearer"},
    )


def _verify_jwt(token: str) -> AuthContext:
    """Verify JWT and extract claims."""
    try:
        import jwt

        public_key = os.getenv("JWT_PUBLIC_KEY", "")
        if not public_key:
            raise ValueError("JWT_PUBLIC_KEY not configured")

        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_exp": True},
        )

        client_id = payload.get("sub", "unknown")
        scopes    = payload.get("scopes", ["read"])

        return AuthContext(
            client_id=client_id,
            scopes=scopes if isinstance(scopes, list) else [scopes],
            auth_type="jwt",
        )

    except ImportError:
        raise ValueError("PyJWT not installed: pip install PyJWT")


# ── Scope-specific dependencies ────────────────────────────────────────────────

async def require_predict_scope(
    auth: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    auth.require_scope("predict")
    return auth


async def require_admin_scope(
    auth: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    auth.require_scope("admin")
    return auth


async def require_read_scope(
    auth: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    auth.require_scope("read")
    return auth