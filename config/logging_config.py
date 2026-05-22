"""
Structured JSON logging configuration for production observability.
"""

import logging
import logging.config
import json
import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """Produces structured JSON log lines for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach extra fields if present
        if hasattr(record, "model_id"):
            log_entry["model_id"] = record.model_id
        if hasattr(record, "pipeline_run_id"):
            log_entry["pipeline_run_id"] = record.pipeline_run_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    settings = get_settings()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if settings.environment == "production" else "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": str(settings.log_dir / "ml_platform.log"),
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 10,
            },
        },
        "root": {
            "level": "DEBUG" if settings.debug else "INFO",
            "handlers": ["console", "file"],
        },
        "loggers": {
            "ml_platform": {"level": "DEBUG" if settings.debug else "INFO", "propagate": True},
            "uvicorn": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)