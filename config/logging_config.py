# config/logging_config.py
"""
Structured JSON logging for production.
JSON format → Loki (via Grafana agent).
Standard format → local development console.
"""
from __future__ import annotations

import json
import logging
import logging.config
import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """
    Produces structured JSON log lines.
    Every log line is a valid JSON object for log aggregation systems.
    """

    RESERVED = frozenset({
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs",
        "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName", "exc_info", "exc_text",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "function":  record.funcName,
            "line":      record.lineno,
        }

        # Standard correlation IDs
        for field in (
            "model_id", "pipeline_run_id", "request_id",
            "model_version", "event_type", "step_name",
            "status", "progress", "duration_ms",
        ):
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        # Any extra fields not in reserved set
        for key, val in record.__dict__.items():
            if key not in self.RESERVED and key not in log_entry:
                if isinstance(val, (str, int, float, bool)):
                    log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    settings  = get_settings()
    is_prod   = settings.environment.value == "production"
    log_level = "DEBUG" if settings.debug else "INFO"

    config = {
        "version":                  1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "standard": {
                "format": (
                    "%(asctime)s [%(levelname)-8s] %(name)s "
                    "[%(module)s:%(lineno)d] — %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class":     "logging.StreamHandler",
                "formatter": "json" if is_prod else "standard",
                "stream":    "ext://sys.stdout",
            },
            "file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "formatter":   "json",
                "filename":    str(settings.log_dir / "ml_platform.log"),
                "maxBytes":    50 * 1024 * 1024,   # 50 MB
                "backupCount": 10,
                "encoding":    "utf-8",
            },
            "error_file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "formatter":   "json",
                "filename":    str(settings.log_dir / "ml_platform_errors.log"),
                "maxBytes":    10 * 1024 * 1024,   # 10 MB
                "backupCount": 5,
                "level":       "ERROR",
                "encoding":    "utf-8",
            },
        },
        "root": {
            "level":    log_level,
            "handlers": ["console", "file", "error_file"],
        },
        "loggers": {
            "ml_platform": {
                "level":     log_level,
                "propagate": True,
            },
            # Suppress noisy third-party loggers
            "uvicorn.access": {
                "level":     "WARNING",
                "propagate": True,
            },
            "mlflow": {
                "level":     "WARNING",
                "propagate": True,
            },
            "great_expectations": {
                "level":     "WARNING",
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(config)

    logger = logging.getLogger("ml_platform")
    logger.info(
        f"Logging configured: level={log_level}, "
        f"format={'json' if is_prod else 'standard'}, "
        f"env={settings.environment.value}"
    )