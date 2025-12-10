"""Logging configuration for the RAG service.

The configuration is intentionally simple but production-friendly:
- JSON-like single-line logs for easy ingestion by log aggregators
- Includes key fields like level, logger, message, and request_id
"""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    log_level = settings.log_level.upper()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "%(message)s | request_id=%(request_id)s"
                )
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "structured",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["default"],
        },
    }

    # Ensure `request_id` is always present in log records
    class RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
            if not hasattr(record, "request_id"):
                record.request_id = "-"
            return True

    dictConfig(config)
    logging.getLogger().addFilter(RequestIdFilter())
