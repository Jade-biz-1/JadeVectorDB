"""
Structured logging configuration using structlog.
- Debug mode  → human-readable coloured console output
- Production  → JSON lines (one log event per line, easy to ship to any log aggregator)

Usage anywhere in the codebase:
    from .logging_config import get_logger
    log = get_logger(__name__)
    log.info("document_uploaded", doc_id=doc_id, filename=filename)
"""

import logging
import sys
import structlog
from .utils.config import settings


def configure_logging() -> None:
    """
    Call once at application startup (from main.py lifespan).
    Idempotent — safe to call multiple times.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.debug:
        # Pretty, coloured output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON lines for production log aggregation
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given name."""
    return structlog.get_logger(name)
