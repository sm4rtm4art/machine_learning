"""Structured logging setup using structlog."""

import logging
import sys
from functools import lru_cache
from typing import Any

import structlog
from structlog.typing import Processor

from ml_portfolio.common.config import get_settings


def setup_logging(level: str | None = None, format_: str | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to settings.
        format_: Output format ('json' or 'console'). Defaults to settings.
    """
    settings = get_settings()
    level = level or settings.log_level
    format_ = format_ or settings.log_format

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_ == "json":
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


def log_context(**kwargs: Any) -> structlog.contextvars.bound_contextvars:
    """Context manager to add context to all logs within the block.

    Example:
        with log_context(run_id="abc123", project="ocr"):
            logger.info("Starting training")  # Includes run_id and project
    """
    return structlog.contextvars.bound_contextvars(**kwargs)
