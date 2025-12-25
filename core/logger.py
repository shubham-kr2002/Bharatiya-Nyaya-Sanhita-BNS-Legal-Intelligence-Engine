"""JSON-formatted logging module for production monitoring."""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from core.config import get_settings


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add ISO-8601 timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service metadata to log events."""
    settings = get_settings()
    event_dict["service"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def setup_logging() -> structlog.BoundLogger:
    """
    Configure structured JSON logging for production.
    
    Returns:
        structlog.BoundLogger: Configured logger instance.
    """
    settings = get_settings()
    
    # Determine log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Shared processors for all logging
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Development vs Production formatting
    if settings.ENVIRONMENT == "development" and settings.DEBUG:
        # Human-readable console output for development
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production monitoring (ELK, CloudWatch, etc.)
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional logger name (e.g., module name).
        
    Returns:
        structlog.BoundLogger: Logger instance.
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger bound with class name."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_async_operation(
    operation_name: str,
    **kwargs: Any,
) -> structlog.BoundLogger:
    """
    Create a logger context for async operations.
    
    Args:
        operation_name: Name of the async operation.
        **kwargs: Additional context to bind.
        
    Returns:
        structlog.BoundLogger: Logger with operation context.
    """
    return get_logger().bind(
        operation=operation_name,
        **kwargs,
    )


# Initialize logging on module import
_logger = setup_logging()


# Example usage and log format demonstration
if __name__ == "__main__":
    logger = get_logger("demo")
    
    logger.info("Application starting", port=8000)
    logger.debug("Debug information", data={"key": "value"})
    logger.warning("Warning message", threshold=0.8)
    logger.error("Error occurred", error_code="E001", details="Something went wrong")
    
    try:
        raise ValueError("Demo exception")
    except Exception:
        logger.exception("Exception caught")
