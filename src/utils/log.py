"""Single source of truth for logging configuration across the project.

Usage:
    # Entry point (CLI script, FastAPI lifespan, etc.):
    from src.utils.log import setup_logging
    setup_logging()                 # reads LOG_LEVEL env, defaults to INFO
    setup_logging(level="DEBUG")    # explicit override

    # Anywhere else — just import the shared logger:
    from src.utils.log import logger
    logger.info("hello")

The format string includes ``module:lineno`` so a single shared logger name
still gives you precise call-site provenance when reading the output.

Why this module exists:
    * One place to tune the format string and date format.
    * One place to silence chatty third-party loggers (httpx, transformers,
      sentence_transformers, ...) that otherwise flood the console.
    * Idempotent — safe to call ``setup_logging`` multiple times; only the
      first invocation actually wires handlers (use ``force=True`` to reset).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable

# ----------------------------------------------------------------- defaults
LOGGER_NAME = "rag"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(module)s:%(lineno)d | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"

# Libraries that emit INFO/DEBUG noise we never want unless explicitly debugging.
# Add entries here rather than tuning each entry point.
NOISY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "filelock",
    "PIL",
    "paddle",
    "paddlex",
    "ppocr",
)

# The single project-wide logger — import this from everywhere.
logger: logging.Logger = logging.getLogger(LOGGER_NAME)

_configured: bool = False


def _coerce_level(value: str | int | None) -> int:
    """Accept a string ('DEBUG'), a logging int, or None and return a level int."""
    if value is None:
        value = os.environ.get("LOG_LEVEL", DEFAULT_LEVEL)
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).strip().upper(), logging.INFO)


def setup_logging(
    level: str | int | None = None,
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    quiet_third_party: bool = True,
    extra_quiet: Iterable[str] = (),
    force: bool = False,
) -> None:
    """Configure root logging exactly once for the whole process.

    Resolution order for ``level``:
        1. Explicit ``level`` argument (string like ``"DEBUG"`` or int).
        2. ``LOG_LEVEL`` environment variable.
        3. ``"INFO"``.

    Subsequent calls are no-ops unless ``force=True`` is passed.
    """
    global _configured
    if _configured and not force:
        return

    resolved = _coerce_level(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    # Drop any prior handlers (e.g., from a stray basicConfig in a dependency).
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(resolved)

    if quiet_third_party:
        for name in (*NOISY_LOGGERS, *extra_quiet):
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    logger.debug("Logging configured (level=%s, quiet_third_party=%s)",
                 logging.getLevelName(resolved), quiet_third_party)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Escape hatch: get a named child logger if you ever need module-scoped control.

    Most code should just ``from src.utils.log import logger`` — this exists for
    the rare case where you want a separate logger for a sub-system.
    """
    return logging.getLogger(name)
