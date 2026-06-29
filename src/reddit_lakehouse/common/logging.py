"""Structured, idempotent logging helper used across all jobs."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler exactly once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
