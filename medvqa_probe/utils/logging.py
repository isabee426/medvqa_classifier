"""Simple logging setup used across the project."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, name: str = "medvqa_probe") -> logging.Logger:
    """Return a configured logger that writes to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
