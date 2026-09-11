"""Minimal logging setup shared by the command-line entry points."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes timestamped INFO messages to stderr."""
    logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)
