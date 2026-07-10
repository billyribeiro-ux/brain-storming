"""Structured logging for Aether.

One consistent logger across ingestion, training, and (later) live trading.
Console output is human-readable; an optional JSONL sink gives machine-
readable run history that later layers (self-diagnosis, autopsies) can mine.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


class JsonLinesHandler(logging.Handler):
    """Appends every record as one JSON object per line."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra structured fields passed via `extra={...}`.
        for key, value in record.__dict__.items():
            if key.startswith("aether_"):
                payload[key[7:]] = value
        with self.path.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")


def get_logger(name: str, jsonl_path: str | Path | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent per name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(console)
    if jsonl_path is not None:
        logger.addHandler(JsonLinesHandler(jsonl_path))
    logger.propagate = False
    return logger
