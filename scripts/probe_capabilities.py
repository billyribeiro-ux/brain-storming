#!/usr/bin/env python3
"""Capability probe CLI — active sensing of the FMP API.

Fires one cheap request at every registered endpoint to discover what the
current API key's plan can actually reach, saves the verdicts to
``data/capabilities.json`` (consumed by backfill/sync to skip dead
endpoints), and prints a human-readable table.

Usage::

    python3 scripts/probe_capabilities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/probe_capabilities.py` from any cwd: the repo root
# (parent of scripts/) must be importable for the `aether` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio

from aether.config import AetherConfig
from aether.data.fmp_client import FMPError
from aether.data.ingestion import IngestionEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe which FMP endpoints this API key can access.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    cfg = AetherConfig.from_env()  # loads .env implicitly
    engine = IngestionEngine(cfg)
    try:
        statuses = asyncio.run(engine.probe_and_save())
    except FMPError as exc:  # e.g. missing FMP_API_KEY
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"{'dataset':<24} {'available':<10} {'http':<6} detail")
    print("-" * 72)
    for name in sorted(statuses):
        s = statuses[name]
        mark = "yes" if s.available else "NO"
        print(f"{name:<24} {mark:<10} {s.http_status:<6} {s.detail[:60]}")
    n_ok = sum(1 for s in statuses.values() if s.available)
    print("-" * 72)
    print(f"{n_ok}/{len(statuses)} endpoints available")
    print(f"Saved to {cfg.data.capabilities_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
