#!/usr/bin/env python3
"""Deep-history backfill CLI.

Downloads everything from ``backfill_start`` to today into the Parquet
lake, then runs the forensic quality audit and prints a summary.

Usage::

    python3 scripts/backfill.py                      # everything
    python3 scripts/backfill.py --tickers AAPL SPX   # a subset of tickers
    python3 scripts/backfill.py --datasets bars_1min bars_5min
    python3 scripts/backfill.py --start 2023-01-01   # shallower history

The run is resumable: chunks are written as they land, and re-running
continues strictly before the lake's low watermark. Snapshot endpoints
(quote, shares float) are point-in-time and are collected by ``sync.py``
instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/backfill.py` from any cwd: the repo root (parent of
# scripts/) must be importable for the `aether` package to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio

from aether.config import TICKER_IDS, AetherConfig
from aether.data.endpoints import ENDPOINTS
from aether.data.fmp_client import FMPError
from aether.data.ingestion import IngestionEngine, RunReport
from aether.data.quality import audit_all
from aether.data.storage import ParquetStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the Aether data lake from FMP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--tickers", nargs="*", default=None, metavar="ALIAS",
        help=f"ticker aliases to ingest (default: all — "
             f"{', '.join(TICKER_IDS)})")
    parser.add_argument(
        "--datasets", nargs="*", default=None, metavar="DATASET",
        help=f"datasets to ingest (default: all — {', '.join(ENDPOINTS)})")
    parser.add_argument(
        "--start", default=None, metavar="YYYY-MM-DD",
        help="override cfg.data.backfill_start for this run")
    return parser.parse_args(argv)


def print_report(report: RunReport) -> None:
    """Render {dataset -> {ticker -> rows_new}} as a fixed-width table."""
    print()
    print(f"{'dataset':<24} {'ticker':<10} {'rows_new':>12}")
    print("-" * 48)
    total = 0
    for dataset in sorted(report):
        for ticker in sorted(report[dataset]):
            rows = report[dataset][ticker]
            total += rows
            print(f"{dataset:<24} {ticker:<10} {rows:>12,}")
    print("-" * 48)
    print(f"{'TOTAL':<35} {total:>12,}")


def print_audit_summary(audit: dict, cfg: AetherConfig) -> None:
    counts = audit.get("counts", {})
    by_sev = counts.get("by_severity", {})
    sev_text = (", ".join(f"{k}={v}" for k, v in sorted(by_sev.items()))
                or "clean")
    print(f"\nQuality audit: {counts.get('total', 0)} issues ({sev_text})")
    print(f"Full report: {cfg.data.root / 'quality_report.json'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = AetherConfig.from_env()  # loads .env implicitly
    if args.start:
        cfg.data.backfill_start = args.start

    store = ParquetStore(cfg.data.root)
    engine = IngestionEngine(cfg, store)
    try:
        report = asyncio.run(
            engine.backfill(datasets=args.datasets, tickers=args.tickers))
    except KeyError as exc:  # unknown dataset/ticker alias
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FMPError as exc:  # e.g. missing FMP_API_KEY
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    audit = audit_all(store, cfg)
    print_audit_summary(audit, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
