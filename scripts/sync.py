#!/usr/bin/env python3
"""Incremental sync CLI.

Tops the Parquet lake up from each (dataset, ticker) high watermark to now,
re-fetching a two-day overlap so late-settling bars get merged in (the
lake's dedup-writes make the overlap free). Snapshot endpoints (quote,
shares float) append one point-in-time row per run. Afterwards the forensic
quality audit runs and a summary is printed.

Usage::

    python3 scripts/sync.py                          # everything
    python3 scripts/sync.py --tickers AAPL NVDA
    python3 scripts/sync.py --datasets bars_1min quote
    python3 scripts/sync.py --start 2024-01-01       # fallback start when a
                                                     # dataset has no watermark
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/sync.py` from any cwd: the repo root (parent of
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
        description="Incrementally sync the Aether data lake from FMP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--tickers", nargs="*", default=None, metavar="ALIAS",
        help=f"ticker aliases to sync (default: all — "
             f"{', '.join(TICKER_IDS)})")
    parser.add_argument(
        "--datasets", nargs="*", default=None, metavar="DATASET",
        help=f"datasets to sync (default: all — {', '.join(ENDPOINTS)})")
    parser.add_argument(
        "--start", default=None, metavar="YYYY-MM-DD",
        help="override cfg.data.backfill_start — only used as the fetch "
             "start for datasets that have no watermark yet")
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
            engine.sync(datasets=args.datasets, tickers=args.tickers))
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
