"""Forensic auditing of the Parquet data lake.

A learning system is only as honest as its data, so before any tensor is
built the lake is audited against the *exchange calendar* — the one source
of truth about when bars should exist. This module answers, per instrument:

* Which trading sessions are missing entirely? (``missing-session``)
* Which sessions are present but suspiciously thin — fewer than 90% of the
  bars the calendar says a full (or half) day should contain?
  (``short-session``)
* Are there duplicated timestamps, impossible prices (``<= 0``), inverted
  ``high < low`` bars, or single-bar moves so violent (|log return| > 0.2 ≈
  ±22% in one bar) that they smell like bad prints? Those are *flagged*,
  never deleted — Aether's philosophy is that the model learns from raw
  reality, and a human (or a later self-diagnosis layer) decides what a
  flag means. Deleting data silently would be a form of hand-crafted
  feature engineering by omission.

Every finding is a :class:`QualityIssue`; :func:`audit_all` sweeps the
whole lake and writes a machine-readable ``quality_report.json`` next to
the data so runs are comparable over time.

Severity vocabulary
-------------------
* ``critical`` — data is self-contradictory (non-positive price,
  high < low): downstream feature math (logs, range fractions) would
  produce NaN/garbage.
* ``error``    — data is absent or ambiguous (missing session, duplicate
  timestamps): windows built over these regions are structurally wrong.
* ``warning``  — data is present but suspicious (short session, extreme
  bar-to-bar move, bars on a non-trading day): usable, but flagged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import AetherConfig
from ..utils.logging import get_logger
from ..utils.market_time import (SESSION_OPEN, expected_session_minutes,
                                 is_trading_day, session_close, trading_days)
from .storage import ParquetStore

logger = get_logger("aether.quality")

#: A session with fewer than this fraction of its calendar-expected bars is
#: flagged short. 90% tolerates the routine handful of zero-volume minutes
#: vendors drop while still catching data-feed outages.
SHORT_SESSION_THRESHOLD: float = 0.90

#: |log return| beyond this between *consecutive intraday bars* is flagged.
#: 0.2 in log space ≈ a ±22% move in one bar — almost always a bad print or
#: an unadjusted corporate action leaking through, not real trading.
EXTREME_LOG_RETURN: float = 0.20

#: OHLC columns subject to the positivity / consistency checks.
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass
class QualityIssue:
    """One audit finding.

    Attributes
    ----------
    kind:
        Machine-readable issue type, e.g. ``missing-session``,
        ``short-session``, ``duplicate-timestamp``, ``non-positive-price``,
        ``high-low-inversion``, ``extreme-return``, ``off-calendar-session``,
        ``missing-day``.
    ticker:
        Storage alias the issue belongs to (may be ``_MARKET``).
    dataset:
        Lake dataset that was audited (e.g. ``bars_1min``).
    detail:
        Human-readable specifics (dates, counts, magnitudes).
    severity:
        ``critical`` | ``error`` | ``warning`` (see module docstring).
    """

    kind: str
    ticker: str
    dataset: str
    detail: str
    severity: str


# --------------------------------------------------------------------------- #
# Intraday audit
# --------------------------------------------------------------------------- #

def audit_intraday(store: ParquetStore, dataset: str, ticker: str,
                   interval_minutes: int) -> list[QualityIssue]:
    """Audit one intraday bar dataset against the NYSE session calendar.

    Bars are grouped by session date. The audited date range is
    [first stored bar, last stored bar] — the lake is judged only on the
    span it claims to cover, so a ticker whose vendor history starts late
    is not blamed for sessions before its first bar.

    Parameters
    ----------
    interval_minutes:
        Bar width (1 for ``bars_1min``, 5 for ``bars_5min``); the expected
        per-session bar count is ``expected_session_minutes(day) //
        interval_minutes``, which automatically handles 13:00 half-days.
    """
    issues: list[QualityIssue] = []
    df = store.read(dataset, ticker)
    if df.empty or "date" not in df.columns:
        return issues

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    day = df["date"].dt.date

    # ---- missing sessions: calendar days with zero bars ----------------- #
    present_days = set(day.unique())
    for d in trading_days(day.iloc[0], day.iloc[-1]):
        if d not in present_days:
            issues.append(QualityIssue(
                "missing-session", ticker, dataset,
                f"no bars for trading session {d.isoformat()}", "error"))

    # ---- per-session checks --------------------------------------------- #
    for d, g in df.groupby(day):
        prefix = f"session {d.isoformat()}"

        if not is_trading_day(d):
            # Bars stamped on a weekend/holiday: the vendor disagrees with
            # the exchange calendar. Flag it and skip completeness checks
            # (there is no 'expected' bar count for a closed market).
            issues.append(QualityIssue(
                "off-calendar-session", ticker, dataset,
                f"{prefix}: {len(g)} bars on a non-trading day", "warning"))
        else:
            # Completeness: count only bars inside regular hours so a feed
            # that includes pre/post-market cannot mask a hollow session.
            close_t = session_close(d)
            times = g["date"].dt.time
            n_in_session = int(((times >= SESSION_OPEN) & (times < close_t)).sum())
            expected = expected_session_minutes(d) // interval_minutes
            if expected > 0 and n_in_session < SHORT_SESSION_THRESHOLD * expected:
                issues.append(QualityIssue(
                    "short-session", ticker, dataset,
                    f"{prefix}: {n_in_session}/{expected} regular-hours bars "
                    f"(<{SHORT_SESSION_THRESHOLD:.0%})", "warning"))

        # Duplicate timestamps: two bars claiming the same minute make the
        # sequence ambiguous (which one is real?).
        n_dup = int(g["date"].duplicated().sum())
        if n_dup:
            issues.append(QualityIssue(
                "duplicate-timestamp", ticker, dataset,
                f"{prefix}: {n_dup} duplicated bar timestamps", "error"))

        issues.extend(_price_integrity_issues(g, ticker, dataset, prefix))

        # Extreme bar-to-bar moves, computed WITHIN the session so genuine
        # overnight gaps between sessions are never falsely flagged.
        if "close" in g.columns and len(g) >= 2:
            closes = g["close"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                # Non-positive closes become NaN; NaN comparisons are False,
                # so corrupt prices don't double-report here (they already
                # raised non-positive-price above).
                log_close = np.log(np.where(closes > 0, closes, np.nan))
            jumps = np.abs(np.diff(log_close))
            n_extreme = int((jumps > EXTREME_LOG_RETURN).sum())
            if n_extreme:
                issues.append(QualityIssue(
                    "extreme-return", ticker, dataset,
                    f"{prefix}: {n_extreme} bar-to-bar |log return| > "
                    f"{EXTREME_LOG_RETURN} (max {float(np.nanmax(jumps)):.3f})",
                    "warning"))
    return issues


def _price_integrity_issues(g: pd.DataFrame, ticker: str, dataset: str,
                            prefix: str) -> list[QualityIssue]:
    """OHLC sanity checks shared by the intraday and daily audits."""
    issues: list[QualityIssue] = []
    price_cols = [c for c in PRICE_COLUMNS if c in g.columns]

    if price_cols:
        n_bad = int((g[price_cols] <= 0).any(axis=1).sum())
        if n_bad:
            issues.append(QualityIssue(
                "non-positive-price", ticker, dataset,
                f"{prefix}: {n_bad} bars with a price <= 0", "critical"))

    if "high" in g.columns and "low" in g.columns:
        n_inv = int((g["high"] < g["low"]).sum())
        if n_inv:
            issues.append(QualityIssue(
                "high-low-inversion", ticker, dataset,
                f"{prefix}: {n_inv} bars with high < low", "critical"))
    return issues


# --------------------------------------------------------------------------- #
# Daily audit
# --------------------------------------------------------------------------- #

def audit_daily(store: ParquetStore, ticker: str) -> list[QualityIssue]:
    """Audit the daily bars of one ticker.

    Checks: every exchange trading day between the first and last stored
    bar must be present (``missing-day``), and prices must be positive /
    self-consistent.
    """
    dataset = "bars_daily"
    issues: list[QualityIssue] = []
    df = store.read(dataset, ticker)
    if df.empty or "date" not in df.columns:
        return issues

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    day = df["date"].dt.date

    present = set(day.unique())
    for d in trading_days(day.iloc[0], day.iloc[-1]):
        if d not in present:
            issues.append(QualityIssue(
                "missing-day", ticker, dataset,
                f"no daily bar for trading day {d.isoformat()}", "error"))

    issues.extend(_price_integrity_issues(
        df, ticker, dataset,
        f"range {day.iloc[0].isoformat()}..{day.iloc[-1].isoformat()}"))
    return issues


# --------------------------------------------------------------------------- #
# Lake-wide sweep
# --------------------------------------------------------------------------- #

#: dataset name -> audit dispatcher. Only bar datasets have calendar
#: expectations; news/fundamentals/macro are event streams with no "should
#: exist at time t" ground truth to audit against.
_INTRADAY_INTERVALS: dict[str, int] = {"bars_1min": 1, "bars_5min": 5}


def audit_all(store: ParquetStore, cfg: AetherConfig) -> dict:
    """Audit every auditable (dataset, ticker) pair present in the lake.

    Walks the store's manifest (so only materialized data is audited),
    writes the full findings to ``cfg.data.root / 'quality_report.json'``,
    and returns ``{'issues': [...], 'counts': {...}}`` where issues are
    plain dicts (JSON-shaped, identical to what was written).
    """
    issues: list[QualityIssue] = []
    for key in sorted(store.catalog()):
        dataset, _, ticker = key.partition("/")
        if dataset in _INTRADAY_INTERVALS:
            issues.extend(audit_intraday(store, dataset, ticker,
                                         _INTRADAY_INTERVALS[dataset]))
        elif dataset == "bars_daily":
            issues.extend(audit_daily(store, ticker))
        # Other datasets: no calendar contract to audit against.

    counts: dict[str, dict[str, int] | int] = {
        "total": len(issues),
        "by_kind": {},
        "by_severity": {},
        "by_dataset": {},
    }
    for issue in issues:
        for bucket, field in (("by_kind", issue.kind),
                              ("by_severity", issue.severity),
                              ("by_dataset", issue.dataset)):
            counts[bucket][field] = counts[bucket].get(field, 0) + 1  # type: ignore[union-attr]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "issues": [asdict(i) for i in issues],
        "counts": counts,
    }
    report_path = cfg.data.root / "quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(report_path)  # atomic on POSIX

    logger.info("quality audit: %d issues (%s) -> %s",
                len(issues),
                ", ".join(f"{k}={v}" for k, v in
                          sorted(counts["by_severity"].items())) or "clean",  # type: ignore[union-attr]
                report_path)
    return report
