"""Shared synthetic-market fixtures and helpers for the Aether test suite.

Everything in this file is built from the *contract* alone
(``aether.perception.interfaces``, ``aether.data.storage``,
``aether.utils.market_time``, ``aether.config``) so the tests pin behavior
without depending on implementation details of the modules under test —
those modules are written concurrently against the same contract.

Synthetic data design
---------------------
``make_synthetic_1min`` produces a geometric random walk sampled on REAL
NYSE trading days (via ``aether.utils.market_time.trading_days``) starting
2026-01-05, one row per regular-session minute (09:30..15:59 ET, naive
timestamps — exactly how FMP quotes intraday bars). Volume follows the
classic U-shaped intraday profile so time-of-day volume normalization has
real structure to normalize away. Every OHLC row is internally consistent
by construction::

    high >= max(open, close) >= min(open, close) >= low > 0

``aggregate_5min`` mirrors vendor convention: each 5-minute bar is stamped
with its window START and aggregates the minutes [T, T+5), i.e. the bar's
trade flow covers (T, T+5] and the bar only *completes* — becomes usable
without look-ahead — at T+5. ``make_daily`` collapses each session into a
single OHLCV row stamped at session-date midnight.

Anchor timestamp decoding
-------------------------
``PerceptionBatch.anchor_ts`` is specified only as "int64 epoch-seconds
(bookkeeping)". A dataset implementation may compute it from naive-ET
timestamps viewed as UTC, or from properly ET-localized timestamps.
``anchor_wall_time`` disambiguates by checking which decoding reproduces
the session minute the batch itself reports in ``minute_of_day`` — the two
candidate wall times differ by the 4-5h UTC offset, so exactly one of them
can match.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aether.data.storage import ParquetStore
from aether.perception.interfaces import PerceptionBatch, WindowSpec
from aether.utils.market_time import (
    expected_session_minutes,
    session_minute,
    trading_days,
)

# --------------------------------------------------------------------------- #
# Constants shared across test modules
# --------------------------------------------------------------------------- #

#: First synthetic session — a real NYSE Monday.
SYNTH_START = date(2026, 1, 5)

#: Number of sessions in the default synthetic lake.
SYNTH_SESSIONS = 6

#: Tickers used by the synthetic lake. Both are real entries of
#: ``aether.config.TICKERS`` so ``TICKER_IDS`` lookups resolve (AAPL=0, SPY=6).
TEST_TICKERS: tuple[str, ...] = ("AAPL", "SPY")

#: Dataset names in the Parquet lake (the ingestion convention).
DATASET_1M = "bars_1min"
DATASET_5M = "bars_5min"
DATASET_DAILY = "bars_daily"

#: Training range used to fit TickerStats and to build datasets. The end
#: timestamp is late in the day so the store's inclusive [start, end] slice
#: keeps the final session's intraday bars.
TRAIN_START = pd.Timestamp("2026-01-05")
TRAIN_END = pd.Timestamp("2026-01-12 23:59:00")

#: Stride used for every PerceptionWindowDataset built in the tests.
#: 7 is coprime with 5, so strided anchors sweep all residues modulo 5 —
#: guaranteeing the 5-minute-aligned anchors the look-ahead tests search for.
STRIDE = 7


def synth_sessions(n: int = SYNTH_SESSIONS) -> list[date]:
    """The first ``n`` real trading days starting at ``SYNTH_START``."""
    days = trading_days(SYNTH_START, SYNTH_START + timedelta(days=4 * n + 40))[:n]
    assert len(days) == n, "not enough trading days in the probe range"
    return days


# --------------------------------------------------------------------------- #
# Synthetic OHLCV builders
# --------------------------------------------------------------------------- #

def make_synthetic_1min(alias: str, sessions: int = SYNTH_SESSIONS,
                        seed: int = 0) -> pd.DataFrame:
    """Geometric-random-walk 1-minute bars on real NYSE sessions.

    Parameters
    ----------
    alias:
        Ticker alias; folded into the RNG seed (and the base price) so each
        instrument gets a distinct but reproducible path.
    sessions:
        Number of consecutive trading sessions starting 2026-01-05.
    seed:
        Base RNG seed; the same (alias, sessions, seed) triple always yields
        the identical frame.

    Returns
    -------
    DataFrame with columns ``date, open, high, low, close, volume`` — one row
    per regular-session minute, naive ET timestamps, strictly increasing.
    """
    rng = np.random.default_rng(seed * 1_000_003 + sum(alias.encode()))
    days = synth_sessions(sessions)

    # Distinct but deterministic base price per instrument.
    price = 50.0 + float(sum(alias.encode()) % 200)
    rows: list[tuple] = []
    for day in days:
        n_min = expected_session_minutes(day)      # 390, or 210 on half days
        price *= float(np.exp(rng.normal(0.0, 2e-3)))   # overnight gap
        half = (n_min - 1) / 2.0
        for m in range(n_min):
            ts = datetime.combine(day, time(9, 30)) + timedelta(minutes=m)
            # Bar path: open near previous close, close via a small log step,
            # then push high/low outward so consistency holds by construction.
            o = price * float(np.exp(rng.normal(0.0, 1e-4)))
            c = o * float(np.exp(rng.normal(0.0, 8e-4)))
            h = max(o, c) * float(np.exp(abs(rng.normal(0.0, 4e-4))))
            lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, 4e-4))))
            # U-shaped volume: heavy at the open/close, light at midday.
            u = (m - half) / half                       # -1 .. +1 over the day
            vol = int(round(3000.0 * (0.4 + 1.2 * u * u)
                            * float(rng.lognormal(0.0, 0.35)))) + 1
            rows.append((ts, o, h, lo, c, vol))
            price = c
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                     "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 5-minute bars labeled by window START.

    The bar stamped ``T`` contains minutes ``[T, T+5)`` — its trade flow is
    ``(T, T+5]`` — so it is complete (usable without look-ahead) only once
    the wall clock reaches ``T + 5min``.
    """
    df = bars_1min.sort_values("date")
    bucket = df["date"].dt.floor("5min")               # 09:30 stays 09:30
    g = df.groupby(bucket)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    out.index.name = "date"
    return out.reset_index()


def make_daily(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """Collapse each session's 1-minute bars into one OHLCV row.

    The row is stamped at session-date midnight, matching how a daily-bars
    endpoint keys its data.
    """
    df = bars_1min.sort_values("date")
    day = df["date"].dt.normalize()
    g = df.groupby(day)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    out.index.name = "date"
    return out.reset_index()


def build_lake(root: Path,
               frames_1m: dict[str, pd.DataFrame],
               frames_5m: dict[str, pd.DataFrame] | None = None,
               frames_daily: dict[str, pd.DataFrame] | None = None,
               ) -> ParquetStore:
    """Materialize a ParquetStore holding 1min/5min/daily bars.

    5-minute and daily frames default to being derived from the 1-minute
    frames; the look-ahead tests pass explicit overrides so they can perturb
    a single timescale in isolation.
    """
    store = ParquetStore(root)
    for alias, df1 in frames_1m.items():
        df5 = (frames_5m or {}).get(alias)
        dfd = (frames_daily or {}).get(alias)
        store.write(DATASET_1M, alias, df1)
        store.write(DATASET_5M, alias, df5 if df5 is not None else aggregate_5min(df1))
        store.write(DATASET_DAILY, alias,
                    dfd if dfd is not None else make_daily(df1))
    return store


# --------------------------------------------------------------------------- #
# Batch helpers (used by the dataset / look-ahead tests)
# --------------------------------------------------------------------------- #

def collate_all(ds) -> PerceptionBatch:
    """Collate every item of a PerceptionWindowDataset into one big batch."""
    from aether.perception.datasets import collate_batch  # concurrent module
    n = len(ds)
    assert n > 0, "dataset produced zero windows from the synthetic lake"
    return collate_batch([ds[i] for i in range(n)])


def anchor_wall_time(epoch_s: int, expected_minute: int) -> pd.Timestamp:
    """Decode an ``anchor_ts`` epoch value back to its naive-ET wall time.

    Tries both plausible encodings (naive-ET viewed as UTC, and true
    UTC-epoch of an ET-localized timestamp) and keeps whichever lands on the
    session minute the batch itself reported. Exactly one can match because
    the two candidates differ by the UTC offset (4-5 hours).
    """
    naive_as_utc = pd.Timestamp(epoch_s, unit="s")
    true_utc = (pd.Timestamp(epoch_s, unit="s", tz="UTC")
                .tz_convert("America/New_York").tz_localize(None))
    for cand in (naive_as_utc, true_utc):
        try:
            minute = session_minute(cand.to_pydatetime())
        except ValueError:
            continue  # candidate lands outside regular hours -> wrong decoding
        if minute == expected_minute:
            return cand
    raise AssertionError(
        f"anchor_ts={epoch_s} decodes to neither convention: "
        f"naive-as-utc={naive_as_utc}, et-from-utc={true_utc}, "
        f"expected session minute {expected_minute}"
    )


def anchor_session_minute(batch: PerceptionBatch, i: int) -> int:
    """Session minute of item ``i``'s anchor bar.

    Windows are LEFT-padded, so the last 1-minute position is always the
    anchor bar t itself (contract: "most recent 1min bars ending at t").
    """
    assert not bool(batch.pad_1m[i, -1]), "anchor position must never be padding"
    return int(batch.minute_of_day[i, -1])


def match_item(batch: PerceptionBatch, ticker_id: int, anchor_ts: int) -> int:
    """Index of the unique batch row with this (ticker, anchor) identity."""
    assert batch.anchor_ts is not None, \
        "PerceptionWindowDataset must populate anchor_ts (needed for auditing)"
    hits = ((batch.ticker_id == ticker_id)
            & (batch.anchor_ts == anchor_ts)).nonzero(as_tuple=True)[0]
    assert hits.numel() == 1, (
        f"expected exactly one item for ticker_id={ticker_id}, "
        f"anchor_ts={anchor_ts}; found {hits.numel()}"
    )
    return int(hits[0])


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def synthetic_frames() -> dict[str, pd.DataFrame]:
    """Deterministic per-ticker 1-minute frames for the synthetic lake."""
    return {a: make_synthetic_1min(a, sessions=SYNTH_SESSIONS, seed=7)
            for a in TEST_TICKERS}


@pytest.fixture()
def synthetic_store(tmp_path: Path, synthetic_frames) -> ParquetStore:
    """ParquetStore with bars_1min / bars_5min / bars_daily for AAPL and SPY."""
    return build_lake(tmp_path / "lake", synthetic_frames)


@pytest.fixture()
def small_spec() -> WindowSpec:
    """Tiny window geometry keeping every test CPU-fast."""
    return WindowSpec(len_1m=30, len_5m=12, len_daily=3, horizon_1m=1)


@pytest.fixture()
def stats(synthetic_store: ParquetStore, tmp_path: Path):
    """TickerStats fitted on the full (== training) synthetic range."""
    from aether.perception.datasets import fit_or_load_stats  # concurrent module
    return fit_or_load_stats(synthetic_store, list(TEST_TICKERS),
                             TRAIN_START, TRAIN_END, tmp_path / "ticker_stats")


@pytest.fixture()
def dataset(synthetic_store: ParquetStore, small_spec: WindowSpec, stats):
    """The reference PerceptionWindowDataset all dataset tests start from."""
    from aether.perception.datasets import PerceptionWindowDataset  # concurrent
    return PerceptionWindowDataset(synthetic_store, list(TEST_TICKERS),
                                   small_spec, TRAIN_START, TRAIN_END, stats,
                                   stride=STRIDE, require_full_1m=True)
