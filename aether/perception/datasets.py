"""Parquet lake -> :class:`PerceptionBatch`.

This module is the bridge between the raw data lake
(:class:`aether.data.storage.ParquetStore`) and the perception model: it
slices the continuous 1min/5min/daily streams into fixed-geometry training
windows (:class:`aether.perception.interfaces.WindowSpec`) and collates them
into contract-exact batches.

Anchoring model
---------------
A *sample* is anchored at one 1-minute bar ``t`` ("now" = that bar's close).
The sample carries three synchronized views of the past:

* the last ``len_1m`` 1min bars **including** the anchor bar itself (its
  close defines "now"),
* the last ``len_5m`` *completed* 5min bars,
* the last ``len_daily`` daily bars of strictly previous sessions,

plus a self-supervised target: the log return from the anchor close to the
close ``horizon_1m`` minutes later **within the same session** (anchors whose
horizon crosses the close are skipped — overnight is a different process and
must not contaminate the intraday target).

The look-ahead invariant (the most important line in this file)
---------------------------------------------------------------
A bar stamped ``T`` covers the *interval after* ``T``: a 5min bar stamped
09:30 aggregates trading in (09:30, 09:35].  It is therefore knowable only
once the market clock passes ``T + 5min``.  The anchor 1min bar stamped ``t``
closes at ``t + 1min``, so a 5min bar may enter the window iff::

    bar_start + 5min <= anchor_time + 1min

A partially-formed 5min bar is never visible.  Similarly, daily context uses
only sessions with ``date < anchor date`` — the anchor day's own daily bar
aggregates the *whole* session, including the anchor's future.

Padding
-------
Streams shorter than their window (early lake history) are LEFT-padded with
zeros and flagged ``True`` in the pad masks, matching ``PerceptionBatch``.
With ``require_full_1m=True`` (the default) the 1m stream is never padded;
anchors without a full 1m history are simply not emitted.

Normalization discipline
------------------------
All feature computation goes through the frozen, training-range-fitted
:class:`TickerStats` — validation loaders reuse the *training* stats
(see :func:`build_dataloaders`), so no statistic ever leaks from the future
or from the validation set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import TICKER_IDS, TICKERS, AetherConfig
from ..data.storage import ParquetStore
from ..utils.logging import get_logger
from ..utils.market_time import BARS_5MIN_PER_SESSION, MINUTES_PER_SESSION
from .interfaces import (
    N_BAR_FEATURES,
    N_DAILY_FEATURES,
    FEATURE_COLUMNS_BAR,
    FEATURE_COLUMNS_DAILY,
    PerceptionBatch,
    PerceptionConfig,
    TrainConfig,
    WindowSpec,
)
from .preprocessing import (
    TickerStats,
    compute_bar_features,
    compute_daily_features,
    filter_regular_session,
)

logger = get_logger("aether.perception.datasets")

#: Lake dataset names (see aether.data.endpoints).
DS_1MIN = "bars_1min"
DS_5MIN = "bars_5min"
DS_DAILY = "bars_daily"

#: Bar durations in seconds — the constants of the look-ahead inequality.
BAR_1M_SECONDS = 60
BAR_5M_SECONDS = 300

_OHLCV = ["date", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
# All lake timestamps are naive US/Eastern wall-clock times.  We encode them
# as int64 "epoch seconds of the ET wall clock" (i.e. the naive datetime
# interpreted as if it were UTC).  This is purely an internal, monotonic,
# arithmetic-friendly encoding used for window math and bookkeeping — every
# comparison happens between values encoded the same way, so the missing
# UTC offset cancels out.

def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    return ts.to_numpy().astype("datetime64[s]").astype(np.int64)


def _day_numbers(ts: pd.Series) -> np.ndarray:
    """Days-since-epoch ints; the session identity used for daily context
    (strictly previous sessions) and same-session target checks."""
    return ts.to_numpy().astype("datetime64[D]").astype(np.int64)


def _ts_seconds(t: pd.Timestamp) -> int:
    return int(t.value // 1_000_000_000)


# --------------------------------------------------------------------------- #
# Per-ticker preprocessed arrays
# --------------------------------------------------------------------------- #

@dataclass
class _TickerArrays:
    """Everything one ticker contributes, precomputed once at init.

    Holding plain numpy arrays (instead of DataFrames) keeps ``__getitem__``
    allocation-light and makes the dataset cheaply picklable into DataLoader
    worker processes.
    """

    alias: str
    ticker_id: int                # index into config.TICKERS order
    feats_1m: np.ndarray          # [N1, N_BAR_FEATURES] float32
    ts_1m: np.ndarray             # [N1] int64 epoch-seconds (bar start)
    sm_1m: np.ndarray             # [N1] int64 session minute 0..389
    day_1m: np.ndarray            # [N1] int64 day number (session id)
    close_1m: np.ndarray          # [N1] float64 raw closes (target math)
    feats_5m: np.ndarray          # [N5, N_BAR_FEATURES] float32
    ts_5m: np.ndarray             # [N5] int64 epoch-seconds (bar start)
    feats_daily: np.ndarray       # [ND, N_DAILY_FEATURES] float32
    day_daily: np.ndarray         # [ND] int64 day number


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class PerceptionWindowDataset(Dataset):
    """Sliding-window dataset over the Parquet lake for all tickers.

    Parameters
    ----------
    store:
        The lake.  Read once during ``__init__``; not retained afterwards.
    tickers:
        Ticker *aliases* (config.TICKERS order defines their integer ids).
        Tickers missing stats or lake data are skipped with a warning —
        a partial universe degrades coverage, never correctness.
    spec:
        Window geometry.  ``spec.horizon_1m`` also defines the SSL target.
    start, end:
        Inclusive anchor date range (``YYYY-MM-DD``).  Extra *context*
        history before ``start`` is read automatically so that anchors near
        ``start`` still see full windows — context is input, anchors are
        what the range selects.
    stats:
        alias -> frozen training-range :class:`TickerStats`.
    stride:
        Emit an anchor every ``stride``-th eligible 1min bar.  Adjacent
        1-minute windows overlap by len_1m-1 bars; striding removes most of
        that redundancy and shrinks the index ~stride-fold.
    require_full_1m:
        If True (default), only anchors with a complete ``len_1m`` history
        are emitted and ``pad_1m`` is all-False.  If False, short 1m
        histories are left-padded like the other streams.
    """

    def __init__(self, store: ParquetStore, tickers: list[str],
                 spec: WindowSpec, start: str, end: str,
                 stats: dict[str, TickerStats], stride: int = 5,
                 require_full_1m: bool = True):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.spec = spec
        self.stride = int(stride)
        self.require_full_1m = bool(require_full_1m)

        start_ts = pd.Timestamp(start).normalize()
        end_excl = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        ctx_start = start_ts - pd.Timedelta(days=self._context_days(spec))
        start_s = _ts_seconds(start_ts)
        end_s = _ts_seconds(end_excl)

        self._data: list[_TickerArrays] = []
        slot_ids: list[np.ndarray] = []
        bar_ids: list[np.ndarray] = []

        for alias in tickers:
            st = stats.get(alias)
            if st is None:
                logger.warning("dataset: no TickerStats for %s — skipped",
                               alias)
                continue
            arrays = self._load_ticker(store, alias, st, ctx_start, end_excl)
            if arrays is None:
                continue
            cand = self._valid_anchors(arrays, start_s, end_s)
            if cand.size == 0:
                logger.warning("dataset: %s has no valid anchors in "
                               "[%s, %s] — skipped", alias, start, end)
                continue
            slot = len(self._data)
            self._data.append(arrays)
            slot_ids.append(np.full(cand.size, slot, dtype=np.int64))
            bar_ids.append(cand)
            logger.info("dataset: %s -> %d anchors (%d 1m bars, %d 5m bars, "
                        "%d daily rows)", alias, cand.size,
                        arrays.ts_1m.size, arrays.ts_5m.size,
                        arrays.day_daily.size)

        if slot_ids:
            self._anchor_slot = np.concatenate(slot_ids)
            self._anchor_idx = np.concatenate(bar_ids)
        else:
            self._anchor_slot = np.zeros(0, dtype=np.int64)
            self._anchor_idx = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _context_days(spec: WindowSpec) -> int:
        """Calendar days of pre-``start`` history to read.

        Enough *sessions* to fill the deepest window, converted to calendar
        days with the 5-trading-days-per-week ratio plus generous slack for
        holiday clusters.  Over-reading costs a little IO once at init;
        under-reading would silently shrink early windows.
        """
        sessions = max(
            math.ceil(spec.len_1m / MINUTES_PER_SESSION) + 1,
            math.ceil(spec.len_5m / BARS_5MIN_PER_SESSION) + 1,
            spec.len_daily + 1,
        )
        return math.ceil(sessions * 7 / 5) + 10

    @staticmethod
    def _load_ticker(store: ParquetStore, alias: str, st: TickerStats,
                     ctx_start: pd.Timestamp, end_excl: pd.Timestamp
                     ) -> _TickerArrays | None:
        """Read + featurize one ticker's three streams. None if unusable."""
        raw1 = store.read(DS_1MIN, alias, ctx_start, end_excl)
        if raw1.empty or not set(_OHLCV) <= set(raw1.columns):
            logger.warning("dataset: %s has no usable 1min bars — skipped",
                           alias)
            return None
        bars1 = filter_regular_session(raw1[_OHLCV])

        raw5 = store.read(DS_5MIN, alias, ctx_start, end_excl)
        bars5 = filter_regular_session(raw5[_OHLCV]) \
            if not raw5.empty and set(_OHLCV) <= set(raw5.columns) \
            else pd.DataFrame(columns=_OHLCV)

        rawd = store.read(DS_DAILY, alias, ctx_start, end_excl)
        dailyd = rawd[[c for c in _OHLCV if c in rawd.columns]] \
            if not rawd.empty else pd.DataFrame(columns=_OHLCV)

        # Featurize each stream ONCE over the full slice; windows are then
        # O(1) views.  Feature row i depends only on rows <= i, so slicing
        # any prefix later is still causal.
        f1 = compute_bar_features(bars1, st, interval_minutes=1)
        f5 = compute_bar_features(bars5, st, interval_minutes=5)
        fd = compute_daily_features(dailyd, st)
        if f1.empty:
            logger.warning("dataset: %s 1min bars all filtered out — skipped",
                           alias)
            return None

        return _TickerArrays(
            alias=alias,
            ticker_id=TICKER_IDS[alias],
            feats_1m=f1[list(FEATURE_COLUMNS_BAR)].to_numpy(np.float32),
            ts_1m=_epoch_seconds(f1["date"]),
            sm_1m=f1["session_minute"].to_numpy(np.int64),
            day_1m=_day_numbers(f1["date"]),
            close_1m=bars1["close"].to_numpy(np.float64),
            feats_5m=(f5[list(FEATURE_COLUMNS_BAR)].to_numpy(np.float32)
                      if not f5.empty
                      else np.zeros((0, N_BAR_FEATURES), np.float32)),
            ts_5m=(_epoch_seconds(f5["date"]) if not f5.empty
                   else np.zeros(0, np.int64)),
            feats_daily=(fd[list(FEATURE_COLUMNS_DAILY)].to_numpy(np.float32)
                         if not fd.empty
                         else np.zeros((0, N_DAILY_FEATURES), np.float32)),
            day_daily=(_day_numbers(fd["date"]) if not fd.empty
                       else np.zeros(0, np.int64)),
        )

    def _valid_anchors(self, d: _TickerArrays, start_s: int,
                       end_s: int) -> np.ndarray:
        """Indices of eligible anchor bars, fully vectorized.

        An anchor is every ``stride``-th 1min bar inside [start, end) that
        additionally satisfies:

        (a) full 1m history if ``require_full_1m`` — the window
            ``[i - len_1m + 1, i]`` must fit in the array;
        (b) a same-session target ``horizon_1m`` bars ahead;
        (c) at least one *completed* 5min bar (look-ahead inequality) and
            at least one strictly-previous daily bar — encoders always get
            at least one real token per stream.
        """
        spec = self.spec
        in_range = np.nonzero((d.ts_1m >= start_s) & (d.ts_1m < end_s))[0]
        cand = in_range[:: self.stride]

        if self.require_full_1m:
            cand = cand[cand >= spec.len_1m - 1]

        # (b) target inside the array AND inside the same session.
        h = spec.horizon_1m
        cand = cand[cand + h < d.ts_1m.size]
        cand = cand[d.day_1m[cand + h] == d.day_1m[cand]]
        if cand.size == 0:
            return cand

        # (c) completed-5min count: bar_start + 300 <= anchor_ts + 60
        #     <=> bar_start <= anchor_ts - 240; ts_5m is sorted, so a right
        #     bisect counts exactly the admissible bars.
        n5 = np.searchsorted(
            d.ts_5m, d.ts_1m[cand] + BAR_1M_SECONDS - BAR_5M_SECONDS,
            side="right")
        cand = cand[n5 >= 1]
        if cand.size == 0:
            return cand

        # (c) prior-daily count: strictly earlier sessions only.
        ndaily = np.searchsorted(d.day_daily, d.day_1m[cand], side="left")
        return cand[ndaily >= 1]

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return int(self._anchor_idx.size)

    def anchor(self, i: int) -> tuple[str, int, int]:
        """(alias, bar index, anchor epoch-seconds) — for tests/audits."""
        slot = int(self._anchor_slot[i])
        idx = int(self._anchor_idx[i])
        return self._data[slot].alias, idx, int(self._data[slot].ts_1m[idx])

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        spec = self.spec
        slot = int(self._anchor_slot[i])
        idx = int(self._anchor_idx[i])
        d = self._data[slot]
        L1, L5, LD = spec.len_1m, spec.len_5m, spec.len_daily

        # ---- 1m window: [idx - L1 + 1, idx], anchor bar included -------- #
        lo1 = max(0, idx - L1 + 1)
        w1 = d.feats_1m[lo1: idx + 1]
        n1 = w1.shape[0]                       # == L1 when require_full_1m
        bars_1m = np.zeros((L1, N_BAR_FEATURES), dtype=np.float32)
        bars_1m[L1 - n1:] = w1
        pad_1m = np.ones(L1, dtype=np.bool_)
        pad_1m[L1 - n1:] = False
        minute_of_day = np.zeros(L1, dtype=np.int64)   # 0 where padded
        minute_of_day[L1 - n1:] = d.sm_1m[lo1: idx + 1]

        # ---- 5m window: only COMPLETED bars (the look-ahead rule) ------- #
        # Admissible iff bar_start + 5min <= anchor_time + 1min.  k is the
        # count of admissible bars; the newest included bar is index k-1.
        anchor_ts = int(d.ts_1m[idx])
        k = int(np.searchsorted(
            d.ts_5m, anchor_ts + BAR_1M_SECONDS - BAR_5M_SECONDS,
            side="right"))
        lo5 = max(0, k - L5)
        w5 = d.feats_5m[lo5:k]
        n5 = w5.shape[0]
        bars_5m = np.zeros((L5, N_BAR_FEATURES), dtype=np.float32)
        bars_5m[L5 - n5:] = w5
        pad_5m = np.ones(L5, dtype=np.bool_)
        pad_5m[L5 - n5:] = False

        # ---- daily window: strictly previous sessions ------------------- #
        nd = int(np.searchsorted(d.day_daily, d.day_1m[idx], side="left"))
        lod = max(0, nd - LD)
        wd = d.feats_daily[lod:nd]
        ndl = wd.shape[0]
        daily = np.zeros((LD, N_DAILY_FEATURES), dtype=np.float32)
        daily[LD - ndl:] = wd
        pad_daily = np.ones(LD, dtype=np.bool_)
        pad_daily[LD - ndl:] = False

        # ---- SSL target: same-session forward log return ----------------- #
        target = math.log(d.close_1m[idx + spec.horizon_1m]
                          / d.close_1m[idx])

        # Keys mirror PerceptionBatch fields; collate_batch tensorizes.
        return {
            "bars_1m": bars_1m,
            "bars_5m": bars_5m,
            "daily": daily,
            "ticker_id": np.int64(d.ticker_id),
            "minute_of_day": minute_of_day,
            "pad_1m": pad_1m,
            "pad_5m": pad_5m,
            "pad_daily": pad_daily,
            "target_ret": np.float32(target),
            "anchor_ts": np.int64(anchor_ts),
        }


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #

def collate_batch(items: list[dict[str, np.ndarray]]) -> PerceptionBatch:
    """Stack per-sample dicts into a contract-exact :class:`PerceptionBatch`."""

    def stack(key: str, dtype: type) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([np.asarray(it[key]) for it in items]).astype(
                dtype, copy=False))

    def scalars(key: str, dtype: type) -> torch.Tensor:
        return torch.from_numpy(np.asarray([it[key] for it in items],
                                           dtype=dtype))

    return PerceptionBatch(
        bars_1m=stack("bars_1m", np.float32),
        bars_5m=stack("bars_5m", np.float32),
        daily=stack("daily", np.float32),
        ticker_id=scalars("ticker_id", np.int64),
        minute_of_day=stack("minute_of_day", np.int64),
        pad_1m=stack("pad_1m", np.bool_),
        pad_5m=stack("pad_5m", np.bool_),
        pad_daily=stack("pad_daily", np.bool_),
        target_ret=scalars("target_ret", np.float32),
        anchor_ts=scalars("anchor_ts", np.int64),
    )


# --------------------------------------------------------------------------- #
# Stats management
# --------------------------------------------------------------------------- #

def fit_or_load_stats(store: ParquetStore, tickers: list[str],
                      train_start: str, train_end: str,
                      stats_root: Path) -> dict[str, TickerStats]:
    """Load cached :class:`TickerStats` or fit them from the lake.

    A cached JSON is reused only if its fitted range *covers* the requested
    training range — a cache fitted on less data than requested would be a
    different (weaker) fingerprint, while one fitted on a superset range
    would leak nothing new but is rejected too if it starts later or ends
    earlier than requested.  Fitting reads ONLY ``[train_start, train_end]``
    from the lake: the frozen stats are the sole channel through which
    training data influences validation preprocessing, and no channel exists
    in the other direction.
    """
    stats_root = Path(stats_root)
    stats_root.mkdir(parents=True, exist_ok=True)
    t_start = pd.Timestamp(train_start).normalize()
    t_end = pd.Timestamp(train_end).normalize()

    out: dict[str, TickerStats] = {}
    for alias in tickers:
        path = stats_root / f"{alias}.json"
        if path.is_file():
            try:
                st = TickerStats.from_json(path)
                if (pd.Timestamp(st.fitted_start) <= t_start
                        and pd.Timestamp(st.fitted_end) >= t_end):
                    out[alias] = st
                    continue
                logger.info("stats: %s cache range [%s, %s] does not cover "
                            "train range [%s, %s] — refitting", alias,
                            st.fitted_start, st.fitted_end,
                            train_start, train_end)
            except Exception as exc:  # corrupt/stale cache file
                logger.warning("stats: failed to load %s (%s) — refitting",
                               path, exc)

        # Intraday read overshoots by a day so train_end's own session is
        # included (store.read compares against midnight); the session
        # filter removes anything outside regular hours.
        bars1 = store.read(DS_1MIN, alias, t_start,
                           t_end + pd.Timedelta(days=1))
        if bars1.empty or not set(_OHLCV) <= set(bars1.columns):
            logger.warning("stats: no 1min bars for %s in train range — "
                           "ticker excluded", alias)
            continue
        bars1 = filter_regular_session(bars1[_OHLCV])
        bars1 = bars1[bars1["date"] < t_end + pd.Timedelta(days=1)]
        if bars1.empty:
            logger.warning("stats: %s train-range bars all filtered out — "
                           "ticker excluded", alias)
            continue
        # Daily rows are stamped at midnight, so an inclusive end is exact.
        daily = store.read(DS_DAILY, alias, t_start, t_end)

        st = TickerStats.fit(bars1, daily, alias,
                             fitted_start=str(t_start.date()),
                             fitted_end=str(t_end.date()))
        st.to_json(path)
        out[alias] = st
    return out


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #

def build_dataloaders(cfg: AetherConfig, perception_cfg: PerceptionConfig,
                      train_cfg: TrainConfig,
                      train_range: tuple[str, str],
                      val_range: tuple[str, str]
                      ) -> tuple[DataLoader, DataLoader]:
    """Assemble (train_loader, val_loader) for perception pretraining.

    * ``TickerStats`` are fitted (or loaded) on the TRAIN range only and
      shared with the validation dataset — the leak-free direction.
    * The train loader shuffles (with a seeded generator for reproducible
      epochs); validation preserves time order for readable eval traces.
    * ``pin_memory`` follows CUDA availability; the code itself stays
      device-agnostic.
    """
    store = ParquetStore(cfg.data.root)
    tickers = [t.alias for t in TICKERS]
    stats = fit_or_load_stats(store, tickers, train_range[0], train_range[1],
                              cfg.data.stats_root)
    if not stats:
        raise RuntimeError(
            "build_dataloaders: no ticker has usable training data in "
            f"[{train_range[0]}, {train_range[1]}] under {cfg.data.root} — "
            "run scripts/backfill.py first")
    available = [a for a in tickers if a in stats]

    spec = perception_cfg.window
    train_ds = PerceptionWindowDataset(store, available, spec,
                                       train_range[0], train_range[1], stats)
    val_ds = PerceptionWindowDataset(store, available, spec,
                                     val_range[0], val_range[1], stats)
    logger.info("dataloaders: %d train / %d val samples across %d tickers",
                len(train_ds), len(val_ds), len(available))

    pin = torch.cuda.is_available()
    gen = torch.Generator()
    gen.manual_seed(train_cfg.seed)
    common = dict(batch_size=train_cfg.batch_size,
                  num_workers=train_cfg.num_workers,
                  collate_fn=collate_batch, pin_memory=pin,
                  persistent_workers=train_cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False,
                              generator=gen, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
