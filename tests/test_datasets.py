"""THE look-ahead tests — the most important invariants in the codebase.

A perception sample anchored at 1-minute bar t may see ONLY data available
at the anchor's close (t + 1min):

* 1min stream:  bars stamped <= t;
* 5min stream:  bars stamped T with T + 5min <= t + 1min (a 5min bar covers
  (T, T+5min] of trading, so it completes at T + 5min);
* daily stream: strictly previous sessions;
* normalization stats: fitted on the training range only (held fixed here).

Strategy: *perturbation audits*. We rebuild the lake after multiplying all
data BEYOND the anchor's information horizon by 3x and assert the sample is
bit-for-bit unchanged; we also multiply the newest datum *inside* the
horizon and assert the sample DOES change (so the exclusion tests cannot
pass vacuously and the windows are maximally fresh). Because vol_z depends
only on the fitted stats, the same stats object is reused for every rebuild
— any residual difference must come from raw-data leakage.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from aether.config import TICKER_IDS
from aether.perception.datasets import PerceptionWindowDataset, collate_batch
from aether.perception.interfaces import (
    FEATURE_COLUMNS_BAR,
    N_BAR_FEATURES,
    N_DAILY_FEATURES,
    PerceptionBatch,
    WindowSpec,
)

from tests.conftest import (
    STRIDE,
    TEST_TICKERS,
    TRAIN_END,
    TRAIN_START,
    aggregate_5min,
    anchor_session_minute,
    anchor_wall_time,
    build_lake,
    collate_all,
    make_daily,
    match_item,
)

LN3 = math.log(3.0)
AAPL_ID = TICKER_IDS["AAPL"]
SPY_ID = TICKER_IDS["SPY"]


# --------------------------------------------------------------------------- #
# Rebuild / perturbation machinery
# --------------------------------------------------------------------------- #

def _rebuild(tmp_path: Path, name: str, stats, spec: WindowSpec,
             frames_1m: dict[str, pd.DataFrame],
             frames_5m: dict[str, pd.DataFrame] | None = None,
             frames_daily: dict[str, pd.DataFrame] | None = None,
             ) -> PerceptionBatch:
    """Write a (possibly perturbed) lake and collate its full dataset.

    Stats, spec, date range and stride are identical to the reference
    dataset, so items line up 1:1 by (ticker_id, anchor_ts).
    """
    store = build_lake(tmp_path / name, frames_1m, frames_5m, frames_daily)
    ds = PerceptionWindowDataset(store, list(TEST_TICKERS), spec,
                                 TRAIN_START, TRAIN_END, stats,
                                 stride=STRIDE, require_full_1m=True)
    return collate_all(ds)


def _scale_rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Multiply the OHLCV of the masked rows by 3 (consistency-preserving)."""
    out = df.copy()
    out.loc[mask, ["open", "high", "low", "close"]] *= 3.0
    out.loc[mask, "volume"] = out.loc[mask, "volume"] * 3
    return out


def _pick_anchor(batch: PerceptionBatch, ticker_id: int,
                 min_minute: int = 60, max_minute: int = 330,
                 min_date: pd.Timestamp | None = None,
                 align5: bool = False) -> tuple[int, pd.Timestamp]:
    """Find a batch item suited to a perturbation audit.

    Mid-session anchors keep the whole 30-bar 1m window inside one session
    and guarantee plenty of perturbable data after the anchor. ``align5``
    selects anchors whose close lands on a 5-minute boundary so the newest
    complete 5min bar is unambiguous. Returns (index, anchor wall time).
    """
    assert batch.anchor_ts is not None, \
        "PerceptionWindowDataset must populate anchor_ts (needed for auditing)"
    for i in range(batch.batch_size):
        if int(batch.ticker_id[i]) != ticker_id:
            continue
        sm = anchor_session_minute(batch, i)
        if not (min_minute <= sm <= max_minute):
            continue
        if align5 and (sm + 1) % 5 != 0:
            continue
        t = anchor_wall_time(int(batch.anchor_ts[i]), sm)
        if min_date is not None and t.normalize() < min_date:
            continue
        return i, t
    pytest.fail(f"no anchor found for ticker_id={ticker_id}, "
                f"minute in [{min_minute},{max_minute}], align5={align5}, "
                f"min_date={min_date}")


def _assert_streams_equal(a: PerceptionBatch, i: int,
                          b: PerceptionBatch, j: int,
                          skip: frozenset[str] = frozenset()) -> None:
    """Assert item i of batch a equals item j of batch b, stream by stream."""
    checks = {
        "bars_1m": (a.bars_1m[i], b.bars_1m[j]),
        "bars_5m": (a.bars_5m[i], b.bars_5m[j]),
        "daily": (a.daily[i], b.daily[j]),
        "target_ret": (a.target_ret[i], b.target_ret[j]),
    }
    for name, (x, y) in checks.items():
        if name in skip:
            continue
        assert torch.allclose(x, y, atol=1e-6), \
            f"{name} changed although only post-anchor data was perturbed"
    assert torch.equal(a.minute_of_day[i], b.minute_of_day[j])
    assert torch.equal(a.pad_1m[i], b.pad_1m[j])
    assert torch.equal(a.pad_5m[i], b.pad_5m[j])
    assert torch.equal(a.pad_daily[i], b.pad_daily[j])


# --------------------------------------------------------------------------- #
# Shapes & dtypes (the batch contract)
# --------------------------------------------------------------------------- #

class TestShapesAndDtypes:
    def test_single_item_collates_to_contract_shapes(self, dataset,
                                                     small_spec) -> None:
        assert len(dataset) > 50
        b = collate_batch([dataset[0]])
        assert b.bars_1m.shape == (1, small_spec.len_1m, N_BAR_FEATURES)
        assert b.bars_5m.shape == (1, small_spec.len_5m, N_BAR_FEATURES)
        assert b.daily.shape == (1, small_spec.len_daily, N_DAILY_FEATURES)
        assert b.minute_of_day.shape == (1, small_spec.len_1m)
        assert b.ticker_id.shape == (1,)
        assert b.target_ret.shape == (1,)

    def test_batch_shapes_dtypes_and_flags(self, dataset, small_spec) -> None:
        k = min(8, len(dataset))
        b = collate_batch([dataset[i] for i in range(k)])

        # ---- shapes ----------------------------------------------------- #
        assert b.bars_1m.shape == (k, small_spec.len_1m, N_BAR_FEATURES)
        assert b.bars_5m.shape == (k, small_spec.len_5m, N_BAR_FEATURES)
        assert b.daily.shape == (k, small_spec.len_daily, N_DAILY_FEATURES)
        assert b.minute_of_day.shape == (k, small_spec.len_1m)
        assert b.pad_1m.shape == (k, small_spec.len_1m)
        assert b.pad_5m.shape == (k, small_spec.len_5m)
        assert b.pad_daily.shape == (k, small_spec.len_daily)
        assert b.batch_size == k

        # ---- dtypes ------------------------------------------------------ #
        assert b.bars_1m.dtype == torch.float32
        assert b.bars_5m.dtype == torch.float32
        assert b.daily.dtype == torch.float32
        assert b.target_ret.dtype == torch.float32
        assert b.ticker_id.dtype == torch.int64
        assert b.minute_of_day.dtype == torch.int64
        assert b.pad_1m.dtype == torch.bool
        assert b.pad_5m.dtype == torch.bool
        assert b.pad_daily.dtype == torch.bool
        assert b.anchor_ts is not None and b.anchor_ts.dtype == torch.int64

        # ---- value ranges / flags ---------------------------------------- #
        assert torch.isfinite(b.bars_1m).all()
        assert torch.isfinite(b.bars_5m).all()
        assert torch.isfinite(b.daily).all()
        assert torch.isfinite(b.target_ret).all()
        assert int(b.minute_of_day.min()) >= 0
        assert int(b.minute_of_day.max()) <= 389
        # require_full_1m=True -> the 1m window is never padded.
        assert not b.pad_1m.any()
        # ticker ids are the config-defined integers for AAPL / SPY.
        assert set(b.ticker_id.tolist()) <= {AAPL_ID, SPY_ID}

    def test_padding_is_left_aligned_zeros(self, dataset) -> None:
        b = collate_all(dataset)
        for pad in (b.pad_1m, b.pad_5m, b.pad_daily):
            p = pad.to(torch.int64)
            # LEFT-padded: once a position is valid, all later ones are too.
            assert (p[:, 1:] <= p[:, :-1]).all()
        # Padded positions hold zeros (contract: zero LEFT-padding).
        if b.pad_daily.any():
            assert float(b.daily[b.pad_daily].abs().max()) == 0.0
        if b.pad_5m.any():
            assert float(b.bars_5m[b.pad_5m].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# Whole-dataset invariants
# --------------------------------------------------------------------------- #

class TestGlobalInvariants:
    def test_anchors_unique_and_both_tickers_present(self, dataset) -> None:
        b = collate_all(dataset)
        assert b.anchor_ts is not None
        keys = set(zip(b.ticker_id.tolist(), b.anchor_ts.tolist()))
        assert len(keys) == b.batch_size          # no duplicate samples
        present = set(b.ticker_id.tolist())
        assert present == {AAPL_ID, SPY_ID}

    def test_no_anchor_lacks_a_same_session_target(self, dataset) -> None:
        # horizon_1m=1: an anchor at the last session minute (389) has no
        # same-session t+1 bar, so it must have been dropped entirely.
        b = collate_all(dataset)
        last_minutes = b.minute_of_day[:, -1]
        assert not b.pad_1m[:, -1].any()
        assert int(last_minutes.max()) <= 388

    def test_minute_of_day_is_contiguous_within_session(self, dataset,
                                                        small_spec) -> None:
        # For anchors at session minute >= len_1m-1, the 30 most recent bars
        # are all in the anchor's session -> minute_of_day must be exactly
        # [sm-29 .. sm]. This pins both window alignment and freshness.
        b = collate_all(dataset)
        sm = b.minute_of_day[:, -1]
        offsets = torch.arange(small_spec.len_1m - 1, -1, -1, dtype=torch.int64)
        expected = sm.unsqueeze(1) - offsets.unsqueeze(0)
        rows = sm >= small_spec.len_1m - 1
        assert bool(rows.any())
        assert torch.equal(b.minute_of_day[rows], expected[rows])

    def test_anchor_bar_clock_features_match_minute_of_day(self, dataset) -> None:
        # The newest 1m feature row is the anchor bar itself; its tod_sin/cos
        # channels must encode the same session minute the batch reports for
        # that position — this cross-checks feature/window alignment.
        i_sin = FEATURE_COLUMNS_BAR.index("tod_sin")
        i_cos = FEATURE_COLUMNS_BAR.index("tod_cos")
        b = collate_all(dataset)
        sm = b.minute_of_day[:, -1].to(torch.float64)
        angle = 2.0 * math.pi * sm / 390.0
        assert torch.allclose(b.bars_1m[:, -1, i_sin].to(torch.float64),
                              torch.sin(angle), atol=1e-4)
        assert torch.allclose(b.bars_1m[:, -1, i_cos].to(torch.float64),
                              torch.cos(angle), atol=1e-4)


# --------------------------------------------------------------------------- #
# Target correctness
# --------------------------------------------------------------------------- #

class TestTarget:
    def test_target_ret_equals_hand_computed_log_return(self, dataset,
                                                        synthetic_frames) -> None:
        b = collate_all(dataset)
        assert b.anchor_ts is not None
        alias_by_id = {AAPL_ID: "AAPL", SPY_ID: "SPY"}
        closes = {a: df.set_index("date")["close"]
                  for a, df in synthetic_frames.items()}

        checked = 0
        step = max(1, b.batch_size // 11)          # spread samples around
        for i in range(0, b.batch_size, step):
            alias = alias_by_id[int(b.ticker_id[i])]
            sm = anchor_session_minute(b, i)
            t = anchor_wall_time(int(b.anchor_ts[i]), sm)
            c_t = float(closes[alias].loc[t])
            c_next = float(closes[alias].loc[t + pd.Timedelta(minutes=1)])
            expect = math.log(c_next / c_t)
            assert float(b.target_ret[i]) == pytest.approx(expect, abs=1e-5)
            checked += 1
        assert checked >= 5


# --------------------------------------------------------------------------- #
# NO-LOOK-AHEAD: 1-minute stream (and everything derived from it)
# --------------------------------------------------------------------------- #

class TestNoLookahead1m:
    def test_future_1m_bars_cannot_reach_the_sample(self, dataset, stats,
                                                    small_spec,
                                                    synthetic_frames,
                                                    tmp_path) -> None:
        base = collate_all(dataset)
        i, t = _pick_anchor(base, AAPL_ID, min_minute=60, max_minute=330)

        # Multiply every AAPL 1min bar strictly AFTER the anchor by 3 and
        # re-derive 5min + daily from the perturbed stream, exactly as a
        # poisoned future would propagate through ingestion.
        frames = {a: df.copy() for a, df in synthetic_frames.items()}
        mask = frames["AAPL"]["date"] > t
        assert mask.any(), "picked anchor has no future bars to perturb"
        frames["AAPL"] = _scale_rows(frames["AAPL"], mask)

        rebuilt = _rebuild(tmp_path, "lake_future_x3", stats, small_spec, frames)
        j = match_item(rebuilt, AAPL_ID, int(base.anchor_ts[i]))

        # Every input stream is unchanged: the sample saw nothing after t.
        _assert_streams_equal(base, i, rebuilt, j, skip=frozenset({"target_ret"}))

        # Non-vacuity: the target reads bar t+1, which WAS scaled by 3, so
        # target_ret must shift by exactly ln(3). If the perturbation had
        # silently missed the data path, this would fail.
        assert float(rebuilt.target_ret[j]) == pytest.approx(
            float(base.target_ret[i]) + LN3, abs=1e-4)


# --------------------------------------------------------------------------- #
# NO-LOOK-AHEAD: 5-minute stream
# --------------------------------------------------------------------------- #

class TestNoLookahead5m:
    """A 5min bar stamped T is usable only once T + 5min <= anchor + 1min.

    Anchors are chosen with (session_minute + 1) % 5 == 0 so the newest
    complete bar sits exactly at T* = t - 4min and completes exactly at the
    anchor's close — probing the boundary with equality on both sides.
    """

    def test_incomplete_5m_bars_are_excluded(self, dataset, stats, small_spec,
                                             synthetic_frames, tmp_path) -> None:
        base = collate_all(dataset)
        i, t = _pick_anchor(base, AAPL_ID, min_minute=60, max_minute=330,
                            align5=True)

        f5 = {a: aggregate_5min(df) for a, df in synthetic_frames.items()}
        # Scale every 5min bar that is NOT complete at the anchor close:
        # T > t - 4min  <=>  T + 5min > t + 1min.
        mask = f5["AAPL"]["date"] > (t - pd.Timedelta(minutes=4))
        assert mask.any()
        f5["AAPL"] = _scale_rows(f5["AAPL"], mask)

        rebuilt = _rebuild(tmp_path, "lake_5m_excl", stats, small_spec,
                           {a: df.copy() for a, df in synthetic_frames.items()},
                           frames_5m=f5)
        j = match_item(rebuilt, AAPL_ID, int(base.anchor_ts[i]))
        # 1m/daily/target come from untouched frames; bars_5m is the real
        # assertion: not one incomplete bar leaked into the window.
        _assert_streams_equal(base, i, rebuilt, j)

    def test_newest_complete_5m_bar_is_included(self, dataset, stats,
                                                small_spec, synthetic_frames,
                                                tmp_path) -> None:
        base = collate_all(dataset)
        i, t = _pick_anchor(base, AAPL_ID, min_minute=60, max_minute=330,
                            align5=True)

        # T* completes exactly at the anchor close — the freshest legal bar.
        t_star = t - pd.Timedelta(minutes=4)
        f5 = {a: aggregate_5min(df) for a, df in synthetic_frames.items()}
        mask = f5["AAPL"]["date"] == t_star
        assert int(mask.sum()) == 1
        f5["AAPL"] = _scale_rows(f5["AAPL"], mask)

        rebuilt = _rebuild(tmp_path, "lake_5m_fresh", stats, small_spec,
                           {a: df.copy() for a, df in synthetic_frames.items()},
                           frames_5m=f5)
        j = match_item(rebuilt, AAPL_ID, int(base.anchor_ts[i]))

        # The 5m window must SEE the perturbed bar (window is maximally
        # fresh, not lagging an extra bar out of misplaced caution)...
        assert not torch.allclose(base.bars_5m[i], rebuilt.bars_5m[j],
                                  atol=1e-6), \
            "newest complete 5min bar (T = anchor+1min-5min) was not included"
        # ...while the other streams remain identical.
        _assert_streams_equal(base, i, rebuilt, j, skip=frozenset({"bars_5m"}))


# --------------------------------------------------------------------------- #
# NO-LOOK-AHEAD: daily stream
# --------------------------------------------------------------------------- #

class TestNoLookaheadDaily:
    """Daily context is strictly previous sessions: the anchor's own session
    (whose daily bar is not final until the close) must never appear."""

    MIN_DATE = pd.Timestamp("2026-01-08")   # >= 3 prior sessions available

    def test_same_day_and_future_daily_bars_excluded(self, dataset, stats,
                                                     small_spec,
                                                     synthetic_frames,
                                                     tmp_path) -> None:
        base = collate_all(dataset)
        i, t = _pick_anchor(base, AAPL_ID, min_minute=60, max_minute=330,
                            min_date=self.MIN_DATE)

        fd = {a: make_daily(df) for a, df in synthetic_frames.items()}
        mask = fd["AAPL"]["date"] >= t.normalize()      # today and later
        assert mask.any()
        fd["AAPL"] = _scale_rows(fd["AAPL"], mask)

        rebuilt = _rebuild(tmp_path, "lake_daily_excl", stats, small_spec,
                           {a: df.copy() for a, df in synthetic_frames.items()},
                           frames_daily=fd)
        j = match_item(rebuilt, AAPL_ID, int(base.anchor_ts[i]))
        _assert_streams_equal(base, i, rebuilt, j)

    def test_previous_session_daily_bar_is_included(self, dataset, stats,
                                                    small_spec,
                                                    synthetic_frames,
                                                    tmp_path) -> None:
        base = collate_all(dataset)
        i, t = _pick_anchor(base, AAPL_ID, min_minute=60, max_minute=330,
                            min_date=self.MIN_DATE)
        # With >= 3 prior sessions the len_daily=3 window is fully valid.
        assert not base.pad_daily[i].any()

        fd = {a: make_daily(df) for a, df in synthetic_frames.items()}
        prior = fd["AAPL"][fd["AAPL"]["date"] < t.normalize()]["date"]
        prev_session = prior.max()                       # yesterday's bar
        mask = fd["AAPL"]["date"] == prev_session
        assert int(mask.sum()) == 1
        fd["AAPL"] = _scale_rows(fd["AAPL"], mask)

        rebuilt = _rebuild(tmp_path, "lake_daily_fresh", stats, small_spec,
                           {a: df.copy() for a, df in synthetic_frames.items()},
                           frames_daily=fd)
        j = match_item(rebuilt, AAPL_ID, int(base.anchor_ts[i]))

        # Yesterday IS part of the visible context -> the window must react.
        assert not torch.allclose(base.daily[i], rebuilt.daily[j], atol=1e-6), \
            "previous session's daily bar was not part of the daily window"
        _assert_streams_equal(base, i, rebuilt, j, skip=frozenset({"daily"}))
