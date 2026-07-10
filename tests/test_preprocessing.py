"""Pins the preprocessing contract: feature columns are *exactly* the
reparameterizations documented in ``FEATURE_COLUMNS_BAR`` /
``FEATURE_COLUMNS_DAILY`` — no NaNs, no hidden indicators, no surprises.

Key invariants covered here:
* output columns match the contract tuple exactly and in order;
* the math of each scale-free column matches its documented formula;
* first bar of a session has ret_log == 0 (no cross-session return leaks
  into the intraday return channel);
* clock encodings agree with the session minute / weekday;
* volume z-scores are centered over the very data the stats were fit on;
* degenerate bars (zero range, zero volume) never produce NaN/inf;
* TickerStats JSON round-trips losslessly;
* daily features use the *previous* day's close (shift correctness);
* filter_regular_session keeps exactly the 09:30..15:59 trading-day rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aether.perception.interfaces import (
    FEATURE_COLUMNS_BAR,
    FEATURE_COLUMNS_DAILY,
)
from aether.perception.preprocessing import (
    TickerStats,
    compute_bar_features,
    compute_daily_features,
    filter_regular_session,
)
from aether.utils.market_time import MINUTES_PER_SESSION, session_minute

from tests.conftest import (
    SYNTH_SESSIONS,
    aggregate_5min,
    make_daily,
    make_synthetic_1min,
)


# --------------------------------------------------------------------------- #
# Module-scoped inputs (never mutated; tests that perturb data take copies)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def bars_1m() -> pd.DataFrame:
    return make_synthetic_1min("AAPL", sessions=SYNTH_SESSIONS, seed=11)


@pytest.fixture(scope="module")
def bars_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    return aggregate_5min(bars_1m)


@pytest.fixture(scope="module")
def daily(bars_1m: pd.DataFrame) -> pd.DataFrame:
    return make_daily(bars_1m)


@pytest.fixture(scope="module")
def tstats(bars_1m: pd.DataFrame, daily: pd.DataFrame):
    return TickerStats.fit(bars_1m, daily, "AAPL")


@pytest.fixture(scope="module")
def feats_1m(bars_1m: pd.DataFrame, tstats) -> pd.DataFrame:
    return compute_bar_features(bars_1m, tstats, 1)


@pytest.fixture(scope="module")
def feats_5m(bars_5m: pd.DataFrame, tstats) -> pd.DataFrame:
    return compute_bar_features(bars_5m, tstats, 5)


def _feature_matrix(feats: pd.DataFrame) -> np.ndarray:
    return feats[list(FEATURE_COLUMNS_BAR)].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# Column contract
# --------------------------------------------------------------------------- #

class TestColumns:
    @pytest.mark.parametrize("which", ["1m", "5m"])
    def test_exact_columns_in_order(self, which, feats_1m, feats_5m,
                                    bars_1m, bars_5m) -> None:
        feats = feats_1m if which == "1m" else feats_5m
        bars = bars_1m if which == "1m" else bars_5m
        # The feature columns appear exactly, in the contract's order.
        present = [c for c in feats.columns if c in FEATURE_COLUMNS_BAR]
        assert present == list(FEATURE_COLUMNS_BAR)
        # Plus the two bookkeeping columns — nothing else.
        assert set(feats.columns) == set(FEATURE_COLUMNS_BAR) | {"date",
                                                                 "session_minute"}
        # One output row per input bar; alignment preserved.
        assert len(feats) == len(bars)
        assert (pd.to_datetime(feats["date"]).to_numpy()
                == bars["date"].to_numpy()).all()

    @pytest.mark.parametrize("which", ["1m", "5m"])
    def test_all_values_finite(self, which, feats_1m, feats_5m) -> None:
        feats = feats_1m if which == "1m" else feats_5m
        assert np.isfinite(_feature_matrix(feats)).all()


# --------------------------------------------------------------------------- #
# Feature math (the "exact reparameterization" pin)
# --------------------------------------------------------------------------- #

def _check_bar_math(bars: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Verify each documented formula against raw OHLC, row by row.

    ``sm > 0`` restricts return/gap checks to rows whose predecessor is the
    previous bar of the SAME session (positions are contiguous per session),
    which is the only place the contract defines them.
    """
    o = bars["open"].to_numpy(dtype=np.float64)
    h = bars["high"].to_numpy(dtype=np.float64)
    lo = bars["low"].to_numpy(dtype=np.float64)
    c = bars["close"].to_numpy(dtype=np.float64)
    sm = feats["session_minute"].to_numpy()

    prev_c = np.roll(c, 1)                    # row 0 wraps; masked out below
    within = sm > 0                           # not the first bar of a session

    ret = feats["ret_log"].to_numpy(dtype=np.float64)
    gap = feats["gap_log"].to_numpy(dtype=np.float64)
    rng = feats["range_log"].to_numpy(dtype=np.float64)
    assert np.allclose(ret[within], np.log(c / prev_c)[within], atol=1e-8)
    assert np.allclose(gap[within], np.log(o / prev_c)[within], atol=1e-8)
    assert np.allclose(rng, np.log(h / lo), atol=1e-8)

    # Fraction features involve an implementation-chosen eps in the
    # denominator; a generous tolerance still pins sign and magnitude.
    span = h - lo
    ok = span > 1e-9
    body = feats["body_frac"].to_numpy(dtype=np.float64)
    upper = feats["upper_wick"].to_numpy(dtype=np.float64)
    lower = feats["lower_wick"].to_numpy(dtype=np.float64)
    assert np.allclose(body[ok], ((c - o) / span)[ok], atol=0.05)
    assert np.allclose(upper[ok], ((h - np.maximum(o, c)) / span)[ok], atol=0.05)
    assert np.allclose(lower[ok], ((np.minimum(o, c) - lo) / span)[ok], atol=0.05)
    # Wick/body fractions live in a bounded band by construction.
    assert (body >= -1.001).all() and (body <= 1.001).all()
    assert (upper >= -0.001).all() and (lower >= -0.001).all()


class TestFeatureMath:
    def test_1m_formulas(self, bars_1m, feats_1m) -> None:
        _check_bar_math(bars_1m, feats_1m)

    def test_5m_formulas(self, bars_5m, feats_5m) -> None:
        _check_bar_math(bars_5m, feats_5m)

    @pytest.mark.parametrize("which", ["1m", "5m"])
    def test_first_bar_of_session_ret_is_zero(self, which, feats_1m,
                                              feats_5m) -> None:
        feats = feats_1m if which == "1m" else feats_5m
        first = feats[feats["session_minute"] == 0]
        # Exactly one session-opening bar per synthetic session.
        assert len(first) == SYNTH_SESSIONS
        assert np.allclose(first["ret_log"].to_numpy(), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Clock encodings
# --------------------------------------------------------------------------- #

class TestClockEncodings:
    def test_session_minute_column_matches_timestamps(self, feats_1m) -> None:
        recomputed = np.array([
            session_minute(ts.to_pydatetime())
            for ts in pd.to_datetime(feats_1m["date"])
        ])
        assert (feats_1m["session_minute"].to_numpy() == recomputed).all()
        assert feats_1m["session_minute"].min() == 0
        assert feats_1m["session_minute"].max() == MINUTES_PER_SESSION - 1

    def test_5m_session_minutes_are_window_starts(self, feats_5m) -> None:
        sm = feats_5m["session_minute"].to_numpy()
        assert (sm % 5 == 0).all()
        recomputed = np.array([
            session_minute(ts.to_pydatetime())
            for ts in pd.to_datetime(feats_5m["date"])
        ])
        assert (sm == recomputed).all()

    @pytest.mark.parametrize("which", ["1m", "5m"])
    def test_tod_encoding(self, which, feats_1m, feats_5m) -> None:
        feats = feats_1m if which == "1m" else feats_5m
        sm = feats["session_minute"].to_numpy(dtype=np.float64)
        angle = 2.0 * np.pi * sm / MINUTES_PER_SESSION
        assert np.allclose(feats["tod_sin"].to_numpy(), np.sin(angle), atol=1e-6)
        assert np.allclose(feats["tod_cos"].to_numpy(), np.cos(angle), atol=1e-6)

    def test_dow_encoding(self, bars_1m, feats_1m) -> None:
        wd = bars_1m["date"].dt.weekday.to_numpy(dtype=np.float64)
        angle = 2.0 * np.pi * wd / 5.0
        assert np.allclose(feats_1m["dow_sin"].to_numpy(), np.sin(angle), atol=1e-6)
        assert np.allclose(feats_1m["dow_cos"].to_numpy(), np.cos(angle), atol=1e-6)


# --------------------------------------------------------------------------- #
# Volume normalization
# --------------------------------------------------------------------------- #

class TestVolumeNormalization:
    def test_vol_z_centered_over_fitted_data(self, feats_1m) -> None:
        # Z-scores computed with stats fit on this very data must be
        # roughly centered; a systematic offset means the time-of-day
        # normalization is broken.
        assert abs(float(feats_1m["vol_z"].mean())) < 0.5
        assert abs(float(feats_1m["dollar_vol_z"].mean())) < 0.5

    def test_vol_z_has_unit_scale(self, feats_1m) -> None:
        # A z-score's dispersion should be O(1) — not collapsed, not huge.
        assert 0.2 < float(feats_1m["vol_z"].std()) < 5.0
        assert 0.2 < float(feats_1m["dollar_vol_z"].std()) < 5.0


# --------------------------------------------------------------------------- #
# Degenerate bars
# --------------------------------------------------------------------------- #

class TestDegenerateBars:
    def test_zero_range_zero_volume_bar_is_finite(self, bars_1m, tstats) -> None:
        mod = bars_1m.copy()
        idx = 200                                # mid-session row
        flat = float(mod["close"].iloc[idx - 1])
        # A completely dead minute: no trades, no movement.
        mod.loc[mod.index[idx], ["open", "high", "low", "close"]] = flat
        mod.loc[mod.index[idx], "volume"] = 0

        feats = compute_bar_features(mod, tstats, 1)
        assert np.isfinite(_feature_matrix(feats)).all()

        row = feats.iloc[idx]
        # o == h == l == c == previous close pins every price feature.
        assert float(row["ret_log"]) == pytest.approx(0.0, abs=1e-9)
        assert float(row["gap_log"]) == pytest.approx(0.0, abs=1e-9)
        assert float(row["range_log"]) == pytest.approx(0.0, abs=1e-9)
        assert float(row["body_frac"]) == pytest.approx(0.0, abs=1e-6)
        assert float(row["upper_wick"]) == pytest.approx(0.0, abs=1e-6)
        assert float(row["lower_wick"]) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Stats serialization
# --------------------------------------------------------------------------- #

class TestStatsRoundTrip:
    def test_json_round_trip_is_lossless(self, tstats, bars_1m) -> None:
        payload = tstats.to_json()
        restored = TickerStats.from_json(payload)
        # Serialized form is stable across a round trip...
        assert restored.to_json() == payload
        # ...and produces bit-identical features.
        a = _feature_matrix(compute_bar_features(bars_1m, tstats, 1))
        b = _feature_matrix(compute_bar_features(bars_1m, restored, 1))
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Daily features
# --------------------------------------------------------------------------- #

class TestDailyFeatures:
    def test_columns_and_finiteness(self, daily, tstats) -> None:
        feats = compute_daily_features(daily, tstats)
        present = [c for c in feats.columns if c in FEATURE_COLUMNS_DAILY]
        assert present == list(FEATURE_COLUMNS_DAILY)
        assert "date" in feats.columns
        assert len(feats) == len(daily)
        mat = feats[list(FEATURE_COLUMNS_DAILY)].to_numpy(dtype=np.float64)
        assert np.isfinite(mat).all()

    def test_shift_correctness_ret_uses_previous_close(self, daily,
                                                       tstats) -> None:
        # Day t's return must be computed against day t-1's close — using
        # the same day's close (identically zero) or day t+1 (look-ahead)
        # both fail loudly here.
        feats = compute_daily_features(daily, tstats)
        c = daily["close"].to_numpy(dtype=np.float64)
        o = daily["open"].to_numpy(dtype=np.float64)
        h = daily["high"].to_numpy(dtype=np.float64)
        lo = daily["low"].to_numpy(dtype=np.float64)

        ret = feats["ret_log"].to_numpy(dtype=np.float64)
        gap = feats["gap_log"].to_numpy(dtype=np.float64)
        expect_ret = np.log(c[1:] / c[:-1])
        expect_gap = np.log(o[1:] / c[:-1])
        assert np.allclose(ret[1:], expect_ret, atol=1e-8)
        assert np.allclose(gap[1:], expect_gap, atol=1e-8)
        assert np.allclose(feats["range_log"].to_numpy(), np.log(h / lo),
                           atol=1e-8)


# --------------------------------------------------------------------------- #
# Session filtering
# --------------------------------------------------------------------------- #

class TestFilterRegularSession:
    def test_keeps_only_regular_session_rows(self) -> None:
        stamps = [
            "2026-01-05 04:00",   # pre-market
            "2026-01-05 09:29",   # one minute before the open
            "2026-01-05 09:30",   # open           (keep)
            "2026-01-05 12:00",   # midday         (keep)
            "2026-01-05 15:59",   # last 1min bar  (keep)
            "2026-01-05 16:00",   # the close itself is NOT a 1min bar
            "2026-01-05 20:00",   # after-hours
            "2026-01-10 12:00",   # Saturday
            "2026-01-19 12:00",   # MLK Day (holiday)
        ]
        df = pd.DataFrame({
            "date": pd.to_datetime(stamps),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000,
        })
        out = filter_regular_session(df)
        kept = sorted(pd.to_datetime(out["date"]))
        assert kept == list(pd.to_datetime([
            "2026-01-05 09:30", "2026-01-05 12:00", "2026-01-05 15:59",
        ]))
        # Filtering must not mangle the payload columns.
        assert set(df.columns) <= set(out.columns)
