"""Raw OHLCV -> the perception contract's canonical representation.

This module is the *input half* of the perception contract
(:mod:`aether.perception.interfaces`).  It turns raw, session-filtered OHLCV
bars from the Parquet lake into the exact ``FEATURE_COLUMNS_BAR`` /
``FEATURE_COLUMNS_DAILY`` representation the encoders consume.

Philosophy — representation hygiene, not indicators
---------------------------------------------------
Nothing in this file is a technical-analysis indicator.  Every transform is a
*reparameterization* of the raw stream into a scale-free, roughly stationary
coordinate system:

* prices  -> log returns / log gaps / log ranges (invariant to price level),
* bar shape -> body/wick fractions (invariant to volatility level),
* volume  -> deviation from the instrument's own time-of-day norm
  (invariant to the instrument's liquidity level *and* to the intraday
  volume smile),
* time    -> sin/cos phase encodings (invariant to wrap-around).

The information content of the raw bars is preserved; only the *units* are
made comparable across tickers, across times of day, and across price
regimes.  All meaning is learned downstream by the network.

The intraday volume smile
-------------------------
US equity volume follows a pronounced U-shape over the session (heavy at
open and close, quiet at lunch), and the exact shape is *ticker-specific*
(SPY's smile differs from NVDA's).  Raw volume therefore mixes two signals:
"what time is it" and "is participation unusual".  ``TickerStats`` fits the
per-minute mean/std of log-volume on TRAINING data only; z-scoring against
that profile removes the predictable smile and leaves exactly the residual —
*abnormal participation* — which is raw signal, not an indicator.

Look-ahead discipline
---------------------
* ``TickerStats`` is fitted on the training date range only; validation and
  live data are normalized with the *frozen* training statistics.
* Every per-bar feature at row ``t`` references only rows ``< t`` of the
  same frame (previous close for returns/gaps).  ``shift``-style logic is
  implemented with explicit ``x[t-1]`` indexing so the direction is
  auditable at a glance.

Robust clipping
---------------
Raw feeds contain occasional data errors (a `0.01` print, a fat-fingered
volume).  For the unbounded log features (``ret_log``, ``gap_log``,
``range_log``) we store the fitted median and IQR and clip transforms to
``median ± 8·IQR``.  Because the IQR is estimated from the *bulk* of the
distribution, 8 IQRs is far outside any normal move (for a Gaussian,
8·IQR ≈ 10.8σ) — real tail events (flash crashes, earnings gaps) survive
almost untouched, while decimal-shift data errors cannot blow up encoder
activations.  Bounds are kept per timescale ("1m", "5m", "daily") because
return dispersion grows with bar duration; a single 1-minute-derived bound
would truncate genuine daily moves.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.market_time import (
    BARS_5MIN_PER_SESSION,
    ET,
    MINUTES_PER_SESSION,
    SESSION_OPEN,
    is_trading_day,
    session_close,
)
from .interfaces import FEATURE_COLUMNS_BAR, FEATURE_COLUMNS_DAILY

logger = get_logger("aether.perception.preprocessing")

# --------------------------------------------------------------------------- #
# Tunable constants (documented invariants, not knobs to sweep)
# --------------------------------------------------------------------------- #

#: Guard for zero-range (high == low) bars in the wick/body fractions.
EPS: float = 1e-9

#: Half-width, in *minutes*, of the smoothing window applied across adjacent
#: minute-of-day bins when fitting the intraday volume profile.  A single
#: minute bin sees only ~1 observation per session; pooling ±7 minutes gives
#: ~15x the sample size while the smile changes slowly enough that the bias
#: is negligible.
SMOOTH_RADIUS_MINUTES: int = 7

#: Clip transforms of ret/gap/range to ``median ± CLIP_IQR_MULT · IQR``.
CLIP_IQR_MULT: float = 8.0

#: Floor on a fitted IQR so a degenerate fit (constant column) can never
#: produce a zero-width clip band that erases real data.
MIN_IQR: float = 1e-4

#: Floor on any fitted standard deviation used as a divisor.
MIN_STD: float = 1e-6

#: Fallback (median, iqr) used when a stream had no data at fit time.  The
#: band ``0 ± 8·0.05 = ±0.4`` in log space (~±49%) is wide enough to pass
#: any legitimate bar of any timescale while still catching decimal-shift
#: data errors.
DEFAULT_BOUNDS: tuple[float, float] = (0.0, 0.05)

_OPEN_MINUTE: int = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute  # 09:30 -> 570


# --------------------------------------------------------------------------- #
# Small vectorized helpers (shared by fit + transform so both use one math)
# --------------------------------------------------------------------------- #

def _session_minutes(ts: pd.Series) -> np.ndarray:
    """Minutes since 09:30 ET for already-session-filtered timestamps.

    Vectorized twin of :func:`aether.utils.market_time.session_minute`;
    valid range is asserted by construction because callers filter to the
    regular session first.
    """
    return (ts.dt.hour * 60 + ts.dt.minute - _OPEN_MINUTE).to_numpy(np.int64)


def _day_ids(ts: pd.Series) -> np.ndarray:
    """Calendar day of each timestamp as int64 days-since-epoch.

    Used as the *session identity*: two bars share a session iff they share
    a day id (US equities have no overnight sessions).
    """
    return ts.to_numpy().astype("datetime64[D]").astype(np.int64)


def _safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``log(num/den)`` where both sides are positive/finite, else 0.

    Zero is the correct neutral element for all three log features and keeps
    the contract's "all features non-NaN" promise even on corrupt rows.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out = np.zeros(num.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(num / den, out=out, where=valid)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ret_and_gap(open_: np.ndarray, close: np.ndarray,
                 day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar log return and log gap versus the *previous row's* close.

    Rows are sorted by time, so the previous row of a session's first bar is
    the *last bar of the previous session*.  That gives exactly the contract
    semantics for free:

    * ``ret_log``: ``log(close_t / close_{t-1})`` within a session; forced
      to 0 on each session's first bar (the overnight move is not an
      intra-session return — it lives in ``gap_log``).
    * ``gap_log``: ``log(open_t / close_{t-1})`` everywhere, including each
      session's first bar where ``close_{t-1}`` is the previous session's
      close — the overnight gap is real information and is kept.  The very
      first row of the frame (no history at all) gets 0.

    Only ``t-1`` is ever referenced: no look-ahead by construction.
    """
    n = close.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan            # no predecessor -> features become 0
    prev_close[1:] = close[:-1]

    first_of_session = np.empty(n, dtype=bool)
    first_of_session[0] = True
    first_of_session[1:] = day[1:] != day[:-1]

    ret_log = _safe_log_ratio(close, prev_close)
    ret_log[first_of_session] = 0.0   # overnight move excluded from ret_log
    gap_log = _safe_log_ratio(open_, prev_close)
    return ret_log, gap_log


def _bar_geometry(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Scale-free candle geometry: (range_log, body_frac, upper, lower wick).

    All three fractions share the denominator ``high - low + EPS`` so a
    zero-range bar (high == low) yields exactly (0, 0, 0) instead of NaN.
    """
    rng = high - low
    denom = rng + EPS
    range_log = _safe_log_ratio(high, low)                      # 0 if h == l
    body_frac = (close - open_) / denom
    upper_wick = (high - np.maximum(open_, close)) / denom
    lower_wick = (np.minimum(open_, close) - low) / denom
    return range_log, body_frac, upper_wick, lower_wick


def _pooled_profile(values: np.ndarray, bins: np.ndarray, n_bins: int,
                    radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean/std with a ±``radius``-bin pooled smoothing window.

    Instead of computing a per-bin estimate and then blurring it, we pool the
    *raw moments* (count, sum, sum of squares) across the window via prefix
    sums, which is the statistically honest way to smooth: each bin's
    estimate is the exact mean/std of every observation that fell within
    ``radius`` bins of it.  Edges clamp (no wrap-around — minute 0 must not
    borrow from minute 389, the auction dynamics differ).

    Bins that remain empty even after pooling fall back to the global
    mean/std; stds are floored so they are always safe divisors.
    """
    mean = np.zeros(n_bins, dtype=np.float64)
    std = np.ones(n_bins, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    values, bins = values[ok], np.asarray(bins)[ok]
    if values.size == 0:
        return mean, std

    cnt = np.bincount(bins, minlength=n_bins).astype(np.float64)
    sums = np.bincount(bins, weights=values, minlength=n_bins)
    sumsq = np.bincount(bins, weights=values ** 2, minlength=n_bins)

    # Prefix sums -> O(1) pooled moments per bin over [b-radius, b+radius].
    c_cnt = np.concatenate(([0.0], np.cumsum(cnt)))
    c_sum = np.concatenate(([0.0], np.cumsum(sums)))
    c_sq = np.concatenate(([0.0], np.cumsum(sumsq)))
    b = np.arange(n_bins)
    lo = np.maximum(b - radius, 0)
    hi = np.minimum(b + radius, n_bins - 1)
    p_cnt = c_cnt[hi + 1] - c_cnt[lo]
    p_sum = c_sum[hi + 1] - c_sum[lo]
    p_sq = c_sq[hi + 1] - c_sq[lo]

    safe_cnt = np.maximum(p_cnt, 1.0)
    mean = p_sum / safe_cnt
    var = np.maximum(p_sq / safe_cnt - mean ** 2, 0.0)
    std = np.sqrt(var)

    # Global fallbacks for empty bins + divisor floor.
    g_mean = float(values.mean())
    g_std = max(float(values.std()), MIN_STD)
    empty = p_cnt == 0
    mean[empty] = g_mean
    std[empty] = g_std
    std = np.maximum(np.where(std < MIN_STD, g_std, std), MIN_STD)
    return mean, std


def _robust_bounds(x: np.ndarray) -> tuple[float, float]:
    """(median, IQR) of the finite entries, with degenerate-fit fallbacks."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:                    # too little data for quartiles to mean much
        return DEFAULT_BOUNDS
    med = float(np.median(x))
    q1, q3 = np.percentile(x, [25.0, 75.0])
    return med, max(float(q3 - q1), MIN_IQR)


def _aggregate_5min(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, volume: np.ndarray, day: np.ndarray,
                    sm: np.ndarray) -> pd.DataFrame:
    """Aggregate 1min bars into 5min bars keyed by (session, 5min bin).

    Used only at *fit* time so the 5min volume profile and clip bounds are
    measured on exactly the shape of data the 5min feature transform will
    see (open=first, high=max, low=min, close=last, volume=sum).  Bar label
    = bin start, matching the lake's 5min convention.
    """
    frame = pd.DataFrame({
        "day": day, "bin": sm // 5, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    agg = (frame.groupby(["day", "bin"], sort=True)
                .agg(open=("open", "first"), high=("high", "max"),
                     low=("low", "min"), close=("close", "last"),
                     volume=("volume", "sum"))
                .reset_index())
    return agg


# --------------------------------------------------------------------------- #
# TickerStats — the per-ticker statistical fingerprint ("DNA v0")
# --------------------------------------------------------------------------- #

@dataclass
class TickerStats:
    """Frozen normalization statistics for one instrument.

    This is the *statistical* half of a ticker's DNA (the learned embedding
    inside the model is the other half).  It is fitted **once, on the
    training date range only**, serialized to JSON under
    ``cfg.data.stats_root/<alias>.json``, and then applied unchanged to
    validation and live data — that asymmetry is what makes normalization
    leak-free.

    Contents
    --------
    * ``vol_*_1m`` / ``dvol_*_1m`` — 390-entry minute-of-day mean/std of
      ``log1p(volume)`` and ``log1p(close·volume)``, smoothed with a
      ±7-minute pooled window (see :func:`_pooled_profile`).  Normalizing
      against this ticker-specific intraday smile turns raw volume into
      "participation vs. this instrument's own norm for this minute" —
      abnormal participation is signal, the smile itself is not.
    * ``vol_*_5m`` / ``dvol_*_5m`` — the same idea over 78 five-minute bins
      (bin = ``session_minute // 5``), smoothed over adjacent bins
      (±7 minutes ≈ ±1 bin at this resolution).
    * ``daily_vol_mean/std`` — mean/std of daily ``log1p(volume)``.
    * ``clip_bounds`` — per-stream ("1m" | "5m" | "daily") robust
      ``(median, IQR)`` for each unbounded log feature; transforms clip to
      ``median ± 8·IQR``.  This protects encoders from data errors without
      destroying real tail events (8·IQR ≈ 10.8σ for Gaussian data).
    * ``fitted_start/end`` — the training range the stats were fitted on,
      so loaders can verify coverage before reusing a cached file.
    """

    alias: str
    fitted_start: str
    fitted_end: str

    # Intraday volume smile, 1min resolution (390 bins).
    vol_mean_1m: list[float]
    vol_std_1m: list[float]
    dvol_mean_1m: list[float]
    dvol_std_1m: list[float]

    # Intraday volume smile, 5min resolution (78 bins).
    vol_mean_5m: list[float]
    vol_std_5m: list[float]
    dvol_mean_5m: list[float]
    dvol_std_5m: list[float]

    # Daily volume level.
    daily_vol_mean: float
    daily_vol_std: float

    # stream -> feature -> [median, iqr]
    clip_bounds: dict[str, dict[str, list[float]]]

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(cls, bars_1min: pd.DataFrame, daily: pd.DataFrame, alias: str,
            *, fitted_start: str | None = None,
            fitted_end: str | None = None) -> "TickerStats":
        """Fit the fingerprint from TRAINING-range data only.

        Parameters
        ----------
        bars_1min:
            Regular-session 1min bars (columns ``date/open/high/low/close/
            volume``), sorted by time, restricted to the training range.
            The 5min statistics are derived by aggregating these same bars,
            so fit needs no separate 5min input.
        daily:
            Daily bars for the same range (may be empty; sane fallbacks are
            substituted and logged).
        fitted_start / fitted_end:
            The *requested* training range, recorded for cache-coverage
            checks.  Defaults to the observed data range when omitted.
        """
        if bars_1min is None or bars_1min.empty:
            raise ValueError(f"TickerStats.fit({alias!r}): no 1min bars given")

        bars = bars_1min.sort_values("date").reset_index(drop=True)
        ts = pd.to_datetime(bars["date"])
        o = bars["open"].to_numpy(np.float64)
        h = bars["high"].to_numpy(np.float64)
        l = bars["low"].to_numpy(np.float64)
        c = bars["close"].to_numpy(np.float64)
        v = bars["volume"].to_numpy(np.float64)
        sm = _session_minutes(ts)
        day = _day_ids(ts)

        # ---- 1min volume smile (390 bins, ±7-minute pooled smoothing) ---- #
        log_v = np.log1p(np.maximum(v, 0.0))
        log_dv = np.log1p(np.maximum(c * v, 0.0))
        vm1, vs1 = _pooled_profile(log_v, sm, MINUTES_PER_SESSION,
                                   SMOOTH_RADIUS_MINUTES)
        dvm1, dvs1 = _pooled_profile(log_dv, sm, MINUTES_PER_SESSION,
                                     SMOOTH_RADIUS_MINUTES)

        # ---- 5min smile (78 bins) from self-aggregated 1min bars --------- #
        agg5 = _aggregate_5min(o, h, l, c, v, day, sm)
        c5 = agg5["close"].to_numpy(np.float64)
        v5 = agg5["volume"].to_numpy(np.float64)
        bins5 = agg5["bin"].to_numpy(np.int64)
        day5 = agg5["day"].to_numpy(np.int64)
        # ±7 minutes at 5min resolution reaches only the adjacent bin.
        radius5 = max(1, SMOOTH_RADIUS_MINUTES // 5)
        log_v5 = np.log1p(np.maximum(v5, 0.0))
        log_dv5 = np.log1p(np.maximum(c5 * v5, 0.0))
        vm5, vs5 = _pooled_profile(log_v5, bins5, BARS_5MIN_PER_SESSION, radius5)
        dvm5, dvs5 = _pooled_profile(log_dv5, bins5, BARS_5MIN_PER_SESSION,
                                     radius5)

        # ---- daily volume level ------------------------------------------ #
        if daily is not None and not daily.empty and "volume" in daily.columns:
            d_sorted = daily.sort_values("date").reset_index(drop=True)
            log_dvol = np.log1p(
                np.maximum(d_sorted["volume"].to_numpy(np.float64), 0.0))
            log_dvol = log_dvol[np.isfinite(log_dvol)]
        else:
            d_sorted = pd.DataFrame()
            log_dvol = np.zeros(0)
            logger.warning("TickerStats.fit(%s): no daily bars — daily "
                           "volume stats fall back to (0, 1)", alias)
        daily_vol_mean = float(log_dvol.mean()) if log_dvol.size else 0.0
        daily_vol_std = max(float(log_dvol.std()), MIN_STD) if log_dvol.size \
            else 1.0

        # ---- robust clip bounds, per timescale --------------------------- #
        def stream_bounds(open_a: np.ndarray, high_a: np.ndarray,
                          low_a: np.ndarray, close_a: np.ndarray,
                          day_a: np.ndarray) -> dict[str, list[float]]:
            ret, gap = _ret_and_gap(open_a, close_a, day_a)
            rng = _safe_log_ratio(high_a, low_a)
            return {"ret_log": list(_robust_bounds(ret)),
                    "gap_log": list(_robust_bounds(gap)),
                    "range_log": list(_robust_bounds(rng))}

        clip_bounds = {"1m": stream_bounds(o, h, l, c, day),
                       "5m": stream_bounds(agg5["open"].to_numpy(np.float64),
                                           agg5["high"].to_numpy(np.float64),
                                           agg5["low"].to_numpy(np.float64),
                                           c5, day5)}
        if not d_sorted.empty and {"open", "high", "low", "close"} <= set(
                d_sorted.columns):
            dd = np.arange(len(d_sorted), dtype=np.int64)  # one "session"/row
            clip_bounds["daily"] = stream_bounds(
                d_sorted["open"].to_numpy(np.float64),
                d_sorted["high"].to_numpy(np.float64),
                d_sorted["low"].to_numpy(np.float64),
                d_sorted["close"].to_numpy(np.float64), dd)
        else:
            clip_bounds["daily"] = {f: list(DEFAULT_BOUNDS)
                                    for f in ("ret_log", "gap_log",
                                              "range_log")}

        start = fitted_start or str(ts.min().date())
        end = fitted_end or str(ts.max().date())
        logger.info("TickerStats.fit(%s): %d 1min bars, %d 5min bars, "
                    "%d daily rows over [%s, %s]", alias, len(bars),
                    len(agg5), len(d_sorted), start, end)
        return cls(
            alias=alias, fitted_start=start, fitted_end=end,
            vol_mean_1m=vm1.tolist(), vol_std_1m=vs1.tolist(),
            dvol_mean_1m=dvm1.tolist(), dvol_std_1m=dvs1.tolist(),
            vol_mean_5m=vm5.tolist(), vol_std_5m=vs5.tolist(),
            dvol_mean_5m=dvm5.tolist(), dvol_std_5m=dvs5.tolist(),
            daily_vol_mean=daily_vol_mean, daily_vol_std=daily_vol_std,
            clip_bounds=clip_bounds,
        )

    # ------------------------------------------------------------------ #
    # Application helpers
    # ------------------------------------------------------------------ #

    def volume_profile(self, interval_minutes: int
                       ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
        """(vol_mean, vol_std, dvol_mean, dvol_std) arrays for an interval."""
        if interval_minutes == 1:
            return (np.asarray(self.vol_mean_1m), np.asarray(self.vol_std_1m),
                    np.asarray(self.dvol_mean_1m), np.asarray(self.dvol_std_1m))
        if interval_minutes == 5:
            return (np.asarray(self.vol_mean_5m), np.asarray(self.vol_std_5m),
                    np.asarray(self.dvol_mean_5m), np.asarray(self.dvol_std_5m))
        raise ValueError(f"No volume profile for interval {interval_minutes}")

    def clip(self, stream: str, feature: str, x: np.ndarray) -> np.ndarray:
        """Clip ``x`` to the stream's fitted ``median ± 8·IQR`` band."""
        med, iqr = self.clip_bounds.get(stream, {}).get(
            feature, list(DEFAULT_BOUNDS))
        half = CLIP_IQR_MULT * max(float(iqr), MIN_IQR)
        return np.clip(x, float(med) - half, float(med) + half)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_json(self, path: Path | str | None = None) -> str:
        """Serialize to human-inspectable JSON.

        With ``path`` given, writes the JSON to that file via an atomic
        replace (temp file + rename) so a crash never leaves a truncated
        stats cache behind.  The serialized payload is returned either way,
        which makes purely in-memory round trips (``from_json(to_json())``)
        possible without touching disk.
        """
        payload = json.dumps(asdict(self), indent=2)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(payload)
            tmp.replace(path)
        return payload

    @classmethod
    def from_json(cls, source: Path | str) -> "TickerStats":
        """Deserialize from a JSON file path OR a raw JSON payload string.

        A raw payload (as returned by :meth:`to_json`) always starts with
        ``{`` after stripping whitespace, which can never be a valid file
        path — so the two forms are unambiguous.
        """
        if isinstance(source, str) and source.lstrip().startswith("{"):
            return cls(**json.loads(source))
        return cls(**json.loads(Path(source).read_text()))


# --------------------------------------------------------------------------- #
# Feature transforms
# --------------------------------------------------------------------------- #

def compute_bar_features(bars: pd.DataFrame, stats: TickerStats,
                         interval_minutes: int) -> pd.DataFrame:
    """Intraday bars -> ``FEATURE_COLUMNS_BAR`` (+ ``date``, ``session_minute``).

    Parameters
    ----------
    bars:
        Regular-session bars with columns ``date/open/high/low/close/volume``,
        already session-filtered (:func:`filter_regular_session`) and sorted.
    stats:
        The frozen training-range fingerprint used for volume normalization
        and robust clipping.
    interval_minutes:
        1 or 5.  Selects the volume-profile resolution (390 vs. 78 bins) and
        the clip-bound stream.  Time-of-day phase always uses the bar-start
        ``session_minute / 390`` regardless of interval, per the contract.

    Row ``t`` of the output depends only on rows ``<= t`` of the input —
    the module-level look-ahead invariant.
    """
    if interval_minutes not in (1, 5):
        raise ValueError(f"interval_minutes must be 1 or 5, got "
                         f"{interval_minutes}")
    cols = list(FEATURE_COLUMNS_BAR) + ["date", "session_minute"]
    if bars is None or bars.empty:
        return pd.DataFrame(columns=cols)

    df = bars.sort_values("date").reset_index(drop=True)
    ts = pd.to_datetime(df["date"])
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    sm = _session_minutes(ts)
    day = _day_ids(ts)

    # --- price reparameterization (previous-row references only) ---------- #
    ret_log, gap_log = _ret_and_gap(o, c, day)
    range_log, body_frac, upper_wick, lower_wick = _bar_geometry(o, h, l, c)

    stream = "1m" if interval_minutes == 1 else "5m"
    ret_log = stats.clip(stream, "ret_log", ret_log)
    gap_log = stats.clip(stream, "gap_log", gap_log)
    range_log = stats.clip(stream, "range_log", range_log)

    # --- volume vs. this ticker's own time-of-day norm -------------------- #
    bins = sm if interval_minutes == 1 else sm // 5
    vm, vs, dvm, dvs = stats.volume_profile(interval_minutes)
    vol_z = (np.log1p(np.maximum(v, 0.0)) - vm[bins]) / vs[bins]
    dollar_vol_z = (np.log1p(np.maximum(c * v, 0.0)) - dvm[bins]) / dvs[bins]

    # --- cyclic clocks ----------------------------------------------------- #
    # Session phase: same denominator (390) for every interval so 1min and
    # 5min tokens agree about "where in the day we are".
    tod = 2.0 * math.pi * sm / MINUTES_PER_SESSION
    dow = 2.0 * math.pi * ts.dt.weekday.to_numpy(np.float64) / 5.0

    out = pd.DataFrame({
        "ret_log": ret_log,
        "gap_log": gap_log,
        "range_log": range_log,
        "body_frac": body_frac,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "vol_z": vol_z,
        "dollar_vol_z": dollar_vol_z,
        "tod_sin": np.sin(tod),
        "tod_cos": np.cos(tod),
        "dow_sin": np.sin(dow),
        "dow_cos": np.cos(dow),
    })
    # Contract: float32, non-NaN, exact column order.
    out = out[list(FEATURE_COLUMNS_BAR)].astype(np.float32).fillna(0.0)
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    out["date"] = ts
    out["session_minute"] = sm
    return out


def compute_daily_features(daily: pd.DataFrame,
                           stats: TickerStats) -> pd.DataFrame:
    """Daily bars -> ``FEATURE_COLUMNS_DAILY`` (+ ``date``).

    ``ret_log``/``gap_log`` reference the *previous day's* close (row ``t-1``
    of the sorted frame — strictly the past; the first row gets 0).
    ``vol_z`` z-scores ``log1p(volume)`` against the fitted daily stats.
    """
    cols = list(FEATURE_COLUMNS_DAILY) + ["date"]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=cols)

    df = daily.sort_values("date").reset_index(drop=True)
    ts = pd.to_datetime(df["date"])
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)

    # Every row is its own "session": previous row = previous trading day,
    # so ret_log = close-to-close and gap_log = open vs. yesterday's close.
    n = len(df)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan
    prev_close[1:] = c[:-1]
    ret_log = _safe_log_ratio(c, prev_close)
    gap_log = _safe_log_ratio(o, prev_close)
    range_log, body_frac, upper_wick, lower_wick = _bar_geometry(o, h, l, c)

    ret_log = stats.clip("daily", "ret_log", ret_log)
    gap_log = stats.clip("daily", "gap_log", gap_log)
    range_log = stats.clip("daily", "range_log", range_log)

    vol_z = (np.log1p(np.maximum(v, 0.0)) - stats.daily_vol_mean) \
        / max(stats.daily_vol_std, MIN_STD)

    out = pd.DataFrame({
        "ret_log": ret_log,
        "gap_log": gap_log,
        "range_log": range_log,
        "body_frac": body_frac,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "vol_z": vol_z,
    })
    out = out[list(FEATURE_COLUMNS_DAILY)].astype(np.float32).fillna(0.0)
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    out["date"] = ts
    return out


# --------------------------------------------------------------------------- #
# Session filtering
# --------------------------------------------------------------------------- #

def filter_regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars inside the regular NYSE session (vectorized).

    Semantics match :func:`aether.utils.market_time.in_regular_session`
    exactly — time-of-day in ``[09:30, close)`` where ``close`` is 16:00 or
    13:00 on half days, and the date is a trading day — but the per-date
    calendar work (holiday sets, half-day close) is done once per unique
    date instead of once per row, which matters at millions of 1min bars.

    Timezone handling: FMP quotes intraday timestamps in US/Eastern local
    time; naive timestamps are taken as ET, aware ones are converted.
    Output is sorted by time with a naive-ET ``date`` column.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=getattr(bars, "columns", []))

    df = bars.copy()
    ts = pd.to_datetime(df["date"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(ET).dt.tz_localize(None)
    df["date"] = ts

    minutes = ts.dt.hour * 60 + ts.dt.minute          # minute of the ET day
    dates = ts.dt.date

    # Close minute per unique date; -1 marks non-trading days so nothing on
    # them can pass the (minutes < close) test.
    close_by_date = {
        d: (session_close(d).hour * 60 + session_close(d).minute
            if is_trading_day(d) else -1)
        for d in pd.unique(dates)
    }
    close_minutes = dates.map(close_by_date).to_numpy(np.int64)

    mask = (minutes.to_numpy(np.int64) >= _OPEN_MINUTE) \
        & (minutes.to_numpy(np.int64) < close_minutes)
    return df.loc[mask].sort_values("date").reset_index(drop=True)
