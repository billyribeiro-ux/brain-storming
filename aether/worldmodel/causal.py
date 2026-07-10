"""The living causal graph over interpretable market channels (Layer 2).

What this module is
-------------------
Perception embeddings are powerful but opaque. This module maintains the
*explainable* counterpart: a small linear structural equation model (SEM)
over human-readable channels — per-ticker 1-minute returns, volume/range
z-scores, 5-minute momentum, plus market-wide treasury and news-flow
channels — refit on a rolling window so the graph stays "alive" as regimes
shift. Its output (:class:`CausalGraphSnapshot`) feeds dashboards and the
autopsy module's attribution, NEVER the trading features.

The structural model
--------------------
With ``X_t ∈ R^d`` the standardized channel vector at minute ``t``::

    X_t ≈ Σ_{l=0..n_lags}  X_{t-l} @ W_l          (W_0 has a zero diagonal)

* ``W_0`` captures *contemporaneous* structure. A node must not explain
  itself at lag 0, so its diagonal is hard-masked to zero, and the
  NOTEARS acyclicity penalty ``h(W_0) = trace(exp(W_0 ∘ W_0)) − d``
  (Zheng et al. 2018) pushes ``W_0`` toward a DAG so contemporaneous
  "causation" cannot run in circles.
* ``W_1..W_L`` are Granger-style lagged effects; time ordering already
  makes them acyclic, so they carry no acyclicity penalty.
* An L1 penalty on every ``W_l`` enforces sparsity: few, strong edges
  instead of a dense correlation soup.

Edge weights are expressed on *standardized* variables (each channel
z-scored over the fitted window), so a weight reads as "one sigma of the
source moves the destination by ``w`` sigmas" — comparable across channels
with wildly different natural units.

Stability (edge confidence)
---------------------------
A single fit on noisy minutely data will hallucinate edges. Every fit is
therefore repeated on ``cfg.stability_bootstraps`` random 60% row
subsamples (distinct seeds); an edge's ``confidence`` is the fraction of
subsample fits in which it survives the weight threshold WITH the same
sign as the full fit. Sign-flipping edges are noise and score ~0.

Honesty notes (read before touching)
------------------------------------
* **In-window normalization is deliberate.** ``vol_z``, ``range_z`` and
  ``news_rate`` are z-scored within the frame being fitted. For a *trading
  feature* this would be look-ahead leakage; for *structure learning* it is
  acceptable because the graph only EXPLAINS a window after the fact — its
  edges never feed model inputs or trading decisions. The affine
  transformation also leaves the regression structure unchanged; only the
  interpretation ("sigmas within this window") depends on it.
* **Treasury is previous-day only.** ``treasury_10y_chg`` on session date D
  is the daily change of the 10-year yield as of the latest date STRICTLY
  BEFORE D. Rates data for D itself may be stamped intraday or after the
  close; using it during D's session would be a same-day leak.
* ``cfg.hidden_dim`` is reserved for a future nonlinear SEM and is unused
  here — the current SEM is deliberately linear so that edge weights stay
  directly interpretable.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aether.data.storage import MARKET_PSEUDO_TICKER, ParquetStore
from aether.perception.preprocessing import filter_regular_session
from aether.utils.logging import get_logger
from aether.worldmodel.interfaces import (
    CAUSAL_MARKET_CHANNELS,
    CAUSAL_TICKER_CHANNELS,
    CausalConfig,
    CausalEdge,
    CausalGraphSnapshot,
)

logger = get_logger("aether.worldmodel.causal")

#: Lake dataset names read by the frame builder (see aether.data.endpoints).
DS_1MIN = "bars_1min"
DS_TREASURY = "treasury_rates"
DS_NEWS_STOCK = "news_stock"
DS_NEWS_GENERAL = "news_general"

#: Trailing window of the news-flow channel: items published in the last
#: 30 minutes (exclusive of t-30, inclusive of t) count toward minute t.
NEWS_WINDOW_SECONDS = 30 * 60

#: Edges with |standardized weight| below this are treated as absent — both
#: when building snapshots and when scoring bootstrap stability.
EDGE_THRESHOLD = 0.01

#: Fraction of design rows used by each stability bootstrap.
BOOTSTRAP_FRACTION = 0.6

#: Numerical floors — never divide by (or take log of) something smaller.
_MIN_STD = 1e-8
_TINY = 1e-12

#: OHLCV columns the frame builder requires from the lake.
_OHLCV = ("date", "open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- #
# Frame construction
# --------------------------------------------------------------------------- #

def _zscore_in_window(x: pd.Series) -> pd.Series:
    """Z-score a series against ITS OWN mean/std (in-window normalization).

    Acceptable here only because the causal frame explains a window after
    the fact — see the module docstring's honesty notes. A constant series
    (e.g. an index with zero volume) degrades to all-zeros instead of NaN.
    """
    sd = float(x.std(ddof=0))
    return (x - float(x.mean())) / max(sd, _MIN_STD)


def _ticker_channels(bars: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Compute the four CAUSAL_TICKER_CHANNELS for one ticker.

    ``bars`` must be regular-session 1min OHLCV, sorted by ``date`` (the
    output of :func:`filter_regular_session`). Returns a frame indexed by
    the naive-ET bar timestamp with columns ``<ALIAS>.<channel>``.
    """
    ts = pd.to_datetime(bars["date"])
    day = ts.dt.normalize()                      # session identity for grouping

    close = np.maximum(bars["close"].to_numpy(np.float64), _TINY)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    volume = np.maximum(bars["volume"].to_numpy(np.float64), 0.0)

    # ret_1m: log close-to-close return WITHIN the session. The first bar of
    # each session has no prior close inside the session, so it is defined
    # as 0 — the overnight gap is deliberately excluded (a different causal
    # animal than intraday flow).
    log_close = pd.Series(np.log(close), index=bars.index)
    ret_1m = log_close.groupby(day).diff().fillna(0.0)

    # ret_5m: 5-bar rolling sum of ret_1m, so bar t aggregates t-4..t ONLY
    # (pandas' trailing window — no look-ahead). Grouped by session so the
    # sum never straddles an overnight boundary; the first bars of a session
    # use however many bars exist (min_periods=1), consistent with ret_1m
    # being 0 at the session start.
    ret_5m = (
        ret_1m.groupby(day)
        .rolling(5, min_periods=1)
        .sum()
        .droplevel(0)          # drop the group key; back to the bar index
        .reindex(bars.index)
    )

    # vol_z: z of log1p(volume) — in-window normalization (see module notes).
    vol_z = _zscore_in_window(pd.Series(np.log1p(volume), index=bars.index))

    # range_z: z of log(high/low). Guard malformed bars (high<low, low<=0)
    # by clamping the ratio at 1 so the log is >= 0 rather than NaN.
    ratio = np.maximum(high, low) / np.maximum(low, _TINY)
    range_z = _zscore_in_window(
        pd.Series(np.log(np.maximum(ratio, 1.0)), index=bars.index)
    )

    # Assemble in the exact CAUSAL_TICKER_CHANNELS order.
    values = {"ret_1m": ret_1m, "vol_z": vol_z, "range_z": range_z,
              "ret_5m": ret_5m}
    out = pd.DataFrame(
        {f"{alias}.{ch}": values[ch] for ch in CAUSAL_TICKER_CHANNELS}
    )
    out.index = pd.DatetimeIndex(ts.to_numpy())
    # The lake dedups, but be defensive: one row per (date, minute).
    return out[~out.index.duplicated(keep="last")]


def _treasury_channel(store: ParquetStore, index: pd.DatetimeIndex) -> pd.Series:
    """``treasury_10y_chg``: previous-day daily change of the 10y yield.

    NO SAME-DAY LEAK: every minute of session date D carries the change
    computed at the latest treasury date STRICTLY BEFORE D (enforced with
    ``merge_asof(..., allow_exact_matches=False)``). Missing data yields a
    zero column (constant ⇒ zero weight after standardization) with a
    warning, rather than silently killing the whole frame in the dropna.
    """
    start = index.min().normalize() - pd.Timedelta(days=14)  # holiday slack
    raw = store.read(DS_TREASURY, MARKET_PSEUDO_TICKER, start, index.max())
    if raw.empty or "year10" not in raw.columns:
        logger.warning("causal frame: no treasury_rates/year10 in the lake — "
                       "treasury_10y_chg filled with 0.0")
        return pd.Series(0.0, index=index)

    daily = (raw[["date", "year10"]]
             .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
             .dropna()
             .drop_duplicates("date", keep="last")
             .sort_values("date"))
    daily["chg"] = daily["year10"].diff()
    daily = daily.dropna(subset=["chg"])
    if daily.empty:
        logger.warning("causal frame: <2 treasury observations — "
                       "treasury_10y_chg filled with 0.0")
        return pd.Series(0.0, index=index)

    # One session date per frame minute -> the last strictly-earlier change.
    minutes = pd.DataFrame({"day": index.normalize()})
    mapped = pd.merge_asof(minutes, daily[["date", "chg"]],
                           left_on="day", right_on="date",
                           allow_exact_matches=False)   # STRICTLY before
    return pd.Series(mapped["chg"].to_numpy(), index=index)


def _news_timestamps(store: ParquetStore, tickers: list[str],
                     start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """All news publication times (naive-ET epoch seconds, sorted).

    Pools per-ticker ``news_stock`` for the requested universe with the
    market-wide ``news_general`` feed. Datasets that are missing or lack a
    ``publishedDate`` column are skipped with a warning — coverage degrades,
    correctness does not.
    """
    stamps: list[np.ndarray] = []
    sources = [(DS_NEWS_STOCK, alias) for alias in tickers]
    sources.append((DS_NEWS_GENERAL, MARKET_PSEUDO_TICKER))
    for dataset, key in sources:
        try:
            df = store.read(dataset, key, start, end,
                            columns=["publishedDate"],
                            time_col="publishedDate")
        except Exception as exc:  # malformed partition: skip, don't die
            logger.warning("causal frame: reading %s/%s failed (%s) — skipped",
                           dataset, key, exc)
            continue
        if df.empty or "publishedDate" not in df.columns:
            continue
        ts = pd.to_datetime(df["publishedDate"])
        if ts.dt.tz is not None:  # FMP quotes ET; normalize aware -> naive ET
            ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
        stamps.append(ts.to_numpy().astype("datetime64[s]").astype(np.int64))
    if not stamps:
        return np.zeros(0, np.int64)
    return np.sort(np.concatenate(stamps))


def _news_channel(store: ParquetStore, tickers: list[str],
                  index: pd.DatetimeIndex) -> pd.Series:
    """``news_rate``: trailing-30min news count per minute, z in-window.

    A minute t counts items with publication time in ``(t-30min, t]`` —
    only news already published at t, never future items. The raw count is
    then z-scored in-window (see module honesty notes). No news in the lake
    ⇒ zero column plus a warning.
    """
    lo = index.min() - pd.Timedelta(seconds=NEWS_WINDOW_SECONDS + 60)
    news_s = _news_timestamps(store, tickers, lo, index.max())
    if news_s.size == 0:
        logger.warning("causal frame: no news items in the lake — "
                       "news_rate filled with 0.0")
        return pd.Series(0.0, index=index)

    minute_s = index.to_numpy().astype("datetime64[s]").astype(np.int64)
    # count((t-30m, t]) = #items <= t  minus  #items <= t-30m.
    hi = np.searchsorted(news_s, minute_s, side="right")
    lo_ct = np.searchsorted(news_s, minute_s - NEWS_WINDOW_SECONDS, side="right")
    counts = (hi - lo_ct).astype(np.float64)
    return _zscore_in_window(pd.Series(counts, index=index))


def build_causal_frame(store: ParquetStore, tickers: list[str],
                       start: str, end: str) -> pd.DataFrame:
    """Build the channel matrix the causal model fits on.

    Parameters
    ----------
    store:
        The Parquet lake.
    tickers:
        Ticker aliases. Each contributes the four CAUSAL_TICKER_CHANNELS;
        tickers with no usable 1min bars are skipped with a warning.
    start, end:
        Inclusive session-date range, ``YYYY-MM-DD``.

    Returns
    -------
    DataFrame indexed by the naive-ET minute timestamp, columns
    ``<ALIAS>.<channel>`` for every kept ticker (in the given order)
    followed by ``MKT.treasury_10y_chg`` and ``MKT.news_rate``. Rows are
    the INNER JOIN over (date, minute) across tickers — only minutes every
    ticker traded survive — and any row with a NaN is dropped (e.g. minutes
    before the first available treasury observation).
    """
    start_ts = pd.Timestamp(start).normalize()
    end_excl = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

    per_ticker: list[pd.DataFrame] = []
    kept: list[str] = []
    for alias in tickers:
        raw = store.read(DS_1MIN, alias, start_ts, end_excl)
        if raw.empty or not set(_OHLCV) <= set(raw.columns):
            logger.warning("causal frame: %s has no usable 1min bars in "
                           "[%s, %s] — skipped", alias, start, end)
            continue
        bars = filter_regular_session(raw[list(_OHLCV)])
        if bars.empty:
            logger.warning("causal frame: %s 1min bars all outside the "
                           "regular session — skipped", alias)
            continue
        per_ticker.append(_ticker_channels(bars, alias))
        kept.append(alias)
    if not per_ticker:
        raise ValueError(f"build_causal_frame: no usable tickers among "
                         f"{tickers} in [{start}, {end}]")

    # Inner join on the minute timestamp == join on (date, minute): only
    # minutes present for EVERY ticker are structurally comparable.
    frame = pd.concat(per_ticker, axis=1, join="inner").sort_index()
    if frame.empty:
        raise ValueError("build_causal_frame: tickers share no common "
                         "regular-session minutes")

    # Market-wide channels, appended in the exact CAUSAL_MARKET_CHANNELS
    # order ("treasury_10y_chg", "news_rate"). Treasury goes on FIRST and
    # its NaN rows (minutes before the first available previous-day change)
    # are dropped BEFORE the news channel is computed, so the news z-score's
    # "window" is exactly the rows the final frame keeps.
    assert CAUSAL_MARKET_CHANNELS == ("treasury_10y_chg", "news_rate")
    frame["MKT.treasury_10y_chg"] = _treasury_channel(store, frame.index)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("build_causal_frame: no minutes survive the "
                         "treasury previous-day-change requirement")
    frame["MKT.news_rate"] = _news_channel(store, kept, frame.index)

    frame = frame.dropna()  # defensive; news_rate introduces no NaNs
    frame.index.name = "date"
    logger.info("causal frame: %d minutes x %d channels (%s .. %s)",
                len(frame), frame.shape[1], start, end)
    return frame


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class CausalModel:
    """Linear lagged SEM with NOTEARS acyclicity and bootstrap stability.

    Life cycle::

        model = CausalModel(cfg)
        model.fit(frame)                          # full fit + bootstraps
        snap = model.snapshot(start, end)         # thresholded, confident edges
        # or, for the rolling loop (windowing is the caller's job):
        snap = model.update(frame, prev_snapshot)

    Determinism: weights are zero-initialized and the optimizer path is
    deterministic, so the full fit depends only on the data; bootstrap
    randomness is confined to the row-subsample seeds (0..B-1).
    """

    def __init__(self, cfg: CausalConfig) -> None:
        self.cfg = cfg
        self.nodes: list[str] = []
        #: [n_lags+1, d, d] standardized-weight tensor of the FULL fit;
        #: weights_[l, i, j] = effect of nodes[i] at lag l on nodes[j] now.
        self.weights_: np.ndarray | None = None
        #: [n_lags+1, d, d] bootstrap stability in [0, 1] (see module doc).
        self.confidence_: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def _design(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardize the frame and lay it out as a lagged design.

        Returns ``(Xlag, Y)`` with ``Xlag[r, l, :] = X[t-l]`` and
        ``Y[r] = X[t]`` for target times ``t = n_lags .. T-1`` (row
        ``r = t - n_lags``). Standardization is internal — snapshot weights
        are therefore in per-sigma units of this window.
        """
        lags = self.cfg.n_lags
        raw = frame.to_numpy(np.float64)
        if raw.shape[0] <= lags + 1:
            raise ValueError(f"CausalModel.fit: need > {lags + 1} rows, "
                             f"got {raw.shape[0]}")
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), _MIN_STD)   # constant col -> zeros
        x = (raw - mean) / std
        # Xlag[r, l] = x[lags - l + r]  ==  x[t - l] with t = lags + r.
        xlag = np.stack([x[lags - l: raw.shape[0] - l] for l in range(lags + 1)],
                        axis=1)
        y = x[lags:]
        return (torch.from_numpy(xlag).to(torch.float32),
                torch.from_numpy(y).to(torch.float32))

    def _fit_once(self, xlag: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One Adam optimization of the SEM weights on the given rows.

        Loss = MSE  +  l1_penalty * Σ|W|  +  acyclic_penalty * h(W_0)
        with h(W) = trace(exp(W∘W)) − d  (NOTEARS; zero iff W is a DAG).
        The lag-0 diagonal is hard-masked (a node cannot contemporaneously
        cause itself), which also keeps h's diagonal contribution honest.
        Returns the masked weight tensor [n_lags+1, d, d], detached.
        """
        n_blocks, d = xlag.shape[1], xlag.shape[2]
        mask = torch.ones(n_blocks, d, d)
        mask[0] -= torch.eye(d)                      # no lag-0 self-loops

        # Zero init: the SEM starts as "nothing causes anything" and only
        # grows edges the data demands — and it makes full fits reproducible
        # without any RNG.
        weights = torch.zeros(n_blocks, d, d, requires_grad=True)
        opt = torch.optim.Adam([weights], lr=self.cfg.lr)
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            w = weights * mask
            # pred[t] = Σ_l  X[t-l] @ W_l   — one einsum for all rows/lags.
            pred = torch.einsum("tld,lde->te", xlag, w)
            mse = torch.mean((pred - y) ** 2)
            l1 = self.cfg.l1_penalty * w.abs().sum()
            w0 = w[0]
            acyclic = torch.trace(torch.matrix_exp(w0 * w0)) - d
            loss = mse + l1 + self.cfg.acyclic_penalty * acyclic
            loss.backward()
            opt.step()
        return (weights * mask).detach()

    def fit(self, frame: pd.DataFrame) -> "CausalModel":
        """Full fit plus stability bootstraps. Returns ``self``.

        The frame must be a :func:`build_causal_frame`-style matrix: one row
        per minute, one column per node. Column order defines node order.
        """
        self.nodes = [str(c) for c in frame.columns]
        xlag, y = self._design(frame)

        full = self._fit_once(xlag, y)

        n_boot = max(0, int(self.cfg.stability_bootstraps))
        if n_boot == 0:
            # Degenerate config: no stability evidence. Treat every
            # above-threshold edge as confidence 1 so snapshots still work,
            # but say so loudly.
            logger.warning("CausalModel.fit: stability_bootstraps=0 — edge "
                           "confidence defaults to presence in the full fit")
            confidence = (full.abs() > EDGE_THRESHOLD).to(torch.float32)
        else:
            n_rows = y.shape[0]
            n_sub = max(1, int(BOOTSTRAP_FRACTION * n_rows))
            agree = torch.zeros_like(full)
            full_sign = torch.sign(full)
            for boot in range(n_boot):
                # Distinct, fixed seeds: reproducible confidence estimates.
                rng = np.random.default_rng(boot)
                idx = np.sort(rng.choice(n_rows, size=n_sub, replace=False))
                rows = torch.from_numpy(idx)
                w_boot = self._fit_once(xlag[rows], y[rows])
                # An edge "survives" a bootstrap only if it is BOTH above
                # threshold and sign-consistent with the full fit; edges
                # that flip sign across subsamples are noise.
                agree += ((w_boot.abs() > EDGE_THRESHOLD)
                          & (torch.sign(w_boot) == full_sign)).to(agree.dtype)
            confidence = agree / n_boot

        self.weights_ = full.numpy()
        self.confidence_ = confidence.numpy()
        return self

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #

    def snapshot(self, fitted_start: str, fitted_end: str) -> CausalGraphSnapshot:
        """Materialize the fitted structure as a timestamped snapshot.

        Only edges with ``|weight| > EDGE_THRESHOLD`` AND nonzero bootstrap
        confidence are emitted (weights are per-sigma, standardized units).
        Edges are sorted by ``|weight|·confidence`` so the strongest,
        most stable structure reads first.
        """
        if self.weights_ is None or self.confidence_ is None:
            raise RuntimeError("CausalModel.snapshot: call fit() first")
        edges: list[CausalEdge] = []
        for lag in range(self.weights_.shape[0]):
            for i, src in enumerate(self.nodes):
                for j, dst in enumerate(self.nodes):
                    weight = float(self.weights_[lag, i, j])
                    conf = float(self.confidence_[lag, i, j])
                    if abs(weight) > EDGE_THRESHOLD and conf > 0.0:
                        edges.append(CausalEdge(src=src, dst=dst, lag=lag,
                                                weight=weight, confidence=conf))
        edges.sort(key=lambda e: -abs(e.weight) * e.confidence)
        return CausalGraphSnapshot(fitted_start=str(fitted_start),
                                   fitted_end=str(fitted_end),
                                   nodes=list(self.nodes), edges=edges)

    def update(self, frame: pd.DataFrame,
               prev_snapshot: CausalGraphSnapshot | None = None
               ) -> CausalGraphSnapshot:
        """Refit on a new window and return the fresh snapshot.

        Rolling-window bookkeeping (what slice of history goes into
        ``frame``) is deliberately the CALLER's job. ``prev_snapshot`` is
        NOT used to warm-start the optimization — warm-starting a NOTEARS
        objective can lock in stale structure; a fresh fit plus bootstrap
        confidence is the honest continuity signal. It is used only to log
        edge churn, the graph's "aliveness" diagnostic.
        """
        self.fit(frame)
        fitted_start, fitted_end = _frame_span(frame)
        snap = self.snapshot(fitted_start, fitted_end)
        if prev_snapshot is not None:
            prev = {(e.src, e.dst, e.lag) for e in prev_snapshot.edges}
            cur = {(e.src, e.dst, e.lag) for e in snap.edges}
            logger.info("causal update: %d edges (+%d appeared, -%d vanished "
                        "vs previous snapshot)", len(cur),
                        len(cur - prev), len(prev - cur))
        return snap


def _frame_span(frame: pd.DataFrame) -> tuple[str, str]:
    """(fitted_start, fitted_end) strings from a frame's index."""
    if len(frame.index) == 0:
        return "", ""
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index.min().isoformat(), frame.index.max().isoformat()
    return str(frame.index[0]), str(frame.index[-1])


# --------------------------------------------------------------------------- #
# JSON serialization + persistence
# --------------------------------------------------------------------------- #

def snapshot_to_json(snap: CausalGraphSnapshot) -> str:
    """Serialize a snapshot to a stable, human-diffable JSON document."""
    payload = {
        "fitted_start": snap.fitted_start,
        "fitted_end": snap.fitted_end,
        "nodes": list(snap.nodes),
        "edges": [dataclasses.asdict(e) for e in snap.edges],
    }
    return json.dumps(payload, indent=2)


def snapshot_from_json(text: str) -> CausalGraphSnapshot:
    """Inverse of :func:`snapshot_to_json`."""
    obj = json.loads(text)
    return CausalGraphSnapshot(
        fitted_start=obj["fitted_start"],
        fitted_end=obj["fitted_end"],
        nodes=list(obj["nodes"]),
        edges=[CausalEdge(**e) for e in obj["edges"]],
    )


# The contract (interfaces.CausalGraphSnapshot) declares ``to_json`` but
# leaves the body to this module. We BIND the module-level serializer as the
# method at import time rather than subclassing: every snapshot instance —
# including ones built by interfaces-only code or ``snapshot_from_json`` —
# gains the real serializer, the dataclass contract stays byte-identical,
# and there is exactly one serialization code path.
CausalGraphSnapshot.to_json = snapshot_to_json  # type: ignore[method-assign]


#: Filename-safe characters for the snapshot timestamp. ISO timestamps map
#: ":" -> "-" (fixed width), so lexicographic filename order still equals
#: chronological order — which load_latest_snapshot relies on.
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_\-T]")


def save_snapshot(snap: CausalGraphSnapshot, data_root: Path | str) -> Path:
    """Write ``<data_root>/causal/snapshot_<fitted_end>.json``; return path."""
    out_dir = Path(data_root) / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _UNSAFE_FILENAME.sub("-", str(snap.fitted_end))
    path = out_dir / f"snapshot_{stamp}.json"
    path.write_text(snapshot_to_json(snap))
    logger.info("saved causal snapshot -> %s (%d edges)", path, len(snap.edges))
    return path


def load_latest_snapshot(data_root: Path | str) -> CausalGraphSnapshot | None:
    """Most recent saved snapshot, or None if none exist.

    "Most recent" = lexicographically last filename, which equals
    chronological order for the sanitized ISO stamps save_snapshot writes.
    """
    out_dir = Path(data_root) / "causal"
    if not out_dir.is_dir():
        return None
    files = sorted(out_dir.glob("snapshot_*.json"))
    if not files:
        return None
    return snapshot_from_json(files[-1].read_text())
