"""Shared builders for the Layer 2-6 test suite (world model, decision core,
execution, evolution, dashboard).

Everything here is derived strictly from the contract modules
(``aether.worldmodel.interfaces``, ``aether.decision.interfaces``,
``aether.execution.interfaces``) plus the Layer 0/1 conventions already pinned
by ``tests/conftest.py``. The implementation modules are written concurrently,
so nothing in this file imports them.

Synthetic embedding store
-------------------------
``make_embedding_npz_dir`` writes per-ticker ``<TICKER>.npz`` files in exactly
the ``EmbeddingStore.load`` format::

    fused          [N, D]  float32   perception state per anchor bar
    aleatoric      [N]     float32
    epistemic      [N]     float32
    anomaly        [N]     float32
    anchor_ts      [N]     int64     naive-ET wall time viewed as UTC epoch
    session_minute [N]     int16     0..n_bars-1 within each session
    close_px       [N]     float64   anchor bar close
    next_open      [N]     float64   next bar's open  (fill price source)
    next_high      [N]     float64   next bar's high
    next_low       [N]     float64   next bar's low
    next_close     [N]     float64   next bar's close

``next_*`` never crosses a session boundary: each session's LAST row holds
NaN, per the EmbeddingStore contract (no same-session fill bar exists).

The price path is a deterministic sine + drift so every fill is predictable:

* bars ``0 .. flat_from-1`` of each session: ``close = base * (1 + 0.05 *
  sin(2*pi*m/60) + 0.0008*m)`` with a generous intra-bar range (+-0.4%), giving
  real swings for stop/target scenarios;
* bars ``flat_from .. n_bars-1``: close frozen at its last sine value with a
  tiny intra-bar range (+-0.08%), a quiet zone where neither a stop nor a
  target at any sane learned distance can be touched — used by fill-math,
  adjust and end-of-day tests.

Sessions are the first ``n_sessions`` real NYSE trading days from the
conftest's ``SYNTH_START`` (2026-01-05), truncated to ``n_bars`` bars each.
``anchor_ts`` follows the lake convention (naive-ET timestamps whose epoch is
taken as if they were UTC — the ``naive_as_utc`` decoding of
``tests.conftest.anchor_wall_time``).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tests.conftest import synth_sessions

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EMBED_DIM = 16          #: fused embedding width used across the tiny stack
N_SESSIONS = 2          #: sessions per synthetic embedding store
SESSION_BARS = 120      #: default bars per (truncated) session
FLAT_FROM = 80          #: session bar where the price path goes flat

#: Intra-bar half-range (fraction of price) in the sine and flat zones.
SINE_PAD = 0.004
FLAT_PAD = 0.0008


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #

def epoch_s(ts) -> int:
    """Naive-timestamp-as-UTC epoch seconds (the lake's anchor convention)."""
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def session_bar_time(day, minute: int) -> datetime:
    """Naive-ET wall time of session ``minute`` (0 == 09:30) on ``day``."""
    return datetime.combine(day, time(9, 30)) + timedelta(minutes=int(minute))


# --------------------------------------------------------------------------- #
# Embedding store builder
# --------------------------------------------------------------------------- #

def make_price_path(n_bars: int, base: float, flat_from: int) -> np.ndarray:
    """Deterministic sine+trend close path, frozen flat from ``flat_from``."""
    m = np.arange(n_bars, dtype=np.float64)
    close = base * (1.0 + 0.05 * np.sin(2.0 * np.pi * m / 60.0) + 0.0008 * m)
    fs = min(max(int(flat_from), 1), n_bars)
    close[fs:] = close[fs - 1]
    return close


def make_embedding_npz_dir(tmp_path,
                           tickers: tuple[str, ...] = ("AAPL", "SPY"),
                           n_bars: int = SESSION_BARS,
                           seed: int = 0,
                           embed_dim: int = EMBED_DIM,
                           n_sessions: int = N_SESSIONS,
                           flat_from: int = FLAT_FROM) -> Path:
    """Write per-ticker ``<TICKER>.npz`` embedding files; return the dir."""
    out = Path(tmp_path) / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    days = synth_sessions(n_sessions)

    for ti, ticker in enumerate(tickers):
        rng = np.random.default_rng(seed * 1_000_003 + sum(ticker.encode()))
        n_total = n_bars * n_sessions

        close = np.empty(n_total)
        open_ = np.empty(n_total)
        high = np.empty(n_total)
        low = np.empty(n_total)
        anchor_ts = np.empty(n_total, dtype=np.int64)
        session_minute = np.empty(n_total, dtype=np.int16)

        for s, day in enumerate(days):
            lo_i, hi_i = s * n_bars, (s + 1) * n_bars
            base = (80.0 + 10.0 * ti) * (1.0 + 0.01 * s)
            c = make_price_path(n_bars, base, flat_from)
            o = np.empty_like(c)
            o[0] = base
            o[1:] = c[:-1]
            pad = np.where(np.arange(n_bars) < flat_from, SINE_PAD, FLAT_PAD)
            close[lo_i:hi_i] = c
            open_[lo_i:hi_i] = o
            high[lo_i:hi_i] = np.maximum(o, c) * (1.0 + pad)
            low[lo_i:hi_i] = np.minimum(o, c) * (1.0 - pad)
            session_minute[lo_i:hi_i] = np.arange(n_bars, dtype=np.int16)
            anchor_ts[lo_i:hi_i] = [epoch_s(session_bar_time(day, m))
                                    for m in range(n_bars)]

        def _next(a: np.ndarray) -> np.ndarray:
            """Shift by one bar; session-LAST rows get NaN ``next_*``.

            The EmbeddingStore contract says the last anchor of each session
            has NaN next_* — there is no same-session fill bar. A plain
            shift would bleed session 2's first bar into session 1's last
            anchor, silently letting positions trade across the overnight
            boundary (this exact bleed masked a real eod bug).
            """
            out = np.append(a[1:].astype(np.float64), np.nan)
            out[n_bars - 1::n_bars] = np.nan
            return out

        np.savez(
            out / f"{ticker}.npz",
            fused=rng.standard_normal((n_total, embed_dim)).astype(np.float32),
            aleatoric=(0.04 + 0.02 * rng.random(n_total)).astype(np.float32),
            epistemic=(0.05 + 0.02 * rng.random(n_total)).astype(np.float32),
            anomaly=(0.10 + 0.05 * rng.random(n_total)).astype(np.float32),
            anchor_ts=anchor_ts,
            session_minute=session_minute,
            close_px=close.astype(np.float64),
            next_open=_next(open_).astype(np.float64),
            next_high=_next(high).astype(np.float64),
            next_low=_next(low).astype(np.float64),
            next_close=_next(close).astype(np.float64),
        )
    return out


def load_npz(dir_, ticker: str) -> dict[str, np.ndarray]:
    """Read one ticker's arrays back as a plain dict."""
    with np.load(Path(dir_) / f"{ticker}.npz") as z:
        return {k: z[k].copy() for k in z.files}


def rewrite_npz(dir_, ticker: str, mutator) -> dict[str, np.ndarray]:
    """Load, mutate in place via ``mutator(arrays_dict)``, and save back."""
    arrays = load_npz(dir_, ticker)
    mutator(arrays)
    np.savez(Path(dir_) / f"{ticker}.npz", **arrays)
    return arrays


# --------------------------------------------------------------------------- #
# TradeRecord factory
# --------------------------------------------------------------------------- #

def make_trade(trade_id: str = "T-0001",
               ticker: str = "AAPL",
               side: str = "long",
               entry_ts: int | None = None,
               exit_ts: int | None = None,
               entry_px: float = 100.0,
               exit_px: float = 99.0,
               qty: float = 10.0,
               pnl: float | None = None,
               fees: float | None = None,
               stop_px: float = 99.0,
               target_px: float = 103.0,
               exit_reason: str = "stop",
               conviction: float = 0.7,
               signal_meta: dict | None = None,
               fees_bps: float = 1.0):
    """Internally consistent TradeRecord; pnl/fees derived unless given."""
    from aether.worldmodel.interfaces import TradeRecord

    day = synth_sessions(1)[0]
    if entry_ts is None:
        entry_ts = epoch_s(session_bar_time(day, 30))
    if exit_ts is None:
        exit_ts = entry_ts + 5 * 60
    if fees is None:
        fees = (entry_px + exit_px) * qty * fees_bps / 1e4
    if pnl is None:
        sign = 1.0 if side == "long" else -1.0
        pnl = sign * (exit_px - entry_px) * qty - fees
    return TradeRecord(
        trade_id=trade_id, ticker=ticker, side=side,
        entry_ts=int(entry_ts), exit_ts=int(exit_ts),
        entry_px=float(entry_px), exit_px=float(exit_px), qty=float(qty),
        pnl=float(pnl), fees=float(fees),
        stop_px=float(stop_px), target_px=float(target_px),
        exit_reason=exit_reason, conviction=float(conviction),
        signal_meta=signal_meta or {},
    )


# --------------------------------------------------------------------------- #
# Tiny config factories (dims 8-32 keep every test CPU-fast)
# --------------------------------------------------------------------------- #

def tiny_dynamics_cfg(embed_dim: int = EMBED_DIM, **overrides):
    from aether.worldmodel.interfaces import DynamicsConfig
    cfg = DynamicsConfig(embed_dim=embed_dim, deter_dim=16, stoch_dim=8,
                         hidden_dim=16, kl_beta=1.0, free_nats=1.0, horizon=5)
    return dataclasses.replace(cfg, **overrides)


def tiny_policy_cfg(obs_market_dim: int = EMBED_DIM, meta_every: int = 5,
                    **overrides):
    from aether.decision.interfaces import PolicyConfig
    cfg = PolicyConfig(obs_market_dim=obs_market_dim, hidden_dim=32,
                       memory_dim=16, meta_every=meta_every)
    return dataclasses.replace(cfg, **overrides)


def tiny_env_cfg(embeddings_dir, **overrides):
    from aether.decision.interfaces import EnvConfig
    days = synth_sessions(N_SESSIONS)
    cfg = EnvConfig(
        embeddings_dir=str(embeddings_dir),
        tickers=("AAPL", "SPY"),
        start=str(days[0]),
        end=str(days[-1]),
        fees_bps=1.0,
        slippage_bps=2.0,
        meta_every=5,
        max_position_frac=1.0,
        episode="session",
        initial_equity=100_000.0,
        seed=0,
    )
    return dataclasses.replace(cfg, **overrides)


# --------------------------------------------------------------------------- #
# Action / observation builders
# --------------------------------------------------------------------------- #

def make_actions(n_envs: int = 1, meta: int = 1, trade: int = 0,
                 size: float = 1.0, stop: float = 1.0,
                 target: float = 1.0) -> dict[str, np.ndarray]:
    """Contract-shaped action dict (meta=1 is hunt_long, trade=0 is hold)."""
    return {
        "meta": np.full(n_envs, meta, dtype=np.int64),
        "trade": np.full(n_envs, trade, dtype=np.int64),
        "size": np.full(n_envs, size, dtype=np.float32),
        "stop": np.full(n_envs, stop, dtype=np.float32),
        "target": np.full(n_envs, target, dtype=np.float32),
    }


def make_policy_obs(batch: int, obs_market_dim: int = EMBED_DIM,
                    flags: float = 1.0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Contract-shaped observation dict of torch tensors for the policy."""
    g = torch.Generator().manual_seed(seed)
    meta = torch.zeros(batch, 3)
    meta[:, 0] = 1.0                                    # stand_aside one-hot
    return {
        "market": torch.randn(batch, obs_market_dim, generator=g),
        "unc": 0.1 * torch.rand(batch, 3, generator=g),
        "position": torch.zeros(batch, 5),
        "portfolio": torch.tensor([[1.0, 0.0, 0.0]]).repeat(batch, 1),
        "clock": torch.tensor([[0.1, 0.9]]).repeat(batch, 1),
        "meta": meta,
        "flags": torch.full((batch, 1), float(flags)),
    }
