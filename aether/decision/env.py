"""The Layer-3 trading environment: a look-ahead-safe historical replay.

Two pieces live here:

* :class:`EmbeddingStore` — runs the frozen perception model over the lake
  once (stride 1) and caches per-ticker ``.npz`` files holding everything
  the environment needs per anchor bar: the fused embedding, uncertainty
  triplet, timestamps, the anchor close (mark price) and the FOLLOWING
  1-minute bar's OHLC (the fill bar). Training then never touches torch's
  perception stack or the Parquet lake — episodes are pure numpy replay.

* :class:`TradingEnv` — a vectorized (over ``n_envs``) bar-by-bar simulator
  where each env instance replays one (ticker, session-date) episode.

The one sacred invariant — decisions at t, consequences at t+1
--------------------------------------------------------------
The observation at bar pointer ``t`` is built exclusively from arrays at
index ``t`` (whose contents the perception dataset already guarantees to be
derived from data ≤ t). Every order decided at ``t`` executes against the
``next_*`` arrays at index ``t`` — which *are* bar ``t+1``'s prices. That is
the point: the ``next_*`` columns exist so the fill math cannot accidentally
index the decision bar. The realized-range statistic that scales stops and
targets is a rolling mean over bar ranges of bars ≤ t (``next_high−next_low``
shifted: index ``j`` holds bar ``j+1``'s range, so bars ≤ t live at indices
``< t``). Nothing in an observation ever reads an index > t.

Execution model (conservative by construction)
----------------------------------------------
Within one ``step`` call at pointer ``t``, per env, in this exact order:

1. **Entry.** A ``trade='enter'`` decided at ``t`` (only while the meta
   intent in force is ``hunt_long``/``hunt_short`` and the env is flat)
   fills at ``next_open ± slippage_bps`` (against the agent: longs buy
   higher, shorts sell lower), pays ``fees_bps``, and sizes
   ``qty = size · max_position_frac · equity / fill_px``. Stop and target
   prices are set at ``fill ∓/± frac · realized_range(t)``. ``'enter'``
   while the intent is ``stand_aside`` is a **no-op**: the meta controller
   owns the *license* to trade and the sub-policy cannot overrule it.
   ``'enter'`` while already positioned is likewise a no-op (no pyramiding
   — one position per env keeps accounting and TradeRecords unambiguous).
2. **Stop/target sweep** of bar ``t+1`` for any position now held —
   including one filled in step 1 (the open is the bar's first print, so an
   entry-bar stop-out is chronologically possible and simulating it is the
   conservative choice). If BOTH the stop and the target are touched inside
   the bar, the **stop fills** (worst case — intra-bar ordering is
   unknowable from OHLC). The stop is a market order: it fills at
   ``stop_px ± slippage`` plus fees. The target is a resting limit order:
   it fills AT ``target_px`` exactly, fees only, no slippage.
3. **Policy exit.** ``trade='exit'`` on a position that survived step 2
   fills at ``next_open ± slippage`` (reason ``policy_exit``). Note the
   deliberate pessimism of the 2→3 order: when a stop/target touch and an
   exit request coincide, the stop/target result stands even though the
   open (the exit's fill) happens first chronologically — for a stop that
   is the worse of the two fills.
4. **Adjust.** ``trade='adjust'`` re-derives stop/target from the action
   fractions × ``realized_range(t)`` around the latest known close. The
   stop may only ever **tighten** (move toward the position), never widen:
   a widening stop is a learned way to run losers, and there is no
   legitimate reversal-hunting reason to grant it. The target re-arms
   freely. The new levels take effect from the next bar's sweep (this
   step's sweep in 2 already used the pre-adjust levels).
5. **Meta re-decision**, only when ``t % meta_every == 0`` (the bars where
   ``obs['flags'] == 1``). The intent the agent *saw* in ``obs['meta']`` at
   bar ``t`` is the intent its bar-``t`` trade action was interpreted under
   (steps 1–4); the newly chosen intent governs from the next bar on.
   Switching intent while positioned does NOT close the position — the
   sub-policy must manage its own exit, otherwise the meta head could
   silently flatten positions without ever learning exit behavior.
6. **Advance** the pointer. On reaching the episode's final bar any open
   position is force-closed at that bar's close ± slippage with fees
   (``exit_reason='eod'`` — Aether does not hold overnight), and the env is
   done. The terminal reward additionally receives the hindsight
   reversal-capture bonus (see :mod:`aether.decision.reward`).

Auto-reset (documented contract decision)
-----------------------------------------
The interface contract is silent on what happens after an episode ends, so
this env commits to SAME-STEP auto-reset — the semantics the PPO trainer
and its reference MiniEnv assume: the step that finishes an episode returns
``done=True`` (exactly once, with the terminal reward including the
hindsight bonus) and the *observation of a freshly sampled episode's first
bar*. The next action therefore applies to the new episode, the trainer
zeroes the recurrent carry at the boundary, and GAE masks the bootstrap
with ``1 − done``. No env is ever inert: consumers that replay a fixed
episode set (backtests) must stop reading an env's stream once it has
reported done.

Every closed round-trip is emitted as a
:class:`~aether.worldmodel.interfaces.TradeRecord` under
``info[i]['trade_closed']``. Equity is marked to market on the anchor close
each bar (``equity = cash + qty · close``), so the accounting identity
``cash + qty·px == equity`` holds exactly at every step.

Vectorization note: the *interface* is vectorized ([n_envs, ...] arrays in,
[n_envs] rewards out) but the per-env transition is a plain scalar loop.
``n_envs`` is small (PPO default 16) and this file is the correctness-
critical heart of the release — a transparent, auditable scalar path beats
a clever broadcasted one here; observation assembly is array-based.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..utils.logging import get_logger
from ..utils.market_time import MINUTES_PER_SESSION
from ..worldmodel.interfaces import TradeRecord
from .interfaces import (
    ACTION_KEYS,
    META_ACTIONS,
    OBS_KEYS,
    TRADE_ACTIONS,
    EnvConfig,
    EnvStep,
    RewardConfig,
)
from .reward import RewardShaper

logger = get_logger("aether.decision.env")

# Action indices derived from the contract tuples (never hardcoded ints).
_META_STAND_ASIDE = META_ACTIONS.index("stand_aside")
_META_HUNT_LONG = META_ACTIONS.index("hunt_long")
_META_HUNT_SHORT = META_ACTIONS.index("hunt_short")
_TRADE_ENTER = TRADE_ACTIONS.index("enter")
_TRADE_EXIT = TRADE_ACTIONS.index("exit")
_TRADE_ADJUST = TRADE_ACTIONS.index("adjust")

#: Arrays every per-ticker npz must contain (the EmbeddingStore format).
NPZ_KEYS: tuple[str, ...] = (
    "fused", "aleatoric", "epistemic", "anomaly", "anchor_ts",
    "session_minute", "close_px",
    "next_open", "next_high", "next_low", "next_close",
)

#: Rolling window (bars ≤ t) for the realized 1-min range statistic that
#: scales stop/target distances.
RANGE_WINDOW_BARS: int = 15

#: Minimum bars for a (ticker, session) group to count as an episode.
#: The floor exists to reject degenerate data holes (a handful of stray
#: anchors), NOT to exclude short-but-real sessions: NYSE half days after
#: perception-window trimming can yield <100 anchors and remain perfectly
#: tradable, and the stack's synthetic fixtures replay 120-bar sessions.
MIN_EPISODE_BARS: int = 30

#: Realized-range fallback for the degenerate first bar of an episode
#: (no completed bar range is observable yet): a conservative 10 bps of
#: price, roughly a quiet large-cap's 1-minute range.
_RANGE_FALLBACK_FRAC: float = 1e-3

#: Floor for the continuous action fractions (contract: (0, 1]).
_FRAC_FLOOR: float = 1e-6

_SECONDS_PER_DAY: int = 86_400
_OHLCV = ["date", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Embedding store
# --------------------------------------------------------------------------- #

class EmbeddingStore:
    """Precomputed perception embeddings + execution prices, per ticker.

    Implements :class:`~aether.decision.interfaces.EmbeddingStoreProtocol`.
    One ``.npz`` per ticker alias under ``out_dir`` with the arrays listed
    in :data:`NPZ_KEYS` (all length N, time-ordered by ``anchor_ts``):

    * ``fused [N, D] float32`` — perception fused embedding per anchor,
    * ``aleatoric / epistemic / anomaly [N] float32`` — uncertainty triplet,
    * ``anchor_ts [N] int64`` — epoch-seconds (lake's naive-ET convention),
    * ``session_minute [N] int16`` — 0..389 anchor session clock,
    * ``close_px [N] float64`` — the anchor bar's close (mark / PnL price),
    * ``next_open/high/low/close [N] float64`` — the FOLLOWING 1-min bar
      (the fill bar). The last bar of each session has NaN ``next_*`` and is
      therefore never tradable — there is no same-session bar to fill on.
    """

    # ------------------------------------------------------------------ #
    # Precompute
    # ------------------------------------------------------------------ #

    @classmethod
    def precompute(cls, perception_ckpt: str, store, tickers: Sequence[str],
                   start: str, end: str, out_dir: str,
                   device: Optional[str] = None, batch_size: int = 256,
                   stats_range: Optional[tuple[str, str]] = None) -> None:
        """Run the frozen perception model over ``[start, end]``, stride 1.

        The model is rebuilt from the checkpoint itself so embeddings are
        always consistent with the network that produced them:

        * architecture from the checkpoint's config snapshot when one is
          recorded (``perception_cfg``), otherwise inferred from the saved
          state-dict shapes (``d_model``, ``dna_dim``, ticker count) with
          contract defaults elsewhere — a strict ``load_state_dict`` then
          fails loudly on any real architecture mismatch;
        * weights from the EMA shadow when present (validation and best-
          checkpoint selection used the EMA view, so inference must too);
        * normalization stats via :func:`fit_or_load_stats` on the exact
          range the checkpoint was trained on. The trainer's snapshot does
          not currently record that range, so unless the checkpoint carries
          one, the caller MUST pass ``stats_range=(train_start, train_end)``
          — silently fitting stats on some other range would normalize
          inputs differently than training did (and fitting them on
          [start, end] itself could even leak future statistics).
        """
        # Perception imports are local so that TradingEnv consumers (pure
        # numpy replay) never pay for — or depend on — the perception stack.
        from torch.utils.data import DataLoader

        from ..perception.datasets import (
            DS_1MIN,
            PerceptionWindowDataset,
            collate_batch,
            fit_or_load_stats,
        )
        from ..perception.model import PerceptionModel
        from ..perception.preprocessing import filter_regular_session

        ckpt: dict[str, Any] = torch.load(
            perception_ckpt, map_location="cpu", weights_only=False)
        pcfg, n_tickers = cls._perception_config_from(ckpt)
        model = PerceptionModel(pcfg, n_tickers=n_tickers)
        model.load_state_dict(ckpt["model"])  # strict: arch mismatch is loud
        shadow = (ckpt.get("ema") or {}).get("shadow") or {}
        if shadow:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in shadow:
                        param.copy_(shadow[name].to(dtype=param.dtype))
        dev = torch.device(device if device is not None
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(dev).eval()

        rng = cls._stats_range_from(ckpt, stats_range)
        stats = fit_or_load_stats(store, list(tickers), rng[0], rng[1],
                                  Path(store.root) / "ticker_stats")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        written = 0
        for alias in tickers:
            if alias not in stats:
                logger.warning("precompute: no stats for %s — skipped", alias)
                continue
            ds = PerceptionWindowDataset(store, [alias], pcfg.window,
                                         start, end, stats, stride=1)
            if len(ds) == 0:
                logger.warning("precompute: %s has no anchors in [%s, %s] — "
                               "skipped", alias, start, end)
                continue
            loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False,
                                collate_fn=collate_batch)
            fused, alea, epis, anom, ts, sm = [], [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    o = model(batch.to(dev))
                    fused.append(o.fused.float().cpu().numpy())
                    alea.append(o.aleatoric.float().cpu().numpy())
                    epis.append(o.epistemic.float().cpu().numpy())
                    anom.append(o.anomaly.float().cpu().numpy())
                    ts.append(batch.anchor_ts.cpu().numpy().astype(np.int64))
                    # require_full_1m=True ⇒ the anchor bar is always the
                    # window's last position, so its session minute is the
                    # last minute_of_day entry.
                    sm.append(batch.minute_of_day[:, -1].cpu().numpy()
                              .astype(np.int16))
            anchor_ts = np.concatenate(ts)
            if not np.all(np.diff(anchor_ts) > 0):
                raise RuntimeError(
                    f"precompute: {alias} anchors are not strictly "
                    "time-ordered — dataset invariant violated")

            close_px, nxt = cls._aligned_lake_prices(
                store, alias, start, end, anchor_ts, DS_1MIN,
                filter_regular_session)
            np.savez(out / f"{alias}.npz",
                     fused=np.concatenate(fused).astype(np.float32),
                     aleatoric=np.concatenate(alea).astype(np.float32),
                     epistemic=np.concatenate(epis).astype(np.float32),
                     anomaly=np.concatenate(anom).astype(np.float32),
                     anchor_ts=anchor_ts,
                     session_minute=np.concatenate(sm),
                     close_px=close_px,
                     next_open=nxt["open"], next_high=nxt["high"],
                     next_low=nxt["low"], next_close=nxt["close"])
            logger.info("precompute: %s -> %d anchors (D=%d) in %s",
                        alias, anchor_ts.size, fused[0].shape[1], out)
            written += 1
        if written == 0:
            raise RuntimeError(
                f"precompute: no ticker produced embeddings in "
                f"[{start}, {end}] — nothing written to {out}")

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, out_dir: str, ticker: str) -> dict[str, np.ndarray]:
        """Load one ticker's precomputed arrays; validates the format."""
        path = Path(out_dir) / f"{ticker}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"EmbeddingStore.load: {path} does not exist — run "
                "EmbeddingStore.precompute first")
        with np.load(path) as z:
            data = {key: z[key] for key in z.files}
        missing = [k for k in NPZ_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"EmbeddingStore.load: {path} is missing arrays {missing}")
        n = data["anchor_ts"].shape[0]
        bad = [k for k in NPZ_KEYS if data[k].shape[0] != n]
        if bad:
            raise ValueError(
                f"EmbeddingStore.load: {path} arrays {bad} have lengths "
                f"inconsistent with anchor_ts ({n})")
        return data

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _perception_config_from(ckpt: dict[str, Any]):
        """(PerceptionConfig, n_tickers) rebuilt from a trainer checkpoint.

        Prefers an explicit ``perception_cfg`` snapshot when the checkpoint
        carries one (forward compatibility). The current trainer snapshot
        records only the TrainConfig, so the fallback recovers the widths
        that the state dict pins unambiguously — ``d_model`` (every head is
        ``d_model → d_model``), ``dna_dim`` and the ticker count from the
        DNA table — and takes contract defaults for the rest; the strict
        ``load_state_dict`` in the caller turns any remaining architecture
        mismatch into an immediate, explicit error instead of silent junk
        embeddings.
        """
        from ..perception.interfaces import PerceptionConfig

        msd = ckpt["model"]
        n_tickers, dna_dim = (int(s) for s in msd["dna_table.weight"].shape)

        snap = ckpt.get("perception_cfg")
        if isinstance(snap, dict):
            return _dataclass_from_dict(PerceptionConfig(), snap), n_tickers

        d_model = int(msd["masked_head.net.0.weight"].shape[1])
        return PerceptionConfig(d_model=d_model, dna_dim=dna_dim), n_tickers

    @staticmethod
    def _stats_range_from(ckpt: dict[str, Any],
                          stats_range: Optional[tuple[str, str]]
                          ) -> tuple[str, str]:
        """Resolve the TickerStats fitting range (see precompute docstring)."""
        if stats_range is not None:
            lo, hi = stats_range
            return str(lo), str(hi)
        cfg = ckpt.get("cfg") or {}
        for src in (ckpt, cfg):
            pair = src.get("stats_range") or src.get("train_range")
            if pair is not None and len(pair) == 2:
                return str(pair[0]), str(pair[1])
            if src.get("train_start") and src.get("train_end"):
                return str(src["train_start"]), str(src["train_end"])
        raise ValueError(
            "EmbeddingStore.precompute: the checkpoint does not record the "
            "range its normalization statistics were fitted on, and no "
            "stats_range was passed. Pass stats_range=(train_start, "
            "train_end) matching the perception training range exactly — "
            "embeddings computed under different normalization than the "
            "network was trained with are silently wrong.")

    # ------------------------------------------------------------------ #
    # Lake alignment
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aligned_lake_prices(store, alias: str, start: str, end: str,
                             anchor_ts: np.ndarray, ds_1min: str,
                             filter_regular_session):
        """close_px + next-bar OHLC aligned to ``anchor_ts`` from the lake.

        ``next_*`` is the lake's very next 1-minute bar **within the same
        session**; the session's last bar gets NaN (no same-session fill
        bar exists — overnight is not tradable in this system).
        """
        raw = store.read(ds_1min, alias, pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize() + pd.Timedelta(days=1))
        if raw.empty or not set(_OHLCV) <= set(raw.columns):
            raise RuntimeError(
                f"precompute: no 1min lake bars for {alias} in "
                f"[{start}, {end}]")
        bars = filter_regular_session(raw[_OHLCV])
        ts = bars["date"].to_numpy().astype("datetime64[s]").astype(np.int64)
        day = bars["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        ohlc = {c: bars[c].to_numpy(np.float64)
                for c in ("open", "high", "low", "close")}

        pos = np.searchsorted(ts, anchor_ts)
        if pos.size and (int(pos.max()) >= ts.size
                         or not np.array_equal(ts[pos], anchor_ts)):
            raise RuntimeError(
                f"precompute: {alias} anchors do not align 1:1 with lake "
                "bars — the dataset and the lake disagree")
        nxt_pos = np.minimum(pos + 1, ts.size - 1)
        has_next = (pos + 1 < ts.size) & (day[nxt_pos] == day[pos])

        def take(col: str) -> np.ndarray:
            arr = np.full(anchor_ts.size, np.nan, dtype=np.float64)
            arr[has_next] = ohlc[col][pos[has_next] + 1]
            return arr

        nxt = {c: take(c) for c in ("open", "high", "low", "close")}
        return ohlc["close"][pos], nxt


def _dataclass_from_dict(obj, payload: dict):
    """Recursively overlay a plain-dict snapshot onto a dataclass instance."""
    for key, value in payload.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _dataclass_from_dict(current, value)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #

@dataclass
class _Episode:
    """One (ticker, session-date) replay path — immutable numpy views."""

    ticker: str
    date: str                  # ISO session date
    fused: np.ndarray          # [T, D] float32
    unc: np.ndarray            # [T, 3] float32 (aleatoric, epistemic, anomaly)
    close: np.ndarray          # [T] float64 — anchor close (mark price)
    next_open: np.ndarray      # [T] float64 — fill-bar OHLC (bar t+1)
    next_high: np.ndarray
    next_low: np.ndarray
    bar_range: np.ndarray      # [T] float64 — next_high−next_low, i.e. index
                               # j holds bar j+1's realized range (shifted!)
    minute: np.ndarray         # [T] int — anchor session minute 0..389
    ts: np.ndarray             # [T] int64 — anchor epoch-seconds

    @property
    def length(self) -> int:
        return int(self.close.shape[0])


# --------------------------------------------------------------------------- #
# Trading environment
# --------------------------------------------------------------------------- #

class TradingEnv:
    """Vectorized intraday replay env (see the module docstring for the
    full execution model). Implements
    :class:`~aether.decision.interfaces.TradingEnvProtocol`.

    Parameters
    ----------
    cfg:
        :class:`~aether.decision.interfaces.EnvConfig`. ``cfg.tickers``
        selects which npz files to index (empty = every npz in
        ``cfg.embeddings_dir``); ``cfg.start``/``cfg.end`` (when non-empty)
        restrict the episode dates sampled.
    n_envs:
        Number of parallel episode replays per step call.
    reward_cfg:
        Optional :class:`RewardConfig` override (defaults are the contract
        weights). Kept out of EnvConfig because Layer-4 evolution mutates
        reward weights independently of market/execution parameters.
    min_episode_bars:
        Sessions with fewer usable bars than this are not indexed as
        episodes (default 30 — enough to reject degenerate data holes
        while keeping half days and truncated test sessions replayable).

    Observation semantics (contract keys, exact shapes; all float32):

    * ``market [D]``   fused embedding at bar t
    * ``unc [3]``      aleatoric, epistemic, anomaly
    * ``position [5]`` signed position notional / equity;
      (px − entry)/entry; unrealized PnL / equity₀; bars-in-position / 390;
      signed stop distance ``dir·(px − stop)/px`` (positive while the stop
      is on its protective side). All zero when flat.
    * ``portfolio [3]`` cash/equity; (equity − equity₀)/equity₀;
      (peak − equity)/equity₀ (≥ 0).
    * ``clock [2]``    session_minute/390; bars-to-episode-end / 390.
    * ``meta [3]``     one-hot of the intent currently in force.
    * ``flags [1]``    1.0 on meta-decision bars (t % meta_every == 0).

    A finished sub-env AUTO-RESETS within the same step call: ``done=True``
    is reported exactly once at the boundary, and the returned obs already
    belongs to a freshly sampled episode (module docstring, "Auto-reset").
    """

    def __init__(self, cfg: EnvConfig, n_envs: int,
                 reward_cfg: Optional[RewardConfig] = None,
                 min_episode_bars: int = MIN_EPISODE_BARS) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if cfg.initial_equity <= 0:
            raise ValueError("cfg.initial_equity must be positive")
        self.cfg = cfg
        self.n_envs = int(n_envs)
        self._fee = cfg.fees_bps * 1e-4
        self._slip = cfg.slippage_bps * 1e-4
        self._eq0 = float(cfg.initial_equity)
        self._shaper = RewardShaper(reward_cfg or RewardConfig(), self._eq0)

        self._episodes = self._index_episodes(cfg, int(min_episode_bars))
        if not self._episodes:
            raise RuntimeError(
                f"TradingEnv: no episodes with >= {min_episode_bars} bars "
                f"under {cfg.embeddings_dir} for tickers={cfg.tickers or 'all'}"
                f" in [{cfg.start or '-inf'}, {cfg.end or '+inf'}]")
        self._embed_dim = int(self._episodes[0].fused.shape[1])

        self._rng = np.random.default_rng(cfg.seed)
        n = self.n_envs
        self._ep: list[_Episode] = [self._episodes[0]] * n  # placeholders
        self._t = np.zeros(n, dtype=np.int64)
        self._cash = np.full(n, self._eq0, dtype=np.float64)
        self._qty = np.zeros(n, dtype=np.float64)            # signed shares
        self._entry_px = np.zeros(n, dtype=np.float64)
        self._stop_px = np.zeros(n, dtype=np.float64)
        self._target_px = np.zeros(n, dtype=np.float64)
        self._entry_fee = np.zeros(n, dtype=np.float64)
        self._entry_ts = np.zeros(n, dtype=np.int64)
        self._entry_bar = np.zeros(n, dtype=np.int64)
        self._entry_intent = np.zeros(n, dtype=np.int64)
        self._entry_size = np.zeros(n, dtype=np.float64)
        self._bars_in_pos = np.zeros(n, dtype=np.int64)
        self._peak = np.full(n, self._eq0, dtype=np.float64)
        self._meta = np.full(n, _META_STAND_ASIDE, dtype=np.int64)
        self._trades: list[list[TradeRecord]] = [[] for _ in range(n)]
        self._was_reset = False

        logger.info("TradingEnv: %d episodes across %d tickers, D=%d, "
                    "n_envs=%d", len(self._episodes),
                    len({e.ticker for e in self._episodes}),
                    self._embed_dim, n)

    # ------------------------------------------------------------------ #
    # Episode indexing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _index_episodes(cfg: EnvConfig, min_bars: int) -> list[_Episode]:
        emb_dir = Path(cfg.embeddings_dir)
        tickers = list(cfg.tickers) if cfg.tickers else sorted(
            p.stem for p in emb_dir.glob("*.npz"))
        if not tickers:
            raise RuntimeError(
                f"TradingEnv: no .npz embedding files under {emb_dir}")

        episodes: list[_Episode] = []
        dim: Optional[int] = None
        for alias in tickers:
            data = EmbeddingStore.load(str(emb_dir), alias)
            fused = data["fused"].astype(np.float32, copy=False)
            if dim is None:
                dim = int(fused.shape[1])
            elif int(fused.shape[1]) != dim:
                raise ValueError(
                    f"TradingEnv: {alias} embed dim {fused.shape[1]} != "
                    f"{dim} of earlier tickers — mixed embedding stores")
            anchor_ts = data["anchor_ts"].astype(np.int64, copy=False)
            if anchor_ts.size and not np.all(np.diff(anchor_ts) > 0):
                raise ValueError(
                    f"TradingEnv: {alias} anchor_ts not strictly increasing")
            unc = np.stack([data["aleatoric"], data["epistemic"],
                            data["anomaly"]], axis=1).astype(np.float32)
            day = anchor_ts // _SECONDS_PER_DAY
            nh = data["next_high"].astype(np.float64, copy=False)
            nl = data["next_low"].astype(np.float64, copy=False)
            for d_val in np.unique(day):
                idx = np.nonzero(day == d_val)[0]
                if idx.size < min_bars:
                    continue
                date = str(np.datetime64(int(d_val), "D"))
                if cfg.start and date < cfg.start:
                    continue
                if cfg.end and date > cfg.end:
                    continue
                s = slice(int(idx[0]), int(idx[-1]) + 1)
                episodes.append(_Episode(
                    ticker=alias, date=date,
                    fused=fused[s], unc=unc[s],
                    close=data["close_px"].astype(np.float64, copy=False)[s],
                    next_open=data["next_open"].astype(np.float64,
                                                       copy=False)[s],
                    next_high=nh[s], next_low=nl[s],
                    bar_range=(nh - nl)[s],
                    minute=data["session_minute"].astype(np.int64,
                                                         copy=False)[s],
                    ts=anchor_ts[s],
                ))
        return episodes

    @property
    def episodes(self) -> list[tuple[str, str]]:
        """Every (ticker, session-date) this env can sample."""
        return [(e.ticker, e.date) for e in self._episodes]

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, episode_ids: Optional[list] = None
              ) -> dict[str, np.ndarray]:
        """Start ``n_envs`` fresh episodes; returns the initial obs dict.

        ``episode_ids`` restricts which episodes may run. Entries are either
        integer indices into :attr:`episodes` or ``(ticker, "YYYY-MM-DD")``
        tuples (the curriculum sampler's currency — resolved here so callers
        never depend on index order). Exactly ``n_envs`` entries pin
        episodes deterministically; a LARGER pool is sampled from with the
        env's seeded RNG — this is how a curriculum stage exposes its whole
        eligible set. ``None`` samples from all episodes.
        """
        m = len(self._episodes)
        if episode_ids is None:
            ids = self._rng.choice(m, size=self.n_envs,
                                   replace=self.n_envs > m)
        else:
            index = getattr(self, "_episode_index", None)
            if index is None:
                index = {(e.ticker, str(e.date)): i
                         for i, e in enumerate(self._episodes)}
                self._episode_index = index
            pool: list[int] = []
            for e in episode_ids:
                if isinstance(e, (tuple, list)) and len(e) == 2:
                    key = (str(e[0]), str(e[1]))
                    if key not in index:
                        raise ValueError(f"reset: unknown episode {key!r}")
                    pool.append(index[key])
                else:
                    pool.append(int(e))
            ids = np.asarray(pool, dtype=np.int64)
            if ids.size == 0:
                raise ValueError("reset: episode_ids is empty")
            if int(ids.min()) < 0 or int(ids.max()) >= m:
                raise ValueError(
                    f"reset: episode ids must lie in [0, {m}), got {ids}")
            if ids.shape != (self.n_envs,):
                # A pool, not a pinning: sample n_envs episodes from it.
                ids = self._rng.choice(ids, size=self.n_envs,
                                       replace=self.n_envs > ids.size)
        for i, e in enumerate(ids):
            self._reset_one(i, int(e))
        self._was_reset = True
        return self._obs()

    def _reset_one(self, i: int, episode_idx: int) -> None:
        """Reset env ``i`` onto episode ``episode_idx`` (fresh state)."""
        self._ep[i] = self._episodes[episode_idx]
        self._t[i] = 0
        self._cash[i] = self._eq0
        self._qty[i] = 0.0
        self._entry_px[i] = 0.0
        self._stop_px[i] = 0.0
        self._target_px[i] = 0.0
        self._entry_fee[i] = 0.0
        self._entry_ts[i] = 0
        self._entry_bar[i] = 0
        self._entry_intent[i] = _META_STAND_ASIDE
        self._entry_size[i] = 0.0
        self._bars_in_pos[i] = 0
        self._peak[i] = self._eq0
        self._meta[i] = _META_STAND_ASIDE
        self._trades[i] = []

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, actions: dict[str, np.ndarray]) -> EnvStep:
        """Advance every env one bar. See the module docstring sequence."""
        if not self._was_reset:
            raise RuntimeError("TradingEnv.step called before reset()")
        acts = self._validate_actions(actions)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.n_envs)]
        m = len(self._episodes)
        for i in range(self.n_envs):
            rewards[i], done = self._step_env(
                i, {key: acts[key][i] for key in ACTION_KEYS}, infos[i])
            if done:
                dones[i] = True
                # SAME-STEP AUTO-RESET (see module docstring): the returned
                # obs for this env is the first bar of a freshly sampled
                # episode; done=True marks the boundary exactly once.
                self._reset_one(i, int(self._rng.integers(m)))
        return EnvStep(obs=self._obs(), reward=rewards,
                       done=dones, info=infos)

    def _step_env(self, i: int, a: dict[str, Any],
                  info: dict) -> tuple[float, bool]:
        """One env's transition t -> t+1: (shaped reward, episode done)."""
        ep = self._ep[i]
        t = int(self._t[i])
        close_t = float(ep.close[t])
        prev_equity = self._cash[i] + self._qty[i] * close_t
        prev_peak = float(self._peak[i])
        opened = False
        entry_fee = 0.0
        intent = int(self._meta[i])  # the intent obs['meta'] showed at t

        trade = int(a["trade"])
        nxt_o = float(ep.next_open[t])
        nxt_h = float(ep.next_high[t])
        nxt_l = float(ep.next_low[t])
        fill_ts = int(ep.ts[t + 1])

        # (1) entry -------------------------------------------------------- #
        if (trade == _TRADE_ENTER and self._qty[i] == 0.0
                and intent != _META_STAND_ASIDE and math.isfinite(nxt_o)):
            d = 1.0 if intent == _META_HUNT_LONG else -1.0
            fill_px = nxt_o * (1.0 + d * self._slip)   # against the agent
            qty_abs = (float(a["size"]) * self.cfg.max_position_frac
                       * prev_equity / fill_px)
            if qty_abs > 0.0:
                fee = self._fee * qty_abs * fill_px
                self._cash[i] -= d * qty_abs * fill_px
                self._cash[i] -= fee
                rng_t = self._realized_range(ep, t)
                self._qty[i] = d * qty_abs
                self._entry_px[i] = fill_px
                self._stop_px[i] = fill_px - d * float(a["stop"]) * rng_t
                self._target_px[i] = fill_px + d * float(a["target"]) * rng_t
                self._entry_fee[i] = fee
                self._entry_ts[i] = fill_ts
                self._entry_bar[i] = t + 1
                self._entry_intent[i] = intent
                self._entry_size[i] = float(a["size"])
                self._bars_in_pos[i] = 0
                opened = True
                entry_fee = fee
                info["entry_fill"] = {"px": fill_px, "qty": self._qty[i],
                                      "bar": t + 1, "ts": fill_ts}

        # (2) stop/target sweep of the fill bar ---------------------------- #
        if self._qty[i] != 0.0:
            d = 1.0 if self._qty[i] > 0.0 else -1.0
            if d > 0.0:
                stop_hit = nxt_l <= self._stop_px[i]
                target_hit = nxt_h >= self._target_px[i]
            else:
                stop_hit = nxt_h >= self._stop_px[i]
                target_hit = nxt_l <= self._target_px[i]
            if stop_hit:      # stop-before-target: worst case wins
                px = self._stop_px[i] * (1.0 - d * self._slip)  # market order
                self._close_position(i, px, fill_ts, t + 1, "stop", info)
            elif target_hit:  # resting limit: fills AT the level, fees only
                self._close_position(i, float(self._target_px[i]), fill_ts,
                                     t + 1, "target", info)

        # (3) policy exit --------------------------------------------------- #
        if self._qty[i] != 0.0 and trade == _TRADE_EXIT:
            d = 1.0 if self._qty[i] > 0.0 else -1.0
            px = nxt_o * (1.0 - d * self._slip)
            self._close_position(i, px, fill_ts, t + 1, "policy_exit", info)

        # (4) adjust: re-derive stop/target; stop tightens only ------------- #
        if self._qty[i] != 0.0 and trade == _TRADE_ADJUST:
            d = 1.0 if self._qty[i] > 0.0 else -1.0
            rng_t = self._realized_range(ep, t)
            cand_stop = close_t - d * float(a["stop"]) * rng_t
            if d > 0.0:
                self._stop_px[i] = max(self._stop_px[i], cand_stop)
            else:
                self._stop_px[i] = min(self._stop_px[i], cand_stop)
            self._target_px[i] = close_t + d * float(a["target"]) * rng_t

        # (5) meta re-decision on meta bars only ----------------------------- #
        if t % self.cfg.meta_every == 0:
            self._meta[i] = int(a["meta"])  # position deliberately untouched

        # (6) advance; force-close at session end ---------------------------- #
        t2 = t + 1
        self._t[i] = t2
        if self._qty[i] != 0.0:
            self._bars_in_pos[i] += 1
        done = t2 >= ep.length - 1
        if done and self._qty[i] != 0.0:
            d = 1.0 if self._qty[i] > 0.0 else -1.0
            px = float(ep.close[t2]) * (1.0 - d * self._slip)
            self._close_position(i, px, int(ep.ts[t2]), t2, "eod", info)

        equity = self._cash[i] + self._qty[i] * float(ep.close[t2])
        self._peak[i] = max(self._peak[i], equity)
        reward = self._shaper.step_reward(prev_equity, equity, prev_peak,
                                          float(self._peak[i]), opened,
                                          entry_fee)
        if done:
            # Hindsight capture bonus: terminal only, never in observations.
            reward += self._shaper.terminal_bonus(self._trades[i], ep.close)
        return float(reward), bool(done)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _close_position(self, i: int, px: float, ts: int, bar: int,
                        reason: str, info: dict) -> None:
        """Close env i's position at ``px``; emit the TradeRecord."""
        qty = float(self._qty[i])
        qty_abs = abs(qty)
        fee = self._fee * qty_abs * px
        self._cash[i] += qty * px
        self._cash[i] -= fee
        fees_total = float(self._entry_fee[i]) + fee
        pnl = qty * (px - float(self._entry_px[i])) - fees_total
        ep = self._ep[i]
        record = TradeRecord(
            trade_id=(f"{ep.ticker}-{ep.date}-e{i}"
                      f"-n{len(self._trades[i])}"),
            ticker=ep.ticker,
            side="long" if qty > 0.0 else "short",
            entry_ts=int(self._entry_ts[i]),
            exit_ts=int(ts),
            entry_px=float(self._entry_px[i]),
            exit_px=float(px),
            qty=qty_abs,
            pnl=float(pnl),
            fees=float(fees_total),
            stop_px=float(self._stop_px[i]),
            target_px=float(self._target_px[i]),
            exit_reason=reason,
            signal_meta={
                # entry_bar feeds RewardShaper.terminal_bonus (episode-local
                # index of the entry fill); the rest is autopsy context.
                "entry_bar": int(self._entry_bar[i]),
                "exit_bar": int(bar),
                "intent": META_ACTIONS[int(self._entry_intent[i])],
                "size_frac": float(self._entry_size[i]),
            },
        )
        self._trades[i].append(record)
        info["trade_closed"] = record
        self._qty[i] = 0.0
        self._entry_px[i] = 0.0
        self._stop_px[i] = 0.0
        self._target_px[i] = 0.0
        self._entry_fee[i] = 0.0
        self._bars_in_pos[i] = 0

    @staticmethod
    def _realized_range(ep: _Episode, t: int) -> float:
        """Mean high−low of the last ≤15 bars known at t (bars ≤ t only).

        ``bar_range[j]`` holds bar ``j+1``'s range, so the bars ``t−14..t``
        live at indices ``t−15..t−1`` — the window never touches index t
        (bar t+1: the future). At t=0 no completed bar range is observable
        within the episode yet; a conservative 10 bps-of-price fallback
        applies (also used if the recorded ranges are degenerate).
        """
        lo = max(0, t - RANGE_WINDOW_BARS)
        window = ep.bar_range[lo:t]
        if window.size:
            mean = float(np.mean(window))
            if math.isfinite(mean) and mean > 0.0:
                return mean
        return float(ep.close[t]) * _RANGE_FALLBACK_FRAC

    def _validate_actions(self, actions: dict[str, np.ndarray]
                          ) -> dict[str, np.ndarray]:
        missing = [k for k in ACTION_KEYS if k not in actions]
        if missing:
            raise ValueError(f"step: actions missing keys {missing}")
        out: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            arr = np.asarray(actions[key]).reshape(-1)
            if arr.shape[0] != self.n_envs:
                raise ValueError(
                    f"step: actions[{key!r}] has length {arr.shape[0]}, "
                    f"expected n_envs={self.n_envs}")
            out[key] = arr
        meta = out["meta"].astype(np.int64)
        trade = out["trade"].astype(np.int64)
        if ((meta < 0) | (meta >= len(META_ACTIONS))).any():
            raise ValueError(f"step: meta actions out of range: {meta}")
        if ((trade < 0) | (trade >= len(TRADE_ACTIONS))).any():
            raise ValueError(f"step: trade actions out of range: {trade}")
        out["meta"], out["trade"] = meta, trade
        for key in ("size", "stop", "target"):
            arr = out[key].astype(np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"step: actions[{key!r}] contains non-"
                                 "finite values")
            # Contract support is (0, 1] (Beta-head outputs); the clip is
            # belt-and-braces against numerical edge cases, not a semantic.
            out[key] = np.clip(arr, _FRAC_FLOOR, 1.0)
        return out

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _obs(self) -> dict[str, np.ndarray]:
        """Contract-exact observation dict, everything derived from ≤ t."""
        n = self.n_envs
        market = np.zeros((n, self._embed_dim), dtype=np.float32)
        unc = np.zeros((n, 3), dtype=np.float32)
        position = np.zeros((n, 5), dtype=np.float32)
        portfolio = np.zeros((n, 3), dtype=np.float32)
        clock = np.zeros((n, 2), dtype=np.float32)
        meta = np.zeros((n, len(META_ACTIONS)), dtype=np.float32)
        flags = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            ep = self._ep[i]
            t = int(self._t[i])
            market[i] = ep.fused[t]
            unc[i] = ep.unc[t]
            px = float(ep.close[t])
            qty = float(self._qty[i])
            equity = self._cash[i] + qty * px
            eq_safe = max(equity, 1e-9)  # guard: sizing cannot bankrupt, but
            if qty != 0.0:               # never divide by ~0 regardless
                d = 1.0 if qty > 0.0 else -1.0
                position[i, 0] = qty * px / eq_safe
                position[i, 1] = (px - self._entry_px[i]) / self._entry_px[i]
                position[i, 2] = qty * (px - self._entry_px[i]) / self._eq0
                position[i, 3] = self._bars_in_pos[i] / MINUTES_PER_SESSION
                position[i, 4] = d * (px - self._stop_px[i]) / px
            portfolio[i, 0] = self._cash[i] / eq_safe
            portfolio[i, 1] = (equity - self._eq0) / self._eq0
            portfolio[i, 2] = max(0.0, self._peak[i] - equity) / self._eq0
            clock[i, 0] = float(ep.minute[t]) / MINUTES_PER_SESSION
            clock[i, 1] = float(ep.length - 1 - t) / MINUTES_PER_SESSION
            meta[i, int(self._meta[i])] = 1.0
            flags[i, 0] = 1.0 if t % self.cfg.meta_every == 0 else 0.0
        obs = {"market": market, "unc": unc, "position": position,
               "portfolio": portfolio, "clock": clock, "meta": meta,
               "flags": flags}
        assert tuple(obs) == OBS_KEYS  # contract order, cheap to keep honest
        return obs
