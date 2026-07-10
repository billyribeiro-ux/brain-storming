"""Event-driven backtesting for Layer 5 — two engines, one accounting core.

* :meth:`Backtester.run_signals` — bar-by-bar replay of the SignalEngine →
  RiskManager → ExecutionTactics stack over precomputed embedding npz files,
  with portfolio-level accounting across tickers.
* :meth:`Backtester.run_policy` — drives ``aether.decision.env.TradingEnv``
  with a policy's greedy actions and collects the trades it closes.

The one sacred invariant applies with full force: at bar ``t`` the strategy
sees only data ``≤ t`` (the anchor's embedding and close), and every order
fills on bar ``t+1`` (the npz's ``next_*`` arrays) under conservative
assumptions — stop-before-target when both are touched inside one bar,
slippage always against the agent, limit fills only when the bar range
trades through the price, never with improvement.

Hard rules enforced here:

* **Accounting identity.** ``cash + Σ qty·mark == equity`` is asserted every
  bar against an independently accumulated ``initial + realized + unrealized
  − open-entry-fees`` when ``debug=True`` (the default).
* **No NaN propagation.** Bars whose ``next_*`` values are not finite are
  skipped for execution (no entries, no exit checks); marks fall back to the
  last finite close.

Sibling modules (``aether.execution.signals``/``risk``,
``aether.worldmodel.autopsy``, ``aether.decision.env``) are developed
concurrently and are therefore imported lazily inside functions; everything
here codes strictly against the published contracts.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aether.decision.interfaces import ACTION_KEYS
from aether.execution.interfaces import (
    BacktestConfig,
    BacktestResult,
    ReversalSignal,
    RiskDecision,
)
from aether.execution.tactics import ExecutionTactics, spread_proxy
from aether.utils.logging import get_logger
from aether.worldmodel.interfaces import TradeRecord

logger = get_logger("aether.backtest")

#: Per-bar → annual scaling for Sharpe/Sortino (390 bars/session, 252 sessions).
ANNUALIZATION: float = math.sqrt(390 * 252)

#: npz keys every embedding file must carry (decision-layer contract).
REQUIRED_EMB_KEYS: tuple[str, ...] = (
    "fused", "aleatoric", "epistemic", "anomaly", "anchor_ts",
    "session_minute", "close_px", "next_open", "next_high", "next_low",
)

_SECONDS_PER_DAY = 86_400


# --------------------------------------------------------------------------- #
# Embedding loading (shared with the paper trader)
# --------------------------------------------------------------------------- #

def load_embeddings(embeddings_dir: str | Path, ticker: str) -> dict[str, np.ndarray]:
    """Load one ticker's precomputed embedding arrays.

    Prefers ``aether.decision.env.EmbeddingStore.load`` (the owning module,
    developed concurrently); falls back to reading ``<dir>/<ticker>.npz``
    directly — the same npz contract either way. ``next_close`` is part of
    the Layer-5 contract but the decision-layer docstring only promises next
    open/high/low, so a missing ``next_close`` is synthesized from the next
    contiguous anchor's ``close_px`` (NaN across gaps). Arrays are returned
    sorted by ``anchor_ts``.
    """
    embeddings_dir = Path(embeddings_dir)
    arrs: dict[str, np.ndarray] | None = None
    try:  # concurrent module — may not exist yet
        from aether.decision.env import EmbeddingStore
        arrs = {k: np.asarray(v)
                for k, v in dict(EmbeddingStore.load(str(embeddings_dir), ticker)).items()}
    except Exception:
        arrs = None
    if arrs is None:
        path = embeddings_dir / f"{ticker}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"no embeddings for {ticker!r}: {path} (run the decision "
                f"layer's EmbeddingStore.precompute first)")
        with np.load(str(path)) as z:
            arrs = {k: np.asarray(z[k]) for k in z.files}

    missing = [k for k in REQUIRED_EMB_KEYS if k not in arrs]
    if missing:
        raise KeyError(f"embeddings for {ticker!r} lack required arrays {missing}")

    order = np.argsort(arrs["anchor_ts"], kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        arrs = _take(arrs, order)

    if "next_close" not in arrs:
        ats = arrs["anchor_ts"].astype(np.int64)
        close = arrs["close_px"].astype(float)
        nxt = np.full(ats.shape, np.nan)
        if ats.size > 1:
            contiguous = (ats[1:] - ats[:-1]) == 60
            nxt[:-1][contiguous] = close[1:][contiguous]
        arrs["next_close"] = nxt
        logger.warning("embeddings for %s carry no next_close — synthesized "
                       "from contiguous anchors (NaN across gaps)", ticker)
    return arrs


def _take(arrs: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Row-select every per-anchor array (leaves scalars/mismatched keys as-is)."""
    n = int(arrs["anchor_ts"].shape[0])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
            for k, v in arrs.items()}


# --------------------------------------------------------------------------- #
# Position lifecycle (shared with the paper trader)
# --------------------------------------------------------------------------- #

def manage_position(pos: dict, next_open: float, next_high: float,
                    next_low: float, next_close: float, session_last: bool,
                    fees_frac: float, slip_frac: float, exit_ts: int,
                    entry_bar: bool = False,
                    allow_target: bool = True) -> TradeRecord | None:
    """Evaluate one position against one execution bar; close it if due.

    ``session_last`` means the EXECUTION bar being evaluated is its
    session's final tradable bar (Aether never holds overnight).

    Event order inside the bar (deliberately conservative):

    1. horizon exit — market at the bar's open (only counted on bars after
       the entry bar),
    2. stop — a stop is a *market* order: fill at the stop price or at the
       open when the bar gaps through it (whichever is worse), slippage
       against the agent,
    3. target — a *limit* order: fills at exactly the target price iff the
       bar range trades through it; stop is checked FIRST, so a bar touching
       both resolves as a stop (stop-before-target). On a MAKER (limit)
       entry's own fill bar the caller passes ``allow_target=False``: the
       bar's favorable extreme may have printed BEFORE the limit entry even
       filled, so crediting a same-bar target would be optimistic. The
       same-bar STOP stays enabled — a deliberately conservative asymmetry,
    4. end of session — force close at the bar's close ("eod").

    Bars with any non-finite OHLC are skipped for exit checks (no bar
    counted — NaN never propagates into fills), EXCEPT that a session-last
    force close must still fire: a position must never survive a session
    boundary just because the final bar's data is holed. In that case the
    close uses the first finite price the bar offers (close preferred);
    only a bar with no finite price at all returns ``None`` and leaves the
    caller's stranded-position sweep as the last resort. ``pos`` is mutated
    (``bars_held``); a :class:`TradeRecord` is returned when the position
    closes, else ``None``. Cash effects are the caller's job.
    """
    o, h, l, c = (float(next_open), float(next_high),
                  float(next_low), float(next_close))
    if not all(math.isfinite(v) for v in (o, h, l, c)):
        # eod BEFORE the NaN early-return: the no-overnight rule outranks
        # the no-NaN-fills rule; close at the best finite price available.
        if session_last:
            for px in (c, o, h, l):
                if math.isfinite(px):
                    return close_position(pos, px, "eod", exit_ts, fees_frac,
                                          slip_frac, detail="nan_bar")
        return None
    d = 1.0 if pos["side"] == "long" else -1.0

    if not entry_bar:
        pos["bars_held"] = int(pos["bars_held"]) + 1
        if pos["bars_held"] >= int(pos["horizon"]):
            return close_position(pos, o, "policy_exit", exit_ts, fees_frac,
                                  slip_frac, detail="horizon")

    stop, target = float(pos["stop_px"]), float(pos["target_px"])
    if (d > 0 and l <= stop) or (d < 0 and h >= stop):
        raw = min(stop, o) if d > 0 else max(stop, o)  # gap through = worse fill
        return close_position(pos, raw, "stop", exit_ts, fees_frac, slip_frac)
    if allow_target and ((d > 0 and h >= target) or (d < 0 and l <= target)):
        return close_position(pos, target, "target", exit_ts, fees_frac,
                              slip_frac, is_limit=True)
    if session_last:
        return close_position(pos, c, "eod", exit_ts, fees_frac, slip_frac)
    return None


def close_position(pos: dict, raw_px: float, reason: str, exit_ts: int,
                   fees_frac: float, slip_frac: float, is_limit: bool = False,
                   detail: str | None = None) -> TradeRecord:
    """Build the :class:`TradeRecord` for a close at ``raw_px``.

    Market closes pay slippage against the agent (long sells lower, short
    covers higher); limit closes (targets) fill at the exact price.
    """
    d = 1.0 if pos["side"] == "long" else -1.0
    px = float(raw_px) if is_limit else float(raw_px) * (1.0 - d * slip_frac)
    q = float(pos["q"])
    exit_fee = abs(q) * px * fees_frac
    entry_fee = float(pos["entry_fee"])
    pnl = q * (px - float(pos["entry_px"])) - entry_fee - exit_fee
    meta = dict(pos.get("meta") or {})
    if detail:
        meta["exit_detail"] = detail
    return TradeRecord(
        trade_id=str(pos["trade_id"]),
        ticker=str(pos["ticker"]),
        side=str(pos["side"]),
        entry_ts=int(pos["entry_ts"]),
        exit_ts=int(exit_ts),
        entry_px=float(pos["entry_px"]),
        exit_px=px,
        qty=abs(q),
        pnl=pnl,
        fees=entry_fee + exit_fee,
        stop_px=float(pos["stop_px"]),
        target_px=float(pos["target_px"]),
        exit_reason=reason,
        conviction=float(pos.get("conviction", 0.0)),
        signal_meta=meta,
    )


# --------------------------------------------------------------------------- #
# Fallback risk
# --------------------------------------------------------------------------- #

class PassThroughRisk:
    """Honest fallback when ``aether.execution.risk`` is not importable yet.

    Approves every signal at its own suggested size (``size_frac`` of
    equity) with the signal's stop/target unchanged, and says so in the
    reasons — it never pretends to be a risk check. Used only when no real
    RiskManager is available; production runs should always inject one.
    """

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    def assess(self, signal: ReversalSignal, equity: float,
               open_positions: dict[str, float], day_pnl_frac: float,
               corr=None) -> RiskDecision:
        px = max(float(signal.entry_px), 1e-9)
        qty = max(0.0, float(signal.size_frac) * float(equity) / px)
        return RiskDecision(
            approved=qty > 0,
            qty=qty,
            reasons=["pass-through sizing: aether.execution.risk unavailable "
                     "— NO risk rails were applied"],
            adjusted_stop_px=float(signal.stop_px),
            adjusted_target_px=float(signal.target_px),
        )

    def reset_day(self) -> None:  # protocol parity
        return None


# --------------------------------------------------------------------------- #
# Stats & persistence (shared helpers)
# --------------------------------------------------------------------------- #

def _trade_stats(trades: list[TradeRecord]) -> dict:
    pnls = [float(t.pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    return {
        "n_trades": len(pnls),
        "hit_rate": (len(wins) / len(pnls)) if pnls else 0.0,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": profit_factor,
        "total_pnl": sum(pnls),
    }


def compute_stats(equity_curve: pd.DataFrame,
                  trades: list[TradeRecord]) -> dict:
    """Shared stats for both engines.

    Sharpe/Sortino are annualized from per-bar equity returns with
    ``sqrt(390·252)``; Sortino uses the root-mean-square of negative returns
    (target 0). ``max_dd`` is the worst peak-to-trough equity fraction.
    Trade-level stats are repeated per ticker under ``per_ticker``.
    """
    sharpe = sortino = max_dd = 0.0
    n_bars = 0
    if equity_curve is not None and len(equity_curve) >= 2:
        eq = np.asarray(equity_curve["equity"], dtype=float)
        n_bars = int(eq.size)
        prev = eq[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
        rets = rets[np.isfinite(rets)]
        if rets.size:
            mu = float(rets.mean())
            sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
            sharpe = mu / sd * ANNUALIZATION if sd > 0 else 0.0
            downside = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))
            sortino = mu / downside * ANNUALIZATION if downside > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, 1.0 - eq / peak, 0.0)
        max_dd = float(np.max(dd)) if dd.size else 0.0

    stats = {"sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
             "n_bars": n_bars}
    stats.update(_trade_stats(trades))
    per_ticker: dict[str, dict] = {}
    for t in trades:
        per_ticker.setdefault(t.ticker, []).append(t)  # type: ignore[arg-type]
    stats["per_ticker"] = {tkr: _trade_stats(ts) for tkr, ts in per_ticker.items()}
    return stats


def to_jsonable(obj):
    """Recursively convert to strict-JSON-safe values.

    Dataclasses become dicts, numpy scalars/arrays become Python values, and
    non-finite floats become ``None`` (strict JSON has no NaN/Infinity).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def save_result(result: BacktestResult, data_root: str | Path,
                name: str) -> Path:
    """Persist a result to ``<data_root>/backtests/<name>/``.

    * ``result.json`` — stats + signals + trades + autopsies as plain dicts,
    * ``equity.parquet`` — the per-bar equity curve ``[ts, equity]``.
    """
    out = Path(data_root) / "backtests" / str(name)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": str(name),
        "stats": to_jsonable(result.stats),
        "signals": [to_jsonable(s) for s in result.signals],
        "trades": [to_jsonable(t) for t in result.trades],
        "autopsies": [to_jsonable(a) for a in result.autopsies],
    }
    (out / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    curve = result.equity_curve
    if not isinstance(curve, pd.DataFrame):
        curve = pd.DataFrame(list(curve or []), columns=["ts", "equity"])
    curve.to_parquet(out / "equity.parquet", index=False)
    logger.info("saved backtest %r: %d trades, %d signals -> %s",
                name, len(result.trades), len(result.signals), out)
    return out


def _bind_kwargs(fn, candidates: dict, what: str) -> dict:
    """House-style adaptive binding: match ``fn``'s parameters by name.

    Concurrently-developed modules own their exact signatures, so callers
    supply a synonym table and every declared parameter is looked up in it.
    A *required* parameter with no known value raises with an actionable
    message — never a silent mis-bind.
    """
    kwargs: dict = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is param.empty:
            raise TypeError(
                f"{what} requires parameter {name!r} which the caller does "
                f"not know how to supply; known: {sorted(candidates)}")
    return kwargs


# --------------------------------------------------------------------------- #
# The backtester
# --------------------------------------------------------------------------- #

class Backtester:
    """Event-driven replay over precomputed embeddings.

    Parameters
    ----------
    cfg:
        :class:`~aether.execution.interfaces.BacktestConfig`.
    embeddings_dir:
        Directory of per-ticker ``.npz`` embedding files (decision-layer
        contract).
    signal_engine:
        Object with ``generate(ticker, t, arrays) -> ReversalSignal | None``
        where ``arrays`` is the ticker's full npz dict and ``t`` the anchor
        index — the engine must only look at rows ``≤ t``. Required for
        :meth:`run_signals`.
    risk:
        A ``RiskManagerProtocol`` implementation. When ``None``, the real
        ``aether.execution.risk.RiskManager`` is imported lazily if it
        exists; otherwise :class:`PassThroughRisk` (loudly documented in
        every decision's reasons).
    autopsist:
        Optional autopsy engine; with ``cfg.autopsy_losses`` every losing
        trade gets its session's 1-min bars loaded from the lake and a
        report produced + saved via ``aether.worldmodel.autopsy``.
    tactics:
        :class:`~aether.execution.tactics.ExecutionTactics`; taker default.
    data_root:
        Lake root for autopsy bar loading (default: ``DataConfig.from_env()``).
    debug:
        Assert the portfolio accounting identity every bar (default True).
    """

    def __init__(self, cfg: BacktestConfig, embeddings_dir: str | Path,
                 signal_engine=None, risk=None, autopsist=None, tactics=None,
                 *, data_root: str | Path | None = None, debug: bool = True):
        self.cfg = cfg
        self.embeddings_dir = Path(embeddings_dir)
        self.signal_engine = signal_engine
        self.risk = risk
        self.autopsist = autopsist
        self.tactics = tactics if tactics is not None else ExecutionTactics(mode="taker")
        self.debug = bool(debug)
        if data_root is None:
            from aether.config import DataConfig  # env-dependent; keep lazy
            data_root = DataConfig.from_env().root
        self.data_root = Path(data_root)
        self._trade_seq = 0

    # ------------------------------------------------------------------ #
    # Engine 1: the signal stack
    # ------------------------------------------------------------------ #

    def run_signals(self) -> BacktestResult:
        """Replay SignalEngine → risk → tactics over ``cfg.tickers``.

        Per global bar (the sorted union of anchor timestamps across
        tickers, intersected with ``[start, end]``):

        1. mark the portfolio at each ticker's anchor close and record the
           equity row (this is the state a signal at bar ``t`` may see),
        2. manage open positions against bar ``t+1`` (``next_*`` arrays):
           horizon → stop → target → eod, conservative throughout,
        3. for flat tickers, ask the signal engine; risk-check with current
           equity/open fractions/day PnL; execute the entry via tactics on
           bar ``t+1``, then apply the same conservative exit checks to the
           entry bar itself (a stop can fire on the bar you entered).

        One position per ticker, max. Signals whose fill bar would be the
        session's last bar are skipped (the entry would be force-closed
        instantly at eod). A ticker whose position EXITED during step 2 (on
        fill bar ``t+1``) may not accept a new entry decided at ``t``: that
        entry would print at bar ``t+1``'s open, chronologically BEFORE the
        stop/target exit it depends on for the book to be flat — the
        decision at ``t`` was made while still positioned. Earliest
        re-entry is a signal decided at ``t+1`` (filling ``t+2``), exactly
        the ordering ``TradingEnv._step_env`` enforces.
        """
        if self.signal_engine is None:
            raise ValueError("run_signals() needs a signal_engine")
        risk = self.risk if self.risk is not None else self._default_risk()
        fees = float(self.cfg.fees_bps) / 1e4
        slip = float(self.cfg.slippage_bps) / 1e4

        data: dict[str, dict[str, np.ndarray]] = {}
        for ticker in self.cfg.tickers:
            arrs = self._load_ticker(ticker)
            if arrs is None:
                continue
            data[ticker] = arrs
        if not data:
            raise ValueError(
                f"no embeddings found for {self.cfg.tickers!r} in "
                f"[{self.cfg.start!r}, {self.cfg.end!r}] under {self.embeddings_dir}")

        timeline = np.unique(np.concatenate(
            [a["anchor_ts"].astype(np.int64) for a in data.values()]))
        ptr = {t: 0 for t in data}
        positions: dict[str, dict] = {}
        last_close: dict[str, float] = {}
        cash = float(self.cfg.initial_equity)
        realized = 0.0
        day = None
        day_start_equity = cash
        rows: list[tuple[int, float]] = []
        trades: list[TradeRecord] = []
        signals: list[ReversalSignal] = []

        for ts in timeline.tolist():
            ts = int(ts)
            active: dict[str, int] = {}
            for ticker, arrs in data.items():
                i = ptr[ticker]
                if i < arrs["anchor_ts"].shape[0] and int(arrs["anchor_ts"][i]) == ts:
                    active[ticker] = i
                    ptr[ticker] = i + 1
                    close = float(arrs["close_px"][i])
                    if math.isfinite(close):
                        last_close[ticker] = close

            # ---- 1) mark at bar-t close; identity check; day roll -------- #
            equity = cash + sum(p["q"] * last_close[t] for t, p in positions.items())
            d = ts // _SECONDS_PER_DAY
            if day is None or d != day:
                day = d
                day_start_equity = equity
                if hasattr(risk, "reset_day"):
                    try:
                        risk.reset_day()
                    except Exception as exc:  # never let bookkeeping kill a run
                        logger.warning("risk.reset_day failed: %s", exc)
            day_pnl_frac = (equity / day_start_equity - 1.0) if day_start_equity else 0.0
            rows.append((ts, float(equity)))
            if self.debug:
                expect = (self.cfg.initial_equity + realized
                          + sum(p["q"] * (last_close[t] - p["entry_px"]) - p["entry_fee"]
                                for t, p in positions.items()))
                assert abs(equity - expect) <= 1e-6 * max(1.0, abs(equity)), (
                    f"accounting identity broken at ts={ts}: "
                    f"equity={equity!r} expected={expect!r}")
            if equity <= 0:
                logger.error("equity non-positive at ts=%d — aborting replay", ts)
                break

            # ---- 2) manage open positions on bar t+1 --------------------- #
            exited_this_bar: set[str] = set()
            for ticker, i in active.items():
                pos = positions.get(ticker)
                if pos is None:
                    continue
                arrs = data[ticker]
                rec = manage_position(
                    pos, arrs["next_open"][i], arrs["next_high"][i],
                    arrs["next_low"][i], arrs["next_close"][i],
                    self._session_last(arrs, i), fees, slip,
                    exit_ts=int(arrs["anchor_ts"][i]) + 60)
                if rec is not None:
                    cash += pos["q"] * rec.exit_px - (rec.fees - pos["entry_fee"])
                    realized += rec.pnl
                    trades.append(rec)
                    del positions[ticker]
                    exited_this_bar.add(ticker)

            # ---- 3) new signals at bar t --------------------------------- #
            for ticker, i in active.items():
                if ticker in positions:
                    continue  # one position per ticker, max
                if ticker in exited_this_bar:
                    # The exit above happened ON fill bar t+1; an entry
                    # decided at t would print at t+1's open, BEFORE that
                    # exit — and the decision was made while the book was
                    # still positioned. No same-bar exit→re-entry; the
                    # earliest re-entry decides at t+1 and fills at t+2
                    # (mirrors TradingEnv._step_env's entry-before-sweep
                    # ordering).
                    continue
                arrs = data[ticker]
                if self._session_last(arrs, i):
                    continue  # fill bar would instantly eod-close
                nxt = (float(arrs["next_open"][i]), float(arrs["next_high"][i]),
                       float(arrs["next_low"][i]), float(arrs["next_close"][i]))
                if not all(math.isfinite(v) for v in nxt):
                    continue  # NaN never propagates into fills
                sig = self.signal_engine.generate(ticker, i, arrs)
                if sig is None:
                    continue
                signals.append(sig)
                open_fracs = ({t: p["q"] * last_close[t] / equity
                               for t, p in positions.items()} if equity > 0 else
                              {t: 0.0 for t in positions})
                decision = risk.assess(sig, equity, open_fracs, day_pnl_frac)
                if not decision.approved or decision.qty <= 0:
                    logger.info("signal %s rejected by risk: %s",
                                sig.signal_id, "; ".join(decision.reasons))
                    continue
                pos = self._enter(ticker, i, arrs, sig, decision, fees)
                if pos is None:
                    continue
                cash -= pos["q"] * pos["entry_px"] + pos["entry_fee"]
                positions[ticker] = pos
                # Conservative same-bar exit check on the entry bar. After a
                # MAKER (limit) entry fill the same-bar TARGET is not
                # credited: the bar's favorable extreme may predate the
                # fill itself; the same-bar stop stays active (documented
                # conservative asymmetry — see manage_position).
                rec = manage_position(
                    pos, *nxt, self._session_last(arrs, i), fees, slip,
                    exit_ts=int(arrs["anchor_ts"][i]) + 60, entry_bar=True,
                    allow_target=pos["meta"].get("order_type") != "limit")
                if rec is not None:
                    cash += pos["q"] * rec.exit_px - (rec.fees - pos["entry_fee"])
                    realized += rec.pnl
                    trades.append(rec)
                    del positions[ticker]

        # Positions stranded by trailing NaN bars / data end: close at the
        # last known mark so no equity is silently held open.
        for ticker, pos in list(positions.items()):
            px = last_close.get(ticker, pos["entry_px"])
            rec = close_position(pos, px, "eod", exit_ts=int(rows[-1][0]) + 60,
                                 fees_frac=fees, slip_frac=slip,
                                 detail="end_of_data")
            cash += pos["q"] * rec.exit_px - (rec.fees - pos["entry_fee"])
            realized += rec.pnl
            trades.append(rec)
            del positions[ticker]

        equity_curve = pd.DataFrame(rows, columns=["ts", "equity"])
        autopsies = self._run_autopsies(trades)
        stats = compute_stats(equity_curve, trades)
        logger.info("run_signals: %d bars, %d signals, %d trades, "
                    "final equity %.2f", len(rows), len(signals), len(trades),
                    rows[-1][1] if rows else float("nan"))
        return BacktestResult(equity_curve=equity_curve, trades=trades,
                              stats=stats, signals=signals, autopsies=autopsies)

    # ------------------------------------------------------------------ #
    # Engine 2: the RL policy
    # ------------------------------------------------------------------ #

    def run_policy(self, policy, chunk_size: int = 16,
                   max_episode_steps: int = 1200) -> BacktestResult:
        """Greedy-replay a policy through ``aether.decision.env.TradingEnv``.

        Episodes (the env's ``(ticker, date)`` list, already filtered by the
        env to ``[start, end]``) are batched sequentially in chunks of
        ``chunk_size`` — each chunk becomes one vectorized env of
        ``n_envs = len(chunk)``. Actions come from ``policy.mode(obs, carry)``
        when the policy exposes a greedy ``mode`` method (falling back to a
        no-arg ``mode()`` switch, then to ``act``). Trade records are
        collected from ``EnvStep.info['trade_closed']``; the ``signals``
        list is empty by design (the policy does not emit ReversalSignals).

        Equity aggregation is best-effort until the env lands: when infos
        carry per-step ``'equity'``, episode paths are chained sequentially
        (each episode's relative path compounds onto the running product);
        otherwise the curve is reconstructed from cumulative trade PnL at
        exit timestamps. Documented, never silently invented.
        """
        # Concurrent modules — lazy by design.
        from aether.decision.env import TradingEnv
        from aether.decision.interfaces import EnvConfig

        env_cfg = EnvConfig(
            embeddings_dir=str(self.embeddings_dir),
            tickers=tuple(self.cfg.tickers),
            start=self.cfg.start, end=self.cfg.end,
            fees_bps=self.cfg.fees_bps, slippage_bps=self.cfg.slippage_bps,
            initial_equity=self.cfg.initial_equity)
        probe = TradingEnv(env_cfg, n_envs=1)
        n_episodes = len(list(probe.episodes))
        if n_episodes == 0:
            raise ValueError("TradingEnv reports no episodes in range")

        trades: list[TradeRecord] = []
        episode_paths: list[list[float]] = []
        for lo in range(0, n_episodes, max(1, int(chunk_size))):
            ids = list(range(lo, min(lo + max(1, int(chunk_size)), n_episodes)))
            env = TradingEnv(env_cfg, n_envs=len(ids))
            obs = env.reset(ids)
            carry = (policy.initial_carry(len(ids))
                     if hasattr(policy, "initial_carry") else None)
            done = np.zeros(len(ids), dtype=bool)
            paths: list[list[float]] = [[] for _ in ids]
            steps = 0
            while not done.all() and steps < max_episode_steps:
                actions, carry = _greedy_actions(policy, obs, carry)
                step = env.step(actions)
                obs = step.obs
                for k, info in enumerate(step.info):
                    if done[k] or not isinstance(info, dict):
                        continue
                    closed = info.get("trade_closed")
                    for rec in (closed if isinstance(closed, (list, tuple))
                                else [closed]):
                        if rec is not None:
                            trades.append(rec)
                    eqv = info.get("equity")
                    if eqv is not None and math.isfinite(float(eqv)):
                        paths[k].append(float(eqv))
                done |= np.asarray(step.done, dtype=bool)
                steps += 1
            if steps >= max_episode_steps and not done.all():
                logger.warning("run_policy: chunk %s hit max_episode_steps=%d "
                               "before all episodes finished", ids, max_episode_steps)
            episode_paths.extend(paths)

        equity_curve, equity_source = self._aggregate_policy_equity(
            episode_paths, trades)
        autopsies = self._run_autopsies(trades)
        stats = compute_stats(equity_curve, trades)
        stats["equity_source"] = equity_source
        if equity_source != "env_step_equity":
            # The fallback curve holds one point per TRADE EXIT (or a flat
            # placeholder), not per bar: Sharpe/Sortino annualized with the
            # per-minute sqrt(390*252) factor would be silently meaningless
            # on it. Marked None rather than invented.
            stats["sharpe"] = None
            stats["sortino"] = None
        logger.info("run_policy: %d episodes, %d trades (equity_source=%s)",
                    n_episodes, len(trades), equity_source)
        return BacktestResult(equity_curve=equity_curve, trades=trades,
                              stats=stats, signals=[], autopsies=autopsies)

    def _aggregate_policy_equity(self, episode_paths: list[list[float]],
                                 trades: list[TradeRecord]
                                 ) -> tuple[pd.DataFrame, str]:
        """Chain per-episode equity paths into one curve (see run_policy doc).

        Returns ``(curve, source)`` where ``source`` labels how the curve
        was built — ``'env_step_equity'`` (per-bar, stats-grade),
        ``'trade_pnl_fallback'`` (one point per trade exit; per-bar stats
        are NOT valid on it) or ``'flat'`` (no data at all).
        """
        initial = float(self.cfg.initial_equity)
        rows: list[tuple[int, float]] = []
        if any(episode_paths):
            running = initial
            idx = 0
            for path in episode_paths:
                if not path:
                    continue
                for v in path:
                    rows.append((idx, running * (v / initial)))
                    idx += 1
                running *= path[-1] / initial
            source = "env_step_equity"
        elif trades:
            logger.warning("run_policy: env infos carried no 'equity' — "
                           "curve reconstructed from cumulative trade PnL "
                           "(stats['equity_source']='trade_pnl_fallback'; "
                           "sharpe/sortino are not computable per-bar and "
                           "are reported as None)")
            eq = initial
            rows.append((int(min(t.entry_ts for t in trades)), eq))
            for t in sorted(trades, key=lambda t: t.exit_ts):
                eq += float(t.pnl)
                rows.append((int(t.exit_ts), eq))
            source = "trade_pnl_fallback"
        else:
            rows.append((0, initial))
            source = "flat"
        return pd.DataFrame(rows, columns=["ts", "equity"]), source

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _default_risk(self):
        """Real RiskManager when importable (concurrent module), else the
        loudly-labelled :class:`PassThroughRisk`."""
        try:
            from aether.execution.interfaces import RiskConfig
            from aether.execution.risk import RiskManager
            return RiskManager(RiskConfig())
        except Exception as exc:
            logger.warning("aether.execution.risk unavailable (%s) — using "
                           "PassThroughRisk (NO risk rails)", exc)
            return PassThroughRisk()

    def _ts_bounds(self) -> tuple[int | None, int | None]:
        """[start, end] ISO dates -> epoch-second bounds (end inclusive)."""
        lo = int(pd.Timestamp(self.cfg.start).timestamp()) if self.cfg.start else None
        hi = (int((pd.Timestamp(self.cfg.end) + pd.Timedelta(days=1)).timestamp())
              if self.cfg.end else None)
        return lo, hi

    def _load_ticker(self, ticker: str) -> dict[str, np.ndarray] | None:
        try:
            arrs = load_embeddings(self.embeddings_dir, ticker)
        except FileNotFoundError as exc:
            logger.warning("run_signals: %s — ticker skipped", exc)
            return None
        lo, hi = self._ts_bounds()
        ats = arrs["anchor_ts"].astype(np.int64)
        mask = np.ones(ats.shape, dtype=bool)
        if lo is not None:
            mask &= ats >= lo
        if hi is not None:
            mask &= ats < hi
        if not mask.any():
            logger.warning("run_signals: no %s bars in [%s, %s] — skipped",
                           ticker, self.cfg.start, self.cfg.end)
            return None
        return _take(arrs, np.nonzero(mask)[0])

    @staticmethod
    def _session_last(arrs: dict[str, np.ndarray], i: int) -> bool:
        """True when anchor ``i``'s EXECUTION bar (bar ``i+1``, the
        ``next_*`` arrays at ``i``) is its session's final tradable bar.

        On a contract store the session's LAST anchor has NaN ``next_*``
        (no same-session fill bar exists), so the anchor whose execution
        bar is the final bar is the SECOND-to-last one — the anchor whose
        FOLLOWING anchor is the session's final anchor. Flagging the last
        anchor itself (the previous behavior) made the eod force-close
        unreachable: ``manage_position`` skipped the NaN bar and positions
        silently survived the session boundary. Both the last anchor and
        the one before it return True here — the last anchor's NaN
        execution bar is untradable either way, and entries are gated on
        this flag so no fill can land on a bar that would be force-closed
        the instant it prints.
        """
        ats = arrs["anchor_ts"]
        sm = arrs["session_minute"]
        n = int(ats.shape[0])

        def _rolls(j: int) -> bool:
            """True when anchor ``j`` has no same-session successor."""
            if j + 1 >= n:
                return True
            same_day = (int(ats[j + 1]) // _SECONDS_PER_DAY
                        == int(ats[j]) // _SECONDS_PER_DAY)
            minute_up = int(sm[j + 1]) > int(sm[j])
            return not (same_day and minute_up)

        # anchor i is itself session-final (NaN execution bar), OR its
        # execution bar — anchor i+1's bar — is the session's final one.
        return _rolls(i) or _rolls(i + 1)

    def _enter(self, ticker: str, i: int, arrs: dict[str, np.ndarray],
               sig: ReversalSignal, decision: RiskDecision,
               fees: float) -> dict | None:
        """Execute an approved entry on bar t+1 via tactics. None = no fill."""
        o = float(arrs["next_open"][i])
        h = float(arrs["next_high"][i])
        l = float(arrs["next_low"][i])
        proxy = spread_proxy(arrs["next_high"], arrs["next_low"], i)
        order = self.tactics.decide(sig.side, o, h, l, proxy)
        filled, px = ExecutionTactics.simulate_fill(
            sig.side, order, o, h, l, slippage_bps=self.cfg.slippage_bps)
        if not filled:
            logger.info("signal %s: %s limit %.4f unfilled on the next bar — "
                        "cancelled", sig.signal_id, sig.side, order["limit_px"])
            return None
        stop = float(decision.adjusted_stop_px)
        target = float(decision.adjusted_target_px)
        if not math.isfinite(stop) or stop <= 0:
            stop = float(sig.stop_px)
        if not math.isfinite(target) or target <= 0:
            target = float(sig.target_px)
        q = float(decision.qty) if sig.side == "long" else -float(decision.qty)
        entry_ts = int(arrs["anchor_ts"][i]) + 60
        self._trade_seq += 1
        return {
            "trade_id": f"{ticker}-{entry_ts}-{self._trade_seq}",
            "ticker": ticker,
            "side": sig.side,
            "q": q,
            "entry_px": float(px),
            "entry_fee": abs(q) * float(px) * fees,
            "entry_ts": entry_ts,
            "stop_px": stop,
            "target_px": target,
            "horizon": max(1, int(sig.horizon_bars)),
            "bars_held": 0,
            "conviction": float(sig.conviction),
            "meta": {"signal_id": sig.signal_id,
                     "order_type": order["order_type"],
                     "tactics_mode": order["mode"],
                     "risk_reasons": list(decision.reasons)},
        }

    # ------------------------------------------------------------------ #
    # Autopsies
    # ------------------------------------------------------------------ #

    def _run_autopsies(self, trades: list[TradeRecord]) -> list:
        """Autopsy every losing trade (cfg.autopsy_losses + autopsist given).

        The trade's session 1-min bars are loaded from the lake and passed
        to the autopsist; reports are saved via
        ``aether.worldmodel.autopsy.save_report``. Both the autopsist call
        and the save are bound adaptively by parameter name (the module is
        developed concurrently) and every failure is a warning, never a
        crash — autopsies must not break a backtest.
        """
        if not self.cfg.autopsy_losses or self.autopsist is None:
            return []
        losers = [t for t in trades if float(t.pnl) < 0]
        if not losers:
            return []
        try:
            from aether.data.storage import ParquetStore
            store = ParquetStore(self.data_root)
        except Exception as exc:
            logger.warning("autopsies skipped: lake unavailable (%s)", exc)
            return []
        try:
            from aether.worldmodel import autopsy as autopsy_mod
        except Exception as exc:
            logger.warning("aether.worldmodel.autopsy not importable (%s) — "
                           "reports will not be saved", exc)
            autopsy_mod = None

        fn = getattr(self.autopsist, "autopsy", self.autopsist)
        reports: list = []
        for trade in losers:
            try:
                start = pd.Timestamp(int(trade.entry_ts), unit="s").normalize()
                end = (pd.Timestamp(int(trade.exit_ts), unit="s").normalize()
                       + pd.Timedelta(days=1))
                bars = store.read("bars_1min", trade.ticker, start=start, end=end)
                candidates = {
                    "trade": trade, "record": trade, "trade_record": trade,
                    "bars": bars, "bars_1min": bars, "df": bars, "frame": bars,
                    "store": store, "ticker": trade.ticker,
                    "data_root": self.data_root,
                    "embeddings_dir": self.embeddings_dir,
                }
                report = fn(**_bind_kwargs(fn, candidates, "autopsist.autopsy"))
                if report is None:
                    continue
                reports.append(report)
                saver = getattr(autopsy_mod, "save_report", None) if autopsy_mod else None
                if callable(saver):
                    save_candidates = {
                        "report": report, "autopsy": report,
                        "data_root": self.data_root, "root": self.data_root,
                        "out_dir": self.data_root, "store": store,
                    }
                    saver(**_bind_kwargs(saver, save_candidates,
                                         "autopsy.save_report"))
            except Exception as exc:
                logger.warning("autopsy failed for trade %s: %s",
                               trade.trade_id, exc)
        return reports


# --------------------------------------------------------------------------- #
# Policy action helper
# --------------------------------------------------------------------------- #

def _greedy_actions(policy, obs: dict, carry):
    """Greedy actions from a policy, tolerant of the exact `mode` spelling.

    Tries ``policy.mode(obs, carry)`` (greedy analog of ``act``); on a
    TypeError, tries a no-argument ``policy.mode()`` switch then falls back
    to ``policy.act(obs, carry)``. Accepts either a bare actions dict or an
    ``(actions, ..., carry)`` tuple; tensors are converted to numpy keyed by
    ``ACTION_KEYS`` for ``TradingEnv.step``.
    """
    import torch  # keep the torch dependency out of the module import path

    obs_t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
             for k, v in obs.items()}
    out = None
    with torch.no_grad():
        mode = getattr(policy, "mode", None)
        if callable(mode):
            try:
                out = mode(obs_t, carry)
            except TypeError:
                try:
                    mode()  # mode() as an eval()-style switch
                except TypeError:
                    pass
                out = None
        if out is None:
            out = policy.act(obs_t, carry)

    if isinstance(out, dict):
        # HierarchicalPolicy.mode returns a flat dict whose action keys sit
        # at the top level alongside distribution stats and the new carry.
        actions, new_carry = out, out.get("carry", carry)
    else:
        actions, new_carry = out[0], out[-1]
    acts: dict[str, np.ndarray] = {}
    for key in ACTION_KEYS:
        if key not in actions:
            continue
        v = actions[key]
        acts[key] = (v.detach().cpu().numpy() if hasattr(v, "detach")
                     else np.asarray(v))
    return acts, new_carry
