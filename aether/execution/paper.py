"""Paper trading — the live-shaped loop that NEVER touches a broker.

One :meth:`PaperTrader.run_once` iteration:

1. (optional) incremental lake sync via ``aether.data.ingestion``
   (``bars_1min`` + ``bars_5min``), lazily imported,
2. load the latest per-ticker embedding npz files; if the lake holds a
   newer bar than the newest embedding, WARN that embeddings need
   re-precompute — stale state is never traded silently (new entries are
   skipped for stale tickers; existing positions are still marked/managed),
3. manage open positions against the newest execution bar (same
   conservative stop-before-target / horizon / eod logic as the backtester,
   shared via :func:`aether.execution.backtest.manage_position`),
4. mark open positions against the newest close,
5. generate signals on the newest bar per flat ticker, risk-check them, and
   simulate fills via tactics (maker limits are single-bar fill-or-cancel,
   exactly as in backtests),
6. append every event as JSONL to the blotter
   (``{ts, kind: order|mark|signal|halt, payload}``) and persist the state
   json (positions, equity, day PnL) atomically.

The blotter and state file are the sole outputs — there is no broker API
anywhere in this module, by design. ``run_loop`` repeats ``run_once`` every
``poll_seconds`` with a KeyboardInterrupt-safe shutdown for scripts.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from aether.execution.backtest import (
    load_embeddings,
    manage_position,
    to_jsonable,
)
from aether.execution.interfaces import PaperConfig
from aether.execution.tactics import ExecutionTactics, spread_proxy
from aether.utils import market_time
from aether.utils.logging import get_logger

logger = get_logger("aether.paper")

DEFAULT_EQUITY: float = 100_000.0

#: Substrings in risk reasons that indicate a circuit breaker, not a mere
#: size rejection — these produce a "halt" blotter event.
_HALT_MARKERS: tuple[str, ...] = ("halt", "daily", "circuit", "loss stop")


class PaperTrader:
    """Simulated live trading against the freshest precomputed embeddings.

    Parameters
    ----------
    cfg:
        :class:`~aether.execution.interfaces.PaperConfig` (blotter/state
        paths, poll cadence).
    aether_cfg:
        Full :class:`~aether.config.AetherConfig` (lake root for sync and
        staleness checks).
    signal_engine:
        ``generate(ticker, t, arrays) -> ReversalSignal | None`` — same
        contract as the backtester's engine.
    risk:
        ``RiskManagerProtocol`` implementation.
    tactics:
        :class:`~aether.execution.tactics.ExecutionTactics`.
    embeddings_dir:
        Where the per-ticker npz files live (default:
        ``<cache_root>/embeddings``, the decision layer's default).
    fees_bps / slippage_bps / initial_equity:
        Simulation costs and the starting equity of a fresh state file.
    """

    def __init__(self, cfg: PaperConfig, aether_cfg, signal_engine, risk,
                 tactics, embeddings_dir: str | Path | None = None,
                 fees_bps: float = 1.0, slippage_bps: float = 2.0,
                 initial_equity: float = DEFAULT_EQUITY) -> None:
        self.cfg = cfg
        self.aether_cfg = aether_cfg
        self.signal_engine = signal_engine
        self.risk = risk
        self.tactics = tactics if tactics is not None else ExecutionTactics()
        self.embeddings_dir = (Path(embeddings_dir) if embeddings_dir is not None
                               else Path(aether_cfg.data.cache_root) / "embeddings")
        self.blotter_path = Path(cfg.blotter_path)
        self.state_path = Path(cfg.state_path)
        self.fees_frac = float(fees_bps) / 1e4
        self.slip_frac = float(slippage_bps) / 1e4
        self.initial_equity = float(initial_equity)
        self._trade_seq = 0

    # ------------------------------------------------------------------ #
    # One iteration
    # ------------------------------------------------------------------ #

    def run_once(self, sync: bool = False) -> dict:
        """One paper-trading iteration; returns a summary dict.

        Idempotent between data updates: per-ticker ``last_bar_ts`` in the
        state file gates signal generation and position management, so
        polling faster than the data arrives never duplicates orders — only
        the portfolio "mark" event is appended each run.
        """
        if sync:
            self._sync_lake()

        state = self._load_state()
        n_events = 0
        tickers = sorted(p.stem for p in self.embeddings_dir.glob("*.npz"))
        if not tickers:
            logger.warning("no embedding npz files under %s — nothing to trade",
                           self.embeddings_dir)

        store = None
        try:
            from aether.data.storage import ParquetStore
            store = ParquetStore(self.aether_cfg.data.root)
        except Exception as exc:
            logger.warning("lake unavailable for staleness checks: %s", exc)

        loaded: dict[str, dict[str, np.ndarray]] = {}
        stale: set[str] = set()
        for ticker in tickers:
            try:
                arrs = load_embeddings(self.embeddings_dir, ticker)
            except Exception as exc:
                logger.warning("embeddings for %s unreadable: %s", ticker, exc)
                continue
            if arrs["anchor_ts"].shape[0] == 0:
                continue
            loaded[ticker] = arrs
            if store is not None:
                try:
                    wm = store.watermark("bars_1min", ticker)
                except Exception:
                    wm = None
                # Newest bar the embeddings know about = the last anchor's
                # NEXT bar (naive-ET epoch convention on both sides).
                emb_newest = pd.Timestamp(int(arrs["anchor_ts"][-1]) + 60, unit="s")
                if wm is not None and wm > emb_newest:
                    stale.add(ticker)
                    logger.warning(
                        "STALE EMBEDDINGS for %s: lake newest bar %s > newest "
                        "embedded bar %s — re-run EmbeddingStore.precompute; "
                        "new entries for this ticker are skipped this run",
                        ticker, wm, emb_newest)

        # ---- marks + day roll --------------------------------------------- #
        marks: dict[str, float] = {}
        for ticker, arrs in loaded.items():
            mark = self._newest_mark(arrs)
            if mark is not None:
                marks[ticker] = mark
        positions: dict[str, dict] = state["positions"]
        for ticker, pos in positions.items():
            if ticker in marks:
                pos["mark"] = marks[ticker]

        equity = self._equity(state)
        if loaded:
            newest_ts = max(int(a["anchor_ts"][-1]) for a in loaded.values())
            day = str(pd.Timestamp(newest_ts, unit="s").date())
            if state.get("day") != day:
                state["day"] = day
                state["day_start_equity"] = equity
                if hasattr(self.risk, "reset_day"):
                    try:
                        self.risk.reset_day()
                    except Exception as exc:
                        logger.warning("risk.reset_day failed: %s", exc)
        day_start = float(state.get("day_start_equity") or equity)
        day_pnl_frac = (equity / day_start - 1.0) if day_start else 0.0

        # ---- manage open positions on the newest execution bar ------------- #
        for ticker in list(positions):
            arrs = loaded.get(ticker)
            if arrs is None:
                continue
            i = int(arrs["anchor_ts"].shape[0]) - 1
            bar_ts = int(arrs["anchor_ts"][i])
            pos = positions[ticker]
            if bar_ts <= int(pos.get("last_managed_ts", -1)):
                continue  # this bar was already processed on a prior run
            pos["last_managed_ts"] = bar_ts
            rec = manage_position(
                pos, arrs["next_open"][i], arrs["next_high"][i],
                arrs["next_low"][i], arrs["next_close"][i],
                self._session_last_bar(arrs, i), self.fees_frac,
                self.slip_frac, exit_ts=bar_ts + 60)
            if rec is not None:
                self._apply_close(state, ticker, rec)
                self._emit("order", {"action": "exit", "trade": asdict(rec)})
                n_events += 1

        # per-position mark events
        for ticker, pos in positions.items():
            mark = float(pos.get("mark", pos["entry_px"]))
            self._emit("mark", {
                "ticker": ticker, "side": pos["side"], "qty": abs(pos["q"]),
                "mark_px": mark, "entry_px": pos["entry_px"],
                "unrealized": pos["q"] * (mark - pos["entry_px"]),
                "bars_held": pos["bars_held"],
            })
            n_events += 1

        # ---- new signals on the newest bar per flat ticker ------------------ #
        for ticker, arrs in loaded.items():
            if ticker in positions or ticker in stale:
                continue
            i = int(arrs["anchor_ts"].shape[0]) - 1
            bar_ts = int(arrs["anchor_ts"][i])
            if bar_ts <= int(state["last_bar_ts"].get(ticker, -1)):
                continue  # no new bar since the last run
            state["last_bar_ts"][ticker] = bar_ts
            nxt = (float(arrs["next_open"][i]), float(arrs["next_high"][i]),
                   float(arrs["next_low"][i]), float(arrs["next_close"][i]))
            if not all(math.isfinite(v) for v in nxt):
                continue  # NaN execution bar — never trade it
            sig = self.signal_engine.generate(ticker, i, arrs)
            if sig is None:
                continue
            equity = self._equity(state)
            open_fracs = ({t: p["q"] * float(p.get("mark", p["entry_px"])) / equity
                           for t, p in positions.items()} if equity > 0 else
                          {t: 0.0 for t in positions})
            decision = self.risk.assess(sig, equity, open_fracs, day_pnl_frac)
            self._emit("signal", {
                "signal": asdict(sig),
                "risk": {"approved": decision.approved, "qty": decision.qty,
                         "reasons": list(decision.reasons),
                         "adjusted_stop_px": decision.adjusted_stop_px,
                         "adjusted_target_px": decision.adjusted_target_px},
            })
            n_events += 1
            if not decision.approved or decision.qty <= 0:
                reasons = " ".join(decision.reasons).lower()
                if any(m in reasons for m in _HALT_MARKERS):
                    self._emit("halt", {"ticker": ticker,
                                        "signal_id": sig.signal_id,
                                        "reasons": list(decision.reasons)})
                    n_events += 1
                continue
            n_events += self._try_entry(state, ticker, i, arrs, sig, decision,
                                        bar_ts)

        # ---- portfolio mark, persist ---------------------------------------- #
        # Re-mark everything (positions opened this run were marked at their
        # entry price) so the persisted equity is against the newest close.
        for ticker, pos in positions.items():
            if ticker in marks:
                pos["mark"] = marks[ticker]
        equity = self._equity(state)
        state["equity"] = equity
        summary = {
            "equity": equity,
            "cash": state["cash"],
            "day": state.get("day"),
            "day_pnl_frac": (equity / float(state.get("day_start_equity") or equity)
                             - 1.0) if state.get("day_start_equity") else 0.0,
            "realized_pnl": state["realized_pnl"],
            "n_positions": len(positions),
            "n_trades": state["n_trades"],
            "stale_tickers": sorted(stale),
        }
        self._emit("mark", {"portfolio": summary,
                            "positions": {t: {k: p.get(k) for k in
                                              ("side", "q", "entry_px", "mark",
                                               "stop_px", "target_px", "bars_held")}
                                          for t, p in positions.items()}})
        n_events += 1
        self._save_state(state)
        summary["n_events"] = n_events
        logger.info("run_once: equity=%.2f positions=%d events=%d%s",
                    equity, len(positions), n_events,
                    f" STALE={sorted(stale)}" if stale else "")
        return summary

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    def run_loop(self, poll_seconds: int | None = None,
                 sync: bool = False) -> None:
        """Repeat :meth:`run_once` forever; Ctrl-C shuts down cleanly.

        State is persisted at the end of every iteration, so an interrupt
        (even mid-sleep) never loses more than the in-flight iteration.
        Non-KeyboardInterrupt exceptions are logged and the loop continues —
        a transient data problem must not kill a long-running paper session.
        """
        poll = int(poll_seconds if poll_seconds is not None
                   else self.cfg.poll_seconds)
        logger.info("paper loop: every %ds, blotter=%s (Ctrl-C to stop)",
                    poll, self.blotter_path)
        try:
            while True:
                try:
                    self.run_once(sync=sync)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.error("run_once failed (loop continues): %s", exc)
                time.sleep(poll)
        except KeyboardInterrupt:
            logger.info("paper loop: shutdown requested — state at %s, "
                        "blotter at %s", self.state_path, self.blotter_path)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sync_lake(self) -> None:
        """Incremental lake top-up (bars only). Failure = warn and carry on
        with the existing lake; a sync outage must not stop marking."""
        try:
            from aether.data.ingestion import IngestionEngine  # lazy: concurrent env
            engine = IngestionEngine(self.aether_cfg)
            report = asyncio.run(engine.sync(datasets=["bars_1min", "bars_5min"]))
            if isinstance(report, dict):
                rows = {d: sum(t.values()) for d, t in report.items()}
                logger.info("lake sync: new rows %s", rows)
        except Exception as exc:
            logger.warning("lake sync failed (continuing with existing lake): %s",
                           exc)

    def _try_entry(self, state: dict, ticker: str, i: int,
                   arrs: dict[str, np.ndarray], sig, decision,
                   bar_ts: int) -> int:
        """Simulate the entry on the newest execution bar. Returns #events."""
        o = float(arrs["next_open"][i])
        h = float(arrs["next_high"][i])
        l = float(arrs["next_low"][i])
        proxy = spread_proxy(arrs["next_high"], arrs["next_low"], i)
        order = self.tactics.decide(sig.side, o, h, l, proxy)
        filled, px = ExecutionTactics.simulate_fill(
            sig.side, order, o, h, l, slippage_bps=self.slip_frac * 1e4)
        if not filled:
            self._emit("order", {"action": "entry_cancelled", "ticker": ticker,
                                 "side": sig.side, "order_type": order["order_type"],
                                 "limit_px": order["limit_px"],
                                 "signal_id": sig.signal_id, "bar_ts": bar_ts})
            return 1
        stop = float(decision.adjusted_stop_px)
        target = float(decision.adjusted_target_px)
        if not math.isfinite(stop) or stop <= 0:
            stop = float(sig.stop_px)
        if not math.isfinite(target) or target <= 0:
            target = float(sig.target_px)
        q = float(decision.qty) if sig.side == "long" else -float(decision.qty)
        entry_fee = abs(q) * px * self.fees_frac
        self._trade_seq += 1
        state["n_trades"] = int(state["n_trades"])  # keep key present
        pos = {
            "trade_id": f"paper-{ticker}-{bar_ts + 60}-{self._trade_seq}",
            "ticker": ticker, "side": sig.side, "q": q,
            "entry_px": float(px), "entry_fee": entry_fee,
            "entry_ts": bar_ts + 60,
            "stop_px": stop, "target_px": target,
            "horizon": max(1, int(sig.horizon_bars)), "bars_held": 0,
            "conviction": float(sig.conviction),
            "mark": float(px),
            "last_managed_ts": bar_ts,  # entry bar handled just below
            "meta": {"signal_id": sig.signal_id,
                     "order_type": order["order_type"],
                     "tactics_mode": order["mode"],
                     "risk_reasons": list(decision.reasons)},
        }
        state["cash"] = float(state["cash"]) - q * px - entry_fee
        state["positions"][ticker] = pos
        events = 1
        self._emit("order", {
            "action": "entry", "ticker": ticker, "side": sig.side,
            "qty": abs(q), "px": px, "order_type": order["order_type"],
            "limit_px": order["limit_px"], "tactics_mode": order["mode"],
            "fee": entry_fee, "signal_id": sig.signal_id,
            "stop_px": stop, "target_px": target, "bar_ts": bar_ts,
        })
        # conservative same-bar exit check on the entry bar (as in backtests)
        rec = manage_position(
            pos, o, h, l, float(arrs["next_close"][i]),
            self._session_last_bar(arrs, i), self.fees_frac, self.slip_frac,
            exit_ts=bar_ts + 60, entry_bar=True)
        if rec is not None:
            self._apply_close(state, ticker, rec)
            self._emit("order", {"action": "exit", "trade": asdict(rec)})
            events += 1
        return events

    def _apply_close(self, state: dict, ticker: str, rec) -> None:
        pos = state["positions"].pop(ticker)
        exit_fee = rec.fees - float(pos["entry_fee"])
        state["cash"] = float(state["cash"]) + pos["q"] * rec.exit_px - exit_fee
        state["realized_pnl"] = float(state["realized_pnl"]) + rec.pnl
        state["n_trades"] = int(state["n_trades"]) + 1

    def _equity(self, state: dict) -> float:
        return float(state["cash"]) + sum(
            p["q"] * float(p.get("mark", p["entry_px"]))
            for p in state["positions"].values())

    @staticmethod
    def _newest_mark(arrs: dict[str, np.ndarray]) -> float | None:
        """Newest known close: the last anchor's next bar close when finite,
        else the anchor close itself."""
        for v in (arrs["next_close"][-1], arrs["close_px"][-1]):
            if math.isfinite(float(v)):
                return float(v)
        return None

    @staticmethod
    def _session_last_bar(arrs: dict[str, np.ndarray], i: int) -> bool:
        """True when anchor ``i``'s NEXT bar is its session's final bar
        (respects half-day early closes via the market calendar)."""
        d = pd.Timestamp(int(arrs["anchor_ts"][i]), unit="s").date()
        n_minutes = market_time.expected_session_minutes(d)
        return int(arrs["session_minute"][i]) + 1 >= n_minutes - 1

    # ---- persistence -------------------------------------------------- #

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text())
        else:
            state = {
                "cash": self.initial_equity,
                "equity": self.initial_equity,
                "realized_pnl": 0.0,
                "n_trades": 0,
                "day": None,
                "day_start_equity": self.initial_equity,
                "positions": {},
                "last_bar_ts": {},
                "created": pd.Timestamp.utcnow().isoformat(),
            }
        state.setdefault("positions", {})
        state.setdefault("last_bar_ts", {})
        state.setdefault("realized_pnl", 0.0)
        state.setdefault("n_trades", 0)
        return state

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(to_jsonable(state), indent=2, default=str))
        os.replace(tmp, self.state_path)  # atomic on POSIX

    def _emit(self, kind: str, payload: dict) -> None:
        """Append one blotter event: ``{ts, kind, payload}`` JSONL."""
        self.blotter_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": int(time.time()), "kind": kind,
                  "payload": to_jsonable(payload)}
        with self.blotter_path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
