"""Hard-railed risk management (Layer 5).

Philosophy (mirrors ``RiskConfig``'s docstring): the rails here are
deliberately NOT learned. Every learned component upstream — policy,
world model, memory — proposes; this module disposes, with fixed arithmetic
that bounds what any of them can ever do to the book. A bug or a delusion in
a neural network must never be able to size past these rails.

Rule order (each consulted rule appends a human-readable line to
``RiskDecision.reasons``; evaluation stops at the first hard rejection):

1. **Circuit breaker** — once the session day's PnL breaches
   ``-daily_loss_stop_frac`` the manager halts NEW entries and the halt is
   STICKY: it persists even if PnL later recovers, until :meth:`reset_day`.
   A recovering book after a max-loss day is exactly when revenge trading
   happens; the rail does not negotiate.
2. **Max open positions** — a new instrument is rejected when the slot
   budget is spent (adding to an already-held instrument consumes no slot).
3. **Per-instrument cap** — ``qty1 = max_position_frac · equity / entry_px``.
4. **Per-trade risk** — ``qty2 = per_trade_risk_frac · equity /
   |entry_px − stop_px|``; a stop AT the entry is rejected outright (risk
   per share would be zero and the formula degenerate).
5. **Gross exposure headroom** — ``qty3 = max(0, max_gross_exposure −
   Σ|held fracs|) · equity / entry_px``.
6. **Correlation haircut** — when a correlation matrix is supplied
   (aligned to ``sorted(open_positions)`` + candidate LAST), size shrinks by
   ``1 / (1 + Σᵢ |corr(candidate, heldᵢ)| · |held fracᵢ|)``: near-duplicate
   exposure is treated as more of the same position, not a new bet.

``qty = floor(min(qty1, qty2, qty3) · haircut)`` in whole shares;
``approved`` iff at least one share survives.

``adjusted_stop_px`` / ``adjusted_target_px`` currently PASS THROUGH the
signal's levels unchanged. They exist as the hook where a future learned
adjustment (e.g. autopsy-informed stop widening) plugs in — the learned part
may move levels, but only through this audited interface, never around it.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from aether.execution.interfaces import ReversalSignal, RiskConfig, RiskDecision
from aether.utils.logging import get_logger

logger = get_logger("aether.risk")


class RiskManager:
    """Stateful hard-rail gatekeeper (``RiskManagerProtocol``).

    The only state is the sticky circuit breaker plus lightweight counters
    for the dashboard; everything else is pure arithmetic on the arguments.
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._halted: bool = False          # sticky until reset_day()
        self._n_assessed: int = 0
        self._n_approved: int = 0
        self._last_reasons: list[str] = []

    # ------------------------------------------------------------------ #
    # Assessment
    # ------------------------------------------------------------------ #

    def assess(self, signal: ReversalSignal, equity: float,
               open_positions: dict[str, float], day_pnl_frac: float,
               corr: Optional[np.ndarray] = None) -> RiskDecision:
        """Run every rail in order; see the module docstring for semantics.

        Parameters
        ----------
        signal:
            The candidate :class:`ReversalSignal`.
        equity:
            Current account equity in dollars.
        open_positions:
            ``ticker -> signed position fraction of equity`` for the book.
        day_pnl_frac:
            Session-day PnL as a fraction of starting equity (negative =
            loss). Trips and re-arms the sticky circuit breaker.
        corr:
            Optional correlation matrix aligned to
            ``sorted(open_positions) + [signal.ticker]`` (candidate LAST) —
            :func:`rolling_correlation` produces exactly this given that
            ticker order.
        """
        cfg = self.cfg
        reasons: list[str] = []
        self._n_assessed += 1

        def reject() -> RiskDecision:
            self._last_reasons = reasons
            logger.info("risk REJECT %s: %s", signal.signal_id, reasons[-1])
            return RiskDecision(approved=False, qty=0.0, reasons=reasons,
                                adjusted_stop_px=signal.stop_px,
                                adjusted_target_px=signal.target_px)

        # --- 1. Circuit breaker (sticky). --------------------------------
        if day_pnl_frac <= -cfg.daily_loss_stop_frac:
            self._halted = True
        if self._halted:
            reasons.append(
                f"[circuit-breaker] HALTED: day PnL {day_pnl_frac:+.2%} vs "
                f"-{cfg.daily_loss_stop_frac:.2%} stop (sticky until "
                f"reset_day) — reject")
            return reject()
        reasons.append(
            f"[circuit-breaker] pass: day PnL {day_pnl_frac:+.2%} above the "
            f"-{cfg.daily_loss_stop_frac:.2%} halt threshold")

        # --- 2. Max open positions. ---------------------------------------
        held = {t: f for t, f in open_positions.items() if f != 0.0}
        needs_slot = signal.ticker not in held
        if needs_slot and len(held) >= cfg.max_open_positions:
            reasons.append(
                f"[max-open] {len(held)}/{cfg.max_open_positions} slots used "
                f"and {signal.ticker} is not held — reject")
            return reject()
        reasons.append(
            f"[max-open] pass: {len(held)}/{cfg.max_open_positions} slots "
            f"used ({signal.ticker} {'already held' if not needs_slot else 'takes a free slot'})")

        # --- 3. Per-instrument cap. -----------------------------------------
        # KNOWN LIMITATION (acknowledged, unmodeled): same-ticker NETTING.
        # qty1 caps the CANDIDATE's notional in isolation — it does not net
        # against (or stack onto) an existing position in the same ticker,
        # and an opposite-side signal is sized as a fresh position rather
        # than a reduce/flip. The execution layers currently enforce one
        # position per ticker (backtest/paper skip signals while
        # positioned), so this path is not exercised today; if pyramiding
        # or flips ever land, this rail must learn |existing + candidate|.
        entry_px = float(signal.entry_px)
        if not math.isfinite(entry_px) or entry_px <= 0:
            reasons.append(
                f"[instrument-cap] entry_px={entry_px!r} is not a positive "
                f"price — reject")
            return reject()
        qty1 = cfg.max_position_frac * equity / entry_px
        reasons.append(
            f"[instrument-cap] qty1 = {cfg.max_position_frac:.0%} x "
            f"{equity:,.0f} / {entry_px:.4f} = {qty1:.1f} sh")

        # --- 4. Per-trade risk. -----------------------------------------------
        stop_dist = abs(entry_px - float(signal.stop_px))
        if stop_dist == 0.0:
            reasons.append(
                "[per-trade-risk] stop equals entry: risk per share is zero, "
                "sizing is undefined — reject")
            return reject()
        qty2 = cfg.per_trade_risk_frac * equity / stop_dist
        reasons.append(
            f"[per-trade-risk] qty2 = {cfg.per_trade_risk_frac:.2%} x "
            f"{equity:,.0f} / |{entry_px:.4f} - {signal.stop_px:.4f}| "
            f"= {qty2:.1f} sh")

        # --- 5. Gross exposure headroom. ---------------------------------------
        gross = sum(abs(f) for f in held.values())
        headroom = max(0.0, cfg.max_gross_exposure - gross)
        qty3 = headroom * equity / entry_px
        reasons.append(
            f"[gross-exposure] gross {gross:.2f} of {cfg.max_gross_exposure:.2f} "
            f"-> headroom {headroom:.2f} -> qty3 = {qty3:.1f} sh")

        # --- 6. Correlation haircut. --------------------------------------------
        haircut = 1.0
        if not cfg.correlation_haircut:
            reasons.append("[correlation] disabled by config — no haircut")
        elif corr is None or not held:
            reasons.append(
                "[correlation] no matrix / no held positions — haircut skipped")
        else:
            held_order = sorted(held)            # matrix row/col convention
            expected = len(held_order) + 1       # + candidate, LAST
            corr = np.asarray(corr, dtype=float)
            if corr.shape != (expected, expected):
                # Fail LOUD in the audit trail but do not silently resize a
                # matrix we cannot interpret; sizing proceeds un-haircut.
                reasons.append(
                    f"[correlation] matrix shape {corr.shape} != "
                    f"({expected}, {expected}) for sorted holdings + "
                    f"candidate — haircut skipped")
            else:
                weighted = sum(
                    abs(float(corr[-1, i])) * abs(held[t])
                    for i, t in enumerate(held_order))
                haircut = 1.0 / (1.0 + weighted)
                reasons.append(
                    f"[correlation] sum |corr x held frac| = {weighted:.3f} "
                    f"-> haircut x{haircut:.3f}")

        # --- Final sizing. --------------------------------------------------------
        qty = float(math.floor(min(qty1, qty2, qty3) * haircut))
        approved = qty >= 1.0
        if approved:
            reasons.append(
                f"[final] qty = floor(min({qty1:.1f}, {qty2:.1f}, {qty3:.1f}) "
                f"x {haircut:.3f}) = {qty:.0f} sh — approved")
            self._n_approved += 1
        else:
            qty = 0.0
            reasons.append(
                f"[final] sized below one whole share "
                f"(min qty {min(qty1, qty2, qty3):.2f} x {haircut:.3f}) — reject")
        self._last_reasons = reasons
        logger.info("risk %s %s: qty=%.0f", "APPROVE" if approved else "REJECT",
                    signal.signal_id, qty)
        # adjusted_* pass through unchanged — the future-learned-adjustment
        # hook documented in the module docstring.
        return RiskDecision(approved=approved, qty=qty, reasons=reasons,
                            adjusted_stop_px=signal.stop_px,
                            adjusted_target_px=signal.target_px)

    # ------------------------------------------------------------------ #
    # Session state
    # ------------------------------------------------------------------ #

    def reset_day(self) -> None:
        """New session day: re-arm the circuit breaker."""
        if self._halted:
            logger.info("risk: circuit breaker re-armed for the new day")
        self._halted = False

    @property
    def state(self) -> dict:
        """Dashboard snapshot: breaker status, counters, last audit trail."""
        return {
            "halted": self._halted,
            "n_assessed": self._n_assessed,
            "n_approved": self._n_approved,
            "last_reasons": list(self._last_reasons),
        }


# --------------------------------------------------------------------------- #
# Correlation helper
# --------------------------------------------------------------------------- #

def rolling_correlation(store, tickers: Sequence[str], end,
                        lookback_days: int = 60) -> np.ndarray:
    """Log-return correlation matrix from ``bars_daily`` closes.

    Parameters
    ----------
    store:
        A ``ParquetStore`` instance, or a lake root path (str/Path) from
        which one is constructed. The import is lazy so ``risk`` stays free
        of pandas/lake dependencies unless correlations are actually used.
    tickers:
        Row/column order of the result. To feed :meth:`RiskManager.assess`,
        pass ``sorted(open_positions) + [candidate_ticker]`` (candidate
        last). Duplicate tickers are allowed and produce duplicate rows.
    end:
        Inclusive end date (anything ``pd.Timestamp`` accepts).
    lookback_days:
        Number of TRADING days of returns used (a ~2.2x calendar buffer is
        read to cover weekends/holidays, then trimmed).

    Returns a ``[len(tickers), len(tickers)]`` float matrix with unit
    diagonal. A zero-variance column (flat closes) would produce NaN
    correlations; those are neutralized to 0 off-diagonal — documented,
    conservative (no haircut from a ticker that carries no signal).
    """
    from pathlib import Path

    import pandas as pd

    from aether.data.storage import ParquetStore  # lazy: see docstring

    if isinstance(store, (str, Path)):
        store = ParquetStore(store)

    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=int(lookback_days * 2.2) + 7)
    series: list[pd.Series] = []
    for ticker in tickers:
        df = store.read("bars_daily", ticker, start=start_ts, end=end_ts,
                        columns=["date", "close"])
        if df.empty:
            raise ValueError(
                f"rolling_correlation: no bars_daily rows for {ticker!r} in "
                f"[{start_ts.date()}, {end_ts.date()}]")
        series.append(df.set_index("date")["close"].rename(ticker))

    # Inner-join on dates so every column covers the same sessions, then keep
    # the newest lookback_days+1 closes -> lookback_days returns.
    px = pd.concat(series, axis=1, join="inner",
                   keys=range(len(series))).sort_index()
    px = px.tail(lookback_days + 1)
    rets = np.log(px.to_numpy(dtype=float))
    rets = np.diff(rets, axis=0)
    if rets.shape[0] < 2:
        raise ValueError(
            f"rolling_correlation: only {rets.shape[0]} overlapping return "
            f"rows for {list(tickers)} — need at least 2")
    corr = np.corrcoef(rets, rowvar=False)
    corr = np.atleast_2d(corr)
    corr[~np.isfinite(corr)] = 0.0       # zero-variance neutralization
    np.fill_diagonal(corr, 1.0)
    return corr
