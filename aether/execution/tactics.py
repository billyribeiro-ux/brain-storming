"""Execution tactics — how an approved signal becomes an order.

v1 ships two *honest*, non-learned tactics plus the scaffold of a learned
tactics model:

* **taker** — a market order at the next bar's open. The environment /
  backtest slippage model (``slippage_bps`` against the agent) applies.
  Always fills; pays the spread.
* **maker** — a limit order pegged ``0.25 · spread_proxy`` inside the next
  open (below it for buys, above it for sells). The fill rule is
  deliberately conservative and single-bar: a buy limit ``L`` fills iff the
  next bar's low trades through it (``next_low <= L``) and the fill price is
  exactly ``L`` (no price improvement is ever assumed); unfilled orders are
  cancelled, not carried. Symmetric for sells.

``spread_proxy`` is the rolling mean bar range (high − low) of the trailing
15 one-minute bars — a scale-free stand-in for the effective spread, since
the lake carries no quote data.

:class:`TacticsModel` is the learned hook: a small MLP mapping execution
context to a taker/maker choice plus a limit-offset fraction. It is
**not yet trained** — v1 never routes live decisions through it, and the
dashboard shows the static ``mode`` instead. It exists so that fill
outcomes recorded by the backtester/paper trader can later become its
training set without an interface change.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

TAKER = "taker"
MAKER = "maker"
MODES: tuple[str, ...] = (TAKER, MAKER)

#: Trailing window (bars) for the spread proxy.
SPREAD_PROXY_WINDOW: int = 15
#: Maker limit offset as a fraction of the spread proxy.
MAKER_OFFSET_FRAC: float = 0.25


def spread_proxy(next_high: np.ndarray, next_low: np.ndarray, t: int,
                 window: int = SPREAD_PROXY_WINDOW) -> float:
    """Rolling mean(high − low) of the trailing ``window`` bars before ``t``.

    The embedding npz stores each anchor's *next* bar OHLC, so the range of
    bar ``j`` lives at index ``j − 1`` of ``next_high``/``next_low``. At
    decision index ``t`` the bars known to the agent are therefore indices
    ``< t`` — using ``next_*[t]`` here would peek one bar into the future.

    Returns NaN when no finite trailing ranges exist (e.g. ``t == 0``);
    callers must treat NaN as "no proxy" and fall back to taker.
    """
    lo = max(0, int(t) - int(window))
    hi = int(t)
    if hi <= lo:
        return float("nan")
    ranges = (np.asarray(next_high[lo:hi], dtype=float)
              - np.asarray(next_low[lo:hi], dtype=float))
    ranges = ranges[np.isfinite(ranges)]
    return float(ranges.mean()) if ranges.size else float("nan")


class ExecutionTactics:
    """Static (non-learned) execution tactics with a conservative fill model.

    Parameters
    ----------
    mode:
        ``"taker"`` (default) or ``"maker"``.
    maker_offset_frac:
        Limit offset as a fraction of the spread proxy (default 0.25).
    model:
        Optional :class:`TacticsModel`. Held for provenance/dashboard only —
        v1 never routes decisions through it because it is untrained.
    """

    def __init__(self, mode: str = TAKER,
                 maker_offset_frac: float = MAKER_OFFSET_FRAC,
                 model: "TacticsModel | None" = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.maker_offset_frac = float(maker_offset_frac)
        self.model = model

    def decide(self, side: str, next_open: float, next_high: float,
               next_low: float, spread_proxy: float) -> dict:
        """Choose the order for one entry.

        Returns ``{"order_type": "market"|"limit", "limit_px": float|None,
        "mode": str}``. In maker mode with an unusable spread proxy (NaN or
        non-positive — e.g. the very first bars of a session) the tactic
        degrades honestly to a market order rather than inventing a peg.
        """
        if self.mode == MAKER and math.isfinite(spread_proxy) and spread_proxy > 0:
            offset = self.maker_offset_frac * float(spread_proxy)
            limit = (float(next_open) - offset if side == "long"
                     else float(next_open) + offset)
            return {"order_type": "limit", "limit_px": float(limit),
                    "mode": MAKER}
        return {"order_type": "market", "limit_px": None, "mode": self.mode}

    @staticmethod
    def simulate_fill(side: str, decision: dict, next_open: float,
                      next_high: float, next_low: float,
                      slippage_bps: float = 0.0) -> tuple[bool, float | None]:
        """Conservative single-bar fill simulation for :meth:`decide` output.

        * market: always fills at ``next_open`` with ``slippage_bps`` applied
          against the agent (buys pay up, sells receive less),
        * limit: a buy limit ``L`` fills iff ``next_low <= L`` at exactly
          ``L`` (no slippage, no improvement); a sell limit iff
          ``next_high >= L``. Unfilled ⇒ ``(False, None)`` — the order is
          cancelled, never carried to later bars.

        KNOWN OPTIMISM (acknowledged, unmodeled): the limit rule is
        touch-equals-fill. In reality a bar whose extreme merely TOUCHES
        the limit price may not fill it — queue position and available
        size at the level decide — so maker fill rates simulated here are
        an upper bound. Downstream consumers (backtests, the paper
        trader) already partially offset this by refusing same-bar target
        credit after a maker fill; the residual optimism stands until a
        queue model exists.
        """
        direction = 1.0 if side == "long" else -1.0
        if decision.get("order_type") == "limit":
            limit = float(decision["limit_px"])
            if side == "long":
                return (True, limit) if float(next_low) <= limit else (False, None)
            return (True, limit) if float(next_high) >= limit else (False, None)
        slip = float(slippage_bps) / 1e4
        return True, float(next_open) * (1.0 + direction * slip)


class TacticsModel(nn.Module):
    """Learned execution-tactics head — scaffold only, **not yet trained**.

    Input features (order matters, all scale-free), shape ``[B, 4]``:

    0. ``spread_proxy_rel`` — spread proxy / price,
    1. ``range_rel``        — most recent bar range / price,
    2. ``session_progress`` — session minute / 390,
    3. ``side``             — +1 long, −1 short.

    Outputs: logits over ``(taker, maker)`` and a limit-offset fraction in
    ``(0, 0.5)`` via ``0.5 · sigmoid`` (bounded so a learned peg can never
    exceed half the spread proxy). Intended training signal: fill outcomes
    (filled?, realized slippage vs. arrival) recorded by the backtester and
    paper trader.
    """

    FEATURES: tuple[str, ...] = (
        "spread_proxy_rel", "range_rel", "session_progress", "side")
    n_features: int = 4

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(self.n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, len(MODES))
        self.offset_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """``[B, 4] -> (mode logits [B, 2], limit-offset frac [B] in (0, 0.5))``."""
        h = self.body(features)
        logits = self.mode_head(h)
        offset = 0.5 * torch.sigmoid(self.offset_head(h)).squeeze(-1)
        return logits, offset

    @torch.no_grad()
    def decide_learned(self, features) -> dict:
        """Greedy decision from the (untrained) model.

        NOT YET TRAINED: the weights are random initialization until a
        fill-outcome training loop exists, so v1 execution never routes
        through this method — :class:`ExecutionTactics` uses its static
        ``mode`` and the dashboard displays that mode. This method exists so
        the calling convention is frozen before learning starts. The
        returned dict carries ``"trained": False`` so any accidental caller
        can see (and log) that the choice is untrained.
        """
        x = torch.as_tensor(features, dtype=torch.float32).reshape(1, self.n_features)
        logits, offset = self.forward(x)
        mode = MODES[int(logits.argmax(dim=-1).item())]
        return {
            "mode": mode,
            "order_type": "market" if mode == TAKER else "limit",
            "limit_offset_frac": float(offset.item()),
            "trained": False,
        }
