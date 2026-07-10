"""Reward shaping for the Layer-3 trading environment.

Implements the :class:`~aether.decision.interfaces.RewardConfig` contract::

    r_t = w_pnl   · Δequity / equity₀
        − w_dd    · max(0, Δdrawdown)                    (drawdown /equity₀)
        − w_churn · 1[trade opened this bar] · fees_paid / equity₀
        + w_capture · terminal reversal-capture bonus    (episode end only)

Every term is normalized by the *initial* episode equity, so the reward
scale is invariant to account size and comparable across episodes — a
prerequisite for stable PPO advantages and for Layer-4 evolution to mutate
the weights meaningfully.

Why the churn term "double-counts" fees
---------------------------------------
Fees already reduce equity, so they appear inside the Δequity term too.
The extra ``w_churn`` penalty is deliberate shaping: pure PnL feedback on
fees is tiny and delayed relative to the noise of one bar, and an agent
exploring entry spam would need thousands of episodes to feel it. The
churn term makes the cost of *opening* a position immediate and explicit.

The terminal bonus uses hindsight — and that is legitimate
----------------------------------------------------------
``terminal_bonus`` looks at the episode's *future* prices relative to each
entry ("what was the best exit this trade could have achieved within the
next 30 bars?"). That is look-ahead — but it is computed ONCE, AFTER the
episode is finished, and flows only into the terminal reward. The agent's
*observations* and *within-episode* rewards never contain it, so nothing
the policy conditions on at decision time leaks the future. Reward shaping
from episode hindsight is standard practice (the return itself is already
"the future"); observations from hindsight would be the crime, and this
module never touches observations.

Purity
------
:class:`RewardShaper` holds only its config and ``equity₀``. Both public
methods are pure functions of their arguments — no environment state, no
randomness — so they are unit-testable with hand-computed numbers.
"""

from __future__ import annotations

import math

import numpy as np

from ..worldmodel.interfaces import TradeRecord
from .interfaces import RewardConfig

#: Hindsight window (bars, inclusive of the entry bar) over which the
#: terminal bonus measures the best achievable exit for each trade. Matches
#: the world model's default imagination horizon (DynamicsConfig.horizon).
CAPTURE_HORIZON_BARS: int = 30


class RewardShaper:
    """Pure reward computations for :class:`~aether.decision.env.TradingEnv`.

    Parameters
    ----------
    cfg:
        The multi-objective weights (contract:
        :class:`~aether.decision.interfaces.RewardConfig`).
    initial_equity:
        ``equity₀`` — the episode's starting equity, the normalizer of every
        term. ``RewardConfig`` deliberately carries only *weights* (Layer 4
        mutates them), so the normalizer arrives separately from
        ``EnvConfig.initial_equity``.
    """

    def __init__(self, cfg: RewardConfig, initial_equity: float) -> None:
        if initial_equity <= 0.0:
            raise ValueError(
                f"initial_equity must be positive, got {initial_equity}")
        self.cfg = cfg
        self.initial_equity = float(initial_equity)

    # ------------------------------------------------------------------ #
    # Per-step reward
    # ------------------------------------------------------------------ #

    def step_reward(self, prev_equity: float, equity: float,
                    prev_peak: float, peak: float,
                    opened_trade: bool, fees_paid: float) -> float:
        """One bar's shaped reward (see the module docstring formula).

        Parameters
        ----------
        prev_equity / equity:
            Mark-to-market equity before and after the bar transition.
        prev_peak / peak:
            Running intraday peak equity before and after (``peak`` must
            already include ``equity``: ``peak >= equity`` is not required,
            but ``peak = max(prev_peak, equity)`` is what the env passes).
        opened_trade:
            True iff a new position was opened during this bar.
        fees_paid:
            Fees charged for that opening fill (0.0 when no trade opened).
            This is the "fees-equivalent" magnitude of the churn penalty.

        Drawdown is measured from the running peak, normalized by
        ``equity₀`` like everything else, and only *increases* are punished
        (``max(0, Δdrawdown)``): recovering from a drawdown is already
        rewarded through the Δequity term, and rewarding it twice would pay
        the agent for round-tripping into holes.
        """
        eq0 = self.initial_equity
        pnl_term = (equity - prev_equity) / eq0
        dd_prev = max(0.0, prev_peak - prev_equity) / eq0
        dd_now = max(0.0, peak - equity) / eq0
        dd_term = max(0.0, dd_now - dd_prev)
        churn_term = (fees_paid / eq0) if opened_trade else 0.0
        return (self.cfg.w_pnl * pnl_term
                - self.cfg.w_dd * dd_term
                - self.cfg.w_churn * churn_term)

    # ------------------------------------------------------------------ #
    # Terminal bonus
    # ------------------------------------------------------------------ #

    def terminal_bonus(self, trades: list[TradeRecord],
                       episode_bars: np.ndarray) -> float:
        """Reversal-capture bonus, added to the LAST step's reward only.

        For each closed trade the *capture ratio* is::

            capture = realized pnl (net, signed, from the TradeRecord)
                      ─────────────────────────────────────────────────
                      best achievable pnl for that side within
                      [entry bar, entry bar + CAPTURE_HORIZON_BARS]

        where "best achievable" is the gross move to the most favorable
        close inside the window at the trade's own size
        (``qty · (max close − entry)`` for longs, mirrored for shorts).
        The ratio is clamped to ``[0, 1]``; if the window offered nothing
        (best ≤ 0 — the entry was simply on the wrong side) the capture is
        0. The bonus is ``w_capture · mean(capture)`` over all trades, and
        0.0 for a flat (trade-less) episode — standing aside earns nothing,
        good or bad, from this term.

        Entry location: the environment stamps every TradeRecord with
        ``signal_meta['entry_bar']`` — the entry fill's bar index within the
        episode — because this method receives only the close-price array
        (``episode_bars``) and prices alone cannot resolve a timestamp.

        HINDSIGHT WARNING: this method reads prices *after* each entry. It
        must only ever be called once the episode is over, and its output
        must only ever flow into the terminal reward — never into an
        observation. See the module docstring.
        """
        if not trades:
            return 0.0
        px = np.asarray(episode_bars, dtype=np.float64).reshape(-1)
        if px.size == 0:
            raise ValueError("terminal_bonus: empty episode_bars")

        captures: list[float] = []
        for tr in trades:
            entry_bar = tr.signal_meta.get("entry_bar")
            if entry_bar is None:
                raise ValueError(
                    "terminal_bonus: TradeRecord.signal_meta lacks "
                    "'entry_bar' (the episode bar index of the entry fill); "
                    "the environment must stamp it when closing the trade")
            i0 = int(entry_bar)
            if not 0 <= i0 < px.size:
                raise ValueError(
                    f"terminal_bonus: entry_bar {i0} outside the episode "
                    f"({px.size} bars)")
            window = px[i0: i0 + CAPTURE_HORIZON_BARS + 1]
            if tr.side == "long":
                best_move = float(window.max()) - tr.entry_px
            else:
                best_move = tr.entry_px - float(window.min())
            best_pnl = abs(tr.qty) * best_move
            if best_pnl <= 0.0 or not math.isfinite(best_pnl):
                captures.append(0.0)
                continue
            captures.append(min(1.0, max(0.0, tr.pnl / best_pnl)))
        return self.cfg.w_capture * float(np.mean(captures))
