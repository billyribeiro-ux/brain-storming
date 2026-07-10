"""THE SIGNAL/RISK/EXECUTION CONTRACT (Layer 5).

Signals must be (a) high-conviction — a consensus gate across independent
evidence sources, and (b) fully explainable — every signal carries the
structured evidence it was built from plus a natural-language rationale
assembled from that evidence (template-composed from real attributions;
honest about being so).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #

@dataclass
class SignalConfig:
    min_conviction: float = 0.65        # consensus gate threshold
    n_imagination: int = 16             # world-model rollouts polled
    memory_k: int = 12                  # analogs consulted
    max_epistemic: float = 2.0          # abstain when the model admits ignorance
    max_anomaly: float = 4.0            # abstain in unprecedented regimes
    horizon_bars: int = 30


@dataclass
class EvidenceBundle:
    """Structured evidence a signal is built from — the audit trail."""
    policy_prob: float                  # policy's own action probability
    imagination_agreement: float        # fraction of rollouts favorable
    analog_winrate: float               # outcome rate among memory analogs
    aleatoric: float
    epistemic: float
    anomaly: float
    saliency: list[dict]                # top perception drivers [{"name","score"}]
    causal_drivers: list[dict]          # top causal edges into this ticker now
    analogs_summary: list[dict]         # compact analog descriptions


@dataclass
class ReversalSignal:
    signal_id: str
    ts: int                             # epoch-seconds (lake convention)
    ticker: str
    side: str                           # "long" | "short"
    conviction: float                   # [0,1] consensus score
    entry_px: float
    stop_px: float
    target_px: float
    horizon_bars: int
    size_frac: float                    # pre-risk suggested size
    rationale: str                      # natural-language explanation
    evidence: EvidenceBundle = None     # populated in real use; None in stubs


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #

@dataclass
class RiskConfig:
    """Hard rails are deliberately NOT learned — they bound what any learned
    component can ever do. Everything inside the rails is learnable."""
    max_position_frac: float = 0.20     # per instrument, of equity
    max_gross_exposure: float = 1.00    # sum |positions| / equity
    per_trade_risk_frac: float = 0.005  # equity fraction at risk to the stop
    daily_loss_stop_frac: float = 0.02  # halt new entries for the day
    max_open_positions: int = 4
    correlation_haircut: bool = True    # shrink size for correlated exposure


@dataclass
class RiskDecision:
    approved: bool
    qty: float                          # shares (0 when rejected)
    reasons: list[str]                  # every rule consulted, pass or fail
    adjusted_stop_px: float
    adjusted_target_px: float


class RiskManagerProtocol:
    """execution.risk.RiskManager must provide:
      __init__(cfg: RiskConfig)
      assess(signal: ReversalSignal, equity: float,
             open_positions: dict[str, float],      # ticker -> signed frac
             day_pnl_frac: float,
             corr: 'np.ndarray | None' = None) -> RiskDecision
      Circuit breaker state persists per session day; reset_day() clears it."""


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

@dataclass
class BacktestConfig:
    tickers: tuple[str, ...] = ()
    start: str = ""
    end: str = ""
    fees_bps: float = 1.0
    slippage_bps: float = 2.0
    initial_equity: float = 100_000.0
    autopsy_losses: bool = True         # run autopsies on every losing trade


@dataclass
class BacktestResult:
    """Everything the dashboard and the audit need. equity_curve is a
    pandas DataFrame [ts, equity]; trades are worldmodel TradeRecords."""
    equity_curve: object
    trades: list
    stats: dict                         # sharpe, sortino, max_dd, hit_rate,
                                        # avg_win, avg_loss, per-ticker dict
    signals: list[ReversalSignal]
    autopsies: list                     # AutopsyReports for flagged trades


# --------------------------------------------------------------------------- #
# Paper trading
# --------------------------------------------------------------------------- #

@dataclass
class PaperConfig:
    blotter_path: str = "data/paper/blotter.jsonl"   # append-only order log
    state_path: str = "data/paper/state.json"        # positions, equity
    poll_seconds: int = 60


class PaperTraderProtocol:
    """execution.paper.PaperTrader: one iteration = sync lake -> embed the
    newest bars -> generate signals -> risk-check -> record simulated orders
    and mark existing positions against the newest bar. NEVER contacts a
    broker; the blotter JSONL is the sole output. run_once() is the unit the
    scheduler (or scripts/run_paper.py --loop) repeats."""
