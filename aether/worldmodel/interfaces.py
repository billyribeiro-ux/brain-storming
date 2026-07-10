"""THE WORLD-MODEL CONTRACT (Layer 2).

Everything in ``aether.worldmodel`` builds against these types, exactly as
``aether/perception/interfaces.py`` governs Layer 1. The world model consumes
perception embeddings (it never touches raw bars directly, except the causal
module which reads interpretable observables from the lake) and produces:

* latent dynamics with imagination rollouts (counterfactual simulation),
* a living causal graph over interpretable driver channels,
* multi-tier memory with outcome-tagged analogs,
* forensic autopsy reports for closed trades.

Design honesty
--------------
Dynamics are ACTION-FREE by design: at Aether's size, its own orders do not
move AAPL/SPY-class instruments, so market-state evolution is independent of
the agent's actions. Position/PnL dynamics are handled exactly in the
trading environment (Layer 3), not approximated here. This is a deliberate,
documented simplification that makes imagination rollouts honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Latent dynamics (RSSM-style)
# --------------------------------------------------------------------------- #

@dataclass
class DynamicsConfig:
    embed_dim: int = 64        # perception fused dim (must match checkpoint)
    deter_dim: int = 192       # GRU deterministic path
    stoch_dim: int = 32        # stochastic latent (diagonal Gaussian)
    hidden_dim: int = 192
    kl_beta: float = 1.0
    free_nats: float = 1.0     # per-STEP budget on the dim-summed KL (Dreamer convention)
    horizon: int = 30          # default imagination depth (bars)


@dataclass
class LatentState:
    """Batched latent state; every tensor is [B, ...]."""
    deter: Tensor              # [B, deter_dim]
    stoch: Tensor              # [B, stoch_dim]

    @property
    def feature(self) -> Tensor:
        """Concatenated feature the decoder/policies consume: [B, D+S]."""
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> "LatentState":
        return LatentState(self.deter.detach(), self.stoch.detach())


@dataclass
class ObserveResult:
    """Posterior filtering of an embedding sequence [B, T, E]."""
    states: LatentState        # tensors shaped [B*T? NO — [B, T, dim] stacked]
    post_mu: Tensor            # [B, T, stoch_dim]
    post_std: Tensor
    prior_mu: Tensor
    prior_std: Tensor


@dataclass
class ImaginedRollout:
    """N stochastic futures from one start state.

    embeds: decoded perception-space trajectories [N, B, H, embed_dim]
    deter/stoch: latent trajectories [N, B, H, dim]
    """
    embeds: Tensor
    deter: Tensor
    stoch: Tensor


class LatentDynamics(nn.Module):
    """Contract for the dynamics model (implemented in dynamics.py).

    Required methods:
      initial(batch) -> LatentState                       (zeros)
      observe(embeds [B,T,E]) -> ObserveResult            (posterior filter)
      imagine(start: LatentState, horizon: int, n: int) -> ImaginedRollout
      decode(feature [.., D+S]) -> embeds_hat [.., E]
      loss(embeds [B,T,E]) -> dict with 'total','recon','kl' scalar tensors
    Device-agnostic; deterministic under torch.manual_seed for tests.
    """

    output_names = ("total", "recon", "kl")


# --------------------------------------------------------------------------- #
# Causal graph
# --------------------------------------------------------------------------- #

#: Interpretable per-ticker observables (computed from the lake with the
#: same scale-free hygiene as perception — representation, not indicators).
CAUSAL_TICKER_CHANNELS: tuple[str, ...] = ("ret_1m", "vol_z", "range_z", "ret_5m")
#: Market-wide observables appended once (from _MARKET datasets).
CAUSAL_MARKET_CHANNELS: tuple[str, ...] = ("treasury_10y_chg", "news_rate")


@dataclass
class CausalConfig:
    tickers: tuple[str, ...] = ()          # aliases; filled from config.TICKERS
    n_lags: int = 5                        # minutes of causal lag considered
    window_minutes: int = 390 * 5          # rolling refit window (5 sessions)
    l1_penalty: float = 1e-3               # sparsity
    acyclic_penalty: float = 10.0          # NOTEARS-style within-lag-0 term
    stability_bootstraps: int = 8          # subsample refits for edge confidence
    hidden_dim: int = 32
    epochs: int = 200
    lr: float = 1e-2


@dataclass
class CausalEdge:
    src: str                   # node name, e.g. "AAPL.ret_1m" or "MKT.news_rate"
    dst: str
    lag: int                   # 0..n_lags (0 = contemporaneous)
    weight: float              # signed strength
    confidence: float          # [0,1] bootstrap stability


@dataclass
class CausalGraphSnapshot:
    """One fitted causal structure, timestamped for the dashboard's history."""
    fitted_start: str
    fitted_end: str
    nodes: list[str]
    edges: list[CausalEdge]

    def top_drivers(self, dst: str, k: int = 5) -> list[CausalEdge]:
        """Strongest incoming edges for a node, by |weight|·confidence."""
        inc = [e for e in self.edges if e.dst == dst]
        return sorted(inc, key=lambda e: -abs(e.weight) * e.confidence)[:k]

    def to_json(self) -> str: ...          # implemented in causal.py helpers
    # from_json(str) -> CausalGraphSnapshot  (module-level function)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

@dataclass
class MemoryConfig:
    dim: int = 64                          # key dim = perception fused dim
    capacity: int = 200_000                # episodic ring size
    persist_dir: str = "data/memory"


@dataclass
class MemoryAnalog:
    """A retrieved historical analog of the current state."""
    similarity: float                      # cosine in [-1, 1]
    ticker: str
    anchor_ts: int                         # epoch-seconds key of the moment
    outcome: dict                          # e.g. {"fwd_ret_5m":..., "fwd_ret_30m":...}
    meta: dict = field(default_factory=dict)


class MemoryBankProtocol:
    """memory.MemoryBank must provide:
      add(keys [N,dim] tensor/ndarray, metas: list[dict]) -> None
      query(key [dim], k=8) -> list[MemoryAnalog]
      __len__; save() / load() using persist_dir
      consolidate(n_clusters=32) -> dict   (semantic tier: cluster stats)
    """


# --------------------------------------------------------------------------- #
# Trades & autopsies
# --------------------------------------------------------------------------- #

@dataclass
class TradeRecord:
    """One closed round-trip, the unit of learning-from-outcomes."""
    trade_id: str
    ticker: str
    side: str                  # "long" | "short"
    entry_ts: int              # epoch-seconds (naive-ET convention of the lake)
    exit_ts: int
    entry_px: float
    exit_px: float
    qty: float
    pnl: float                 # signed, after fees
    fees: float
    stop_px: float
    target_px: float
    exit_reason: str           # "stop" | "target" | "policy_exit" | "eod"
    conviction: float = 0.0
    signal_meta: dict = field(default_factory=dict)


@dataclass
class Counterfactual:
    description: str           # e.g. "enter 5 bars later"
    pnl: float                 # simulated PnL under the variation
    delta: float               # pnl - actual pnl


@dataclass
class AutopsyReport:
    """Deep forensic analysis of one trade. JSON-serializable via asdict."""
    trade: TradeRecord
    verdict: str               # "good_loss" | "bad_loss" | "good_win" | "lucky_win"
    narrative: str             # natural-language post-mortem
    drivers: list[dict]        # [{"name","attribution"}] perception saliency
    causal_context: list[dict] # top causal edges active around entry
    counterfactuals: list[Counterfactual]
    analogs: list[MemoryAnalog]
    lessons: list[dict]        # structured entries for the lessons buffer


#: Path (relative to data root) where autopsy lessons accumulate; training
#: consumers upweight/replay these states. Append-only JSONL.
LESSONS_PATH = "lessons.jsonl"
