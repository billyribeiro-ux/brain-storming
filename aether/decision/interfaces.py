"""THE DECISION-CORE CONTRACT (Layer 3).

Types shared by the trading environment, policies, PPO trainer, replay,
curriculum, and (Layer 5) the signal/risk/execution stack.

The one sacred invariant applies with full force here: the environment is a
bar-by-bar replay in which the agent, at bar t, sees ONLY state derived from
data ≤ t, and every order it places fills at bar t+1 or later under
conservative assumptions (stop-before-target when both prices are touched
inside one bar; slippage always against the agent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Precomputed embeddings (the env's observation source)
# --------------------------------------------------------------------------- #

class EmbeddingStoreProtocol:
    """decision.env.EmbeddingStore must provide:

      precompute(perception_ckpt: str, store: ParquetStore, tickers, start, end,
                 out_dir: str, device=None, batch_size=256) -> None   [classmethod]
          Runs the perception model over every valid anchor (stride=1) and
          saves per-ticker .npz: fused [N,D], aleatoric [N], epistemic [N],
          anomaly [N], anchor_ts [N] int64, session_minute [N] int16,
          close_px [N] float64 (anchor close, for fills/PnL), plus bar
          open/high/low of the NEXT bar for execution simulation.
      load(out_dir, ticker) -> dict of numpy arrays  [classmethod]

    Stats/config are read from the perception checkpoint so embeddings are
    always consistent with the network that produced them.
    """


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

#: Meta-controller intents (hierarchical level 1, re-decided every
#: ``meta_every`` bars).
META_ACTIONS: tuple[str, ...] = ("stand_aside", "hunt_long", "hunt_short")
#: Sub-policy trade actions (level 2, every bar).
TRADE_ACTIONS: tuple[str, ...] = ("hold", "enter", "exit", "adjust")


@dataclass
class EnvConfig:
    embeddings_dir: str = "data/cache/embeddings"
    tickers: tuple[str, ...] = ()          # aliases
    start: str = ""                        # ISO dates (episode sampling range)
    end: str = ""
    fees_bps: float = 1.0                  # per side
    slippage_bps: float = 2.0              # applied against the agent
    meta_every: int = 15                   # bars between meta decisions
    max_position_frac: float = 1.0         # of equity (risk layer tightens this)
    episode: str = "session"               # one full session per episode
    initial_equity: float = 100_000.0
    seed: int = 0


@dataclass
class EnvStep:
    """Vectorized step result; every array is [n_envs, ...]."""
    obs: dict[str, np.ndarray]
    reward: np.ndarray                     # [n_envs] float32
    done: np.ndarray                       # [n_envs] bool
    info: list[dict]                       # fills, trades closed this step, etc.


#: Observation dict keys and shapes (per env):
#:   "market"    [embed_dim]      perception fused embedding at bar t
#:   "unc"       [3]              aleatoric, epistemic, anomaly
#:   "position"  [5]              signed pos frac, entry px rel, unrealized rel,
#:                                bars in pos, stop distance rel
#:   "portfolio" [3]              cash frac, day pnl rel, drawdown rel
#:   "clock"     [2]              session progress, bars to close (both /390)
#:   "meta"      [len(META_ACTIONS)] one-hot of current intent
OBS_KEYS = ("market", "unc", "position", "portfolio", "clock", "meta")


#: Action dict (per env):
#:   "meta"   int in [0, len(META_ACTIONS))   (used every meta_every bars)
#:   "trade"  int in [0, len(TRADE_ACTIONS))
#:   "size"   float in (0, 1]   fraction of allowed position
#:   "stop"   float in (0, 1]   stop distance as multiple of recent realized
#:                              1-min range (learned scale, no fixed ticks)
#:   "target" float in (0, 1]   take-profit distance, same unit
ACTION_KEYS = ("meta", "trade", "size", "stop", "target")


class TradingEnvProtocol:
    """decision.env.TradingEnv must provide:
      __init__(cfg: EnvConfig, n_envs: int)
      reset(episode_ids: list | None = None) -> dict obs      (samples episodes)
      step(actions: dict[str, np.ndarray]) -> EnvStep
      episodes: list of (ticker, date) it can sample           (property)
    Every closed round-trip appears in EnvStep.info as a TradeRecord
    (worldmodel.interfaces.TradeRecord) under key 'trade_closed'.
    Positions force-close at end of session ('eod')."""


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #

@dataclass
class RewardConfig:
    """Multi-objective shaping. All terms are per-step unless noted.

    Reward = w_pnl · Δequity/equity₀
           − w_dd · max(0, Δdrawdown)
           − w_churn · |trade opened this bar| · fees-equivalent
           + w_capture · terminal reversal-capture bonus (episode end):
             realized fraction of the best adverse-to-favorable swing the
             episode offered around each entry (0 when flat all day; never
             computed from future data DURING the episode — terminal only).
    Weights are config now; Layer 4 evolution mutates them.
    """
    w_pnl: float = 1.0
    w_dd: float = 0.5
    w_churn: float = 0.05
    w_capture: float = 0.3


# --------------------------------------------------------------------------- #
# Policies & PPO
# --------------------------------------------------------------------------- #

@dataclass
class PolicyConfig:
    obs_market_dim: int = 64
    hidden_dim: int = 256
    memory_dim: int = 128                  # GRU carry inside the policy
    meta_every: int = 15


@dataclass
class PPOConfig:
    n_envs: int = 16
    rollout_len: int = 390                 # one session
    epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03                # early-stop epochs beyond this
    total_updates: int = 1000
    checkpoint_dir: str = "checkpoints/decision"
    log_jsonl: str = "runs/rl_train.jsonl"
    seed: int = 0


class PolicyProtocol:
    """decision.policies.HierarchicalPolicy must provide:
      act(obs: dict[str, Tensor], carry) -> (actions dict[str, Tensor],
          logprobs dict, value Tensor, new_carry)          # sampling
      evaluate(obs, actions, carry) -> (logprobs dict, entropy Tensor,
          value Tensor, new_carry)                         # for PPO updates
      initial_carry(batch) -> carry
    Discrete heads: Categorical. Continuous heads (size/stop/target):
    Beta distributions rescaled to (0,1] — bounded support, no clipping bias.
    The meta head is only sampled/evaluated on meta-decision bars; between
    them the previous intent is carried and its logprob contribution is zero.
    """
