"""Hierarchical actor-critic policy for the Aether decision core (Layer 3).

Architecture
------------
One shared recurrent torso feeds five heads::

    obs dict ── concat ──> LayerNorm ──> MLP(hidden) ──> GRUCell ──> feature
    (OBS_KEYS order)                                     (carry in/out = feature)
                 ┌────────────┬─────────────┬────────┬────────┬────────┐
               meta          trade         size     stop    target   value
            Categorical   Categorical      Beta     Beta     Beta    Linear
            |META_ACTIONS| |TRADE_ACTIONS| (α,β)    (α,β)    (α,β)   scalar

The GRU cell *is* the policy's memory: the carry passed between bars is the
cell's hidden state, and the feature every head consumes is the *new* carry.
``initial_carry`` returns zeros, so a fresh episode starts memory-less.

Continuous heads — Beta with a rescaling jacobian
-------------------------------------------------
Each continuous head (size / stop / target) emits raw ``[B, 2]`` from which

    α = softplus(raw₀) + 1,      β = softplus(raw₁) + 1.

Adding 1 keeps both concentrations ≥ 1, which forces the Beta density to be
**unimodal and bounded** — no U-shaped densities piling infinite mass onto
the action bounds, which would make boundary actions degenerate attractors.

The contract wants actions in ``(0, 1]`` (a zero size/stop/target is
meaningless), so the Beta variable ``U ~ Beta(α, β)`` on ``(0, 1)`` is
affinely rescaled:

    A = ε + (1 − ε) · U,          ε = ACTION_EPS.

Change of variables for a strictly monotone map ``A = f(U)``:

    p_A(a) = p_U(f⁻¹(a)) · |d f⁻¹/da| = p_U(u) / (1 − ε)
    ⇒  log p_A(a) = log p_U(u) − log(1 − ε)          (jacobian correction)
    ⇒  H(A)       = H(U)       + log(1 − ε)          (differential entropy
                                                       shifts by log|f′|)

Both corrections are constants, so they do not change gradients — they are
applied anyway so that stored log-probabilities are *true* densities of the
actions the environment actually received (importance ratios, KL estimates
and diagnostics all stay honest). ``U`` is clamped to ``[1e-6, 1 − 1e-6]``
before ``log_prob`` on both the sampling and evaluation paths: with α > 1
the density vanishes at the support edge, so an un-clamped boundary value
would yield ``log 0 = −inf``.

Meta masking (hierarchical timing)
----------------------------------
The meta head re-decides intent only on meta-decision bars, signalled by
``obs["flags"][:, 0] > 0.5``. On every other bar the meta *action* is the
previous intent decoded from the ``obs["meta"]`` one-hot, and its log-prob
contribution is **exactly zero** (a detached zero tensor): between decision
bars the intent is a deterministic carry-over, not a stochastic choice, so
it must not contribute to PPO ratios or entropy. ``evaluate`` applies the
identical mask, which makes old/new log-probs consistent by construction.

Determinism: sampling draws from torch's global generator only, so results
are reproducible under ``torch.manual_seed``. ``mode`` is sampling-free
(argmax / Beta mean) for evaluation and backtests.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Beta, Categorical

from aether.decision.interfaces import (
    META_ACTIONS,
    OBS_KEYS,
    PolicyConfig,
    TRADE_ACTIONS,
)

#: Continuous action heads, in ACTION_KEYS order (after "meta", "trade").
CONTINUOUS_KEYS: tuple[str, ...] = ("size", "stop", "target")

#: Lower edge of the rescaled continuous-action support: actions ∈ (ε, 1].
ACTION_EPS: float = 1e-3

#: Clamp for the underlying Beta variable u ∈ (0, 1). Keeps log-densities
#: finite at numerical boundaries (Beta.log_prob(0) = −inf when α > 1).
_U_EPS: float = 1e-6

#: Observation widths that do not depend on config ("market" comes from cfg).
#: Mirrors the OBS_KEYS shape table in aether.decision.interfaces.
_STATIC_OBS_DIMS: dict[str, int] = {
    "unc": 3,
    "position": 5,
    "portfolio": 3,
    "clock": 2,
    "meta": len(META_ACTIONS),
    "flags": 1,
}


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal weight init + zero bias — the standard PPO recipe.

    Small gains (0.01) on the action heads keep the initial policy close to
    uniform (near-zero logits / near-symmetric Beta), so early exploration
    is driven by the entropy bonus rather than by random initialization.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class HierarchicalPolicy(nn.Module):
    """Recurrent hierarchical policy implementing ``PolicyProtocol``."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obs_market_dim + sum(_STATIC_OBS_DIMS.values())

        # ---- Torso: LayerNorm -> MLP -> GRUCell ---------------------------
        # The LayerNorm sits on the *raw concatenated observation* so wildly
        # different obs scales (embedding vs. one-hots vs. fractions) enter
        # the MLP on comparable footing.
        self.obs_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            _orthogonal(nn.Linear(in_dim, cfg.hidden_dim), math.sqrt(2)),
            nn.Tanh(),
            _orthogonal(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), math.sqrt(2)),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.memory_dim)

        # ---- Heads ---------------------------------------------------------
        self.meta_head = _orthogonal(
            nn.Linear(cfg.memory_dim, len(META_ACTIONS)), gain=0.01)
        self.trade_head = _orthogonal(
            nn.Linear(cfg.memory_dim, len(TRADE_ACTIONS)), gain=0.01)
        # Each Beta head outputs (raw_alpha, raw_beta); softplus(·)+1 downstream.
        self.beta_heads = nn.ModuleDict({
            key: _orthogonal(nn.Linear(cfg.memory_dim, 2), gain=0.01)
            for key in CONTINUOUS_KEYS
        })
        self.value_head = _orthogonal(nn.Linear(cfg.memory_dim, 1), gain=1.0)

    # ------------------------------------------------------------------ #
    # Carry & torso
    # ------------------------------------------------------------------ #

    def initial_carry(
        self, batch: int, device: torch.device | str | None = None
    ) -> Tensor:
        """Zero GRU state ``[batch, memory_dim]`` — a memory-less fresh start."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch, self.cfg.memory_dim, device=device)

    def _prep(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validate keys and coerce every observation tensor to float32."""
        missing = [key for key in OBS_KEYS if key not in obs]
        if missing:
            raise KeyError(f"observation dict is missing keys {missing}; "
                           f"expected all of {OBS_KEYS}")
        return {key: obs[key].float() for key in OBS_KEYS}

    def _features(self, obs: dict[str, Tensor], carry: Tensor) -> Tensor:
        """Concat obs (OBS_KEYS order) -> norm -> MLP -> GRU. Returns the new
        carry, which doubles as the feature every head consumes."""
        x = torch.cat([obs[key] for key in OBS_KEYS], dim=-1)
        h = self.mlp(self.obs_norm(x))
        return self.gru(h, carry)

    # ------------------------------------------------------------------ #
    # Distributions
    # ------------------------------------------------------------------ #

    def _beta_dist(self, key: str, feature: Tensor) -> Beta:
        """Beta(α, β) with α, β = softplus(raw) + 1 ≥ 1 (unimodal, bounded)."""
        raw = self.beta_heads[key](feature)
        alpha = F.softplus(raw[..., 0]) + 1.0
        beta = F.softplus(raw[..., 1]) + 1.0
        return Beta(alpha, beta)

    @staticmethod
    def _meta_context(obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """(mask, prev_intent): mask is True on meta-decision bars; the
        previous intent is decoded from the obs["meta"] one-hot."""
        mask = obs["flags"][:, 0] > 0.5
        prev = obs["meta"].argmax(dim=-1)
        return mask, prev

    # ------------------------------------------------------------------ #
    # PolicyProtocol API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act(
        self, obs: dict[str, Tensor], carry: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor]:
        """Sample actions for one bar.

        Returns ``(actions, logprobs, value, new_carry)``. Runs under
        ``no_grad`` — PPO recomputes gradients through :meth:`evaluate`, so
        rollouts store detached tensors only.
        """
        obs = self._prep(obs)
        feature = self._features(obs, carry)
        mask, prev = self._meta_context(obs)

        # Meta: sample only on decision bars; elsewhere carry the previous
        # intent forward with a hard-zero logprob (deterministic carry-over
        # contributes nothing to importance ratios).
        meta_dist = Categorical(logits=self.meta_head(feature))
        meta = torch.where(mask, meta_dist.sample(), prev)
        zeros = torch.zeros_like(feature[:, 0])
        lp_meta = torch.where(mask, meta_dist.log_prob(meta), zeros)

        trade_dist = Categorical(logits=self.trade_head(feature))
        trade = trade_dist.sample()

        actions: dict[str, Tensor] = {"meta": meta, "trade": trade}
        logprobs: dict[str, Tensor] = {
            "meta": lp_meta, "trade": trade_dist.log_prob(trade)}
        for key in CONTINUOUS_KEYS:
            dist = self._beta_dist(key, feature)
            # Clamp BEFORE log_prob so the stored density is the density of
            # the action actually emitted (and finite at the boundaries).
            u = dist.sample().clamp(_U_EPS, 1.0 - _U_EPS)
            actions[key] = ACTION_EPS + (1.0 - ACTION_EPS) * u
            # Jacobian correction: log p_A(a) = log p_U(u) − log(1 − ε).
            logprobs[key] = dist.log_prob(u) - math.log1p(-ACTION_EPS)

        value = self.value_head(feature).squeeze(-1)
        return actions, logprobs, value, feature

    def evaluate(
        self, obs: dict[str, Tensor], actions: dict[str, Tensor], carry: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        """Log-probs / entropy / value of *given* actions, with gradients.

        Applies the same meta masking as :meth:`act`: on non-decision bars
        the meta logprob is exactly zero (``torch.where`` also zeroes its
        gradient there) and the meta entropy is excluded from the total.

        Returns ``(logprobs, entropy, value, new_carry)`` where entropy is
        the per-sample sum over active heads ``[B]``.
        """
        obs = self._prep(obs)
        feature = self._features(obs, carry)
        mask, _ = self._meta_context(obs)
        mask_f = mask.to(feature.dtype)

        meta_dist = Categorical(logits=self.meta_head(feature))
        zeros = torch.zeros_like(feature[:, 0])
        lp_meta = torch.where(
            mask, meta_dist.log_prob(actions["meta"].long()), zeros)

        trade_dist = Categorical(logits=self.trade_head(feature))
        lp_trade = trade_dist.log_prob(actions["trade"].long())

        logprobs: dict[str, Tensor] = {"meta": lp_meta, "trade": lp_trade}
        entropy = mask_f * meta_dist.entropy() + trade_dist.entropy()
        for key in CONTINUOUS_KEYS:
            dist = self._beta_dist(key, feature)
            # Invert the affine rescale, clamped for boundary safety; the
            # jacobian correction mirrors act() exactly.
            u = (actions[key].to(feature.dtype) - ACTION_EPS) / (1.0 - ACTION_EPS)
            u = u.clamp(_U_EPS, 1.0 - _U_EPS)
            logprobs[key] = dist.log_prob(u) - math.log1p(-ACTION_EPS)
            # H(A) = H(U) + log(1 − ε): constant shift, kept for honesty.
            entropy = entropy + dist.entropy() + math.log1p(-ACTION_EPS)

        value = self.value_head(feature).squeeze(-1)
        return logprobs, entropy, value, feature

    @torch.no_grad()
    def mode(
        self, obs: dict[str, Tensor], carry: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Greedy (deterministic) actions + head distributions, sampling-free.

        Discrete heads: argmax of logits (meta still masked by flags).
        Continuous heads: Beta mean α/(α+β), rescaled to (ε, 1].

        Returns one FLAT dict of tensors (a documented cross-layer
        reconciliation: the backtester consumes the greedy actions, the
        signal engine consumes the distributions, and the test suite
        iterates every value as a tensor — so everything lives at the top
        level rather than in nested sub-dicts):

        * ``meta`` / ``trade`` — greedy discrete actions ``[B]`` (int64),
        * ``size`` / ``stop`` / ``target`` — Beta means rescaled ``[B]``,
        * ``meta_probs [B, len(META_ACTIONS)]`` / ``trade_probs
          [B, len(TRADE_ACTIONS)]`` — full softmax probabilities,
        * ``size_mean`` / ``stop_mean`` / ``target_mean`` ``[B]`` — aliases
          of the greedy continuous actions (the Beta means),
        * ``value [B]`` — critic estimate,
        * ``carry [B, memory_dim]`` — the new recurrent state.

        ``carry=None`` starts from :meth:`initial_carry` (stateless probes,
        e.g. one-shot signal generation on a flat book).
        """
        obs = self._prep(obs)
        if carry is None:
            carry = self.initial_carry(obs["market"].shape[0],
                                       obs["market"].device)
        feature = self._features(obs, carry)
        mask, prev = self._meta_context(obs)

        meta_probs = torch.softmax(self.meta_head(feature), dim=-1)
        trade_probs = torch.softmax(self.trade_head(feature), dim=-1)
        meta = torch.where(mask, meta_probs.argmax(dim=-1), prev)
        out: dict[str, Tensor] = {
            "meta": meta,
            "trade": trade_probs.argmax(dim=-1),
            "meta_probs": meta_probs,
            "trade_probs": trade_probs,
            "value": self.value_head(feature).squeeze(-1),
            "carry": feature,
        }
        for key in CONTINUOUS_KEYS:
            dist = self._beta_dist(key, feature)
            mean = ACTION_EPS + (1.0 - ACTION_EPS) * dist.mean
            out[key] = mean
            out[f"{key}_mean"] = mean
        return out

    def value(
        self, obs: dict[str, Tensor], carry: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Critic value ``[B]`` plus the new carry, WITH gradients.

        The dream phase (Layer 4) regresses the value head through this
        path, so unlike :meth:`act`/:meth:`mode` it must not run under
        ``no_grad`` — gradients flow into the value head and the shared
        torso. ``carry=None`` starts from :meth:`initial_carry`.
        """
        obs = self._prep(obs)
        if carry is None:
            carry = self.initial_carry(obs["market"].shape[0],
                                       obs["market"].device)
        feature = self._features(obs, carry)
        return self.value_head(feature).squeeze(-1), feature
