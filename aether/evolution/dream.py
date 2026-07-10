"""Imagination-phase training inside the world model's dreams (Layer 4).

HONEST v1 SCOPE — READ THIS BEFORE TRUSTING RESULTS
---------------------------------------------------
The world model's dynamics decode imagined futures into PERCEPTION
EMBEDDINGS, not prices. Without prices there are no fills, no PnL, and
therefore **no imagined reward**. This phase does NOT train action
selection, does NOT improve the policy's trading decisions, and produces
NO evidence about profitability. What it *does* do:

* **Value-consistency training** — the policy's value head is regressed
  toward the (stop-gradient) discounted mean of its own value estimates
  along imagined latent trajectories. A value function whose estimate at
  ``t`` disagrees wildly with its estimates over plausible futures
  ``t+1..H`` is noisy; smoothing it over the dynamics model's futures is
  classical self-consistency regularization.
* **Torso representation shaping** — the same loss backpropagates through
  the policy's shared torso, pulling its state abstraction toward features
  that are stable under imagined dynamics. Action heads receive NO gradient
  (the loss never touches them).

Full dream-policy training (imagined rewards → policy-gradient through
imagination, Dreamer-style) arrives when the world model gains a price
decoder. Until then, treat this phase as a value/representation stabilizer
to interleave with real-environment PPO — nothing more.

Mechanics
---------
Observations fed to the policy during dreams follow the Layer-3 obs
contract (:data:`aether.decision.interfaces.OBS_KEYS`): ``obs['market']``
is the dynamics model's DECODED embedding at each imagined step; every
other entry is neutral — zeros for ``unc``/``position``/``portfolio``/
``clock``/``flags``, and ``meta`` a one-hot of ``stand_aside``. Dreaming a
flat book is deliberate: position dynamics live in the real environment
(Layer 3), and pretending otherwise here would be fiction.

The dynamics model is a hard invariant: its parameters are frozen
(``requires_grad`` toggled off and restored afterwards) and a parameter
fingerprint is asserted unchanged after :meth:`DreamPhase.run`. Imagined
tensors are detached before entering the policy, so no gradient can reach
the world model even by accident. The dynamics' ``imagine`` is called with
gradients enabled (the contract requires it to support that); v1 simply
does not need world-model gradients because nothing upstream of the policy
is trained.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aether.decision.interfaces import META_ACTIONS
from aether.utils.logging import get_logger
from aether.worldmodel.interfaces import LatentState

#: Neutral (all-zero) observation entries and their per-env dims, per the
#: Layer-3 obs contract in ``aether.decision.interfaces``. ``market`` and
#: ``meta`` are built separately.
_NEUTRAL_OBS_DIMS: dict[str, int] = {
    "unc": 3,        # aleatoric, epistemic, anomaly
    "position": 5,   # flat book
    "portfolio": 3,
    "clock": 2,
    "flags": 1,      # never a meta-decision bar while dreaming
}


class DreamPhase:
    """Train the policy's value head + torso on imagined trajectories.

    Parameters
    ----------
    dynamics:
        A ``worldmodel.dynamics.LatentDynamics`` (or anything honoring the
        contract): ``imagine(start, horizon, n) -> ImaginedRollout`` whose
        ``embeds`` are decoded perception-space trajectories
        ``[N, B, H, embed_dim]``, callable with autograd enabled. Frozen for
        the duration of :meth:`run`.
    policy:
        A ``decision.policies.HierarchicalPolicy``-shaped module. The value
        estimate is obtained via ``policy.value(obs, carry) -> (value,
        carry)`` when available, else ``policy.act(obs, carry)``'s third
        output. Either path MUST allow gradients to flow into the value
        head and torso (no internal ``no_grad`` around the value path).
    horizon:
        Imagination depth in bars (≥ 2: consistency needs a future).
    n_rollouts:
        Stochastic futures per start state; the loss averages over all.
    gamma:
        Discount used in the future-value weighting.
    log_jsonl:
        JSONL sink for per-step metrics (``None`` = console only).
    """

    def __init__(self, dynamics: nn.Module, policy: nn.Module,
                 horizon: int = 15, n_rollouts: int = 32,
                 gamma: float = 0.99,
                 log_jsonl: str | None = "runs/dream.jsonl") -> None:
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2, got {horizon}")
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")
        self.dynamics = dynamics
        self.policy = policy
        self.horizon = int(horizon)
        self.n_rollouts = int(n_rollouts)
        self.gamma = float(gamma)
        self.logger = get_logger("aether.evolution.dream", log_jsonl)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _fingerprint(self) -> Tensor:
        """Cheap invariant over dynamics weights: per-parameter (sum, |sum|)
        pairs. ``requires_grad=False`` is the real guard; this catches
        in-place edits too."""
        with torch.no_grad():
            parts = []
            for p in self.dynamics.parameters():
                d = p.detach().double()
                parts.append(torch.stack([d.sum(), d.abs().sum()]))
            if not parts:
                return torch.zeros(0, dtype=torch.float64)
            return torch.stack(parts).cpu()

    def _neutral_obs(self, batch: int, device: torch.device,
                     dtype: torch.dtype) -> dict[str, Tensor]:
        """All obs entries except ``market``: zeros + stand_aside one-hot."""
        obs = {key: torch.zeros(batch, dim, device=device, dtype=dtype)
               for key, dim in _NEUTRAL_OBS_DIMS.items()}
        meta = torch.zeros(batch, len(META_ACTIONS), device=device, dtype=dtype)
        meta[:, META_ACTIONS.index("stand_aside")] = 1.0
        obs["meta"] = meta
        return obs

    def _policy_value(self, obs: dict[str, Tensor], carry: Any
                      ) -> tuple[Tensor, Any]:
        """Value estimate for one dream step, threading the policy carry."""
        value_fn = getattr(self.policy, "value", None)
        if callable(value_fn):
            out = value_fn(obs, carry)
            if isinstance(out, tuple):
                return out[0], out[-1]
            return out, carry
        _actions, _logprobs, value, carry = self.policy.act(obs, carry)
        return value, carry

    def _initial_carry(self, batch: int) -> Any:
        init = getattr(self.policy, "initial_carry", None)
        return init(batch) if callable(init) else None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, start_states: LatentState, optimizer: torch.optim.Optimizer,
            steps: int) -> list[dict[str, float]]:
        """Run ``steps`` dream-training updates from ``start_states``.

        Each step: (1) imagine ``n_rollouts`` fresh stochastic futures of
        ``horizon`` bars from every start state, (2) decode them into
        dream observations, (3) evaluate the policy's value along each
        trajectory, (4) minimize

            L = MSE( V(s_t),  sg[ Σ_{k>t} γ^{k−t−1} V(s_k) / Σ γ^{k−t−1} ] )

        averaged over rollouts, batch, and ``t ∈ [0, H−2]`` (``sg`` =
        stop-gradient). ``optimizer`` should hold the policy's parameters;
        only the value head and torso ever receive gradient.

        Returns per-step history: ``{"step", "value_consistency",
        "value_mean"}``. Raises ``RuntimeError`` if the dynamics weights
        changed (they are frozen and fingerprinted).
        """
        # ---- freeze dynamics (restore flags afterwards, even on error) ----
        prior_flags = {name: p.requires_grad
                       for name, p in self.dynamics.named_parameters()}
        for p in self.dynamics.parameters():
            p.requires_grad_(False)
        fingerprint = self._fingerprint()

        history: list[dict[str, float]] = []
        try:
            for step in range(int(steps)):
                rollout = self.dynamics.imagine(
                    start_states, self.horizon, self.n_rollouts)
                # Detach: dreams are training DATA for the policy; no
                # gradient may flow back into the (frozen) world model.
                market = rollout.embeds.detach()            # [N, B, H, E]
                n, b, h, e = market.shape
                flat = market.reshape(n * b, h, e)
                neutral = self._neutral_obs(n * b, flat.device, flat.dtype)

                carry = self._initial_carry(n * b)
                per_step_values: list[Tensor] = []
                for t in range(h):
                    obs = dict(neutral)
                    obs["market"] = flat[:, t]
                    value, carry = self._policy_value(obs, carry)
                    per_step_values.append(value.reshape(-1))
                values = torch.stack(per_step_values, dim=1)  # [N·B, H]

                # ---- discounted mean of FUTURE values, stop-gradient ------
                with torch.no_grad():
                    v = values.detach()
                    targets = torch.zeros(n * b, h - 1, device=v.device,
                                          dtype=v.dtype)
                    running_num = torch.zeros(n * b, device=v.device,
                                              dtype=v.dtype)
                    running_den = 0.0
                    for t in range(h - 2, -1, -1):
                        running_num = v[:, t + 1] + self.gamma * running_num
                        running_den = 1.0 + self.gamma * running_den
                        targets[:, t] = running_num / running_den

                loss = F.mse_loss(values[:, : h - 1], targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                entry = {
                    "step": float(step),
                    "value_consistency": float(loss.detach()),
                    "value_mean": float(values.detach().mean()),
                }
                history.append(entry)
                self.logger.info(
                    "dream step %4d/%d  value_consistency=%.6f  value_mean=%.4f",
                    step + 1, steps, entry["value_consistency"],
                    entry["value_mean"],
                    extra={"aether_dream_step": step,
                           "aether_value_consistency": entry["value_consistency"],
                           "aether_value_mean": entry["value_mean"]},
                )
        finally:
            for name, p in self.dynamics.named_parameters():
                p.requires_grad_(prior_flags[name])

        after = self._fingerprint()
        if not torch.equal(fingerprint, after):
            raise RuntimeError(
                "DreamPhase invariant violated: dynamics weights changed "
                "during imagination training"
            )
        return history
