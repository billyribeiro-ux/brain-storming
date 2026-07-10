"""RSSM-style latent dynamics over perception embeddings.

This module implements the :class:`~aether.worldmodel.interfaces.LatentDynamics`
contract: a Recurrent State-Space Model (RSSM, Hafner et al. 2019 "Dream to
Control") that learns how the market state — as summarized by the perception
layer's fused embedding — evolves from one bar to the next, and can then roll
that learned dynamic forward to *imagine* plausible futures.

Architecture (one filtering step)
---------------------------------
::

                       stoch_t ─────────────┐
                                            ▼
    deter_t ──────────────────────► GRUCell ──► deter_{t+1}
                                                  │
                       ┌──────────────────────────┤
                       ▼                          ▼
                 Prior MLP                  Posterior MLP ◄── embed_{t+1}
              (deter → μ_p, σ_p)       (deter ⊕ embed → μ_q, σ_q)
                                                  │
                                     stoch_{t+1} ~ N(μ_q, σ_q)   (rsample)

    feature_t = deter_t ⊕ stoch_t  ──► Decoder MLP ──► embed_hat_t

The latent state is the classic RSSM split:

* ``deter`` — a deterministic GRU path that carries long-range context
  reliably (gradients flow through it un-noised),
* ``stoch`` — a diagonal-Gaussian stochastic state that forces the model to
  represent genuine uncertainty about the next market state instead of
  averaging futures into a blur.

Action-free by design
---------------------
Unlike the control-oriented RSSM, the transition takes **no action input**
(``deter_{t+1} = GRU(stoch_t, deter_t)``). The contract's rationale: at
Aether's size its own orders do not move AAPL/SPY-class instruments, so the
market evolves independently of the agent. Position/PnL bookkeeping is exact
in the Layer-3 environment; modeling it here would only add approximation
error to imagination rollouts.

The math of training (``loss``)
-------------------------------
Filtering a sequence produces, at each step, a *prior* ``p(s_t | h_t)``
(what the model expected before seeing the bar) and a *posterior*
``q(s_t | h_t, e_t)`` (belief after seeing the perception embedding
``e_t``). Training minimizes an ELBO-style objective::

    recon = MSE( decode(deter ⊕ stoch_posterior), e )              (fit data)
    kl    = mean_t clamp( Σ_dims KL(q_t ‖ p_t) − free_nats, 0 )    (regularize)
    total = recon + kl_beta · kl

The KL term is what makes imagination honest: it drags the prior toward the
posterior so open-loop rollouts (which only have the prior) match the
filtered dynamics, while simultaneously keeping the posterior from encoding
information the prior could never predict.

Free-nats convention (documented choice)
----------------------------------------
``free_nats`` is applied as a **per-step budget on the dim-summed KL**::

    kl_per_step = KL(q ‖ p).sum(dim=-1)                # [B, T]
    kl          = clamp(kl_per_step − free_nats, min=0).mean()

i.e. each timestep may spend up to ``free_nats`` nats of total divergence
across all ``stoch_dim`` dimensions before any penalty applies. This is the
original Dreamer convention and prevents posterior collapse (the optimizer
cannot earn reward by squeezing an already-tiny KL to zero). The alternative
reading — ``free_nats`` *per dimension* — corresponds to a budget of
``free_nats · stoch_dim`` and is deliberately NOT used; with the default
config (free_nats=1, stoch_dim=32) that would exempt 32 nats per step, far
more slack than the dynamics should be given.

Determinism & device policy
---------------------------
All sampling goes through ``torch.randn_like`` (reparameterized), which
draws from torch's global generator — so runs are reproducible under
``torch.manual_seed``. Nothing here names a device: tensors are created
with ``*_like`` factories or explicit ``device=`` derived from inputs or
parameters, and no in-place operation touches a tensor that autograd needs.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence

from aether.worldmodel.interfaces import (
    DynamicsConfig,
    ImaginedRollout,
    LatentDynamics,
    LatentState,
    ObserveResult,
)

#: Floor added to every predicted standard deviation (after softplus).
#: Guarantees a strictly positive σ so log-densities, KL terms and the
#: reparameterized sample are always finite, and bounds the sharpness the
#: prior/posterior can claim (an exactly-zero σ would make KL explode).
_MIN_STD: float = 1e-3


class RSSM(LatentDynamics):
    """Action-free Recurrent State-Space Model over perception embeddings.

    Parameters
    ----------
    cfg:
        :class:`~aether.worldmodel.interfaces.DynamicsConfig`. ``embed_dim``
        must equal the perception model's fused dimension; the rest sizes
        the deterministic/stochastic paths and the loss weights.

    The four submodules:

    * ``cell`` — ``GRUCell(stoch_dim → deter_dim)``; the deterministic
      transition ``deter_{t+1} = GRU(stoch_t, deter_t)``,
    * ``prior_net`` — ``deter → (μ, σ_raw)`` of the transition prior,
    * ``post_net`` — ``deter ⊕ embed → (μ, σ_raw)`` of the filtering
      posterior,
    * ``decoder`` — ``deter ⊕ stoch → embed_hat``, mapping latent features
      back to perception space (the reconstruction target and the output
      of imagination rollouts).
    """

    def __init__(self, cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Deterministic path. The GRU consumes only the previous stochastic
        # state (action-free); its hidden state IS `deter`.
        self.cell = nn.GRUCell(cfg.stoch_dim, cfg.deter_dim)

        # Prior: what the next stochastic state should look like given only
        # the deterministic history — this is all imagination has.
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.deter_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )

        # Posterior: refines the prior with the actually-observed embedding.
        self.post_net = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )

        # Decoder: latent feature (deter ⊕ stoch) back to perception space.
        self.decoder = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.stoch_dim, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_stats(raw: Tensor) -> tuple[Tensor, Tensor]:
        """Split an MLP head's output into ``(μ, σ)``.

        ``σ = softplus(raw) + _MIN_STD`` — softplus keeps the map smooth
        (unlike ``exp`` it does not explode for large activations early in
        training) and the additive floor keeps σ strictly positive.
        """
        mu, std_raw = raw.chunk(2, dim=-1)
        return mu, F.softplus(std_raw) + _MIN_STD

    @staticmethod
    def _rsample(mu: Tensor, std: Tensor) -> Tensor:
        """Reparameterized Gaussian sample ``μ + σ·ε``, ``ε ~ N(0, I)``.

        The pathwise (reparameterization) form keeps the sample
        differentiable w.r.t. μ and σ, which both the ELBO and the dream
        phase (``imagine(..., grad=True)``) require. Uses the global torch
        generator ⇒ deterministic under ``torch.manual_seed``.
        """
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------ #
    # Contract API
    # ------------------------------------------------------------------ #

    def initial(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> LatentState:
        """All-zeros latent state for ``batch_size`` sequences.

        Zeros are the canonical RSSM start: the GRU's null hidden state plus
        a degenerate stochastic state. The first filtering step immediately
        replaces both with data-driven values. ``device`` defaults to
        wherever the model's parameters live (device-agnostic policy).
        """
        if device is None:
            device = next(self.parameters()).device
        cfg = self.cfg
        return LatentState(
            deter=torch.zeros(batch_size, cfg.deter_dim, device=device),
            stoch=torch.zeros(batch_size, cfg.stoch_dim, device=device),
        )

    def observe(self, embeds: Tensor) -> ObserveResult:
        """Posterior-filter an embedding sequence.

        Parameters
        ----------
        embeds:
            ``[B, T, embed_dim]`` perception embeddings, one per bar, in
            time order (historical replay — no look-ahead by construction).

        Returns
        -------
        ObserveResult
            ``states`` holds the filtered trajectory with every field
            stacked to ``[B, T, dim]``; the four ``*_mu`` / ``*_std``
            tensors are ``[B, T, stoch_dim]``.

        Filtering is sequential by necessity — ``deter_{t}`` depends on the
        *sampled* posterior ``stoch_{t-1}`` — so the loop over T cannot be
        parallelized. During filtering the **posterior** sample drives the
        next transition (the prior is only computed for the KL target):
        this is closed-loop filtering, the regime under which the model is
        trained to reconstruct.
        """
        if embeds.dim() != 3 or embeds.shape[-1] != self.cfg.embed_dim:
            raise ValueError(
                f"observe() expects [B, T, {self.cfg.embed_dim}] embeddings, "
                f"got {tuple(embeds.shape)}"
            )
        batch, horizon = embeds.shape[0], embeds.shape[1]
        state = self.initial(batch, embeds.device)

        deters: list[Tensor] = []
        stochs: list[Tensor] = []
        post_mus: list[Tensor] = []
        post_stds: list[Tensor] = []
        prior_mus: list[Tensor] = []
        prior_stds: list[Tensor] = []

        for t in range(horizon):
            # Deterministic transition from the previous (posterior) state.
            deter = self.cell(state.stoch, state.deter)
            # Prior — prediction before seeing bar t (KL target only here).
            prior_mu, prior_std = self._split_stats(self.prior_net(deter))
            # Posterior — belief after fusing the observed embedding.
            post_mu, post_std = self._split_stats(
                self.post_net(torch.cat([deter, embeds[:, t]], dim=-1))
            )
            stoch = self._rsample(post_mu, post_std)
            state = LatentState(deter=deter, stoch=stoch)

            deters.append(deter)
            stochs.append(stoch)
            post_mus.append(post_mu)
            post_stds.append(post_std)
            prior_mus.append(prior_mu)
            prior_stds.append(prior_std)

        return ObserveResult(
            states=LatentState(
                deter=torch.stack(deters, dim=1),      # [B, T, deter_dim]
                stoch=torch.stack(stochs, dim=1),      # [B, T, stoch_dim]
            ),
            post_mu=torch.stack(post_mus, dim=1),      # [B, T, stoch_dim]
            post_std=torch.stack(post_stds, dim=1),
            prior_mu=torch.stack(prior_mus, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
        )

    def imagine(
        self,
        start: LatentState,
        horizon: Optional[int] = None,
        n: int = 1,
        grad: bool = False,
    ) -> ImaginedRollout:
        """Roll ``n`` independent stochastic futures from one start state.

        Parameters
        ----------
        start:
            Latent state with ``[B, dim]`` fields (e.g. the last filtered
            step of :meth:`observe`). Not modified.
        horizon:
            Steps to imagine; defaults to ``cfg.horizon``.
        n:
            Number of independent rollouts per batch element. Each draws
            its own noise at every step, so the fan of futures reflects the
            prior's learned uncertainty.
        grad:
            Gradients are recorded only when ``grad=True`` **and** the
            module is in training mode (the Dreamer-style "dream phase",
            where a policy is optimized through imagined trajectories).
            Everywhere else — evaluation, counterfactual probes, dashboard
            fans — the rollout runs under ``no_grad`` so a long horizon
            never builds an autograd graph by accident.

        Returns
        -------
        ImaginedRollout
            ``embeds [N, B, H, embed_dim]`` decoded trajectories plus the
            latent paths ``deter``/``stoch`` shaped ``[N, B, H, dim]``.

        Implementation note: the ``n`` rollouts are folded into the batch
        dimension (``[n·B, dim]``, sample-major) so the whole fan advances
        in one GRU/MLP call per step; independence comes from each row
        drawing its own ``ε``.
        """
        if horizon is None:
            horizon = self.cfg.horizon
        if horizon < 1 or n < 1:
            raise ValueError(f"imagine() needs horizon ≥ 1 and n ≥ 1, "
                             f"got horizon={horizon}, n={n}")
        batch = start.deter.shape[0]

        wants_grad = grad and self.training
        ctx: ContextManager = nullcontext() if wants_grad else torch.no_grad()
        with ctx:
            # Tile the start state n times: row i·B + b is rollout i of
            # batch element b, which is exactly the layout a final
            # .view(n, B, ...) unpacks.
            deter = start.deter.repeat(n, 1)               # [n·B, deter_dim]
            stoch = start.stoch.repeat(n, 1)               # [n·B, stoch_dim]

            deters: list[Tensor] = []
            stochs: list[Tensor] = []
            for _ in range(horizon):
                # Open-loop: transition, then sample from the PRIOR — there
                # are no observations in the future, so the prior is the
                # model's entire knowledge of what comes next.
                deter = self.cell(stoch, deter)
                mu, std = self._split_stats(self.prior_net(deter))
                stoch = self._rsample(mu, std)
                deters.append(deter)
                stochs.append(stoch)

            deter_path = torch.stack(deters, dim=1)        # [n·B, H, deter]
            stoch_path = torch.stack(stochs, dim=1)        # [n·B, H, stoch]
            feature = torch.cat([deter_path, stoch_path], dim=-1)
            embeds = self.decode(feature)                  # [n·B, H, embed]

            return ImaginedRollout(
                embeds=embeds.view(n, batch, horizon, self.cfg.embed_dim),
                deter=deter_path.view(n, batch, horizon, self.cfg.deter_dim),
                stoch=stoch_path.view(n, batch, horizon, self.cfg.stoch_dim),
            )

    def decode(self, feature: Tensor) -> Tensor:
        """Map latent features ``[..., deter_dim + stoch_dim]`` back to
        perception space ``[..., embed_dim]``.

        Pure MLP ⇒ arbitrary leading dimensions pass through unchanged.
        """
        return self.decoder(feature)

    def loss(self, embeds: Tensor) -> dict[str, Tensor]:
        """One training objective evaluation on a batch of sequences.

        Parameters
        ----------
        embeds:
            ``[B, T, embed_dim]`` embedding sequences (session-contiguous;
            the training script guarantees windows never cross a session
            boundary, so the GRU never has to explain an overnight gap as
            if it were one bar).

        Returns
        -------
        dict
            ``total`` — the scalar to optimize,
            ``recon`` — mean-squared reconstruction error of the decoded
            posterior features against the true embeddings,
            ``kl`` — the free-nats-clamped KL that actually enters
            ``total`` (so ``total == recon + kl_beta·kl`` holds exactly),
            ``kl_raw`` — the unclamped KL, detached, for monitoring
            posterior/prior agreement (extra key beyond the contract).

        The free-nats clamp is applied to the **dim-summed per-step KL**
        (see module docstring for the full rationale): each timestep gets a
        budget of ``cfg.free_nats`` nats across all stochastic dimensions
        before contributing to the loss.
        """
        result = self.observe(embeds)

        # --- Reconstruction: decode the filtered (posterior) features ------
        recon_hat = self.decode(result.states.feature)     # [B, T, embed_dim]
        recon = F.mse_loss(recon_hat, embeds)

        # --- KL(q ‖ p) between diagonal Gaussians, analytic per dim --------
        posterior = Normal(result.post_mu, result.post_std)
        prior = Normal(result.prior_mu, result.prior_std)
        kl_per_dim = kl_divergence(posterior, prior)       # [B, T, stoch_dim]
        kl_per_step = kl_per_dim.sum(dim=-1)               # [B, T]
        # Free-nats: clamp is elementwise per (sequence, step) — a step that
        # is already cheap cannot subsidize an expensive one.
        kl_clamped = torch.clamp(
            kl_per_step - self.cfg.free_nats, min=0.0
        ).mean()

        total = recon + self.cfg.kl_beta * kl_clamped
        return {
            "total": total,
            "recon": recon,
            "kl": kl_clamped,
            "kl_raw": kl_per_step.mean().detach(),
        }

    @property
    def num_parameters(self) -> int:
        """Total number of parameters (buffers excluded)."""
        return sum(p.numel() for p in self.parameters())
