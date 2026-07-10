"""Continual learning without catastrophic forgetting (Layer 4).

Markets shift regimes; Aether must keep finetuning on new data forever. The
danger is *catastrophic forgetting*: a few thousand SGD steps on the latest
regime can erase competence on every regime that came before. The classic
remedy implemented here is **Elastic Weight Consolidation** (EWC,
Kirkpatrick et al. 2017).

The EWC math
------------
Treat the parameters ``θ*`` learned on old data as a mode of the posterior
``p(θ | old data)``. A Laplace approximation around that mode has precision
given by the Fisher information, whose diagonal is cheap to estimate:

    F_i = E[ (∂L/∂θ_i)² ]            (empirical diagonal Fisher)

Staying probable under the old posterior while training on new data then
means adding a quadratic penalty — stiff springs on the parameters that
mattered, loose springs on the ones that never carried gradient:

    L_total(θ) = L_new(θ) + strength · Σ_i F_i · (θ_i − θ*_i)²

(The textbook form carries λ/2; here the ½ is folded into ``strength``.)
This estimator uses the caller's actual training loss rather than sampled
model labels, i.e. the *empirical* Fisher — the standard, honest shortcut.

The expectation above is over SAMPLES. :meth:`FisherRegularizer.snapshot`
therefore computes the true diagonal only when its ``loss_fn`` returns
per-sample losses (1-D ``[B]``); a scalar batch-mean loss can only yield
the squared MEAN gradient, a ~1/B underestimate near convergence — see the
``snapshot`` docstring for the exact contract and its limits.

When to snapshot
----------------
Call :meth:`FisherRegularizer.snapshot` on the CONVERGED model, using
batches from the OLD regime, immediately BEFORE finetuning on a new regime
(new tickers, post-shock data, an evolved reward). Then add
``regularizer.penalty(model)`` to every finetuning loss. Snapshotting after
finetuning has begun anchors to a point you no longer care about.

Architecture evolution interplay: Layer-4 evolution may change layer shapes
between snapshot and finetune. Parameters whose name is missing or whose
shape changed are skipped with a warning — the penalty protects what still
exists and never blocks growth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from aether.utils.logging import get_logger


class FisherRegularizer:
    """Diagonal-Fisher (EWC) anchor for one model.

    Typical lifecycle::

        reg = FisherRegularizer()
        reg.snapshot(model, loss_fn=lambda: model.loss(next(old_batches)),
                     n_batches=64)
        reg.save("checkpoints/perception/ewc.pt")
        ...
        loss = new_loss + reg.penalty(model, strength=100.0)

    All snapshot state (Fisher diagonal + parameter copies) lives on CPU in
    float32: it is read rarely, and keeping it off the training device costs
    one transfer per penalty call instead of permanent memory.
    """

    def __init__(self) -> None:
        self._fisher: dict[str, Tensor] = {}
        self._theta_star: dict[str, Tensor] = {}
        self._warned: set[str] = set()
        self.logger = get_logger("aether.evolution.continual")

    @property
    def has_snapshot(self) -> bool:
        return bool(self._fisher)

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self, model: nn.Module, loss_fn: Callable[[], Tensor],
                 n_batches: int = 32) -> None:
        """Estimate the diagonal Fisher and anchor the current parameters.

        Parameters
        ----------
        model:
            The trained model to anchor. Left in its incoming train/eval
            mode; gradients are zeroed before returning.
        loss_fn:
            Zero-argument callable returning the loss for ONE batch of
            *old-regime* data (the caller closes over its data iterator).
            It is called ``n_batches`` times; each call must build a fresh
            graph over ``model``'s parameters. TWO contracts, chosen by the
            returned tensor's shape:

            * **1-D per-sample losses ``[B]``** — the CORRECT empirical
              Fisher: ``F_i = E_samples[(∂ℓ/∂θ_i)²]`` is accumulated by
              backpropagating each sample separately (a documented O(B)
              backward cost per batch). Prefer this whenever your loss can
              be computed per sample.
            * **scalar (batch-mean) loss** — HONESTY WARNING, READ THIS:
              a scalar path can only observe the batch-mean gradient, so
              what gets accumulated is the SQUARED MEAN gradient
              ``(E[∂ℓ/∂θ_i])²``, NOT the Fisher's mean of squares
              ``E[(∂ℓ/∂θ_i)²]``. Around a converged optimum the mean
              gradient is near zero while per-sample gradients are not, so
              this UNDERESTIMATES the Fisher by roughly a factor of the
              batch size B (empirically ~87x at B=64 on this stack). It is
              retained because its *relative* magnitudes still rank
              parameter importance usably — but it is only comparable
              across snapshots taken with the SAME batch size, and the
              resulting ``penalty()`` strength is not calibrated in
              Fisher units. No silent rescaling is applied: multiplying by
              B would fake a quantity that was never measured.
        n_batches:
            Number of batches to average over. More batches →
            lower-variance estimate. 32–128 is a sensible range (default
            32).

        Notes
        -----
        Accumulates into CPU float32 buffers, plus a detached CPU copy of
        every parameter (``θ*``). Parameters with ``requires_grad=False``
        are excluded — they cannot drift, so they need no spring. A
        parameter that never receives a gradient keeps Fisher 0 (a free
        parameter, unpenalized).
        """
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {n_batches}")
        named = [(name, p) for name, p in model.named_parameters()
                 if p.requires_grad]
        fisher = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                  for name, p in named}
        used_scalar_path = False
        for _ in range(int(n_batches)):
            model.zero_grad(set_to_none=True)
            loss = loss_fn()
            if loss.dim() == 0:
                # Scalar (batch-mean) path: accumulates the SQUARED MEAN
                # gradient — see the loss_fn contract above for why this
                # is a biased (≈1/B) stand-in for the true Fisher.
                used_scalar_path = True
                loss.backward()
                for name, p in named:
                    if p.grad is not None:
                        fisher[name] += p.grad.detach().float().pow(2).cpu()
            elif loss.dim() == 1:
                # Per-sample path: the honest empirical Fisher
                # F_i = mean_b (∂ℓ_b/∂θ_i)², one backward per sample.
                bsz = int(loss.shape[0])
                if bsz == 0:
                    raise ValueError("loss_fn returned an empty per-sample "
                                     "loss tensor")
                for b in range(bsz):
                    model.zero_grad(set_to_none=True)
                    loss[b].backward(retain_graph=b < bsz - 1)
                    for name, p in named:
                        if p.grad is not None:
                            fisher[name] += (p.grad.detach().float().pow(2)
                                             .cpu() / float(bsz))
            else:
                raise ValueError(
                    "loss_fn must return a scalar (batch-mean) loss or a "
                    f"1-D per-sample loss tensor; got shape "
                    f"{tuple(loss.shape)}")
        model.zero_grad(set_to_none=True)
        if used_scalar_path:
            self.logger.warning(
                "EWC snapshot used the SCALAR loss path: the accumulated "
                "quantity is the squared BATCH-MEAN gradient, ~1/B of the "
                "true diagonal Fisher — comparable only across identical "
                "batch sizes. Return per-sample losses from loss_fn for "
                "the honest estimator.")

        self._fisher = {name: buf / float(n_batches)
                        for name, buf in fisher.items()}
        self._theta_star = {name: p.detach().float().cpu().clone()
                            for name, p in named}
        self._warned.clear()
        total = sum(f.sum().item() for f in self._fisher.values())
        self.logger.info(
            "EWC snapshot: %d params anchored over %d batches "
            "(total Fisher mass %.3e)",
            len(self._fisher), n_batches, total,
        )

    # ------------------------------------------------------------------ #
    # Penalty
    # ------------------------------------------------------------------ #

    def penalty(self, model: nn.Module, strength: float = 1.0) -> Tensor:
        """``strength · Σ_i F_i (θ_i − θ*_i)²`` as a scalar on the model's
        device — add it to the finetuning loss.

        Exactly zero at the snapshot point. Parameters that are new since
        the snapshot, or whose shape changed (architecture evolved), are
        skipped with a one-time warning per parameter; parameters that
        disappeared are simply no longer constrained.
        """
        if not self._fisher:
            raise RuntimeError("penalty() called before snapshot() / load()")
        total: Tensor | None = None
        device = torch.device("cpu")
        for name, p in model.named_parameters():
            device = p.device
            if not p.requires_grad:
                continue
            anchor = self._theta_star.get(name)
            if anchor is None:
                self._warn_once(name, "not in snapshot (new parameter)")
                continue
            if anchor.shape != p.shape:
                self._warn_once(
                    name,
                    f"shape changed {tuple(anchor.shape)} -> {tuple(p.shape)}"
                    " (architecture evolved)",
                )
                continue
            fisher = self._fisher[name].to(device)
            term = (fisher * (p.float() - anchor.to(device)).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            # Nothing matched — a fully evolved model. Penalty is honestly 0.
            return torch.zeros((), device=device)
        return strength * total

    def _warn_once(self, name: str, reason: str) -> None:
        if name not in self._warned:
            self._warned.add(name)
            self.logger.warning("EWC: skipping %s — %s", name, reason)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Write Fisher + anchors to a ``.pt`` file (atomic tmp + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "fisher": self._fisher,
            "theta_star": self._theta_star,
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    def load(self, path: str | Path) -> None:
        """Restore a snapshot written by :meth:`save` (CPU tensors only,
        loaded with ``weights_only=True`` — the payload is pure tensors)."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self._fisher = dict(payload["fisher"])
        self._theta_star = dict(payload["theta_star"])
        self._warned.clear()
        self.logger.info("EWC snapshot loaded: %d params from %s",
                         len(self._fisher), path)
