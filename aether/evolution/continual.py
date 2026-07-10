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
            Zero-argument callable returning the scalar training loss for
            ONE batch of *old-regime* data (the caller closes over its data
            iterator). It is called ``n_batches`` times; each call must
            build a fresh graph over ``model``'s parameters.
        n_batches:
            Number of batches to average ``grad²`` over. More batches →
            lower-variance Fisher. 32–128 is a sensible range (default 32).

        Notes
        -----
        Accumulates ``F_i = (1/N) Σ_batches (∂L/∂θ_i)²`` per parameter into
        CPU float32 buffers, plus a detached CPU copy of every parameter
        (``θ*``). Parameters with ``requires_grad=False`` are excluded —
        they cannot drift, so they need no spring. A parameter that never
        receives a gradient keeps Fisher 0 (a free parameter, unpenalized).
        """
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {n_batches}")
        named = [(name, p) for name, p in model.named_parameters()
                 if p.requires_grad]
        fisher = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                  for name, p in named}
        for _ in range(int(n_batches)):
            model.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            for name, p in named:
                if p.grad is not None:
                    fisher[name] += p.grad.detach().float().pow(2).cpu()
        model.zero_grad(set_to_none=True)

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
