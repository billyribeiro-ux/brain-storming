"""Self-supervised trainer for the perception model.

This module owns the full optimization loop for
:class:`aether.perception.model.PerceptionModel`:

* **AdamW with decay hygiene** — weight decay is applied ONLY to genuine
  weight matrices. Biases, normalization gains, embedding rows, and learned
  tokens/latents are exempt (see :func:`build_param_groups` for the why).
* **Warmup + cosine LR schedule** — linear warmup for ``cfg.warmup_steps``
  optimizer steps, then a cosine decay that lands on 5% of the peak LR
  exactly at ``cfg.max_steps``.
* **Mixed precision** — ``torch.autocast`` (bfloat16 where the GPU supports
  it, float16 otherwise) plus a ``GradScaler``, active only when training on
  CUDA with ``cfg.amp``. On CPU everything runs in plain float32.
* **EMA weights** — an exponential moving average of all parameters is
  maintained alongside training; validation and checkpoints use it because
  the EMA point is a smoother, better-generalizing model than any single
  SGD iterate (Polyak averaging).
* **Checkpointing** — ``last.pt`` at every validation, ``best.pt`` whenever
  the validation total loss improves. Checkpoints contain *everything*
  (model, EMA, optimizer, scheduler, scaler, step, config snapshot) so a
  resumed run is bit-for-bit a continuation, not an approximation.
* **Structured logging** — human-readable console lines plus a JSONL stream
  (via :func:`aether.utils.logging.get_logger`) that later self-diagnosis
  layers can mine for loss curves and LR traces.

The trainer is deliberately model-agnostic beyond one contract point: the
model must expose ``self_supervised_losses(batch) -> (dict[str, Tensor],
PerceptionOutput)`` where ``losses["total"]`` is the scalar to optimize.
Batches must implement ``.to(device)`` (see
:class:`aether.perception.interfaces.PerceptionBatch`).

Device policy (project rule #4): nothing in this file ever hardcodes
``.cuda()``. The device is chosen once in ``__init__`` ("cuda" if available,
else "cpu", unless the caller overrides) and everything else derives from it.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager, Iterable, Iterator, TypeVar

import numpy as np
import torch
from torch import Tensor, nn

from aether.perception.interfaces import PerceptionBatch, TrainConfig
from aether.utils.logging import get_logger

T = TypeVar("T")

#: How often (in optimizer steps) to print a human-readable console line.
_CONSOLE_EVERY: int = 50


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def cycle(loader: Iterable[T]) -> Iterator[T]:
    """Endlessly re-iterate ``loader``.

    A *fresh* iterator is created for each pass, so a
    ``torch.utils.data.DataLoader`` with ``shuffle=True`` re-shuffles on
    every epoch boundary (an ``itertools.cycle`` over the first epoch's
    items would freeze one shuffle order forever — subtly wrong for SGD).

    Raises
    ------
    ValueError
        If the underlying loader yields nothing — an infinite busy-loop
        would otherwise hang training silently.
    """
    while True:
        yielded = False
        for item in loader:
            yielded = True
            yield item
        if not yielded:
            raise ValueError("cycle(): the loader produced no batches")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs (all devices) for reproducibility.

    Note this makes *initial conditions* deterministic; bitwise determinism
    on GPU additionally depends on kernel choices outside our control.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA generators


def build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into AdamW groups: weight-decayed vs. not.

    Decay hygiene — the reasoning
    -----------------------------
    Weight decay (L2-toward-zero) is a sensible regularizer for *matrices*
    that mix features: shrinking them bounds the layer's gain and combats
    overfitting. It is actively harmful for:

    * **biases** — decaying a bias just mis-calibrates the layer's operating
      point; it does not reduce model capacity,
    * **normalization gains/offsets** (LayerNorm/BatchNorm/RMSNorm weights)
      — these *re-introduce* scale after normalization by design; pulling
      them to zero fights the normalization itself,
    * **embedding rows** (ticker DNA table) — each row is a lookup value,
      not a mixing matrix; decay would erase identity information for
      rarely-sampled tickers,
    * **learned tokens/latents** (mask token, fusion latent queries,
      positional embeddings) — these are free vectors whose *position* in
      representation space is the whole point; zero is not a meaningful
      prior for them.

    Classification rules, applied in order:

    1. any parameter owned by an ``nn.Embedding`` or a normalization module
       → no decay,
    2. any parameter with fewer than 2 dimensions (biases, norm gains,
       scalar/vector tokens such as the model's ``mask_token``) → no decay,
    3. any parameter whose name contains ``token``, ``latent`` or
       ``pos_emb`` (learned queries / positional tables that are ≥2-D and
       so slip past rule 2) → no decay,
    4. everything else (true weight matrices, conv kernels) → decay.
    """
    # Norm classes to exempt. `_NormBase` covers all BatchNorm/InstanceNorm
    # variants; RMSNorm only exists in newer torch, hence the getattr guard.
    norm_types: tuple[type, ...] = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.batchnorm._NormBase,
        nn.LocalResponseNorm,
    )
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        norm_types = norm_types + (rms_norm,)

    # Rule 1: collect fully-qualified names of params owned by exempt modules.
    exempt_by_module: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag, *norm_types)):
            for p_name, _ in module.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                exempt_by_module.add(full)

    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if (
            name in exempt_by_module            # rule 1
            or param.ndim < 2                   # rule 2
            or "token" in lowered               # rule 3
            or "latent" in lowered
            or "pos_emb" in lowered
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_lr_lambda(warmup_steps: int, max_steps: int, floor: float = 0.05):
    """LR multiplier schedule: linear warmup → cosine decay to ``floor``.

    Returns a function ``step -> multiplier`` for
    :class:`torch.optim.lr_scheduler.LambdaLR`, where ``step`` counts
    completed optimizer steps (LambdaLR's ``last_epoch``).

    * steps ``0 .. warmup-1``: multiplier ramps ``1/warmup .. 1`` — starting
      at a nonzero value so the very first update is not wasted,
    * steps ``warmup .. max_steps``: half-cosine from 1 down to ``floor``
      (5% of peak by default) reached exactly at ``max_steps``,
    * beyond ``max_steps`` (shouldn't happen; training stops there): held
      at ``floor``.
    """
    warmup_steps = max(0, int(warmup_steps))
    span = max(1, int(max_steps) - warmup_steps)  # cosine phase length

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
        return floor + (1.0 - floor) * cosine

    return lr_lambda


# --------------------------------------------------------------------------- #
# Parameter EMA
# --------------------------------------------------------------------------- #

class ParameterEMA:
    """Exponential moving average of a model's parameters (Polyak averaging).

    ``shadow[p] ← decay·shadow[p] + (1−decay)·p`` after every optimizer
    step. The shadow copy is kept as detached **float32** tensors on the
    model's own device (CPU-friendly: on a CPU run everything already lives
    on CPU; checkpoints always store the shadow on CPU for portability).
    float32 is used even under mixed precision so that thousands of tiny
    ``(1−decay)`` increments don't get rounded away in half precision.

    Buffers (e.g. the model's anomaly EMA statistics) are *not* averaged:
    they are already running statistics, and averaging them with stale
    values would corrupt them. During :meth:`swap` the live buffers remain
    in place.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        # Detached float32 clones, keyed by parameter name for exact
        # save/load round-trips regardless of parameter ordering.
        self.shadow: dict[str, Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """One EMA step. Call after ``optimizer.step()``."""
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            # `.float()` guards against a half-precision model; lerp-style
            # update keeps everything out-of-place-safe for the shadow.
            self.shadow[name].mul_(self.decay).add_(
                param.detach().float(), alpha=one_minus
            )

    @contextmanager
    def swap(self, model: nn.Module) -> Iterator[None]:
        """Context manager: EMA weights in, training weights restored after.

        Used for validation and for exporting checkpoints' ``best`` view.
        The training weights are backed up, overwritten by the shadow (cast
        back to each parameter's dtype/device), and restored on exit — even
        if the body raises.
        """
        backup: dict[str, Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(param.device, param.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(backup[name])

    # -- (de)serialization -------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        """CPU snapshot: portable across devices and small to store."""
        return {
            "decay": self.decay,
            "shadow": {name: t.detach().cpu() for name, t in self.shadow.items()},
        }

    def load_state_dict(
        self, state: dict[str, Any], device: torch.device | str = "cpu"
    ) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name: t.to(device=device, dtype=torch.float32)
            for name, t in state["shadow"].items()
        }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class PerceptionTrainer:
    """Runs the self-supervised optimization loop for a perception model.

    Parameters
    ----------
    model:
        A :class:`~aether.perception.model.PerceptionModel` (or anything
        exposing ``self_supervised_losses``). Moved to ``device`` here.
    train_loader / val_loader:
        Iterables yielding :class:`PerceptionBatch`. The train loader is
        cycled indefinitely (see :func:`cycle`); the val loader is consumed
        in full at every validation.
    cfg:
        :class:`~aether.perception.interfaces.TrainConfig`.
    device:
        Explicit device string (``"cpu"``, ``"cuda"``, ``"cuda:1"`` …).
        ``None`` auto-selects CUDA when available, else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[PerceptionBatch],
        val_loader: Iterable[PerceptionBatch],
        cfg: TrainConfig,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- Optimizer with decay hygiene (see build_param_groups) --------
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.weight_decay),
            lr=cfg.lr,
            betas=(0.9, 0.95),  # slightly low β2: noisy SSL losses benefit
            weight_decay=cfg.weight_decay,  # per-group values override this
        )

        # ---- LR schedule: warmup → cosine to 5% of peak --------------------
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_lr_lambda(cfg.warmup_steps, cfg.max_steps)
        )

        # ---- Mixed precision -----------------------------------------------
        # Autocast + GradScaler only on CUDA with cfg.amp; plain fp32 on CPU.
        # bfloat16 is preferred (same exponent range as fp32 ⇒ no loss-scale
        # tuning issues); fp16 is the fallback on pre-Ampere GPUs. The
        # GradScaler is enabled alongside autocast per the training contract;
        # under bf16 its scaling is effectively benign (bf16's dynamic range
        # makes overflow a non-issue), under fp16 it is essential.
        self.amp_enabled: bool = self.device.type == "cuda" and cfg.amp
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.amp_dtype: torch.dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        # Constructing a disabled CUDA scaler is safe on CPU-only machines.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # ---- EMA weights ----------------------------------------------------
        self.ema = ParameterEMA(self.model, cfg.ema_decay)

        # ---- Bookkeeping ----------------------------------------------------
        self.step: int = 0                    # completed optimizer steps
        self.best_val: float = math.inf       # best validation total so far
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        # NOTE: get_logger is idempotent per name — the first TrainConfig in
        # a process pins the JSONL path for the 'aether.train' logger.
        self.logger = get_logger("aether.train", cfg.log_jsonl)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _autocast(self) -> ContextManager[Any]:
        """The forward-pass precision context for this trainer's device."""
        if self.amp_enabled:
            return torch.autocast(self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def _train_step(self, batch: PerceptionBatch) -> dict[str, float]:
        """One optimizer step; returns the (detached) loss dictionary."""
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            losses, _ = self.model.self_supervised_losses(batch)
            total: Tensor = losses["total"]

        # Backward through the (possibly) scaled loss. With the scaler
        # disabled (CPU / no AMP) scale() is the identity, so one code path
        # serves both precisions.
        self.scaler.scale(total).backward()

        if self.cfg.grad_clip > 0:
            # Gradients must be unscaled BEFORE clipping, otherwise the
            # clip threshold would be compared against scaled magnitudes
            # and effectively become grad_clip/scale.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

        self.scaler.step(self.optimizer)   # skips the step on inf/nan (fp16)
        self.scaler.update()
        self.scheduler.step()              # per-step (not per-epoch) schedule
        self.ema.update(self.model)

        return {k: float(v.detach()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate(self) -> dict[str, float]:
        """Mean SSL losses over the entire validation loader.

        Runs with **EMA weights swapped in**, ``model.eval()`` and
        ``no_grad``: eval mode also freezes the model's internal running
        statistics (e.g. the anomaly EMA baseline), so validation never
        contaminates training-time state. Training weights and train mode
        are restored before returning.

        Returns an empty dict if the val loader yields nothing.
        """
        was_training = self.model.training
        self.model.eval()
        sums: dict[str, float] = defaultdict(float)
        n_batches = 0
        with self.ema.swap(self.model), torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                with self._autocast():
                    losses, _ = self.model.self_supervised_losses(batch)
                for key, value in losses.items():
                    sums[key] += float(value)
                n_batches += 1
        if was_training:
            self.model.train()
        if n_batches == 0:
            return {}
        return {key: total / n_batches for key, total in sums.items()}

    def fit(self) -> dict[str, float]:
        """Run training until ``cfg.max_steps`` optimizer steps.

        The loop cycles the train loader indefinitely; every
        ``cfg.val_every`` steps (and at the final step) it validates,
        checkpoints ``last.pt``, and refreshes ``best.pt`` when the
        validation total improves. Console lines every 50 steps; JSONL
        metrics on every console line and every validation.

        Returns a flat summary dict: final step, last train total, best
        validation total, and the last validation losses (``val_``-prefixed).
        """
        seed_everything(self.cfg.seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg

        self.model.train()
        train_iter = cycle(self.train_loader)
        last_train: dict[str, float] = {}
        last_val: dict[str, float] = {}

        self.logger.info(
            "fit: device=%s amp=%s steps=%d..%d val_every=%d",
            self.device, self.amp_enabled, self.step, cfg.max_steps,
            cfg.val_every,
        )

        while self.step < cfg.max_steps:
            batch = next(train_iter).to(self.device)
            last_train = self._train_step(batch)
            self.step += 1
            lr = self.optimizer.param_groups[0]["lr"]

            # Human-readable heartbeat + machine-readable JSONL extras.
            if self.step % _CONSOLE_EVERY == 0 or self.step == 1:
                self.logger.info(
                    "step %6d/%d  total=%.4f  masked=%.4f  contr=%.4f  "
                    "dna=%.4f  evid=%.4f  lr=%.2e",
                    self.step, cfg.max_steps,
                    last_train.get("total", float("nan")),
                    last_train.get("masked", float("nan")),
                    last_train.get("contrastive", float("nan")),
                    last_train.get("dna", float("nan")),
                    last_train.get("evidential", float("nan")),
                    lr,
                    extra={
                        "aether_step": self.step,
                        "aether_train_total": last_train.get("total"),
                        "aether_lr": lr,
                    },
                )

            # Validation + checkpointing cadence (always fires on the final
            # step so a run never ends without a fresh last.pt / metrics).
            if self.step % cfg.val_every == 0 or self.step == cfg.max_steps:
                last_val = self.validate()
                val_total = last_val.get("total", math.inf)
                self.logger.info(
                    "val   %6d/%d  total=%.4f  (best %.4f)",
                    self.step, cfg.max_steps, val_total,
                    min(self.best_val, val_total),
                    extra={
                        "aether_step": self.step,
                        "aether_train_total": last_train.get("total"),
                        "aether_val_total": val_total,
                        "aether_lr": lr,
                    },
                )
                # Update best BEFORE writing last.pt so a resume from
                # last.pt carries the true best-so-far (a stale value would
                # let a resumed run overwrite best.pt with a worse model).
                improved = val_total < self.best_val
                if improved:
                    self.best_val = val_total
                self.save_checkpoint(self.checkpoint_dir / "last.pt")
                if improved:
                    self.save_checkpoint(self.checkpoint_dir / "best.pt")

        summary: dict[str, float] = {
            "step": float(self.step),
            "train_total": last_train.get("total", float("nan")),
            "best_val_total": self.best_val,
        }
        summary.update({f"val_{k}": v for k, v in last_val.items()})
        return summary

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a complete, resumable training snapshot to ``path``.

        Contents: model weights, EMA shadow, optimizer/scheduler/scaler
        state, step counter, best-val bookkeeping, and a plain-dict snapshot
        of the :class:`TrainConfig` (for provenance and mismatch warnings on
        resume). Written atomically (temp file + rename) so a crash mid-save
        never corrupts an existing checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "step": self.step,
            "best_val_total": self.best_val,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "cfg": dataclasses.asdict(self.cfg),
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore a snapshot written by :meth:`save_checkpoint`.

        Restores model, EMA, optimizer, scheduler, scaler, step and best-val
        state — a subsequent :meth:`fit` continues exactly where the saved
        run stopped (``fit`` loops ``while self.step < max_steps``). The
        checkpoint's config snapshot is compared against the live config and
        differences are logged as warnings (they are allowed — e.g. resuming
        with a longer ``max_steps`` — but should never be silent).

        Returns the raw checkpoint dict for callers that want extra fields.
        """
        path = Path(path)
        # weights_only=False: the payload includes plain dicts (cfg snapshot,
        # scheduler state). The file is our own artifact, not untrusted input.
        ckpt: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"], device=self.device)
        # Optimizer state tensors are re-cast onto the parameters' device by
        # load_state_dict; loading from a CPU map_location is the safe path.
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt["step"])
        self.best_val = float(ckpt.get("best_val_total", math.inf))

        saved_cfg = ckpt.get("cfg", {})
        live_cfg = dataclasses.asdict(self.cfg)
        for key in sorted(set(saved_cfg) | set(live_cfg)):
            if saved_cfg.get(key) != live_cfg.get(key):
                self.logger.warning(
                    "resume: cfg.%s differs (checkpoint=%r, live=%r)",
                    key, saved_cfg.get(key), live_cfg.get(key),
                )
        self.logger.info("resumed from %s at step %d", path, self.step)
        return ckpt
