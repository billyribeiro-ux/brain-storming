"""Learnable multi-resolution filterbank encoder (wavelet-inspired, causal).

This module implements :class:`WaveletEncoder`, the filterbank member of the
encoder family. Its default role in
:class:`~aether.perception.interfaces.PerceptionConfig` is the *front half*
of the 1-minute stream's ``"wavelet+transformer"`` stack: decompose the raw
bar sequence into several temporal frequency bands cheaply and causally, so
the attention layers behind it start from tokens that already separate
"fast" microstructure wiggles from "slow" session-scale drifts.

Why "wavelet-inspired", and why it is NOT hardcoded analysis
------------------------------------------------------------
A discrete wavelet transform decomposes a signal with a fixed pair of
average/difference filters applied at dyadic (power-of-two) scales. We keep
exactly two ideas from that construction and learn everything else:

1. **Dyadic multi-scale structure.** Branch s uses dilation ``2**s``, so its
   receptive field covers ``(kernel_size − 1)·2**s + 1`` timesteps — the
   branches tile the time axis from bar-to-bar detail (s = 0) up to
   multi-hour context, in octaves, exactly like a wavelet pyramid but
   computed in parallel rather than by recursive decimation (so we keep one
   output token per input step, as the ``SequenceEncoder`` contract
   requires).
2. **Haar-like initialization.** Each branch's depthwise filters start as a
   noisy local *average* (low-pass, the Haar scaling function's shape) on
   half the channels and a noisy old-vs-recent *difference* (band-pass, the
   Haar wavelet's shape) on the other half. This is an *initialization
   prior*, not a rule: identical-random filters at every scale would start
   highly correlated and waste the multi-branch capacity, whereas
   average/difference at dyadic dilations start near-orthogonal in frequency
   (each branch owns an octave of the spectrum). Every filter weight remains
   a free ``nn.Parameter`` — gradient descent can (and will) reshape them
   into whatever filters the data demands. Nothing here computes a named
   indicator or imposes a trading rule; it is representation scaffolding, in
   the same spirit as edge-filter-like inits in vision stems.

Architecture
------------
::

    x ─ Linear(input_dim → d_model) ─ h            (+ zero pad positions)
    h ─┬─ branch s=0: causal depthwise Conv1d(k, dilation=1)   ─┐
       ├─ branch s=1: causal depthwise Conv1d(k, dilation=2)   ─┤ concat
       ├─ ...                                                  ─┤ [B, S·D, L]
       └─ branch s=S−1: causal depthwise Conv1d(k, dil=2^(S−1))─┘
         ─ pointwise Conv1d(S·D → D) ─ GELU ─ Dropout
         ─ + h (residual) ─ LayerNorm ─▶ tokens [B, L, d_model]

Depthwise-separable factorization: each branch is the *depthwise* (per
channel, time-only) half of a separable convolution, and the shared
pointwise 1×1 convolution after concatenation is the *channel-mixing* half —
applied once across all scales. Splitting it this way (rather than giving
each branch its own pointwise conv) matters mathematically: two stacked linear
maps with no nonlinearity between them collapse into one, so per-branch
pointwise convs before the shared mixer would add parameters without adding
expressivity.

Causality
---------
Each branch left-pads by ``(kernel_size − 1) · dilation`` and convolves with
no internal padding, so output position t reads inputs
``t, t − dilation, …, t − (k−1)·dilation`` — strictly ≤ t. Everything else in
the block (1×1 conv, GELU, dropout, residual, LayerNorm) is pointwise in
time. Output token t therefore depends only on inputs ≤ t.

Padding
-------
Windows are LEFT-padded with zeros (see ``PerceptionBatch``). Pad positions
are zeroed after the input projection (the projection's bias would otherwise
make them nonzero) so the causal convolutions see genuine zeros — the same
"no history" semantics as the explicit causal left-padding itself. Outputs at
pad positions are zeroed again before returning (cosmetic; the contract says
consumers must mask).

Autocast / device notes
-----------------------
Convolutions, GELU and LayerNorm all compose cleanly with ``torch.autocast``;
all ops are out-of-place (``masked_fill``, not ``masked_fill_``), float32 by
default, and no device is ever hard-coded — everything follows the input.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aether.perception.interfaces import SequenceEncoder, WaveletEncoderConfig

__all__ = ["WaveletEncoder"]


class _CausalScaleBranch(nn.Module):
    """One dilated causal depthwise Conv1d branch: [B, D, L] -> [B, D, L].

    ``nn.Conv1d`` computes cross-correlation, so with a pure LEFT pad of
    ``(k − 1)·dilation`` and no internal padding, weight index ``k − 1``
    multiplies the *current* timestep and weight index 0 the *oldest* one:

        out[t] = Σ_{j=0}^{k−1} w[j] · in[t − (k − 1 − j)·dilation]

    which is causal by construction. The Haar-like init below is written
    against exactly this index convention (index 0 = far past, k−1 = now).
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        # groups=d_model => depthwise: one length-k filter per channel, no
        # channel mixing here (that is the shared pointwise conv's job).
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            dilation=dilation, groups=d_model, bias=True,
        )
        self._init_haar_like(d_model, kernel_size)

    def _init_haar_like(self, d_model: int, kernel_size: int) -> None:
        """Haar-like average/difference init plus noise (see module docstring).

        * Even channels: local AVERAGE, w = 1/k everywhere — unit DC gain
          low-pass, the (stretched) Haar scaling function.
        * Odd channels: old-vs-recent DIFFERENCE, −1/k on the older half of
          the taps and +1/k on the recent half — zero DC gain band-pass, the
          (stretched) Haar wavelet. Positive sign on the *recent* taps so a
          rising input yields a positive response at init (sign is
          arbitrary but fixing it keeps the init interpretable).
        * Both get N(0, (0.5/k)²) noise so channels within a branch start
          decorrelated and gradient descent breaks symmetry immediately.

        Filters remain fully learnable parameters; this only sets t=0 values.
        """
        k = kernel_size
        recent_half = k // 2  # taps closest to "now" (weight indices k//2..k-1)

        average = torch.full((k,), 1.0 / k)
        difference = torch.cat([
            torch.full((k - recent_half,), -1.0 / k),   # older taps
            torch.full((recent_half,), +1.0 / k),       # recent taps
        ])

        # Depthwise Conv1d weight shape: [d_model, 1, k].
        pattern = torch.where(
            (torch.arange(d_model) % 2 == 0).view(-1, 1),
            average.view(1, k),
            difference.view(1, k),
        ).view(d_model, 1, k)
        noise = torch.randn(d_model, 1, k) * (0.5 / k)

        with torch.no_grad():
            self.depthwise.weight.copy_(pattern + noise)
            if self.depthwise.bias is not None:
                self.depthwise.bias.zero_()

    def forward(self, u: Tensor) -> Tensor:
        """Causally filter u [B, D, L] -> [B, D, L] at this branch's scale."""
        # Left-only zero pad = causal: position t never reads inputs > t.
        return self.depthwise(F.pad(u, (self.left_pad, 0)))


class WaveletEncoder(SequenceEncoder):
    """Learnable dyadic filterbank block. See module docstring.

    Contract (``SequenceEncoder``):
        forward(x [B, L, input_dim], pad_mask [B, L] bool or None)
            -> tokens [B, L, d_model]

    * Output token t depends ONLY on inputs at positions <= t (per-position
      causal — left-padded dilated convs; everything else is pointwise).
    * ``pad_mask`` (True = padding): pad positions are zeroed before the
      convolutions and in the returned tokens.
    * ``output_dim == cfg.d_model``.
    """

    def __init__(self, input_dim: int, cfg: WaveletEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        # Parallel dyadic branches: dilation 1, 2, 4, ... 2^(n_scales-1).
        self.branches = nn.ModuleList(
            _CausalScaleBranch(cfg.d_model, cfg.kernel_size, dilation=2 ** s)
            for s in range(cfg.n_scales)
        )
        # Shared pointwise (1x1) conv: the channel-mixing half of the
        # depthwise-separable factorization, fusing all scales at once.
        self.mix = nn.Conv1d(cfg.n_scales * cfg.d_model, cfg.d_model, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output_dim: int = cfg.d_model

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """Encode x [B, L, input_dim] -> contextual tokens [B, L, d_model]."""
        h = self.in_proj(x)                                # [B, L, D]
        if pad_mask is not None:
            # The projection bias makes pad rows nonzero; restore true zeros
            # so the causal convs see "no history" at padded (left) steps.
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        u = h.transpose(1, 2)                              # [B, D, L] for Conv1d
        bands = torch.cat([branch(u) for branch in self.branches],
                          dim=1)                           # [B, S*D, L]
        y = self.mix(bands)                                # [B, D, L]
        y = F.gelu(y)
        y = self.dropout(y.transpose(1, 2))                # [B, L, D]

        out = self.norm(h + y)                             # residual + LN
        if pad_mask is not None:
            # Cosmetic zeroing (contract allows arbitrary values at pads).
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out
