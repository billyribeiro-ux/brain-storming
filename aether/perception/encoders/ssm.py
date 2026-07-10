"""S4D-style diagonal state-space sequence encoder (long-memory, causal).

This module implements :class:`SSMEncoder`, the state-space member of the
encoder family. Its default assignment in
:class:`~aether.perception.interfaces.PerceptionConfig` is the 5-minute
stream (two full sessions, 156 bars), where the job is *long-memory context
compression*: carry information across hundreds of steps at O(L log L) cost
without attention's O(L²) blow-up, and with per-position causality that makes
the "no look-ahead" invariant structurally impossible to violate inside the
encoder.

The continuous-time model
-------------------------
A (single-input single-output) state-space model is the linear ODE

    x'(t) = A x(t) + B u(t)          x(t) ∈ ℂ^N   (latent state)
    y(t)  = Re( C x(t) ) + D u(t)    u, y ∈ ℝ     (input / output signal)

Solving it shows y is a *causal convolution* of u with the impulse-response
kernel ``K(t) = C e^{tA} B`` — the state is literally an exponentially
weighted memory of the entire past input. S4D (Gu et al., 2022, "On the
Parameterization and Initialization of Diagonal State Space Models") makes
this practical by taking **A diagonal**, so every state coordinate evolves
independently and the kernel is a sum of N complex exponentials.

We run one independent SSM per model channel ("depthwise", H = d_model of
them), each with its own N = d_state diagonal states — mixing across channels
is delegated to the pointwise (GLU) linear that follows in each block.

Parameterization (per channel h, state n) — faithful to S4D-Lin
---------------------------------------------------------------
* ``A_n = -exp(log_A_re) + i·A_im`` with **S4D-Lin init** ``A_n = -1/2 + iπn``.
  The imaginary parts ``πn`` place the states at the harmonics of a Fourier
  basis (this is where the "linear" in S4D-Lin comes from: frequencies grow
  linearly with n), so at initialization the kernel is a damped Fourier
  series — an expressive, well-conditioned basis for arbitrary smooth
  kernels. The real part −1/2 sets the decay envelope ``e^{-t/2}``: a soft
  memory horizon of a couple of *time units*, which the per-channel step size
  dt (below) rescales into steps. Storing ``log(−Re A)`` keeps
  ``Re A < 0`` — i.e. a *stable* (decaying, non-exploding) system — for every
  possible parameter value, so no clamping is ever needed. Both the log-real
  and imaginary parts are learnable.
* ``B_n``: learnable, init 1 (complex, stored as re/im pair). With B ≡ 1 the
  init reduces to the canonical S4D kernel; learning B lets each state
  re-weight how strongly it ingests the input.
* ``C_n``: learnable, init standard complex normal (re, im ~ N(0, 1/2), so
  E|C|² = 1). C mixes the N state trajectories into the scalar output —
  random phases/magnitudes decorrelate the channels at init.
* ``dt``: learnable per channel, stored as ``log dt``, init log-uniform in
  [1e-3, 1e-1]. dt is the *time scale* of the discretization: a channel with
  dt = 1e-3 integrates ~1000 steps into its state (slow context), one with
  dt = 1e-1 reacts within ~10 steps (fast transients). The log-uniform spread
  gives the encoder a bank of memory horizons spanning two orders of
  magnitude before any learning happens.
* ``D``: learnable real skip (init 1), the instantaneous feed-through term of
  the state-space model: ``y += D·u``.

Discretization (zero-order hold) and the convolution kernel
-----------------------------------------------------------
Assuming u is held constant over each step of length dt (ZOH), the exact
discrete recurrence has

    Ā = exp(dt·A)                       (state transition per step)
    B̄ = (Ā − 1) / A · B                (integrated input weight)

and unrolling ``x_k = Ā x_{k−1} + B̄ u_k`` gives the length-L causal kernel

    K[l] = 2·Re( Σ_n C_n · B̄_n · Ā_nˡ ),      l = 0 … L−1.

The factor 2·Re(·) is the standard conjugate-pair trick: a real system needs
eigenvalues in conjugate pairs (λ, λ̄) with conjugate coefficients, and the
pair's contribution is exactly twice the real part of one member — so we
store only half the spectrum and double the real part.

``Ā_nˡ`` for l = 0…L−1 is a Vandermonde matrix. Powering a complex number L
times is numerically hazardous; instead we exploit ``log Ā = dt·A`` (exact,
by construction) and evaluate **in log space**:

    Ā^l = exp(l · dt·A)

which is stable because ``Re(dt·A) < 0`` ⇒ every entry has magnitude ≤ 1
(pure decaying exponentials — no overflow, graceful underflow to 0).

Causal FFT convolution
----------------------
``y = K * u`` (causal, per channel) is computed in O(L log L) via FFT. FFT
multiplication computes *circular* convolution, which would wrap the end of
the window back onto its start — a causality violation. Zero-padding both K
and u to length 2L makes the circular convolution of the padded signals equal
to the *linear* convolution on its first 2L−1 samples; we keep outputs
[0, L), which are exactly ``y[t] = Σ_{l≤t} K[l]·u[t−l]``: position t depends
only on inputs ≤ t. This 2L padding is the load-bearing causality guarantee
of the whole module.

Block structure (× cfg.n_layers)
--------------------------------
::

    x ─ Linear(input_dim → d_model)
      ─ n_layers × [ mask pads → S4D conv (+D·u) → GELU
                     → Linear(d → 2d) → GLU → Dropout → +residual → LayerNorm ]
      ─▶ tokens [B, L, d_model]

Every non-S4D op (GELU, GLU linear, LayerNorm, residual) is *pointwise in
time*, so the block as a whole stays per-position causal.

Padding
-------
Windows are LEFT-padded (see ``PerceptionBatch``). Padded positions are
zeroed *before every S4D convolution* — not just once at the input — because
LayerNorm's affine bias re-inflates pad positions to nonzero values between
blocks, and the causal kernel of the *next* block would otherwise smear those
phantom values into the valid region. Zero inputs contribute exactly nothing
to the convolution sum, which is the correct "no history" semantics for a
short window. Outputs at padded positions are zeroed before returning
(cosmetic — the contract lets consumers mask).

Autocast / dtype guard
----------------------
The kernel involves complex exponentials and FFTs — math that is unstable (or
outright unsupported) in bf16/fp16. All S4D-internal math is therefore pinned
to float32/complex64: inputs are ``.float()``-ed on entry, parameters live in
float32, and the result is cast back to the incoming activation dtype at the
end, so the module composes cleanly with ``torch.autocast``. No in-place ops
touch autocast outputs, and no device is ever hard-coded.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aether.perception.interfaces import SequenceEncoder, SSMEncoderConfig

__all__ = ["SSMEncoder"]


class _S4DLayer(nn.Module):
    """One depthwise diagonal-SSM long convolution: [B, L, H] -> [B, L, H].

    Holds the continuous-time parameters (A, B, C, dt, D) for H = d_model
    independent single-channel SSMs with N = d_state diagonal states each,
    materializes the length-L ZOH kernel on every forward (L is not known in
    advance and the parameters change every optimizer step), and applies it
    as a causal depthwise convolution via 2L-padded FFT. See the module
    docstring for the full derivation of every formula used here.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        h, n = d_model, d_state

        # --- A = -exp(log_A_re) + i*A_im,  S4D-Lin init A_n = -1/2 + i*pi*n.
        # log of the NEGATED real part guarantees Re(A) < 0 (stability) by
        # construction for any parameter value the optimizer reaches.
        self.log_A_re = nn.Parameter(torch.full((h, n), math.log(0.5)))
        # Imaginary parts pi*n, identical across channels at init; channels
        # differentiate through dt, B, C and training.
        self.A_im = nn.Parameter(
            (math.pi * torch.arange(n, dtype=torch.float32)).repeat(h, 1)
        )

        # --- B: complex, init 1 + 0i (canonical S4D uses fixed B = 1; ours
        # is learnable per the spec, stored as separate real tensors so the
        # optimizer never sees complex parameters).
        self.B_re = nn.Parameter(torch.ones(h, n))
        self.B_im = nn.Parameter(torch.zeros(h, n))

        # --- C: standard complex normal — re, im ~ N(0, 1/2) => E|C|^2 = 1.
        self.C_re = nn.Parameter(torch.randn(h, n) * math.sqrt(0.5))
        self.C_im = nn.Parameter(torch.randn(h, n) * math.sqrt(0.5))

        # --- dt: per-channel step size, log-uniform in [1e-3, 1e-1] so the
        # layer starts with memory horizons spanning ~10 to ~1000 steps.
        log_dt = torch.rand(h) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        self.log_dt = nn.Parameter(log_dt)

        # --- D: real instantaneous skip, y += D * u.
        self.D = nn.Parameter(torch.ones(h))

    def _kernel(self, length: int) -> Tensor:
        """Materialize the causal ZOH convolution kernel K, float32 [H, L].

        K[h, l] = 2 * Re( sum_n C_hn * Bbar_hn * Abar_hn^l ), with the
        Vandermonde powers Abar^l computed in log space as exp(l * dt*A)
        (exact since log Abar = dt*A), which only ever *decays* because
        Re(dt*A) < 0.
        """
        # All in float32/complex64 regardless of any ambient autocast: these
        # are parameter-only ops (params are float32) plus complex math that
        # autocast does not (and must not) down-cast.
        dt = torch.exp(self.log_dt.float())                        # [H]
        A = torch.complex(-torch.exp(self.log_A_re.float()),
                          self.A_im.float())                       # [H, N] c64
        B = torch.complex(self.B_re.float(), self.B_im.float())    # [H, N]
        C = torch.complex(self.C_re.float(), self.C_im.float())    # [H, N]

        dtA = A * dt.unsqueeze(-1)                                 # [H, N]
        Abar = torch.exp(dtA)                                      # ZOH: e^{dt A}
        # ZOH input matrix: Bbar = (Abar - 1)/A * B. A is never 0 because
        # Re(A) = -exp(...) < 0 strictly.
        Bbar = (Abar - 1.0) / A * B                                # [H, N]

        # Vandermonde in log space: Abar^l = exp(l * dtA), l = 0..L-1.
        steps = torch.arange(length, dtype=torch.float32,
                             device=dtA.device)                    # [L]
        vand = torch.exp(dtA.unsqueeze(-1) * steps)                # [H, N, L]

        # Contract the state dimension. Deliberately written as mul+sum (not
        # einsum): elementwise complex ops are dtype-transparent under
        # autocast, whereas einsum is an autocast-eligible op we would have
        # to fence off explicitly.
        coeff = (C * Bbar).unsqueeze(-1)                           # [H, N, 1]
        kernel = 2.0 * (coeff * vand).sum(dim=1).real              # [H, L] f32
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        """Causal depthwise SSM convolution, [B, L, H] -> [B, L, H].

        Runs in float32 internally (see module docstring) and returns in the
        caller's activation dtype so autocast graphs stay consistent.
        """
        in_dtype = x.dtype
        u = x.float().transpose(1, 2)                              # [B, H, L] f32
        length = u.shape[-1]

        kernel = self._kernel(length)                              # [H, L] f32

        # Causal linear convolution via FFT. Padding both operands to 2L
        # turns circular convolution into linear convolution on the first L
        # outputs — this is what prevents future samples wrapping around
        # into the past (see module docstring, "Causal FFT convolution").
        n_fft = 2 * length
        k_f = torch.fft.rfft(kernel, n=n_fft)                      # [H, F]
        u_f = torch.fft.rfft(u, n=n_fft)                           # [B, H, F]
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :length]      # [B, H, L]

        # Instantaneous skip connection D*u — the feed-through term of the
        # state-space model (out-of-place: autocast-safe).
        y = y + u * self.D.float().view(1, -1, 1)

        return y.transpose(1, 2).to(in_dtype)                      # [B, L, H]


class _S4DBlock(nn.Module):
    """S4D conv → GELU → GLU pointwise mix → dropout → residual + LayerNorm.

    The S4D layer is purely depthwise (channels never interact inside it);
    the GLU linear afterwards is the channel mixer: Linear(d → 2d) split into
    value/gate halves, ``out = value * sigmoid(gate)`` — a multiplicative
    gate that lets the block modulate how much of the long-memory signal
    passes through, per channel and per timestep. Post-norm residual wiring
    (``LayerNorm(x + f(x))``) per the assignment spec. Every op besides the
    S4D convolution acts pointwise in time, preserving causality.
    """

    def __init__(self, cfg: SSMEncoderConfig) -> None:
        super().__init__()
        self.s4d = _S4DLayer(cfg.d_model, cfg.d_state)
        # 2*d_model so F.glu can split into (value, gate) halves.
        self.mix = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        # Zero pad positions BEFORE the convolution — LayerNorm's bias makes
        # them nonzero between blocks, and the causal kernel would otherwise
        # smear those phantom values forward into valid positions.
        u = x if pad_mask is None else x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        y = self.s4d(u)                       # [B, L, D] long-memory features
        y = F.gelu(y)
        y = F.glu(self.mix(y), dim=-1)        # gated pointwise channel mix
        y = self.dropout(y)
        # Residual + post-LayerNorm. Both are pointwise in time: no leakage.
        return self.norm(x + y)


class SSMEncoder(SequenceEncoder):
    """Stack of S4D blocks behind an input projection. See module docstring.

    Contract (``SequenceEncoder``):
        forward(x [B, L, input_dim], pad_mask [B, L] bool or None)
            -> tokens [B, L, d_model]

    * Output token t depends ONLY on inputs at positions <= t (per-position
      causal — enforced structurally by the 2L-padded FFT convolution).
    * ``pad_mask`` (True = padding): pad positions are zeroed before every
      convolution and in the returned tokens.
    * ``output_dim == cfg.d_model``.
    """

    def __init__(self, input_dim: int, cfg: SSMEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        self.blocks = nn.ModuleList(
            _S4DBlock(cfg) for _ in range(cfg.n_layers)
        )
        self.output_dim: int = cfg.d_model

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """Encode x [B, L, input_dim] -> contextual tokens [B, L, d_model]."""
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, pad_mask)
        if pad_mask is not None:
            # Cosmetic zeroing of pad positions (contract allows arbitrary
            # values there; zeros make debugging dumps less misleading).
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return h
