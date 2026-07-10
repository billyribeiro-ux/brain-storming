"""Bar-level Pre-LN transformer encoder with rotary position embeddings.

This module implements :class:`BarTransformerEncoder`, the attention-based
sequence encoder used for intraday bar streams (and available to any stream
via :class:`~aether.perception.interfaces.PerceptionConfig`). It is written
from scratch — deliberately NOT on top of ``nn.TransformerEncoder`` — because
we need two things the stock implementation cannot give us:

1. **Rotary position embeddings (RoPE)** applied to queries/keys inside every
   attention layer, and
2. **explicit control of the padding mask** so we can guarantee that padded
   key positions contribute exactly ``-inf`` to the attention logits.

Why bidirectional attention is safe here
----------------------------------------
Every window handed to this encoder is *entirely in the past* relative to the
sample's anchor time: the dataset only assembles bars whose closing timestamp
is ``<= anchor_close_time`` (see ``WindowSpec`` in ``interfaces.py``). A token
at window position ``j > i`` is therefore *later past data*, not future data.
Letting position ``i`` attend to position ``j`` leaks nothing about the
post-anchor future — it only lets the encoder build a fully contextual
representation of the already-observed window, exactly like a bidirectional
text encoder reading a completed sentence. Bidirectional (non-causal)
attention within the window is intentional.

Architecture (Pre-LN)
---------------------
::

    x  ─ Linear(input_dim → d_model) ─ [+ sinusoidal positions if not RoPE]
       ─ n_layers × [ x + Drop(MHSA(LN(x)))  ;  x + Drop(FFN(LN(x))) ]
       ─ LayerNorm ─▶ tokens [B, L, d_model]

Pre-LN (normalize *before* each sublayer, residual around it) is used instead
of the original Post-LN because it keeps residual-branch magnitudes bounded,
trains stably without a learning-rate warmup cliff, and is the de-facto
standard for modern transformers.

Rotary position embeddings — the math
-------------------------------------
RoPE (Su et al., 2021, "RoFormer") encodes *absolute* positions as rotations
whose effect on the attention dot product depends only on the *relative*
offset. Each head dimension pair ``(i, i + d_head/2)`` is treated as 2-D
coordinates and rotated by the position-dependent angle

    ``m · θ_i``  with  ``θ_i = 10000^(-2i / d_head)``,  ``i ∈ [0, d_head/2)``

for a token at position ``m``. Using the *rotate-half* formulation, the
rotation of a vector ``x`` is computed as

    ``RoPE(x, m) = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)``

where ``rotate_half([x₁, x₂]) = [-x₂, x₁]`` (``x₁``/``x₂`` are the first and
second halves of the head dimension) and the ``cos``/``sin`` tables repeat the
``d_head/2`` frequencies over both halves. Because rotations are orthogonal,
``⟨RoPE(q, m), RoPE(k, n)⟩`` depends only on ``m − n`` — the attention scores
are relative-position aware. A pleasant corollary for us: LEFT-padding shifts
all *valid* tokens by the same constant offset, and since only relative
offsets matter, RoPE is completely insensitive to how much left-padding a
short-history sample carries. (The additive-sinusoid fallback does not enjoy
this property; it is kept for ablations.)

The ``cos``/``sin`` tables are precomputed lazily per ``(L, device, dtype)``
and cached on the module so repeated forwards with the same sequence length
pay the trigonometry exactly once.

Autocast / device notes
-----------------------
No ``.cuda()`` / hard-coded devices anywhere — every lazily built tensor
derives its device and dtype from the activations flowing through. All ops on
autocast-produced tensors are out-of-place, so the module runs cleanly under
``torch.autocast`` in bf16/fp16.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aether.perception.interfaces import SequenceEncoder, TransformerEncoderConfig

__all__ = ["BarTransformerEncoder"]


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #

def _rotate_half(x: Tensor) -> Tensor:
    """Map ``[x₁, x₂] → [-x₂, x₁]`` over the last dimension.

    Together with the duplicated cos/sin tables this realizes the 2-D rotation
    of each ``(i, i + d/2)`` coordinate pair:
    ``(x₁cosθ − x₂sinθ,  x₂cosθ + x₁sinθ)``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin tables; rotates q/k per head.

    Frequencies follow the original RoFormer schedule
    ``θ_i = base^(-2i/d_head)`` with ``base = 10000``. Tables are built lazily
    for each distinct ``(seq_len, device, dtype)`` triple and memoized in a
    plain dict (NOT a persistent buffer — they are pure functions of shape and
    carry no learnable state, so they must not pollute checkpoints).
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"RoPE requires an even head dim, got d_head={d_head}")
        self.d_head = d_head
        self.base = base
        # inv_freq[i] = base^(-2i/d_head), i = 0 .. d_head/2 - 1.
        # Registered as a NON-persistent buffer so `.to(device)` moves it with
        # the module but it never enters the state_dict.
        inv_freq = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # (L, device, dtype) -> (cos [L, d_head], sin [L, d_head])
        self._cache: dict[tuple[int, torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}

    def _tables(self, seq_len: int, device: torch.device,
                dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return (cos, sin) tables of shape [seq_len, d_head], memoized."""
        key = (seq_len, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Angles are always computed in float32: at large positions, low
        # frequencies lose precision badly in half dtypes; we cast only the
        # final table to the activation dtype.
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device,
                                                         dtype=torch.float32))
        # Duplicate the d_head/2 frequencies over both halves so the tables
        # line up with the rotate-half layout ([first half | second half]).
        angles = torch.cat((angles, angles), dim=-1)          # [L, d_head]
        tables = (angles.cos().to(dtype), angles.sin().to(dtype))
        self._cache[key] = tables
        return tables

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate q and k. Both are [B, n_heads, L, d_head]; shapes preserved."""
        seq_len = q.shape[-2]
        cos, sin = self._tables(seq_len, q.device, q.dtype)
        # [L, d_head] broadcasts over batch and head dims of [B, H, L, d_head].
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k


# --------------------------------------------------------------------------- #
# Sinusoidal absolute positions (fallback when cfg.rope is False)
# --------------------------------------------------------------------------- #

class SinusoidalPositions(nn.Module):
    """Classic additive sin/cos absolute position encoding (Vaswani 2017).

    ``pe[m, 2i] = sin(m / 10000^(2i/d))``, ``pe[m, 2i+1] = cos(...)`` — added
    once to the projected input. Tables are cached per (L, device, dtype)
    exactly like the RoPE tables. Note: absolute positions are *not* invariant
    to left-padding (a short-history sample sees its real bars at shifted
    positions); RoPE is the preferred, default mechanism.
    """

    def __init__(self, d_model: int, base: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.base = base
        self._cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}

    def _table(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (seq_len, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Even channel index 2i -> frequency base^(-2i/d).
        inv_freq = self.base ** (
            -torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            / self.d_model
        )
        angles = torch.outer(positions, inv_freq)             # [L, ceil(d/2)]
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = angles.sin()
        pe[:, 1::2] = angles.cos()[:, : self.d_model // 2]    # guard odd d_model
        pe = pe.to(dtype)
        self._cache[key] = pe
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Add positions to x [B, L, d_model] (out-of-place; autocast-safe)."""
        return x + self._table(x.shape[1], x.device, x.dtype)


# --------------------------------------------------------------------------- #
# Attention + FFN sublayers
# --------------------------------------------------------------------------- #

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE and explicit key masking.

    Uses ``F.scaled_dot_product_attention`` so the fused/flash kernels are
    picked automatically per device. The boolean ``attn_mask`` we pass has
    True = "this key participates"; SDPA implements False positions by adding
    ``-inf`` to the corresponding attention logits before the softmax, which
    is precisely the padding semantics the contract requires.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_dropout = dropout
        # One fused projection for q, k, v — fewer kernel launches, same math.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor],
                rope: Optional[RotaryEmbedding]) -> Tensor:
        """x: [B, L, d_model]; attn_mask: [B, 1, 1, L] bool (True = attend) or None.

        ``rope`` is passed functionally from the encoder (a single shared
        instance, so its trig cache is shared across all layers).
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)                                     # [B, L, 3*d_model]
        # -> [3, B, n_heads, L, d_head]
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope is not None:
            q, k = rope(q, k)                                 # relative positions
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,                              # bool: False -> -inf logits
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                     # [B, H, L, d_head]
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear(d → ffn_mult·d) → GELU → Dropout → Linear(→ d)."""

    def __init__(self, d_model: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = ffn_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One Pre-LN block: two normalized sublayers with residual dropout.

    ``x = x + Drop(Attn(LN(x)))`` then ``x = x + Drop(FFN(LN(x)))``.
    All residual adds are out-of-place so the block is safe on tensors
    produced under ``torch.autocast``.
    """

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.drop_attn = nn.Dropout(cfg.dropout)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout)
        self.drop_ffn = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor],
                rope: Optional[RotaryEmbedding]) -> Tensor:
        x = x + self.drop_attn(self.attn(self.norm_attn(x), attn_mask, rope))
        x = x + self.drop_ffn(self.ffn(self.norm_ffn(x)))
        return x


# --------------------------------------------------------------------------- #
# The encoder
# --------------------------------------------------------------------------- #

class BarTransformerEncoder(SequenceEncoder):
    """Pre-LN transformer over bar sequences. See module docstring for math.

    Contract (``SequenceEncoder``):
        forward(x [B, L, input_dim], pad_mask [B, L] bool or None)
            -> tokens [B, L, d_model]

    * Attention is **bidirectional within the window** — safe by construction,
      because the whole window is past data relative to the anchor (the
      dataset enforces the ``<= anchor_close_time`` invariant); a token
      attending "rightward" only sees later *past* bars, never the future.
    * Padded key positions (``pad_mask == True``) receive ``-inf`` attention
      logits, so no valid token's representation is ever contaminated by
      left-pad zeros. Outputs *at* padded positions are unspecified by the
      contract; consumers mask them (e.g. via ``masked_mean``).
    * ``cfg.rope=True`` (default): rotary embeddings on q/k, relative-position
      aware and left-padding invariant. ``cfg.rope=False``: additive
      sinusoidal absolute positions at the input instead.
    """

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        # Exactly one positioning mechanism is active, per the config flag.
        # The RoPE module is single and shared across layers (passed
        # functionally in forward) so its trig cache is built once.
        d_head = cfg.d_model // cfg.n_heads
        self.rope: Optional[RotaryEmbedding] = (
            RotaryEmbedding(d_head) if cfg.rope else None
        )
        self.positions: Optional[SinusoidalPositions] = (
            None if cfg.rope else SinusoidalPositions(cfg.d_model)
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output_dim: int = cfg.d_model

    @staticmethod
    def _build_attn_mask(pad_mask: Optional[Tensor]) -> Optional[Tensor]:
        """[B, L] pad mask (True = padding) -> [B, 1, 1, L] SDPA bool mask.

        The returned mask has True = "key may be attended to" and broadcasts
        over heads and query positions, so every query in every head is
        forbidden (−inf logit) from looking at padded keys.

        Degenerate guard: if a sample is *entirely* padding, masking all keys
        would make every softmax row NaN and poison gradients batch-wide.
        For such rows we un-mask everything — their outputs are garbage over
        garbage (all-zero pad inputs) and consumers discard them via the pad
        mask anyway, but the arithmetic stays finite.
        """
        if pad_mask is None:
            return None
        keep = ~pad_mask                                       # True = real bar
        all_pad = ~keep.any(dim=1, keepdim=True)               # [B, 1]
        keep = keep | all_pad                                  # NaN guard
        return keep[:, None, None, :]                          # [B, 1, 1, L]

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """Encode x [B, L, input_dim] -> contextual tokens [B, L, d_model]."""
        h = self.in_proj(x)
        if self.positions is not None:                         # sinusoidal path
            h = self.positions(h)
        attn_mask = self._build_attn_mask(pad_mask)
        for block in self.blocks:
            h = block(h, attn_mask, self.rope)
        return self.final_norm(h)
