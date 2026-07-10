"""Cross-modal fusion: Perceiver-style latent bottleneck over all streams.

Problem
-------
After per-stream encoding, one sample carries hundreds of contextual tokens
(``len_1m + len_5m + len_daily`` of them, e.g. 240 + 156 + 60 = 456 at the
default window). Downstream layers want ONE state vector per sample, and we
want every timescale to be able to talk to every other while producing it.

Naive choice: full self-attention over the concatenated token soup — that is
O(L_total²) in both compute and memory, and mixes streams in a single
undifferentiated pool.

Perceiver-style solution (Jaegle et al., 2021)
----------------------------------------------
Keep a small *learned latent array* of ``n_latents`` query tokens (16 by
default). Each fusion layer:

(a) **cross-attention** — the latents attend to the concatenation of all
    stream tokens plus one DNA token, cost O(n_latents · L_total): each of
    the 456 stream tokens is *read once per layer* by 16 queries instead of
    by 456 of its peers. That is a ~L_total / n_latents (≈28×) reduction in
    attention cost versus full self-attention, and it scales linearly if
    window lengths grow;
(b) **latent self-attention** — the latents exchange what they read,
    cost O(n_latents²) which is negligible;
(c) **position-wise FFN** on the latents.

All three sub-blocks are pre-LN residual (LayerNorm on the input of the
sub-block, output added back to the un-normalized stream), which keeps deep
stacks stable without warmup tricks.

The DNA token
-------------
The instrument's learned DNA embedding is linearly projected to ``d_model``
and appended to the key/value context as one extra, always-valid token. The
latents can therefore condition *what they read* on *which instrument* they
are reading — e.g. weighting overnight-gap bars differently for a leveraged
ETF than for a mega-cap — without a separate conditioning pathway.

Padding
-------
Each stream arrives with a ``pad_mask`` (True = padding). Masks are
concatenated in the same order as the tokens and handed to
``nn.MultiheadAttention`` as ``key_padding_mask``, so padded bars are
invisible to the latents. The appended DNA token is always valid, which also
guarantees at least one attendable key per sample (no all-masked-row NaNs).

Output
------
The latent array is mean-pooled into a single [B, d_model] fused state
vector — the primary hand-off to the world model. The pool needs no mask:
latents are learned queries and are never padded.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from aether.perception.interfaces import FusionConfig

#: FFN expansion factor inside fusion layers. FusionConfig deliberately does
#: not expose this knob (the fusion stack is shallow; 4× is the standard
#: transformer choice and there is little to gain tuning it).
_FFN_MULT: int = 4


class _PerceiverFusionLayer(nn.Module):
    """One fusion layer: cross-attend → latent self-attend → FFN (pre-LN).

    Pre-LN residual form: every sub-block computes
    ``x = x + SubBlock(LayerNorm(x))``, so the residual stream itself is
    never normalized in place — gradients flow through an identity path and
    the stack trains stably from step 0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        # (a) latents (queries) read the stream tokens + DNA token (keys/values)
        self.ln_query = nn.LayerNorm(d_model)
        self.ln_context = nn.LayerNorm(d_model)   # normalize K/V independently
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # (b) latents exchange information among themselves
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # (c) position-wise feed-forward on each latent
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, _FFN_MULT * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_FFN_MULT * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: Tensor,
        context: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """latents [B, N, D], context [B, L_total, D], mask [B, L_total] bool.

        Returns updated latents [B, N, D]. No in-place ops on activations —
        safe under ``torch.autocast``.
        """
        # (a) cross-attention: latents pull information out of the streams.
        q = self.ln_query(latents)
        kv = self.ln_context(context)
        read, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        latents = latents + read
        # (b) latent self-attention: mix what the latents just read.
        s = self.ln_self(latents)
        mixed, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + mixed
        # (c) feed-forward.
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class CrossModalFusion(nn.Module):
    """Fuse per-stream token sequences + ticker DNA into one state vector.

    Parameters
    ----------
    cfg:
        :class:`FusionConfig` — width, heads, number of latents/layers.
    dna_dim:
        Width of the ticker DNA embedding; projected to ``d_model`` and
        appended to the attended context as one extra token.
    """

    def __init__(self, cfg: FusionConfig, dna_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        # Learned latent array — the Perceiver "query bottleneck". Small
        # normal init (like token embeddings) so early attention is diffuse.
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, cfg.d_model) * 0.02)
        # DNA embedding → one context token in model width.
        self.dna_proj = nn.Linear(dna_dim, cfg.d_model)
        self.layers = nn.ModuleList(
            _PerceiverFusionLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        # Final LayerNorm before pooling — standard pre-LN closing norm; the
        # residual stream is otherwise never normalized.
        self.ln_out = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        streams: list[Tensor],
        pad_masks: list[Optional[Tensor]],
        dna: Tensor,
    ) -> Tensor:
        """Fuse streams into a per-sample state vector.

        Parameters
        ----------
        streams:
            List of [B, L_i, d_model] token tensors (any number of streams,
            any lengths). Concatenated along the length axis.
        pad_masks:
            Parallel list of [B, L_i] bool masks (True = padding) or ``None``
            for a stream with no padding.
        dna:
            [B, dna_dim] ticker DNA embedding, appended as one always-valid
            context token.

        Returns
        -------
        Tensor
            [B, d_model] fused state — mean over the latent array (a
            mask-free mean: latents are learned queries, never padded).
        """
        if len(streams) != len(pad_masks):
            raise ValueError(
                f"streams ({len(streams)}) and pad_masks ({len(pad_masks)}) "
                "must be parallel lists"
            )
        batch = dna.shape[0]
        device = dna.device

        # Build the combined key/value context and its padding mask, keeping
        # token order and mask order identical: [stream_0 | stream_1 | ... | DNA].
        mask_parts: list[Tensor] = []
        for tokens, mask in zip(streams, pad_masks):
            if mask is None:
                # No padding info ⇒ every position is a valid key.
                mask = torch.zeros(
                    tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=device
                )
            mask_parts.append(mask)
        dna_token = self.dna_proj(dna).unsqueeze(1)               # [B, 1, D]
        context = torch.cat([*streams, dna_token], dim=1)          # [B, L_total+1, D]
        key_padding_mask = torch.cat(
            mask_parts
            + [torch.zeros(batch, 1, dtype=torch.bool, device=device)],  # DNA valid
            dim=1,
        )                                                          # [B, L_total+1]

        # Broadcast the learned latent array across the batch. `expand` is a
        # view; the residual adds inside each layer allocate fresh tensors,
        # so the parameter itself is never written to.
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)  # [B, N, D]
        for layer in self.layers:
            latents = layer(latents, context, key_padding_mask)

        # Mask-free mean pool over latents → the fused state vector.
        return self.ln_out(latents).mean(dim=1)                    # [B, D]
