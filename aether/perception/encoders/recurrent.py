"""LSTM-based sequence encoder for slow, low-frequency context streams.

This module implements :class:`RecurrentEncoder`, the recurrent member of the
encoder family. Its default assignment in
:class:`~aether.perception.interfaces.PerceptionConfig` is the DAILY context
stream (~60 previous sessions), where a gated recurrence is a natural fit:
the sequence is short, strictly ordered, and what downstream layers mostly
need from it is an accumulated "state of the regime" rather than fine-grained
token-to-token relations.

Architecture
------------
::

    x ─ Linear(input_dim → d_model)
      ─ nn.LSTM(d_model, d_model, n_layers, batch_first,
                dropout between layers, optionally bidirectional)
      ─ [Linear(2·d_model → d_model) iff bidirectional, else identity]
      ─ LayerNorm ─▶ tokens [B, L, d_model]

The LSTM recurrence (per gate: ``i, f, g, o``) maintains a cell state ``c_t``
through multiplicative gates — ``c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t``,
``h_t = o_t ⊙ tanh(c_t)`` — which gives each output token a summary of
*everything up to and including* its own time step. A unidirectional LSTM is
therefore causal *per position*; combined with the dataset guarantee that the
entire window predates the anchor, no configuration of this encoder can leak
post-anchor information. (``cfg.bidirectional`` must stay ``False`` for
streams where per-position causality itself matters — see the note on the
config dataclass; when it is ``True``, each token also summarizes later
*past* bars of the same window, which is still anchor-safe but no longer
per-position causal.)

Left-padding behaviour — why we do NOT pack sequences
-----------------------------------------------------
Batches LEFT-pad short histories with zeros (see ``PerceptionBatch``). The
LSTM simply consumes those zero rows first: with zero input and zero initial
state the recurrence produces small, quickly-forgotten activations, and by
the time real bars arrive the state has "washed in" — the forget gate has had
every opportunity to discard the pad transient. This is a deliberate,
documented trade-off: ``pack_padded_sequence`` machinery assumes RIGHT
padding, complicates ONNX/torchscript export, and buys nothing measurable on
windows this short. ``pad_mask`` is accepted for interface compatibility and
**ignored for computation**; purely for cleanliness the returned tokens are
zeroed at padded positions (the contract allows arbitrary values there, but
zeros make debugging dumps and accidental un-masked pooling less misleading).

Autocast / device notes
-----------------------
No device is ever hard-coded; everything derives from the input. All ops on
tensors produced under ``torch.autocast`` are out-of-place
(``masked_fill``, not ``masked_fill_``), so bf16/fp16 autocast runs cleanly.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from aether.perception.interfaces import RecurrentEncoderConfig, SequenceEncoder

__all__ = ["RecurrentEncoder"]


class RecurrentEncoder(SequenceEncoder):
    """Linear → LSTM → (merge directions) → LayerNorm. See module docstring.

    Contract (``SequenceEncoder``):
        forward(x [B, L, input_dim], pad_mask [B, L] bool or None)
            -> tokens [B, L, d_model]

    * ``cfg.bidirectional=False`` (default): output token *t* summarizes bars
      ``0..t`` — causal per position.
    * ``cfg.bidirectional=True``: the LSTM runs both directions and the
      concatenated ``2·d_model`` features are mixed back to ``d_model`` with a
      learned linear map. Only use this where per-position causality is not
      required (the whole window is still pre-anchor data).
    * ``pad_mask`` does not alter the computation (left-pad wash-in, see
      module docstring) but padded output positions are zeroed before return.
    """

    def __init__(self, input_dim: int, cfg: RecurrentEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            batch_first=True,
            # nn.LSTM applies dropout only BETWEEN stacked layers; passing a
            # nonzero value with n_layers == 1 triggers a warning, so gate it.
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        # A bidirectional LSTM emits [B, L, 2*d_model] (forward ‖ backward);
        # project back to the contract width. Identity keeps the
        # unidirectional path free of an extra matmul.
        self.merge: nn.Module = (
            nn.Linear(2 * cfg.d_model, cfg.d_model)
            if cfg.bidirectional
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output_dim: int = cfg.d_model

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """Encode x [B, L, input_dim] -> contextual tokens [B, L, d_model].

        ``pad_mask`` ([B, L] bool, True = padding) is ignored for the
        recurrence itself (see module docstring) and used only to zero the
        returned tokens at padded positions.
        """
        h = self.in_proj(x)
        # Default zero (h0, c0) initial state: correct for windows that start
        # "cold", and the left-pad wash-in handles short histories.
        out, _ = self.lstm(h)                       # [B, L, D or 2D]
        out = self.merge(out)                       # [B, L, d_model]
        out = self.norm(out)
        if pad_mask is not None:
            # Cosmetic only — the contract says consumers must mask padded
            # positions anyway. Out-of-place fill keeps autocast happy.
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out
