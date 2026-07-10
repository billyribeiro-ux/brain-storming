"""Encoder factory for the perception layer.

The perception model never instantiates a concrete encoder class directly —
it asks :func:`build_encoder` for one by *name*. This indirection is what
makes the architecture NAS-swappable: ``PerceptionConfig.encoder_1m/5m/daily``
are plain strings, so a later self-evolution loop can mutate the architecture
by editing config alone, with zero code changes.

Available kinds
---------------
``transformer``
    :class:`~aether.perception.encoders.bar_transformer.BarTransformerEncoder`
    — self-attention over bars (RoPE or sinusoidal positions).
``ssm``
    :class:`~aether.perception.encoders.ssm.SSMEncoder` — diagonal
    state-space (S4D-style) long-memory encoder.
``lstm``
    :class:`~aether.perception.encoders.recurrent.RecurrentEncoder` —
    stacked (causal, unidirectional) LSTM.
``wavelet``
    :class:`~aether.perception.encoders.wavelet.WaveletEncoder` — learned
    multi-resolution causal filterbank.
``wavelet+transformer``
    :class:`WaveletTransformerEncoder` (defined below) — the filterbank as a
    front-end tokenizer feeding the transformer; multi-scale local structure
    first, global context second.

Import discipline
-----------------
Concrete encoder modules are imported *lazily* — inside
:func:`build_encoder` / :class:`WaveletTransformerEncoder`'s ``__init__`` and
via module-level ``__getattr__`` (PEP 562) for the re-exports — so importing
``aether.perception.encoders`` never drags in every architecture (and cannot
participate in an import cycle with sibling modules).
"""

from __future__ import annotations

from importlib import import_module
from typing import Optional

from torch import Tensor

from aether.perception.interfaces import (
    PerceptionConfig,
    SequenceEncoder,
    TransformerEncoderConfig,
    WaveletEncoderConfig,
)

#: The exact set of strings accepted by :func:`build_encoder` (and therefore
#: valid values for ``PerceptionConfig.encoder_1m/5m/daily``).
ENCODER_KINDS: tuple[str, ...] = (
    "transformer",
    "ssm",
    "lstm",
    "wavelet",
    "wavelet+transformer",
)


class WaveletTransformerEncoder(SequenceEncoder):
    """Composite encoder: learned multi-resolution filterbank → transformer.

    Motivation
    ----------
    A raw transformer attends over *single-bar* tokens; it must spend early
    layers just discovering local, multi-scale structure (bursts, drifts,
    volatility clusters spanning a handful of bars). The wavelet front-end
    hands it that structure for free: each position's token already mixes
    ``n_scales`` learned causal frequency bands, so self-attention operates
    on scale-aware tokens and can devote its capacity to *global* relations
    across the window.

    Both stages honor the :class:`SequenceEncoder` contract — one output
    token per input step, padding handled via ``pad_mask`` — so the composite
    trivially satisfies it too. The wavelet stage is strictly causal
    (dilated causal convolutions), preserving the no-look-ahead invariant.

    Parameters
    ----------
    input_dim:
        Number of raw features per bar.
    wavelet_cfg:
        Config for the filterbank front-end (defines the shared ``d_model``).
    transformer_cfg:
        Config for the transformer back-end. Its ``d_model`` must equal the
        wavelet stage's output width (``PerceptionConfig.__post_init__``
        enforces one width everywhere).
    """

    def __init__(
        self,
        input_dim: int,
        wavelet_cfg: WaveletEncoderConfig,
        transformer_cfg: TransformerEncoderConfig,
    ) -> None:
        super().__init__()
        # Lazy imports: keep the package importable without pulling every
        # architecture module at import time (see module docstring).
        from aether.perception.encoders.bar_transformer import BarTransformerEncoder
        from aether.perception.encoders.wavelet import WaveletEncoder

        self.wavelet = WaveletEncoder(input_dim, wavelet_cfg)
        # The transformer consumes the filterbank's d_model-wide tokens.
        self.transformer = BarTransformerEncoder(
            self.wavelet.output_dim, transformer_cfg
        )
        self.output_dim: int = self.transformer.output_dim

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """[B, L, input_dim] → [B, L, d_model]; ``pad_mask`` True = padding."""
        # Same pad_mask for both stages: neither changes sequence length.
        return self.transformer(self.wavelet(x, pad_mask), pad_mask)


def build_encoder(kind: str, input_dim: int, cfg: PerceptionConfig) -> SequenceEncoder:
    """Instantiate a sequence encoder by config name.

    Parameters
    ----------
    kind:
        One of :data:`ENCODER_KINDS`.
    input_dim:
        Per-step feature count of the stream this encoder will consume
        (``N_BAR_FEATURES`` for intraday, ``N_DAILY_FEATURES`` for daily).
    cfg:
        The full :class:`PerceptionConfig`; the relevant per-family
        sub-config is selected here so callers stay architecture-agnostic.

    Raises
    ------
    ValueError
        If ``kind`` is not a recognized encoder family.
    """
    # Imports live inside the branches so that (a) unused architectures are
    # never imported and (b) no import cycle can form with sibling modules.
    if kind == "transformer":
        from aether.perception.encoders.bar_transformer import BarTransformerEncoder

        return BarTransformerEncoder(input_dim, cfg.transformer)
    if kind == "ssm":
        from aether.perception.encoders.ssm import SSMEncoder

        return SSMEncoder(input_dim, cfg.ssm)
    if kind == "lstm":
        from aether.perception.encoders.recurrent import RecurrentEncoder

        return RecurrentEncoder(input_dim, cfg.recurrent)
    if kind == "wavelet":
        from aether.perception.encoders.wavelet import WaveletEncoder

        return WaveletEncoder(input_dim, cfg.wavelet)
    if kind == "wavelet+transformer":
        return WaveletTransformerEncoder(input_dim, cfg.wavelet, cfg.transformer)
    raise ValueError(
        f"Unknown encoder kind {kind!r}; valid kinds are: "
        + ", ".join(repr(k) for k in ENCODER_KINDS)
    )


# --------------------------------------------------------------------------- #
# Lazy re-exports (PEP 562): `from aether.perception.encoders import
# BarTransformerEncoder` works without importing every sibling at package
# import time.
# --------------------------------------------------------------------------- #

_LAZY_EXPORTS: dict[str, str] = {
    "BarTransformerEncoder": "aether.perception.encoders.bar_transformer",
    "RecurrentEncoder": "aether.perception.encoders.recurrent",
    "SSMEncoder": "aether.perception.encoders.ssm",
    "WaveletEncoder": "aether.perception.encoders.wavelet",
}

__all__ = [
    "ENCODER_KINDS",
    "BarTransformerEncoder",
    "RecurrentEncoder",
    "SSMEncoder",
    "WaveletEncoder",
    "WaveletTransformerEncoder",
    "build_encoder",
]


def __getattr__(name: str):  # noqa: ANN202 — PEP 562 module hook
    """Resolve lazily re-exported encoder classes on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        return getattr(import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
