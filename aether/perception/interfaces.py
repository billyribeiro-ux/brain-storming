"""THE PERCEPTION CONTRACT.

Every module in ``aether.perception`` builds against the types in this file.
It pins down, exactly:

* the canonical input representation (feature columns per timescale),
* window geometry (how much 1min / 5min / daily context a sample carries),
* the batch structure (`PerceptionBatch`),
* the encoder interface (`SequenceEncoder`),
* all configuration dataclasses,
* the output structure (`PerceptionOutput`) consumed by the world model
  and decision core in later releases.

Philosophy — "no hardcoded indicators"
--------------------------------------
Aether's mandate is that the network *invents* its own features. What the
preprocessing provides is therefore NOT technical analysis; it is a
scale-free, stationarity-friendly *reparameterization* of raw OHLCV (log
returns, bar geometry fractions, volume vs. that ticker's own time-of-day
norm). This is representation hygiene — the equivalent of pixel
normalization in vision — and preserves the information content of the raw
stream. No RSI, no MACD, no moving averages, no rules. Meaning is learned.

Ticker DNA
----------
Each instrument gets (a) a statistical fingerprint used only for input
normalization (`TickerStats`, fitted on TRAINING data only), and (b) a
learned embedding inside the model that lets every layer condition on the
instrument's identity. Both together are how Aether learns per-ticker
behavior instead of one averaged market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

# --------------------------------------------------------------------------- #
# Canonical feature representation
# --------------------------------------------------------------------------- #

#: Per-bar features for intraday (1min & 5min) streams, in this exact order.
#: All are scale-free or normalized per ticker; produced by
#: ``aether.perception.preprocessing.compute_bar_features``.
FEATURE_COLUMNS_BAR: tuple[str, ...] = (
    "ret_log",       # log(close_t / close_{t-1})            — price change
    "gap_log",       # log(open_t / close_{t-1})             — inter-bar gap
    "range_log",     # log(high_t / low_t)                   — bar expansion
    "body_frac",     # (close-open)/(high-low+eps)           — direction of travel
    "upper_wick",    # (high-max(o,c))/(high-low+eps)        — rejection above
    "lower_wick",    # (min(o,c)-low)/(high-low+eps)         — rejection below
    "vol_z",         # log1p(volume) z-scored vs this ticker's time-of-day norm
    "dollar_vol_z",  # log1p(close*volume) z-scored the same way
    "tod_sin",       # sin(2π · session_minute/390)          — session clock
    "tod_cos",       # cos(2π · session_minute/390)
    "dow_sin",       # sin(2π · weekday/5)                   — weekly clock
    "dow_cos",       # cos(2π · weekday/5)
)

#: Per-day features for the daily context stream, in this exact order.
FEATURE_COLUMNS_DAILY: tuple[str, ...] = (
    "ret_log",
    "gap_log",
    "range_log",
    "body_frac",
    "upper_wick",
    "lower_wick",
    "vol_z",         # z-scored vs this ticker's trailing daily volume stats
)

N_BAR_FEATURES: int = len(FEATURE_COLUMNS_BAR)
N_DAILY_FEATURES: int = len(FEATURE_COLUMNS_DAILY)


@dataclass
class WindowSpec:
    """Geometry of one perception sample, anchored at a 1-minute bar t.

    The sample sees ONLY information available at the anchor's close:
    * ``len_1m`` most recent 1min bars ending at t (inclusive),
    * ``len_5m`` most recent completed 5min bars,
    * ``len_daily`` most recent completed daily bars (yesterday backwards).

    Look-ahead safety is a hard invariant enforced by dataset tests.
    """

    len_1m: int = 240      # 4 trading hours of 1min bars
    len_5m: int = 156      # 2 full sessions of 5min bars
    len_daily: int = 60    # ~3 months of daily context
    horizon_1m: int = 1    # bars ahead for the self-supervised density target


# --------------------------------------------------------------------------- #
# Batch structure
# --------------------------------------------------------------------------- #

@dataclass
class PerceptionBatch:
    """One training/inference batch. Shapes use B=batch, L=length, F=features.

    Padding: sequences shorter than the window (early history) are LEFT-padded
    with zeros and flagged in the pad masks (True = padding position).
    """

    bars_1m: Tensor          # [B, len_1m, N_BAR_FEATURES] float32
    bars_5m: Tensor          # [B, len_5m, N_BAR_FEATURES] float32
    daily: Tensor            # [B, len_daily, N_DAILY_FEATURES] float32
    ticker_id: Tensor        # [B] int64 — index into config.TICKERS order
    minute_of_day: Tensor    # [B, len_1m] int64 in [0, 389]; 0 where padded
    pad_1m: Tensor           # [B, len_1m] bool — True = padding
    pad_5m: Tensor           # [B, len_5m] bool
    pad_daily: Tensor        # [B, len_daily] bool
    target_ret: Tensor       # [B] float32 — next-horizon log return (SSL target)
    anchor_ts: Optional[Tensor] = None  # [B] int64 epoch-seconds (bookkeeping)

    def to(self, device: torch.device | str) -> "PerceptionBatch":
        move = lambda t: t.to(device) if t is not None else None
        return PerceptionBatch(
            bars_1m=move(self.bars_1m), bars_5m=move(self.bars_5m),
            daily=move(self.daily), ticker_id=move(self.ticker_id),
            minute_of_day=move(self.minute_of_day),
            pad_1m=move(self.pad_1m), pad_5m=move(self.pad_5m),
            pad_daily=move(self.pad_daily), target_ret=move(self.target_ret),
            anchor_ts=move(self.anchor_ts),
        )

    @property
    def batch_size(self) -> int:
        return int(self.bars_1m.shape[0])


# --------------------------------------------------------------------------- #
# Encoder interface
# --------------------------------------------------------------------------- #

class SequenceEncoder(nn.Module):
    """Common interface for all sequence encoders.

    Subclasses MUST:
    * accept ``(input_dim: int, cfg: <their config dataclass>)`` in __init__,
    * set ``self.output_dim: int`` before __init__ returns,
    * implement ``forward(x, pad_mask=None) -> Tensor``
        x:        [B, L, input_dim] float
        pad_mask: [B, L] bool, True = padding (may be None)
        returns:  [B, L, output_dim] — one contextual token per input step.
    Padding positions may hold arbitrary values in the output; consumers mask.
    """

    output_dim: int

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


def masked_mean(tokens: Tensor, pad_mask: Optional[Tensor]) -> Tensor:
    """Mean-pool [B, L, D] over valid (non-pad) positions -> [B, D]."""
    if pad_mask is None:
        return tokens.mean(dim=1)
    valid = (~pad_mask).float().unsqueeze(-1)            # [B, L, 1]
    denom = valid.sum(dim=1).clamp_min(1.0)              # [B, 1]
    return (tokens * valid).sum(dim=1) / denom


# --------------------------------------------------------------------------- #
# Encoder configurations
# --------------------------------------------------------------------------- #

@dataclass
class TransformerEncoderConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    rope: bool = True          # rotary position embeddings (else sinusoidal)


@dataclass
class SSMEncoderConfig:
    """Diagonal state-space (S4D-style) long-memory encoder."""
    d_model: int = 256
    d_state: int = 64          # latent state size per channel
    n_layers: int = 4
    dropout: float = 0.1


@dataclass
class RecurrentEncoderConfig:
    d_model: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False   # MUST stay False for causal streams


@dataclass
class WaveletEncoderConfig:
    """Learnable multi-resolution filterbank front-end (wavelet-inspired).

    ``n_scales`` dilated causal convolution banks with kernel ``kernel_size``
    and exponentially increasing dilation decompose the sequence into
    frequency bands whose filters are LEARNED, then are mixed into d_model.
    """
    d_model: int = 256
    n_scales: int = 5
    kernel_size: int = 8
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Cross-modal fusion (Perceiver-style latent queries + cross-attention)."""
    d_model: int = 256
    n_heads: int = 8
    n_latents: int = 16        # learned query tokens that read all streams
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class HeadsConfig:
    """Self-supervised objective heads and loss weights."""
    mask_ratio: float = 0.25          # masked-bar-modeling mask fraction
    temperature: float = 0.07         # InfoNCE temperature
    evidential_coef: float = 0.01     # evidence regularizer strength
    w_masked: float = 1.0             # loss weights ->  total loss
    w_contrastive: float = 1.0
    w_dna: float = 0.2
    w_evidential: float = 1.0


@dataclass
class PerceptionConfig:
    """Assembles the whole perception model.

    ``encoder_1m/5m/daily`` select the encoder family per stream, so later
    self-evolution (NAS) can swap architectures via config alone:
        "transformer" | "ssm" | "lstm" | "wavelet" | "wavelet+transformer"
    """

    d_model: int = 256
    dna_dim: int = 64                 # learned per-ticker DNA embedding size
    window: WindowSpec = field(default_factory=WindowSpec)
    encoder_1m: str = "wavelet+transformer"
    encoder_5m: str = "ssm"
    encoder_daily: str = "lstm"
    transformer: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    ssm: SSMEncoderConfig = field(default_factory=SSMEncoderConfig)
    recurrent: RecurrentEncoderConfig = field(default_factory=RecurrentEncoderConfig)
    wavelet: WaveletEncoderConfig = field(default_factory=WaveletEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    def __post_init__(self) -> None:
        # One width everywhere keeps fusion simple and NAS-swappable.
        for sub in (self.transformer, self.ssm, self.recurrent,
                    self.wavelet, self.fusion):
            sub.d_model = self.d_model


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 50_000
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True                  # bf16/fp16 autocast on CUDA
    val_every: int = 1_000
    checkpoint_dir: str = "checkpoints/perception"
    log_jsonl: str = "runs/perception_train.jsonl"
    num_workers: int = 2
    seed: int = 1337


# --------------------------------------------------------------------------- #
# Output structure (the hand-off to the world model / decision core)
# --------------------------------------------------------------------------- #

@dataclass
class PerceptionOutput:
    """Everything the perception layer knows about a batch of moments.

    ``fused`` is the primary state vector consumed by later layers.
    Uncertainty follows the evidential (Normal-Inverse-Gamma) formulation:
    aleatoric  = β / (α - 1)         — irreducible market noise
    epistemic  = β / (ν (α - 1))     — model ignorance (shrinks with data)
    """

    tokens_1m: Tensor                 # [B, len_1m, D] contextual 1min tokens
    tokens_5m: Tensor                 # [B, len_5m, D]
    tokens_daily: Tensor              # [B, len_daily, D]
    fused: Tensor                     # [B, D] cross-modal state embedding
    dna: Tensor                       # [B, dna_dim] ticker DNA embedding used
    evidential: dict[str, Tensor]     # {"gamma","nu","alpha","beta"} each [B]
    aleatoric: Tensor                 # [B] variance — market noise
    epistemic: Tensor                 # [B] variance — model ignorance
    anomaly: Tensor                   # [B] ≥0 — normalized reconstruction surprise


# --------------------------------------------------------------------------- #
# Synthetic data helper (tests + smoke runs, no network / lake required)
# --------------------------------------------------------------------------- #

def synthetic_batch(batch_size: int = 4, spec: WindowSpec | None = None,
                    n_tickers: int = 10, seed: int = 0,
                    device: str = "cpu") -> PerceptionBatch:
    """Random but shape/dtype-faithful batch for unit tests."""
    spec = spec or WindowSpec()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g) * 0.1
    batch = PerceptionBatch(
        bars_1m=rnd(batch_size, spec.len_1m, N_BAR_FEATURES),
        bars_5m=rnd(batch_size, spec.len_5m, N_BAR_FEATURES),
        daily=rnd(batch_size, spec.len_daily, N_DAILY_FEATURES),
        ticker_id=torch.randint(0, n_tickers, (batch_size,), generator=g),
        minute_of_day=torch.randint(0, 390, (batch_size, spec.len_1m), generator=g),
        pad_1m=torch.zeros(batch_size, spec.len_1m, dtype=torch.bool),
        pad_5m=torch.zeros(batch_size, spec.len_5m, dtype=torch.bool),
        pad_daily=torch.zeros(batch_size, spec.len_daily, dtype=torch.bool),
        target_ret=rnd(batch_size) * 0.01,
        anchor_ts=torch.arange(batch_size, dtype=torch.int64),
    )
    return batch.to(device)
