"""The assembled perception model.

Architecture (one forward pass)
-------------------------------
::

    ticker_id ──► DNA embedding table ──────────────────────────┐
                        │                                       │
                        ▼ FiLM (per stream)                     ▼
    bars_1m  ──► x·(1+scale)+shift ──► encoder_1m  ──► tokens ──┤
    bars_5m  ──► x·(1+scale)+shift ──► encoder_5m  ──► tokens ──┼─► CrossModal
    daily    ──► x·(1+scale)+shift ──► encoder_dly ──► tokens ──┘   Fusion
                                                                      │
                                              fused state [B, D] ◄───┘
                                                │
              ┌──────────────┬──────────────────┼───────────────┐
              ▼              ▼                  ▼               ▼
        MaskedBarHead  Contrastive           DNAHead      EvidentialHead
        (1m tokens)    (1m vs 5m pool)       (fused)      (fused → γ,ν,α,β)

FiLM conditioning ("one brain, many instruments")
-------------------------------------------------
All tickers share ONE set of encoder weights — that is what lets structure
learned on liquid names transfer to the whole universe. But instruments have
genuinely different dynamics (an index has no real "volume", a high-beta
name has fatter intraday tails). Feature-wise Linear Modulation (FiLM,
Perez et al. 2018) reconciles the two: a per-stream linear map turns the
ticker's learned DNA embedding into a per-feature ``(scale, shift)`` applied
to the raw features *before* encoding::

    x ← x · (1 + scale(dna)) + shift(dna)

so the SAME encoder weights see instrument-recalibrated inputs and can
specialize per instrument without per-instrument parameters. The FiLM maps
are zero-initialized ⇒ they start as the identity and only bend away from it
where the data demands.

Masked-bar modeling & the mask token
------------------------------------
For self-supervision, random 1-minute bars are replaced by one learned mask
token (in raw feature space, before FiLM — so even the mask token gets
instrument-conditioned, keeping the encoder's input distribution
consistent). The encoder must infer the hidden bars from context; the
MaskedBarHead decodes its tokens back to the TRUE features.

Anomaly score
-------------
The same reconstruction machinery yields a market-surprise signal for free:
on the UNMASKED 1-minute stream, decode every token back to feature space
and measure the per-sample mean squared residual. A model that has seen a
regime reconstructs it well; residuals spike when the tape "looks wrong".
Raw residual scale drifts during training, so it is normalized by running
EMA statistics (mean/variance buffers, updated only in training mode) and
clamped at zero::

    anomaly = max(0, (err − EMA_mean) / sqrt(EMA_var))

giving a non-negative "standard deviations above normal surprise" score.
Only the *upper* tail is interesting — reconstructing unusually well is not
an anomaly — hence the clamp.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from aether.config import TICKERS
from aether.perception.encoders import build_encoder
from aether.perception.encoders.fusion import CrossModalFusion
from aether.perception.heads import (
    CrossScaleContrastiveHead,
    DNAHead,
    EvidentialHead,
    MaskedBarHead,
)
from aether.perception.interfaces import (
    N_BAR_FEATURES,
    N_DAILY_FEATURES,
    PerceptionBatch,
    PerceptionConfig,
    PerceptionOutput,
    masked_mean,
)

#: Decay of the running (EMA) statistics that normalize the anomaly score.
#: 0.99 ≈ a ~100-step effective memory: slow enough to be stable, fast
#: enough to track the falling reconstruction error early in training.
_ANOMALY_EMA_DECAY: float = 0.99


class PerceptionModel(nn.Module):
    """Multi-stream, self-supervised market perception model.

    Parameters
    ----------
    cfg:
        Full :class:`PerceptionConfig` (window geometry, encoder families,
        fusion and heads hyper-parameters).
    n_tickers:
        Size of the DNA embedding table / DNA classification head. Defaults
        to the fixed Aether universe (``len(config.TICKERS)``); tests may
        pass a different value.
    """

    def __init__(self, cfg: PerceptionConfig, n_tickers: int = len(TICKERS)) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Ticker DNA ---------------------------------------------------
        # One learned vector per instrument. It conditions the encoders
        # (FiLM), the fusion (DNA context token) and is supervised by the
        # DNAHead — three gradient paths shaping one identity code.
        self.dna_table = nn.Embedding(n_tickers, cfg.dna_dim)

        # ---- FiLM conditioners (one per stream) ---------------------------
        # dna → per-feature (scale, shift), applied as x·(1+scale)+shift.
        # Zero init ⇒ exact identity at step 0: training starts from the
        # instrument-agnostic model and *learns* where instruments differ.
        self.film_1m = nn.Linear(cfg.dna_dim, 2 * N_BAR_FEATURES)
        self.film_5m = nn.Linear(cfg.dna_dim, 2 * N_BAR_FEATURES)
        self.film_daily = nn.Linear(cfg.dna_dim, 2 * N_DAILY_FEATURES)
        for film in (self.film_1m, self.film_5m, self.film_daily):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        # ---- Per-stream encoders (config-selected families) ----------------
        self.encoder_1m = build_encoder(cfg.encoder_1m, N_BAR_FEATURES, cfg)
        self.encoder_5m = build_encoder(cfg.encoder_5m, N_BAR_FEATURES, cfg)
        self.encoder_daily = build_encoder(cfg.encoder_daily, N_DAILY_FEATURES, cfg)

        # ---- Mask token for masked-bar modeling on the 1m stream -----------
        # Lives in RAW feature space (replaces a bar's features before FiLM
        # and encoding). Zero init = "an average bar" under the normalized
        # feature scheme; it drifts to whatever placeholder the encoder finds
        # most informative.
        self.mask_token = nn.Parameter(torch.zeros(N_BAR_FEATURES))

        # ---- Fusion + heads -------------------------------------------------
        self.fusion = CrossModalFusion(cfg.fusion, cfg.dna_dim)
        self.masked_head = MaskedBarHead(cfg.d_model, N_BAR_FEATURES)
        self.contrastive_head = CrossScaleContrastiveHead(
            cfg.d_model, proj_dim=128, temperature=cfg.heads.temperature
        )
        self.dna_head = DNAHead(cfg.d_model, n_tickers)
        self.evidential_head = EvidentialHead(cfg.d_model, coef=cfg.heads.evidential_coef)

        # ---- Anomaly EMA statistics (buffers: saved with the checkpoint,
        # moved with .to(device), NOT optimized). Updated only in training
        # mode so evaluation/inference never contaminates the baseline.
        self.register_buffer("anomaly_mean", torch.zeros(()))
        self.register_buffer("anomaly_var", torch.ones(()))
        self.register_buffer(
            "anomaly_initialized", torch.zeros((), dtype=torch.bool)
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _film(x: Tensor, film: nn.Linear, dna: Tensor) -> Tensor:
        """Apply FiLM conditioning: ``x·(1+scale)+shift`` per feature.

        x [B, L, F], dna [B, dna_dim] → [B, L, F]. The ``1+scale``
        parameterization keeps zero-initialized FiLM an exact identity.
        """
        scale, shift = film(dna).chunk(2, dim=-1)          # each [B, F]
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _encode_1m(
        self, batch: PerceptionBatch, mask_positions: Optional[Tensor], dna: Tensor
    ) -> Tensor:
        """Encode the 1m stream, optionally with masked-bar substitution."""
        x = batch.bars_1m
        if mask_positions is not None:
            # Replace masked bars with the learned mask token in raw feature
            # space. torch.where broadcasts the [F] token over [B, L, F];
            # no in-place writes (autocast-safe).
            x = torch.where(
                mask_positions.unsqueeze(-1),
                self.mask_token.to(dtype=x.dtype),
                x,
            )
        return self.encoder_1m(self._film(x, self.film_1m, dna), batch.pad_1m)

    def _anomaly_score(
        self,
        batch: PerceptionBatch,
        dna: Tensor,
        tokens_1m: Tensor,
        was_masked: bool,
    ) -> Tensor:
        """Non-negative reconstruction-surprise score, [B] float32.

        Decodes the UNMASKED 1m encoding back to feature space and measures
        the per-sample mean squared residual over valid bars, then
        normalizes by the EMA baseline (see module docstring).

        Runs entirely under ``no_grad``: anomaly is a diagnostic output, not
        a training signal — reconstruction learning happens through the
        masked-bar loss, and letting this path carry gradient would let the
        model optimize its own surprise meter.
        """
        with torch.no_grad():
            if was_masked:
                # The masked forward corrupted the inputs; re-encode the
                # clean stream (extra compute paid only during SSL training).
                clean_tokens = self._encode_1m(batch, None, dna)
            else:
                clean_tokens = tokens_1m
            recon = self.masked_head(clean_tokens)                     # [B, L, F]
            # Residuals in float32: EMA buffers are float32 and bf16 squared
            # errors are too coarse near convergence.
            residual = (recon.float() - batch.bars_1m.float()).pow(2).mean(dim=-1)
            valid = (~batch.pad_1m).float()                            # [B, L]
            err = (residual * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

            if self.training:
                # EMA update of the surprise baseline (training only).
                batch_mean = err.mean()
                # Population variance; a 1-sample batch contributes 0 var.
                batch_var = err.var(unbiased=False).clamp_min(1e-12)
                if bool(self.anomaly_initialized):
                    self.anomaly_mean.mul_(_ANOMALY_EMA_DECAY).add_(
                        batch_mean * (1.0 - _ANOMALY_EMA_DECAY)
                    )
                    self.anomaly_var.mul_(_ANOMALY_EMA_DECAY).add_(
                        batch_var * (1.0 - _ANOMALY_EMA_DECAY)
                    )
                else:
                    # First training batch seeds the statistics directly —
                    # avoids a long burn-in from the arbitrary (0, 1) prior.
                    self.anomaly_mean.copy_(batch_mean)
                    self.anomaly_var.copy_(batch_var)
                    self.anomaly_initialized.fill_(True)

            # z-score against the running baseline; clamp at 0 because only
            # ABOVE-normal surprise is anomalous.
            z = (err - self.anomaly_mean) / torch.sqrt(self.anomaly_var + 1e-8)
            return z.clamp_min(0.0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def forward(
        self, batch: PerceptionBatch, mask_positions: Optional[Tensor] = None
    ) -> PerceptionOutput:
        """Encode a batch of market moments into a :class:`PerceptionOutput`.

        Parameters
        ----------
        batch:
            The input :class:`PerceptionBatch`.
        mask_positions:
            Optional [B, len_1m] bool — True where 1m bars should be replaced
            by the learned mask token (masked-bar modeling). ``None`` for
            plain inference. When provided, ``tokens_1m`` in the output come
            from the *masked* forward (that is what the reconstruction loss
            needs); the anomaly score always uses a clean encoding.
        """
        dna = self.dna_table(batch.ticker_id)                     # [B, dna_dim]

        # Per-stream: FiLM-condition on DNA, then encode. Same weights for
        # every instrument; FiLM makes them behave per-instrument.
        tokens_1m = self._encode_1m(batch, mask_positions, dna)
        tokens_5m = self.encoder_5m(
            self._film(batch.bars_5m, self.film_5m, dna), batch.pad_5m
        )
        tokens_daily = self.encoder_daily(
            self._film(batch.daily, self.film_daily, dna), batch.pad_daily
        )

        # Cross-modal fusion → the single state vector for this moment.
        fused = self.fusion(
            [tokens_1m, tokens_5m, tokens_daily],
            [batch.pad_1m, batch.pad_5m, batch.pad_daily],
            dna,
        )

        # Evidential parameters + uncertainty decomposition.
        evidential = self.evidential_head(fused)
        aleatoric, epistemic = EvidentialHead.uncertainties(evidential)

        # Reconstruction-surprise anomaly score (no_grad diagnostic).
        anomaly = self._anomaly_score(
            batch, dna, tokens_1m, was_masked=mask_positions is not None
        )

        return PerceptionOutput(
            tokens_1m=tokens_1m,
            tokens_5m=tokens_5m,
            tokens_daily=tokens_daily,
            fused=fused,
            dna=dna,
            evidential=evidential,
            aleatoric=aleatoric,
            epistemic=epistemic,
            anomaly=anomaly,
        )

    def self_supervised_losses(
        self, batch: PerceptionBatch
    ) -> tuple[dict[str, Tensor], PerceptionOutput]:
        """One full SSL step: sample a mask, forward, compute all losses.

        Returns ``(losses, output)`` where ``losses`` holds scalar tensors
        for ``'masked'``, ``'contrastive'``, ``'dna'``, ``'evidential'`` and
        their weighted sum ``'total'`` (weights from :class:`HeadsConfig`).
        """
        heads_cfg = self.cfg.heads
        pad_1m = batch.pad_1m

        # --- Sample mask positions: Bernoulli(mask_ratio) over NON-PAD 1m
        # bars. Padding must never be "masked" (there is nothing to
        # reconstruct there and the loss only averages over mask positions).
        rand = torch.rand(pad_1m.shape, device=pad_1m.device)
        mask_positions = (rand < heads_cfg.mask_ratio) & ~pad_1m
        # Never mask the final (anchor) bar: with left-padding it is always
        # the last position. The anchor bar is the state hand-off point —
        # the freshest observable information at decision time t. Hiding it
        # would train the model to act on a market state whose most recent
        # bar it was never allowed to see, corrupting exactly the embedding
        # the world model / decision core consumes.
        mask_positions[:, -1] = False

        out = self.forward(batch, mask_positions=mask_positions)

        losses: dict[str, Tensor] = {}
        # 1. Masked-bar reconstruction: decode masked-forward tokens back to
        #    the TRUE (uncorrupted) features at the masked positions.
        pred = self.masked_head(out.tokens_1m)
        losses["masked"] = self.masked_head.loss(pred, batch.bars_1m, mask_positions)
        # 2. Cross-scale contrastive: pooled 1m view vs pooled 5m view of the
        #    same moment (masking doubles as augmentation on the 1m side).
        pooled_1m = masked_mean(out.tokens_1m, pad_1m)
        pooled_5m = masked_mean(out.tokens_5m, batch.pad_5m)
        losses["contrastive"] = self.contrastive_head.loss(pooled_1m, pooled_5m)
        # 3. Instrument identity from the fused state.
        losses["dna"] = self.dna_head.loss(out.fused, batch.ticker_id)
        # 4. Evidential regression on the next-horizon log return.
        losses["evidential"] = self.evidential_head.loss(
            out.evidential, batch.target_ret
        )
        # Weighted total — the single scalar the optimizer sees.
        losses["total"] = (
            heads_cfg.w_masked * losses["masked"]
            + heads_cfg.w_contrastive * losses["contrastive"]
            + heads_cfg.w_dna * losses["dna"]
            + heads_cfg.w_evidential * losses["evidential"]
        )
        return losses, out

    @property
    def num_parameters(self) -> int:
        """Total number of parameters (trainable or not; buffers excluded)."""
        return sum(p.numel() for p in self.parameters())
