"""Self-supervised objective heads for the perception model.

The perception layer trains WITHOUT labels. Four complementary objectives
force the encoders to build a rich, calibrated market representation:

1. :class:`MaskedBarHead` — masked-bar modeling (the BERT idea on price
   bars): hide random bars, reconstruct their true features from context.
   Solves "what does the local microstructure imply about a missing bar?"
2. :class:`CrossScaleContrastiveHead` — symmetric InfoNCE between the pooled
   1-minute view and the pooled 5-minute view of the SAME moment. The two
   timescales are two "augmentations" of one underlying market state; making
   them agree (against in-batch negatives) forces scale-consistent features.
3. :class:`DNAHead` — classify which instrument produced the window from the
   fused state. Keeps instrument identity linearly decodable, which in turn
   gives the DNA embedding table a gradient signal to differentiate tickers.
4. :class:`EvidentialHead` — Deep Evidential Regression (Amini et al., 2020)
   on the next-horizon log return. Instead of predicting a point value, the
   head predicts the parameters of a Normal-Inverse-Gamma (NIG) distribution
   over Gaussian outcomes, yielding *calibrated* aleatoric and epistemic
   uncertainty in a single forward pass — no ensembles, no MC dropout.

Every ``loss`` method returns a scalar tensor and is hardened against NaNs:
all ``log`` arguments are clamped away from zero, all mean denominators are
clamped ≥ 1, and the numerically delicate math (NIG likelihood, InfoNCE
logits) is computed in float32 even under ``torch.autocast``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: Floor for arguments of ``log`` — keeps every loss finite even in the
#: pathological corner where a softplus output underflows.
_EPS: float = 1e-12


# --------------------------------------------------------------------------- #
# 1. Masked-bar modeling
# --------------------------------------------------------------------------- #

class MaskedBarHead(nn.Module):
    """Reconstruct true bar features at masked positions.

    A 2-layer MLP decodes each contextual token back to the raw feature
    space::

        pred = W2 · GELU(W1 · token)          # [B, L, D] → [B, L, F]

    The loss is mean-squared error, averaged **only over masked positions**::

        L = Σ_{b,l: mask} ‖pred_{b,l} − target_{b,l}‖² / F   ÷   max(#masked, 1)

    i.e. first a mean over the F feature channels per position, then a mean
    over all masked positions in the batch. Averaging over the masked count
    (not B·L) keeps the loss scale independent of mask ratio and sequence
    length; the ``max(·, 1)`` clamp makes a mask-free batch yield 0, not NaN.

    Callers must pass masks that already exclude padding positions (the
    model samples masks over non-pad positions only), so "masked & non-pad"
    reduces to ``mask_positions`` here.
    """

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """[B, L, d_model] contextual tokens → [B, L, n_features] predictions."""
        return self.net(tokens)

    def loss(
        self, pred: Tensor, target_features: Tensor, mask_positions: Tensor
    ) -> Tensor:
        """MSE over masked (non-pad) positions only.

        Parameters
        ----------
        pred:
            [B, L, F] reconstructed features (from :meth:`forward`).
        target_features:
            [B, L, F] the TRUE (unmasked) input features.
        mask_positions:
            [B, L] bool — True where the input bar was replaced by the mask
            token. Must not include padding positions.
        """
        # float32 math regardless of autocast: squared errors in bf16 lose
        # precision exactly where the loss should be smallest.
        per_position = (pred.float() - target_features.float()).pow(2).mean(dim=-1)
        weights = mask_positions.to(per_position.dtype)              # [B, L]
        denom = weights.sum().clamp_min(1.0)                         # ≥ 1
        return (per_position * weights).sum() / denom


# --------------------------------------------------------------------------- #
# 2. Cross-scale contrastive alignment
# --------------------------------------------------------------------------- #

class CrossScaleContrastiveHead(nn.Module):
    """Symmetric InfoNCE between pooled 1-minute and 5-minute views.

    Each scale gets its own projection MLP (the standard SimCLR-style
    "projection head" — contrastive structure lives in the projected space so
    the backbone tokens stay information-rich)::

        z¹ = normalize(g₁(pooled_1m)),  z⁵ = normalize(g₅(pooled_5m))

    With batch size B, the logit matrix is ``S = z¹ z⁵ᵀ / τ`` ∈ ℝ^{B×B}.
    Sample i's positive is the *same sample's* other-scale view (diagonal);
    the other B−1 rows/columns are negatives. Symmetric InfoNCE::

        L = ½ [ CE(S, diag) + CE(Sᵀ, diag) ]

    L2 normalization puts logits on the unit sphere so the temperature τ is
    the only scale knob; small τ (default 0.07) sharpens the softmax and
    focuses gradient on hard negatives.
    """

    def __init__(
        self, d_model: int, proj_dim: int = 128, temperature: float = 0.07
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        # Separate projections: the two scales live in differently-shaped
        # token statistics; forcing one shared map would waste capacity on
        # undoing that mismatch.
        self.proj_1m = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim)
        )
        self.proj_5m = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim)
        )

    def loss(self, pooled_1m: Tensor, pooled_5m: Tensor) -> Tensor:
        """Symmetric InfoNCE across the batch.

        Parameters
        ----------
        pooled_1m / pooled_5m:
            [B, d_model] masked-mean-pooled tokens of each stream, same
            sample order (row i of both tensors is the same moment).
        """
        # Project, then normalize in float32 — the division by a near-zero
        # norm and the τ-scaled logits are precision-sensitive under autocast.
        z1 = F.normalize(self.proj_1m(pooled_1m).float(), dim=-1, eps=1e-6)
        z5 = F.normalize(self.proj_5m(pooled_5m).float(), dim=-1, eps=1e-6)
        logits = z1 @ z5.t() / self.temperature                      # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        # Both directions: 1m retrieves its 5m partner AND vice versa.
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )


# --------------------------------------------------------------------------- #
# 3. Ticker-identity (DNA) classification
# --------------------------------------------------------------------------- #

class DNAHead(nn.Module):
    """Classify the instrument from the fused state vector.

    A single linear probe ``d_model → n_tickers`` trained with cross-entropy
    against ``batch.ticker_id``. Two purposes:

    * it back-propagates an identity-separating gradient into the DNA
      embedding table (via FiLM and the fusion DNA token), so instruments
      with different dynamics get genuinely different DNA vectors;
    * it acts as a regularizing probe — the fused state must retain "which
      instrument is this" rather than collapsing to an averaged market.

    The weight for this loss is small (``HeadsConfig.w_dna``): identity must
    be *decodable*, not dominate the representation.
    """

    def __init__(self, d_model: int, n_tickers: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, n_tickers)

    def forward(self, fused: Tensor) -> Tensor:
        """[B, d_model] fused state → [B, n_tickers] logits."""
        return self.classifier(fused)

    def loss(self, fused: Tensor, ticker_id: Tensor) -> Tensor:
        """Cross-entropy of the linear probe vs. the true ticker ids ([B] int64)."""
        return F.cross_entropy(self.forward(fused), ticker_id)


# --------------------------------------------------------------------------- #
# 4. Deep Evidential Regression
# --------------------------------------------------------------------------- #

class EvidentialHead(nn.Module):
    """Normal-Inverse-Gamma head — Deep Evidential Regression (Amini 2020).

    Model
    -----
    Assume the target (next-horizon log return) is Gaussian with unknown
    mean μ and variance σ², and place the conjugate NIG prior over (μ, σ²)::

        μ  ~ 𝒩(γ, σ²/ν),      σ² ~ Γ⁻¹(α, β)

    The head predicts the four evidential parameters from the fused state
    with a 2-layer MLP (``d_model → d_model → 4``), constrained to their
    valid domains::

        γ = raw₀                        (mean — unconstrained)
        ν = softplus(raw₁) + 1e-4       (> 0, virtual evidence for the mean)
        α = softplus(raw₂) + 1 + 1e-4   (> 1, virtual evidence for variance)
        β = softplus(raw₃) + 1e-4       (> 0, variance scale)

    Loss
    ----
    Marginalizing (μ, σ²) gives a Student-t predictive; its negative log
    likelihood, with Ω = 2β(1+ν)::

        NLL(y) = ½·log(π/ν) − α·log Ω + (α+½)·log((y−γ)²·ν + Ω)
                 + lgamma(α) − lgamma(α+½)

    Fitting NLL alone lets the network claim high evidence everywhere, so an
    evidence regularizer penalizes confidence in proportion to the error::

        R(y)  = |y − γ| · (2ν + α)          (total evidence Φ = 2ν + α)
        Loss  = mean(NLL) + coef · mean(R)

    Uncertainty decomposition
    -------------------------
    ::

        aleatoric  = 𝔼[σ²]      = β / (α − 1)        irreducible market noise
        epistemic  = Var[μ]     = β / (ν (α − 1))    model ignorance — shrinks
                                                     as evidence ν grows

    All the delicate math runs in float32 (raw outputs are upcast before the
    softplus) so the head is autocast-safe, and every log argument is clamped
    ≥ 1e-12.
    """

    #: Names and order of the evidential parameters, as they appear in the
    #: dict returned by :meth:`forward` (and in ``PerceptionOutput.evidential``).
    PARAM_NAMES: tuple[str, ...] = ("gamma", "nu", "alpha", "beta")

    def __init__(self, d_model: int, coef: float = 0.01) -> None:
        super().__init__()
        self.coef = float(coef)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, fused: Tensor) -> dict[str, Tensor]:
        """[B, d_model] fused state → NIG parameters, each [B] float32."""
        # Upcast BEFORE the constraint transforms: softplus in bf16 flushes
        # small evidence values to zero, which the NLL's log(π/ν) hates.
        raw = self.net(fused).float()                       # [B, 4]
        gamma = raw[..., 0]
        nu = F.softplus(raw[..., 1]) + 1e-4                 # > 0
        alpha = F.softplus(raw[..., 2]) + 1.0 + 1e-4        # > 1 (E[σ²] finite)
        beta = F.softplus(raw[..., 3]) + 1e-4               # > 0
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def nig_nll(self, params: dict[str, Tensor], y: Tensor) -> Tensor:
        """Per-sample NIG negative log likelihood, [B] float32."""
        y = y.float()
        gamma, nu = params["gamma"], params["nu"]
        alpha, beta = params["alpha"], params["beta"]
        omega = 2.0 * beta * (1.0 + nu)
        return (
            0.5 * (math.log(math.pi) - torch.log(nu.clamp_min(_EPS)))
            - alpha * torch.log(omega.clamp_min(_EPS))
            + (alpha + 0.5)
            * torch.log(((y - gamma).pow(2) * nu + omega).clamp_min(_EPS))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

    def evidence_regularizer(self, params: dict[str, Tensor], y: Tensor) -> Tensor:
        """Per-sample penalty |y−γ|·Φ with total evidence Φ = 2ν + α, [B]."""
        return (y.float() - params["gamma"]).abs() * (
            2.0 * params["nu"] + params["alpha"]
        )

    def loss(self, params: dict[str, Tensor], y: Tensor) -> Tensor:
        """Scalar total loss: mean NLL + ``coef`` · mean evidence penalty.

        Parameters
        ----------
        params:
            Output of :meth:`forward` — {"gamma","nu","alpha","beta"} each [B].
        y:
            [B] regression target (next-horizon log return).
        """
        return self.nig_nll(params, y).mean() + self.coef * self.evidence_regularizer(
            params, y
        ).mean()

    @staticmethod
    def uncertainties(params: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Decompose predictive uncertainty → (aleatoric, epistemic), each [B].

        aleatoric = β/(α−1) — expected observation variance (market noise);
        epistemic = β/(ν(α−1)) — variance of the predicted mean itself,
        i.e. how little evidence the model has; the 1e-6 keeps the division
        safe even though α > 1 by construction.
        """
        denom = params["alpha"] - 1.0 + 1e-6
        aleatoric = params["beta"] / denom
        epistemic = params["beta"] / (params["nu"] * denom)
        return aleatoric, epistemic
