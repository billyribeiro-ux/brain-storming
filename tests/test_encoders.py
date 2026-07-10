"""Pins the SequenceEncoder contract for every encoder family.

For each kind reachable through ``build_encoder`` ("transformer", "ssm",
"lstm", "wavelet", "wavelet+transformer"):

* [B, L, input_dim] (+ optional pad_mask) -> [B, L, d_model], float32;
* ``output_dim`` is set and truthful;
* gradients flow back to the inputs (i.e. through the input projection);
* deterministic under a fixed seed in eval mode.

Causality: the ssm / lstm / wavelet families are causal by design — output
at position p may depend only on inputs at positions <= p. The transformer
is *intentionally bidirectional* (it contextualizes a completed window; the
look-ahead boundary is enforced at the dataset level, not inside the
window) — a test below documents that design decision by asserting the
transformer is NOT causal.
"""

from __future__ import annotations

import pytest
import torch

from aether.perception.encoders import (
    BarTransformerEncoder,
    RecurrentEncoder,
    SSMEncoder,
    WaveletEncoder,
    build_encoder,
)
from aether.perception.interfaces import (
    N_BAR_FEATURES,
    PerceptionConfig,
    RecurrentEncoderConfig,
    SequenceEncoder,
    SSMEncoderConfig,
    TransformerEncoderConfig,
    WaveletEncoderConfig,
    FusionConfig,
    WindowSpec,
)

D_MODEL = 64
INPUT_DIM = N_BAR_FEATURES
B, L = 2, 40

ALL_KINDS = ("transformer", "ssm", "lstm", "wavelet", "wavelet+transformer")
CAUSAL_KINDS = ("ssm", "lstm", "wavelet")


def small_cfg() -> PerceptionConfig:
    """Tiny PerceptionConfig; __post_init__ propagates d_model=64 everywhere.

    Dropout is zeroed so eval/train mode cannot blur the determinism and
    causality comparisons.
    """
    return PerceptionConfig(
        d_model=D_MODEL,
        dna_dim=16,
        window=WindowSpec(len_1m=30, len_5m=12, len_daily=3, horizon_1m=1),
        transformer=TransformerEncoderConfig(n_heads=4, n_layers=2, dropout=0.0),
        ssm=SSMEncoderConfig(d_state=16, n_layers=2, dropout=0.0),
        recurrent=RecurrentEncoderConfig(n_layers=1, dropout=0.0),
        wavelet=WaveletEncoderConfig(n_scales=3, kernel_size=4, dropout=0.0),
        fusion=FusionConfig(n_heads=4, n_latents=4, n_layers=1, dropout=0.0),
    )


def _build(kind: str, seed: int = 1234) -> torch.nn.Module:
    torch.manual_seed(seed)
    enc = build_encoder(kind, INPUT_DIM, small_cfg())
    enc.eval()
    return enc


def _x(seed: int = 0, length: int = L) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, length, INPUT_DIM, generator=g)


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #

class TestInterface:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_output_shape_dtype_and_output_dim(self, kind: str) -> None:
        enc = _build(kind)
        assert hasattr(enc, "output_dim")
        assert enc.output_dim == D_MODEL
        with torch.no_grad():
            out = enc(_x())
        assert out.shape == (B, L, D_MODEL)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_pad_mask_accepted(self, kind: str) -> None:
        enc = _build(kind)
        pad = torch.zeros(B, L, dtype=torch.bool)
        pad[:, :6] = True                    # LEFT padding, per the contract
        with torch.no_grad():
            out = enc(_x(), pad_mask=pad)
        assert out.shape == (B, L, D_MODEL)
        # Valid positions must stay finite; padded positions may be anything.
        assert torch.isfinite(out[:, 6:]).all()

    def test_named_classes_follow_sequence_encoder(self) -> None:
        cfg = small_cfg()
        torch.manual_seed(0)
        instances = (
            BarTransformerEncoder(INPUT_DIM, cfg.transformer),
            SSMEncoder(INPUT_DIM, cfg.ssm),
            RecurrentEncoder(INPUT_DIM, cfg.recurrent),
            WaveletEncoder(INPUT_DIM, cfg.wavelet),
        )
        for enc in instances:
            assert isinstance(enc, SequenceEncoder)
            assert enc.output_dim == D_MODEL
            enc.eval()
            with torch.no_grad():
                assert enc(_x()).shape == (B, L, D_MODEL)


# --------------------------------------------------------------------------- #
# Gradient flow
# --------------------------------------------------------------------------- #

class TestGradients:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_gradients_reach_the_input(self, kind: str) -> None:
        enc = _build(kind)
        x = _x().requires_grad_(True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # A dead input projection would show up as an all-zero gradient.
        assert float(x.grad.abs().sum()) > 0.0

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_gradients_reach_parameters(self, kind: str) -> None:
        enc = _build(kind)
        out = enc(_x())
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(float(g.abs().sum()) > 0.0 for g in grads)


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #

def _outputs_before_and_after_perturbation(kind: str,
                                           k: int) -> tuple[torch.Tensor,
                                                            torch.Tensor]:
    """Run the same encoder on x and on x perturbed at positions >= k."""
    enc = _build(kind)
    x1 = _x(seed=3)
    x2 = x1.clone()
    g = torch.Generator().manual_seed(99)
    x2[:, k:] = x2[:, k:] + torch.randn(B, L - k, INPUT_DIM, generator=g)
    with torch.no_grad():
        return enc(x1), enc(x2)


class TestCausality:
    K = L // 2

    @pytest.mark.parametrize("kind", CAUSAL_KINDS)
    def test_causal_encoders_ignore_the_future(self, kind: str) -> None:
        y1, y2 = _outputs_before_and_after_perturbation(kind, self.K)
        # Positions BEFORE the perturbation must be untouched...
        assert torch.allclose(y1[:, :self.K], y2[:, :self.K], atol=1e-4), \
            f"{kind}: outputs before position {self.K} changed when only " \
            f"inputs at positions >= {self.K} were perturbed (future leak)"
        # ...and the perturbation must actually reach later positions,
        # otherwise the assertion above would be vacuous.
        assert not torch.allclose(y1[:, self.K:], y2[:, self.K:], atol=1e-4)

    def test_transformer_is_intentionally_bidirectional(self) -> None:
        # DESIGN DECISION, documented as a test: the bar transformer attends
        # in both directions inside the (already look-ahead-safe) window.
        # If this ever fails, someone made the transformer causal — update
        # the contract docs and this test together, deliberately.
        y1, y2 = _outputs_before_and_after_perturbation("transformer", self.K)
        assert not torch.allclose(y1[:, :self.K], y2[:, :self.K], atol=1e-4)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

class TestDeterminism:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_same_module_same_input_same_output(self, kind: str) -> None:
        enc = _build(kind)
        x = _x(seed=5)
        with torch.no_grad():
            a = enc(x)
            b = enc(x)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_same_seed_rebuild_gives_same_output(self, kind: str) -> None:
        x = _x(seed=6)
        with torch.no_grad():
            a = _build(kind, seed=777)(x)
            b = _build(kind, seed=777)(x)
        assert torch.allclose(a, b, atol=1e-7)
