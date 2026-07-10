"""Pins the PerceptionModel contract: output structure, evidential
uncertainty algebra, self-supervised losses, gradient reach, and ticker-DNA
conditioning.

Everything runs on ``synthetic_batch`` from the contract file, so this test
needs no data lake and stays CPU-fast at d_model=64.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from aether.config import TICKERS
from aether.perception.interfaces import (
    FusionConfig,
    PerceptionConfig,
    PerceptionOutput,
    RecurrentEncoderConfig,
    SSMEncoderConfig,
    TransformerEncoderConfig,
    WaveletEncoderConfig,
    WindowSpec,
    synthetic_batch,
)
from aether.perception.model import PerceptionModel

N_TICKERS = len(TICKERS)
D_MODEL = 64
DNA_DIM = 16
SPEC = WindowSpec(len_1m=30, len_5m=12, len_daily=3, horizon_1m=1)
LOSS_KEYS = ("masked", "contrastive", "dna", "evidential", "total")
EVID_KEYS = ("gamma", "nu", "alpha", "beta")


def small_model_cfg() -> PerceptionConfig:
    """Tiny full-stack config using the default (mixed) encoder families."""
    return PerceptionConfig(
        d_model=D_MODEL,
        dna_dim=DNA_DIM,
        window=SPEC,
        transformer=TransformerEncoderConfig(n_heads=4, n_layers=1, dropout=0.0),
        ssm=SSMEncoderConfig(d_state=16, n_layers=1, dropout=0.0),
        recurrent=RecurrentEncoderConfig(n_layers=1, dropout=0.0),
        wavelet=WaveletEncoderConfig(n_scales=2, kernel_size=3, dropout=0.0),
        fusion=FusionConfig(n_heads=4, n_latents=4, n_layers=1, dropout=0.0),
    )


@pytest.fixture(scope="module")
def model() -> PerceptionModel:
    torch.manual_seed(0)
    return PerceptionModel(small_model_cfg(), N_TICKERS)


@pytest.fixture()
def batch():
    return synthetic_batch(batch_size=4, spec=SPEC, n_tickers=N_TICKERS, seed=1)


# --------------------------------------------------------------------------- #
# Forward pass / PerceptionOutput
# --------------------------------------------------------------------------- #

class TestForward:
    def test_output_shapes(self, model, batch) -> None:
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert isinstance(out, PerceptionOutput)
        assert out.tokens_1m.shape == (4, SPEC.len_1m, D_MODEL)
        assert out.tokens_5m.shape == (4, SPEC.len_5m, D_MODEL)
        assert out.tokens_daily.shape == (4, SPEC.len_daily, D_MODEL)
        assert out.fused.shape == (4, D_MODEL)
        assert out.dna.shape == (4, DNA_DIM)
        assert out.fused.dtype == torch.float32
        assert torch.isfinite(out.fused).all()
        assert torch.isfinite(out.tokens_1m).all()

    def test_evidential_structure_and_positivity(self, model, batch) -> None:
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert set(out.evidential.keys()) == set(EVID_KEYS)
        for key in EVID_KEYS:
            assert out.evidential[key].shape == (4,)
            assert torch.isfinite(out.evidential[key]).all()
        # Normal-Inverse-Gamma validity constraints.
        assert (out.evidential["nu"] > 0).all()
        assert (out.evidential["alpha"] > 1).all()
        assert (out.evidential["beta"] > 0).all()

    def test_uncertainty_algebra(self, model, batch) -> None:
        # The contract defines the decompositions exactly:
        #   aleatoric = beta / (alpha - 1),  epistemic = beta / (nu (alpha-1))
        model.eval()
        with torch.no_grad():
            out = model(batch)
        nu, alpha, beta = (out.evidential["nu"], out.evidential["alpha"],
                           out.evidential["beta"])
        assert out.aleatoric.shape == (4,)
        assert out.epistemic.shape == (4,)
        assert torch.allclose(out.aleatoric, beta / (alpha - 1),
                              rtol=1e-4, atol=1e-6)
        assert torch.allclose(out.epistemic, beta / (nu * (alpha - 1)),
                              rtol=1e-4, atol=1e-6)
        # Both are variances -> strictly positive given the constraints.
        assert (out.aleatoric > 0).all()
        assert (out.epistemic > 0).all()

    def test_anomaly_is_nonnegative(self, model, batch) -> None:
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.anomaly.shape == (4,)
        assert torch.isfinite(out.anomaly).all()
        assert (out.anomaly >= 0).all()


# --------------------------------------------------------------------------- #
# Self-supervised losses
# --------------------------------------------------------------------------- #

class TestLosses:
    def test_losses_are_finite_scalars(self, model, batch) -> None:
        model.train()
        losses, out = model.self_supervised_losses(batch)
        assert isinstance(out, PerceptionOutput)
        for key in LOSS_KEYS:
            assert key in losses, f"missing loss component {key!r}"
            val = losses[key]
            assert torch.is_tensor(val) and val.ndim == 0, \
                f"{key} must be a scalar tensor"
            assert torch.isfinite(val), f"{key} is not finite"

    def test_total_backward_reaches_dna_embedding(self, model, batch) -> None:
        model.train()
        model.zero_grad(set_to_none=True)
        losses, _ = model.self_supervised_losses(batch)
        losses["total"].backward()
        # The ticker DNA table is the (N_TICKERS, dna_dim) parameter.
        dna_params = [p for _, p in model.named_parameters()
                      if tuple(p.shape) == (N_TICKERS, DNA_DIM)]
        assert dna_params, \
            "no (n_tickers, dna_dim) parameter found — where is the DNA table?"
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0
                   for p in dna_params), "total loss puts no gradient on DNA"
        model.zero_grad(set_to_none=True)

    def test_each_loss_component_touches_the_model(self, model, batch) -> None:
        # Every component must be wired to parameters — a constant or
        # detached component would silently train nothing.
        model.train()
        losses, _ = model.self_supervised_losses(batch)
        params = [p for p in model.parameters() if p.requires_grad]
        for key in ("masked", "contrastive", "dna", "evidential"):
            grads = torch.autograd.grad(losses[key], params,
                                        retain_graph=True, allow_unused=True)
            live = [g for g in grads if g is not None]
            assert live, f"loss {key!r} reaches no parameters"
            assert any(float(g.abs().sum()) > 0.0 for g in live), \
                f"loss {key!r} has all-zero gradients"

    def test_evidential_outputs_are_differentiable(self, model, batch) -> None:
        # The four raw evidential outputs must be produced by learnable
        # layers (not detached): each must carry gradient back to params.
        model.train()
        out = model(batch)
        params = [p for p in model.parameters() if p.requires_grad]
        for key in EVID_KEYS:
            t = out.evidential[key]
            assert t.requires_grad, f"evidential[{key!r}] is detached"
            grads = torch.autograd.grad(t.sum(), params,
                                        retain_graph=True, allow_unused=True)
            assert any(g is not None and float(g.abs().sum()) > 0.0
                       for g in grads), \
                f"evidential[{key!r}] has no gradient path to parameters"


# --------------------------------------------------------------------------- #
# Ticker DNA conditioning
# --------------------------------------------------------------------------- #

class TestDNAConditioning:
    def test_different_tickers_give_different_outputs(self, model, batch) -> None:
        # Identical market data, different instrument identity: the DNA
        # embedding must condition the fused state, otherwise Aether learns
        # one averaged market instead of per-ticker behavior.
        model.eval()
        b0 = dataclasses.replace(batch,
                                 ticker_id=torch.zeros_like(batch.ticker_id))
        b1 = dataclasses.replace(batch,
                                 ticker_id=torch.ones_like(batch.ticker_id))
        with torch.no_grad():
            out0, out1 = model(b0), model(b1)
        assert not torch.allclose(out0.dna, out1.dna, atol=1e-6), \
            "DNA embeddings identical across tickers"
        assert not torch.allclose(out0.fused, out1.fused, atol=1e-6), \
            "fused state ignores ticker identity — DNA conditioning is dead"


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #

class TestParameters:
    def test_num_parameters_property(self, model) -> None:
        n = model.num_parameters
        assert isinstance(n, int)                       # property, not method
        assert n == sum(p.numel() for p in model.parameters())
        assert n > 0
