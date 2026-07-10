"""Layer 2 latent dynamics (RSSM-style) contract tests.

Pins: observe shapes, loss keys/finiteness, gradient flow, imagination
rollout shapes, seed determinism, and the free-nats KL floor.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

dynmod = pytest.importorskip("aether.worldmodel.dynamics")

from aether.worldmodel.dynamics import LatentDynamics
from aether.worldmodel.interfaces import ImaginedRollout, LatentState, ObserveResult

from tests.helpers_stack import tiny_dynamics_cfg

B, T = 2, 8


@pytest.fixture()
def cfg():
    return tiny_dynamics_cfg()


@pytest.fixture()
def dyn(cfg):
    torch.manual_seed(0)
    return LatentDynamics(cfg)


def _embeds(cfg, batch=B, steps=T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(batch, steps, cfg.embed_dim, generator=g)


# --------------------------------------------------------------------------- #
# observe
# --------------------------------------------------------------------------- #

class TestObserve:
    def test_shapes(self, dyn, cfg):
        res = dyn.observe(_embeds(cfg))
        assert isinstance(res, ObserveResult)
        assert tuple(res.states.deter.shape) == (B, T, cfg.deter_dim)
        assert tuple(res.states.stoch.shape) == (B, T, cfg.stoch_dim)
        for name in ("post_mu", "post_std", "prior_mu", "prior_std"):
            assert tuple(getattr(res, name).shape) == (B, T, cfg.stoch_dim), name
            assert torch.isfinite(getattr(res, name)).all(), name
        assert (res.post_std > 0).all()
        assert (res.prior_std > 0).all()

    def test_initial_state_is_zeros(self, dyn, cfg):
        st = dyn.initial(B)
        assert isinstance(st, LatentState)
        assert tuple(st.deter.shape) == (B, cfg.deter_dim)
        assert tuple(st.stoch.shape) == (B, cfg.stoch_dim)
        assert torch.count_nonzero(st.deter) == 0
        assert torch.count_nonzero(st.stoch) == 0
        assert tuple(st.feature.shape) == (B, cfg.deter_dim + cfg.stoch_dim)


# --------------------------------------------------------------------------- #
# loss + gradients
# --------------------------------------------------------------------------- #

class TestLoss:
    def test_keys_and_finiteness(self, dyn, cfg):
        losses = dyn.loss(_embeds(cfg))
        for key in ("total", "recon", "kl"):
            assert key in losses, f"loss dict missing {key!r}"
            val = losses[key]
            assert torch.is_tensor(val) and val.numel() == 1
            assert torch.isfinite(val).all(), f"{key} not finite"

    def test_backward_produces_finite_grads(self, dyn, cfg):
        dyn.zero_grad()
        losses = dyn.loss(_embeds(cfg))
        losses["total"].backward()
        grads = [p.grad for p in dyn.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient from loss['total']"
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_free_nats_collapse_kl_penalty(self, cfg):
        """A huge free-nats floor absorbs the whole KL: near-zero penalty."""
        embeds = _embeds(cfg)
        torch.manual_seed(0)
        dyn_small = LatentDynamics(dataclasses.replace(cfg, free_nats=0.0))
        torch.manual_seed(0)
        dyn_free = LatentDynamics(dataclasses.replace(cfg, free_nats=100.0))

        torch.manual_seed(7)
        kl_small = float(dyn_small.loss(embeds)["kl"])
        torch.manual_seed(7)
        kl_free = float(dyn_free.loss(embeds)["kl"])

        assert kl_free <= kl_small + 1e-6
        assert kl_free < 1e-3, "free_nats=100 must collapse the KL penalty to ~0"


# --------------------------------------------------------------------------- #
# imagine / decode
# --------------------------------------------------------------------------- #

class TestImagine:
    def test_rollout_shapes(self, dyn, cfg):
        n, horizon = 3, 4
        roll = dyn.imagine(dyn.initial(B), horizon, n)
        assert isinstance(roll, ImaginedRollout)
        assert tuple(roll.embeds.shape) == (n, B, horizon, cfg.embed_dim)
        assert tuple(roll.deter.shape) == (n, B, horizon, cfg.deter_dim)
        assert tuple(roll.stoch.shape) == (n, B, horizon, cfg.stoch_dim)
        for t_ in (roll.embeds, roll.deter, roll.stoch):
            assert torch.isfinite(t_).all()

    def test_imagine_from_observed_state(self, dyn, cfg):
        res = dyn.observe(_embeds(cfg))
        last = LatentState(res.states.deter[:, -1], res.states.stoch[:, -1])
        roll = dyn.imagine(last.detach(), 3, 2)
        assert tuple(roll.embeds.shape) == (2, B, 3, cfg.embed_dim)

    def test_decode_maps_features_to_embedding_space(self, dyn, cfg):
        st = dyn.initial(B)
        out = dyn.decode(st.feature)
        assert tuple(out.shape) == (B, cfg.embed_dim)
        # decode must broadcast over arbitrary leading dims (contract: [.., D+S])
        roll = dyn.imagine(st, 4, 3)
        feat = torch.cat([roll.deter, roll.stoch], dim=-1)
        out4 = dyn.decode(feat)
        assert tuple(out4.shape) == (3, B, 4, cfg.embed_dim)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

class TestDeterminism:
    def test_loss_deterministic_under_manual_seed(self, dyn, cfg):
        embeds = _embeds(cfg)
        torch.manual_seed(123)
        l1 = {k: float(v) for k, v in dyn.loss(embeds).items()}
        torch.manual_seed(123)
        l2 = {k: float(v) for k, v in dyn.loss(embeds).items()}
        assert l1 == l2

    def test_imagine_deterministic_under_manual_seed(self, dyn):
        st = dyn.initial(B)
        torch.manual_seed(11)
        r1 = dyn.imagine(st, 4, 3)
        torch.manual_seed(11)
        r2 = dyn.imagine(st, 4, 3)
        assert torch.equal(r1.embeds, r2.embeds)
        assert torch.equal(r1.stoch, r2.stoch)

    def test_imagine_is_stochastic_across_seeds(self, dyn):
        st = dyn.initial(B)
        torch.manual_seed(11)
        r1 = dyn.imagine(st, 4, 3)
        torch.manual_seed(12)
        r2 = dyn.imagine(st, 4, 3)
        assert not torch.equal(r1.stoch, r2.stoch), \
            "imagination must sample stochastic futures"
