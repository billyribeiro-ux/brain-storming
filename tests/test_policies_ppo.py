"""Layer 3 policy + PPO trainer tests.

Policy: contract shapes/finiteness, hierarchical meta gating (zero meta
logprob between decision bars), Beta-head support (0,1], deterministic mode.
PPO: two updates on a tiny in-test vectorized env honoring the env protocol,
then a checkpoint round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

polmod = pytest.importorskip("aether.decision.policies")
ppomod = pytest.importorskip("aether.decision.ppo")

from aether.decision.interfaces import (
    ACTION_KEYS,
    META_ACTIONS,
    TRADE_ACTIONS,
    EnvStep,
    PPOConfig,
)
from aether.decision.policies import HierarchicalPolicy
from aether.decision.ppo import PPOTrainer

from tests.helpers_stack import EMBED_DIM, make_policy_obs, tiny_policy_cfg

B = 6
META_EVERY = 5


@pytest.fixture()
def policy() -> HierarchicalPolicy:
    torch.manual_seed(0)
    return HierarchicalPolicy(tiny_policy_cfg(meta_every=META_EVERY))


def _extract_actions(ret):
    """Pull the action dict out of whatever ``mode`` returns."""
    if isinstance(ret, dict) and "trade" in ret:
        return ret
    if isinstance(ret, (tuple, list)):
        for item in ret:
            if isinstance(item, dict) and "trade" in item:
                return item
    pytest.fail(f"could not locate an action dict in {type(ret)!r}")


def _assert_discrete(t: torch.Tensor, n: int) -> None:
    vals = t.float()
    assert torch.all(vals == vals.round()), "discrete action must be integral"
    assert vals.min() >= 0 and vals.max() < n


def _assert_unit_interval(t: torch.Tensor) -> None:
    assert torch.isfinite(t).all()
    assert t.min() > 0.0, "Beta-head samples must be strictly positive"
    assert t.max() <= 1.0 + 1e-6, "Beta-head samples must stay within (0, 1]"


# --------------------------------------------------------------------------- #
# act / evaluate
# --------------------------------------------------------------------------- #

class TestActEvaluate:
    def test_act_shapes_and_ranges(self, policy):
        obs = make_policy_obs(B, flags=1.0)
        actions, logprobs, value, carry = policy.act(obs, policy.initial_carry(B))
        assert set(actions.keys()) == set(ACTION_KEYS)
        _assert_discrete(actions["meta"], len(META_ACTIONS))
        _assert_discrete(actions["trade"], len(TRADE_ACTIONS))
        for k in ("size", "stop", "target"):
            _assert_unit_interval(actions[k])
            assert actions[k].numel() == B
        assert value.numel() == B and torch.isfinite(value).all()
        assert isinstance(logprobs, dict) and "trade" in logprobs
        for k, lp in logprobs.items():
            assert torch.isfinite(lp).all(), f"logprob[{k}] not finite"

    def test_evaluate_reproduces_act_logprobs(self, policy):
        obs = make_policy_obs(B, flags=1.0)
        carry0 = policy.initial_carry(B)
        actions, lp_act, value, _ = policy.act(obs, carry0)
        lp_eval, entropy, value_eval, _ = policy.evaluate(obs, actions, carry0)
        assert "trade" in lp_eval
        for k in set(lp_act) & set(lp_eval):
            assert torch.allclose(lp_act[k].float(), lp_eval[k].float(),
                                  atol=1e-4), f"logprob[{k}] act/eval mismatch"
        assert torch.isfinite(entropy).all()
        assert torch.allclose(value.float(), value_eval.float(), atol=1e-5)

    def test_meta_logprob_zero_between_decision_bars(self, policy):
        obs = make_policy_obs(B, flags=0.0)
        carry0 = policy.initial_carry(B)
        actions, lp_act, _, _ = policy.act(obs, carry0)
        assert torch.allclose(lp_act["meta"].float(),
                              torch.zeros(B), atol=1e-8), \
            "meta logprob must be exactly 0 when flags == 0"
        lp_eval, _, _, _ = policy.evaluate(obs, actions, carry0)
        assert torch.allclose(lp_eval["meta"].float(),
                              torch.zeros(B), atol=1e-8)

    def test_beta_heads_sample_inside_unit_interval(self, policy):
        obs = make_policy_obs(512, flags=1.0, seed=3)
        actions, _, _, _ = policy.act(obs, policy.initial_carry(512))
        for k in ("size", "stop", "target"):
            _assert_unit_interval(actions[k])
            assert actions[k].float().unique().numel() > 1, \
                f"{k} head does not appear to be sampling"


# --------------------------------------------------------------------------- #
# mode (deterministic head)
# --------------------------------------------------------------------------- #

class TestMode:
    def test_mode_is_deterministic(self, policy):
        obs = make_policy_obs(B, flags=1.0)
        carry = policy.initial_carry(B)
        a1 = _extract_actions(policy.mode(obs, carry))
        a2 = _extract_actions(policy.mode(obs, carry))
        assert set(a1.keys()) == set(a2.keys())
        for k in a1:
            assert torch.equal(torch.as_tensor(a1[k]), torch.as_tensor(a2[k])), \
                f"mode action {k!r} is not deterministic"
        for k in ("size", "stop", "target"):
            _assert_unit_interval(torch.as_tensor(a1[k]))


# --------------------------------------------------------------------------- #
# tiny protocol-honoring vectorized env for the trainer
# --------------------------------------------------------------------------- #

class MiniEnv:
    """20-bar deterministic env: holding pays, everything else costs."""

    EP_LEN = 20

    def __init__(self, n_envs: int, obs_dim: int = EMBED_DIM,
                 meta_every: int = META_EVERY, seed: int = 0):
        self.n_envs = n_envs
        self.meta_every = meta_every
        rng = np.random.default_rng(seed)
        self._market = rng.standard_normal((self.EP_LEN, obs_dim)).astype(np.float32)
        self._bar = np.zeros(n_envs, dtype=np.int64)

    @property
    def episodes(self):
        return [("AAPL", "2026-01-05")]

    def _obs(self) -> dict[str, np.ndarray]:
        b = self._bar % self.EP_LEN
        n = self.n_envs
        return {
            "market": self._market[b],
            "unc": np.full((n, 3), 0.1, dtype=np.float32),
            "position": np.zeros((n, 5), dtype=np.float32),
            "portfolio": np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32),
                                 (n, 1)),
            "clock": np.stack([b / 390.0, (390.0 - b) / 390.0],
                              axis=1).astype(np.float32),
            "meta": np.tile(np.eye(len(META_ACTIONS), dtype=np.float32)[0],
                            (n, 1)),
            "flags": ((b % self.meta_every) == 0).astype(np.float32)[:, None],
        }

    def reset(self, episode_ids=None) -> dict[str, np.ndarray]:
        self._bar[:] = 0
        return self._obs()

    def step(self, actions: dict[str, np.ndarray]) -> EnvStep:
        trade = np.asarray(actions["trade"]).reshape(self.n_envs)
        reward = np.where(trade == 0, 0.05, -0.05).astype(np.float32)
        self._bar += 1
        done = (self._bar % self.EP_LEN) == 0
        self._bar[done] = 0                      # auto-reset finished envs
        return EnvStep(obs=self._obs(), reward=reward, done=done,
                       info=[{} for _ in range(self.n_envs)])


def _call_first(fn, variants):
    """Call ``fn`` with the first arg tuple its signature accepts."""
    last = None
    for args in variants:
        try:
            return fn(*args)
        except TypeError as exc:
            last = exc
    raise last


def _numeric_leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numeric_leaves(v)
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        yield float(obj)
    elif torch.is_tensor(obj) and obj.numel() == 1:
        yield float(obj)


# --------------------------------------------------------------------------- #
# PPO trainer
# --------------------------------------------------------------------------- #

class TestPPOTrainer:
    def _build(self, tmp_path):
        torch.manual_seed(0)
        np.random.seed(0)
        policy = HierarchicalPolicy(tiny_policy_cfg(meta_every=META_EVERY))
        env = MiniEnv(n_envs=4)
        cfg = PPOConfig(
            n_envs=4, rollout_len=MiniEnv.EP_LEN, epochs=2, minibatch_size=16,
            lr=1e-3, total_updates=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_jsonl=str(tmp_path / "runs" / "rl_train.jsonl"), seed=0,
        )
        return policy, PPOTrainer(policy, env, cfg), cfg

    def test_two_updates_finite_and_learning_moves_weights(self, tmp_path):
        policy, trainer, cfg = self._build(tmp_path)
        before = {k: v.detach().clone() for k, v in policy.state_dict().items()}

        result = trainer.train(2)

        after = policy.state_dict()
        assert any(not torch.allclose(before[k], after[k])
                   for k in before if before[k].is_floating_point()), \
            "two PPO updates left every parameter untouched"
        for k, v in after.items():
            if v.is_floating_point():
                assert torch.isfinite(v).all(), f"non-finite weights in {k}"

        leaves = list(_numeric_leaves(result)) if result is not None else []
        log_path = Path(cfg.log_jsonl)
        if log_path.is_file():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    leaves.extend(_numeric_leaves(json.loads(line)))
        for val in leaves:
            assert np.isfinite(val), "PPO reported a non-finite metric/loss"

    def test_checkpoint_roundtrip_restores_weights(self, tmp_path):
        policy, trainer, cfg = self._build(tmp_path)
        trainer.train(1)
        snap = {k: v.detach().clone() for k, v in policy.state_dict().items()}

        ckpt = tmp_path / "roundtrip.pt"
        _call_first(trainer.save, [(str(ckpt),), ()])
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(0.37)
        assert any(not torch.allclose(snap[k], v)
                   for k, v in policy.state_dict().items()
                   if v.is_floating_point()), "perturbation did not apply"

        _call_first(trainer.load, [(str(ckpt),), ()])
        for k, v in policy.state_dict().items():
            if v.is_floating_point():
                assert torch.allclose(v, snap[k], atol=1e-7), \
                    f"checkpoint load failed to restore {k}"
