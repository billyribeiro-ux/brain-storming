"""Recurrent PPO trainer for the Aether decision core (Layer 3).

This module owns the on-policy optimization loop for
:class:`aether.decision.policies.HierarchicalPolicy` against a vectorized
trading environment (``TradingEnvProtocol``; reward shaping lives *inside*
the env, this trainer only sees scalars).

Key design points
-----------------
* **Recurrent rollouts** — :meth:`collect` steps all ``n_envs`` for
  ``cfg.rollout_len`` bars, storing observations, actions, per-head
  log-probs, values, rewards, dones, meta flags *and the GRU carry at every
  step* (all detached). The carry is zeroed whenever an episode ends, so a
  fresh episode never inherits memory from the previous one — the exact rule
  the update replay reproduces.
* **GAE(γ, λ)** with a bootstrap value on non-done rollout tails; advantages
  normalized over the whole batch (mean 0 / std 1) for scale-free clipping.
* **Sequence-intact minibatches** — minibatches partition *env indices*, and
  each selected env's trajectory is replayed step-by-step, in time order,
  from the STORED initial carry of the rollout (see :meth:`_replay` for why
  time order is mandatory for a recurrent policy).
* **Clipped surrogate + clipped value loss + entropy bonus**, ratios built
  from the *sum* of active heads' log-probs (the meta head contributes zero
  off decision bars by construction — see policies.py), gradient-norm clip,
  and an early stop of remaining epochs when the approximate KL between the
  behaviour and current policy exceeds ``cfg.target_kl``.
* **JSONL logging** via :func:`aether.utils.logging.get_logger` — mean
  reward, episode PnL statistics mined from ``info['trade_closed']``
  records, losses, KL, LR.
* **Checkpointing** — ``last.pt`` every update and ``best.pt`` whenever the
  running mean episode reward improves; checkpoints carry the policy,
  optimizer, config snapshot, reward window and all RNG states so
  :meth:`load` resumes a run *exactly* (same future action samples, same
  minibatch permutations).
* **Device-agnostic** (project rule): the device is chosen once in
  ``__init__`` ("cuda" if available unless overridden); env observations are
  numpy and converted to torch on that device at the boundary.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from aether.decision.interfaces import ACTION_KEYS, OBS_KEYS, PPOConfig
from aether.utils.logging import get_logger

#: Discrete action keys (emitted to the env as int64; the rest as float32).
_DISCRETE_KEYS = ("meta", "trade")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs (all devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA generators


class PPOTrainer:
    """Runs collect → update cycles of recurrent PPO.

    Parameters
    ----------
    policy:
        A :class:`~aether.decision.policies.HierarchicalPolicy` (anything
        satisfying ``PolicyProtocol`` with an ``initial_carry(batch, device)``
        and a ``cfg`` attribute). Moved to ``device`` here.
    env:
        A vectorized ``TradingEnvProtocol`` environment. Reward shaping is
        the env's job; every closed round-trip must appear in step infos
        under ``'trade_closed'``. The trainer assumes finished envs
        auto-reset inside ``step`` (the obs returned for a done env is the
        first obs of its next episode) — GAE masks the boundary with
        ``1 − done`` and the GRU carry is zeroed there.
    cfg:
        :class:`~aether.decision.interfaces.PPOConfig`.
    episode_sampler:
        Optional curriculum object with ``advance(update_idx)`` and
        ``eligible() -> list`` (see decision.curriculum). Each collect asks
        it which episode ids the env may sample from.
    device:
        Explicit device string; ``None`` auto-selects CUDA when available.
    """

    def __init__(
        self,
        policy: nn.Module,
        env: Any,
        cfg: PPOConfig,
        episode_sampler: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = policy.to(self.device)
        self.env = env
        self.episode_sampler = episode_sampler

        # The env is the source of truth for parallelism; warn on mismatch
        # rather than silently shaping buffers wrong.
        self.n_envs = int(getattr(env, "n_envs", cfg.n_envs))

        # Adam with the PPO-conventional eps (default 1e-8 can destabilize
        # updates when advantage-scaled gradients get small).
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        # ---- Bookkeeping ----------------------------------------------------
        self.update_idx: int = 0
        self.best_reward: float = -math.inf
        #: Recent completed-episode returns; its mean is the "running mean
        #: episode reward" that gates best.pt.
        self._ep_returns: deque[float] = deque(maxlen=100)
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.logger = get_logger("aether.rl", cfg.log_jsonl)
        if self.n_envs != cfg.n_envs:
            self.logger.warning(
                "env exposes n_envs=%d but cfg.n_envs=%d — using the env's",
                self.n_envs, cfg.n_envs)

    # ------------------------------------------------------------------ #
    # Torch/numpy boundary helpers
    # ------------------------------------------------------------------ #

    def _obs_to_torch(self, obs: dict[str, Any]) -> dict[str, Tensor]:
        """Numpy observation dict -> float32 tensors on the trainer device."""
        return {
            key: torch.as_tensor(
                np.asarray(obs[key]), dtype=torch.float32, device=self.device)
            for key in OBS_KEYS
        }

    @staticmethod
    def _actions_to_numpy(actions: dict[str, Tensor]) -> dict[str, np.ndarray]:
        """Torch action dict -> numpy dict with the env-facing dtypes."""
        out: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            arr = actions[key].detach().cpu().numpy()
            out[key] = arr.astype(
                np.int64 if key in _DISCRETE_KEYS else np.float32)
        return out

    @staticmethod
    def _trade_pnls(step_infos: Sequence[dict]) -> list[float]:
        """Extract closed-trade PnLs from one step's info list.

        ``info['trade_closed']`` may hold a single TradeRecord, a list of
        them, or plain dicts — all are accepted.
        """
        pnls: list[float] = []
        for info in step_infos:
            if not isinstance(info, dict) or "trade_closed" not in info:
                continue
            records = info["trade_closed"]
            if not isinstance(records, (list, tuple)):
                records = [records]
            for rec in records:
                pnl = rec.get("pnl") if isinstance(rec, dict) \
                    else getattr(rec, "pnl", None)
                if pnl is not None:
                    pnls.append(float(pnl))
        return pnls

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def collect(self, episode_ids: Optional[list] = None) -> dict[str, Any]:
        """One on-policy rollout of ``cfg.rollout_len`` steps × ``n_envs``.

        Resets the env (optionally restricted to curriculum-eligible
        ``episode_ids``) and the policy carry, then steps and stores
        everything an update needs — including the detached carry at *every*
        step so the recurrent replay can start from ground truth rather
        than an approximation.

        Returns the rollout dict with GAE ``advantages`` (batch-normalized)
        and ``returns`` already attached, plus a ``stats`` sub-dict (mean
        reward, completed episode returns, trade PnL stats from infos).
        """
        cfg, N, T = self.cfg, self.n_envs, self.cfg.rollout_len
        obs = self._obs_to_torch(self.env.reset(episode_ids=episode_ids))
        carry = self.policy.initial_carry(N, self.device)

        obs_seq: list[dict[str, Tensor]] = []
        carry_seq: list[Tensor] = []
        act_seq: list[dict[str, Tensor]] = []
        lp_seq: list[dict[str, Tensor]] = []
        val_seq: list[Tensor] = []
        rew_seq: list[Tensor] = []
        done_seq: list[Tensor] = []
        infos: list[Sequence[dict]] = []

        ep_return = torch.zeros(N, device=self.device)
        episode_returns: list[float] = []
        trade_pnls: list[float] = []

        for _ in range(T):
            carry_seq.append(carry.clone())      # carry *entering* this step
            obs_seq.append(obs)
            actions, logprobs, value, carry = self.policy.act(obs, carry)
            step = self.env.step(self._actions_to_numpy(actions))

            reward = torch.as_tensor(
                np.asarray(step.reward), dtype=torch.float32, device=self.device)
            done = torch.as_tensor(
                np.asarray(step.done), dtype=torch.float32, device=self.device)

            act_seq.append(actions)
            lp_seq.append(logprobs)
            val_seq.append(value)
            rew_seq.append(reward)
            done_seq.append(done)
            infos.append(step.info)
            trade_pnls.extend(self._trade_pnls(step.info))

            # Episode-return accounting (per env, reset at boundaries).
            ep_return = ep_return + reward
            for i in torch.nonzero(done > 0.5).flatten().tolist():
                episode_returns.append(float(ep_return[i]))
                ep_return[i] = 0.0

            # A finished env auto-resets: its next obs starts a NEW episode,
            # so the recurrent memory must not leak across the boundary.
            carry = carry * (1.0 - done).unsqueeze(-1)
            obs = self._obs_to_torch(step.obs)

        # Bootstrap value of the final (non-stored) obs for GAE tails. act()
        # also samples, which merely advances the seeded global RNG — still
        # fully deterministic run-to-run.
        _, _, last_value, _ = self.policy.act(obs, carry)

        rollout: dict[str, Any] = {
            "obs": {k: torch.stack([o[k] for o in obs_seq]) for k in OBS_KEYS},
            "actions": {k: torch.stack([a[k] for a in act_seq])
                        for k in ACTION_KEYS},
            "logprobs": {k: torch.stack([l[k] for l in lp_seq])
                         for k in ACTION_KEYS},
            "values": torch.stack(val_seq),          # [T, N]
            "rewards": torch.stack(rew_seq),         # [T, N]
            "dones": torch.stack(done_seq),          # [T, N] float 0/1
            "carries": torch.stack(carry_seq),       # [T, N, memory_dim]
            "infos": infos,
        }
        # Meta-decision flags, stored explicitly for masking diagnostics
        # (identical to rollout["obs"]["flags"][..., 0]).
        rollout["flags"] = rollout["obs"]["flags"][..., 0]

        # ---- GAE(γ, λ) ------------------------------------------------------
        # δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
        # A_t = δ_t + γλ·(1−done_t)·A_{t+1}
        # done_t masks both terms: past an episode end neither the bootstrap
        # value nor the advantage recursion may cross the boundary (the
        # auto-reset obs at t+1 belongs to a different episode).
        values, rewards, dones = (
            rollout["values"], rollout["rewards"], rollout["dones"])
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = (rewards[t] + cfg.gamma * next_value * non_terminal
                     - values[t])
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        rollout["returns"] = advantages + values
        # Batch normalization: clipping thresholds become scale-free.
        rollout["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8))

        n_trades = len(trade_pnls)
        rollout["stats"] = {
            "mean_reward": float(rewards.mean()),
            "episode_returns": episode_returns,
            "n_trades": n_trades,
            "trade_pnl_sum": float(np.sum(trade_pnls)) if n_trades else 0.0,
            "trade_pnl_mean": float(np.mean(trade_pnls)) if n_trades else 0.0,
            "win_rate": (float(np.mean([p > 0 for p in trade_pnls]))
                         if n_trades else 0.0),
        }
        return rollout

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def _replay(
        self, rollout: dict[str, Any], env_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-evaluate the selected envs' trajectories with gradients.

        Why strict time order is mandatory: the GRU carry at step ``t`` is a
        deterministic function of *every* observation the policy saw at
        steps ``0..t−1``. Evaluating steps out of order (or from arbitrary
        carries) would score actions against hidden states the behaviour
        policy never actually had — the importance ratios would compare two
        different conditional distributions and the PPO objective would be
        silently wrong. So minibatches partition ENV indices, each sequence
        stays intact, and we re-walk it step-by-step from the STORED initial
        carry of the rollout, re-applying the same carry-zeroing at episode
        boundaries that collect() applied.

        Returns ``(logprob_sum [T, n], entropy [T, n], values [T, n])``
        where logprob_sum is summed over the active heads (the meta head is
        exactly zero off decision bars in both old and new logprobs, so the
        sum stays consistent with the behaviour policy's).
        """
        T = rollout["rewards"].shape[0]
        # The stored t=0 carry is detached ground truth (zeros after reset).
        carry = rollout["carries"][0, env_idx].clone()
        lp_steps, ent_steps, val_steps = [], [], []
        for t in range(T):
            obs_t = {k: v[t, env_idx] for k, v in rollout["obs"].items()}
            act_t = {k: v[t, env_idx] for k, v in rollout["actions"].items()}
            logprobs, entropy, value, carry = self.policy.evaluate(
                obs_t, act_t, carry)
            lp_steps.append(sum(logprobs[k] for k in ACTION_KEYS))
            ent_steps.append(entropy)
            val_steps.append(value)
            # Mirror collect(): no memory across episode boundaries.
            carry = carry * (1.0 - rollout["dones"][t, env_idx]).unsqueeze(-1)
        return (torch.stack(lp_steps), torch.stack(ent_steps),
                torch.stack(val_steps))

    def update(self, rollout: dict[str, Any]) -> dict[str, float]:
        """cfg.epochs PPO passes over the rollout; returns mean loss stats.

        Early-stops the remaining epochs when the epoch-mean approximate KL
        (the low-variance estimator ``E[(r−1) − log r]``) exceeds
        ``cfg.target_kl`` — past that point the importance ratios no longer
        trust-region-hold and further reuse of the batch harms the policy.
        """
        cfg = self.cfg
        T = rollout["rewards"].shape[0]
        old_lp_sum = sum(rollout["logprobs"][k] for k in ACTION_KEYS)  # [T,N]

        # Minibatches partition env indices; granularity is whole sequences
        # because the recurrent replay cannot split a sequence (see _replay).
        # cfg.minibatch_size counts samples, so envs per minibatch ≈ size/T.
        envs_per_mb = max(1, cfg.minibatch_size // max(1, T))

        totals: dict[str, float] = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0}
        n_minibatches = 0
        epochs_run = 0

        for _ in range(cfg.epochs):
            epoch_kls: list[float] = []
            perm = torch.randperm(self.n_envs, device=self.device)
            for start in range(0, self.n_envs, envs_per_mb):
                idx = perm[start:start + envs_per_mb]
                new_lp_sum, entropy, values = self._replay(rollout, idx)

                adv = rollout["advantages"][:, idx]
                ratio = torch.exp(new_lp_sum - old_lp_sum[:, idx])

                # Clipped surrogate (pessimistic min of the two).
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss: keeps the critic from chasing return
                # targets that moved because the policy moved.
                v_old = rollout["values"][:, idx]
                returns = rollout["returns"][:, idx]
                v_clipped = v_old + torch.clamp(
                    values - v_old, -cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * torch.max(
                    (values - returns) ** 2,
                    (v_clipped - returns) ** 2).mean()

                entropy_mean = entropy.mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_mean)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - ratio.log()).mean())
                    clip_frac = float(
                        ((ratio - 1.0).abs() > cfg.clip).float().mean())
                epoch_kls.append(approx_kl)

                totals["policy_loss"] += float(policy_loss)
                totals["value_loss"] += float(value_loss)
                totals["entropy"] += float(entropy_mean)
                totals["approx_kl"] += approx_kl
                totals["clip_frac"] += clip_frac
                n_minibatches += 1

            epochs_run += 1
            if epoch_kls and float(np.mean(epoch_kls)) > cfg.target_kl:
                break  # trust region exhausted — skip remaining epochs

        stats = {k: v / max(1, n_minibatches) for k, v in totals.items()}
        stats["epochs_run"] = float(epochs_run)
        return stats

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self, total_updates: int | None = None) -> dict[str, float]:
        """Run collect+update until ``total_updates`` (default from cfg).

        Logs one JSONL record per update, checkpoints ``last.pt`` every
        update, and refreshes ``best.pt`` whenever the running mean episode
        reward (over the last 100 completed episodes) improves. Resumable:
        after :meth:`load`, training continues from the stored update index
        with restored RNG streams.
        """
        cfg = self.cfg
        total = int(total_updates) if total_updates is not None \
            else cfg.total_updates
        if self.update_idx == 0:
            # Fresh run only: a resumed run's RNG streams were restored by
            # load() and must not be re-seeded (that would replay the past).
            seed_everything(cfg.seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "train: device=%s n_envs=%d rollout=%d updates=%d..%d",
            self.device, self.n_envs, cfg.rollout_len, self.update_idx, total)

        last_record: dict[str, float] = {}
        while self.update_idx < total:
            u = self.update_idx

            episode_ids: Optional[list] = None
            if self.episode_sampler is not None:
                if hasattr(self.episode_sampler, "advance"):
                    self.episode_sampler.advance(u)
                episode_ids = list(self.episode_sampler.eligible())

            rollout = self.collect(episode_ids)
            losses = self.update(rollout)
            self.update_idx = u + 1

            roll = rollout["stats"]
            self._ep_returns.extend(roll["episode_returns"])
            running = (float(np.mean(self._ep_returns))
                       if self._ep_returns else float("-inf"))
            lr = self.optimizer.param_groups[0]["lr"]

            last_record = {
                "update": float(self.update_idx),
                "mean_reward": roll["mean_reward"],
                "ep_reward_running": running,
                "n_trades": float(roll["n_trades"]),
                "trade_pnl_mean": roll["trade_pnl_mean"],
                "trade_pnl_sum": roll["trade_pnl_sum"],
                "win_rate": roll["win_rate"],
                **{k: float(v) for k, v in losses.items()},
                "lr": lr,
            }
            self.logger.info(
                "update %4d/%d  reward=%.5f  ep_run=%.5f  trades=%d  "
                "pi=%.4f  vf=%.4f  ent=%.4f  kl=%.4f  lr=%.2e",
                self.update_idx, total, roll["mean_reward"],
                running if self._ep_returns else float("nan"),
                roll["n_trades"], losses["policy_loss"], losses["value_loss"],
                losses["entropy"], losses["approx_kl"], lr,
                extra={f"aether_{k}": v for k, v in last_record.items()},
            )

            # Update best BEFORE writing last.pt so a resume from last.pt
            # carries the true best-so-far (stale best would let a resumed
            # run overwrite best.pt with a worse policy).
            improved = bool(self._ep_returns) and running > self.best_reward
            if improved:
                self.best_reward = running
            self.save(self.checkpoint_dir / "last.pt")
            if improved:
                self.save(self.checkpoint_dir / "best.pt")

        summary = dict(last_record)
        summary["best_reward"] = self.best_reward
        return summary

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Write a complete, resumable snapshot (atomic temp-file + rename).

        Includes every RNG stream (python/numpy/torch/cuda) so a resumed run
        draws the *same* future action samples and minibatch permutations —
        exact resume, not an approximation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy_cfg = getattr(self.policy, "cfg", None)
        payload: dict[str, Any] = {
            "update": self.update_idx,
            "best_reward": self.best_reward,
            "reward_window": list(self._ep_returns),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": dataclasses.asdict(self.cfg),
            "policy_cfg": (dataclasses.asdict(policy_cfg)
                           if dataclasses.is_dataclass(policy_cfg) else None),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else []),
            },
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    def load(self, path: str | Path) -> dict[str, Any]:
        """Restore a snapshot written by :meth:`save` for exact resume."""
        path = Path(path)
        # weights_only=False: the payload holds plain dicts and RNG state
        # tuples; the file is our own artifact, not untrusted input.
        ckpt: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_idx = int(ckpt["update"])
        self.best_reward = float(ckpt["best_reward"])
        self._ep_returns = deque(ckpt.get("reward_window", []), maxlen=100)

        rng = ckpt.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        saved_cfg = ckpt.get("cfg", {})
        live_cfg = dataclasses.asdict(self.cfg)
        for key in sorted(set(saved_cfg) | set(live_cfg)):
            if saved_cfg.get(key) != live_cfg.get(key):
                self.logger.warning(
                    "resume: cfg.%s differs (checkpoint=%r, live=%r)",
                    key, saved_cfg.get(key), live_cfg.get(key))
        self.logger.info("resumed from %s at update %d", path, self.update_idx)
        return ckpt
