#!/usr/bin/env python3
"""Layer-3 RL training CLI: hierarchical PPO over the trading environment.

Wires together the curriculum sampler (reversal-clarity staged episode
exposure), the look-ahead-safe TradingEnv over precomputed perception
embeddings, the hierarchical recurrent policy, and the PPO trainer, then
runs the full loop and prints final statistics.

Usage::

    python3 scripts/train_rl.py \\
        --embeddings-dir data/cache/embeddings \\
        --tickers AAPL,MSFT,SPY \\
        --start 2024-01-01 --end 2024-12-31

    # quick smoke
    python3 scripts/train_rl.py \\
        --embeddings-dir data/cache/embeddings --tickers AAPL \\
        --start 2024-01-01 --end 2024-03-31 \\
        --updates 5 --n-envs 4 --rollout-len 64 --device cpu

Look-ahead hygiene is enforced by the environment (bar t sees only data
≤ t; fills at t+1 under conservative assumptions). The curriculum only
reorders which historical episodes train first — it never leaks future
data into an observation (see aether.decision.curriculum).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/train_rl.py` from any cwd: the repo root (parent
# of scripts/) must be importable for the `aether` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import torch

from aether.decision.curriculum import CurriculumSampler
from aether.decision.interfaces import EnvConfig, PolicyConfig, PPOConfig
from aether.decision.policies import HierarchicalPolicy
from aether.decision.ppo import PPOTrainer, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical PPO training for the Aether decision core.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir", default=EnvConfig.embeddings_dir, metavar="PATH",
        help="directory of precomputed per-ticker embedding .npz files")
    parser.add_argument(
        "--tickers", required=True, metavar="A,B,C",
        help="comma-separated ticker aliases to trade")
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD",
        help="first session of the episode sampling range (inclusive)")
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD",
        help="last session of the episode sampling range (inclusive)")
    parser.add_argument(
        "--updates", type=int, default=PPOConfig.total_updates,
        help="total PPO updates (collect+optimize cycles)")
    parser.add_argument(
        "--n-envs", type=int, default=PPOConfig.n_envs,
        help="parallel environment copies")
    parser.add_argument(
        "--rollout-len", type=int, default=PPOConfig.rollout_len,
        help="steps per env per rollout (390 = one full session)")
    parser.add_argument(
        "--device", default=None, metavar="DEV",
        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--seed", type=int, default=PPOConfig.seed, help="global RNG seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tickers = tuple(t.strip() for t in args.tickers.split(",") if t.strip())
    if not tickers:
        raise SystemExit("--tickers produced an empty list")

    # ---- Environment --------------------------------------------------------
    try:
        from aether.decision.env import TradingEnv
    except ImportError as exc:
        raise SystemExit(
            "aether.decision.env.TradingEnv is not available yet "
            f"({exc}). The environment module is required to train; "
            "policies/ppo/replay/curriculum are importable without it."
        ) from exc

    env_cfg = EnvConfig(
        embeddings_dir=args.embeddings_dir,
        tickers=tickers,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
    env = TradingEnv(env_cfg, n_envs=args.n_envs)

    # ---- Curriculum ----------------------------------------------------------
    sampler = CurriculumSampler(args.embeddings_dir, tickers)
    print(f"curriculum: {len(sampler)} episodes across {len(tickers)} "
          f"tickers, {sampler.stages} stages")

    # ---- Policy (market width inferred from real observations) --------------
    # A throwaway reset tells us the embedding width the env actually emits,
    # so the policy always matches the perception checkpoint that produced
    # the store — no silently divergent config constant.
    seed_everything(args.seed)  # deterministic policy init AND env sampling
    probe_obs = env.reset(episode_ids=None)
    market_dim = int(np.asarray(probe_obs["market"]).shape[-1])
    policy_cfg = PolicyConfig(
        obs_market_dim=market_dim, meta_every=env_cfg.meta_every)
    policy = HierarchicalPolicy(policy_cfg)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"HierarchicalPolicy: {n_params:,} parameters "
          f"(market_dim={market_dim}, memory_dim={policy_cfg.memory_dim})")

    # ---- Trainer --------------------------------------------------------------
    ppo_cfg = PPOConfig(
        n_envs=args.n_envs,
        rollout_len=args.rollout_len,
        total_updates=args.updates,
        seed=args.seed,
    )
    trainer = PPOTrainer(
        policy, env, ppo_cfg, episode_sampler=sampler, device=args.device)
    print(f"training on {trainer.device}: {args.updates} updates × "
          f"{args.n_envs} envs × {args.rollout_len} steps")

    summary = trainer.train()

    # ---- Final report ----------------------------------------------------------
    print("\nfinal stats")
    print("-" * 40)
    for key in sorted(summary):
        print(f"{key:<24} {summary[key]:>14.6f}")
    print("-" * 40)
    print(f"checkpoints: {trainer.checkpoint_dir / 'last.pt'} | "
          f"{trainer.checkpoint_dir / 'best.pt'}")
    print(f"log: {ppo_cfg.log_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
