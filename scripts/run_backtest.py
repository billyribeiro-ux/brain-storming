#!/usr/bin/env python3
"""Backtest CLI — replay the signal stack or a trained policy over history.

Usage::

    # signal-stack replay
    python3 scripts/run_backtest.py --name june --tickers AAPL NVDA \\
        --start 2025-06-01 --end 2025-06-30

    # RL-policy replay
    python3 scripts/run_backtest.py --mode policy \\
        --policy-ckpt checkpoints/decision/best.pt \\
        --start 2025-06-01 --end 2025-06-30

The SignalEngine is assembled from **whatever components exist**: the policy
is loaded from ``--policy-ckpt`` when given; dynamics, memory, and causal
artifacts are loaded when their modules import and their artifacts are on
disk, else each component is passed as ``None`` — the engine's consensus
gate is expected to degrade gracefully (fewer independent evidence sources,
stricter abstention). Every skipped component is logged, never silent.
Results are printed as a stats table and persisted under
``<data_root>/backtests/<name>/`` (``result.json`` + ``equity.parquet``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/run_backtest.py` from any cwd: the repo root
# (parent of scripts/) must be importable for the `aether` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

from aether.config import TICKER_IDS, AetherConfig
from aether.execution.backtest import (
    Backtester,
    PassThroughRisk,
    _bind_kwargs,
    save_result,
)
from aether.execution.interfaces import BacktestConfig, RiskConfig, SignalConfig
from aether.utils.logging import get_logger

logger = get_logger("aether.run_backtest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Aether signal stack or a trained policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default=None,
                        help="run name (default: <mode>-<timestamp>); "
                             "artifacts land in <data_root>/backtests/<name>/")
    parser.add_argument("--tickers", nargs="*", default=None, metavar="ALIAS",
                        help=f"ticker aliases (default: all — {', '.join(TICKER_IDS)})")
    parser.add_argument("--start", default="", metavar="YYYY-MM-DD",
                        help="first session (inclusive); empty = full range")
    parser.add_argument("--end", default="", metavar="YYYY-MM-DD",
                        help="last session (inclusive); empty = full range")
    parser.add_argument("--embeddings-dir", default="data/cache/embeddings",
                        help="per-ticker embedding npz directory")
    parser.add_argument("--data-root", default=None, metavar="PATH",
                        help="data lake root (default: $AETHER_DATA_ROOT or ./data)")
    parser.add_argument("--mode", choices=("signals", "policy"),
                        default="signals", help="which engine to run")
    parser.add_argument("--policy-ckpt", default=None, metavar="PATH",
                        help="decision-policy checkpoint (required for "
                             "--mode policy; optional evidence source for "
                             "--mode signals)")
    parser.add_argument("--perception-ckpt", default=None, metavar="PATH",
                        help="perception checkpoint path, forwarded to the "
                             "SignalEngine if it wants one")
    parser.add_argument("--no-autopsy", action="store_true",
                        help="skip autopsies of losing trades")
    parser.add_argument("--fees-bps", type=float, default=BacktestConfig.fees_bps)
    parser.add_argument("--slippage-bps", type=float,
                        default=BacktestConfig.slippage_bps)
    parser.add_argument("--initial-equity", type=float,
                        default=BacktestConfig.initial_equity)
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="episodes per vectorized env chunk (policy mode)")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Component loaders — each returns None (with a log line) when the module or
# its artifact does not exist yet. The Layer-5 stack must run with partial
# brains: missing components mean fewer evidence sources, never a crash.
# --------------------------------------------------------------------------- #

def load_policy(ckpt_path: str | None):
    """Policy from a checkpoint, or None. Tolerates the checkpoint layout
    (state under 'policy'/'model'/'state_dict' or the raw dict) and builds
    the config from any 'policy_cfg'/'cfg' snapshot it finds."""
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.is_file():
        logger.warning("policy checkpoint %s not found — policy=None", path)
        return None
    try:
        import dataclasses

        import torch

        from aether.decision.interfaces import PolicyConfig
        from aether.decision.policies import HierarchicalPolicy
    except Exception as exc:
        logger.warning("aether.decision.policies unavailable (%s) — policy=None", exc)
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_snap = ckpt.get("policy_cfg") or ckpt.get("cfg") or {}
        field_names = {f.name for f in dataclasses.fields(PolicyConfig)}
        pcfg = PolicyConfig(**{k: v for k, v in dict(cfg_snap).items()
                               if k in field_names})
        policy = HierarchicalPolicy(**_bind_kwargs(
            HierarchicalPolicy,
            {"cfg": pcfg, "config": pcfg, "policy_cfg": pcfg},
            "HierarchicalPolicy"))
        state = (ckpt.get("policy") or ckpt.get("model")
                 or ckpt.get("state_dict") or ckpt)
        policy.load_state_dict(state)
        if hasattr(policy, "eval"):
            policy.eval()
        logger.info("policy loaded from %s", path)
        return policy
    except Exception as exc:
        logger.warning("could not load policy from %s (%s) — policy=None",
                       path, exc)
        return None


def load_dynamics():
    """Latent dynamics from the first checkpoint found in the usual spots,
    or None (module missing, no artifact, or incompatible layout)."""
    candidates = [
        # scripts/train_worldmodel.py writes {model, cfg, step} here:
        Path("checkpoints/worldmodel/best.pt"),
        Path("checkpoints/worldmodel/last.pt"),
        *sorted(Path("checkpoints/worldmodel").glob("*dynamics*.pt")),
        Path("checkpoints/dynamics/best.pt"),
        Path("checkpoints/dynamics/last.pt"),
    ]
    ckpt_path = next((p for p in candidates if p.is_file()), None)
    if ckpt_path is None:
        logger.info("no dynamics checkpoint found — dynamics=None")
        return None
    try:
        import dataclasses

        import torch

        from aether.worldmodel import dynamics as dyn_mod
        from aether.worldmodel.interfaces import DynamicsConfig
    except Exception as exc:
        logger.warning("aether.worldmodel.dynamics unavailable (%s) — "
                       "dynamics=None", exc)
        return None
    try:
        cls = next((getattr(dyn_mod, n) for n in
                    ("LatentDynamicsModel", "LatentDynamics", "RSSM", "Dynamics")
                    if hasattr(dyn_mod, n)), None)
        if cls is None:
            logger.warning("no dynamics class found in worldmodel.dynamics — "
                           "dynamics=None")
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_snap = ckpt.get("cfg") or {}
        field_names = {f.name for f in dataclasses.fields(DynamicsConfig)}
        dcfg = DynamicsConfig(**{k: v for k, v in dict(cfg_snap).items()
                                 if k in field_names})
        model = cls(**_bind_kwargs(cls, {"cfg": dcfg, "config": dcfg},
                                   str(cls.__name__)))
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state)
        model.eval()
        logger.info("dynamics loaded from %s", ckpt_path)
        return model
    except Exception as exc:
        logger.warning("could not load dynamics from %s (%s) — dynamics=None",
                       ckpt_path, exc)
        return None


def load_memory(data_root: Path):
    """MemoryBank restored from its persist dir, or None."""
    persist_dir = Path(data_root) / "memory"
    if not persist_dir.exists():
        logger.info("no memory artifacts under %s — memory=None", persist_dir)
        return None
    try:
        from aether.worldmodel.interfaces import MemoryConfig
        from aether.worldmodel.memory import MemoryBank
        mcfg = MemoryConfig(persist_dir=str(persist_dir))
        bank = MemoryBank(**_bind_kwargs(MemoryBank,
                                         {"cfg": mcfg, "config": mcfg},
                                         "MemoryBank"))
        bank.load()
        logger.info("memory loaded: %d entries", len(bank))
        return bank
    except Exception as exc:
        logger.warning("could not load memory (%s) — memory=None", exc)
        return None


def load_causal(data_root: Path):
    """Newest causal graph snapshot json, or None."""
    causal_dir = Path(data_root) / "causal"
    snaps = sorted(causal_dir.glob("*.json")) if causal_dir.exists() else []
    if not snaps:
        logger.info("no causal snapshots under %s — causal=None", causal_dir)
        return None
    try:
        from aether.worldmodel import causal as causal_mod
        snapshot = causal_mod.from_json(snaps[-1].read_text())
        logger.info("causal graph loaded from %s", snaps[-1])
        return snapshot
    except Exception as exc:
        logger.warning("could not load causal snapshot (%s) — causal=None", exc)
        return None


def build_risk():
    """Real RiskManager when importable, else the loud PassThroughRisk."""
    try:
        from aether.execution.risk import RiskManager
        return RiskManager(RiskConfig())
    except Exception as exc:
        logger.warning("aether.execution.risk unavailable (%s) — "
                       "PassThroughRisk (NO risk rails)", exc)
        return PassThroughRisk()


def build_autopsist(data_root: Path):
    """An autopsy engine if the concurrent module provides one, else None."""
    try:
        from aether.worldmodel import autopsy as autopsy_mod
    except Exception as exc:
        logger.info("aether.worldmodel.autopsy unavailable (%s) — "
                    "autopsies skipped", exc)
        return None
    cls = next((getattr(autopsy_mod, n) for n in
                ("Autopsist", "TradeAutopsist", "AutopsyEngine", "Autopsy")
                if hasattr(autopsy_mod, n)), None)
    if cls is None:
        logger.info("no autopsist class in worldmodel.autopsy — autopsies skipped")
        return None
    try:
        return cls(**_bind_kwargs(cls, {"data_root": data_root,
                                        "root": data_root,
                                        "cfg": None, "config": None},
                                  str(cls.__name__)))
    except Exception as exc:
        logger.warning("could not build autopsist (%s) — autopsies skipped", exc)
        return None


def build_signal_engine(aether_cfg: AetherConfig, embeddings_dir: str,
                        policy=None, perception_ckpt: str | None = None):
    """Assemble ``aether.execution.signals.SignalEngine`` adaptively.

    Constructor arguments are bound by parameter name against a synonym
    table (house style — the module is developed concurrently), so the
    engine receives exactly the components it declares: any of policy /
    dynamics / memory / causal it asks for arrives loaded-or-None.
    """
    from aether.execution.signals import SignalEngine  # concurrent module

    data_root = Path(aether_cfg.data.root)
    dynamics = load_dynamics()
    memory = load_memory(data_root)
    causal = load_causal(data_root)
    sig_cfg = SignalConfig()
    candidates = {
        "cfg": sig_cfg, "config": sig_cfg, "signal_cfg": sig_cfg,
        "aether_cfg": aether_cfg, "aether_config": aether_cfg,
        "policy": policy,
        "dynamics": dynamics, "world_model": dynamics, "worldmodel": dynamics,
        "memory": memory, "memory_bank": memory,
        "causal": causal, "causal_graph": causal, "causal_snapshot": causal,
        "perception_ckpt": perception_ckpt, "perception": None,
        "perception_model": None,
        "embeddings_dir": str(embeddings_dir),
        "data_root": data_root, "root": data_root,
    }
    engine = SignalEngine(**_bind_kwargs(SignalEngine, candidates, "SignalEngine"))
    logger.info("SignalEngine built (policy=%s dynamics=%s memory=%s causal=%s)",
                *("yes" if c is not None else "None"
                  for c in (policy, dynamics, memory, causal)))
    return engine


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def print_stats(stats: dict) -> None:
    print("\nstats")
    print("-" * 44)
    for key in sorted(k for k in stats if k != "per_ticker"):
        print(f"{key:<20} {_fmt(stats[key]):>22}")
    per_ticker = stats.get("per_ticker") or {}
    if per_ticker:
        print("\nper ticker")
        header = ("ticker", "n", "hit", "avg_win", "avg_loss", "pf", "pnl")
        print(f"{header[0]:<8}{header[1]:>5}{header[2]:>8}{header[3]:>12}"
              f"{header[4]:>12}{header[5]:>10}{header[6]:>14}")
        for ticker in sorted(per_ticker):
            s = per_ticker[ticker]
            pf = s["profit_factor"]
            pf_txt = "inf" if pf == float("inf") else f"{pf:.2f}"
            print(f"{ticker:<8}{s['n_trades']:>5}{s['hit_rate']:>8.2f}"
                  f"{s['avg_win']:>12.2f}{s['avg_loss']:>12.2f}{pf_txt:>10}"
                  f"{s['total_pnl']:>14.2f}")
    print("-" * 44)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = AetherConfig.from_env()  # loads .env implicitly
    if args.data_root:
        cfg.data.root = Path(args.data_root)
    tickers = tuple(args.tickers) if args.tickers else tuple(TICKER_IDS)
    name = args.name or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"

    bt_cfg = BacktestConfig(
        tickers=tickers, start=args.start, end=args.end,
        fees_bps=args.fees_bps, slippage_bps=args.slippage_bps,
        initial_equity=args.initial_equity,
        autopsy_losses=not args.no_autopsy)

    policy = load_policy(args.policy_ckpt)
    if args.mode == "policy" and policy is None:
        print("error: --mode policy requires a loadable --policy-ckpt",
              file=sys.stderr)
        return 2

    signal_engine = None
    if args.mode == "signals":
        try:
            signal_engine = build_signal_engine(
                cfg, args.embeddings_dir, policy=policy,
                perception_ckpt=args.perception_ckpt)
        except ImportError as exc:
            print(f"error: aether.execution.signals is not available yet "
                  f"({exc}); --mode signals needs it", file=sys.stderr)
            return 2

    autopsist = None if args.no_autopsy else build_autopsist(Path(cfg.data.root))
    backtester = Backtester(
        bt_cfg, args.embeddings_dir, signal_engine=signal_engine,
        risk=build_risk(), autopsist=autopsist, data_root=cfg.data.root)

    if args.mode == "signals":
        result = backtester.run_signals()
    else:
        result = backtester.run_policy(policy, chunk_size=args.chunk_size)

    print_stats(result.stats)
    out = save_result(result, cfg.data.root, name)
    print(f"\nsignals: {len(result.signals)}  trades: {len(result.trades)}  "
          f"autopsies: {len(result.autopsies)}")
    print(f"artifacts: {out / 'result.json'} | {out / 'equity.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
