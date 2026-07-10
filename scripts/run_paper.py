#!/usr/bin/env python3
"""Paper trading CLI — simulated orders only, never a broker.

Usage::

    python3 scripts/run_paper.py --once --sync        # one iteration
    python3 scripts/run_paper.py --loop --sync        # poll forever (Ctrl-C)
    python3 scripts/run_paper.py --once \\
        --embeddings-dir data/cache/embeddings --tactics-mode maker

Each iteration optionally syncs the lake (1min/5min bars), loads the latest
per-ticker embeddings, warns loudly when they are stale relative to the
lake, marks/manages open positions, generates + risk-checks signals on the
newest bar, and appends everything as JSONL events to the blotter. The
SignalEngine is assembled exactly as in scripts/run_backtest.py — with
whatever components exist (missing ones are None, logged, never silent).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/run_paper.py` from any cwd: the repo root (parent
# of scripts/) must be importable for the `aether` package to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from aether.config import AetherConfig
from aether.execution.interfaces import PaperConfig
from aether.execution.paper import PaperTrader
from aether.execution.tactics import MODES, ExecutionTactics
from aether.utils.logging import get_logger
from scripts.run_backtest import build_risk, build_signal_engine, load_policy

logger = get_logger("aether.run_paper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aether paper trading (simulated orders; no broker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true",
                       help="run a single iteration (default)")
    group.add_argument("--loop", action="store_true",
                       help="poll forever every --poll-seconds")
    parser.add_argument("--sync", action="store_true",
                        help="incrementally sync bars_1min/bars_5min from FMP "
                             "before each iteration")
    parser.add_argument("--embeddings-dir", default="data/cache/embeddings",
                        help="per-ticker embedding npz directory")
    parser.add_argument("--data-root", default=None, metavar="PATH",
                        help="data lake root (default: $AETHER_DATA_ROOT or ./data)")
    parser.add_argument("--blotter-path", default=None, metavar="PATH",
                        help=f"blotter JSONL (default: {PaperConfig.blotter_path})")
    parser.add_argument("--state-path", default=None, metavar="PATH",
                        help=f"state json (default: {PaperConfig.state_path})")
    parser.add_argument("--poll-seconds", type=int,
                        default=PaperConfig.poll_seconds,
                        help="loop cadence")
    parser.add_argument("--tactics-mode", choices=MODES, default="taker",
                        help="execution tactic for entries")
    parser.add_argument("--policy-ckpt", default=None, metavar="PATH",
                        help="optional decision-policy checkpoint fed to the "
                             "SignalEngine as an evidence source")
    parser.add_argument("--perception-ckpt", default=None, metavar="PATH",
                        help="perception checkpoint path, forwarded to the "
                             "SignalEngine if it wants one")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = AetherConfig.from_env()  # loads .env implicitly
    if args.data_root:
        cfg.data.root = Path(args.data_root)

    # Default blotter/state under the (possibly overridden) data root.
    paper_cfg = PaperConfig(
        blotter_path=args.blotter_path or str(Path(cfg.data.root) / "paper"
                                              / "blotter.jsonl"),
        state_path=args.state_path or str(Path(cfg.data.root) / "paper"
                                          / "state.json"),
        poll_seconds=args.poll_seconds)

    try:
        signal_engine = build_signal_engine(
            cfg, args.embeddings_dir,
            policy=load_policy(args.policy_ckpt),
            perception_ckpt=args.perception_ckpt)
    except ImportError as exc:
        print(f"error: aether.execution.signals is not available yet ({exc}); "
              f"paper trading needs a SignalEngine", file=sys.stderr)
        return 2

    trader = PaperTrader(
        paper_cfg, cfg, signal_engine, build_risk(),
        ExecutionTactics(mode=args.tactics_mode),
        embeddings_dir=args.embeddings_dir)

    if args.loop:
        trader.run_loop(poll_seconds=args.poll_seconds, sync=args.sync)
        return 0
    summary = trader.run_once(sync=args.sync)
    print(json.dumps(summary, indent=2, default=str))
    print(f"blotter: {trader.blotter_path}\nstate:   {trader.state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
