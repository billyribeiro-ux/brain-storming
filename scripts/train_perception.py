#!/usr/bin/env python3
"""Perception self-supervised pre-training CLI.

Builds the data pipeline (from the Parquet lake), a
:class:`~aether.perception.model.PerceptionModel`, and a
:class:`~aether.perception.train.PerceptionTrainer`, then runs the full
self-supervised fit and prints the final metrics.

Usage::

    python3 scripts/train_perception.py \\
        --train-start 2020-01-01 --train-end 2024-12-31 \\
        --val-start 2025-01-01 --val-end 2025-06-30

    # quick smoke on a small model
    python3 scripts/train_perception.py \\
        --train-start 2024-01-01 --train-end 2024-10-31 \\
        --val-start 2024-11-01 --val-end 2024-12-31 \\
        --steps 500 --batch-size 32 --d-model 128 --device cpu

Look-ahead hygiene is enforced upstream: the train/val date split is passed
to the dataset builder, which fits all normalization statistics on the
TRAINING range only and slices windows so no sample sees past its anchor.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/train_perception.py` from any cwd: the repo root
# (parent of scripts/) must be importable for the `aether` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import inspect
from typing import Any, Iterable

import torch

from aether.config import AetherConfig
from aether.perception.interfaces import PerceptionConfig, TrainConfig
from aether.perception.model import PerceptionModel
from aether.perception.train import PerceptionTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised pre-training of the Aether perception model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="data lake root (default: $AETHER_DATA_ROOT or ./data)")
    parser.add_argument(
        "--train-start", required=True, metavar="YYYY-MM-DD",
        help="first session of the TRAINING range (inclusive); "
             "normalization statistics are fitted on this range only")
    parser.add_argument(
        "--train-end", required=True, metavar="YYYY-MM-DD",
        help="last session of the training range (inclusive)")
    parser.add_argument(
        "--val-start", required=True, metavar="YYYY-MM-DD",
        help="first session of the validation range (inclusive)")
    parser.add_argument(
        "--val-end", required=True, metavar="YYYY-MM-DD",
        help="last session of the validation range (inclusive)")
    parser.add_argument(
        "--steps", type=int, default=TrainConfig.max_steps,
        help="total optimizer steps")
    parser.add_argument(
        "--batch-size", type=int, default=TrainConfig.batch_size,
        help="samples per training batch")
    parser.add_argument(
        "--device", default=None, metavar="DEV",
        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--d-model", type=int, default=PerceptionConfig.d_model,
        help="model width used by every encoder and the fusion")
    return parser.parse_args(argv)


def build_loaders(
    cfg: AetherConfig,
    perception_cfg: PerceptionConfig,
    train_cfg: TrainConfig,
    args: argparse.Namespace,
) -> tuple[Iterable, Iterable]:
    """Construct (train_loader, val_loader) via ``aether.perception.datasets``.

    The dataset module is developed independently of this CLI, so instead of
    hard-wiring one exact signature we bind arguments **by parameter name**
    against the real ``build_dataloaders`` signature: every parameter it
    declares is looked up in a synonym table (config objects, window spec,
    date ranges, batch size, …). A required parameter with no known value is
    a hard error with an actionable message — never a silent mis-bind.
    """
    from aether.perception.datasets import build_dataloaders

    # Everything the builder could reasonably ask for, keyed by the
    # parameter names it might use. Unused entries are simply ignored.
    candidates: dict[str, Any] = {
        # config objects / paths
        "cfg": cfg, "config": cfg, "aether_cfg": cfg, "aether_config": cfg,
        "data_cfg": cfg.data, "data_config": cfg.data,
        "data_root": cfg.data.root, "root": cfg.data.root,
        "store": None,  # replaced below if the builder wants a ParquetStore
        "perception_cfg": perception_cfg, "perception_config": perception_cfg,
        "train_cfg": train_cfg, "train_config": train_cfg,
        "window": perception_cfg.window, "spec": perception_cfg.window,
        "window_spec": perception_cfg.window,
        # date ranges (both flat and tuple spellings)
        "train_start": args.train_start, "train_end": args.train_end,
        "val_start": args.val_start, "val_end": args.val_end,
        "train_range": (args.train_start, args.train_end),
        "val_range": (args.val_start, args.val_end),
        # loader knobs
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "seed": train_cfg.seed,
    }
    signature = inspect.signature(build_dataloaders)
    if "store" in signature.parameters:
        from aether.data.storage import ParquetStore
        candidates["store"] = ParquetStore(cfg.data.root)

    kwargs: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is param.empty:
            raise TypeError(
                f"build_dataloaders() requires parameter {name!r} which "
                f"scripts/train_perception.py does not know how to supply; "
                f"add it to the candidates table in build_loaders()."
            )

    result = build_dataloaders(**kwargs)
    # Accept (train, val) or a longer tuple whose first two are the loaders.
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return result[0], result[1]
    raise TypeError(
        "build_dataloaders() must return (train_loader, val_loader, ...); "
        f"got {type(result).__name__}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Configuration ----------------------------------------------------
    cfg = AetherConfig.from_env()  # loads .env implicitly
    if args.data_root:
        cfg.data.root = Path(args.data_root)

    perception_cfg = PerceptionConfig(d_model=args.d_model)
    # Warmup / validation cadence scale down with very short runs so smoke
    # trainings still warm up and validate; at the default 50k steps these
    # reduce to the TrainConfig defaults (1000 each).
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        warmup_steps=min(TrainConfig.warmup_steps, max(1, args.steps // 10)),
        val_every=min(TrainConfig.val_every, max(1, args.steps // 10)),
    )

    # ---- Data -------------------------------------------------------------
    train_loader, val_loader = build_loaders(cfg, perception_cfg, train_cfg, args)

    # ---- Model + trainer ----------------------------------------------------
    model = PerceptionModel(perception_cfg)
    print(f"PerceptionModel: {model.num_parameters:,} parameters "
          f"(d_model={perception_cfg.d_model})")

    trainer = PerceptionTrainer(
        model, train_loader, val_loader, train_cfg, device=args.device
    )
    print(f"training on {trainer.device} for {train_cfg.max_steps:,} steps "
          f"(train {args.train_start}..{args.train_end}, "
          f"val {args.val_start}..{args.val_end})")

    metrics = trainer.fit()

    # ---- Final report --------------------------------------------------------
    print("\nfinal metrics")
    print("-" * 40)
    for key in sorted(metrics):
        print(f"{key:<24} {metrics[key]:>14.6f}")
    print("-" * 40)
    print(f"parameters: {model.num_parameters:,}")
    print(f"checkpoints: {trainer.checkpoint_dir / 'last.pt'} | "
          f"{trainer.checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
