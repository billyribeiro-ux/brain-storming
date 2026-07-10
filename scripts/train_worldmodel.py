#!/usr/bin/env python3
"""World-model latent dynamics training CLI.

Trains the :class:`aether.worldmodel.dynamics.RSSM` on precomputed
perception embeddings: the per-ticker ``.npz`` files produced by
``aether.decision.env.EmbeddingStore.precompute`` (``fused [N, D]`` plus
``anchor_ts`` / ``session_minute`` bookkeeping). The dynamics model never
sees raw bars — Layer 2 lives strictly downstream of perception.

Usage::

    python3 scripts/train_worldmodel.py \\
        --start 2020-01-01 --end 2024-12-31

    # quick smoke on two tickers
    python3 scripts/train_worldmodel.py \\
        --tickers AAPL SPY --steps 200 --batch-size 8 --seq-len 64 \\
        --device cpu

Session hygiene
---------------
Training windows are sampled ONLY from within single sessions: the anchor
stream is split into contiguous session segments wherever ``session_minute``
fails to increase or the calendar date changes, and every ``[seq_len]``
window must fit inside one segment. A window spanning a session boundary
would ask the GRU to explain a 17.5-hour overnight gap (plus news, plus the
opening auction) as if it were one ordinary bar-to-bar transition —
systematically corrupting the learned dynamics.

Look-ahead note: sequences are historical replay consumed in time order
within each window; the model at step t is conditioned only on embeddings
≤ t inside the window. There is no target leakage — the objective is
next-step filtering/reconstruction of the same stream.

Checkpoints (``{model, cfg, step}``) go to ``<checkpoint-dir>/last.pt``
every checkpoint interval and to ``best.pt`` whenever the *running* (EMA)
reconstruction loss reaches a new low — the running mean, not one lucky
batch, decides "best".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scripts/train_worldmodel.py` from any cwd: the repo root
# (parent of scripts/) must be importable for the `aether` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import dataclasses
import os
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import torch

from aether.config import TICKERS
from aether.perception.train import (
    build_param_groups,
    make_lr_lambda,
    seed_everything,
)
from aether.utils.logging import get_logger
from aether.worldmodel.dynamics import RSSM
from aether.worldmodel.interfaces import DynamicsConfig

#: How often (in optimizer steps) to print a human-readable console line.
_CONSOLE_EVERY: int = 50
#: Decay of the running reconstruction mean that selects best.pt. 0.99 ≈ a
#: ~100-step effective memory — long enough to iron out batch noise, short
#: enough to reward genuine late-training improvement.
_RECON_EMA_DECAY: float = 0.99

logger = get_logger("aether.worldmodel.train", "runs/worldmodel_train.jsonl")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Aether world-model latent dynamics (RSSM) "
                    "on precomputed perception embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir", default="data/cache/embeddings", metavar="PATH",
        help="directory of per-ticker .npz embedding files "
             "(EmbeddingStore.precompute output)")
    parser.add_argument(
        "--tickers", nargs="*", default=None, metavar="ALIAS",
        help="ticker aliases to train on (default: the full universe; "
             "tickers with no embedding file are skipped with a warning)")
    parser.add_argument(
        "--start", default=None, metavar="YYYY-MM-DD",
        help="first session to include (inclusive; default: all history)")
    parser.add_argument(
        "--end", default=None, metavar="YYYY-MM-DD",
        help="last session to include (inclusive; default: all history)")
    parser.add_argument(
        "--steps", type=int, default=2000, help="total optimizer steps")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="sequences per training batch")
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="bars per training sequence (must fit inside one session)")
    parser.add_argument(
        "--device", default=None, metavar="DEV",
        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/worldmodel", metavar="PATH",
        help="where last.pt / best.pt are written")
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="peak AdamW learning rate")
    parser.add_argument(
        "--seed", type=int, default=1337, help="RNG seed (torch/numpy/python)")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Embedding loading
# --------------------------------------------------------------------------- #

def _load_npz_fallback(embeddings_dir: str, ticker: str) -> dict[str, np.ndarray]:
    """Local reader for EmbeddingStore's npz layout: ``<dir>/<ticker>.npz``.

    Used only when ``aether.decision.env`` (developed concurrently, Layer 3)
    is not importable yet. Reads the identical on-disk format, so swapping
    the real loader in changes nothing about training.
    """
    path = Path(embeddings_dir) / f"{ticker}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no embedding file for {ticker!r} at {path} — run the "
            f"embedding precompute first"
        )
    with np.load(path) as z:
        return {name: z[name] for name in z.files}


def resolve_loader() -> Callable[[str, str], dict[str, np.ndarray]]:
    """Prefer the canonical ``EmbeddingStore.load``; fall back to the local
    npz reader while Layer 3 is still being written (lazy import by design —
    this script must not hard-depend on a module that lands independently).
    """
    try:
        from aether.decision.env import EmbeddingStore  # noqa: PLC0415
        return EmbeddingStore.load
    except (ImportError, AttributeError):
        logger.info("aether.decision.env not available yet — using the "
                    "local npz loader (same file format)")
        return _load_npz_fallback


def _iso_to_epoch(day: str, end_of_day: bool = False) -> int:
    """ISO date → epoch-seconds under the lake's naive-ET convention.

    ``anchor_ts`` values are epoch-seconds computed from naive Eastern-Time
    wall clocks *as if they were UTC* (the lake's documented convention), so
    the correct inverse is a UTC-anchored conversion — machine-timezone
    independent by construction.
    """
    ts = int(datetime.strptime(day, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp())
    return ts + (86_399 if end_of_day else 0)


# --------------------------------------------------------------------------- #
# Session-safe window index
# --------------------------------------------------------------------------- #

def session_segments(anchor_ts: np.ndarray,
                     session_minute: np.ndarray) -> list[tuple[int, int]]:
    """Split an anchor stream into ``[start, end)`` single-session runs.

    A new session begins wherever

    * ``session_minute`` fails to strictly increase (the intraday clock
      only ever counts up within a session, so any reset/repeat marks a
      boundary), or
    * the calendar date of ``anchor_ts`` changes (guards the degenerate
      case of two consecutive sessions whose minute values happen to
      continue, e.g. after a data gap).

    Both tests are pure bookkeeping on data ≤ t — no look-ahead.
    """
    n = len(anchor_ts)
    if n == 0:
        return []
    day = anchor_ts.astype(np.int64) // 86_400          # naive-ET day ordinal
    minute = session_minute.astype(np.int64)
    boundary = np.ones(n, dtype=bool)                   # index 0 always starts
    if n > 1:
        boundary[1:] = (minute[1:] <= minute[:-1]) | (day[1:] != day[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], n)
    return list(zip(starts.tolist(), ends.tolist()))


class WindowIndex:
    """All session-safe ``[seq_len]`` windows across the loaded tickers.

    Holds one fused-embedding array per ticker plus two flat numpy vectors
    (``ticker index``, ``window start``) enumerating every valid window —
    compact enough for millions of windows, and sampling a batch is one
    RNG call plus B slice-copies.
    """

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.fused: list[np.ndarray] = []          # per ticker, [N, D] float32
        self.names: list[str] = []
        self._w_ticker: list[np.ndarray] = []
        self._w_start: list[np.ndarray] = []
        self.w_ticker: np.ndarray = np.zeros(0, dtype=np.int32)
        self.w_start: np.ndarray = np.zeros(0, dtype=np.int64)

    def add_ticker(self, name: str, fused: np.ndarray,
                   anchor_ts: np.ndarray, session_minute: np.ndarray) -> int:
        """Register one ticker's stream; returns the window count added."""
        idx = len(self.fused)
        self.fused.append(np.ascontiguousarray(fused, dtype=np.float32))
        self.names.append(name)
        added = 0
        for seg_start, seg_end in session_segments(anchor_ts, session_minute):
            n_windows = (seg_end - seg_start) - self.seq_len + 1
            if n_windows <= 0:
                continue                        # session shorter than seq_len
            starts = np.arange(seg_start, seg_start + n_windows, dtype=np.int64)
            self._w_start.append(starts)
            self._w_ticker.append(np.full(n_windows, idx, dtype=np.int32))
            added += n_windows
        return added

    def finalize(self) -> None:
        """Flatten the per-segment lists into the two sampling vectors."""
        if self._w_start:
            self.w_ticker = np.concatenate(self._w_ticker)
            self.w_start = np.concatenate(self._w_start)
        self._w_ticker, self._w_start = [], []

    def __len__(self) -> int:
        return len(self.w_start)

    def sample_batch(self, rng: np.random.Generator, batch_size: int,
                     device: torch.device) -> torch.Tensor:
        """Uniformly sample ``[B, seq_len, D]`` session-contiguous windows."""
        picks = rng.integers(0, len(self.w_start), size=batch_size)
        seqs = np.stack([
            self.fused[self.w_ticker[i]][self.w_start[i]:
                                         self.w_start[i] + self.seq_len]
            for i in picks
        ])
        return torch.from_numpy(seqs).to(device)


def build_window_index(args: argparse.Namespace,
                       loader: Callable[[str, str], dict[str, np.ndarray]],
                       ) -> WindowIndex:
    """Load embeddings for every requested ticker and index valid windows.

    Date filtering happens BEFORE segmentation, on ``anchor_ts``; since the
    stream is time-sorted per ticker, a date-range mask selects a contiguous
    run and the boundary detector re-derives sessions from what remains.
    """
    explicit = args.tickers is not None and len(args.tickers) > 0
    tickers = args.tickers if explicit else [t.alias for t in TICKERS]
    lo = _iso_to_epoch(args.start) if args.start else None
    hi = _iso_to_epoch(args.end, end_of_day=True) if args.end else None

    index = WindowIndex(args.seq_len)
    embed_dim: int | None = None
    for name in tickers:
        try:
            data = loader(args.embeddings_dir, name)
        except FileNotFoundError as exc:
            if explicit:
                raise  # the user asked for this ticker by name — hard error
            logger.warning("skipping %s: %s", name, exc)
            continue
        fused = np.asarray(data["fused"], dtype=np.float32)
        anchor_ts = np.asarray(data["anchor_ts"], dtype=np.int64)
        session_minute = np.asarray(data["session_minute"], dtype=np.int64)

        if embed_dim is None:
            embed_dim = int(fused.shape[1])
        elif int(fused.shape[1]) != embed_dim:
            raise ValueError(
                f"{name}: embedding dim {fused.shape[1]} differs from "
                f"{embed_dim} seen earlier — mixed precompute runs?"
            )

        mask = np.ones(len(anchor_ts), dtype=bool)
        if lo is not None:
            mask &= anchor_ts >= lo
        if hi is not None:
            mask &= anchor_ts <= hi
        added = index.add_ticker(
            name, fused[mask], anchor_ts[mask], session_minute[mask]
        )
        logger.info("%s: %d anchors in range -> %d windows",
                    name, int(mask.sum()), added)

    index.finalize()
    if len(index) == 0:
        raise SystemExit(
            "no training windows found — check --embeddings-dir/--start/"
            "--end, or lower --seq-len (every window must fit inside one "
            "session, i.e. seq_len <= bars per session)"
        )
    return index


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(path: Path, model: RSSM, cfg: DynamicsConfig,
                    step: int,
                    train_range: dict[str, str] | None = None) -> None:
    """Write ``{model, cfg, step, train_range}`` atomically (tmp + rename).

    ``train_range`` ({"start", "end"} ISO dates, empty = unbounded) is
    provenance metadata: downstream guards (scripts/run_backtest.py) use it
    to warn when a dynamics model is evaluated on the window it was
    trained on. Loaders must stay tolerant of its absence (older files).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "step": step,
        "train_range": train_range,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- Data: session-safe window index over precomputed embeddings ------
    # The canonical loader lives in Layer 3 (developed concurrently), hence
    # the lazy resolution inside main() rather than a top-level import.
    loader = resolve_loader()
    index = build_window_index(args, loader)
    embed_dim = int(index.fused[0].shape[1])
    logger.info("window index: %d windows across %d tickers (seq_len=%d, "
                "embed_dim=%d)", len(index), len(index.names), args.seq_len,
                embed_dim)

    # ---- Model / optimizer / schedule --------------------------------------
    cfg = DynamicsConfig(embed_dim=embed_dim)
    model = RSSM(cfg).to(device)
    model.train()
    print(f"RSSM: {model.num_parameters:,} parameters "
          f"(deter={cfg.deter_dim}, stoch={cfg.stoch_dim}, "
          f"embed={cfg.embed_dim})")

    # Decay hygiene + warmup→cosine schedule, reused from the perception
    # trainer — same optimizer policy across layers, one implementation.
    optimizer = torch.optim.AdamW(
        build_param_groups(model, weight_decay=0.01),
        lr=args.lr, betas=(0.9, 0.95),
    )
    warmup = max(1, min(100, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup, args.steps)
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_every = max(1, min(200, args.steps // 10))
    rng = np.random.default_rng(args.seed)

    # ---- Training loop ------------------------------------------------------
    running_recon: float | None = None   # EMA of recon — the "best" criterion
    best_recon = float("inf")
    last: dict[str, float] = {}
    logger.info("fit: device=%s steps=%d batch=%d seq_len=%d lr=%.2e",
                device, args.steps, args.batch_size, args.seq_len, args.lr)

    for step in range(1, args.steps + 1):
        batch = index.sample_batch(rng, args.batch_size, device)
        losses = model.loss(batch)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        last = {k: float(v.detach()) for k, v in losses.items()}
        recon = last["recon"]
        # Running mean of recon: seeded by the first batch, then EMA — the
        # smoothed value decides best.pt so one lucky batch cannot win.
        running_recon = recon if running_recon is None else (
            _RECON_EMA_DECAY * running_recon + (1.0 - _RECON_EMA_DECAY) * recon
        )

        if step % _CONSOLE_EVERY == 0 or step == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "step %6d/%d  total=%.4f  recon=%.4f  kl=%.4f  "
                "kl_raw=%.4f  recon_ema=%.4f  lr=%.2e",
                step, args.steps, last["total"], last["recon"], last["kl"],
                last.get("kl_raw", float("nan")), running_recon, lr_now,
                extra={
                    "aether_step": step,
                    "aether_total": last["total"],
                    "aether_recon": last["recon"],
                    "aether_kl": last["kl"],
                    "aether_recon_ema": running_recon,
                    "aether_lr": lr_now,
                },
            )

        # Checkpoint cadence (always fires on the final step): last.pt every
        # time, best.pt only on a new running-recon low.
        if step % checkpoint_every == 0 or step == args.steps:
            train_range = {"start": str(args.start or ""),
                           "end": str(args.end or "")}
            save_checkpoint(checkpoint_dir / "last.pt", model, cfg, step,
                            train_range=train_range)
            if running_recon < best_recon:
                best_recon = running_recon
                save_checkpoint(checkpoint_dir / "best.pt", model, cfg, step,
                                train_range=train_range)
                logger.info("step %d: new best recon_ema=%.4f -> best.pt",
                            step, best_recon,
                            extra={"aether_step": step,
                                   "aether_best_recon": best_recon})

    # ---- Final report --------------------------------------------------------
    print("\nfinal metrics")
    print("-" * 40)
    for key in sorted(last):
        print(f"{key:<24} {last[key]:>14.6f}")
    print(f"{'best_recon_ema':<24} {best_recon:>14.6f}")
    print("-" * 40)
    print(f"checkpoints: {checkpoint_dir / 'last.pt'} | "
          f"{checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
