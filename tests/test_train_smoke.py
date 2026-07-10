"""End-to-end training smoke test: a 6-step fit on synthetic batches must
produce finite losses, JSONL logs, checkpoints, and a restorable step
counter — all in temp dirs, all on CPU, in seconds.

The loaders are plain lists of PerceptionBatch objects: anything the
trainer needs from a loader (len(), repeated iteration) a list provides,
without coupling the test to torch DataLoader specifics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aether.config import TICKERS
from aether.perception.interfaces import (
    FusionConfig,
    PerceptionConfig,
    RecurrentEncoderConfig,
    SSMEncoderConfig,
    TrainConfig,
    TransformerEncoderConfig,
    WaveletEncoderConfig,
    WindowSpec,
    synthetic_batch,
)
from aether.perception.model import PerceptionModel
from aether.perception.train import PerceptionTrainer

N_TICKERS = len(TICKERS)
SPEC = WindowSpec(len_1m=30, len_5m=12, len_daily=3, horizon_1m=1)
MAX_STEPS = 6


def _model_cfg() -> PerceptionConfig:
    """Smallest sensible full-stack model (d_model=32, single layers)."""
    return PerceptionConfig(
        d_model=32,
        dna_dim=8,
        window=SPEC,
        transformer=TransformerEncoderConfig(n_heads=4, n_layers=1, dropout=0.0),
        ssm=SSMEncoderConfig(d_state=8, n_layers=1, dropout=0.0),
        recurrent=RecurrentEncoderConfig(n_layers=1, dropout=0.0),
        wavelet=WaveletEncoderConfig(n_scales=2, kernel_size=3, dropout=0.0),
        fusion=FusionConfig(n_heads=4, n_latents=4, n_layers=1, dropout=0.0),
    )


def _train_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        max_steps=MAX_STEPS,
        warmup_steps=2,
        grad_clip=1.0,
        ema_decay=0.99,
        amp=False,                 # CPU test: no autocast surprises
        val_every=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_jsonl=str(tmp_path / "runs" / "train.jsonl"),
        num_workers=0,
        seed=7,
    )


def _loaders() -> tuple[list, list]:
    train = [synthetic_batch(4, SPEC, n_tickers=N_TICKERS, seed=i)
             for i in range(3)]
    val = [synthetic_batch(4, SPEC, n_tickers=N_TICKERS, seed=100 + i)
           for i in range(2)]
    return train, val


def _trainer_step(trainer, loaded=None) -> int:
    """Best-effort read of the trainer's global step counter.

    Accepts either the return value of load_checkpoint (int or dict with a
    'step' entry) or one of the conventional attribute names.
    """
    if isinstance(loaded, int):
        return loaded
    if isinstance(loaded, dict) and "step" in loaded:
        return int(loaded["step"])
    for name in ("step", "global_step", "_step", "steps_done"):
        value = getattr(trainer, name, None)
        if isinstance(value, int):
            return value
        if torch.is_tensor(value):
            return int(value)
    pytest.fail("cannot determine trainer step: load_checkpoint returned "
                f"{type(loaded).__name__} and none of step/global_step/_step/"
                "steps_done exist as int attributes")


def _load_checkpoint(trainer, path: Path):
    """Call load_checkpoint tolerating a path-arg or no-arg signature."""
    try:
        return trainer.load_checkpoint(str(path))
    except TypeError:
        return trainer.load_checkpoint()


def test_six_step_fit_checkpoints_and_restore(tmp_path: Path) -> None:
    cfg = _train_cfg(tmp_path)
    train_loader, val_loader = _loaders()

    torch.manual_seed(cfg.seed)
    model = PerceptionModel(_model_cfg(), N_TICKERS)
    trainer = PerceptionTrainer(model, train_loader, val_loader, cfg)

    # ---- validation works before any training --------------------------- #
    metrics0 = trainer.validate()
    assert isinstance(metrics0, dict)
    assert "total" in metrics0
    assert float(metrics0["total"]) == float(metrics0["total"])  # not NaN

    # ---- the 6-step fit -------------------------------------------------- #
    trainer.fit()
    assert _trainer_step(trainer) == MAX_STEPS

    # ---- JSONL training log ---------------------------------------------- #
    log_path = Path(cfg.log_jsonl)
    assert log_path.is_file(), "fit() must write the JSONL log"
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1, "JSONL log is empty"
    for ln in lines:
        record = json.loads(ln)              # every line is valid JSON
        assert isinstance(record, dict)

    # ---- checkpoints exist ------------------------------------------------ #
    ckpt_files = [p for p in Path(cfg.checkpoint_dir).rglob("*") if p.is_file()]
    assert ckpt_files, "fit() must leave at least one checkpoint on disk"

    # ---- restoring into a fresh trainer brings the step back -------------- #
    torch.manual_seed(123)                   # deliberately different init
    model2 = PerceptionModel(_model_cfg(), N_TICKERS)
    before = {n: p.detach().clone() for n, p in model2.named_parameters()}
    trainer2 = PerceptionTrainer(model2, train_loader, val_loader, cfg)

    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    loaded = _load_checkpoint(trainer2, latest)
    assert _trainer_step(trainer2, loaded) == MAX_STEPS, \
        "load_checkpoint must restore the global step"

    # The checkpoint actually moved weights (fresh init was overwritten).
    changed = any(not torch.equal(before[n], p.detach())
                  for n, p in model2.named_parameters())
    assert changed, "load_checkpoint left every parameter at fresh-init values"

    # ---- restored trainer can still validate ------------------------------ #
    metrics = trainer2.validate()
    assert isinstance(metrics, dict) and "total" in metrics
    total = float(metrics["total"])
    assert total == total and abs(total) != float("inf")
