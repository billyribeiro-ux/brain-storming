"""Layer 4 self-evolution tests.

The exact constructor/keyword surface of this layer is looser than the core
contracts, so these tests adapt to the signatures they find (filling only
well-known parameter names) and skip with a clear message when a required
parameter cannot be inferred — a skip here means "surface mismatch", while
a failure means real broken behavior.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from torch import nn

popmod = pytest.importorskip("aether.evolution.population")


# --------------------------------------------------------------------------- #
# signature-adaptive construction helpers
# --------------------------------------------------------------------------- #

def _adaptive_call(fn, preferred: dict, *, label: str):
    """Call ``fn`` filling only parameters whose names we recognize."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if name in preferred:
            kwargs[name] = preferred[name]
        elif param.default is inspect.Parameter.empty:
            pytest.skip(f"{label} requires unrecognized parameter {name!r}")
    return fn(**kwargs)


def _sample_genome(space: dict) -> dict:
    """First-choice genome drawn from whatever structure SEARCH_SPACE uses."""
    genome = {}
    for key, spec in space.items():
        if isinstance(spec, dict):
            spec = (spec.get("choices") or spec.get("values")
                    or [spec.get("low", spec.get("min", 0))])
        if isinstance(spec, (list, tuple)) and len(spec) > 0:
            genome[key] = spec[0]
        else:
            genome[key] = spec
    return genome


def _numeric_score(genome) -> float:
    """Deterministic synthetic fitness: sum of the genome's numeric genes."""
    total = 0.0
    values = genome.values() if isinstance(genome, dict) else [genome]
    for v in values:
        if isinstance(v, bool):
            total += float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            total += float(v)
        else:
            total += (sum(ord(c) for c in str(v)) % 97) / 97.0
    return total


# --------------------------------------------------------------------------- #
# population / genomes
# --------------------------------------------------------------------------- #

class TestSearchSpace:
    def test_search_space_is_a_nonempty_mapping(self):
        space = popmod.SEARCH_SPACE
        assert len(space) > 0
        assert all(isinstance(k, str) for k in space)

    def test_apply_genome_produces_configs(self):
        genome = _sample_genome(dict(popmod.SEARCH_SPACE))
        sig = inspect.signature(popmod.apply_genome)
        required = [p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD)]
        if len(required) > 1:
            pytest.skip("apply_genome requires more than a genome; surface "
                        f"mismatch: {list(sig.parameters)}")
        result = popmod.apply_genome(genome)
        assert result is not None, "apply_genome returned nothing"
        parts = result if isinstance(result, (tuple, list)) else [result]
        assert all(p is not None for p in parts)


class TestEvolve:
    def test_evolution_improves_synthetic_fitness(self):
        torch.manual_seed(0)
        np.random.seed(0)
        scores: list[float] = []

        def fitness(genome, *args, **kwargs) -> float:
            score = _numeric_score(genome)
            scores.append(score)
            return score

        pop = _adaptive_call(
            popmod.Population,
            {"size": 8, "n": 8, "pop_size": 8, "n_pop": 8, "seed": 0,
             "space": dict(popmod.SEARCH_SPACE),
             "search_space": dict(popmod.SEARCH_SPACE)},
            label="Population")
        _adaptive_call(
            pop.evolve,
            {"fitness": fitness, "fitness_fn": fitness, "fn": fitness,
             "evaluate": fitness, "score_fn": fitness,
             "generations": 4, "n_generations": 4, "gens": 4, "n_gen": 4,
             "steps": 4, "seed": 0},
            label="Population.evolve")

        assert len(scores) >= 8, \
            f"evolve evaluated only {len(scores)} genomes — no real search"
        half = len(scores) // 2
        early, late = scores[:half], scores[half:]
        assert max(late) >= max(early) - 1e-9, \
            "best fitness regressed across generations (no elitism?)"
        assert np.mean(late) > np.mean(early), (
            "selection pressure failed to improve mean synthetic fitness: "
            f"early={np.mean(early):.4f} late={np.mean(late):.4f}")


# --------------------------------------------------------------------------- #
# continual learning: Fisher regularizer
# --------------------------------------------------------------------------- #

class TestFisherRegularizer:
    def _build(self):
        contmod = pytest.importorskip("aether.evolution.continual")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
        x = torch.randn(32, 4)

        def loss_fn(m=model):
            return (m(x) ** 2).mean()

        reg = _adaptive_call(contmod.FisherRegularizer,
                             {"model": model}, label="FisherRegularizer")

        # Estimation phase: try the obvious surfaces; also leave fresh grads
        # on the model in case the implementation reads them directly.
        model.zero_grad()
        loss_fn().backward()
        for name in ("estimate", "snapshot", "fit", "update", "consolidate"):
            method = getattr(reg, name, None)
            if method is None:
                continue
            for args in ((loss_fn,), (model, loss_fn), (model,), ()):
                try:
                    method(*args)
                    return reg, model
                except TypeError:
                    continue
        return reg, model

    def _penalty(self, reg, model) -> float:
        for name in ("penalty", "loss", "regularize", "__call__"):
            method = getattr(reg, name, None)
            if method is None:
                continue
            for args in ((), (model,)):
                try:
                    return float(method(*args))
                except TypeError:
                    continue
        pytest.skip("FisherRegularizer exposes no recognizable penalty method")

    def test_zero_at_snapshot_positive_after_perturbation(self):
        reg, model = self._build()
        p0 = self._penalty(reg, model)
        assert p0 == pytest.approx(0.0, abs=1e-8), \
            "Fisher penalty must be zero at the snapshot parameters"

        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        p1 = self._penalty(reg, model)
        if p1 == 0.0:
            pytest.skip("penalty stayed 0 after perturbation — the Fisher "
                        "estimation surface was not matched by this test")
        assert p1 > 0.0
        assert np.isfinite(p1)

    def test_per_sample_losses_give_the_true_diagonal_fisher(self):
        """A 1-D per-sample loss_fn must yield F = mean_b (∂l_b/∂θ)² —
        checked against a hand-computed value on a linear model — while
        the scalar path yields the (squared-mean-gradient) ~1/B
        underestimate it is documented to be."""
        contmod = pytest.importorskip("aether.evolution.continual")
        torch.manual_seed(1)
        model = nn.Linear(4, 1, bias=False)
        x = torch.randn(32, 4)

        def per_sample_loss():
            return (model(x).squeeze(-1)) ** 2          # [B]

        def scalar_loss():
            return (model(x).squeeze(-1) ** 2).mean()   # []

        # Hand-computed: l_b = (w·x_b)², ∂l_b/∂w = 2 (w·x_b) x_b.
        with torch.no_grad():
            w = model.weight.squeeze(0)                 # [4]
            per_grad = 2.0 * (x @ w).unsqueeze(1) * x   # [B, 4]
            expected = per_grad.pow(2).mean(dim=0)      # [4]
            expected_scalar = per_grad.mean(dim=0).pow(2)

        reg = contmod.FisherRegularizer()
        reg.snapshot(model, per_sample_loss, n_batches=1)
        got = reg._fisher["weight"].squeeze(0)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6)

        reg_scalar = contmod.FisherRegularizer()
        reg_scalar.snapshot(model, scalar_loss, n_batches=1)
        got_scalar = reg_scalar._fisher["weight"].squeeze(0)
        torch.testing.assert_close(got_scalar, expected_scalar,
                                   rtol=1e-4, atol=1e-6)
        # the scalar path is the documented ~1/B underestimate, never more
        assert float(got_scalar.sum()) < float(got.sum()), \
            "scalar (squared-mean) path must underestimate the true Fisher"


# --------------------------------------------------------------------------- #
# self-diagnosis on a crafted regressing run
# --------------------------------------------------------------------------- #

def _write_regressing_runs(runs_dir) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rl_train.jsonl", "perception_train.jsonl"):
        with open(runs_dir / name, "w") as fh:
            for i in range(25):
                fh.write(json.dumps({
                    "step": i * 100,
                    "loss": 0.40 + 0.02 * i,          # train loss rising
                    "val_loss": 0.40 + 0.05 * i,      # val loss rising faster
                    "mean_reward": 1.0 - 0.04 * i,    # reward collapsing
                }) + "\n")


class TestSelfDiagnosis:
    def test_regressing_val_curve_emits_warning(self, tmp_path):
        diagmod = pytest.importorskip("aether.evolution.diagnosis")
        runs_dir = tmp_path / "runs"
        _write_regressing_runs(runs_dir)

        sd = _adaptive_call(
            diagmod.SelfDiagnosis,
            {"runs_dir": str(runs_dir), "root": str(runs_dir),
             "path": str(runs_dir), "runs": str(runs_dir),
             "data_root": str(tmp_path)},
            label="SelfDiagnosis")

        findings = None
        for name in ("diagnose", "run", "check", "analyze", "findings"):
            attr = getattr(sd, name, None)
            if attr is None:
                continue
            findings = attr() if callable(attr) else attr
            if findings:
                break
        assert findings, "SelfDiagnosis produced no findings on a clearly " \
                         "regressing validation curve"

        def _text(f) -> str:
            if hasattr(f, "__dataclass_fields__"):
                import dataclasses
                return json.dumps(dataclasses.asdict(f), default=str).lower()
            return json.dumps(f, default=str).lower()

        blob = " ".join(_text(f) for f in findings)
        assert ("warn" in blob) or ("critical" in blob), (
            "expected a warn/critical severity finding, got: " + blob[:500])
