"""Configuration-genome evolution — Aether's NAS-lite (Layer 4).

Aether does not rewrite its own code. What it *does* evolve is its
**configuration genome**: a flat dict of dotted keys into the three config
dataclasses that shape the whole stack —

* :class:`aether.perception.interfaces.PerceptionConfig` (architecture:
  which encoder family per stream, model width, SSL objective weights),
* :class:`aether.decision.interfaces.RewardConfig` (what the agent is
  rewarded for — drawdown aversion, churn penalty, reversal capture),
* :class:`aether.decision.interfaces.PPOConfig` (how the agent is trained).

:data:`SEARCH_SPACE` bounds every gene; :func:`apply_genome` turns a genome
into three *fresh* config objects, so a genome is a complete, reproducible
recipe for a candidate system.

Honesty about scope
-------------------
This module is the SEARCH LOOP ONLY. It never trains a network, never runs a
backtest, never touches data. The fitness function is supplied by the caller
and is expected to be expensive and honest — e.g. "train a small perception
model for N steps and return −(validation SSL loss)" or "train a policy and
return the out-of-sample simulated Sharpe". Fitness is MAXIMIZED. Evolution
is exactly as good as the fitness signal it is given: a leaky fitness (e.g.
in-sample Sharpe) will be exploited, not exposed, by this loop.

The algorithm is a deliberately small genetic algorithm (population ~8,
tournament-2 selection, uniform crossover, per-gene mutation, elitism) —
appropriate for the tiny, cheap-to-enumerate space above, not a claim of
state-of-the-art NAS.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable

from aether.decision.interfaces import PPOConfig, RewardConfig
from aether.perception.interfaces import PerceptionConfig
from aether.utils.logging import get_logger

#: A genome: dotted config key -> value drawn from SEARCH_SPACE's bound.
Genome = dict[str, Any]

# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

#: Every evolvable gene and its bound.
#:
#: * ``list``  → categorical choice (resampled on mutation),
#: * ``tuple`` ``(lo, hi)`` → continuous range (uniform init, clipped
#:   gaussian jitter on mutation). Continuous genes whose bounds are both
#:   positive and span ≥ :data:`_LOG_SCALE_RATIO`× are treated as
#:   log-uniform (learning rates live on a log scale, not a linear one).
#:
#: Keys are dotted paths rooted at ``perception.`` / ``reward.`` / ``ppo.``.
#: NOT evolved on purpose: ``n_heads`` (must divide every ``d_model``
#: choice; fixed 8 divides 64/128/256), window geometry (changing it
#: invalidates cached datasets/embeddings), and anything in ``TrainConfig``
#: that is budget rather than behavior (max_steps, batch_size).
SEARCH_SPACE: dict[str, list | tuple] = {
    # -- perception: architecture ----------------------------------------- #
    "perception.d_model": [64, 128, 256],
    "perception.dna_dim": [16, 32, 64],
    "perception.encoder_1m": ["transformer", "ssm", "wavelet+transformer"],
    "perception.encoder_5m": ["transformer", "ssm", "lstm"],
    "perception.encoder_daily": ["lstm", "ssm", "transformer"],
    "perception.transformer.n_layers": [2, 3, 4, 6],
    "perception.transformer.dropout": (0.0, 0.3),
    "perception.ssm.d_state": [32, 64, 128],
    "perception.fusion.n_latents": [8, 16, 32],
    # -- perception: SSL objective ----------------------------------------- #
    "perception.heads.mask_ratio": (0.1, 0.4),
    "perception.heads.temperature": (0.03, 0.2),
    "perception.heads.w_contrastive": (0.25, 2.0),
    # -- reward shaping ------------------------------------------------------ #
    "reward.w_dd": (0.1, 1.5),
    "reward.w_churn": (0.0, 0.2),
    "reward.w_capture": (0.0, 0.8),
    # -- PPO ----------------------------------------------------------------- #
    "ppo.lr": (1e-5, 1e-3),               # log-scale (spans 100×)
    "ppo.entropy_coef": (0.001, 0.05),    # log-scale (spans 50×)
    "ppo.clip": (0.1, 0.3),
    "ppo.gamma": [0.99, 0.995, 0.999],
    "ppo.gae_lambda": (0.9, 0.99),
}

#: Continuous genes with positive bounds spanning at least this ratio are
#: sampled and jittered in log10 space.
_LOG_SCALE_RATIO: float = 20.0

#: Gaussian mutation std as a fraction of the gene's (possibly log) span.
_JITTER_FRAC: float = 0.15


def _is_log_scale(lo: float, hi: float) -> bool:
    return lo > 0 and hi / lo >= _LOG_SCALE_RATIO


def _sample_gene(key: str, rng: random.Random,
                 space: dict[str, list | tuple]) -> Any:
    """Draw one gene uniformly from its bound (log-uniform where flagged)."""
    bound = space[key]
    if isinstance(bound, list):
        return rng.choice(bound)
    lo, hi = float(bound[0]), float(bound[1])
    if _is_log_scale(lo, hi):
        return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))
    return rng.uniform(lo, hi)


def _perturb_gene(key: str, value: Any, rng: random.Random,
                  space: dict[str, list | tuple]) -> Any:
    """Mutate one gene within its bound.

    Categorical: resample among the *other* choices (a mutation that cannot
    change anything is not a mutation). Continuous: gaussian jitter with std
    ``_JITTER_FRAC × span``, in log10 space for log-scale genes, clipped.
    """
    bound = space[key]
    if isinstance(bound, list):
        others = [c for c in bound if c != value]
        return rng.choice(others) if others else value
    lo, hi = float(bound[0]), float(bound[1])
    if _is_log_scale(lo, hi):
        llo, lhi = math.log10(lo), math.log10(hi)
        jittered = math.log10(float(value)) + rng.gauss(0.0, _JITTER_FRAC * (lhi - llo))
        return 10.0 ** min(lhi, max(llo, jittered))
    jittered = float(value) + rng.gauss(0.0, _JITTER_FRAC * (hi - lo))
    return min(hi, max(lo, jittered))


# --------------------------------------------------------------------------- #
# Genome operators
# --------------------------------------------------------------------------- #

def random_genome(rng: random.Random,
                  space: dict[str, list | tuple] | None = None) -> Genome:
    """A full genome sampled uniformly (log-uniform where flagged) from
    every gene's bound."""
    space = space if space is not None else SEARCH_SPACE
    return {key: _sample_gene(key, rng, space) for key in space}


def mutate(genome: Genome, rng: random.Random, rate: float = 0.3,
           space: dict[str, list | tuple] | None = None) -> Genome:
    """Return a mutated copy: each gene mutates independently with
    probability ``rate``.

    If the coin flips leave every gene untouched, one random gene is forced
    to mutate — offspring identical to a parent would silently shrink the
    effective population.
    """
    space = space if space is not None else SEARCH_SPACE
    child = dict(genome)
    mutated = False
    for key in child:
        if key in space and rng.random() < rate:
            child[key] = _perturb_gene(key, child[key], rng, space)
            mutated = True
    if not mutated and child:
        key = rng.choice(sorted(k for k in child if k in space))
        child[key] = _perturb_gene(key, child[key], rng, space)
    return child


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene comes from parent ``a`` or ``b`` with
    equal probability. Genes present in only one parent are inherited
    from that parent (tolerates genomes from differently-versioned
    search spaces)."""
    child: Genome = {}
    for key in sorted(set(a) | set(b)):
        if key in a and key in b:
            child[key] = a[key] if rng.random() < 0.5 else b[key]
        else:
            child[key] = a.get(key, b.get(key))
    return child


# --------------------------------------------------------------------------- #
# Genome -> configs
# --------------------------------------------------------------------------- #

def _validated(key: str, value: Any) -> Any:
    """Check ``value`` against the gene's bound; normalize continuous genes.

    Categorical values must be one of the listed choices (exact match).
    Continuous values are cast to float and clipped into ``[lo, hi]`` —
    tolerant of tiny numeric drift from JSON round-trips.
    """
    bound = SEARCH_SPACE.get(key)
    if bound is None:
        # Not an evolvable gene — allow any dotted override, unchecked.
        # (apply_genome is also the config-override utility for callers.)
        return value
    if isinstance(bound, list):
        if value not in bound:
            raise ValueError(f"genome[{key!r}] = {value!r} not in choices {bound}")
        return value
    lo, hi = float(bound[0]), float(bound[1])
    return min(hi, max(lo, float(value)))


def apply_genome(genome: Genome) -> tuple[PerceptionConfig, RewardConfig, PPOConfig]:
    """Build fresh ``(PerceptionConfig, RewardConfig, PPOConfig)`` with the
    genome's dotted overrides applied.

    Partial genomes are fine: unspecified genes keep their config defaults.
    Unknown attribute paths raise ``KeyError`` (a genome must never silently
    no-op). After all overrides, ``PerceptionConfig.__post_init__`` is
    re-run so an evolved ``d_model`` propagates into every encoder/fusion
    sub-config exactly as it does at normal construction time.
    """
    roots: dict[str, Any] = {
        "perception": PerceptionConfig(),
        "reward": RewardConfig(),
        "ppo": PPOConfig(),
    }
    for key, raw in genome.items():
        root_name, _, rest = key.partition(".")
        if root_name not in roots or not rest:
            raise KeyError(f"genome key {key!r} must start with "
                           f"'perception.'/'reward.'/'ppo.'")
        value = _validated(key, raw)
        obj = roots[root_name]
        *mids, leaf = rest.split(".")
        for attr in mids:
            if not hasattr(obj, attr):
                raise KeyError(f"genome key {key!r}: no attribute {attr!r} "
                               f"on {type(obj).__name__}")
            obj = getattr(obj, attr)
        if not hasattr(obj, leaf):
            raise KeyError(f"genome key {key!r}: no attribute {leaf!r} "
                           f"on {type(obj).__name__}")
        setattr(obj, leaf, value)
    # Re-propagate the (possibly evolved) shared width into sub-configs.
    roots["perception"].__post_init__()
    return roots["perception"], roots["reward"], roots["ppo"]


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #

def _genome_key(genome: Genome) -> tuple:
    """Hashable identity of a genome (for the fitness cache)."""
    return tuple(sorted(genome.items()))


class Population:
    """A small genetic algorithm over configuration genomes.

    Selection: tournament of 2 (pick two uniformly, keep the fitter) — mild
    selection pressure that preserves diversity in a population this small.
    Variation: uniform crossover of two tournament winners, then per-gene
    mutation. Elitism: the top ``elite`` genomes survive unchanged, so the
    best-so-far fitness is monotone under a deterministic fitness function.

    Fitness caching: results are memoized by genome, so surviving elites are
    not re-evaluated. This assumes the caller's fitness is stable for a
    given genome (deterministic or seed-pinned). For a noisy fitness the
    cached elite value is exactly the standard "lucky elite" bias of
    elitism — re-evaluate inside your fitness function if that matters.

    Every evaluation and generation summary is logged through
    :func:`aether.utils.logging.get_logger`, with a JSONL sink (default
    ``runs/evolution.jsonl``) that :mod:`aether.evolution.diagnosis` and the
    dashboard can mine. NOTE: ``get_logger`` is idempotent per name — the
    first ``log_jsonl`` used in a process pins the sink for this logger.

    Parameters
    ----------
    pop_size:
        Number of genomes per generation.
    elite:
        How many top genomes are copied unchanged into the next generation.
    seed:
        Seeds this population's private ``random.Random``.
    log_jsonl:
        JSONL sink path, or ``None`` for console-only logging.
    """

    def __init__(self, pop_size: int = 8, elite: int = 2, seed: int = 0,
                 log_jsonl: str | Path | None = "runs/evolution.jsonl") -> None:
        if pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {pop_size}")
        if not 0 <= elite < pop_size:
            raise ValueError(f"elite must be in [0, pop_size), got {elite}")
        self.pop_size = int(pop_size)
        self.elite = int(elite)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.generation: int = 0
        self.genomes: list[Genome] = [random_genome(self.rng)
                                      for _ in range(self.pop_size)]
        self.history: list[dict[str, Any]] = []
        self._cache: dict[tuple, float] = {}
        self.logger = get_logger("aether.evolution", log_jsonl)

    # ------------------------------------------------------------------ #
    # Evolution loop
    # ------------------------------------------------------------------ #

    def _tournament(self, scored: list[tuple[float, Genome]]) -> Genome:
        """Tournament-2: the fitter of two uniform picks (copies it)."""
        a = self.rng.choice(scored)
        b = self.rng.choice(scored)
        return dict((a if a[0] >= b[0] else b)[1])

    def evolve(self, fitness_fn: Callable[[Genome], float],
               generations: int) -> list[dict[str, Any]]:
        """Run ``generations`` rounds of evaluate → select → vary.

        ``fitness_fn`` maps a genome to a scalar that is MAXIMIZED (return
        e.g. −val_loss or simulated out-of-sample Sharpe). Returns this
        call's history: one record per generation with keys ``gen``,
        ``best_fitness``, ``best_genome``, ``mean``.
        """
        run_history: list[dict[str, Any]] = []
        for _ in range(int(generations)):
            # ---- evaluate --------------------------------------------------
            scored: list[tuple[float, Genome]] = []
            for idx, genome in enumerate(self.genomes):
                key = _genome_key(genome)
                cached = key in self._cache
                if cached:
                    fitness = self._cache[key]
                else:
                    fitness = float(fitness_fn(genome))
                    self._cache[key] = fitness
                scored.append((fitness, genome))
                self.logger.info(
                    "gen %3d  genome %d/%d  fitness=%.6f%s",
                    self.generation, idx + 1, self.pop_size, fitness,
                    "  (cached)" if cached else "",
                    extra={
                        "aether_gen": self.generation,
                        "aether_index": idx,
                        "aether_fitness": fitness,
                        "aether_cached": cached,
                        "aether_genome": genome,
                    },
                )

            # ---- record ----------------------------------------------------
            scored.sort(key=lambda pair: pair[0], reverse=True)
            best_fitness, best_genome = scored[0]
            mean = sum(f for f, _ in scored) / len(scored)
            record = {
                "gen": self.generation,
                "best_fitness": best_fitness,
                "best_genome": dict(best_genome),
                "mean": mean,
            }
            run_history.append(record)
            self.history.append(record)
            self.logger.info(
                "gen %3d done: best=%.6f mean=%.6f",
                self.generation, best_fitness, mean,
                extra={
                    "aether_gen": self.generation,
                    "aether_best_fitness": best_fitness,
                    "aether_mean_fitness": mean,
                    "aether_best_genome": dict(best_genome),
                },
            )

            # ---- next generation: elites + mutated crossovers --------------
            next_gen: list[Genome] = [dict(g) for _, g in scored[: self.elite]]
            while len(next_gen) < self.pop_size:
                child = crossover(self._tournament(scored),
                                  self._tournament(scored), self.rng)
                next_gen.append(mutate(child, self.rng))
            self.genomes = next_gen
            self.generation += 1
        return run_history

    # ------------------------------------------------------------------ #
    # Persistence (plain JSON — a population is data, not weights)
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Write the full population state (genomes, generation counter,
        fitness cache, RNG state, history) as JSON. Atomic (tmp + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        version, internal, gauss_next = self.rng.getstate()
        state = {
            "pop_size": self.pop_size,
            "elite": self.elite,
            "seed": self.seed,
            "generation": self.generation,
            "genomes": self.genomes,
            "history": self.history,
            "cache": [[dict(key), val] for key, val in self._cache.items()],
            "rng_state": [version, list(internal), gauss_next],
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)  # atomic on POSIX

    @classmethod
    def load(cls, path: str | Path,
             log_jsonl: str | Path | None = "runs/evolution.jsonl") -> "Population":
        """Restore a population saved by :meth:`save`. A subsequent
        :meth:`evolve` continues exactly where the saved run stopped
        (same genomes, same RNG stream, warm fitness cache)."""
        state = json.loads(Path(path).read_text())
        pop = cls(pop_size=state["pop_size"], elite=state["elite"],
                  seed=state["seed"], log_jsonl=log_jsonl)
        pop.generation = int(state["generation"])
        pop.genomes = [dict(g) for g in state["genomes"]]
        pop.history = list(state["history"])
        pop._cache = {_genome_key(genome): float(fit)
                      for genome, fit in state["cache"]}
        version, internal, gauss_next = state["rng_state"]
        pop.rng.setstate((version, tuple(internal), gauss_next))
        return pop
