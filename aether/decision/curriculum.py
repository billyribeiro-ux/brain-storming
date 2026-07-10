"""Reversal-clarity curriculum over precomputed embedding sessions.

Idea: PPO learns faster when early updates see episodes whose reward signal
is *legible* — sessions with one clean, deep intraday reversal — before
being exposed to choppy, ambiguous tape. The sampler ranks every available
(ticker, date) episode by a "reversal clarity" score and unlocks deeper
quantiles of that ranking in stages as training progresses.

Reversal clarity
----------------
For a session close path ``c_0..c_T`` (``c_0`` = session open reference)::

    max_drawup   = max_t ( c_t − min_{s ≤ t} c_s ) / c_0   (best long swing)
    max_drawdown = max_t ( max_{s ≤ t} c_s − c_t ) / c_0   (best short swing)
    clarity      = (max_drawup + max_drawdown) / 2

Both terms use *running* extrema, so each measures a swing that actually
unfolded in time order (a fall to the running max after a rise, a recovery
from the running min after a fall) — exactly the structure a reversal
hunter can capture, unlike raw high−low range which also rewards one-way
drift.

Look-ahead safety
-----------------
Clarity is computed from each episode's own session data, which is PAST
data at training time — the whole embedding store is historical. Mining
history to decide the *order* in which training episodes are presented is
look-ahead-safe: no future information ever enters an observation, a
reward, or any statistic the agent conditions on *within* an episode. The
one sacred invariant (bar t sees only data ≤ t) governs what happens inside
the env; curriculum ordering happens strictly outside it.

Data source: per-ticker ``.npz`` files written by the EmbeddingStore
(``close_px`` float64, ``anchor_ts`` int64 epoch-seconds, naive-ET), grouped
into sessions by calendar date of ``anchor_ts``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger("aether.rl.curriculum")


def reversal_clarity(closes: np.ndarray) -> float:
    """Reversal clarity of one session close path (see module docstring).

    Returns 0.0 for degenerate paths (fewer than 2 bars or a non-positive
    open) rather than raising — such sessions simply rank last.
    """
    c = np.asarray(closes, dtype=np.float64)
    if c.size < 2 or c[0] <= 0.0:
        return 0.0
    running_min = np.minimum.accumulate(c)
    running_max = np.maximum.accumulate(c)
    max_drawup = float(np.max(c - running_min)) / float(c[0])
    max_drawdown = float(np.max(running_max - c)) / float(c[0])
    return 0.5 * (max_drawup + max_drawdown)


class CurriculumSampler:
    """Stage-gated episode sampler ordered by reversal clarity.

    Parameters
    ----------
    embeddings_dir:
        Directory of per-ticker ``.npz`` embedding files (EmbeddingStore
        layout; ``{ticker}.npz`` with ``close_px`` and ``anchor_ts``).
    tickers:
        Ticker aliases to scan. Missing files are skipped with a warning;
        an entirely empty result is a hard error (a silent empty curriculum
        would starve the env of episodes).
    stages:
        Number of curriculum stages. Stage ``k`` (0-based) exposes the top
        ``(k + 1) / stages`` clarity quantile — the final stage exposes
        everything, so the curriculum biases *order*, never final coverage.

    Episode ids are ``(ticker, "YYYY-MM-DD")`` tuples, matching the
    ``TradingEnv.episodes`` convention; ``TradingEnv.reset(episode_ids=...)``
    consumes them directly (the trainer asks :meth:`eligible` each collect).
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        tickers: tuple[str, ...] | list[str],
        stages: int = 4,
    ) -> None:
        if stages < 1:
            raise ValueError(f"stages must be >= 1, got {stages}")
        self.embeddings_dir = Path(embeddings_dir)
        self.tickers = tuple(tickers)
        self.stages = int(stages)
        self.stage = 0

        scored = self._score_episodes()
        if not scored:
            raise FileNotFoundError(
                f"CurriculumSampler found no episodes for tickers "
                f"{self.tickers} under {self.embeddings_dir} — run the "
                f"embedding precompute first")
        # Rank by clarity (descending); deterministic tie-break on the id.
        scored.sort(key=lambda item: (-item[1], item[0]))
        self._ranked: list[tuple[str, str]] = [eid for eid, _ in scored]
        self._clarity: dict[tuple[str, str], float] = dict(scored)
        logger.info(
            "curriculum: %d episodes over %d tickers, %d stages "
            "(clarity %.4f .. %.4f)",
            len(self._ranked), len(self.tickers), self.stages,
            scored[0][1], scored[-1][1])

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _score_episodes(self) -> list[tuple[tuple[str, str], float]]:
        """Load every ticker's store and score each session's clarity."""
        scored: list[tuple[tuple[str, str], float]] = []
        for ticker in self.tickers:
            path = self.embeddings_dir / f"{ticker}.npz"
            if not path.exists():
                logger.warning("curriculum: no embeddings for %s (%s missing)",
                               ticker, path)
                continue
            with np.load(path) as data:
                if "close_px" in data:
                    closes = np.asarray(data["close_px"], dtype=np.float64)
                elif "close" in data:                     # tolerated alias
                    closes = np.asarray(data["close"], dtype=np.float64)
                else:
                    logger.warning(
                        "curriculum: %s has no close_px array — skipped", path)
                    continue
                anchor_ts = np.asarray(data["anchor_ts"], dtype=np.int64)

            # Chronological order, then group bars into sessions by the
            # calendar date of the (naive-ET) anchor timestamp.
            order = np.argsort(anchor_ts, kind="stable")
            anchor_ts, closes = anchor_ts[order], closes[order]
            days = anchor_ts // 86_400
            for day in np.unique(days):
                sel = days == day
                date = str(np.datetime64(int(day), "D"))
                score = reversal_clarity(closes[sel])
                scored.append(((ticker, date), score))
        return scored

    # ------------------------------------------------------------------ #
    # Stage control & queries
    # ------------------------------------------------------------------ #

    def advance(self, update_idx: int, every: int = 50) -> int:
        """Set the stage from the training clock: one stage per ``every``
        updates, capped at the final (everything-unlocked) stage."""
        self.stage = min(self.stages - 1, int(update_idx) // int(every))
        return self.stage

    def eligible(self) -> list[tuple[str, str]]:
        """Episode ids unlocked at the current stage: the top
        ``(stage + 1) / stages`` quantile of the clarity ranking."""
        fraction = (self.stage + 1) / self.stages
        k = max(1, math.ceil(fraction * len(self._ranked)))
        return list(self._ranked[:k])

    def clarity(self, episode_id: tuple[str, str]) -> float:
        """Clarity score of one episode id (KeyError if unknown)."""
        return self._clarity[episode_id]

    @property
    def episodes(self) -> list[tuple[str, str]]:
        """All episode ids, best-ranked first."""
        return list(self._ranked)

    def __len__(self) -> int:
        return len(self._ranked)
