"""Episode replay buffer and hindsight mining for the decision core.

Honest v1 scope
---------------
This is deliberately NOT full Hindsight Experience Replay. HER-style
relabeling ("pretend the goal was what actually happened") requires a
goal-conditioned policy, which Layer 3's current policy is not. What v1
*does* provide, and what downstream consumers can rely on today:

* :class:`EpisodeBuffer` — a capacity-bounded ring of complete episodes
  with per-episode sampling weights, so value-refit / auxiliary passes can
  oversample hard episodes.
* :class:`HindsightMiner` — scans stored episodes for the single most
  actionable counterfactual regret an intraday reversal trader has: the
  agent stayed **flat all session** while the session offered a large swing
  (``(max close − min close) / open`` above a threshold). Each hit emits a
  structured lesson ``{"kind": "missed_reversal", "ticker", "anchor_ts",
  "weight"}`` for the lessons buffer (worldmodel.interfaces.LESSONS_PATH
  consumers) and bumps the episode's sampling weight so the value function
  gets refit against these states more often.

Full HER-style relabeling arrives with goal-conditioned policies later;
until then hindsight feeds *sampling weights* and the *lessons buffer* only.

Episode schema
--------------
An episode is a plain dict. Keys the buffer/miner understand (all optional
except the trainer-provided core):

* ``obs``      dict[str, ndarray]  stacked per-step observations ``[T, ...]``
  (``obs["position"][:, 0]`` — signed position fraction — is how the miner
  detects flatness).
* ``actions``  dict[str, ndarray]  per-step actions.
* ``rewards``  ndarray ``[T]``.
* ``infos``    list[dict], length T — the env's per-step info dicts
  (``'trade_closed'`` records mark round-trips; a per-bar ``'close'`` /
  ``'close_px'`` is used for the swing when ``closes`` is absent).
* ``closes``   ndarray ``[T]`` — the session close path (preferred source).
* ``anchor_ts`` ndarray ``[T]`` int64 — per-bar anchor timestamps.
* ``ticker``   str.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Optional

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort ndarray view of an array-like (handles torch tensors
    without importing torch)."""
    if hasattr(value, "detach"):          # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


# --------------------------------------------------------------------------- #
# Episode buffer
# --------------------------------------------------------------------------- #

class EpisodeBuffer:
    """Capacity-bounded FIFO store of episodes with weighted sampling.

    Every episode gets a monotonically increasing integer id (stable across
    evictions — ids are never reused, so lessons and hard-example lists stay
    valid references even after the underlying episode is evicted).
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._episodes: "OrderedDict[int, dict]" = OrderedDict()
        self._weights: dict[int, float] = {}
        self._next_id = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    def ids(self) -> list[int]:
        """Ids of all currently stored episodes, oldest first."""
        return list(self._episodes.keys())

    def add(self, episode: dict) -> int:
        """Store an episode (evicting the oldest at capacity); returns its id."""
        eid = self._next_id
        self._next_id += 1
        self._episodes[eid] = episode
        self._weights[eid] = 1.0
        while len(self._episodes) > self.capacity:
            old_id, _ = self._episodes.popitem(last=False)  # FIFO eviction
            self._weights.pop(old_id, None)
        return eid

    def get(self, eid: int) -> Optional[dict]:
        return self._episodes.get(eid)

    def set_weight(self, eid: int, weight: float) -> None:
        """Set an episode's sampling weight (must be > 0)."""
        if eid in self._episodes:
            self._weights[eid] = max(1e-6, float(weight))

    def sample(self, n: int) -> list[dict]:
        """Draw ``n`` episodes, probability ∝ weight.

        Sampling is *with replacement* (standard for replay: hard episodes
        should legitimately appear multiple times in a refit batch). Raises
        on an empty buffer rather than silently returning nothing.
        """
        if not self._episodes:
            raise ValueError("cannot sample from an empty EpisodeBuffer")
        ids = self.ids()
        weights = np.array([self._weights[i] for i in ids], dtype=np.float64)
        probs = weights / weights.sum()
        chosen = self._rng.choice(len(ids), size=int(n), replace=True, p=probs)
        return [self._episodes[ids[i]] for i in chosen]


# --------------------------------------------------------------------------- #
# Hindsight mining
# --------------------------------------------------------------------------- #

class HindsightMiner:
    """Finds sessions where the agent sat flat through a large swing.

    Parameters
    ----------
    swing_threshold:
        Minimum ``(max close − min close) / session open`` for a session to
        count as a missed opportunity (default 1%: an intraday swing worth
        hunting after fees/slippage).
    flat_tol:
        Absolute tolerance on the signed position fraction below which the
        agent is considered flat.
    """

    def __init__(self, swing_threshold: float = 0.01,
                 flat_tol: float = 1e-6) -> None:
        self.swing_threshold = float(swing_threshold)
        self.flat_tol = float(flat_tol)

    # -- episode field extraction ---------------------------------------- #

    @staticmethod
    def _closes(episode: dict) -> Optional[np.ndarray]:
        """Session close path: prefer ``closes``, fall back to per-bar info."""
        if "closes" in episode:
            return _to_numpy(episode["closes"]).astype(np.float64)
        closes = []
        for info in episode.get("infos", []):
            if not isinstance(info, dict):
                return None
            px = info.get("close", info.get("close_px"))
            if px is None:
                return None
            closes.append(float(px))
        return np.asarray(closes, dtype=np.float64) if closes else None

    def _stayed_flat(self, episode: dict) -> bool:
        """True only when we can *prove* the agent held no position.

        Evidence, in order: the position-fraction observation channel
        (obs["position"][:, 0]), else the absence of any 'trade_closed'
        info record. With neither source available we conservatively return
        False — never mint a lesson from an unverifiable premise.
        """
        obs = episode.get("obs")
        if isinstance(obs, dict) and "position" in obs:
            pos = _to_numpy(obs["position"])
            if pos.ndim >= 2 and pos.shape[-1] >= 1:
                return bool(np.max(np.abs(pos[..., 0])) <= self.flat_tol)
        infos = episode.get("infos")
        if infos is not None:
            return not any(
                isinstance(info, dict) and info.get("trade_closed")
                for info in infos)
        return False

    # -- mining ------------------------------------------------------------ #

    def mine(
        self, buffer: EpisodeBuffer, update_weights: bool = True
    ) -> tuple[list[dict], list[int]]:
        """Scan the buffer for missed-reversal sessions.

        Returns ``(lessons, hard_ids)``:

        * ``lessons`` — entries ``{"kind": "missed_reversal", "ticker",
          "anchor_ts", "weight"}``; ``anchor_ts`` is the timestamp of the
          session's *first* price extreme (the pivot the reversal turned
          on — for a V-shaped session that is the low, for an inverted V
          the high; bar index when the episode has no ``anchor_ts``), and
          ``weight`` is the swing expressed in threshold multiples, so a 3%
          swing at a 1% threshold weighs 3.0.
        * ``hard_ids`` — buffer ids for value-refit oversampling.

        With ``update_weights`` (default) each hit's buffer sampling weight
        is bumped to ``1 + weight``, so :meth:`EpisodeBuffer.sample`
        immediately favors the missed sessions.
        """
        lessons: list[dict] = []
        hard_ids: list[int] = []
        for eid in buffer.ids():
            episode = buffer.get(eid)
            if episode is None:
                continue
            closes = self._closes(episode)
            if closes is None or closes.size < 2 or closes[0] <= 0.0:
                continue
            if not self._stayed_flat(episode):
                continue
            swing = float((closes.max() - closes.min()) / closes[0])
            if swing <= self.swing_threshold:
                continue

            # The reversal pivot is the first extreme reached: everything
            # after it is the move the agent failed to capture.
            pivot = int(min(np.argmin(closes), np.argmax(closes)))
            anchor = episode.get("anchor_ts")
            anchor_ts = (int(_to_numpy(anchor)[pivot])
                         if anchor is not None else pivot)

            weight = swing / self.swing_threshold
            lessons.append({
                "kind": "missed_reversal",
                "ticker": str(episode.get("ticker", "")),
                "anchor_ts": anchor_ts,
                "weight": float(weight),
            })
            hard_ids.append(eid)
            if update_weights:
                buffer.set_weight(eid, 1.0 + weight)
        return lessons, hard_ids
