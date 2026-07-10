"""Multi-tier memory: outcome-tagged episodic recall + semantic consolidation.

This module implements the
:class:`~aether.worldmodel.interfaces.MemoryBankProtocol`. It is the part of
the world model that answers two different questions:

* **Episodic tier** (``add`` / ``query``) — *"when has the market looked
  like THIS before, and how did those moments end?"* Every stored entry is
  a perception embedding (the key) tagged with its instrument, timestamp
  and realized outcome. Retrieval is cosine nearest-neighbor: two moments
  whose fused embeddings point the same way in representation space were,
  by the perception model's judgment, the same kind of moment. Live/replay
  consumers pass ``as_of`` (the query bar's anchor time) so retrieval is
  embargo-filtered: an analog whose outcome window overlaps — or follows —
  the query moment would leak the future (see :meth:`MemoryBank.query`).

* **Semantic tier** (``consolidate``) — *"what KINDS of moments exist, and
  what happens after each kind on average?"* Periodic k-means over the
  stored keys compresses hundreds of thousands of episodes into a few dozen
  regime prototypes, each carrying the mean realized outcome of its
  members. This is the analog of sleep-time memory consolidation: raw
  episodes distilled into transferable structure the decision layer can
  condition on cheaply.

Storage design
--------------
Keys live in one pre-allocated ``float32 [capacity, dim]`` numpy matrix used
as a **ring buffer**: writes advance a pointer and wrap, so once full the
bank holds the most recent ``capacity`` episodes and inserts stay O(N·dim)
with zero allocation churn. Metadata rides in a parallel Python list —
row ``i`` of the matrix and element ``i`` of the list always describe the
same episode.

The similarity math
-------------------
Cosine similarity is used instead of raw dot product or L2 because
perception embeddings are not norm-calibrated — training shapes their
*direction*, while the norm drifts with training dynamics. Normalizing both
sides makes retrieval scores comparable across checkpoints and lie in
``[-1, 1]``::

    sim(q, k_i) = ⟨q / ‖q‖, k_i / ‖k_i‖⟩

Persistence
-----------
``save()`` writes ``<persist_dir>/keys.npz`` (the live key rows plus ring
pointer and size) and ``<persist_dir>/metas.jsonl`` (one JSON object per
line, same physical order as the key rows). Both are written atomically
(temp file + rename) so a crash mid-save never corrupts the previous state.
``load()`` restores the exact ring layout when the capacity is unchanged;
under a changed capacity the entries survive but rotation-age ordering
inside the ring is no longer exact (documented in :meth:`load`).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from aether.worldmodel.interfaces import MemoryAnalog, MemoryBankProtocol, MemoryConfig

#: Every stored meta MUST carry these keys; ``outcome`` must be a dict.
#: They are the minimum needed to turn a retrieved row into a MemoryAnalog.
_REQUIRED_META_KEYS: tuple[str, ...] = ("ticker", "anchor_ts", "outcome")

_KEYS_FILE = "keys.npz"
_METAS_FILE = "metas.jsonl"

#: Norm floor for cosine normalization — an (unlikely) all-zero key must not
#: produce NaN similarities, it just scores ~0 against everything.
_NORM_EPS: float = 1e-8


class MemoryBank(MemoryBankProtocol):
    """Ring-buffered episodic memory with a k-means semantic tier.

    Parameters
    ----------
    cfg:
        :class:`~aether.worldmodel.interfaces.MemoryConfig` — key
        dimensionality (must equal the perception fused dim), ring
        capacity, and the persistence directory.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        # Pre-allocated ring storage. Rows [0, _size) are live; _ptr is the
        # next write position. Before the first wrap _ptr == _size; after
        # wrapping, _size stays == capacity and _ptr marks the oldest row.
        self._keys: np.ndarray = np.zeros((cfg.capacity, cfg.dim), dtype=np.float32)
        self._metas: list[dict] = [None] * cfg.capacity  # type: ignore[list-item]
        self._ptr: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_meta(meta: dict, index: int) -> None:
        """Enforce the outcome-tagging contract on one meta dict.

        A memory without its outcome is useless to every consumer
        (autopsies rank analogs by what happened next; consolidation
        averages outcomes per cluster), so malformed entries are rejected
        loudly at insertion time instead of surfacing as silent Nones at
        retrieval time.
        """
        if not isinstance(meta, dict):
            raise ValueError(
                f"metas[{index}] must be a dict, got {type(meta).__name__}"
            )
        missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise ValueError(
                f"metas[{index}] is missing required key(s) {missing}; every "
                f"memory meta must contain {list(_REQUIRED_META_KEYS)}"
            )
        if not isinstance(meta["outcome"], dict):
            raise ValueError(
                f"metas[{index}]['outcome'] must be a dict of realized "
                f"outcomes (e.g. {{'fwd_ret_5m': ...}}), got "
                f"{type(meta['outcome']).__name__}"
            )

    def _as_key_matrix(self, keys: Tensor | np.ndarray) -> np.ndarray:
        """Coerce ``keys`` to a float32 ``[N, dim]`` numpy matrix (copy)."""
        if isinstance(keys, Tensor):
            arr = keys.detach().cpu().numpy()
        else:
            arr = np.asarray(keys)
        if arr.ndim != 2 or arr.shape[1] != self.cfg.dim:
            raise ValueError(
                f"keys must be [N, {self.cfg.dim}], got {tuple(arr.shape)}"
            )
        # astype always copies here — the ring must own its rows (a view
        # into caller memory could be mutated after insertion).
        return arr.astype(np.float32, copy=True)

    # ------------------------------------------------------------------ #
    # Episodic tier
    # ------------------------------------------------------------------ #

    def add(self, keys: Tensor | np.ndarray, metas: Sequence[dict]) -> None:
        """Insert ``N`` episodes into the ring.

        Parameters
        ----------
        keys:
            ``[N, dim]`` torch tensor (any device) or numpy array.
        metas:
            ``N`` dicts, each containing at least ``'ticker'``,
            ``'anchor_ts'`` and ``'outcome'`` (a dict). Values should be
            JSON-serializable — persistence is JSONL. Extra keys are kept
            and returned in :class:`MemoryAnalog.meta`.

        All metas are validated **before** any row is written, so a bad
        batch never leaves the bank half-mutated. Inserting more rows than
        ``capacity`` keeps only the trailing ``capacity`` of them (the ring
        semantics applied eagerly).
        """
        arr = self._as_key_matrix(keys)
        n = arr.shape[0]
        if len(metas) != n:
            raise ValueError(
                f"len(metas)={len(metas)} does not match keys N={n}"
            )
        for i, meta in enumerate(metas):
            self._validate_meta(meta, i)
        if n == 0:
            return
        # Shallow-copy each meta so later caller-side mutation of the dict
        # cannot silently rewrite stored history.
        metas = [dict(m) for m in metas]

        capacity = self.cfg.capacity
        if n >= capacity:
            # The batch alone overflows the ring: only its trailing
            # `capacity` rows survive; the layout resets to a fresh ring.
            self._keys[:] = arr[n - capacity:]
            self._metas = list(metas[n - capacity:])
            self._ptr = 0
            self._size = capacity
            return

        first = min(n, capacity - self._ptr)       # rows before the wrap
        self._keys[self._ptr:self._ptr + first] = arr[:first]
        self._metas[self._ptr:self._ptr + first] = metas[:first]
        rest = n - first                           # rows wrapped to the front
        if rest:
            self._keys[:rest] = arr[first:]
            self._metas[:rest] = metas[first:]
        self._ptr = (self._ptr + n) % capacity
        self._size = min(self._size + n, capacity)

    def query(self, key: Tensor | np.ndarray, k: int = 8,
              as_of: int | None = None,
              embargo_s: int = 1800) -> list[MemoryAnalog]:
        """Retrieve the ``k`` most cosine-similar episodes to ``key``.

        Parameters
        ----------
        key:
            ``[dim]`` (or ``[1, dim]``) tensor/array — typically the
            current fused perception embedding.
        k:
            Maximum analogs to return; silently truncated to the bank size.
        as_of:
            Optional query-moment anchor timestamp (epoch-seconds). When
            given, only episodes with ``anchor_ts <= as_of - embargo_s``
            are eligible. This is the no-look-ahead invariant applied to
            retrieval: a stored outcome such as ``fwd_ret_30m`` covers the
            30 minutes AFTER its own anchor, so an analog anchored inside
            the embargo window (or later) would leak the query moment's
            own future into the evidence. ``None`` (the default) keeps the
            unfiltered behavior for offline consumers that query with
            hindsight on purpose (autopsies, consolidation audits).
        embargo_s:
            Width of the exclusion window in seconds; defaults to 1800
            (30 minutes — the longest outcome horizon stored per episode).

        Returns
        -------
        list[MemoryAnalog]
            Sorted by similarity, descending. Empty bank — or no episode
            surviving the embargo — ⇒ ``[]`` (a young system simply has no
            analogs yet — not an error).
        """
        if self._size == 0:
            return []
        if isinstance(key, Tensor):
            q_arr = key.detach().cpu().numpy()
        else:
            q_arr = np.asarray(key)
        q_arr = q_arr.reshape(-1)
        if q_arr.shape[0] != self.cfg.dim:
            raise ValueError(
                f"query key must have {self.cfg.dim} elements, "
                f"got {q_arr.shape[0]}"
            )

        if as_of is None:
            live: np.ndarray | None = None                # all rows, zero-copy
            bank_np = self._keys[: self._size]
        else:
            cutoff = int(as_of) - int(embargo_s)
            live = np.array(
                [i for i in range(self._size)
                 if int(self._metas[i]["anchor_ts"]) <= cutoff],
                dtype=np.int64)
            if live.size == 0:
                return []
            bank_np = self._keys[live]

        # Normalized matmul == cosine. Retrieval math runs in torch (CPU)
        # so topk and normalization reuse one well-tested code path.
        bank = torch.from_numpy(bank_np)                           # [S, dim]
        q = torch.from_numpy(q_arr.astype(np.float32, copy=False)) # [dim]
        bank_n = bank / bank.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
        q_n = q / q.norm().clamp_min(_NORM_EPS)
        sims = bank_n @ q_n                                        # [S]

        top = torch.topk(sims, k=min(k, int(bank_np.shape[0])))
        analogs: list[MemoryAnalog] = []
        for sim, pos in zip(top.values.tolist(), top.indices.tolist()):
            meta = self._metas[pos if live is None else int(live[pos])]
            extras = {
                key_: val for key_, val in meta.items()
                if key_ not in _REQUIRED_META_KEYS
            }
            analogs.append(MemoryAnalog(
                similarity=float(sim),
                ticker=meta["ticker"],
                anchor_ts=int(meta["anchor_ts"]),
                outcome=meta["outcome"],
                meta=extras,
            ))
        return analogs

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Persist the bank to ``cfg.persist_dir`` (atomic per file).

        Layout: ``keys.npz`` holds the live key rows in *physical* ring
        order plus the ring pointer and size; ``metas.jsonl`` holds one
        JSON object per line in the same physical order, so row i of the
        matrix and line i of the JSONL always pair up.
        """
        persist = Path(self.cfg.persist_dir)
        persist.mkdir(parents=True, exist_ok=True)

        keys_path = persist / _KEYS_FILE
        tmp = keys_path.with_name(keys_path.name + ".tmp")
        # Write through a file handle: np.savez would append ".npz" to a
        # bare temp *path* and break the rename pairing.
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                keys=self._keys[: self._size],
                ptr=np.int64(self._ptr),
                size=np.int64(self._size),
            )
        os.replace(tmp, keys_path)  # atomic on POSIX

        metas_path = persist / _METAS_FILE
        tmp = metas_path.with_name(metas_path.name + ".tmp")
        with open(tmp, "w") as fh:
            for i in range(self._size):
                fh.write(json.dumps(self._metas[i], default=str) + "\n")
        os.replace(tmp, metas_path)

    def load(self) -> None:
        """Restore state written by :meth:`save` from ``cfg.persist_dir``.

        Raises
        ------
        FileNotFoundError
            If either persistence file is missing.
        ValueError
            On dimension mismatch, on a saved size exceeding the current
            capacity, or on a keys/metas count mismatch (corruption).

        Ring-pointer semantics: when the saved size equals the current
        capacity (a full ring) the saved pointer is restored verbatim, so
        overwrite order continues exactly. When the bank was not full the
        pointer is placed after the last live row. If the capacity changed
        between save and load, entries all survive but the ring's
        oldest-first overwrite order is only approximate — acceptable for
        a recency buffer, and flagged here for honesty.
        """
        persist = Path(self.cfg.persist_dir)
        keys_path = persist / _KEYS_FILE
        metas_path = persist / _METAS_FILE
        if not keys_path.is_file() or not metas_path.is_file():
            raise FileNotFoundError(
                f"MemoryBank.load(): missing {keys_path} and/or {metas_path}"
            )

        with np.load(keys_path) as z:
            keys = z["keys"].astype(np.float32, copy=True)
            saved_ptr = int(z["ptr"])
            saved_size = int(z["size"])
        if keys.ndim != 2 or keys.shape[1] != self.cfg.dim:
            raise ValueError(
                f"saved keys have shape {keys.shape}, expected "
                f"[*, {self.cfg.dim}] — config/persist mismatch"
            )
        if saved_size > self.cfg.capacity:
            raise ValueError(
                f"saved bank holds {saved_size} entries but capacity is "
                f"{self.cfg.capacity}; raise MemoryConfig.capacity or "
                f"re-persist with the smaller bank"
            )
        if keys.shape[0] != saved_size:
            raise ValueError(
                f"keys.npz is inconsistent: {keys.shape[0]} rows vs "
                f"size={saved_size}"
            )

        metas: list[dict] = []
        with open(metas_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))
        if len(metas) != saved_size:
            raise ValueError(
                f"metas.jsonl holds {len(metas)} entries but keys.npz says "
                f"{saved_size} — the persisted pair is corrupt"
            )
        for i, meta in enumerate(metas):
            self._validate_meta(meta, i)

        self._keys = np.zeros((self.cfg.capacity, self.cfg.dim), dtype=np.float32)
        self._keys[:saved_size] = keys
        self._metas = list(metas) + [None] * (self.cfg.capacity - saved_size)  # type: ignore[list-item]
        self._size = saved_size
        # Full ring under unchanged capacity ⇒ exact pointer; otherwise
        # append after the last live row (see docstring).
        self._ptr = saved_ptr if saved_size == self.cfg.capacity \
            else saved_size % self.cfg.capacity

    # ------------------------------------------------------------------ #
    # Semantic tier
    # ------------------------------------------------------------------ #

    def consolidate(self, n_clusters: int = 32, seed: int = 0) -> dict:
        """Distill the episodic tier into regime prototypes (semantic tier).

        Runs plain Lloyd's k-means (20 iterations, seeded, in torch) over
        every stored key and aggregates the realized outcomes per cluster.
        20 iterations is deliberate: k-means over a few hundred thousand
        64-d embeddings is near-converged well before that, and a fixed
        budget keeps consolidation's cost predictable for scheduled
        (sleep-time) runs.

        Parameters
        ----------
        n_clusters:
            Prototype count; truncated to the bank size when the bank is
            smaller.
        seed:
            Seed for the centroid initialization (random distinct
            episodes), so consolidation is reproducible.

        Returns
        -------
        dict
            ``{"centroids": [[float]*dim]*K,
               "cluster_stats": [{"size": int,
                                  "mean_outcome": {key: float}}]*K}``
            where ``mean_outcome`` averages every *numeric* outcome value
            over the cluster's members (non-numeric outcome entries and
            bools are skipped; a key missing from some members is averaged
            over the members that do carry it). Empty bank ⇒ empty lists.
        """
        if self._size == 0:
            return {"centroids": [], "cluster_stats": []}

        keys = torch.from_numpy(self._keys[: self._size].copy())  # [S, dim]
        k = min(n_clusters, self._size)

        # --- Lloyd's algorithm, seeded ------------------------------------
        gen = torch.Generator().manual_seed(seed)
        init = torch.randperm(self._size, generator=gen)[:k]
        centroids = keys[init].clone()                             # [K, dim]
        assign = torch.zeros(self._size, dtype=torch.long)
        for _ in range(20):
            # Assign: nearest centroid in Euclidean distance.
            assign = torch.cdist(keys, centroids).argmin(dim=1)    # [S]
            # Update: mean of members; an emptied cluster keeps its previous
            # centroid (it can re-acquire members on a later iteration).
            for j in range(k):
                members = assign == j
                if bool(members.any()):
                    centroids[j] = keys[members].mean(dim=0)

        # --- Per-cluster outcome aggregation ------------------------------
        stats: list[dict] = []
        for j in range(k):
            member_idx = torch.nonzero(assign == j, as_tuple=False).flatten()
            sums: dict[str, float] = {}
            counts: dict[str, int] = {}
            for idx in member_idx.tolist():
                for key_, val in self._metas[idx]["outcome"].items():
                    # bool is an int subclass in Python — exclude it, and
                    # skip non-finite values so one bad entry cannot poison
                    # a cluster mean.
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        continue
                    if not math.isfinite(float(val)):
                        continue
                    sums[key_] = sums.get(key_, 0.0) + float(val)
                    counts[key_] = counts.get(key_, 0) + 1
            stats.append({
                "size": int(member_idx.numel()),
                "mean_outcome": {
                    key_: sums[key_] / counts[key_] for key_ in sorted(sums)
                },
            })

        return {"centroids": centroids.tolist(), "cluster_stats": stats}
