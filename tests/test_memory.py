"""Layer 2 memory bank tests: cosine retrieval correctness, ring-buffer
capacity semantics, persistence round-trip, consolidation, empty-bank query.
"""

from __future__ import annotations

import numpy as np
import pytest

memmod = pytest.importorskip("aether.worldmodel.memory")

from aether.worldmodel.interfaces import MemoryAnalog, MemoryConfig

DIM = 8


def _cfg(tmp_path, capacity: int = 64) -> MemoryConfig:
    return MemoryConfig(dim=DIM, capacity=capacity,
                        persist_dir=str(tmp_path / "memory"))


def _one_hots(n: int) -> np.ndarray:
    """n distinct, mutually orthogonal keys (cosine similarity exactly 0)."""
    assert n <= DIM
    keys = np.zeros((n, DIM), dtype=np.float32)
    keys[np.arange(n), np.arange(n)] = 1.0
    return keys


def _metas(n: int, start_ts: int = 1_000) -> list[dict]:
    return [{"ticker": "AAPL", "anchor_ts": start_ts + i,
             "outcome": {"fwd_ret_5m": 0.01 * i, "fwd_ret_30m": 0.02 * i}}
            for i in range(n)]


# --------------------------------------------------------------------------- #
# add / query
# --------------------------------------------------------------------------- #

class TestQuery:
    def test_exact_key_is_top_hit_with_cosine_one(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path))
        keys, metas = _one_hots(6), _metas(6)
        bank.add(keys, metas)
        assert len(bank) == 6

        hits = bank.query(keys[3], k=3)
        assert hits and isinstance(hits[0], MemoryAnalog)
        top = hits[0]
        assert top.similarity == pytest.approx(1.0, abs=1e-4)
        assert top.anchor_ts == metas[3]["anchor_ts"]
        assert top.ticker == "AAPL"
        assert top.outcome["fwd_ret_5m"] == pytest.approx(0.03)

    def test_results_sorted_by_similarity_desc(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path))
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((10, DIM)).astype(np.float32)
        bank.add(keys, _metas(10))
        hits = bank.query(keys[0], k=5)
        sims = [h.similarity for h in hits]
        assert sims == sorted(sims, reverse=True)
        assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in sims)

    def test_k_bounds_result_count(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path))
        bank.add(_one_hots(4), _metas(4))
        assert len(bank.query(_one_hots(1)[0], k=2)) == 2
        assert len(bank.query(_one_hots(1)[0], k=100)) <= 4

    def test_empty_bank_returns_empty_list(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path))
        assert len(bank) == 0
        assert bank.query(np.ones(DIM, dtype=np.float32), k=8) == []


# --------------------------------------------------------------------------- #
# capacity ring
# --------------------------------------------------------------------------- #

class TestCapacityRing:
    def test_oldest_entries_are_overwritten(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path, capacity=4))
        keys, metas = _one_hots(6), _metas(6)
        bank.add(keys[:4], metas[:4])
        bank.add(keys[4:], metas[4:])          # ring wraps: evicts 0 and 1
        assert len(bank) == 4

        # newest key must still be retrievable at cosine ~1
        newest = bank.query(keys[5], k=1)[0]
        assert newest.similarity == pytest.approx(1.0, abs=1e-4)
        assert newest.anchor_ts == metas[5]["anchor_ts"]

        # evicted keys are orthogonal to everything remaining: best match ~0
        evicted = bank.query(keys[0], k=1)[0]
        assert evicted.similarity < 0.5, \
            "capacity ring failed to evict the oldest entry"


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = _cfg(tmp_path)
        bank = memmod.MemoryBank(cfg)
        keys, metas = _one_hots(5), _metas(5)
        bank.add(keys, metas)
        bank.save()

        bank2 = memmod.MemoryBank(cfg)
        bank2.load()
        assert len(bank2) == len(bank) == 5
        top = bank2.query(keys[2], k=1)[0]
        assert top.similarity == pytest.approx(1.0, abs=1e-4)
        assert top.anchor_ts == metas[2]["anchor_ts"]
        assert top.outcome["fwd_ret_30m"] == pytest.approx(0.04)


# --------------------------------------------------------------------------- #
# consolidation (semantic tier)
# --------------------------------------------------------------------------- #

def _cluster_count(result: dict) -> int:
    if "clusters" in result:
        return len(result["clusters"])
    if "n_clusters" in result:
        return int(result["n_clusters"])
    return len(result)


class TestConsolidate:
    def test_cluster_count_at_most_requested(self, tmp_path):
        bank = memmod.MemoryBank(_cfg(tmp_path))
        rng = np.random.default_rng(1)
        keys = rng.standard_normal((12, DIM)).astype(np.float32)
        bank.add(keys, _metas(12))
        result = bank.consolidate(n_clusters=3)
        assert isinstance(result, dict)
        n = _cluster_count(result)
        assert 1 <= n <= 3, f"consolidate produced {n} clusters for n_clusters=3"
