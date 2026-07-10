"""Layer 2 causal graph tests.

The core claim: on a synthetic system where X drives Y at lag 1
(``y_t = 0.8 * x_{t-1} + noise``, x white), the fitted snapshot contains a
confident x->y lag-1 edge and no comparable reverse edge. Plus: JSON
round-trip fidelity, snapshot persistence, top_drivers ordering, and the
lake-backed frame builder.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

causal = pytest.importorskip("aether.worldmodel.causal")

from aether.worldmodel.interfaces import (
    CAUSAL_TICKER_CHANNELS,
    CausalConfig,
    CausalEdge,
    CausalGraphSnapshot,
)

from tests.conftest import TEST_TICKERS, TRAIN_END, TRAIN_START


# --------------------------------------------------------------------------- #
# Synthetic lag-1 system
# --------------------------------------------------------------------------- #

def _xy_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    y[1:] = 0.8 * x[:-1] + 0.1 * rng.standard_normal(n - 1)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="1min")
    return pd.DataFrame({"x": x, "y": y}, index=idx)


@pytest.fixture(scope="module")
def fitted_snapshot() -> CausalGraphSnapshot:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = CausalConfig(n_lags=2, l1_penalty=1e-3, stability_bootstraps=4,
                       hidden_dim=8, epochs=150, lr=1e-2)
    model = causal.CausalModel(cfg)
    model.fit(_xy_frame())
    return model.snapshot("2026-01-05", "2026-01-05")


def _edges(snap, src, dst, lag):
    return [e for e in snap.edges if e.src == src and e.dst == dst
            and int(e.lag) == lag]


class TestLagOneRecovery:
    def test_true_edge_found_strong_and_confident(self, fitted_snapshot):
        hits = _edges(fitted_snapshot, "x", "y", 1)
        assert hits, "fitted graph is missing the true x->y lag-1 edge"
        best = max(hits, key=lambda e: abs(e.weight))
        assert abs(best.weight) > 0.1, f"x->y lag1 too weak: {best.weight}"
        assert best.confidence > 0.5, f"x->y lag1 unstable: {best.confidence}"
        assert 0.0 <= best.confidence <= 1.0

    def test_no_strong_reverse_edge(self, fitted_snapshot):
        for e in _edges(fitted_snapshot, "y", "x", 1):
            assert abs(e.weight) < 0.1 / 3.0, \
                f"spurious reverse edge y->x lag1 with weight {e.weight}"

    def test_snapshot_metadata(self, fitted_snapshot):
        assert fitted_snapshot.fitted_start == "2026-01-05"
        assert fitted_snapshot.fitted_end == "2026-01-05"
        assert set(fitted_snapshot.nodes) >= {"x", "y"}
        for e in fitted_snapshot.edges:
            assert e.src in fitted_snapshot.nodes
            assert e.dst in fitted_snapshot.nodes
            assert 0 <= int(e.lag)
            assert np.isfinite(e.weight)
            assert 0.0 <= e.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

def _edge_key(e: CausalEdge) -> tuple:
    return (e.src, e.dst, int(e.lag), round(float(e.weight), 6),
            round(float(e.confidence), 6))


class TestSerialization:
    def test_json_roundtrip_preserves_edges(self, fitted_snapshot):
        payload = causal.snapshot_to_json(fitted_snapshot)
        json.loads(payload)  # must be genuine JSON
        back = causal.snapshot_from_json(payload)
        assert back.fitted_start == fitted_snapshot.fitted_start
        assert back.fitted_end == fitted_snapshot.fitted_end
        assert list(back.nodes) == list(fitted_snapshot.nodes)
        assert ({_edge_key(e) for e in back.edges}
                == {_edge_key(e) for e in fitted_snapshot.edges})

    def test_save_and_load_latest(self, fitted_snapshot, tmp_path):
        causal.save_snapshot(fitted_snapshot, tmp_path)
        back = causal.load_latest_snapshot(tmp_path)
        assert back is not None
        assert ({_edge_key(e) for e in back.edges}
                == {_edge_key(e) for e in fitted_snapshot.edges})


# --------------------------------------------------------------------------- #
# top_drivers ordering (interface-level behavior)
# --------------------------------------------------------------------------- #

class TestTopDrivers:
    def test_ordering_by_weight_times_confidence(self):
        snap = CausalGraphSnapshot(
            fitted_start="2026-01-05", fitted_end="2026-01-05",
            nodes=["a", "b", "c", "y"],
            edges=[
                CausalEdge("a", "y", 1, weight=0.9, confidence=0.5),   # 0.45
                CausalEdge("b", "y", 1, weight=-0.6, confidence=1.0),  # 0.60
                CausalEdge("c", "y", 0, weight=0.2, confidence=0.9),   # 0.18
                CausalEdge("y", "a", 1, weight=5.0, confidence=1.0),   # outgoing
            ],
        )
        top = snap.top_drivers("y", k=2)
        assert [e.src for e in top] == ["b", "a"]
        assert all(e.dst == "y" for e in top)

    def test_k_truncates(self):
        snap = CausalGraphSnapshot(
            "s", "e", ["a", "b", "y"],
            [CausalEdge("a", "y", 1, 0.5, 0.5), CausalEdge("b", "y", 1, 0.4, 0.5)],
        )
        assert len(snap.top_drivers("y", k=1)) == 1
        assert snap.top_drivers("missing") == []


# --------------------------------------------------------------------------- #
# Lake-backed frame builder
# --------------------------------------------------------------------------- #

class TestBuildCausalFrame:
    def test_frame_has_ticker_channel_columns(self, synthetic_store):
        frame = causal.build_causal_frame(
            synthetic_store, list(TEST_TICKERS), TRAIN_START, TRAIN_END)
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 100, "frame should cover the synthetic sessions"
        cols = [f"{t}.{ch}" for t in TEST_TICKERS
                for ch in CAUSAL_TICKER_CHANNELS]
        for col in cols:
            assert col in frame.columns, f"missing channel column {col}"
        assert np.isfinite(frame[cols].to_numpy(dtype=float)).all(), \
            "causal frame must be finite (no NaNs leaking from the lake)"
