"""Layer 6 dashboard view-builder tests.

Views must be pure builders: synthetic inputs in, plotly Figures out — no
Streamlit runtime required. training_curves must survive a malformed JSONL
line (real training logs get truncated mid-write), and append_feedback must
produce an append-only JSONL audit trail.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

views = pytest.importorskip("aether.dashboard.views")
go = pytest.importorskip("plotly.graph_objects")

from aether.worldmodel.interfaces import CausalEdge, CausalGraphSnapshot

from tests.conftest import make_synthetic_1min


def _find_figure(ret) -> "go.Figure":
    if isinstance(ret, go.Figure):
        return ret
    if isinstance(ret, (tuple, list)):
        for item in ret:
            if isinstance(item, go.Figure):
                return item
    pytest.fail(f"no plotly Figure in return value of type {type(ret)!r}")


# --------------------------------------------------------------------------- #
# figure builders
# --------------------------------------------------------------------------- #

class TestFigures:
    def test_candles_figure(self):
        bars = make_synthetic_1min("AAPL", sessions=1, seed=3).head(120)
        try:
            fig = views.candles_figure(bars)
        except TypeError:
            fig = views.candles_figure(bars, "AAPL")
        fig = _find_figure(fig)
        assert len(fig.data) >= 1, "candles figure has no traces"

    def test_causal_figure(self):
        snap = CausalGraphSnapshot(
            fitted_start="2026-01-05", fitted_end="2026-01-09",
            nodes=["AAPL.ret_1m", "SPY.ret_1m", "MKT.news_rate"],
            edges=[
                CausalEdge("SPY.ret_1m", "AAPL.ret_1m", 1, 0.55, 0.9),
                CausalEdge("MKT.news_rate", "AAPL.ret_1m", 2, -0.30, 0.7),
                CausalEdge("AAPL.ret_1m", "SPY.ret_1m", 1, 0.10, 0.4),
            ])
        fig = _find_figure(views.causal_figure(snap))
        assert len(fig.data) >= 1

    def test_equity_figure(self):
        rng = np.random.default_rng(0)
        ts = pd.date_range("2026-01-05 09:30", periods=100, freq="1min")
        equity = 100_000.0 * np.cumprod(1.0 + 1e-4 * rng.standard_normal(100))
        eq = pd.DataFrame({"ts": ts, "equity": equity})
        fig = _find_figure(views.equity_figure(eq))
        assert len(fig.data) >= 1


# --------------------------------------------------------------------------- #
# training curves from JSONL (with one corrupted line)
# --------------------------------------------------------------------------- #

class TestTrainingCurves:
    def test_parses_jsonl_and_survives_malformed_line(self, tmp_path):
        log = tmp_path / "train.jsonl"
        with open(log, "w") as fh:
            for i in range(15):
                fh.write(json.dumps({"step": i * 100, "loss": 1.0 / (i + 1),
                                     "val_loss": 1.1 / (i + 1)}) + "\n")
            fh.write("{this line was truncated mid-write and is not json\n")
            for i in range(15, 30):
                fh.write(json.dumps({"step": i * 100, "loss": 1.0 / (i + 1),
                                     "val_loss": 1.1 / (i + 1)}) + "\n")
        fig = _find_figure(views.training_curves(str(log)))
        assert len(fig.data) >= 1, \
            "training_curves produced an empty figure from a valid log"


# --------------------------------------------------------------------------- #
# feedback sink
# --------------------------------------------------------------------------- #

class TestAppendFeedback:
    @staticmethod
    def _append(path, entry):
        try:
            return views.append_feedback(path, entry)
        except TypeError:
            return views.append_feedback(entry, path)

    def test_appends_valid_json_lines(self, tmp_path):
        target = tmp_path / "feedback.jsonl"
        first = {"signal_id": "FB-1", "verdict": "good_call", "note": "unit"}
        second = {"signal_id": "FB-2", "verdict": "bad_call", "note": "unit"}
        self._append(str(target), first)
        self._append(str(target), second)

        candidates = [p for p in tmp_path.rglob("*.jsonl") if p.is_file()]
        assert candidates, "append_feedback wrote no JSONL file"
        sink = next((p for p in candidates if "FB-1" in p.read_text()), None)
        assert sink is not None, "feedback entries not found in any JSONL"

        lines = [ln for ln in sink.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2, "append_feedback must be append-only"
        parsed = [json.loads(ln) for ln in lines]
        by_id = {p.get("signal_id"): p for p in parsed}
        assert set(by_id) >= {"FB-1", "FB-2"}
        for entry in (first, second):
            stored = by_id[entry["signal_id"]]
            assert entry.items() <= stored.items(), \
                "stored feedback lost fields from the submitted entry"


def test_filter_records_period_and_summary():
    """Period (from/to) filtering on the trades list + the period summary
    the dashboard's Trades page shows for the filtered window."""
    views = pytest.importorskip("aether.dashboard.views")
    trades = [
        {"trade_id": "A1", "ticker": "AAPL", "entry_ts": 1781602320, "pnl": 10.0, "fees": 1.0},  # 2026-06-16
        {"trade_id": "A2", "ticker": "AAPL", "entry_ts": 1782207120, "pnl": -5.0, "fees": 1.0},  # 2026-06-23
        {"trade_id": "S1", "ticker": "SPY", "entry_ts": 1782811920, "pnl": 3.0, "fees": 1.0},    # 2026-06-30
        {"trade_id": "X0", "ticker": "SPY", "pnl": 99.0},  # undated: excluded by time filters
    ]
    mid = views.filter_records(trades, start="2026-06-20", end="2026-06-29",
                               ts_keys=("entry_ts",))
    assert [t["trade_id"] for t in mid] == ["A2"]
    # open-ended bounds
    assert [t["trade_id"] for t in
            views.filter_records(trades, end="2026-06-16", ts_keys=("entry_ts",))] == ["A1"]
    assert [t["trade_id"] for t in
            views.filter_records(trades, start="2026-06-30", ts_keys=("entry_ts",))] == ["S1"]
    # inclusive on both ends; ticker filter composes
    both = views.filter_records(trades, start="2026-06-16", end="2026-06-30",
                                ticker="AAPL", ts_keys=("entry_ts",))
    assert [t["trade_id"] for t in both] == ["A1", "A2"]
    summary = views.trades_period_summary(mid)
    assert summary == {"n_trades": 1, "net_pnl": -5.0, "win_rate": 0.0,
                       "avg_pnl": -5.0, "fees": 1.0}
    # empty window is well-defined, not an error
    assert views.trades_period_summary([]) == {
        "n_trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0, "fees": 0.0}
