"""Layer 6 paper-trader tests: position-management catch-up.

Pins the fix that ``run_once`` must manage EVERY completed execution bar
since the previous run: a stop touched during downtime has to fire at the
bar that touched it — the old newest-bar-only management silently walked
positions straight past their stops.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

papermod = pytest.importorskip("aether.execution.paper")

from aether.execution.interfaces import PaperConfig
from aether.execution.paper import PaperTrader

from tests.helpers_stack import load_npz, make_embedding_npz_dir

FEES_BPS = 1.0
SLIP_BPS = 2.0


class NoSignals:
    """Signal engine that always abstains (isolates position management)."""

    def generate(self, *args, **kwargs):
        return None


class NoRisk:
    def assess(self, *args, **kwargs):  # pragma: no cover - must not be hit
        raise AssertionError("no signals were expected in this test")

    def reset_day(self) -> None:
        return None


def _trader(tmp_path, emb_dir) -> PaperTrader:
    cfg = PaperConfig(blotter_path=str(tmp_path / "paper" / "blotter.jsonl"),
                      state_path=str(tmp_path / "paper" / "state.json"))
    aether_cfg = types.SimpleNamespace(data=types.SimpleNamespace(
        root=str(tmp_path / "lake"), cache_root=str(tmp_path / "cache")))
    return PaperTrader(cfg, aether_cfg, NoSignals(), NoRisk(), tactics=None,
                       embeddings_dir=emb_dir, fees_bps=FEES_BPS,
                       slippage_bps=SLIP_BPS)


class TestManagementCatchUp:
    def test_stop_touched_during_downtime_fires_at_that_bar(self, tmp_path):
        emb_dir = make_embedding_npz_dir(tmp_path, seed=0)
        arrays = load_npz(emb_dir, "AAPL")
        trader = _trader(tmp_path, emb_dir)

        # Open long position, last managed at anchor 25; the stop sits just
        # above bar 31's low (the execution bar of anchor 30) — it is
        # touched there, ~90 bars before the newest anchor, and nowhere
        # near the newest (flat-zone) prices.
        stop_i = 30
        stop_px = float(arrays["next_low"][stop_i]) * 1.0001
        assert float(np.nanmin(arrays["next_low"][-40:])) > stop_px, \
            "fixture: the newest bars must NOT touch the stop"
        entry_px = float(arrays["close_px"][20])
        q = 100.0
        pos = {
            "trade_id": "paper-AAPL-test-1", "ticker": "AAPL", "side": "long",
            "q": q, "entry_px": entry_px,
            "entry_fee": q * entry_px * FEES_BPS / 1e4,
            "entry_ts": int(arrays["anchor_ts"][20]) + 60,
            "stop_px": stop_px, "target_px": entry_px * 10.0,
            "horizon": 10_000, "bars_held": 0, "conviction": 0.9,
            "mark": entry_px,
            "last_managed_ts": int(arrays["anchor_ts"][25]),
            "meta": {"order_type": "market"},
        }
        state = {
            "cash": 100_000.0 - q * entry_px - pos["entry_fee"],
            "equity": 100_000.0, "realized_pnl": 0.0, "n_trades": 0,
            "day": None, "day_start_equity": 100_000.0,
            "positions": {"AAPL": pos}, "last_bar_ts": {},
        }
        trader.state_path.parent.mkdir(parents=True, exist_ok=True)
        trader.state_path.write_text(json.dumps(state))

        trader.run_once(sync=False)

        saved = json.loads(trader.state_path.read_text())
        assert saved["positions"] == {}, \
            "the stop touched during downtime must close the position"
        assert saved["n_trades"] == 1

        exits = []
        for line in trader.blotter_path.read_text().splitlines():
            ev = json.loads(line)
            if ev["kind"] == "order" and ev["payload"].get("action") == "exit":
                exits.append(ev["payload"]["trade"])
        assert len(exits) == 1
        rec = exits[0]
        assert rec["exit_reason"] == "stop"
        # ... at the bar that touched it, not at the newest bar:
        assert rec["exit_ts"] == int(arrays["anchor_ts"][stop_i]) + 60
        slip = SLIP_BPS / 1e4
        raw = min(stop_px, float(arrays["next_open"][stop_i]))
        assert rec["exit_px"] == pytest.approx(raw * (1.0 - slip), rel=1e-9)

    def test_pending_newest_bar_is_not_marked_managed(self, tmp_path):
        """The newest anchor's execution bar is unknown (NaN next_*): it
        must stay pending — last_managed_ts may only advance through bars
        that were actually evaluated (the one-bar operational lag)."""
        emb_dir = make_embedding_npz_dir(tmp_path, seed=0)
        arrays = load_npz(emb_dir, "AAPL")
        trader = _trader(tmp_path, emb_dir)

        entry_px = float(arrays["close_px"][20])
        q = 10.0
        pos = {
            "trade_id": "paper-AAPL-test-2", "ticker": "AAPL", "side": "long",
            "q": q, "entry_px": entry_px,
            "entry_fee": q * entry_px * FEES_BPS / 1e4,
            "entry_ts": int(arrays["anchor_ts"][20]) + 60,
            # stop far below / target far above: nothing fires
            "stop_px": entry_px * 0.01, "target_px": entry_px * 100.0,
            "horizon": 10_000, "bars_held": 0, "conviction": 0.9,
            "mark": entry_px,
            "last_managed_ts": int(arrays["anchor_ts"][25]),
            "meta": {"order_type": "market"},
        }
        state = {
            "cash": 100_000.0 - q * entry_px - pos["entry_fee"],
            "equity": 100_000.0, "realized_pnl": 0.0, "n_trades": 0,
            "day": None, "day_start_equity": 100_000.0,
            "positions": {"AAPL": pos}, "last_bar_ts": {},
        }
        trader.state_path.parent.mkdir(parents=True, exist_ok=True)
        trader.state_path.write_text(json.dumps(state))

        trader.run_once(sync=False)

        saved = json.loads(trader.state_path.read_text())
        kept = saved["positions"]["AAPL"]
        # newest anchor has NaN next_* (session-last row): it stays pending
        assert not np.isfinite(float(arrays["next_open"][-1]))
        assert int(kept["last_managed_ts"]) == int(arrays["anchor_ts"][-2]), \
            "management must stop at the last COMPLETED execution bar"
        # marking uses the newest completed close (close_px), never NaN
        assert kept["mark"] == pytest.approx(float(arrays["close_px"][-1]))
