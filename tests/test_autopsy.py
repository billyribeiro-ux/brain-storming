"""Layer 2 trade autopsy tests.

Two crafted scenarios pin the verdict logic:

* bad_loss — a stopped-out long where the same idea 15 bars later (and/or a
  wider stop) would have profited: the loss was avoidable.
* good_loss — price never recovers; no variation helps: the stop did its job.

Plus: counterfactual bookkeeping (both stop variations, delta identity),
JSON-serializability via dataclasses.asdict, and the lessons/report sinks.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

autopsymod = pytest.importorskip("aether.worldmodel.autopsy")

from aether.worldmodel.autopsy import Autopsist, append_lessons, save_report
from aether.worldmodel.interfaces import LESSONS_PATH, AutopsyReport

from tests.conftest import synth_sessions
from tests.helpers_stack import epoch_s, make_trade, session_bar_time

DAY = synth_sessions(1)[0]


def _bars_from_closes(closes: np.ndarray) -> pd.DataFrame:
    """OHLCV frame (lake convention) with open = previous close."""
    closes = np.asarray(closes, dtype=np.float64)
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    dates = [session_bar_time(DAY, m) for m in range(len(closes))]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": opens, "high": high, "low": low, "close": closes,
        "volume": np.full(len(closes), 1_000, dtype=np.int64),
    })


def _avoidable_loss_bars() -> pd.DataFrame:
    """Flat 100 -> dip to ~98.6 (stops the trade) -> rally to 105."""
    closes = np.concatenate([
        np.full(30, 100.0),                    # minutes 0..29: flat
        np.linspace(100.0, 98.7, 7),           # 30..36: dip through the 99 stop
        np.full(9, 98.6),                      # 37..45: base builds
        np.linspace(98.6, 105.0, 45),          # 46..90: strong rally
        np.full(60, 105.0),                    # 91..150: holds the gains
    ])
    return _bars_from_closes(closes)


def _hopeless_loss_bars() -> pd.DataFrame:
    """Flat 100, then a relentless decline: every re-entry loses too."""
    closes = np.concatenate([
        np.full(30, 100.0),
        np.linspace(100.0, 88.0, 121),
    ])
    return _bars_from_closes(closes)


def _losing_long(exit_minute: int):
    return make_trade(
        trade_id="T-loss", ticker="AAPL", side="long",
        entry_ts=epoch_s(session_bar_time(DAY, 30)),
        exit_ts=epoch_s(session_bar_time(DAY, exit_minute)),
        entry_px=100.0, exit_px=99.0, qty=10.0,
        stop_px=99.0, target_px=103.0, exit_reason="stop",
    )


@pytest.fixture(scope="module")
def bad_loss_report() -> AutopsyReport:
    return Autopsist().autopsy(_losing_long(34), _avoidable_loss_bars())


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #

class TestVerdicts:
    def test_avoidable_loss_is_bad_loss(self, bad_loss_report):
        assert bad_loss_report.verdict == "bad_loss"

    def test_hopeless_loss_is_good_loss(self):
        report = Autopsist().autopsy(_losing_long(42), _hopeless_loss_bars())
        assert report.verdict == "good_loss"

    def test_verdict_vocabulary(self, bad_loss_report):
        assert bad_loss_report.verdict in {
            "good_loss", "bad_loss", "good_win", "lucky_win"}


# --------------------------------------------------------------------------- #
# report structure
# --------------------------------------------------------------------------- #

class TestReportStructure:
    def test_counterfactuals_include_both_stop_variations(self, bad_loss_report):
        stop_cfs = [cf for cf in bad_loss_report.counterfactuals
                    if "stop" in cf.description.lower()]
        assert len(stop_cfs) >= 2, (
            "expected at least two stop-variation counterfactuals, got "
            f"{[cf.description for cf in bad_loss_report.counterfactuals]}")
        assert len({cf.description for cf in stop_cfs}) >= 2

    def test_counterfactual_delta_identity(self, bad_loss_report):
        actual = bad_loss_report.trade.pnl
        assert bad_loss_report.counterfactuals, "no counterfactuals produced"
        for cf in bad_loss_report.counterfactuals:
            assert np.isfinite(cf.pnl) and np.isfinite(cf.delta)
            assert cf.delta == pytest.approx(cf.pnl - actual, abs=1e-6)

    def test_narrative_and_lists(self, bad_loss_report):
        assert isinstance(bad_loss_report.narrative, str)
        assert bad_loss_report.narrative.strip()
        assert isinstance(bad_loss_report.drivers, list)
        assert isinstance(bad_loss_report.causal_context, list)
        assert isinstance(bad_loss_report.lessons, list)
        for lesson in bad_loss_report.lessons:
            assert isinstance(lesson, dict)

    def test_report_is_json_serializable(self, bad_loss_report):
        payload = json.dumps(dataclasses.asdict(bad_loss_report))
        back = json.loads(payload)
        assert back["verdict"] == bad_loss_report.verdict
        assert back["trade"]["trade_id"] == "T-loss"


# --------------------------------------------------------------------------- #
# sinks: lessons buffer + report files
# --------------------------------------------------------------------------- #

class TestSinks:
    def test_append_lessons_writes_append_only_jsonl(self, bad_loss_report,
                                                     tmp_path):
        append_lessons([bad_loss_report], tmp_path)
        lessons_file = tmp_path / LESSONS_PATH
        assert lessons_file.is_file(), f"expected {LESSONS_PATH} under data root"
        lines = [ln for ln in lessons_file.read_text().splitlines() if ln.strip()]
        assert lines, "lessons file is empty"
        for ln in lines:
            json.loads(ln)

        append_lessons([bad_loss_report], tmp_path)   # append-only: grows
        lines2 = [ln for ln in lessons_file.read_text().splitlines()
                  if ln.strip()]
        assert len(lines2) > len(lines)

    def test_save_report_persists_valid_json(self, bad_loss_report, tmp_path):
        before = set(tmp_path.rglob("*"))
        save_report(bad_loss_report, tmp_path)
        new_files = [p for p in set(tmp_path.rglob("*")) - before if p.is_file()]
        assert new_files, "save_report created no file under the data root"
        parsed_any = False
        for p in new_files:
            text = p.read_text()
            try:
                json.loads(text)                     # one JSON document
                parsed_any = True
            except json.JSONDecodeError:
                for ln in text.splitlines():         # or JSONL: every line
                    if ln.strip():
                        json.loads(ln)
                        parsed_any = True
        assert parsed_any, "no JSON payload found in saved report artifacts"
