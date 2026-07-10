"""Layer 5 backtester tests.

A stub signal engine fires exactly one long signal at the top of the sine
swing, so the resulting trade is a predictable stopped-out loss: this pins
fill math, per-trade accounting, the equity-curve identity, stats keys,
loss autopsies, and the save_result artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

btmod = pytest.importorskip("aether.execution.backtest")
tactmod = pytest.importorskip("aether.execution.tactics")
riskmod = pytest.importorskip("aether.execution.risk")
autopsymod = pytest.importorskip("aether.worldmodel.autopsy")

from aether.execution.backtest import Backtester, PassThroughRisk, save_result
from aether.execution.interfaces import BacktestConfig, ReversalSignal, RiskConfig
from aether.execution.risk import RiskManager
from aether.worldmodel.autopsy import Autopsist
from aether.worldmodel.interfaces import TradeRecord

from tests.conftest import synth_sessions
from tests.helpers_stack import SESSION_BARS, load_npz, make_embedding_npz_dir

FEES_BPS = 1.0
SLIP_BPS = 2.0
INITIAL_EQ = 100_000.0

#: fire inside the falling flank of session 1's sine swing: entries here drop
#: >= 2% (the stop) long before they could ever gain 5% (the target).
FIRE_LO, FIRE_HI = 16, 28


class OneShotEngine:
    """Emits exactly one long AAPL signal, then stays silent."""

    def __init__(self, arrays: dict):
        self.arrays = arrays
        self.calls = 0
        self.fired_idx: int | None = None

    def generate(self, ticker, idx, emb=None, *args, **kwargs):
        self.calls += 1
        idx = int(idx)
        if ticker != "AAPL" or self.fired_idx is not None:
            return None
        if not FIRE_LO <= idx <= FIRE_HI:
            return None
        self.fired_idx = idx
        px = float(self.arrays["close_px"][idx])
        return ReversalSignal(
            signal_id="BT-1", ts=int(self.arrays["anchor_ts"][idx]),
            ticker="AAPL", side="long", conviction=0.9,
            entry_px=px, stop_px=px * 0.98, target_px=px * 1.05,
            horizon_bars=30, size_frac=0.5, rationale="stub signal",
            evidence=None)


class AlwaysLongEngine:
    """Fires a long AAPL signal at EVERY bar it is consulted on."""

    def __init__(self, arrays: dict, stop_frac: float, target_frac: float,
                 horizon: int = 10_000):
        self.arrays = arrays
        self.stop_frac = float(stop_frac)
        self.target_frac = float(target_frac)
        self.horizon = int(horizon)

    def generate(self, ticker, idx, emb=None, *args, **kwargs):
        if ticker != "AAPL":
            return None
        idx = int(idx)
        px = float(self.arrays["close_px"][idx])
        return ReversalSignal(
            signal_id=f"AL-{idx}", ts=int(self.arrays["anchor_ts"][idx]),
            ticker="AAPL", side="long", conviction=0.9,
            entry_px=px, stop_px=px * self.stop_frac,
            target_px=px * self.target_frac,
            horizon_bars=self.horizon, size_frac=0.1,
            rationale="always-long stub", evidence=None)


def _tactics(cfg):
    try:
        return tactmod.ExecutionTactics()
    except TypeError:
        return tactmod.ExecutionTactics(cfg)


def _downtrend(arrays):
    """Rewrite AAPL into a clean −0.3%/bar downtrend per session (next_*
    stays session-bounded: last row NaN, per the EmbeddingStore contract)."""
    n = SESSION_BARS
    n_sessions = arrays["close_px"].shape[0] // n
    for s in range(n_sessions):
        lo, hi = s * n, (s + 1) * n
        base = 100.0 * (1.0 + 0.01 * s)
        m = np.arange(n, dtype=np.float64)
        c = base * (1.0 - 0.003) ** m
        o = np.empty_like(c)
        o[0] = base
        o[1:] = c[:-1]
        h = np.maximum(o, c) * 1.001
        l = np.minimum(o, c) * 0.999
        arrays["close_px"][lo:hi] = c
        arrays["next_open"][lo:hi] = np.append(o[1:], np.nan)
        arrays["next_high"][lo:hi] = np.append(h[1:], np.nan)
        arrays["next_low"][lo:hi] = np.append(l[1:], np.nan)
        arrays["next_close"][lo:hi] = np.append(c[1:], np.nan)


def _always_long_run(tmp_path, stop_frac: float, target_frac: float,
                     mutator=None):
    from tests.helpers_stack import rewrite_npz

    emb_dir = make_embedding_npz_dir(tmp_path, seed=0)
    if mutator is not None:
        rewrite_npz(emb_dir, "AAPL", mutator)
    arrays = load_npz(emb_dir, "AAPL")
    days = synth_sessions(2)
    cfg = BacktestConfig(
        tickers=("AAPL",), start=str(days[0]), end=str(days[-1]),
        fees_bps=FEES_BPS, slippage_bps=SLIP_BPS,
        initial_equity=INITIAL_EQ, autopsy_losses=False)
    engine = AlwaysLongEngine(arrays, stop_frac, target_frac)
    bt = Backtester(cfg, str(emb_dir), engine, PassThroughRisk(), None,
                    _tactics(cfg), data_root=tmp_path)
    return cfg, arrays, bt.run_signals()


@pytest.fixture(scope="module")
def bt_run(tmp_path_factory):
    emb_dir = make_embedding_npz_dir(tmp_path_factory.mktemp("emb"), seed=0)
    arrays = load_npz(emb_dir, "AAPL")
    days = synth_sessions(2)
    cfg = BacktestConfig(
        tickers=("AAPL", "SPY"), start=str(days[0]), end=str(days[-1]),
        fees_bps=FEES_BPS, slippage_bps=SLIP_BPS,
        initial_equity=INITIAL_EQ, autopsy_losses=True)
    engine = OneShotEngine(arrays)
    bt = Backtester(cfg, str(emb_dir), engine, RiskManager(RiskConfig()),
                    Autopsist(), _tactics(cfg))
    result = bt.run_signals()
    return cfg, engine, arrays, result


class TestRunSignals:
    def test_engine_swept_and_signal_recorded(self, bt_run):
        _, engine, _, result = bt_run
        assert engine.calls > 50, "backtester barely consulted the engine"
        assert engine.fired_idx is not None, "stub never got a chance to fire"
        assert any(s.signal_id == "BT-1" for s in result.signals)

    def test_trade_fill_math(self, bt_run):
        cfg, engine, arrays, result = bt_run
        assert result.trades, "the accepted signal must produce a trade"
        rec = next(t for t in result.trades if t.ticker == "AAPL")
        assert isinstance(rec, TradeRecord)
        assert rec.side == "long"
        slip = cfg.slippage_bps / 1e4
        assert rec.entry_px == pytest.approx(
            arrays["next_open"][engine.fired_idx] * (1.0 + slip), rel=1e-6), \
            "backtest entry must fill at the next bar's open plus slippage"
        assert rec.qty > 0

    def test_engineered_loss_stops_out(self, bt_run):
        _, _, _, result = bt_run
        rec = next(t for t in result.trades if t.ticker == "AAPL")
        assert rec.exit_reason == "stop"
        assert rec.pnl < 0
        assert rec.exit_px < rec.entry_px

    def test_per_trade_accounting_identity(self, bt_run):
        cfg, _, _, result = bt_run
        for rec in result.trades:
            sign = 1.0 if rec.side == "long" else -1.0
            fees = (rec.entry_px + rec.exit_px) * rec.qty * cfg.fees_bps / 1e4
            assert rec.fees == pytest.approx(fees, rel=1e-6)
            assert rec.pnl == pytest.approx(
                sign * (rec.exit_px - rec.entry_px) * rec.qty - fees,
                rel=1e-6, abs=1e-6)

    def test_equity_curve_identity(self, bt_run):
        cfg, _, _, result = bt_run
        eq = result.equity_curve
        assert isinstance(eq, pd.DataFrame)
        assert len(eq) > 0
        assert "equity" in eq.columns
        final = float(eq["equity"].iloc[-1])
        expected = cfg.initial_equity + sum(t.pnl for t in result.trades)
        assert final == pytest.approx(expected, rel=1e-9), \
            "final equity must equal initial equity plus realized pnl"
        assert np.isfinite(eq["equity"].to_numpy(dtype=float)).all()

    def test_stats_keys_complete(self, bt_run):
        _, _, _, result = bt_run
        stats = result.stats
        for key in ("sharpe", "sortino", "max_dd", "hit_rate",
                    "avg_win", "avg_loss"):
            assert key in stats, f"stats missing {key!r}"
        assert 0.0 <= float(stats["hit_rate"]) <= 1.0
        assert any(isinstance(v, dict) for v in stats.values()), \
            "stats must include a per-ticker breakdown dict"

    def test_losses_are_autopsied(self, bt_run):
        _, _, _, result = bt_run
        assert any(t.pnl < 0 for t in result.trades)
        assert len(result.autopsies) >= 1, \
            "autopsy_losses=True must autopsy the losing trade"


class TestSessionBoundary:
    def test_no_position_survives_a_session_boundary(self, tmp_path):
        """MAJOR-2 regression: with contract-conform stores (session-last
        rows carry NaN ``next_*``) the eod force-close must fire at the
        session's LAST REAL bar; positions must never cross the overnight
        boundary."""
        # stop far below / target far above: nothing exits intraday, so
        # every position rides to the session end.
        _, arrays, result = _always_long_run(tmp_path, stop_frac=0.50,
                                             target_frac=2.00)
        trades = [t for t in result.trades if t.ticker == "AAPL"]
        assert trades, "the always-long engine must produce trades"
        eod = [t for t in trades if t.exit_reason == "eod"]
        assert len(eod) == 2, \
            f"expected one eod close per session, got {len(eod)}"
        slip = SLIP_BPS / 1e4
        for k, rec in enumerate(sorted(eod, key=lambda t: t.exit_ts)):
            # eod fill = final REAL bar's close (the bar after the
            # second-to-last anchor), slipped against the agent.
            i = (k + 1) * SESSION_BARS - 2
            assert rec.exit_ts == int(arrays["anchor_ts"][i]) + 60
            assert rec.exit_px == pytest.approx(
                float(arrays["next_close"][i]) * (1.0 - slip), rel=1e-9)
        for t in trades:
            entry_day = pd.Timestamp(int(t.entry_ts), unit="s").date()
            exit_day = pd.Timestamp(int(t.exit_ts), unit="s").date()
            assert entry_day == exit_day, \
                f"trade {t.trade_id} survived a session boundary"


class TestNoSameBarReentry:
    def test_no_entry_prints_at_another_trades_exit(self, tmp_path):
        """MAJOR-3 regression: a ticker whose position exits on fill bar
        t+1 may not accept a new entry decided at t — the entry would
        print (at t+1's open) BEFORE the exit it depends on. Earliest
        re-entry decides at t+1 and fills at t+2."""
        # clean downtrend + tight stop: repeated stop-outs with immediate
        # re-entry pressure from the always-long stub.
        _, _, result = _always_long_run(tmp_path, stop_frac=0.99,
                                        target_frac=10.0,
                                        mutator=_downtrend)
        trades = result.trades
        stops = [t for t in trades if t.exit_reason == "stop"]
        assert len(stops) >= 2, \
            "fixture must produce repeated stop-outs to exercise re-entry"
        for a in trades:
            for b in trades:
                if a is b or a.ticker != b.ticker:
                    continue
                assert int(a.entry_ts) != int(b.exit_ts), (
                    f"entry {a.trade_id} prints on the same bar as exit "
                    f"{b.trade_id} — same-bar exit→re-entry leaked through")


class TestSaveResult:
    def test_artifacts_exist_and_reload(self, bt_run, tmp_path):
        _, _, _, result = bt_run
        save_result(result, tmp_path, "bt_unit")
        files = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert files, "save_result wrote nothing under the data root"

        json_files = [p for p in files if p.suffix in (".json", ".jsonl")]
        parquet_files = [p for p in files if p.suffix == ".parquet"]
        assert json_files, "expected at least one JSON artifact"
        assert parquet_files, "expected at least one parquet artifact"
        for p in json_files:
            text = p.read_text()
            try:
                json.loads(text)
            except json.JSONDecodeError:
                for ln in text.splitlines():
                    if ln.strip():
                        json.loads(ln)
        for p in parquet_files:
            frame = pd.read_parquet(p)
            assert len(frame) > 0
