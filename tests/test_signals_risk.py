"""Layer 5 signal engine + risk manager tests.

SignalEngine is driven with a duck-typed deterministic policy (its ``mode``
returns a fixed intent with fixed probabilities) over the hand-controlled
embedding store, so emitted signals are fully predictable. RiskManager is
pure contract: every hard rail gets its own scenario, including the daily
circuit breaker's stickiness and the correlation haircut's monotonicity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

sigmod = pytest.importorskip("aether.execution.signals")
riskmod = pytest.importorskip("aether.execution.risk")

from aether.execution.interfaces import (
    ReversalSignal,
    RiskConfig,
    RiskDecision,
    SignalConfig,
)
from aether.execution.risk import RiskManager
from aether.execution.signals import SignalEngine

from tests.helpers_stack import (
    epoch_s,
    load_npz,
    make_embedding_npz_dir,
    session_bar_time,
)
from tests.conftest import synth_sessions

IDX = 100                      # flat-zone bar with 100 bars of history


# --------------------------------------------------------------------------- #
# deterministic fake policy
# --------------------------------------------------------------------------- #

class FakePolicy:
    """Duck-typed policy: mode() returns a fixed intent with fixed probs."""

    def __init__(self, meta: int = 1, trade: int = 1, prob: float = 0.9):
        self.meta, self.trade, self.prob = meta, trade, prob

    def initial_carry(self, batch: int = 1):
        return None

    @staticmethod
    def _batch(obs) -> int:
        if isinstance(obs, dict) and "market" in obs:
            market = torch.as_tensor(obs["market"])
            return market.shape[0] if market.dim() > 1 else 1
        return 1

    def mode(self, obs=None, carry=None, *args, **kwargs):
        b = self._batch(obs)
        actions = {
            "meta": torch.full((b,), self.meta, dtype=torch.long),
            "trade": torch.full((b,), self.trade, dtype=torch.long),
            "size": torch.full((b,), 0.5),
            "stop": torch.full((b,), 0.5),
            "target": torch.full((b,), 0.5),
        }
        logprobs = {k: torch.full((b,), math.log(self.prob)) for k in actions}
        return actions, logprobs, torch.zeros(b), carry

    act = mode


@pytest.fixture(scope="module")
def emb(tmp_path_factory):
    dir_ = make_embedding_npz_dir(tmp_path_factory.mktemp("emb"), seed=0)
    return load_npz(dir_, "AAPL")


def _engine(cfg: SignalConfig | None = None, policy: FakePolicy | None = None):
    return SignalEngine(policy=policy or FakePolicy(), dynamics=None,
                        memory=None, causal_snapshot=None, perception=None,
                        cfg=cfg or SignalConfig(min_conviction=0.2))


# --------------------------------------------------------------------------- #
# signal emission
# --------------------------------------------------------------------------- #

class TestSignalEmission:
    def test_long_signal_with_sane_levels(self, emb):
        sig = _engine().generate("AAPL", IDX, emb)
        assert isinstance(sig, ReversalSignal)
        assert sig.ticker == "AAPL"
        assert sig.side == "long"
        assert sig.signal_id
        assert 0.0 <= sig.conviction <= 1.0
        assert sig.conviction >= 0.2
        assert sig.horizon_bars > 0
        assert 0.0 < sig.size_frac <= 1.0
        assert isinstance(sig.rationale, str) and sig.rationale.strip()
        assert sig.ts in (int(emb["anchor_ts"][IDX]), int(emb["anchor_ts"][IDX + 1]))
        # levels: long means stop < entry < target, near the anchor price
        assert sig.stop_px < sig.entry_px < sig.target_px
        px = float(emb["close_px"][IDX])
        assert 0.8 * px < sig.entry_px < 1.2 * px

    def test_short_signal_levels_invert(self, emb):
        sig = _engine(policy=FakePolicy(meta=2)).generate("AAPL", IDX, emb)
        assert sig is not None
        assert sig.side == "short"
        assert sig.target_px < sig.entry_px < sig.stop_px

    def test_evidence_bundle_complete(self, emb):
        sig = _engine().generate("AAPL", IDX, emb)
        ev = sig.evidence
        assert ev is not None, "real signals must carry their evidence"
        for name in ("policy_prob", "imagination_agreement", "analog_winrate"):
            val = getattr(ev, name)
            assert 0.0 <= float(val) <= 1.0, f"{name}={val} outside [0,1]"
        for name in ("aleatoric", "epistemic", "anomaly"):
            val = float(getattr(ev, name))
            assert np.isfinite(val) and val >= 0.0, name
        assert isinstance(ev.saliency, list)
        assert isinstance(ev.causal_drivers, list)
        assert isinstance(ev.analogs_summary, list)


class RecordingDictPolicy:
    """HierarchicalPolicy-shaped (dict) mode() that records every obs it is
    shown — pins the two-pass policy query."""

    def __init__(self):
        self.seen: list[dict] = []

    def mode(self, obs, carry=None):
        self.seen.append({k: torch.as_tensor(v).clone()
                          for k, v in obs.items()})
        b = int(torch.as_tensor(obs["market"]).shape[0])
        return {
            "meta": torch.ones(b, dtype=torch.long),          # hunt_long
            "trade": torch.ones(b, dtype=torch.long),         # enter
            "meta_probs": torch.tensor([[0.1, 0.8, 0.1]]).repeat(b, 1),
            "trade_probs": torch.tensor([[0.1, 0.7, 0.1, 0.1]]).repeat(b, 1),
            "size_mean": torch.full((b,), 0.5),
            "stop_mean": torch.full((b,), 0.5),
            "target_mean": torch.full((b,), 0.5),
        }


class TestTwoPassPolicyQuery:
    def test_enter_prob_is_queried_with_the_intent_in_force(self, emb):
        """policy_prob must be p(intent) from a neutral meta-decision state
        times p(enter) from the state where the chosen intent is IN FORCE
        (flags=0, intent one-hot) — the env gates entries on the
        pre-update intent, so pass 1's trade head is reward-inert noise."""
        pol = RecordingDictPolicy()
        sig = _engine(policy=pol).generate("AAPL", IDX, emb)
        assert sig is not None and sig.side == "long"
        assert len(pol.seen) == 2, "exactly two policy passes expected"
        o1, o2 = pol.seen
        # pass 1: meta-decision bar, neutral stand_aside prior
        assert float(o1["flags"][0, 0]) == pytest.approx(1.0)
        assert int(torch.argmax(o1["meta"][0])) == 0          # stand_aside
        # pass 2: intent in force, NOT a meta-decision bar
        assert float(o2["flags"][0, 0]) == pytest.approx(0.0)
        assert int(torch.argmax(o2["meta"][0])) == 1          # hunt_long
        assert sig.evidence.policy_prob == pytest.approx(0.8 * 0.7)


class TestAbstention:
    def _mutated(self, emb, key, value):
        out = dict(emb)
        out[key] = emb[key].copy()
        out[key][IDX] = value
        return out

    def test_abstains_on_huge_epistemic(self, emb):
        engine = _engine()
        assert engine.generate("AAPL", IDX, emb) is not None
        assert engine.generate("AAPL", IDX,
                               self._mutated(emb, "epistemic", 100.0)) is None, \
            "the engine must abstain when the model admits ignorance"

    def test_abstains_on_huge_anomaly(self, emb):
        assert _engine().generate(
            "AAPL", IDX, self._mutated(emb, "anomaly", 100.0)) is None, \
            "the engine must abstain in unprecedented regimes"

    def test_conviction_gate_blocks_weak_consensus(self, emb):
        assert _engine(SignalConfig(min_conviction=0.999)).generate(
            "AAPL", IDX, emb) is None
        assert _engine(SignalConfig(min_conviction=0.65),
                       policy=FakePolicy(prob=0.05)).generate(
            "AAPL", IDX, emb) is None, \
            "a low-probability policy vote cannot clear the consensus gate"


# --------------------------------------------------------------------------- #
# risk manager
# --------------------------------------------------------------------------- #

EQ = 100_000.0
DAY = synth_sessions(1)[0]


def _signal(side="long", entry=100.0, stop=95.0, target=110.0,
            size_frac=1.0, conviction=0.9, ticker="AAPL") -> ReversalSignal:
    return ReversalSignal(
        signal_id="S-1", ts=epoch_s(session_bar_time(DAY, 60)), ticker=ticker,
        side=side, conviction=conviction, entry_px=entry, stop_px=stop,
        target_px=target, horizon_bars=30, size_frac=size_frac,
        rationale="test", evidence=None)


def _rm(**overrides) -> RiskManager:
    return RiskManager(RiskConfig(**overrides))


class TestRiskRules:
    def test_normal_approval_structure(self):
        dec = _rm().assess(_signal(), EQ, {}, 0.0)
        assert isinstance(dec, RiskDecision)
        assert dec.approved
        assert dec.qty > 0
        assert dec.adjusted_stop_px > 0 and dec.adjusted_target_px > 0
        assert dec.adjusted_stop_px < _signal().entry_px  # long stop below entry
        assert dec.reasons and all(isinstance(r, str) for r in dec.reasons)

    def test_per_trade_risk_qty_hand_check(self):
        # risk budget = 100k * 0.005 = $500; stop distance = $5 -> 100 shares,
        # comfortably under the 20% notional cap (200 shares at $100).
        dec = _rm().assess(_signal(entry=100.0, stop=95.0), EQ, {}, 0.0)
        assert dec.approved
        assert dec.qty <= 100.0 + 1e-6, \
            f"qty {dec.qty} exceeds the per-trade risk budget (expected <=100)"
        assert dec.qty >= 95.0, \
            f"qty {dec.qty} far below the hand-computed 100 shares"

    def test_max_position_notional_cap(self):
        # $1 stop distance -> risk rule alone would allow 500 shares, but the
        # 20% per-instrument cap allows only 200 at $100.
        dec = _rm().assess(_signal(entry=100.0, stop=99.0), EQ, {}, 0.0)
        assert dec.qty <= 200.0 + 1e-6, \
            f"qty {dec.qty} breaches max_position_frac"

    def test_max_open_positions_rejects(self):
        open_pos = {"NVDA": 0.04, "TSLA": 0.04, "QQQ": 0.04, "IWM": 0.04}
        dec = _rm().assess(_signal(), EQ, open_pos, 0.0)
        assert not dec.approved
        assert dec.qty == 0.0

    def test_gross_exposure_rejects(self):
        dec = _rm().assess(_signal(), EQ, {"NVDA": 0.6, "QQQ": 0.45}, 0.0)
        assert not dec.approved
        assert dec.qty == 0.0

    def test_daily_loss_circuit_breaker_is_sticky(self):
        rm = _rm()
        tripped = rm.assess(_signal(), EQ, {}, -0.03)
        assert not tripped.approved, "-3% day must trip the -2% breaker"
        assert tripped.qty == 0.0

        recovered = rm.assess(_signal(), EQ, {}, 0.0)
        assert not recovered.approved, \
            "circuit breaker must stay latched for the rest of the day"

        rm.reset_day()
        fresh = rm.assess(_signal(), EQ, {}, 0.0)
        assert fresh.approved, "reset_day() must clear the breaker"

    def test_correlation_haircut_monotonicity(self):
        open_pos = {"SPY": 0.10}

        def qty(rho: float | None) -> float:
            corr = None if rho is None else np.array([[1.0, rho], [rho, 1.0]])
            dec = _rm().assess(_signal(entry=100.0, stop=95.0), EQ,
                               open_pos, 0.0, corr=corr)
            assert dec.approved
            return dec.qty

        q_none, q_lo, q_mid, q_hi = qty(None), qty(0.1), qty(0.5), qty(0.9)
        assert q_hi <= q_mid + 1e-9 <= q_lo + 2e-9, \
            f"haircut not monotone in correlation: {q_lo}, {q_mid}, {q_hi}"
        assert q_hi < q_lo, "high correlation must shrink size strictly"
        assert q_lo <= q_none + 1e-9
