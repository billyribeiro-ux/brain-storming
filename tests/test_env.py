"""Layer 3 trading environment tests — the sacred invariants.

Uses the hand-controlled embedding store from ``tests.helpers_stack``:
a deterministic sine+trend price path per session whose final 40 bars are
frozen flat, so fills, stops and end-of-day behavior are all predictable.

The driver locates the env's current bar by matching ``obs['market']``
against the fused array it wrote, so no assumption is made about where the
env starts an episode or how it counts bars internally.
"""

from __future__ import annotations

import numpy as np
import pytest

envmod = pytest.importorskip("aether.decision.env")

from aether.decision.env import EmbeddingStore, TradingEnv
from aether.decision.interfaces import META_ACTIONS, OBS_KEYS, TRADE_ACTIONS, EnvStep
from aether.worldmodel.interfaces import TradeRecord

from tests.helpers_stack import (
    EMBED_DIM,
    SESSION_BARS,
    load_npz,
    make_actions,
    make_embedding_npz_dir,
    rewrite_npz,
    tiny_env_cfg,
)

HOLD, ENTER, EXIT, ADJUST = (TRADE_ACTIONS.index(a) for a in
                             ("hold", "enter", "exit", "adjust"))
STAND, LONG, SHORT = (META_ACTIONS.index(a) for a in
                      ("stand_aside", "hunt_long", "hunt_short"))

META_EVERY = 5
ENTER_MINUTE = 100          # inside the flat zone (bars 80..119 of session 1)

REQUIRED_NPZ_KEYS = ("fused", "aleatoric", "epistemic", "anomaly", "anchor_ts",
                     "session_minute", "close_px", "next_open", "next_high",
                     "next_low", "next_close")

OBS_SHAPES = {"market": (EMBED_DIM,), "unc": (3,), "position": (5,),
              "portfolio": (3,), "clock": (2,), "meta": (len(META_ACTIONS),),
              "flags": (1,)}


# --------------------------------------------------------------------------- #
# Fixtures & driver
# --------------------------------------------------------------------------- #

@pytest.fixture()
def emb_dir(tmp_path):
    return make_embedding_npz_dir(tmp_path, seed=0)


@pytest.fixture()
def aapl(emb_dir):
    return load_npz(emb_dir, "AAPL")


def _cfg(emb_dir, **kw):
    return tiny_env_cfg(emb_dir, meta_every=META_EVERY, **kw)


def _episode_index(env, ticker="AAPL", date_str="2026-01-05") -> int:
    for i, ep in enumerate(env.episodes):
        s = str(ep)
        if ticker in s and date_str in s:
            return i
    pytest.fail(f"no ({ticker}, {date_str}) episode in {list(env.episodes)!r}")


class Driver:
    """Scripted single-env driver that tracks bars, obs, rewards, trades."""

    def __init__(self, env: TradingEnv, arrays: dict):
        self.env = env
        self.fused = np.asarray(arrays["fused"], dtype=np.float64)
        self.session_minute = np.asarray(arrays["session_minute"])
        self.obs = None
        self.obs_log: list[dict] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.records: list[TradeRecord] = []

    def reset(self, episode_ids=None):
        self.obs = self.env.reset(episode_ids)
        self.obs_log.append(self.obs)
        return self.obs

    def bar(self) -> int:
        """Global array index of the currently observed bar."""
        row = np.asarray(self.obs["market"], dtype=np.float64)[0]
        dist = np.linalg.norm(self.fused - row[None, :], axis=1)
        idx = int(np.argmin(dist))
        assert dist[idx] < 1e-2, (
            "obs['market'] does not match any fused row — the contract says "
            "it is the perception fused embedding at bar t")
        return idx

    def minute(self) -> int:
        return int(self.session_minute[self.bar()])

    def step(self, trade=HOLD, meta=LONG, size=1.0, stop=1.0, target=1.0):
        step = self.env.step(make_actions(1, meta=meta, trade=trade,
                                          size=size, stop=stop, target=target))
        assert isinstance(step, EnvStep)
        for info in step.info:
            closed = info.get("trade_closed")
            if closed is None:
                continue
            self.records.extend(closed if isinstance(closed, (list, tuple))
                                else [closed])
        self.obs = step.obs
        self.obs_log.append(step.obs)
        self.rewards.append(float(step.reward[0]))
        self.dones.append(bool(step.done[0]))
        return step

    def run_to_minute(self, minute: int, **kw) -> int:
        for _ in range(3 * SESSION_BARS):
            if self.minute() >= minute:
                return self.bar()
            self.step(trade=HOLD, **kw)
        pytest.fail(f"never reached session minute {minute}")

    def run_to_done(self, max_steps: int = 3 * SESSION_BARS, **kw):
        for _ in range(max_steps):
            step = self.step(**kw)
            if bool(step.done[0]):
                return
        pytest.fail("episode never terminated")


def _driver(emb_dir, arrays, **cfg_kw) -> Driver:
    env = TradingEnv(_cfg(emb_dir, **cfg_kw), n_envs=1)
    d = Driver(env, arrays)
    d.reset([_episode_index(env)])
    return d


def _fee(cfg, notional: float) -> float:
    return notional * cfg.fees_bps / 1e4


# --------------------------------------------------------------------------- #
# EmbeddingStore.load — the npz handshake
# --------------------------------------------------------------------------- #

class TestEmbeddingStore:
    def test_load_returns_contract_arrays(self, emb_dir):
        d = EmbeddingStore.load(str(emb_dir), "AAPL")
        raw = load_npz(emb_dir, "AAPL")
        for key in REQUIRED_NPZ_KEYS:
            assert key in d, f"EmbeddingStore.load missing {key!r}"
        n = d["fused"].shape[0]
        assert d["fused"].shape == (n, EMBED_DIM)
        for key in REQUIRED_NPZ_KEYS[1:]:
            assert np.asarray(d[key]).shape[0] == n, key
        assert np.asarray(d["anchor_ts"]).dtype.kind == "i"
        assert np.asarray(d["session_minute"]).dtype.kind == "i"
        np.testing.assert_allclose(d["fused"], raw["fused"], rtol=1e-6)
        np.testing.assert_allclose(d["close_px"], raw["close_px"], rtol=1e-12)
        np.testing.assert_array_equal(d["anchor_ts"], raw["anchor_ts"])


# --------------------------------------------------------------------------- #
# (d) observation contract
# --------------------------------------------------------------------------- #

class TestObsContract:
    def test_keys_shapes_finiteness(self, emb_dir):
        env = TradingEnv(_cfg(emb_dir), n_envs=2)
        obs = env.reset()
        assert set(obs.keys()) == set(OBS_KEYS)
        for key, shape in OBS_SHAPES.items():
            arr = np.asarray(obs[key])
            assert arr.shape == (2, *shape), f"{key}: {arr.shape}"
            assert np.isfinite(arr.astype(np.float64)).all(), key

        step = env.step(make_actions(2))
        assert isinstance(step, EnvStep)
        assert set(step.obs.keys()) == set(OBS_KEYS)
        assert np.asarray(step.reward).shape == (2,)
        assert np.isfinite(np.asarray(step.reward, dtype=np.float64)).all()
        assert np.asarray(step.done).shape == (2,)
        assert np.asarray(step.done).dtype == np.bool_
        assert len(step.info) == 2
        assert all(isinstance(i, dict) for i in step.info)

    def test_meta_obs_is_one_hot_of_intent(self, emb_dir, aapl):
        d = _driver(emb_dir, aapl)
        m0 = np.asarray(d.obs["meta"])[0]
        assert m0.sum() == pytest.approx(1.0)
        # reset lands on a meta-decision bar (flags=1); choose hunt_long there
        assert float(np.asarray(d.obs["flags"])[0, 0]) == pytest.approx(1.0)
        d.step(meta=LONG, trade=HOLD)
        m1 = np.asarray(d.obs["meta"])[0]
        assert m1.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(m1, np.eye(len(META_ACTIONS))[LONG])

    def test_flags_every_meta_every_bars(self, emb_dir, aapl):
        d = _driver(emb_dir, aapl)
        flags = [float(np.asarray(d.obs["flags"])[0, 0])]
        for _ in range(2 * META_EVERY + 2):
            d.step(trade=HOLD)
            flags.append(float(np.asarray(d.obs["flags"])[0, 0]))
        for bar, f in enumerate(flags):
            expected = 1.0 if bar % META_EVERY == 0 else 0.0
            assert f == pytest.approx(expected), \
                f"flags at episode bar {bar}: {f} (meta_every={META_EVERY})"

    def test_clock_progresses(self, emb_dir, aapl):
        d = _driver(emb_dir, aapl)
        c0 = np.asarray(d.obs["clock"])[0].copy()
        for _ in range(5):
            d.step(trade=HOLD)
        c1 = np.asarray(d.obs["clock"])[0]
        assert c1[0] > c0[0], "session progress must advance"
        assert c1[1] < c0[1], "bars-to-close must shrink"


# --------------------------------------------------------------------------- #
# (a) fill price: next bar's open, slippage against the agent
# --------------------------------------------------------------------------- #

class TestFillMath:
    def test_long_fills_at_next_open_with_slippage(self, emb_dir, aapl):
        cfg = _cfg(emb_dir)
        d = _driver(emb_dir, aapl)
        slip = cfg.slippage_bps / 1e4

        d.run_to_minute(ENTER_MINUTE)
        b_enter = d.bar()
        d.step(trade=ENTER, size=1.0, stop=1.0, target=1.0)
        assert np.asarray(d.obs["position"])[0, 0] > 0, \
            "position obs must show the long after the fill"
        d.step(trade=HOLD)
        b_exit = d.bar()
        d.step(trade=EXIT)
        for _ in range(3):
            if d.records:
                break
            d.step(trade=HOLD)
        assert len(d.records) == 1, f"expected one round-trip, got {d.records}"
        rec = d.records[0]
        assert isinstance(rec, TradeRecord)
        assert rec.ticker == "AAPL"
        assert rec.side == "long"
        assert rec.exit_reason == "policy_exit"
        assert rec.qty > 0
        assert rec.entry_px == pytest.approx(
            aapl["next_open"][b_enter] * (1.0 + slip), rel=1e-9), \
            "long entry must fill at next_open * (1 + slippage)"
        assert rec.exit_px == pytest.approx(
            aapl["next_open"][b_exit] * (1.0 - slip), rel=1e-9), \
            "long exit must fill at next_open * (1 - slippage)"
        expected_fees = _fee(cfg, (rec.entry_px + rec.exit_px) * rec.qty)
        assert rec.fees == pytest.approx(expected_fees, rel=1e-6)
        assert rec.pnl == pytest.approx(
            (rec.exit_px - rec.entry_px) * rec.qty - expected_fees,
            rel=1e-6, abs=1e-6)


# --------------------------------------------------------------------------- #
# (b) stop and target inside one bar -> the stop fills (conservatism)
# --------------------------------------------------------------------------- #

class TestStopBeforeTarget:
    def test_wide_bar_touching_both_levels_fills_stop(self, emb_dir):
        def widen(arrays):
            sl = slice(ENTER_MINUTE, ENTER_MINUTE + 7)
            arrays["next_high"][sl] = arrays["close_px"][sl] * 1.10
            arrays["next_low"][sl] = arrays["close_px"][sl] * 0.90

        arrays = rewrite_npz(emb_dir, "AAPL", widen)
        d = _driver(emb_dir, arrays)
        d.run_to_minute(ENTER_MINUTE)
        d.step(trade=ENTER, size=1.0, stop=1.0, target=1.0)
        for _ in range(10):
            if d.records:
                break
            d.step(trade=HOLD)
        assert d.records, "stop/target bar never closed the trade"
        rec = d.records[0]
        assert rec.exit_reason == "stop", \
            "when stop and target are both touched in one bar, stop must win"
        assert rec.exit_px <= rec.stop_px * (1.0 + 1e-9), \
            "long stop fill can never improve on the stop price"
        assert rec.pnl < 0


# --------------------------------------------------------------------------- #
# (c) end-of-day force close
# --------------------------------------------------------------------------- #

class TestEndOfDay:
    def test_open_position_force_closes_with_exact_accounting(self, emb_dir,
                                                              aapl):
        cfg = _cfg(emb_dir)
        d = _driver(emb_dir, aapl)
        slip = cfg.slippage_bps / 1e4

        d.run_to_minute(ENTER_MINUTE)
        b_enter = d.bar()
        d.step(trade=ENTER, size=1.0, stop=1.0, target=1.0)
        d.run_to_done()

        assert len(d.records) == 1, (
            "flat-zone entry must survive untouched until the session ends; "
            f"got {[r.exit_reason for r in d.records]}")
        rec = d.records[0]
        assert rec.exit_reason == "eod"
        assert rec.entry_px == pytest.approx(
            aapl["next_open"][b_enter] * (1.0 + slip), rel=1e-9)
        assert rec.exit_px > 0
        assert rec.exit_ts >= rec.entry_ts
        expected_fees = _fee(cfg, (rec.entry_px + rec.exit_px) * rec.qty)
        assert rec.fees == pytest.approx(expected_fees, rel=1e-6)
        assert rec.pnl == pytest.approx(
            (rec.exit_px - rec.entry_px) * rec.qty - expected_fees,
            rel=1e-6, abs=1e-6), "eod pnl must include fees on both sides"
        assert d.dones[-1] is True


# --------------------------------------------------------------------------- #
# (e) accounting identity: cash + position value == equity, every step
# --------------------------------------------------------------------------- #

class TestEquityIdentity:
    def test_cash_frac_plus_position_frac_is_one(self, emb_dir, aapl):
        """cash + qty*close == equity  <=>  cash/equity + qty*close/equity == 1,
        i.e. portfolio[0] + position[0] == 1 under the contract's obs layout."""
        d = _driver(emb_dir, aapl)
        assert np.asarray(d.obs["portfolio"])[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert np.asarray(d.obs["position"])[0, 0] == pytest.approx(0.0, abs=1e-9)

        script = ([dict(trade=HOLD)] * 3 + [dict(trade=ENTER, size=0.7)]
                  + [dict(trade=HOLD)] * 6 + [dict(trade=EXIT)]
                  + [dict(trade=HOLD)] * 4)
        for kw in script:
            d.step(**kw)
        for i, obs in enumerate(d.obs_log):
            cash_frac = float(np.asarray(obs["portfolio"])[0, 0])
            pos_frac = float(np.asarray(obs["position"])[0, 0])
            assert cash_frac + pos_frac == pytest.approx(1.0, abs=1e-4), \
                f"equity identity broken at step {i}"


# --------------------------------------------------------------------------- #
# (f) NO LOOK-AHEAD: mutating the future changes nothing about the present
# --------------------------------------------------------------------------- #

T_MUT = 40


class TestNoLookAhead:
    def test_future_mutation_leaves_present_untouched(self, tmp_path):
        dir_a = make_embedding_npz_dir(tmp_path / "a", seed=5)
        dir_b = make_embedding_npz_dir(tmp_path / "b", seed=5)

        def corrupt_future(arrays):
            sl = slice(T_MUT + 1, None)
            rng = np.random.default_rng(999)
            arrays["fused"][sl] = rng.standard_normal(
                arrays["fused"][sl].shape).astype(np.float32)
            for k in ("aleatoric", "epistemic", "anomaly"):
                arrays[k][sl] = (arrays[k][sl] + 1.7).astype(arrays[k].dtype)
            for k in ("close_px", "next_open", "next_high", "next_low",
                      "next_close"):
                arrays[k][sl] = arrays[k][sl] * 3.0

        for ticker in ("AAPL", "SPY"):
            rewrite_npz(dir_b, ticker, corrupt_future)

        arrays_a = load_npz(dir_a, "AAPL")
        env_a = TradingEnv(_cfg(dir_a), n_envs=1)
        env_b = TradingEnv(_cfg(dir_b), n_envs=1)
        drv_a = Driver(env_a, arrays_a)
        drv_b = Driver(env_b, arrays_a)      # match against the clean fused
        drv_a.reset([_episode_index(env_a)])
        drv_b.reset([_episode_index(env_b)])

        b0 = drv_a.bar()
        n_steps = T_MUT - 5 - b0
        assert n_steps >= 10, f"episode starts too late (bar {b0}) to test"

        script = [dict(trade=HOLD)] * n_steps
        script[3] = dict(trade=ENTER, size=0.8)
        script[12] = dict(trade=EXIT)
        if n_steps > 20:
            script[20] = dict(trade=ENTER, size=0.5)

        for k in OBS_KEYS:
            np.testing.assert_array_equal(
                np.asarray(drv_a.obs[k]), np.asarray(drv_b.obs[k]),
                err_msg=f"reset obs[{k}] differs")
        for s, kw in enumerate(script):
            step_a = drv_a.step(**kw)
            step_b = drv_b.step(**kw)
            for k in OBS_KEYS:
                np.testing.assert_array_equal(
                    np.asarray(step_a.obs[k]), np.asarray(step_b.obs[k]),
                    err_msg=(f"LOOK-AHEAD LEAK: obs[{k}] at step {s} depends "
                             f"on data beyond bar {T_MUT}"))
            np.testing.assert_array_equal(step_a.reward, step_b.reward,
                                          err_msg=f"reward differs at step {s}")
            np.testing.assert_array_equal(step_a.done, step_b.done)


# --------------------------------------------------------------------------- #
# (g) enter while stand_aside is a no-op
# --------------------------------------------------------------------------- #

class TestStandAside:
    def test_enter_under_stand_aside_does_nothing(self, emb_dir, aapl):
        d = _driver(emb_dir, aapl)
        for _ in range(2 * META_EVERY):
            d.step(meta=STAND, trade=ENTER, size=1.0)
        assert not d.records, "stand_aside must veto entries"
        for i, obs in enumerate(d.obs_log):
            assert np.asarray(obs["position"])[0, 0] == pytest.approx(0.0, abs=1e-9), \
                f"a position appeared under stand_aside at step {i}"
        assert np.asarray(d.obs["portfolio"])[0, 0] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# (h) adjust may tighten the stop, never widen it
# --------------------------------------------------------------------------- #

class TestAdjustStop:
    def test_stop_tightens_but_never_widens(self, emb_dir, aapl):
        d = _driver(emb_dir, aapl)
        d.run_to_minute(ENTER_MINUTE)
        d.step(trade=ENTER, size=1.0, stop=1.0, target=1.0)   # widest stop
        s0 = float(np.asarray(d.obs["position"])[0, 4])
        assert s0 > 0, "stop distance obs must be positive while in a trade"

        d.step(trade=ADJUST, stop=0.3, target=1.0)            # tighten
        s1 = float(np.asarray(d.obs["position"])[0, 4])
        assert s1 < 0.75 * s0, f"adjust failed to tighten the stop: {s0} -> {s1}"

        d.step(trade=HOLD)
        d.step(trade=ADJUST, stop=1.0, target=1.0)            # try to widen
        s2 = float(np.asarray(d.obs["position"])[0, 4])
        assert s2 <= s1 + 0.25 * (s0 - s1), \
            f"adjust widened the stop: {s1} -> {s2} (initial was {s0})"
