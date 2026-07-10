"""High-conviction reversal signal synthesis (Layer 5).

The SignalEngine turns everything the earlier layers learned into a single,
auditable artifact: a :class:`~aether.execution.interfaces.ReversalSignal`
that either clears a consensus gate across INDEPENDENT evidence sources or
does not exist at all. There is no "weak signal" output — abstention is the
default state and every abstention is explainable (:meth:`explain_abstention`).

Consensus design
----------------
Three evidence sources are polled, each producing a value in [0, 1]:

* ``policy_prob``            — the trained policy's own modal action
                               probability (weight **2.0**: it is the only
                               source trained end-to-end on the task),
* ``imagination_agreement``  — fraction of world-model rollouts whose decoded
                               embedding drifts the signal's way (weight 1.0),
* ``analog_winrate``         — 30-minute outcome agreement among the nearest
                               historical analogs in memory (weight 1.0).

Conviction is the WEIGHTED GEOMETRIC MEAN of the available sources — a
geometric mean because consensus should be vetoed by any single source near
zero, not averaged away — times a soft aleatoric penalty
``(1 - min(1, aleatoric/5))``: when perception says the moment is inherently
noisy, even unanimous evidence is worth less.

Every component is OPTIONAL. A missing component's evidence source simply
drops out of the consensus and the remaining weights are renormalized
(``exp(Σ w·ln v / Σ w)`` over available sources only), so a bare engine with
just a policy degrades gracefully instead of failing. The EvidenceBundle
records unavailable sources as the neutral no-information likelihood 0.5
(the bundle contract pins every source to [0, 1]); the rationale names the
missing sources explicitly, so the audit trail still distinguishes
"source measured 0.5" from "source absent".

Honesty notes (read before trusting a rationale)
------------------------------------------------
* **Imagination is a proxy.** Rollouts are scored by projecting the decoded
  embedding delta onto the embedding-space direction that recently co-moved
  with rising closes. That direction is estimated, and embeddings are not
  prices: agreement means "the world model expects the state to keep drifting
  the way it drifts when price rises", NOT a price forecast.
* **Entry is a proxy.** ``entry_px`` is the anchor close standing in for the
  next bar's open (the earliest fill the environment allows). The backtester
  applies slippage on top.
* **Range fallback.** Stop/target distances scale the policy's Beta means by
  the mean high-low range of the trailing 15 bars, reconstructed from the
  next-bar high/low arrays in the embedding store. When those arrays are
  absent, 0.3% of the entry price is used — a deliberately blunt fallback,
  flagged in the rationale.
* **No policy attached?** The directional intent then falls back to fading
  the trailing move (a documented reversal prior, not a learned view) and
  neutral 0.5 Beta means size the levels. This exists so the engine stays
  usable in partial deployments; a production signal should always carry
  policy evidence.

The rationale string is TEMPLATE-COMPOSED exclusively from the numbers in the
EvidenceBundle — it never asserts anything that was not computed.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from aether.decision.interfaces import META_ACTIONS, TRADE_ACTIONS
from aether.execution.interfaces import EvidenceBundle, ReversalSignal, SignalConfig
from aether.perception.interfaces import FEATURE_COLUMNS_BAR, PerceptionBatch
from aether.utils.logging import get_logger
from aether.worldmodel.interfaces import LatentState

logger = get_logger("aether.signals")

#: Consensus weights per evidence source. The policy carries double weight —
#: it is the only source optimized end-to-end for the trading objective; the
#: world model and memory are corroborating witnesses, not principals.
_EVIDENCE_WEIGHTS: dict[str, float] = {
    "policy_prob": 2.0,
    "imagination_agreement": 1.0,
    "analog_winrate": 1.0,
}

#: Aleatoric variance at which the soft penalty saturates (conviction -> 0).
_ALEATORIC_SCALE: float = 5.0

#: Trailing fused-embedding window fed to dynamics.observe (bars).
_OBSERVE_WINDOW: int = 64

#: Trailing bars over which the realized high-low range is averaged.
_RANGE_BARS: int = 15
#: Fraction of entry price used as the range when high/low arrays are absent.
_RANGE_FALLBACK_FRAC: float = 0.003

#: Floor for evidence values inside the geometric mean: ln(0) would collapse
#: conviction to -inf; 1e-6 keeps the veto behavior (conviction ~0) while
#: staying numerically defined.
_EVIDENCE_EPS: float = 1e-6

#: Neutral no-information likelihood recorded in the EvidenceBundle for
#: sources that were absent from the consensus (contract pins [0, 1]).
_NEUTRAL_EVIDENCE: float = 0.5


def _to_1d(x) -> np.ndarray:
    """Coerce a tensor/array/scalar from a policy head to a flat float array."""
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


class SignalEngine:
    """Consensus-gated reversal signal generator.

    Parameters
    ----------
    policy:
        Decision-core policy exposing ``mode(obs)``. Preferred shape: the
        flat dict of ``HierarchicalPolicy.mode`` — greedy actions plus
        ``"meta_probs"`` [B, len(META_ACTIONS)], ``"trade_probs"``
        [B, len(TRADE_ACTIONS)] (softmax probabilities) and ``"size_mean"``,
        ``"stop_mean"``, ``"target_mean"`` [B] (Beta means in (0, 1]).
        ``act``-style tuples ``(actions, logprobs, value, carry)`` from
        duck-typed policies are also accepted (see :meth:`_policy_view`).
    dynamics:
        ``worldmodel.interfaces.LatentDynamics`` implementation
        (``observe`` / ``imagine``).
    memory:
        ``MemoryBankProtocol`` implementation (``query``).
    causal_snapshot:
        A fitted ``CausalGraphSnapshot`` (``top_drivers``).
    perception:
        The perception model, used ONLY for saliency attribution (one extra
        forward+backward per signal, and only when ``generate`` receives the
        matching ``batch``).
    cfg:
        :class:`SignalConfig` thresholds; defaults used when omitted.

    Any subset may be ``None`` — see the module docstring for the degradation
    semantics.
    """

    def __init__(self, policy=None, dynamics=None, memory=None,
                 causal_snapshot=None, perception=None,
                 cfg: Optional[SignalConfig] = None) -> None:
        self.policy = policy
        self.dynamics = dynamics
        self.memory = memory
        self.causal_snapshot = causal_snapshot
        self.perception = perception
        self.cfg = cfg or SignalConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, ticker: str, idx: int, emb: dict[str, np.ndarray],
                 equity_hint: float = 100_000,
                 batch: Optional[PerceptionBatch] = None,
                 ) -> Optional[ReversalSignal]:
        """Generate a signal at anchor bar ``idx``, or ``None`` (abstain).

        Parameters
        ----------
        ticker:
            Instrument alias (matches the embedding store partition).
        idx:
            Index into the embedding arrays. Must have a valid NEXT bar
            (``idx + 1`` in range) because fills happen at t+1; otherwise
            the engine abstains.
        emb:
            ``EmbeddingStore.load`` output — dict of aligned numpy arrays
            (``fused``, ``aleatoric``, ``epistemic``, ``anomaly``,
            ``anchor_ts``, ``session_minute``, ``close_px`` plus next-bar
            OHL arrays).
        equity_hint:
            Reserved. Signals express size as a FRACTION; dollar sizing is
            the risk layer's job. Kept in the signature so callers can wire
            equity through without a later interface break.
        batch:
            Optional :class:`PerceptionBatch` for the SAME anchor moment
            (row 0). Only used for saliency attribution when a perception
            model is attached; omit to skip saliency.
        """
        signal, _reason = self._evaluate(ticker, idx, emb, batch)
        return signal

    def explain_abstention(self, ticker: str, idx: int,
                           emb: dict[str, np.ndarray]) -> str:
        """Human-readable reason the engine abstains at ``idx`` (dashboard).

        Re-runs the evaluation without saliency and reports the FIRST gate
        that failed, with its margin. If nothing fails, says so explicitly.
        """
        signal, reason = self._evaluate(ticker, idx, emb, batch=None)
        if signal is not None:
            return (f"no abstention: a {signal.side} signal would be generated "
                    f"(conviction {signal.conviction:.3f} ≥ "
                    f"{self.cfg.min_conviction:.2f})")
        return reason

    # ------------------------------------------------------------------ #
    # Core evaluation (shared by generate / explain_abstention)
    # ------------------------------------------------------------------ #

    def _evaluate(self, ticker: str, idx: int, emb: dict[str, np.ndarray],
                  batch: Optional[PerceptionBatch],
                  ) -> tuple[Optional[ReversalSignal], str]:
        """Returns ``(signal, "")`` or ``(None, abstention_reason)``."""
        cfg = self.cfg
        fused_all = np.asarray(emb["fused"], dtype=np.float64)
        close_all = np.asarray(emb["close_px"], dtype=np.float64)
        n = len(close_all)

        # --- 0. Anchor validity: a fill needs bar idx+1 to exist. ----------
        if not (0 <= idx < n - 1):
            return None, (f"invalid anchor: idx={idx} has no valid next bar "
                          f"(store holds {n} bars; last usable anchor is {n - 2})")
        close_px = float(close_all[idx])
        if not np.isfinite(close_px) or close_px <= 0:
            return None, f"invalid anchor: close_px[{idx}]={close_px!r} is not a positive price"

        aleatoric = float(emb["aleatoric"][idx])
        epistemic = float(emb["epistemic"][idx])
        anomaly = float(emb["anomaly"][idx])

        # --- 1. Hard uncertainty gates: the model admits it doesn't know. --
        # These come FIRST: no amount of evidence agreement rescues a moment
        # perception itself cannot vouch for.
        if epistemic > cfg.max_epistemic:
            return None, (f"epistemic gate: {epistemic:.3f} exceeds "
                          f"max_epistemic {cfg.max_epistemic:.2f} by "
                          f"{epistemic - cfg.max_epistemic:.3f} (model ignorance)")
        if anomaly > cfg.max_anomaly:
            return None, (f"anomaly gate: {anomaly:.3f} exceeds max_anomaly "
                          f"{cfg.max_anomaly:.2f} by "
                          f"{anomaly - cfg.max_anomaly:.3f} (unprecedented regime)")

        # --- 2. Policy evidence: intent, enter-probability, level means. ---
        w0 = max(0, idx - (_OBSERVE_WINDOW - 1))
        policy_prob: Optional[float] = None
        intent_name = ""
        p_intent = p_enter = float("nan")
        if self.policy is not None:
            obs = self._build_obs(emb, fused_all, idx)
            with torch.no_grad():
                view = self._policy_view(self.policy.mode(obs))
            intent = int(view["intent"])
            intent_name = META_ACTIONS[intent]
            p_intent = float(view["p_intent"])
            if intent_name == "stand_aside":
                return None, (f"policy intent: stand_aside is the modal "
                              f"intent (p={p_intent:.3f})")
            side = "long" if intent_name == "hunt_long" else "short"
            if view["p_enter"] is None:
                return None, (f"policy trade appetite: the modal trade "
                              f"action is "
                              f"{TRADE_ACTIONS[int(view['trade'])]!r}, not "
                              f"'enter' — the policy does not want a fill "
                              f"here")
            p_enter = float(view["p_enter"])
            policy_prob = p_intent * p_enter
            size_mean = float(view["size_mean"])
            stop_mean = float(view["stop_mean"])
            target_mean = float(view["target_mean"])
        else:
            # Documented fallback (module docstring): fade the trailing move.
            trailing_chg = close_px - float(close_all[w0])
            if trailing_chg == 0.0 or idx == w0:
                return None, ("no directional view: policy absent and the "
                              "trailing move is flat, so the reversal prior "
                              "has nothing to fade")
            side = "long" if trailing_chg < 0 else "short"
            size_mean = stop_mean = target_mean = 0.5  # neutral Beta means

        # --- 3. Imagination evidence (world-model rollouts). ---------------
        imagination = self._imagination_agreement(fused_all, close_all, idx, w0, side)

        # --- 4. Analog evidence (memory outcomes). --------------------------
        analog_winrate, analogs_summary = self._analog_evidence(fused_all[idx], side)

        # --- 5. Consensus: renormalized weighted geometric mean. ------------
        available: dict[str, float] = {}
        if policy_prob is not None:
            available["policy_prob"] = policy_prob
        if imagination is not None:
            available["imagination_agreement"] = imagination
        if analog_winrate is not None:
            available["analog_winrate"] = analog_winrate
        if not available:
            return None, ("no evidence: policy, dynamics and memory are all "
                          "absent or returned nothing — consensus is undefined")
        total_w = sum(_EVIDENCE_WEIGHTS[k] for k in available)
        log_sum = sum(
            _EVIDENCE_WEIGHTS[k] * np.log(max(v, _EVIDENCE_EPS))
            for k, v in available.items()
        )
        aleatoric_penalty = 1.0 - min(1.0, aleatoric / _ALEATORIC_SCALE)
        conviction = float(np.exp(log_sum / total_w) * aleatoric_penalty)
        conviction = min(1.0, max(0.0, conviction))
        if conviction < cfg.min_conviction:
            parts = ", ".join(f"{k}={v:.3f}" for k, v in available.items())
            return None, (f"conviction gate: {conviction:.3f} falls short of "
                          f"min_conviction {cfg.min_conviction:.2f} by "
                          f"{cfg.min_conviction - conviction:.3f} "
                          f"({parts}; aleatoric penalty x{aleatoric_penalty:.3f})")

        # --- 6. Levels. ------------------------------------------------------
        entry_px = close_px  # next-open proxy (see module docstring)
        rng, rng_fallback = self._recent_range(emb, idx, entry_px)
        sign = 1.0 if side == "long" else -1.0
        stop_px = entry_px - sign * stop_mean * rng
        target_px = entry_px + sign * target_mean * rng
        size_frac = float(np.clip(size_mean * conviction, 0.0, 1.0))

        # --- 7. Attribution extras (saliency, causal drivers). ---------------
        saliency = self._saliency(batch)
        causal_drivers = self._causal_drivers(ticker)

        # Absent sources are recorded as the neutral 0.5 likelihood (the
        # bundle contract pins [0, 1]); the rationale names them missing.
        evidence = EvidenceBundle(
            policy_prob=(float(policy_prob) if policy_prob is not None
                         else _NEUTRAL_EVIDENCE),
            imagination_agreement=(float(imagination) if imagination is not None
                                   else _NEUTRAL_EVIDENCE),
            analog_winrate=(float(analog_winrate) if analog_winrate is not None
                            else _NEUTRAL_EVIDENCE),
            aleatoric=aleatoric,
            epistemic=epistemic,
            anomaly=anomaly,
            saliency=saliency,
            causal_drivers=causal_drivers,
            analogs_summary=analogs_summary,
        )
        rationale = self._compose_rationale(
            ticker=ticker, side=side, conviction=conviction,
            available=available, intent_name=intent_name,
            p_intent=p_intent, p_enter=p_enter,
            analogs_summary=analogs_summary,
            aleatoric=aleatoric, epistemic=epistemic, anomaly=anomaly,
            aleatoric_penalty=aleatoric_penalty,
            entry_px=entry_px, stop_px=stop_px, target_px=target_px,
            stop_mean=stop_mean, target_mean=target_mean,
            size_mean=size_mean, size_frac=size_frac,
            rng=rng, rng_fallback=rng_fallback,
        )

        anchor_ts = int(emb["anchor_ts"][idx])
        signal = ReversalSignal(
            signal_id=f"{ticker}-{anchor_ts}",
            ts=anchor_ts,
            ticker=ticker,
            side=side,
            conviction=conviction,
            entry_px=entry_px,
            stop_px=stop_px,
            target_px=target_px,
            horizon_bars=cfg.horizon_bars,
            size_frac=size_frac,
            rationale=rationale,
            evidence=evidence,
        )
        logger.info("signal %s: %s conviction=%.3f entry=%.4f",
                    signal.signal_id, side, conviction, entry_px)
        return signal, ""

    # ------------------------------------------------------------------ #
    # Evidence gathering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _policy_view(mode_out) -> dict:
        """Normalize a policy's ``mode`` output to one evidence view.

        Accepted shapes (documented reconciliation across builders):

        * the rich FLAT dict of ``HierarchicalPolicy.mode`` — greedy
          actions plus ``meta_probs`` / ``trade_probs`` /
          ``size_mean`` / ``stop_mean`` / ``target_mean``. The intent is
          the meta argmax and ``p_enter`` is read off ``trade_probs``
          regardless of the modal trade action;
        * an ``act``-style tuple ``(actions, logprobs, value, carry)``
          from duck-typed policies. Only the modal action and its own
          probability are knowable here, so ``p_enter`` is ``None``
          (unknown) unless the modal trade action IS ``enter`` — the
          caller abstains on ``None`` (conservative: no invented
          probabilities).

        Returns ``{"intent", "p_intent", "trade", "p_enter", "size_mean",
        "stop_mean", "target_mean"}`` with row 0 of each batch.
        """
        enter_idx = TRADE_ACTIONS.index("enter")
        if isinstance(mode_out, dict):
            actions = mode_out
            meta_probs = (_to_1d(mode_out["meta_probs"])
                          if "meta_probs" in mode_out else None)
            trade_probs = (_to_1d(mode_out["trade_probs"])
                           if "trade_probs" in mode_out else None)
            intent = (int(np.argmax(meta_probs)) if meta_probs is not None
                      else int(_to_1d(actions["meta"])[0]))
            trade = (int(np.argmax(trade_probs)) if trade_probs is not None
                     else int(_to_1d(actions["trade"])[0]))
            # A probability-less dict gets the neutral 0.5 likelihood —
            # a deterministic action report carries no confidence and must
            # not masquerade as certainty.
            p_intent = (float(meta_probs[intent]) if meta_probs is not None
                        else _NEUTRAL_EVIDENCE)
            p_enter = (float(trade_probs[enter_idx])
                       if trade_probs is not None else _NEUTRAL_EVIDENCE)
        else:
            seq = list(mode_out)
            dicts = [x for x in seq if isinstance(x, dict) and "trade" in x]
            if not dicts:
                raise TypeError(
                    f"policy.mode returned {type(mode_out).__name__} without "
                    "an actions dict — cannot extract policy evidence")
            actions = dicts[0]
            logprobs = dicts[1] if len(dicts) > 1 else None
            intent = int(_to_1d(actions["meta"])[0])
            trade = int(_to_1d(actions["trade"])[0])

            def _prob(key: str) -> float:
                if logprobs is None or key not in logprobs:
                    return _NEUTRAL_EVIDENCE
                return float(np.clip(np.exp(_to_1d(logprobs[key])[0]),
                                     0.0, 1.0))

            p_intent = _prob("meta")
            p_enter = _prob("trade") if trade == enter_idx else None

        def _mean(key: str) -> float:
            for name in (f"{key}_mean", key):
                if name in actions:
                    return float(_to_1d(actions[name])[0])
            return _NEUTRAL_EVIDENCE  # neutral Beta mean

        return {
            "intent": intent, "p_intent": p_intent,
            "trade": trade, "p_enter": p_enter,
            "size_mean": _mean("size"), "stop_mean": _mean("stop"),
            "target_mean": _mean("target"),
        }

    def _build_obs(self, emb: dict, fused_all: np.ndarray, idx: int,
                   ) -> dict[str, Tensor]:
        """Observation dict exactly per ``decision.interfaces.OBS_KEYS``.

        The book is FLAT (zero position, all-cash portfolio), ``flags`` is
        1.0 so the policy treats this as a meta-decision bar, and ``meta``
        is the stand_aside one-hot: we ask the meta head to decide from a
        neutral stance rather than nudging it with a prior intent.
        """
        minute = float(emb["session_minute"][idx]) if "session_minute" in emb else 195.0
        meta = np.zeros(len(META_ACTIONS), dtype=np.float32)
        meta[META_ACTIONS.index("stand_aside")] = 1.0
        obs_np: dict[str, np.ndarray] = {
            "market": fused_all[idx].astype(np.float32),
            "unc": np.array([emb["aleatoric"][idx], emb["epistemic"][idx],
                             emb["anomaly"][idx]], dtype=np.float32),
            "position": np.zeros(5, dtype=np.float32),          # flat
            "portfolio": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # all cash
            "clock": np.array([minute / 390.0, (390.0 - minute) / 390.0],
                              dtype=np.float32),
            "meta": meta,
            "flags": np.ones(1, dtype=np.float32),              # meta head decides
        }
        return {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs_np.items()}

    def _imagination_agreement(self, fused_all: np.ndarray,
                               close_all: np.ndarray, idx: int, w0: int,
                               side: str) -> Optional[float]:
        """Fraction of rollouts whose decoded drift favors ``side``.

        The "recent up-move direction" is estimated from the trailing window
        as the mean fused-embedding delta SIGNED by the concurrent close
        change: the embedding-space direction that co-moved with rising
        prices. Each rollout's decoded end-minus-start delta is projected on
        it; positive projection favors long, negative favors short. This is
        a decoded-latent PROXY for price direction, not a price forecast —
        the rationale says so verbatim.
        """
        if self.dynamics is None or idx - w0 < 1:
            return None
        seg_f = fused_all[w0:idx + 1]                       # [T, E]
        seg_c = close_all[w0:idx + 1]                       # [T]
        deltas = np.diff(seg_f, axis=0)                     # [T-1, E]
        signs = np.sign(np.diff(seg_c))[:, None]            # [T-1, 1]
        up_dir = (deltas * signs).mean(axis=0)
        norm = float(np.linalg.norm(up_dir))
        if norm < 1e-12:
            return None  # no resolvable direction -> source drops out
        up_dir = up_dir / norm

        with torch.no_grad():
            embeds = torch.as_tensor(seg_f, dtype=torch.float32).unsqueeze(0)
            result = self.dynamics.observe(embeds)          # ObserveResult
            last = LatentState(deter=result.states.deter[:, -1],
                               stoch=result.states.stoch[:, -1])
            rollout = self.dynamics.imagine(
                last, horizon=self.cfg.horizon_bars, n=self.cfg.n_imagination)
            traj = rollout.embeds[:, 0]                      # [N, H, E]
            drift = (traj[:, -1, :] - traj[:, 0, :]).cpu().numpy()  # [N, E]
        proj = drift @ up_dir                                # [N]
        favorable = proj > 0 if side == "long" else proj < 0
        return float(np.mean(favorable))

    def _analog_evidence(self, key: np.ndarray, side: str,
                         ) -> tuple[Optional[float], list[dict]]:
        """30-minute outcome agreement among memory analogs.

        A win is a strictly favorable ``fwd_ret_30m`` for the side (zero
        counts against — conservative). Analogs without the outcome key are
        excluded from the rate but still listed in the summary.
        """
        if self.memory is None:
            return None, []
        analogs = self.memory.query(np.asarray(key, dtype=np.float32),
                                    k=self.cfg.memory_k)
        summary = [{
            "ticker": a.ticker,
            "anchor_ts": int(a.anchor_ts),
            "similarity": round(float(a.similarity), 4),
            "fwd_ret_30m": (float(a.outcome["fwd_ret_30m"])
                            if "fwd_ret_30m" in a.outcome else None),
        } for a in analogs]
        rets = [s["fwd_ret_30m"] for s in summary
                if s["fwd_ret_30m"] is not None and np.isfinite(s["fwd_ret_30m"])]
        if not rets:
            return None, summary
        wins = sum(1 for r in rets if (r > 0) == (side == "long") and r != 0)
        return wins / len(rets), summary

    def _recent_range(self, emb: dict, idx: int, entry_px: float,
                      ) -> tuple[float, bool]:
        """Mean high-low range of the trailing ``_RANGE_BARS`` bars.

        The store carries NEXT-bar OHL under the canonical EmbeddingStore
        keys (``next_high`` / ``next_low``), so bar ``t``'s high/low live at
        index ``t - 1``: the trailing window ending at the anchor is the
        slice ``[idx - _RANGE_BARS, idx)``. Returns ``(range, used_fallback)``
        — the fallback is ``_RANGE_FALLBACK_FRAC`` of the entry price when
        the arrays are missing or degenerate.
        """
        highs = np.asarray(emb["next_high"]) if "next_high" in emb else None
        lows = np.asarray(emb["next_low"]) if "next_low" in emb else None
        if highs is not None and lows is not None:
            lo_i = max(0, idx - _RANGE_BARS)
            spans = (np.asarray(highs[lo_i:idx], dtype=np.float64)
                     - np.asarray(lows[lo_i:idx], dtype=np.float64))
            spans = spans[np.isfinite(spans) & (spans > 0)]
            if len(spans) > 0:
                return float(spans.mean()), False
        return _RANGE_FALLBACK_FRAC * entry_px, True

    def _saliency(self, batch: Optional[PerceptionBatch]) -> list[dict]:
        """Grad×input attribution of ‖fused‖ onto the 1-minute features.

        One forward+backward on the provided batch (row 0 must be the anchor
        moment). Scores are grad×input summed over time per feature column,
        ranked by magnitude, signed values kept. Skipped (empty list) when
        either the perception model or the batch is absent.
        """
        if self.perception is None or batch is None:
            return []
        model = self.perception
        was_training = model.training
        model.eval()  # freeze anomaly EMA / dropout during attribution
        try:
            x = batch.bars_1m.detach().clone().requires_grad_(True)
            probe = dataclasses.replace(batch, bars_1m=x)
            with torch.enable_grad():
                out = model(probe)
                out.fused[0].norm().backward()
            gxi = (x.grad[0] * x.detach()[0]).sum(dim=0)     # [F] over time
            scores = gxi.cpu().numpy()
        finally:
            if was_training:
                model.train()
        order = np.argsort(-np.abs(scores))[:5]
        return [{"name": FEATURE_COLUMNS_BAR[int(i)],
                 "score": float(scores[int(i)])} for i in order]

    def _causal_drivers(self, ticker: str) -> list[dict]:
        """Top causal edges into this ticker's 1-minute return, as dicts."""
        if self.causal_snapshot is None:
            return []
        edges = self.causal_snapshot.top_drivers(f"{ticker}.ret_1m", 5)
        return [dataclasses.asdict(e) for e in edges]

    # ------------------------------------------------------------------ #
    # Rationale
    # ------------------------------------------------------------------ #

    def _compose_rationale(self, *, ticker: str, side: str, conviction: float,
                           available: dict[str, float], intent_name: str,
                           p_intent: float, p_enter: float,
                           analogs_summary: list[dict],
                           aleatoric: float, epistemic: float, anomaly: float,
                           aleatoric_penalty: float,
                           entry_px: float, stop_px: float, target_px: float,
                           stop_mean: float, target_mean: float,
                           size_mean: float, size_frac: float,
                           rng: float, rng_fallback: bool) -> str:
        """4-8 sentence explanation assembled ONLY from computed evidence.

        Template composition, not generation: every number below is a value
        that participated in the decision; unavailable sources are named as
        unavailable rather than papered over.
        """
        cfg = self.cfg
        sentences: list[str] = [
            (f"{side.capitalize()} reversal signal on {ticker}: consensus "
             f"conviction {conviction:.3f} clears the {cfg.min_conviction:.2f} "
             f"gate with margin +{conviction - cfg.min_conviction:.3f}.")
        ]
        if "policy_prob" in available:
            sentences.append(
                f"The policy's modal intent is {intent_name} "
                f"(p={p_intent:.3f}) with enter-probability {p_enter:.3f}, "
                f"giving policy evidence {available['policy_prob']:.3f} at "
                f"consensus weight {_EVIDENCE_WEIGHTS['policy_prob']:.0f}.")
        else:
            sentences.append(
                f"No policy was attached, so the {side} side comes from "
                f"fading the trailing move (documented reversal prior) and "
                f"neutral 0.5 means size the levels.")
        if "imagination_agreement" in available:
            frac = available["imagination_agreement"]
            sentences.append(
                f"{round(frac * cfg.n_imagination)} of {cfg.n_imagination} "
                f"imagination rollouts ({frac:.0%}) drift favorably for the "
                f"{side} side over {cfg.horizon_bars} bars — a decoded-latent "
                f"proxy for direction, not a price forecast.")
        if "analog_winrate" in available:
            n_rel = sum(1 for s in analogs_summary if s["fwd_ret_30m"] is not None)
            sentences.append(
                f"Among {n_rel} memory analogs with a 30-minute outcome, "
                f"{available['analog_winrate']:.0%} resolved in the {side} "
                f"side's favor (analog winrate "
                f"{available['analog_winrate']:.3f}).")
        missing = [k for k in _EVIDENCE_WEIGHTS if k not in available]
        if missing:
            sentences.append(
                f"Evidence from {', '.join(missing)} was unavailable; its "
                f"weight was renormalized over the remaining sources.")
        sentences.append(
            f"Uncertainty gates passed with margin: epistemic {epistemic:.3f} "
            f"vs {cfg.max_epistemic:.2f} (margin "
            f"{cfg.max_epistemic - epistemic:.3f}) and anomaly {anomaly:.3f} "
            f"vs {cfg.max_anomaly:.2f} (margin {cfg.max_anomaly - anomaly:.3f}), "
            f"while aleatoric {aleatoric:.3f} applied a x{aleatoric_penalty:.3f} "
            f"soft penalty.")
        sentences.append(
            f"Entry {entry_px:.4f} uses the anchor close as a next-open proxy; "
            f"stop {stop_px:.4f} and target {target_px:.4f} scale the "
            f"stop/target means ({stop_mean:.2f}/{target_mean:.2f}) by the "
            f"realized {_RANGE_BARS}-bar range {rng:.4f}"
            f"{' (0.3% price fallback — high/low arrays absent)' if rng_fallback else ''}.")
        sentences.append(
            f"Suggested size fraction {size_frac:.3f} is the policy size mean "
            f"{size_mean:.2f} scaled by conviction {conviction:.3f}; hard risk "
            f"rails apply downstream.")
        return " ".join(sentences)
