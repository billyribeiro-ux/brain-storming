"""Forensic post-mortems of closed trades (Layer 2).

LOOK-AHEAD DISCLAIMER — READ THIS FIRST
---------------------------------------
This module deliberately uses the ACTUAL bar path AFTER a trade's entry —
including bars after its exit. That would be a fatal leak in any feature or
training-input pipeline; here it is exactly correct, because an autopsy is
*post-trade forensics*: the trade is closed, the future it traded into is
now history, and the whole point is to replay that realized history under
"what if" variations. Nothing produced here (reports, counterfactuals,
narratives) may ever be fed back as a model INPUT for a time before the
trade closed. Lessons are consumed by TRAINING loops (replay weighting),
which see them strictly after the fact.

What an autopsy contains
------------------------
* **Counterfactuals** — the trade replayed against the real bar path with
  perturbed timing (enter 5/15 bars earlier/later), perturbed brackets
  (stop and target at 0.5x/2x their original distance), and a "hold to
  exit+30min with no stop" variant. Intrabar ambiguity is resolved
  CONSERVATIVELY: if a bar's range spans both stop and target, the stop is
  assumed to fill first; gaps through a stop fill at the (worse) open, gaps
  through a target still fill at the target (never credit upside we cannot
  prove).
* **Verdict** — a two-axis honesty label. A loss whose counterfactual
  neighborhood also loses (>= 70%) is a *good_loss* (the thesis was wrong
  or unlucky; parameters would not have saved it). A loss whose
  neighborhood mostly profits is a *bad_loss* (execution/parameters are the
  fixable culprit). Wins mirror to *good_win* / *lucky_win*. The grey zone
  (50–70% agreement) resolves toward the ACTIONABLE label (bad_loss /
  lucky_win) so lessons err on the side of scrutiny.
* **Drivers** — interpretable attribution WITHOUT the neural net. We rank
  the causal snapshot's top incoming edges for ``<TICKER>.ret_1m`` together
  with realized statistics around entry (volume z, range expansion, 5-minute
  trend), normalized to sum to 1. This is deliberately shallow: this module
  stays dependency-light so autopsies run anywhere, any time. Deeper,
  gradient-based saliency over the perception embedding lives with the
  signal producer (``execution.signals``) — do not mistake these ranks for
  that analysis.
* **Analogs** — retrieved from the (optional) memory bank via a caller-
  provided key vector; **lessons** — structured entries appended to the
  lessons buffer that training consumers upweight/replay.

Graceful degradation: every collaborator (memory, causal snapshot, even a
usable bar path) is optional. Missing evidence is skipped and NAMED in the
narrative instead of raising — a partially-informed autopsy beats none.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aether.utils.logging import get_logger
from aether.worldmodel.interfaces import (
    CAUSAL_TICKER_CHANNELS,
    LESSONS_PATH,
    AutopsyReport,
    CausalGraphSnapshot,
    Counterfactual,
    MemoryAnalog,
    TradeRecord,
)

logger = get_logger("aether.worldmodel.autopsy")

#: The "hold with no stop" counterfactual runs this far past the actual exit.
HOLD_EXTRA_SECONDS = 30 * 60

#: Timing counterfactuals: entry shifted by this many 1-minute bars.
ENTRY_SHIFTS = (-15, -5, 5, 15)

#: Bracket counterfactuals: stop/target distance multipliers.
BRACKET_SCALES = (0.5, 2.0)

#: Verdict agreement threshold: fraction of counterfactuals that must share
#: the trade's fate for it to be called "good" (thesis-driven).
VERDICT_AGREEMENT = 0.7

#: $PnL that earns a hard example full weight 1.0 (capped at 3.0), i.e.
#: weight = min(3, |pnl| / 50): a $150+ bad loss is maximally upweighted.
LESSON_PNL_SCALE = 50.0

_MIN_STD = 1e-8
_OHLCV = ("date", "open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- #
# Bar-path plumbing (pure helpers)
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class _BarPath:
    """Column-major numpy view of a 1min OHLCV frame, sorted by time.

    ``ts`` uses the lake's naive-ET epoch-seconds convention (the naive
    wall-clock datetime interpreted as if UTC) — the same encoding as
    ``TradeRecord.entry_ts``/``exit_ts``, so comparisons are direct ints.
    """

    ts: np.ndarray      # [N] int64 epoch-seconds (bar start)
    open: np.ndarray    # [N] float64
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.ts.size)

    @classmethod
    def from_frame(cls, bars: pd.DataFrame) -> "_BarPath | None":
        """Parse/sort a bars frame; None if unusable (degraded autopsy)."""
        if bars is None or bars.empty or not set(_OHLCV) <= set(bars.columns):
            return None
        df = bars.sort_values("date")
        ts = pd.to_datetime(df["date"])
        if ts.dt.tz is not None:  # normalize aware stamps to naive ET
            ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
        return cls(
            ts=ts.to_numpy().astype("datetime64[s]").astype(np.int64),
            open=df["open"].to_numpy(np.float64),
            high=df["high"].to_numpy(np.float64),
            low=df["low"].to_numpy(np.float64),
            close=df["close"].to_numpy(np.float64),
            volume=df["volume"].to_numpy(np.float64),
        )

    def index_at(self, epoch: int) -> int:
        """Index of the first bar at/after ``epoch`` (== len if past data)."""
        return int(np.searchsorted(self.ts, epoch, side="left"))

    def last_index_at_or_before(self, epoch: int) -> int:
        """Index of the last bar with ts <= epoch (-1 if none)."""
        return int(np.searchsorted(self.ts, epoch, side="right")) - 1


def _direction(side: str) -> float:
    """+1 for long, -1 for short; anything else is a caller bug."""
    if side == "long":
        return 1.0
    if side == "short":
        return -1.0
    raise ValueError(f"TradeRecord.side must be 'long'|'short', got {side!r}")


def simulate_bracket(path: _BarPath, side: str, entry_idx: int,
                     entry_px: float, qty: float, fees: float,
                     horizon_ts: int,
                     stop_px: float | None = None,
                     target_px: float | None = None) -> float:
    """Replay a bracket order against the REAL bar path; return $PnL.

    Semantics (all deliberately conservative):

    * The fill happens at ``entry_px`` "at the close" of bar ``entry_idx``;
      triggers are therefore evaluated from bar ``entry_idx + 1`` onward —
      the entry bar's extremes may predate the fill, and using them would be
      time travel.
    * Per bar, the STOP is checked BEFORE the target: a bar whose range
      spans both is assumed to stop us out (worst case for the trader).
    * A gap through the stop fills at the bar's open (worse than the stop);
      a gap through the target still fills AT the target (we never credit
      better-than-target fills we cannot prove).
    * If nothing triggers by ``horizon_ts``, exit at the close of the last
      bar with ``ts <= horizon_ts`` (the moment the real policy went flat).
    * ``fees`` are charged in full — every counterfactual is a round trip
      with the same cost structure, so deltas vs the actual trade are
      apples-to-apples.

    Pure function of its arguments; ``stop_px``/``target_px`` of ``None``
    disable that leg (used by the "hold with no stop" counterfactual).
    """
    direction = _direction(side)
    exit_px = entry_px  # degenerate default: no post-entry bars => flat + fees
    for i in range(entry_idx + 1, len(path)):
        if path.ts[i] > horizon_ts:
            break
        # --- stop first (conservative intrabar ordering) ------------------
        if stop_px is not None:
            hit = (path.low[i] <= stop_px) if direction > 0 \
                else (path.high[i] >= stop_px)
            if hit:
                # Gap through the stop: fill at the (worse) open.
                if direction > 0:
                    exit_px = min(stop_px, path.open[i])
                else:
                    exit_px = max(stop_px, path.open[i])
                break
        # --- then target ---------------------------------------------------
        if target_px is not None:
            hit = (path.high[i] >= target_px) if direction > 0 \
                else (path.low[i] <= target_px)
            if hit:
                exit_px = target_px  # never credit a better-than-target gap
                break
        exit_px = path.close[i]      # mark to close; horizon exit uses this
    return direction * (exit_px - entry_px) * qty - fees


def _excursions(path: _BarPath, entry_idx: int, exit_idx: int,
                entry_px: float, direction: float) -> tuple[float, float]:
    """(MAE, MFE) in price terms over bars [entry_idx, exit_idx].

    MFE = furthest the path moved IN the trade's favor; MAE = furthest it
    moved AGAINST. Both are >= 0 (clamped) and 0 when the slice is empty.
    """
    if entry_idx >= len(path) or exit_idx < entry_idx:
        return 0.0, 0.0
    hi = float(path.high[entry_idx:exit_idx + 1].max())
    lo = float(path.low[entry_idx:exit_idx + 1].min())
    if direction > 0:
        mfe, mae = hi - entry_px, entry_px - lo
    else:
        mfe, mae = entry_px - lo, hi - entry_px
    return max(mae, 0.0), max(mfe, 0.0)


def _fmt_ts(epoch: int) -> str:
    """Naive-ET epoch-seconds -> 'YYYY-MM-DD HH:MM' for narratives."""
    return pd.Timestamp(int(epoch), unit="s").strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# The Autopsist
# --------------------------------------------------------------------------- #

class Autopsist:
    """Performs forensic autopsies on closed trades.

    Parameters
    ----------
    memory:
        Optional :class:`~aether.worldmodel.interfaces.MemoryBankProtocol`
        implementation. Without it (or without an ``analog_key``), analog
        retrieval is skipped and noted in the narrative.
    causal_snapshot:
        Optional fitted :class:`CausalGraphSnapshot` (see ``causal.py``).
        Without it, causal attribution/context are skipped and noted.
    """

    def __init__(self, memory: object | None = None,
                 causal_snapshot: CausalGraphSnapshot | None = None) -> None:
        self.memory = memory
        self.causal_snapshot = causal_snapshot

    # ------------------------------------------------------------------ #
    # Counterfactual battery
    # ------------------------------------------------------------------ #

    def _counterfactuals(self, trade: TradeRecord,
                         path: _BarPath) -> list[Counterfactual]:
        """The standard perturbation battery around one trade.

        Every variation changes EXACTLY ONE thing (entry timing, stop
        distance, target distance, or exit discipline) while holding the
        rest of the plan fixed, so each delta isolates one decision. The
        policy's chosen exit moment (``exit_ts``) is kept as the forced
        horizon for all bracketed variants — we are grading the bracket,
        not inventing a new exit policy — except the explicit hold-longer
        variant, which extends the horizon by 30 minutes.
        """
        direction = _direction(trade.side)
        entry_idx = path.index_at(trade.entry_ts)
        if entry_idx >= len(path):
            logger.warning("autopsy %s: bar path ends before entry — "
                           "no counterfactuals", trade.trade_id)
            return []
        stop_dist = abs(trade.entry_px - trade.stop_px)
        target_dist = abs(trade.target_px - trade.entry_px)

        out: list[Counterfactual] = []

        def add(description: str, pnl: float) -> None:
            out.append(Counterfactual(description=description,
                                      pnl=round(float(pnl), 6),
                                      delta=round(float(pnl) - trade.pnl, 6)))

        # --- timing: same bracket DISTANCES re-anchored at the new fill ----
        for shift in ENTRY_SHIFTS:
            when = "earlier" if shift < 0 else "later"
            desc = f"enter {abs(shift)} bars {when}"
            j = entry_idx + shift
            # Skip impossible timings instead of faking them: before the
            # provided history, after the data, or at/after the policy exit.
            if j < 0 or j >= len(path) or path.ts[j] >= trade.exit_ts:
                logger.warning("autopsy %s: '%s' outside the bar path — "
                               "skipped", trade.trade_id, desc)
                continue
            fill = float(path.close[j])
            add(desc, simulate_bracket(
                path, trade.side, j, fill, trade.qty, trade.fees,
                horizon_ts=trade.exit_ts,
                stop_px=fill - direction * stop_dist,
                target_px=fill + direction * target_dist))

        # --- stop distance x0.5 / x2 (entry & target unchanged) ------------
        for scale in BRACKET_SCALES:
            add(f"stop at {scale:g}x distance", simulate_bracket(
                path, trade.side, entry_idx, trade.entry_px, trade.qty,
                trade.fees, horizon_ts=trade.exit_ts,
                stop_px=trade.entry_px - direction * stop_dist * scale,
                target_px=trade.target_px))

        # --- target distance x0.5 / x2 (entry & stop unchanged) ------------
        for scale in BRACKET_SCALES:
            add(f"target at {scale:g}x distance", simulate_bracket(
                path, trade.side, entry_idx, trade.entry_px, trade.qty,
                trade.fees, horizon_ts=trade.exit_ts,
                stop_px=trade.stop_px,
                target_px=trade.entry_px + direction * target_dist * scale))

        # --- discipline: hold to exit+30min, no stop, no target ------------
        add("hold to exit+30min with no stop", simulate_bracket(
            path, trade.side, entry_idx, trade.entry_px, trade.qty,
            trade.fees, horizon_ts=trade.exit_ts + HOLD_EXTRA_SECONDS,
            stop_px=None, target_px=None))
        return out

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #

    @staticmethod
    def _verdict(trade: TradeRecord,
                 cfs: list[Counterfactual]) -> tuple[str, str]:
        """(verdict, one-sentence reasoning) from the counterfactual fates.

        No counterfactuals (degraded bar path) means no evidence AGAINST
        the thesis, so the outcome defaults to the "good" label — and the
        narrative separately flags the missing evidence.
        """
        n = len(cfs)
        n_loss = sum(1 for c in cfs if c.pnl < 0)
        n_win = sum(1 for c in cfs if c.pnl > 0)
        if trade.pnl < 0:
            if n == 0 or n_loss / n >= VERDICT_AGREEMENT:
                return "good_loss", (
                    f"{n_loss}/{n} counterfactuals also lost — the thesis "
                    "was wrong or unlucky; no nearby parameter change "
                    "rescued it")
            return "bad_loss", (
                f"{n_win}/{n} counterfactuals were profitable — execution "
                "or bracket parameters, not the thesis, drove this loss")
        if n == 0 or n_win / n >= VERDICT_AGREEMENT:
            return "good_win", (
                f"{n_win}/{n} counterfactuals also profited — the edge was "
                "robust to timing and bracket perturbations")
        return "lucky_win", (
            f"only {n_win}/{n} counterfactuals profited — this win leaned "
            "on luck rather than a robust edge")

    # ------------------------------------------------------------------ #
    # Drivers & causal context (no neural net — see module docstring)
    # ------------------------------------------------------------------ #

    def _drivers(self, trade: TradeRecord, path: _BarPath | None,
                 entry_idx: int) -> list[dict]:
        """Interpretable attribution: causal edges + realized entry stats.

        Scores: a causal edge contributes ``|weight| * confidence``; each
        realized statistic contributes its |z|-like magnitude vs the hour
        of bars before entry. Scores are normalized to sum to 1. This is
        an honest RANKING of interpretable evidence, not saliency — the
        gradient-based analysis lives in execution.signals.
        """
        scored: list[tuple[str, float]] = []

        # 1. Causal snapshot: strongest stable edges into this ticker's
        #    1-minute return node.
        if self.causal_snapshot is not None:
            for e in self.causal_snapshot.top_drivers(
                    dst=f"{trade.ticker}.ret_1m", k=5):
                scored.append((f"causal:{e.src}@lag{e.lag}",
                               abs(e.weight) * e.confidence))

        # 2. Realized statistics around entry: how unusual were volume,
        #    range and short-term trend at the moment we pulled the trigger,
        #    scored against the pre-entry hour as the local baseline.
        #    (Statistical descriptions of the tape, not TA indicator rules.)
        if path is not None and 0 < entry_idx <= len(path):
            base = slice(max(0, entry_idx - 60), entry_idx)   # pre-entry hour
            last = slice(max(0, entry_idx - 4), min(entry_idx + 1, len(path)))

            def z_of_recent(series: np.ndarray) -> float:
                ref = series[base]
                if ref.size < 2:
                    return 0.0
                sd = max(float(ref.std()), _MIN_STD)
                return (float(series[last].mean()) - float(ref.mean())) / sd

            log_vol = np.log1p(np.maximum(path.volume, 0.0))
            log_rng = np.log(np.maximum(
                np.maximum(path.high, path.low) / np.maximum(path.low, _MIN_STD),
                1.0))
            rets = np.diff(np.log(np.maximum(path.close, _MIN_STD)),
                           prepend=np.log(max(path.close[0], _MIN_STD)))
            ref = rets[base]
            sd = max(float(ref.std()), _MIN_STD) if ref.size >= 2 else _MIN_STD
            # 5-minute trend in "sigmas of 1m returns": sign & magnitude.
            trend = float(rets[last].sum()) / (sd * np.sqrt(5.0))

            scored.append(("realized:volume_z", abs(z_of_recent(log_vol))))
            scored.append(("realized:range_expansion", abs(z_of_recent(log_rng))))
            scored.append((f"realized:trend_5m({'+' if trend >= 0 else '-'})",
                           abs(trend)))

        if not scored:
            return []
        total = sum(s for _, s in scored)
        if total <= 0.0:                       # all-zero evidence -> uniform
            return [{"name": n, "attribution": 1.0 / len(scored)}
                    for n, _ in scored]
        return sorted(({"name": n, "attribution": s / total}
                       for n, s in scored),
                      key=lambda d: -d["attribution"])

    def _causal_context(self, ticker: str) -> list[dict]:
        """Top stable incoming edges for each of the ticker's channels."""
        if self.causal_snapshot is None:
            return []
        out: list[dict] = []
        for ch in CAUSAL_TICKER_CHANNELS:
            for e in self.causal_snapshot.top_drivers(dst=f"{ticker}.{ch}", k=3):
                out.append(dataclasses.asdict(e))
        return out

    # ------------------------------------------------------------------ #
    # Analogs & lessons
    # ------------------------------------------------------------------ #

    def _analogs(self, analog_key, notes: list[str]) -> list[MemoryAnalog]:
        """Query the memory bank if both it and a key vector are present."""
        if self.memory is None:
            notes.append("no memory bank attached (analog retrieval skipped)")
            return []
        if analog_key is None:
            notes.append("no analog key provided (analog retrieval skipped)")
            return []
        try:
            return list(self.memory.query(analog_key, k=8))
        except Exception as exc:  # a broken memory must not kill forensics
            logger.warning("autopsy: memory.query failed (%s)", exc)
            notes.append(f"memory query failed: {exc}")
            return []

    @staticmethod
    def _lessons(trade: TradeRecord, verdict: str) -> list[dict]:
        """Structured lessons for the training-side replay buffer.

        * ``bad_loss`` -> a HARD EXAMPLE anchored at the entry moment,
          weighted by loss size (|pnl|/$50, capped at 3.0): fixable
          mistakes deserve replay pressure proportional to their cost.
        * ``good_loss`` -> a REGIME NOTE: nothing to fix, but the state is
          worth remembering as "markets where this thesis fails".
        * Wins teach nothing actionable through this channel (yet).
        """
        if verdict == "bad_loss":
            return [{
                "kind": "hard_example",
                "trade_id": trade.trade_id,
                "ticker": trade.ticker,
                "anchor_ts": trade.entry_ts,
                "weight": min(3.0, abs(trade.pnl) / LESSON_PNL_SCALE),
            }]
        if verdict == "good_loss":
            return [{
                "kind": "regime_note",
                "trade_id": trade.trade_id,
                "ticker": trade.ticker,
                "anchor_ts": trade.entry_ts,
                "weight": 1.0,
                "note": "thesis-consistent loss: counterfactual neighborhood "
                        "also lost — regime, not execution",
            }]
        return []

    # ------------------------------------------------------------------ #
    # Narrative
    # ------------------------------------------------------------------ #

    @staticmethod
    def _narrative(trade: TradeRecord, verdict: str, reasoning: str,
                   mae: float, mfe: float, cfs: list[Counterfactual],
                   drivers: list[dict], analogs: list[MemoryAnalog],
                   lessons: list[dict], notes: list[str]) -> str:
        """Compose the 5-10 sentence post-mortem from the actual numbers."""
        s: list[str] = []
        s.append(
            f"{trade.ticker} {trade.side} {trade.qty:g} @ {trade.entry_px:.2f} "
            f"entered {_fmt_ts(trade.entry_ts)}, exited {_fmt_ts(trade.exit_ts)} "
            f"@ {trade.exit_px:.2f} ({trade.exit_reason}) for "
            f"{trade.pnl:+.2f} after {trade.fees:.2f} fees.")
        s.append(
            f"While open, the path ran {mfe:+.2f}/share "
            f"({mfe / max(trade.entry_px, _MIN_STD):+.2%}) in the trade's favor "
            f"at best and {mae:.2f}/share "
            f"({mae / max(trade.entry_px, _MIN_STD):.2%}) against it at worst.")
        s.append(f"Verdict: {verdict} — {reasoning}.")
        if cfs:
            best = max(cfs, key=lambda c: c.pnl)
            worst = min(cfs, key=lambda c: c.pnl)
            s.append(
                f"The best variation was '{best.description}' "
                f"({best.pnl:+.2f}, {best.delta:+.2f} vs actual); the worst "
                f"was '{worst.description}' ({worst.pnl:+.2f}).")
        else:
            s.append("No counterfactuals could be simulated because the bar "
                     "path did not usably cover the entry.")
        if drivers:
            top = ", ".join(f"{d['name']} ({d['attribution']:.0%})"
                            for d in drivers[:3])
            s.append(f"Ranked interpretable drivers around entry: {top}.")
        else:
            s.append("No interpretable drivers could be ranked (no causal "
                     "snapshot and no usable bar path).")
        if analogs:
            avg = float(np.mean([a.similarity for a in analogs]))
            s.append(f"Memory returned {len(analogs)} historical analogs "
                     f"(mean similarity {avg:+.2f}).")
        if notes:
            s.append(f"Evidence gaps: {'; '.join(notes)}.")
        if lessons:
            lesson = lessons[0]
            s.append(f"Lesson recorded: {lesson['kind']} anchored at "
                     f"{_fmt_ts(lesson['anchor_ts'])} with weight "
                     f"{lesson['weight']:.2f}.")
        else:
            s.append("No lesson recorded for this verdict.")
        return " ".join(s)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def autopsy(self, trade: TradeRecord, bars: pd.DataFrame,
                analog_key=None) -> AutopsyReport:
        """Full forensic post-mortem of one closed trade.

        Parameters
        ----------
        trade:
            The closed round-trip.
        bars:
            That ticker's 1min OHLCV covering ``[entry_ts - 60min,
            exit_ts + horizon]`` (horizon >= 30min so the hold-longer
            counterfactual has road). Using ACTUAL post-entry bars is
            correct here — see the module-level look-ahead disclaimer.
        analog_key:
            Optional key vector (perception fused embedding at entry) for
            memory retrieval; without it, analogs are skipped.
        """
        notes: list[str] = []

        path = _BarPath.from_frame(bars)
        if path is None:
            notes.append("bars missing or malformed (counterfactuals and "
                         "realized stats skipped)")
            entry_idx = 0
            cfs: list[Counterfactual] = []
            mae = mfe = 0.0
        else:
            entry_idx = path.index_at(trade.entry_ts)
            cfs = self._counterfactuals(trade, path)
            exit_idx = path.last_index_at_or_before(trade.exit_ts)
            mae, mfe = _excursions(path, entry_idx, exit_idx,
                                   trade.entry_px, _direction(trade.side))

        verdict, reasoning = self._verdict(trade, cfs)
        drivers = self._drivers(trade, path, entry_idx)
        if self.causal_snapshot is None:
            notes.append("no causal snapshot supplied (causal attribution "
                         "skipped)")
        causal_context = self._causal_context(trade.ticker)
        analogs = self._analogs(analog_key, notes)
        lessons = self._lessons(trade, verdict)
        narrative = self._narrative(trade, verdict, reasoning, mae, mfe,
                                    cfs, drivers, analogs, lessons, notes)
        return AutopsyReport(trade=trade, verdict=verdict, narrative=narrative,
                             drivers=drivers, causal_context=causal_context,
                             counterfactuals=cfs, analogs=analogs,
                             lessons=lessons)


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #

def save_report(report: AutopsyReport, data_root: Path | str) -> Path:
    """Write ``<data_root>/autopsies/<trade_id>.json``; return the path.

    The report is a tree of dataclasses, so ``dataclasses.asdict`` yields a
    plain JSON-serializable dict (``default=str`` guards exotic values that
    may ride in ``signal_meta`` / analog ``meta``).
    """
    out_dir = Path(data_root) / "autopsies"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.trade.trade_id}.json"
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2,
                               default=str))
    logger.info("saved autopsy -> %s (verdict=%s)", path, report.verdict)
    return path


def append_lessons(reports: list[AutopsyReport],
                   data_root: Path | str) -> int:
    """Append every report's lessons to ``<data_root>/lessons.jsonl``.

    Append-only JSONL (one lesson per line) per the contract's
    ``LESSONS_PATH`` — training consumers tail this file to upweight or
    replay the flagged states. Returns the number of lessons written.
    """
    path = Path(data_root) / LESSONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lessons = [lesson for report in reports for lesson in report.lessons]
    if lessons:
        with path.open("a") as fh:
            for lesson in lessons:
                fh.write(json.dumps(lesson, default=str) + "\n")
    logger.info("appended %d lessons -> %s", len(lessons), path)
    return len(lessons)
