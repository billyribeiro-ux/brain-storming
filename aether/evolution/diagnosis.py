"""Self-diagnosis over Aether's run artifacts (Layer 4).

Every trainer in this codebase writes structured JSONL run logs (via
:func:`aether.utils.logging.get_logger`) and resumable checkpoints
(``last.pt`` / ``best.pt``). :class:`SelfDiagnosis` mines those artifacts —
it never touches live models — and produces a :class:`DiagnosisReport`:
machine-readable findings plus concrete, human-actionable recommendations
("rollback to best.pt", "reduce lr", ...).

Checks (one method each, see their docstrings):

1. validation-loss regression (late-run val_total > 5% above its best),
2. NaN/inf events anywhere in any log,
3. LR anomalies (lr == 0 early; lr never decaying by the end of a run),
4. stale ``best.pt`` (training long past its last improvement — plateau),
5. train/val divergence (val ≫ train ⇒ overfitting),
6. missing artifacts (training logs exist but no checkpoints do).

Design rules: a malformed or missing file is a FINDING, never a crash —
a diagnostician that dies on corrupt evidence is useless exactly when it is
needed. All thresholds are class attributes, documented and overridable.
Severity semantics: ``info`` = worth knowing, ``warn`` = degradation that
needs a human/evolution decision, ``critical`` = the run's outputs should
not be trusted. ``healthy`` means no warn/critical findings at all.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aether.utils.logging import get_logger

# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #

@dataclass
class DiagnosisFinding:
    """One detected condition, with the evidence that triggered it."""

    kind: str                      # e.g. "val_regression", "nan_inf"
    severity: str                  # "info" | "warn" | "critical"
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class DiagnosisReport:
    """The full result of one :meth:`SelfDiagnosis.scan`."""

    created: str                   # ISO-8601 UTC timestamp
    findings: list[DiagnosisFinding]
    recommendations: list[str]
    healthy: bool                  # True iff no warn/critical findings

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: "nan"/"inf" as standalone tokens in message text ("info" does NOT match).
_NAN_INF_RE = re.compile(r"(?<![a-z0-9_])(nan|inf|-inf)(?![a-z0-9_])", re.I)


def _walk_numbers(value: Any):
    """Yield every float found anywhere inside a parsed JSON value."""
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_numbers(v)


def _finite_points(records: list[dict], x_key: str, y_key: str
                   ) -> list[tuple[float, float]]:
    """Sorted ``(x, y)`` pairs from records carrying both keys, finite only."""
    pts = []
    for rec in records:
        x, y = rec.get(x_key), rec.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and math.isfinite(x) and math.isfinite(y):
            pts.append((float(x), float(y)))
    pts.sort(key=lambda p: p[0])
    return pts


# --------------------------------------------------------------------------- #
# The diagnostician
# --------------------------------------------------------------------------- #

class SelfDiagnosis:
    """Scans ``runs_dir`` JSONL logs and ``checkpoints_dir`` for pathology.

    Stateless between scans; construct once, call :meth:`scan` whenever a
    health check is wanted (end of a training run, a Layer-4 evolution
    round, a cron tick on the live box).
    """

    #: Late-run val_total this fraction above the best ⇒ regression (check 1).
    VAL_REGRESSION_PCT: float = 0.05
    #: Final LR at least this fraction of peak LR ⇒ "never decayed" (check 3).
    LR_END_FLAT_FRAC: float = 0.9
    #: best.pt this many steps behind last.pt ⇒ plateau (check 4).
    STALE_STEPS: int = 2_000
    #: val/train loss ratio beyond this ⇒ overfitting (check 5).
    DIVERGENCE_RATIO: float = 3.0
    #: Minimum points before curve-shape checks are meaningful.
    MIN_POINTS: int = 4

    def __init__(self, runs_dir: str | Path = "runs",
                 checkpoints_dir: str | Path = "checkpoints") -> None:
        self.runs_dir = Path(runs_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        # Console-only on purpose: writing our own JSONL into runs_dir would
        # make the diagnostician read (and potentially flag) its own output.
        self.logger = get_logger("aether.evolution.diagnosis")

    # ------------------------------------------------------------------ #
    # Scan
    # ------------------------------------------------------------------ #

    def scan(self) -> DiagnosisReport:
        """Run every check and assemble the report. Never raises for bad
        or missing artifacts — those become findings."""
        findings: list[DiagnosisFinding] = []
        logs = self._load_logs(findings)

        for path, records in logs.items():
            name = str(path)
            findings += self._check_nan_inf(name, records)
            findings += self._check_val_regression(name, records)
            findings += self._check_lr(name, records)
            findings += self._check_train_val_divergence(name, records)
        findings += self._check_stale_checkpoints()
        findings += self._check_missing_artifacts(logs)

        recommendations = self._recommend(findings)
        healthy = not any(f.severity in ("warn", "critical") for f in findings)
        report = DiagnosisReport(
            created=datetime.now(timezone.utc).isoformat(),
            findings=findings,
            recommendations=recommendations,
            healthy=healthy,
        )
        self.logger.info(
            "scan: %d findings (%d warn, %d critical) — %s",
            len(findings),
            sum(f.severity == "warn" for f in findings),
            sum(f.severity == "critical" for f in findings),
            "healthy" if healthy else "NOT healthy",
        )
        return report

    # ------------------------------------------------------------------ #
    # Log loading (robustness layer)
    # ------------------------------------------------------------------ #

    def _load_logs(self, findings: list[DiagnosisFinding]
                   ) -> dict[Path, list[dict]]:
        """Parse every ``*.jsonl`` under ``runs_dir``. Malformed lines and a
        missing directory are findings, not exceptions."""
        logs: dict[Path, list[dict]] = {}
        if not self.runs_dir.is_dir():
            findings.append(DiagnosisFinding(
                kind="missing_runs_dir", severity="info",
                message=f"runs directory {self.runs_dir} does not exist — "
                        "nothing to diagnose",
                evidence={"runs_dir": str(self.runs_dir)},
            ))
            return logs
        for path in sorted(self.runs_dir.rglob("*.jsonl")):
            records: list[dict] = []
            bad = 0
            try:
                text = path.read_text(errors="replace")
            except OSError as exc:
                findings.append(DiagnosisFinding(
                    kind="malformed_log", severity="warn",
                    message=f"could not read log {path}: {exc}",
                    evidence={"file": str(path), "error": str(exc)},
                ))
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    bad += 1
            if bad:
                findings.append(DiagnosisFinding(
                    kind="malformed_log", severity="info",
                    message=f"{path}: {bad} unparseable line(s) skipped",
                    evidence={"file": str(path), "bad_lines": bad,
                              "good_lines": len(records)},
                ))
            logs[path] = records
        return logs

    # ------------------------------------------------------------------ #
    # Check 1: validation-loss regression
    # ------------------------------------------------------------------ #

    def _check_val_regression(self, name: str, records: list[dict]
                              ) -> list[DiagnosisFinding]:
        """Flag runs whose LATE validation loss sits >5% above its best.

        Points: every record with both ``step`` and ``val_total``. The
        "last quarter" is the final 25% of the step RANGE; if even the best
        val_total inside that window exceeds ``(1+PCT) × global best``, the
        model has genuinely regressed (not just wiggled) — usually LR too
        hot late in training, or a data/regime shift mid-run. Skipped for
        short traces and for non-positive losses (ratios are meaningless).
        """
        pts = _finite_points(records, "step", "val_total")
        if len(pts) < self.MIN_POINTS:
            return []
        best = min(v for _, v in pts)
        if best <= 0:
            return []
        lo, hi = pts[0][0], pts[-1][0]
        cutoff = hi - (hi - lo) / 4.0
        recent = [v for s, v in pts if s >= cutoff]
        recent_best = min(recent)
        rise = recent_best / best - 1.0
        if rise > self.VAL_REGRESSION_PCT:
            return [DiagnosisFinding(
                kind="val_regression", severity="warn",
                message=(f"{name}: val_total in the last quarter of the run "
                         f"is {rise:.1%} above its best ({recent_best:.4f} "
                         f"vs {best:.4f})"),
                evidence={"file": name, "best_val_total": best,
                          "recent_best_val_total": recent_best,
                          "rise_pct": rise, "last_quarter_from_step": cutoff},
            )]
        return []

    # ------------------------------------------------------------------ #
    # Check 2: NaN / inf events
    # ------------------------------------------------------------------ #

    def _check_nan_inf(self, name: str, records: list[dict]
                       ) -> list[DiagnosisFinding]:
        """Flag any non-finite number in structured fields, or a standalone
        'nan'/'inf' token in message text.

        Structured fields: ``json.loads`` decodes the ``NaN``/``Infinity``
        tokens our JSONL writer emits into real non-finite floats, so every
        numeric leaf is walked with ``isfinite``. Messages: word-boundary
        regex, so "info"/"infra" never match. Non-finite training numbers
        mean the run's math broke — critical, whatever else looks fine.
        """
        hits: list[dict] = []
        for i, rec in enumerate(records):
            bad_fields = [v for v in _walk_numbers(rec) if not math.isfinite(v)]
            msg = rec.get("msg", "")
            token = _NAN_INF_RE.search(msg) if isinstance(msg, str) else None
            if bad_fields or token:
                hits.append({"line": i + 1,
                             "non_finite_values": len(bad_fields),
                             "msg_token": token.group(0) if token else None})
        if not hits:
            return []
        return [DiagnosisFinding(
            kind="nan_inf", severity="critical",
            message=f"{name}: {len(hits)} log line(s) contain NaN/inf",
            evidence={"file": name, "n_lines": len(hits),
                      "first_hits": hits[:5]},
        )]

    # ------------------------------------------------------------------ #
    # Check 3: learning-rate anomalies
    # ------------------------------------------------------------------ #

    def _check_lr(self, name: str, records: list[dict]
                  ) -> list[DiagnosisFinding]:
        """Two LR pathologies from the ``lr`` trace.

        * ``lr == 0`` within the first quarter of the step range: with
          warmup starting at a nonzero multiplier (see
          ``perception.train.make_lr_lambda``), an early zero LR means a
          misconfigured scheduler and silently frozen training — critical.
        * Final LR still ≥ 90% of the peak LR: the cosine decay never
          happened. Either ``scheduler.step()`` isn't being called, or the
          run stopped far short of ``max_steps`` (so its checkpoints are
          hot-LR iterates, not converged ones) — warn.
        """
        pts = _finite_points(records, "step", "lr")
        if len(pts) < self.MIN_POINTS:
            return []
        findings: list[DiagnosisFinding] = []
        lo, hi = pts[0][0], pts[-1][0]
        early_cut = lo + (hi - lo) / 4.0
        early_zero = [(s, v) for s, v in pts if s <= early_cut and v == 0.0]
        if early_zero:
            findings.append(DiagnosisFinding(
                kind="lr_zero", severity="critical",
                message=(f"{name}: lr == 0 during the first quarter of the "
                         f"run — training is frozen"),
                evidence={"file": name, "first_occurrence_step": early_zero[0][0],
                          "n_zero_points": len(early_zero)},
            ))
        peak = max(v for _, v in pts)
        final = pts[-1][1]
        if peak > 0 and final >= self.LR_END_FLAT_FRAC * peak:
            findings.append(DiagnosisFinding(
                kind="lr_no_decay", severity="warn",
                message=(f"{name}: final lr {final:.2e} is still "
                         f"{final / peak:.0%} of peak {peak:.2e} — schedule "
                         "never decayed (run ended early, or scheduler not "
                         "stepping)"),
                evidence={"file": name, "peak_lr": peak, "final_lr": final,
                          "final_step": hi},
            ))
        return findings

    # ------------------------------------------------------------------ #
    # Check 4: stale best checkpoint (plateau)
    # ------------------------------------------------------------------ #

    def _check_stale_checkpoints(self) -> list[DiagnosisFinding]:
        """Compare the ``step`` recorded inside each ``best.pt``/``last.pt``
        pair; ``last`` more than ``STALE_STEPS`` beyond ``best`` means the
        run kept burning compute long after its last improvement — a
        plateau the evolution layer should react to. Unreadable
        checkpoints become info findings.
        """
        findings: list[DiagnosisFinding] = []
        if not self.checkpoints_dir.is_dir():
            return findings
        candidates = [self.checkpoints_dir,
                      *sorted(p for p in self.checkpoints_dir.iterdir()
                              if p.is_dir())]
        for ckpt_dir in candidates:
            best_p, last_p = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
            if not (best_p.is_file() and last_p.is_file()):
                continue
            try:
                # Our own artifacts (train.py writes dicts incl. configs):
                # weights_only=False mirrors PerceptionTrainer.load_checkpoint.
                best_step = torch.load(best_p, map_location="cpu",
                                       weights_only=False).get("step")
                last_step = torch.load(last_p, map_location="cpu",
                                       weights_only=False).get("step")
            except Exception as exc:  # noqa: BLE001 — diagnostics never crash
                findings.append(DiagnosisFinding(
                    kind="checkpoint_unreadable", severity="info",
                    message=f"could not read step from checkpoints in "
                            f"{ckpt_dir}: {exc}",
                    evidence={"dir": str(ckpt_dir), "error": str(exc)},
                ))
                continue
            if not isinstance(best_step, int) or not isinstance(last_step, int):
                continue
            gap = last_step - best_step
            if gap > self.STALE_STEPS:
                findings.append(DiagnosisFinding(
                    kind="stale_best", severity="warn",
                    message=(f"{ckpt_dir}: best.pt is {gap} steps behind "
                             f"last.pt — validation has plateaued"),
                    evidence={"dir": str(ckpt_dir), "best_step": best_step,
                              "last_step": last_step, "gap_steps": gap,
                              "threshold": self.STALE_STEPS},
                ))
        return findings

    # ------------------------------------------------------------------ #
    # Check 5: train/val divergence (overfitting)
    # ------------------------------------------------------------------ #

    def _check_train_val_divergence(self, name: str, records: list[dict]
                                    ) -> list[DiagnosisFinding]:
        """Flag the run when, at its most recent joint measurement, the
        validation loss exceeds ``DIVERGENCE_RATIO ×`` the train loss.

        Uses the LAST record carrying both ``train_total`` and
        ``val_total`` (both positive): a persistent end-of-run gap is the
        overfitting signature; early-run gaps are normal while the model
        warms up, which is why only the latest point is judged.
        """
        joint = [rec for rec in records
                 if isinstance(rec.get("train_total"), (int, float))
                 and isinstance(rec.get("val_total"), (int, float))
                 and math.isfinite(rec["train_total"])
                 and math.isfinite(rec["val_total"])]
        if len(joint) < self.MIN_POINTS:
            return []
        last = joint[-1]
        train, val = float(last["train_total"]), float(last["val_total"])
        if train <= 0 or val <= 0:
            return []
        ratio = val / train
        if ratio > self.DIVERGENCE_RATIO:
            return [DiagnosisFinding(
                kind="overfit", severity="warn",
                message=(f"{name}: val/train loss ratio {ratio:.2f}x at the "
                         f"latest validation (val={val:.4f}, "
                         f"train={train:.4f}) — overfitting"),
                evidence={"file": name, "train_total": train,
                          "val_total": val, "ratio": ratio,
                          "step": last.get("step")},
            )]
        return []

    # ------------------------------------------------------------------ #
    # Check 6: missing artifacts
    # ------------------------------------------------------------------ #

    def _check_missing_artifacts(self, logs: dict[Path, list[dict]]
                                 ) -> list[DiagnosisFinding]:
        """Training logs without any checkpoint on disk mean the run's work
        is unrecoverable (crash before the first save, or checkpointing
        misconfigured) — critical. A "training log" is any log containing
        train_total/val_total fields."""
        training_logs = [str(p) for p, records in logs.items()
                         if any("train_total" in r or "val_total" in r
                                for r in records)]
        if not training_logs:
            return []
        has_ckpt = self.checkpoints_dir.is_dir() and \
            any(self.checkpoints_dir.rglob("*.pt"))
        if has_ckpt:
            return []
        return [DiagnosisFinding(
            kind="missing_artifacts", severity="critical",
            message=(f"training logs exist ({len(training_logs)} file(s)) but "
                     f"no .pt checkpoint found under {self.checkpoints_dir}"),
            evidence={"training_logs": training_logs,
                      "checkpoints_dir": str(self.checkpoints_dir)},
        )]

    # ------------------------------------------------------------------ #
    # Recommendations
    # ------------------------------------------------------------------ #

    #: Finding kind -> concrete actions, most urgent first.
    _RECOMMENDATIONS: dict[str, list[str]] = {
        "nan_inf": [
            "rollback to best.pt (the post-NaN iterates are poisoned)",
            "reduce lr (halve the peak) and verify grad_clip > 0",
            "refit ticker stats — stale normalization can produce extreme "
            "z-scores that overflow",
        ],
        "val_regression": [
            "rollback to best.pt",
            "reduce lr for the remainder of training",
        ],
        "lr_zero": [
            "fix the scheduler wiring (warmup_steps vs max_steps) — lr must "
            "never be 0 during warmup",
        ],
        "lr_no_decay": [
            "verify scheduler.step() runs once per optimizer step",
            "align max_steps with the actual run length so the cosine decay "
            "completes (deploy from best.pt, not a hot-lr last.pt)",
        ],
        "stale_best": [
            "rollback to best.pt — the extra steps bought nothing",
            "increase mask_ratio (a harder SSL task can break the plateau)",
            "let Population evolution mutate the config instead of extending "
            "this run",
        ],
        "overfit": [
            "increase mask_ratio",
            "increase dropout / weight_decay",
            "widen the training window (more tickers / history)",
        ],
        "missing_artifacts": [
            "re-run training — verify checkpoint_dir is writable and "
            "val_every is small enough to reach a first save",
        ],
        "malformed_log": [
            "rotate or repair the corrupted JSONL file (concurrent writers "
            "and crashes both truncate lines)",
        ],
        "checkpoint_unreadable": [
            "re-save the checkpoint from a live trainer, or delete the "
            "corrupt pair and retrain",
        ],
    }

    def _recommend(self, findings: list[DiagnosisFinding]) -> list[str]:
        """Deduplicated action list: critical findings' actions first, then
        warns, then infos; original order preserved within a tier."""
        rank = {"critical": 0, "warn": 1, "info": 2}
        seen: set[str] = set()
        actions: list[str] = []
        ordered = sorted(findings, key=lambda f: rank.get(f.severity, 3))
        for finding in ordered:
            for action in self._RECOMMENDATIONS.get(finding.kind, ()):
                if action not in seen:
                    seen.add(action)
                    actions.append(action)
        return actions
