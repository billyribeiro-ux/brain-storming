"""The Aether bridge: HTTP + WebSocket API for the SvelteKit cockpit.

One FastAPI service exposing the lake, embeddings, backtests, autopsies,
causal snapshots, brain status, an imagination endpoint, and a session
REPLAY engine that streams historical bars over the WebSocket so the
cockpit gets a truthful "live" experience from recorded markets (no fake
data — every tick replayed is a real lake bar; signals pushed during replay
are the ones the recorded backtest actually emitted at that bar).

Run:  python3 scripts/run_api.py            (default port 8600)

Time convention: identical to the lake — naive-ET wall-clock epoch seconds.
The frontend formats them with UTC formatters; nobody converts zones.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..config import TICKERS, AetherConfig
from ..data.storage import ParquetStore
from ..utils.logging import get_logger
from . import views

logger = get_logger("aether.api")

app = FastAPI(title="Aether Bridge", version="1.0")
CFG = AetherConfig.from_env()
DATA = CFG.data.root
EMB_DIR = Path("data/cache/embeddings")


def _store() -> ParquetStore:
    return ParquetStore(DATA)


def _ts(series: pd.Series) -> list[int]:
    """pandas datetimes -> naive-ET epoch seconds (the lake convention)."""
    return (series.astype("datetime64[s]").astype("int64")).tolist()


# --------------------------------------------------------------------------- #
# Universe / bars
# --------------------------------------------------------------------------- #

def _regime_label(daily: pd.DataFrame) -> str:
    """Coarse regime tag from the last 20 daily bars — descriptive stats for
    the sidebar, not a trading input."""
    if daily is None or len(daily) < 21:
        return "unknown"
    px = daily["close"].tail(21).to_numpy()
    rets = np.diff(np.log(px))
    trend = float(np.sum(rets))
    vol = float(np.std(rets) * math.sqrt(252))
    if vol > 0.35:
        return "volatile"
    if trend > 0.03:
        return "trending-up"
    if trend < -0.03:
        return "trending-down"
    return "ranging"


@app.get("/api/tickers")
def tickers() -> list[dict]:
    store = _store()
    out = []
    latest_signals = _latest_backtest_signal_prob()
    for t in TICKERS:
        daily = store.read("bars_daily", t.alias, columns=["date", "close", "open"])
        last_px = day_ret = None
        if len(daily) >= 2:
            last_px = float(daily["close"].iloc[-1])
            day_ret = float(daily["close"].iloc[-1] / daily["close"].iloc[-2] - 1.0)
        wm = store.watermark("bars_1min", t.alias)
        out.append({
            "symbol": t.symbol, "alias": t.alias, "kind": t.kind,
            "last_px": last_px, "day_ret": day_ret,
            "regime": _regime_label(daily),
            "signal_prob": latest_signals.get(t.alias),
            "last_bar_ts": int(wm.timestamp()) if wm is not None else None,
        })
    return out


@app.get("/api/dates")
def dates(ticker: str) -> list[str]:
    emb = _emb(ticker)
    if emb is None:
        raise HTTPException(404, f"no embeddings for {ticker}")
    days = pd.to_datetime(emb["anchor_ts"], unit="s").normalize().unique()
    return [str(d.date()) for d in sorted(days)]


@app.get("/api/bars")
def bars(ticker: str, date: str, tf: str = "1min") -> list[dict]:
    if tf not in ("1min", "5min"):
        raise HTTPException(400, "tf must be 1min|5min")
    df = _store().read(f"bars_{tf}", ticker, start=date, end=f"{date} 23:59:59")
    if df.empty:
        return []
    return [
        {"t": t, "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)}
        for t, o, h, l, c, v in zip(_ts(df["date"]), df["open"], df["high"],
                                    df["low"], df["close"], df["volume"])
    ]


@app.get("/api/uncertainty")
def uncertainty(ticker: str, date: str) -> list[dict]:
    emb = _emb(ticker)
    if emb is None:
        return []
    ts = emb["anchor_ts"]
    day = pd.Timestamp(date)
    mask = (ts >= day.value // 10**9) & (ts < (day + pd.Timedelta(days=1)).value // 10**9)
    return [
        {"t": int(t), "aleatoric": float(a), "epistemic": float(e), "anomaly": float(x)}
        for t, a, e, x in zip(ts[mask], emb["aleatoric"][mask],
                              emb["epistemic"][mask], emb["anomaly"][mask])
    ]


@lru_cache(maxsize=16)
def _emb_cached(ticker: str, mtime: float) -> Optional[dict]:
    path = EMB_DIR / f"{ticker}.npz"
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _emb(ticker: str) -> Optional[dict]:
    path = EMB_DIR / f"{ticker}.npz"
    return _emb_cached(ticker, path.stat().st_mtime if path.is_file() else 0.0)


# --------------------------------------------------------------------------- #
# Backtests / signals / trades / autopsies
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def _backtest_cached(name: str, mtime: float) -> dict:
    path = DATA / "backtests" / name / "result.json"
    return json.loads(path.read_text())


def _backtest(name: str) -> dict:
    path = DATA / "backtests" / name / "result.json"
    if not path.is_file():
        raise HTTPException(404, f"backtest {name!r} not found")
    return _backtest_cached(name, path.stat().st_mtime)


def _backtest_names() -> list[str]:
    root = DATA / "backtests"
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if (p / "result.json").is_file()),
                  key=lambda n: (root / n / "result.json").stat().st_mtime, reverse=True)


def _newest_backtest() -> Optional[str]:
    names = _backtest_names()
    return names[0] if names else None


def _latest_backtest_signal_prob() -> dict[str, float]:
    """Sidebar 'signal probability': share of the newest backtest's final
    session on which each ticker had a live signal."""
    name = _newest_backtest()
    if name is None:
        return {}
    sigs = _backtest(name).get("signals") or []
    if not sigs:
        return {}
    last_day = max(pd.Timestamp(s["ts"], unit="s").date() for s in sigs)
    out: dict[str, float] = {}
    for t in TICKERS:
        day_sigs = [s for s in sigs if s["ticker"] == t.alias
                    and pd.Timestamp(s["ts"], unit="s").date() == last_day]
        out[t.alias] = min(1.0, len(day_sigs) / 390)
    return out


@app.get("/api/backtests")
def backtests() -> list[dict]:
    out = []
    for name in _backtest_names():
        r = _backtest(name)
        trades = r.get("trades") or []
        entry_ts = [t["entry_ts"] for t in trades if t.get("entry_ts")]
        out.append({
            "name": name,
            "stats": {k: v for k, v in (r.get("stats") or {}).items() if k != "per_ticker"},
            "n_trades": len(trades),
            "n_signals": len(r.get("signals") or []),
            "start": str(pd.Timestamp(min(entry_ts), unit="s").date()) if entry_ts else None,
            "end": str(pd.Timestamp(max(entry_ts), unit="s").date()) if entry_ts else None,
        })
    return out


def _autopsy_ids() -> set[str]:
    root = DATA / "autopsies"
    return {p.stem for p in root.glob("*.json")} if root.is_dir() else set()


@app.get("/api/signals")
def signals(backtest: str = "", ticker: str = "",
            from_: str = Query("", alias="from"), to: str = "",
            limit: int = 500) -> list[dict]:
    name = backtest or _newest_backtest()
    if name is None:
        return []
    recs = views.filter_records(_backtest(name).get("signals"),
                                ticker=ticker or None,
                                start=from_ or None, end=to or None, ts_keys=("ts",))
    out = []
    for s in recs[-limit:]:
        ev = s.get("evidence") or {}
        drivers = [{"name": d.get("name", "?"), "attribution": float(d.get("attribution", d.get("score", 0)) or 0)}
                   for d in (ev.get("causal_drivers") or [])[:6]]
        out.append({**{k: s[k] for k in ("signal_id", "ts", "ticker", "side", "conviction",
                                          "entry_px", "stop_px", "target_px",
                                          "horizon_bars", "size_frac")},
                    "rationale": s.get("rationale", ""),
                    "drivers": drivers,
                    "evidence": {k: (None if ev.get(k) is None or
                                     (isinstance(ev.get(k), float) and math.isnan(ev[k]))
                                     else ev.get(k))
                                 for k in ("policy_prob", "imagination_agreement",
                                           "analog_winrate", "aleatoric", "epistemic",
                                           "anomaly")}})
    return out


@app.get("/api/trades")
def trades(backtest: str = "", ticker: str = "",
           from_: str = Query("", alias="from"), to: str = "") -> list[dict]:
    name = backtest or _newest_backtest()
    if name is None:
        return []
    recs = views.filter_records(_backtest(name).get("trades"),
                                ticker=ticker or None,
                                start=from_ or None, end=to or None,
                                ts_keys=("entry_ts",))
    ids = _autopsy_ids()
    keys = ("trade_id", "ticker", "side", "entry_ts", "exit_ts", "entry_px", "exit_px",
            "qty", "pnl", "fees", "stop_px", "target_px", "exit_reason", "conviction")
    return [{**{k: t.get(k) for k in keys}, "has_autopsy": str(t.get("trade_id")) in ids}
            for t in recs]


@app.get("/api/equity")
def equity(backtest: str = "") -> list[dict]:
    name = backtest or _newest_backtest()
    if name is None:
        return []
    path = DATA / "backtests" / name / "equity.parquet"
    if not path.is_file():
        return []
    df = pd.read_parquet(path)
    tcol = "ts" if "ts" in df.columns else df.columns[0]
    return [{"t": t, "equity": float(e)} for t, e in
            zip(_ts(pd.to_datetime(df[tcol])), df["equity"])]


@app.get("/api/autopsies")
def autopsies(ticker: str = "", verdict: str = "", limit: int = 200) -> list[dict]:
    root = DATA / "autopsies"
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            rep = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tr = rep.get("trade") or {}
        if ticker and tr.get("ticker") != ticker:
            continue
        if verdict and rep.get("verdict") != verdict:
            continue
        out.append(rep)
        if len(out) >= limit:
            break
    return out


@app.get("/api/causal/latest")
def causal_latest() -> Optional[dict]:
    root = DATA / "causal"
    snaps = sorted(root.glob("snapshot_*.json")) if root.is_dir() else []
    return json.loads(snaps[-1].read_text()) if snaps else None


# --------------------------------------------------------------------------- #
# Brain status
# --------------------------------------------------------------------------- #

@app.get("/api/status")
def status() -> dict:
    store = _store()
    cat = store.catalog()
    newest = max((v.get("max_time") or "" for v in cat.values()), default=None)

    ckpts = []
    for layer, pattern in (("perception", "checkpoints/perception/best.pt"),
                           ("worldmodel", "checkpoints/worldmodel/best.pt"),
                           ("decision", "checkpoints/decision/best.pt")):
        p = Path(pattern)
        if p.is_file():
            ckpts.append({"layer": layer, "path": str(p),
                          "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                          "steps": None})

    training: dict[str, Any] = {}
    runs = Path("runs")
    if runs.is_dir():
        for f in runs.glob("*.jsonl"):
            steps, totals, best_val = [], [], None
            for line in f.read_text().splitlines()[-400:]:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = rec.get("step") or rec.get("update")
                tot = rec.get("train_total") or rec.get("total") or rec.get("reward_mean")
                v = rec.get("val_total")
                if v is not None:
                    best_val = min(best_val, float(v)) if best_val is not None else float(v)
                if s is not None and tot is not None:
                    steps.append(float(s)); totals.append(float(tot))
            training[f.stem] = {
                "last_step": steps[-1] if steps else None,
                "last_total": totals[-1] if totals else None,
                "best_val": best_val,
                "series": [{"step": s, "total": t} for s, t in zip(steps[-120:], totals[-120:])],
            }

    try:
        from ..evolution.diagnosis import SelfDiagnosis
        rep = SelfDiagnosis().scan()
        diagnosis = {"healthy": bool(rep.healthy),
                     "findings": [{"kind": f.kind, "severity": f.severity, "message": f.message}
                                  for f in rep.findings][:20]}
    except Exception as exc:  # diagnosis must never take the API down
        diagnosis = {"healthy": True, "findings": [{"kind": "diagnosis_error",
                                                    "severity": "info", "message": str(exc)[:200]}]}

    regime = {t.alias: _regime_label(store.read("bars_daily", t.alias,
                                                 columns=["date", "close"]))
              for t in TICKERS}

    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lake": {"datasets": len({k.split('/')[0] for k in cat}),
                     "rows": int(sum(v.get("rows", 0) for v in cat.values())),
                     "newest_bar": str(newest)[:19] if newest else None},
            "checkpoints": ckpts, "training": training,
            "diagnosis": diagnosis, "regime": regime}


# --------------------------------------------------------------------------- #
# Playground: imagination + replay
# --------------------------------------------------------------------------- #

class ImagineBody(BaseModel):
    ticker: str
    anchor_ts: int
    n: int = 20
    horizon: int = 30


@app.post("/api/playground/imagine")
def imagine(body: ImagineBody) -> dict:
    emb = _emb(body.ticker)
    if emb is None:
        raise HTTPException(404, f"no embeddings for {body.ticker}")
    idx = int(np.searchsorted(emb["anchor_ts"], body.anchor_ts, side="right")) - 1
    if idx < 8:
        raise HTTPException(400, "anchor too early — not enough context")
    try:
        import torch
        from ..worldmodel.dynamics import RSSM
        from ..worldmodel.interfaces import DynamicsConfig
        ck = torch.load("checkpoints/worldmodel/best.pt", map_location="cpu",
                        weights_only=False)
        cfg = DynamicsConfig(**{k: v for k, v in (ck.get("cfg") or {}).items()
                                if k in DynamicsConfig.__dataclass_fields__})
        dyn = RSSM(cfg); dyn.load_state_dict(ck["model"]); dyn.eval()
        w0 = max(0, idx - 63)
        seq = torch.from_numpy(emb["fused"][w0:idx + 1]).float().unsqueeze(0)
        with torch.no_grad():
            obs = dyn.observe(seq)
            from ..worldmodel.interfaces import LatentState
            start = LatentState(obs.states.deter[:, -1], obs.states.stoch[:, -1])
            roll = dyn.imagine(start, horizon=body.horizon, n=body.n)
        paths = views.project_rollouts(roll.embeds)  # [N, H] latent divergence
        arr = np.asarray(paths, dtype=float)
    except FileNotFoundError:
        raise HTTPException(409, "world-model checkpoint missing — train it first")
    q = lambda p: np.quantile(arr, p, axis=0).tolist()
    return {"ticker": body.ticker, "anchor_ts": int(emb["anchor_ts"][idx]),
            "horizon": body.horizon, "paths": arr.tolist(),
            "quantiles": {"p10": q(0.10), "p50": q(0.50), "p90": q(0.90)}}


class _Hub:
    """WebSocket fan-out + the session replay engine."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.replay_task: Optional[asyncio.Task] = None
        self.replay_state = {"running": False, "ticker": None, "date": None,
                             "cursor": None, "speed": 1.0}

    async def broadcast(self, event: dict) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def replay(self, ticker: str, date: str, speed: float) -> None:
        """Stream one recorded session bar-by-bar; push the backtest's real
        signals/trades as their timestamps pass. 1 bar = 60/speed seconds."""
        bars_ = bars(ticker, date, "1min")
        name = _newest_backtest()
        sig_by_ts: dict[int, list] = {}
        trade_by_ts: dict[int, list] = {}
        if name:
            r = _backtest(name)
            for s in r.get("signals") or []:
                if s["ticker"] == ticker:
                    sig_by_ts.setdefault(int(s["ts"]), []).append(s)
            for t in r.get("trades") or []:
                if t["ticker"] == ticker:
                    trade_by_ts.setdefault(int(t["exit_ts"]), []).append(t)
        self.replay_state.update(running=True, ticker=ticker, date=date, speed=speed)
        await self.broadcast({"type": "replay_state", "payload": self.replay_state})
        try:
            for bar in bars_:
                self.replay_state["cursor"] = bar["t"]
                await self.broadcast({"type": "bar", "ticker": ticker, "payload": bar})
                for s in sig_by_ts.get(bar["t"], []):
                    await self.broadcast({"type": "signal", "payload": signals_shape(s)})
                for t in trade_by_ts.get(bar["t"], []):
                    await self.broadcast({"type": "trade", "payload": {**t, "has_autopsy":
                                          str(t.get("trade_id")) in _autopsy_ids()}})
                await asyncio.sleep(max(0.02, 60.0 / speed / 60.0))
        finally:
            self.replay_state.update(running=False, cursor=None)
            await self.broadcast({"type": "replay_state", "payload": self.replay_state})


def signals_shape(s: dict) -> dict:
    """Reshape a stored signal into the wire schema (drivers list, clean NaNs)."""
    ev = s.get("evidence") or {}
    return {**{k: s[k] for k in ("signal_id", "ts", "ticker", "side", "conviction",
                                  "entry_px", "stop_px", "target_px", "horizon_bars",
                                  "size_frac")},
            "rationale": s.get("rationale", ""),
            "drivers": [{"name": d.get("name", "?"),
                         "attribution": float(d.get("attribution", 0) or 0)}
                        for d in (ev.get("causal_drivers") or [])[:6]],
            "evidence": None}


HUB = _Hub()


class ReplayBody(BaseModel):
    action: str
    ticker: str = "AAPL"
    date: str = ""
    speed: float = 60.0  # 60x -> one session in ~6.5 minutes


@app.post("/api/playground/replay")
async def replay_ctl(body: ReplayBody) -> dict:
    if body.action == "stop":
        if HUB.replay_task:
            HUB.replay_task.cancel()
            HUB.replay_task = None
        return {"ok": True}
    if HUB.replay_task and not HUB.replay_task.done():
        HUB.replay_task.cancel()
    date = body.date or dates(body.ticker)[-1]
    HUB.replay_task = asyncio.create_task(HUB.replay(body.ticker, date, body.speed))
    return {"ok": True}


class FeedbackBody(BaseModel):
    context: str
    ticker: str = ""
    signal_id: str = ""
    rating: int
    comment: str = ""


@app.post("/api/feedback")
def feedback(body: FeedbackBody) -> dict:
    views.append_feedback(DATA, body.model_dump())
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    HUB.clients.add(sock)
    await sock.send_text(json.dumps({"type": "hello",
                                     "server_time": time.strftime("%H:%M:%S")}))
    try:
        while True:
            # Heartbeat keeps intermediaries from idling the socket out; the
            # client never needs to send anything.
            try:
                await asyncio.wait_for(sock.receive_text(), timeout=10.0)
            except asyncio.TimeoutError:
                await sock.send_text(json.dumps({"type": "heartbeat",
                                                 "server_time": time.strftime("%H:%M:%S")}))
    except WebSocketDisconnect:
        pass
    finally:
        HUB.clients.discard(sock)
