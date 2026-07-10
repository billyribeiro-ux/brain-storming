"""Pure data-access and figure builders for the Aether dashboard (Layer 6).

Deliberately Streamlit-free: every function here takes plain inputs (paths,
dicts, DataFrames, arrays) and returns plotly Figures, DataFrames or dicts, so
the whole module is unit-testable without a Streamlit runtime. ``app.py`` is a
thin presentation shell over these builders.

Defensive by design: sibling layers (backtester, causal fitter, paper trader,
autopsy engine) are built concurrently, so every loader tolerates missing or
malformed artifacts and codes against the *dict shapes* in the interface
contracts, never against sibling code.

The only write path in the entire dashboard is :func:`append_feedback`.
"""

from __future__ import annotations

import json
import math
import time as _time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Palette (validated categorical order — see dataviz reference palette)
# --------------------------------------------------------------------------- #

SERIES = ("#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834")
GOOD = "#0ca30c"          # status colors — reserved, never used as "series 4"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
POS = "#2a78d6"           # diverging pair: blue <-> red, gray midpoint
NEG = "#d03b3b"
MUTED = "#898781"
GRID = "#e1e0d9"
SEQ_BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b")

VERDICT_COLORS = {
    "good_win": GOOD,
    "good_loss": "#2a78d6",   # a good loss is process-positive: cool, not red
    "bad_loss": CRITICAL,
    "lucky_win": WARNING,
}


def _seq_colorscale() -> list:
    steps = ("#fcfcfb",) + SEQ_BLUES
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def _base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=48, r=24, t=48 if title else 24, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=24, b=24),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------- #
# Small shape-tolerant helpers
# --------------------------------------------------------------------------- #

def _as_dict(obj: Any) -> dict:
    """Record -> dict, tolerating dataclass instances and mappings."""
    if isinstance(obj, Mapping):
        return dict(obj)
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def _get(d: Mapping, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _num(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _to_dt(value: Any):
    """Epoch-seconds (naive-ET lake convention) or parseable stamp -> Timestamp."""
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.to_datetime(int(value), unit="s")
        return pd.to_datetime(value)
    except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return None


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(Path(path).read_text())
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Robust JSONL reader — malformed lines are skipped, never fatal."""
    out: list[dict] = []
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_lake_census(data_root: str | Path) -> pd.DataFrame:
    """Everything the lake holds, one row per (dataset, ticker)."""
    from aether.data.storage import ParquetStore  # lazy: keeps import light
    cat = ParquetStore(Path(data_root)).catalog()
    rows = []
    for key, entry in (cat or {}).items():
        dataset, _, ticker = str(key).partition("/")
        entry = entry if isinstance(entry, Mapping) else {}
        rows.append({
            "dataset": dataset,
            "ticker": ticker,
            "rows": int(entry.get("rows") or 0),
            "min_time": str(entry.get("min_time") or ""),
            "max_time": str(entry.get("max_time") or ""),
        })
    df = pd.DataFrame(rows, columns=["dataset", "ticker", "rows", "min_time", "max_time"])
    return df.sort_values(["dataset", "ticker"]).reset_index(drop=True) if len(df) else df


def load_quality(data_root: str | Path) -> dict | None:
    return _read_json(Path(data_root) / "quality_report.json")


def load_capabilities(data_root: str | Path) -> dict | None:
    return _read_json(Path(data_root) / "capabilities.json")


def quality_summary(quality: Mapping | None) -> pd.DataFrame:
    """Issue counts per report section, tolerant of unknown report shapes."""
    def count(val: Any) -> int:
        if isinstance(val, list):
            return len(val)
        if isinstance(val, Mapping):
            issues = val.get("issues")
            if isinstance(issues, list):
                return len(issues)
            return sum(count(v) for v in val.values()
                       if isinstance(v, (list, Mapping)))
        return 0

    rows = []
    if isinstance(quality, Mapping):
        for key, val in quality.items():
            rows.append({"section": str(key), "issues": count(val)})
    return pd.DataFrame(rows, columns=["section", "issues"])


def available_dates(store, ticker: str, dataset: str = "bars_1min") -> list[str]:
    """Sorted session dates present in the lake for a ticker."""
    try:
        df = store.read(dataset, ticker, columns=["date"])
    except Exception:
        return []
    if df is None or len(df) == 0 or "date" not in df.columns:
        return []
    return sorted({str(d) for d in pd.to_datetime(df["date"]).dt.date})


def checkpoint_inventory(root: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(root)
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            st_ = path.stat()
            rows.append({
                "checkpoint": str(path.relative_to(root)),
                "size_mb": round(st_.st_size / 1e6, 3),
                "modified": pd.Timestamp(st_.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            })
    df = pd.DataFrame(rows, columns=["checkpoint", "size_mb", "modified"])
    return df.sort_values("modified", ascending=False).reset_index(drop=True) if len(df) else df


def load_backtests(data_root: str | Path) -> dict[str, dict]:
    """data/backtests/<name>/{result.json, equity.parquet} -> {name: entry}."""
    root = Path(data_root) / "backtests"
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        entry: dict[str, Any] = {"name": d.name, "path": str(d),
                                 "result": _read_json(d / "result.json"), "equity": None}
        eq = d / "equity.parquet"
        if eq.is_file():
            try:
                entry["equity"] = pd.read_parquet(eq)
            except Exception:
                entry["equity"] = None
        out[d.name] = entry
    return out


def load_autopsies(data_root: str | Path) -> list[dict]:
    """Every data/autopsies/<trade_id>.json, newest exits first."""
    root = Path(data_root) / "autopsies"
    reports: list[dict] = []
    if not root.is_dir():
        return reports
    for path in sorted(root.glob("*.json")):
        rep = _read_json(path)
        if rep is not None:
            rep = dict(rep)
            rep["_file"] = path.stem
            reports.append(rep)

    def exit_ts(rep: dict) -> float:
        trade = rep.get("trade") if isinstance(rep.get("trade"), Mapping) else {}
        return _num(trade.get("exit_ts")) or 0.0

    return sorted(reports, key=exit_ts, reverse=True)


def load_blotter(blotter_path: str | Path) -> pd.DataFrame | None:
    recs = _read_jsonl(Path(blotter_path))
    return pd.DataFrame(recs) if recs else None


def load_paper_state(state_path: str | Path) -> dict | None:
    return _read_json(Path(state_path))


def load_lessons(data_root: str | Path) -> pd.DataFrame | None:
    recs = _read_jsonl(Path(data_root) / "lessons.jsonl")
    return pd.DataFrame(recs) if recs else None


def load_feedback(data_root: str | Path) -> pd.DataFrame | None:
    recs = _read_jsonl(Path(data_root) / "feedback.jsonl")
    return pd.DataFrame(recs) if recs else None


def append_feedback(data_root: str | Path, entry: dict) -> None:
    """Append one human-feedback record — the dashboard's ONLY write path."""
    rec = dict(entry)
    rec.setdefault("ts", int(_time.time()))
    rec.setdefault("ts_iso", pd.Timestamp.now().isoformat(timespec="seconds"))
    path = Path(data_root) / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Embeddings (npz produced by decision.env.EmbeddingStore)
# --------------------------------------------------------------------------- #

def load_embeddings(cache_dir: str | Path, ticker: str) -> dict[str, np.ndarray] | None:
    path = Path(cache_dir) / f"{ticker}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def embeddings_session_slice(embs: Mapping[str, np.ndarray] | None, date) -> dict[str, np.ndarray]:
    """Restrict per-anchor arrays to one session date (naive-ET epoch convention)."""
    if not embs or "anchor_ts" not in embs:
        return {}
    ts = np.asarray(embs["anchor_ts"]).astype("int64")
    day = pd.Timestamp(date).normalize()
    t0 = int(day.timestamp())            # naive stamp treated as UTC == lake convention
    mask = (ts >= t0) & (ts < t0 + 86400)
    if not mask.any():
        return {}
    n = ts.shape[0]
    out = {}
    for key, val in embs.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask]
    return out


def embedding_dates(embs: Mapping[str, np.ndarray] | None) -> list[str]:
    if not embs or "anchor_ts" not in embs:
        return []
    ts = pd.to_datetime(np.asarray(embs["anchor_ts"]).astype("int64"), unit="s")
    return sorted({str(d) for d in ts.date})


def uncertainty_figure(session_embs: Mapping[str, np.ndarray]) -> go.Figure | None:
    """Aleatoric / epistemic / anomaly strip for one session, aligned on time."""
    keys = [k for k in ("aleatoric", "epistemic", "anomaly") if k in (session_embs or {})]
    if not keys or "anchor_ts" not in (session_embs or {}):
        return None
    x = pd.to_datetime(np.asarray(session_embs["anchor_ts"]).astype("int64"), unit="s")
    fig = go.Figure()
    for i, key in enumerate(keys):
        fig.add_trace(go.Scatter(
            x=x, y=np.asarray(session_embs[key], dtype=float), mode="lines",
            name=key, line=dict(width=2, color=SERIES[i]),
        ))
    _base_layout(fig, "Model uncertainty (perception heads)", height=240)
    fig.update_yaxes(title_text="score")
    return fig


# --------------------------------------------------------------------------- #
# Candles + overlays
# --------------------------------------------------------------------------- #

def _saliency_scores(saliency: Any, bar_times: pd.Series) -> np.ndarray | None:
    """Align saliency (sequence or {ts: score} mapping) to the session bars."""
    if saliency is None:
        return None
    n = len(bar_times)
    if isinstance(saliency, Mapping):
        lookup = {}
        for k, v in saliency.items():
            dt = _to_dt(_num(k) if _num(k) is not None else k)
            val = _num(v)
            if dt is not None and val is not None:
                lookup[dt] = val
        if not lookup:
            return None
        return np.array([lookup.get(t, np.nan) for t in bar_times], dtype=float)
    try:
        arr = np.asarray(saliency, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    out = np.full(n, np.nan)
    out[: min(n, arr.size)] = arr[: min(n, arr.size)]
    return out


def candles_figure(store, ticker: str, date, overlays: dict | None = None) -> go.Figure:
    """1-min candlestick + volume for one session, with optional overlays.

    ``overlays`` (all optional, all shape-tolerant):
      signals: list of dicts with ts / side / entry(_px) / stop(_px) / target(_px)
      trades:  list of dicts with entry_ts / exit_ts / entry_px / exit_px / side / pnl
      saliency: per-bar scores (sequence aligned to session bars, or {ts: score})
    """
    overlays = overlays or {}
    day = pd.Timestamp(date).normalize()
    try:
        bars = store.read("bars_1min", ticker, start=day,
                          end=day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    except Exception:
        bars = pd.DataFrame()
    if bars is None or len(bars) == 0 or "date" not in getattr(bars, "columns", []):
        return _empty_figure(f"No 1min bars for {ticker} on {day.date()}")

    x = pd.to_datetime(bars["date"])
    scores = _saliency_scores(overlays.get("saliency"), x)
    has_sal = scores is not None and np.isfinite(scores).any()

    rows = 3 if has_sal else 2
    heights = [0.60, 0.10, 0.30] if has_sal else [0.68, 0.32]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights)

    fig.add_trace(go.Candlestick(
        x=x, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
        name=f"{ticker} 1min",
        increasing=dict(line=dict(color=GOOD, width=1), fillcolor=GOOD),
        decreasing=dict(line=dict(color=CRITICAL, width=1), fillcolor=CRITICAL),
    ), row=1, col=1)

    if has_sal:
        fig.add_trace(go.Heatmap(
            z=[np.nan_to_num(scores, nan=0.0)], x=x, y=["saliency"],
            colorscale=_seq_colorscale(), showscale=False,
            hovertemplate="%{x|%H:%M} saliency=%{z:.3f}<extra></extra>",
        ), row=2, col=1)

    up = (bars["close"].to_numpy() >= bars["open"].to_numpy())
    vol_colors = np.where(up, "rgba(12,163,12,0.55)", "rgba(208,59,59,0.55)")
    fig.add_trace(go.Bar(
        x=x, y=bars["volume"], name="volume", marker_color=vol_colors, showlegend=False,
        hovertemplate="%{x|%H:%M} vol=%{y:,.0f}<extra></extra>",
    ), row=rows, col=1)

    shown: set[str] = set()

    def _legend(group: str) -> bool:
        first = group not in shown
        shown.add(group)
        return first

    # -- signal overlays --------------------------------------------------- #
    for sig in overlays.get("signals") or []:
        d = _as_dict(sig)
        ts = _to_dt(_get(d, "ts", "signal_ts"))
        entry = _num(_get(d, "entry", "entry_px"))
        if ts is None or entry is None:
            continue
        side = str(_get(d, "side", default="long")).lower()
        stop = _num(_get(d, "stop", "stop_px"))
        target = _num(_get(d, "target", "target_px"))
        horizon = int(_num(_get(d, "horizon", "horizon_bars", default=30)) or 30)
        conviction = _num(_get(d, "conviction"))
        x1 = ts + pd.Timedelta(minutes=horizon)
        fig.add_trace(go.Scatter(
            x=[ts], y=[entry], mode="markers", name="signal",
            legendgroup="signal", showlegend=_legend("signal"),
            marker=dict(symbol="triangle-up" if side == "long" else "triangle-down",
                        size=13, color=GOOD if side == "long" else CRITICAL,
                        line=dict(width=1, color="#ffffff")),
            hovertext=f"signal {side} @ {entry:.2f}"
                      + (f" (conviction {conviction:.2f})" if conviction is not None else ""),
            hoverinfo="text",
        ), row=1, col=1)
        for level, role, dash, color in ((entry, "entry", "solid", MUTED),
                                         (stop, "stop", "dot", CRITICAL),
                                         (target, "target", "dot", GOOD)):
            if level is None:
                continue
            fig.add_trace(go.Scatter(
                x=[ts, x1], y=[level, level], mode="lines",
                name=f"signal {role}", legendgroup=f"sig-{role}",
                showlegend=_legend(f"sig-{role}"),
                line=dict(width=1.5, dash=dash, color=color), hoverinfo="skip",
            ), row=1, col=1)

    # -- trade overlays ---------------------------------------------------- #
    for trade in overlays.get("trades") or []:
        d = _as_dict(trade)
        t0, t1 = _to_dt(d.get("entry_ts")), _to_dt(d.get("exit_ts"))
        p0, p1 = _num(d.get("entry_px")), _num(d.get("exit_px"))
        if t0 is None or p0 is None:
            continue
        side = str(_get(d, "side", default="long")).lower()
        pnl = _num(d.get("pnl"))
        won = pnl is not None and pnl >= 0
        fig.add_trace(go.Scatter(
            x=[t0], y=[p0], mode="markers", name="trade entry",
            legendgroup="trade-entry", showlegend=_legend("trade-entry"),
            marker=dict(symbol="triangle-up" if side == "long" else "triangle-down",
                        size=12, color="#4a3aa7", line=dict(width=1, color="#ffffff")),
            hovertext=f"entry {side} @ {p0:.2f}", hoverinfo="text",
        ), row=1, col=1)
        if t1 is not None and p1 is not None:
            fig.add_trace(go.Scatter(
                x=[t1], y=[p1], mode="markers", name="trade exit",
                legendgroup="trade-exit", showlegend=_legend("trade-exit"),
                marker=dict(symbol="x", size=11,
                            color=GOOD if won else CRITICAL),
                hovertext=f"exit @ {p1:.2f}"
                          + (f", pnl {pnl:+.2f}" if pnl is not None else "")
                          + (f" ({d.get('exit_reason')})" if d.get("exit_reason") else ""),
                hoverinfo="text",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[p0, p1], mode="lines", legendgroup="trade-path",
                name="trade path", showlegend=_legend("trade-path"),
                line=dict(width=1.5, dash="dash", color=GOOD if won else CRITICAL),
                hoverinfo="skip",
            ), row=1, col=1)

    fig.update_layout(
        title=f"{ticker} — {day.date()} (1min)",
        height=640, margin=dict(l=48, r=24, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="vol", row=rows, col=1)
    if has_sal:
        fig.update_yaxes(showticklabels=False, row=2, col=1)
    return fig


# --------------------------------------------------------------------------- #
# Equity / drawdown / stats
# --------------------------------------------------------------------------- #

def _equity_xy(equity_df) -> tuple[Any, Any]:
    if equity_df is None or len(equity_df) == 0:
        return None, None
    df = pd.DataFrame(equity_df)
    tcol = next((c for c in ("ts", "date", "time", "timestamp") if c in df.columns), None)
    x = df[tcol].map(_to_dt) if tcol else df.index
    ycol = next((c for c in ("equity", "value", "nav") if c in df.columns), None)
    if ycol is None:
        nums = df.select_dtypes(include="number").columns
        ycol = nums[0] if len(nums) else None
    if ycol is None:
        return None, None
    return x, pd.to_numeric(df[ycol], errors="coerce")


def equity_figure(equity_df) -> go.Figure:
    x, y = _equity_xy(equity_df)
    if y is None:
        return _empty_figure("No equity curve")
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="equity",
                               line=dict(width=2, color=SERIES[0])))
    _base_layout(fig, "Equity curve", height=340)
    fig.update_yaxes(title_text="equity ($)", tickformat=",.0f")
    return fig


def drawdown_figure(equity_df) -> go.Figure:
    x, y = _equity_xy(equity_df)
    if y is None:
        return _empty_figure("No equity curve")
    eq = y.to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.where(np.isfinite(eq), eq, -np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, eq / peak - 1.0, 0.0)
    fig = go.Figure(go.Scatter(x=x, y=dd, mode="lines", name="drawdown",
                               line=dict(width=1.5, color=CRITICAL),
                               fill="tozeroy", fillcolor="rgba(208,59,59,0.18)"))
    _base_layout(fig, "Drawdown", height=260)
    fig.update_yaxes(title_text="drawdown", tickformat=".1%")
    return fig


def stats_table(stats: Mapping | None) -> pd.DataFrame:
    """Flatten a (possibly nested) stats dict into a metric/value table."""
    rows: list[tuple[str, Any]] = []

    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, Mapping):
            for k, v in val.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(val, (list, tuple)):
            rows.append((prefix, json.dumps(val, default=str)[:120]))
        else:
            if isinstance(val, float):
                val = round(val, 6)
            rows.append((prefix, val))

    walk("", stats or {})
    return pd.DataFrame(rows, columns=["metric", "value"])


def trades_table(trades: Iterable | None) -> pd.DataFrame:
    """TradeRecord dicts (or dataclasses) -> display table."""
    rows = []
    for t in trades or []:
        d = _as_dict(t)
        if not d:
            continue
        row = dict(d)
        for key in ("entry_ts", "exit_ts", "ts"):
            dt = _to_dt(row.get(key))
            if dt is not None:
                row[key] = dt.strftime("%Y-%m-%d %H:%M")
        for key in ("pnl", "fees", "entry_px", "exit_px", "stop_px", "target_px", "conviction"):
            v = _num(row.get(key))
            if v is not None:
                row[key] = round(v, 4)
        row.pop("signal_meta", None)
        rows.append(row)
    df = pd.DataFrame(rows)
    preferred = [c for c in ("trade_id", "ticker", "side", "entry_ts", "exit_ts",
                             "entry_px", "exit_px", "qty", "pnl", "exit_reason",
                             "conviction") if c in df.columns]
    return df[preferred + [c for c in df.columns if c not in preferred]] if len(df) else df


def filter_records(records: Iterable | None, ticker: str | None = None, date=None,
                   ts_keys: tuple[str, ...] = ("ts", "entry_ts")) -> list[dict]:
    """Filter signal/trade dicts to one ticker and/or one session date."""
    day = pd.Timestamp(date).date() if date is not None else None
    out = []
    for rec in records or []:
        d = _as_dict(rec)
        if not d:
            continue
        if ticker and str(d.get("ticker", "")) != str(ticker):
            continue
        if day is not None:
            ts = next((d.get(k) for k in ts_keys if d.get(k) is not None), None)
            dt = _to_dt(ts)
            if dt is None or dt.date() != day:
                continue
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Causal graph
# --------------------------------------------------------------------------- #

def _snapshot_edges(snapshot: Mapping | None) -> list[dict]:
    edges = []
    for e in (snapshot or {}).get("edges") or []:
        d = _as_dict(e)
        src, dst = str(d.get("src", "")), str(d.get("dst", ""))
        w, conf = _num(d.get("weight")), _num(d.get("confidence"))
        if not src or not dst or w is None:
            continue
        conf = conf if conf is not None else 0.0
        edges.append({"src": src, "dst": dst, "lag": int(_num(d.get("lag")) or 0),
                      "weight": w, "confidence": conf, "strength": abs(w) * conf})
    return edges


def edges_table(snapshot: Mapping | None) -> pd.DataFrame:
    df = pd.DataFrame(_snapshot_edges(snapshot),
                      columns=["src", "dst", "lag", "weight", "confidence", "strength"])
    if len(df):
        df = df.sort_values("strength", ascending=False).reset_index(drop=True)
        df[["weight", "confidence", "strength"]] = df[["weight", "confidence", "strength"]].round(4)
    return df


def causal_figure(snapshot: Mapping | None, max_edges: int = 120) -> go.Figure:
    """Directed causal graph: spring layout, width prop. to |weight|*confidence,
    color by sign (blue positive / red negative), arrows via annotations."""
    import networkx as nx  # lazy

    edges = sorted(_snapshot_edges(snapshot), key=lambda e: -e["strength"])[:max_edges]
    nodes = [str(n) for n in (snapshot or {}).get("nodes") or []]
    nodes = sorted(set(nodes) | {e["src"] for e in edges} | {e["dst"] for e in edges})
    if not nodes:
        return _empty_figure("Empty causal snapshot")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e["src"], e["dst"])
    pos = nx.spring_layout(G, seed=7, k=1.6 / math.sqrt(max(len(nodes), 1)))

    max_s = max((e["strength"] for e in edges), default=1.0) or 1.0
    fig = go.Figure()
    annotations, mx, my, mtext = [], [], [], []
    for e in edges:
        (x0, y0), (x1, y1) = pos[e["src"]], pos[e["dst"]]
        width = 0.8 + 5.0 * e["strength"] / max_s
        color = POS if e["weight"] >= 0 else NEG
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1, axref="x", ayref="y", xref="x", yref="y",
            showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=width,
            arrowcolor=color, opacity=min(1.0, 0.35 + 0.65 * e["confidence"]),
            standoff=10, startstandoff=6,
        ))
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        mtext.append(f"{e['src']} -> {e['dst']} (lag {e['lag']})"
                     f"<br>weight {e['weight']:+.3f}, confidence {e['confidence']:.2f}")
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", hovertext=mtext, hoverinfo="text",
                             marker=dict(size=9, opacity=0.0), showlegend=False))

    prefixes = sorted({n.split(".", 1)[0] for n in nodes})
    prefix_color = {p: (SERIES[i] if i < len(SERIES) else MUTED)
                    for i, p in enumerate(prefixes)}
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(size=10, color=MUTED),
        marker=dict(size=13, color=[prefix_color[n.split(".", 1)[0]] for n in nodes],
                    line=dict(width=1, color="#ffffff")),
        hovertext=nodes, hoverinfo="text", showlegend=False,
    ))
    n_total = len(_snapshot_edges(snapshot))
    subtitle = f"{len(edges)} of {n_total} edges shown" if n_total > len(edges) else f"{len(edges)} edges"
    fig.update_layout(
        title=f"Causal graph — blue positive, red negative, width = |weight| x confidence ({subtitle})",
        annotations=annotations, height=620,
        margin=dict(l=24, r=24, t=48, b=24),
        xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
    )
    return fig


def driver_heatmap(snapshot: Mapping | None, tickers: Iterable[str] | None = None) -> go.Figure:
    """Incoming edge strength (sum over lags of |weight|*confidence): dst-channel x src."""
    edges = _snapshot_edges(snapshot)
    if not edges:
        return _empty_figure("Empty causal snapshot")
    aliases = {str(t) for t in tickers} if tickers else None
    dst_nodes = sorted({e["dst"] for e in edges})
    if aliases:
        filtered = [n for n in dst_nodes if n.split(".", 1)[0] in aliases]
        dst_nodes = filtered or dst_nodes
    src_nodes = sorted({e["src"] for e in edges})
    mat = np.zeros((len(dst_nodes), len(src_nodes)))
    d_idx = {n: i for i, n in enumerate(dst_nodes)}
    s_idx = {n: i for i, n in enumerate(src_nodes)}
    for e in edges:
        if e["dst"] in d_idx and e["src"] in s_idx:
            mat[d_idx[e["dst"]], s_idx[e["src"]]] += e["strength"]
    fig = go.Figure(go.Heatmap(
        z=mat, x=src_nodes, y=dst_nodes, colorscale=_seq_colorscale(),
        colorbar=dict(title="strength", thickness=12),
        hovertemplate="%{x} -> %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Driver heatmap — incoming edge strength (sum of |weight| x confidence)",
        height=max(360, 24 * len(dst_nodes) + 140),
        margin=dict(l=140, r=24, t=48, b=110),
        xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Imagination (world-model rollouts)
# --------------------------------------------------------------------------- #

def project_rollouts(embeds) -> np.ndarray:
    """Project imagined embedding trajectories [N, 1, H, D] (or [N, H, D]) to
    scalar paths [N, H] via the norm-delta projection:

        path[n, t] = || e[n, t] - e[n, 0] ||_2

    This is a LATENT-SPACE DIVERGENCE proxy — how far the imagined world state
    drifts from the start state — NOT a price forecast. Spread across rollouts
    reads as model uncertainty about how the situation evolves.
    """
    if hasattr(embeds, "detach"):  # torch tensor without importing torch
        embeds = embeds.detach().cpu().numpy()
    arr = np.asarray(embeds, dtype=np.float64)
    if arr.ndim == 4:              # [N, B, H, D] -> first batch element
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"expected [N,1,H,D] or [N,H,D] embeds, got shape {arr.shape}")
    return np.linalg.norm(arr - arr[:, :1, :], axis=-1)


def imagination_fanchart(fused_traj: np.ndarray) -> go.Figure:
    """Per-rollout projected paths + 10-90% quantile band + median."""
    arr = np.asarray(fused_traj, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.size == 0:
        return _empty_figure("No rollouts to display")
    n, horizon = arr.shape
    x = np.arange(1, horizon + 1)

    fig = go.Figure()
    xs: list[Any] = []
    ys: list[Any] = []
    for i in range(n):                      # one trace, None-separated: fast
        xs.extend(x.tolist() + [None])
        ys.extend(arr[i].tolist() + [None])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="rollouts",
                             line=dict(width=1, color="rgba(42,120,214,0.30)"),
                             hoverinfo="skip"))
    q10, q50, q90 = np.nanquantile(arr, [0.10, 0.50, 0.90], axis=0)
    fig.add_trace(go.Scatter(x=x, y=q90, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q10, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.18)",
                             name="10-90% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q50, mode="lines", name="median",
                             line=dict(width=2.5, color=SERIES[0]),
                             hovertemplate="t+%{x}: %{y:.3f}<extra>median</extra>"))
    _base_layout(fig, f"Imagined futures — {n} rollouts, {horizon} bars", height=420)
    fig.update_xaxes(title_text="bars ahead")
    fig.update_yaxes(title_text="latent divergence  ||e(t) - e(0)||")
    fig.add_annotation(
        text="Latent-space divergence proxy (world-model rollouts) — not prices.",
        xref="paper", yref="paper", x=0, y=-0.18, showarrow=False,
        font=dict(size=11, color=MUTED),
    )
    return fig


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #

def training_curves(runs_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Parse every runs/*.jsonl into a DataFrame of step + numeric fields.
    Robust to malformed lines and non-numeric fields."""
    out: dict[str, pd.DataFrame] = {}
    runs = Path(runs_dir)
    if not runs.is_dir():
        return out
    for path in sorted(runs.glob("*.jsonl")):
        rows = []
        for rec in _read_jsonl(path):
            if "step" not in rec:
                continue
            row = {}
            for k, v in rec.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if math.isfinite(float(v)):
                    row[k] = float(v)
            if "step" in row:
                rows.append(row)
        if rows:
            out[path.stem] = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return out


def training_curve_figure(df: pd.DataFrame, metrics: Iterable[str],
                          title: str = "") -> go.Figure:
    metrics = [m for m in metrics if m in df.columns][:8]  # fixed slots, never cycled
    if not metrics or "step" not in df.columns:
        return _empty_figure("No numeric metrics to plot")
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Scatter(x=df["step"], y=df[m], mode="lines", name=m,
                                 line=dict(width=2, color=SERIES[i])))
    _base_layout(fig, title, height=340)
    fig.update_xaxes(title_text="step")
    return fig


# --------------------------------------------------------------------------- #
# Autopsy / memory renderers
# --------------------------------------------------------------------------- #

def counterfactuals_table(counterfactuals: Iterable | None) -> pd.DataFrame:
    rows = []
    for cf in counterfactuals or []:
        d = _as_dict(cf)
        rows.append({"counterfactual": str(d.get("description", "")),
                     "pnl": _num(d.get("pnl")),
                     "delta": _num(d.get("delta"))})
    df = pd.DataFrame(rows, columns=["counterfactual", "pnl", "delta"])
    if len(df):
        df[["pnl", "delta"]] = df[["pnl", "delta"]].round(2)
    return df


def driver_bars_figure(drivers: Iterable | None, title: str = "Perception drivers") -> go.Figure:
    """Horizontal attribution bars, diverging blue/red by sign."""
    names, vals = [], []
    for d in drivers or []:
        dd = _as_dict(d)
        v = _num(_get(dd, "attribution", "score", "weight"))
        name = str(_get(dd, "name", "feature", default=""))
        if name and v is not None:
            names.append(name)
            vals.append(v)
    if not names:
        return _empty_figure("No driver attributions", height=220)
    order = np.argsort(np.abs(vals))          # largest at top of a horizontal bar
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=[POS if v >= 0 else NEG for v in vals],
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    _base_layout(fig, title, height=max(220, 30 * len(names) + 110))
    fig.update_xaxes(title_text="attribution")
    fig.update_layout(showlegend=False)
    return fig


def analogs_table(analogs: Iterable | None) -> pd.DataFrame:
    """MemoryAnalog records (dicts or dataclasses) -> display table."""
    rows = []
    for a in analogs or []:
        d = _as_dict(a)
        if not d:
            continue
        sim = _num(d.get("similarity"))
        dt = _to_dt(d.get("anchor_ts"))
        row = {"similarity": round(sim, 4) if sim is not None else None,
               "ticker": str(d.get("ticker", "")),
               "anchor": dt.strftime("%Y-%m-%d %H:%M") if dt is not None else ""}
        outcome = d.get("outcome")
        if isinstance(outcome, Mapping):
            for k, v in outcome.items():
                v = _num(v)
                row[f"outcome.{k}"] = round(v, 5) if v is not None else None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("similarity", ascending=False).reset_index(drop=True) \
        if len(df) and "similarity" in df.columns else df


# --------------------------------------------------------------------------- #
# Causal snapshot files
# --------------------------------------------------------------------------- #

def list_snapshots(data_root: str | Path) -> list[Path]:
    """Causal snapshot files, newest first (timestamped filenames sort)."""
    root = Path(data_root) / "causal"
    if not root.is_dir():
        return []
    return sorted(root.glob("snapshot_*.json"), reverse=True)


def load_snapshot(path: str | Path) -> dict | None:
    snap = _read_json(Path(path))
    if snap is None or not isinstance(snap.get("edges"), list):
        return None
    return snap
