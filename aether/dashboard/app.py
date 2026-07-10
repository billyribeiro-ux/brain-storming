"""Aether — Market Intelligence Brain: the Streamlit cockpit (Layer 6).

Launch:  streamlit run aether/dashboard/app.py   (or scripts/run_dashboard.py)

Read-only over the lake, checkpoints, runs, blotters, autopsies and causal
snapshots. One write path: human feedback -> data/feedback.jsonl. Every page
degrades gracefully when a sibling layer has not produced its artifact yet,
and tells you the exact command that generates it. All heavy imports (torch,
world-model modules) happen lazily inside the pages that need them.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# --- sys.path bootstrap: `streamlit run aether/dashboard/app.py` executes this
# file as a plain script, so make the repo root importable first.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aether.dashboard import views  # noqa: E402  (needs the bootstrap above)

_DATA_ROOT = Path(os.environ.get("AETHER_DATA_ROOT", "data"))
DATA_ROOT = _DATA_ROOT if _DATA_ROOT.is_absolute() else _REPO_ROOT / _DATA_ROOT
RUNS_DIR = _REPO_ROOT / "runs"
CKPT_ROOT = _REPO_ROOT / "checkpoints"
EMB_DIR = DATA_ROOT / "cache" / "embeddings"
WORLDMODEL_CKPT = CKPT_ROOT / "worldmodel" / "best.pt"

PAGES = ("Overview", "Charts", "Causal Brain", "Playground",
         "Trades & Autopsies", "Paper Blotter", "Feedback", "Training")


def _tickers() -> list[str]:
    from aether.config import TICKERS
    return [t.alias for t in TICKERS]


def _store():
    from aether.data.storage import ParquetStore
    return ParquetStore(DATA_ROOT)


# --------------------------------------------------------------------------- #
# Cached loaders (ttl=60 so artifacts written by sibling processes show up)
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=60)
def cached_census(data_root: str) -> pd.DataFrame:
    return views.load_lake_census(data_root)


@st.cache_data(ttl=60)
def cached_quality(data_root: str):
    return views.load_quality(data_root)


@st.cache_data(ttl=60)
def cached_capabilities(data_root: str):
    return views.load_capabilities(data_root)


@st.cache_data(ttl=60)
def cached_dates(data_root: str, ticker: str) -> list[str]:
    from aether.data.storage import ParquetStore
    return views.available_dates(ParquetStore(data_root), ticker)


@st.cache_data(ttl=60)
def cached_runs(runs_dir: str) -> dict[str, pd.DataFrame]:
    return views.training_curves(runs_dir)


@st.cache_data(ttl=60)
def cached_backtests(data_root: str) -> dict[str, dict]:
    return views.load_backtests(data_root)


@st.cache_data(ttl=60)
def cached_autopsies(data_root: str) -> list[dict]:
    return views.load_autopsies(data_root)


@st.cache_data(ttl=60)
def cached_embeddings(emb_dir: str, ticker: str):
    return views.load_embeddings(emb_dir, ticker)


@st.cache_data(ttl=60)
def cached_blotter(path: str):
    return views.load_blotter(path)


@st.cache_data(ttl=60)
def cached_feedback(data_root: str):
    return views.load_feedback(data_root)


@st.cache_data(ttl=60)
def cached_checkpoints(root: str) -> pd.DataFrame:
    return views.checkpoint_inventory(root)


def _missing(what: str, command: str) -> None:
    st.info(f"{what} not found yet. Generate it with:\n\n`{command}`")


def _badge(label: str, color: str, text_color: str = "#ffffff") -> str:
    return (f"<span style='background:{color};color:{text_color};padding:2px 10px;"
            f"border-radius:10px;font-size:0.85em;white-space:nowrap;'>{label}</span>")


# --------------------------------------------------------------------------- #
# 1 Overview
# --------------------------------------------------------------------------- #

def page_overview() -> None:
    st.header("Overview")

    st.subheader("Data lake census")
    census = cached_census(str(DATA_ROOT))
    if len(census) == 0:
        _missing("Data lake", "python scripts/backfill.py  &&  python scripts/sync.py")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Datasets", census["dataset"].nunique())
        c2.metric("(dataset, ticker) partitions", len(census))
        c3.metric("Total rows", f"{int(census['rows'].sum()):,}")
        st.dataframe(census, width="stretch", height=320)

    st.subheader("Capabilities")
    caps = cached_capabilities(str(DATA_ROOT))
    if not caps:
        _missing("data/capabilities.json", "python scripts/probe_capabilities.py")
    else:
        chips = []
        for name, info in sorted(caps.items()):
            ok = bool(info.get("available")) if isinstance(info, dict) else bool(info)
            chips.append(_badge(f"{name}: {'ok' if ok else 'unavailable'}",
                                views.GOOD if ok else views.CRITICAL))
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    st.subheader("Data quality")
    quality = cached_quality(str(DATA_ROOT))
    if quality is None:
        _missing("data/quality_report.json", "python scripts/audit_quality.py")
    else:
        summary = views.quality_summary(quality)
        total = int(summary["issues"].sum()) if len(summary) else 0
        st.metric("Issues flagged", total)
        if len(summary):
            st.dataframe(summary, width="stretch")
        with st.expander("Raw quality report"):
            st.json(quality)

    st.subheader("Latest training metrics")
    curves = cached_runs(str(RUNS_DIR))
    if not curves:
        _missing("runs/*.jsonl", "python scripts/train_perception.py")
    else:
        rows = []
        for name, df in curves.items():
            last = df.iloc[-1].to_dict()
            row = {"run": name, "step": int(last.pop("step", 0))}
            for k, v in last.items():
                if k != "ts":
                    row[k] = round(float(v), 5)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Checkpoints")
    ckpts = cached_checkpoints(str(CKPT_ROOT))
    if len(ckpts) == 0:
        _missing("checkpoints/", "python scripts/train_perception.py")
    else:
        st.dataframe(ckpts, width="stretch")


# --------------------------------------------------------------------------- #
# 2 Charts
# --------------------------------------------------------------------------- #

def page_charts() -> None:
    st.header("Charts")
    census = cached_census(str(DATA_ROOT))
    have_1min = sorted(census[census["dataset"] == "bars_1min"]["ticker"].unique()) \
        if len(census) else []
    tickers = [t for t in _tickers() if t in have_1min] or have_1min
    if not tickers:
        _missing("1min bars", "python scripts/backfill.py")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    ticker = c1.selectbox("Ticker", tickers)
    dates = cached_dates(str(DATA_ROOT), ticker)
    if not dates:
        _missing(f"1min bars for {ticker}", "python scripts/sync.py")
        return
    date = c2.selectbox("Session", list(reversed(dates)))

    backtests = cached_backtests(str(DATA_ROOT))
    bt_name = c3.selectbox("Overlay source (backtest)", ["(none)"] + sorted(backtests))

    o1, o2, o3 = st.columns(3)
    show_signals = o1.checkbox("Signals", value=True)
    show_trades = o2.checkbox("Trades", value=True)
    embs = cached_embeddings(str(EMB_DIR), ticker)
    has_sal = bool(embs) and "saliency" in embs
    show_sal = o3.checkbox("Saliency strip", value=has_sal, disabled=not has_sal,
                           help="Needs a 'saliency' array in the embeddings npz "
                                f"(data/cache/embeddings/{ticker}.npz).")

    overlays: dict = {}
    if bt_name != "(none)":
        result = backtests[bt_name].get("result") or {}
        if show_signals:
            overlays["signals"] = views.filter_records(
                result.get("signals"), ticker, date, ts_keys=("ts",))
        if show_trades:
            overlays["trades"] = views.filter_records(
                result.get("trades"), ticker, date, ts_keys=("entry_ts",))
    elif show_signals or show_trades:
        st.caption("Select a backtest above to overlay its signals and trades. "
                   "Generate one with `python scripts/run_backtest.py`.")

    session_embs = views.embeddings_session_slice(embs, date) if embs else {}
    if show_sal and has_sal:
        sal = session_embs.get("saliency")
        if sal is not None and "anchor_ts" in session_embs:
            overlays["saliency"] = dict(zip(
                (int(t) for t in session_embs["anchor_ts"]),
                (float(s) for s in sal.reshape(len(sal), -1).mean(axis=1)),
            ))

    st.plotly_chart(views.candles_figure(_store(), ticker, date, overlays),
                    width="stretch")

    if embs is None:
        _missing(f"Embeddings (data/cache/embeddings/{ticker}.npz)",
                 "python scripts/precompute_embeddings.py")
    else:
        unc = views.uncertainty_figure(session_embs)
        if unc is None:
            st.info(f"No embedding anchors for {ticker} on {date} — the uncertainty "
                    "strip needs precomputed embeddings covering this session.")
        else:
            st.plotly_chart(unc, width="stretch")


# --------------------------------------------------------------------------- #
# 3 Causal Brain
# --------------------------------------------------------------------------- #

def page_causal() -> None:
    st.header("Causal Brain")
    snapshots = views.list_snapshots(DATA_ROOT)
    if not snapshots:
        _missing("Causal snapshots (data/causal/snapshot_*.json)",
                 "python scripts/fit_causal.py")
        return
    labels = [p.stem.replace("snapshot_", "") for p in snapshots]
    choice = st.selectbox("Snapshot (newest first)", range(len(snapshots)),
                          format_func=lambda i: labels[i])
    snap = views.load_snapshot(snapshots[choice])
    if snap is None:
        st.error(f"Could not parse {snapshots[choice].name} as a causal snapshot.")
        return
    st.caption(f"Fitted window: {snap.get('fitted_start', '?')} to "
               f"{snap.get('fitted_end', '?')} - {len(snap.get('nodes') or [])} nodes, "
               f"{len(snap.get('edges') or [])} edges")

    st.plotly_chart(views.causal_figure(snap), width="stretch")
    st.plotly_chart(views.driver_heatmap(snap, _tickers()), width="stretch")

    st.subheader("Edges (by |weight| x confidence)")
    edges = views.edges_table(snap)
    if len(edges) == 0:
        st.info("Snapshot contains no edges.")
    else:
        st.dataframe(edges, width="stretch", height=380)


# --------------------------------------------------------------------------- #
# 4 Playground
# --------------------------------------------------------------------------- #

def page_playground() -> None:
    st.header("Playground — imagine futures from a chosen moment")

    ticker = st.selectbox("Ticker", _tickers())
    embs = cached_embeddings(str(EMB_DIR), ticker)
    if embs is None or "fused" not in embs or "anchor_ts" not in embs:
        _missing(f"Embeddings (data/cache/embeddings/{ticker}.npz)",
                 "python scripts/precompute_embeddings.py")
        return
    if not WORLDMODEL_CKPT.is_file():
        _missing("World-model checkpoint (checkpoints/worldmodel/best.pt)",
                 "python scripts/train_worldmodel.py")
        return

    dates = views.embedding_dates(embs)
    c1, c2 = st.columns(2)
    date = c1.selectbox("Session", list(reversed(dates)))
    session = views.embeddings_session_slice(embs, date)
    if "anchor_ts" not in session:
        st.info("No embedding anchors on that session.")
        return
    times = pd.to_datetime(session["anchor_ts"].astype("int64"), unit="s")
    pick = c2.select_slider("Moment", options=list(range(len(times))),
                            value=len(times) - 1,
                            format_func=lambda i: times[i].strftime("%H:%M"))
    anchor_ts = int(session["anchor_ts"][pick])

    all_ts = embs["anchor_ts"].astype("int64")
    idx = int((all_ts == anchor_ts).nonzero()[0][0])
    context = 64
    if idx + 1 < context:
        st.info(f"Need {context} trailing embeddings before this moment "
                f"(have {idx + 1}). Pick a later moment.")
        return

    n_roll, horizon = 20, 30
    if st.button("Imagine", type="primary"):
        try:
            import torch
            from aether.worldmodel.dynamics import LatentDynamics
            from aether.worldmodel.interfaces import DynamicsConfig, LatentState

            fused = embs["fused"]
            ckpt = torch.load(WORLDMODEL_CKPT, map_location="cpu")
            raw_cfg = ckpt.get("config") or ckpt.get("cfg") if isinstance(ckpt, dict) else None
            if isinstance(raw_cfg, DynamicsConfig):
                cfg = raw_cfg
            elif isinstance(raw_cfg, dict):
                fields = set(DynamicsConfig.__dataclass_fields__)
                cfg = DynamicsConfig(**{k: v for k, v in raw_cfg.items() if k in fields})
            else:
                cfg = DynamicsConfig(embed_dim=int(fused.shape[1]))
            model = LatentDynamics(cfg)
            state_dict = None
            if isinstance(ckpt, dict):
                for key in ("model", "state_dict", "model_state_dict"):
                    if isinstance(ckpt.get(key), dict):
                        state_dict = ckpt[key]
                        break
                if state_dict is None and all(hasattr(v, "shape") for v in ckpt.values()):
                    state_dict = ckpt
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
            model.eval()

            seq = torch.as_tensor(fused[idx - context + 1: idx + 1],
                                  dtype=torch.float32).unsqueeze(0)  # [1, 64, E]
            with torch.no_grad():
                observed = model.observe(seq)
                start = LatentState(deter=observed.states.deter[:, -1],
                                    stoch=observed.states.stoch[:, -1])
                rollout = model.imagine(start, horizon=horizon, n=n_roll)
            traj = views.project_rollouts(rollout.embeds)
            st.plotly_chart(views.imagination_fanchart(traj), width="stretch")
            st.caption("Norm-delta projection of decoded rollout embeddings: a "
                       "latent-space divergence proxy, not a price forecast.")
        except Exception:
            st.error("Imagination failed — the world-model checkpoint may not match "
                     "the interface contract yet.")
            st.code(traceback.format_exc())
            return

        st.subheader("Nearest memory analogs")
        try:
            from aether.worldmodel.memory import MemoryBank
            from aether.worldmodel.interfaces import MemoryConfig
            bank = MemoryBank(MemoryConfig(dim=int(embs["fused"].shape[1]),
                                           persist_dir=str(DATA_ROOT / "memory")))
            bank.load()
            analogs = bank.query(embs["fused"][idx], k=8)
            table = views.analogs_table(analogs)
            if len(table):
                st.dataframe(table, width="stretch")
            else:
                st.info("Memory bank returned no analogs for this moment.")
        except Exception as exc:
            st.info(f"Memory bank unavailable ({type(exc).__name__}: {exc}). "
                    "It is persisted under data/memory/ by world-model training.")


# --------------------------------------------------------------------------- #
# 5 Trades & Autopsies
# --------------------------------------------------------------------------- #

def _render_autopsy(report: dict) -> None:
    verdict = str(report.get("verdict", "unknown"))
    color = views.VERDICT_COLORS.get(verdict, views.MUTED)
    st.markdown(_badge(verdict, color), unsafe_allow_html=True)
    narrative = report.get("narrative")
    if narrative:
        st.markdown(f"> {narrative}")

    trade = report.get("trade") or {}
    if trade:
        st.dataframe(views.trades_table([trade]), width="stretch")

    cfs = views.counterfactuals_table(report.get("counterfactuals"))
    if len(cfs):
        st.subheader("Counterfactuals")
        st.dataframe(cfs, width="stretch")

    drivers = report.get("drivers") or []
    if drivers:
        st.plotly_chart(views.driver_bars_figure(drivers), width="stretch")

    causal_ctx = views.driver_bars_figure(report.get("causal_context"),
                                          title="Causal context at entry")
    if report.get("causal_context"):
        st.plotly_chart(causal_ctx, width="stretch")

    analogs = views.analogs_table(report.get("analogs"))
    if len(analogs):
        st.subheader("Historical analogs")
        st.dataframe(analogs, width="stretch")


def page_trades() -> None:
    st.header("Trades & Autopsies")
    backtests = cached_backtests(str(DATA_ROOT))
    autopsies = cached_autopsies(str(DATA_ROOT))
    autopsy_by_id = {}
    for rep in autopsies:
        trade = rep.get("trade") if isinstance(rep.get("trade"), dict) else {}
        autopsy_by_id[str(trade.get("trade_id") or rep.get("_file"))] = rep

    if not backtests:
        _missing("Backtests (data/backtests/<name>/)", "python scripts/run_backtest.py")
    else:
        name = st.selectbox("Backtest", sorted(backtests))
        entry = backtests[name]
        result = entry.get("result") or {}

        stats = result.get("stats")
        if isinstance(stats, dict) and stats:
            st.subheader("Stats")
            st.dataframe(views.stats_table(stats), width="stretch", height=280)
        else:
            st.info("result.json has no stats dict for this backtest.")

        equity = entry.get("equity")
        if equity is not None and len(equity):
            st.plotly_chart(views.equity_figure(equity), width="stretch")
            st.plotly_chart(views.drawdown_figure(equity), width="stretch")
        else:
            st.info(f"No equity.parquet under data/backtests/{name}/ — re-run "
                    "`python scripts/run_backtest.py` to regenerate it.")

        trades = result.get("trades") or []
        for rep in result.get("autopsies") or []:
            if isinstance(rep, dict):
                trade = rep.get("trade") if isinstance(rep.get("trade"), dict) else {}
                tid = str(trade.get("trade_id") or "")
                if tid:
                    autopsy_by_id.setdefault(tid, rep)
        st.subheader(f"Trades ({len(trades)})")
        if trades:
            table = views.trades_table(trades)
            st.dataframe(table, width="stretch", height=300)
            ids = [str(t.get("trade_id", i)) for i, t in
                   enumerate(views.filter_records(trades))]
            marked = [f"{tid}  [autopsy]" if tid in autopsy_by_id else tid for tid in ids]
            pick = st.selectbox("Inspect trade", range(len(ids)),
                                format_func=lambda i: marked[i])
            tid = ids[pick]
            if tid in autopsy_by_id:
                _render_autopsy(autopsy_by_id[tid])
            else:
                st.info(f"No autopsy for trade {tid} (data/autopsies/{tid}.json). "
                        "Autopsies are produced by the backtester for flagged trades.")
        else:
            st.info("This backtest closed no trades.")

    if autopsies:
        st.divider()
        st.subheader(f"All autopsies on disk ({len(autopsies)})")
        labels = [f"{a.get('_file')} - {a.get('verdict', '?')}" for a in autopsies]
        pick = st.selectbox("Autopsy", range(len(autopsies)),
                            format_func=lambda i: labels[i])
        _render_autopsy(autopsies[pick])
    elif not backtests:
        _missing("Autopsies (data/autopsies/*.json)",
                 "python scripts/run_backtest.py  (autopsies are written for losses)")


# --------------------------------------------------------------------------- #
# 6 Paper Blotter
# --------------------------------------------------------------------------- #

def page_blotter() -> None:
    st.header("Paper Blotter")
    state = views.load_paper_state(DATA_ROOT / "paper" / "state.json")
    if state is None:
        _missing("Paper state (data/paper/state.json)", "python scripts/run_paper.py --loop")
    else:
        cols = st.columns(4)
        equity = state.get("equity")
        if equity is not None:
            cols[0].metric("Equity", f"${float(equity):,.0f}")
        positions = state.get("positions")
        if isinstance(positions, dict):
            cols[1].metric("Open positions", len(positions))
        day_pnl = state.get("day_pnl") or state.get("day_pnl_frac")
        if day_pnl is not None:
            cols[2].metric("Day PnL", f"{float(day_pnl):,.4f}")
        halted = state.get("halted") or state.get("halt")
        if halted is not None:
            cols[3].metric("Halted", str(bool(halted)))
        with st.expander("Raw state"):
            st.json(state)

    blotter = cached_blotter(str(DATA_ROOT / "paper" / "blotter.jsonl"))
    if blotter is None:
        _missing("Blotter (data/paper/blotter.jsonl)", "python scripts/run_paper.py --loop")
        return

    text_cols = [c for c in blotter.columns if blotter[c].dtype == object]
    halt_mask = pd.Series(False, index=blotter.index)
    for col in text_cols:
        halt_mask |= blotter[col].astype(str).str.contains("halt", case=False, na=False)
    n_halts = int(halt_mask.sum())
    if n_halts:
        st.warning(f"{n_halts} halt event(s) in the blotter (highlighted below).")

    tail = blotter.tail(200)
    tail_mask = halt_mask.tail(200)

    def _highlight(row):
        style = "background-color: rgba(208, 59, 59, 0.25)" if tail_mask.loc[row.name] else ""
        return [style] * len(row)

    st.subheader(f"Last {len(tail)} of {len(blotter)} blotter events")
    st.dataframe(tail.style.apply(_highlight, axis=1), width="stretch", height=480)


# --------------------------------------------------------------------------- #
# 7 Feedback
# --------------------------------------------------------------------------- #

def page_feedback() -> None:
    st.header("Feedback")
    st.caption("Feedback appended here flows into the lessons buffer "
               "(data/lessons.jsonl) and from there directly into future "
               "training — this is the human half of the learning loop.")

    with st.form("feedback_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        context = c1.selectbox("Context", ("signal", "autopsy", "general"))
        ticker = c2.selectbox("Ticker", ["(none)"] + _tickers())
        rating = c3.slider("Rating", 1, 5, 3,
                           help="1 = model badly wrong, 5 = model exactly right")
        text = st.text_area("What did you observe?",
                            placeholder="e.g. the 14:30 AAPL signal ignored an "
                                        "obvious news-driven regime shift...")
        if st.form_submit_button("Submit feedback", type="primary"):
            if text.strip():
                views.append_feedback(DATA_ROOT, {
                    "context": context,
                    "ticker": None if ticker == "(none)" else ticker,
                    "text": text.strip(),
                    "rating": int(rating),
                    "source": "dashboard",
                })
                cached_feedback.clear()
                st.success("Feedback recorded in data/feedback.jsonl.")
            else:
                st.error("Feedback text is empty — nothing recorded.")

    st.subheader("Feedback history")
    history = cached_feedback(str(DATA_ROOT))
    if history is None or len(history) == 0:
        st.info("No feedback yet — submit the first entry above.")
    else:
        if "ts" in history.columns:
            history = history.sort_values("ts", ascending=False)
        st.dataframe(history, width="stretch", height=360)


# --------------------------------------------------------------------------- #
# 8 Training
# --------------------------------------------------------------------------- #

def page_training() -> None:
    st.header("Training")
    curves = cached_runs(str(RUNS_DIR))
    if not curves:
        _missing("Training runs (runs/*.jsonl)", "python scripts/train_perception.py")
    for name, df in curves.items():
        st.subheader(name)
        metrics = [c for c in df.columns if c not in ("step", "ts")]
        default = [m for m in metrics if m not in ("lr",)][:3] or metrics[:1]
        chosen = st.multiselect(f"Metrics ({name})", metrics, default=default,
                                max_selections=8, key=f"metrics_{name}")
        if chosen:
            st.plotly_chart(views.training_curve_figure(df, chosen, title=name),
                            width="stretch")

    st.divider()
    st.subheader("Self-diagnosis")
    try:
        import importlib
        diagnosis = importlib.import_module("aether.evolution.diagnosis")
        SelfDiagnosis = getattr(diagnosis, "SelfDiagnosis")
    except Exception:
        st.info("Self-diagnosis module not built yet (aether.evolution.diagnosis). "
                "It will appear here automatically once Layer 4 lands.")
        return
    if st.button("Run self-diagnosis"):
        try:
            report = SelfDiagnosis().scan()
        except Exception:
            st.error("SelfDiagnosis().scan() raised:")
            st.code(traceback.format_exc())
            return
        findings = report if isinstance(report, list) else \
            report.get("findings") if isinstance(report, dict) else None
        severity_color = {"critical": views.CRITICAL, "serious": views.SERIOUS,
                          "error": views.CRITICAL, "warning": views.WARNING,
                          "warn": views.WARNING, "info": views.SERIES[0],
                          "ok": views.GOOD, "good": views.GOOD}
        if isinstance(findings, list) and findings and \
                all(isinstance(f, dict) for f in findings):
            for f in findings:
                sev = str(f.get("severity", "info")).lower()
                color = severity_color.get(sev, views.MUTED)
                msg = f.get("message") or f.get("msg") or json.dumps(f, default=str)
                st.markdown(f"{_badge(sev.upper(), color)}&nbsp; {msg}",
                            unsafe_allow_html=True)
        else:
            st.json(report)


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Aether — Market Intelligence Brain",
                       layout="wide")
    st.sidebar.title("Aether")
    st.sidebar.caption("Market Intelligence Brain — Layer 6 cockpit")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.caption(f"data root: `{DATA_ROOT}`")

    dispatch = {
        "Overview": page_overview,
        "Charts": page_charts,
        "Causal Brain": page_causal,
        "Playground": page_playground,
        "Trades & Autopsies": page_trades,
        "Paper Blotter": page_blotter,
        "Feedback": page_feedback,
        "Training": page_training,
    }
    try:
        dispatch[page]()
    except Exception:
        st.error("This page hit an unexpected error — the artifact it reads may "
                 "be mid-write by another Aether process. Details:")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
