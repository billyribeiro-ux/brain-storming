# Aether — Market Intelligence Brain

Aether is a self-learning market intelligence system for intraday **reversal
trading** (catching tops and bottoms) on a fixed universe of ten instruments:

`AAPL · NVDA · TSLA · AMZN · NFLX · CSCO · SPY · QQQ · IWM · SPX (^GSPC)`

It is built on one uncompromising idea: **nothing about trading is hardcoded.**
No indicators, no chart patterns, no rules. Aether ingests everything its data
plan exposes, learns its own features, builds its own causal understanding,
and improves itself relentlessly — especially from its losses.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 6. Dashboard          Streamlit: charts, attention maps, causal │
│                       graphs, autopsies, human feedback          │
├─────────────────────────────────────────────────────────────────┤
│ 5. Signal/Risk/Exec   consensus signals, learned stops/targets/  │
│                       sizing, learned execution tactics          │
├─────────────────────────────────────────────────────────────────┤
│ 4. Meta-Evolution     NAS, continual learning, dream phases,     │
│                       self-diagnosis                             │
├─────────────────────────────────────────────────────────────────┤
│ 3. Decision Core      hierarchical meta-RL: entries, sizing,     │
│                       dynamic stops/targets, trade management    │
├─────────────────────────────────────────────────────────────────┤
│ 2. World Model        neural causal graphs, counterfactual       │
│                       simulation, memory tiers, loss autopsies   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Perception         multi-modal encoders → hierarchical        │  ◀ this release
│                       embeddings, uncertainty, anomaly, DNA      │
├─────────────────────────────────────────────────────────────────┤
│ 0. Data               FMP stable API → Parquet lake              │  ◀ this release
└─────────────────────────────────────────────────────────────────┘
```

This repository currently implements **layers 0–1**. Higher layers land
module by module on top of the `PerceptionOutput` contract.

### Layer 0 — Data (`aether/data/`)

* **`fmp_client.py`** — async FMP *stable*-API client with token-bucket rate
  limiting, retry/backoff, and intraday pagination calibrated against the
  measured behavior of the live API (the server returns a fixed calendar
  window anchored at `to`; the client marches deterministic chunks and never
  mistakes a holiday-empty response for end-of-history).
* **`endpoints.py`** — a single registry of every decoded endpoint: intraday
  and daily bars, ticker/press/general news, quarterly fundamentals, earnings
  with estimates, analyst grade changes, share float, dividends/splits,
  treasury curve, and macro indicators. Each entry declares how it paginates,
  which instrument kinds it applies to, and its **probed availability** —
  Aether actively senses what the API key can reach
  (`scripts/probe_capabilities.py`) instead of assuming. Options endpoints
  are registered but 404 on the current plan; they light up automatically
  after an upgrade.
* **`storage.py`** — merge-writing, year-partitioned Parquet lake with a
  watermark manifest. Every write is idempotent; backfills are resumable and
  incremental syncs are O(1) to plan.
* **`ingestion.py`** — deep backfill + incremental sync orchestration across
  all endpoints and tickers. **`quality.py`** — forensic audits of the lake
  against the exchange calendar (missing/short sessions, duplicates,
  impossible prices).
* **`utils/market_time.py`** — NYSE session calendar computed from exchange
  rules (including Good Friday computus and half-days), 390-minute session
  arithmetic.

### Layer 1 — Perception (`aether/perception/`)

The contract lives in **`interfaces.py`** and is the single source of truth
for every module above it.

* **Representation, not indicators** (`preprocessing.py`) — raw OHLCV is
  reparameterized into scale-free form (log returns, bar geometry fractions,
  volume vs. that ticker's own time-of-day norm). This is normalization
  hygiene, like pixel scaling in vision; all *meaning* is learned.
* **Ticker DNA** — each instrument gets (a) a statistical fingerprint
  (`TickerStats`, fitted on training data only) used for normalization, and
  (b) a learned embedding that FiLM-conditions every encoder, so shared
  weights specialize per instrument.
* **Multi-scale encoders** (`encoders/`) — one `SequenceEncoder` contract,
  four families, swappable per stream via config (the future NAS layer
  mutates config, not code):
  * `bar_transformer.py` — pre-LN transformer with rotary embeddings,
  * `ssm.py` — S4D-style diagonal state-space model (long memory, causal),
  * `recurrent.py` — stacked LSTM,
  * `wavelet.py` — learnable multi-resolution causal filterbank,
  * `fusion.py` — Perceiver-style cross-attention fusing the 1min stream,
    5min stream, daily context, and DNA into one state embedding.
* **Self-supervised objectives** (`heads.py`, `model.py`) — no labels, no
  human targets:
  * masked-bar modeling (reconstruct hidden bars from context),
  * cross-scale contrastive alignment (1min ↔ 5min InfoNCE),
  * ticker-DNA discrimination (embeddings must know *who* they are),
  * evidential regression on next-bar returns (Normal-Inverse-Gamma) giving
    calibrated **aleatoric** (market noise) vs **epistemic** (model
    ignorance) uncertainty,
  * reconstruction-based **anomaly scoring** (normalized surprise).
* **Trainer** (`train.py`) — AMP autocast, cosine schedule with warmup, EMA
  weights, exact-resume checkpointing, JSONL metrics. GPU-ready; runs on CPU.

### The one sacred invariant

**No look-ahead.** Anything computed for anchor time *t* may only use data
timestamped ≤ *t*: 5-minute bars must be *completed* by *t*, daily context is
strictly previous sessions, and normalization statistics are fitted on the
training range only. The test suite pins this down by perturbing future bars
and asserting sample invariance. Every future module inherits this rule.

## Quickstart

```bash
pip install -e ".[dev]"           # plus the torch build for your hardware
cp .env.example .env              # put your FMP_API_KEY inside

python3 scripts/probe_capabilities.py        # what can this key reach?
python3 scripts/backfill.py                  # deep history for all 10 tickers
python3 scripts/sync.py                      # incremental top-up (cron-able)

python3 scripts/train_perception.py \
    --train-start 2020-01-01 --train-end 2025-06-30 \
    --val-start 2025-07-01 --val-end 2026-06-30
```

```bash
python3 -m pytest tests/ -q       # the invariant suite
```

## Roadmap

| Module | Status |
|---|---|
| 0. FMP data pipeline (bars, news, fundamentals, macro) | ✅ this release |
| 1. Perception: encoders, DNA, uncertainty, anomaly | ✅ this release |
| 2. World model: causal graphs, counterfactuals, autopsies | next |
| 3. Hierarchical meta-RL decision core | planned |
| 4. Meta-learning & self-evolution (NAS, continual learning) | planned |
| 5. Signal/risk/execution layer | planned |
| 6. Streamlit dashboard & human feedback | planned |
| Options flow & surfaces | blocked on data plan (auto-enables via probe) |

## Honesty notes

* Options-flow decoding is registered but **not active** — the current FMP
  plan returns 404 for options endpoints. The capability probe records this;
  upgrading the plan activates ingestion with no code changes.
* Perception embeddings are trained self-supervised; they become *tradable*
  only once the world model and decision core (next modules) sit on top.
  Nothing in this release emits trade signals yet — by design.
