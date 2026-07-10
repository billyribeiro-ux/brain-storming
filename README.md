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

All seven layers are implemented. Layers 0–1 are the load-bearing
foundation; layers 2–6 are first-generation implementations, adversarially
audited and hardened (see "Honesty notes").

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

## Layer guide (2–6)

* **World model** (`aether/worldmodel/`) — action-free RSSM latent dynamics
  (imagination rollouts for counterfactual simulation; honest by design:
  our order flow does not move these instruments), NOTEARS-style lagged
  causal graphs with bootstrap edge confidence, ring-buffer memory with
  outcome-tagged analogs (as-of + 30-min embargo on retrieval), and
  forensic trade autopsies: counterfactual variants replayed against real
  bars, verdict taxonomy (`good_loss`/`bad_loss`/`good_win`/`lucky_win`),
  lessons appended for training consumers.
* **Decision core** (`aether/decision/`) — `EmbeddingStore` (frozen
  perception → per-ticker npz replay), the conservative trading env
  (decisions at t fill at t+1; stop fills at the worse of stop and open;
  stop-before-target intrabar; whole-share fee-inclusive entries; eod
  force-close; same-step auto-reset honoring the curriculum pool),
  hierarchical policy (meta intent + Beta-distributed size/stop/target),
  sequence-preserving recurrent PPO with exact resume, hindsight mining,
  reversal-clarity curriculum.
* **Meta-evolution** (`aether/evolution/`) — 20-gene config-genome search,
  EWC anti-forgetting (true per-sample Fisher path), dream phase (value
  consistency on imagined latents; provably cannot train action heads),
  six-check self-diagnosis over run artifacts.
* **Signals/risk/execution** (`aether/execution/`) — consensus conviction
  (policy × imagination agreement × embargoed analogs, uncertainty
  abstention gates, full `EvidenceBundle` audit trail + template rationale
  from computed evidence only), hard-rail risk (sticky circuit breaker,
  per-trade risk sizing, correlation haircut), event-driven backtester
  (no same-bar re-entry after stops, session-boundary force-close,
  automatic loss autopsies), taker/maker tactics, paper trader that
  replays every missed bar (downtime cannot walk past stops).
* **Dashboard** (`aether/dashboard/`) — `streamlit run aether/dashboard/app.py`:
  overview, candles with signal/trade/saliency overlays, causal brain,
  imagination playground, autopsy browser, paper blotter, feedback →
  lessons pipeline, training curves + self-diagnosis.

## Roadmap

| Module | Status |
|---|---|
| 0. FMP data pipeline (bars, news, fundamentals, macro) | ✅ |
| 1. Perception: encoders, DNA, uncertainty, anomaly | ✅ |
| 2. World model: dynamics, causal graphs, memory, autopsies | ✅ v1 |
| 3. Hierarchical RL decision core (env, PPO, curriculum) | ✅ v1 |
| 4. Meta-evolution (genome search, EWC, dreams, diagnosis) | ✅ v1 |
| 5. Signals, risk, backtest, tactics, paper trading | ✅ v1 |
| 6. Streamlit dashboard & human feedback | ✅ v1 |
| Options flow & surfaces | blocked on data plan (auto-enables via probe) |
| GPU-scale training, meta-controller depth, live NAS loops | next iterations |

## The Cockpit (`dashboard-web/`) — institutional web UI

A SvelteKit cockpit (Svelte 5 runes, TypeScript strict, Tailwind v4,
Lightweight Charts, TanStack Query, Zod, phosphor icons, Node 24 LTS, pnpm)
fed by a FastAPI bridge (`aether/dashboard/api.py`) over HTTP + WebSocket.

```bash
# 1. the Python bridge (REST + WS on :8600)
python3 scripts/run_api.py

# 2. the cockpit (dev on :5173, or build+preview on :4173)
cd dashboard-web && pnpm install
pnpm dev            # development
pnpm build && pnpm preview   # production build behind the same proxy
```

Pages: **Deck** (candles with signal arrows, stop/target lines, brain-
attention strip, live signal feed, driver compass), **Autopsies** (period-
filterable trade log with CSV export, autopsy detail, equity + drawdown),
**Playground** (session replay streaming real recorded bars + the signals
the backtest actually emitted; imagination fan honestly labeled as latent
divergence), **Brain** (status grid, training sparklines, interactive
42-node causal graph). Real-time via native WebSocket (`/ws`): replayed
bars, signals, trades, heartbeats. Feedback posts land in
`data/feedback.jsonl` and flow to the lessons buffer. The e2e walkthrough
lives at `dashboard-web/e2e/walkthrough.mjs` (Playwright).

The legacy Streamlit dashboard (`aether/dashboard/app.py`) remains for
quick local inspection; the cockpit is the primary interface.

## Honesty notes

* Options-flow decoding is registered but **not active** — the current FMP
  plan returns 404 for options endpoints. The capability probe records this;
  upgrading the plan activates ingestion with no code changes.
* Layers 2–6 are **first-generation**: every architectural organ exists,
  is tested (213-test suite incl. look-ahead batteries and hand-computed
  fill math), was adversarially audited by independent review passes
  (look-ahead, RL math, accounting realism — 18 findings applied), and runs
  end-to-end on the real lake. What they are NOT yet is *converged*:
  checkpoints in this repo come from short CPU validation runs, not
  GPU-scale training. Treat any backtest numbers as machinery proof, not
  alpha claims.
* Dream phases train value consistency only (no imagined prices → no
  imagined PnL) until a price decoder is added to the world model.
* Signal rationales are template-composed from real computed evidence
  (attribution, causal drivers, analogs, uncertainty) — not free-form
  generation, and honest about being so.
* The paper trader writes an order blotter; **no broker integration
  exists anywhere** — live capital requires explicit human integration by
  design.
