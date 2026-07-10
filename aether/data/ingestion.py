"""Ingestion engine — fills and refreshes the Parquet data lake from FMP.

The engine sits between the thin transport client (:class:`FMPClient`) and
the merge-writing lake (:class:`ParquetStore`) and owns every acquisition
*policy*. The client guarantees transport (rate limiting, retries, honest
errors); the store guarantees durability (idempotent merge-writes,
watermarks); this module decides *what* to ask for and *when*.

Core mechanics
--------------
* **Chunked backward marches.** Deep intraday history cannot be pulled in a
  single request (the FMP server caps each response at ~3 trading days of
  1min bars), so :meth:`IngestionEngine.backfill` marches a date cursor
  backwards from today toward ``cfg.data.backfill_start`` in bounded chunks
  (~30 calendar days for intraday bars, 90 days for treasury rates, calendar
  months for news) and writes every chunk to the lake the moment it lands.
  Progress is therefore durable: killing the process loses at most the one
  in-flight chunk.

* **Resumability via watermarks.** The store records the oldest
  (``low_watermark``) and newest (``watermark``) timestamp per
  (dataset, ticker). A resumed backfill continues strictly *before* the low
  watermark, skipping the already-covered middle. An incremental
  :meth:`IngestionEngine.sync` fetches only *after* the high watermark minus
  a two-day overlap, so late-settling bars (vendor restatements, delayed
  prints) are re-merged; the lake's dedup-on-write makes the overlap free.

* **Capability awareness.** Endpoints this key's plan cannot reach are
  recorded in ``capabilities.json`` — either proactively by
  :meth:`IngestionEngine.probe_and_save` ("active sensing" of the API) or
  reactively when a live request raises :class:`FMPAccessError`. Known-dead
  endpoints are skipped on later runs, and a mid-run discovery flips an
  in-memory flag so sibling jobs stop burning rate limit on the same wall.
  A paywalled endpoint never crashes a run.

* **Bounded concurrency.** Every (dataset, ticker) pair becomes one job; all
  jobs run concurrently under an ``asyncio.Semaphore(cfg.fmp.max_concurrency)``
  while ONE shared :class:`FMPClient` enforces the global requests-per-minute
  budget with its token bucket. Parallelism can therefore never violate the
  plan's rate limit.

No interpretation happens here: frames are stored exactly as FMP returns
them (plus canonical timestamp parsing done by the client). Feature
construction belongs to the perception layer — Aether learns meaning;
ingestion guarantees supply.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..config import TICKERS, AetherConfig, TickerSpec, ticker_by_alias
from ..utils.logging import get_logger
from ..utils.market_time import ET
from .endpoints import ECONOMIC_INDICATOR_NAMES, ENDPOINTS, EndpointSpec
from .fmp_client import CapabilityStatus, FMPAccessError, FMPClient, FMPError
from .storage import MARKET_PSEUDO_TICKER, ParquetStore, WriteResult

logger = get_logger("aether.ingestion")

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Calendar-day span of one intraday-bars fetch chunk. ~30 days ≈ 21 trading
#: sessions ≈ 8.2k one-minute bars, i.e. roughly 7–8 paginated requests per
#: chunk — big enough to amortize request overhead, small enough that a
#: killed process re-downloads at most a month.
INTRADAY_CHUNK_DAYS: int = 30

#: Treasury rates are one row per day across the whole curve, so 90-day
#: chunks keep request counts tiny while staying inside any server caps.
TREASURY_CHUNK_DAYS: int = 90

#: Incremental sync re-fetches this many days *before* the stored high
#: watermark. Vendors occasionally restate the most recent bars (late
#: consolidated prints, corrections); a two-day overlap re-merges them and
#: the store's dedup-on-write keeps the operation idempotent.
SYNC_OVERLAP_DAYS: int = 2

#: Signature of a windowed fetch: (chunk_start, chunk_end) -> DataFrame.
FetchFn = Callable[[date, date], Awaitable[pd.DataFrame]]
#: Signature of a chunk writer: DataFrame -> WriteResult.
WriteFn = Callable[[pd.DataFrame], WriteResult]

#: Report shape returned by backfill/sync: {dataset -> {ticker -> rows_new}}.
RunReport = dict[str, dict[str, int]]


def _today_et() -> date:
    """Today's date on the exchange clock (America/New_York).

    All FMP intraday timestamps are quoted in US/Eastern, so the march
    cursors must live on the same calendar — a UTC host after midnight would
    otherwise ask for a "today" the exchange has not reached yet (harmless,
    but sloppy).
    """
    return datetime.now(ET).date()


def _utcnow_iso() -> str:
    """UTC timestamp for capability bookkeeping."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IngestionEngine:
    """Orchestrates all data acquisition into the Parquet lake.

    Parameters
    ----------
    cfg:
        Full Aether configuration (FMP credentials/limits + lake paths).
    store:
        An existing :class:`ParquetStore` to write into, or ``None`` to open
        one rooted at ``cfg.data.root``.
    """

    def __init__(self, cfg: AetherConfig, store: ParquetStore | None = None):
        self.cfg = cfg
        self.store = store or ParquetStore(cfg.data.root)
        #: In-memory capability view for the *current* run. Loaded from
        #: capabilities.json at run start and flipped to False the moment a
        #: live request reveals an endpoint is unreachable, so concurrent
        #: sibling jobs short-circuit instead of re-hitting the paywall.
        self._caps: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Capabilities — active sensing of what this API key can reach
    # ------------------------------------------------------------------ #

    async def probe_and_save(self) -> dict[str, CapabilityStatus]:
        """Probe every registered endpoint and persist the results.

        Runs :meth:`FMPClient.probe_capabilities` (one cheap request per
        endpoint), writes the outcome to ``cfg.data.capabilities_path`` as
        JSON, and returns the raw statuses. Later backfill/sync runs consult
        this file to skip endpoints the plan cannot access.
        """
        async with FMPClient(self.cfg.fmp) as client:
            statuses = await client.probe_capabilities()

        payload = {
            name: {
                "available": s.available,
                "http_status": s.http_status,
                "detail": s.detail,
                "checked_at": _utcnow_iso(),
            }
            for name, s in statuses.items()
        }
        path = self.cfg.data.capabilities_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)  # atomic on POSIX — a crash never truncates the file

        n_ok = sum(1 for s in statuses.values() if s.available)
        logger.info("capabilities probe: %d/%d endpoints available -> %s",
                    n_ok, len(statuses), path)
        return statuses

    def load_capabilities(self) -> dict[str, bool]:
        """Read ``capabilities.json`` into ``{dataset: available}``.

        Returns ``{}`` when the file is absent or unreadable, which the
        engine interprets as *unknown*: attempt every endpoint and record
        failures as they surface. Values may be stored either as bare
        booleans or as rich dicts with an ``available`` key; both parse.
        """
        path = self.cfg.data.capabilities_path
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not parse %s (%s) — treating capabilities "
                           "as unknown", path, exc)
            return {}
        out: dict[str, bool] = {}
        for name, value in raw.items():
            if isinstance(value, dict):
                out[name] = bool(value.get("available", True))
            else:
                out[name] = bool(value)
        return out

    def _record_capability(self, dataset: str, available: bool,
                           detail: str = "") -> None:
        """Merge one runtime capability discovery into capabilities.json.

        Called when a live request raised :class:`FMPAccessError` — the API
        itself just told us the plan cannot reach this endpoint, which is
        strictly better information than any assumption.
        """
        path = self.cfg.data.capabilities_path
        raw: dict[str, Any] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                raw = {}  # corrupt file: rebuild from this single fact
        raw[dataset] = {
            "available": available,
            "detail": detail,
            "checked_at": _utcnow_iso(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, indent=2))
        tmp.replace(path)

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    async def backfill(self, datasets: list[str] | None = None,
                       tickers: list[str] | None = None) -> RunReport:
        """Deep historical download back to ``cfg.data.backfill_start``.

        Windowed/paged endpoints march backwards from today, writing every
        chunk immediately (durable, resumable via the low watermark).
        Series endpoints (fundamentals, macro) are pulled whole — their
        full history fits in one response. Snapshot endpoints are skipped:
        a snapshot taken today says nothing about the past.

        Returns ``{dataset: {ticker: rows_new}}``.
        """
        return await self._run("backfill", datasets, tickers)

    async def sync(self, datasets: list[str] | None = None,
                   tickers: list[str] | None = None) -> RunReport:
        """Incremental top-up: high watermark (minus overlap) -> now.

        Also collects snapshot endpoints (quote, shares float), which
        accumulate one point-in-time row per sync into the lake.

        Returns ``{dataset: {ticker: rows_new}}``.
        """
        return await self._run("sync", datasets, tickers)

    # ------------------------------------------------------------------ #
    # Run orchestration
    # ------------------------------------------------------------------ #

    def _resolve_specs(self, datasets: list[str] | None) -> list[EndpointSpec]:
        """Dataset names -> EndpointSpecs (all registered when None)."""
        if datasets is None:
            return list(ENDPOINTS.values())
        unknown = [d for d in datasets if d not in ENDPOINTS]
        if unknown:
            raise KeyError(
                f"Unknown dataset(s) {unknown}; known: {sorted(ENDPOINTS)}")
        return [ENDPOINTS[d] for d in datasets]

    @staticmethod
    def _resolve_tickers(tickers: list[str] | None) -> list[TickerSpec]:
        """Aliases -> TickerSpecs (full universe when None)."""
        if tickers is None:
            return list(TICKERS)
        return [ticker_by_alias(alias) for alias in tickers]

    @staticmethod
    def _targets(spec: EndpointSpec,
                 tickers: list[TickerSpec]) -> list[TickerSpec | None]:
        """Which instruments one endpoint applies to.

        Market-wide endpoints (``symbol_param is None``: treasury rates,
        economic indicators, general news) yield the single pseudo-target
        ``None``, stored under :data:`MARKET_PSEUDO_TICKER`. Per-symbol
        endpoints are filtered by ticker kind so we never burn rate limit
        asking for e.g. an income statement of the S&P 500 index.
        """
        if spec.symbol_param is None:
            return [None]
        return [t for t in tickers if t.kind in spec.kinds]

    async def _run(self, phase: str, datasets: list[str] | None,
                   tickers: list[str] | None) -> RunReport:
        """Shared driver behind :meth:`backfill` and :meth:`sync`."""
        specs = self._resolve_specs(datasets)
        tspecs = self._resolve_tickers(tickers)
        self._caps = self.load_capabilities()
        report: RunReport = {}

        async with FMPClient(self.cfg.fmp) as client:
            # Ticker-level concurrency bound. The client additionally caps
            # in-flight HTTP requests and paces them with a token bucket, so
            # this semaphore mainly keeps memory (pending DataFrames) and
            # log interleaving civilized.
            sem = asyncio.Semaphore(self.cfg.fmp.max_concurrency)
            jobs: list[Awaitable[None]] = []
            for spec in specs:
                if self._caps.get(spec.dataset) is False:
                    logger.info("skip %s: marked unavailable in %s",
                                spec.dataset, self.cfg.data.capabilities_path)
                    continue
                if phase == "backfill" and spec.mode == "snapshot":
                    # Snapshots are point-in-time: today's quote cannot
                    # backfill yesterday. They accumulate via sync instead.
                    logger.info("skip %s during backfill: snapshot endpoints "
                                "are point-in-time (collected on sync)",
                                spec.dataset)
                    continue
                for tspec in self._targets(spec, tspecs):
                    jobs.append(self._guarded_job(client, sem, spec, tspec,
                                                  phase, report))
            logger.info("%s: launching %d jobs (max %d concurrent)",
                        phase, len(jobs), self.cfg.fmp.max_concurrency)
            await asyncio.gather(*jobs)
            n_requests = client.request_count

        total_new = sum(sum(per_ticker.values()) for per_ticker in report.values())
        logger.info("%s complete: %d jobs reported, %d new rows, "
                    "%d API requests", phase,
                    sum(len(v) for v in report.values()), total_new, n_requests)
        return report

    async def _guarded_job(self, client: FMPClient, sem: asyncio.Semaphore,
                           spec: EndpointSpec, tspec: TickerSpec | None,
                           phase: str, report: RunReport) -> None:
        """One (dataset, ticker) unit of work with full failure isolation.

        Failure policy — the run must NEVER crash on API problems:
        * :class:`FMPAccessError` (402/403/404): the plan cannot reach this
          endpoint. Log it, persist ``{dataset: False}`` to
          capabilities.json, flip the in-memory flag so queued sibling jobs
          bail out, and move on.
        * Any other :class:`FMPError` (retries exhausted, malformed
          response): log and continue — partial progress is already durable
          in the lake, and the next run's watermarks will pick up the slack.
        Programming errors are deliberately NOT swallowed; bugs should fail
        loudly, not silently produce a hollow lake.
        """
        key = tspec.alias if tspec is not None else MARKET_PSEUDO_TICKER
        async with sem:
            # A sibling job may have discovered mid-run that this dataset is
            # paywalled — don't spend more requests confirming it.
            if self._caps.get(spec.dataset) is False:
                return
            try:
                rows_new = await self._run_job(client, spec, tspec, phase)
            except FMPAccessError as exc:
                logger.warning("%s/%s: access denied (%s) — recording "
                               "capability=False and continuing",
                               spec.dataset, key, exc)
                self._caps[spec.dataset] = False
                self._record_capability(spec.dataset, False, str(exc)[:200])
                return
            except FMPError as exc:
                logger.error("%s/%s: failed after retries: %s — continuing "
                             "with remaining jobs", spec.dataset, key, exc)
                return
            report.setdefault(spec.dataset, {})[key] = rows_new

    async def _run_job(self, client: FMPClient, spec: EndpointSpec,
                       tspec: TickerSpec | None, phase: str) -> int:
        """Dispatch one job to the mode-specific ingestion strategy."""
        dataset = spec.dataset
        if dataset in ("bars_1min", "bars_5min"):
            assert tspec is not None  # bars are always per-instrument
            return await self._ingest_intraday(client, spec, tspec, phase)
        if dataset == "bars_daily":
            assert tspec is not None
            return await self._ingest_daily(client, spec, tspec, phase)
        if dataset == "treasury_rates":
            return await self._ingest_treasury(client, spec, phase)
        if dataset == "economic_indicators":
            return await self._ingest_economic(client, spec)
        if spec.mode == "paged":
            return await self._ingest_news(client, spec, tspec, phase)
        if spec.mode in ("series", "snapshot"):
            assert tspec is not None  # market-wide series handled above
            return await self._ingest_series(client, spec, tspec)
        raise ValueError(f"No ingestion strategy for {dataset!r} "
                         f"(mode={spec.mode!r})")

    # ------------------------------------------------------------------ #
    # Date-window bookkeeping
    # ------------------------------------------------------------------ #

    def _history_start(self) -> date:
        return date.fromisoformat(self.cfg.data.backfill_start)

    def _backfill_window(self, dataset: str, key: str) -> tuple[date, date] | None:
        """Remaining [start, end] for a backwards backfill, or None if done.

        If a low watermark exists, prior runs already cover
        [low_watermark, high_watermark]; the march resumes strictly *before*
        the low watermark's day. (The chunk-then-write discipline guarantees
        the low-watermark day itself is fully covered: a chunk is only in
        the lake if its entire fetch completed.) The gap between the high
        watermark and now is sync's job, keeping the two entry points'
        responsibilities disjoint.
        """
        start = self._history_start()
        low = self.store.low_watermark(dataset, key)
        end = _today_et() if low is None else (low.date() - timedelta(days=1))
        if end < start:
            return None  # history already reaches backfill_start
        return start, end

    def _sync_window(self, dataset: str, key: str) -> tuple[date, date]:
        """[start, end] for an incremental sync.

        Starts :data:`SYNC_OVERLAP_DAYS` before the stored high watermark so
        late-settling rows get re-merged (dedup makes this idempotent). With
        no watermark at all — nothing ingested yet — sync degrades to a full
        fetch from ``backfill_start``.
        """
        wm = self.store.watermark(dataset, key)
        start = (self._history_start() if wm is None
                 else (wm - timedelta(days=SYNC_OVERLAP_DAYS)).date())
        return start, _today_et()

    def _window_for(self, phase: str, dataset: str,
                    key: str) -> tuple[date, date] | None:
        """Phase-appropriate fetch window (None => nothing left to do)."""
        if phase == "backfill":
            window = self._backfill_window(dataset, key)
            if window is None:
                logger.info("%s/%s: backfill already reaches %s — nothing "
                            "to do", dataset, key, self.cfg.data.backfill_start)
            return window
        return self._sync_window(dataset, key)

    # ------------------------------------------------------------------ #
    # The backward march (shared by bars / treasury / news)
    # ------------------------------------------------------------------ #

    async def _march_backwards(self, dataset: str, key: str,
                               fetch: FetchFn, write: WriteFn,
                               start: date, end: date,
                               chunk_start_of: Callable[[date], date],
                               stop_on_empty: bool) -> int:
        """Fetch [start, end] in chunks, newest chunk first, writing each.

        Marching *backwards* means the most recent (most valuable) data
        lands first and an interrupted run leaves a contiguous
        [low_watermark, end] block behind — exactly what the resume logic in
        :meth:`_backfill_window` expects.

        ``chunk_start_of`` maps a chunk's end date to its natural start
        (fixed span for bars/treasury, first-of-month for news); the true
        start is clamped to ``start``.

        ``stop_on_empty`` handles vendors whose history simply ends: every
        ~30-calendar-day window contains trading days, so an empty response
        for price-like data means the front of the vendor's history (or the
        instrument's listing date) was reached and further requests are
        wasted. News, in contrast, can legitimately have empty months, so
        news marches keep going.
        """
        rows_new = 0
        n_chunks = 0
        cursor_end = end
        while cursor_end >= start:
            chunk_start = max(start, chunk_start_of(cursor_end))
            df = await fetch(chunk_start, cursor_end)
            n_chunks += 1
            if df is None or df.empty:
                if stop_on_empty:
                    logger.info("%s/%s: empty response for %s..%s — front of "
                                "vendor history reached, stopping march",
                                dataset, key, chunk_start, cursor_end)
                    break
            else:
                rows_new += write(df).rows_new
            # Continue strictly before this chunk. chunk_start <= cursor_end
            # always holds, so the cursor strictly decreases => terminates.
            cursor_end = chunk_start - timedelta(days=1)
        logger.info("%s/%s: march finished (%d chunks, +%d new rows)",
                    dataset, key, n_chunks, rows_new)
        return rows_new

    # ------------------------------------------------------------------ #
    # Mode-specific strategies
    # ------------------------------------------------------------------ #

    async def _ingest_intraday(self, client: FMPClient, spec: EndpointSpec,
                               tspec: TickerSpec, phase: str) -> int:
        """1min/5min bars: ~30-day chunks of backward-paginated history."""
        interval = spec.path.rsplit("/", 1)[-1]  # "1min" | "5min"
        window = self._window_for(phase, spec.dataset, tspec.alias)
        if window is None:
            return 0
        start, end = window

        async def fetch(s: date, e: date) -> pd.DataFrame:
            return await client.intraday_bars(tspec.symbol, interval, s, e)

        def write(df: pd.DataFrame) -> WriteResult:
            return self.store.write(spec.dataset, tspec.alias, df,
                                    dedup_keys=spec.dedup_keys)

        return await self._march_backwards(
            spec.dataset, tspec.alias, fetch, write, start, end,
            chunk_start_of=lambda ce: ce - timedelta(days=INTRADAY_CHUNK_DAYS - 1),
            stop_on_empty=True,
        )

    async def _ingest_daily(self, client: FMPClient, spec: EndpointSpec,
                            tspec: TickerSpec, phase: str) -> int:
        """Daily EOD bars: cheap enough for ONE call over the whole range.

        Backfill always requests the full configured history (a single
        request; the idempotent merge makes repeats free). Sync requests
        only the watermark-overlap tail.
        """
        if phase == "backfill":
            start, end = self._history_start(), _today_et()
        else:
            start, end = self._sync_window(spec.dataset, tspec.alias)
        df = await client.daily_bars(tspec.symbol, start, end)
        if df.empty:
            return 0
        return self.store.write(spec.dataset, tspec.alias, df,
                                dedup_keys=spec.dedup_keys).rows_new

    async def _ingest_treasury(self, client: FMPClient, spec: EndpointSpec,
                               phase: str) -> int:
        """Treasury curve: market-wide, 90-day chunks under ``_MARKET``."""
        window = self._window_for(phase, spec.dataset, MARKET_PSEUDO_TICKER)
        if window is None:
            return 0
        start, end = window

        async def fetch(s: date, e: date) -> pd.DataFrame:
            rows = await client.get(spec.path,
                                    **{"from": s.isoformat(),
                                       "to": e.isoformat()})
            df = pd.DataFrame(rows or [])
            if "date" in df.columns:
                # Canonical timestamp parsing (the raw JSON carries strings).
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
            return df

        def write(df: pd.DataFrame) -> WriteResult:
            return self.store.write(spec.dataset, MARKET_PSEUDO_TICKER, df,
                                    dedup_keys=spec.dedup_keys)

        return await self._march_backwards(
            spec.dataset, MARKET_PSEUDO_TICKER, fetch, write, start, end,
            chunk_start_of=lambda ce: ce - timedelta(days=TREASURY_CHUNK_DAYS - 1),
            stop_on_empty=True,
        )

    async def _ingest_news(self, client: FMPClient, spec: EndpointSpec,
                           tspec: TickerSpec | None, phase: str) -> int:
        """News: month-by-month backward drain of a paged endpoint.

        ``news_general`` has no symbol and lands under ``_MARKET``; stock
        news / press releases are per-ticker. News frames are timestamped by
        ``publishedDate``, so that column drives partitioning and
        watermarks. Empty months are normal (quiet tickers), so the march
        never stops early.
        """
        key = tspec.alias if tspec is not None else MARKET_PSEUDO_TICKER
        symbol = tspec.symbol if tspec is not None else None
        window = self._window_for(phase, spec.dataset, key)
        if window is None:
            return 0
        start, end = window
        limit = int(spec.extra_params.get("limit", 250))

        async def fetch(s: date, e: date) -> pd.DataFrame:
            return await client.paged_news(
                spec.path, symbol, s, e,
                symbol_param=spec.symbol_param or "symbols", limit=limit)

        def write(df: pd.DataFrame) -> WriteResult:
            return self.store.write(spec.dataset, key, df,
                                    time_col="publishedDate",
                                    dedup_keys=spec.dedup_keys)

        return await self._march_backwards(
            spec.dataset, key, fetch, write, start, end,
            chunk_start_of=lambda ce: ce.replace(day=1),  # calendar months
            stop_on_empty=False,
        )

    async def _ingest_economic(self, client: FMPClient,
                               spec: EndpointSpec) -> int:
        """Macro indicators: one series fetch per registered name.

        The endpoint takes ``name=<indicator>`` and returns that
        indicator's full history; all names are concatenated and stored
        under ``_MARKET`` with dedup key ``(name, date)``. Some payload
        variants omit the ``name`` column, so it is injected when missing —
        without it the dedup key would silently degrade to ``date`` alone
        and different indicators sharing a date would clobber each other.

        A single denied indicator name is logged and skipped; only if EVERY
        name is denied does the job re-raise, marking the whole dataset
        unavailable.
        """
        frames: list[pd.DataFrame] = []
        denied: list[str] = []
        last_denied: FMPAccessError | None = None
        for name in ECONOMIC_INDICATOR_NAMES:
            try:
                df = await client.fetch_series(spec, None, name=name)
            except FMPAccessError as exc:
                denied.append(name)
                last_denied = exc
                logger.warning("economic indicator %r denied: %s", name, exc)
                continue
            if df.empty:
                continue
            if "name" not in df.columns:
                df = df.assign(name=name)
            frames.append(df)
        if last_denied is not None and len(denied) == len(ECONOMIC_INDICATOR_NAMES):
            raise last_denied  # endpoint fully inaccessible for this key
        if not frames:
            return 0
        out = pd.concat(frames, ignore_index=True)
        return self.store.write(spec.dataset, MARKET_PSEUDO_TICKER, out,
                                dedup_keys=spec.dedup_keys).rows_new

    async def _ingest_series(self, client: FMPClient, spec: EndpointSpec,
                             tspec: TickerSpec) -> int:
        """Series & snapshot endpoints: one fetch, one merge-write.

        * ``series`` (fundamentals, earnings, grades, dividends, splits):
          the full history arrives in a single response; re-fetching on
          every run is cheap and the merge-write keeps it idempotent.
        * ``snapshot`` (quote, shares float): each sync appends one
          point-in-time row; the dedup keys (e.g. ``(symbol, timestamp)``)
          let repeated syncs within the same instant collapse to one row.
        """
        df = await client.fetch_series(spec, tspec.symbol)
        if df.empty:
            return 0
        return self.store.write(spec.dataset, tspec.alias, df,
                                dedup_keys=spec.dedup_keys).rows_new
