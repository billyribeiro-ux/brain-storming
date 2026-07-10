"""Async client for the FMP *stable* API.

Engineered against measured behavior of the live API (probed 2026-07-10):

* Intraday ``historical-chart`` responses are **server-capped** (~1170 rows
  ≈ 3 trading days for 1min) and returned **newest-first** within the
  requested window. Deep history therefore requires *backward pagination*:
  ask for ``[start, to]``, take what comes back, move ``to`` just before the
  oldest returned timestamp, repeat until the window is exhausted or the API
  runs out of history.
* Legacy ``/api/v3`` is rejected (403) for post-Aug-2025 keys; only the
  stable base URL is used.
* Rate limits are plan-based per minute; a token bucket paces requests so
  multi-day backfills run at full speed without tripping 429s.

The client is deliberately *thin and faithful*: it returns data exactly as
FMP provides it (plus canonical timestamp parsing). All interpretation is
left to the perception layer — Aether learns meaning; the client only
guarantees transport, completeness, and honesty about failures.
"""

from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd
from tenacity import (AsyncRetrying, retry_if_exception, stop_after_attempt,
                      wait_exponential_jitter)

from ..config import FMPConfig
from ..utils.logging import get_logger
from .endpoints import ENDPOINTS, EndpointSpec

logger = get_logger("aether.fmp")


class FMPError(Exception):
    """Base error for FMP transport problems."""


class FMPAccessError(FMPError):
    """Endpoint exists but this key's plan cannot access it (402/403/404)."""


class FMPTransientError(FMPError):
    """Retryable condition: 429 rate limit, 5xx, network timeouts."""


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, FMPTransientError)


class _TokenBucket:
    """Async token bucket pacing requests to ``rate`` per 60 seconds.

    Two deliberate choices keep any fixed 60-second window at (or barely
    above) the plan limit, because FMP may enforce its per-minute quota on
    fixed windows rather than a rolling average:

    * **The bucket starts EMPTY.** Starting full (the naive choice) would
      let a fresh backfill fire ``capacity`` requests instantly *plus*
      ``fill_rate * 60`` refills — i.e. ~2x the plan limit inside the first
      minute — guaranteeing a 429 storm at every run start, and the retry
      schedule (~15-20 s of total backoff over 5 attempts) can exhaust
      itself inside that same rate window, abandoning jobs outright.
    * **Burst capacity is a few seconds of budget, not a full minute.**
      Accrual during idle gaps mid-run is capped at ~2 seconds' worth of
      tokens, so a post-idle burst can exceed the steady rate by only a few
      requests instead of doubling it.
    """

    def __init__(self, rate_per_minute: int):
        fill_rate = rate_per_minute / 60.0
        # Burst allowance ≈ 2 seconds of steady-state budget (min 1 token so
        # tiny plans still make progress one request at a time).
        self.capacity = max(1.0, fill_rate * 2.0)
        self.tokens = 0.0                     # start empty: pace from t=0
        self.fill_rate = fill_rate
        self.updated = _time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = _time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.updated) * self.fill_rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.fill_rate)


@dataclass
class CapabilityStatus:
    dataset: str
    available: bool
    http_status: int
    detail: str = ""


class FMPClient:
    """Rate-limited, retrying async client. Use as an async context manager:

    >>> async with FMPClient(cfg) as fmp:
    ...     bars = await fmp.intraday_bars("AAPL", "1min",
    ...                                    date(2026, 1, 2), date(2026, 7, 9))
    """

    def __init__(self, cfg: FMPConfig | None = None):
        self.cfg = cfg or FMPConfig.from_env()
        if not self.cfg.api_key:
            raise FMPError(
                "FMP_API_KEY is not set. Put it in the environment or a .env file."
            )
        self._bucket = _TokenBucket(self.cfg.requests_per_minute)
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._client: httpx.AsyncClient | None = None
        self.request_count = 0

    async def __aenter__(self) -> "FMPClient":
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout_seconds,
            headers={"User-Agent": "aether/0.1"},
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Core transport
    # ------------------------------------------------------------------ #

    async def get(self, path: str, **params: Any) -> Any:
        """One rate-limited, retried GET returning parsed JSON.

        Raises FMPAccessError for plan/permission problems (not retried) and
        FMPTransientError after retries are exhausted.
        """
        if self._client is None:
            raise FMPError("Client used outside `async with` context")
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self.cfg.api_key

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_transient),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            stop=stop_after_attempt(self.cfg.max_retries),
            reraise=True,
        ):
            with attempt:
                await self._bucket.acquire()
                async with self._sem:
                    try:
                        resp = await self._client.get(path, params=params)
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        raise FMPTransientError(f"{path}: {exc}") from exc
                self.request_count += 1
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (402, 403, 404):
                    raise FMPAccessError(
                        f"{path} -> HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise FMPTransientError(f"{path} -> HTTP {resp.status_code}")
                raise FMPError(f"{path} -> HTTP {resp.status_code}: {resp.text[:200]}")

    # ------------------------------------------------------------------ #
    # Bars
    # ------------------------------------------------------------------ #

    #: Measured server behavior (probed live 2026-07-10): a request returns
    #: ONLY bars inside a fixed calendar window anchored at ``to`` —
    #: ``[to - (W-1) days, to]`` — regardless of ``from`` (which merely clips).
    #: W was measured as 3 calendar days for 1min and 10 for 5min. ONLY
    #: measured intervals are listed: a guessed-too-large W would make the
    #: march step past bars the server never returned — silent data loss —
    #: whereas the too-small fallback of 3 merely costs extra requests. Any
    #: unmeasured interval (15min/30min/1hour/4hour) therefore uses the
    #: safe fallback until someone measures its true window and adds it.
    INTRADAY_WINDOW_DAYS: dict[str, int] = {
        "1min": 3, "5min": 10,
    }

    async def intraday_bars(self, symbol: str, interval: str,
                            start: date, end: date) -> pd.DataFrame:
        """Full intraday history in [start, end] via calendar-chunk marching.

        The cursor marches backwards in fixed windows of
        ``INTRADAY_WINDOW_DAYS[interval]`` calendar days. Progress NEVER
        depends on response content, because the server legitimately returns
        empty payloads when a window lands on weekends/holidays (measured:
        ``to`` on the Sunday after July 4th → 0 rows). Chunks containing no
        trading sessions are skipped without a request. Marching stops early
        only after ``empty_chunk_limit`` consecutive empty *trading* chunks —
        i.e. months of silence — which marks the true front of the symbol's
        history.

        Caution: the server silently IGNORES malformed date params (a
        datetime-granular ``to`` was measured to behave like "no ``to`` at
        all"), so only pure ISO dates are ever sent.

        Returns a DataFrame with columns
        ``date (datetime64, ET-naive), open, high, low, close, volume``
        sorted oldest→newest, deduplicated on ``date``.
        """
        from ..utils.market_time import trading_days  # local import: avoid cycles

        window = self.INTRADAY_WINDOW_DAYS.get(interval, 3)
        frames: list[pd.DataFrame] = []
        to_cursor = end
        consecutive_empty = 0
        empty_chunk_limit = max(1, 120 // window)   # ~4 months of nothing

        while to_cursor >= start:
            chunk_from = max(start, to_cursor - timedelta(days=window - 1))
            next_cursor = chunk_from - timedelta(days=1)
            # Weekend/holiday-only chunks cannot contain bars: skip the call.
            if not trading_days(chunk_from, to_cursor):
                to_cursor = next_cursor
                continue
            rows = await self.get(
                f"historical-chart/{interval}",
                symbol=symbol,
                **{"from": chunk_from.isoformat(), "to": to_cursor.isoformat()},
            )
            if rows:
                consecutive_empty = 0
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                frames.append(df)
            else:
                consecutive_empty += 1
                if consecutive_empty >= empty_chunk_limit:
                    logger.info("%s %s: history front reached near %s "
                                "(%d consecutive empty chunks)",
                                symbol, interval, to_cursor, consecutive_empty)
                    break
            to_cursor = next_cursor

        if not frames:
            return _empty_bars_frame()
        out = pd.concat(frames, ignore_index=True)
        out = (out.drop_duplicates(subset="date")
                  .sort_values("date")
                  .reset_index(drop=True))
        # Enforce the requested window (the API may round outward).
        mask = (out["date"] >= pd.Timestamp(start)) & \
               (out["date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        return out.loc[mask].reset_index(drop=True)

    async def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Adjusted EOD bars (single call covers years)."""
        rows = await self.get(
            "historical-price-eod/full", symbol=symbol,
            **{"from": start.isoformat(), "to": end.isoformat()},
        )
        if not rows:
            return _empty_bars_frame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # News (page-based)
    # ------------------------------------------------------------------ #

    async def paged_news(self, path: str, symbol: str | None,
                         start: date, end: date,
                         symbol_param: str = "symbols",
                         limit: int = 250, max_pages: int = 400) -> pd.DataFrame:
        """Drain a paged news endpoint over a date window.

        If the window holds more than ``max_pages * limit`` rows the drain
        is TRUNCATED at the newest ``max_pages * limit`` articles — the
        pages arrive newest-first, so the window's older tail is lost. This
        is loudly logged (a caller marching month-by-month would otherwise
        step past the window and never revisit the missing tail); callers
        seeing the warning should re-fetch with a narrower window.
        """
        frames: list[pd.DataFrame] = []
        drained = False  # saw a short/empty page => window fully consumed
        for page in range(max_pages):
            params: dict[str, Any] = {
                "page": page, "limit": limit,
                "from": start.isoformat(), "to": end.isoformat(),
            }
            if symbol is not None:
                params[symbol_param] = symbol
            rows = await self.get(path, **params)
            if not rows:
                drained = True
                break
            frames.append(pd.DataFrame(rows))
            if len(rows) < limit:
                drained = True
                break
        if not drained:
            # Every page (including the last one) came back full: the window
            # almost certainly holds more rows than we were allowed to page
            # through. Do not fail — partial news is useful — but make the
            # loss impossible to miss.
            logger.warning(
                "%s: window %s..%s exceeded max_pages=%d x limit=%d "
                "(~%d rows) — OLDER articles in this window were TRUNCATED; "
                "re-fetch with a narrower window to recover them",
                path, start, end, max_pages, limit, max_pages * limit,
            )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "publishedDate" in out.columns:
            out["publishedDate"] = pd.to_datetime(out["publishedDate"])
            out = (out.drop_duplicates(subset=["publishedDate", "title"])
                      .sort_values("publishedDate")
                      .reset_index(drop=True))
        return out

    # ------------------------------------------------------------------ #
    # Generic endpoint pull + capability probe
    # ------------------------------------------------------------------ #

    async def fetch_series(self, spec: EndpointSpec, symbol: str | None,
                           **extra: Any) -> pd.DataFrame:
        """Pull a `series`/`snapshot` endpoint into a DataFrame as-is."""
        params = dict(spec.extra_params) | extra
        if spec.symbol_param and symbol is not None:
            params[spec.symbol_param] = symbol
        rows = await self.get(spec.path, **params)
        if isinstance(rows, dict):
            rows = [rows]
        df = pd.DataFrame(rows or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    async def probe_capabilities(
        self, probe_symbol: str = "AAPL"
    ) -> dict[str, CapabilityStatus]:
        """Active sensing of the API: test every registered endpoint with a
        cheap request and report what this key can actually reach.

        Classification of outcomes — chosen so that a persisted ``False``
        always means *the plan genuinely cannot reach this endpoint*
        (ingestion skips ``False`` datasets on every future run, so a wrong
        ``False`` silently starves the lake):

        * 200 with data → available.
        * 200 with an EMPTY payload → **available**. Emptiness is evidence
          of *no data in the probe window* (e.g. a quiet news week for the
          probe symbol), not of no access — the request itself succeeded.
        * :class:`FMPAccessError` (402/403/404) → unavailable: the API
          explicitly refused.
        * :class:`FMPTransientError` (timeout, 5xx, 429 after retries) →
          **omitted from the results entirely** (capability *unknown*).
          A network blip must not be persisted as a permanent verdict;
          absent entries are treated by ingestion as "attempt and see".
        """
        results: dict[str, CapabilityStatus] = {}
        for name, spec in ENDPOINTS.items():
            params: dict[str, Any] = dict(spec.probe_params)
            if spec.symbol_param:
                params[spec.symbol_param] = probe_symbol
            if spec.mode in ("window", "paged"):
                today = date.today()
                params.setdefault("from", (today - timedelta(days=7)).isoformat())
                params.setdefault("to", today.isoformat())
            if name == "economic_indicators":
                params["name"] = "federalFunds"
            if name in ("income_statement", "key_metrics", "ratios",
                        "earnings", "grades", "dividends", "splits",
                        "balance_sheet", "cash_flow"):
                params["limit"] = 1
            try:
                payload = await self.get(spec.path, **params)
                # A 200 proves the endpoint is reachable even when the probe
                # window happens to contain no rows (see docstring).
                detail = "" if payload else "empty response (reachable)"
                results[name] = CapabilityStatus(name, True, 200, detail)
            except FMPAccessError as exc:
                results[name] = CapabilityStatus(name, False, 403, str(exc)[:200])
            except FMPTransientError as exc:
                # Transient ≠ unavailable: leave the entry absent (unknown)
                # so ingestion still attempts it rather than skipping the
                # dataset forever on the strength of one network blip.
                logger.warning("probe %-22s -> transient failure (%s) — "
                               "capability unknown, not recorded", name, exc)
                continue
            except FMPError as exc:
                results[name] = CapabilityStatus(name, False, 0, str(exc)[:200])
            logger.info("probe %-22s -> %s", name,
                        "OK" if results[name].available else "unavailable")
        return results


def _empty_bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"),
         "open": pd.Series(dtype="float64"),
         "high": pd.Series(dtype="float64"),
         "low": pd.Series(dtype="float64"),
         "close": pd.Series(dtype="float64"),
         "volume": pd.Series(dtype="float64")}
    )
