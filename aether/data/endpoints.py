"""Registry of every FMP stable-API endpoint Aether decodes.

Aether's mandate is to read *everything beneath the surface* that the data
plan exposes. Rather than scattering URL strings through the codebase, every
endpoint is declared here once, with:

* the dataset name used for Parquet partitioning,
* how requests are parameterized (symbol / date-window / paged),
* which ticker kinds it applies to (fundamentals make no sense for ``^GSPC``),
* its verified availability for the current API key.

``probe`` support ("active sensing" of the API itself) lets Aether discover
what a key can access and record it in ``data/capabilities.json`` instead of
assuming. Endpoints verified live on 2026-07-10 for this project's key are
marked ``verified=True``; options endpoints returned 404 (not included in the
current plan) and are kept in the registry as aspirational so a plan upgrade
lights them up with zero code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EndpointSpec:
    """One FMP stable endpoint.

    Attributes
    ----------
    dataset:
        Storage dataset name (e.g. ``bars_1min``); also the registry key.
    path:
        Path under the stable base URL, e.g. ``historical-chart/1min``.
    mode:
        ``window``  — supports from/to; fetched by backward date pagination.
        ``paged``   — supports page/limit pagination (news).
        ``snapshot``— point-in-time pull, re-fetched on every sync.
        ``series``  — full history in one call (fundamentals, macro).
    symbol_param:
        Query-parameter name carrying the symbol, or None for market-wide
        endpoints (treasury rates, economic indicators).
    kinds:
        Ticker kinds this endpoint applies to.
    extra_params:
        Constant query parameters always sent.
    probe_params:
        Cheap parameters used by the capability probe.
    verified:
        True if confirmed live for the current key (2026-07-10 probe).
    """

    dataset: str
    path: str
    mode: str
    symbol_param: str | None = "symbol"
    kinds: tuple[str, ...] = ("equity", "etf", "index")
    extra_params: dict = field(default_factory=dict)
    probe_params: dict = field(default_factory=dict)
    verified: bool = False
    dedup_keys: tuple[str, ...] = ("date",)
    notes: str = ""


ENDPOINTS: dict[str, EndpointSpec] = {e.dataset: e for e in [
    # ------------------------------------------------------------------ #
    # Price microstructure — the heartbeat.
    # Server caps each response (~3 trading days of 1min, newest-first),
    # so ingestion pages backwards on `to`.
    # ------------------------------------------------------------------ #
    EndpointSpec("bars_1min", "historical-chart/1min", "window", verified=True,
                 notes="Server returns only a 3-calendar-day window anchored "
                       "at `to` (measured); client marches the cursor."),
    EndpointSpec("bars_5min", "historical-chart/5min", "window", verified=True,
                 notes="10-calendar-day window anchored at `to` (measured)."),
    EndpointSpec("bars_daily", "historical-price-eod/full", "window", verified=True,
                 notes="Split/dividend-adjusted EOD with vwap."),
    EndpointSpec("quote", "quote", "snapshot", verified=True,
                 dedup_keys=("symbol", "timestamp"),
                 notes="Real-time quote incl. day range, avg volume, market cap."),

    # ------------------------------------------------------------------ #
    # Narrative / sentiment raw material.
    # ------------------------------------------------------------------ #
    EndpointSpec("news_stock", "news/stock", "paged", symbol_param="symbols",
                 kinds=("equity", "etf"), verified=True,
                 extra_params={"limit": 250},
                 dedup_keys=("publishedDate", "title"),
                 notes="Ticker-tagged articles; paged, from/to supported."),
    EndpointSpec("news_press_releases", "news/press-releases", "paged",
                 symbol_param="symbols", kinds=("equity",), verified=True,
                 extra_params={"limit": 250},
                 dedup_keys=("publishedDate", "title")),
    EndpointSpec("news_general", "news/general-latest", "paged",
                 symbol_param=None, verified=True,
                 extra_params={"limit": 250},
                 dedup_keys=("publishedDate", "title"),
                 notes="Macro narrative context shared across all tickers."),

    # ------------------------------------------------------------------ #
    # Fundamentals — slow-moving drivers of each ticker's DNA.
    # ------------------------------------------------------------------ #
    EndpointSpec("income_statement", "income-statement", "series",
                 kinds=("equity",), extra_params={"period": "quarter", "limit": 60},
                 verified=True),
    EndpointSpec("balance_sheet", "balance-sheet-statement", "series",
                 kinds=("equity",), extra_params={"period": "quarter", "limit": 60}),
    EndpointSpec("cash_flow", "cash-flow-statement", "series",
                 kinds=("equity",), extra_params={"period": "quarter", "limit": 60}),
    EndpointSpec("key_metrics", "key-metrics", "series",
                 kinds=("equity",), extra_params={"period": "quarter", "limit": 60},
                 verified=True),
    EndpointSpec("ratios", "ratios", "series",
                 kinds=("equity",), extra_params={"period": "quarter", "limit": 60}),
    EndpointSpec("earnings", "earnings", "series",
                 kinds=("equity",), extra_params={"limit": 60}, verified=True,
                 notes="Past + upcoming earnings with estimates — event risk clock."),
    EndpointSpec("grades", "grades", "series",
                 kinds=("equity",), extra_params={"limit": 500}, verified=True,
                 dedup_keys=("date", "gradingCompany"),
                 notes="Analyst upgrades/downgrades — sentiment shocks."),
    EndpointSpec("shares_float", "shares-float", "snapshot",
                 kinds=("equity",), verified=True,
                 dedup_keys=("symbol", "date"),
                 notes="Float — squeeze/liquidity structure."),
    EndpointSpec("dividends", "dividends", "series",
                 kinds=("equity", "etf"), extra_params={"limit": 100}),
    EndpointSpec("splits", "splits", "series",
                 kinds=("equity",), extra_params={"limit": 100}),

    # ------------------------------------------------------------------ #
    # Macro fabric — market-wide, stored once under the pseudo-ticker
    # ``_MARKET`` and cross-attended by every instrument.
    # ------------------------------------------------------------------ #
    EndpointSpec("treasury_rates", "treasury-rates", "window",
                 symbol_param=None, verified=True,
                 notes="Full curve daily — rate regime context."),
    EndpointSpec("economic_indicators", "economic-indicators", "series",
                 symbol_param=None,
                 extra_params={},  # requires name=<indicator>; expanded in ingestion
                 verified=True,
                 dedup_keys=("name", "date"),
                 notes="GDP, CPI, fed funds, unemployment, ... one call per name."),

    # ------------------------------------------------------------------ #
    # Options — NOT included in the current plan (probed 404 on
    # 2026-07-10). Kept for automatic activation via capability probe.
    # ------------------------------------------------------------------ #
    EndpointSpec("options_chain", "options-chain", "snapshot",
                 kinds=("equity", "etf"), verified=False,
                 notes="404 on current plan; re-probe after upgrades."),
]}

#: Economic indicator names pulled through the ``economic_indicators`` endpoint.
ECONOMIC_INDICATOR_NAMES: tuple[str, ...] = (
    "GDP", "realGDP", "CPI", "inflationRate", "federalFunds",
    "unemploymentRate", "retailSales", "consumerSentiment",
    "durableGoods", "initialClaims", "industrialProductionTotalIndex",
    "totalNonfarmPayroll",
)
