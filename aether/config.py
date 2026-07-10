"""Central configuration for Aether.

Everything that identifies *what* the system watches (tickers), *where* data
lives (paths), and *how* it talks to the outside world (FMP API) is defined
here. No trading logic lives in this file — Aether learns that itself.

Design notes
------------
* The FMP API key is read from the environment (``FMP_API_KEY``) or from a
  local ``.env`` file. It is never stored in code or in the data lake.
* Index symbols like ``^GSPC`` are queried from FMP with their caret symbol
  but stored under a filesystem-safe ``alias`` (``SPX``).
* The verified FMP tier for this project is the **stable** API
  (``https://financialmodelingprep.com/stable``). Legacy ``/api/v3`` is not
  available for post-Aug-2025 keys and is deliberately not used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Environment loading
# --------------------------------------------------------------------------- #

def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no external dependency).

    Existing environment variables win over file values so that shell-level
    overrides behave as expected.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# --------------------------------------------------------------------------- #
# Ticker universe
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TickerSpec:
    """A single instrument in Aether's universe.

    Attributes
    ----------
    symbol:
        The exact symbol FMP expects (e.g. ``^GSPC`` for the S&P 500 index).
    alias:
        Filesystem/model-safe identifier used for storage partitions,
        embedding ids, and display (e.g. ``SPX``).
    kind:
        ``equity`` | ``etf`` | ``index``. Some FMP endpoints (fundamentals,
        float, grades) only make sense for equities; ingestion uses this to
        skip impossible requests instead of burning rate limit on 404s.
    """

    symbol: str
    alias: str
    kind: str  # "equity" | "etf" | "index"

    @property
    def has_fundamentals(self) -> bool:
        return self.kind == "equity"


#: The exact, fixed universe. Aether learns a distinct "DNA" embedding per
#: entry — order matters because it defines the integer ticker id used by
#: the perception model's embedding table. Append only; never reorder.
TICKERS: tuple[TickerSpec, ...] = (
    TickerSpec("AAPL", "AAPL", "equity"),
    TickerSpec("NVDA", "NVDA", "equity"),
    TickerSpec("TSLA", "TSLA", "equity"),
    TickerSpec("AMZN", "AMZN", "equity"),
    TickerSpec("NFLX", "NFLX", "equity"),
    TickerSpec("CSCO", "CSCO", "equity"),
    TickerSpec("SPY", "SPY", "etf"),
    TickerSpec("QQQ", "QQQ", "etf"),
    TickerSpec("IWM", "IWM", "etf"),
    TickerSpec("^GSPC", "SPX", "index"),
)

#: alias -> integer id used by embedding tables. Stable by construction.
TICKER_IDS: dict[str, int] = {t.alias: i for i, t in enumerate(TICKERS)}


def ticker_by_alias(alias: str) -> TickerSpec:
    for t in TICKERS:
        if t.alias == alias:
            return t
    raise KeyError(f"Unknown ticker alias: {alias!r}")


# --------------------------------------------------------------------------- #
# FMP API configuration
# --------------------------------------------------------------------------- #

@dataclass
class FMPConfig:
    """Connection + throttling parameters for the FMP stable API.

    ``requests_per_minute`` should match the subscription tier
    (Starter=300, Premium=750, Ultimate=3000). The client enforces this
    with a token bucket so backfills can run flat-out without 429s.
    """

    base_url: str = "https://financialmodelingprep.com/stable"
    api_key: str = ""
    requests_per_minute: int = 300
    max_concurrency: int = 8          # simultaneous in-flight requests
    timeout_seconds: float = 30.0
    max_retries: int = 5              # exponential backoff on 429/5xx/timeouts

    @classmethod
    def from_env(cls) -> "FMPConfig":
        load_dotenv()
        key = os.environ.get("FMP_API_KEY", "")
        rpm = int(os.environ.get("FMP_REQUESTS_PER_MINUTE", "300"))
        return cls(api_key=key, requests_per_minute=rpm)


# --------------------------------------------------------------------------- #
# Data lake configuration
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    """Where and how raw market data is persisted."""

    root: Path = Path("data")
    backfill_start: str = "2020-01-01"   # earliest date to attempt for history
    intraday_intervals: tuple[str, ...] = ("1min", "5min")

    @property
    def parquet_root(self) -> Path:
        return self.root / "parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def capabilities_path(self) -> Path:
        return self.root / "capabilities.json"

    @property
    def stats_root(self) -> Path:
        """Per-ticker statistical fingerprints used by perception preprocessing."""
        return self.root / "ticker_stats"

    @property
    def cache_root(self) -> Path:
        """Materialized tensor caches for fast DataLoader startup."""
        return self.root / "cache"

    @classmethod
    def from_env(cls) -> "DataConfig":
        load_dotenv()
        return cls(
            root=Path(os.environ.get("AETHER_DATA_ROOT", "data")),
            backfill_start=os.environ.get("AETHER_BACKFILL_START", "2020-01-01"),
        )


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #

@dataclass
class AetherConfig:
    fmp: FMPConfig = field(default_factory=FMPConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_env(cls) -> "AetherConfig":
        return cls(fmp=FMPConfig.from_env(), data=DataConfig.from_env())
