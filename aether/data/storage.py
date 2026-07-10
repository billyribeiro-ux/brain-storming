"""The Parquet data lake.

Layout::

    data/parquet/<dataset>/ticker=<ALIAS>/year=<YYYY>/data.parquet
    data/manifest.json          # per (dataset, ticker) watermarks + row counts

Design choices
--------------
* **Year-partitioned single files.** Even 1min bars are only ~98k rows per
  ticker-year; rewriting one year file on merge is cheap, keeps the lake
  free of small-file sprawl, and makes reads trivially predictable.
* **Idempotent merge-writes.** Every write merges with what exists, dedups
  on the dataset's natural key, and sorts by time. Backfills and incremental
  syncs can overlap or be re-run without corruption.
* **Watermarks in a manifest.** Incremental sync asks "what is the newest
  timestamp I hold?" in O(1) instead of scanning Parquet.
* Market-wide datasets (treasury rates, general news) are stored under the
  pseudo-ticker ``_MARKET``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger("aether.storage")

MARKET_PSEUDO_TICKER = "_MARKET"

_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def _safe(name: str) -> str:
    """Filesystem-safe partition value (defensive; aliases are already safe)."""
    return _SAFE.sub("_", name)


@dataclass
class WriteResult:
    dataset: str
    ticker: str
    rows_written: int      # rows in the incoming frame
    rows_new: int          # rows that were not already in the lake
    total_rows: int        # rows now stored for this (dataset, ticker)


class ParquetStore:
    """Merge-writing, watermark-tracking Parquet lake."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.parquet_root = self.root / "parquet"
        self.manifest_path = self.root / "manifest.json"
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict = self._load_manifest()

    # ------------------------------------------------------------------ #
    # Manifest / watermarks
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> dict:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str))
        tmp.replace(self.manifest_path)  # atomic on POSIX

    @staticmethod
    def _key(dataset: str, ticker: str) -> str:
        return f"{dataset}/{ticker}"

    def watermark(self, dataset: str, ticker: str) -> pd.Timestamp | None:
        """Newest time value stored for (dataset, ticker), or None."""
        entry = self._manifest.get(self._key(dataset, ticker))
        return pd.Timestamp(entry["max_time"]) if entry and entry.get("max_time") else None

    def low_watermark(self, dataset: str, ticker: str) -> pd.Timestamp | None:
        """Oldest time value stored — lets a resumed backfill skip done work."""
        entry = self._manifest.get(self._key(dataset, ticker))
        return pd.Timestamp(entry["min_time"]) if entry and entry.get("min_time") else None

    def row_count(self, dataset: str, ticker: str) -> int:
        entry = self._manifest.get(self._key(dataset, ticker))
        return int(entry["rows"]) if entry else 0

    def catalog(self) -> dict:
        """Snapshot of everything the lake holds (for dashboards/reports)."""
        return json.loads(json.dumps(self._manifest, default=str))

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #

    def _partition_dir(self, dataset: str, ticker: str) -> Path:
        return self.parquet_root / dataset / f"ticker={_safe(ticker)}"

    def write(self, dataset: str, ticker: str, df: pd.DataFrame,
              time_col: str = "date",
              dedup_keys: tuple[str, ...] = ("date",)) -> WriteResult:
        """Merge ``df`` into the lake. Idempotent: re-writing the same rows
        changes nothing. Frames without the time column are stored as a
        single un-partitioned file (snapshots)."""
        if df is None or df.empty:
            return WriteResult(dataset, ticker, 0, 0, self.row_count(dataset, ticker))
        df = df.copy()

        part_dir = self._partition_dir(dataset, ticker)
        part_dir.mkdir(parents=True, exist_ok=True)

        has_time = time_col in df.columns
        if has_time:
            df[time_col] = pd.to_datetime(df[time_col])
            groups = df.groupby(df[time_col].dt.year)
        else:
            groups = [(0, df)]  # single "year 0" bucket for snapshot data

        new_rows = 0
        for year, chunk in groups:
            path = (part_dir / f"year={int(year)}" / "data.parquet") if has_time \
                else (part_dir / "data.parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                existing = pd.read_parquet(path)
                before = len(existing)
                merged = pd.concat([existing, chunk], ignore_index=True)
            else:
                before = 0
                merged = chunk
            keys = [k for k in dedup_keys if k in merged.columns] or None
            merged = merged.drop_duplicates(subset=keys, keep="last")
            if has_time:
                merged = merged.sort_values(time_col)
            merged = merged.reset_index(drop=True)
            merged.to_parquet(path, index=False)
            new_rows += len(merged) - before

        # ---- update manifest ------------------------------------------- #
        key = self._key(dataset, ticker)
        entry = self._manifest.get(key, {"rows": 0, "min_time": None, "max_time": None})
        entry["rows"] = int(entry["rows"]) + new_rows
        if has_time:
            lo, hi = df[time_col].min(), df[time_col].max()
            entry["min_time"] = str(min(pd.Timestamp(entry["min_time"]), lo)) \
                if entry["min_time"] else str(lo)
            entry["max_time"] = str(max(pd.Timestamp(entry["max_time"]), hi)) \
                if entry["max_time"] else str(hi)
        self._manifest[key] = entry
        self._save_manifest()

        logger.info("write %s/%s: +%d new (%d in frame, %d total)",
                    dataset, ticker, new_rows, len(df), entry["rows"])
        return WriteResult(dataset, ticker, len(df), new_rows, entry["rows"])

    def read(self, dataset: str, ticker: str,
             start: pd.Timestamp | str | None = None,
             end: pd.Timestamp | str | None = None,
             columns: list[str] | None = None,
             time_col: str = "date") -> pd.DataFrame:
        """Read (dataset, ticker), optionally sliced to [start, end].

        Year partitions outside the slice are never opened.
        """
        part_dir = self._partition_dir(dataset, ticker)
        if not part_dir.exists():
            return pd.DataFrame()
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        frames: list[pd.DataFrame] = []
        for path in sorted(part_dir.rglob("data.parquet")):
            year_dir = path.parent.name
            if year_dir.startswith("year="):
                year = int(year_dir.split("=", 1)[1])
                if year != 0:  # snapshot bucket has no time meaning
                    if start_ts is not None and year < start_ts.year:
                        continue
                    if end_ts is not None and year > end_ts.year:
                        continue
            frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if time_col in out.columns:
            out[time_col] = pd.to_datetime(out[time_col])
            if start_ts is not None:
                out = out[out[time_col] >= start_ts]
            if end_ts is not None:
                out = out[out[time_col] <= end_ts]
            out = out.sort_values(time_col).reset_index(drop=True)
        return out

    def datasets_for(self, ticker: str) -> list[str]:
        return sorted({k.split("/", 1)[0] for k in self._manifest
                       if k.endswith(f"/{ticker}")})
