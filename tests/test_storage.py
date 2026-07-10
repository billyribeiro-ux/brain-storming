"""Pins the ParquetStore contract: idempotent merge-writes, watermarks,
year partitioning, dedup semantics, snapshot round-trips, and the catalog.

The lake is the foundation every training run reads from; a silent
double-write or a broken year-boundary read would corrupt everything
downstream, so these invariants get their own file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aether.data.storage import MARKET_PSEUDO_TICKER, ParquetStore

from tests.conftest import DATASET_1M, TEST_TICKERS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _daily_frame(dates: list[str], values: list[float] | None = None) -> pd.DataFrame:
    """Minimal time-series frame keyed by 'date'."""
    if values is None:
        values = list(np.arange(float(len(dates))))
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "lake")


# --------------------------------------------------------------------------- #
# Merge idempotence
# --------------------------------------------------------------------------- #

class TestIdempotence:
    def test_rewrite_same_rows_adds_nothing(self, store: ParquetStore) -> None:
        df = _daily_frame(["2026-01-05", "2026-01-06", "2026-01-07"])
        w1 = store.write("ds", "T", df)
        assert w1.rows_written == 3
        assert w1.rows_new == 3
        assert w1.total_rows == 3

        w2 = store.write("ds", "T", df)          # exact same frame again
        assert w2.rows_written == 3
        assert w2.rows_new == 0                  # nothing new merged in
        assert w2.total_rows == 3

        out = store.read("ds", "T")
        assert len(out) == 3
        assert out["date"].is_monotonic_increasing

    def test_partial_overlap_merges_only_new(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-05", "2026-01-06"]))
        w = store.write("ds", "T", _daily_frame(["2026-01-06", "2026-01-07"],
                                                [1.0, 9.0]))
        assert w.rows_new == 1                   # only Jan 7 is new
        assert len(store.read("ds", "T")) == 3

    def test_synthetic_lake_rewrite_is_idempotent(self, synthetic_store,
                                                  synthetic_frames) -> None:
        # Re-writing the exact bars the fixture already stored is a no-op.
        alias = TEST_TICKERS[0]
        w = synthetic_store.write(DATASET_1M, alias, synthetic_frames[alias])
        assert w.rows_new == 0


# --------------------------------------------------------------------------- #
# Watermarks
# --------------------------------------------------------------------------- #

class TestWatermarks:
    def test_high_and_low_watermarks(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-06", "2026-01-08"]))
        assert store.watermark("ds", "T") == pd.Timestamp("2026-01-08")
        assert store.low_watermark("ds", "T") == pd.Timestamp("2026-01-06")

    def test_backfill_extends_low_watermark_only(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-06", "2026-01-08"]))
        store.write("ds", "T", _daily_frame(["2025-12-30"]))   # older backfill
        assert store.low_watermark("ds", "T") == pd.Timestamp("2025-12-30")
        assert store.watermark("ds", "T") == pd.Timestamp("2026-01-08")

    def test_missing_key_returns_none(self, store: ParquetStore) -> None:
        assert store.watermark("nope", "T") is None
        assert store.low_watermark("nope", "T") is None


# --------------------------------------------------------------------------- #
# Year partition boundaries
# --------------------------------------------------------------------------- #

class TestYearPartitions:
    DEC = ["2025-12-29", "2025-12-30", "2025-12-31"]
    JAN = ["2026-01-02", "2026-01-05", "2026-01-06"]

    def _write_spanning(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(self.DEC + self.JAN))

    def test_partition_files_split_by_year(self, store: ParquetStore) -> None:
        self._write_spanning(store)
        base = store.parquet_root / "ds" / "ticker=T"
        assert (base / "year=2025" / "data.parquet").is_file()
        assert (base / "year=2026" / "data.parquet").is_file()

    def test_read_slices_across_the_boundary(self, store: ParquetStore) -> None:
        self._write_spanning(store)

        only_2026 = store.read("ds", "T", start="2026-01-01")
        assert list(only_2026["date"].dt.year.unique()) == [2026]
        assert len(only_2026) == len(self.JAN)

        only_2025 = store.read("ds", "T", end="2025-12-31")
        assert list(only_2025["date"].dt.year.unique()) == [2025]
        assert len(only_2025) == len(self.DEC)

        window = store.read("ds", "T", start="2025-12-30", end="2026-01-02")
        assert [str(d.date()) for d in window["date"]] == \
            ["2025-12-30", "2025-12-31", "2026-01-02"]

    def test_full_read_is_sorted(self, store: ParquetStore) -> None:
        self._write_spanning(store)
        out = store.read("ds", "T")
        assert len(out) == len(self.DEC) + len(self.JAN)
        assert out["date"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Dedup semantics
# --------------------------------------------------------------------------- #

class TestDedup:
    def test_dedup_keeps_latest_write(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-05"], [1.0]))
        w = store.write("ds", "T", _daily_frame(["2026-01-05"], [2.0]))
        # The row count did not grow, but the value was replaced.
        assert w.rows_new == 0
        out = store.read("ds", "T")
        assert len(out) == 1
        assert float(out["value"].iloc[0]) == 2.0

    def test_custom_dedup_keys(self, store: ParquetStore) -> None:
        # News-like data: several rows per timestamp, keyed by (date, url).
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "url": ["a", "b"],
            "title": ["one", "two"],
        })
        store.write("news", "T", df, dedup_keys=("date", "url"))
        w = store.write("news", "T", df, dedup_keys=("date", "url"))
        assert w.rows_new == 0
        assert len(store.read("news", "T")) == 2


# --------------------------------------------------------------------------- #
# Snapshots (no time column)
# --------------------------------------------------------------------------- #

class TestSnapshots:
    def test_snapshot_round_trip(self, store: ParquetStore) -> None:
        snap = pd.DataFrame({"symbol": ["AAPL", "SPY"],
                             "market_cap": [3.1e12, float("nan")]})
        w = store.write("profile_snapshot", MARKET_PSEUDO_TICKER, snap)
        assert w.rows_new == 2
        # Snapshots carry no watermark (there is no time column).
        assert store.watermark("profile_snapshot", MARKET_PSEUDO_TICKER) is None

        out = store.read("profile_snapshot", MARKET_PSEUDO_TICKER)
        assert list(out["symbol"]) == ["AAPL", "SPY"]
        assert float(out["market_cap"].iloc[0]) == pytest.approx(3.1e12)

    def test_snapshot_rewrite_is_idempotent(self, store: ParquetStore) -> None:
        snap = pd.DataFrame({"symbol": ["AAPL"], "beta": [1.2]})
        store.write("snap", MARKET_PSEUDO_TICKER, snap)
        w = store.write("snap", MARKET_PSEUDO_TICKER, snap)
        assert w.rows_new == 0
        assert len(store.read("snap", MARKET_PSEUDO_TICKER)) == 1

    def test_empty_frame_is_a_noop(self, store: ParquetStore) -> None:
        w = store.write("ds", "T", pd.DataFrame())
        assert w.rows_written == 0
        assert w.rows_new == 0
        assert store.row_count("ds", "T") == 0


# --------------------------------------------------------------------------- #
# Catalog / manifest
# --------------------------------------------------------------------------- #

class TestCatalog:
    def test_catalog_reflects_writes(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-05", "2026-01-06"]))
        cat = store.catalog()
        assert "ds/T" in cat
        assert int(cat["ds/T"]["rows"]) == 2

    def test_row_count_and_datasets_for(self, store: ParquetStore) -> None:
        store.write("a", "T", _daily_frame(["2026-01-05"]))
        store.write("b", "T", _daily_frame(["2026-01-05"]))
        assert store.row_count("a", "T") == 1
        assert store.datasets_for("T") == ["a", "b"]

    def test_manifest_survives_reopen(self, store: ParquetStore) -> None:
        store.write("ds", "T", _daily_frame(["2026-01-05"]))
        reopened = ParquetStore(store.root)
        assert reopened.row_count("ds", "T") == 1
        assert reopened.watermark("ds", "T") == pd.Timestamp("2026-01-05")
