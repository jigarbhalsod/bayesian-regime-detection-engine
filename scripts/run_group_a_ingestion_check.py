"""Run a small Phase 5 Group A real-source ingestion check.

This script intentionally performs a bounded validation pull instead of a full
historical backfill. It proves connector/storage/manifest behavior against live
sources while keeping downloaded data out of Git.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collectors import HistoricalMarketDataCollector, MacroExternalDataCollector
from src.data.config import DEFAULT_START_DATE
from src.data.schemas import IngestionRequest


CHECK_DATASETS = (
    "nifty_50",
    "india_vix",
    "cpi_india",
    "sp500",
    "cboe_vix",
    "crude_oil",
)


def parse_date(value: str | None) -> date | None:
    if value in (None, "", "latest_available"):
        return None
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default="latest_available")
    parser.add_argument("--dataset", action="append", choices=CHECK_DATASETS)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    datasets = tuple(args.dataset or CHECK_DATASETS)
    market_collector = HistoricalMarketDataCollector()
    macro_external_collector = MacroExternalDataCollector()

    failures = []
    for dataset_id in datasets:
        request = IngestionRequest(
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            allow_fallback=not args.no_fallback,
        )
        collector = (
            market_collector
            if dataset_id in {"nifty_50", "india_vix"}
            else macro_external_collector
        )
        try:
            result = collector.collect(request)
            print(
                "OK",
                dataset_id,
                f"source={result.source.source_id}",
                f"fallback={result.fallback_used}",
                f"records={result.record_count}",
                f"raw={result.raw_path}",
                f"manifest={result.manifest_path}",
            )
        except Exception as exc:
            failures.append((dataset_id, str(exc)))
            print("FAIL", dataset_id, exc)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
