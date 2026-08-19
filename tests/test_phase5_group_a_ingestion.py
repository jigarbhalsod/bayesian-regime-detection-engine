from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.collectors import HistoricalMarketDataCollector, MacroExternalDataCollector
from src.data.registry import DatasetRegistry
from src.data.schemas import (
    DatasetClassification,
    DatasetDomain,
    IngestionRequest,
)
from src.data.storage import RawDataStore


class Phase5GroupAIngestionTests(unittest.TestCase):
    def test_registry_loads_phase_4_group_a_datasets(self) -> None:
        registry = DatasetRegistry.from_json()

        core_ids = {
            dataset.dataset_id
            for dataset in registry.datasets(classification=DatasetClassification.CORE)
        }
        supporting_ids = {
            dataset.dataset_id
            for dataset in registry.datasets(
                classification=DatasetClassification.SUPPORTING
            )
        }

        self.assertEqual(core_ids, {"india_vix", "nifty_50"})
        self.assertTrue(
            {
                "nifty_breadth",
                "repo_rate",
                "cpi_india",
                "usd_inr",
                "sp500",
                "cboe_vix",
                "crude_oil",
            }.issubset(supporting_ids)
        )
        self.assertEqual(registry.preferred_source_for("nifty_50").source_id, "nse")

    def test_historical_market_collector_writes_raw_file_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_csv = tmp_path / "nifty.csv"
            source_csv.write_text(
                "date,open,high,low,close\n2020-01-01,100,110,95,105\n",
                encoding="utf-8",
            )
            store = RawDataStore(
                raw_root=tmp_path / "raw",
                manifest_root=tmp_path / "metadata" / "manifests",
            )
            collector = HistoricalMarketDataCollector(store=store)

            result = collector.collect(
                IngestionRequest(
                    dataset_id="nifty_50",
                    source_id="local_file",
                    source_reference=source_csv,
                )
            )

            self.assertTrue(result.raw_path.exists())
            self.assertEqual(result.raw_path.read_bytes(), source_csv.read_bytes())
            self.assertEqual(
                result.raw_path.parts[-5:],
                (
                    "market",
                    "nifty_50",
                    "daily",
                    result.raw_path.parts[-2],
                    result.raw_path.name,
                ),
            )
            self.assertEqual(result.record_count, 1)

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_id"], "nifty_50")
            self.assertEqual(manifest["source_id"], "local_file")
            self.assertEqual(manifest["source_class"], "fallback")
            self.assertEqual(manifest["source_reference"], str(source_csv))
            self.assertTrue(manifest["fallback_used"])
            self.assertEqual(
                manifest["fallback_reason"],
                None,
            )
            self.assertEqual(manifest["actual_first_observation"], "2020-01-01")
            self.assertEqual(manifest["actual_last_observation"], "2020-01-01")
            self.assertEqual(manifest["availability_status"], "READY")
            self.assertEqual(manifest["schema_version"], "raw.v1")
            self.assertEqual(manifest["file_hash"], result.file_hash)

    def test_market_collector_rejects_macro_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_csv = tmp_path / "repo.csv"
            source_csv.write_text("date,rate\n2020-01-01,5.15\n", encoding="utf-8")
            collector = HistoricalMarketDataCollector(
                store=RawDataStore(
                    raw_root=tmp_path / "raw",
                    manifest_root=tmp_path / "metadata" / "manifests",
                )
            )

            with self.assertRaisesRegex(ValueError, "repo_rate is a macro dataset"):
                collector.collect(
                    IngestionRequest(
                        dataset_id="repo_rate",
                        source_id="local_file",
                        source_reference=source_csv,
                    )
                )

    def test_macro_external_collector_accepts_macro_and_external_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_csv = tmp_path / "usd_inr.csv"
            source_csv.write_text("date,rate\n2020-01-01,71.4\n", encoding="utf-8")
            collector = MacroExternalDataCollector(
                store=RawDataStore(
                    raw_root=tmp_path / "raw",
                    manifest_root=tmp_path / "metadata" / "manifests",
                )
            )

            result = collector.collect(
                IngestionRequest(
                    dataset_id="usd_inr",
                    source_id="local_file",
                    source_reference=source_csv,
                )
            )

            self.assertEqual(result.dataset.domain, DatasetDomain.EXTERNAL)
            self.assertTrue(result.raw_path.exists())
            self.assertTrue(result.manifest_path.exists())

    def test_unknown_dataset_fails_explicitly(self) -> None:
        registry = DatasetRegistry.from_json()

        with self.assertRaisesRegex(KeyError, "Unknown dataset_id"):
            registry.get_dataset("unknown")


if __name__ == "__main__":
    unittest.main()
