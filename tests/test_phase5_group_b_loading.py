"""Tests for Phase 5 Group B data processing."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.cleaning import RecordCleaner
from src.data.loading import RawDataLoader
from src.data.parsing import DatasetParser
from src.data.pipeline import DataProcessingPipeline
from src.data.standardization import RecordStandardizer
from src.data.validation import DataQualityValidator


def test_raw_data_loader_reads_json(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    source.write_text(
        json.dumps([{"Date": "2026-01-01", "Close": 100.0}]),
        encoding="utf-8",
    )

    content = RawDataLoader().load(source)

    assert content


def test_parser_parses_json_records(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    source.write_text(
        json.dumps(
            [
                {"Date": "2026-01-01", "Close": 100.0},
                {"Date": "2026-01-02", "Close": 101.0},
            ]
        ),
        encoding="utf-8",
    )

    raw_content = RawDataLoader().load(source)
    records = DatasetParser().parse(
        dataset_id="nifty_50",
        raw_content=raw_content,
        source_path=source,
    )

    assert len(records) == 2


def test_standardizer_normalizes_column_names() -> None:
    records = [
        {
            "Date": "2026-01-01",
            "Open": "100",
            "High": "110",
            "Low": "95",
            "Close": "105",
        }
    ]

    standardized = RecordStandardizer().standardize(
        dataset_id="nifty_50",
        records=records,
    )

    assert "date" in standardized[0]
    assert "close" in standardized[0]


def test_cleaner_handles_missing_values() -> None:
    records = [
        {"date": "2026-01-01", "close": "100"},
        {"date": "", "close": "101"},
        {"date": "2026-01-03", "close": ""},
    ]

    cleaned = RecordCleaner().clean(
        dataset_id="nifty_50",
        records=records,
    )

    assert len(cleaned) <= len(records)
    assert all(record.get("date") for record in cleaned)


def test_validator_returns_quality_report() -> None:
    records = [
        {"date": "2026-01-01", "close": 100.0},
        {"date": "2026-01-02", "close": 101.0},
    ]

    report = DataQualityValidator().validate(
        dataset_id="nifty_50",
        records=records,
    )

    assert report is not None


def test_end_to_end_processing_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "nifty_sample.json"
    source.write_text(
        json.dumps(
            [
                {
                    "Date": "2026-01-01",
                    "Open": "100",
                    "High": "110",
                    "Low": "95",
                    "Close": "105",
                },
                {
                    "Date": "2026-01-02",
                    "Open": "105",
                    "High": "115",
                    "Low": "100",
                    "Close": "110",
                },
            ]
        ),
        encoding="utf-8",
    )

    pipeline = DataProcessingPipeline()
    result = pipeline.run(
        dataset_id="nifty_50",
        source_path=source,
    )

    assert result.dataset_id == "nifty_50"
    assert result.input_record_count == 2
    assert result.output_record_count > 0
    assert result.storage_result.processed_path.exists()
    assert result.storage_result.metadata_path.exists()