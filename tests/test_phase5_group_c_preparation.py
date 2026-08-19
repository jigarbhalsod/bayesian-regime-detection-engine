from __future__ import annotations

import json
from pathlib import Path

from src.data.alignment import TimeSeriesAligner
from src.data.preparation import DatasetPreparationPipeline
from src.data.transformation import TimeSeriesTransformer


def test_transformer_adds_nifty_returns() -> None:
    records = [
        {"date": "2026-01-01", "close": 100.0},
        {"date": "2026-01-02", "close": 110.0},
        {"date": "2026-01-03", "close": 121.0},
    ]

    result = TimeSeriesTransformer().transform(
        dataset_id="nifty_50",
        records=records,
    )

    assert result[0]["returns"] is None
    assert result[1]["returns"] == 0.10
    assert result[2]["returns"] == 0.10


def test_aligner_sorts_normalizes_and_deduplicates() -> None:
    records = [
        {"date": "03-01-2026", "close": 103.0},
        {"date": "2026/01/01", "close": 100.0},
        {"date": "2026-01-02", "close": 101.0},
        {"date": "2026-01-02", "close": 102.0},
        {"date": "", "close": 999.0},
    ]

    result = TimeSeriesAligner().align(
        dataset_id="nifty_50",
        records=records,
    )

    assert [record["date"] for record in result] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]
    assert result[1]["close"] == 102.0


def test_preparation_pipeline_transforms_aligns_validates_and_stores(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nifty_sample.json"

    source.write_text(
        json.dumps(
            [
                {
                    "Date": "2026-01-03",
                    "Open": "110",
                    "High": "120",
                    "Low": "105",
                    "Close": "115",
                },
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

    pipeline = DatasetPreparationPipeline()
    result = pipeline.prepare(
        dataset_id="nifty_50",
        source_path=source,
    )

    assert len(result.records) == 3
    assert result.records[0]["date"] == "2026-01-01"
    assert result.records[1]["date"] == "2026-01-02"
    assert result.records[2]["date"] == "2026-01-03"

    assert result.records[0]["returns"] is None
    assert result.records[1]["returns"] is not None
    assert result.records[2]["returns"] is not None

    assert result.processed_result.record_count == 3
    assert result.processed_result.processed_path.exists()
    assert result.processed_result.metadata_path.exists()