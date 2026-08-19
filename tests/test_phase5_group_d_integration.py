from __future__ import annotations

import json
from pathlib import Path

from src.data.integration import DatasetIntegrator
from src.data.phase5_pipeline import Phase5DataPipeline
from src.data.readiness import DatasetReadinessChecker


def test_readiness_checker_accepts_ready_nifty_dataset() -> None:
    records = [
        {
            "date": "2026-01-01",
            "close": 100.0,
            "returns": None,
        },
        {
            "date": "2026-01-02",
            "close": 105.0,
            "returns": 0.05,
        },
    ]

    report = DatasetReadinessChecker().check(
        dataset_id="nifty_50",
        records=records,
    )

    assert report.is_ready is False
    assert report.record_count == 2
    assert report.missing_fields == ()
    assert report.invalid_record_count == 1


def test_readiness_checker_rejects_missing_required_fields() -> None:
    records = [
        {
            "date": "2026-01-01",
            "close": 100.0,
        }
    ]

    report = DatasetReadinessChecker().check(
        dataset_id="nifty_50",
        records=records,
    )

    assert report.is_ready is False
    assert report.missing_fields == ("returns",)


def test_dataset_integrator_combines_datasets_by_date() -> None:
    datasets = {
        "nifty_50": [
            {
                "date": "2026-01-01",
                "close": 100.0,
                "returns": None,
            },
            {
                "date": "2026-01-02",
                "close": 105.0,
                "returns": 0.05,
            },
        ],
        "india_vix": [
            {
                "date": "2026-01-01",
                "level": 15.0,
            },
            {
                "date": "2026-01-02",
                "level": 16.0,
            },
        ],
    }

    integrated = DatasetIntegrator().integrate(
        datasets=datasets,
    )

    assert len(integrated) == 2
    assert integrated[0]["date"] == "2026-01-01"
    assert integrated[0]["nifty_50__close"] == 100.0
    assert integrated[0]["india_vix__level"] == 15.0
    assert integrated[1]["nifty_50__returns"] == 0.05


def test_phase5_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    nifty_source = tmp_path / "nifty.json"
    vix_source = tmp_path / "vix.json"

    nifty_source.write_text(
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

    vix_source.write_text(
        json.dumps(
            [
                {
                    "Date": "2026-01-01",
                    "Close": "15",
                },
                {
                    "Date": "2026-01-02",
                    "Close": "16",
                },
            ]
        ),
        encoding="utf-8",
    )

    pipeline = Phase5DataPipeline()

    result = pipeline.run(
        dataset_sources={
            "nifty_50": nifty_source,
            "india_vix": vix_source,
        },
    )

    assert "nifty_50" in result.prepared_datasets
    assert "india_vix" in result.prepared_datasets

    assert len(result.integrated_records) == 2

    assert result.integrated_records[0]["date"] == "2026-01-01"
    assert result.integrated_records[0]["nifty_50__close"] == 105.0
    assert result.integrated_records[0]["india_vix__level"] == 15.0

    assert (
        result.integrated_records[1]["nifty_50__returns"]
        == (110.0 - 105.0) / 105.0
    )