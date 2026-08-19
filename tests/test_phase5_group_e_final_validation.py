import json
from pathlib import Path

from src.data.phase5_pipeline import Phase5DataPipeline


def test_phase5_final_pipeline_output_is_model_ready(
    tmp_path: Path,
) -> None:
    """The final Phase 5 pipeline should produce model-ready records."""

    source = tmp_path / "nifty_final.json"

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

    pipeline = Phase5DataPipeline()

    result = pipeline.run(
        dataset_sources={
            "nifty_50": source,
        }
    )

    assert "nifty_50" in result.prepared_datasets
    assert "nifty_50" in result.readiness_reports

    records = result.prepared_datasets["nifty_50"]

    assert len(records) == 3
    assert records[0]["date"] == "2026-01-01"
    assert records[1]["date"] == "2026-01-02"
    assert records[2]["date"] == "2026-01-03"

    readiness_report = result.readiness_reports["nifty_50"]
    assert readiness_report.is_ready is True

    assert len(result.integrated_records) == 3

    for record in result.integrated_records:
        assert "date" in record
        assert record["date"] is not None