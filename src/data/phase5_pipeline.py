"""Final Phase 5 orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.integration import DatasetIntegrator
from src.data.preparation import DatasetPreparationPipeline
from src.data.readiness import (
    DatasetReadinessChecker,
    DatasetReadinessReport,
)


@dataclass(frozen=True)
class Phase5PipelineResult:
    """Result returned after preparing and integrating datasets."""

    prepared_datasets: dict[str, list[dict[str, Any]]]
    readiness_reports: dict[str, Any]
    integrated_records: list[dict[str, Any]]


class Phase5DataPipeline:
    """Runs the complete Phase 5 data preparation workflow."""

    def __init__(
        self,
        preparation_pipeline: DatasetPreparationPipeline | None = None,
        readiness_checker: DatasetReadinessChecker | None = None,
        integrator: DatasetIntegrator | None = None,
    ) -> None:
        self.preparation_pipeline = (
            preparation_pipeline or DatasetPreparationPipeline()
        )
        self.readiness_checker = (
            readiness_checker or DatasetReadinessChecker()
        )
        self.integrator = integrator or DatasetIntegrator()

    def run(
        self,
        *,
        dataset_sources: dict[str, str | Path],
    ) -> Phase5PipelineResult:
        """Prepare, validate readiness, and integrate all datasets."""

        prepared_datasets: dict[str, list[dict[str, Any]]] = {}
        readiness_reports: dict[str, Any] = {}

        for dataset_id, source_path in dataset_sources.items():
            preparation_result = self.preparation_pipeline.prepare(
                dataset_id=dataset_id,
                source_path=source_path,
            )

            records = preparation_result.records

            readiness_report = self._check_final_readiness(
                dataset_id=dataset_id,
                records=records,
            )

            prepared_datasets[dataset_id] = records
            readiness_reports[dataset_id] = readiness_report

        integrated_records = self.integrator.integrate(
            datasets=prepared_datasets,
        )

        return Phase5PipelineResult(
            prepared_datasets=prepared_datasets,
            readiness_reports=readiness_reports,
            integrated_records=integrated_records,
        )

    def _check_final_readiness(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> DatasetReadinessReport:
        """Check final model readiness with expected boundary values allowed."""

        report = self.readiness_checker.check(
            dataset_id=dataset_id,
            records=records,
        )

        if dataset_id != "nifty_50":
            return report

        invalid_record_count = sum(
            1
            for index, record in enumerate(records)
            if not self._is_valid_final_nifty_record(
                record=record,
                index=index,
            )
        )

        return DatasetReadinessReport(
            dataset_id=report.dataset_id,
            is_ready=(
                len(records) > 0
                and not report.missing_fields
                and invalid_record_count == 0
            ),
            record_count=report.record_count,
            required_fields=report.required_fields,
            missing_fields=report.missing_fields,
            invalid_record_count=invalid_record_count,
        )

    @staticmethod
    def _is_valid_final_nifty_record(
        *,
        record: dict[str, Any],
        index: int,
    ) -> bool:
        """Allow the first return value to be undefined by design."""

        if (
            index == 0
            and record.get("returns") is None
        ):
            return (
                record.get("date") not in (None, "")
                and record.get("close") not in (None, "")
            )

        return (
            record.get("date") not in (None, "")
            and record.get("close") not in (None, "")
            and record.get("returns") not in (None, "")
        )