"""Dataset preparation pipeline for Phase 5 Group C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.alignment import TimeSeriesAligner
from src.data.pipeline import DataProcessingPipeline
from src.data.processed_storage import (
    ProcessedDataResult,
    ProcessedDataStore,
)
from src.data.transformation import TimeSeriesTransformer
from src.data.validation import DataQualityValidator


@dataclass(frozen=True)
class DatasetPreparationResult:
    """Result produced by the complete Phase 5 Group C pipeline."""

    dataset_id: str
    records: list[dict[str, Any]]
    processed_result: ProcessedDataResult
    quality_report: Any


class DatasetPreparationPipeline:
    """Runs processing, transformation, alignment, validation and storage."""

    def __init__(
        self,
        processing_pipeline: DataProcessingPipeline | None = None,
        transformer: TimeSeriesTransformer | None = None,
        aligner: TimeSeriesAligner | None = None,
        validator: DataQualityValidator | None = None,
        processed_store: ProcessedDataStore | None = None,
    ) -> None:
        self.processing_pipeline = (
            processing_pipeline or DataProcessingPipeline()
        )
        self.transformer = transformer or TimeSeriesTransformer()
        self.aligner = aligner or TimeSeriesAligner()
        self.validator = validator or DataQualityValidator()
        self.processed_store = processed_store or ProcessedDataStore()

    def prepare(
        self,
        *,
        dataset_id: str,
        source_path: str | Path,
    ) -> DatasetPreparationResult:
        """Prepare and persist one dataset."""

        source_path = Path(source_path)

        processing_result = self.processing_pipeline.run(
            dataset_id=dataset_id,
            source_path=source_path,
        )

        transformed_records = self.transformer.transform(
            dataset_id=dataset_id,
            records=processing_result.records,
        )

        aligned_records = self.aligner.align(
            dataset_id=dataset_id,
            records=transformed_records,
        )

        quality_report = self.validator.validate(
            dataset_id=dataset_id,
            records=aligned_records,
        )

        processed_result = self.processed_store.write(
            dataset_id=dataset_id,
            records=aligned_records,
            source_path=source_path,
            quality_report=quality_report,
            processing_steps=[
                "loading",
                "parsing",
                "standardization",
                "cleaning",
                "transformation",
                "alignment",
                "validation",
                "processed_storage",
            ],
        )

        return DatasetPreparationResult(
            dataset_id=dataset_id,
            records=aligned_records,
            processed_result=processed_result,
            quality_report=quality_report,
        )