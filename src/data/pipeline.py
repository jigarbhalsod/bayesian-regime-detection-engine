"""End-to-end Phase 5 Group B data processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.cleaning import RecordCleaner
from src.data.loading import RawDataLoader
from src.data.parsing import DatasetParser
from src.data.processed_storage import ProcessedDataResult, ProcessedDataStore
from src.data.standardization import RecordStandardizer
from src.data.validation import DataQualityValidator


@dataclass(frozen=True)
class ProcessingPipelineResult:
    """Result returned after processing one raw dataset."""

    dataset_id: str
    source_path: Path
    input_record_count: int
    output_record_count: int
    quality_report: Any
    storage_result: ProcessedDataResult


class DataProcessingPipeline:
    """Runs raw data through the complete Phase 5 Group B pipeline."""

    def __init__(
        self,
        loader: RawDataLoader | None = None,
        parser: DatasetParser | None = None,
        standardizer: RecordStandardizer | None = None,
        cleaner: RecordCleaner | None = None,
        validator: DataQualityValidator | None = None,
        store: ProcessedDataStore | None = None,
    ) -> None:
        self.loader = loader or RawDataLoader()
        self.parser = parser or DatasetParser()
        self.standardizer = standardizer or RecordStandardizer()
        self.cleaner = cleaner or RecordCleaner()
        self.validator = validator or DataQualityValidator()
        self.store = store or ProcessedDataStore()

    def run(
        self,
        *,
        dataset_id: str,
        source_path: str | Path,
    ) -> ProcessingPipelineResult:
        """Run loading through processed-data storage."""

        source_path = Path(source_path)

        raw_content = self.loader.load(source_path)
        parsed_records = self.parser.parse(
            dataset_id=dataset_id,
            raw_content=raw_content,
            source_path=source_path,
        )

        standardized_records = self.standardizer.standardize(
            dataset_id=dataset_id,
            records=parsed_records,
        )

        cleaned_records = self.cleaner.clean(
            dataset_id=dataset_id,
            records=standardized_records,
        )

        quality_report = self.validator.validate(
            dataset_id=dataset_id,
            records=cleaned_records,
        )

        storage_result = self.store.write(
            dataset_id=dataset_id,
            records=cleaned_records,
            source_path=source_path,
            quality_report=quality_report,
        )

        return ProcessingPipelineResult(
            dataset_id=dataset_id,
            source_path=source_path,
            input_record_count=len(parsed_records),
            output_record_count=len(cleaned_records),
            quality_report=quality_report,
            storage_result=storage_result,
        )