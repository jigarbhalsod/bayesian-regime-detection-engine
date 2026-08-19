"""Processed data storage utilities for Phase 5 Group B."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.config import PROCESSED_DATA_DIR, METADATA_DIR


@dataclass(frozen=True)
class ProcessedDataResult:
    """Details of one processed dataset write."""

    dataset_id: str
    processed_path: Path
    metadata_path: Path
    record_count: int
    file_hash: str


class ProcessedDataStore:
    """Writes validated records and processing metadata."""

    def __init__(
        self,
        processed_data_dir: Path = PROCESSED_DATA_DIR,
        metadata_dir: Path = METADATA_DIR,
    ) -> None:
        self.processed_data_dir = processed_data_dir
        self.metadata_dir = metadata_dir

    def write(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
        source_path: str | Path,
        quality_report: Any,
    ) -> ProcessedDataResult:
        """Write processed records and associated metadata."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dataset_dir = self.processed_data_dir / dataset_id
        metadata_dir = self.metadata_dir / "processed" / dataset_id

        dataset_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        processed_path = dataset_dir / f"{dataset_id}_{timestamp}.csv"
        metadata_path = metadata_dir / f"{dataset_id}_{timestamp}.json"

        fieldnames = self._fieldnames(records)

        with processed_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        content = processed_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        metadata = {
            "dataset_id": dataset_id,
            "source_path": str(source_path),
            "processed_path": str(processed_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "file_hash_sha256": file_hash,
            "quality_report": self._serialize_quality_report(
                quality_report
            ),
        }

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        return ProcessedDataResult(
            dataset_id=dataset_id,
            processed_path=processed_path,
            metadata_path=metadata_path,
            record_count=len(records),
            file_hash=file_hash,
        )

    @staticmethod
    def _fieldnames(
        records: list[dict[str, Any]],
    ) -> list[str]:
        """Build a stable union of record fields."""
        fields: set[str] = set()

        for record in records:
            fields.update(record.keys())

        return sorted(fields)

    @staticmethod
    def _serialize_quality_report(
        quality_report: Any,
    ) -> dict[str, Any]:
        """Convert a dataclass quality report into JSON-safe metadata."""
        try:
            return asdict(quality_report)
        except TypeError:
            return {"value": str(quality_report)}