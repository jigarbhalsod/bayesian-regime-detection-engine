"""Raw data storage and manifest creation."""

from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime
from io import StringIO
from pathlib import Path

from src.data.config import MANIFEST_DIR, RAW_DATA_DIR
from src.data.schemas import (
    DatasetDefinition,
    IngestionAvailabilityStatus,
    RawIngestionResult,
    RawPayload,
    SourceDefinition,
    IngestionRequest,
    utc_now,
)


class RawDataStore:
    """Persists source-faithful payloads without mutating their content."""

    def __init__(
        self,
        raw_root: Path = RAW_DATA_DIR,
        manifest_root: Path = MANIFEST_DIR,
    ) -> None:
        self.raw_root = raw_root
        self.manifest_root = manifest_root

    def write_payload(
        self,
        *,
        dataset: DatasetDefinition,
        source: SourceDefinition,
        payload: RawPayload,
        availability_status: IngestionAvailabilityStatus = IngestionAvailabilityStatus.PARTIAL,
        ingestion_timestamp: datetime | None = None,
        request: IngestionRequest | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> RawIngestionResult:
        timestamp = ingestion_timestamp or utc_now()
        file_hash = hashlib.sha256(payload.content).hexdigest()
        extension = self._extension_for(payload.content_type)
        raw_path = self._raw_path(dataset, timestamp, file_hash, extension)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload.content)
        actual_start, actual_end = self._actual_dates(payload)

        manifest = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "domain": dataset.domain.value,
            "classification": dataset.classification.value,
            "frequency": dataset.frequency.value,
            "source_id": source.source_id,
            "source": source.name,
            "source_class": source.source_class.value,
            "source_reference": payload.source_reference,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "requested_start_date": request.start_date.isoformat()
            if request and request.start_date
            else None,
            "requested_end_date": request.end_date.isoformat()
            if request and request.end_date
            else "latest_available",
            "actual_first_observation": actual_start.isoformat() if actual_start else None,
            "actual_last_observation": actual_end.isoformat() if actual_end else None,
            "ingestion_timestamp": timestamp.isoformat(),
            "source_timestamp": payload.source_timestamp.isoformat()
            if payload.source_timestamp
            else None,
            "availability_status": availability_status.value,
            "record_count": self._count_records(payload.content, payload.content_type),
            "schema_version": "raw.v1",
            "dataset_version": "v0.1.0",
            "file_hash": file_hash,
            "raw_path": str(raw_path),
            "provider_metadata": payload.metadata or {},
        }

        manifest_path = self._manifest_path(dataset, timestamp, file_hash)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return RawIngestionResult(
            dataset=dataset,
            source=source,
            raw_path=raw_path,
            manifest_path=manifest_path,
            ingestion_timestamp=timestamp,
            file_hash=file_hash,
            record_count=manifest["record_count"],
            availability_status=availability_status,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _raw_path(
        self,
        dataset: DatasetDefinition,
        timestamp: datetime,
        file_hash: str,
        extension: str,
    ) -> Path:
        year = timestamp.strftime("%Y")
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{dataset.dataset_id}_{stamp}_{file_hash[:12]}{extension}"
        return (
            self.raw_root
            / dataset.domain.value
            / dataset.dataset_id
            / dataset.frequency.value
            / f"year={year}"
            / filename
        )

    def _manifest_path(
        self,
        dataset: DatasetDefinition,
        timestamp: datetime,
        file_hash: str,
    ) -> Path:
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{dataset.dataset_id}_{stamp}_{file_hash[:12]}.json"
        return self.manifest_root / dataset.dataset_id / filename

    @staticmethod
    def _extension_for(content_type: str) -> str:
        if "json" in content_type:
            return ".json"
        if "parquet" in content_type:
            return ".parquet"
        return ".csv"

    @staticmethod
    def _count_records(content: bytes, content_type: str) -> int | None:
        if "csv" not in content_type:
            return None
        text = content.decode("utf-8-sig", errors="replace").strip()
        if not text:
            return 0
        return max(len(text.splitlines()) - 1, 0)

    @staticmethod
    def _actual_dates(payload: RawPayload) -> tuple[object | None, object | None]:
        if payload.actual_first_observation or payload.actual_last_observation:
            return payload.actual_first_observation, payload.actual_last_observation
        if "csv" not in payload.content_type:
            return None, None

        text = payload.content.decode("utf-8-sig", errors="replace").strip()
        if not text:
            return None, None

        reader = csv.DictReader(StringIO(text))
        date_columns = ("date", "Date", "DATE", "observation_date")
        values = []
        for row in reader:
            raw_date = next((row.get(column) for column in date_columns if row.get(column)), None)
            if not raw_date:
                continue
            try:
                values.append(datetime.fromisoformat(raw_date.strip()).date())
            except ValueError:
                continue
        if not values:
            return None, None
        return min(values), max(values)
