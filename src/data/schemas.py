"""Schemas and contracts for Phase 5 data ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetClassification(StrEnum):
    CORE = "core"
    SUPPORTING = "supporting"
    OPTIONAL = "optional"


class DatasetFrequency(StrEnum):
    DAILY = "daily"
    MONTHLY = "monthly"
    EVENT_BASED = "event_based"


class DatasetDomain(StrEnum):
    MARKET = "market"
    MACRO = "macro"
    EXTERNAL = "external"


class IngestionAvailabilityStatus(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class SourceClass(StrEnum):
    OFFICIAL = "official"
    STRUCTURED = "structured"
    FALLBACK = "fallback"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    name: str
    domain: DatasetDomain
    classification: DatasetClassification
    mvp_decision: str
    frequency: DatasetFrequency
    key_variables: tuple[str, ...]
    purpose: str
    preferred_source_id: str
    fallback_source_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetDefinition":
        return cls(
            dataset_id=payload["dataset_id"],
            name=payload["name"],
            domain=DatasetDomain(payload["domain"]),
            classification=DatasetClassification(payload["classification"]),
            mvp_decision=payload["mvp_decision"],
            frequency=DatasetFrequency(payload["frequency"]),
            key_variables=tuple(payload["key_variables"]),
            purpose=payload["purpose"],
            preferred_source_id=payload["preferred_source_id"],
            fallback_source_ids=tuple(payload.get("fallback_source_ids", ())),
        )


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    name: str
    source_class: SourceClass
    access_method: str
    connector_type: str
    base_url: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SourceDefinition":
        return cls(
            source_id=payload["source_id"],
            name=payload["name"],
            source_class=SourceClass(payload["source_class"]),
            access_method=payload["access_method"],
            connector_type=payload["connector_type"],
            base_url=payload.get("base_url"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class IngestionRequest:
    dataset_id: str
    source_id: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    source_reference: str | Path | None = None
    params: dict[str, str] | None = None
    allow_fallback: bool = True


@dataclass(frozen=True)
class RawPayload:
    content: bytes
    source_reference: str
    content_type: str = "text/csv"
    source_timestamp: datetime | None = None
    actual_first_observation: date | None = None
    actual_last_observation: date | None = None
    metadata: dict[str, str | bool | None] | None = None


@dataclass(frozen=True)
class RawIngestionResult:
    dataset: DatasetDefinition
    source: SourceDefinition
    raw_path: Path
    manifest_path: Path
    ingestion_timestamp: datetime
    file_hash: str
    record_count: int | None
    availability_status: IngestionAvailabilityStatus
    fallback_used: bool
    fallback_reason: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
