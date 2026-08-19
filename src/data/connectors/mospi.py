"""MOSPI connector for official Indian statistics source downloads."""

from __future__ import annotations

from dataclasses import replace

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


MOSPI_REFERENCE_BY_DATASET = {
    "cpi_india": "/data",
}


class MospiOfficialConnector(HttpCsvConnector):
    """Fetches raw MOSPI pages/downloads for Indian statistics datasets."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        source_reference = request.source_reference or MOSPI_REFERENCE_BY_DATASET.get(
            request.dataset_id
        )
        if not source_reference:
            raise ValueError(f"No MOSPI source reference configured for {request.dataset_id}")

        payload = super().fetch(replace(request, source_reference=source_reference))
        return RawPayload(
            content=payload.content,
            content_type=payload.content_type,
            source_reference=payload.source_reference,
            source_timestamp=payload.source_timestamp,
            metadata={"official_download": True},
        )
