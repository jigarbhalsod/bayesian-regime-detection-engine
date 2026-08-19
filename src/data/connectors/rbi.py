"""RBI connector for official Indian monetary and FX source downloads."""

from __future__ import annotations

from dataclasses import replace

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


RBI_REFERENCE_BY_DATASET = {
    "repo_rate": "/Scripts/PublicationsView.aspx?Id=22517",
    "usd_inr": "/home.aspx",
}


class RbiOfficialConnector(HttpCsvConnector):
    """Fetches raw RBI pages/downloads for approved Indian macro datasets."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        source_reference = request.source_reference or RBI_REFERENCE_BY_DATASET.get(
            request.dataset_id
        )
        if not source_reference:
            raise ValueError(f"No RBI source reference configured for {request.dataset_id}")

        payload = super().fetch(replace(request, source_reference=source_reference))
        return RawPayload(
            content=payload.content,
            content_type=payload.content_type,
            source_reference=payload.source_reference,
            source_timestamp=payload.source_timestamp,
            metadata={"official_download": True},
        )
