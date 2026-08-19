"""FRED connector for public CSV graph downloads."""

from __future__ import annotations

from dataclasses import replace

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


FRED_SERIES_BY_DATASET = {
    "cpi_india": "INDCPIALLMINMEI",
    "cboe_vix": "VIXCLS",
}


class FredCsvConnector(HttpCsvConnector):
    """Fetches FRED CSV data without requiring an API key."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        series_id = (request.params or {}).get(
            "series_id",
            FRED_SERIES_BY_DATASET.get(request.dataset_id),
        )
        if not series_id:
            raise ValueError(f"No FRED series configured for {request.dataset_id}")

        params = dict(request.params or {})
        params["id"] = series_id
        source_reference = request.source_reference or "/graph/fredgraph.csv"
        fred_request = replace(request, source_reference=source_reference, params=params)
        payload = super().fetch(fred_request)
        return RawPayload(
            content=payload.content,
            content_type=payload.content_type,
            source_reference=payload.source_reference,
            source_timestamp=payload.source_timestamp,
            metadata={"fred_series_id": series_id},
        )
