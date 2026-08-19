"""Stooq connector for research/MVP market fallback data."""

from __future__ import annotations

from dataclasses import replace

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


STOOQ_SYMBOL_BY_DATASET = {
    "sp500": "^spx",
    "crude_oil": "cl.f",
}


class StooqCsvConnector(HttpCsvConnector):
    """Fetches daily CSV data from Stooq for approved external series."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        symbol = (request.params or {}).get(
            "symbol",
            STOOQ_SYMBOL_BY_DATASET.get(request.dataset_id),
        )
        if not symbol:
            raise ValueError(f"No Stooq symbol configured for {request.dataset_id}")

        params = {"s": symbol, "i": "d"}
        if request.start_date is not None:
            params["d1"] = request.start_date.strftime("%Y%m%d")
        if request.end_date is not None:
            params["d2"] = request.end_date.strftime("%Y%m%d")

        stooq_request = replace(
            request,
            source_reference=request.source_reference or "/q/d/l/",
            params={**params, **(request.params or {})},
        )
        payload = super().fetch(stooq_request)
        return RawPayload(
            content=payload.content,
            content_type=payload.content_type,
            source_reference=payload.source_reference,
            source_timestamp=payload.source_timestamp,
            metadata={"stooq_symbol": symbol},
        )
