"""NSE connector for official Indian market data."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from urllib.request import HTTPCookieProcessor, Request, build_opener

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


NSE_INDEX_BY_DATASET = {
    "nifty_50": "NIFTY 50",
    "india_vix": "INDIA VIX",
}


class NseOfficialConnector(HttpCsvConnector):
    """Fetches official NSE index history through NSE's session-aware API path."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        index_type = (request.params or {}).get(
            "indexType",
            NSE_INDEX_BY_DATASET.get(request.dataset_id),
        )
        if not index_type:
            raise ValueError(f"No NSE indexType configured for {request.dataset_id}")

        params = {
            "indexType": index_type,
            "from": self._nse_date(request.start_date),
            "to": self._nse_date(request.end_date or date.today()),
        }
        params.update(request.params or {})
        nse_request = replace(
            request,
            source_reference=request.source_reference or "/api/historical/indicesHistory",
            params=params,
        )
        url = self._build_url(str(nse_request.source_reference), nse_request)

        opener = build_opener(HTTPCookieProcessor())
        headers = {
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
            "User-Agent": "Mozilla/5.0 project-1a-data-ingestion/0.1",
        }
        opener.open(Request(self.source.base_url or "https://www.nseindia.com", headers=headers), timeout=30)
        with opener.open(Request(url, headers=headers), timeout=30) as response:
            content = response.read()
            content_type = response.headers.get_content_type() or "application/json"

        return RawPayload(
            content=content,
            content_type=content_type,
            source_reference=url,
            metadata={"nse_index_type": index_type},
        )

    @staticmethod
    def _nse_date(value: date | None) -> str:
        if value is None:
            return "01-01-2010"
        return value.strftime("%d-%m-%Y")
