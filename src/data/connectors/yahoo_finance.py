"""Yahoo Finance research fallback connector."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timezone

from src.data.connectors.http import HttpCsvConnector
from src.data.schemas import IngestionRequest, RawPayload


YAHOO_SYMBOL_BY_DATASET = {
    "nifty_50": "^NSEI",
    "india_vix": "^INDIAVIX",
    "sp500": "^GSPC",
    "cboe_vix": "^VIX",
    "crude_oil": "CL=F",
}


class YahooFinanceCsvConnector(HttpCsvConnector):
    """Downloads Yahoo Finance CSV data for research/MVP fallback use only."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        symbol = (request.params or {}).get(
            "symbol",
            YAHOO_SYMBOL_BY_DATASET.get(request.dataset_id),
        )
        if not symbol:
            raise ValueError(f"No Yahoo Finance symbol configured for {request.dataset_id}")

        start = self._to_epoch(request.start_date or date(2010, 1, 1))
        end = self._to_epoch(request.end_date or date.today())
        params = {
            "period1": str(start),
            "period2": str(end),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
        yahoo_request = replace(
            request,
            source_reference=f"/v7/finance/download/{symbol}",
            params=params,
        )
        payload = super().fetch(yahoo_request)
        return RawPayload(
            content=payload.content,
            content_type=payload.content_type,
            source_reference=payload.source_reference,
            source_timestamp=payload.source_timestamp,
            metadata={"yahoo_symbol": symbol, "research_only": True},
        )

    @staticmethod
    def _to_epoch(value: date) -> int:
        return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())
