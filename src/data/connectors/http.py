"""HTTP connector for CSV-like source downloads."""

from __future__ import annotations

from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.data.connectors.base import SourceConnector
from src.data.schemas import IngestionRequest, RawPayload


class HttpCsvConnector(SourceConnector):
    """Retrieves CSV content from an explicit URL or a source-relative path."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        if request.source_reference is None and self.source.base_url is None:
            raise ValueError("HTTP ingestion requires source_reference or source base_url.")

        url = self._build_url(str(request.source_reference or self.source.base_url), request)
        http_request = Request(
            url,
            headers={
                "Accept": "text/csv,application/csv,text/plain,*/*",
                "User-Agent": "project-1a-data-ingestion/0.1",
            },
        )
        with urlopen(http_request, timeout=30) as response:
            content_type = response.headers.get_content_type() or "text/csv"
            content = response.read()

        return RawPayload(
            content=content,
            content_type=content_type,
            source_reference=url,
        )

    def _build_url(self, source_reference: str, request: IngestionRequest) -> str:
        if source_reference.startswith(("http://", "https://")):
            url = source_reference
        elif self.source.base_url:
            url = f"{self.source.base_url.rstrip('/')}/{source_reference.lstrip('/')}"
        else:
            url = source_reference

        params = dict(request.params or {})
        if request.start_date is not None:
            params.setdefault("start_date", request.start_date.isoformat())
        if request.end_date is not None:
            params.setdefault("end_date", request.end_date.isoformat())
        if not params:
            return url

        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{urlencode(params)}"
