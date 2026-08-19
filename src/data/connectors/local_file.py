"""Connector for manually downloaded source files."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from src.data.connectors.base import SourceConnector
from src.data.schemas import IngestionRequest, RawPayload


class LocalFileConnector(SourceConnector):
    """Loads a raw file from disk without altering its bytes."""

    def fetch(self, request: IngestionRequest) -> RawPayload:
        if request.source_reference is None:
            raise ValueError("Local file ingestion requires source_reference.")

        path = Path(request.source_reference)
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")

        if path.suffix.lower() == ".csv":
            content_type = "text/csv"
        else:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return RawPayload(
            content=path.read_bytes(),
            content_type=content_type,
            source_reference=str(path),
        )
