"""Base connector contract for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas import IngestionRequest, RawPayload, SourceDefinition


class SourceConnector(ABC):
    """Fetches a raw payload from one source without applying business transforms."""

    def __init__(self, source: SourceDefinition) -> None:
        self.source = source

    @abstractmethod
    def fetch(self, request: IngestionRequest) -> RawPayload:
        """Return source-faithful raw content for the request."""
