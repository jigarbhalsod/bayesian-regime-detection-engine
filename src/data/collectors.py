"""Phase 5 Group A data collectors."""

from __future__ import annotations

from src.data.connectors import (
    FredCsvConnector,
    HttpCsvConnector,
    LocalFileConnector,
    MospiOfficialConnector,
    NseOfficialConnector,
    RbiOfficialConnector,
    SourceConnector,
    StooqCsvConnector,
    YahooFinanceCsvConnector,
)
from src.data.registry import DatasetRegistry
from src.data.schemas import (
    DatasetDomain,
    IngestionAvailabilityStatus,
    IngestionRequest,
    RawIngestionResult,
)
from src.data.storage import RawDataStore


class SourceConnectorFactory:
    """Builds connector instances from source registry definitions."""

    CONNECTORS = {
        "fred_csv": FredCsvConnector,
        "http_csv": HttpCsvConnector,
        "local_file": LocalFileConnector,
        "mospi_official": MospiOfficialConnector,
        "nse_official": NseOfficialConnector,
        "rbi_official": RbiOfficialConnector,
        "stooq_csv": StooqCsvConnector,
        "yahoo_finance_csv": YahooFinanceCsvConnector,
    }

    def build(self, source_id: str, registry: DatasetRegistry) -> SourceConnector:
        source = registry.get_source(source_id)
        try:
            connector_cls = self.CONNECTORS[source.connector_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported connector_type: {source.connector_type}") from exc
        return connector_cls(source)


class BaseCollector:
    """Coordinates registry lookup, source fetch and raw storage."""

    allowed_domains: tuple[DatasetDomain, ...] = ()

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        store: RawDataStore | None = None,
        connector_factory: SourceConnectorFactory | None = None,
    ) -> None:
        self.registry = registry or DatasetRegistry.from_json()
        self.store = store or RawDataStore()
        self.connector_factory = connector_factory or SourceConnectorFactory()

    def collect(self, request: IngestionRequest) -> RawIngestionResult:
        dataset = self.registry.get_dataset(request.dataset_id)
        if self.allowed_domains and dataset.domain not in self.allowed_domains:
            allowed = ", ".join(domain.value for domain in self.allowed_domains)
            raise ValueError(
                f"{dataset.dataset_id} is a {dataset.domain.value} dataset; "
                f"collector supports: {allowed}"
            )

        source_ids = [request.source_id or dataset.preferred_source_id]
        if request.allow_fallback and request.source_id is None:
            source_ids.extend(dataset.fallback_source_ids)

        payload = None
        source = None
        fallback_reason = None
        selected_source_id = None
        errors: list[str] = []
        for index, source_id in enumerate(source_ids):
            try:
                source = self.registry.get_source(source_id)
                connector = self.connector_factory.build(source_id, self.registry)
                payload = connector.fetch(request)
                selected_source_id = source_id
                fallback_reason = "; ".join(errors) if index > 0 else None
                break
            except Exception as exc:
                errors.append(f"{source_id}: {exc}")

        if payload is None or source is None or selected_source_id is None:
            detail = "; ".join(errors) or "no sources configured"
            raise RuntimeError(f"Unable to ingest {dataset.dataset_id}: {detail}")

        status = (
            IngestionAvailabilityStatus.READY
            if payload.content
            else IngestionAvailabilityStatus.UNAVAILABLE
        )
        return self.store.write_payload(
            dataset=dataset,
            source=source,
            payload=payload,
            availability_status=status,
            request=request,
            fallback_used=selected_source_id != dataset.preferred_source_id,
            fallback_reason=fallback_reason,
        )


class HistoricalMarketDataCollector(BaseCollector):
    """Collects market datasets such as NIFTY 50, India VIX and breadth inputs."""

    allowed_domains = (DatasetDomain.MARKET,)


class MacroExternalDataCollector(BaseCollector):
    """Collects macro, FX, global equity, volatility and commodity datasets."""

    allowed_domains = (DatasetDomain.MACRO, DatasetDomain.EXTERNAL)
