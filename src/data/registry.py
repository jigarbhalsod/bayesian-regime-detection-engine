"""Dataset and source registries for Phase 5 ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.config import DEFAULT_DATA_SOURCE_CONFIG
from src.data.schemas import (
    DatasetClassification,
    DatasetDefinition,
    DatasetDomain,
    SourceDefinition,
)


class DatasetRegistry:
    """Loads approved Phase 4 dataset and source mappings."""

    def __init__(
        self,
        datasets: dict[str, DatasetDefinition],
        sources: dict[str, SourceDefinition],
    ) -> None:
        self._datasets = datasets
        self._sources = sources

    @classmethod
    def from_json(cls, path: Path = DEFAULT_DATA_SOURCE_CONFIG) -> "DatasetRegistry":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        datasets = {
            item["dataset_id"]: DatasetDefinition.from_mapping(item)
            for item in payload["datasets"]
        }
        sources = {
            item["source_id"]: SourceDefinition.from_mapping(item)
            for item in payload["sources"]
        }
        return cls(datasets=datasets, sources=sources)

    def get_dataset(self, dataset_id: str) -> DatasetDefinition:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset_id: {dataset_id}") from exc

    def get_source(self, source_id: str) -> SourceDefinition:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"Unknown source_id: {source_id}") from exc

    def preferred_source_for(self, dataset_id: str) -> SourceDefinition:
        dataset = self.get_dataset(dataset_id)
        return self.get_source(dataset.preferred_source_id)

    def datasets(
        self,
        *,
        domain: DatasetDomain | None = None,
        classification: DatasetClassification | None = None,
    ) -> list[DatasetDefinition]:
        datasets = list(self._datasets.values())
        if domain is not None:
            datasets = [item for item in datasets if item.domain == domain]
        if classification is not None:
            datasets = [item for item in datasets if item.classification == classification]
        return sorted(datasets, key=lambda item: item.dataset_id)

    def sources(self) -> list[SourceDefinition]:
        return sorted(self._sources.values(), key=lambda item: item.source_id)
