"""Feature engineering pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.base import FeatureResult
from src.features.registry import FeatureRegistry


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Result of running the feature engineering pipeline."""

    records: list[dict[str, Any]]
    created_features: tuple[str, ...]
    executed_transformers: tuple[str, ...]


class FeatureEngineeringPipeline:
    """Runs registered feature transformers in deterministic order."""

    def __init__(
        self,
        registry: FeatureRegistry,
    ) -> None:
        self.registry = registry

    def run(
        self,
        records: list[dict[str, Any]],
    ) -> FeaturePipelineResult:
        """Run all registered transformers on chronological records."""

        working_records = [
            dict(record)
            for record in records
        ]

        created_features: list[str] = []
        executed_transformers: list[str] = []

        for transformer in self.registry.values():
            result: FeatureResult = transformer.transform(
                working_records
            )

            working_records = result.records

            created_features.extend(
                result.created_features
            )

            executed_transformers.append(
                transformer.name
            )

        return FeaturePipelineResult(
            records=working_records,
            created_features=tuple(created_features),
            executed_transformers=tuple(executed_transformers),
        )