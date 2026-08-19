"""Base contracts and utilities for feature engineering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureResult:
    """Result returned by one feature transformer."""

    records: list[dict[str, Any]]
    created_features: tuple[str, ...]


class BaseFeatureTransformer(ABC):
    """Abstract base class for all feature transformers."""

    name: str

    @abstractmethod
    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create features from chronological records."""