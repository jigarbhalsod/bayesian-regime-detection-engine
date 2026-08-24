from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


def _validate_feature_name(feature_name: str) -> str:
    """Validate and normalize a feature name."""
    if not isinstance(feature_name, str):
        raise TypeError("feature_name must be a string.")

    feature_name = feature_name.strip()

    if not feature_name:
        raise ValueError("feature_name must not be empty.")

    return feature_name


def _validate_finite_number(value: Any, field_name: str) -> float:
    """Validate that a value is a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a numeric value.")

    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    return value


def _validate_rank(rank: int | None) -> int | None:
    """Validate an optional positive integer rank."""
    if rank is None:
        return None

    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be a positive integer or None.")

    if rank <= 0:
        raise ValueError("rank must be greater than 0.")

    return rank


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")

    return dict(metadata)


def _validate_warnings(warnings: list[str]) -> list[str]:
    """Validate warning messages."""
    if not isinstance(warnings, list):
        raise TypeError("warnings must be a list.")

    for warning in warnings:
        if not isinstance(warning, str):
            raise TypeError("each warning must be a string.")

    return list(warnings)


@dataclass
class FeatureAttribution:
    """
    Represents the contribution of a single feature
    to a specific prediction.
    """

    feature_name: str
    attribution: float
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_name = _validate_feature_name(self.feature_name)
        self.attribution = _validate_finite_number(
            self.attribution,
            "attribution",
        )
        self.rank = _validate_rank(self.rank)
        self.metadata = _validate_metadata(self.metadata)


@dataclass
class FeatureImportance:
    """
    Represents the global importance of a single feature.
    """

    feature_name: str
    importance: float
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_name = _validate_feature_name(self.feature_name)
        self.importance = _validate_finite_number(
            self.importance,
            "importance",
        )
        self.rank = _validate_rank(self.rank)
        self.metadata = _validate_metadata(self.metadata)


@dataclass
class ExplanationResult:
    """
    Standardized result returned by an explainer.
    """

    attributions: list[FeatureAttribution] = field(default_factory=list)
    importances: list[FeatureImportance] = field(default_factory=list)
    prediction: Any = None
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attributions, list):
            raise TypeError("attributions must be a list.")

        for attribution in self.attributions:
            if not isinstance(attribution, FeatureAttribution):
                raise TypeError(
                    "each attribution must be a FeatureAttribution instance."
                )

        self.attributions = list(self.attributions)

        if not isinstance(self.importances, list):
            raise TypeError("importances must be a list.")

        for importance in self.importances:
            if not isinstance(importance, FeatureImportance):
                raise TypeError(
                    "each importance must be a FeatureImportance instance."
                )

        self.importances = list(self.importances)

        if self.confidence is not None:
            self.confidence = _validate_finite_number(
                self.confidence,
                "confidence",
            )

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0.0 and 1.0."
                )

        self.warnings = _validate_warnings(self.warnings)
        self.metadata = _validate_metadata(self.metadata)