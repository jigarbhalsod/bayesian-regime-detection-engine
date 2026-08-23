from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EnsembleResult:
    """
    Standard output contract for an ensemble prediction.
    """

    prediction: Any
    confidence: float
    uncertainty: float
    model_weights: Mapping[str, float] = field(
        default_factory=dict
    )
    model_outputs: Mapping[str, Any] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        confidence = self._validate_probability(
            self.confidence,
            "confidence",
        )

        uncertainty = self._validate_probability(
            self.uncertainty,
            "uncertainty",
        )

        model_weights = self._validate_model_weights(
            self.model_weights
        )

        model_outputs = self._validate_mapping(
            self.model_outputs,
            "model_outputs",
        )

        metadata = self._validate_mapping(
            self.metadata,
            "metadata",
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "uncertainty",
            uncertainty,
        )

        object.__setattr__(
            self,
            "model_weights",
            model_weights,
        )

        object.__setattr__(
            self,
            "model_outputs",
            model_outputs,
        )

        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @staticmethod
    def _validate_probability(
        value: Any,
        field_name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

        return value

    @staticmethod
    def _validate_mapping(
        value: Any,
        field_name: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping."
            )

        return dict(value)

    @classmethod
    def _validate_model_weights(
        cls,
        value: Any,
    ) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise TypeError(
                "model_weights must be a mapping."
            )

        validated: dict[str, float] = {}

        for model_name, weight in value.items():
            if not isinstance(model_name, str):
                raise TypeError(
                    "model weight names must be strings."
                )

            if not model_name.strip():
                raise ValueError(
                    "model weight names cannot be empty."
                )

            if isinstance(weight, bool) or not isinstance(
                weight,
                (int, float),
            ):
                raise TypeError(
                    "model weights must be numeric."
                )

            weight = float(weight)

            if not math.isfinite(weight):
                raise ValueError(
                    "model weights must be finite."
                )

            if weight < 0.0:
                raise ValueError(
                    "model weights cannot be negative."
                )

            validated[model_name] = weight

        return validated

    def to_dict(self) -> dict[str, Any]:
        """
        Return the result as a plain dictionary.
        """
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "model_weights": dict(
                self.model_weights
            ),
            "model_outputs": dict(
                self.model_outputs
            ),
            "metadata": dict(
                self.metadata
            ),
        }