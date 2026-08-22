from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class AdvancedModelResult:
    """
    Standardized result returned by advanced regime models.

    Attributes
    ----------
    regimes:
        Predicted regime labels.
    probabilities:
        Optional probability distributions corresponding to predictions.
    confidence:
        Optional confidence values.
    timestamps:
        Optional timestamps aligned with regime predictions.
    model_name:
        Name of the model producing this result.
    metadata:
        General model metadata.
    diagnostics:
        Model-specific diagnostic information.
    """

    regimes: Sequence[Any] = field(default_factory=list)
    probabilities: Optional[Sequence[Sequence[Any]]] = None
    confidence: Optional[Sequence[Any]] = None
    timestamps: Optional[Sequence[Any]] = None
    model_name: str = "advanced_model"
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.regimes = self._validate_regimes(self.regimes)
        self.model_name = self._validate_model_name(self.model_name)
        self.metadata = self._validate_dict(
            self.metadata,
            "metadata",
        )
        self.diagnostics = self._validate_dict(
            self.diagnostics,
            "diagnostics",
        )

        self.probabilities = self._validate_probabilities(
            self.probabilities
        )
        self.confidence = self._validate_confidence(
            self.confidence
        )
        self.timestamps = self._validate_timestamps(
            self.timestamps
        )

        self._validate_alignment()

    @staticmethod
    def _validate_regimes(value: Any) -> List[Any]:
        if value is None:
            raise TypeError("regimes cannot be None.")

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "regimes must be a sequence, not a string."
            )

        try:
            regimes = list(value)
        except TypeError as exc:
            raise TypeError(
                "regimes must be an iterable."
            ) from exc

        return regimes

    @staticmethod
    def _validate_model_name(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("model_name must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("model_name cannot be empty.")

        return value

    @staticmethod
    def _validate_dict(
        value: Any,
        name: str,
    ) -> Dict[str, Any]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary.")

        return dict(value)

    def _validate_probabilities(
        self,
        value: Optional[Sequence[Sequence[Any]]],
    ) -> Optional[List[List[float]]]:
        if value is None:
            return None

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "probabilities must be a sequence of sequences."
            )

        try:
            rows = list(value)
        except TypeError as exc:
            raise TypeError(
                "probabilities must be an iterable."
            ) from exc

        validated: List[List[float]] = []

        for row in rows:
            if isinstance(row, (str, bytes)):
                raise TypeError(
                    "Each probability row must be a sequence."
                )

            try:
                converted = [float(item) for item in row]
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Probability values must be numeric."
                ) from exc

            if not converted:
                raise ValueError(
                    "Probability rows cannot be empty."
                )

            for probability in converted:
                if probability < 0.0 or probability > 1.0:
                    raise ValueError(
                        "Probability values must be between 0 and 1."
                    )

            total = sum(converted)

            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "Each probability row must sum to 1."
                )

            validated.append(converted)

        return validated

    def _validate_confidence(
        self,
        value: Optional[Sequence[Any]],
    ) -> Optional[List[float]]:
        if value is None:
            return None

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "confidence must be a sequence."
            )

        try:
            confidence = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Confidence values must be numeric."
            ) from exc

        for item in confidence:
            if item < 0.0 or item > 1.0:
                raise ValueError(
                    "Confidence values must be between 0 and 1."
                )

        return confidence

    @staticmethod
    def _validate_timestamps(
        value: Optional[Sequence[Any]],
    ) -> Optional[List[Any]]:
        if value is None:
            return None

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "timestamps must be a sequence."
            )

        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(
                "timestamps must be an iterable."
            ) from exc

    def _validate_alignment(self) -> None:
        expected_length = len(self.regimes)

        if (
            self.probabilities is not None
            and len(self.probabilities) != expected_length
        ):
            raise ValueError(
                "probabilities length must match regimes length."
            )

        if (
            self.confidence is not None
            and len(self.confidence) != expected_length
        ):
            raise ValueError(
                "confidence length must match regimes length."
            )

        if (
            self.timestamps is not None
            and len(self.timestamps) != expected_length
        ):
            raise ValueError(
                "timestamps length must match regimes length."
            )

    @property
    def n_predictions(self) -> int:
        """
        Number of regime predictions.
        """
        return len(self.regimes)

    @property
    def has_probabilities(self) -> bool:
        return self.probabilities is not None

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None

    @property
    def has_timestamps(self) -> bool:
        return self.timestamps is not None

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a serializable representation of the result.
        """
        return {
            "regimes": list(self.regimes),
            "probabilities": (
                [list(row) for row in self.probabilities]
                if self.probabilities is not None
                else None
            ),
            "confidence": (
                list(self.confidence)
                if self.confidence is not None
                else None
            ),
            "timestamps": (
                list(self.timestamps)
                if self.timestamps is not None
                else None
            ),
            "model_name": self.model_name,
            "metadata": dict(self.metadata),
            "diagnostics": dict(self.diagnostics),
        }