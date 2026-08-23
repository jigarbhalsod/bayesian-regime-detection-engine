from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UncertaintyResult:
    """Represents the output of the uncertainty pipeline."""

    prediction: Any
    uncertainty: float
    confidence: float
    level: str
    abstained: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be between 0.0 and 1.0.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        if self.level not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "level must be one of: low, medium, high."
            )