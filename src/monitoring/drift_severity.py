from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DriftSeverityResult:
    score: float
    severity: DriftSeverity
    thresholds: dict[str, float]


class DriftSeverityEvaluator:
    """
    Evaluate a drift score against ordered severity thresholds.

    Thresholds are provided as:

        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
        }

    Scores below the medium threshold are classified as LOW.
    """

    REQUIRED_THRESHOLDS = (
        "medium",
        "high",
        "critical",
    )

    def __init__(
        self,
        thresholds: dict[str, Any],
    ) -> None:
        self._thresholds = self._validate_thresholds(
            thresholds
        )

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def evaluate(
        self,
        score: Any,
    ) -> DriftSeverityResult:
        numeric_score = self._validate_score(score)

        if (
            numeric_score
            >= self._thresholds["critical"]
        ):
            severity = DriftSeverity.CRITICAL
        elif (
            numeric_score
            >= self._thresholds["high"]
        ):
            severity = DriftSeverity.HIGH
        elif (
            numeric_score
            >= self._thresholds["medium"]
        ):
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return DriftSeverityResult(
            score=numeric_score,
            severity=severity,
            thresholds=self.thresholds,
        )

    @classmethod
    def _validate_thresholds(
        cls,
        thresholds: Any,
    ) -> dict[str, float]:
        if not isinstance(thresholds, dict):
            raise TypeError(
                "thresholds must be a dictionary."
            )

        if set(thresholds.keys()) != set(
            cls.REQUIRED_THRESHOLDS
        ):
            raise ValueError(
                "thresholds must contain exactly: "
                "medium, high, critical."
            )

        normalized: dict[str, float] = {}

        for name in cls.REQUIRED_THRESHOLDS:
            value = thresholds[name]

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} threshold must be numeric."
                )

            numeric_value = float(value)

            if numeric_value < 0.0:
                raise ValueError(
                    f"{name} threshold must be non-negative."
                )

            normalized[name] = numeric_value

        if not (
            normalized["medium"]
            <= normalized["high"]
            <= normalized["critical"]
        ):
            raise ValueError(
                "thresholds must satisfy "
                "medium <= high <= critical."
            )

        return normalized

    @staticmethod
    def _validate_score(
        score: Any,
    ) -> float:
        if isinstance(score, bool) or not isinstance(
            score,
            (int, float),
        ):
            raise TypeError(
                "score must be numeric."
            )

        numeric_score = float(score)

        if numeric_score < 0.0:
            raise ValueError(
                "score must be non-negative."
            )

        return numeric_score