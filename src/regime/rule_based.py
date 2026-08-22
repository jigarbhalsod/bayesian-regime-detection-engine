import math
from typing import Any

from .base import BaseRegimeDetector
from .config import RegimeConfig
from .result import RegimeResult


class RuleBasedRegimeDetector(BaseRegimeDetector):
    """
    Interpretable rule-based baseline regime detector.
    """

    def __init__(
        self,
        config: RegimeConfig | None = None,
        return_column: str = "return",
        volatility_column: str = "volatility",
        positive_return_threshold: float = 0.0,
        high_volatility_threshold: float = 0.02,
    ) -> None:
        super().__init__(config=config)

        self.return_column = return_column
        self.volatility_column = volatility_column
        self.positive_return_threshold = positive_return_threshold
        self.high_volatility_threshold = high_volatility_threshold

    @property
    def name(self) -> str:
        return "rule_based"

    def detect(
        self,
        records: list[dict[str, Any]],
    ) -> RegimeResult:
        labels: list[str] = []
        output_records = [dict(record) for record in records]

        for record in output_records:
            return_value = self._to_float(
                record.get(self.return_column)
            )
            volatility_value = self._to_float(
                record.get(self.volatility_column)
            )

            label = self._classify(
                return_value,
                volatility_value,
            )

            record["regime"] = label
            labels.append(label)

        return RegimeResult(
            records=output_records,
            regime_labels=labels,
            metadata={
                "detector": self.name,
                "return_column": self.return_column,
                "volatility_column": self.volatility_column,
                "positive_return_threshold": (
                    self.positive_return_threshold
                ),
                "high_volatility_threshold": (
                    self.high_volatility_threshold
                ),
            },
        )

    def _classify(
        self,
        return_value: float | None,
        volatility_value: float | None,
    ) -> str:
        if return_value is None or volatility_value is None:
            return "unknown"

        if (
            return_value > self.positive_return_threshold
            and volatility_value < self.high_volatility_threshold
        ):
            return "risk_on"

        if (
            return_value <= self.positive_return_threshold
            and volatility_value >= self.high_volatility_threshold
        ):
            return "risk_off"

        return "transitional"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return number