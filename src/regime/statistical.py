import math
import statistics
from typing import Any

from .base import BaseRegimeDetector
from .config import RegimeConfig
from .result import RegimeResult


class StatisticalRegimeDetector(BaseRegimeDetector):
    """
    Statistical threshold-based baseline regime detector.
    """

    def __init__(
        self,
        config: RegimeConfig | None = None,
        return_column: str = "return",
        volatility_column: str = "volatility",
    ) -> None:
        super().__init__(config=config)

        self.return_column = return_column
        self.volatility_column = volatility_column

    @property
    def name(self) -> str:
        return "statistical"

    def detect(
        self,
        records: list[dict[str, Any]],
    ) -> RegimeResult:
        output_records = [dict(record) for record in records]

        valid_returns = [
            value
            for record in output_records
            if (
                value := self._to_float(
                    record.get(self.return_column)
                )
            ) is not None
        ]

        valid_volatilities = [
            value
            for record in output_records
            if (
                value := self._to_float(
                    record.get(self.volatility_column)
                )
            ) is not None
        ]

        return_threshold = (
            statistics.median(valid_returns)
            if valid_returns
            else None
        )

        volatility_threshold = (
            statistics.median(valid_volatilities)
            if valid_volatilities
            else None
        )

        labels: list[str] = []

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
                return_threshold,
                volatility_threshold,
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
                "return_threshold": return_threshold,
                "volatility_threshold": volatility_threshold,
                "valid_return_count": len(valid_returns),
                "valid_volatility_count": len(valid_volatilities),
            },
        )

    @staticmethod
    def _classify(
        return_value: float | None,
        volatility_value: float | None,
        return_threshold: float | None,
        volatility_threshold: float | None,
    ) -> str:
        if (
            return_value is None
            or volatility_value is None
            or return_threshold is None
            or volatility_threshold is None
        ):
            return "unknown"

        if (
            return_value >= return_threshold
            and volatility_value < volatility_threshold
        ):
            return "risk_on"

        if (
            return_value < return_threshold
            and volatility_value >= volatility_threshold
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