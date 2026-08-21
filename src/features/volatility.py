"""Volatility-based feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class VolatilityFeatureTransformer(BaseFeatureTransformer):
    """Creates historical volatility features without look-ahead bias."""

    name = "volatility"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create volatility features from historical daily returns."""

        close_column = self.config.target_close_column

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()

        returns = self._daily_returns(
            records=transformed_records,
            close_column=close_column,
        )

        for index, record in enumerate(transformed_records):
            current_close = self._to_float(
                record.get(close_column)
            )

            if current_close is None or current_close <= 0:
                self._set_missing_features(
                    record,
                    feature_names,
                )
                continue

            for period in self.config.volatility_periods:
                self._add_period_features(
                    record=record,
                    returns=returns,
                    index=index,
                    period=period,
                )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _add_period_features(
        self,
        *,
        record: dict[str, Any],
        returns: list[float | None],
        index: int,
        period: int,
    ) -> None:
        """Add volatility statistics for one complete return window."""

        variance_name = f"feature__return_variance_{period}d"
        volatility_name = f"feature__return_volatility_{period}d"
        realized_name = f"feature__realized_volatility_{period}d"
        downside_name = f"feature__downside_volatility_{period}d"

        window = self._complete_return_window(
            returns=returns,
            end_index=index,
            period=period,
        )

        if window is None:
            record[variance_name] = None
            record[volatility_name] = None
            record[realized_name] = None
            record[downside_name] = None
            return

        mean_return = sum(window) / len(window)

        variance = sum(
            (value - mean_return) ** 2
            for value in window
        ) / len(window)

        volatility = math.sqrt(variance)

        realized_volatility = math.sqrt(
            sum(value ** 2 for value in window)
        )

        negative_returns = [
            value
            for value in window
            if value < 0
        ]

        if negative_returns:
            downside_variance = sum(
                value ** 2
                for value in negative_returns
            ) / len(window)

            downside_volatility = math.sqrt(
                downside_variance
            )
        else:
            downside_volatility = 0.0

        record[variance_name] = variance
        record[volatility_name] = volatility
        record[realized_name] = realized_volatility
        record[downside_name] = downside_volatility

    @staticmethod
    def _complete_return_window(
        *,
        returns: list[float | None],
        end_index: int,
        period: int,
    ) -> list[float] | None:
        """Return a complete historical window of valid returns."""

        start_index = end_index - period + 1

        if start_index < 1:
            return None

        window = returns[
            start_index:end_index + 1
        ]

        if len(window) != period:
            return None

        if any(value is None for value in window):
            return None

        return [
            float(value)
            for value in window
            if value is not None
        ]

    def _daily_returns(
        self,
        *,
        records: list[dict[str, Any]],
        close_column: str,
    ) -> list[float | None]:
        """Calculate one-day returns using only adjacent history."""

        returns: list[float | None] = []

        for index, record in enumerate(records):
            current_close = self._to_float(
                record.get(close_column)
            )

            if index == 0:
                returns.append(None)
                continue

            previous_close = self._to_float(
                records[index - 1].get(close_column)
            )

            if (
                current_close is None
                or previous_close is None
                or current_close <= 0
                or previous_close <= 0
            ):
                returns.append(None)
                continue

            returns.append(
                (current_close / previous_close) - 1.0
            )

        return returns

    def _feature_names(self) -> tuple[str, ...]:
        """Return every generated volatility feature name."""

        names: list[str] = []

        for period in self.config.volatility_periods:
            names.extend(
                (
                    f"feature__return_variance_{period}d",
                    f"feature__return_volatility_{period}d",
                    f"feature__realized_volatility_{period}d",
                    f"feature__downside_volatility_{period}d",
                )
            )

        return tuple(names)

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:
        """Convert a value to a valid finite float."""

        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    @staticmethod
    def _set_missing_features(
        record: dict[str, Any],
        feature_names: tuple[str, ...],
    ) -> None:
        """Set all generated features to missing."""

        for feature_name in feature_names:
            record[feature_name] = None