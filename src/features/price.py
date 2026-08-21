"""Price-based feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class PriceFeatureTransformer(BaseFeatureTransformer):
    """Creates historical price features without look-ahead bias."""

    name = "price"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create price features from chronological market records."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()
        close_column = self.config.target_close_column

        closes = [
            self._to_positive_float(
                record.get(close_column)
            )
            for record in transformed_records
        ]

        for index, record in enumerate(transformed_records):
            current_close = closes[index]

            if current_close is None:
                self._set_missing_features(
                    record=record,
                    feature_names=feature_names,
                )
                continue

            self._add_price_change_features(
                record=record,
                closes=closes,
                index=index,
            )

            self._add_rolling_price_features(
                record=record,
                closes=closes,
                index=index,
            )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _add_price_change_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
    ) -> None:
        """Add absolute historical price changes."""

        current_close = closes[index]

        for period in self.config.resolved_price_change_periods:
            feature_name = (
                f"feature__price_change_{period}d"
            )

            historical_index = index - period

            if (
                historical_index < 0
                or current_close is None
                or closes[historical_index] is None
            ):
                record[feature_name] = None
                continue

            historical_close = closes[historical_index]

            record[feature_name] = (
                current_close - historical_close
            )

    def _add_rolling_price_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
    ) -> None:
        """Add rolling high, low, position and drawdown features."""

        current_close = closes[index]

        for period in self.config.resolved_price_rolling_periods:
            max_name = (
                f"feature__rolling_max_{period}d"
            )
            min_name = (
                f"feature__rolling_min_{period}d"
            )
            position_name = (
                f"feature__price_position_{period}d"
            )
            drawdown_name = (
                f"feature__drawdown_{period}d"
            )

            window = self._historical_window(
                values=closes,
                end_index=index,
                period=period,
            )

            if window is None or current_close is None:
                record[max_name] = None
                record[min_name] = None
                record[position_name] = None
                record[drawdown_name] = None
                continue

            rolling_max = max(window)
            rolling_min = min(window)

            record[max_name] = rolling_max
            record[min_name] = rolling_min

            price_range = (
                rolling_max - rolling_min
            )

            if price_range == 0:
                record[position_name] = None
            else:
                record[position_name] = (
                    (current_close - rolling_min)
                    / price_range
                )

            if rolling_max <= 0:
                record[drawdown_name] = None
            else:
                record[drawdown_name] = (
                    (current_close / rolling_max)
                    - 1.0
                )

    def _feature_names(self) -> tuple[str, ...]:
        """Return the complete set of generated feature names."""

        names: list[str] = []

        for period in self.config.resolved_price_change_periods:
            names.append(
                f"feature__price_change_{period}d"
            )

        for period in self.config.resolved_price_rolling_periods:
            names.extend(
                (
                    f"feature__rolling_max_{period}d",
                    f"feature__rolling_min_{period}d",
                    f"feature__price_position_{period}d",
                    f"feature__drawdown_{period}d",
                )
            )

        return tuple(names)

    @staticmethod
    def _historical_window(
        *,
        values: list[float | None],
        end_index: int,
        period: int,
    ) -> list[float] | None:
        """Return a complete valid rolling window."""

        start_index = end_index - period + 1

        if start_index < 0:
            return None

        window = values[
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

    @staticmethod
    def _to_positive_float(
        value: Any,
    ) -> float | None:
        """Convert a value to a valid positive finite float."""

        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
        ):
            return None

        return numeric_value

    @staticmethod
    def _set_missing_features(
        *,
        record: dict[str, Any],
        feature_names: tuple[str, ...],
    ) -> None:
        """Set every generated feature to missing."""

        for feature_name in feature_names:
            record[feature_name] = None