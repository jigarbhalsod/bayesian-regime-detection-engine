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
    """Creates historical price-based features without look-ahead bias."""

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
        """Create price features using only current and past prices."""

        close_column = self.config.target_close_column

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()

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

            previous_close = self._value_at(
                transformed_records,
                index - 1,
                close_column,
            )

            if previous_close is None or previous_close <= 0:
                record["feature__price_change_1d"] = None
            else:
                record["feature__price_change_1d"] = (
                    current_close - previous_close
                )

            for period in self.config.return_periods:
                if period == 1:
                    continue

                feature_name = (
                    f"feature__price_change_{period}d"
                )

                past_close = self._value_at(
                    transformed_records,
                    index - period,
                    close_column,
                )

                if past_close is None or past_close <= 0:
                    record[feature_name] = None
                else:
                    record[feature_name] = (
                        current_close - past_close
                    )

            for period in self.config.return_periods:
                window = self._historical_window(
                    records=transformed_records,
                    end_index=index,
                    period=period,
                    column=close_column,
                )

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

                if window is None:
                    record[max_name] = None
                    record[min_name] = None
                    record[position_name] = None
                    record[drawdown_name] = None
                    continue

                rolling_max = max(window)
                rolling_min = min(window)

                record[max_name] = rolling_max
                record[min_name] = rolling_min
                record[drawdown_name] = (
                    current_close / rolling_max
                ) - 1.0

                price_range = rolling_max - rolling_min

                if price_range == 0:
                    record[position_name] = None
                else:
                    record[position_name] = (
                        (current_close - rolling_min)
                        / price_range
                    )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _feature_names(self) -> tuple[str, ...]:
        """Return the complete set of generated feature names."""

        names = [
            "feature__price_change_1d",
        ]

        for period in self.config.return_periods:
            if period != 1:
                names.append(
                    f"feature__price_change_{period}d"
                )

        for period in self.config.return_periods:
            names.extend(
                (
                    f"feature__rolling_max_{period}d",
                    f"feature__rolling_min_{period}d",
                    f"feature__price_position_{period}d",
                    f"feature__drawdown_{period}d",
                )
            )

        return tuple(names)

    def _historical_window(
        self,
        *,
        records: list[dict[str, Any]],
        end_index: int,
        period: int,
        column: str,
    ) -> list[float] | None:
        """Return a complete trailing window ending at end_index."""

        start_index = end_index - period + 1

        if start_index < 0:
            return None

        values: list[float] = []

        for index in range(start_index, end_index + 1):
            value = self._value_at(
                records,
                index,
                column,
            )

            if value is None or value <= 0:
                return None

            values.append(value)

        return values

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

    def _value_at(
        self,
        records: list[dict[str, Any]],
        index: int,
        column: str,
    ) -> float | None:
        """Read one valid historical numeric value."""

        if index < 0 or index >= len(records):
            return None

        return self._to_float(
            records[index].get(column)
        )

    @staticmethod
    def _set_missing_features(
        record: dict[str, Any],
        feature_names: tuple[str, ...],
    ) -> None:
        """Set every generated feature to missing."""

        for feature_name in feature_names:
            record[feature_name] = None