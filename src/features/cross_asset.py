"""Cross-asset feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class CrossAssetFeatureTransformer(BaseFeatureTransformer):
    """Create relative and relationship features across market assets."""

    name = "cross_asset"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create chronological cross-asset features."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        target_closes = [
            self._valid_price(
                record.get(self.config.target_close_column)
            )
            for record in transformed_records
        ]

        target_returns = self._calculate_returns(
            target_closes,
            self.config.resolved_cross_asset_return_period,
        )

        asset_returns: dict[str, list[float | None]] = {}

        for column in self.config.cross_asset_columns:
            closes = [
                self._valid_price(record.get(column))
                for record in transformed_records
            ]

            asset_returns[column] = self._calculate_returns(
                closes,
                self.config.resolved_cross_asset_return_period,
            )

        for index, record in enumerate(transformed_records):
            for column in self.config.cross_asset_columns:
                prefix = self._feature_prefix(column)

                target_return = target_returns[index]
                asset_return = asset_returns[column][index]

                record[
                    f"feature__cross_asset__{prefix}__return"
                ] = asset_return

                if (
                    target_return is None
                    or asset_return is None
                ):
                    record[
                        f"feature__cross_asset__{prefix}__return_spread"
                    ] = None
                else:
                    record[
                        f"feature__cross_asset__{prefix}__return_spread"
                    ] = (
                        target_return - asset_return
                    )

                for period in (
                    self.config.cross_asset_correlation_periods
                ):
                    correlation = self._rolling_correlation(
                        target_returns,
                        asset_returns[column],
                        index,
                        period,
                    )

                    record[
                        f"feature__cross_asset__{prefix}"
                        f"__correlation_{period}"
                    ] = correlation

        return FeatureResult(
            records=transformed_records,
            created_features=self._feature_names(),
        )

    @staticmethod
    def _valid_price(
        value: Any,
    ) -> float | None:
        """Convert a value to a valid positive finite price."""

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
    def _calculate_returns(
        prices: list[float | None],
        period: int,
    ) -> list[float | None]:
        """Calculate simple returns without using future observations."""

        values: list[float | None] = []

        for index, current_price in enumerate(prices):
            historical_index = index - period

            if historical_index < 0:
                values.append(None)
                continue

            historical_price = prices[historical_index]

            if (
                current_price is None
                or historical_price is None
                or historical_price == 0
            ):
                values.append(None)
                continue

            values.append(
                (current_price - historical_price)
                / historical_price
            )

        return values

    @staticmethod
    def _rolling_correlation(
        first: list[float | None],
        second: list[float | None],
        index: int,
        period: int,
    ) -> float | None:
        """Calculate correlation using a complete trailing window."""

        start_index = index - period + 1

        if start_index < 0:
            return None

        first_window = first[
            start_index:index + 1
        ]
        second_window = second[
            start_index:index + 1
        ]

        if (
            len(first_window) != period
            or len(second_window) != period
            or any(value is None for value in first_window)
            or any(value is None for value in second_window)
        ):
            return None

        first_values = [
            float(value)
            for value in first_window
            if value is not None
        ]
        second_values = [
            float(value)
            for value in second_window
            if value is not None
        ]

        first_mean = (
            sum(first_values)
            / period
        )
        second_mean = (
            sum(second_values)
            / period
        )

        numerator = sum(
            (first_value - first_mean)
            * (second_value - second_mean)
            for first_value, second_value in zip(
                first_values,
                second_values,
            )
        )

        first_variance = sum(
            (value - first_mean) ** 2
            for value in first_values
        )

        second_variance = sum(
            (value - second_mean) ** 2
            for value in second_values
        )

        denominator = math.sqrt(
            first_variance * second_variance
        )

        if denominator == 0:
            return None

        return numerator / denominator

    def _feature_names(self) -> tuple[str, ...]:
        """Return all generated cross-asset feature names."""

        names: list[str] = []

        for column in self.config.cross_asset_columns:
            prefix = self._feature_prefix(column)

            names.append(
                f"feature__cross_asset__{prefix}__return"
            )
            names.append(
                f"feature__cross_asset__{prefix}__return_spread"
            )

            for period in (
                self.config.cross_asset_correlation_periods
            ):
                names.append(
                    f"feature__cross_asset__{prefix}"
                    f"__correlation_{period}"
                )

        return tuple(names)

    @staticmethod
    def _feature_prefix(
        column: str,
    ) -> str:
        """Convert a source column into a stable feature prefix."""

        return (
            column
            .replace("__close", "")
            .replace("__", "_")
        )