"""Macro-economic feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import BaseFeatureTransformer, FeatureResult
from src.features.config import FeatureConfig


class MacroFeatureTransformer(BaseFeatureTransformer):
    """Create chronological features from aligned macro-economic variables."""

    name = "macro"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create macro features using only current and historical values."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        for column in self.config.macro_columns:
            values = [
                self._valid_value(record.get(column))
                for record in transformed_records
            ]

            prefix = self._feature_prefix(column)

            for index, record in enumerate(transformed_records):
                current_value = values[index]

                for period in self.config.resolved_macro_change_periods:
                    change = self._period_change(
                        values,
                        index,
                        period,
                    )

                    record[
                        f"feature__macro__{prefix}"
                        f"__change_{period}"
                    ] = change

                for period in self.config.macro_rolling_periods:
                    mean = self._rolling_mean(
                        values,
                        index,
                        period,
                    )

                    deviation = None

                    if (
                        current_value is not None
                        and mean is not None
                    ):
                        deviation = current_value - mean

                    zscore = self._rolling_zscore(
                        values,
                        index,
                        period,
                    )

                    record[
                        f"feature__macro__{prefix}"
                        f"__mean_{period}"
                    ] = mean

                    record[
                        f"feature__macro__{prefix}"
                        f"__deviation_{period}"
                    ] = deviation

                    record[
                        f"feature__macro__{prefix}"
                        f"__zscore_{period}"
                    ] = zscore

        return FeatureResult(
            records=transformed_records,
            created_features=self._feature_names(),
        )

    @staticmethod
    def _valid_value(
        value: Any,
    ) -> float | None:
        """Convert a value into a finite numeric macro observation."""

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
    def _period_change(
        values: list[float | None],
        index: int,
        period: int,
    ) -> float | None:
        """Calculate absolute change over a historical period."""

        historical_index = index - period

        if historical_index < 0:
            return None

        current_value = values[index]
        historical_value = values[historical_index]

        if (
            current_value is None
            or historical_value is None
        ):
            return None

        return current_value - historical_value

    @staticmethod
    def _rolling_mean(
        values: list[float | None],
        index: int,
        period: int,
    ) -> float | None:
        """Calculate a complete trailing rolling mean."""

        start_index = index - period + 1

        if start_index < 0:
            return None

        window = values[start_index:index + 1]

        if (
            len(window) != period
            or any(value is None for value in window)
        ):
            return None

        return sum(
            float(value)
            for value in window
            if value is not None
        ) / period

    @staticmethod
    def _rolling_zscore(
        values: list[float | None],
        index: int,
        period: int,
    ) -> float | None:
        """Calculate z-score from a complete trailing window."""

        start_index = index - period + 1

        if start_index < 0:
            return None

        window = values[start_index:index + 1]

        if (
            len(window) != period
            or any(value is None for value in window)
        ):
            return None

        numeric_window = [
            float(value)
            for value in window
            if value is not None
        ]

        mean = sum(numeric_window) / period

        variance = sum(
            (value - mean) ** 2
            for value in numeric_window
        ) / period

        standard_deviation = math.sqrt(variance)

        if standard_deviation == 0:
            return None

        return (
            numeric_window[-1] - mean
        ) / standard_deviation

    def _feature_names(self) -> tuple[str, ...]:
        """Return all generated macro feature names."""

        names: list[str] = []

        for column in self.config.macro_columns:
            prefix = self._feature_prefix(column)

            for period in self.config.resolved_macro_change_periods:
                names.append(
                    f"feature__macro__{prefix}"
                    f"__change_{period}"
                )

            for period in self.config.macro_rolling_periods:
                names.append(
                    f"feature__macro__{prefix}"
                    f"__mean_{period}"
                )
                names.append(
                    f"feature__macro__{prefix}"
                    f"__deviation_{period}"
                )
                names.append(
                    f"feature__macro__{prefix}"
                    f"__zscore_{period}"
                )

        return tuple(names)

    @staticmethod
    def _feature_prefix(
        column: str,
    ) -> str:
        """Convert a source column into a stable feature prefix."""

        return (
            column
            .replace("__", "_")
            .replace("-", "_")
            .lower()
        )