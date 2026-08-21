"""Volume and liquidity feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class VolumeFeatureTransformer(BaseFeatureTransformer):
    """Creates historical volume and liquidity features."""

    name = "volume"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create volume features using current and historical records."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()

        volume_column = self.config.target_volume_column
        close_column = self.config.target_close_column

        volumes = [
            self._valid_positive(
                record.get(volume_column)
            )
            for record in transformed_records
        ]

        closes = [
            self._valid_positive(
                record.get(close_column)
            )
            for record in transformed_records
        ]

        for index, record in enumerate(transformed_records):
            self._add_one_day_features(
                record=record,
                volumes=volumes,
                closes=closes,
                index=index,
            )

            for period in self.config.volume_periods:
                self._add_period_features(
                    record=record,
                    volumes=volumes,
                    index=index,
                    period=period,
                )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _add_one_day_features(
        self,
        *,
        record: dict[str, Any],
        volumes: list[float | None],
        closes: list[float | None],
        index: int,
    ) -> None:
        """Add one-day volume and traded-value features."""

        current_volume = volumes[index]
        current_close = closes[index]

        if (
            index == 0
            or current_volume is None
            or volumes[index - 1] is None
        ):
            record["feature__volume_change_1d"] = None
            record["feature__volume_return_1d"] = None
        else:
            previous_volume = volumes[index - 1]

            record["feature__volume_change_1d"] = (
                current_volume - previous_volume
            )

            record["feature__volume_return_1d"] = (
                current_volume / previous_volume
            ) - 1.0

        if current_volume is None or current_close is None:
            record["feature__traded_value"] = None
        else:
            record["feature__traded_value"] = (
                current_volume * current_close
            )

    def _add_period_features(
        self,
        *,
        record: dict[str, Any],
        volumes: list[float | None],
        index: int,
        period: int,
    ) -> None:
        """Add historical volume features for one complete window."""

        average_name = f"feature__average_volume_{period}d"
        relative_name = f"feature__relative_volume_{period}d"
        volatility_name = f"feature__volume_volatility_{period}d"
        trend_name = f"feature__volume_trend_{period}d"

        window = self._complete_window(
            values=volumes,
            end_index=index,
            period=period,
        )

        if window is None:
            record[average_name] = None
            record[relative_name] = None
            record[volatility_name] = None
            record[trend_name] = None
            return

        average_volume = sum(window) / len(window)

        variance = sum(
            (value - average_volume) ** 2
            for value in window
        ) / len(window)

        current_volume = volumes[index]

        record[average_name] = average_volume

        if average_volume == 0 or current_volume is None:
            record[relative_name] = None
        else:
            record[relative_name] = (
                current_volume / average_volume
            )

        record[volatility_name] = math.sqrt(variance)

        first_volume = window[0]

        if first_volume <= 0:
            record[trend_name] = None
        else:
            record[trend_name] = (
                window[-1] / first_volume
            ) - 1.0

    @staticmethod
    def _complete_window(
        *,
        values: list[float | None],
        end_index: int,
        period: int,
    ) -> list[float] | None:
        """Return a complete valid historical window."""

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

    def _feature_names(self) -> tuple[str, ...]:
        """Return every feature created by this transformer."""

        names: list[str] = [
            "feature__volume_change_1d",
            "feature__volume_return_1d",
            "feature__traded_value",
        ]

        for period in self.config.volume_periods:
            names.extend(
                (
                    f"feature__average_volume_{period}d",
                    f"feature__relative_volume_{period}d",
                    f"feature__volume_volatility_{period}d",
                    f"feature__volume_trend_{period}d",
                )
            )

        return tuple(names)

    @staticmethod
    def _valid_positive(
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