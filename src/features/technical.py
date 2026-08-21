"""Technical indicator feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class TechnicalFeatureTransformer(BaseFeatureTransformer):
    """Creates technical indicators using chronological market history."""

    name = "technical"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create RSI, MACD, and Bollinger Band features."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        closes = [
            self._valid_close(
                record.get(self.config.target_close_column)
            )
            for record in transformed_records
        ]

        rsi_values = self._calculate_rsi(closes)

        ema_fast = self._calculate_ema(
            closes,
            self.config.macd_fast_period,
        )

        ema_slow = self._calculate_ema(
            closes,
            self.config.macd_slow_period,
        )

        macd_values = self._subtract_series(
            ema_fast,
            ema_slow,
        )

        macd_signal = self._calculate_ema(
            macd_values,
            self.config.macd_signal_period,
        )

        feature_names = self._feature_names()

        for index, record in enumerate(transformed_records):
            macd_value = macd_values[index]
            signal_value = macd_signal[index]

            record["feature__rsi"] = rsi_values[index]
            record["feature__macd"] = macd_value
            record["feature__macd_signal"] = signal_value

            if (
                macd_value is None
                or signal_value is None
            ):
                record["feature__macd_histogram"] = None
            else:
                record["feature__macd_histogram"] = (
                    macd_value - signal_value
                )

            self._add_bollinger_features(
                record=record,
                closes=closes,
                index=index,
            )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _calculate_rsi(
        self,
        closes: list[float | None],
    ) -> list[float | None]:
        """Calculate RSI from complete historical close windows."""

        period = self.config.resolved_rsi_period

        values: list[float | None] = [
            None
        ] * len(closes)

        for index in range(len(closes)):
            start_index = index - period

            if start_index < 0:
                continue

            window = closes[
                start_index:index + 1
            ]

            if (
                len(window) != period + 1
                or any(
                    value is None
                    for value in window
                )
            ):
                continue

            numeric_window = [
                float(value)
                for value in window
                if value is not None
            ]

            gains: list[float] = []
            losses: list[float] = []

            for previous, current in zip(
                numeric_window[:-1],
                numeric_window[1:],
            ):
                change = current - previous

                gains.append(
                    max(change, 0.0)
                )

                losses.append(
                    abs(min(change, 0.0))
                )

            average_gain = (
                sum(gains) / period
            )

            average_loss = (
                sum(losses) / period
            )

            if average_loss == 0:
                values[index] = (
                    100.0
                    if average_gain > 0
                    else 50.0
                )
                continue

            relative_strength = (
                average_gain / average_loss
            )

            values[index] = (
                100.0
                - (
                    100.0
                    / (
                        1.0
                        + relative_strength
                    )
                )
            )

        return values

    @staticmethod
    def _calculate_ema(
        values: list[float | None],
        period: int,
    ) -> list[float | None]:
        """Calculate EMA using a complete initial window."""

        result: list[float | None] = [
            None
        ] * len(values)

        multiplier = (
            2.0 / (period + 1.0)
        )

        for index in range(len(values)):
            start_index = (
                index - period + 1
            )

            if start_index < 0:
                continue

            window = values[
                start_index:index + 1
            ]

            if (
                len(window) != period
                or any(
                    value is None
                    for value in window
                )
            ):
                continue

            if index == period - 1:
                result[index] = (
                    sum(
                        float(value)
                        for value in window
                        if value is not None
                    )
                    / period
                )
                continue

            previous_ema = result[index - 1]
            current_value = values[index]

            if (
                previous_ema is None
                or current_value is None
            ):
                continue

            result[index] = (
                (
                    current_value
                    - previous_ema
                )
                * multiplier
                + previous_ema
            )

        return result

    @staticmethod
    def _subtract_series(
        first: list[float | None],
        second: list[float | None],
    ) -> list[float | None]:
        """Subtract two aligned feature series."""

        values: list[float | None] = []

        for first_value, second_value in zip(
            first,
            second,
        ):
            if (
                first_value is None
                or second_value is None
            ):
                values.append(None)
            else:
                values.append(
                    first_value - second_value
                )

        return values

    def _add_bollinger_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
    ) -> None:
        """Add Bollinger Band features using a complete rolling window."""

        period = self.config.resolved_bollinger_period

        start_index = (
            index - period + 1
        )

        if start_index < 0:
            self._set_bollinger_missing(record)
            return

        window = closes[
            start_index:index + 1
        ]

        if (
            len(window) != period
            or any(
                value is None
                for value in window
            )
        ):
            self._set_bollinger_missing(record)
            return

        numeric_window = [
            float(value)
            for value in window
            if value is not None
        ]

        middle_band = (
            sum(numeric_window)
            / period
        )

        variance = (
            sum(
                (
                    value - middle_band
                ) ** 2
                for value in numeric_window
            )
            / period
        )

        standard_deviation = math.sqrt(
            variance
        )

        offset = (
            self.config.bollinger_std_multiplier
            * standard_deviation
        )

        upper_band = (
            middle_band + offset
        )

        lower_band = (
            middle_band - offset
        )

        current_close = numeric_window[-1]

        record[
            "feature__bollinger_middle"
        ] = middle_band

        record[
            "feature__bollinger_upper"
        ] = upper_band

        record[
            "feature__bollinger_lower"
        ] = lower_band

        band_width = (
            upper_band - lower_band
        )

        if band_width == 0:
            record[
                "feature__bollinger_position"
            ] = None
        else:
            record[
                "feature__bollinger_position"
            ] = (
                (
                    current_close - lower_band
                )
                / band_width
            )

    @staticmethod
    def _set_bollinger_missing(
        record: dict[str, Any],
    ) -> None:
        """Set Bollinger Band features to missing."""

        record[
            "feature__bollinger_middle"
        ] = None

        record[
            "feature__bollinger_upper"
        ] = None

        record[
            "feature__bollinger_lower"
        ] = None

        record[
            "feature__bollinger_position"
        ] = None

    @staticmethod
    def _valid_close(
        value: Any,
    ) -> float | None:
        """Convert a close value to a valid positive finite float."""

        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
        ):
            return None

        return numeric_value

    @staticmethod
    def _feature_names() -> tuple[str, ...]:
        """Return every feature created by this transformer."""

        return (
            "feature__rsi",
            "feature__macd",
            "feature__macd_signal",
            "feature__macd_histogram",
            "feature__bollinger_middle",
            "feature__bollinger_upper",
            "feature__bollinger_lower",
            "feature__bollinger_position",
        )


# Backward-compatible alias.
TechnicalIndicatorTransformer = TechnicalFeatureTransformer