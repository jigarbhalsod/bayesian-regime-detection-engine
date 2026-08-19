"""Trend and momentum feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class MomentumFeatureTransformer(BaseFeatureTransformer):
    """Creates trend and momentum features without look-ahead bias."""

    name = "momentum"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create SMA, EMA, momentum, and ROC features."""

        close_column = self.config.target_close_column

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()

        closes = [
            self._to_float(record.get(close_column))
            for record in transformed_records
        ]

        for index, record in enumerate(transformed_records):
            current_close = closes[index]

            if current_close is None or current_close <= 0:
                self._set_missing_features(
                    record,
                    feature_names,
                )
                continue

            self._add_sma_features(
                record=record,
                closes=closes,
                index=index,
                current_close=current_close,
            )

            self._add_ema_features(
                record=record,
                closes=closes,
                index=index,
                current_close=current_close,
            )

            self._add_momentum_features(
                record=record,
                closes=closes,
                index=index,
                current_close=current_close,
            )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _add_sma_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
        current_close: float,
    ) -> None:
        """Add simple moving average and price/SMA features."""

        for period in self.config.sma_periods:
            sma_name = f"feature__sma_{period}d"
            ratio_name = f"feature__price_to_sma_{period}d"

            window = self._complete_window(
                closes=closes,
                end_index=index,
                period=period,
            )

            if window is None:
                record[sma_name] = None
                record[ratio_name] = None
                continue

            sma = sum(window) / period

            record[sma_name] = sma
            record[ratio_name] = (
                current_close / sma
                if sma != 0
                else None
            )

    def _add_ema_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
        current_close: float,
    ) -> None:
        """Add EMA and price/EMA features using historical data only."""

        for period in self.config.ema_periods:
            ema_name = f"feature__ema_{period}d"
            ratio_name = f"feature__price_to_ema_{period}d"

            window = self._complete_window(
                closes=closes,
                end_index=index,
                period=period,
            )

            if window is None:
                record[ema_name] = None
                record[ratio_name] = None
                continue

            multiplier = 2.0 / (period + 1.0)

            ema = window[0]

            for price in window[1:]:
                ema = (
                    (price - ema) * multiplier
                ) + ema

            record[ema_name] = ema
            record[ratio_name] = (
                current_close / ema
                if ema != 0
                else None
            )

    def _add_momentum_features(
        self,
        *,
        record: dict[str, Any],
        closes: list[float | None],
        index: int,
        current_close: float,
    ) -> None:
        """Add absolute momentum and rate-of-change features."""

        for period in self.config.momentum_periods:
            momentum_name = f"feature__momentum_{period}d"
            roc_name = f"feature__roc_{period}d"

            past_index = index - period

            if past_index < 0:
                record[momentum_name] = None
                record[roc_name] = None
                continue

            past_close = closes[past_index]

            if past_close is None or past_close <= 0:
                record[momentum_name] = None
                record[roc_name] = None
                continue

            record[momentum_name] = (
                current_close - past_close
            )

            record[roc_name] = (
                (current_close / past_close) - 1.0
            )

    @staticmethod
    def _complete_window(
        *,
        closes: list[float | None],
        end_index: int,
        period: int,
    ) -> list[float] | None:
        """Return a complete valid trailing price window."""

        start_index = end_index - period + 1

        if start_index < 0:
            return None

        window = closes[
            start_index:end_index + 1
        ]

        if len(window) != period:
            return None

        if any(
            value is None or value <= 0
            for value in window
        ):
            return None

        return [
            float(value)
            for value in window
        ]

    def _feature_names(self) -> tuple[str, ...]:
        """Return all generated feature names."""

        names: list[str] = []

        for period in self.config.sma_periods:
            names.extend(
                (
                    f"feature__sma_{period}d",
                    f"feature__price_to_sma_{period}d",
                )
            )

        for period in self.config.ema_periods:
            names.extend(
                (
                    f"feature__ema_{period}d",
                    f"feature__price_to_ema_{period}d",
                )
            )

        for period in self.config.momentum_periods:
            names.extend(
                (
                    f"feature__momentum_{period}d",
                    f"feature__roc_{period}d",
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
        """Set every generated feature to missing."""

        for feature_name in feature_names:
            record[feature_name] = None