"""Return-based feature engineering for Phase 6."""

from __future__ import annotations

import math
from typing import Any

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig


class ReturnFeatureTransformer(BaseFeatureTransformer):
    """Creates historical return-based features without look-ahead bias."""

    name = "returns"

    def __init__(
        self,
        config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create return features using only current and past prices."""

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

            if previous_close is not None and previous_close > 0:
                record["feature__return_1d"] = (
                    current_close / previous_close
                ) - 1.0

                record["feature__log_return_1d"] = math.log(
                    current_close / previous_close
                )
            else:
                record["feature__return_1d"] = None
                record["feature__log_return_1d"] = None

            for period in self.config.return_periods:
                feature_name = (
                    f"feature__return_{period}d"
                )

                if period == 1:
                    continue

                past_close = self._value_at(
                    transformed_records,
                    index - period,
                    close_column,
                )

                if past_close is None or past_close <= 0:
                    record[feature_name] = None
                    continue

                record[feature_name] = (
                    current_close / past_close
                ) - 1.0

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    def _feature_names(self) -> tuple[str, ...]:
        """Return the complete set of features created."""

        names = [
            "feature__return_1d",
            "feature__log_return_1d",
        ]

        for period in self.config.return_periods:
            if period != 1:
                names.append(
                    f"feature__return_{period}d"
                )

        return tuple(names)

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:
        """Convert a numeric value to float when valid."""

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
        """Return a valid numeric value at a historical index."""

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
        """Set all generated features to missing."""

        for feature_name in feature_names:
            record[feature_name] = None