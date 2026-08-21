"""Calendar and market context feature engineering for Phase 6."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.features.base import BaseFeatureTransformer, FeatureResult


class CalendarFeatureTransformer(BaseFeatureTransformer):
    """Create calendar-based market context features."""

    name = "calendar"

    def __init__(
        self,
        date_column: str = "date",
    ) -> None:
        self.date_column = date_column

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        """Create calendar features from each record's date."""

        transformed_records = [
            dict(record)
            for record in records
        ]

        feature_names = self._feature_names()

        for record in transformed_records:
            parsed_date = self._parse_date(
                record.get(self.date_column)
            )

            if parsed_date is None:
                for feature_name in feature_names:
                    record[feature_name] = None
                continue

            record[
                "feature__calendar__day_of_week"
            ] = parsed_date.weekday()

            record[
                "feature__calendar__month"
            ] = parsed_date.month

            record[
                "feature__calendar__quarter"
            ] = (
                (parsed_date.month - 1) // 3
            ) + 1

            record[
                "feature__calendar__is_month_start"
            ] = parsed_date.day == 1

            record[
                "feature__calendar__is_month_end"
            ] = self._is_month_end(parsed_date)

            record[
                "feature__calendar__is_quarter_end"
            ] = (
                parsed_date.month in (3, 6, 9, 12)
                and self._is_month_end(parsed_date)
            )

        return FeatureResult(
            records=transformed_records,
            created_features=feature_names,
        )

    @staticmethod
    def _parse_date(
        value: Any,
    ) -> date | None:
        """Parse supported date values safely."""

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if not isinstance(value, str):
            return None

        value = value.strip()

        if not value:
            return None

        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).date()
        except ValueError:
            return None

    @staticmethod
    def _is_month_end(
        value: date,
    ) -> bool:
        """Return whether a date is the final calendar day of its month."""

        if value.month == 12:
            next_month = date(
                value.year + 1,
                1,
                1,
            )
        else:
            next_month = date(
                value.year,
                value.month + 1,
                1,
            )

        return (
            next_month - value
        ).days == 1

    @staticmethod
    def _feature_names() -> tuple[str, ...]:
        """Return all generated calendar feature names."""

        return (
            "feature__calendar__day_of_week",
            "feature__calendar__month",
            "feature__calendar__quarter",
            "feature__calendar__is_month_start",
            "feature__calendar__is_month_end",
            "feature__calendar__is_quarter_end",
        )