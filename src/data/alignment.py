"""Date and frequency alignment utilities for Phase 5 Group C."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class TimeSeriesAligner:
    """Sorts time-series records and aligns dates to project conventions."""

    def align(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Align records chronologically and remove invalid date records."""
        aligned = [
            dict(record)
            for record in records
            if self._normalize_date(record.get("date")) is not None
        ]

        for record in aligned:
            record["date"] = self._normalize_date(record.get("date"))

        aligned.sort(key=lambda record: record["date"])

        return self._deduplicate_dates(aligned)

    @staticmethod
    def _normalize_date(value: Any) -> str | None:
        """Normalize supported dates to ISO YYYY-MM-DD."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date().isoformat()

        text = str(value).strip()

        if not text:
            return None

        formats = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d %b %Y",
        )

        for format_string in formats:
            try:
                return datetime.strptime(
                    text,
                    format_string,
                ).date().isoformat()
            except ValueError:
                continue

        return None

    @staticmethod
    def _deduplicate_dates(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep the latest record when duplicate dates are present."""
        by_date: dict[str, dict[str, Any]] = {}

        for record in records:
            by_date[record["date"]] = record

        return [
            by_date[date_value]
            for date_value in sorted(by_date)
        ]