"""Data cleaning and type handling utilities for Phase 5 Group B."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


class RecordCleaner:
    """Cleans standardized records and converts common data types."""

    DATE_FIELDS = {
        "date",
        "change_date",
        "release_date",
        "publication_date",
    }

    def clean(
        self,
        records: list[dict[str, Any]],
        *,
        dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Clean records, convert values and remove invalid records."""
        cleaned_records = [
            self._clean_record(record)
            for record in records
        ]

        return [
            record
            for record in cleaned_records
            if self._is_valid_record(record, dataset_id=dataset_id)
        ]

    def _clean_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Clean one standardized record."""
        cleaned: dict[str, Any] = {}

        for key, value in record.items():
            if key in self.DATE_FIELDS:
                cleaned[key] = self._parse_date(value)
            else:
                cleaned[key] = self._parse_value(value)

        return cleaned

    @staticmethod
    def _is_valid_record(
        record: dict[str, Any],
        *,
        dataset_id: str | None = None,
    ) -> bool:
        """Keep records that contain a valid date and at least one value."""
        if record.get("date") is None:
            return False

        return any(
            value is not None
            for key, value in record.items()
            if key != "date"
        )

    @staticmethod
    def _parse_date(value: Any) -> str | None:
        """Convert common date formats to ISO YYYY-MM-DD."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date().isoformat()

        if isinstance(value, date):
            return value.isoformat()

        text = str(value).strip()

        if not text:
            return None

        formats = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
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

        return text

    @staticmethod
    def _parse_value(value: Any) -> Any:
        """Convert numeric-looking values while preserving other text."""
        if value is None or isinstance(value, (int, float, bool)):
            return value

        text = str(value).strip()

        if not text:
            return None

        normalized = (
            text.replace(",", "")
            .replace("%", "")
        )

        try:
            return float(normalized)
        except ValueError:
            return text