"""Column and value standardization utilities for Phase 5 Group B."""

from __future__ import annotations

from typing import Any


class RecordStandardizer:
    """Standardizes parsed records into consistent project conventions."""

    def standardize(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize column names and common missing values."""

        return [
            self._standardize_record(record)
            for record in records
        ]

    @staticmethod
    def _standardize_record(record: dict[str, Any]) -> dict[str, Any]:
        """Standardize one record."""

        standardized: dict[str, Any] = {}

        for key, value in record.items():
            normalized_key = RecordStandardizer._normalize_key(key)
            standardized[normalized_key] = RecordStandardizer._normalize_value(value)

        return standardized

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Convert source column names to snake_case."""

        return (
            str(key)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize common empty and textual missing values."""

        if isinstance(value, str):
            value = value.strip()

            if value.lower() in {
                "",
                "na",
                "n/a",
                "nan",
                "null",
                "none",
                "-",
            }:
                return None

        return value