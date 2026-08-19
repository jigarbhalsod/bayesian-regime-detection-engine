"""Dataset transformation utilities for Phase 5 Group C."""

from __future__ import annotations

from typing import Any


class TimeSeriesTransformer:
    """Adds dataset-specific derived features."""

    def transform(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Transform records and add dataset-specific features."""

        sorted_records = sorted(
            [dict(record) for record in records],
            key=lambda record: str(record.get("date") or ""),
        )

        transformer = getattr(
            self,
            f"_transform_{dataset_id}",
            self._transform_generic,
        )

        return transformer(sorted_records)

    @staticmethod
    def _transform_generic(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return generic records without additional transformations."""
        return [dict(record) for record in records]

    @staticmethod
    def _transform_nifty_50(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add daily returns for chronologically ordered NIFTY 50 records."""

        transformed: list[dict[str, Any]] = []
        previous_close: float | None = None

        for record in records:
            transformed_record = dict(record)

            close = TimeSeriesTransformer._to_float(
                transformed_record.get("close")
            )

            if (
                previous_close is None
                or close is None
                or previous_close == 0
            ):
                transformed_record["returns"] = None
            else:
                transformed_record["returns"] = (
                    close - previous_close
                ) / previous_close

            if close is not None:
                previous_close = close

            transformed.append(transformed_record)

        return transformed

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Safely convert a value to float."""

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None