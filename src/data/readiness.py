"""Dataset readiness checks for Phase 5 Group D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetReadinessReport:
    """Summary of whether a prepared dataset is ready for downstream use."""

    dataset_id: str
    is_ready: bool
    record_count: int
    required_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    invalid_record_count: int


class DatasetReadinessChecker:
    """Checks whether prepared records are ready for downstream modelling."""

    REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
        "nifty_50": ("date", "close", "returns"),
        "india_vix": ("date", "level"),
        "cboe_vix": ("date", "level"),
        "sp500": ("date", "close"),
        "crude_oil": ("date", "price"),
        "cpi_india": ("date",),
        "repo_rate": ("date",),
    }

    def check(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> DatasetReadinessReport:
        """Check dataset structure and record-level completeness."""

        required_fields = self.REQUIRED_FIELDS.get(
            dataset_id,
            ("date",),
        )

        available_fields: set[str] = set()

        for record in records:
            available_fields.update(record.keys())

        missing_fields = tuple(
            field
            for field in required_fields
            if field not in available_fields
        )

        invalid_record_count = sum(
            1
            for record in records
            if not self._is_valid_record(
                record=record,
                required_fields=required_fields,
            )
        )

        is_ready = (
            len(records) > 0
            and not missing_fields
            and invalid_record_count == 0
        )

        return DatasetReadinessReport(
            dataset_id=dataset_id,
            is_ready=is_ready,
            record_count=len(records),
            required_fields=required_fields,
            missing_fields=missing_fields,
            invalid_record_count=invalid_record_count,
        )

    @staticmethod
    def _is_valid_record(
        *,
        record: dict[str, Any],
        required_fields: tuple[str, ...],
    ) -> bool:
        """Check that required values exist in one record."""

        for field in required_fields:
            value = record.get(field)

            if value is None or value == "":
                return False

        return True