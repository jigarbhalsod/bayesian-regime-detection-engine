"""Data quality validation utilities for Phase 5 Group B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of validation results for a dataset."""

    dataset_id: str
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    missing_required_fields: dict[str, int]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return True when no validation errors were found."""
        return not self.errors


class DataQualityValidator:
    """Validates cleaned records before processed storage."""

    REQUIRED_FIELDS = {
        "nifty_50": ("date", "close"),
        "india_vix": ("date", "level"),
        "repo_rate": ("change_date", "rate"),
        "cpi_india": ("date",),
        "usd_inr": ("date", "rate"),
        "sp500": ("date", "close"),
        "cboe_vix": ("date", "level"),
        "crude_oil": ("date", "price"),
    }

    def validate(
        self,
        records: list[dict[str, Any]],
        *,
        dataset_id: str,
    ) -> DataQualityReport:
        """Validate required fields and duplicate records."""
        required_fields = self.REQUIRED_FIELDS.get(dataset_id, ("date",))

        missing_counts = {field: 0 for field in required_fields}
        valid_records = 0
        seen_records: set[tuple[tuple[str, str], ...]] = set()
        duplicate_records = 0

        for record in records:
            missing = False

            for field in required_fields:
                if record.get(field) is None:
                    missing_counts[field] += 1
                    missing = True

            if not missing:
                valid_records += 1

            fingerprint = tuple(
                sorted((str(key), str(value)) for key, value in record.items())
            )

            if fingerprint in seen_records:
                duplicate_records += 1
            else:
                seen_records.add(fingerprint)

        invalid_records = len(records) - valid_records

        errors: list[str] = []

        if not records:
            errors.append("Dataset contains no records.")

        if invalid_records:
            errors.append(
                f"{invalid_records} record(s) are missing required fields."
            )

        if duplicate_records:
            errors.append(
                f"{duplicate_records} duplicate record(s) detected."
            )

        return DataQualityReport(
            dataset_id=dataset_id,
            total_records=len(records),
            valid_records=valid_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_records,
            missing_required_fields=missing_counts,
            errors=tuple(errors),
        )