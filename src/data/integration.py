"""Cross-dataset integration utilities for Phase 5 Group D."""

from __future__ import annotations

from typing import Any


class DatasetIntegrator:
    """Integrates multiple prepared datasets using their date field."""

    def integrate(
        self,
        *,
        datasets: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Combine datasets into date-aligned records."""

        integrated: dict[str, dict[str, Any]] = {}

        for dataset_id, records in datasets.items():
            for record in records:
                date = record.get("date")

                if date is None or date == "":
                    continue

                date_key = str(date)

                if date_key not in integrated:
                    integrated[date_key] = {"date": date_key}

                for key, value in record.items():
                    if key == "date":
                        continue

                    integrated_key = f"{dataset_id}__{key}"
                    integrated[date_key][integrated_key] = value

        return [
            integrated[date]
            for date in sorted(integrated)
        ]