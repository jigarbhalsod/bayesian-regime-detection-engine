"""Source-specific parsing utilities for Phase 5 Group B."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


class DatasetParser:
    """Parses loaded source content into consistent record dictionaries."""

    def parse(
        self,
        *,
        dataset_id: str,
        raw_content: str | bytes | list[dict[str, Any]],
        source_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Parse loaded source content and dispatch dataset-specific parsing."""
        records = self._load_records(
            raw_content=raw_content,
            source_path=source_path,
        )

        parser = getattr(
            self,
            f"_parse_{dataset_id}",
            self._parse_generic,
        )

        return parser(records)

    @staticmethod
    def _load_records(
        *,
        raw_content: str | bytes | list[dict[str, Any]],
        source_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Convert JSON or CSV source content into record dictionaries."""
        if isinstance(raw_content, list):
            return [
                dict(record)
                for record in raw_content
                if isinstance(record, dict)
            ]

        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8-sig")

        path = Path(source_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            parsed = json.loads(raw_content)

            if isinstance(parsed, dict):
                for key in ("data", "records", "results"):
                    if isinstance(parsed.get(key), list):
                        parsed = parsed[key]
                        break

            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected a JSON list of records in {path}"
                )

            return [
                dict(record)
                for record in parsed
                if isinstance(record, dict)
            ]

        if suffix == ".csv":
            return [
                dict(record)
                for record in csv.DictReader(
                    io.StringIO(raw_content)
                )
            ]

        raise ValueError(
            f"Unsupported source format: {suffix or 'unknown'}"
        )

    @staticmethod
    def _parse_generic(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return generic records without source-specific assumptions."""
        return [dict(record) for record in records]

    @staticmethod
    def _parse_nifty_50(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse NIFTY 50 market records."""
        return [
            {
                "date": record.get("Date") or record.get("date"),
                "open": record.get("Open") or record.get("open"),
                "high": record.get("High") or record.get("high"),
                "low": record.get("Low") or record.get("low"),
                "close": (
                    record.get("Close")
                    or record.get("close")
                    or record.get("Index Value")
                ),
                "volume": record.get("Volume") or record.get("volume"),
            }
            for record in records
        ]

    @staticmethod
    def _parse_india_vix(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse India VIX records."""
        return [
            {
                "date": record.get("Date") or record.get("date"),
                "level": (
                    record.get("Close")
                    or record.get("close")
                    or record.get("Index Value")
                    or record.get("level")
                ),
            }
            for record in records
        ]

    @staticmethod
    def _parse_sp500(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse S&P 500 records."""
        return [
            {
                "date": record.get("Date") or record.get("date"),
                "close": record.get("Close") or record.get("close"),
            }
            for record in records
        ]

    @staticmethod
    def _parse_cboe_vix(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse CBOE VIX records."""
        return [
            {
                "date": (
                    record.get("DATE")
                    or record.get("Date")
                    or record.get("date")
                ),
                "level": (
                    record.get("VALUE")
                    or record.get("Close")
                    or record.get("close")
                    or record.get("level")
                ),
            }
            for record in records
        ]

    @staticmethod
    def _parse_crude_oil(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse crude oil records."""
        return [
            {
                "date": record.get("Date") or record.get("date"),
                "price": (
                    record.get("Close")
                    or record.get("close")
                    or record.get("price")
                ),
            }
            for record in records
        ]