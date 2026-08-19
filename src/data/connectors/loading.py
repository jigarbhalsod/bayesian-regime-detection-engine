"""Raw data loading utilities for Phase 5 Group B."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path


class RawDataLoader:
    """Loads source-faithful raw datasets for downstream processing."""

    SUPPORTED_SUFFIXES = {".csv", ".json"}

    def load(self, path: str | Path) -> list[dict[str, str]]:
        """Load a supported raw file into a list of records."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file does not exist: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported raw data format: {suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_SUFFIXES))}"
            )

        content = file_path.read_text(encoding="utf-8-sig")

        if suffix == ".csv":
            return self._load_csv(content)

        return self._load_json(content)

    @staticmethod
    def _load_csv(content: str) -> list[dict[str, str]]:
        """Load CSV content while preserving source column names."""
        if not content.strip():
            return []

        return list(csv.DictReader(StringIO(content)))

    @staticmethod
    def _load_json(content: str) -> list[dict[str, str]]:
        """Load JSON records into a common record-list representation."""
        if not content.strip():
            return []

        payload = json.loads(content)

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("data", "records", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError(
            "JSON raw data must contain a list of records or a "
            "'data', 'records', or 'results' list."
        )