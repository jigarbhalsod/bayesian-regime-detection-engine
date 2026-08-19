"""Lightweight data access interface specified by Phase 4."""

from __future__ import annotations

from pathlib import Path

from src.data.config import CURATED_DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR


class DataAccess:
    """Read-side facade for approved datasets.

    Phase 5 Group A creates the interface boundary. Later Phase 5 groups will
    fill in processed and curated dataset readers once validation exists.
    """

    def __init__(
        self,
        processed_root: Path = PROCESSED_DATA_DIR,
        curated_root: Path = CURATED_DATA_DIR,
        metadata_root: Path = METADATA_DIR,
    ) -> None:
        self.processed_root = processed_root
        self.curated_root = curated_root
        self.metadata_root = metadata_root

    def get_dataset(self, dataset_id: str) -> Path:
        return self.processed_root / dataset_id

    def get_market_data(self, start_date: str, end_date: str) -> Path:
        _ = (start_date, end_date)
        return self.processed_root / "market"

    def get_indicator_data(self, indicator_id: str, start_date: str, end_date: str) -> Path:
        _ = (start_date, end_date)
        return self.processed_root / indicator_id

    def get_regime_dataset(self, as_of_date: str | None = None) -> Path:
        _ = as_of_date
        return self.curated_root / "regime_dataset"

    def get_dataset_metadata(self, dataset_id: str) -> Path:
        return self.metadata_root / "manifests" / dataset_id
