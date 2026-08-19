"""Shared data-layer configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CURATED_DATA_DIR = DATA_DIR / "curated"
METADATA_DIR = DATA_DIR / "metadata"
MANIFEST_DIR = METADATA_DIR / "manifests"
DEFAULT_DATA_SOURCE_CONFIG = CONFIG_DIR / "data_sources.json"
DEFAULT_START_DATE = "2010-01-01"
