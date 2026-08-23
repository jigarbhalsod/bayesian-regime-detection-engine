from __future__ import annotations

from enum import Enum


class LifecycleStage(str, Enum):
    """
    Lifecycle stages for a registered model version.
    """

    NONE = "none"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"