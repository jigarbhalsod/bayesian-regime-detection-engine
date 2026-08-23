from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationConfig:
    """
    Configuration shared by validation components.

    This class provides common validation settings and metadata that
    can be extended by concrete validation implementations.
    """

    name: str = "validation"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")

        if not self.name.strip():
            raise ValueError("name cannot be empty")

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")