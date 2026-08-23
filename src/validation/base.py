from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.validation.config import ValidationConfig
from src.validation.result import ValidationResult


class BaseValidator(ABC):
    """
    Abstract base class for all validation components.

    Concrete validators must implement the validate() method and
    return a standardized ValidationResult.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        if config is None:
            config = ValidationConfig()

        if not isinstance(config, ValidationConfig):
            raise TypeError("config must be a ValidationConfig instance")

        self.config = config

    @property
    def name(self) -> str:
        """Return the validator name."""
        return self.config.name

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input consumed by the concrete validator.

        Returns:
            A standardized ValidationResult.
        """
        raise NotImplementedError