from abc import ABC, abstractmethod
from typing import Any

from .config import RegimeConfig
from .result import RegimeResult


class BaseRegimeDetector(ABC):
    """
    Abstract base class for all regime detectors.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique detector name.
        """
        raise NotImplementedError

    @abstractmethod
    def detect(
        self,
        records: list[dict[str, Any]],
    ) -> RegimeResult:
        """
        Detect market regimes from input records.
        """
        raise NotImplementedError