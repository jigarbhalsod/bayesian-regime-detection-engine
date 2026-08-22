from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .config import AnalysisConfig
from .result import AnalysisResult


class BaseFinancialAnalyzer(ABC):
    """
    Abstract base class for Phase 7 financial analyzers.
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name of the analyzer.
        """

    @abstractmethod
    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Analyze financial records and return an AnalysisResult.
        """