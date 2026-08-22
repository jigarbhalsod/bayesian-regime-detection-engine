from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

from .base import BaseFinancialAnalyzer


class AnalyzerRegistry:
    """
    Registry for managing financial analyzers.
    """

    def __init__(self) -> None:
        self._analyzers: OrderedDict[str, BaseFinancialAnalyzer] = OrderedDict()

    def register(self, analyzer: BaseFinancialAnalyzer) -> None:
        """
        Register a financial analyzer.

        Raises:
            ValueError: If an analyzer with the same name is already registered.
        """
        if analyzer.name in self._analyzers:
            raise ValueError(
                f"Analyzer '{analyzer.name}' is already registered."
            )

        self._analyzers[analyzer.name] = analyzer

    def get(self, name: str) -> BaseFinancialAnalyzer:
        """
        Resolve a registered analyzer by name.

        Raises:
            KeyError: If the analyzer is not registered.
        """
        if name not in self._analyzers:
            raise KeyError(
                f"Analyzer '{name}' is not registered."
            )

        return self._analyzers[name]

    def names(self) -> list[str]:
        """
        Return registered analyzer names in registration order.
        """
        return list(self._analyzers.keys())

    def analyzers(self) -> list[BaseFinancialAnalyzer]:
        """
        Return registered analyzers in registration order.
        """
        return list(self._analyzers.values())

    def __iter__(self) -> Iterator[BaseFinancialAnalyzer]:
        """
        Iterate over analyzers in registration order.
        """
        return iter(self._analyzers.values())

    def __len__(self) -> int:
        """
        Return the number of registered analyzers.
        """
        return len(self._analyzers)