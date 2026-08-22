from __future__ import annotations

from typing import Any, Sequence

from .base import BaseFinancialAnalyzer
from .config import AnalysisConfig
from .registry import AnalyzerRegistry
from .result import AnalysisResult


class FinancialAnalysisPipeline:
    """
    Executes registered financial analyzers in registration order.
    """

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        registry: AnalyzerRegistry | None = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self.registry = registry or AnalyzerRegistry()

    def register(self, analyzer: BaseFinancialAnalyzer) -> None:
        """
        Register an analyzer with the pipeline.
        """
        self.registry.register(analyzer)

    def run(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Run all registered analyzers sequentially.

        The records produced by each analyzer become the input
        records for the next analyzer.
        """
        current_records = [dict(record) for record in records]

        all_metrics: dict[str, Any] = {}
        all_metric_names: list[str] = []
        all_metadata: dict[str, Any] = {}

        for analyzer in self.registry:
            result = analyzer.analyze(current_records)

            current_records = result.records
            all_metrics.update(result.metrics)
            all_metric_names.extend(result.metric_names)

            all_metadata[analyzer.name] = result.metadata

        return AnalysisResult(
            records=current_records,
            metrics=all_metrics,
            metric_names=all_metric_names,
            metadata=all_metadata,
        )