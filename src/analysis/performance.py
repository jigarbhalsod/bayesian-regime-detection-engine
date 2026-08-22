from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.analysis.base import BaseFinancialAnalyzer
from src.analysis.result import AnalysisResult


class PerformanceAnalyzer(BaseFinancialAnalyzer):
    """
    Calculates core and rolling performance metrics from return data.
    """

    @property
    def name(self) -> str:
        return "performance"

    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Analyze total, mean, and rolling returns.
        """

        output_records = [dict(record) for record in records]

        valid_returns: list[float] = []
        parsed_returns: list[float | None] = []

        # Parse and validate returns.
        for record in records:
            value = record.get(self.config.return_column)

            if value is None:
                parsed_returns.append(None)
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                parsed_returns.append(None)
                continue

            if not math.isfinite(numeric_value):
                parsed_returns.append(None)
                continue

            parsed_returns.append(numeric_value)
            valid_returns.append(numeric_value)

        metadata = {
            "valid_return_count": len(valid_returns),
            "return_column": self.config.return_column,
            "rolling_windows": list(self.config.rolling_windows),
        }

        # Preserve original records when no valid returns exist.
        if not valid_returns:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        # Add rolling compounded return columns only when enough
        # records exist to evaluate at least one configured window.
        for window in self.config.rolling_windows:
            if len(output_records) < window:
                continue

            column = f"analysis__rolling_return_{window}"

            for index in range(len(output_records)):
                if index < window - 1:
                    output_records[index][column] = None
                    continue

                window_values = parsed_returns[
                    index - window + 1:index + 1
                ]

                # Every value in the rolling window must be valid.
                if any(value is None for value in window_values):
                    output_records[index][column] = None
                    continue

                output_records[index][column] = float(
                    np.prod(
                        1.0 + np.asarray(
                            window_values,
                            dtype=float,
                        )
                    ) - 1.0
                )

        returns = np.asarray(valid_returns, dtype=float)

        total_return = float(
            np.prod(1.0 + returns) - 1.0
        )

        mean_return = float(
            np.mean(returns)
        )

        metrics = {
            "analysis__total_return": total_return,
            "analysis__mean_return": mean_return,
        }

        return AnalysisResult(
            records=output_records,
            metrics=metrics,
            metric_names=list(metrics.keys()),
            metadata=metadata,
        )