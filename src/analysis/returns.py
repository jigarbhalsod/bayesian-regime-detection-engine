from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseFinancialAnalyzer
from .result import AnalysisResult


class ReturnsAnalyzer(BaseFinancialAnalyzer):
    """
    Analyze return characteristics from financial records.
    """

    @property
    def name(self) -> str:
        return "returns"

    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Calculate core and rolling return metrics.
        """
        output_records = [dict(record) for record in records]

        returns = self._extract_returns(output_records)

        self._add_rolling_returns(output_records)

        if len(returns) == 0:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata={
                    "valid_return_count": 0,
                },
            )

        mean_return = float(np.mean(returns))

        cumulative_return = float(
            np.prod(1.0 + returns) - 1.0
        )

        positive_return_ratio = float(
            np.mean(returns > 0.0)
        )

        negative_return_ratio = float(
            np.mean(returns < 0.0)
        )

        metrics = {
            "analysis__mean_return": mean_return,
            "analysis__cumulative_return": cumulative_return,
            "analysis__positive_return_ratio": positive_return_ratio,
            "analysis__negative_return_ratio": negative_return_ratio,
        }

        return AnalysisResult(
            records=output_records,
            metrics=metrics,
            metric_names=list(metrics.keys()),
            metadata={
                "valid_return_count": int(len(returns)),
                "return_column": self.config.return_column,
                "rolling_windows": list(self.config.rolling_windows),
            },
        )

    def _extract_returns(
        self,
        records: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract finite numeric return values.
        """
        values: list[float] = []

        for record in records:
            value = self._parse_return(
                record.get(self.config.return_column)
            )

            if value is not None:
                values.append(value)

        return np.asarray(values, dtype=float)

    def _add_rolling_returns(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """
        Add rolling cumulative returns to output records.

        A rolling value is only calculated when every return
        in the corresponding window is valid.
        """
        parsed_returns = [
            self._parse_return(
                record.get(self.config.return_column)
            )
            for record in records
        ]

        for window in self.config.rolling_windows:
            column_name = f"analysis__rolling_return_{window}"

            for index, record in enumerate(records):
                if index + 1 < window:
                    record[column_name] = None
                    continue

                window_returns = parsed_returns[
                    index - window + 1:index + 1
                ]

                if any(value is None for value in window_returns):
                    record[column_name] = None
                    continue

                record[column_name] = float(
                    np.prod(
                        1.0 + np.asarray(
                            window_returns,
                            dtype=float,
                        )
                    )
                    - 1.0
                )

    @staticmethod
    def _parse_return(value: Any) -> float | None:
        """
        Convert a value into a finite float return.
        """
        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(numeric_value):
            return None

        return numeric_value