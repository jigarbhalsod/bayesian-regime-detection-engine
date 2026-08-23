from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.base import BaseMonitor
from src.monitoring.models import (
    MetricResult,
    MonitoringResult,
    MonitoringStatus,
)


class StatisticalDataMonitor(BaseMonitor):
    """
    Monitor statistical properties of numeric DataFrame columns.
    """

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        numeric_data = data.select_dtypes(
            include=[np.number]
        )

        column_statistics: dict[str, dict[str, float]] = {}

        for column in numeric_data.columns:
            series = numeric_data[column]

            non_missing = series.dropna()

            missing_ratio = (
                float(series.isna().mean())
                if len(series) > 0
                else 0.0
            )

            if non_missing.empty:
                column_statistics[column] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "q25": float("nan"),
                    "q50": float("nan"),
                    "q75": float("nan"),
                    "missing_ratio": missing_ratio,
                }
                continue

            column_statistics[column] = {
                "mean": float(non_missing.mean()),
                "median": float(non_missing.median()),
                "std": float(non_missing.std(ddof=0)),
                "min": float(non_missing.min()),
                "max": float(non_missing.max()),
                "q25": float(non_missing.quantile(0.25)),
                "q50": float(non_missing.quantile(0.50)),
                "q75": float(non_missing.quantile(0.75)),
                "missing_ratio": missing_ratio,
            }

        metrics = (
            MetricResult(
                name="numeric_column_count",
                value=float(len(numeric_data.columns)),
                threshold=self.config.get_threshold(
                    "numeric_column_count"
                ),
            ),
            MetricResult(
                name="max_missing_ratio",
                value=self._get_max_missing_ratio(
                    column_statistics
                ),
                threshold=self.config.get_threshold(
                    "max_missing_ratio"
                ),
            ),
        )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "column_statistics": column_statistics,
                "numeric_columns": list(
                    numeric_data.columns
                ),
            },
        )

    @staticmethod
    def _get_max_missing_ratio(
        column_statistics: dict[str, dict[str, float]],
    ) -> float:
        if not column_statistics:
            return 0.0

        return max(
            statistics["missing_ratio"]
            for statistics in column_statistics.values()
        )

    @staticmethod
    def _determine_status(
        metrics: tuple[MetricResult, ...],
    ) -> MonitoringStatus:
        for metric in metrics:
            if (
                metric.threshold is not None
                and metric.value > metric.threshold
            ):
                return MonitoringStatus.WARNING

        return MonitoringStatus.OK