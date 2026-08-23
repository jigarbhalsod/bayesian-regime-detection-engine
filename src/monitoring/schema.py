from __future__ import annotations

from typing import Any

import pandas as pd

from src.monitoring.base import BaseMonitor
from src.monitoring.models import (
    MetricResult,
    MonitoringResult,
    MonitoringStatus,
)


class SchemaMonitor(BaseMonitor):
    """
    Monitor a pandas DataFrame against an expected schema.

    Expected columns and dtypes are provided through
    MonitoringConfig.metadata:

        {
            "expected_schema": {
                "price": "float64",
                "volume": "int64",
            }
        }
    """

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        expected_schema = self.config.metadata.get(
            "expected_schema",
            {},
        )

        if not isinstance(expected_schema, dict):
            raise TypeError(
                "expected_schema metadata must be a dictionary."
            )

        expected_columns = set(expected_schema.keys())
        actual_columns = set(data.columns)

        missing_columns = sorted(
            expected_columns - actual_columns
        )

        unexpected_columns = sorted(
            actual_columns - expected_columns
        )

        dtype_mismatches = []

        for column, expected_dtype in expected_schema.items():
            if column not in data.columns:
                continue

            actual_dtype = str(data[column].dtype)

            if actual_dtype != str(expected_dtype):
                dtype_mismatches.append(
                    {
                        "column": column,
                        "expected": str(expected_dtype),
                        "actual": actual_dtype,
                    }
                )

        metrics = (
            MetricResult(
                name="missing_column_count",
                value=float(len(missing_columns)),
                threshold=self.config.get_threshold(
                    "missing_column_count"
                ),
            ),
            MetricResult(
                name="unexpected_column_count",
                value=float(len(unexpected_columns)),
                threshold=self.config.get_threshold(
                    "unexpected_column_count"
                ),
            ),
            MetricResult(
                name="dtype_mismatch_count",
                value=float(len(dtype_mismatches)),
                threshold=self.config.get_threshold(
                    "dtype_mismatch_count"
                ),
            ),
        )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "expected_columns": sorted(expected_columns),
                "actual_columns": sorted(actual_columns),
                "missing_columns": missing_columns,
                "unexpected_columns": unexpected_columns,
                "dtype_mismatches": dtype_mismatches,
            },
        )

    @staticmethod
    def _determine_status(
        metrics: tuple[MetricResult, ...],
    ) -> MonitoringStatus:
        """
        Determine status from configured thresholds.
        """

        for metric in metrics:
            if (
                metric.threshold is not None
                and metric.value > metric.threshold
            ):
                return MonitoringStatus.WARNING

        return MonitoringStatus.OK