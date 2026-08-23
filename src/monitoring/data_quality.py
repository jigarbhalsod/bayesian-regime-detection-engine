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


class DataQualityMonitor(BaseMonitor):
    """
    Monitor tabular data for basic data quality issues.
    """

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        total_cells = int(data.shape[0] * data.shape[1])

        if total_cells == 0:
            missing_ratio = 0.0
        else:
            missing_ratio = float(
                data.isna().sum().sum() / total_cells
            )

        duplicate_count = int(data.duplicated().sum())

        numeric_data = data.select_dtypes(
            include=[np.number]
        )

        if numeric_data.empty:
            infinite_count = 0
        else:
            infinite_count = int(
                np.isinf(
                    numeric_data.to_numpy(
                        dtype=float,
                        copy=False,
                    )
                ).sum()
            )

        metrics = (
            MetricResult(
                name="missing_ratio",
                value=missing_ratio,
                threshold=self.config.get_threshold(
                    "missing_ratio"
                ),
            ),
            MetricResult(
                name="duplicate_count",
                value=float(duplicate_count),
                threshold=self.config.get_threshold(
                    "duplicate_count"
                ),
            ),
            MetricResult(
                name="infinite_count",
                value=float(infinite_count),
                threshold=self.config.get_threshold(
                    "infinite_count"
                ),
            ),
        )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "row_count": int(data.shape[0]),
                "column_count": int(data.shape[1]),
                "total_cells": total_cells,
            },
        )

    @staticmethod
    def _determine_status(
        metrics: tuple[MetricResult, ...],
    ) -> MonitoringStatus:
        """
        Determine status based on configured thresholds.
        """

        has_warning = False

        for metric in metrics:
            if (
                metric.threshold is not None
                and metric.value > metric.threshold
            ):
                has_warning = True

        if has_warning:
            return MonitoringStatus.WARNING

        return MonitoringStatus.OK