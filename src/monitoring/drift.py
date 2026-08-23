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


class FeatureDriftMonitor(BaseMonitor):
    """
    Monitor distribution drift between reference and current
    numeric feature data.

    The reference dataset must be supplied through config metadata:

        {
            "reference_data": reference_dataframe
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

        reference_data = self.config.metadata.get(
            "reference_data"
        )

        if not isinstance(reference_data, pd.DataFrame):
            raise TypeError(
                "reference_data metadata must be a pandas DataFrame."
            )

        common_columns = sorted(
            set(reference_data.columns)
            & set(data.columns)
        )

        feature_metrics: list[dict[str, Any]] = []

        drift_scores: list[float] = []

        for column in common_columns:
            reference_series = reference_data[column]
            current_series = data[column]

            if not (
                pd.api.types.is_numeric_dtype(reference_series)
                and pd.api.types.is_numeric_dtype(current_series)
            ):
                continue

            reference_values = (
                reference_series.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            current_values = (
                current_series.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            if (
                len(reference_values) == 0
                or len(current_values) == 0
            ):
                continue

            reference_mean = float(
                np.mean(reference_values)
            )
            current_mean = float(
                np.mean(current_values)
            )

            reference_std = float(
                np.std(reference_values)
            )
            current_std = float(
                np.std(current_values)
            )

            mean_difference = abs(
                current_mean - reference_mean
            )

            std_difference = abs(
                current_std - reference_std
            )

            scale = max(
                abs(reference_mean),
                reference_std,
                1e-12,
            )

            drift_score = float(
                mean_difference / scale
            )

            drift_scores.append(drift_score)

            feature_metrics.append(
                {
                    "feature": column,
                    "reference_mean": reference_mean,
                    "current_mean": current_mean,
                    "reference_std": reference_std,
                    "current_std": current_std,
                    "mean_difference": float(
                        mean_difference
                    ),
                    "std_difference": float(
                        std_difference
                    ),
                    "drift_score": drift_score,
                }
            )

        if drift_scores:
            max_drift_score = float(max(drift_scores))
            mean_drift_score = float(
                np.mean(drift_scores)
            )
        else:
            max_drift_score = 0.0
            mean_drift_score = 0.0

        metrics = (
            MetricResult(
                name="feature_count",
                value=float(len(feature_metrics)),
                threshold=self.config.get_threshold(
                    "feature_count"
                ),
            ),
            MetricResult(
                name="max_feature_drift",
                value=max_drift_score,
                threshold=self.config.get_threshold(
                    "max_feature_drift"
                ),
            ),
            MetricResult(
                name="mean_feature_drift",
                value=mean_drift_score,
                threshold=self.config.get_threshold(
                    "mean_feature_drift"
                ),
            ),
        )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "feature_metrics": feature_metrics,
                "reference_row_count": int(
                    len(reference_data)
                ),
                "current_row_count": int(
                    len(data)
                ),
                "common_column_count": len(
                    common_columns
                ),
            },
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

class PredictionDriftMonitor(BaseMonitor):
    """
    Monitor drift between reference and current model predictions.

    Reference predictions must be supplied through config metadata:

        {
            "reference_predictions": [...]
        }
    """

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        reference_predictions = self.config.metadata.get(
            "reference_predictions"
        )

        reference_values = self._validate_predictions(
            reference_predictions,
            "reference_predictions",
        )

        current_values = self._validate_predictions(
            data,
            "data",
        )

        reference_mean = float(np.mean(reference_values))
        current_mean = float(np.mean(current_values))

        reference_std = float(np.std(reference_values))
        current_std = float(np.std(current_values))

        mean_difference = abs(
            current_mean - reference_mean
        )

        std_difference = abs(
            current_std - reference_std
        )

        scale = max(
            abs(reference_mean),
            reference_std,
            1e-12,
        )

        drift_score = float(
            mean_difference / scale
        )

        metrics = (
            MetricResult(
                name="prediction_drift_score",
                value=drift_score,
                threshold=self.config.get_threshold(
                    "prediction_drift_score"
                ),
            ),
            MetricResult(
                name="prediction_mean_difference",
                value=float(mean_difference),
                threshold=self.config.get_threshold(
                    "prediction_mean_difference"
                ),
            ),
            MetricResult(
                name="prediction_std_difference",
                value=float(std_difference),
                threshold=self.config.get_threshold(
                    "prediction_std_difference"
                ),
            ),
        )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "reference_count": int(
                    len(reference_values)
                ),
                "current_count": int(
                    len(current_values)
                ),
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "reference_std": reference_std,
                "current_std": current_std,
            },
        )

    @staticmethod
    def _validate_predictions(
        values: Any,
        name: str,
    ) -> np.ndarray:
        if isinstance(
            values,
            (str, bytes, dict),
        ):
            raise TypeError(
                f"{name} must be a numeric sequence."
            )

        try:
            array = np.asarray(
                values,
                dtype=float,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name} must be a numeric sequence."
            ) from exc

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if len(array) == 0:
            raise ValueError(
                f"{name} must not be empty."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} must contain only finite values."
            )

        return array

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