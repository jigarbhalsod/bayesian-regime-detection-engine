from __future__ import annotations

from typing import Any

import numpy as np

from src.monitoring.models import MetricResult
from src.monitoring.performance import ModelPerformanceMonitor


class RegressionPerformanceMonitor(ModelPerformanceMonitor):
    """
    Monitor regression model performance.

    Calculates MAE, MSE, RMSE and R-squared.
    """

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[MetricResult, ...]:
        self._validate_numeric_arrays(
            y_true,
            y_pred,
        )

        errors = y_true.astype(float) - y_pred.astype(float)

        mae = float(
            np.mean(np.abs(errors))
        )

        mse = float(
            np.mean(np.square(errors))
        )

        rmse = float(
            np.sqrt(mse)
        )

        true_values = y_true.astype(float)

        total_sum_squares = float(
            np.sum(
                np.square(
                    true_values - np.mean(true_values)
                )
            )
        )

        residual_sum_squares = float(
            np.sum(np.square(errors))
        )

        if total_sum_squares == 0.0:
            r2_score = (
                1.0
                if residual_sum_squares == 0.0
                else 0.0
            )
        else:
            r2_score = float(
                1.0
                - (
                    residual_sum_squares
                    / total_sum_squares
                )
            )

        return (
            MetricResult(
                name="mae",
                value=mae,
                threshold=self.config.get_threshold(
                    "mae"
                ),
            ),
            MetricResult(
                name="mse",
                value=mse,
                threshold=self.config.get_threshold(
                    "mse"
                ),
            ),
            MetricResult(
                name="rmse",
                value=rmse,
                threshold=self.config.get_threshold(
                    "rmse"
                ),
            ),
            MetricResult(
                name="r2_score",
                value=r2_score,
                threshold=self.config.get_threshold(
                    "r2_score"
                ),
            ),
        )

    @staticmethod
    def _validate_numeric_arrays(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        for name, values in (
            ("y_true", y_true),
            ("y_pred", y_pred),
        ):
            if not np.issubdtype(
                values.dtype,
                np.number,
            ):
                raise TypeError(
                    f"{name} must contain numeric values."
                )

            numeric_values = values.astype(float)

            if not np.all(
                np.isfinite(numeric_values)
            ):
                raise ValueError(
                    f"{name} must contain only finite values."
                )

    def monitor(
        self,
        data: Any,
    ):
        return super().monitor(data)