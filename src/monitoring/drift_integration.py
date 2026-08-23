from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.monitoring.drift import (
    FeatureDriftMonitor,
    PredictionDriftMonitor,
)
from src.monitoring.drift_severity import (
    DriftSeverityEvaluator,
    DriftSeverityResult,
)
from src.monitoring.models import MonitoringResult


@dataclass(frozen=True)
class DriftMonitoringReport:
    """
    Combined report for feature and prediction drift monitoring.
    """

    feature_result: MonitoringResult
    prediction_result: MonitoringResult
    feature_severity: DriftSeverityResult
    prediction_severity: DriftSeverityResult


class DriftMonitoringIntegration:
    """
    Execute feature and prediction drift monitoring together.

    The provided MonitoringConfig objects are passed to their
    respective monitors. A single DriftSeverityEvaluator is used
    to classify both drift scores.
    """

    def __init__(
        self,
        feature_monitor: FeatureDriftMonitor,
        prediction_monitor: PredictionDriftMonitor,
        severity_evaluator: DriftSeverityEvaluator,
    ) -> None:
        if not isinstance(
            feature_monitor,
            FeatureDriftMonitor,
        ):
            raise TypeError(
                "feature_monitor must be a FeatureDriftMonitor."
            )

        if not isinstance(
            prediction_monitor,
            PredictionDriftMonitor,
        ):
            raise TypeError(
                "prediction_monitor must be a PredictionDriftMonitor."
            )

        if not isinstance(
            severity_evaluator,
            DriftSeverityEvaluator,
        ):
            raise TypeError(
                "severity_evaluator must be a DriftSeverityEvaluator."
            )

        self._feature_monitor = feature_monitor
        self._prediction_monitor = prediction_monitor
        self._severity_evaluator = severity_evaluator

    def execute(
        self,
        feature_data: Any,
        predictions: Any,
    ) -> DriftMonitoringReport:
        if not isinstance(feature_data, pd.DataFrame):
            raise TypeError(
                "feature_data must be a pandas DataFrame."
            )

        feature_result = self._feature_monitor.execute(
            feature_data
        )

        prediction_result = self._prediction_monitor.execute(
            predictions
        )

        feature_score = self._get_metric_value(
            feature_result,
            "max_feature_drift",
        )

        prediction_score = self._get_metric_value(
            prediction_result,
            "prediction_drift_score",
        )

        feature_severity = self._severity_evaluator.evaluate(
            feature_score
        )

        prediction_severity = self._severity_evaluator.evaluate(
            prediction_score
        )

        return DriftMonitoringReport(
            feature_result=feature_result,
            prediction_result=prediction_result,
            feature_severity=feature_severity,
            prediction_severity=prediction_severity,
        )

    @staticmethod
    def _get_metric_value(
        result: MonitoringResult,
        metric_name: str,
    ) -> float:
        for metric in result.metrics:
            if metric.name == metric_name:
                return float(metric.value)

        raise ValueError(
            f"Required metric not found: {metric_name}"
        )