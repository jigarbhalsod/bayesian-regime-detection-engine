import pandas as pd
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.drift import (
    FeatureDriftMonitor,
    PredictionDriftMonitor,
)
from src.monitoring.drift_integration import (
    DriftMonitoringIntegration,
    DriftMonitoringReport,
)
from src.monitoring.drift_severity import (
    DriftSeverity,
    DriftSeverityEvaluator,
)


def create_integration():
    reference_data = pd.DataFrame(
        {
            "price": [100.0, 100.0],
            "volume": [1000.0, 1000.0],
        }
    )

    feature_monitor = FeatureDriftMonitor(
        MonitoringConfig(
            name="feature_drift",
            metadata={
                "reference_data": reference_data,
            },
        )
    )

    prediction_monitor = PredictionDriftMonitor(
        MonitoringConfig(
            name="prediction_drift",
            metadata={
                "reference_predictions": [
                    0.2,
                    0.2,
                ],
            },
        )
    )

    severity_evaluator = DriftSeverityEvaluator(
        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
        }
    )

    return DriftMonitoringIntegration(
        feature_monitor=feature_monitor,
        prediction_monitor=prediction_monitor,
        severity_evaluator=severity_evaluator,
    )


def test_integration_returns_combined_report():
    integration = create_integration()

    feature_data = pd.DataFrame(
        {
            "price": [100.0, 100.0],
            "volume": [1000.0, 1000.0],
        }
    )

    predictions = [0.2, 0.2]

    report = integration.execute(
        feature_data,
        predictions,
    )

    assert isinstance(
        report,
        DriftMonitoringReport,
    )


def test_integration_executes_both_monitors():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.2, 0.2],
    )

    assert report.feature_result.monitor_name == (
        "feature_drift"
    )

    assert report.prediction_result.monitor_name == (
        "prediction_drift"
    )


def test_integration_classifies_low_drift():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [105.0, 105.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.21, 0.21],
    )

    assert (
        report.feature_severity.severity
        == DriftSeverity.LOW
    )

    assert (
        report.prediction_severity.severity
        == DriftSeverity.LOW
    )


def test_integration_classifies_medium_drift():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [115.0, 115.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.24, 0.24],
    )

    assert (
        report.feature_severity.severity
        == DriftSeverity.MEDIUM
    )

    assert (
        report.prediction_severity.severity
        == DriftSeverity.MEDIUM
    )


def test_integration_classifies_high_drift():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [135.0, 135.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.28, 0.28],
    )

    assert (
        report.feature_severity.severity
        == DriftSeverity.HIGH
    )

    assert (
        report.prediction_severity.severity
        == DriftSeverity.HIGH
    )


def test_integration_classifies_critical_drift():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [160.0, 160.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.35, 0.35],
    )

    assert (
        report.feature_severity.severity
        == DriftSeverity.CRITICAL
    )

    assert (
        report.prediction_severity.severity
        == DriftSeverity.CRITICAL
    )


def test_integration_rejects_invalid_feature_monitor():
    prediction_monitor = PredictionDriftMonitor(
        MonitoringConfig(
            name="prediction_drift",
            metadata={
                "reference_predictions": [0.2],
            },
        )
    )

    evaluator = DriftSeverityEvaluator(
        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
        }
    )

    with pytest.raises(
        TypeError,
        match="feature_monitor",
    ):
        DriftMonitoringIntegration(
            feature_monitor="invalid",
            prediction_monitor=prediction_monitor,
            severity_evaluator=evaluator,
        )


def test_integration_rejects_invalid_prediction_monitor():
    feature_monitor = FeatureDriftMonitor(
        MonitoringConfig(
            name="feature_drift",
            metadata={
                "reference_data": pd.DataFrame(
                    {"price": [100.0]}
                ),
            },
        )
    )

    evaluator = DriftSeverityEvaluator(
        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
        }
    )

    with pytest.raises(
        TypeError,
        match="prediction_monitor",
    ):
        DriftMonitoringIntegration(
            feature_monitor=feature_monitor,
            prediction_monitor="invalid",
            severity_evaluator=evaluator,
        )


def test_integration_rejects_invalid_severity_evaluator():
    feature_monitor = FeatureDriftMonitor(
        MonitoringConfig(
            name="feature_drift",
            metadata={
                "reference_data": pd.DataFrame(
                    {"price": [100.0]}
                ),
            },
        )
    )

    prediction_monitor = PredictionDriftMonitor(
        MonitoringConfig(
            name="prediction_drift",
            metadata={
                "reference_predictions": [0.2],
            },
        )
    )

    with pytest.raises(
        TypeError,
        match="severity_evaluator",
    ):
        DriftMonitoringIntegration(
            feature_monitor=feature_monitor,
            prediction_monitor=prediction_monitor,
            severity_evaluator="invalid",
        )


def test_integration_rejects_non_dataframe_feature_data():
    integration = create_integration()

    with pytest.raises(
        TypeError,
        match="feature_data",
    ):
        integration.execute(
            {"price": [100.0]},
            [0.2],
        )


def test_integration_propagates_invalid_predictions():
    integration = create_integration()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        integration.execute(
            pd.DataFrame(
                {
                    "price": [100.0],
                    "volume": [1000.0],
                }
            ),
            [],
        )


def test_integration_includes_severity_scores():
    integration = create_integration()

    report = integration.execute(
        pd.DataFrame(
            {
                "price": [150.0, 150.0],
                "volume": [1000.0, 1000.0],
            }
        ),
        [0.3, 0.3],
    )

    assert report.feature_severity.score == 0.5
    assert report.prediction_severity.score == pytest.approx(
        0.5
    )