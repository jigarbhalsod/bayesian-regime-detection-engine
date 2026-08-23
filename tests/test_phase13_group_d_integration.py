import pytest

from src.monitoring.classification_performance import (
    ClassificationPerformanceMonitor,
)
from src.monitoring.config import MonitoringConfig
from src.monitoring.models import (
    MonitoringStatus,
)
from src.monitoring.performance_degradation import (
    PerformanceDegradationMonitor,
)
from src.monitoring.performance_integration import (
    PerformanceMonitoringIntegration,
)
from src.monitoring.regression_performance import (
    RegressionPerformanceMonitor,
)


def create_classification_monitor(
    thresholds=None,
):
    return ClassificationPerformanceMonitor(
        MonitoringConfig(
            name="classification_performance",
            thresholds=thresholds or {},
        )
    )


def create_regression_monitor(
    thresholds=None,
):
    return RegressionPerformanceMonitor(
        MonitoringConfig(
            name="regression_performance",
            thresholds=thresholds or {},
        )
    )


def create_degradation_monitor(
    thresholds=None,
):
    return PerformanceDegradationMonitor(
        MonitoringConfig(
            name="performance_degradation",
            thresholds=thresholds or {},
        )
    )


def test_integration_executes_classification_monitor():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor()
    )

    result = integration.execute(
        {
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 1, 0, 0],
        }
    )

    assert result["performance"].status == MonitoringStatus.OK
    assert result["degradation"] is None
    assert result["status"] == MonitoringStatus.OK


def test_integration_executes_regression_monitor():
    integration = PerformanceMonitoringIntegration(
        create_regression_monitor()
    )

    result = integration.execute(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 2.0],
        }
    )

    assert result["performance"].status == MonitoringStatus.OK
    assert result["degradation"] is None


def test_integration_executes_degradation_when_reference_provided():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor(),
        create_degradation_monitor(
            thresholds={
                "accuracy": 0.05,
            }
        ),
    )

    result = integration.execute(
        data={
            "y_true": [0, 1, 1, 1, 0],
            "y_pred": [0, 0, 0, 1, 0],
        },
        reference_metrics={
            "accuracy": 0.90,
            "precision": 0.90,
            "recall": 0.90,
            "f1_score": 0.90,
        },
    )

    assert result["degradation"] is not None
    assert result["status"] == MonitoringStatus.WARNING


def test_integration_returns_combined_ok_status():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor(),
        create_degradation_monitor(
            thresholds={
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
            }
        ),
    )

    result = integration.execute(
        data={
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 1, 0, 0],
        },
        reference_metrics={
            "accuracy": 0.75,
            "precision": 1.0,
            "recall": 0.50,
            "f1_score": 2 / 3,
        },
    )

    assert result["status"] == MonitoringStatus.OK


def test_integration_combines_performance_warning():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor(
            thresholds={
                "accuracy": 0.95,
            }
        ),
        create_degradation_monitor(),
    )

    result = integration.execute(
        data={
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 1, 0, 0],
        },
        reference_metrics={
            "accuracy": 0.75,
            "precision": 1.0,
            "recall": 0.50,
            "f1_score": 2 / 3,
        },
    )

    assert (
        result["performance"].status
        == MonitoringStatus.WARNING
    )
    assert result["status"] == MonitoringStatus.WARNING


def test_integration_combines_degradation_warning():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor(),
        create_degradation_monitor(
            thresholds={
                "accuracy": 0.05,
            }
        ),
    )

    result = integration.execute(
        data={
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 0, 0, 0],
        },
        reference_metrics={
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        },
    )

    assert (
        result["degradation"].status
        == MonitoringStatus.WARNING
    )
    assert result["status"] == MonitoringStatus.WARNING


def test_integration_rejects_invalid_performance_monitor():
    with pytest.raises(TypeError):
        PerformanceMonitoringIntegration(
            performance_monitor="invalid"
        )


def test_integration_rejects_invalid_degradation_monitor():
    with pytest.raises(TypeError):
        PerformanceMonitoringIntegration(
            performance_monitor=create_classification_monitor(),
            degradation_monitor="invalid",
        )


def test_integration_rejects_reference_without_degradation_monitor():
    integration = PerformanceMonitoringIntegration(
        create_classification_monitor()
    )

    with pytest.raises(ValueError):
        integration.execute(
            data={
                "y_true": [0, 1],
                "y_pred": [0, 1],
            },
            reference_metrics={
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
            },
        )


def test_integration_propagates_invalid_performance_data():
    integration = PerformanceMonitoringIntegration(
        create_regression_monitor()
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        integration.execute(
            {
                "y_true": [1.0, 2.0],
                "y_pred": [1.0],
            }
        )


def test_integration_propagates_invalid_reference_metrics():
    integration = PerformanceMonitoringIntegration(
        create_regression_monitor(),
        create_degradation_monitor(),
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        integration.execute(
            data={
                "y_true": [1.0, 2.0],
                "y_pred": [1.0, 2.0],
            },
            reference_metrics={
                "invalid": 1.0,
            },
        )


def test_integration_extracts_metrics_correctly():
    integration = PerformanceMonitoringIntegration(
        create_regression_monitor()
    )

    result = integration.execute(
        {
            "y_true": [1.0, 2.0],
            "y_pred": [1.0, 3.0],
        }
    )

    metrics = integration._extract_metrics(
        result["performance"]
    )

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["mse"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(
        0.5 ** 0.5
    )
    assert "r2_score" in metrics