import pytest

from src.monitoring.classification_performance import (
    ClassificationPerformanceMonitor,
)
from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringStatus


def create_monitor(
    thresholds=None,
):
    return ClassificationPerformanceMonitor(
        MonitoringConfig(
            name="classification_performance",
            thresholds=thresholds or {},
        )
    )


def get_metric(
    result,
    name,
):
    for metric in result.metrics:
        if metric.name == name:
            return metric.value

    raise AssertionError(
        f"Metric not found: {name}"
    )


def test_classification_monitor_perfect_predictions():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 0, 1],
        }
    )

    assert get_metric(result, "accuracy") == 1.0
    assert get_metric(result, "precision") == 1.0
    assert get_metric(result, "recall") == 1.0
    assert get_metric(result, "f1_score") == 1.0
    assert result.status == MonitoringStatus.OK


def test_classification_monitor_calculates_binary_metrics():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert get_metric(result, "accuracy") == 0.5
    assert get_metric(result, "precision") == 0.5
    assert get_metric(result, "recall") == 0.5
    assert get_metric(result, "f1_score") == 0.5


def test_classification_monitor_calculates_multiclass_metrics():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [0, 1, 2, 0, 1, 2],
            "y_pred": [0, 2, 2, 0, 1, 1],
        }
    )

    assert get_metric(
        result,
        "accuracy",
    ) == pytest.approx(4 / 6)

    assert get_metric(
        result,
        "precision",
    ) == pytest.approx(
        (1.0 + 0.5 + 0.5) / 3
    )

    assert get_metric(
        result,
        "recall",
    ) == pytest.approx(
        (1.0 + 0.5 + 0.5) / 3
    )


def test_classification_monitor_calculates_f1_score():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
        }
    )

    assert get_metric(
        result,
        "f1_score",
    ) == pytest.approx(
        (2 / 3 + 4 / 5) / 2
    )


def test_classification_monitor_handles_single_class():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1, 1, 1],
            "y_pred": [1, 1, 1],
        }
    )

    assert get_metric(result, "accuracy") == 1.0
    assert get_metric(result, "precision") == 1.0
    assert get_metric(result, "recall") == 1.0
    assert get_metric(result, "f1_score") == 1.0


def test_classification_monitor_handles_missing_predicted_class():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 0, 0, 0],
        }
    )

    assert get_metric(
        result,
        "accuracy",
    ) == 0.5

    assert get_metric(
        result,
        "precision",
    ) == pytest.approx(0.25)

    assert get_metric(
        result,
        "recall",
    ) == pytest.approx(0.5)


def test_classification_monitor_uses_accuracy_threshold():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_classification_monitor_uses_precision_threshold():
    monitor = create_monitor(
        thresholds={
            "precision": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_classification_monitor_uses_recall_threshold():
    monitor = create_monitor(
        thresholds={
            "recall": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_classification_monitor_uses_f1_threshold():
    monitor = create_monitor(
        thresholds={
            "f1_score": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_classification_monitor_uses_ok_at_exact_threshold():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )

    assert result.status == MonitoringStatus.OK


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
        {},
        {"y_true": [1]},
        {"y_pred": [1]},
    ],
)
def test_classification_monitor_rejects_invalid_data(
    invalid_data,
):
    monitor = create_monitor()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(invalid_data)