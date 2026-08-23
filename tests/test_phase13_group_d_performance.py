import pytest

from src.monitoring.models import (
    MetricResult,
    MonitoringStatus,
)
from src.monitoring.performance import (
    ModelPerformanceMonitor,
)
from src.monitoring.config import MonitoringConfig


class DummyPerformanceMonitor(
    ModelPerformanceMonitor):
    def calculate_metrics(
        self,
        y_true,
        y_pred,
    ):
        return (
            MetricResult(
                name="accuracy",
                value=0.8,
                threshold=self.config.get_threshold(
                    "accuracy"
                ),
            ),
        )


class InvalidReturnPerformanceMonitor(
    ModelPerformanceMonitor):
    def calculate_metrics(
        self,
        y_true,
        y_pred,
    ):
        return []


class InvalidMetricPerformanceMonitor(
    ModelPerformanceMonitor):
    def calculate_metrics(
        self,
        y_true,
        y_pred,
    ):
        return (
            "invalid",
        )


def create_monitor(
    thresholds=None,
):
    return DummyPerformanceMonitor(
        MonitoringConfig(
            name="performance",
            thresholds=thresholds or {},
        )
    )


def test_performance_monitor_executes_successfully():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1, 0, 1],
            "y_pred": [1, 0, 0],
        }
    )

    assert result.monitor_name == "performance"
    assert result.status == MonitoringStatus.OK


def test_performance_monitor_includes_sample_count():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 0, 1],
        }
    )

    assert result.metadata["sample_count"] == 4


def test_performance_monitor_returns_metrics():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
        }
    )

    assert len(result.metrics) == 1
    assert result.metrics[0].name == "accuracy"
    assert result.metrics[0].value == 0.8


def test_performance_monitor_warns_when_higher_is_better_metric_falls_below_threshold():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_performance_monitor_uses_ok_at_exact_higher_is_better_threshold():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.8,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
        }
    )

    assert result.status == MonitoringStatus.OK


class LossPerformanceMonitor(
    ModelPerformanceMonitor):
    def calculate_metrics(
        self,
        y_true,
        y_pred,
    ):
        return (
            MetricResult(
                name="model_loss",
                value=0.8,
                threshold=self.config.get_threshold(
                    "model_loss"
                ),
            ),
        )


def test_performance_monitor_warns_when_lower_is_better_metric_exceeds_threshold():
    monitor = LossPerformanceMonitor(
        MonitoringConfig(
            name="loss_performance",
            thresholds={
                "model_loss": 0.5,
            },
        )
    )

    result = monitor.execute(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_performance_monitor_uses_ok_at_exact_lower_is_better_threshold():
    monitor = LossPerformanceMonitor(
        MonitoringConfig(
            name="loss_performance",
            thresholds={
                "model_loss": 0.8,
            },
        )
    )

    result = monitor.execute(
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
        }
    )

    assert result.status == MonitoringStatus.OK


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
        "invalid",
        [1, 2],
    ],
)
def test_performance_monitor_rejects_non_dictionary_data(
    invalid_data,
):
    monitor = create_monitor()

    with pytest.raises(TypeError):
        monitor.execute(invalid_data)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {},
        {"y_true": [1, 0]},
        {"y_pred": [1, 0]},
        {
            "y_true": [1, 0],
            "y_pred": [1, 0],
            "extra": [],
        },
    ],
)
def test_performance_monitor_rejects_missing_or_extra_keys(
    invalid_data,
):
    monitor = create_monitor()

    with pytest.raises(ValueError):
        monitor.execute(invalid_data)


def test_performance_monitor_rejects_length_mismatch():
    monitor = create_monitor()

    with pytest.raises(ValueError):
        monitor.execute(
            {
                "y_true": [1, 0],
                "y_pred": [1],
            }
        )


@pytest.mark.parametrize(
    "invalid_values",
    [
        [],
        [[1, 0]],
        "invalid",
        {"value": 1},
    ],
)
def test_performance_monitor_rejects_invalid_arrays(
    invalid_values,
):
    monitor = create_monitor()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(
            {
                "y_true": invalid_values,
                "y_pred": [1],
            }
        )


def test_performance_monitor_rejects_invalid_metric_return_type():
    monitor = InvalidReturnPerformanceMonitor(
        MonitoringConfig(
            name="invalid_return",
        )
    )

    with pytest.raises(
        TypeError,
        match="tuple",
    ):
        monitor.execute(
            {
                "y_true": [1],
                "y_pred": [1],
            }
        )


def test_performance_monitor_rejects_invalid_metric_instances():
    monitor = InvalidMetricPerformanceMonitor(
        MonitoringConfig(
            name="invalid_metric",
        )
    )

    with pytest.raises(
        TypeError,
        match="MetricResult",
    ):
        monitor.execute(
            {
                "y_true": [1],
                "y_pred": [1],
            }
        )