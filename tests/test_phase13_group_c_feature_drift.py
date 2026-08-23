import numpy as np
import pandas as pd
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.drift import FeatureDriftMonitor
from src.monitoring.models import MonitoringStatus


def create_monitor(
    reference_data,
    thresholds=None,
):
    return FeatureDriftMonitor(
        MonitoringConfig(
            name="feature_drift",
            thresholds=thresholds or {},
            metadata={
                "reference_data": reference_data,
            },
        )
    )


def get_metric(result, name):
    return next(
        metric
        for metric in result.metrics
        if metric.name == name
    )


def test_feature_drift_returns_ok_for_identical_data():
    reference = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(reference.copy())

    assert result.status == MonitoringStatus.OK
    assert (
        get_metric(
            result,
            "max_feature_drift",
        ).value
        == 0.0
    )


def test_feature_drift_detects_mean_shift():
    reference = pd.DataFrame(
        {
            "price": [100.0, 100.0, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [150.0, 150.0, 150.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    feature = result.metadata["feature_metrics"][0]

    assert feature["mean_difference"] == 50.0
    assert feature["drift_score"] == 0.5


def test_feature_drift_uses_reference_scale():
    reference = pd.DataFrame(
        {
            "price": [10.0, 10.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [15.0, 15.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    feature = result.metadata["feature_metrics"][0]

    assert feature["drift_score"] == 0.5


def test_feature_drift_handles_multiple_features():
    reference = pd.DataFrame(
        {
            "a": [10.0, 10.0],
            "b": [100.0, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "a": [15.0, 15.0],
            "b": [110.0, 110.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "feature_count",
        ).value
        == 2.0
    )

    assert len(
        result.metadata["feature_metrics"]
    ) == 2


def test_feature_drift_uses_max_and_mean_scores():
    reference = pd.DataFrame(
        {
            "a": [10.0, 10.0],
            "b": [100.0, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "a": [15.0, 15.0],
            "b": [110.0, 110.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "max_feature_drift",
        ).value
        == 0.5
    )

    assert (
        get_metric(
            result,
            "mean_feature_drift",
        ).value
        == 0.3
    )


def test_feature_drift_ignores_non_numeric_columns():
    reference = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "symbol": ["A", "B"],
        }
    )

    current = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "symbol": ["C", "D"],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "feature_count",
        ).value
        == 1.0
    )


def test_feature_drift_ignores_non_common_columns():
    reference = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "reference_only": [1.0, 2.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "current_only": [5.0, 6.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "feature_count",
        ).value
        == 1.0
    )

    assert result.metadata["common_column_count"] == 1


def test_feature_drift_handles_missing_values():
    reference = pd.DataFrame(
        {
            "price": [100.0, np.nan, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [110.0, np.nan, 110.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    feature = result.metadata["feature_metrics"][0]

    assert feature["mean_difference"] == 10.0
    assert feature["drift_score"] == 0.1


def test_feature_drift_handles_infinite_values():
    reference = pd.DataFrame(
        {
            "price": [100.0, np.inf, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [110.0, -np.inf, 110.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    feature = result.metadata["feature_metrics"][0]

    assert feature["drift_score"] == 0.1


def test_feature_drift_warns_when_threshold_exceeded():
    reference = pd.DataFrame(
        {
            "price": [100.0, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [150.0, 150.0],
        }
    )

    monitor = create_monitor(
        reference,
        thresholds={
            "max_feature_drift": 0.4,
        },
    )

    result = monitor.execute(current)

    assert result.status == MonitoringStatus.WARNING


def test_feature_drift_uses_ok_at_exact_threshold():
    reference = pd.DataFrame(
        {
            "price": [100.0, 100.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [150.0, 150.0],
        }
    )

    monitor = create_monitor(
        reference,
        thresholds={
            "max_feature_drift": 0.5,
        },
    )

    result = monitor.execute(current)

    assert result.status == MonitoringStatus.OK


def test_feature_drift_rejects_non_dataframe_data():
    reference = pd.DataFrame(
        {
            "price": [100.0],
        }
    )

    monitor = create_monitor(reference)

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        monitor.execute(
            {"price": [110.0]}
        )


def test_feature_drift_rejects_invalid_reference_data():
    config = MonitoringConfig(
        name="feature_drift",
        metadata={
            "reference_data": [1, 2, 3],
        },
    )

    monitor = FeatureDriftMonitor(config)

    with pytest.raises(
        TypeError,
        match="reference_data",
    ):
        monitor.execute(
            pd.DataFrame(
                {"price": [100.0]}
            )
        )


def test_feature_drift_handles_no_numeric_features():
    reference = pd.DataFrame(
        {
            "symbol": ["A", "B"],
        }
    )

    current = pd.DataFrame(
        {
            "symbol": ["C", "D"],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "feature_count",
        ).value
        == 0.0
    )

    assert (
        get_metric(
            result,
            "max_feature_drift",
        ).value
        == 0.0
    )


def test_feature_drift_includes_dataset_metadata():
    reference = pd.DataFrame(
        {
            "price": [100.0, 101.0],
        }
    )

    current = pd.DataFrame(
        {
            "price": [102.0, 103.0, 104.0],
        }
    )

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert result.metadata["reference_row_count"] == 2
    assert result.metadata["current_row_count"] == 3