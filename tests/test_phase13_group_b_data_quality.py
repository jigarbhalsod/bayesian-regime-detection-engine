import numpy as np
import pandas as pd
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.data_quality import DataQualityMonitor
from src.monitoring.models import MonitoringStatus


def create_monitor(
    thresholds=None,
):
    return DataQualityMonitor(
        MonitoringConfig(
            name="data_quality",
            thresholds=thresholds or {},
        )
    )


def get_metric(result, name):
    return next(
        metric
        for metric in result.metrics
        if metric.name == name
    )


def test_data_quality_monitor_returns_ok_for_clean_data():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK
    assert result.monitor_name == "data_quality"


def test_data_quality_monitor_calculates_missing_ratio():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "a": [1.0, np.nan],
            "b": [np.nan, 4.0],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(result, "missing_ratio")

    assert metric.value == 0.5


def test_data_quality_monitor_counts_duplicates():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [10, 10, 20],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(result, "duplicate_count")

    assert metric.value == 1.0


def test_data_quality_monitor_counts_infinite_values():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf],
            "b": [10.0, 20.0, 30.0],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(result, "infinite_count")

    assert metric.value == 2.0


def test_data_quality_monitor_handles_non_numeric_data():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "symbol": ["NIFTY", "BANKNIFTY"],
            "regime": ["risk_on", "risk_off"],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(result, "infinite_count")

    assert metric.value == 0.0


def test_data_quality_monitor_rejects_non_dataframe():
    monitor = create_monitor()

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        monitor.execute(
            {"price": [100, 101]}
        )


def test_data_quality_monitor_warns_when_missing_threshold_exceeded():
    monitor = create_monitor(
        thresholds={"missing_ratio": 0.2}
    )

    data = pd.DataFrame(
        {
            "a": [1.0, np.nan],
            "b": [3.0, 4.0],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_data_quality_monitor_warns_when_duplicate_threshold_exceeded():
    monitor = create_monitor(
        thresholds={"duplicate_count": 0.0}
    )

    data = pd.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_data_quality_monitor_warns_when_infinite_threshold_exceeded():
    monitor = create_monitor(
        thresholds={"infinite_count": 0.0}
    )

    data = pd.DataFrame(
        {
            "a": [1.0, np.inf],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_data_quality_monitor_uses_ok_at_exact_threshold():
    monitor = create_monitor(
        thresholds={"duplicate_count": 1.0}
    )

    data = pd.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK


def test_data_quality_monitor_handles_empty_dataframe():
    monitor = create_monitor()

    data = pd.DataFrame()

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK
    assert result.metadata["row_count"] == 0
    assert result.metadata["column_count"] == 0
    assert result.metadata["total_cells"] == 0


def test_data_quality_monitor_includes_dataset_metadata():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [100, 101, 102],
            "volume": [10, 20, 30],
        }
    )

    result = monitor.execute(data)

    assert result.metadata["row_count"] == 3
    assert result.metadata["column_count"] == 2
    assert result.metadata["total_cells"] == 6