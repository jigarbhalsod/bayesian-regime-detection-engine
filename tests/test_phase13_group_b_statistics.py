import math

import numpy as np
import pandas as pd
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringStatus
from src.monitoring.statistics import (
    StatisticalDataMonitor,
)


def create_monitor(
    thresholds=None,
):
    return StatisticalDataMonitor(
        MonitoringConfig(
            name="statistics_monitor",
            thresholds=thresholds or {},
        )
    )


def get_metric(result, name):
    return next(
        metric
        for metric in result.metrics
        if metric.name == name
    )


def test_statistics_monitor_calculates_basic_statistics():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = monitor.execute(data)

    stats = result.metadata["column_statistics"][
        "price"
    ]

    assert stats["mean"] == 2.5
    assert stats["median"] == 2.5
    assert stats["std"] == pytest.approx(
        1.11803398875
    )
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


def test_statistics_monitor_calculates_quantiles():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = monitor.execute(data)

    stats = result.metadata["column_statistics"][
        "price"
    ]

    assert stats["q25"] == 1.75
    assert stats["q50"] == 2.5
    assert stats["q75"] == 3.25


def test_statistics_monitor_tracks_missing_ratio():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [1.0, np.nan, 3.0, np.nan],
        }
    )

    result = monitor.execute(data)

    stats = result.metadata["column_statistics"][
        "price"
    ]

    assert stats["missing_ratio"] == 0.5

    metric = get_metric(
        result,
        "max_missing_ratio",
    )

    assert metric.value == 0.5


def test_statistics_monitor_handles_multiple_columns():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [1.0, 2.0],
            "volume": [100, 200],
            "symbol": ["A", "B"],
        }
    )

    result = monitor.execute(data)

    assert result.metadata["numeric_columns"] == [
        "price",
        "volume",
    ]

    assert len(
        result.metadata["column_statistics"]
    ) == 2


def test_statistics_monitor_handles_all_missing_numeric_column():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "price": [np.nan, np.nan],
        }
    )

    result = monitor.execute(data)

    stats = result.metadata["column_statistics"][
        "price"
    ]

    assert math.isnan(stats["mean"])
    assert math.isnan(stats["median"])
    assert math.isnan(stats["std"])
    assert math.isnan(stats["min"])
    assert math.isnan(stats["max"])
    assert stats["missing_ratio"] == 1.0


def test_statistics_monitor_handles_non_numeric_data():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "symbol": ["NIFTY", "BANKNIFTY"],
        }
    )

    result = monitor.execute(data)

    assert result.metadata["numeric_columns"] == []
    assert result.metadata["column_statistics"] == {}

    metric = get_metric(
        result,
        "numeric_column_count",
    )

    assert metric.value == 0.0


def test_statistics_monitor_handles_empty_dataframe():
    monitor = create_monitor()

    result = monitor.execute(pd.DataFrame())

    assert result.status == MonitoringStatus.OK
    assert result.metadata["numeric_columns"] == []
    assert result.metadata["column_statistics"] == {}


def test_statistics_monitor_rejects_non_dataframe():
    monitor = create_monitor()

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        monitor.execute(
            {"price": [1.0, 2.0]}
        )


def test_statistics_monitor_warns_when_missing_ratio_exceeds_threshold():
    monitor = create_monitor(
        thresholds={
            "max_missing_ratio": 0.25,
        }
    )

    data = pd.DataFrame(
        {
            "price": [1.0, np.nan],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_statistics_monitor_uses_ok_at_exact_missing_ratio_threshold():
    monitor = create_monitor(
        thresholds={
            "max_missing_ratio": 0.5,
        }
    )

    data = pd.DataFrame(
        {
            "price": [1.0, np.nan],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK


def test_statistics_monitor_includes_numeric_column_count():
    monitor = create_monitor()

    data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3.0, 4.0],
            "c": ["x", "y"],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(
        result,
        "numeric_column_count",
    )

    assert metric.value == 2.0


def test_statistics_monitor_warns_when_numeric_column_count_exceeds_threshold():
    monitor = create_monitor(
        thresholds={
            "numeric_column_count": 1.0,
        }
    )

    data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING