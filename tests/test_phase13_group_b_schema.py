import pandas as pd
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringStatus
from src.monitoring.schema import SchemaMonitor


def create_monitor(
    expected_schema,
    thresholds=None,
):
    return SchemaMonitor(
        MonitoringConfig(
            name="schema_monitor",
            thresholds=thresholds or {},
            metadata={
                "expected_schema": expected_schema,
            },
        )
    )


def get_metric(result, name):
    return next(
        metric
        for metric in result.metrics
        if metric.name == name
    )


def test_schema_monitor_returns_ok_for_matching_schema():
    monitor = create_monitor(
        {
            "price": "float64",
            "volume": "int64",
        }
    )

    data = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "volume": [1000, 1100],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK
    assert result.metadata["missing_columns"] == []
    assert result.metadata["unexpected_columns"] == []
    assert result.metadata["dtype_mismatches"] == []


def test_schema_monitor_detects_missing_columns():
    monitor = create_monitor(
        {
            "price": "float64",
            "volume": "int64",
        }
    )

    data = pd.DataFrame(
        {
            "price": [100.0, 101.0],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(
        result,
        "missing_column_count",
    )

    assert metric.value == 1.0
    assert result.metadata["missing_columns"] == [
        "volume"
    ]


def test_schema_monitor_detects_unexpected_columns():
    monitor = create_monitor(
        {
            "price": "float64",
        }
    )

    data = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "extra": [1, 2],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(
        result,
        "unexpected_column_count",
    )

    assert metric.value == 1.0
    assert result.metadata["unexpected_columns"] == [
        "extra"
    ]


def test_schema_monitor_detects_dtype_mismatch():
    monitor = create_monitor(
        {
            "price": "float64",
            "volume": "int64",
        }
    )

    data = pd.DataFrame(
        {
            "price": ["100", "101"],
            "volume": [1000, 1100],
        }
    )

    result = monitor.execute(data)

    metric = get_metric(
        result,
        "dtype_mismatch_count",
    )

    assert metric.value == 1.0

    mismatch = result.metadata["dtype_mismatches"][0]

    assert mismatch["column"] == "price"
    assert mismatch["expected"] == "float64"
    assert mismatch["actual"] == "object"


def test_schema_monitor_does_not_check_dtype_for_missing_column():
    monitor = create_monitor(
        {
            "price": "float64",
        }
    )

    data = pd.DataFrame(
        {
            "volume": [1000, 1100],
        }
    )

    result = monitor.execute(data)

    assert (
        get_metric(
            result,
            "missing_column_count",
        ).value
        == 1.0
    )

    assert (
        get_metric(
            result,
            "dtype_mismatch_count",
        ).value
        == 0.0
    )


def test_schema_monitor_rejects_non_dataframe():
    monitor = create_monitor(
        {
            "price": "float64",
        }
    )

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        monitor.execute(
            {"price": [100.0]}
        )


def test_schema_monitor_warns_for_missing_columns():
    monitor = create_monitor(
        {
            "price": "float64",
        },
        thresholds={
            "missing_column_count": 0.0,
        },
    )

    data = pd.DataFrame(
        {
            "volume": [1000],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_schema_monitor_warns_for_unexpected_columns():
    monitor = create_monitor(
        {
            "price": "float64",
        },
        thresholds={
            "unexpected_column_count": 0.0,
        },
    )

    data = pd.DataFrame(
        {
            "price": [100.0],
            "extra": [1],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_schema_monitor_warns_for_dtype_mismatch():
    monitor = create_monitor(
        {
            "price": "float64",
        },
        thresholds={
            "dtype_mismatch_count": 0.0,
        },
    )

    data = pd.DataFrame(
        {
            "price": ["100"],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.WARNING


def test_schema_monitor_uses_ok_at_exact_threshold():
    monitor = create_monitor(
        {
            "price": "float64",
        },
        thresholds={
            "missing_column_count": 1.0,
        },
    )

    data = pd.DataFrame(
        {
            "volume": [100],
        }
    )

    result = monitor.execute(data)

    assert result.status == MonitoringStatus.OK


def test_schema_monitor_handles_empty_dataframe():
    monitor = create_monitor(
        {
            "price": "float64",
        }
    )

    data = pd.DataFrame()

    result = monitor.execute(data)

    assert (
        get_metric(
            result,
            "missing_column_count",
        ).value
        == 1.0
    )


def test_schema_monitor_rejects_invalid_expected_schema():
    config = MonitoringConfig(
        name="schema_monitor",
        metadata={
            "expected_schema": ["price"],
        },
    )

    monitor = SchemaMonitor(config)

    with pytest.raises(
        TypeError,
        match="expected_schema",
    ):
        monitor.execute(
            pd.DataFrame(
                {"price": [100.0]}
            )
        )