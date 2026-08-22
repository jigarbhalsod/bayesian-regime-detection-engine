from __future__ import annotations

import math

import pytest

from src.analysis.performance import PerformanceAnalyzer


def test_performance_analyzer_name() -> None:
    analyzer = PerformanceAnalyzer()

    assert analyzer.name == "performance"


def test_performance_analyzer_calculates_core_metrics() -> None:
    analyzer = PerformanceAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": 0.02},
    ]

    result = analyzer.analyze(records)

    expected_total_return = (
        (1.01 * 0.98 * 1.03 * 1.02) - 1.0
    )

    expected_mean_return = 0.01

    assert result.metrics[
        "analysis__total_return"
    ] == pytest.approx(expected_total_return)

    assert result.metrics[
        "analysis__mean_return"
    ] == pytest.approx(expected_mean_return)

    assert result.metadata["valid_return_count"] == 4


def test_performance_analyzer_handles_empty_records() -> None:
    analyzer = PerformanceAnalyzer()

    result = analyzer.analyze([])

    assert result.records == []
    assert result.metrics == {}
    assert result.metric_names == []


def test_performance_analyzer_ignores_invalid_values() -> None:
    analyzer = PerformanceAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": "invalid"},
        {"feature__return_1d": float("nan")},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["valid_return_count"] == 2
    assert result.metrics["analysis__mean_return"] == pytest.approx(
        0.02
    )


def test_performance_analyzer_uses_custom_return_column() -> None:
    from src.analysis.config import AnalysisConfig

    config = AnalysisConfig(
        return_column="custom_return",
    )

    analyzer = PerformanceAnalyzer(config=config)

    records = [
        {"custom_return": 0.01},
        {"custom_return": 0.02},
    ]

    result = analyzer.analyze(records)

    assert result.metrics[
        "analysis__mean_return"
    ] == pytest.approx(0.015)

    assert result.metadata[
        "return_column"
    ] == "custom_return"


def test_performance_analyzer_does_not_mutate_input() -> None:
    analyzer = PerformanceAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.02},
    ]

    original_records = [dict(record) for record in records]

    result = analyzer.analyze(records)

    assert records == original_records
    assert result.records == original_records


def test_performance_analyzer_handles_numeric_strings() -> None:
    analyzer = PerformanceAnalyzer()

    records = [
        {"feature__return_1d": "0.01"},
        {"feature__return_1d": "-0.02"},
        {"feature__return_1d": "invalid"},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["valid_return_count"] == 2
    assert result.metrics[
        "analysis__mean_return"
    ] == pytest.approx(-0.005)


def test_performance_analyzer_ignores_missing_return_column() -> None:
    analyzer = PerformanceAnalyzer()

    records = [
        {"other_column": 0.01},
        {"another_column": 0.02},
    ]

    result = analyzer.analyze(records)

    assert result.records == records
    assert result.metrics == {}
    assert result.metric_names == []

from src.analysis.config import AnalysisConfig


def test_performance_analyzer_adds_rolling_returns() -> None:
    config = AnalysisConfig(
        rolling_windows=(3,),
    )

    analyzer = PerformanceAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.02},
        {"feature__return_1d": -0.01},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_return_3"

    assert result.records[0][column] is None
    assert result.records[1][column] is None
    assert result.records[2][column] == pytest.approx(
        (1.01 * 1.02 * 0.99) - 1.0
    )
    assert result.records[3][column] == pytest.approx(
        (1.02 * 0.99 * 1.03) - 1.0
    )


def test_performance_analyzer_supports_multiple_rolling_windows() -> None:
    config = AnalysisConfig(
        rolling_windows=(2, 3),
    )

    analyzer = PerformanceAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.02},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    assert result.records[1][
        "analysis__rolling_return_2"
    ] == pytest.approx((1.01 * 1.02) - 1.0)

    assert result.records[2][
        "analysis__rolling_return_3"
    ] == pytest.approx((1.01 * 1.02 * 1.03) - 1.0)


def test_performance_analyzer_rolling_returns_require_valid_window() -> None:
    config = AnalysisConfig(
        rolling_windows=(2,),
    )

    analyzer = PerformanceAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": "invalid"},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_return_2"

    assert result.records[0][column] is None
    assert result.records[1][column] is None
    assert result.records[2][column] is None


def test_performance_analyzer_reports_rolling_windows_metadata() -> None:
    config = AnalysisConfig(
        rolling_windows=(3, 10),
    )

    analyzer = PerformanceAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.02},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["rolling_windows"] == [3, 10]