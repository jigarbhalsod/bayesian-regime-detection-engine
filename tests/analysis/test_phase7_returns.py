import math

import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.returns import ReturnsAnalyzer


def test_returns_analyzer_name() -> None:
    analyzer = ReturnsAnalyzer()

    assert analyzer.name == "returns"


def test_returns_analyzer_calculates_core_metrics() -> None:
    analyzer = ReturnsAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.05},
        {"feature__return_1d": 0.02},
        {"feature__return_1d": 0.00},
    ]

    result = analyzer.analyze(records)

    expected_mean = (0.10 - 0.05 + 0.02 + 0.00) / 4
    expected_cumulative = (
        (1.10 * 0.95 * 1.02 * 1.00) - 1.0
    )

    assert result.metrics["analysis__mean_return"] == pytest.approx(
        expected_mean
    )

    assert result.metrics["analysis__cumulative_return"] == pytest.approx(
        expected_cumulative
    )

    assert result.metrics[
        "analysis__positive_return_ratio"
    ] == pytest.approx(2 / 4)

    assert result.metrics[
        "analysis__negative_return_ratio"
    ] == pytest.approx(1 / 4)

    assert result.metric_names == [
        "analysis__mean_return",
        "analysis__cumulative_return",
        "analysis__positive_return_ratio",
        "analysis__negative_return_ratio",
    ]

    assert result.metadata["valid_return_count"] == 4
    assert result.metadata["return_column"] == "feature__return_1d"


def test_returns_analyzer_handles_empty_records() -> None:
    analyzer = ReturnsAnalyzer()

    result = analyzer.analyze([])

    assert result.records == []
    assert result.metrics == {}
    assert result.metric_names == []
    assert result.metadata == {
        "valid_return_count": 0,
    }


def test_returns_analyzer_ignores_invalid_values() -> None:
    analyzer = ReturnsAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": None},
        {"feature__return_1d": "invalid"},
        {"feature__return_1d": float("nan")},
        {"feature__return_1d": float("inf")},
        {"feature__return_1d": float("-inf")},
        {"feature__return_1d": -0.05},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["valid_return_count"] == 2

    assert result.metrics["analysis__mean_return"] == pytest.approx(
        0.025
    )

    assert result.metrics[
        "analysis__positive_return_ratio"
    ] == pytest.approx(0.5)

    assert result.metrics[
        "analysis__negative_return_ratio"
    ] == pytest.approx(0.5)


def test_returns_analyzer_uses_custom_return_column() -> None:
    config = AnalysisConfig(
        return_column="custom_return",
    )

    analyzer = ReturnsAnalyzer(config=config)

    records = [
        {"custom_return": 0.02},
        {"custom_return": -0.01},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["return_column"] == "custom_return"
    assert result.metadata["valid_return_count"] == 2

    assert result.metrics["analysis__mean_return"] == pytest.approx(
        0.005
    )


def test_returns_analyzer_does_not_mutate_input() -> None:
    analyzer = ReturnsAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.05},
    ]

    original_records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.05},
    ]

    result = analyzer.analyze(records)

    # Original input must remain unchanged.
    assert records == original_records

    # Output is a separate collection.
    assert result.records is not records

    # Original values are preserved in the output.
    assert result.records[0]["feature__return_1d"] == 0.10
    assert result.records[1]["feature__return_1d"] == -0.05

    # Rolling columns are added only to output records.
    for window in (5, 20, 60):
        column = f"analysis__rolling_return_{window}"

        assert result.records[0][column] is None
        assert result.records[1][column] is None


def test_returns_analyzer_handles_numeric_strings() -> None:
    analyzer = ReturnsAnalyzer()

    records = [
        {"feature__return_1d": "0.10"},
        {"feature__return_1d": "-0.05"},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["valid_return_count"] == 2
    assert result.metrics["analysis__mean_return"] == pytest.approx(
        0.025
    )


def test_returns_analyzer_ignores_missing_return_column() -> None:
    analyzer = ReturnsAnalyzer()

    records = [
        {"value": 100.0},
        {"value": 101.0},
    ]

    result = analyzer.analyze(records)

    # No valid returns means no summary metrics.
    assert result.metrics == {}
    assert result.metric_names == []

    assert result.metadata == {
        "valid_return_count": 0,
    }

    # Original data is preserved.
    assert result.records[0]["value"] == 100.0
    assert result.records[1]["value"] == 101.0

    # Rolling columns still exist in the enriched output, but are None
    # because the configured return column is missing.
    for window in (5, 20, 60):
        column = f"analysis__rolling_return_{window}"

        assert result.records[0][column] is None
        assert result.records[1][column] is None

def test_returns_analyzer_adds_rolling_returns() -> None:
    config = AnalysisConfig(
        rolling_windows=(3,),
    )

    analyzer = ReturnsAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.05},
        {"feature__return_1d": 0.02},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_return_3"

    assert result.records[0][column] is None
    assert result.records[1][column] is None

    assert result.records[2][column] == pytest.approx(
        (1.10 * 0.95 * 1.02) - 1.0
    )

    assert result.records[3][column] == pytest.approx(
        (0.95 * 1.02 * 1.03) - 1.0
    )


def test_returns_analyzer_supports_multiple_rolling_windows() -> None:
    config = AnalysisConfig(
        rolling_windows=(2, 3),
    )

    analyzer = ReturnsAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": 0.20},
        {"feature__return_1d": -0.10},
    ]

    result = analyzer.analyze(records)

    rolling_2 = "analysis__rolling_return_2"
    rolling_3 = "analysis__rolling_return_3"

    assert result.records[0][rolling_2] is None
    assert result.records[0][rolling_3] is None

    assert result.records[1][rolling_2] == pytest.approx(
        (1.10 * 1.20) - 1.0
    )
    assert result.records[1][rolling_3] is None

    assert result.records[2][rolling_2] == pytest.approx(
        (1.20 * 0.90) - 1.0
    )
    assert result.records[2][rolling_3] == pytest.approx(
        (1.10 * 1.20 * 0.90) - 1.0
    )


def test_returns_analyzer_rolling_returns_require_valid_window() -> None:
    config = AnalysisConfig(
        rolling_windows=(2,),
    )

    analyzer = ReturnsAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": None},
        {"feature__return_1d": 0.05},
        {"feature__return_1d": 0.02},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_return_2"

    assert result.records[0][column] is None
    assert result.records[1][column] is None
    assert result.records[2][column] is None

    assert result.records[3][column] == pytest.approx(
        (1.05 * 1.02) - 1.0
    )


def test_returns_analyzer_reports_rolling_windows_metadata() -> None:
    config = AnalysisConfig(
        rolling_windows=(3, 10),
    )

    analyzer = ReturnsAnalyzer(config=config)

    result = analyzer.analyze(
        [{"feature__return_1d": 0.01}]
    )

    assert result.metadata["rolling_windows"] == [3, 10]