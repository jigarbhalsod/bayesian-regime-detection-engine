import pytest

from src.analysis.drawdown import DrawdownAnalyzer


def test_drawdown_analyzer_name() -> None:
    analyzer = DrawdownAnalyzer()

    assert analyzer.name == "drawdown"


def test_drawdown_analyzer_calculates_core_metrics() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.20},
        {"feature__return_1d": 0.05},
    ]

    result = analyzer.analyze(records)

    assert result.records[0]["analysis__cumulative_wealth"] == pytest.approx(
        1.10
    )
    assert result.records[1]["analysis__cumulative_wealth"] == pytest.approx(
        0.88
    )
    assert result.records[2]["analysis__cumulative_wealth"] == pytest.approx(
        0.924
    )

    assert result.records[1]["analysis__drawdown"] == pytest.approx(
        -0.20
    )

    assert result.metrics["analysis__maximum_drawdown"] == pytest.approx(
        -0.20
    )


def test_drawdown_analyzer_handles_empty_records() -> None:
    analyzer = DrawdownAnalyzer()

    result = analyzer.analyze([])

    assert result.records == []
    assert result.metrics == {}
    assert result.metric_names == []


def test_drawdown_analyzer_ignores_invalid_values() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": "invalid"},
        {"feature__return_1d": -0.10},
    ]

    result = analyzer.analyze(records)

    assert result.metadata["valid_return_count"] == 2
    assert result.records[1]["analysis__drawdown"] is None


def test_drawdown_analyzer_does_not_mutate_input() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
    ]

    original_records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
    ]

    result = analyzer.analyze(records)

    assert records == original_records
    assert result.records != records


def test_drawdown_analyzer_uses_threshold() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.20},
        {"feature__return_1d": 0.05},
    ]

    result = analyzer.analyze(records)

    assert (
        result.metrics[
            "analysis__drawdown_threshold_breach_count"
        ]
        >= 1
    )

def test_drawdown_analyzer_tracks_drawdown_duration() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": 0.30},
    ]

    result = analyzer.analyze(records)

    assert result.records[0]["analysis__drawdown_duration"] == 0
    assert result.records[1]["analysis__drawdown_duration"] == 1
    assert result.records[2]["analysis__drawdown_duration"] == 2
    assert result.records[3]["analysis__drawdown_duration"] == 0


def test_drawdown_analyzer_reports_max_drawdown_duration() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": -0.05},
        {"feature__return_1d": 0.30},
    ]

    result = analyzer.analyze(records)

    assert result.metrics[
        "analysis__maximum_drawdown_duration"
    ] == 2


def test_drawdown_analyzer_tracks_recovery() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": 0.25},
    ]

    result = analyzer.analyze(records)

    assert result.records[1]["analysis__recovered"] is False
    assert result.records[2]["analysis__recovered"] is False
    assert result.records[3]["analysis__recovered"] is True


def test_drawdown_analyzer_invalid_values_reset_duration() -> None:
    analyzer = DrawdownAnalyzer()

    records = [
        {"feature__return_1d": 0.10},
        {"feature__return_1d": -0.10},
        {"feature__return_1d": "invalid"},
        {"feature__return_1d": -0.05},
    ]

    result = analyzer.analyze(records)

    assert result.records[1]["analysis__drawdown_duration"] == 1
    assert result.records[2]["analysis__drawdown_duration"] is None
    assert result.records[3]["analysis__drawdown_duration"] == 2