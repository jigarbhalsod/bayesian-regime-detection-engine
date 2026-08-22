from __future__ import annotations

import math

import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.cross_asset import CrossAssetAnalyzer


def test_cross_asset_analyzer_name() -> None:
    analyzer = CrossAssetAnalyzer()

    assert analyzer.name == "cross_asset"


def test_cross_asset_analyzer_handles_empty_records() -> None:
    analyzer = CrossAssetAnalyzer()

    result = analyzer.analyze([])

    assert result.records == []
    assert result.metrics == {}
    assert result.metric_names == []


def test_cross_asset_analyzer_requires_multiple_return_columns() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
    ]

    result = analyzer.analyze(records)

    assert result.records == records
    assert result.metrics == {}
    assert result.metric_names == []


def test_cross_asset_analyzer_calculates_correlation() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": 0.02,
        },
        {
            "feature__return_1d": 0.02,
            "feature__asset_b_return": 0.04,
        },
        {
            "feature__return_1d": 0.03,
            "feature__asset_b_return": 0.06,
        },
    ]

    result = analyzer.analyze(records)

    assert result.metrics[
        "analysis__correlation__feature__return_1d__feature__asset_b_return"
    ] == pytest.approx(1.0)


def test_cross_asset_analyzer_handles_negative_correlation() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": -0.01,
        },
        {
            "feature__return_1d": 0.02,
            "feature__asset_b_return": -0.02,
        },
        {
            "feature__return_1d": 0.03,
            "feature__asset_b_return": -0.03,
        },
    ]

    result = analyzer.analyze(records)

    assert result.metrics[
        "analysis__correlation__feature__return_1d__feature__asset_b_return"
    ] == pytest.approx(-1.0)


def test_cross_asset_analyzer_ignores_invalid_values() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": 0.02,
        },
        {
            "feature__return_1d": "invalid",
            "feature__asset_b_return": 0.04,
        },
        {
            "feature__return_1d": 0.03,
            "feature__asset_b_return": 0.06,
        },
        {
            "feature__return_1d": 0.04,
            "feature__asset_b_return": 0.08,
        },
    ]

    result = analyzer.analyze(records)

    key = (
        "analysis__correlation__"
        "feature__return_1d__feature__asset_b_return"
    )

    assert result.metrics[key] == pytest.approx(1.0)
    assert result.metadata["valid_pair_count"] == 3


def test_cross_asset_analyzer_handles_numeric_strings() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {
            "feature__return_1d": "0.01",
            "feature__asset_b_return": "0.02",
        },
        {
            "feature__return_1d": "0.02",
            "feature__asset_b_return": "0.04",
        },
        {
            "feature__return_1d": "0.03",
            "feature__asset_b_return": "0.06",
        },
    ]

    result = analyzer.analyze(records)

    key = (
        "analysis__correlation__"
        "feature__return_1d__feature__asset_b_return"
    )

    assert result.metrics[key] == pytest.approx(1.0)


def test_cross_asset_analyzer_does_not_mutate_input() -> None:
    analyzer = CrossAssetAnalyzer()

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": 0.02,
        },
        {
            "feature__return_1d": 0.02,
            "feature__asset_b_return": 0.04,
        },
    ]

    original_records = [dict(record) for record in records]

    result = analyzer.analyze(records)

    assert records == original_records
    assert result.records == original_records


def test_cross_asset_analyzer_adds_rolling_correlation() -> None:
    config = AnalysisConfig(
        correlation_windows=(3,),
    )

    analyzer = CrossAssetAnalyzer(config=config)

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": 0.02,
        },
        {
            "feature__return_1d": 0.02,
            "feature__asset_b_return": 0.04,
        },
        {
            "feature__return_1d": 0.03,
            "feature__asset_b_return": 0.06,
        },
        {
            "feature__return_1d": 0.04,
            "feature__asset_b_return": 0.08,
        },
    ]

    result = analyzer.analyze(records)

    column = (
        "analysis__rolling_correlation_3__"
        "feature__return_1d__feature__asset_b_return"
    )

    assert result.records[0][column] is None
    assert result.records[1][column] is None
    assert result.records[2][column] == pytest.approx(1.0)
    assert result.records[3][column] == pytest.approx(1.0)


def test_cross_asset_analyzer_reports_correlation_windows_metadata() -> None:
    config = AnalysisConfig(
        correlation_windows=(3, 10),
    )

    analyzer = CrossAssetAnalyzer(config=config)

    records = [
        {
            "feature__return_1d": 0.01,
            "feature__asset_b_return": 0.02,
        },
        {
            "feature__return_1d": 0.02,
            "feature__asset_b_return": 0.04,
        },
        {
            "feature__return_1d": 0.03,
            "feature__asset_b_return": 0.06,
        },
    ]

    result = analyzer.analyze(records)

    assert result.metadata["correlation_windows"] == [3, 10]