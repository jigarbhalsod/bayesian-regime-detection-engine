"""Tests for Phase 6 technical indicators."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.technical import (
    TechnicalIndicatorTransformer,
)


def build_records(
    closes: list[float | None],
) -> list[dict[str, object]]:
    """Build chronological market records."""

    return [
        {
            "date": f"2024-01-{index + 1:02d}",
            "nifty_50__close": close,
        }
        for index, close in enumerate(closes)
    ]


def test_creates_expected_technical_features() -> None:
    """Transformer should report all technical features."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0, 130.0])
    )

    assert result.created_features == (
        "feature__rsi",
        "feature__macd",
        "feature__macd_signal",
        "feature__macd_histogram",
        "feature__bollinger_middle",
        "feature__bollinger_upper",
        "feature__bollinger_lower",
        "feature__bollinger_position",
    )


def test_calculates_rsi_for_all_gains() -> None:
    """RSI should reach 100 for a complete all-gain window."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0])
    )

    assert result.records[2][
        "feature__rsi"
    ] == pytest.approx(100.0)


def test_calculates_rsi_for_mixed_moves() -> None:
    """RSI should correctly combine gains and losses."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    result = transformer.transform(
        build_records([100.0, 110.0, 105.0])
    )

    assert result.records[2][
        "feature__rsi"
    ] == pytest.approx(66.6666666667)


def test_calculates_bollinger_features() -> None:
    """Bollinger Bands should use rolling population deviation."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
            bollinger_std_multiplier=2.0,
        )
    )

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    record = result.records[1]

    assert record[
        "feature__bollinger_middle"
    ] == pytest.approx(105.0)

    assert record[
        "feature__bollinger_upper"
    ] == pytest.approx(115.0)

    assert record[
        "feature__bollinger_lower"
    ] == pytest.approx(95.0)

    assert record[
        "feature__bollinger_position"
    ] == pytest.approx(0.75)


def test_flat_bollinger_range_has_missing_position() -> None:
    """Flat prices should produce no Bollinger position."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    result = transformer.transform(
        build_records([100.0, 100.0])
    )

    assert result.records[1][
        "feature__bollinger_position"
    ] is None


def test_insufficient_history_produces_missing_features() -> None:
    """Indicators should require complete historical windows."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=3,
            macd_fast_period=3,
            macd_slow_period=4,
            macd_signal_period=2,
            bollinger_period=3,
        )
    )

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    for record in result.records:
        for feature_name in result.created_features:
            assert record[feature_name] is None


def test_missing_close_breaks_affected_windows() -> None:
    """Missing prices should invalidate dependent indicators."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    result = transformer.transform(
        build_records([100.0, None, 120.0, 130.0])
    )

    assert result.records[2][
        "feature__rsi"
    ] is None

    assert result.records[2][
        "feature__bollinger_middle"
    ] is None


def test_transformer_does_not_mutate_input() -> None:
    """Technical transformer should preserve input records."""

    transformer = TechnicalIndicatorTransformer(
        FeatureConfig(
            rsi_period=2,
            macd_fast_period=2,
            macd_slow_period=3,
            macd_signal_period=2,
            bollinger_period=2,
        )
    )

    records = build_records(
        [100.0, 110.0, 120.0, 130.0]
    )

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )