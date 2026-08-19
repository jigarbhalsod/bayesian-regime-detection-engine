"""Tests for Phase 6 price features."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.price import PriceFeatureTransformer


def build_records(
    closes: list[float | None],
) -> list[dict[str, object]]:
    """Build chronological Phase 5-style records."""

    return [
        {
            "date": f"2024-01-{index + 1:02d}",
            "nifty_50__close": close,
        }
        for index, close in enumerate(closes)
    ]


def test_creates_expected_price_features() -> None:
    """Transformer should report all configured price features."""

    config = FeatureConfig(
        return_periods=(1, 2),
    )

    transformer = PriceFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    assert result.created_features == (
        "feature__price_change_1d",
        "feature__price_change_2d",
        "feature__rolling_max_1d",
        "feature__rolling_min_1d",
        "feature__price_position_1d",
        "feature__drawdown_1d",
        "feature__rolling_max_2d",
        "feature__rolling_min_2d",
        "feature__price_position_2d",
        "feature__drawdown_2d",
    )


def test_calculates_one_day_price_change() -> None:
    """One-day absolute price change should be correct."""

    transformer = PriceFeatureTransformer()

    result = transformer.transform(
        build_records([100.0, 108.0])
    )

    assert result.records[1][
        "feature__price_change_1d"
    ] == pytest.approx(8.0)


def test_calculates_multi_period_price_change() -> None:
    """Multi-period price changes should use the correct lag."""

    config = FeatureConfig(
        return_periods=(1, 2, 3),
    )

    transformer = PriceFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0, 135.0])
    )

    record = result.records[3]

    assert record["feature__price_change_1d"] == pytest.approx(15.0)
    assert record["feature__price_change_2d"] == pytest.approx(25.0)
    assert record["feature__price_change_3d"] == pytest.approx(35.0)


def test_calculates_rolling_price_features() -> None:
    """Rolling highs, lows, position and drawdown should be correct."""

    config = FeatureConfig(
        return_periods=(3,),
    )

    transformer = PriceFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 120.0, 110.0])
    )

    record = result.records[2]

    assert record["feature__rolling_max_3d"] == pytest.approx(120.0)
    assert record["feature__rolling_min_3d"] == pytest.approx(100.0)
    assert record["feature__price_position_3d"] == pytest.approx(0.5)
    assert record["feature__drawdown_3d"] == pytest.approx(
        (110.0 / 120.0) - 1.0
    )


def test_insufficient_history_produces_missing_rolling_features() -> None:
    """Rolling features should remain missing until the window exists."""

    config = FeatureConfig(
        return_periods=(3,),
    )

    transformer = PriceFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    for record in result.records:
        assert record["feature__rolling_max_3d"] is None
        assert record["feature__rolling_min_3d"] is None
        assert record["feature__price_position_3d"] is None
        assert record["feature__drawdown_3d"] is None


def test_flat_price_range_has_missing_position() -> None:
    """Position should be undefined when rolling range is zero."""

    config = FeatureConfig(
        return_periods=(3,),
    )

    transformer = PriceFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 100.0, 100.0])
    )

    record = result.records[2]

    assert record["feature__price_position_3d"] is None
    assert record["feature__drawdown_3d"] == pytest.approx(0.0)


def test_missing_current_price_produces_missing_features() -> None:
    """Missing current price should invalidate generated features."""

    transformer = PriceFeatureTransformer()

    result = transformer.transform(
        build_records([100.0, None])
    )

    record = result.records[1]

    for feature_name in result.created_features:
        assert record[feature_name] is None


def test_transformer_does_not_mutate_input() -> None:
    """Price transformer should preserve input records."""

    transformer = PriceFeatureTransformer()

    records = build_records([100.0, 110.0])

    transformer.transform(records)

    assert "feature__price_change_1d" not in records[0]
    assert "feature__price_change_1d" not in records[1]