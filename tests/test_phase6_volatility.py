"""Tests for Phase 6 volatility features."""

from __future__ import annotations

import math

import pytest

from src.features.config import FeatureConfig
from src.features.volatility import VolatilityFeatureTransformer


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


def test_creates_expected_volatility_features() -> None:
    """Transformer should report all configured features."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 99.0])
    )

    assert result.created_features == (
        "feature__return_variance_2d",
        "feature__return_volatility_2d",
        "feature__realized_volatility_2d",
        "feature__downside_volatility_2d",
    )


def test_calculates_return_variance_and_volatility() -> None:
    """Variance and volatility should be calculated correctly."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 99.0])
    )

    # Returns: +10%, -10%
    record = result.records[2]

    assert record[
        "feature__return_variance_2d"
    ] == pytest.approx(0.01)

    assert record[
        "feature__return_volatility_2d"
    ] == pytest.approx(0.1)


def test_calculates_realized_volatility() -> None:
    """Realized volatility should use squared returns."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 99.0])
    )

    expected = math.sqrt(
        (0.10 ** 2) + ((-0.10) ** 2)
    )

    assert result.records[2][
        "feature__realized_volatility_2d"
    ] == pytest.approx(expected)


def test_calculates_downside_volatility() -> None:
    """Downside volatility should use negative returns only."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 99.0])
    )

    expected = math.sqrt(
        ((-0.10) ** 2) / 2
    )

    assert result.records[2][
        "feature__downside_volatility_2d"
    ] == pytest.approx(expected)


def test_insufficient_history_produces_missing_features() -> None:
    """Features should remain missing before enough returns exist."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    for record in result.records:
        for feature_name in result.created_features:
            assert record[feature_name] is None


def test_missing_price_breaks_return_window() -> None:
    """Invalid prices should invalidate affected volatility windows."""

    config = FeatureConfig(
        volatility_periods=(2,),
    )

    transformer = VolatilityFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, None, 110.0])
    )

    record = result.records[2]

    for feature_name in result.created_features:
        assert record[feature_name] is None


def test_transformer_does_not_mutate_input() -> None:
    """Volatility transformer should preserve input records."""

    transformer = VolatilityFeatureTransformer()

    records = build_records([100.0, 110.0, 99.0])

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )