"""Tests for Phase 6 trend and momentum features."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.momentum import MomentumFeatureTransformer


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


def test_creates_expected_trend_and_momentum_features() -> None:
    """Transformer should report all configured features."""

    config = FeatureConfig(
        sma_periods=(2,),
        ema_periods=(2,),
        momentum_periods=(1, 2),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0])
    )

    assert result.created_features == (
        "feature__sma_2d",
        "feature__price_to_sma_2d",
        "feature__ema_2d",
        "feature__price_to_ema_2d",
        "feature__momentum_1d",
        "feature__roc_1d",
        "feature__momentum_2d",
        "feature__roc_2d",
    )


def test_calculates_sma_and_price_ratio() -> None:
    """SMA and price-to-SMA ratio should be correct."""

    config = FeatureConfig(
        sma_periods=(3,),
        ema_periods=(),
        momentum_periods=(),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0])
    )

    record = result.records[2]

    assert record["feature__sma_3d"] == pytest.approx(110.0)
    assert record["feature__price_to_sma_3d"] == pytest.approx(
        120.0 / 110.0
    )


def test_calculates_ema_and_price_ratio() -> None:
    """EMA should use only the available historical window."""

    config = FeatureConfig(
        sma_periods=(),
        ema_periods=(3,),
        momentum_periods=(),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0])
    )

    record = result.records[2]

    # Multiplier for period 3 = 2 / (3 + 1) = 0.5
    # EMA starts at 100:
    # EMA1 = 105
    # EMA2 = 112.5
    assert record["feature__ema_3d"] == pytest.approx(112.5)
    assert record["feature__price_to_ema_3d"] == pytest.approx(
        120.0 / 112.5
    )


def test_calculates_momentum_and_roc() -> None:
    """Momentum and ROC should use the correct historical lag."""

    config = FeatureConfig(
        sma_periods=(),
        ema_periods=(),
        momentum_periods=(1, 2),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 130.0])
    )

    record = result.records[2]

    assert record["feature__momentum_1d"] == pytest.approx(20.0)
    assert record["feature__roc_1d"] == pytest.approx(
        (130.0 / 110.0) - 1.0
    )

    assert record["feature__momentum_2d"] == pytest.approx(30.0)
    assert record["feature__roc_2d"] == pytest.approx(
        (130.0 / 100.0) - 1.0
    )


def test_insufficient_history_produces_missing_features() -> None:
    """Features should not be calculated without enough history."""

    config = FeatureConfig(
        sma_periods=(3,),
        ema_periods=(3,),
        momentum_periods=(2,),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    for record in result.records:
        assert record["feature__sma_3d"] is None
        assert record["feature__price_to_sma_3d"] is None
        assert record["feature__ema_3d"] is None
        assert record["feature__price_to_ema_3d"] is None

    assert result.records[0]["feature__momentum_2d"] is None
    assert result.records[1]["feature__momentum_2d"] is None


def test_missing_price_produces_missing_features() -> None:
    """Missing current price should invalidate all generated features."""

    config = FeatureConfig(
        sma_periods=(2,),
        ema_periods=(2,),
        momentum_periods=(1,),
    )

    transformer = MomentumFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, None])
    )

    record = result.records[1]

    for feature_name in result.created_features:
        assert record[feature_name] is None


def test_transformer_does_not_mutate_input() -> None:
    """Transformer should preserve original records."""

    transformer = MomentumFeatureTransformer()

    records = build_records([100.0, 110.0])

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )