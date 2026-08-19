"""Tests for Phase 6 return features."""

from __future__ import annotations

import math

import pytest

from src.features.config import FeatureConfig
from src.features.returns import ReturnFeatureTransformer


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


def test_creates_expected_return_features() -> None:
    """Transformer should report all configured return features."""

    transformer = ReturnFeatureTransformer()

    result = transformer.transform(
        build_records([100.0] * 21)
    )

    assert result.created_features == (
        "feature__return_1d",
        "feature__log_return_1d",
        "feature__return_5d",
        "feature__return_10d",
        "feature__return_20d",
    )


def test_calculates_simple_and_log_return() -> None:
    """One-day returns should be calculated correctly."""

    transformer = ReturnFeatureTransformer()

    result = transformer.transform(
        build_records([100.0, 110.0])
    )

    record = result.records[1]

    assert record["feature__return_1d"] == pytest.approx(0.10)
    assert record["feature__log_return_1d"] == pytest.approx(
        math.log(1.10)
    )


def test_calculates_multi_period_returns() -> None:
    """Configured historical returns should use the correct lag."""

    config = FeatureConfig(
        return_periods=(1, 2, 3),
    )

    transformer = ReturnFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 110.0, 120.0, 130.0])
    )

    final_record = result.records[3]

    assert final_record["feature__return_1d"] == pytest.approx(
        (130.0 / 120.0) - 1.0
    )

    assert final_record["feature__return_2d"] == pytest.approx(
        (130.0 / 110.0) - 1.0
    )

    assert final_record["feature__return_3d"] == pytest.approx(
        (130.0 / 100.0) - 1.0
    )


def test_first_record_has_missing_return_features() -> None:
    """The first record should not invent historical returns."""

    transformer = ReturnFeatureTransformer()

    result = transformer.transform(
        build_records([100.0])
    )

    record = result.records[0]

    assert record["feature__return_1d"] is None
    assert record["feature__log_return_1d"] is None
    assert record["feature__return_5d"] is None
    assert record["feature__return_10d"] is None
    assert record["feature__return_20d"] is None


def test_missing_close_produces_missing_features() -> None:
    """Invalid current close should produce missing features."""

    transformer = ReturnFeatureTransformer()

    result = transformer.transform(
        build_records([100.0, None])
    )

    record = result.records[1]

    for feature_name in result.created_features:
        assert record[feature_name] is None


def test_invalid_historical_close_produces_missing_return() -> None:
    """Missing historical prices should not be silently filled."""

    transformer = ReturnFeatureTransformer()

    result = transformer.transform(
        build_records([100.0, None, 120.0])
    )

    record = result.records[2]

    assert record["feature__return_1d"] is None
    assert record["feature__log_return_1d"] is None


def test_transformer_does_not_mutate_input() -> None:
    """Return transformer should preserve the original records."""

    transformer = ReturnFeatureTransformer()

    records = build_records([100.0, 110.0])

    transformer.transform(records)

    assert "feature__return_1d" not in records[0]
    assert "feature__return_1d" not in records[1]