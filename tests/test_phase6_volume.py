"""Tests for Phase 6 volume and liquidity features."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.volume import VolumeFeatureTransformer


def build_records(
    volumes: list[float | None],
    closes: list[float | None] | None = None,
) -> list[dict[str, object]]:
    """Build chronological market records."""

    if closes is None:
        closes = [
            100.0
            for _ in volumes
        ]

    return [
        {
            "date": f"2024-01-{index + 1:02d}",
            "nifty_50__close": closes[index],
            "nifty_50__volume": volume,
        }
        for index, volume in enumerate(volumes)
    ]


def test_creates_expected_volume_features() -> None:
    """Transformer should report configured features."""

    config = FeatureConfig(
        volume_periods=(2,),
    )

    transformer = VolumeFeatureTransformer(config)

    result = transformer.transform(
        build_records([100.0, 200.0])
    )

    assert result.created_features == (
        "feature__volume_change_1d",
        "feature__volume_return_1d",
        "feature__traded_value",
        "feature__average_volume_2d",
        "feature__relative_volume_2d",
        "feature__volume_volatility_2d",
        "feature__volume_trend_2d",
    )


def test_calculates_one_day_volume_features() -> None:
    """One-day volume features should be correct."""

    transformer = VolumeFeatureTransformer(
        FeatureConfig(volume_periods=(2,))
    )

    result = transformer.transform(
        build_records([100.0, 150.0])
    )

    record = result.records[1]

    assert record[
        "feature__volume_change_1d"
    ] == pytest.approx(50.0)

    assert record[
        "feature__volume_return_1d"
    ] == pytest.approx(0.5)


def test_calculates_traded_value() -> None:
    """Traded value should equal close times volume."""

    transformer = VolumeFeatureTransformer()

    result = transformer.transform(
        build_records(
            volumes=[1000.0],
            closes=[250.0],
        )
    )

    assert result.records[0][
        "feature__traded_value"
    ] == pytest.approx(250000.0)


def test_calculates_rolling_volume_features() -> None:
    """Rolling volume statistics should be correct."""

    transformer = VolumeFeatureTransformer(
        FeatureConfig(volume_periods=(2,))
    )

    result = transformer.transform(
        build_records([100.0, 200.0])
    )

    record = result.records[1]

    assert record[
        "feature__average_volume_2d"
    ] == pytest.approx(150.0)

    assert record[
        "feature__relative_volume_2d"
    ] == pytest.approx(200.0 / 150.0)

    assert record[
        "feature__volume_volatility_2d"
    ] == pytest.approx(50.0)

    assert record[
        "feature__volume_trend_2d"
    ] == pytest.approx(1.0)


def test_insufficient_history_produces_missing_rolling_features() -> None:
    """Rolling features require a complete historical window."""

    transformer = VolumeFeatureTransformer(
        FeatureConfig(volume_periods=(3,))
    )

    result = transformer.transform(
        build_records([100.0, 200.0])
    )

    for record in result.records:
        assert record[
            "feature__average_volume_3d"
        ] is None
        assert record[
            "feature__relative_volume_3d"
        ] is None
        assert record[
            "feature__volume_volatility_3d"
        ] is None
        assert record[
            "feature__volume_trend_3d"
        ] is None


def test_missing_volume_produces_missing_features() -> None:
    """Missing volume should invalidate dependent features."""

    transformer = VolumeFeatureTransformer(
        FeatureConfig(volume_periods=(2,))
    )

    result = transformer.transform(
        build_records([100.0, None])
    )

    record = result.records[1]

    assert record[
        "feature__volume_change_1d"
    ] is None

    assert record[
        "feature__volume_return_1d"
    ] is None

    assert record[
        "feature__average_volume_2d"
    ] is None

    assert record[
        "feature__relative_volume_2d"
    ] is None


def test_missing_close_only_breaks_traded_value() -> None:
    """Volume-only features should not require close prices."""

    transformer = VolumeFeatureTransformer(
        FeatureConfig(volume_periods=(2,))
    )

    result = transformer.transform(
        build_records(
            volumes=[100.0, 200.0],
            closes=[100.0, None],
        )
    )

    record = result.records[1]

    assert record[
        "feature__traded_value"
    ] is None

    assert record[
        "feature__average_volume_2d"
    ] == pytest.approx(150.0)


def test_transformer_does_not_mutate_input() -> None:
    """Transformer should preserve the original records."""

    transformer = VolumeFeatureTransformer()

    records = build_records([100.0, 200.0])

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )