"""Tests for Phase 6 cross-asset features."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.cross_asset import (
    CrossAssetFeatureTransformer,
)


def build_records(
    target_closes: list[float | None],
    bank_closes: list[float | None],
    vix_closes: list[float | None],
) -> list[dict[str, object]]:
    """Build aligned chronological multi-asset records."""

    return [
        {
            "date": f"2024-01-{index + 1:02d}",
            "nifty_50__close": target_close,
            "bank_nifty__close": bank_close,
            "india_vix__close": vix_close,
        }
        for index, (
            target_close,
            bank_close,
            vix_close,
        ) in enumerate(
            zip(
                target_closes,
                bank_closes,
                vix_closes,
            )
        )
    ]


def make_config() -> FeatureConfig:
    """Create a compact test configuration."""

    return FeatureConfig(
        cross_asset_columns=(
            "bank_nifty__close",
            "india_vix__close",
        ),
        cross_asset_return_period=1,
        cross_asset_correlation_periods=(2,),
    )


def test_creates_expected_cross_asset_features() -> None:
    """Transformer should report all configured features."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0, 121.0],
            [200.0, 220.0, 242.0],
            [20.0, 18.0, 19.8],
        )
    )

    assert result.created_features == (
        "feature__cross_asset__bank_nifty__return",
        "feature__cross_asset__bank_nifty__return_spread",
        "feature__cross_asset__bank_nifty__correlation_2",
        "feature__cross_asset__india_vix__return",
        "feature__cross_asset__india_vix__return_spread",
        "feature__cross_asset__india_vix__correlation_2",
    )


def test_calculates_cross_asset_returns_and_spreads() -> None:
    """Cross-asset returns and target-relative spreads should be correct."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0],
            [200.0, 210.0],
            [20.0, 18.0],
        )
    )

    record = result.records[1]

    assert record[
        "feature__cross_asset__bank_nifty__return"
    ] == pytest.approx(0.05)

    assert record[
        "feature__cross_asset__bank_nifty__return_spread"
    ] == pytest.approx(0.05)

    assert record[
        "feature__cross_asset__india_vix__return"
    ] == pytest.approx(-0.10)

    assert record[
        "feature__cross_asset__india_vix__return_spread"
    ] == pytest.approx(0.20)


def test_calculates_rolling_correlation() -> None:
    """Rolling correlation should use only trailing aligned returns."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0, 132.0],
            [200.0, 220.0, 264.0],
            [20.0, 18.0, 14.4],
        )
    )

    assert result.records[2][
        "feature__cross_asset__bank_nifty__correlation_2"
    ] == pytest.approx(1.0)

    assert result.records[2][
        "feature__cross_asset__india_vix__correlation_2"
    ] == pytest.approx(-1.0)


def test_insufficient_history_produces_missing_correlation() -> None:
    """Correlation should remain missing before enough returns exist."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0],
            [200.0, 210.0],
            [20.0, 18.0],
        )
    )

    for record in result.records:
        assert record[
            "feature__cross_asset__bank_nifty__correlation_2"
        ] is None

        assert record[
            "feature__cross_asset__india_vix__correlation_2"
        ] is None


def test_missing_asset_price_breaks_dependent_features() -> None:
    """Missing asset prices should invalidate affected features."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0, 121.0],
            [200.0, None, 242.0],
            [20.0, 18.0, 19.8],
        )
    )

    assert result.records[1][
        "feature__cross_asset__bank_nifty__return"
    ] is None

    assert result.records[1][
        "feature__cross_asset__bank_nifty__return_spread"
    ] is None

    assert result.records[2][
        "feature__cross_asset__bank_nifty__return"
    ] is None

    assert result.records[2][
        "feature__cross_asset__bank_nifty__correlation_2"
    ] is None


def test_flat_return_window_has_missing_correlation() -> None:
    """Zero-variance return windows should not produce correlation."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [100.0, 110.0, 121.0],
            [200.0, 220.0, 242.0],
            [20.0, 20.0, 20.0],
        )
    )

    assert result.records[2][
        "feature__cross_asset__india_vix__correlation_2"
    ] is None


def test_transformer_does_not_mutate_input() -> None:
    """Cross-asset transformer should preserve input records."""

    transformer = CrossAssetFeatureTransformer(
        make_config()
    )

    records = build_records(
        [100.0, 110.0, 121.0],
        [200.0, 220.0, 242.0],
        [20.0, 18.0, 19.8],
    )

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )