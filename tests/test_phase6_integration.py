"""End-to-end integration tests for Phase 6 feature engineering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.features.calendar import CalendarFeatureTransformer
from src.features.config import FeatureConfig
from src.features.cross_asset import CrossAssetFeatureTransformer
from src.features.macro import MacroFeatureTransformer
from src.features.momentum import MomentumFeatureTransformer
from src.features.pipeline import FeatureEngineeringPipeline
from src.features.price import PriceFeatureTransformer
from src.features.registry import FeatureRegistry
from src.features.returns import ReturnFeatureTransformer
from src.features.technical import TechnicalIndicatorTransformer
from src.features.volatility import VolatilityFeatureTransformer
from src.features.volume import VolumeFeatureTransformer


def build_records() -> list[dict[str, Any]]:
    """Build realistic Phase 5-compatible records for integration testing."""

    return [
        {
            "date": "2024-01-29",
            "close": 100.0,
            "high": 102.0,
            "low": 98.0,
            "volume": 1000.0,
            "bank_nifty_close": 200.0,
            "india_vix_close": 20.0,
            "repo_rate": 6.50,
        },
        {
            "date": "2024-01-30",
            "close": 105.0,
            "high": 107.0,
            "low": 103.0,
            "volume": 1200.0,
            "bank_nifty_close": 210.0,
            "india_vix_close": 19.0,
            "repo_rate": 6.50,
        },
        {
            "date": "2024-01-31",
            "close": 110.0,
            "high": 112.0,
            "low": 108.0,
            "volume": 1500.0,
            "bank_nifty_close": 220.0,
            "india_vix_close": 18.0,
            "repo_rate": 6.50,
        },
        {
            "date": "2024-02-01",
            "close": 108.0,
            "high": 111.0,
            "low": 106.0,
            "volume": 1300.0,
            "bank_nifty_close": 216.0,
            "india_vix_close": 18.5,
            "repo_rate": 6.50,
        },
        {
            "date": "2024-02-02",
            "close": 115.0,
            "high": 117.0,
            "low": 113.0,
            "volume": 1800.0,
            "bank_nifty_close": 230.0,
            "india_vix_close": 17.0,
            "repo_rate": 6.75,
        },
        {
            "date": "2024-02-05",
            "close": 120.0,
            "high": 123.0,
            "low": 118.0,
            "volume": 2000.0,
            "bank_nifty_close": 240.0,
            "india_vix_close": 16.0,
            "repo_rate": 6.75,
        },
    ]


def make_config() -> FeatureConfig:
    """Create a compact configuration for end-to-end testing."""

    return FeatureConfig(
        target_close_column="close",
        target_returns_column="close",
        target_volume_column="volume",
        return_periods=(1, 2),
        price_periods=(1, 2),
        price_rolling_periods=(2,),
        momentum_periods=(2,),
        trend_periods=(2,),
        volatility_periods=(2,),
        volume_periods=(2,),
        technical_periods=(2,),
        bollinger_periods=(2,),
        cross_asset_columns=(
            "bank_nifty_close",
            "india_vix_close",
        ),
        cross_asset_periods=(2,),
        macro_columns=("repo_rate",),
        macro_periods=(2,),
    )


def build_full_pipeline(
    config: FeatureConfig,
) -> FeatureEngineeringPipeline:
    """Register every Phase 6 transformer in deterministic order."""

    registry = FeatureRegistry()

    registry.register(ReturnFeatureTransformer(config))
    registry.register(PriceFeatureTransformer(config))
    registry.register(MomentumFeatureTransformer(config))
    registry.register(VolatilityFeatureTransformer(config))
    registry.register(VolumeFeatureTransformer(config))
    registry.register(TechnicalIndicatorTransformer(config))
    registry.register(CrossAssetFeatureTransformer(config))
    registry.register(MacroFeatureTransformer(config))
    registry.register(CalendarFeatureTransformer())

    return FeatureEngineeringPipeline(registry)


def test_full_phase6_pipeline_runs_successfully() -> None:
    """Every Phase 6 transformer should run together."""

    pipeline = build_full_pipeline(make_config())

    result = pipeline.run(build_records())

    assert len(result.records) == 6
    assert len(result.created_features) > 0

    assert result.executed_transformers == (
        "returns",
        "price",
        "momentum",
        "volatility",
        "volume",
        "technical",
        "cross_asset",
        "macro",
        "calendar",
    )


def test_full_pipeline_creates_features_from_every_group() -> None:
    """Integration output should contain features from every transformer."""

    pipeline = build_full_pipeline(make_config())

    result = pipeline.run(build_records())

    feature_names = set(result.created_features)

    expected_prefixes = (
        "feature__return_",
        "feature__price_",
        "feature__momentum_",
        "feature__return_volatility_",
        "feature__volume_",
        "feature__rsi",
        "feature__cross_asset__",
        "feature__macro__",
        "feature__calendar__",
    )

    for prefix in expected_prefixes:
        assert any(
            feature_name.startswith(prefix)
            for feature_name in feature_names
        )


def test_full_pipeline_preserves_original_columns() -> None:
    """Feature engineering should preserve all original Phase 5 columns."""

    pipeline = build_full_pipeline(make_config())

    records = build_records()
    result = pipeline.run(records)

    for original, transformed in zip(
        records,
        result.records,
        strict=True,
    ):
        for key, value in original.items():
            assert transformed[key] == value


def test_full_pipeline_does_not_mutate_input() -> None:
    """End-to-end processing must not modify source records."""

    pipeline = build_full_pipeline(make_config())

    records = build_records()
    original_records = deepcopy(records)

    pipeline.run(records)

    assert records == original_records


def test_full_pipeline_has_consistent_feature_schema() -> None:
    """Every output record should expose the complete feature schema."""

    pipeline = build_full_pipeline(make_config())

    result = pipeline.run(build_records())

    for record in result.records:
        for feature_name in result.created_features:
            assert feature_name in record


def test_full_pipeline_handles_early_history_with_missing_features() -> None:
    """Early records should remain valid without enough historical data."""

    pipeline = build_full_pipeline(make_config())

    result = pipeline.run(build_records())

    first_record = result.records[0]

    assert first_record[
        "feature__return_1d"
    ] is None

    assert first_record[
        "feature__return_variance_2d"
    ] is None

    assert first_record[
        "feature__calendar__month"
    ] == 1


def test_full_pipeline_generates_later_rolling_features() -> None:
    """Later records should contain computed rolling features."""

    pipeline = build_full_pipeline(make_config())

    result = pipeline.run(build_records())

    last_record = result.records[-1]

    assert last_record[
        "feature__return_1d"
    ] is not None

    assert last_record[
        "feature__rolling_max_2d"
    ] is not None

    assert last_record[
        "feature__return_volatility_2d"
    ] is not None

    assert last_record[
        "feature__average_volume_2d"
    ] is not None


def test_full_pipeline_handles_missing_data_end_to_end() -> None:
    """Missing values should produce missing dependent features, not crashes."""

    pipeline = build_full_pipeline(make_config())

    records = build_records()

    records[3]["close"] = None
    records[3]["volume"] = None
    records[3]["bank_nifty_close"] = None
    records[3]["repo_rate"] = None

    result = pipeline.run(records)

    affected_record = result.records[3]

    assert affected_record[
        "feature__return_1d"
    ] is None

    assert affected_record[
        "feature__price_change_1d"
    ] is None

    assert affected_record[
        "feature__volume_change_1d"
    ] is None

    assert affected_record[
        "feature__calendar__month"
    ] == 2



