"""Tests for Phase 6 Group A feature engineering architecture."""

from __future__ import annotations

from typing import Any

import pytest

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig
from src.features.pipeline import FeatureEngineeringPipeline
from src.features.registry import FeatureRegistry
from src.features.price import PriceFeatureTransformer
from src.features.returns import ReturnFeatureTransformer
from src.features.momentum import MomentumFeatureTransformer

class DummyTransformer(BaseFeatureTransformer):
    """Simple transformer used for architecture tests."""

    name = "dummy"

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        transformed_records = []

        for record in records:
            updated_record = dict(record)
            updated_record["dummy__feature"] = 1
            transformed_records.append(updated_record)

        return FeatureResult(
            records=transformed_records,
            created_features=("dummy__feature",),
        )


class SecondDummyTransformer(BaseFeatureTransformer):
    """Second transformer used to verify execution order."""

    name = "second_dummy"

    def transform(
        self,
        records: list[dict[str, Any]],
    ) -> FeatureResult:
        transformed_records = []

        for record in records:
            updated_record = dict(record)
            updated_record["second_dummy__feature"] = (
                updated_record["dummy__feature"] + 1
            )
            transformed_records.append(updated_record)

        return FeatureResult(
            records=transformed_records,
            created_features=("second_dummy__feature",),
        )


def test_feature_config_builds_phase5_compatible_columns() -> None:
    """Configuration should match Phase 5 integrated naming."""

    config = FeatureConfig()

    assert config.target_close_column == "nifty_50__close"
    assert config.target_returns_column == "nifty_50__returns"


def test_registry_registers_and_resolves_transformer() -> None:
    """Registered transformers should be retrievable."""

    registry = FeatureRegistry()
    transformer = DummyTransformer()

    registry.register(transformer)

    assert registry.get("dummy") is transformer
    assert registry.names() == ("dummy",)


def test_registry_rejects_duplicate_transformer_names() -> None:
    """Duplicate transformer names should fail."""

    registry = FeatureRegistry()

    registry.register(DummyTransformer())

    with pytest.raises(ValueError):
        registry.register(DummyTransformer())


def test_registry_rejects_unknown_transformer() -> None:
    """Unknown transformer lookup should fail."""

    registry = FeatureRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_pipeline_runs_transformers_in_registration_order() -> None:
    """Pipeline should execute transformers deterministically."""

    registry = FeatureRegistry()

    registry.register(DummyTransformer())
    registry.register(SecondDummyTransformer())

    pipeline = FeatureEngineeringPipeline(registry)

    records = [
        {
            "date": "2024-01-01",
            "nifty_50__close": 100.0,
        }
    ]

    result = pipeline.run(records)

    assert result.executed_transformers == (
        "dummy",
        "second_dummy",
    )

    assert result.created_features == (
        "dummy__feature",
        "second_dummy__feature",
    )

    assert result.records[0]["dummy__feature"] == 1
    assert result.records[0]["second_dummy__feature"] == 2


def test_pipeline_does_not_mutate_input_records() -> None:
    """Pipeline should preserve the original input records."""

    registry = FeatureRegistry()
    registry.register(DummyTransformer())

    pipeline = FeatureEngineeringPipeline(registry)

    records: list[dict[str, Any]] = [
        {
            "date": "2024-01-01",
            "nifty_50__close": 100.0,
        }
    ]

    pipeline.run(records)

    assert "dummy__feature" not in records[0]


def test_empty_registry_returns_input_unchanged() -> None:
    """An empty registry should safely return copied input records."""

    registry = FeatureRegistry()
    pipeline = FeatureEngineeringPipeline(registry)

    records = [
        {
            "date": "2024-01-01",
            "nifty_50__close": 100.0,
        }
    ]

    result = pipeline.run(records)

    assert result.records == records
    assert result.records is not records
    assert result.created_features == ()
    assert result.executed_transformers == ()

def test_pipeline_integrates_return_and_price_transformers() -> None:
    """Pipeline should run real Phase 6.2 transformers together."""

    config = FeatureConfig(
        return_periods=(1, 2),
    )

    registry = FeatureRegistry()

    registry.register(
        ReturnFeatureTransformer(config)
    )
    registry.register(
        PriceFeatureTransformer(config)
    )

    pipeline = FeatureEngineeringPipeline(registry)

    records = [
        {
            "date": "2024-01-01",
            "nifty_50__close": 100.0,
        },
        {
            "date": "2024-01-02",
            "nifty_50__close": 110.0,
        },
        {
            "date": "2024-01-03",
            "nifty_50__close": 121.0,
        },
    ]

    result = pipeline.run(records)

    final_record = result.records[2]

    assert result.executed_transformers == (
        "returns",
        "price",
    )

    assert "feature__return_1d" in result.created_features
    assert "feature__return_2d" in result.created_features

    assert "feature__price_change_1d" in result.created_features
    assert "feature__price_change_2d" in result.created_features

    assert final_record["feature__return_1d"] == pytest.approx(0.10)
    assert final_record["feature__return_2d"] == pytest.approx(0.21)

    assert final_record[
        "feature__price_change_1d"
    ] == pytest.approx(11.0)

    assert final_record[
        "feature__price_change_2d"
    ] == pytest.approx(21.0)

def test_pipeline_integrates_all_group_a_transformers() -> None:
    """Pipeline should run all Phase 6 Group A transformers."""

    config = FeatureConfig(
        return_periods=(1, 2),
        sma_periods=(2,),
        ema_periods=(2,),
        momentum_periods=(1, 2),
    )

    registry = FeatureRegistry()

    registry.register(
        ReturnFeatureTransformer(config)
    )
    registry.register(
        PriceFeatureTransformer(config)
    )
    registry.register(
        MomentumFeatureTransformer(config)
    )

    pipeline = FeatureEngineeringPipeline(registry)

    records = [
        {
            "date": "2024-01-01",
            "nifty_50__close": 100.0,
        },
        {
            "date": "2024-01-02",
            "nifty_50__close": 110.0,
        },
        {
            "date": "2024-01-03",
            "nifty_50__close": 121.0,
        },
    ]

    result = pipeline.run(records)

    final_record = result.records[2]

    assert result.executed_transformers == (
        "returns",
        "price",
        "momentum",
    )

    assert final_record["feature__return_1d"] == pytest.approx(
        0.10
    )
    assert final_record["feature__return_2d"] == pytest.approx(
        0.21
    )

    assert final_record["feature__price_change_1d"] == pytest.approx(
        11.0
    )
    assert final_record["feature__price_change_2d"] == pytest.approx(
        21.0
    )

    assert final_record["feature__sma_2d"] == pytest.approx(
        115.5
    )
    assert final_record["feature__momentum_1d"] == pytest.approx(
        11.0
    )
    assert final_record["feature__momentum_2d"] == pytest.approx(
        21.0
    )