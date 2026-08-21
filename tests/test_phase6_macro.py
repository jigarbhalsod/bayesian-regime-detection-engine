"""Tests for Phase 6 macro-economic features."""

from __future__ import annotations

import pytest

from src.features.config import FeatureConfig
from src.features.macro import MacroFeatureTransformer


def build_records(
    repo_rates: list[float | None],
    cpi_values: list[float | None],
    usd_inr_values: list[float | None],
) -> list[dict[str, object]]:
    """Build aligned chronological macro records."""

    return [
        {
            "date": f"2024-{index + 1:02d}-01",
            "repo_rate": repo_rate,
            "cpi_inflation": cpi_value,
            "usd_inr": usd_inr_value,
        }
        for index, (
            repo_rate,
            cpi_value,
            usd_inr_value,
        ) in enumerate(
            zip(
                repo_rates,
                cpi_values,
                usd_inr_values,
            )
        )
    ]


def make_config() -> FeatureConfig:
    """Create a compact configuration for deterministic tests."""

    return FeatureConfig(
        macro_columns=(
            "repo_rate",
            "cpi_inflation",
            "usd_inr",
        ),
        macro_change_periods=(1,),
        macro_rolling_periods=(2,),
    )


def test_creates_expected_macro_features() -> None:
    """Transformer should report every configured macro feature."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.5, 6.5],
            [5.0, 5.5],
            [80.0, 81.0],
        )
    )

    assert result.created_features == (
        "feature__macro__repo_rate__change_1",
        "feature__macro__repo_rate__mean_2",
        "feature__macro__repo_rate__deviation_2",
        "feature__macro__repo_rate__zscore_2",
        "feature__macro__cpi_inflation__change_1",
        "feature__macro__cpi_inflation__mean_2",
        "feature__macro__cpi_inflation__deviation_2",
        "feature__macro__cpi_inflation__zscore_2",
        "feature__macro__usd_inr__change_1",
        "feature__macro__usd_inr__mean_2",
        "feature__macro__usd_inr__deviation_2",
        "feature__macro__usd_inr__zscore_2",
    )


def test_calculates_macro_change() -> None:
    """Macro changes should compare current and historical values."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.0, 6.5],
            [5.0, 5.5],
            [80.0, 81.0],
        )
    )

    record = result.records[1]

    assert record[
        "feature__macro__repo_rate__change_1"
    ] == pytest.approx(0.5)

    assert record[
        "feature__macro__cpi_inflation__change_1"
    ] == pytest.approx(0.5)

    assert record[
        "feature__macro__usd_inr__change_1"
    ] == pytest.approx(1.0)


def test_calculates_rolling_mean_and_deviation() -> None:
    """Rolling mean and current deviation should be calculated correctly."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.0, 7.0],
            [5.0, 7.0],
            [80.0, 84.0],
        )
    )

    record = result.records[1]

    assert record[
        "feature__macro__repo_rate__mean_2"
    ] == pytest.approx(6.5)

    assert record[
        "feature__macro__repo_rate__deviation_2"
    ] == pytest.approx(0.5)


def test_calculates_rolling_zscore() -> None:
    """Rolling z-score should standardize against the trailing window."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.0, 8.0],
            [5.0, 7.0],
            [80.0, 84.0],
        )
    )

    assert result.records[1][
        "feature__macro__repo_rate__zscore_2"
    ] == pytest.approx(1.0)


def test_insufficient_history_produces_missing_rolling_features() -> None:
    """Rolling features should remain missing before a full window exists."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.0],
            [5.0],
            [80.0],
        )
    )

    record = result.records[0]

    assert record[
        "feature__macro__repo_rate__change_1"
    ] is None

    assert record[
        "feature__macro__repo_rate__mean_2"
    ] is None

    assert record[
        "feature__macro__repo_rate__deviation_2"
    ] is None

    assert record[
        "feature__macro__repo_rate__zscore_2"
    ] is None


def test_missing_macro_value_breaks_affected_features() -> None:
    """Missing observations should invalidate dependent windows."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.0, None, 7.0],
            [5.0, 5.5, 6.0],
            [80.0, 81.0, 82.0],
        )
    )

    assert result.records[1][
        "feature__macro__repo_rate__change_1"
    ] is None

    assert result.records[1][
        "feature__macro__repo_rate__mean_2"
    ] is None

    assert result.records[2][
        "feature__macro__repo_rate__change_1"
    ] is None

    assert result.records[2][
        "feature__macro__repo_rate__mean_2"
    ] is None


def test_flat_macro_window_has_missing_zscore() -> None:
    """A zero-variance macro window should not produce a z-score."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    result = transformer.transform(
        build_records(
            [6.5, 6.5],
            [5.0, 5.0],
            [80.0, 80.0],
        )
    )

    assert result.records[1][
        "feature__macro__repo_rate__zscore_2"
    ] is None


def test_transformer_does_not_mutate_input() -> None:
    """Macro transformer should preserve input records."""

    transformer = MacroFeatureTransformer(
        make_config()
    )

    records = build_records(
        [6.0, 6.5],
        [5.0, 5.5],
        [80.0, 81.0],
    )

    transformer.transform(records)

    assert all(
        not key.startswith("feature__")
        for record in records
        for key in record
    )