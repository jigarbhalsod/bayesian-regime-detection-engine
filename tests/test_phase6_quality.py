"""Feature quality validation tests for Phase 6."""

from __future__ import annotations

import math

from tests.test_phase6_integration import (
    build_full_pipeline,
    build_records,
    make_config,
)


def test_final_feature_matrix_has_no_infinite_values() -> None:
    """Generated numeric features must never contain positive or negative infinity."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    for record_index, record in enumerate(result.records):
        for feature_name in result.created_features:
            value = record[feature_name]

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert not math.isinf(value), (
                    f"Infinite value found in {feature_name!r} "
                    f"at record index {record_index}: {value!r}"
                )


def test_final_feature_matrix_has_unique_feature_columns() -> None:
    """Every created feature name must be unique."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    assert len(result.created_features) == len(set(result.created_features))


def test_final_feature_matrix_preserves_record_count() -> None:
    """Feature engineering must preserve one output record per input record."""

    pipeline = build_full_pipeline(make_config())
    records = build_records()

    result = pipeline.run(records)

    assert len(result.records) == len(records)


def test_final_feature_matrix_preserves_record_alignment() -> None:
    """Output records must remain aligned with input chronology."""

    pipeline = build_full_pipeline(make_config())
    records = build_records()

    result = pipeline.run(records)

    for original, transformed in zip(
        records,
        result.records,
        strict=True,
    ):
        assert transformed["date"] == original["date"]


def test_final_feature_schema_is_present_on_every_record() -> None:
    """Every record must expose every created feature column."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    for record_index, record in enumerate(result.records):
        missing_features = [
            feature_name
            for feature_name in result.created_features
            if feature_name not in record
        ]

        assert not missing_features, (
            f"Record index {record_index} is missing features: "
            f"{missing_features}"
        )


def test_mature_rows_have_no_unexpected_missing_features() -> None:
    """Features with sufficient history should be populated in the final record."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    last_record = result.records[-1]

    expected_available_features = (
        "feature__return_1d",
        "feature__return_2d",
        "feature__price_change_1d",
        "feature__price_change_2d",
        "feature__rolling_max_2d",
        "feature__momentum_2d",
        "feature__return_volatility_2d",
        "feature__average_volume_2d",
        "feature__calendar__month",
        "feature__calendar__day_of_week",
    )

    for feature_name in expected_available_features:
        assert feature_name in last_record, (
            f"Expected feature {feature_name!r} is missing "
            "from the final record."
        )

        assert last_record[feature_name] is not None, (
            f"Unexpected missing value for {feature_name!r} "
            "in the final record despite sufficient history."
        )

    expected_rsi_features = [
        feature_name
        for feature_name in result.created_features
        if feature_name.startswith("feature__rsi")
    ]

    assert expected_rsi_features, "No RSI features were generated."

    for feature_name in expected_rsi_features:
        assert last_record[feature_name] is not None, (
            f"Unexpected missing value for RSI feature "
            f"{feature_name!r} in the final record."
        )


def test_early_missing_values_are_limited_to_warmup_dependent_features() -> None:
    """Early history may have missing rolling features but stable features remain available."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    first_record = result.records[0]

    assert first_record["feature__return_1d"] is None
    assert first_record["feature__return_volatility_2d"] is None

    assert first_record["feature__calendar__month"] is not None
    assert first_record["feature__calendar__day_of_week"] is not None


def test_feature_groups_are_represented_in_final_schema() -> None:
    """The final schema must contain features from every Phase 6 group."""

    pipeline = build_full_pipeline(make_config())
    result = pipeline.run(build_records())

    feature_names = result.created_features

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
        ), (
            f"No features found for expected group prefix {prefix!r}"
        )