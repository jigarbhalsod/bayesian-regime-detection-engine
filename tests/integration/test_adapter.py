import pytest

from src.integration.adapter import (
    ModelOutputAdapter,
)
from src.integration.result import (
    IntegrationResult,
)


def test_adapt_plain_value():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "hmm",
        0.75,
    )

    assert result == {
        "model_name": "hmm",
        "prediction": 0.75,
        "probabilities": None,
        "uncertainty": None,
        "metadata": {},
    }


def test_adapt_mapping_with_standard_fields():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "bayesian",
        {
            "prediction": "up",
            "probabilities": {
                "up": 0.7,
                "down": 0.3,
            },
            "uncertainty": 0.12,
            "metadata": {
                "samples": 100,
            },
        },
    )

    assert result == {
        "model_name": "bayesian",
        "prediction": "up",
        "probabilities": {
            "up": 0.7,
            "down": 0.3,
        },
        "uncertainty": 0.12,
        "metadata": {
            "samples": 100,
        },
    }


def test_adapt_value_alias():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "foundation",
        {
            "value": 0.25,
        },
    )

    assert result["prediction"] == 0.25


def test_adapt_probability_alias():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "hmm",
        {
            "probability": 0.8,
        },
    )

    assert result["probabilities"] == 0.8


def test_extra_mapping_values_become_metadata():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "particle_filter",
        {
            "prediction": "down",
            "effective_sample_size": 75.5,
            "particles": 100,
        },
    )

    assert result["prediction"] == "down"

    assert result["metadata"] == {
        "effective_sample_size": 75.5,
        "particles": 100,
    }


def test_supplied_metadata_and_extra_values_merge():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "regime_var",
        {
            "prediction": 1,
            "metadata": {
                "lag_order": 2,
            },
            "aic": 100.5,
        },
    )

    assert result["metadata"] == {
        "lag_order": 2,
        "aic": 100.5,
    }


def test_output_model_name_is_not_used():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "registered_name",
        {
            "model_name": "different_name",
            "prediction": 1,
        },
    )

    assert result["model_name"] == (
        "registered_name"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_empty_model_name_rejected(model_name):
    adapter = ModelOutputAdapter()

    with pytest.raises(ValueError):
        adapter.adapt(
            model_name,
            1,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        None,
        123,
        True,
        [],
    ],
)
def test_invalid_model_name_type_rejected(
    model_name,
):
    adapter = ModelOutputAdapter()

    with pytest.raises(TypeError):
        adapter.adapt(
            model_name,
            1,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "invalid",
        [],
        123,
        True,
    ],
)
def test_invalid_output_metadata_rejected(
    metadata,
):
    adapter = ModelOutputAdapter()

    with pytest.raises(TypeError):
        adapter.adapt(
            "hmm",
            {
                "prediction": 1,
                "metadata": metadata,
            },
        )


def test_none_output_metadata_allowed():
    adapter = ModelOutputAdapter()

    result = adapter.adapt(
        "hmm",
        {
            "prediction": 1,
            "metadata": None,
        },
    )

    assert result["metadata"] == {}


def test_adapt_all():
    adapter = ModelOutputAdapter()

    integration_result = IntegrationResult(
        outputs={
            "hmm": {
                "prediction": "up",
                "probability": 0.8,
            },
            "bayesian": 0.6,
        },
        errors={},
        metadata={},
    )

    result = adapter.adapt_all(
        integration_result
    )

    assert result == {
        "hmm": {
            "model_name": "hmm",
            "prediction": "up",
            "probabilities": 0.8,
            "uncertainty": None,
            "metadata": {},
        },
        "bayesian": {
            "model_name": "bayesian",
            "prediction": 0.6,
            "probabilities": None,
            "uncertainty": None,
            "metadata": {},
        },
    }


def test_adapt_all_ignores_errors():
    adapter = ModelOutputAdapter()

    integration_result = IntegrationResult(
        outputs={
            "hmm": 0.7,
        },
        errors={
            "bayesian": "failed",
        },
        metadata={},
    )

    result = adapter.adapt_all(
        integration_result
    )

    assert tuple(result.keys()) == (
        "hmm",
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "invalid",
        123,
    ],
)
def test_adapt_all_requires_integration_result(
    value,
):
    adapter = ModelOutputAdapter()

    with pytest.raises(TypeError):
        adapter.adapt_all(value)