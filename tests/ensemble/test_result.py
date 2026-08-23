import math

import pytest

from src.ensemble.result import EnsembleResult


def test_minimal_result():
    result = EnsembleResult(
        prediction="risk_on",
        confidence=0.8,
        uncertainty=0.2,
    )

    assert result.prediction == "risk_on"
    assert result.confidence == 0.8
    assert result.uncertainty == 0.2
    assert result.model_weights == {}
    assert result.model_outputs == {}
    assert result.metadata == {}


def test_complete_result():
    result = EnsembleResult(
        prediction="risk_off",
        confidence=0.75,
        uncertainty=0.25,
        model_weights={
            "hmm": 0.4,
            "bayesian": 0.6,
        },
        model_outputs={
            "hmm": "risk_off",
            "bayesian": "risk_off",
        },
        metadata={
            "method": "weighted_average",
        },
    )

    assert result.model_weights == {
        "hmm": 0.4,
        "bayesian": 0.6,
    }

    assert result.model_outputs["hmm"] == "risk_off"
    assert result.metadata["method"] == (
        "weighted_average"
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "confidence",
        "uncertainty",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_probability_values_rejected(
    field_name,
    value,
):
    kwargs = {
        "prediction": "risk_on",
        "confidence": 0.5,
        "uncertainty": 0.5,
        field_name: value,
    }

    with pytest.raises(ValueError):
        EnsembleResult(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "confidence",
        "uncertainty",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        "0.5",
        True,
    ],
)
def test_invalid_probability_types_rejected(
    field_name,
    value,
):
    kwargs = {
        "prediction": "risk_on",
        "confidence": 0.5,
        "uncertainty": 0.5,
        field_name: value,
    }

    with pytest.raises(TypeError):
        EnsembleResult(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "model_outputs",
        "metadata",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "invalid",
    ],
)
def test_invalid_mapping_types_rejected(
    field_name,
    value,
):
    kwargs = {
        "prediction": "risk_on",
        "confidence": 0.5,
        "uncertainty": 0.5,
        field_name: value,
    }

    with pytest.raises(TypeError):
        EnsembleResult(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "invalid",
    ],
)
def test_invalid_model_weights_mapping_rejected(
    value,
):
    with pytest.raises(TypeError):
        EnsembleResult(
            prediction="risk_on",
            confidence=0.5,
            uncertainty=0.5,
            model_weights=value,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_empty_model_weight_name_rejected(
    model_name,
):
    with pytest.raises(ValueError):
        EnsembleResult(
            prediction="risk_on",
            confidence=0.5,
            uncertainty=0.5,
            model_weights={
                model_name: 0.5,
            },
        )


@pytest.mark.parametrize(
    "model_name",
    [
        None,
        123,
        True,
    ],
)
def test_invalid_model_weight_name_type_rejected(
    model_name,
):
    with pytest.raises(TypeError):
        EnsembleResult(
            prediction="risk_on",
            confidence=0.5,
            uncertainty=0.5,
            model_weights={
                model_name: 0.5,
            },
        )


@pytest.mark.parametrize(
    "weight",
    [
        -0.1,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_model_weight_values_rejected(
    weight,
):
    with pytest.raises(ValueError):
        EnsembleResult(
            prediction="risk_on",
            confidence=0.5,
            uncertainty=0.5,
            model_weights={
                "hmm": weight,
            },
        )


@pytest.mark.parametrize(
    "weight",
    [
        None,
        "0.5",
        True,
    ],
)
def test_invalid_model_weight_types_rejected(
    weight,
):
    with pytest.raises(TypeError):
        EnsembleResult(
            prediction="risk_on",
            confidence=0.5,
            uncertainty=0.5,
            model_weights={
                "hmm": weight,
            },
        )


def test_to_dict():
    result = EnsembleResult(
        prediction="risk_on",
        confidence=0.8,
        uncertainty=0.2,
        model_weights={
            "hmm": 1.0,
        },
        model_outputs={
            "hmm": "risk_on",
        },
        metadata={
            "source": "test",
        },
    )

    assert result.to_dict() == {
        "prediction": "risk_on",
        "confidence": 0.8,
        "uncertainty": 0.2,
        "model_weights": {
            "hmm": 1.0,
        },
        "model_outputs": {
            "hmm": "risk_on",
        },
        "metadata": {
            "source": "test",
        },
    }


def test_result_is_frozen():
    result = EnsembleResult(
        prediction="risk_on",
        confidence=0.8,
        uncertainty=0.2,
    )

    with pytest.raises(Exception):
        result.confidence = 0.9