import pytest
from pydantic import ValidationError

from src.ensemble import (
    UncertaintyLevel,
    UncertaintyResult,
)


def make_result(**overrides):
    data = {
        "prediction": "risk_on",
        "uncertainty_score": 0.25,
        "confidence_score": 0.75,
        "entropy": 0.35,
        "uncertainty_level": UncertaintyLevel.LOW,
        "probabilities": {
            "risk_on": 0.75,
            "risk_off": 0.25,
        },
        "components": {
            "predictive": 0.25,
        },
    }

    data.update(overrides)

    return UncertaintyResult(**data)


def test_uncertainty_level_values():
    assert UncertaintyLevel.LOW.value == "low"
    assert UncertaintyLevel.MEDIUM.value == "medium"
    assert UncertaintyLevel.HIGH.value == "high"


def test_uncertainty_result_creates_successfully():
    result = make_result()

    assert result.prediction == "risk_on"
    assert result.uncertainty_score == pytest.approx(0.25)
    assert result.confidence_score == pytest.approx(0.75)


def test_uncertainty_score_rejects_negative_value():
    with pytest.raises(ValidationError):
        make_result(uncertainty_score=-0.1)


def test_uncertainty_score_rejects_value_above_one():
    with pytest.raises(ValidationError):
        make_result(uncertainty_score=1.1)


def test_confidence_score_rejects_negative_value():
    with pytest.raises(ValidationError):
        make_result(confidence_score=-0.1)


def test_confidence_score_rejects_value_above_one():
    with pytest.raises(ValidationError):
        make_result(confidence_score=1.1)


def test_entropy_rejects_negative_value():
    with pytest.raises(ValidationError):
        make_result(entropy=-0.01)


def test_entropy_rejects_value_above_one():
    with pytest.raises(ValidationError):
        make_result(entropy=1.01)


def test_probability_rejects_negative_value():
    with pytest.raises(
        ValidationError,
        match="cannot be negative",
    ):
        make_result(
            probabilities={
                "risk_on": -0.1,
                "risk_off": 1.1,
            }
        )


def test_probability_rejects_value_above_one():
    with pytest.raises(
        ValidationError,
        match="cannot exceed 1.0",
    ):
        make_result(
            probabilities={
                "risk_on": 1.1,
                "risk_off": 0.0,
            }
        )


def test_probability_must_sum_to_one():
    with pytest.raises(
        ValidationError,
        match="sum to 1.0",
    ):
        make_result(
            probabilities={
                "risk_on": 0.7,
                "risk_off": 0.2,
            }
        )


def test_empty_probabilities_are_allowed():
    result = make_result(
        probabilities={},
    )

    assert result.probabilities == {}


def test_components_reject_negative_value():
    with pytest.raises(
        ValidationError,
        match="cannot be negative",
    ):
        make_result(
            components={
                "predictive": -0.1,
            }
        )


def test_uncertainty_level_properties():
    low = make_result(
        uncertainty_level=UncertaintyLevel.LOW,
    )
    medium = make_result(
        uncertainty_level=UncertaintyLevel.MEDIUM,
    )
    high = make_result(
        uncertainty_level=UncertaintyLevel.HIGH,
    )

    assert low.is_low_uncertainty is True
    assert low.is_medium_uncertainty is False
    assert low.is_high_uncertainty is False

    assert medium.is_low_uncertainty is False
    assert medium.is_medium_uncertainty is True
    assert medium.is_high_uncertainty is False

    assert high.is_low_uncertainty is False
    assert high.is_medium_uncertainty is False
    assert high.is_high_uncertainty is True


def test_optional_metric_properties():
    result = make_result(
        confidence_score=None,
        entropy=None,
    )

    assert result.has_confidence is False
    assert result.has_entropy is False


def test_component_names():
    result = make_result(
        components={
            "predictive": 0.2,
            "model_disagreement": 0.1,
        }
    )

    assert result.component_names == [
        "predictive",
        "model_disagreement",
    ]


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        UncertaintyResult(
            prediction="risk_on",
            uncertainty_score=0.2,
            uncertainty_level=UncertaintyLevel.LOW,
            unexpected_field="invalid",
        )