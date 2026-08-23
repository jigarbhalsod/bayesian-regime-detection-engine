import pytest

from src.uncertainty.result import UncertaintyResult


def test_valid_result():
    result = UncertaintyResult(
        prediction="risk_on",
        uncertainty=0.2,
        confidence=0.8,
        level="medium",
    )

    assert result.prediction == "risk_on"
    assert result.uncertainty == 0.2
    assert result.confidence == 0.8
    assert result.level == "medium"
    assert result.abstained is False


@pytest.mark.parametrize(
    "uncertainty",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_uncertainty_raises_error(uncertainty):
    with pytest.raises(ValueError):
        UncertaintyResult(
            prediction=1.0,
            uncertainty=uncertainty,
            confidence=0.5,
            level="low",
        )


@pytest.mark.parametrize(
    "confidence",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_confidence_raises_error(confidence):
    with pytest.raises(ValueError):
        UncertaintyResult(
            prediction=1.0,
            uncertainty=0.5,
            confidence=confidence,
            level="medium",
        )


def test_invalid_level_raises_error():
    with pytest.raises(ValueError):
        UncertaintyResult(
            prediction=1.0,
            uncertainty=0.5,
            confidence=0.5,
            level="unknown",
        )


def test_abstained_result():
    result = UncertaintyResult(
        prediction=1.0,
        uncertainty=0.8,
        confidence=0.2,
        level="high",
        abstained=True,
    )

    assert result.abstained is True