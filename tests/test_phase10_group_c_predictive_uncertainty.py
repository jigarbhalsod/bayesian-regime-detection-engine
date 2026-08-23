import pytest

from src.ensemble import (
    EnsembleResult,
    PredictiveUncertaintyEstimator,
    UncertaintyLevel,
)


def make_ensemble_result(
    probabilities=None,
    prediction="risk_on",
):
    if probabilities is None:
        probabilities = {
            "risk_on": 0.8,
            "risk_off": 0.2,
        }

    return EnsembleResult(
        prediction=prediction,
        probabilities=probabilities,
        participating_models=["hmm"],
        strategy="weighted",
    )


def test_estimator_defaults():
    estimator = PredictiveUncertaintyEstimator()

    assert estimator.low_threshold == pytest.approx(0.25)
    assert estimator.high_threshold == pytest.approx(0.50)


def test_estimator_calculates_uncertainty():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(make_ensemble_result())

    assert result.uncertainty_score == pytest.approx(0.2)
    assert result.confidence_score == pytest.approx(0.8)


def test_estimator_preserves_prediction():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(
        make_ensemble_result(prediction="risk_on")
    )

    assert result.prediction == "risk_on"


def test_estimator_preserves_probabilities():
    estimator = PredictiveUncertaintyEstimator()

    probabilities = {
        "risk_on": 0.7,
        "risk_off": 0.3,
    }

    result = estimator.estimate(
        make_ensemble_result(probabilities=probabilities)
    )

    assert result.probabilities == probabilities


def test_estimator_classifies_low_uncertainty():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.LOW


def test_estimator_classifies_medium_uncertainty():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.7,
                "risk_off": 0.3,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.MEDIUM


def test_estimator_classifies_high_uncertainty():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.4,
                "risk_off": 0.3,
                "transitional": 0.3,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.HIGH


def test_estimator_uses_custom_thresholds():
    estimator = PredictiveUncertaintyEstimator(
        low_threshold=0.10,
        high_threshold=0.30,
    )

    result = estimator.estimate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.75,
                "risk_off": 0.25,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.MEDIUM


def test_estimator_includes_predictive_component():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(make_ensemble_result())

    assert result.components["predictive"] == pytest.approx(0.2)


def test_estimator_includes_metadata():
    estimator = PredictiveUncertaintyEstimator()

    result = estimator.estimate(make_ensemble_result())

    assert result.metadata["estimator"] == "predictive"
    assert result.metadata["max_probability"] == pytest.approx(0.8)


def test_estimator_rejects_invalid_input_type():
    estimator = PredictiveUncertaintyEstimator()

    with pytest.raises(
        TypeError,
        match="EnsembleResult",
    ):
        estimator.estimate({})


def test_estimator_rejects_empty_probabilities():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={},
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="empty probabilities",
    ):
        estimator.estimate(result)


def test_estimator_rejects_negative_probability():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": -0.1,
            "risk_off": 1.1,
        },
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        estimator.estimate(result)


def test_estimator_rejects_probability_above_one():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": 1.1,
            "risk_off": -0.1,
        },
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed 1.0",
    ):
        estimator.estimate(result)


def test_estimator_rejects_non_normalized_probabilities():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": 0.6,
            "risk_off": 0.2,
        },
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        estimator.estimate(result)


def test_estimator_rejects_probability_above_one():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": 1.1,
            "risk_off": -0.1,
        },
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed 1.0",
    ):
        estimator.estimate(result)


def test_estimator_rejects_non_normalized_probabilities():
    estimator = PredictiveUncertaintyEstimator()

    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": 0.6,
            "risk_off": 0.2,
        },
        participating_models=["hmm"],
        strategy="weighted",
    )

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        estimator.estimate(result)


def test_estimator_rejects_invalid_low_threshold_type():
    with pytest.raises(
        TypeError,
        match="low_threshold must be numeric",
    ):
        PredictiveUncertaintyEstimator(
            low_threshold="invalid",
        )


def test_estimator_rejects_invalid_high_threshold_type():
    with pytest.raises(
        TypeError,
        match="high_threshold must be numeric",
    ):
        PredictiveUncertaintyEstimator(
            high_threshold="invalid",
        )


def test_estimator_rejects_low_threshold_below_zero():
    with pytest.raises(
        ValueError,
        match="low_threshold must be between",
    ):
        PredictiveUncertaintyEstimator(
            low_threshold=-0.1,
        )


def test_estimator_rejects_high_threshold_above_one():
    with pytest.raises(
        ValueError,
        match="high_threshold must be between",
    ):
        PredictiveUncertaintyEstimator(
            high_threshold=1.1,
        )


def test_estimator_rejects_inverted_thresholds():
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        PredictiveUncertaintyEstimator(
            low_threshold=0.7,
            high_threshold=0.3,
        )