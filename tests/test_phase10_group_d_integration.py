import pytest

from src.ensemble import (
    ConfidenceEntropyAnalyzer,
    EnsembleResult,
    EnsembleUncertaintyIntegrator,
    PredictiveUncertaintyEstimator,
    TemperatureCalibrator,
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
        participating_models=["hmm", "bayesian_nn"],
        strategy="weighted",
    )


def test_integrator_defaults():
    integrator = EnsembleUncertaintyIntegrator()

    assert isinstance(
        integrator.predictive_estimator,
        PredictiveUncertaintyEstimator,
    )

    assert isinstance(
        integrator.confidence_analyzer,
        ConfidenceEntropyAnalyzer,
    )

    assert integrator.calibrator is None


def test_integrator_returns_unified_result():
    integrator = EnsembleUncertaintyIntegrator()

    result = integrator.integrate(
        make_ensemble_result()
    )

    assert result.ensemble_result.prediction == "risk_on"
    assert result.predictive_uncertainty is not None
    assert result.confidence_entropy is not None


def test_integrator_runs_predictive_uncertainty():
    integrator = EnsembleUncertaintyIntegrator()

    result = integrator.integrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    assert (
        result.predictive_uncertainty.uncertainty_score
        == pytest.approx(0.2)
    )

    assert (
        result.predictive_uncertainty.confidence_score
        == pytest.approx(0.8)
    )


def test_integrator_runs_confidence_entropy_analysis():
    integrator = EnsembleUncertaintyIntegrator()

    result = integrator.integrate(
        make_ensemble_result()
    )

    assert (
        result.confidence_entropy.confidence_score
        == pytest.approx(0.8)
    )

    assert result.confidence_entropy.entropy is not None


def test_integrator_without_calibrator_is_not_calibrated():
    integrator = EnsembleUncertaintyIntegrator()

    result = integrator.integrate(
        make_ensemble_result()
    )

    assert result.calibrated is False


def test_integrator_with_calibrator_marks_calibrated():
    integrator = EnsembleUncertaintyIntegrator(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    result = integrator.integrate(
        make_ensemble_result()
    )

    assert result.calibrated is True
    assert result.ensemble_result.metadata["calibrated"] is True


def test_integrator_uses_calibrated_probabilities():
    integrator = EnsembleUncertaintyIntegrator(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    result = integrator.integrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    assert (
        result.ensemble_result.probabilities["risk_on"]
        < 0.9
    )


def test_predictive_uncertainty_uses_processed_result():
    integrator = EnsembleUncertaintyIntegrator(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    result = integrator.integrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    max_probability = max(
        result.ensemble_result.probabilities.values()
    )

    assert (
        result.predictive_uncertainty.confidence_score
        == pytest.approx(max_probability)
    )


def test_confidence_entropy_uses_processed_result():
    integrator = EnsembleUncertaintyIntegrator(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    result = integrator.integrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    max_probability = max(
        result.ensemble_result.probabilities.values()
    )

    assert (
        result.confidence_entropy.confidence_score
        == pytest.approx(max_probability)
    )


def test_integrator_preserves_models_and_strategy():
    integrator = EnsembleUncertaintyIntegrator(
        calibrator=TemperatureCalibrator()
    )

    result = integrator.integrate(
        make_ensemble_result()
    )

    assert result.ensemble_result.participating_models == [
        "hmm",
        "bayesian_nn",
    ]

    assert result.ensemble_result.strategy == "weighted"


def test_integrator_rejects_invalid_input_type():
    integrator = EnsembleUncertaintyIntegrator()

    with pytest.raises(
        TypeError,
        match="EnsembleResult",
    ):
        integrator.integrate({})


def test_integrator_rejects_invalid_predictive_estimator():
    with pytest.raises(
        TypeError,
        match="predictive_estimator",
    ):
        EnsembleUncertaintyIntegrator(
            predictive_estimator="invalid",
        )


def test_integrator_rejects_invalid_confidence_analyzer():
    with pytest.raises(
        TypeError,
        match="confidence_analyzer",
    ):
        EnsembleUncertaintyIntegrator(
            confidence_analyzer="invalid",
        )


def test_integrator_rejects_invalid_calibrator():
    with pytest.raises(
        TypeError,
        match="calibrator",
    ):
        EnsembleUncertaintyIntegrator(
            calibrator="invalid",
        )


def test_integrator_accepts_custom_components():
    predictive_estimator = (
        PredictiveUncertaintyEstimator(
            low_threshold=0.10,
            high_threshold=0.40,
        )
    )

    confidence_analyzer = (
        ConfidenceEntropyAnalyzer(
            low_threshold=0.20,
            high_threshold=0.80,
        )
    )

    calibrator = TemperatureCalibrator(
        temperature=1.5,
    )

    integrator = EnsembleUncertaintyIntegrator(
        predictive_estimator=predictive_estimator,
        confidence_analyzer=confidence_analyzer,
        calibrator=calibrator,
    )

    assert (
        integrator.predictive_estimator
        is predictive_estimator
    )

    assert (
        integrator.confidence_analyzer
        is confidence_analyzer
    )

    assert integrator.calibrator is calibrator


def test_integrator_handles_three_class_distribution():
    integrator = EnsembleUncertaintyIntegrator()

    result = integrator.integrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.5,
                "risk_off": 0.3,
                "transitional": 0.2,
            }
        )
    )

    assert (
        result.predictive_uncertainty.confidence_score
        == pytest.approx(0.5)
    )

    assert (
        result.ensemble_result.probabilities
        == {
            "risk_on": 0.5,
            "risk_off": 0.3,
            "transitional": 0.2,
        }
    )