import pytest

from src.ensemble import (
    ConfidenceEntropyAnalyzer,
    EnsembleResult,
    Phase10Pipeline,
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
        participating_models=[
            "hmm",
            "bayesian_nn",
            "var",
        ],
        strategy="weighted",
    )


def test_pipeline_initializes_with_defaults():
    pipeline = Phase10Pipeline()

    assert isinstance(
        pipeline.integrator.predictive_estimator,
        PredictiveUncertaintyEstimator,
    )

    assert isinstance(
        pipeline.integrator.confidence_analyzer,
        ConfidenceEntropyAnalyzer,
    )

    assert pipeline.integrator.calibrator is None


def test_pipeline_runs_end_to_end_without_calibration():
    pipeline = Phase10Pipeline()

    output = pipeline.run(make_ensemble_result())

    assert output.calibrated is False
    assert output.ensemble_result.prediction == "risk_on"
    assert (
        output.predictive_uncertainty.confidence_score
        == pytest.approx(0.8)
    )
    assert output.confidence_entropy.entropy is not None


def test_pipeline_runs_end_to_end_with_calibration():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    assert output.calibrated is True
    assert output.ensemble_result.metadata["calibrated"] is True
    assert output.ensemble_result.probabilities["risk_on"] < 0.9


def test_pipeline_preserves_models():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator()
    )

    output = pipeline.run(make_ensemble_result())

    assert output.ensemble_result.participating_models == [
        "hmm",
        "bayesian_nn",
        "var",
    ]


def test_pipeline_preserves_strategy():
    pipeline = Phase10Pipeline()

    output = pipeline.run(make_ensemble_result())

    assert output.ensemble_result.strategy == "weighted"


def test_pipeline_predictive_uncertainty_matches_output():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator(
            temperature=1.5,
        )
    )

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    max_probability = max(
        output.ensemble_result.probabilities.values()
    )

    expected_uncertainty = 1.0 - max_probability

    assert (
        output.predictive_uncertainty.uncertainty_score
        == pytest.approx(expected_uncertainty)
    )


def test_pipeline_confidence_matches_output():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator(
            temperature=1.5,
        )
    )

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    max_probability = max(
        output.ensemble_result.probabilities.values()
    )

    assert (
        output.confidence_entropy.confidence_score
        == pytest.approx(max_probability)
    )


def test_pipeline_handles_three_class_distribution():
    pipeline = Phase10Pipeline()

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.5,
                "risk_off": 0.3,
                "transitional": 0.2,
            }
        )
    )

    assert len(output.ensemble_result.probabilities) == 3
    assert (
        output.predictive_uncertainty.confidence_score
        == pytest.approx(0.5)
    )
    assert 0.0 <= output.confidence_entropy.entropy <= 1.0


def test_pipeline_handles_deterministic_distribution():
    pipeline = Phase10Pipeline()

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 1.0,
                "risk_off": 0.0,
            }
        )
    )

    assert (
        output.predictive_uncertainty.uncertainty_score
        == pytest.approx(0.0)
    )
    assert (
        output.confidence_entropy.entropy
        == pytest.approx(0.0)
    )


def test_pipeline_accepts_custom_components():
    predictive_estimator = PredictiveUncertaintyEstimator(
        low_threshold=0.1,
        high_threshold=0.4,
    )

    confidence_analyzer = ConfidenceEntropyAnalyzer(
        low_threshold=0.2,
        high_threshold=0.8,
    )

    calibrator = TemperatureCalibrator(
        temperature=1.5,
    )

    pipeline = Phase10Pipeline(
        predictive_estimator=predictive_estimator,
        confidence_analyzer=confidence_analyzer,
        calibrator=calibrator,
    )

    assert (
        pipeline.integrator.predictive_estimator
        is predictive_estimator
    )
    assert (
        pipeline.integrator.confidence_analyzer
        is confidence_analyzer
    )
    assert pipeline.integrator.calibrator is calibrator


def test_pipeline_rejects_invalid_input():
    pipeline = Phase10Pipeline()

    with pytest.raises(
        TypeError,
        match="EnsembleResult",
    ):
        pipeline.run({})


def test_pipeline_calibration_preserves_normalization():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.6,
                "risk_off": 0.3,
                "transitional": 0.1,
            }
        )
    )

    assert sum(
        output.ensemble_result.probabilities.values()
    ) == pytest.approx(1.0)


def test_pipeline_uncertainty_results_reference_final_prediction():
    pipeline = Phase10Pipeline(
        calibrator=TemperatureCalibrator(
            temperature=2.0,
        )
    )

    output = pipeline.run(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.7,
                "risk_off": 0.3,
            },
            prediction="risk_off",
        )
    )

    final_prediction = output.ensemble_result.prediction

    assert (
        output.predictive_uncertainty.prediction
        == final_prediction
    )
    assert (
        output.confidence_entropy.prediction
        == final_prediction
    )