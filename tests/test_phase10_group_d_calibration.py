import pytest

from src.ensemble import (
    EnsembleResult,
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


def test_calibrator_defaults():
    calibrator = TemperatureCalibrator()

    assert calibrator.temperature == pytest.approx(1.0)


def test_temperature_one_preserves_probabilities():
    calibrator = TemperatureCalibrator(
        temperature=1.0,
    )

    original = {
        "risk_on": 0.8,
        "risk_off": 0.2,
    }

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities=original,
        )
    )

    assert result.probabilities["risk_on"] == pytest.approx(
        0.8
    )
    assert result.probabilities["risk_off"] == pytest.approx(
        0.2
    )


def test_calibrator_preserves_participating_models():
    calibrator = TemperatureCalibrator()

    result = calibrator.calibrate(
        make_ensemble_result()
    )

    assert result.participating_models == [
        "hmm",
        "bayesian_nn",
    ]


def test_calibrator_preserves_strategy():
    calibrator = TemperatureCalibrator()

    result = calibrator.calibrate(
        make_ensemble_result()
    )

    assert result.strategy == "weighted"


def test_calibrator_marks_result_as_calibrated():
    calibrator = TemperatureCalibrator()

    result = calibrator.calibrate(
        make_ensemble_result()
    )

    assert result.metadata["calibrated"] is True
    assert (
        result.metadata["calibration_method"]
        == "temperature_scaling"
    )


def test_calibrator_records_temperature():
    calibrator = TemperatureCalibrator(
        temperature=1.5,
    )

    result = calibrator.calibrate(
        make_ensemble_result()
    )

    assert result.metadata["temperature"] == pytest.approx(
        1.5
    )


def test_higher_temperature_softens_distribution():
    calibrator = TemperatureCalibrator(
        temperature=2.0,
    )

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    assert result.probabilities["risk_on"] < 0.9
    assert result.probabilities["risk_on"] > 0.5


def test_lower_temperature_sharpens_distribution():
    calibrator = TemperatureCalibrator(
        temperature=0.5,
    )

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    assert result.probabilities["risk_on"] > 0.8


def test_calibrated_probabilities_sum_to_one():
    calibrator = TemperatureCalibrator(
        temperature=1.7,
    )

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.6,
                "risk_off": 0.3,
                "transitional": 0.1,
            }
        )
    )

    assert sum(result.probabilities.values()) == pytest.approx(
        1.0
    )


def test_calibrator_updates_prediction_from_calibrated_distribution():
    calibrator = TemperatureCalibrator(
        temperature=1.5,
    )

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.7,
                "risk_off": 0.3,
            },
            prediction="risk_off",
        )
    )

    assert result.prediction == "risk_on"


def test_calibrator_rejects_invalid_input_type():
    calibrator = TemperatureCalibrator()

    with pytest.raises(
        TypeError,
        match="EnsembleResult",
    ):
        calibrator.calibrate({})


def test_calibrator_rejects_empty_probabilities():
    calibrator = TemperatureCalibrator()

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
        calibrator.calibrate(result)


def test_calibrator_rejects_non_numeric_temperature():
    with pytest.raises(
        TypeError,
        match="temperature must be numeric",
    ):
        TemperatureCalibrator(
            temperature="invalid",
        )


def test_calibrator_rejects_zero_temperature():
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        TemperatureCalibrator(
            temperature=0.0,
        )


def test_calibrator_rejects_negative_temperature():
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        TemperatureCalibrator(
            temperature=-1.0,
        )


def test_calibrator_handles_zero_probability():
    calibrator = TemperatureCalibrator(
        temperature=1.0,
    )

    result = calibrator.calibrate(
        make_ensemble_result(
            probabilities={
                "risk_on": 1.0,
                "risk_off": 0.0,
            }
        )
    )

    assert result.probabilities["risk_on"] == pytest.approx(
        1.0
    )
    assert result.probabilities["risk_off"] >= 0.0