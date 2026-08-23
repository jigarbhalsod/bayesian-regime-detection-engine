import pytest

from src.uncertainty.config import UncertaintyConfig
from src.uncertainty.pipeline import UncertaintyPipeline


def test_single_prediction_has_zero_uncertainty():
    pipeline = UncertaintyPipeline()

    result = pipeline.predict([10.0])

    assert result.prediction == pytest.approx(10.0)
    assert result.uncertainty == pytest.approx(0.0)
    assert result.confidence == pytest.approx(1.0)
    assert result.level == "low"
    assert result.abstained is False


def test_identical_predictions_have_zero_uncertainty():
    pipeline = UncertaintyPipeline()

    result = pipeline.predict([10.0, 10.0, 10.0])

    assert result.prediction == pytest.approx(10.0)
    assert result.uncertainty == pytest.approx(0.0)
    assert result.confidence == pytest.approx(1.0)
    assert result.level == "low"


def test_pipeline_returns_ensemble_prediction():
    pipeline = UncertaintyPipeline()

    result = pipeline.predict([10.0, 20.0, 30.0])

    assert result.prediction == pytest.approx(20.0)
    assert 0.0 <= result.uncertainty <= 1.0
    assert result.confidence == pytest.approx(
        1.0 - result.uncertainty
    )


def test_explicit_weights_affect_prediction():
    pipeline = UncertaintyPipeline()

    result = pipeline.predict(
        predictions=[10.0, 20.0, 30.0],
        weights=[0.2, 0.3, 0.5],
    )

    assert result.prediction == pytest.approx(23.0)


def test_empty_predictions_raise_error():
    pipeline = UncertaintyPipeline()

    with pytest.raises(ValueError):
        pipeline.predict([])


def test_low_uncertainty_level():
    config = UncertaintyConfig(
        low_uncertainty_threshold=0.4,
        high_uncertainty_threshold=0.8,
    )

    pipeline = UncertaintyPipeline(config=config)

    result = pipeline.predict([10.0, 11.0, 10.0])

    assert result.level == "low"


def test_high_uncertainty_can_abstain():
    config = UncertaintyConfig(
        low_uncertainty_threshold=0.01,
        high_uncertainty_threshold=0.20,
        abstain_on_high_uncertainty=True,
    )

    pipeline = UncertaintyPipeline(config=config)

    result = pipeline.predict([0.0, 100.0])

    assert result.level == "high"
    assert result.abstained is True


def test_high_uncertainty_can_continue_when_abstention_disabled():
    config = UncertaintyConfig(
        low_uncertainty_threshold=0.01,
        high_uncertainty_threshold=0.20,
        abstain_on_high_uncertainty=False,
    )

    pipeline = UncertaintyPipeline(config=config)

    result = pipeline.predict([0.0, 100.0])

    assert result.level == "high"
    assert result.abstained is False


def test_medium_uncertainty_level():
    config = UncertaintyConfig(
        low_uncertainty_threshold=0.10,
        high_uncertainty_threshold=0.40,
    )

    pipeline = UncertaintyPipeline(config=config)

    result = pipeline.predict([0.0, 10.0, 20.0])

    assert result.level == "medium"