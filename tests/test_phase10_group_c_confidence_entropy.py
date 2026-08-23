import math

import pytest

from src.ensemble import (
    ConfidenceEntropyAnalyzer,
    EnsembleResult,
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


def test_analyzer_defaults():
    analyzer = ConfidenceEntropyAnalyzer()

    assert analyzer.low_threshold == pytest.approx(0.33)
    assert analyzer.high_threshold == pytest.approx(0.66)


def test_analyzer_calculates_confidence():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(make_ensemble_result())

    assert result.confidence_score == pytest.approx(0.8)


def test_analyzer_calculates_normalized_entropy():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(make_ensemble_result())

    expected = (
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        / math.log(2)
    )

    assert result.entropy == pytest.approx(expected)
    assert result.uncertainty_score == pytest.approx(expected)


def test_analyzer_preserves_prediction():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(prediction="risk_off")
    )

    assert result.prediction == "risk_off"


def test_analyzer_preserves_probabilities():
    analyzer = ConfidenceEntropyAnalyzer()

    probabilities = {
        "risk_on": 0.7,
        "risk_off": 0.3,
    }

    result = analyzer.analyze(
        make_ensemble_result(probabilities=probabilities)
    )

    assert result.probabilities == probabilities


def test_analyzer_calculates_confidence_margin():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    assert result.components["confidence_margin"] == pytest.approx(
        0.6
    )


def test_analyzer_handles_three_classes():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.6,
                "risk_off": 0.3,
                "transitional": 0.1,
            }
        )
    )

    assert result.confidence_score == pytest.approx(0.6)
    assert result.components["confidence_margin"] == pytest.approx(
        0.3
    )
    assert 0.0 <= result.entropy <= 1.0


def test_analyzer_uniform_distribution_has_maximum_entropy():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.5,
                "risk_off": 0.5,
            }
        )
    )

    assert result.entropy == pytest.approx(1.0)
    assert result.uncertainty_level == UncertaintyLevel.HIGH


def test_analyzer_deterministic_distribution_has_zero_entropy():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 1.0,
                "risk_off": 0.0,
            }
        )
    )

    assert result.entropy == pytest.approx(0.0)
    assert result.uncertainty_level == UncertaintyLevel.LOW


def test_analyzer_classifies_medium_uncertainty():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.MEDIUM


def test_analyzer_uses_custom_thresholds():
    analyzer = ConfidenceEntropyAnalyzer(
        low_threshold=0.80,
        high_threshold=0.95,
    )

    result = analyzer.analyze(
        make_ensemble_result(
            probabilities={
                "risk_on": 0.8,
                "risk_off": 0.2,
            }
        )
    )

    assert result.uncertainty_level == UncertaintyLevel.LOW


def test_analyzer_includes_entropy_component():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(make_ensemble_result())

    assert result.components["entropy"] == pytest.approx(
        result.entropy
    )


def test_analyzer_includes_metadata():
    analyzer = ConfidenceEntropyAnalyzer()

    result = analyzer.analyze(make_ensemble_result())

    assert result.metadata["analyzer"] == "confidence_entropy"
    assert result.metadata["class_count"] == 2
    assert result.metadata["confidence_margin"] == pytest.approx(
        0.6
    )


def test_analyzer_rejects_invalid_input_type():
    analyzer = ConfidenceEntropyAnalyzer()

    with pytest.raises(
        TypeError,
        match="EnsembleResult",
    ):
        analyzer.analyze({})


def test_analyzer_rejects_empty_probabilities():
    analyzer = ConfidenceEntropyAnalyzer()

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
        analyzer.analyze(result)


def test_analyzer_rejects_invalid_low_threshold_type():
    with pytest.raises(
        TypeError,
        match="low_threshold must be numeric",
    ):
        ConfidenceEntropyAnalyzer(
            low_threshold="invalid",
        )


def test_analyzer_rejects_invalid_high_threshold_type():
    with pytest.raises(
        TypeError,
        match="high_threshold must be numeric",
    ):
        ConfidenceEntropyAnalyzer(
            high_threshold="invalid",
        )


def test_analyzer_rejects_low_threshold_below_zero():
    with pytest.raises(
        ValueError,
        match="low_threshold must be between",
    ):
        ConfidenceEntropyAnalyzer(
            low_threshold=-0.1,
        )


def test_analyzer_rejects_high_threshold_above_one():
    with pytest.raises(
        ValueError,
        match="high_threshold must be between",
    ):
        ConfidenceEntropyAnalyzer(
            high_threshold=1.1,
        )


def test_analyzer_rejects_inverted_thresholds():
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        ConfidenceEntropyAnalyzer(
            low_threshold=0.8,
            high_threshold=0.3,
        )