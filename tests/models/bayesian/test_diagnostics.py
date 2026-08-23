import pytest
import torch

from src.models.bayesian.diagnostics import (
    BayesianDiagnostics,
    BayesianDiagnosticsResult,
)
from src.models.bayesian.model import BayesianPrediction
from src.models.bayesian.uncertainty import UncertaintyResult


def make_prediction(
    *,
    predictions=None,
    probabilities=None,
    confidence=None,
    predictive_entropy=None,
    expected_entropy=None,
    mutual_information=None,
    probability_std=None,
    epistemic_uncertainty=None,
):
    predictions = (
        torch.tensor([0, 1, 1, 0], dtype=torch.long)
        if predictions is None
        else predictions
    )

    probabilities = (
        torch.tensor(
            [
                [0.80, 0.20],
                [0.30, 0.70],
                [0.40, 0.60],
                [0.90, 0.10],
            ],
            dtype=torch.float32,
        )
        if probabilities is None
        else probabilities
    )

    confidence = (
        torch.tensor(
            [0.80, 0.70, 0.60, 0.90],
            dtype=torch.float32,
        )
        if confidence is None
        else confidence
    )

    predictive_entropy = (
        torch.tensor(
            [0.50, 0.60, 0.65, 0.30],
            dtype=torch.float32,
        )
        if predictive_entropy is None
        else predictive_entropy
    )

    expected_entropy = (
        torch.tensor(
            [0.40, 0.45, 0.50, 0.20],
            dtype=torch.float32,
        )
        if expected_entropy is None
        else expected_entropy
    )

    mutual_information = (
        torch.tensor(
            [0.10, 0.15, 0.15, 0.10],
            dtype=torch.float32,
        )
        if mutual_information is None
        else mutual_information
    )

    probability_std = (
        torch.tensor(
            [0.10, 0.20, 0.25, 0.05],
            dtype=torch.float32,
        )
        if probability_std is None
        else probability_std
    )

    normalized_predictive_entropy = (
        predictive_entropy
        / torch.log(
            torch.tensor(
                float(probabilities.shape[1]),
                dtype=predictive_entropy.dtype,
                device=predictive_entropy.device,
            )
        )
    )

    epistemic_uncertainty = (
        torch.tensor(
            [0.12, 0.18, 0.22, 0.08],
            dtype=torch.float32,
        )
        if epistemic_uncertainty is None
        else epistemic_uncertainty
    )

    uncertainty = UncertaintyResult(
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        confidence=confidence,
        probability_std=probability_std,
        normalized_predictive_entropy=normalized_predictive_entropy,
        epistemic_uncertainty=epistemic_uncertainty,
    )

    return BayesianPrediction(
        predictions=predictions,
        probabilities=probabilities,
        confidence=confidence,
        uncertainty=uncertainty,
    )


# ============================================================
# CREATION
# ============================================================


def test_default_creation():
    diagnostics = BayesianDiagnostics()

    assert diagnostics.confidence_threshold == 0.50


def test_custom_creation():
    diagnostics = BayesianDiagnostics(
        confidence_threshold=0.75,
    )

    assert diagnostics.confidence_threshold == 0.75


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        None,
        True,
        [],
    ],
)
def test_invalid_threshold_type_rejected(value):
    with pytest.raises(TypeError):
        BayesianDiagnostics(
            confidence_threshold=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        1.1,
        2,
    ],
)
def test_invalid_threshold_range_rejected(value):
    with pytest.raises(ValueError):
        BayesianDiagnostics(
            confidence_threshold=value,
        )


# ============================================================
# ANALYZE
# ============================================================


def test_analyze_returns_expected_result_type():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    assert isinstance(
        result,
        BayesianDiagnosticsResult,
    )


def test_analyze_contains_expected_fields():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    assert hasattr(result, "n_samples")
    assert hasattr(result, "n_regimes")
    assert hasattr(result, "regime_counts")
    assert hasattr(result, "regime_frequencies")
    assert hasattr(result, "mean_confidence")
    assert hasattr(result, "mean_predictive_entropy")
    assert hasattr(result, "mean_expected_entropy")
    assert hasattr(result, "mean_mutual_information")
    assert hasattr(result, "mean_probability_std")
    assert hasattr(result, "mean_epistemic_uncertainty")
    assert hasattr(result, "low_confidence_count")
    assert hasattr(result, "low_confidence_fraction")
    assert hasattr(result, "low_confidence_mask")


def test_analyze_sample_count():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    assert result.n_samples == 4


def test_analyze_regime_count():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    assert result.n_regimes == 2


def test_analyze_regime_counts():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [2, 2],
        dtype=torch.long,
    )

    assert torch.equal(
        result.regime_counts,
        expected,
    )


def test_analyze_regime_frequencies():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.50, 0.50],
        dtype=torch.float32,
    )

    assert torch.allclose(
        result.regime_frequencies,
        expected,
    )


def test_analyze_mean_confidence():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.80, 0.70, 0.60, 0.90],
        dtype=torch.float32,
    ).mean().item()

    assert result.mean_confidence == pytest.approx(
        expected
    )


def test_analyze_mean_predictive_entropy():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.50, 0.60, 0.65, 0.30],
        dtype=torch.float32,
    ).mean().item()

    assert (
        result.mean_predictive_entropy
        == pytest.approx(expected)
    )


def test_analyze_mean_expected_entropy():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.40, 0.45, 0.50, 0.20],
        dtype=torch.float32,
    ).mean().item()

    assert (
        result.mean_expected_entropy
        == pytest.approx(expected)
    )


def test_analyze_mean_mutual_information():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.10, 0.15, 0.15, 0.10],
        dtype=torch.float32,
    ).mean().item()

    assert (
        result.mean_mutual_information
        == pytest.approx(expected)
    )


def test_analyze_mean_probability_std():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.10, 0.20, 0.25, 0.05],
        dtype=torch.float32,
    ).mean().item()

    assert result.mean_probability_std == pytest.approx(
        expected
    )


def test_analyze_mean_epistemic_uncertainty():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [0.12, 0.18, 0.22, 0.08],
        dtype=torch.float32,
    ).mean().item()

    assert (
        result.mean_epistemic_uncertainty
        == pytest.approx(expected)
    )


def test_analyze_low_confidence_count():
    diagnostics = BayesianDiagnostics(
        confidence_threshold=0.75,
    )

    result = diagnostics.analyze(
        make_prediction()
    )

    # 0.70 and 0.60 are below 0.75
    assert result.low_confidence_count == 2


def test_analyze_low_confidence_fraction():
    diagnostics = BayesianDiagnostics(
        confidence_threshold=0.75,
    )

    result = diagnostics.analyze(
        make_prediction()
    )

    assert (
        result.low_confidence_fraction
        == pytest.approx(0.50)
    )


def test_analyze_low_confidence_mask():
    diagnostics = BayesianDiagnostics(
        confidence_threshold=0.75,
    )

    result = diagnostics.analyze(
        make_prediction()
    )

    expected = torch.tensor(
        [False, True, True, False],
        dtype=torch.bool,
    )

    assert torch.equal(
        result.low_confidence_mask,
        expected,
    )


def test_analyze_uses_custom_threshold():
    diagnostics = BayesianDiagnostics(
        confidence_threshold=0.65,
    )

    result = diagnostics.analyze(
        make_prediction()
    )

    # Only 0.60 is below 0.65
    assert result.low_confidence_count == 1

    assert (
        result.low_confidence_fraction
        == pytest.approx(0.25)
    )


# ============================================================
# SUMMARY
# ============================================================


def test_summary_returns_expected_keys():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.summary(
        make_prediction()
    )

    expected_keys = {
        "n_samples",
        "n_regimes",
        "regime_counts",
        "regime_frequencies",
        "mean_confidence",
        "mean_predictive_entropy",
        "mean_expected_entropy",
        "mean_mutual_information",
        "mean_epistemic_uncertainty",
        "mean_probability_std",
        "low_confidence_count",
        "low_confidence_fraction",
    }

    assert set(result.keys()) == expected_keys


def test_summary_returns_python_friendly_values():
    diagnostics = BayesianDiagnostics()

    result = diagnostics.summary(
        make_prediction()
    )

    assert isinstance(
        result["regime_counts"],
        list,
    )

    assert isinstance(
        result["regime_frequencies"],
        list,
    )

    assert result["n_samples"] == 4
    assert result["n_regimes"] == 2


# ============================================================
# VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "value",
    [
        None,
        "invalid",
        123,
        {},
        [],
    ],
)
def test_analyze_rejects_invalid_prediction_type(value):
    diagnostics = BayesianDiagnostics()

    with pytest.raises(TypeError):
        diagnostics.analyze(value)


def test_analyze_rejects_empty_prediction():
    diagnostics = BayesianDiagnostics()

    prediction = BayesianPrediction(
        predictions=torch.empty(
            0,
            dtype=torch.long,
        ),
        probabilities=torch.empty(
            (0, 2),
            dtype=torch.float32,
        ),
        confidence=torch.empty(
            0,
            dtype=torch.float32,
        ),
        uncertainty=UncertaintyResult(
            predictive_entropy=torch.empty(
                0,
                dtype=torch.float32,
            ),
            expected_entropy=torch.empty(
                0,
                dtype=torch.float32,
            ),
            mutual_information=torch.empty(
                0,
                dtype=torch.float32,
            ),
            confidence=torch.empty(
                0,
                dtype=torch.float32,
            ),
            probability_std=torch.empty(
                0,
                dtype=torch.float32,
            ),
            normalized_predictive_entropy=torch.empty(
                0,
                dtype=torch.float32,
            ),
            epistemic_uncertainty=torch.empty(
                0,
                dtype=torch.float32,
            ),
        ),
    )

    with pytest.raises(ValueError):
        diagnostics.analyze(prediction)