import pytest
import torch

from src.models.bayesian.uncertainty import (
    BayesianUncertaintyEstimator,
    UncertaintyResult,
)


@pytest.fixture
def estimator():
    return BayesianUncertaintyEstimator()


@pytest.fixture
def samples():
    return torch.tensor(
        [
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.80, 0.10],
            ],
            [
                [0.60, 0.30, 0.10],
                [0.20, 0.70, 0.10],
            ],
            [
                [0.80, 0.10, 0.10],
                [0.15, 0.75, 0.10],
            ],
            [
                [0.75, 0.15, 0.10],
                [0.05, 0.85, 0.10],
            ],
        ],
        dtype=torch.float32,
    )


def test_default_epsilon(estimator):
    assert estimator.epsilon == pytest.approx(1e-8)


def test_custom_epsilon():
    estimator = BayesianUncertaintyEstimator(
        epsilon=1e-6
    )

    assert estimator.epsilon == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "epsilon",
    [
        0,
        -1e-8,
        -1,
    ],
)
def test_invalid_epsilon_value_rejected(epsilon):
    with pytest.raises(ValueError):
        BayesianUncertaintyEstimator(
            epsilon=epsilon
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        "1e-8",
        None,
        True,
    ],
)
def test_invalid_epsilon_type_rejected(epsilon):
    with pytest.raises(TypeError):
        BayesianUncertaintyEstimator(
            epsilon=epsilon
        )


def test_mean_probabilities(estimator, samples):
    probabilities = estimator.mean_probabilities(
        samples
    )

    assert probabilities.shape == (2, 3)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_predictive_entropy_shape_and_range(
    estimator,
    samples,
):
    entropy = estimator.predictive_entropy(samples)

    assert entropy.shape == (2,)
    assert torch.all(entropy >= 0)
    assert torch.all(
        entropy <= torch.log(torch.tensor(3.0))
    )


def test_expected_entropy_shape_and_range(
    estimator,
    samples,
):
    entropy = estimator.expected_entropy(samples)

    assert entropy.shape == (2,)
    assert torch.all(entropy >= 0)
    assert torch.all(
        entropy <= torch.log(torch.tensor(3.0))
    )


def test_mutual_information_shape_and_non_negative(
    estimator,
    samples,
):
    value = estimator.mutual_information(samples)

    assert value.shape == (2,)
    assert torch.all(value >= 0)


def test_confidence(estimator, samples):
    confidence = estimator.confidence(samples)

    expected = torch.tensor(
        [0.7125, 0.7750]
    )

    assert torch.allclose(
        confidence,
        expected,
        atol=1e-6,
    )


def test_probability_std_shape_and_non_negative(
    estimator,
    samples,
):
    std = estimator.probability_std(samples)

    assert std.shape == (2,)
    assert torch.all(std >= 0)


def test_normalized_predictive_entropy_range(
    estimator,
    samples,
):
    value = estimator.normalized_predictive_entropy(
        samples
    )

    assert value.shape == (2,)
    assert torch.all(value >= 0)
    assert torch.all(value <= 1)


def test_epistemic_uncertainty_range(
    estimator,
    samples,
):
    value = estimator.epistemic_uncertainty(samples)

    assert value.shape == (2,)
    assert torch.all(value >= 0)
    assert torch.all(value <= 1)


def test_estimate_returns_result(
    estimator,
    samples,
):
    result = estimator.estimate(samples)

    assert isinstance(result, UncertaintyResult)

    assert result.predictive_entropy.shape == (2,)
    assert result.expected_entropy.shape == (2,)
    assert result.mutual_information.shape == (2,)
    assert result.confidence.shape == (2,)
    assert result.probability_std.shape == (2,)
    assert (
        result.normalized_predictive_entropy.shape
        == (2,)
    )
    assert result.epistemic_uncertainty.shape == (2,)


def test_integer_samples_converted_to_float(
    estimator,
):
    samples = torch.tensor(
        [
            [
                [1, 0],
                [0, 1],
            ],
            [
                [1, 0],
                [0, 1],
            ],
        ],
        dtype=torch.int64,
    )

    probabilities = estimator.mean_probabilities(
        samples
    )

    assert probabilities.dtype == torch.float32


def test_non_tensor_samples_rejected(estimator):
    with pytest.raises(TypeError):
        estimator.mean_probabilities(
            [[[0.5, 0.5]]]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 2),
        (2, 3, 2, 2),
    ],
)
def test_invalid_sample_dimensions_rejected(
    estimator,
    shape,
):
    samples = torch.ones(*shape)

    with pytest.raises(ValueError):
        estimator.mean_probabilities(samples)


def test_single_regime_rejected(estimator):
    samples = torch.ones(
        3,
        2,
        1,
    )

    with pytest.raises(ValueError):
        estimator.mean_probabilities(samples)


def test_negative_probability_rejected(estimator):
    samples = torch.tensor(
        [
            [
                [1.1, -0.1],
            ],
        ]
    )

    with pytest.raises(ValueError):
        estimator.mean_probabilities(samples)


def test_probabilities_not_summing_to_one_rejected(
    estimator,
):
    samples = torch.tensor(
        [
            [
                [0.2, 0.2, 0.2],
            ],
        ]
    )

    with pytest.raises(ValueError):
        estimator.mean_probabilities(samples)


def test_non_finite_values_rejected(estimator):
    samples = torch.tensor(
        [
            [
                [float("nan"), 0.0],
            ],
        ]
    )

    with pytest.raises(ValueError):
        estimator.mean_probabilities(samples)


def test_deterministic_samples_have_zero_epistemic_uncertainty(
    estimator,
):
    samples = torch.tensor(
        [
            [
                [0.8, 0.1, 0.1],
            ],
            [
                [0.8, 0.1, 0.1],
            ],
            [
                [0.8, 0.1, 0.1],
            ],
        ]
    )

    assert torch.allclose(
        estimator.mutual_information(samples),
        torch.zeros(1),
        atol=1e-6,
    )


def test_uniform_mean_distribution_has_high_entropy(
    estimator,
):
    samples = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
            ],
        ]
    )

    normalized_entropy = (
        estimator.normalized_predictive_entropy(
            samples
        )
    )

    assert torch.allclose(
        normalized_entropy,
        torch.ones(1),
        atol=1e-6,
    )