import math

import pytest

from src.validation.bootstrap import (
    BootstrapConfidenceInterval,
    BootstrapConfidenceIntervalEstimator,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_configuration():
    estimator = BootstrapConfidenceIntervalEstimator()

    assert estimator.n_resamples == 1000
    assert estimator.confidence_level == 0.95
    assert estimator.random_state is None


def test_custom_configuration():
    estimator = BootstrapConfidenceIntervalEstimator(
        n_resamples=500,
        confidence_level=0.90,
        random_state=42,
    )

    assert estimator.n_resamples == 500
    assert estimator.confidence_level == 0.90
    assert estimator.random_state == 42


@pytest.mark.parametrize(
    "n_resamples",
    [
        0,
        1,
        -1,
        10.5,
        "invalid",
        None,
        True,
    ],
)
def test_invalid_n_resamples(n_resamples):
    with pytest.raises((TypeError, ValueError)):
        BootstrapConfidenceIntervalEstimator(
            n_resamples=n_resamples
        )


@pytest.mark.parametrize(
    "confidence_level",
    [
        0,
        1,
        -0.1,
        1.1,
        "invalid",
        None,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_confidence_level(confidence_level):
    with pytest.raises((TypeError, ValueError)):
        BootstrapConfidenceIntervalEstimator(
            confidence_level=confidence_level
        )


@pytest.mark.parametrize(
    "random_state",
    [
        1.5,
        "invalid",
        True,
        [],
    ],
)
def test_invalid_random_state(random_state):
    with pytest.raises(TypeError):
        BootstrapConfidenceIntervalEstimator(
            random_state=random_state
        )


# ============================================================
# Result Tests
# ============================================================


def test_estimate_returns_result():
    estimator = BootstrapConfidenceIntervalEstimator(
        random_state=42
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0]
    )

    assert isinstance(
        result,
        BootstrapConfidenceInterval,
    )


def test_result_sample_size():
    estimator = BootstrapConfidenceIntervalEstimator(
        random_state=42
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert result.sample_size == 4


def test_original_estimate():
    estimator = BootstrapConfidenceIntervalEstimator(
        random_state=42
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0]
    )

    assert result.original_estimate == 2.0


def test_result_configuration_is_preserved():
    estimator = BootstrapConfidenceIntervalEstimator(
        n_resamples=100,
        confidence_level=0.90,
        random_state=42,
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0]
    )

    assert result.n_resamples == 100
    assert result.confidence_level == 0.90


# ============================================================
# Bootstrap Behavior Tests
# ============================================================


def test_reproducibility_with_random_state():
    samples = [1.0, 2.0, 3.0, 4.0]

    estimator_a = BootstrapConfidenceIntervalEstimator(
        n_resamples=100,
        random_state=42,
    )

    estimator_b = BootstrapConfidenceIntervalEstimator(
        n_resamples=100,
        random_state=42,
    )

    result_a = estimator_a.estimate(samples)
    result_b = estimator_b.estimate(samples)

    assert result_a == result_b


def test_bootstrap_mean_is_reasonable():
    estimator = BootstrapConfidenceIntervalEstimator(
        n_resamples=2000,
        random_state=42,
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    assert math.isclose(
        result.bootstrap_mean,
        result.original_estimate,
        abs_tol=0.15,
    )


def test_standard_error_is_non_negative():
    estimator = BootstrapConfidenceIntervalEstimator(
        random_state=42
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert result.standard_error >= 0.0


def test_confidence_bounds_are_ordered():
    estimator = BootstrapConfidenceIntervalEstimator(
        random_state=42
    )

    result = estimator.estimate(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert result.lower_bound <= result.upper_bound


def test_constant_samples_have_zero_standard_error():
    estimator = BootstrapConfidenceIntervalEstimator(
        n_resamples=100,
        random_state=42,
    )

    result = estimator.estimate(
        [5.0, 5.0, 5.0]
    )

    assert result.original_estimate == 5.0
    assert result.bootstrap_mean == 5.0
    assert result.standard_error == 0.0
    assert result.lower_bound == 5.0
    assert result.upper_bound == 5.0


def test_single_sample_is_supported():
    estimator = BootstrapConfidenceIntervalEstimator(
        n_resamples=100,
        random_state=42,
    )

    result = estimator.estimate([3.0])

    assert result.sample_size == 1
    assert result.original_estimate == 3.0
    assert result.bootstrap_mean == 3.0
    assert result.standard_error == 0.0
    assert result.lower_bound == 3.0
    assert result.upper_bound == 3.0


# ============================================================
# Confidence Level Tests
# ============================================================


def test_wider_confidence_level_not_narrower():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    result_90 = (
        BootstrapConfidenceIntervalEstimator(
            n_resamples=1000,
            confidence_level=0.90,
            random_state=42,
        ).estimate(samples)
    )

    result_99 = (
        BootstrapConfidenceIntervalEstimator(
            n_resamples=1000,
            confidence_level=0.99,
            random_state=42,
        ).estimate(samples)
    )

    width_90 = (
        result_90.upper_bound
        - result_90.lower_bound
    )

    width_99 = (
        result_99.upper_bound
        - result_99.lower_bound
    )

    assert width_99 >= width_90


# ============================================================
# Input Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "samples",
    [
        [],
    ],
)
def test_empty_samples(samples):
    estimator = BootstrapConfidenceIntervalEstimator()

    with pytest.raises(ValueError):
        estimator.estimate(samples)


@pytest.mark.parametrize(
    "samples",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_samples_type(samples):
    estimator = BootstrapConfidenceIntervalEstimator()

    with pytest.raises(TypeError):
        estimator.estimate(samples)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        None,
        True,
        [],
    ],
)
def test_invalid_sample_value_type(value):
    estimator = BootstrapConfidenceIntervalEstimator()

    with pytest.raises(TypeError):
        estimator.estimate(
            [1.0, value]
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_non_finite_sample_values(value):
    estimator = BootstrapConfidenceIntervalEstimator()

    with pytest.raises(ValueError):
        estimator.estimate(
            [1.0, value]
        )


# ============================================================
# Helper Tests
# ============================================================


def test_mean_helper():
    assert (
        BootstrapConfidenceIntervalEstimator._mean(
            [1.0, 2.0, 3.0]
        )
        == 2.0
    )


def test_sample_standard_deviation_helper():
    result = (
        BootstrapConfidenceIntervalEstimator
        ._sample_standard_deviation(
            [1.0, 3.0, 5.0],
            3.0,
        )
    )

    assert result == 2.0


def test_sample_standard_deviation_single_value():
    result = (
        BootstrapConfidenceIntervalEstimator
        ._sample_standard_deviation(
            [1.0],
            1.0,
        )
    )

    assert result == 0.0


def test_percentile_exact_position():
    result = (
        BootstrapConfidenceIntervalEstimator
        ._percentile(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            0.5,
        )
    )

    assert result == 3.0


def test_percentile_interpolation():
    result = (
        BootstrapConfidenceIntervalEstimator
        ._percentile(
            [0.0, 10.0],
            0.25,
        )
    )

    assert result == 2.5


def test_percentile_single_value():
    result = (
        BootstrapConfidenceIntervalEstimator
        ._percentile(
            [7.0],
            0.95,
        )
    )

    assert result == 7.0