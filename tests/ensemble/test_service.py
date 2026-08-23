from __future__ import annotations

import math

import pytest

from src.ensemble.service import EnsembleService


# ============================================================
# Initialization
# ============================================================


def test_service_creates_default_config():
    service = EnsembleService()

    assert service.config is not None


def test_service_accepts_custom_config():
    from src.ensemble.config import EnsembleConfig

    config = EnsembleConfig()
    service = EnsembleService(config=config)

    assert service.config is config


# ============================================================
# Scalar Aggregation
# ============================================================


def test_aggregate_equal_weights():
    service = EnsembleService()

    result = service.aggregate(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert result.prediction == pytest.approx(20.0)


def test_aggregate_explicit_weights():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            20.0,
            30.0,
        ],
        weights=[
            0.2,
            0.3,
            0.5,
        ],
    )

    assert result.prediction == pytest.approx(23.0)


def test_aggregate_normalizes_explicit_weights():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            30.0,
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result.prediction == pytest.approx(25.0)


def test_aggregate_with_single_prediction():
    service = EnsembleService()

    result = service.aggregate(
        [
            42.0,
        ]
    )

    assert result.prediction == pytest.approx(42.0)


# ============================================================
# Vector Aggregation
# ============================================================


def test_aggregate_vectors_equal_weights():
    service = EnsembleService()

    result = service.aggregate_vectors(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert result.prediction == pytest.approx(
        (
            2.0,
            3.0,
        )
    )


def test_aggregate_vectors_explicit_weights():
    service = EnsembleService()

    result = service.aggregate_vectors(
        predictions=[
            [1.0, 2.0],
            [5.0, 6.0],
        ],
        weights=[
            0.25,
            0.75,
        ],
    )

    assert result.prediction == pytest.approx(
        (
            4.0,
            5.0,
        )
    )


def test_aggregate_vectors_normalizes_weights():
    service = EnsembleService()

    result = service.aggregate_vectors(
        predictions=[
            [0.0, 10.0],
            [10.0, 20.0],
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result.prediction == pytest.approx(
        (
            7.5,
            17.5,
        )
    )


# ============================================================
# Voting
# ============================================================


def test_vote_uses_majority_vote_without_weights():
    service = EnsembleService()

    result = service.vote(
        [
            "bull",
            "bear",
            "bull",
        ]
    )

    assert result.prediction == "bull"


def test_vote_uses_weighted_vote_with_explicit_weights():
    service = EnsembleService()

    result = service.vote(
        predictions=[
            "bull",
            "bear",
            "bull",
        ],
        weights=[
            0.2,
            0.7,
            0.1,
        ],
    )

    assert result.prediction == "bear"


def test_vote_with_single_prediction():
    service = EnsembleService()

    result = service.vote(
        [
            "risk_on",
        ]
    )

    assert result.prediction == "risk_on"


# ============================================================
# Scalar Input Validation
# ============================================================


@pytest.mark.parametrize(
    "predictions",
    [
        None,
        "invalid",
        123,
        12.5,
    ],
)
def test_aggregate_rejects_non_sequence_predictions(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.aggregate(predictions)


def test_aggregate_rejects_empty_predictions():
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate([])


@pytest.mark.parametrize(
    "predictions",
    [
        [True, 1.0],
        [False, 2.0],
        [None, 1.0],
        ["1.0", 2.0],
    ],
)
def test_aggregate_rejects_invalid_prediction_values(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.aggregate(predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        [math.nan],
        [math.inf],
        [-math.inf],
        [1.0, math.nan],
    ],
)
def test_aggregate_rejects_non_finite_predictions(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate(predictions)


# ============================================================
# Vector Input Validation
# ============================================================


@pytest.mark.parametrize(
    "predictions",
    [
        None,
        "invalid",
        123,
    ],
)
def test_aggregate_vectors_rejects_non_sequence(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.aggregate_vectors(predictions)


def test_aggregate_vectors_rejects_empty_predictions():
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate_vectors([])


@pytest.mark.parametrize(
    "predictions",
    [
        [
            [1.0, 2.0],
            [3.0],
        ],
        [
            [1.0],
            [2.0, 3.0],
        ],
    ],
)
def test_aggregate_vectors_rejects_different_lengths(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate_vectors(predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        [
            [],
        ],
        [
            [1.0],
            [],
        ],
    ],
)
def test_aggregate_vectors_rejects_empty_vectors(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate_vectors(predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        [
            [1.0, True],
        ],
        [
            [None],
        ],
        [
            ["1.0"],
        ],
    ],
)
def test_aggregate_vectors_rejects_invalid_values(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.aggregate_vectors(predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        [
            [math.nan],
        ],
        [
            [math.inf],
        ],
        [
            [-math.inf],
        ],
    ],
)
def test_aggregate_vectors_rejects_non_finite_values(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.aggregate_vectors(predictions)


# ============================================================
# Voting Input Validation
# ============================================================


@pytest.mark.parametrize(
    "predictions",
    [
        None,
        "bull",
        123,
    ],
)
def test_vote_rejects_non_sequence_predictions(
    predictions,
):
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.vote(predictions)


def test_vote_rejects_empty_predictions():
    service = EnsembleService()

    with pytest.raises(ValueError):
        service.vote([])


def test_vote_rejects_none_prediction_value():
    service = EnsembleService()

    with pytest.raises(TypeError):
        service.vote(
            [
                "bull",
                None,
            ]
        )


# ============================================================
# Weight Handling
# ============================================================


def test_aggregate_accepts_weights_that_need_normalization():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            20.0,
        ],
        weights=[
            2.0,
            6.0,
        ],
    )

    assert result.prediction == pytest.approx(17.5)


def test_vector_aggregate_accepts_weights_that_need_normalization():
    service = EnsembleService()

    result = service.aggregate_vectors(
        predictions=[
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result.prediction == pytest.approx(
        (
            2.5,
            3.5,
        )
    )


def test_vote_accepts_weights_that_need_normalization():
    service = EnsembleService()

    result = service.vote(
        predictions=[
            "bull",
            "bear",
        ],
        weights=[
            2.0,
            6.0,
        ],
    )

    assert result.prediction == "bear"


# ============================================================
# Result Integrity
# ============================================================


def test_aggregate_returns_result_with_confidence():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            20.0,
        ],
    )

    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_aggregate_returns_result_with_uncertainty():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            20.0,
        ],
    )

    assert isinstance(result.uncertainty, float)
    assert 0.0 <= result.uncertainty <= 1.0


def test_confidence_and_uncertainty_sum_to_one():
    service = EnsembleService()

    result = service.aggregate(
        predictions=[
            10.0,
            20.0,
            30.0,
        ],
        weights=[
            0.1,
            0.2,
            0.7,
        ],
    )

    assert (
        result.confidence + result.uncertainty
    ) == pytest.approx(1.0)