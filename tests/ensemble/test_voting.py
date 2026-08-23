import math

import pytest

from src.ensemble.voting import EnsembleVoting


# ============================================================
# Majority Voting
# ============================================================


def test_majority_vote_basic():
    result = EnsembleVoting.majority_vote(
        [
            "bull",
            "bear",
            "bull",
        ]
    )

    assert result == "bull"


def test_majority_vote_integer_labels():
    result = EnsembleVoting.majority_vote(
        [
            1,
            2,
            1,
            1,
        ]
    )

    assert result == 1


def test_majority_vote_single_prediction():
    result = EnsembleVoting.majority_vote(
        [
            "neutral",
        ]
    )

    assert result == "neutral"


def test_majority_vote_first_occurrence_breaks_tie():
    result = EnsembleVoting.majority_vote(
        [
            "bull",
            "bear",
        ]
    )

    assert result == "bull"


def test_majority_vote_first_occurrence_breaks_multiway_tie():
    result = EnsembleVoting.majority_vote(
        [
            "neutral",
            "bull",
            "bear",
        ]
    )

    assert result == "neutral"


def test_majority_vote_tie_after_repeated_values():
    result = EnsembleVoting.majority_vote(
        [
            "bear",
            "bull",
            "bull",
            "bear",
        ]
    )

    assert result == "bear"


# ============================================================
# Vote Counts
# ============================================================


def test_vote_counts_basic():
    result = EnsembleVoting.vote_counts(
        [
            "bull",
            "bear",
            "bull",
            "neutral",
            "bull",
        ]
    )

    assert result == {
        "bull": 3,
        "bear": 1,
        "neutral": 1,
    }


def test_vote_counts_preserves_first_appearance_order():
    result = EnsembleVoting.vote_counts(
        [
            "bear",
            "bull",
            "bear",
            "neutral",
        ]
    )

    assert list(result.keys()) == [
        "bear",
        "bull",
        "neutral",
    ]


def test_vote_counts_single_prediction():
    result = EnsembleVoting.vote_counts(
        [
            "bull",
        ]
    )

    assert result == {
        "bull": 1,
    }


# ============================================================
# Vote Shares
# ============================================================


def test_vote_shares_basic():
    result = EnsembleVoting.vote_shares(
        [
            "bull",
            "bear",
            "bull",
            "neutral",
        ]
    )

    assert result["bull"] == pytest.approx(0.5)
    assert result["bear"] == pytest.approx(0.25)
    assert result["neutral"] == pytest.approx(0.25)


def test_vote_shares_sum_to_one():
    result = EnsembleVoting.vote_shares(
        [
            "bull",
            "bull",
            "bear",
        ]
    )

    assert sum(result.values()) == pytest.approx(1.0)


def test_vote_shares_single_prediction():
    result = EnsembleVoting.vote_shares(
        [
            "neutral",
        ]
    )

    assert result == {
        "neutral": 1.0,
    }


# ============================================================
# Weighted Voting
# ============================================================


def test_weighted_vote_equal_weights():
    result = EnsembleVoting.weighted_vote(
        [
            "bull",
            "bear",
            "bull",
        ]
    )

    assert result == "bull"


def test_weighted_vote_custom_weights():
    result = EnsembleVoting.weighted_vote(
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

    assert result == "bear"


def test_weighted_vote_combines_prediction_weights():
    result = EnsembleVoting.weighted_vote(
        predictions=[
            "bull",
            "bear",
            "bull",
        ],
        weights=[
            0.4,
            0.3,
            0.3,
        ],
    )

    assert result == "bull"


def test_weighted_vote_normalizes_weights():
    result = EnsembleVoting.weighted_vote(
        predictions=[
            "bull",
            "bear",
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result == "bear"


def test_weighted_vote_first_occurrence_breaks_tie():
    result = EnsembleVoting.weighted_vote(
        predictions=[
            "bear",
            "bull",
        ],
        weights=[
            1.0,
            1.0,
        ],
    )

    assert result == "bear"


# ============================================================
# Weighted Vote Shares
# ============================================================


def test_weighted_vote_shares_basic():
    result = EnsembleVoting.weighted_vote_shares(
        predictions=[
            "bull",
            "bear",
            "bull",
        ],
        weights=[
            0.2,
            0.5,
            0.3,
        ],
    )

    assert result["bull"] == pytest.approx(0.5)
    assert result["bear"] == pytest.approx(0.5)


def test_weighted_vote_shares_equal_weights():
    result = EnsembleVoting.weighted_vote_shares(
        [
            "bull",
            "bear",
            "bull",
        ]
    )

    assert result["bull"] == pytest.approx(2 / 3)
    assert result["bear"] == pytest.approx(1 / 3)


def test_weighted_vote_shares_normalizes_weights():
    result = EnsembleVoting.weighted_vote_shares(
        predictions=[
            "bull",
            "bear",
        ],
        weights=[
            2.0,
            6.0,
        ],
    )

    assert result["bull"] == pytest.approx(0.25)
    assert result["bear"] == pytest.approx(0.75)


def test_weighted_vote_shares_sum_to_one():
    result = EnsembleVoting.weighted_vote_shares(
        predictions=[
            "bull",
            "bear",
            "bull",
        ],
        weights=[
            1.0,
            2.0,
            3.0,
        ],
    )

    assert sum(result.values()) == pytest.approx(1.0)


# ============================================================
# Invalid Prediction Containers
# ============================================================


@pytest.mark.parametrize(
    "predictions",
    [
        [],
    ],
)
def test_empty_predictions_rejected(predictions):
    with pytest.raises(ValueError):
        EnsembleVoting.majority_vote(predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        "bull",
        b"bull",
        123,
        1.0,
        None,
    ],
)
def test_invalid_prediction_container_rejected(predictions):
    with pytest.raises(TypeError):
        EnsembleVoting.majority_vote(predictions)


# ============================================================
# Invalid Prediction Values
# ============================================================


def test_none_prediction_rejected():
    with pytest.raises(TypeError):
        EnsembleVoting.majority_vote(
            [
                "bull",
                None,
            ]
        )


@pytest.mark.parametrize(
    "prediction",
    [
        [],
        {},
        set(),
    ],
)
def test_unhashable_prediction_rejected(prediction):
    with pytest.raises(TypeError):
        EnsembleVoting.majority_vote(
            [
                "bull",
                prediction,
            ]
        )


# ============================================================
# Validation Shared Across Methods
# ============================================================


@pytest.mark.parametrize(
    "method_name",
    [
        "majority_vote",
        "vote_counts",
        "vote_shares",
        "weighted_vote",
        "weighted_vote_shares",
    ],
)
def test_empty_predictions_rejected_by_all_methods(
    method_name,
):
    method = getattr(
        EnsembleVoting,
        method_name,
    )

    with pytest.raises(ValueError):
        method([])


@pytest.mark.parametrize(
    "method_name",
    [
        "majority_vote",
        "vote_counts",
        "vote_shares",
        "weighted_vote",
        "weighted_vote_shares",
    ],
)
def test_invalid_container_rejected_by_all_methods(
    method_name,
):
    method = getattr(
        EnsembleVoting,
        method_name,
    )

    with pytest.raises(TypeError):
        method("bull")


@pytest.mark.parametrize(
    "method_name",
    [
        "majority_vote",
        "vote_counts",
        "vote_shares",
        "weighted_vote",
        "weighted_vote_shares",
    ],
)
def test_none_prediction_rejected_by_all_methods(
    method_name,
):
    method = getattr(
        EnsembleVoting,
        method_name,
    )

    with pytest.raises(TypeError):
        method(
            [
                "bull",
                None,
            ]
        )


def test_weighted_vote_invalid_weight_type_rejected():
    with pytest.raises(TypeError):
        EnsembleVoting.weighted_vote(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
                "invalid",
            ],
        )


def test_weighted_vote_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleVoting.weighted_vote(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
            ],
        )


def test_weighted_vote_negative_weight_rejected():
    with pytest.raises(ValueError):
        EnsembleVoting.weighted_vote(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
                -1.0,
            ],
        )


def test_weighted_vote_non_finite_weight_rejected():
    with pytest.raises(ValueError):
        EnsembleVoting.weighted_vote(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
                math.nan,
            ],
        )


def test_weighted_vote_shares_invalid_weight_type_rejected():
    with pytest.raises(TypeError):
        EnsembleVoting.weighted_vote_shares(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
                "invalid",
            ],
        )


def test_weighted_vote_shares_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleVoting.weighted_vote_shares(
            predictions=[
                "bull",
                "bear",
            ],
            weights=[
                1.0,
            ],
        )