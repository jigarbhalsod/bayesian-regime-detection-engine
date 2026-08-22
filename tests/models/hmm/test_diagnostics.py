import numpy as np
import pytest

from src.models.hmm.config import HMMConfig
from src.models.hmm.diagnostics import HMMRegimeDiagnostics
from src.models.hmm.model import GaussianHMMRegimeModel


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)

    regime_one = rng.normal(
        loc=[-1.0, 1.0],
        scale=[0.2, 0.2],
        size=(40, 2),
    )

    regime_two = rng.normal(
        loc=[1.0, -1.0],
        scale=[0.2, 0.2],
        size=(40, 2),
    )

    return np.vstack(
        [
            regime_one,
            regime_two,
        ]
    )


@pytest.fixture
def fitted_model(sample_data):
    config = HMMConfig(
        n_regimes=2,
        n_iter=50,
        random_state=42,
    )

    model = GaussianHMMRegimeModel(config)
    model.fit(sample_data)

    return model


@pytest.fixture
def diagnostics(fitted_model):
    return HMMRegimeDiagnostics(fitted_model)


def test_creation_with_fitted_model(
    diagnostics,
    fitted_model,
):
    assert diagnostics.model is fitted_model
    assert diagnostics.n_regimes == 2


def test_invalid_model_type_rejected():
    with pytest.raises(TypeError):
        HMMRegimeDiagnostics(model={})


def test_unfitted_model_rejected():
    model = GaussianHMMRegimeModel(
        HMMConfig(n_regimes=2)
    )

    with pytest.raises(RuntimeError):
        HMMRegimeDiagnostics(model)


def test_state_statistics(
    diagnostics,
    sample_data,
):
    statistics = diagnostics.get_state_statistics(
        sample_data
    )

    assert len(statistics) == 2

    total_count = 0

    for item in statistics:
        assert "state" in item
        assert "count" in item
        assert "proportion" in item
        assert "mean" in item
        assert "std" in item

        assert item["state"] in [0, 1]
        assert item["count"] >= 0
        assert 0.0 <= item["proportion"] <= 1.0

        assert len(item["mean"]) == 2
        assert len(item["std"]) == 2

        total_count += item["count"]

    assert total_count == len(sample_data)


def test_state_proportions_sum_to_one(
    diagnostics,
    sample_data,
):
    statistics = diagnostics.get_state_statistics(
        sample_data
    )

    total = sum(
        item["proportion"]
        for item in statistics
    )

    assert total == pytest.approx(1.0)


def test_transition_diagnostics(
    diagnostics,
):
    results = diagnostics.get_transition_diagnostics()

    assert len(results) == 2

    for item in results:
        assert "state" in item
        assert "self_transition_probability" in item
        assert "exit_probability" in item
        assert "expected_duration" in item

        assert 0.0 <= (
            item["self_transition_probability"]
        ) <= 1.0

        assert 0.0 <= item["exit_probability"] <= 1.0

        assert (
            item["self_transition_probability"]
            + item["exit_probability"]
        ) == pytest.approx(1.0)

        assert item["expected_duration"] >= 1.0


def test_expected_duration_matches_transition_matrix(
    diagnostics,
    fitted_model,
):
    matrix = fitted_model.get_transition_matrix()

    results = diagnostics.get_transition_diagnostics()

    for item in results:
        state = item["state"]
        probability = matrix[state][state]

        if np.isclose(probability, 1.0):
            assert np.isinf(
                item["expected_duration"]
            )
        else:
            expected = 1.0 / (
                1.0 - probability
            )

            assert (
                item["expected_duration"]
                == pytest.approx(expected)
            )


def test_state_ordering_ascending(
    diagnostics,
    sample_data,
):
    ordering = diagnostics.get_state_ordering(
        sample_data,
        feature_index=0,
        ascending=True,
    )

    assert len(ordering) == 2
    assert sorted(ordering) == [0, 1]


def test_state_ordering_descending(
    diagnostics,
    sample_data,
):
    ordering = diagnostics.get_state_ordering(
        sample_data,
        feature_index=0,
        ascending=False,
    )

    assert len(ordering) == 2
    assert sorted(ordering) == [0, 1]


def test_ascending_and_descending_orderings_reverse(
    diagnostics,
    sample_data,
):
    ascending = diagnostics.get_state_ordering(
        sample_data,
        ascending=True,
    )

    descending = diagnostics.get_state_ordering(
        sample_data,
        ascending=False,
    )

    assert descending == list(
        reversed(ascending)
    )


@pytest.mark.parametrize(
    "feature_index",
    [
        -1,
        2,
        100,
    ],
)
def test_invalid_feature_index_value_rejected(
    diagnostics,
    sample_data,
    feature_index,
):
    with pytest.raises(ValueError):
        diagnostics.get_state_ordering(
            sample_data,
            feature_index=feature_index,
        )


@pytest.mark.parametrize(
    "feature_index",
    [
        0.5,
        True,
        "0",
        None,
    ],
)
def test_invalid_feature_index_type_rejected(
    diagnostics,
    sample_data,
    feature_index,
):
    with pytest.raises(TypeError):
        diagnostics.get_state_ordering(
            sample_data,
            feature_index=feature_index,
        )


@pytest.mark.parametrize(
    "ascending",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_invalid_ascending_rejected(
    diagnostics,
    sample_data,
    ascending,
):
    with pytest.raises(TypeError):
        diagnostics.get_state_ordering(
            sample_data,
            ascending=ascending,
        )


def test_summary(
    diagnostics,
    sample_data,
):
    summary = diagnostics.get_summary(
        sample_data
    )

    assert summary["n_regimes"] == 2
    assert summary["n_features"] == 2
    assert summary["n_samples"] == len(sample_data)

    assert len(
        summary["state_counts"]
    ) == 2

    assert sum(
        summary["state_counts"]
    ) == len(sample_data)

    assert len(
        summary["state_statistics"]
    ) == 2

    assert len(
        summary["transition_diagnostics"]
    ) == 2


def test_summary_state_counts_match_statistics(
    diagnostics,
    sample_data,
):
    summary = diagnostics.get_summary(
        sample_data
    )

    expected_counts = [
        item["count"]
        for item in summary["state_statistics"]
    ]

    assert (
        summary["state_counts"]
        == expected_counts
    )


def test_invalid_data_rejected(
    diagnostics,
):
    with pytest.raises(ValueError):
        diagnostics.get_state_statistics([])