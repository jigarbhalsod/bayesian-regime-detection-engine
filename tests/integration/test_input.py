import pytest

from src.integration.input import (
    AdvancedModelInput,
)


def test_minimal_input():
    model_input = AdvancedModelInput(
        features=[1, 2, 3],
    )

    assert model_input.features == [1, 2, 3]
    assert model_input.timestamps is None
    assert model_input.target is None
    assert model_input.regime_context is None
    assert model_input.metadata == {}


def test_complete_input():
    model_input = AdvancedModelInput(
        features=[[1, 2], [3, 4]],
        timestamps=["t1", "t2"],
        target={"direction": "up"},
        regime_context="risk_on",
        metadata={"source": "test"},
    )

    assert model_input.n_observations == 2
    assert model_input.has_timestamps is True
    assert model_input.has_target is True
    assert model_input.has_regime_context is True
    assert model_input.metadata == {
        "source": "test",
    }


def test_n_observations():
    model_input = AdvancedModelInput(
        features=[10, 20, 30, 40],
    )

    assert model_input.n_observations == 4


@pytest.mark.parametrize(
    "features",
    [
        [],
        (),
    ],
)
def test_empty_features_rejected(features):
    with pytest.raises(ValueError):
        AdvancedModelInput(
            features=features
        )


@pytest.mark.parametrize(
    "features",
    [
        "invalid",
        b"invalid",
        123,
        None,
        {"a": 1},
    ],
)
def test_invalid_features_type_rejected(features):
    with pytest.raises(TypeError):
        AdvancedModelInput(
            features=features
        )


def test_matching_timestamps_accepted():
    model_input = AdvancedModelInput(
        features=[1, 2, 3],
        timestamps=["a", "b", "c"],
    )

    assert model_input.timestamps == [
        "a",
        "b",
        "c",
    ]


def test_mismatched_timestamps_rejected():
    with pytest.raises(
        ValueError,
        match="timestamps length must match",
    ):
        AdvancedModelInput(
            features=[1, 2, 3],
            timestamps=["a", "b"],
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_timestamps_type_rejected(
    timestamps,
):
    with pytest.raises(TypeError):
        AdvancedModelInput(
            features=[1, 2],
            timestamps=timestamps,
        )


def test_none_timestamps_allowed():
    model_input = AdvancedModelInput(
        features=[1, 2],
        timestamps=None,
    )

    assert model_input.timestamps is None
    assert model_input.has_timestamps is False


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        "invalid",
        123,
        None,
    ],
)
def test_invalid_metadata_type_rejected(
    metadata,
):
    with pytest.raises(TypeError):
        AdvancedModelInput(
            features=[1],
            metadata=metadata,
        )


def test_metadata_is_copied():
    metadata = {
        "source": "original",
    }

    model_input = AdvancedModelInput(
        features=[1, 2],
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert model_input.metadata == {
        "source": "original",
    }


def test_to_dict():
    model_input = AdvancedModelInput(
        features=[1, 2],
        timestamps=["a", "b"],
        target="up",
        regime_context="risk_on",
        metadata={"source": "test"},
    )

    result = model_input.to_dict()

    assert result == {
        "features": [1, 2],
        "timestamps": ["a", "b"],
        "target": "up",
        "regime_context": "risk_on",
        "metadata": {"source": "test"},
    }


def test_to_dict_returns_metadata_copy():
    model_input = AdvancedModelInput(
        features=[1],
        metadata={"source": "test"},
    )

    result = model_input.to_dict()
    result["metadata"]["source"] = "changed"

    assert model_input.metadata == {
        "source": "test",
    }