import pytest
import torch

from src.models.foundation.sequence import (
    SequenceDataset,
    SequencePreparer,
)


@pytest.fixture
def preparer():
    return SequencePreparer(
        context_length=4,
        forecast_horizon=2,
        n_features=3,
        output_dim=1,
        stride=1,
    )


@pytest.fixture
def data():
    return torch.arange(
        30,
        dtype=torch.float32,
    ).reshape(10, 3)


def test_default_metadata(preparer):
    assert preparer.get_metadata() == {
        "context_length": 4,
        "forecast_horizon": 2,
        "n_features": 3,
        "output_dim": 1,
        "stride": 1,
    }


def test_prepare_returns_sequence_dataset(
    preparer,
    data,
):
    result = preparer.prepare(data)

    assert isinstance(result, SequenceDataset)


def test_sequence_shapes(
    preparer,
    data,
):
    result = preparer.prepare(data)

    # 10 - 4 - 2 + 1 = 5 sequences
    assert result.inputs.shape == (5, 4, 3)
    assert result.targets.shape == (5, 2, 1)


def test_first_sequence_alignment(
    preparer,
    data,
):
    result = preparer.prepare(data)

    assert torch.equal(
        result.inputs[0],
        data[0:4],
    )

    assert torch.equal(
        result.targets[0],
        data[4:6, :1],
    )


def test_last_sequence_alignment(
    preparer,
    data,
):
    result = preparer.prepare(data)

    assert torch.equal(
        result.inputs[-1],
        data[4:8],
    )

    assert torch.equal(
        result.targets[-1],
        data[8:10, :1],
    )


def test_custom_targets(preparer, data):
    targets = torch.arange(
        10,
        dtype=torch.float32,
    ).reshape(10, 1)

    result = preparer.prepare(
        data,
        targets=targets,
    )

    assert torch.equal(
        result.targets[0],
        targets[4:6],
    )


def test_stride_reduces_sequence_count(data):
    preparer = SequencePreparer(
        context_length=4,
        forecast_horizon=2,
        n_features=3,
        output_dim=1,
        stride=2,
    )

    result = preparer.prepare(data)

    # Valid starts: 0, 2, 4
    assert result.inputs.shape[0] == 3


def test_integer_data_converted_to_float(preparer):
    data = torch.ones(
        10,
        3,
        dtype=torch.int64,
    )

    result = preparer.prepare(data)

    assert result.inputs.dtype == torch.float32
    assert result.targets.dtype == torch.float32


@pytest.mark.parametrize(
    "field_name",
    [
        "context_length",
        "forecast_horizon",
        "n_features",
        "output_dim",
        "stride",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_positive_integer_values_rejected(
    field_name,
    value,
):
    kwargs = {
        "context_length": 4,
        "forecast_horizon": 2,
        "n_features": 3,
        "output_dim": 1,
        "stride": 1,
        field_name: value,
    }

    with pytest.raises(ValueError):
        SequencePreparer(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "context_length",
        "forecast_horizon",
        "n_features",
        "output_dim",
        "stride",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "4",
        None,
        True,
    ],
)
def test_invalid_positive_integer_types_rejected(
    field_name,
    value,
):
    kwargs = {
        "context_length": 4,
        "forecast_horizon": 2,
        "n_features": 3,
        "output_dim": 1,
        "stride": 1,
        field_name: value,
    }

    with pytest.raises(TypeError):
        SequencePreparer(**kwargs)


def test_output_dim_greater_than_features_rejected():
    with pytest.raises(ValueError):
        SequencePreparer(
            context_length=4,
            forecast_horizon=2,
            n_features=2,
            output_dim=3,
        )


def test_non_tensor_data_rejected(preparer):
    with pytest.raises(TypeError):
        preparer.prepare(
            [[1, 2, 3]],
        )


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (2, 5, 3),
    ],
)
def test_invalid_data_dimensions_rejected(
    preparer,
    shape,
):
    with pytest.raises(ValueError):
        preparer.prepare(
            torch.ones(*shape),
        )


def test_empty_data_rejected(preparer):
    with pytest.raises(ValueError):
        preparer.prepare(
            torch.empty(0, 3),
        )


def test_wrong_feature_count_rejected(preparer):
    with pytest.raises(ValueError):
        preparer.prepare(
            torch.ones(10, 2),
        )


def test_non_finite_data_rejected(preparer):
    data = torch.ones(10, 3)
    data[0, 0] = float("nan")

    with pytest.raises(ValueError):
        preparer.prepare(data)


def test_insufficient_observations_rejected(preparer):
    with pytest.raises(ValueError):
        preparer.prepare(
            torch.ones(5, 3),
        )


def test_non_tensor_targets_rejected(
    preparer,
    data,
):
    with pytest.raises(TypeError):
        preparer.prepare(
            data,
            targets=[[1]],
        )


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (10, 1, 1),
    ],
)
def test_invalid_target_dimensions_rejected(
    preparer,
    data,
    shape,
):
    with pytest.raises(ValueError):
        preparer.prepare(
            data,
            targets=torch.ones(*shape),
        )


def test_target_observation_count_mismatch_rejected(
    preparer,
    data,
):
    with pytest.raises(ValueError):
        preparer.prepare(
            data,
            targets=torch.ones(9, 1),
        )


def test_wrong_target_output_dim_rejected(
    preparer,
    data,
):
    with pytest.raises(ValueError):
        preparer.prepare(
            data,
            targets=torch.ones(10, 2),
        )


def test_non_finite_targets_rejected(
    preparer,
    data,
):
    targets = torch.ones(10, 1)
    targets[0, 0] = float("inf")

    with pytest.raises(ValueError):
        preparer.prepare(
            data,
            targets=targets,
        )