import pytest

from src.models.result import AdvancedModelResult


def test_default_result():
    result = AdvancedModelResult()

    assert result.regimes == []
    assert result.probabilities is None
    assert result.confidence is None
    assert result.timestamps is None
    assert result.model_name == "advanced_model"
    assert result.metadata == {}
    assert result.diagnostics == {}
    assert result.n_predictions == 0


def test_complete_result():
    result = AdvancedModelResult(
        regimes=["RISK_ON", "RISK_OFF"],
        probabilities=[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ],
        confidence=[0.8, 0.8],
        timestamps=["2026-01-01", "2026-01-02"],
        model_name="hmm",
        metadata={"n_states": 3},
        diagnostics={"converged": True},
    )

    assert result.n_predictions == 2
    assert result.has_probabilities is True
    assert result.has_confidence is True
    assert result.has_timestamps is True
    assert result.model_name == "hmm"


def test_regimes_none_rejected():
    with pytest.raises(TypeError):
        AdvancedModelResult(regimes=None)


def test_string_regimes_rejected():
    with pytest.raises(TypeError):
        AdvancedModelResult(regimes="RISK_ON")


def test_empty_regimes_allowed():
    result = AdvancedModelResult(regimes=[])

    assert result.n_predictions == 0


def test_model_name_is_normalized():
    result = AdvancedModelResult(
        model_name=" hmm "
    )

    assert result.model_name == "hmm"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
    ],
)
def test_empty_model_name_rejected(value):
    with pytest.raises(ValueError):
        AdvancedModelResult(model_name=value)


def test_non_string_model_name_rejected():
    with pytest.raises(TypeError):
        AdvancedModelResult(model_name=123)


def test_probabilities_are_converted_to_float():
    result = AdvancedModelResult(
        regimes=["A"],
        probabilities=[["0.2", "0.3", "0.5"]],
    )

    assert result.probabilities == [
        [0.2, 0.3, 0.5]
    ]


def test_probability_out_of_range_rejected():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A"],
            probabilities=[[1.2, -0.2]],
        )


def test_probability_row_must_sum_to_one():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A"],
            probabilities=[[0.2, 0.2, 0.2]],
        )


def test_empty_probability_row_rejected():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A"],
            probabilities=[[]],
        )


def test_probability_row_string_rejected():
    with pytest.raises(TypeError):
        AdvancedModelResult(
            regimes=["A"],
            probabilities=["0.2"],
        )


def test_probability_length_must_match_regimes():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A", "B"],
            probabilities=[
                [0.2, 0.3, 0.5],
            ],
        )


def test_confidence_values_are_converted_to_float():
    result = AdvancedModelResult(
        regimes=["A"],
        confidence=["0.9"],
    )

    assert result.confidence == [0.9]


@pytest.mark.parametrize(
    "value",
    [
        [-0.1],
        [1.1],
    ],
)
def test_invalid_confidence_rejected(value):
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A"],
            confidence=value,
        )


def test_confidence_length_must_match_regimes():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A", "B"],
            confidence=[0.8],
        )


def test_timestamp_length_must_match_regimes():
    with pytest.raises(ValueError):
        AdvancedModelResult(
            regimes=["A", "B"],
            timestamps=["2026-01-01"],
        )


def test_metadata_none_becomes_empty_dict():
    result = AdvancedModelResult(metadata=None)

    assert result.metadata == {}


def test_diagnostics_none_becomes_empty_dict():
    result = AdvancedModelResult(diagnostics=None)

    assert result.diagnostics == {}


@pytest.mark.parametrize(
    "field_name",
    [
        "metadata",
        "diagnostics",
    ],
)
def test_invalid_dict_fields_rejected(field_name):
    kwargs = {
        field_name: [],
    }

    with pytest.raises(TypeError):
        AdvancedModelResult(**kwargs)


def test_to_dict_returns_independent_collections():
    result = AdvancedModelResult(
        regimes=["A"],
        probabilities=[[0.2, 0.3, 0.5]],
        confidence=[0.8],
        timestamps=["2026-01-01"],
        metadata={"value": 1},
        diagnostics={"status": "ok"},
    )

    data = result.to_dict()

    data["regimes"].append("B")
    data["probabilities"][0][0] = 0.9
    data["metadata"]["value"] = 99

    assert result.regimes == ["A"]
    assert result.probabilities == [
        [0.2, 0.3, 0.5]
    ]
    assert result.metadata == {"value": 1}