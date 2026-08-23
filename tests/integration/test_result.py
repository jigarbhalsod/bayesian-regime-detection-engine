import pytest

from src.integration.result import IntegrationResult


def test_default_result():
    result = IntegrationResult()

    assert result.outputs == {}
    assert result.errors == {}
    assert result.metadata == {}

    assert result.successful_models == 0
    assert result.failed_models == 0
    assert result.total_models == 0
    assert result.success is True


def test_successful_result():
    result = IntegrationResult(
        outputs={
            "hmm": {"regime": "risk_on"},
            "bayesian": {"probability": 0.8},
        },
        metadata={
            "integration_name": "advanced_models",
        },
    )

    assert result.successful_models == 2
    assert result.failed_models == 0
    assert result.total_models == 2
    assert result.success is True

    assert result.get_output("hmm") == {
        "regime": "risk_on",
    }

    assert result.get_output("missing") is None


def test_failed_result():
    result = IntegrationResult(
        outputs={
            "hmm": "success",
        },
        errors={
            "particle_filter": "execution failed",
        },
    )

    assert result.successful_models == 1
    assert result.failed_models == 1
    assert result.total_models == 2
    assert result.success is False

    assert (
        result.get_error("particle_filter")
        == "execution failed"
    )

    assert result.get_error("missing") is None


def test_get_output_default():
    result = IntegrationResult()

    assert (
        result.get_output(
            "missing",
            default="fallback",
        )
        == "fallback"
    )


def test_get_error_default():
    result = IntegrationResult()

    assert (
        result.get_error(
            "missing",
            default="fallback",
        )
        == "fallback"
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("outputs", []),
        ("outputs", None),
        ("errors", []),
        ("errors", None),
        ("metadata", []),
        ("metadata", None),
    ],
)
def test_invalid_mapping_types(
    field_name,
    value,
):
    kwargs = {
        field_name: value,
    }

    with pytest.raises(TypeError):
        IntegrationResult(**kwargs)


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_empty_output_model_names(model_name):
    with pytest.raises(ValueError):
        IntegrationResult(
            outputs={
                model_name: "result",
            }
        )


@pytest.mark.parametrize(
    "model_name",
    [
        123,
        None,
        True,
    ],
)
def test_invalid_output_model_name_types(
    model_name,
):
    with pytest.raises(TypeError):
        IntegrationResult(
            outputs={
                model_name: "result",
            }
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_empty_error_model_names(model_name):
    with pytest.raises(ValueError):
        IntegrationResult(
            errors={
                model_name: "failed",
            }
        )


@pytest.mark.parametrize(
    "error",
    [
        "",
        "   ",
    ],
)
def test_empty_error_messages(error):
    with pytest.raises(ValueError):
        IntegrationResult(
            errors={
                "hmm": error,
            }
        )


@pytest.mark.parametrize(
    "error",
    [
        123,
        None,
        True,
    ],
)
def test_invalid_error_message_types(error):
    with pytest.raises(TypeError):
        IntegrationResult(
            errors={
                "hmm": error,
            }
        )


def test_duplicate_model_names_rejected():
    with pytest.raises(ValueError):
        IntegrationResult(
            outputs={
                "hmm": "success",
            },
            errors={
                "hmm": "failed",
            },
        )


def test_to_dict():
    result = IntegrationResult(
        outputs={
            "hmm": "success",
        },
        errors={
            "bayesian": "failed",
        },
        metadata={
            "run_id": "123",
        },
    )

    assert result.to_dict() == {
        "outputs": {
            "hmm": "success",
        },
        "errors": {
            "bayesian": "failed",
        },
        "metadata": {
            "run_id": "123",
        },
        "successful_models": 1,
        "failed_models": 1,
        "total_models": 2,
        "success": False,
    }


def test_input_mappings_are_copied():
    outputs = {
        "hmm": "success",
    }

    errors = {}

    metadata = {
        "run_id": "123",
    }

    result = IntegrationResult(
        outputs=outputs,
        errors=errors,
        metadata=metadata,
    )

    outputs["bayesian"] = "new"
    errors["foundation"] = "failed"
    metadata["changed"] = True

    assert result.outputs == {
        "hmm": "success",
    }

    assert result.errors == {}

    assert result.metadata == {
        "run_id": "123",
    }