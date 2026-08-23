import pytest

from src.validation.pipeline import (
    ValidationPipeline,
    ValidationPipelineResult,
)


# ============================================================
# Result Tests
# ============================================================


def test_result_defaults():
    result = ValidationPipelineResult(
        results={},
        executed_validators=(),
        failed_validators=(),
    )

    assert result.results == {}
    assert result.executed_validators == ()
    assert result.failed_validators == ()
    assert result.is_valid is True
    assert result.validator_count == 0


def test_result_with_failed_validator():
    result = ValidationPipelineResult(
        results={"first": 10},
        executed_validators=("first", "second"),
        failed_validators=("second",),
    )

    assert result.is_valid is False
    assert result.validator_count == 2


# ============================================================
# Pipeline Registration Tests
# ============================================================


def test_pipeline_starts_empty():
    pipeline = ValidationPipeline()

    assert pipeline.validator_names == ()


def test_register_validator():
    pipeline = ValidationPipeline()

    pipeline.register(
        "identity",
        lambda value: value,
    )

    assert pipeline.validator_names == ("identity",)
    assert pipeline.has_validator("identity") is True


def test_registration_order_is_preserved():
    pipeline = ValidationPipeline()

    pipeline.register("first", lambda: 1)
    pipeline.register("second", lambda: 2)
    pipeline.register("third", lambda: 3)

    assert pipeline.validator_names == (
        "first",
        "second",
        "third",
    )


def test_duplicate_validator_name():
    pipeline = ValidationPipeline()

    pipeline.register("validator", lambda: 1)

    with pytest.raises(ValueError):
        pipeline.register("validator", lambda: 2)


@pytest.mark.parametrize(
    "name",
    [
        123,
        None,
        True,
        [],
    ],
)
def test_invalid_validator_name_type(name):
    with pytest.raises(TypeError):
        ValidationPipeline().register(
            name,
            lambda: None,
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "\t",
        "\n",
    ],
)
def test_empty_validator_name(name):
    with pytest.raises(ValueError):
        ValidationPipeline().register(
            name,
            lambda: None,
        )


@pytest.mark.parametrize(
    "validator",
    [
        123,
        None,
        "invalid",
        [],
    ],
)
def test_invalid_validator(validator):
    with pytest.raises(TypeError):
        ValidationPipeline().register(
            "validator",
            validator,
        )


# ============================================================
# Unregistration Tests
# ============================================================


def test_unregister_validator():
    pipeline = ValidationPipeline()

    pipeline.register("first", lambda: 1)
    pipeline.register("second", lambda: 2)

    pipeline.unregister("first")

    assert pipeline.validator_names == ("second",)
    assert pipeline.has_validator("first") is False


def test_unregister_unknown_validator():
    with pytest.raises(KeyError):
        ValidationPipeline().unregister("unknown")


def test_unregister_invalid_name():
    with pytest.raises(TypeError):
        ValidationPipeline().unregister(123)


# ============================================================
# Pipeline Execution Tests
# ============================================================


def test_run_empty_pipeline():
    result = ValidationPipeline().run()

    assert isinstance(result, ValidationPipelineResult)
    assert result.results == {}
    assert result.executed_validators == ()
    assert result.failed_validators == ()
    assert result.is_valid is True


def test_run_single_validator():
    pipeline = ValidationPipeline()

    pipeline.register(
        "double",
        lambda value: value * 2,
    )

    result = pipeline.run(5)

    assert result.results == {"double": 10}
    assert result.executed_validators == ("double",)
    assert result.failed_validators == ()
    assert result.is_valid is True


def test_run_multiple_validators():
    pipeline = ValidationPipeline()

    pipeline.register(
        "double",
        lambda value: value * 2,
    )
    pipeline.register(
        "triple",
        lambda value: value * 3,
    )

    result = pipeline.run(5)

    assert result.results == {
        "double": 10,
        "triple": 15,
    }
    assert result.executed_validators == (
        "double",
        "triple",
    )
    assert result.failed_validators == ()


def test_run_passes_keyword_arguments():
    pipeline = ValidationPipeline()

    pipeline.register(
        "combine",
        lambda left, right: left + right,
    )

    result = pipeline.run(
        left=2,
        right=3,
    )

    assert result.results["combine"] == 5


def test_failed_validator_is_recorded():
    pipeline = ValidationPipeline()

    def failing_validator():
        raise RuntimeError("failure")

    pipeline.register("failing", failing_validator)
    pipeline.register("successful", lambda: "ok")

    result = pipeline.run()

    assert result.results == {
        "successful": "ok"
    }
    assert result.executed_validators == (
        "failing",
        "successful",
    )
    assert result.failed_validators == ("failing",)
    assert result.is_valid is False


def test_pipeline_continues_after_failure():
    pipeline = ValidationPipeline()

    def failing_validator():
        raise ValueError("failure")

    pipeline.register("first", failing_validator)
    pipeline.register("second", lambda: 2)
    pipeline.register("third", lambda: 3)

    result = pipeline.run()

    assert result.results == {
        "second": 2,
        "third": 3,
    }
    assert result.executed_validators == (
        "first",
        "second",
        "third",
    )
    assert result.failed_validators == ("first",)


def test_raise_on_error_stops_execution():
    pipeline = ValidationPipeline()

    pipeline.register(
        "failing",
        lambda: 1 / 0,
    )
    pipeline.register(
        "later",
        lambda: "should not execute",
    )

    with pytest.raises(ZeroDivisionError):
        pipeline.run(raise_on_error=True)


def test_invalid_raise_on_error_type():
    pipeline = ValidationPipeline()

    with pytest.raises(TypeError):
        pipeline.run(raise_on_error="yes")


# ============================================================
# has_validator Tests
# ============================================================


def test_has_validator_returns_false():
    pipeline = ValidationPipeline()

    assert pipeline.has_validator("missing") is False


def test_has_validator_invalid_name():
    pipeline = ValidationPipeline()

    with pytest.raises(ValueError):
        pipeline.has_validator("   ")


# ============================================================
# Internal Validation Tests
# ============================================================


def test_validate_name_returns_none():
    result = ValidationPipeline._validate_name(
        "valid"
    )

    assert result is None


def test_validate_validator_returns_none():
    result = (
        ValidationPipeline._validate_validator(
            lambda: None
        )
    )

    assert result is None


def test_result_is_immutable():
    result = ValidationPipelineResult(
        results={},
        executed_validators=(),
        failed_validators=(),
    )

    with pytest.raises(Exception):
        result.failed_validators = ("failed",)