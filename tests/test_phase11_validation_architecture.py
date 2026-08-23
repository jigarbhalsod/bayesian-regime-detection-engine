import pytest

from src.validation import (
    BaseValidator,
    ValidationConfig,
    ValidationResult,
)


# ============================================================
# ValidationConfig Tests
# ============================================================


def test_validation_config_defaults():
    config = ValidationConfig()

    assert config.name == "validation"
    assert config.enabled is True
    assert config.metadata == {}


def test_validation_config_custom_values():
    config = ValidationConfig(
        name="custom_validator",
        enabled=False,
        metadata={"version": 1},
    )

    assert config.name == "custom_validator"
    assert config.enabled is False
    assert config.metadata == {"version": 1}


@pytest.mark.parametrize(
    "name, expected_exception",
    [
        (123, TypeError),
        ("", ValueError),
        ("   ", ValueError),
    ],
)
def test_validation_config_invalid_name(name, expected_exception):
    with pytest.raises(expected_exception):
        ValidationConfig(name=name)


def test_validation_config_invalid_enabled():
    with pytest.raises(TypeError):
        ValidationConfig(enabled="yes")


def test_validation_config_invalid_metadata():
    with pytest.raises(TypeError):
        ValidationConfig(metadata=["invalid"])


# ============================================================
# ValidationResult Tests
# ============================================================


def test_validation_result_valid_defaults():
    result = ValidationResult(
        validator_name="test_validator",
        is_valid=True,
    )

    assert result.validator_name == "test_validator"
    assert result.is_valid is True
    assert result.metrics == {}
    assert result.metadata == {}


def test_validation_result_custom_values():
    result = ValidationResult(
        validator_name="performance_validator",
        is_valid=True,
        metrics={
            "accuracy": 0.95,
            "f1_score": 0.92,
        },
        metadata={
            "samples": 100,
        },
    )

    assert result.metrics["accuracy"] == 0.95
    assert result.metrics["f1_score"] == 0.92
    assert result.metadata["samples"] == 100


@pytest.mark.parametrize(
    "validator_name, expected_exception",
    [
        (123, TypeError),
        ("", ValueError),
        ("   ", ValueError),
    ],
)
def test_validation_result_invalid_validator_name(
    validator_name,
    expected_exception,
):
    with pytest.raises(expected_exception):
        ValidationResult(
            validator_name=validator_name,
            is_valid=True,
        )


def test_validation_result_invalid_is_valid():
    with pytest.raises(TypeError):
        ValidationResult(
            validator_name="validator",
            is_valid="true",
        )


def test_validation_result_invalid_metrics_type():
    with pytest.raises(TypeError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metrics=["invalid"],
        )


def test_validation_result_invalid_metric_name_type():
    with pytest.raises(TypeError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metrics={123: 0.9},
        )


def test_validation_result_empty_metric_name():
    with pytest.raises(ValueError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metrics={"": 0.9},
        )


def test_validation_result_whitespace_metric_name():
    with pytest.raises(ValueError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metrics={"   ": 0.9},
        )


@pytest.mark.parametrize(
    "metric_value",
    [
        "0.9",
        None,
        True,
        [],
    ],
)
def test_validation_result_invalid_metric_value(metric_value):
    with pytest.raises(TypeError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metrics={"accuracy": metric_value},
        )


def test_validation_result_invalid_metadata_type():
    with pytest.raises(TypeError):
        ValidationResult(
            validator_name="validator",
            is_valid=True,
            metadata=["invalid"],
        )


# ============================================================
# BaseValidator Tests
# ============================================================


class ConcreteValidator(BaseValidator):
    def validate(self, data):
        return ValidationResult(
            validator_name=self.name,
            is_valid=True,
            metrics={"samples": len(data)},
        )


def test_base_validator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseValidator()


def test_concrete_validator_default_config():
    validator = ConcreteValidator()

    assert validator.name == "validation"
    assert isinstance(validator.config, ValidationConfig)


def test_concrete_validator_custom_config():
    config = ValidationConfig(name="custom_validator")

    validator = ConcreteValidator(config=config)

    assert validator.config is config
    assert validator.name == "custom_validator"


def test_base_validator_invalid_config():
    with pytest.raises(TypeError):
        ConcreteValidator(config="invalid")


def test_concrete_validator_validate():
    validator = ConcreteValidator(
        ValidationConfig(name="sample_validator")
    )

    result = validator.validate([1, 2, 3])

    assert isinstance(result, ValidationResult)
    assert result.validator_name == "sample_validator"
    assert result.is_valid is True
    assert result.metrics["samples"] == 3


# ============================================================
# Public API Tests
# ============================================================


def test_validation_public_imports():
    assert BaseValidator is not None
    assert ValidationConfig is not None
    assert ValidationResult is not None