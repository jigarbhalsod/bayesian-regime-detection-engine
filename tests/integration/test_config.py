import pytest

from src.integration.config import IntegrationConfig


def test_default_config():
    config = IntegrationConfig()

    assert config.integration_name == "model_integration"
    assert config.min_models == 1
    assert config.fail_fast is True
    assert config.include_metadata is True


def test_custom_config():
    config = IntegrationConfig(
        integration_name="advanced_models",
        min_models=3,
        fail_fast=False,
        include_metadata=False,
    )

    assert config.integration_name == "advanced_models"
    assert config.min_models == 3
    assert config.fail_fast is False
    assert config.include_metadata is False


@pytest.mark.parametrize(
    ("integration_name", "exception"),
    [
        (123, TypeError),
        ("", ValueError),
        ("   ", ValueError),
    ],
)
def test_invalid_integration_name(
    integration_name,
    exception,
):
    with pytest.raises(exception):
        IntegrationConfig(
            integration_name=integration_name
        )


@pytest.mark.parametrize(
    "min_models",
    [
        0,
        -1,
    ],
)
def test_invalid_min_models_values(min_models):
    with pytest.raises(ValueError):
        IntegrationConfig(
            min_models=min_models
        )


@pytest.mark.parametrize(
    "min_models",
    [
        1.5,
        True,
        "3",
        None,
    ],
)
def test_invalid_min_models_types(min_models):
    with pytest.raises(TypeError):
        IntegrationConfig(
            min_models=min_models
        )


@pytest.mark.parametrize(
    "fail_fast",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_invalid_fail_fast(fail_fast):
    with pytest.raises(TypeError):
        IntegrationConfig(
            fail_fast=fail_fast
        )


@pytest.mark.parametrize(
    "include_metadata",
    [
        1,
        0,
        "false",
        None,
    ],
)
def test_invalid_include_metadata(include_metadata):
    with pytest.raises(TypeError):
        IntegrationConfig(
            include_metadata=include_metadata
        )


def test_to_dict():
    config = IntegrationConfig(
        integration_name="test_integration",
        min_models=2,
        fail_fast=False,
        include_metadata=True,
    )

    assert config.to_dict() == {
        "integration_name": "test_integration",
        "min_models": 2,
        "fail_fast": False,
        "include_metadata": True,
    }


def test_config_is_frozen():
    config = IntegrationConfig()

    with pytest.raises(
        (AttributeError, TypeError),
    ):
        config.min_models = 10