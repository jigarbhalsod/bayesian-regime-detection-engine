import pytest

from src.integration.registry import ModelRegistry


class DummyModel:
    pass


class AnotherModel:
    pass


def test_empty_registry():
    registry = ModelRegistry()

    assert registry.count == 0
    assert len(registry) == 0
    assert registry.names == ()


def test_register_and_get_model():
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(
        "hmm",
        model,
    )

    assert registry.count == 1
    assert registry.get("hmm") is model
    assert registry.require("hmm") is model


def test_register_name_is_stripped():
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(
        "  hmm  ",
        model,
    )

    assert registry.names == ("hmm",)
    assert registry.get(" hmm ") is model


def test_register_none_model_rejected():
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            "hmm",
            None,
        )


@pytest.mark.parametrize(
    "name",
    [
        None,
        123,
        True,
    ],
)
def test_invalid_name_types_rejected(name):
    registry = ModelRegistry()

    with pytest.raises(TypeError):
        registry.register(
            name,
            DummyModel(),
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_empty_names_rejected(name):
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.register(
            name,
            DummyModel(),
        )


def test_duplicate_registration_rejected():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        DummyModel(),
    )

    with pytest.raises(ValueError):
        registry.register(
            "hmm",
            AnotherModel(),
        )


def test_overwrite_registration():
    registry = ModelRegistry()

    first_model = DummyModel()
    second_model = AnotherModel()

    registry.register(
        "hmm",
        first_model,
    )

    registry.register(
        "hmm",
        second_model,
        overwrite=True,
    )

    assert registry.count == 1
    assert registry.require("hmm") is second_model


@pytest.mark.parametrize(
    "overwrite",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_invalid_overwrite_type(overwrite):
    registry = ModelRegistry()

    with pytest.raises(TypeError):
        registry.register(
            "hmm",
            DummyModel(),
            overwrite=overwrite,
        )


def test_get_missing_returns_none():
    registry = ModelRegistry()

    assert registry.get("missing") is None


def test_get_missing_returns_default():
    registry = ModelRegistry()

    assert (
        registry.get(
            "missing",
            default="fallback",
        )
        == "fallback"
    )


def test_require_missing_raises_key_error():
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.require("missing")


def test_unregister_model():
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(
        "hmm",
        model,
    )

    removed = registry.unregister("hmm")

    assert removed is model
    assert registry.count == 0
    assert registry.names == ()


def test_unregister_missing_raises_key_error():
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.unregister("missing")


def test_clear_registry():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        DummyModel(),
    )

    registry.register(
        "bayesian",
        AnotherModel(),
    )

    registry.clear()

    assert registry.count == 0
    assert registry.names == ()


def test_contains():
    registry = ModelRegistry()
    registry.register(
        "hmm",
        DummyModel(),
    )

    assert registry.contains("hmm") is True
    assert registry.contains(" hmm ") is True
    assert registry.contains("missing") is False


def test_contains_operator():
    registry = ModelRegistry()
    registry.register(
        "hmm",
        DummyModel(),
    )

    assert "hmm" in registry
    assert " hmm " in registry
    assert "missing" not in registry
    assert 123 not in registry
    assert None not in registry


def test_invalid_name_for_get_rejected():
    registry = ModelRegistry()

    with pytest.raises(TypeError):
        registry.get(None)


def test_empty_name_for_require_rejected():
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.require("")


def test_multiple_models_preserve_order():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        DummyModel(),
    )

    registry.register(
        "regime_var",
        AnotherModel(),
    )

    registry.register(
        "bayesian",
        DummyModel(),
    )

    assert registry.names == (
        "hmm",
        "regime_var",
        "bayesian",
    )