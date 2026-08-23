from __future__ import annotations

import pytest

from src.mlops import (
    ModelRegistry,
    ModelVersion,
    RegisteredModel,
    Version,
)


# ---------------------------------------------------------------------------
# RegisteredModel
# ---------------------------------------------------------------------------


def test_registered_model_creation() -> None:
    model = RegisteredModel(
        name="regime_engine",
        description="Market regime detection model",
        tags=("HMM", "Bayesian"),
        metadata={"regimes": 5},
    )

    assert model.name == "regime_engine"
    assert model.description == "Market regime detection model"
    assert model.tags == ("HMM", "Bayesian")
    assert model.metadata == {"regimes": 5}


def test_registered_model_strips_fields() -> None:
    model = RegisteredModel(
        name=" regime_engine ",
        description=" Market regime detection ",
        tags=(" HMM ", " Bayesian "),
    )

    assert model.name == "regime_engine"
    assert model.description == "Market regime detection"
    assert model.tags == ("HMM", "Bayesian")


def test_registered_model_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        RegisteredModel(name=" ")


def test_registered_model_rejects_empty_tags() -> None:
    with pytest.raises(ValueError, match="tags must not contain empty"):
        RegisteredModel(
            name="regime_engine",
            tags=("HMM", " "),
        )


def test_registered_model_rejects_invalid_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        RegisteredModel(
            name="regime_engine",
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_registered_model_is_immutable() -> None:
    model = RegisteredModel(name="regime_engine")

    with pytest.raises(AttributeError):
        model.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


def test_registry_registers_model() -> None:
    registry = ModelRegistry()
    model = RegisteredModel(name="regime_engine")

    result = registry.register_model(model)

    assert result is model
    assert registry.get_model("regime_engine") is model


def test_registry_rejects_invalid_model() -> None:
    registry = ModelRegistry()

    with pytest.raises(TypeError, match="RegisteredModel"):
        registry.register_model("invalid")  # type: ignore[arg-type]


def test_registry_rejects_duplicate_model() -> None:
    registry = ModelRegistry()
    model = RegisteredModel(name="regime_engine")

    registry.register_model(model)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(model)


def test_registry_retrieves_model() -> None:
    registry = ModelRegistry()
    model = RegisteredModel(name="regime_engine")

    registry.register_model(model)

    assert registry.get_model("regime_engine") == model


def test_registry_lookup_strips_name() -> None:
    registry = ModelRegistry()
    model = RegisteredModel(name="regime_engine")

    registry.register_model(model)

    assert registry.get_model(" regime_engine ") == model


def test_registry_rejects_empty_model_lookup() -> None:
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="model name must not be empty"):
        registry.get_model(" ")


def test_registry_rejects_unknown_model_lookup() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get_model("unknown")


# ---------------------------------------------------------------------------
# Model Versions
# ---------------------------------------------------------------------------


def test_registry_registers_model_version() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    version = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    result = registry.register_version(version)

    assert result is version


def test_registry_rejects_invalid_model_version() -> None:
    registry = ModelRegistry()

    with pytest.raises(TypeError, match="ModelVersion"):
        registry.register_version("invalid")  # type: ignore[arg-type]


def test_registry_rejects_version_for_unknown_model() -> None:
    registry = ModelRegistry()

    version = ModelVersion(
        model_name="unknown",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    with pytest.raises(KeyError, match="not registered"):
        registry.register_version(version)


def test_registry_rejects_duplicate_model_version() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    version = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_version(version)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_version(version)


def test_registry_retrieves_model_version() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    version = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_version(version)

    assert registry.get_version("regime_engine", "1.0.0") == version


def test_registry_version_lookup_strips_fields() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    version = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_version(version)

    assert (
        registry.get_version(
            " regime_engine ",
            " 1.0.0 ",
        )
        == version
    )


def test_registry_rejects_empty_version_lookup_fields() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    with pytest.raises(ValueError, match="model_name must not be empty"):
        registry.get_version(" ", "1.0.0")

    with pytest.raises(ValueError, match="version must not be empty"):
        registry.get_version("regime_engine", " ")


def test_registry_rejects_unknown_version() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    with pytest.raises(KeyError, match="not registered"):
        registry.get_version("regime_engine", "1.0.0")


def test_registry_lists_versions_in_order() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    v2 = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 1, 0),
        model_type="BayesianHMM",
    )
    v1 = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_version(v2)
    registry.register_version(v1)

    assert registry.list_versions("regime_engine") == [v1, v2]


def test_registry_returns_empty_version_list_for_model() -> None:
    registry = ModelRegistry()
    registry.register_model(RegisteredModel(name="regime_engine"))

    assert registry.list_versions("regime_engine") == []


def test_registry_rejects_unknown_model_version_list() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.list_versions("unknown")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_registry_public_exports() -> None:
    from src.mlops.registry import (
        ModelRegistry as ExportedModelRegistry,
        RegisteredModel as ExportedRegisteredModel,
    )

    assert ExportedModelRegistry is ModelRegistry
    assert ExportedRegisteredModel is RegisteredModel