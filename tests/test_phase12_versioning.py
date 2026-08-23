from __future__ import annotations

import pytest

from src.mlops import (
    ModelVersion,
    Version,
    VersionedArtifact,
    VersionRegistry,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_creation() -> None:
    version = Version(1, 2, 3)

    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3


def test_version_string() -> None:
    assert str(Version(1, 2, 3)) == "1.2.3"


def test_version_parsing() -> None:
    assert Version.parse("2.4.1") == Version(2, 4, 1)


def test_version_parsing_strips_whitespace() -> None:
    assert Version.parse(" 1.2.3 ") == Version(1, 2, 3)


def test_version_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        Version(-1, 0, 0)


def test_version_rejects_non_integer_values() -> None:
    with pytest.raises(TypeError):
        Version(1.0, 0, 0)  # type: ignore[arg-type]


def test_version_parse_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        Version.parse("1.2")


def test_version_bump_major() -> None:
    assert Version(1, 2, 3).bump_major() == Version(2, 0, 0)


def test_version_bump_minor() -> None:
    assert Version(1, 2, 3).bump_minor() == Version(1, 3, 0)


def test_version_bump_patch() -> None:
    assert Version(1, 2, 3).bump_patch() == Version(1, 2, 4)


def test_version_ordering() -> None:
    assert Version(1, 0, 0) < Version(2, 0, 0)
    assert Version(1, 2, 0) < Version(1, 3, 0)
    assert Version(1, 2, 3) < Version(1, 2, 4)


# ---------------------------------------------------------------------------
# VersionedArtifact
# ---------------------------------------------------------------------------


def test_versioned_artifact_creation() -> None:
    artifact = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
        source_run_id="run_001",
        metadata={"features": 68},
    )

    assert artifact.name == "feature_schema"
    assert artifact.version == Version(1, 0, 0)
    assert artifact.artifact_type == "schema"
    assert artifact.source_run_id == "run_001"
    assert artifact.metadata["features"] == 68


def test_versioned_artifact_strips_name_and_type() -> None:
    artifact = VersionedArtifact(
        name="  feature_schema  ",
        version=Version(1, 0, 0),
        artifact_type="  schema  ",
    )

    assert artifact.name == "feature_schema"
    assert artifact.artifact_type == "schema"


def test_versioned_artifact_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        VersionedArtifact(
            name=" ",
            version=Version(1, 0, 0),
            artifact_type="schema",
        )


def test_versioned_artifact_rejects_empty_type() -> None:
    with pytest.raises(ValueError, match="artifact_type"):
        VersionedArtifact(
            name="feature_schema",
            version=Version(1, 0, 0),
            artifact_type=" ",
        )


def test_versioned_artifact_requires_version() -> None:
    with pytest.raises(TypeError, match="Version"):
        VersionedArtifact(
            name="feature_schema",
            version="1.0.0",  # type: ignore[arg-type]
            artifact_type="schema",
        )


def test_versioned_artifact_is_immutable() -> None:
    artifact = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
    )

    with pytest.raises(AttributeError):
        artifact.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------


def test_model_version_creation() -> None:
    model = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
        experiment_id="exp_001",
        source_run_id="run_001",
        metadata={"regimes": 5},
    )

    assert model.model_name == "regime_engine"
    assert model.version == Version(1, 0, 0)
    assert model.model_type == "HMM"
    assert model.experiment_id == "exp_001"
    assert model.source_run_id == "run_001"
    assert model.metadata["regimes"] == 5


def test_model_version_strips_fields() -> None:
    model = ModelVersion(
        model_name="  regime_engine  ",
        version=Version(1, 0, 0),
        model_type="  HMM  ",
    )

    assert model.model_name == "regime_engine"
    assert model.model_type == "HMM"


def test_model_version_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        ModelVersion(
            model_name=" ",
            version=Version(1, 0, 0),
            model_type="HMM",
        )


def test_model_version_rejects_empty_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        ModelVersion(
            model_name="regime_engine",
            version=Version(1, 0, 0),
            model_type=" ",
        )


def test_model_version_is_immutable() -> None:
    model = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    with pytest.raises(AttributeError):
        model.model_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VersionRegistry
# ---------------------------------------------------------------------------


def test_registry_registers_artifact() -> None:
    registry = VersionRegistry()

    artifact = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
    )

    registry.register_artifact(artifact)

    assert registry.get_artifact(
        "feature_schema",
        "1.0.0",
    ) is artifact


def test_registry_registers_model() -> None:
    registry = VersionRegistry()

    model = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_model(model)

    assert registry.get_model(
        "regime_engine",
        "1.0.0",
    ) is model


def test_registry_rejects_duplicate_artifact_version() -> None:
    registry = VersionRegistry()

    artifact = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
    )

    registry.register_artifact(artifact)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_artifact(artifact)


def test_registry_rejects_duplicate_model_version() -> None:
    registry = VersionRegistry()

    model = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )

    registry.register_model(model)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(model)


def test_registry_lists_artifact_versions() -> None:
    registry = VersionRegistry()

    v1 = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
    )
    v2 = VersionedArtifact(
        name="feature_schema",
        version=Version(2, 0, 0),
        artifact_type="schema",
    )

    registry.register_artifact(v1)
    registry.register_artifact(v2)

    versions = registry.list_artifact_versions(
        "feature_schema"
    )

    assert len(versions) == 2
    assert {str(item.version) for item in versions} == {
        "1.0.0",
        "2.0.0",
    }


def test_registry_lists_model_versions() -> None:
    registry = VersionRegistry()

    v1 = ModelVersion(
        model_name="regime_engine",
        version=Version(1, 0, 0),
        model_type="HMM",
    )
    v2 = ModelVersion(
        model_name="regime_engine",
        version=Version(2, 0, 0),
        model_type="BayesianHMM",
    )

    registry.register_model(v1)
    registry.register_model(v2)

    versions = registry.list_model_versions(
        "regime_engine"
    )

    assert len(versions) == 2
    assert {str(item.version) for item in versions} == {
        "1.0.0",
        "2.0.0",
    }


def test_registry_rejects_unknown_artifact() -> None:
    registry = VersionRegistry()

    with pytest.raises(KeyError, match="does not exist"):
        registry.get_artifact(
            "unknown",
            "1.0.0",
        )


def test_registry_rejects_unknown_model() -> None:
    registry = VersionRegistry()

    with pytest.raises(KeyError, match="does not exist"):
        registry.get_model(
            "unknown",
            "1.0.0",
        )


def test_registry_supports_multiple_independent_artifacts() -> None:
    registry = VersionRegistry()

    feature_schema = VersionedArtifact(
        name="feature_schema",
        version=Version(1, 0, 0),
        artifact_type="schema",
    )
    metrics_report = VersionedArtifact(
        name="metrics_report",
        version=Version(1, 0, 0),
        artifact_type="report",
    )

    registry.register_artifact(feature_schema)
    registry.register_artifact(metrics_report)

    assert registry.get_artifact(
        "feature_schema",
        "1.0.0",
    ) is feature_schema

    assert registry.get_artifact(
        "metrics_report",
        "1.0.0",
    ) is metrics_report