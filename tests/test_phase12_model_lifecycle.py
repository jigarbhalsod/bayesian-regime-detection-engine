from __future__ import annotations

import pytest

from src.mlops import (
    LifecycleStage,
    ModelLifecycleManager,
    ModelLifecycleState,
    ModelVersion,
    Version,
)


def make_model_version(
    version: tuple[int, int, int] = (1, 0, 0),
) -> ModelVersion:
    return ModelVersion(
        model_name="regime_engine",
        version=Version(*version),
        model_type="HMM",
    )


# ---------------------------------------------------------------------------
# LifecycleStage
# ---------------------------------------------------------------------------


def test_lifecycle_stage_values() -> None:
    assert LifecycleStage.NONE.value == "none"
    assert LifecycleStage.DEVELOPMENT.value == "development"
    assert LifecycleStage.STAGING.value == "staging"
    assert LifecycleStage.PRODUCTION.value == "production"
    assert LifecycleStage.ARCHIVED.value == "archived"


def test_lifecycle_stage_is_string_enum() -> None:
    assert isinstance(LifecycleStage.PRODUCTION, str)


# ---------------------------------------------------------------------------
# ModelLifecycleState
# ---------------------------------------------------------------------------


def test_lifecycle_state_default_stage() -> None:
    state = ModelLifecycleState(
        model_version=make_model_version(),
    )

    assert state.stage is LifecycleStage.NONE


def test_lifecycle_state_explicit_stage() -> None:
    state = ModelLifecycleState(
        model_version=make_model_version(),
        stage=LifecycleStage.DEVELOPMENT,
    )

    assert state.stage is LifecycleStage.DEVELOPMENT


def test_lifecycle_state_rejects_invalid_model_version() -> None:
    with pytest.raises(TypeError, match="model_version must be a ModelVersion"):
        ModelLifecycleState(
            model_version="invalid",  # type: ignore[arg-type]
        )


def test_lifecycle_state_rejects_invalid_stage() -> None:
    with pytest.raises(TypeError, match="stage must be a LifecycleStage"):
        ModelLifecycleState(
            model_version=make_model_version(),
            stage="production",  # type: ignore[arg-type]
        )


def test_lifecycle_state_is_immutable() -> None:
    state = ModelLifecycleState(
        model_version=make_model_version(),
    )

    with pytest.raises(AttributeError):
        state.stage = LifecycleStage.PRODUCTION  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelLifecycleManager - Creation and Lookup
# ---------------------------------------------------------------------------


def test_manager_creates_initial_state() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    state = manager.create_state(version)

    assert state.model_version is version
    assert state.stage is LifecycleStage.NONE


def test_manager_rejects_invalid_model_version_creation() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(TypeError, match="model_version must be a ModelVersion"):
        manager.create_state("invalid")  # type: ignore[arg-type]


def test_manager_rejects_duplicate_state_creation() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    manager.create_state(version)

    with pytest.raises(ValueError, match="already exists"):
        manager.create_state(version)


def test_manager_gets_existing_state() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    created = manager.create_state(version)

    assert manager.get_state(version) == created


def test_manager_rejects_unknown_state_lookup() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(KeyError, match="not registered"):
        manager.get_state(make_model_version())


def test_manager_rejects_invalid_state_lookup_argument() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(TypeError, match="model_version must be a ModelVersion"):
        manager.get_state("invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Transition Rules
# ---------------------------------------------------------------------------


def test_allowed_transition_none_to_development() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.NONE,
        LifecycleStage.DEVELOPMENT,
    )


def test_allowed_transition_development_to_staging() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.DEVELOPMENT,
        LifecycleStage.STAGING,
    )


def test_allowed_transition_staging_to_production() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.STAGING,
        LifecycleStage.PRODUCTION,
    )


def test_allowed_transition_staging_to_development() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.STAGING,
        LifecycleStage.DEVELOPMENT,
    )


def test_allowed_transition_production_to_staging() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.PRODUCTION,
        LifecycleStage.STAGING,
    )


def test_allowed_transition_production_to_archived() -> None:
    manager = ModelLifecycleManager()

    assert manager.can_transition(
        LifecycleStage.PRODUCTION,
        LifecycleStage.ARCHIVED,
    )


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (LifecycleStage.NONE, LifecycleStage.STAGING),
        (LifecycleStage.NONE, LifecycleStage.PRODUCTION),
        (LifecycleStage.DEVELOPMENT, LifecycleStage.NONE),
        (LifecycleStage.DEVELOPMENT, LifecycleStage.PRODUCTION),
        (LifecycleStage.STAGING, LifecycleStage.NONE),
        (LifecycleStage.STAGING, LifecycleStage.ARCHIVED),
        (LifecycleStage.PRODUCTION, LifecycleStage.NONE),
        (LifecycleStage.PRODUCTION, LifecycleStage.DEVELOPMENT),
        (LifecycleStage.ARCHIVED, LifecycleStage.DEVELOPMENT),
        (LifecycleStage.ARCHIVED, LifecycleStage.PRODUCTION),
    ],
)
def test_invalid_transitions(
    current: LifecycleStage,
    target: LifecycleStage,
) -> None:
    manager = ModelLifecycleManager()

    assert not manager.can_transition(current, target)


def test_can_transition_rejects_invalid_current_stage() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(TypeError, match="current must be a LifecycleStage"):
        manager.can_transition(
            "none",  # type: ignore[arg-type]
            LifecycleStage.DEVELOPMENT,
        )


def test_can_transition_rejects_invalid_target_stage() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(TypeError, match="target must be a LifecycleStage"):
        manager.can_transition(
            LifecycleStage.NONE,
            "development",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# State Transitions
# ---------------------------------------------------------------------------


def test_manager_transitions_state() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    state = manager.create_state(version)

    result = manager.transition(
        state,
        LifecycleStage.DEVELOPMENT,
    )

    assert result.stage is LifecycleStage.DEVELOPMENT
    assert manager.get_state(version) == result


def test_manager_transitions_through_full_promotion_path() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    state = manager.create_state(version)

    state = manager.transition(
        state,
        LifecycleStage.DEVELOPMENT,
    )
    state = manager.transition(
        state,
        LifecycleStage.STAGING,
    )
    state = manager.transition(
        state,
        LifecycleStage.PRODUCTION,
    )
    state = manager.transition(
        state,
        LifecycleStage.ARCHIVED,
    )

    assert state.stage is LifecycleStage.ARCHIVED
    assert manager.get_state(version).stage is LifecycleStage.ARCHIVED


def test_manager_rejects_invalid_transition() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    state = manager.create_state(version)

    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        manager.transition(
            state,
            LifecycleStage.PRODUCTION,
        )


def test_manager_rejects_unregistered_state_transition() -> None:
    manager = ModelLifecycleManager()

    state = ModelLifecycleState(
        model_version=make_model_version(),
        stage=LifecycleStage.NONE,
    )

    with pytest.raises(KeyError, match="not registered"):
        manager.transition(
            state,
            LifecycleStage.DEVELOPMENT,
        )


def test_manager_rejects_invalid_state_transition_argument() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(TypeError, match="state must be a ModelLifecycleState"):
        manager.transition(
            "invalid",  # type: ignore[arg-type]
            LifecycleStage.DEVELOPMENT,
        )


def test_manager_rejects_invalid_transition_target() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()
    state = manager.create_state(version)

    with pytest.raises(TypeError, match="target must be a LifecycleStage"):
        manager.transition(
            state,
            "development",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_manager_promotes_model_version() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    manager.create_state(version)

    state = manager.promote(
        version,
        LifecycleStage.DEVELOPMENT,
    )

    assert state.stage is LifecycleStage.DEVELOPMENT
    assert manager.get_state(version) == state


def test_manager_promotes_to_production_through_valid_path() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    manager.create_state(version)

    manager.promote(
        version,
        LifecycleStage.DEVELOPMENT,
    )
    manager.promote(
        version,
        LifecycleStage.STAGING,
    )
    result = manager.promote(
        version,
        LifecycleStage.PRODUCTION,
    )

    assert result.stage is LifecycleStage.PRODUCTION


def test_manager_rejects_promotion_of_unknown_version() -> None:
    manager = ModelLifecycleManager()

    with pytest.raises(KeyError, match="not registered"):
        manager.promote(
            make_model_version(),
            LifecycleStage.DEVELOPMENT,
        )


def test_manager_rejects_invalid_promotion_target() -> None:
    manager = ModelLifecycleManager()
    version = make_model_version()

    manager.create_state(version)

    with pytest.raises(TypeError, match="target must be a LifecycleStage"):
        manager.promote(
            version,
            "development",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Multiple Model Versions
# ---------------------------------------------------------------------------


def test_manager_tracks_multiple_versions_independently() -> None:
    manager = ModelLifecycleManager()

    v1 = make_model_version((1, 0, 0))
    v2 = make_model_version((2, 0, 0))

    manager.create_state(v1)
    manager.create_state(v2)

    manager.promote(
        v1,
        LifecycleStage.DEVELOPMENT,
    )

    assert manager.get_state(v1).stage is LifecycleStage.DEVELOPMENT
    assert manager.get_state(v2).stage is LifecycleStage.NONE


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------


def test_lifecycle_public_exports() -> None:
    from src.mlops.lifecycle import (
        LifecycleStage as ExportedLifecycleStage,
        ModelLifecycleManager as ExportedModelLifecycleManager,
        ModelLifecycleState as ExportedModelLifecycleState,
    )

    assert ExportedLifecycleStage is LifecycleStage
    assert ExportedModelLifecycleManager is ModelLifecycleManager
    assert ExportedModelLifecycleState is ModelLifecycleState