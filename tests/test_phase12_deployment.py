from __future__ import annotations

import pytest

from src.mlops import (
    DeploymentConfig,
    DeploymentManager,
    DeploymentState,
    DeploymentStatus,
)


def make_config(
    deployment_name: str = "regime-engine-prod",
) -> DeploymentConfig:
    return DeploymentConfig(
        deployment_name=deployment_name,
        environment="production",
        model_name="regime_engine",
        model_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# DeploymentConfig
# ---------------------------------------------------------------------------


def test_deployment_config_creation() -> None:
    config = make_config()

    assert config.deployment_name == "regime-engine-prod"
    assert config.environment == "production"
    assert config.model_name == "regime_engine"
    assert config.model_version == "1.0.0"
    assert config.replicas == 1
    assert config.endpoint is None
    assert dict(config.metadata) == {}


def test_deployment_config_strips_fields() -> None:
    config = DeploymentConfig(
        deployment_name=" regime-prod ",
        environment=" production ",
        model_name=" regime_engine ",
        model_version=" 1.0.0 ",
        endpoint=" /predict ",
    )

    assert config.deployment_name == "regime-prod"
    assert config.environment == "production"
    assert config.model_name == "regime_engine"
    assert config.model_version == "1.0.0"
    assert config.endpoint == "/predict"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deployment_name", "   "),
        ("environment", "   "),
        ("model_name", "   "),
        ("model_version", "   "),
    ],
)
def test_deployment_config_rejects_empty_required_fields(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "deployment_name": "regime-prod",
        "environment": "production",
        "model_name": "regime_engine",
        "model_version": "1.0.0",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        DeploymentConfig(**kwargs)


def test_deployment_config_rejects_invalid_replicas_type() -> None:
    with pytest.raises(TypeError, match="replicas must be an integer"):
        DeploymentConfig(
            deployment_name="regime-prod",
            environment="production",
            model_name="regime_engine",
            model_version="1.0.0",
            replicas="3",  # type: ignore[arg-type]
        )


def test_deployment_config_rejects_boolean_replicas() -> None:
    with pytest.raises(TypeError, match="replicas must be an integer"):
        DeploymentConfig(
            deployment_name="regime-prod",
            environment="production",
            model_name="regime_engine",
            model_version="1.0.0",
            replicas=True,
        )


def test_deployment_config_rejects_replicas_less_than_one() -> None:
    with pytest.raises(ValueError, match="replicas must be at least 1"):
        DeploymentConfig(
            deployment_name="regime-prod",
            environment="production",
            model_name="regime_engine",
            model_version="1.0.0",
            replicas=0,
        )


def test_deployment_config_rejects_invalid_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint cannot be empty"):
        DeploymentConfig(
            deployment_name="regime-prod",
            environment="production",
            model_name="regime_engine",
            model_version="1.0.0",
            endpoint="   ",
        )


def test_deployment_config_rejects_invalid_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        DeploymentConfig(
            deployment_name="regime-prod",
            environment="production",
            model_name="regime_engine",
            model_version="1.0.0",
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_deployment_config_is_immutable() -> None:
    config = make_config()

    with pytest.raises(AttributeError):
        config.environment = "staging"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeploymentStatus
# ---------------------------------------------------------------------------


def test_deployment_status_values() -> None:
    assert DeploymentStatus.PENDING.value == "pending"
    assert DeploymentStatus.DEPLOYING.value == "deploying"
    assert DeploymentStatus.ACTIVE.value == "active"
    assert DeploymentStatus.FAILED.value == "failed"
    assert DeploymentStatus.STOPPED.value == "stopped"


def test_deployment_status_is_string_enum() -> None:
    assert isinstance(DeploymentStatus.ACTIVE, str)


# ---------------------------------------------------------------------------
# DeploymentState
# ---------------------------------------------------------------------------


def test_deployment_state_defaults() -> None:
    state = DeploymentState(config=make_config())

    assert state.status is DeploymentStatus.PENDING
    assert state.message is None
    assert dict(state.metadata) == {}


def test_deployment_state_normalizes_message() -> None:
    state = DeploymentState(
        config=make_config(),
        status=DeploymentStatus.ACTIVE,
        message=" Deployment successful ",
        metadata={"ready": True},
    )

    assert state.message == "Deployment successful"
    assert dict(state.metadata) == {"ready": True}


def test_deployment_state_rejects_invalid_config() -> None:
    with pytest.raises(TypeError, match="config must be a DeploymentConfig"):
        DeploymentState(
            config="invalid",  # type: ignore[arg-type]
        )


def test_deployment_state_rejects_invalid_status() -> None:
    with pytest.raises(TypeError, match="status must be a DeploymentStatus"):
        DeploymentState(
            config=make_config(),
            status="active",  # type: ignore[arg-type]
        )


def test_deployment_state_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="message cannot be empty"):
        DeploymentState(
            config=make_config(),
            message="   ",
        )


def test_deployment_state_rejects_invalid_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        DeploymentState(
            config=make_config(),
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_deployment_state_is_immutable() -> None:
    state = DeploymentState(config=make_config())

    with pytest.raises(AttributeError):
        state.status = DeploymentStatus.ACTIVE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeploymentManager - Creation and Lookup
# ---------------------------------------------------------------------------


def test_manager_creates_deployment() -> None:
    manager = DeploymentManager()

    state = manager.create_deployment(make_config())

    assert state.status is DeploymentStatus.PENDING


def test_manager_rejects_invalid_config() -> None:
    manager = DeploymentManager()

    with pytest.raises(TypeError, match="config must be a DeploymentConfig"):
        manager.create_deployment("invalid")  # type: ignore[arg-type]


def test_manager_rejects_duplicate_deployment() -> None:
    manager = DeploymentManager()
    config = make_config()

    manager.create_deployment(config)

    with pytest.raises(ValueError, match="already exists"):
        manager.create_deployment(config)


def test_manager_gets_existing_deployment() -> None:
    manager = DeploymentManager()
    created = manager.create_deployment(make_config())

    result = manager.get_deployment("regime-engine-prod")

    assert result == created


def test_manager_lookup_strips_name() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())

    result = manager.get_deployment(" regime-engine-prod ")

    assert result.status is DeploymentStatus.PENDING


def test_manager_rejects_empty_deployment_lookup() -> None:
    manager = DeploymentManager()

    with pytest.raises(ValueError, match="deployment_name cannot be empty"):
        manager.get_deployment("   ")


def test_manager_rejects_invalid_deployment_lookup_type() -> None:
    manager = DeploymentManager()

    with pytest.raises(TypeError, match="deployment_name must be a string"):
        manager.get_deployment(123)  # type: ignore[arg-type]


def test_manager_rejects_unknown_deployment() -> None:
    manager = DeploymentManager()

    with pytest.raises(KeyError, match="not registered"):
        manager.get_deployment("unknown")


# ---------------------------------------------------------------------------
# Deployment Transition Rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (DeploymentStatus.PENDING, DeploymentStatus.DEPLOYING),
        (DeploymentStatus.PENDING, DeploymentStatus.STOPPED),
        (DeploymentStatus.DEPLOYING, DeploymentStatus.ACTIVE),
        (DeploymentStatus.DEPLOYING, DeploymentStatus.FAILED),
        (DeploymentStatus.ACTIVE, DeploymentStatus.STOPPED),
        (DeploymentStatus.FAILED, DeploymentStatus.DEPLOYING),
        (DeploymentStatus.FAILED, DeploymentStatus.STOPPED),
        (DeploymentStatus.STOPPED, DeploymentStatus.DEPLOYING),
    ],
)
def test_allowed_deployment_transitions(
    current: DeploymentStatus,
    target: DeploymentStatus,
) -> None:
    manager = DeploymentManager()

    assert manager.can_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (DeploymentStatus.PENDING, DeploymentStatus.ACTIVE),
        (DeploymentStatus.PENDING, DeploymentStatus.FAILED),
        (DeploymentStatus.DEPLOYING, DeploymentStatus.PENDING),
        (DeploymentStatus.ACTIVE, DeploymentStatus.DEPLOYING),
        (DeploymentStatus.ACTIVE, DeploymentStatus.FAILED),
        (DeploymentStatus.FAILED, DeploymentStatus.ACTIVE),
        (DeploymentStatus.STOPPED, DeploymentStatus.ACTIVE),
        (DeploymentStatus.STOPPED, DeploymentStatus.FAILED),
    ],
)
def test_invalid_deployment_transitions(
    current: DeploymentStatus,
    target: DeploymentStatus,
) -> None:
    manager = DeploymentManager()

    assert not manager.can_transition(current, target)


def test_can_transition_rejects_invalid_current() -> None:
    manager = DeploymentManager()

    with pytest.raises(TypeError, match="current must be a DeploymentStatus"):
        manager.can_transition(
            "pending",  # type: ignore[arg-type]
            DeploymentStatus.DEPLOYING,
        )


def test_can_transition_rejects_invalid_target() -> None:
    manager = DeploymentManager()

    with pytest.raises(TypeError, match="target must be a DeploymentStatus"):
        manager.can_transition(
            DeploymentStatus.PENDING,
            "deploying",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Deployment Lifecycle Operations
# ---------------------------------------------------------------------------


def test_manager_deploys_and_activates() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())

    deploying = manager.deploy("regime-engine-prod")
    active = manager.activate(
        "regime-engine-prod",
        "Deployment successful",
    )

    assert deploying.status is DeploymentStatus.DEPLOYING
    assert active.status is DeploymentStatus.ACTIVE
    assert active.message == "Deployment successful"


def test_manager_fails_deployment() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())
    manager.deploy("regime-engine-prod")

    result = manager.fail(
        "regime-engine-prod",
        "Health check failed",
    )

    assert result.status is DeploymentStatus.FAILED
    assert result.message == "Health check failed"


def test_manager_retries_failed_deployment() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())
    manager.deploy("regime-engine-prod")
    manager.fail("regime-engine-prod", "Health check failed")

    result = manager.deploy("regime-engine-prod")

    assert result.status is DeploymentStatus.DEPLOYING


def test_manager_stops_active_deployment() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())
    manager.deploy("regime-engine-prod")
    manager.activate("regime-engine-prod")

    result = manager.stop(
        "regime-engine-prod",
        "Manual shutdown",
    )

    assert result.status is DeploymentStatus.STOPPED
    assert result.message == "Manual shutdown"


def test_manager_stops_pending_deployment() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())

    result = manager.stop("regime-engine-prod")

    assert result.status is DeploymentStatus.STOPPED


def test_manager_rejects_invalid_lifecycle_transition() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())

    with pytest.raises(ValueError, match="Invalid deployment transition"):
        manager.activate("regime-engine-prod")


def test_manager_rejects_invalid_transition_target() -> None:
    manager = DeploymentManager()
    manager.create_deployment(make_config())

    with pytest.raises(TypeError, match="target must be a DeploymentStatus"):
        manager.transition(
            "regime-engine-prod",
            "active",  # type: ignore[arg-type]
        )


def test_manager_rejects_unknown_transition() -> None:
    manager = DeploymentManager()

    with pytest.raises(KeyError, match="not registered"):
        manager.deploy("unknown")


# ---------------------------------------------------------------------------
# Multiple Deployments
# ---------------------------------------------------------------------------


def test_manager_tracks_multiple_deployments_independently() -> None:
    manager = DeploymentManager()

    prod = make_config("regime-engine-prod")
    staging = DeploymentConfig(
        deployment_name="regime-engine-staging",
        environment="staging",
        model_name="regime_engine",
        model_version="1.1.0",
    )

    manager.create_deployment(prod)
    manager.create_deployment(staging)

    manager.deploy("regime-engine-prod")
    manager.activate("regime-engine-prod")

    assert (
        manager.get_deployment(
            "regime-engine-prod"
        ).status
        is DeploymentStatus.ACTIVE
    )
    assert (
        manager.get_deployment(
            "regime-engine-staging"
        ).status
        is DeploymentStatus.PENDING
    )


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------


def test_deployment_public_exports() -> None:
    from src.mlops.deployment import (
        DeploymentConfig as ExportedDeploymentConfig,
        DeploymentManager as ExportedDeploymentManager,
        DeploymentState as ExportedDeploymentState,
        DeploymentStatus as ExportedDeploymentStatus,
    )

    assert ExportedDeploymentConfig is DeploymentConfig
    assert ExportedDeploymentManager is DeploymentManager
    assert ExportedDeploymentState is DeploymentState
    assert ExportedDeploymentStatus is DeploymentStatus