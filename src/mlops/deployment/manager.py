from __future__ import annotations

from .config import DeploymentConfig
from .state import DeploymentState
from .status import DeploymentStatus


class DeploymentManager:
    """
    Manages deployment states and valid deployment lifecycle transitions.
    """

    _ALLOWED_TRANSITIONS: dict[
        DeploymentStatus,
        set[DeploymentStatus],
    ] = {
        DeploymentStatus.PENDING: {
            DeploymentStatus.DEPLOYING,
            DeploymentStatus.STOPPED,
        },
        DeploymentStatus.DEPLOYING: {
            DeploymentStatus.ACTIVE,
            DeploymentStatus.FAILED,
        },
        DeploymentStatus.ACTIVE: {
            DeploymentStatus.STOPPED,
        },
        DeploymentStatus.FAILED: {
            DeploymentStatus.DEPLOYING,
            DeploymentStatus.STOPPED,
        },
        DeploymentStatus.STOPPED: {
            DeploymentStatus.DEPLOYING,
        },
    }

    def __init__(self) -> None:
        self._states: dict[str, DeploymentState] = {}

    @staticmethod
    def _key(config: DeploymentConfig) -> str:
        """
        Build a unique key for a deployment.
        """
        return config.deployment_name

    def can_transition(
        self,
        current: DeploymentStatus,
        target: DeploymentStatus,
    ) -> bool:
        """
        Return whether a deployment status transition is allowed.
        """
        if not isinstance(current, DeploymentStatus):
            raise TypeError(
                "current must be a DeploymentStatus."
            )

        if not isinstance(target, DeploymentStatus):
            raise TypeError(
                "target must be a DeploymentStatus."
            )

        return target in self._ALLOWED_TRANSITIONS[current]

    def create_deployment(
        self,
        config: DeploymentConfig,
    ) -> DeploymentState:
        """
        Create and register a new deployment in PENDING state.
        """
        if not isinstance(config, DeploymentConfig):
            raise TypeError(
                "config must be a DeploymentConfig."
            )

        key = self._key(config)

        if key in self._states:
            raise ValueError(
                "Deployment already exists."
            )

        state = DeploymentState(
            config=config,
            status=DeploymentStatus.PENDING,
        )

        self._states[key] = state

        return state

    def get_deployment(
        self,
        deployment_name: str,
    ) -> DeploymentState:
        """
        Retrieve the current state of a deployment.
        """
        if not isinstance(deployment_name, str):
            raise TypeError(
                "deployment_name must be a string."
            )

        deployment_name = deployment_name.strip()

        if not deployment_name:
            raise ValueError(
                "deployment_name cannot be empty."
            )

        if deployment_name not in self._states:
            raise KeyError(
                "Deployment is not registered."
            )

        return self._states[deployment_name]

    def transition(
        self,
        deployment_name: str,
        target: DeploymentStatus,
        message: str | None = None,
    ) -> DeploymentState:
        """
        Transition a deployment to a new valid status.
        """
        if not isinstance(target, DeploymentStatus):
            raise TypeError(
                "target must be a DeploymentStatus."
            )

        current_state = self.get_deployment(
            deployment_name,
        )

        if not self.can_transition(
            current_state.status,
            target,
        ):
            raise ValueError(
                f"Invalid deployment transition: "
                f"{current_state.status.value} -> {target.value}."
            )

        new_state = DeploymentState(
            config=current_state.config,
            status=target,
            message=message,
            metadata=current_state.metadata,
        )

        key = self._key(current_state.config)
        self._states[key] = new_state

        return new_state

    def deploy(
        self,
        deployment_name: str,
    ) -> DeploymentState:
        """
        Begin deployment by moving to DEPLOYING.
        """
        return self.transition(
            deployment_name,
            DeploymentStatus.DEPLOYING,
        )

    def activate(
        self,
        deployment_name: str,
        message: str | None = None,
    ) -> DeploymentState:
        """
        Mark a deploying deployment as ACTIVE.
        """
        return self.transition(
            deployment_name,
            DeploymentStatus.ACTIVE,
            message=message,
        )

    def fail(
        self,
        deployment_name: str,
        message: str,
    ) -> DeploymentState:
        """
        Mark a deploying deployment as FAILED.
        """
        return self.transition(
            deployment_name,
            DeploymentStatus.FAILED,
            message=message,
        )

    def stop(
        self,
        deployment_name: str,
        message: str | None = None,
    ) -> DeploymentState:
        """
        Stop a deployment when the transition is valid.
        """
        return self.transition(
            deployment_name,
            DeploymentStatus.STOPPED,
            message=message,
        )