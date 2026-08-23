from __future__ import annotations

from src.mlops.versioning import ModelVersion

from .stage import LifecycleStage
from .state import ModelLifecycleState


class ModelLifecycleManager:
    """
    Manages lifecycle states and valid stage transitions
    for model versions.
    """

    _ALLOWED_TRANSITIONS: dict[LifecycleStage, set[LifecycleStage]] = {
        LifecycleStage.NONE: {
            LifecycleStage.DEVELOPMENT,
        },
        LifecycleStage.DEVELOPMENT: {
            LifecycleStage.STAGING,
        },
        LifecycleStage.STAGING: {
            LifecycleStage.DEVELOPMENT,
            LifecycleStage.PRODUCTION,
        },
        LifecycleStage.PRODUCTION: {
            LifecycleStage.STAGING,
            LifecycleStage.ARCHIVED,
        },
        LifecycleStage.ARCHIVED: set(),
    }

    def __init__(self) -> None:
        self._states: dict[str, ModelLifecycleState] = {}

    @staticmethod
    def _key(model_version: ModelVersion) -> str:
        """
        Build a unique lifecycle key for a model version.
        """
        return (
            f"{model_version.model_name}:"
            f"{model_version.version}"
        )

    def can_transition(
        self,
        current: LifecycleStage,
        target: LifecycleStage,
    ) -> bool:
        """
        Return whether a lifecycle transition is allowed.
        """
        if not isinstance(current, LifecycleStage):
            raise TypeError(
                "current must be a LifecycleStage."
            )

        if not isinstance(target, LifecycleStage):
            raise TypeError(
                "target must be a LifecycleStage."
            )

        return target in self._ALLOWED_TRANSITIONS[current]

    def create_state(
        self,
        model_version: ModelVersion,
    ) -> ModelLifecycleState:
        """
        Create and register an initial lifecycle state.
        """
        if not isinstance(model_version, ModelVersion):
            raise TypeError(
                "model_version must be a ModelVersion."
            )

        key = self._key(model_version)

        if key in self._states:
            raise ValueError(
                "Lifecycle state already exists for this model version."
            )

        state = ModelLifecycleState(
            model_version=model_version,
            stage=LifecycleStage.NONE,
        )

        self._states[key] = state

        return state

    def get_state(
        self,
        model_version: ModelVersion,
    ) -> ModelLifecycleState:
        """
        Retrieve the lifecycle state for a model version.
        """
        if not isinstance(model_version, ModelVersion):
            raise TypeError(
                "model_version must be a ModelVersion."
            )

        key = self._key(model_version)

        if key not in self._states:
            raise KeyError(
                "Lifecycle state is not registered for this model version."
            )

        return self._states[key]

    def transition(
        self,
        state: ModelLifecycleState,
        target: LifecycleStage,
    ) -> ModelLifecycleState:
        """
        Transition and persist a model lifecycle state.
        """
        if not isinstance(state, ModelLifecycleState):
            raise TypeError(
                "state must be a ModelLifecycleState."
            )

        if not isinstance(target, LifecycleStage):
            raise TypeError(
                "target must be a LifecycleStage."
            )

        key = self._key(state.model_version)

        if key not in self._states:
            raise KeyError(
                "Lifecycle state is not registered for this model version."
            )

        current_state = self._states[key]

        if not self.can_transition(
            current_state.stage,
            target,
        ):
            raise ValueError(
                f"Invalid lifecycle transition: "
                f"{current_state.stage.value} -> {target.value}."
            )

        new_state = ModelLifecycleState(
            model_version=current_state.model_version,
            stage=target,
        )

        self._states[key] = new_state

        return new_state

    def promote(
        self,
        model_version: ModelVersion,
        target: LifecycleStage,
    ) -> ModelLifecycleState:
        """
        Promote a registered model version to a valid lifecycle stage.
        """
        state = self.get_state(model_version)

        return self.transition(
            state=state,
            target=target,
        )