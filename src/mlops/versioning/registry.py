from __future__ import annotations

from .artifact import VersionedArtifact
from .model_version import ModelVersion


class VersionRegistry:
    """
    In-memory registry for model and artifact versions.
    """

    def __init__(self) -> None:
        self._artifacts: dict[
            tuple[str, str],
            VersionedArtifact,
        ] = {}

        self._models: dict[
            tuple[str, str],
            ModelVersion,
        ] = {}

    def register_artifact(
        self,
        artifact: VersionedArtifact,
    ) -> None:
        """
        Register a versioned artifact.
        """
        if not isinstance(artifact, VersionedArtifact):
            raise TypeError(
                "artifact must be a VersionedArtifact."
            )

        key = (
            artifact.name,
            str(artifact.version),
        )

        if key in self._artifacts:
            raise ValueError(
                f"Artifact '{artifact.name}' version "
                f"'{artifact.version}' is already registered."
            )

        self._artifacts[key] = artifact

    def register_model(
        self,
        model: ModelVersion,
    ) -> None:
        """
        Register a model version.
        """
        if not isinstance(model, ModelVersion):
            raise TypeError(
                "model must be a ModelVersion."
            )

        key = (
            model.model_name,
            str(model.version),
        )

        if key in self._models:
            raise ValueError(
                f"Model '{model.model_name}' version "
                f"'{model.version}' is already registered."
            )

        self._models[key] = model

    def get_artifact(
        self,
        name: str,
        version: str,
    ) -> VersionedArtifact:
        """
        Retrieve a specific artifact version.
        """
        name = name.strip()
        version = version.strip()

        if not name:
            raise ValueError("name must not be empty.")

        if not version:
            raise ValueError("version must not be empty.")

        key = (name, version)

        try:
            return self._artifacts[key]
        except KeyError as exc:
            raise KeyError(
                f"Artifact '{name}' version '{version}' "
                "does not exist."
            ) from exc

    def get_model(
        self,
        model_name: str,
        version: str,
    ) -> ModelVersion:
        """
        Retrieve a specific model version.
        """
        model_name = model_name.strip()
        version = version.strip()

        if not model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        if not version:
            raise ValueError(
                "version must not be empty."
            )

        key = (model_name, version)

        try:
            return self._models[key]
        except KeyError as exc:
            raise KeyError(
                f"Model '{model_name}' version '{version}' "
                "does not exist."
            ) from exc

    def list_artifact_versions(
        self,
        name: str,
    ) -> list[VersionedArtifact]:
        """
        Return all registered versions of an artifact.
        """
        name = name.strip()

        if not name:
            raise ValueError("name must not be empty.")

        return [
            artifact
            for (artifact_name, _),
            artifact in self._artifacts.items()
            if artifact_name == name
        ]

    def list_model_versions(
        self,
        model_name: str,
    ) -> list[ModelVersion]:
        """
        Return all registered versions of a model.
        """
        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        return [
            model
            for (registered_name, _),
            model in self._models.items()
            if registered_name == model_name
        ]