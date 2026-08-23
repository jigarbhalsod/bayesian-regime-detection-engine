from __future__ import annotations

from collections import defaultdict

from src.mlops.versioning import ModelVersion

from .model import RegisteredModel


class ModelRegistry:
    """
    In-memory registry for registered models and their versions.
    """

    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}
        self._versions: dict[str, list[ModelVersion]] = defaultdict(list)

    def register_model(
        self,
        model: RegisteredModel,
    ) -> RegisteredModel:
        """
        Register a new model.
        """
        if not isinstance(model, RegisteredModel):
            raise TypeError("model must be a RegisteredModel.")

        if model.name in self._models:
            raise ValueError(
                f"Model '{model.name}' is already registered."
            )

        self._models[model.name] = model

        return model

    def register_version(
        self,
        model_version: ModelVersion,
    ) -> ModelVersion:
        """
        Register a version for an existing model.
        """
        if not isinstance(model_version, ModelVersion):
            raise TypeError(
                "model_version must be a ModelVersion."
            )

        model_name = model_version.model_name

        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' is not registered."
            )

        versions = self._versions[model_name]

        if any(
            existing.version == model_version.version
            for existing in versions
        ):
            raise ValueError(
                f"Version '{model_version.version}' for model "
                f"'{model_name}' is already registered."
            )

        versions.append(model_version)
        versions.sort(key=lambda item: item.version)

        return model_version

    def get_model(
        self,
        name: str,
    ) -> RegisteredModel:
        """
        Retrieve a registered model by name.
        """
        if not isinstance(name, str):
            raise TypeError("model name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("model name must not be empty.")

        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(
                f"Model '{name}' is not registered."
            ) from exc

    def get_version(
        self,
        model_name: str,
        version: str,
    ) -> ModelVersion:
        """
        Retrieve a specific registered model version.
        """
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        if not isinstance(version, str):
            raise TypeError("version must be a string.")

        model_name = model_name.strip()
        version = version.strip()

        if not model_name:
            raise ValueError("model_name must not be empty.")

        if not version:
            raise ValueError("version must not be empty.")

        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' is not registered."
            )

        for model_version in self._versions[model_name]:
            if str(model_version.version) == version:
                return model_version

        raise KeyError(
            f"Version '{version}' for model "
            f"'{model_name}' is not registered."
        )

    def list_versions(
        self,
        model_name: str,
    ) -> list[ModelVersion]:
        """
        List all registered versions for a model in version order.
        """
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        model_name = model_name.strip()

        if not model_name:
            raise ValueError("model_name must not be empty.")

        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' is not registered."
            )

        return list(self._versions[model_name])