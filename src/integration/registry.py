from __future__ import annotations

from typing import Any


class ModelRegistry:
    """
    Registry for model instances used by the integration layer.

    Models are stored by unique string names. The registry deliberately
    does not impose a common model interface; execution compatibility is
    handled by the integration service.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    @staticmethod
    def _validate_name(name: Any) -> str:
        """
        Validate and normalize a model name.
        """
        if not isinstance(name, str):
            raise TypeError(
                "model name must be a string."
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "model name must not be empty."
            )

        return name

    def register(
        self,
        name: str,
        model: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model under a unique name.

        Parameters
        ----------
        name:
            Unique model identifier.
        model:
            Model instance to store. None is not allowed.
        overwrite:
            Replace an existing model with the same name when True.
        """
        name = self._validate_name(name)

        if model is None:
            raise ValueError(
                "model must not be None."
            )

        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean."
            )

        if name in self._models and not overwrite:
            raise ValueError(
                f"model '{name}' is already registered."
            )

        self._models[name] = model

    def get(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Return a registered model or the supplied default.
        """
        name = self._validate_name(name)

        return self._models.get(
            name,
            default,
        )

    def require(
        self,
        name: str,
    ) -> Any:
        """
        Return a registered model.

        Raises
        ------
        KeyError
            If no model exists under the supplied name.
        """
        name = self._validate_name(name)

        if name not in self._models:
            raise KeyError(
                f"model '{name}' is not registered."
            )

        return self._models[name]

    def unregister(
        self,
        name: str,
    ) -> Any:
        """
        Remove and return a registered model.

        Raises
        ------
        KeyError
            If no model exists under the supplied name.
        """
        name = self._validate_name(name)

        if name not in self._models:
            raise KeyError(
                f"model '{name}' is not registered."
            )

        return self._models.pop(name)

    def clear(self) -> None:
        """
        Remove all registered models.
        """
        self._models.clear()

    def contains(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a model is registered under the supplied name.
        """
        name = self._validate_name(name)

        return name in self._models

    @property
    def names(self) -> tuple[str, ...]:
        """
        Return registered model names in registration order.
        """
        return tuple(self._models.keys())

    @property
    def count(self) -> int:
        """
        Return the number of registered models.
        """
        return len(self._models)

    def __len__(self) -> int:
        return self.count

    def __contains__(
        self,
        name: object,
    ) -> bool:
        """
        Support: ``name in registry``.

        Invalid names simply return False, matching normal container
        membership semantics.
        """
        if not isinstance(name, str):
            return False

        normalized_name = name.strip()

        if not normalized_name:
            return False

        return normalized_name in self._models