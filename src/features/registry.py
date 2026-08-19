"""Registry for managing feature transformers."""

from __future__ import annotations

from typing import Iterable

from src.features.base import BaseFeatureTransformer


class FeatureRegistry:
    """Registers and resolves feature transformers."""

    def __init__(self) -> None:
        self._transformers: dict[str, BaseFeatureTransformer] = {}

    def register(
        self,
        transformer: BaseFeatureTransformer,
    ) -> None:
        """Register a feature transformer by its unique name."""

        name = transformer.name

        if not name:
            raise ValueError(
                "Feature transformer must define a non-empty name."
            )

        if name in self._transformers:
            raise ValueError(
                f"Feature transformer '{name}' is already registered."
            )

        self._transformers[name] = transformer

    def get(
        self,
        name: str,
    ) -> BaseFeatureTransformer:
        """Return a registered transformer."""

        try:
            return self._transformers[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown feature transformer: '{name}'."
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered transformer names."""

        return tuple(self._transformers.keys())

    def values(self) -> Iterable[BaseFeatureTransformer]:
        """Return registered transformers in registration order."""

        return self._transformers.values()