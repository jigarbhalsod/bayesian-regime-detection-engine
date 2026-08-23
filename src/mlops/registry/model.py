from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class RegisteredModel:
    """
    Immutable metadata describing a registered model.
    """

    name: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()

        if not name:
            raise ValueError("name must not be empty.")

        tags = tuple(tag.strip() for tag in self.tags)

        if any(not tag for tag in tags):
            raise ValueError("tags must not contain empty values.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            self.description.strip()
            if self.description is not None
            else None,
        )
        object.__setattr__(self, "tags", tags)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )