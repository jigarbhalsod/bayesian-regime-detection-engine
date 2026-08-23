from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .config import DeploymentConfig
from .status import DeploymentStatus


@dataclass(frozen=True)
class DeploymentState:
    """
    Immutable snapshot of a deployment's current state.
    """

    config: DeploymentConfig
    status: DeploymentStatus = DeploymentStatus.PENDING
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, DeploymentConfig):
            raise TypeError("config must be a DeploymentConfig.")

        if not isinstance(self.status, DeploymentStatus):
            raise TypeError("status must be a DeploymentStatus.")

        message = self.message

        if message is not None:
            if not isinstance(message, str):
                raise TypeError("message must be a string or None.")

            message = message.strip()

            if not message:
                raise ValueError(
                    "message cannot be empty when provided."
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )