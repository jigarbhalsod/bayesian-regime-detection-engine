from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class DeploymentConfig:
    """
    Immutable configuration describing a model deployment.
    """

    deployment_name: str
    environment: str
    model_name: str
    model_version: str
    replicas: int = 1
    endpoint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        deployment_name = self.deployment_name.strip()
        environment = self.environment.strip()
        model_name = self.model_name.strip()
        model_version = self.model_version.strip()

        if not deployment_name:
            raise ValueError("deployment_name cannot be empty.")

        if not environment:
            raise ValueError("environment cannot be empty.")

        if not model_name:
            raise ValueError("model_name cannot be empty.")

        if not model_version:
            raise ValueError("model_version cannot be empty.")

        if (
            not isinstance(self.replicas, int)
            or isinstance(self.replicas, bool)
        ):
            raise TypeError("replicas must be an integer.")

        if self.replicas < 1:
            raise ValueError("replicas must be at least 1.")

        endpoint = self.endpoint

        if endpoint is not None:
            if not isinstance(endpoint, str):
                raise TypeError("endpoint must be a string or None.")

            endpoint = endpoint.strip()

            if not endpoint:
                raise ValueError(
                    "endpoint cannot be empty when provided."
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(
            self,
            "deployment_name",
            deployment_name,
        )
        object.__setattr__(
            self,
            "environment",
            environment,
        )
        object.__setattr__(
            self,
            "model_name",
            model_name,
        )
        object.__setattr__(
            self,
            "model_version",
            model_version,
        )
        object.__setattr__(
            self,
            "endpoint",
            endpoint,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )