from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .version import Version


@dataclass(frozen=True)
class VersionedArtifact:
    """
    Immutable representation of a versioned experiment artifact.
    """

    name: str
    version: Version
    artifact_type: str
    source_run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        artifact_type = self.artifact_type.strip()

        if not name:
            raise ValueError("name must not be empty.")

        if not artifact_type:
            raise ValueError("artifact_type must not be empty.")

        if not isinstance(self.version, Version):
            raise TypeError("version must be a Version.")

        if self.source_run_id is not None:
            source_run_id = self.source_run_id.strip()

            if not source_run_id:
                raise ValueError(
                    "source_run_id must not be empty when provided."
                )

            object.__setattr__(
                self,
                "source_run_id",
                source_run_id,
            )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )