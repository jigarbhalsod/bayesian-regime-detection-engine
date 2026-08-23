from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    Immutable snapshot of the execution environment.
    """

    python_version: str
    platform: str
    system: str
    machine: str
    packages: Mapping[str, str]

    @classmethod
    def capture(
        cls,
        packages: tuple[str, ...] = (),
    ) -> "EnvironmentSnapshot":
        package_versions: dict[str, str] = {}

        for package in packages:
            package = package.strip()

            if not package:
                raise ValueError(
                    "package names must not be empty."
                )

            try:
                package_versions[package] = (
                    importlib.metadata.version(package)
                )
            except importlib.metadata.PackageNotFoundError as exc:
                raise ValueError(
                    f"Package '{package}' is not installed."
                ) from exc

        return cls(
            python_version=sys.version,
            platform=platform.platform(),
            system=platform.system(),
            machine=platform.machine(),
            packages=package_versions,
        )