from __future__ import annotations

from typing import Any, Mapping

from .environment import EnvironmentSnapshot
from .seed import SeedManager
from .snapshot import ReproducibilitySnapshot


class ReproducibilityManager:
    """
    Coordinates creation of reproducibility snapshots.
    """

    def __init__(
        self,
        seed_manager: SeedManager,
    ) -> None:
        if not isinstance(seed_manager, SeedManager):
            raise TypeError(
                "seed_manager must be a SeedManager."
            )

        self._seed_manager = seed_manager

    @property
    def seed_manager(self) -> SeedManager:
        return self._seed_manager

    def capture(
        self,
        *,
        experiment_id: str | None = None,
        run_id: str | None = None,
        dataset_version: str | None = None,
        feature_version: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        packages: tuple[str, ...] = (),
    ) -> ReproducibilitySnapshot:
        """
        Capture a complete reproducibility snapshot.
        """

        environment = EnvironmentSnapshot.capture(
            packages=packages,
        )

        return ReproducibilitySnapshot(
            seed=self._seed_manager.state,
            environment=environment,
            experiment_id=experiment_id,
            run_id=run_id,
            dataset_version=dataset_version,
            feature_version=feature_version,
            configuration=configuration,
        )

    def apply_seed(self) -> None:
        """
        Apply the configured seed to supported random generators.
        """
        self._seed_manager.apply()