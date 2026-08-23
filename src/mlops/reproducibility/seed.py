from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """
    Immutable record of the random seed configuration.
    """

    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


class SeedManager:
    """
    Centralized manager for deterministic random-state initialization.
    """

    def __init__(self, seed: int) -> None:
        self._state = SeedState(seed)

    @property
    def seed(self) -> int:
        return self._state.seed

    @property
    def state(self) -> SeedState:
        return self._state

    def apply(self) -> SeedState:
        """
        Apply the configured seed to supported random generators.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        return self._state