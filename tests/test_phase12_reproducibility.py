from __future__ import annotations

import random

import numpy as np
import pytest

from src.mlops import (
    EnvironmentSnapshot,
    ReproducibilityManager,
    ReproducibilitySnapshot,
    SeedManager,
    SeedState,
)


# ---------------------------------------------------------------------------
# SeedState
# ---------------------------------------------------------------------------


def test_seed_state_creation() -> None:
    state = SeedState(42)

    assert state.seed == 42


def test_seed_state_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SeedState(-1)


def test_seed_state_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="integer"):
        SeedState(1.5)  # type: ignore[arg-type]


def test_seed_manager_exposes_seed() -> None:
    manager = SeedManager(42)

    assert manager.seed == 42
    assert manager.state == SeedState(42)


def test_seed_manager_applies_deterministic_numpy_seed() -> None:
    manager = SeedManager(42)

    manager.apply()
    first = np.random.rand(5)

    manager.apply()
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_seed_manager_applies_deterministic_python_random_seed() -> None:
    manager = SeedManager(42)

    manager.apply()
    first = [random.random() for _ in range(5)]

    manager.apply()
    second = [random.random() for _ in range(5)]

    assert first == second


# ---------------------------------------------------------------------------
# EnvironmentSnapshot
# ---------------------------------------------------------------------------


def test_environment_snapshot_capture() -> None:
    snapshot = EnvironmentSnapshot.capture(
        packages=("numpy", "pytest"),
    )

    assert snapshot.python_version
    assert snapshot.platform
    assert snapshot.system
    assert snapshot.machine
    assert snapshot.packages["numpy"]
    assert snapshot.packages["pytest"]


def test_environment_snapshot_rejects_empty_package_name() -> None:
    with pytest.raises(ValueError, match="package names"):
        EnvironmentSnapshot.capture(packages=("numpy", ""))


def test_environment_snapshot_rejects_missing_package() -> None:
    with pytest.raises(ValueError, match="not installed"):
        EnvironmentSnapshot.capture(
            packages=("this_package_should_not_exist_xyz",),
        )


def test_environment_snapshot_is_immutable() -> None:
    snapshot = EnvironmentSnapshot.capture()

    with pytest.raises(AttributeError):
        snapshot.system = "Linux"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReproducibilitySnapshot
# ---------------------------------------------------------------------------


def _build_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot.capture(
        packages=("numpy", "pytest"),
    )


def test_reproducibility_snapshot_creation() -> None:
    seed = SeedState(42)
    environment = _build_environment()

    snapshot = ReproducibilitySnapshot(
        seed=seed,
        environment=environment,
        experiment_id="exp_001",
        run_id="run_001",
        dataset_version="1.0.0",
        feature_version="1.0.0",
        configuration={"model": "HMM"},
    )

    assert snapshot.seed == seed
    assert snapshot.environment == environment
    assert snapshot.experiment_id == "exp_001"
    assert snapshot.run_id == "run_001"
    assert snapshot.dataset_version == "1.0.0"
    assert snapshot.feature_version == "1.0.0"
    assert snapshot.configuration == {"model": "HMM"}


def test_reproducibility_snapshot_strips_identifiers() -> None:
    snapshot = ReproducibilitySnapshot(
        seed=SeedState(42),
        environment=_build_environment(),
        experiment_id="  exp_001  ",
        run_id="  run_001  ",
        dataset_version=" 1.0.0 ",
        feature_version=" 1.0.0 ",
    )

    assert snapshot.experiment_id == "exp_001"
    assert snapshot.run_id == "run_001"
    assert snapshot.dataset_version == "1.0.0"
    assert snapshot.feature_version == "1.0.0"


def test_reproducibility_snapshot_rejects_invalid_seed() -> None:
    with pytest.raises(TypeError, match="SeedState"):
        ReproducibilitySnapshot(
            seed=42,  # type: ignore[arg-type]
            environment=_build_environment(),
        )


def test_reproducibility_snapshot_rejects_invalid_environment() -> None:
    with pytest.raises(TypeError, match="EnvironmentSnapshot"):
        ReproducibilitySnapshot(
            seed=SeedState(42),
            environment="invalid",  # type: ignore[arg-type]
        )


def test_reproducibility_snapshot_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        ReproducibilitySnapshot(
            seed=SeedState(42),
            environment=_build_environment(),
            experiment_id=" ",
        )


def test_reproducibility_snapshot_is_immutable() -> None:
    snapshot = ReproducibilitySnapshot(
        seed=SeedState(42),
        environment=_build_environment(),
    )

    with pytest.raises(AttributeError):
        snapshot.run_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReproducibilityManager
# ---------------------------------------------------------------------------


def test_reproducibility_manager_creation() -> None:
    manager = ReproducibilityManager(
        SeedManager(42),
    )

    assert manager.seed_manager.seed == 42


def test_reproducibility_manager_rejects_invalid_seed_manager() -> None:
    with pytest.raises(TypeError, match="SeedManager"):
        ReproducibilityManager(42)  # type: ignore[arg-type]


def test_reproducibility_manager_capture() -> None:
    manager = ReproducibilityManager(
        SeedManager(42),
    )

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        dataset_version="1.0.0",
        feature_version="1.0.0",
        configuration={"model": "HMM"},
        packages=("numpy", "pytest"),
    )

    assert isinstance(snapshot, ReproducibilitySnapshot)
    assert snapshot.seed.seed == 42
    assert snapshot.experiment_id == "exp_001"
    assert snapshot.run_id == "run_001"
    assert snapshot.dataset_version == "1.0.0"
    assert snapshot.feature_version == "1.0.0"
    assert snapshot.configuration == {"model": "HMM"}
    assert snapshot.environment.packages["numpy"]
    assert snapshot.environment.packages["pytest"]


def test_reproducibility_manager_apply_seed() -> None:
    manager = ReproducibilityManager(
        SeedManager(42),
    )

    manager.apply_seed()
    first = np.random.rand(5)

    manager.apply_seed()
    second = np.random.rand(5)

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_reproducibility_public_exports() -> None:
    from src.mlops.reproducibility import (
        EnvironmentSnapshot as ExportedEnvironmentSnapshot,
        ReproducibilityManager as ExportedManager,
        ReproducibilitySnapshot as ExportedSnapshot,
        SeedManager as ExportedSeedManager,
        SeedState as ExportedSeedState,
    )

    assert ExportedEnvironmentSnapshot is EnvironmentSnapshot
    assert ExportedManager is ReproducibilityManager
    assert ExportedSnapshot is ReproducibilitySnapshot
    assert ExportedSeedManager is SeedManager
    assert ExportedSeedState is SeedState