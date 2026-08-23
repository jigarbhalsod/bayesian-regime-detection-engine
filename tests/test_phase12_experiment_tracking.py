from __future__ import annotations

import pytest

from src.mlops import (
    BaseExperimentTracker,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    ExperimentStatus,
    InMemoryExperimentTracker,
)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


def test_experiment_config_defaults() -> None:
    config = ExperimentConfig(experiment_name="test")

    assert config.experiment_name == "test"
    assert config.description is None
    assert config.tags == ()
    assert dict(config.metadata) == {}


def test_experiment_config_strips_name_and_tags() -> None:
    config = ExperimentConfig(
        experiment_name="  regime_model  ",
        tags=("  bayesian  ", "finance"),
    )

    assert config.experiment_name == "regime_model"
    assert config.tags == ("bayesian", "finance")


def test_experiment_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        ExperimentConfig(experiment_name="   ")


def test_experiment_config_rejects_invalid_tags() -> None:
    with pytest.raises(ValueError, match="tags"):
        ExperimentConfig(
            experiment_name="test",
            tags=("valid", ""),
        )


def test_experiment_config_is_immutable() -> None:
    config = ExperimentConfig(experiment_name="test")

    with pytest.raises(AttributeError):
        config.experiment_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


def test_experiment_result_success() -> None:
    result = ExperimentResult(
        experiment_id="exp_001",
        run_id="run_001",
        success=True,
        metrics={"accuracy": 0.95},
    )

    assert result.experiment_id == "exp_001"
    assert result.run_id == "run_001"
    assert result.success is True
    assert result.metrics["accuracy"] == 0.95
    assert result.error is None


def test_experiment_result_failure() -> None:
    result = ExperimentResult(
        experiment_id="exp_001",
        run_id="run_001",
        success=False,
        error="training failed",
    )

    assert result.success is False
    assert result.error == "training failed"


def test_experiment_result_rejects_success_with_error() -> None:
    with pytest.raises(ValueError, match="error must be None"):
        ExperimentResult(
            experiment_id="exp_001",
            run_id="run_001",
            success=True,
            error="unexpected error",
        )


def test_experiment_result_rejects_empty_ids() -> None:
    with pytest.raises(ValueError):
        ExperimentResult(
            experiment_id=" ",
            run_id="run_001",
            success=True,
        )

    with pytest.raises(ValueError):
        ExperimentResult(
            experiment_id="exp_001",
            run_id=" ",
            success=True,
        )


# ---------------------------------------------------------------------------
# ExperimentRun
# ---------------------------------------------------------------------------


def test_experiment_run_initial_state() -> None:
    run = ExperimentRun(experiment_id="exp_001")

    assert run.status is ExperimentStatus.CREATED
    assert run.started_at is None
    assert run.completed_at is None
    assert run.metrics == {}
    assert run.artifacts == {}
    assert run.error is None
    assert run.run_id


def test_experiment_run_lifecycle() -> None:
    run = ExperimentRun(experiment_id="exp_001")

    run.start()

    assert run.status is ExperimentStatus.RUNNING
    assert run.started_at is not None

    run.complete()

    assert run.status is ExperimentStatus.COMPLETED
    assert run.completed_at is not None


def test_experiment_run_cannot_start_twice() -> None:
    run = ExperimentRun(experiment_id="exp_001")

    run.start()

    with pytest.raises(RuntimeError, match="created"):
        run.start()


def test_experiment_run_can_log_metric() -> None:
    run = ExperimentRun(experiment_id="exp_001")
    run.start()

    run.log_metric("accuracy", 0.95)

    assert run.metrics["accuracy"] == 0.95


def test_experiment_run_can_log_artifact() -> None:
    run = ExperimentRun(experiment_id="exp_001")
    run.start()

    run.log_artifact("model", "model.pkl")

    assert run.artifacts["model"] == "model.pkl"


def test_experiment_run_cannot_log_metric_when_not_running() -> None:
    run = ExperimentRun(experiment_id="exp_001")

    with pytest.raises(RuntimeError, match="running"):
        run.log_metric("accuracy", 0.95)


def test_experiment_run_cannot_log_artifact_when_not_running() -> None:
    run = ExperimentRun(experiment_id="exp_001")

    with pytest.raises(RuntimeError, match="running"):
        run.log_artifact("model", "model.pkl")


def test_experiment_run_failure_lifecycle() -> None:
    run = ExperimentRun(experiment_id="exp_001")
    run.start()

    run.fail("training failed")

    assert run.status is ExperimentStatus.FAILED
    assert run.error == "training failed"
    assert run.completed_at is not None


def test_experiment_run_rejects_empty_failure_error() -> None:
    run = ExperimentRun(experiment_id="exp_001")
    run.start()

    with pytest.raises(ValueError, match="error"):
        run.fail("   ")


# ---------------------------------------------------------------------------
# BaseExperimentTracker
# ---------------------------------------------------------------------------


def test_base_tracker_is_abstract() -> None:
    assert BaseExperimentTracker.__abstractmethods__ == frozenset(
        {
            "create_experiment",
            "start_run",
            "get_run",
            "complete_run",
            "fail_run",
        }
    )


# ---------------------------------------------------------------------------
# InMemoryExperimentTracker
# ---------------------------------------------------------------------------


def test_tracker_creates_experiment() -> None:
    tracker = InMemoryExperimentTracker()

    experiment_id = tracker.create_experiment(
        ExperimentConfig(experiment_name="regime_test")
    )

    assert experiment_id
    assert isinstance(experiment_id, str)


def test_tracker_starts_run() -> None:
    tracker = InMemoryExperimentTracker()

    experiment_id = tracker.create_experiment(
        ExperimentConfig(experiment_name="regime_test")
    )

    run = tracker.start_run(
        experiment_id,
        metadata={"model": "HMM"},
    )

    assert isinstance(run, ExperimentRun)
    assert run.experiment_id == experiment_id
    assert run.status is ExperimentStatus.RUNNING
    assert run.metadata["model"] == "HMM"


def test_tracker_retrieves_existing_run() -> None:
    tracker = InMemoryExperimentTracker()

    experiment_id = tracker.create_experiment(
        ExperimentConfig(experiment_name="regime_test")
    )

    run = tracker.start_run(experiment_id)

    retrieved = tracker.get_run(run.run_id)

    assert retrieved is run


def test_tracker_rejects_unknown_experiment() -> None:
    tracker = InMemoryExperimentTracker()

    with pytest.raises(KeyError, match="does not exist"):
        tracker.start_run("unknown")


def test_tracker_completes_run() -> None:
    tracker = InMemoryExperimentTracker()

    experiment_id = tracker.create_experiment(
        ExperimentConfig(experiment_name="regime_test")
    )

    run = tracker.start_run(experiment_id)
    run.log_metric("accuracy", 0.95)
    run.log_artifact("model", "model.pkl")

    result = tracker.complete_run(run.run_id)

    assert isinstance(result, ExperimentResult)
    assert result.success is True
    assert result.metrics["accuracy"] == 0.95
    assert result.artifacts["model"] == "model.pkl"
    assert result.error is None


def test_tracker_fails_run() -> None:
    tracker = InMemoryExperimentTracker()

    experiment_id = tracker.create_experiment(
        ExperimentConfig(experiment_name="regime_test")
    )

    run = tracker.start_run(experiment_id)

    result = tracker.fail_run(
        run.run_id,
        "training failed",
    )

    assert isinstance(result, ExperimentResult)
    assert result.success is False
    assert result.error == "training failed"


def test_tracker_rejects_unknown_run() -> None:
    tracker = InMemoryExperimentTracker()

    with pytest.raises(KeyError, match="does not exist"):
        tracker.get_run("unknown")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_public_exports() -> None:
    from src.mlops import (
        BaseExperimentTracker as ExportedBaseTracker,
        ExperimentConfig as ExportedConfig,
        ExperimentResult as ExportedResult,
        ExperimentRun as ExportedRun,
        ExperimentStatus as ExportedStatus,
        InMemoryExperimentTracker as ExportedTracker,
    )

    assert ExportedBaseTracker is BaseExperimentTracker
    assert ExportedConfig is ExperimentConfig
    assert ExportedResult is ExperimentResult
    assert ExportedRun is ExperimentRun
    assert ExportedStatus is ExperimentStatus
    assert ExportedTracker is InMemoryExperimentTracker