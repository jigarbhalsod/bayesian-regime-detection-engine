import pytest

from src.ensemble import (
    EnsembleConfig,
    EnsembleInputPreparer,
    EnsembleModelOutput,
    PreparedEnsembleInput,
)


def test_model_output_validation():
    output = EnsembleModelOutput(
        model_name="hmm",
        prediction="risk_on",
        probabilities={
            "risk_on": 0.8,
            "risk_off": 0.2,
        },
    )

    output.validate()


def test_model_output_rejects_empty_name():
    output = EnsembleModelOutput(
        model_name="",
    )

    with pytest.raises(ValueError, match="model_name"):
        output.validate()


def test_model_output_rejects_negative_probability():
    output = EnsembleModelOutput(
        model_name="hmm",
        probabilities={
            "risk_on": -0.2,
        },
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        output.validate()


def test_prepared_input_model_count():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(model_name="hmm"),
            EnsembleModelOutput(model_name="bayesian"),
        ]
    )

    assert prepared.model_count == 2


def test_prepared_input_participating_models():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(model_name="hmm"),
            EnsembleModelOutput(model_name="bayesian"),
        ]
    )

    assert prepared.participating_models == [
        "hmm",
        "bayesian",
    ]


def test_prepared_input_rejects_insufficient_models():
    prepared = PreparedEnsembleInput()

    with pytest.raises(ValueError, match="At least 1"):
        prepared.validate(min_models=1)


def test_prepared_input_rejects_duplicate_models():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(model_name="hmm"),
            EnsembleModelOutput(model_name="hmm"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepared.validate()


def test_preparer_normalizes_probabilities():
    preparer = EnsembleInputPreparer()

    prepared = preparer.prepare(
        [
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 8.0,
                    "risk_off": 2.0,
                },
            )
        ]
    )

    probabilities = prepared.outputs[0].probabilities

    assert probabilities["risk_on"] == pytest.approx(0.8)
    assert probabilities["risk_off"] == pytest.approx(0.2)


def test_preparer_collects_all_regime_labels():
    preparer = EnsembleInputPreparer()

    prepared = preparer.prepare(
        [
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 0.8,
                    "risk_off": 0.2,
                },
            ),
            EnsembleModelOutput(
                model_name="bayesian",
                prediction="transitional",
                probabilities={
                    "transitional": 0.7,
                    "risk_off": 0.3,
                },
            ),
        ]
    )

    assert prepared.regime_labels == [
        "risk_on",
        "risk_off",
        "transitional",
    ]


def test_preparer_skips_unsuccessful_model_when_allowed():
    config = EnsembleConfig(
        allow_missing_models=True,
    )

    preparer = EnsembleInputPreparer(config)

    prepared = preparer.prepare(
        [
            EnsembleModelOutput(
                model_name="hmm",
                success=True,
                probabilities={
                    "risk_on": 1.0,
                },
            ),
            EnsembleModelOutput(
                model_name="bayesian",
                success=False,
            ),
        ]
    )

    assert prepared.model_count == 1
    assert prepared.skipped_models == ["bayesian"]


def test_preparer_rejects_unsuccessful_model_when_not_allowed():
    config = EnsembleConfig(
        allow_missing_models=False,
    )

    preparer = EnsembleInputPreparer(config)

    with pytest.raises(ValueError, match="unsuccessful"):
        preparer.prepare(
            [
                EnsembleModelOutput(
                    model_name="hmm",
                    success=False,
                )
            ]
        )


def test_preparer_respects_enabled_models():
    config = EnsembleConfig(
        enabled_models=["hmm"],
    )

    preparer = EnsembleInputPreparer(config)

    prepared = preparer.prepare(
        [
            EnsembleModelOutput(
                model_name="hmm",
                probabilities={
                    "risk_on": 1.0,
                },
            ),
            EnsembleModelOutput(
                model_name="bayesian",
                probabilities={
                    "risk_off": 1.0,
                },
            ),
        ]
    )

    assert prepared.model_count == 1
    assert prepared.participating_models == ["hmm"]


def test_preparer_requires_minimum_models():
    config = EnsembleConfig(
        min_models=2,
    )

    preparer = EnsembleInputPreparer(config)

    with pytest.raises(ValueError, match="At least 2"):
        preparer.prepare(
            [
                EnsembleModelOutput(
                    model_name="hmm",
                    probabilities={
                        "risk_on": 1.0,
                    },
                )
            ]
        )


def test_preparer_rejects_invalid_output_type():
    preparer = EnsembleInputPreparer()

    with pytest.raises(TypeError, match="EnsembleModelOutput"):
        preparer.prepare(["invalid"])


def test_preparer_rejects_none_outputs():
    preparer = EnsembleInputPreparer()

    with pytest.raises(ValueError, match="cannot be None"):
        preparer.prepare(None)


def test_preparer_rejects_zero_probability_total():
    preparer = EnsembleInputPreparer()

    with pytest.raises(ValueError, match="positive total"):
        preparer.prepare(
            [
                EnsembleModelOutput(
                    model_name="hmm",
                    probabilities={
                        "risk_on": 0.0,
                        "risk_off": 0.0,
                    },
                )
            ]
        )