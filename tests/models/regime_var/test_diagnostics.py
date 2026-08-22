import numpy as np
import pytest

from src.models.regime_var.config import RegimeVARConfig
from src.models.regime_var.diagnostics import (
    RegimeVARDiagnostics,
)
from src.models.regime_var.model import RegimeSwitchingVAR


def make_data():
    data = np.arange(
        1,
        13,
        dtype=np.float64,
    ).reshape(-1, 1)

    regimes = np.array(
        [
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
        ],
        dtype=np.int64,
    )

    return data, regimes


def make_fitted_model():
    data, regimes = make_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(
        data=data,
        regimes=regimes,
    )

    return model


def test_creation():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    assert isinstance(
        diagnostics.model,
        RegimeSwitchingVAR,
    )


def test_invalid_model_rejected():
    with pytest.raises(TypeError):
        RegimeVARDiagnostics("invalid")


def test_unfitted_model_evaluation_rejected():
    model = RegimeSwitchingVAR()

    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    with pytest.raises(RuntimeError):
        diagnostics.evaluate(data, regimes)


def test_evaluate_returns_expected_keys():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert set(result.keys()) == {
        "n_samples",
        "n_features",
        "mse",
        "rmse",
        "mae",
        "per_feature",
        "per_regime",
    }


def test_evaluate_sample_and_feature_counts():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert result["n_samples"] == 11
    assert result["n_features"] == 1


def test_evaluate_error_metrics_are_valid():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert np.isfinite(result["mse"])
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["mae"])

    assert result["mse"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["mae"] >= 0.0


def test_rmse_matches_mse():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert np.isclose(
        result["rmse"],
        np.sqrt(result["mse"]),
    )


def test_per_feature_metrics():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert 0 in result["per_feature"]

    metrics = result["per_feature"][0]

    assert set(metrics.keys()) == {
        "mse",
        "rmse",
        "mae",
    }


def test_per_regime_metrics():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    assert set(result["per_regime"].keys()) == {
        0,
        1,
        2,
    }


def test_per_regime_error_metrics():
    model = make_fitted_model()
    diagnostics = RegimeVARDiagnostics(model)

    data, regimes = make_data()

    result = diagnostics.evaluate(data, regimes)

    for metrics in result["per_regime"].values():
        assert metrics["sample_count"] > 0
        assert metrics["mse"] >= 0.0
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0


def test_evaluation_skips_unfitted_regimes():
    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    train_regimes = np.array(
        [0, 0, 0, 0, 0],
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=2,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, train_regimes)

    diagnostics = RegimeVARDiagnostics(model)

    evaluation_regimes = np.array(
        [0, 0, 1, 1, 1],
        dtype=np.int64,
    )

    result = diagnostics.evaluate(
        data,
        evaluation_regimes,
    )

    assert result["n_samples"] == 1
    assert set(result["per_regime"].keys()) == {0}


def test_no_evaluable_samples_rejected():
    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    train_regimes = np.array(
        [0, 0, 0],
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=2,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, train_regimes)

    diagnostics = RegimeVARDiagnostics(model)

    evaluation_regimes = np.array(
        [1, 1, 1],
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        diagnostics.evaluate(
            data,
            evaluation_regimes,
        )


def test_missing_regimes_rejected():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    data, _ = make_data()

    with pytest.raises(ValueError):
        diagnostics.evaluate(
            data=data,
            regimes=None,
        )


def test_wrong_regime_length_rejected():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    data, _ = make_data()

    with pytest.raises(ValueError):
        diagnostics.evaluate(
            data=data,
            regimes=np.array([0, 1]),
        )


def test_wrong_feature_count_rejected():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    regimes = np.array(
        [0, 0, 0],
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        diagnostics.evaluate(
            data=data,
            regimes=regimes,
        )


def test_model_summary_before_fit():
    model = RegimeSwitchingVAR()

    diagnostics = RegimeVARDiagnostics(model)

    summary = diagnostics.model_summary()

    assert summary["is_fitted"] is False
    assert summary["regime_parameters"] == {}


def test_model_summary_after_fit():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    summary = diagnostics.model_summary()

    assert summary["is_fitted"] is True
    assert set(summary["regime_parameters"].keys()) == {
        0,
        1,
        2,
    }


def test_model_summary_parameter_shapes():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    summary = diagnostics.model_summary()

    for parameters in summary[
        "regime_parameters"
    ].values():
        assert parameters["coefficient_shape"] == (1, 1)
        assert parameters["intercept_shape"] == (1,)


def test_model_summary_sample_counts_match_parameters():
    diagnostics = RegimeVARDiagnostics(
        make_fitted_model()
    )

    summary = diagnostics.model_summary()

    for regime, parameters in summary[
        "regime_parameters"
    ].items():
        assert parameters["sample_count"] == summary[
            "sample_counts"
        ][regime]