import pytest
import torch

from src.models.foundation.config import (
    FoundationModelConfig,
)
from src.models.foundation.integration import (
    FoundationIntegrationResult,
    FoundationModelIntegration,
)
from src.models.foundation.sequence import (
    SequenceDataset,
)


@pytest.fixture
def config():
    return FoundationModelConfig(
        model_name="integration_model",
        context_length=4,
        forecast_horizon=2,
        n_features=3,
        output_dim=1,
    )


@pytest.fixture
def integration(config):
    torch.manual_seed(42)

    return FoundationModelIntegration(
        config=config,
        hidden_dim=8,
        stride=1,
    )


@pytest.fixture
def data():
    torch.manual_seed(123)

    return torch.randn(12, 3)


@pytest.fixture
def targets():
    torch.manual_seed(456)

    return torch.randn(12, 1)


def test_default_creation():
    integration = FoundationModelIntegration()

    assert isinstance(
        integration.config,
        FoundationModelConfig,
    )

    assert integration.is_fitted is False


def test_custom_creation(integration, config):
    assert integration.config is config
    assert integration.preparer.context_length == 4
    assert integration.preparer.forecast_horizon == 2
    assert integration.preparer.n_features == 3
    assert integration.preparer.output_dim == 1
    assert integration.model.hidden_dim == 8


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        FoundationModelIntegration(
            config={},
        )


def test_prepare_returns_sequence_dataset(
    integration,
    data,
):
    result = integration.prepare(data)

    assert isinstance(
        result,
        SequenceDataset,
    )


def test_prepare_sequence_shapes(
    integration,
    data,
):
    result = integration.prepare(data)

    # 12 - 4 - 2 + 1 = 7
    assert result.inputs.shape == (7, 4, 3)
    assert result.targets.shape == (7, 2, 1)


def test_prepare_with_custom_targets(
    integration,
    data,
    targets,
):
    result = integration.prepare(
        data,
        targets=targets,
    )

    assert result.targets.shape == (7, 2, 1)


def test_fit_returns_self(
    integration,
    data,
):
    result = integration.fit(
        data,
        epochs=2,
    )

    assert result is integration
    assert integration.is_fitted is True


def test_fit_with_custom_targets(
    integration,
    data,
    targets,
):
    result = integration.fit(
        data,
        targets=targets,
        epochs=2,
    )

    assert result is integration
    assert integration.is_fitted is True


def test_predict_before_fit_rejected(
    integration,
):
    x = torch.randn(2, 4, 3)

    with pytest.raises(RuntimeError):
        integration.predict(x)


def test_predict_after_fit_shape(
    integration,
    data,
):
    integration.fit(
        data,
        epochs=2,
    )

    x = torch.randn(3, 4, 3)

    predictions = integration.predict(x)

    assert predictions.shape == (3, 2, 1)


def test_fit_predict_returns_result(
    integration,
    data,
):
    result = integration.fit_predict(
        data,
        epochs=2,
    )

    assert isinstance(
        result,
        FoundationIntegrationResult,
    )


def test_fit_predict_dataset(
    integration,
    data,
):
    result = integration.fit_predict(
        data,
        epochs=2,
    )

    assert isinstance(
        result.dataset,
        SequenceDataset,
    )

    assert result.dataset.inputs.shape == (7, 4, 3)
    assert result.dataset.targets.shape == (7, 2, 1)


def test_fit_predict_prediction_shape(
    integration,
    data,
):
    result = integration.fit_predict(
        data,
        epochs=2,
    )

    assert result.predictions.shape == (7, 2, 1)


def test_fit_predict_with_custom_targets(
    integration,
    data,
    targets,
):
    result = integration.fit_predict(
        data,
        targets=targets,
        epochs=2,
    )

    assert result.predictions.shape == (7, 2, 1)


def test_metadata_before_fit(
    integration,
):
    metadata = integration.get_metadata()

    assert metadata["is_fitted"] is False
    assert metadata["config"]["model_name"] == (
        "integration_model"
    )
    assert metadata["config"]["context_length"] == 4
    assert metadata["sequence"]["stride"] == 1
    assert metadata["model"]["hidden_dim"] == 8
    assert metadata["model"]["is_fitted"] is False


def test_metadata_after_fit(
    integration,
    data,
):
    integration.fit(
        data,
        epochs=1,
    )

    metadata = integration.get_metadata()

    assert metadata["is_fitted"] is True
    assert metadata["model"]["is_fitted"] is True


def test_fit_rejects_insufficient_data(
    integration,
):
    data = torch.randn(5, 3)

    with pytest.raises(ValueError):
        integration.fit(
            data,
            epochs=1,
        )


def test_fit_rejects_wrong_feature_count(
    integration,
):
    data = torch.randn(12, 2)

    with pytest.raises(ValueError):
        integration.fit(
            data,
            epochs=1,
        )


def test_fit_rejects_invalid_custom_targets(
    integration,
    data,
):
    targets = torch.randn(12, 2)

    with pytest.raises(ValueError):
        integration.fit(
            data,
            targets=targets,
            epochs=1,
        )


def test_predict_rejects_invalid_input_shape(
    integration,
    data,
):
    integration.fit(
        data,
        epochs=1,
    )

    invalid_x = torch.randn(3, 4)

    with pytest.raises(ValueError):
        integration.predict(invalid_x)


def test_predict_rejects_wrong_context_length(
    integration,
    data,
):
    integration.fit(
        data,
        epochs=1,
    )

    invalid_x = torch.randn(3, 5, 3)

    with pytest.raises(ValueError):
        integration.predict(invalid_x)


def test_predict_rejects_wrong_feature_count(
    integration,
    data,
):
    integration.fit(
        data,
        epochs=1,
    )

    invalid_x = torch.randn(3, 4, 2)

    with pytest.raises(ValueError):
        integration.predict(invalid_x)