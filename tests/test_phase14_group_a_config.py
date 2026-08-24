import math

import pytest

from src.explainability.config import ExplainabilityConfig


class TestExplainabilityConfig:
    def test_default_configuration(self):
        config = ExplainabilityConfig()

        assert config.top_k == 10
        assert config.importance_threshold == 0.0
        assert config.attribution_threshold == 0.0
        assert config.normalize is True
        assert config.method == "default"

    def test_valid_custom_configuration(self):
        config = ExplainabilityConfig(
            top_k=5,
            importance_threshold=0.2,
            attribution_threshold=0.1,
            normalize=False,
            method="permutation",
        )

        assert config.top_k == 5
        assert config.importance_threshold == 0.2
        assert config.attribution_threshold == 0.1
        assert config.normalize is False
        assert config.method == "permutation"

    @pytest.mark.parametrize(
        "top_k",
        [0, -1, -10],
    )
    def test_invalid_top_k_value_rejected(self, top_k):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            ExplainabilityConfig(top_k=top_k)

    @pytest.mark.parametrize(
        "top_k",
        [1.5, True, False, "10", None],
    )
    def test_invalid_top_k_type_rejected(self, top_k):
        with pytest.raises(TypeError, match="top_k must be a positive integer"):
            ExplainabilityConfig(top_k=top_k)

    @pytest.mark.parametrize(
        "field_name",
        [
            "importance_threshold",
            "attribution_threshold",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [-0.01, -1.0, -100.0],
    )
    def test_negative_threshold_rejected(self, field_name, value):
        kwargs = {field_name: value}

        with pytest.raises(
            ValueError,
            match=f"{field_name} must be greater than or equal to 0",
        ):
            ExplainabilityConfig(**kwargs)

    @pytest.mark.parametrize(
        "field_name",
        [
            "importance_threshold",
            "attribution_threshold",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [math.inf, -math.inf, math.nan],
    )
    def test_non_finite_threshold_rejected(self, field_name, value):
        kwargs = {field_name: value}

        with pytest.raises(
            ValueError,
            match=f"{field_name} must be finite",
        ):
            ExplainabilityConfig(**kwargs)

    @pytest.mark.parametrize(
        "field_name",
        [
            "importance_threshold",
            "attribution_threshold",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [True, False, "0.5", None, []],
    )
    def test_invalid_threshold_type_rejected(self, field_name, value):
        kwargs = {field_name: value}

        with pytest.raises(
            TypeError,
            match=f"{field_name} must be a numeric value",
        ):
            ExplainabilityConfig(**kwargs)

    def test_zero_thresholds_are_allowed(self):
        config = ExplainabilityConfig(
            importance_threshold=0.0,
            attribution_threshold=0.0,
        )

        assert config.importance_threshold == 0.0
        assert config.attribution_threshold == 0.0

    def test_positive_thresholds_are_allowed(self):
        config = ExplainabilityConfig(
            importance_threshold=0.75,
            attribution_threshold=0.25,
        )

        assert config.importance_threshold == 0.75
        assert config.attribution_threshold == 0.25

    @pytest.mark.parametrize(
        "normalize",
        [1, 0, "true", None, []],
    )
    def test_invalid_normalize_rejected(self, normalize):
        with pytest.raises(
            TypeError,
            match="normalize must be a boolean",
        ):
            ExplainabilityConfig(normalize=normalize)

    def test_normalize_false_is_allowed(self):
        config = ExplainabilityConfig(normalize=False)

        assert config.normalize is False

    def test_method_is_stripped(self):
        config = ExplainabilityConfig(
            method="  permutation  ",
        )

        assert config.method == "permutation"

    @pytest.mark.parametrize(
        "method",
        ["", "   "],
    )
    def test_empty_method_rejected(self, method):
        with pytest.raises(
            ValueError,
            match="method must not be empty",
        ):
            ExplainabilityConfig(method=method)

    @pytest.mark.parametrize(
        "method",
        [123, True, None, []],
    )
    def test_invalid_method_type_rejected(self, method):
        with pytest.raises(
            TypeError,
            match="method must be a string",
        ):
            ExplainabilityConfig(method=method)

    def test_numeric_thresholds_are_converted_to_float(self):
        config = ExplainabilityConfig(
            importance_threshold=1,
            attribution_threshold=2,
        )

        assert config.importance_threshold == 1.0
        assert config.attribution_threshold == 2.0