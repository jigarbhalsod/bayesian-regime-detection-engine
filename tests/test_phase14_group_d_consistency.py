import pytest

from src.explainability.consistency import (
    ExplanationConsistencyValidator,
)


class TestExplanationConsistencyValidator:
    def test_valid_summary_is_consistent(self):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": ["UP"],
            "confidences": [0.85],
            "feature_drivers": ["momentum", "volatility"],
            "regimes": ["RISK_ON"],
            "metadata": {"model": "ensemble"},
        }

        result = validator.validate(summary)

        assert result == {
            "is_consistent": True,
            "issues": [],
        }

    def test_empty_valid_summary_is_consistent(self):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }

        result = validator.validate(summary)

        assert result["is_consistent"] is True
        assert result["issues"] == []

    @pytest.mark.parametrize(
        "field",
        [
            "predictions",
            "confidences",
            "feature_drivers",
            "regimes",
            "metadata",
        ],
    )
    def test_missing_required_field_is_reported(self, field):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }
        del summary[field]

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert f"Missing required field: {field}." in result["issues"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("predictions", "UP"),
            ("confidences", 0.85),
            ("feature_drivers", {}),
            ("regimes", ("RISK_ON",)),
        ],
    )
    def test_invalid_collection_type_is_reported(
        self,
        field,
        value,
    ):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }
        summary[field] = value

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert f"{field} must be a list." in result["issues"]

    @pytest.mark.parametrize(
        "confidence",
        [
            -0.1,
            1.1,
            float("inf"),
            float("-inf"),
            float("nan"),
            "invalid",
            True,
            None,
        ],
    )
    def test_invalid_confidence_is_reported(self, confidence):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [confidence],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert (
            "Confidence values must be finite numbers between 0 and 1."
            in result["issues"]
        )

    @pytest.mark.parametrize(
        "confidence",
        [
            0,
            0.0,
            0.5,
            1,
            1.0,
        ],
    )
    def test_valid_confidence_boundaries_are_accepted(
        self,
        confidence,
    ):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [confidence],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }

        result = validator.validate(summary)

        assert result["is_consistent"] is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("predictions", None),
            ("predictions", ""),
            ("predictions", "   "),
            ("predictions", 123),
            ("predictions", True),
            ("feature_drivers", None),
            ("feature_drivers", ""),
            ("feature_drivers", "   "),
            ("feature_drivers", 123),
            ("feature_drivers", True),
            ("regimes", None),
            ("regimes", ""),
            ("regimes", "   "),
            ("regimes", 123),
            ("regimes", True),
        ],
    )
    def test_invalid_string_collection_values_are_reported(
        self,
        field,
        value,
    ):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }
        summary[field] = [value]

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert (
            f"{field} must contain non-empty strings."
            in result["issues"]
        )

    def test_metadata_must_be_dictionary(self):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": "invalid",
        }

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert "metadata must be a dictionary." in result["issues"]

    def test_multiple_issues_are_all_reported(self):
        validator = ExplanationConsistencyValidator()

        summary = {
            "predictions": ["UP", ""],
            "confidences": [0.8, 1.5],
            "feature_drivers": [123],
            "regimes": ["RISK_ON", None],
            "metadata": [],
        }

        result = validator.validate(summary)

        assert result["is_consistent"] is False
        assert len(result["issues"]) == 5

    def test_none_summary_is_rejected(self):
        validator = ExplanationConsistencyValidator()

        with pytest.raises(ValueError):
            validator.validate(None)

    @pytest.mark.parametrize(
        "summary",
        [
            "invalid",
            123,
            True,
            [],
            (),
        ],
    )
    def test_non_dictionary_summary_is_rejected(
        self,
        summary,
    ):
        validator = ExplanationConsistencyValidator()

        with pytest.raises(TypeError):
            validator.validate(summary)