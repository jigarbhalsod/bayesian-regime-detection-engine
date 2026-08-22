from src.regime import RegimeResult
from src.regime.validation import RegimeValidator


def test_validator_uses_default_allowed_regimes():
    validator = RegimeValidator()

    assert validator.allowed_regimes == {
        "risk_on",
        "risk_off",
        "transitional",
        "unknown",
    }


def test_validator_accepts_custom_allowed_regimes():
    validator = RegimeValidator(
        allowed_regimes={"bull", "bear"}
    )

    assert validator.allowed_regimes == {"bull", "bear"}


def test_validate_returns_true_for_valid_regime_result():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", "risk_off", "transitional"]
    )

    assert validator.validate(result) is True


def test_validate_returns_false_for_invalid_regime_label():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", "invalid_regime"]
    )

    assert validator.validate(result) is False


def test_validate_accepts_unknown_regime():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", "unknown", "risk_off"]
    )

    assert validator.validate(result) is True


def test_validate_handles_empty_result():
    validator = RegimeValidator()

    result = RegimeResult()

    assert validator.validate(result) is True


def test_validate_rejects_non_regime_result():
    validator = RegimeValidator()

    assert validator.validate({"labels": ["risk_on"]}) is False


def test_validate_rejects_non_string_labels():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", 123, None]
    )

    assert validator.validate(result) is False


def test_validate_rejects_invalid_labels_container():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels="risk_on"
    )

    assert validator.validate(result) is False


def test_validate_custom_allowed_regimes():
    validator = RegimeValidator(
        allowed_regimes={"bull", "bear"}
    )

    result = RegimeResult(
        regime_labels=["bull", "bear"]
    )

    assert validator.validate(result) is True


def test_validate_rejects_default_label_with_custom_regimes():
    validator = RegimeValidator(
        allowed_regimes={"bull", "bear"}
    )

    result = RegimeResult(
        regime_labels=["risk_on"]
    )

    assert validator.validate(result) is False


def test_validate_normalizes_whitespace_and_case():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=[" RISK_ON ", "Risk_Off", "TRANSITIONAL"]
    )

    assert validator.validate(result) is True


def test_invalid_labels_returns_all_invalid_labels():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", "bad", "another_bad", "risk_off"]
    )

    assert validator.invalid_labels(result) == [
        "bad",
        "another_bad",
    ]


def test_invalid_labels_returns_empty_for_valid_result():
    validator = RegimeValidator()

    result = RegimeResult(
        regime_labels=["risk_on", "unknown"]
    )

    assert validator.invalid_labels(result) == []


def test_invalid_labels_handles_non_regime_result():
    validator = RegimeValidator()

    assert validator.invalid_labels(
        {"labels": ["bad"]}
    ) == []