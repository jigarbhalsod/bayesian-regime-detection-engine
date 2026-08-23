import pytest

from src.validation.regime_analysis import (
    ClassificationAnalysisResult,
    Misclassification,
    RegimeClassificationAnalyzer,
    RegimeStatistics,
)


# ============================================================
# Basic Analysis Tests
# ============================================================


def test_analyze_returns_result():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["risk_on", "risk_off"],
        predicted=["risk_on", "risk_off"],
    )

    assert isinstance(result, ClassificationAnalysisResult)


def test_labels_preserve_first_seen_order():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["risk_off", "risk_on"],
        predicted=["transition", "risk_on"],
    )

    assert result.labels == [
        "risk_off",
        "risk_on",
        "transition",
    ]


# ============================================================
# Confusion Matrix Tests
# ============================================================


def test_confusion_matrix():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    assert result.confusion_matrix == {
        "a": {
            "a": 1,
            "b": 1,
        },
        "b": {
            "a": 0,
            "b": 2,
        },
    }


def test_confusion_matrix_includes_predicted_only_label():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a"],
        predicted=["a", "c"],
    )

    assert "c" in result.labels
    assert result.confusion_matrix["c"] == {
        "a": 0,
        "c": 0,
    }


def test_confusion_matrix_is_square():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "b"],
        predicted=["a", "c"],
    )

    label_count = len(result.labels)

    assert len(result.confusion_matrix) == label_count

    assert all(
        len(row) == label_count
        for row in result.confusion_matrix.values()
    )


# ============================================================
# Per-Regime Statistics Tests
# ============================================================


def test_per_regime_statistics():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    class_a = result.per_regime["a"]
    class_b = result.per_regime["b"]

    assert isinstance(class_a, RegimeStatistics)

    assert class_a.support == 2
    assert class_a.correct_predictions == 1
    assert class_a.accuracy == 0.5

    assert class_b.support == 2
    assert class_b.correct_predictions == 2
    assert class_b.accuracy == 1.0


def test_predicted_only_label_has_zero_support():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a"],
        predicted=["a", "c"],
    )

    class_c = result.per_regime["c"]

    assert class_c.support == 0
    assert class_c.correct_predictions == 0
    assert class_c.accuracy == 0.0


# ============================================================
# Overall Count Tests
# ============================================================


def test_overall_prediction_counts():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "a"],
    )

    assert result.total_observations == 4
    assert result.correct_predictions == 2
    assert result.incorrect_predictions == 2


def test_perfect_predictions_have_no_incorrect():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "b", "c"],
        predicted=["a", "b", "c"],
    )

    assert result.correct_predictions == 3
    assert result.incorrect_predictions == 0


# ============================================================
# Misclassification Tests
# ============================================================


def test_most_common_misclassification():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=[
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_off",
            "risk_off",
        ],
        predicted=[
            "transition",
            "transition",
            "risk_on",
            "transition",
            "risk_off",
        ],
    )

    misclassification = (
        result.most_common_misclassification
    )

    assert isinstance(
        misclassification,
        Misclassification,
    )
    assert misclassification.actual_regime == "risk_on"
    assert misclassification.predicted_regime == "transition"
    assert misclassification.count == 2


def test_no_misclassification_for_perfect_predictions():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "b", "c"],
        predicted=["a", "b", "c"],
    )

    assert result.most_common_misclassification is None


def test_misclassification_ignores_correct_predictions():
    analyzer = RegimeClassificationAnalyzer()

    result = analyzer.analyze(
        actual=["a", "a", "a", "b"],
        predicted=["a", "a", "a", "a"],
    )

    misclassification = (
        result.most_common_misclassification
    )

    assert misclassification.actual_regime == "b"
    assert misclassification.predicted_regime == "a"
    assert misclassification.count == 1


# ============================================================
# Input Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "actual, predicted",
    [
        ([], []),
        ([], ["a"]),
        (["a"], []),
    ],
)
def test_empty_inputs(actual, predicted):
    analyzer = RegimeClassificationAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(actual, predicted)


def test_mismatched_lengths():
    analyzer = RegimeClassificationAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(
            actual=["a", "b"],
            predicted=["a"],
        )


@pytest.mark.parametrize(
    "actual",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_actual_type(actual):
    analyzer = RegimeClassificationAnalyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(
            actual=actual,
            predicted=["a"],
        )


@pytest.mark.parametrize(
    "predicted",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_predicted_type(predicted):
    analyzer = RegimeClassificationAnalyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(
            actual=["a"],
            predicted=predicted,
        )


# ============================================================
# Helper Method Tests
# ============================================================


def test_get_labels():
    labels = RegimeClassificationAnalyzer._get_labels(
        actual=["b", "a", "b"],
        predicted=["c", "a", "d"],
    )

    assert labels == ["b", "a", "c", "d"]


def test_find_most_common_misclassification_returns_none():
    matrix = {
        "a": {"a": 2, "b": 0},
        "b": {"a": 0, "b": 1},
    }

    result = (
        RegimeClassificationAnalyzer
        ._find_most_common_misclassification(
            matrix,
            ["a", "b"],
        )
    )

    assert result is None