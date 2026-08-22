from src.regime import RegimeConfig, RegimeResult
from src.regime.clustering import ClusteringRegimeDetector


def make_config(
    n_regimes: int = 2,
) -> RegimeConfig:
    return RegimeConfig(
        n_regimes=n_regimes,
        feature_columns=("return", "volatility"),
        random_state=42,
    )


def make_records() -> list[dict[str, float]]:
    return [
        {"return": -0.03, "volatility": 0.05},
        {"return": -0.02, "volatility": 0.04},
        {"return": -0.01, "volatility": 0.03},
        {"return": 0.01, "volatility": 0.01},
        {"return": 0.02, "volatility": 0.01},
        {"return": 0.03, "volatility": 0.02},
    ]


def test_clustering_detector_name():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    assert detector.name == "clustering"


def test_clustering_detector_requires_feature_columns():
    config = RegimeConfig()

    try:
        ClusteringRegimeDetector(config=config)
    except ValueError as error:
        assert "feature column" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_clustering_detector_returns_regime_result():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(make_records())

    assert isinstance(result, RegimeResult)


def test_clustering_detector_returns_label_for_each_record():
    records = make_records()

    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(records)

    assert len(result.regime_labels) == len(records)
    assert len(result.records) == len(records)


def test_clustering_detector_uses_expected_number_of_clusters():
    detector = ClusteringRegimeDetector(
        config=make_config(n_regimes=2)
    )

    result = detector.detect(make_records())

    valid_labels = {
        label
        for label in result.regime_labels
        if label != "unknown"
    }

    assert len(valid_labels) == 2


def test_clustering_detector_returns_cluster_ids_as_strings():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(make_records())

    assert all(
        label in {"0", "1"}
        for label in result.regime_labels
    )


def test_clustering_detector_stores_labels_in_records():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(make_records())

    assert all(
        record["regime"] == label
        for record, label in zip(
            result.records,
            result.regime_labels,
        )
    )


def test_clustering_detector_handles_empty_records():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect([])

    assert result.records == []
    assert result.regime_labels == []
    assert result.metadata["n_valid_records"] == 0


def test_clustering_detector_marks_invalid_records_unknown():
    records = [
        {"return": -0.03, "volatility": 0.05},
        {"return": -0.02, "volatility": 0.04},
        {"return": 0.01, "volatility": 0.01},
        {"return": 0.02, "volatility": 0.01},
        {"return": None, "volatility": 0.02},
    ]

    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(records)

    assert result.regime_labels[-1] == "unknown"


def test_clustering_detector_handles_invalid_numeric_values():
    records = [
        {"return": -0.03, "volatility": 0.05},
        {"return": -0.02, "volatility": 0.04},
        {"return": 0.01, "volatility": 0.01},
        {"return": 0.02, "volatility": 0.01},
        {"return": float("nan"), "volatility": 0.02},
        {"return": 0.01, "volatility": float("inf")},
    ]

    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(records)

    assert result.regime_labels[-2:] == [
        "unknown",
        "unknown",
    ]


def test_clustering_detector_handles_numeric_strings():
    records = [
        {"return": "-0.03", "volatility": "0.05"},
        {"return": "-0.02", "volatility": "0.04"},
        {"return": "0.01", "volatility": "0.01"},
        {"return": "0.02", "volatility": "0.01"},
    ]

    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(records)

    assert all(
        label != "unknown"
        for label in result.regime_labels
    )


def test_clustering_detector_marks_all_unknown_when_insufficient_samples():
    detector = ClusteringRegimeDetector(
        config=make_config(n_regimes=3)
    )

    records = [
        {"return": 0.01, "volatility": 0.01},
        {"return": 0.02, "volatility": 0.02},
    ]

    result = detector.detect(records)

    assert result.regime_labels == [
        "unknown",
        "unknown",
    ]


def test_clustering_detector_does_not_mutate_input():
    records = make_records()
    original = [dict(record) for record in records]

    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    detector.detect(records)

    assert records == original


def test_clustering_detector_returns_metadata():
    detector = ClusteringRegimeDetector(
        config=make_config()
    )

    result = detector.detect(make_records())

    assert result.metadata["detector"] == "clustering"
    assert result.metadata["feature_columns"] == (
        "return",
        "volatility",
    )
    assert result.metadata["n_clusters"] == 2
    assert result.metadata["n_valid_records"] == 6
    assert "cluster_centers" in result.metadata


def test_clustering_detector_is_deterministic():
    records = make_records()

    detector_one = ClusteringRegimeDetector(
        config=make_config()
    )
    detector_two = ClusteringRegimeDetector(
        config=make_config()
    )

    result_one = detector_one.detect(records)
    result_two = detector_two.detect(records)

    assert result_one.regime_labels == result_two.regime_labels