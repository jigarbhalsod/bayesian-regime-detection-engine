import pytest

from src.regime.labeling import RegimeLabelMapper


def test_mapper_uses_default_score_column():
    mapper = RegimeLabelMapper()

    assert mapper.score_column == "return"


def test_mapper_accepts_custom_score_column():
    mapper = RegimeLabelMapper(
        score_column="momentum"
    )

    assert mapper.score_column == "momentum"


def test_build_mapping_requires_matching_lengths():
    mapper = RegimeLabelMapper()

    records = [
        {"return": 0.01},
    ]
    labels = ["0", "1"]

    with pytest.raises(ValueError):
        mapper.build_mapping(records, labels)


def test_build_mapping_returns_empty_for_no_valid_data():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": None},
            {"return": float("nan")},
        ],
        cluster_labels=["0", "1"],
    )

    assert mapping == {}


def test_single_cluster_maps_to_transitional():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": 0.01},
            {"return": 0.02},
        ],
        cluster_labels=["0", "0"],
    )

    assert mapping == {
        "0": "transitional",
    }


def test_two_clusters_map_to_risk_off_and_risk_on():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.03},
            {"return": -0.02},
            {"return": 0.02},
            {"return": 0.03},
        ],
        cluster_labels=["0", "0", "1", "1"],
    )

    assert mapping["0"] == "risk_off"
    assert mapping["1"] == "risk_on"


def test_three_clusters_include_transitional():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.03},
            {"return": 0.00},
            {"return": 0.03},
        ],
        cluster_labels=["2", "0", "1"],
    )

    assert mapping["2"] == "risk_off"
    assert mapping["0"] == "transitional"
    assert mapping["1"] == "risk_on"


def test_middle_clusters_map_to_transitional():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.04},
            {"return": -0.01},
            {"return": 0.01},
            {"return": 0.04},
        ],
        cluster_labels=["0", "1", "2", "3"],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "transitional",
        "2": "transitional",
        "3": "risk_on",
    }


def test_mapping_is_based_on_average_score():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": 0.01},
            {"return": -0.03},
            {"return": 0.03},
            {"return": -0.01},
        ],
        cluster_labels=["0", "1", "0", "1"],
    )

    assert mapping["1"] == "risk_off"
    assert mapping["0"] == "risk_on"


def test_mapper_supports_integer_cluster_labels():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.02},
            {"return": 0.02},
        ],
        cluster_labels=[0, 1],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "risk_on",
    }


def test_mapper_normalizes_numeric_string_labels():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.02},
            {"return": 0.02},
        ],
        cluster_labels=[" 00 ", " 01 "],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "risk_on",
    }


def test_mapper_ignores_invalid_cluster_labels_when_building():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.02},
            {"return": 0.00},
            {"return": 0.02},
        ],
        cluster_labels=["0", "unknown", "1"],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "risk_on",
    }


def test_mapper_ignores_invalid_scores():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": -0.02},
            {"return": float("nan")},
            {"return": 0.02},
            {"return": float("inf")},
        ],
        cluster_labels=["0", "0", "1", "1"],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "risk_on",
    }


def test_map_labels_applies_mapping():
    mapper = RegimeLabelMapper()

    regimes = mapper.map_labels(
        cluster_labels=["0", "1", "2"],
        mapping={
            "0": "risk_off",
            "1": "transitional",
            "2": "risk_on",
        },
    )

    assert regimes == [
        "risk_off",
        "transitional",
        "risk_on",
    ]


def test_map_labels_returns_unknown_for_unmapped_clusters():
    mapper = RegimeLabelMapper()

    regimes = mapper.map_labels(
        cluster_labels=["0", "1", "2"],
        mapping={
            "0": "risk_off",
            "1": "risk_on",
        },
    )

    assert regimes == [
        "risk_off",
        "risk_on",
        "unknown",
    ]


def test_map_labels_returns_unknown_for_invalid_labels():
    mapper = RegimeLabelMapper()

    regimes = mapper.map_labels(
        cluster_labels=["0", "unknown", None, "1"],
        mapping={
            "0": "risk_off",
            "1": "risk_on",
        },
    )

    assert regimes == [
        "risk_off",
        "unknown",
        "unknown",
        "risk_on",
    ]


def test_custom_score_column_is_used():
    mapper = RegimeLabelMapper(
        score_column="momentum"
    )

    mapping = mapper.build_mapping(
        records=[
            {"return": 0.50, "momentum": -2.0},
            {"return": -0.50, "momentum": 2.0},
        ],
        cluster_labels=["0", "1"],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "risk_on",
    }


def test_mapping_is_deterministic_for_equal_scores():
    mapper = RegimeLabelMapper()

    mapping = mapper.build_mapping(
        records=[
            {"return": 0.01},
            {"return": 0.01},
            {"return": 0.01},
        ],
        cluster_labels=["2", "1", "0"],
    )

    assert mapping == {
        "0": "risk_off",
        "1": "transitional",
        "2": "risk_on",
    }