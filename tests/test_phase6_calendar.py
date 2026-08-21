"""Tests for Phase 6 calendar and market context features."""

from __future__ import annotations

from datetime import date, datetime

from src.features.calendar import CalendarFeatureTransformer


def test_creates_expected_calendar_features() -> None:
    """Transformer should report all calendar features."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {
                "date": "2024-01-15",
            }
        ]
    )

    assert result.created_features == (
        "feature__calendar__day_of_week",
        "feature__calendar__month",
        "feature__calendar__quarter",
        "feature__calendar__is_month_start",
        "feature__calendar__is_month_end",
        "feature__calendar__is_quarter_end",
    )


def test_calculates_day_month_and_quarter() -> None:
    """Calendar components should be derived correctly."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {
                "date": "2024-08-15",
            }
        ]
    )

    record = result.records[0]

    # Thursday = 3 because Monday = 0.
    assert record[
        "feature__calendar__day_of_week"
    ] == 3

    assert record[
        "feature__calendar__month"
    ] == 8

    assert record[
        "feature__calendar__quarter"
    ] == 3


def test_identifies_month_start() -> None:
    """First calendar day should be marked as month start."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {"date": "2024-02-01"},
            {"date": "2024-02-02"},
        ]
    )

    assert result.records[0][
        "feature__calendar__is_month_start"
    ] is True

    assert result.records[1][
        "feature__calendar__is_month_start"
    ] is False


def test_identifies_month_end_including_leap_year() -> None:
    """Month-end detection should correctly handle leap years."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {"date": "2024-02-28"},
            {"date": "2024-02-29"},
            {"date": "2024-03-31"},
        ]
    )

    assert result.records[0][
        "feature__calendar__is_month_end"
    ] is False

    assert result.records[1][
        "feature__calendar__is_month_end"
    ] is True

    assert result.records[2][
        "feature__calendar__is_month_end"
    ] is True


def test_identifies_quarter_end() -> None:
    """Quarter-end should only be true on the final day of a quarter."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {"date": "2024-03-30"},
            {"date": "2024-03-31"},
            {"date": "2024-06-30"},
            {"date": "2024-12-31"},
        ]
    )

    assert result.records[0][
        "feature__calendar__is_quarter_end"
    ] is False

    assert result.records[1][
        "feature__calendar__is_quarter_end"
    ] is True

    assert result.records[2][
        "feature__calendar__is_quarter_end"
    ] is True

    assert result.records[3][
        "feature__calendar__is_quarter_end"
    ] is True


def test_supports_datetime_and_date_objects() -> None:
    """Transformer should accept standard date and datetime objects."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {"date": date(2024, 1, 1)},
            {"date": datetime(2024, 12, 31, 10, 30)},
        ]
    )

    assert result.records[0][
        "feature__calendar__month"
    ] == 1

    assert result.records[1][
        "feature__calendar__month"
    ] == 12

    assert result.records[1][
        "feature__calendar__is_month_end"
    ] is True

    assert result.records[1][
        "feature__calendar__is_quarter_end"
    ] is True


def test_invalid_date_produces_missing_features() -> None:
    """Invalid or missing dates should produce missing calendar features."""

    transformer = CalendarFeatureTransformer()

    result = transformer.transform(
        [
            {"date": "not-a-date"},
            {"date": None},
            {},
        ]
    )

    for record in result.records:
        assert record[
            "feature__calendar__day_of_week"
        ] is None

        assert record[
            "feature__calendar__month"
        ] is None

        assert record[
            "feature__calendar__quarter"
        ] is None

        assert record[
            "feature__calendar__is_month_start"
        ] is None

        assert record[
            "feature__calendar__is_month_end"
        ] is None

        assert record[
            "feature__calendar__is_quarter_end"
        ] is None


def test_transformer_does_not_mutate_input() -> None:
    """Calendar transformer should preserve the original input records."""

    transformer = CalendarFeatureTransformer()

    records = [
        {"date": "2024-03-31"},
        {"date": "2024-04-01"},
    ]

    transformer.transform(records)

    assert records == [
        {"date": "2024-03-31"},
        {"date": "2024-04-01"},
    ]