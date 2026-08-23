import pytest

from src.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardWindow,
)


# ============================================================
# WalkForwardWindow Tests
# ============================================================


def test_walk_forward_window_sizes():
    window = WalkForwardWindow(
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
    )

    assert window.train_size == 10
    assert window.test_size == 5


def test_walk_forward_window_is_immutable():
    window = WalkForwardWindow(
        train_start=0,
        train_end=10,
        test_start=10,
        test_end=15,
    )

    with pytest.raises(Exception):
        window.train_end = 20


# ============================================================
# Configuration Tests
# ============================================================


def test_default_step_size_uses_test_size():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    assert validator.step_size == 5
    assert validator.expanding is True


def test_custom_configuration():
    validator = WalkForwardValidator(
        train_size=20,
        test_size=5,
        step_size=2,
        expanding=False,
    )

    assert validator.train_size == 20
    assert validator.test_size == 5
    assert validator.step_size == 2
    assert validator.expanding is False


@pytest.mark.parametrize(
    "parameter",
    [
        "train_size",
        "test_size",
        "step_size",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
        1.5,
        "invalid",
        True,
    ],
)
def test_invalid_window_sizes(parameter, invalid_value):
    kwargs = {
        "train_size": 10,
        "test_size": 5,
        "step_size": 2,
    }

    kwargs[parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        WalkForwardValidator(**kwargs)


def test_invalid_expanding_type():
    with pytest.raises(TypeError):
        WalkForwardValidator(
            train_size=10,
            test_size=5,
            expanding="yes",
        )


# ============================================================
# Expanding Window Tests
# ============================================================


def test_expanding_windows():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    windows = validator.get_windows(data_size=10)

    assert len(windows) == 3

    assert windows[0] == WalkForwardWindow(0, 4, 4, 6)
    assert windows[1] == WalkForwardWindow(0, 6, 6, 8)
    assert windows[2] == WalkForwardWindow(0, 8, 8, 10)


def test_expanding_training_window_grows():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    windows = validator.get_windows(data_size=10)

    assert [window.train_size for window in windows] == [
        4,
        6,
        8,
    ]


def test_expanding_windows_preserve_temporal_order():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    windows = validator.get_windows(data_size=10)

    for window in windows:
        assert window.train_start < window.train_end
        assert window.train_end == window.test_start
        assert window.test_start < window.test_end


# ============================================================
# Rolling Window Tests
# ============================================================


def test_rolling_windows():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
        expanding=False,
    )

    windows = validator.get_windows(data_size=10)

    assert len(windows) == 3

    assert windows[0] == WalkForwardWindow(0, 4, 4, 6)
    assert windows[1] == WalkForwardWindow(2, 6, 6, 8)
    assert windows[2] == WalkForwardWindow(4, 8, 8, 10)


def test_rolling_training_window_has_fixed_size():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
        expanding=False,
    )

    windows = validator.get_windows(data_size=10)

    assert [window.train_size for window in windows] == [
        4,
        4,
        4,
    ]


# ============================================================
# Step Size Tests
# ============================================================


def test_custom_step_size():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
        step_size=1,
    )

    windows = validator.get_windows(data_size=8)

    assert len(windows) == 3

    assert windows[0] == WalkForwardWindow(0, 4, 4, 6)
    assert windows[1] == WalkForwardWindow(0, 5, 5, 7)
    assert windows[2] == WalkForwardWindow(0, 6, 6, 8)


def test_step_size_preserves_future_testing():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
        step_size=1,
    )

    windows = validator.get_windows(data_size=8)

    for window in windows:
        assert window.train_end <= window.test_start


# ============================================================
# Boundary Tests
# ============================================================


def test_exact_minimum_data_size():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    windows = validator.get_windows(data_size=6)

    assert len(windows) == 1
    assert windows[0] == WalkForwardWindow(0, 4, 4, 6)


def test_insufficient_data_size():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    with pytest.raises(ValueError):
        validator.get_windows(data_size=5)


@pytest.mark.parametrize(
    "invalid_data_size",
    [
        0,
        -1,
        1.5,
        "invalid",
        True,
    ],
)
def test_invalid_data_size(invalid_data_size):
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    with pytest.raises((TypeError, ValueError)):
        validator.get_windows(invalid_data_size)


def test_all_windows_within_data_boundaries():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
        step_size=1,
    )

    data_size = 10
    windows = validator.get_windows(data_size)

    for window in windows:
        assert window.train_start >= 0
        assert window.train_end <= data_size
        assert window.test_start >= 0
        assert window.test_end <= data_size


# ============================================================
# Generator Tests
# ============================================================


def test_generate_windows_returns_iterator():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    generator = validator.generate_windows(data_size=10)

    assert hasattr(generator, "__iter__")
    assert hasattr(generator, "__next__")


def test_get_windows_matches_generator_output():
    validator = WalkForwardValidator(
        train_size=4,
        test_size=2,
    )

    from_generator = list(
        validator.generate_windows(data_size=10)
    )
    from_method = validator.get_windows(data_size=10)

    assert from_generator == from_method