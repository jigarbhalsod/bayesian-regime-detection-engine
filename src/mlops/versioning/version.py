from __future__ import annotations

from dataclasses import dataclass
import re


_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
)


@dataclass(frozen=True, order=True)
class Version:
    """
    Immutable semantic version.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> "Version":
        """
        Parse a semantic version string such as '1.2.3'.
        """
        if not isinstance(value, str):
            raise TypeError("version must be a string.")

        value = value.strip()

        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError(
                "version must follow semantic version format "
                "'MAJOR.MINOR.PATCH'."
            )

        major, minor, patch = map(int, value.split("."))

        return cls(
            major=major,
            minor=minor,
            patch=patch,
        )

    def bump_major(self) -> "Version":
        return Version(
            major=self.major + 1,
            minor=0,
            patch=0,
        )

    def bump_minor(self) -> "Version":
        return Version(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
        )

    def bump_patch(self) -> "Version":
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
        )