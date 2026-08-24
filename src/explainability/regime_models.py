from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RegimeContribution(BaseModel):
    """
    Represents the contribution of a single feature toward
    a specific market regime classification.
    """

    feature_name: str
    contribution: float
    rank: int | None = None

    @field_validator("feature_name", mode="before")
    @classmethod
    def validate_feature_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("feature_name must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("feature_name must not be empty.")

        return value

    @field_validator("contribution", mode="before")
    @classmethod
    def validate_contribution(cls, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("contribution must be a number.")

        if not isinstance(value, (int, float)):
            raise ValueError("contribution must be a number.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError("contribution must be finite.")

        return value

    @field_validator("rank", mode="before")
    @classmethod
    def validate_rank(cls, value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("rank must be an integer.")

        if value < 1:
            raise ValueError(
                "rank must be greater than or equal to 1."
            )

        return value


class RegimeExplanation(BaseModel):
    """
    Structured explanation describing why a particular
    market regime was assigned.
    """

    regime: str | None = None
    probability: float | None = None
    contributions: list[RegimeContribution] = Field(
        default_factory=list
    )
    key_drivers: list[str] = Field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("regime", mode="before")
    @classmethod
    def validate_regime(cls, value: Any) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("regime must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("regime must not be empty.")

        return value

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(
        cls,
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError("probability must be a number.")

        if not isinstance(value, (int, float)):
            raise ValueError("probability must be a number.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError("probability must be finite.")

        if value < 0 or value > 1:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        return value

    @field_validator("key_drivers", mode="before")
    @classmethod
    def validate_key_drivers(
        cls,
        value: Any,
    ) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("key_drivers must be a list.")

        normalized_drivers = []

        for driver in value:
            if not isinstance(driver, str):
                raise ValueError(
                    "each key driver must be a string."
                )

            driver = driver.strip()

            if not driver:
                raise ValueError(
                    "key drivers must not contain empty values."
                )

            normalized_drivers.append(driver)

        return normalized_drivers

    @field_validator("summary", mode="before")
    @classmethod
    def validate_summary(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("summary must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("summary must not be empty.")

        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(
        cls,
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("metadata must be a dictionary.")

        return value