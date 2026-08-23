from __future__ import annotations

from typing import Any, Mapping

from src.integration.result import IntegrationResult


class ModelOutputAdapter:
    """
    Normalize heterogeneous advanced-model outputs into
    a shared output contract.
    """

    def adapt(
        self,
        model_name: str,
        output: Any,
    ) -> dict[str, Any]:
        """
        Adapt one model output.

        The returned contract always contains:

        - model_name
        - prediction
        - probabilities
        - uncertainty
        - metadata
        """
        model_name = self._validate_model_name(
            model_name
        )

        if isinstance(output, Mapping):
            return self._adapt_mapping(
                model_name,
                output,
            )

        return {
            "model_name": model_name,
            "prediction": output,
            "probabilities": None,
            "uncertainty": None,
            "metadata": {},
        }

    def adapt_all(
        self,
        result: IntegrationResult,
    ) -> dict[str, dict[str, Any]]:
        """
        Adapt every successful output in an IntegrationResult.
        """
        if not isinstance(
            result,
            IntegrationResult,
        ):
            raise TypeError(
                "result must be an IntegrationResult."
            )

        return {
            model_name: self.adapt(
                model_name,
                output,
            )
            for model_name, output
            in result.outputs.items()
        }

    @staticmethod
    def _validate_model_name(
        model_name: Any,
    ) -> str:
        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        return model_name

    @staticmethod
    def _adapt_mapping(
        model_name: str,
        output: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Adapt a mapping-based model output.

        Supports:
        - prediction
        - value (prediction alias)
        - probabilities
        - probability (probabilities alias)
        - uncertainty
        - metadata

        Any other fields are preserved in metadata.
        """
        reserved_keys = {
            "model_name",
            "prediction",
            "value",
            "probabilities",
            "probability",
            "uncertainty",
            "metadata",
        }

        metadata: dict[str, Any] = {}

        supplied_metadata = output.get(
            "metadata",
            {},
        )

        if supplied_metadata is not None:
            if not isinstance(
                supplied_metadata,
                Mapping,
            ):
                raise TypeError(
                    "output metadata must be a mapping."
                )

            metadata.update(
                supplied_metadata
            )

        for key, value in output.items():
            if key not in reserved_keys:
                metadata[key] = value

        return {
            "model_name": model_name,
            "prediction": output.get(
                "prediction",
                output.get("value"),
            ),
            "probabilities": output.get(
                "probabilities",
                output.get("probability"),
            ),
            "uncertainty": output.get(
                "uncertainty",
            ),
            "metadata": metadata,
        }