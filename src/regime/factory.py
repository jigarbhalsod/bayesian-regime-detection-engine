from .clustering import ClusteringRegimeDetector
from .rule_based import RuleBasedRegimeDetector
from .statistical import StatisticalRegimeDetector


class RegimeDetectorFactory:
    """Registry and factory for regime detectors."""

    _registry = {
        "rule_based": RuleBasedRegimeDetector,
        "statistical": StatisticalRegimeDetector,
        "clustering": ClusteringRegimeDetector,
    }

    @classmethod
    def _normalize_name(cls, name):
        """Normalize and validate a detector name."""
        if not isinstance(name, str):
            raise ValueError("Detector name must be a string")

        return name.strip().lower()

    @classmethod
    def available(cls):
        """Return registered detector names."""
        return tuple(sorted(cls._registry.keys()))

    @classmethod
    def create(cls, name, config=None):
        """Create a detector by its registered name."""
        normalized_name = cls._normalize_name(name)

        if normalized_name not in cls._registry:
            raise ValueError(
                f"Unknown regime detector: {name}"
            )

        detector_class = cls._registry[normalized_name]

        return detector_class(config=config)

    @classmethod
    def register(cls, name, detector_class):
        """Register a custom detector class."""
        normalized_name = cls._normalize_name(name)

        if not isinstance(detector_class, type):
            raise TypeError("Detector must be a class")

        if normalized_name in cls._registry:
            raise ValueError(
                f"Detector '{normalized_name}' is already registered"
            )

        cls._registry[normalized_name] = detector_class