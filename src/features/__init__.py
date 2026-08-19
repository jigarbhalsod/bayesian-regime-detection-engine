"""Feature engineering package for Phase 6."""

from src.features.base import (
    BaseFeatureTransformer,
    FeatureResult,
)
from src.features.config import FeatureConfig
from src.features.momentum import MomentumFeatureTransformer
from src.features.pipeline import (
    FeatureEngineeringPipeline,
    FeaturePipelineResult,
)
from src.features.price import PriceFeatureTransformer
from src.features.registry import FeatureRegistry
from src.features.returns import ReturnFeatureTransformer

__all__ = [
    "BaseFeatureTransformer",
    "FeatureConfig",
    "FeatureEngineeringPipeline",
    "FeaturePipelineResult",
    "FeatureRegistry",
    "FeatureResult",
    "MomentumFeatureTransformer",
    "PriceFeatureTransformer",
    "ReturnFeatureTransformer",
]