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
from src.features.volatility import VolatilityFeatureTransformer
from src.features.volume import VolumeFeatureTransformer
from src.features.technical import TechnicalIndicatorTransformer

__all__ = [
    "ReturnFeatureTransformer",
    "VolatilityFeatureTransformer",
    "VolumeFeatureTransformer",
    "TechnicalIndicatorTransformer",
    "BaseFeatureTransformer",
    "FeatureConfig",
    "FeatureEngineeringPipeline",
    "FeaturePipelineResult",
    "FeatureRegistry",
    "FeatureResult",
    "MomentumFeatureTransformer",
    "PriceFeatureTransformer",
    "ReturnFeatureTransformer",
    "VolatilityFeatureTransformer",
    "VolumeFeatureTransformer",
]