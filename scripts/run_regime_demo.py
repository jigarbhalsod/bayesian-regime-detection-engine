from __future__ import annotations

import sys
from pathlib import Path

import torch


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.model import BayesianRegimeModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

torch.manual_seed(42)

REGIME_NAMES = [
    "RISK_ON",
    "LATE_CYCLE",
    "TRANSITIONAL",
    "POST_SHOCK",
    "RISK_OFF",
]

INPUT_DIM = 8


def create_demo_data(
    samples_per_regime: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic market feature data for demonstrating
    the Bayesian regime classification pipeline.

    Feature layout:
    0: short-term return
    1: medium-term return
    2: momentum
    3: volatility
    4: volume change
    5: RSI
    6: VIX-like risk measure
    7: trend strength
    """

    regime_centers = torch.tensor(
        [
            # RISK_ON
            [0.8, 0.7, 0.9, -0.5, 0.4, 0.7, -0.7, 0.9],

            # LATE_CYCLE
            [0.4, 0.3, 0.2, 0.1, 0.2, 0.5, 0.0, 0.3],

            # TRANSITIONAL
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0],

            # POST_SHOCK
            [-0.3, -0.2, -0.4, 0.8, 0.5, -0.3, 0.9, -0.4],

            # RISK_OFF
            [-0.8, -0.7, -0.9, 0.7, -0.4, -0.7, 0.8, -0.9],
        ],
        dtype=torch.float32,
    )

    features = []
    labels = []

    for regime_id, center in enumerate(regime_centers):
        noise = torch.randn(
            samples_per_regime,
            INPUT_DIM,
        ) * 0.25

        regime_features = center + noise

        features.append(regime_features)

        labels.append(
            torch.full(
                (samples_per_regime,),
                regime_id,
                dtype=torch.long,
            )
        )

    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    indices = torch.randperm(x.shape[0])

    return x[indices], y[indices]


def print_prediction(
    prediction,
) -> None:
    print("\n" + "=" * 60)
    print(" BAYESIAN REGIME DETECTION ENGINE — LIVE MODEL DEMO")
    print("=" * 60)

    regime_id = prediction.predictions[0].item()
    confidence = prediction.confidence[0].item()

    print(f"\nDetected Regime : {REGIME_NAMES[regime_id]}")
    print(f"Regime ID       : {regime_id}")
    print(f"Confidence      : {confidence:.4f}")

    print("\nRegime Probabilities:")

    probabilities = prediction.probabilities[0]

    for regime_name, probability in zip(
        REGIME_NAMES,
        probabilities,
    ):
        print(
            f"  {regime_name:<15}: "
            f"{probability.item():.4f}"
        )

    print("\nInput Market Features:")

    feature_names = [
        "short_return",
        "medium_return",
        "momentum",
        "volatility",
        "volume_change",
        "rsi",
        "vix_risk",
        "trend_strength",
    ]

    print("\n" + "=" * 60)


def main() -> None:

    # -----------------------------------------------------
    # 1. Create configuration
    # -----------------------------------------------------

    config = BayesianModelConfig(
        model_name="bayesian_regime_demo",
        n_features=INPUT_DIM,
        n_outputs=5,
        n_regimes=5,
        hidden_dims=(32, 16),
        dropout_rate=0.20,
        mc_samples=30,
        random_seed=42,
    )

    # -----------------------------------------------------
    # 2. Create model
    # -----------------------------------------------------

    model = BayesianRegimeModel(
        input_dim=INPUT_DIM,
        config=config,
    )

    print("\nCreating synthetic market regime dataset...")

    x, y = create_demo_data(
        samples_per_regime=120,
    )

    print(f"Training samples : {x.shape[0]}")
    print(f"Input features   : {x.shape[1]}")
    print(f"Number of regimes: {config.n_regimes}")

    # -----------------------------------------------------
    # 3. Train model
    # -----------------------------------------------------

    print("\nTraining Bayesian regime model...")

    model.fit(
        x=x,
        y=y,
        epochs=250,
        learning_rate=0.005,
    )

    print("Training complete.")

    # -----------------------------------------------------
    # 4. Create new market observation
    # -----------------------------------------------------

    market_observation = torch.tensor(
        [
            [
                0.75,   # short_return
                0.65,   # medium_return
                0.85,   # momentum
                -0.40,  # volatility
                0.35,   # volume_change
                0.65,   # RSI
                -0.60,  # VIX risk
                0.80,   # trend strength
            ]
        ],
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # 5. Predict with uncertainty
    # -----------------------------------------------------

    print("\nRunning Monte Carlo Dropout inference...")

    prediction = model.predict_with_uncertainty(
        market_observation,
    )

    # -----------------------------------------------------
    # 6. Display output
    # -----------------------------------------------------

    print_prediction(prediction)


if __name__ == "__main__":
    main()