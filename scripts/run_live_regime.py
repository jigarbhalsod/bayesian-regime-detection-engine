from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.model import BayesianRegimeModel


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "live_regime_model.pt"
)


# ---------------------------------------------------------
# Market data download
# ---------------------------------------------------------

def download_market_data() -> pd.DataFrame:
    """
    Download recent NIFTY 50 and India VIX data required
    to calculate the latest market features.
    """

    print("Downloading latest NIFTY 50 data...")

    nifty = yf.download(
        "^NSEI",
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    print("Downloading latest India VIX data...")

    vix = yf.download(
        "^INDIAVIX",
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if nifty.empty:
        raise RuntimeError(
            "Unable to download latest NIFTY 50 data."
        )

    if vix.empty:
        raise RuntimeError(
            "Unable to download latest India VIX data."
        )

    # Handle MultiIndex columns returned by yfinance.
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    nifty = nifty[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    nifty = nifty.rename(
        columns={
            "Close": "nifty_close",
            "Volume": "nifty_volume",
        }
    )

    vix = vix[
        ["Close"]
    ].copy()

    vix = vix.rename(
        columns={
            "Close": "india_vix",
        }
    )

    # Join NIFTY and India VIX using common market dates.
    data = nifty.join(
        vix,
        how="inner",
    )

    data = data.sort_index()

    data = data[
        [
            "nifty_close",
            "nifty_volume",
            "india_vix",
        ]
    ].copy()

    # Remove invalid price/VIX observations.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "nifty_close",
            "india_vix",
        ]
    )

    if data.empty:
        raise RuntimeError(
            "No valid overlapping NIFTY 50 and "
            "India VIX observations were found."
        )

    return data


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the exact same 8 features used during training.

    This function intentionally matches the feature logic
    in train_live_regime_model.py to maintain training and
    inference feature parity.
    """

    frame = data.copy()

    # -----------------------------------------------------
    # Data-quality handling
    # -----------------------------------------------------
    # yfinance may occasionally return zero or non-positive
    # NIFTY index volume values. Treat these as unavailable
    # and forward-fill the latest valid volume.
    # -----------------------------------------------------

    frame["nifty_volume"] = frame[
        "nifty_volume"
    ].where(
        frame["nifty_volume"] > 0,
        np.nan,
    )

    frame["nifty_volume"] = frame[
        "nifty_volume"
    ].ffill()

    # -----------------------------------------------------
    # 1. Short-term return
    # -----------------------------------------------------

    frame["short_return"] = (
        frame["nifty_close"].pct_change(1)
    )

    # -----------------------------------------------------
    # 2. Medium-term return
    # -----------------------------------------------------

    frame["medium_return"] = (
        frame["nifty_close"].pct_change(20)
    )

    # -----------------------------------------------------
    # 3. Momentum
    # -----------------------------------------------------

    frame["momentum"] = (
        frame["nifty_close"].pct_change(10)
    )

    # -----------------------------------------------------
    # 4. Volatility
    # -----------------------------------------------------

    frame["volatility"] = (
        frame["short_return"]
        .rolling(20)
        .std()
    )

    # -----------------------------------------------------
    # 5. Volume change
    # -----------------------------------------------------

    frame["volume_change"] = (
        frame["nifty_volume"].pct_change(1)
    )

    # -----------------------------------------------------
    # 6. RSI
    # -----------------------------------------------------

    delta = frame["nifty_close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    frame["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # -----------------------------------------------------
    # 7. VIX risk
    # -----------------------------------------------------

    frame["vix_risk"] = (
        frame["india_vix"].pct_change(5)
    )

    # -----------------------------------------------------
    # 8. Trend strength
    # -----------------------------------------------------

    sma_20 = (
        frame["nifty_close"]
        .rolling(20)
        .mean()
    )

    frame["trend_strength"] = (
        frame["nifty_close"] / sma_20
    ) - 1.0

    # -----------------------------------------------------
    # Select final features
    # -----------------------------------------------------

    feature_columns = [
        "short_return",
        "medium_return",
        "momentum",
        "volatility",
        "volume_change",
        "rsi",
        "vix_risk",
        "trend_strength",
    ]

    features = frame[
        feature_columns
    ].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.dropna()

    if features.empty:
        raise RuntimeError(
            "No valid feature observations available "
            "for inference."
        )

    return features


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

def load_model_and_checkpoint():
    """
    Load the saved model checkpoint and reconstruct the
    Bayesian regime model.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Model checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Run train_live_regime_model.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config_data = checkpoint["config"]

    config = BayesianModelConfig(
        model_name=config_data["model_name"],
        n_features=config_data["n_features"],
        n_outputs=config_data["n_outputs"],
        n_regimes=config_data["n_regimes"],
        hidden_dims=tuple(
            config_data["hidden_dims"]
        ),
        dropout_rate=config_data["dropout_rate"],
        mc_samples=config_data["mc_samples"],
        random_seed=config_data["random_seed"],
    )

    model = BayesianRegimeModel(
        input_dim=config.n_features,
        config=config,
    )

    # Load the trained neural-network weights.
    model.network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, checkpoint


# ---------------------------------------------------------
# Normalize latest observation
# ---------------------------------------------------------

def normalize_latest_features(
    latest_features: pd.Series,
    checkpoint: dict,
) -> torch.Tensor:
    """
    Apply the exact normalization statistics saved during
    model training.
    """

    feature_columns = checkpoint[
        "feature_columns"
    ]

    # Ensure exact training-time feature order.
    latest_features = latest_features[
        feature_columns
    ]

    raw_features = latest_features.to_numpy(
        dtype=np.float32
    )

    means = np.array(
        checkpoint["feature_means"],
        dtype=np.float32,
    )

    stds = np.array(
        checkpoint["feature_stds"],
        dtype=np.float32,
    )

    # Prevent division by zero as an additional safeguard.
    stds = np.where(
        stds == 0,
        1.0,
        stds,
    )

    normalized_features = (
        raw_features - means
    ) / stds

    if not np.isfinite(
        normalized_features
    ).all():
        raise RuntimeError(
            "Latest normalized features contain "
            "invalid values."
        )

    return torch.tensor(
        normalized_features,
        dtype=torch.float32,
    ).unsqueeze(0)


# ---------------------------------------------------------
# Display prediction
# ---------------------------------------------------------

def print_prediction(
    prediction,
    regime_names,
    latest_date,
    latest_features,
    training_date,
) -> None:
    """
    Display the latest-available market regime assessment.
    """

    regime_id = (
        prediction.predictions[0].item()
    )

    confidence = (
        prediction.confidence[0].item()
    )

    probabilities = (
        prediction.probabilities[0]
    )

    print("\n" + "=" * 60)
    print(
        " BAYESIAN REGIME DETECTION ENGINE — LIVE INFERENCE"
    )
    print("=" * 60)

    print(
        f"\nLatest Market Date : {latest_date}"
    )

    print(
        f"Model Training Date: {training_date}"
    )

    print(
        f"\nDetected Regime    : "
        f"{regime_names[regime_id]}"
    )

    print(
        f"Regime ID          : {regime_id}"
    )

    print(
        f"Confidence         : {confidence:.4f}"
    )

    print("\nRegime Probabilities:")

    for regime_name, probability in zip(
        regime_names,
        probabilities,
    ):
        print(
            f"  {regime_name:<15}: "
            f"{probability.item():.4f}"
        )

    print("\nLatest Market Features:")

    for name, value in latest_features.items():
        print(
            f"  {name:<15}: {value:.6f}"
        )

    print("\n" + "=" * 60)


# ---------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print(
        " BAYESIAN REGIME DETECTION ENGINE"
    )
    print(
        " LATEST-AVAILABLE MARKET REGIME INFERENCE"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Download latest market data
    # -----------------------------------------------------

    data = download_market_data()

    print(
        f"Market observations downloaded: {len(data)}"
    )

    # -----------------------------------------------------
    # 2. Build features
    # -----------------------------------------------------

    print("\nBuilding latest market features...")

    features = build_features(data)

    latest_features = features.iloc[-1].copy()

    latest_date = features.index[-1].date()

    print(
        f"Latest valid feature date: {latest_date}"
    )

    # -----------------------------------------------------
    # 3. Load trained model
    # -----------------------------------------------------

    print(
        "\nLoading trained Bayesian regime model..."
    )

    model, checkpoint = (
        load_model_and_checkpoint()
    )

    # -----------------------------------------------------
    # 4. Validate feature schema
    # -----------------------------------------------------

    checkpoint_features = checkpoint[
        "feature_columns"
    ]

    if list(features.columns) != checkpoint_features:
        raise RuntimeError(
            "Feature schema mismatch between "
            "live inference and model checkpoint.\n"
            f"Live features: {list(features.columns)}\n"
            f"Model features: {checkpoint_features}"
        )

    # -----------------------------------------------------
    # 5. Apply training normalization
    # -----------------------------------------------------

    x = normalize_latest_features(
        latest_features=latest_features,
        checkpoint=checkpoint,
    )

    # -----------------------------------------------------
    # 6. Monte Carlo Dropout inference
    # -----------------------------------------------------

    print(
        "\nRunning Monte Carlo Dropout inference..."
    )

    prediction = (
        model.predict_with_uncertainty(x)
    )

    # -----------------------------------------------------
    # 7. Display results
    # -----------------------------------------------------

    print_prediction(
        prediction=prediction,
        regime_names=checkpoint["regime_names"],
        latest_date=latest_date,
        latest_features=latest_features,
        training_date=checkpoint[
            "latest_training_date"
        ],
    )


if __name__ == "__main__":
    main()