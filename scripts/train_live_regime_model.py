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

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "live_regime_model.pt"

REGIME_NAMES = [
    "RISK_ON",
    "LATE_CYCLE",
    "TRANSITIONAL",
    "POST_SHOCK",
    "RISK_OFF",
]

FEATURE_COLUMNS = [
    "short_return",
    "medium_return",
    "momentum",
    "volatility",
    "volume_change",
    "rsi",
    "vix_risk",
    "trend_strength",
]


# ---------------------------------------------------------
# Market data download
# ---------------------------------------------------------

def download_market_data() -> pd.DataFrame:
    """
    Download historical NIFTY 50 and India VIX data and
    combine them into one aligned market dataset.
    """

    print("Downloading NIFTY 50 historical data...")

    nifty = yf.download(
        "^NSEI",
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    print("Downloading India VIX historical data...")

    vix = yf.download(
        "^INDIAVIX",
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if nifty.empty:
        raise RuntimeError(
            "Unable to download NIFTY 50 historical data."
        )

    if vix.empty:
        raise RuntimeError(
            "Unable to download India VIX historical data."
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

    data = nifty.join(
        vix,
        how="inner",
    )

    data = data.sort_index()

    # Keep only the columns required by this pipeline.
    data = data[
        [
            "nifty_close",
            "nifty_volume",
            "india_vix",
        ]
    ].copy()

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

    return data


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the 8 market features used by the Bayesian
    regime model.

    These exact definitions are also used by
    run_live_regime.py to maintain training/inference parity.
    """

    frame = data.copy()

    # -----------------------------------------------------
    # Data-quality handling
    # -----------------------------------------------------
    # yfinance may occasionally return zero or non-positive
    # NIFTY index volume values. Treat these as unavailable
    # rather than allowing them to create artificial -1.0
    # volume changes.
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
    # Final data cleaning
    # -----------------------------------------------------

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return frame


# ---------------------------------------------------------
# Regime labelling
# ---------------------------------------------------------

def create_regime_labels(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Create transparent rule-based training labels.

    Labels are based on combinations of trend, momentum,
    volatility and VIX behaviour.

    Regime IDs:
        0 -> RISK_ON
        1 -> LATE_CYCLE
        2 -> TRANSITIONAL
        3 -> POST_SHOCK
        4 -> RISK_OFF
    """

    labels = pd.Series(
        2,
        index=frame.index,
        dtype=np.int64,
    )

    # Risk-off:
    # Negative trend/momentum with elevated volatility or VIX.
    risk_off = (
        (frame["short_return"] < -0.01)
        & (
            (frame["volatility"] > frame["volatility"].quantile(0.70))
            | (frame["vix_risk"] > 0.05)
        )
    )

    # Post-shock:
    # High volatility and VIX stress with weak market momentum.
    post_shock = (
        (frame["volatility"] > frame["volatility"].quantile(0.90))
        & (frame["vix_risk"] > 0.05)
        & (frame["momentum"] < 0)
    )

    # Risk-on:
    # Positive trend and momentum with comparatively lower risk.
    risk_on = (
        (frame["trend_strength"] > 0)
        & (frame["momentum"] > 0)
        & (frame["short_return"] > 0)
        & (
            frame["volatility"]
            < frame["volatility"].quantile(0.60)
        )
        & (frame["vix_risk"] <= 0.02)
    )

    # Late-cycle:
    # Positive trend remains, but momentum/risk conditions
    # are less supportive than a clear risk-on environment.
    late_cycle = (
        (frame["trend_strength"] > 0)
        & (
            (frame["momentum"] <= 0)
            | (
                frame["volatility"]
                >= frame["volatility"].quantile(0.60)
            )
            | (frame["vix_risk"] > 0.02)
        )
    )

    # Apply labels.
    labels.loc[risk_on] = 0
    labels.loc[late_cycle] = 1
    labels.loc[post_shock] = 3
    labels.loc[risk_off] = 4

    return labels


# ---------------------------------------------------------
# Prepare training data
# ---------------------------------------------------------

def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Select valid feature rows, create regime labels and
    return the training matrix.
    """

    valid_frame = frame.copy()

    valid_frame = valid_frame.dropna(
        subset=FEATURE_COLUMNS,
    )

    labels = create_regime_labels(
        valid_frame
    )

    x_raw = valid_frame[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y = labels.to_numpy(
        dtype=np.int64
    )

    # Extra safety check.
    valid_mask = np.isfinite(x_raw).all(
        axis=1
    )

    x_raw = x_raw[valid_mask]
    y = y[valid_mask]

    valid_frame = valid_frame.iloc[
        np.where(valid_mask)[0]
    ].copy()

    return x_raw, y, valid_frame


# ---------------------------------------------------------
# Normalize features
# ---------------------------------------------------------

def normalize_features(
    x_raw: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate training normalization statistics and
    standardize the feature matrix.
    """

    means = x_raw.mean(
        axis=0
    )

    stds = x_raw.std(
        axis=0
    )

    # Prevent division by zero.
    stds = np.where(
        stds == 0,
        1.0,
        stds,
    )

    x_normalized = (
        x_raw - means
    ) / stds

    if not np.isfinite(
        x_normalized
    ).all():
        raise RuntimeError(
            "Normalized training features contain "
            "invalid values."
        )

    return (
        x_normalized.astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
    )


# ---------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print(" BAYESIAN REGIME DETECTION ENGINE — LIVE MODEL TRAINING")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Download market data
    # -----------------------------------------------------

    data = download_market_data()

    print(
        f"Historical observations: {len(data)}"
    )

    # -----------------------------------------------------
    # 2. Build features
    # -----------------------------------------------------

    print("\nBuilding market features...")

    frame = build_features(data)

    print(
        f"Feature observations: {len(frame)}"
    )

    # -----------------------------------------------------
    # 3. Prepare training dataset
    # -----------------------------------------------------

    print("\nPreparing training dataset...")

    x_raw, y, training_frame = (
        prepare_training_data(frame)
    )

    x, feature_means, feature_stds = (
        normalize_features(x_raw)
    )

    print(
        f"Training samples : {x.shape[0]}"
    )

    print(
        f"Input features   : {x.shape[1]}"
    )

    print(
        f"Number of regimes: {len(REGIME_NAMES)}"
    )

    print("\nRegime distribution:")

    for regime_id, regime_name in enumerate(
        REGIME_NAMES
    ):
        count = int(
            (y == regime_id).sum()
        )

        print(
            f"  {regime_name:<15}: {count}"
        )

    # -----------------------------------------------------
    # 4. Configure model
    # -----------------------------------------------------

    torch.manual_seed(42)

    config = BayesianModelConfig(
        model_name="live_bayesian_regime_model",
        n_features=len(FEATURE_COLUMNS),
        n_outputs=len(REGIME_NAMES),
        n_regimes=len(REGIME_NAMES),
        hidden_dims=(32, 16),
        dropout_rate=0.20,
        mc_samples=50,
        random_seed=42,
    )

    # -----------------------------------------------------
    # 5. Create model
    # -----------------------------------------------------

    model = BayesianRegimeModel(
        input_dim=len(FEATURE_COLUMNS),
        config=config,
    )

    # -----------------------------------------------------
    # 6. Convert data to tensors
    # -----------------------------------------------------

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # 7. Train model
    # -----------------------------------------------------

    print(
        "\nTraining Bayesian regime model..."
    )

    model.fit(
        x=x_tensor,
        y=y_tensor,
        epochs=250,
        learning_rate=0.005,
    )

    print("Training complete.")

    # -----------------------------------------------------
    # 8. Save checkpoint
    # -----------------------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.network.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "regime_names": REGIME_NAMES,
        "config": {
            "model_name": config.model_name,
            "n_features": config.n_features,
            "n_outputs": config.n_outputs,
            "n_regimes": config.n_regimes,
            "hidden_dims": tuple(config.hidden_dims),
            "dropout_rate": config.dropout_rate,
            "mc_samples": config.mc_samples,
            "random_seed": config.random_seed,
        },
        "training_samples": int(x.shape[0]),
        "latest_training_date": str(
            training_frame.index[-1].date()
        ),
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    print("\nModel saved successfully:")
    print(
        CHECKPOINT_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        "\nLatest training market date: "
        f"{training_frame.index[-1].date()}"
    )


if __name__ == "__main__":
    main()