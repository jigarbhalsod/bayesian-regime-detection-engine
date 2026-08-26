# Bayesian Regime Detection Engine for Equity Direction Forecasting

> A modular, uncertainty-aware financial machine learning system for identifying the latest available market regime in Indian equities.

**Project 1A | Zetheta**

---

## Overview

Financial markets do not operate under one stable environment. A model that performs well during a broad market rally may behave very differently during a volatility shock, market drawdown, transitional period, or post-shock recovery.

Instead of attempting to predict an exact future market price, this project focuses on identifying the **current/latest available market regime** and representing the result probabilistically.

The engine is designed to answer:

- What market environment is represented by the latest available data?
- Which regime is most likely?
- How probable are the alternative regimes?
- How confident is the model?
- How uncertain is the prediction?

> **Direction over price. Probability over unsupported certainty. Uncertainty-aware modelling over a single deterministic label.**

---

# Project Status

## Current Status: COMPLETE AND RUNNABLE END-TO-END

The repository was developed incrementally across a broader 15-phase roadmap covering business understanding, research, architecture, data, features, modelling, uncertainty, validation, decision support, deployment foundations, documentation, and final integration.

The final practical workflow has been verified end-to-end:

```text
Historical Market Data
        |
        v
Download NIFTY 50 + India VIX
        |
        v
Build Market Features
        |
        v
Clean Missing / Non-Finite Values
        |
        v
Create Rule-Based Regime Labels
        |
        v
Standardize Features
        |
        v
Train Bayesian Regime Model
        |
        v
Save Model Checkpoint
        |
        v
Download Latest Available Market Data
        |
        v
Build Latest Feature Vector
        |
        v
Load Trained Model
        |
        v
Monte Carlo Dropout Inference
        |
        v
Regime Probabilities + Confidence
```

---

# Quick Start

Run the complete project from the repository root:

```powershell
python scripts\train_live_regime_model.py; if ($LASTEXITCODE -eq 0) { python scripts\run_live_regime.py }
```

This single command:

1. Downloads historical NIFTY 50 data.
2. Downloads historical India VIX data.
3. Builds the eight market features.
4. Removes invalid, missing, and non-finite observations.
5. Creates transparent rule-based training labels.
6. Standardizes the training features.
7. Trains the Bayesian regime model.
8. Saves the trained checkpoint.
9. Downloads the latest available market observations.
10. Builds the latest valid feature vector.
11. Loads the trained model.
12. Runs Monte Carlo Dropout inference.
13. Prints the dominant regime, probabilities, confidence, and latest feature values.

Run it from:

```text
D:\PROJECT\Kotak Projects\bayesian-regime-detection-engine
```

---

# Latest Verified End-to-End Run

The latest verified workflow completed successfully with:

```text
Historical observations: 4511
Feature observations   : 4511
Training samples       : 3322
Input features         : 8
Number of regimes      : 5
Latest training date   : 2026-08-25
Latest market date     : 2026-08-25
```

Example verified inference:

```text
Detected Regime    : TRANSITIONAL
Regime ID          : 2
Confidence         : 0.7848

Regime Probabilities:
  RISK_ON        : 0.0919
  LATE_CYCLE     : 0.0897
  TRANSITIONAL   : 0.7848
  POST_SHOCK     : 0.0284
  RISK_OFF       : 0.0051
```

Because inference uses **Monte Carlo Dropout**, repeated runs on the same market observation can produce slightly different probability estimates. Verified confidence values included approximately:

```text
80.73%
79.64%
80.97%
78.48%
```

The dominant regime remained **TRANSITIONAL** across those runs.

---

# Important Data Terminology

The current runnable implementation uses:

> **latest-available market data**

It should not be described as guaranteed real-time tick-by-tick inference. The workflow uses the latest available daily observations returned by its data source.

---

# Market Regimes

| Regime | Description |
|---|---|
| **RISK_ON** | Constructive market environment with positive trend and stronger risk appetite |
| **LATE_CYCLE** | Mature expansion with possible signs of fatigue or stretched conditions |
| **TRANSITIONAL** | Conflicting signals or movement between broader market states |
| **POST_SHOCK** | Stabilisation or recovery following elevated market stress |
| **RISK_OFF** | Defensive or stressed environment with deteriorating risk appetite |

The output preserves the complete regime probability distribution instead of only returning a hard label.

---

# System Architecture

```text
                         Market Data
                    NIFTY 50 + India VIX
                              |
                              v
                         Data Layer
                Ingestion • Quality • Preparation
                              |
                              v
                    Feature Engineering
         Returns • Momentum • Volatility • RSI • Trend
                              |
                              v
                    Financial Analysis
                              |
                              v
                      Regime Model Layer
                Baselines • Bayesian • Advanced Models
                              |
                              v
                   Ensemble & Uncertainty
                              |
                              v
                Validation & Explainability
                              |
                              v
                 Decision Support + Monitoring
                              |
                              v
                   Integration / MLOps Foundations
```

The current verified end-to-end path uses the concrete Bayesian model workflow below.

---

# Bayesian Regime Model

The implemented Bayesian-style neural classification path uses:

- PyTorch
- Bayesian regime network architecture
- Dropout
- Monte Carlo Dropout inference
- Predictive probability estimation
- Confidence estimation
- Predictive uncertainty estimation

Conceptually:

```text
Market Features
      |
      v
Bayesian Regime Network
      |
      v
Multiple Stochastic Forward Passes
(Monte Carlo Dropout)
      |
      v
Regime Probability Distribution
      |
      +-- Dominant Regime
      +-- Confidence
      +-- Predictive Uncertainty
```

Monte Carlo Dropout keeps dropout active during repeated inference passes. This is why exact confidence values can vary slightly across repeated runs.

---

# Historical Training Workflow

The historical training workflow downloads:

- **NIFTY 50**
- **India VIX**

and constructs a model-ready dataset.

```text
Historical NIFTY 50
        +
Historical India VIX
        |
        v
    Data Alignment
        |
        v
Feature Engineering
        |
        v
Remove Missing / Non-Finite Values
        |
        v
Rule-Based Regime Labelling
        |
        v
Feature Standardization
        |
        v
Bayesian Regime Model Training
        |
        v
Save Model + Metadata
```

The trained checkpoint stores:

- model state dictionary,
- feature columns,
- feature means,
- feature standard deviations,
- regime names,
- model configuration,
- training sample count,
- latest training market date.

Generated checkpoint location:

```text
artifacts/live_regime_model.pt
```

The artifact can be recreated by running the training script.

---

# Current Model Features

The live workflow uses eight features:

| # | Feature | Description |
|---|---|---|
| 1 | `short_return` | One-period NIFTY 50 percentage return |
| 2 | `medium_return` | 20-period NIFTY 50 percentage return |
| 3 | `momentum` | 10-period NIFTY 50 percentage return |
| 4 | `volatility` | Rolling standard deviation of short-term returns |
| 5 | `volume_change` | Change in market trading volume with invalid values handled safely |
| 6 | `rsi` | Relative Strength Index |
| 7 | `vix_risk` | Five-period percentage change in India VIX |
| 8 | `trend_strength` | Distance from the rolling 20-period market trend |

Before training:

1. Missing values are removed.
2. Infinite and non-finite values are removed.
3. Features are standardized.
4. Normalization statistics are stored with the checkpoint.

Inference uses the same stored normalization statistics.

---

# Latest Verified Training Distribution

```text
RISK_ON        : 527
LATE_CYCLE     : 1179
TRANSITIONAL   : 1295
POST_SHOCK     : 44
RISK_OFF       : 277
```

Current labels are generated through transparent rule-based logic using combinations of trend, momentum, RSI, volatility, and VIX behaviour.

They should therefore be interpreted as **supervised modelling targets based on explicit assumptions**, not as perfectly objective market ground truth.

---

# Latest-Available Inference Workflow

The dedicated inference script is:

```text
scripts/run_live_regime.py
```

It performs:

```text
Latest Available NIFTY 50 Data
              +
Latest Available India VIX Data
              |
              v
       Build 8 Market Features
              |
              v
Select Latest Valid Observation
              |
              v
      Load Model Checkpoint
              |
              v
Apply Stored Standardization
              |
              v
    Monte Carlo Dropout Inference
              |
              v
Regime + Probabilities + Confidence
```

Output includes:

- Latest market date
- Model training date
- Detected regime
- Regime ID
- Confidence
- Probability for each regime
- Latest market feature values

---

# Repository Structure

```text
bayesian-regime-detection-engine/
|
+-- artifacts/                  # Generated trained model artifacts
+-- configs/                    # Configuration files
+-- data/                       # Data storage and processing layers
+-- docs/                       # Research, architecture and documentation
+-- notebooks/                  # Exploration and experiments
+-- reports/                    # Generated reports
|
+-- scripts/
|   +-- run_regime_demo.py
|   +-- train_live_regime_model.py
|   +-- run_live_regime.py
|
+-- src/
|   +-- analysis/
|   +-- data/
|   +-- features/
|   +-- models/
|   |   +-- bayesian/
|   +-- ensemble/
|   +-- uncertainty/
|   +-- explainability/
|   +-- validation/
|   +-- decision_support/
|   +-- monitoring/
|   +-- integration/
|   +-- mlops/
|
+-- tests/
+-- README.md
+-- .gitignore
```

---

# Scripts

## Full End-to-End Run

```powershell
python scripts\train_live_regime_model.py; if ($LASTEXITCODE -eq 0) { python scripts\run_live_regime.py }
```

## Train Only

```powershell
python scripts\train_live_regime_model.py
```

## Latest-Available Inference Only

```powershell
python scripts\run_live_regime.py
```

This requires:

```text
artifacts/live_regime_model.pt
```

If the checkpoint does not exist, run training first.

## Original Synthetic Demo

```powershell
python scripts\run_regime_demo.py
```

This is separate from the real historical-data training workflow.

---

# Example Successful Output

```text
BAYESIAN REGIME DETECTION ENGINE — LIVE MODEL TRAINING

Downloading NIFTY 50 historical data...
Downloading India VIX historical data...
Building market features...
Preparing training dataset...
Training Bayesian regime model...
Training complete.

Model saved successfully:
artifacts\live_regime_model.pt

BAYESIAN REGIME DETECTION ENGINE — LIVE INFERENCE

Latest Market Date : YYYY-MM-DD
Model Training Date: YYYY-MM-DD

Detected Regime    : TRANSITIONAL
Regime ID          : 2
Confidence         : 0.XXXX

Regime Probabilities:
  RISK_ON        : 0.XXXX
  LATE_CYCLE     : 0.XXXX
  TRANSITIONAL   : 0.XXXX
  POST_SHOCK     : 0.XXXX
  RISK_OFF       : 0.XXXX
```

Actual values depend on the latest available market data and stochastic Monte Carlo inference.

---

# Technology Stack

- Python
- PyTorch
- NumPy
- Pandas
- yfinance
- Monte Carlo Dropout
- Git / GitHub

---

# Environment Check

```powershell
python -c "import yfinance, pandas, numpy, torch; print('All required packages are installed')"
```

The verified workflow was run with Python 3.11.

---

# 15-Phase Development Roadmap

1. Business & BFSI Understanding
2. Research & Literature
3. Solution Architecture
4. Data Layer
5. Data Engineering
6. Feature Engineering
7. Financial Analysis
8. Baseline Regime Engine
9. Advanced Models
10. Ensemble & Uncertainty
11. Validation
12. Decision Support
13. Deployment
14. Documentation
15. Final Integration & Review

The repository contains modular components and architectural foundations developed across these phases.

The final practical integration focused on making the project demonstrable through a complete training-to-inference workflow using historical Indian market data and the latest available daily observation.

---

# Development Principles

### Modular by Design
Responsibilities are separated across data, features, analysis, modelling, validation, uncertainty, explainability, monitoring, and integration.

### Time-Aware
Financial time-series processing should respect chronological ordering and avoid look-ahead bias.

### Probability-Aware
The output preserves alternative regime probabilities rather than only returning a hard label.

### Uncertainty-Aware
Confidence is not treated as certainty.

### Explainable
The broader architecture includes components intended to keep outputs interpretable and traceable.

### Reproducible
Configuration, feature normalization metadata, model artifacts, and scripts support repeatable workflows.

### Human-in-the-Loop
The system is a decision-support and research tool, not an autonomous trading system.

---

# Current Implementation Summary

- [x] Historical NIFTY 50 ingestion
- [x] Historical India VIX ingestion
- [x] Data alignment
- [x] Eight-feature engineering pipeline
- [x] Missing-value handling
- [x] Non-finite value handling
- [x] Safe handling of invalid latest volume observations
- [x] Rule-based regime labelling
- [x] Five-regime framework
- [x] Feature standardization
- [x] Bayesian neural regime model
- [x] Monte Carlo Dropout inference
- [x] Probability output
- [x] Confidence estimation
- [x] Model checkpoint persistence
- [x] Checkpoint metadata
- [x] Latest-available inference script
- [x] Single-command end-to-end execution
- [x] Repeated successful end-to-end verification

---

# Limitations

## Latest Available Is Not Guaranteed Real-Time

The current pipeline uses the latest available daily observations returned by its data source. It should not be described as guaranteed tick-level or real-time market inference.

## Rule-Based Training Labels

The current supervised targets are generated from explicit rules and selected market signals. They are modelling assumptions rather than perfectly objective ground truth.

## Current Verified Live Path Uses Selected Inputs

The verified end-to-end path currently uses NIFTY 50 and India VIX as primary inputs. Broader repository architecture may support additional components, but they should not automatically be claimed as part of the same verified live path unless separately executed and validated.

## Confidence Is Not Certainty

A high probability does not guarantee a correct regime interpretation.

## Not Financial Advice

This project is for educational, research, and decision-support purposes. It does not provide investment advice or automatically execute trades.

---

# Resume / Portfolio Description

> **Developed an end-to-end Bayesian Regime Detection Engine that trains on historical NIFTY 50 and India VIX data, engineers market features, classifies market conditions into five regimes, and performs latest-available market regime inference using Monte Carlo Dropout-based probabilistic uncertainty estimation.**

Short version:

> **Built a Bayesian market regime detection system for Indian equities using NIFTY 50 and India VIX data, with five-state classification and Monte Carlo Dropout-based uncertainty-aware inference.**

---

# Final Demonstration

Run:

```powershell
python scripts\train_live_regime_model.py; if ($LASTEXITCODE -eq 0) { python scripts\run_live_regime.py }
```

A successful run demonstrates:

```text
Real Historical Market Data
        +
Feature Engineering
        +
Bayesian Model Training
        +
Model Persistence
        +
Latest-Available Data Retrieval
        +
Model Loading
        +
Monte Carlo Dropout Inference
        =
End-to-End Regime Detection Workflow
```

---

## Project 1A — Final Status

**Status:** Complete and runnable end-to-end  
**Primary data path:** Historical NIFTY 50 + India VIX  
**Inference mode:** Latest available daily market observation  
**Regimes:** 5  
**Features:** 8  
**Model:** Bayesian-style neural classifier with Monte Carlo Dropout inference  
**Output:** Regime probabilities, dominant regime, confidence, and latest feature values

---

**Project 1A | Bayesian Regime Detection Engine | Zetheta**
