# Bayesian Regime Detection Engine for Equity Direction Forecasting

> A modular, uncertainty-aware financial machine learning system for identifying changing market regimes in Indian equities.

**Project 1A | Zetheta Algorithms Private Limited**

---

## Overview

Financial markets do not operate under a single stable environment.

A model that performs well during a broad market rally may behave very differently during a volatility shock, market drawdown, transitional period, or post-crisis recovery. Instead of attempting to predict an exact future market price, this project focuses on identifying the **current market regime** and representing the result probabilistically.

The Bayesian Regime Detection Engine is designed to answer:

- What market environment is currently being observed?
- Which regime is most likely?
- How probable are the alternative regimes?
- How confident is the model?
- How uncertain is the prediction?
- Can the result be traced back to its data, features, model, and execution context?

The project is designed as a **research and decision-support system** for the Indian equity market.

> **Direction over price. Probability over unsupported certainty. Complementary models over a single black box.**

---

## Key Features

- Five-state Indian equity market regime framework
- Historical market-data ingestion and preparation
- Modular financial feature engineering
- Financial analysis and risk components
- Interpretable baseline regime models
- Bayesian neural regime classification
- Monte Carlo Dropout-based predictive uncertainty
- Ensemble and uncertainty architecture
- Calibration and validation components
- Explainability layer
- Decision-support abstractions
- Monitoring and model-health foundations
- Integration and orchestration components
- Experiment tracking and model versioning support
- Modular architecture with dedicated test coverage

---

## Market Regimes

The project uses a five-state regime taxonomy for Indian equity markets.

| Regime | Description |
|---|---|
| **RISK_ON** | Constructive market environment with positive trend, supportive participation and stronger risk appetite |
| **LATE_CYCLE** | Mature expansion with possible signs of fatigue, narrow leadership or stretched conditions |
| **TRANSITIONAL** | Conflicting signals or movement between broader market states |
| **POST_SHOCK** | Stabilisation or recovery following a period of elevated market stress |
| **RISK_OFF** | Defensive or stressed environment with deteriorating risk appetite |

The objective is not simply to produce a hard label.

A regime assessment is intended to preserve the complete probability distribution:

```text
RISK_ON        0.11
LATE_CYCLE     0.49
TRANSITIONAL   0.29
POST_SHOCK     0.05
RISK_OFF       0.06
```

In this example, `LATE_CYCLE` is the dominant regime, but the probability assigned to `TRANSITIONAL` remains relevant. The system is designed to preserve that ambiguity rather than hiding it behind a single overconfident classification.

---

# System Architecture

The project follows a modular end-to-end architecture:

```text
                         Market Data
                NIFTY • VIX • Macro • Flows
                              |
                              v
                         Data Layer
                Ingestion • Quality • Preparation
                              |
                              v
                    Feature Engineering
             Returns • Momentum • Volatility • Macro
                              |
                              v
                    Financial Analysis
                Risk • Drawdown • Performance
                              |
                              v
                      Regime Model Layer
          Baselines • Bayesian • Neural • HMM • Sequential
                              |
                              v
                   Ensemble & Uncertainty
                         Calibration
                              |
                              v
                 Validation & Explainability
                              |
                              v
                  Decision Support + Health
                              |
                              v
                Monitoring • MLOps • Deployment
```

---

# Current Implementation

The repository has been developed incrementally across a broader **15-phase project roadmap**, covering:

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

The repository includes modules and architectural foundations across data processing, feature generation, analysis, modelling, ensemble logic, uncertainty estimation, explainability, validation, monitoring, integration, and MLOps.

---

# Bayesian Regime Model

One concrete model path implemented in the project uses a Bayesian-style neural classification workflow built around:

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

Monte Carlo Dropout keeps dropout active during repeated inference passes, allowing the model to generate a distribution of predictions rather than relying on one deterministic forward pass.

---

# Historical Training Workflow

A historical training workflow has been added to move beyond the original synthetic demonstration.

The current implementation downloads historical:

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

### Latest verified training run

```text
Historical observations: 4071
Valid training samples : 3295
Input features         : 8
Number of regimes      : 5
Latest training date   : 2026-08-25
```

The trained checkpoint is saved at:

```text
artifacts/live_regime_model.pt
```

---

# Current Model Features

The current historical Bayesian training workflow uses eight features:

| # | Feature | Description |
|---|---|---|
| 1 | `short_return` | Short-term market return |
| 2 | `medium_return` | Medium-term market return |
| 3 | `momentum` | Intermediate market momentum |
| 4 | `volatility` | Rolling return volatility |
| 5 | `volume_change` | Change in trading volume |
| 6 | `rsi` | Relative Strength Index |
| 7 | `vix_risk` | Change in India VIX risk signal |
| 8 | `trend_strength` | Distance from the rolling market trend |

Before training:

1. Missing values are removed.
2. Infinite and non-finite values are removed.
3. Features are standardized.
4. Normalization statistics are stored with the checkpoint.

This ensures that future inference can apply the same feature transformation used during training.

---

# Regime Distribution in Latest Training Run

The latest verified training run produced the following class distribution:

```text
RISK_ON        : 620
LATE_CYCLE     : 824
TRANSITIONAL   : 1427
POST_SHOCK     : 147
RISK_OFF       : 277
```

These labels are currently generated through transparent rule-based logic using combinations of market trend, momentum, RSI, volatility, and VIX behaviour.

They should therefore be interpreted as **supervised modelling targets based on explicit assumptions**, not as an objectively observable ground truth.

---

# Repository Structure

The project is organised by responsibility to keep individual concerns modular and testable.

```text
bayesian-regime-detection-engine/
|
+-- artifacts/                  # Saved model artifacts
|   +-- live_regime_model.pt
|
+-- configs/                    # Configuration files
|
+-- data/                       # Data storage and processing layers
|
+-- docs/                       # Research, architecture and documentation
|
+-- notebooks/                  # Exploration and experiments
|
+-- reports/                    # Generated analysis and reporting artifacts
|
+-- scripts/
|   +-- run_regime_demo.py
|   +-- train_live_regime_model.py
|
+-- src/
|   +-- analysis/               # Financial analysis
|   +-- data/                   # Data ingestion and preparation
|   +-- features/               # Feature engineering
|   +-- models/                 # Baseline and advanced models
|   |   +-- bayesian/           # Bayesian regime model
|   +-- ensemble/               # Model combination
|   +-- uncertainty/            # Confidence and uncertainty
|   +-- explainability/         # Model interpretation
|   +-- validation/             # Validation and diagnostics
|   +-- decision_support/       # Decision-support layer
|   +-- monitoring/             # Data/model monitoring
|   +-- integration/            # Pipeline integration
|   +-- mlops/                  # Versioning and experiment support
|
+-- tests/                      # Automated tests
|
+-- README.md
+-- requirements.txt
+-- pyproject.toml
```

---

# Technology Stack

### Core

- Python
- NumPy
- Pandas
- PyTorch
- yfinance
- pytest

### Broader Architecture / Research Components

Depending on the module and experiment, the project architecture also considers or supports tools and approaches involving:

- scikit-learn
- SciPy
- statsmodels
- hmmlearn
- PyMC / NumPyro
- ArviZ
- TensorFlow Probability
- sequential inference
- changepoint detection
- conformal prediction
- model explainability

Not every package is required to execute the current historical Bayesian training workflow.

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd bayesian-regime-detection-engine
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the project is configured for editable installation:

```powershell
python -m pip install -e .
```

---

# Verify the Environment

Check the primary packages used by the current historical workflow:

```powershell
python -c "import yfinance, pandas, numpy, torch; print('All required packages are installed')"
```

Expected output:

```text
All required packages are installed
```

---

# How to Run

## 1. Bayesian Synthetic Demo

The original reproducible demonstration trains the Bayesian model on synthetic regime data and evaluates a fixed demonstration observation.

```powershell
python scripts\run_regime_demo.py
```

Workflow:

```text
Synthetic Market Data
        |
        v
Bayesian Model Training
        |
        v
Fixed Market Observation
        |
        v
Monte Carlo Dropout Inference
        |
        v
Regime Probabilities + Confidence
```

Because the script uses reproducible data generation and a fixed observation, repeated runs can produce the same output.

This is expected behaviour.

---

## 2. Historical Model Training

Train the Bayesian regime model using historical NIFTY 50 and India VIX data:

```powershell
python scripts\train_live_regime_model.py
```

A successful run:

```text
Downloads historical data
        |
        v
Builds market features
        |
        v
Cleans invalid observations
        |
        v
Creates regime labels
        |
        v
Normalizes features
        |
        v
Trains Bayesian model
        |
        v
Saves checkpoint
```

Expected final output includes:

```text
Training complete.

Model saved successfully:
artifacts\live_regime_model.pt
```

---

# Testing

The project includes automated tests across major modules and development phases.

Run the complete test suite:

```powershell
python -m pytest -v
```

Run a specific test file:

```powershell
python -m pytest tests\<test_file>.py -v
```

Representative verified milestones include:

- **Phase 5 – Data Engineering:** 19 tests passed
- **Phase 8 – Baseline Regime Engine:** 106 tests passed
- Dedicated module and integration testing across subsequent architectural components

The project was developed incrementally so that module-level correctness and interface compatibility could be checked throughout implementation.

---

# Validation Philosophy

Regime detection is different from conventional supervised classification because market regimes are not directly observable in the same way as ordinary class labels.

The project therefore considers multiple forms of evaluation, including:

- chronological validation,
- walk-forward evaluation,
- leakage prevention,
- regime stability,
- probability quality,
- calibration,
- stress-period behaviour,
- model robustness,
- model agreement and disagreement,
- operational health.

Accuracy alone is not sufficient for a probabilistic financial regime model.

---

# Explainability and Decision Support

The system architecture separates:

```text
Model Output
     |
     v
Probability Distribution
     |
     v
Uncertainty Assessment
     |
     v
Validation / Model Health
     |
     v
Explainability
     |
     v
Human Review
     |
     v
Decision Within Constraints
```

The project does **not** treat a model prediction as an automatic trading instruction.

Instead, the intended decision-support output can include:

- current regime assessment,
- complete regime probability distribution,
- confidence,
- uncertainty,
- model-health status,
- explanatory drivers,
- changes from previous assessments,
- potential decision context.

This preserves the distinction between **analytical evidence** and the **final investment decision**.

---

# Monitoring and MLOps

The architecture includes components for making the project reproducible and operationally traceable.

These areas include:

- experiment tracking,
- model versioning,
- version registry support,
- deployment state,
- lifecycle management,
- data quality monitoring,
- drift detection,
- performance monitoring,
- degradation detection,
- integration and orchestration.

The goal is that a historical regime assessment can eventually be traced back to the relevant model and execution context.

---

# Development Principles

The project follows several engineering principles.

### Modular by Design

Individual responsibilities are separated across data, features, analysis, modelling, validation and deployment layers.

### Time-Aware

Financial time-series operations should respect chronological ordering and avoid look-ahead bias.

### Probability-Aware

The output should preserve alternative regime probabilities rather than only returning a hard label.

### Uncertainty-Aware

Confidence is not treated as certainty. Predictive uncertainty is an explicit part of the modelling architecture.

### Explainable

Outputs are intended to remain interpretable and traceable.

### Reproducible

Configurations, models, data handling and execution components are structured to support repeatability.

### Human-in-the-Loop

The system supports decision-making rather than autonomously executing investment actions.

---

# Current Status

### Completed and Implemented

- [x] Business and domain understanding
- [x] Research and methodology foundation
- [x] Modular solution architecture
- [x] Data-layer architecture
- [x] Data engineering components
- [x] Feature engineering framework
- [x] Financial analysis components
- [x] Baseline regime modelling
- [x] Bayesian regime model
- [x] Monte Carlo Dropout inference
- [x] Predictive uncertainty components
- [x] Ensemble and uncertainty architecture
- [x] Explainability modules
- [x] Validation framework
- [x] Decision-support architecture
- [x] Monitoring foundations
- [x] Integration/orchestration components
- [x] MLOps and model versioning foundations
- [x] Historical NIFTY 50 + India VIX training workflow
- [x] 8-feature historical training pipeline
- [x] Saved trained model checkpoint

### Next Operational Step

- [ ] Dedicated tested live-inference script for fetching the latest market observation and loading the saved checkpoint

This distinction is intentional: the repository should document only functionality that has actually been implemented and verified.

---

# Limitations

This project has important limitations.

- Financial markets can undergo structural breaks.
- Historical relationships can weaken or reverse.
- Market regimes are latent and involve modelling assumptions.
- Current supervised training labels are rule-based constructs.
- Regime boundaries can be ambiguous.
- Model confidence does not guarantee correctness.
- Historical performance does not guarantee future performance.
- A trained model can become stale as market behaviour changes.
- Data availability and revisions can affect results.
- Production investment use would require additional validation, governance and controls.

---

# Disclaimer

This project is intended for:

- research,
- education,
- quantitative experimentation,
- financial analytics,
- machine learning experimentation,
- decision-support research.

It does **not** provide financial advice and does not guarantee future market behaviour or investment performance.

Model outputs should be interpreted with appropriate validation, risk controls, portfolio constraints and responsible human judgement.

---

## Project Philosophy

> **Do not ask a model to pretend it knows the future.**
>
> **Ask what the market environment most likely is, quantify what is uncertain, and preserve enough evidence for a human to make a defensible decision.**
