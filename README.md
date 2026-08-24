# Bayesian Regime Detection Engine for Equity Direction Forecasting

A quantitative research and decision-support system for identifying
market regimes in Indian equities using probabilistic, time-aware
models.

> **Direction over price. Calibrated probabilities over unsupported
> certainty. Complementary models over a single black box.**

------------------------------------------------------------------------

## Overview

This project addresses a practical problem in quantitative investing:
financial markets do not operate under one stable set of conditions.

A model that works during a broad market expansion can behave very
differently during a liquidity shock, a volatility spike, persistent
foreign outflows, or a post-crisis recovery. Instead of trying to
produce an exact price target, this project focuses on identifying the
**current market regime**, estimating the probability of alternative
regimes, and communicating how uncertain that assessment is.

The engine is designed around the Indian equity-market context and
combines market, volatility, breadth, institutional-flow, currency,
yield, and macroeconomic information.

The intended output is not an automated trading instruction. It is a
structured input to an investment process, supporting activities such
as:

-   tactical allocation tilts,
-   cash-buffer sizing,
-   sleeve weighting,
-   market-risk assessment,
-   regime-aware backtesting and scenario analysis.

The project is designed with the expectations of a serious quantitative
workflow in mind: reproducibility, point-in-time data discipline,
calibration, validation, explainability, model diagnostics, and
traceability.

------------------------------------------------------------------------

## The Problem

Point forecasting is attractive because it produces a simple answer:

> "Where will the market be?"

The problem is that multi-period financial price forecasts are noisy,
relationships are non-stationary, and a precise-looking forecast can
imply more confidence than the evidence justifies.

This project takes a different approach.

Instead of asking only:

``` text
What will the NIFTY price be?
```

the system asks:

``` text
What kind of market environment are we currently in?

How likely is each possible regime?

Is the system confident, or are the signals ambiguous?

Are the underlying models in agreement?

Can the assessment be reconstructed and defended later?
```

That distinction drives the architecture of the entire repository.

------------------------------------------------------------------------

## Market Regimes

The project uses a five-state regime taxonomy for Indian equity markets.

  -----------------------------------------------------------------------
  Regime                              Interpretation
  ----------------------------------- -----------------------------------
  **Risk-On**                         Broadly constructive market
                                      conditions with positive
                                      participation and supportive risk
                                      appetite

  **Risk-Off**                        Defensive or stressed conditions,
                                      usually associated with drawdowns,
                                      elevated volatility, and
                                      deteriorating risk signals

  **Transitional**                    Conflicting evidence or a regime
                                      boundary where no single state has
                                      strong dominance

  **Late-Cycle**                      Continued expansion with signs of
                                      narrowing leadership, stretched
                                      conditions, or market fatigue

  **Post-Shock**                      Stabilisation and recovery
                                      following an acute market stress
                                      event
  -----------------------------------------------------------------------

A regime call is represented as a probability distribution rather than a
hard label.

For example:

``` text
Risk-On       0.11
Risk-Off      0.06
Transitional  0.29
Late-Cycle    0.49
Post-Shock    0.05
```

Here, `Late-Cycle` is the most likely state, but the probability
assigned to `Transitional` remains important. The system should not hide
that ambiguity by pretending the classification is certain.

------------------------------------------------------------------------

## What the System Produces

At the decision-support boundary, the system is expected to produce a
standardised regime assessment containing information such as:

``` text
Regime probability distribution
        +
Dominant regime
        +
Prediction confidence / uncertainty
        +
Conformal prediction set
        +
Model agreement or disagreement
        +
Model-health indicators
        +
Relevant diagnostics and lineage
```

A representative output contract looks like:

``` json
{
  "as_of_date": "YYYY-MM-DD",
  "dominant_regime": "Late-Cycle",
  "regime_probabilities": {
    "Risk-On": 0.11,
    "Risk-Off": 0.06,
    "Transitional": 0.29,
    "Late-Cycle": 0.49,
    "Post-Shock": 0.05
  },
  "prediction_set": [
    "Late-Cycle",
    "Transitional"
  ],
  "model_health": "healthy"
}
```

The exact fields depend on the final implementation, but the principle
is fixed: **the final result must preserve probability, uncertainty, and
traceability rather than collapsing everything into one unsupported
label**.

------------------------------------------------------------------------

# Project Scope

The project covers the full path from business understanding to final
delivery.

``` text
Business Context
      ↓
Research
      ↓
Architecture
      ↓
Data Foundation
      ↓
Data Engineering
      ↓
Feature Engineering
      ↓
Financial Analysis
      ↓
Baseline Regime Modelling
      ↓
Advanced Regime Models
      ↓
Ensemble + Uncertainty
      ↓
Validation
      ↓
Decision Support
      ↓
Deployment
      ↓
Documentation
      ↓
Final Review
```

The implementation is organised into 15 phases so that advanced
modelling is built on a validated foundation rather than being
introduced prematurely.

------------------------------------------------------------------------

# Project Phases

## Phase 1 --- Business & BFSI Understanding

The first phase establishes the business context.

The project is framed around a quantitative investment workflow in an
Indian asset-management environment. The key objective is to understand
what decision the regime engine supports, who consumes the output, and
where the system must stop.

Key areas include:

-   Indian mutual-fund and asset-management context
-   Indian equity-market structure
-   user and stakeholder requirements
-   regime definitions
-   tactical allocation use cases
-   cash-buffer and sleeve-weighting context
-   governance and regulatory expectations
-   product boundaries and success criteria

**Output:** a clear product definition and decision workflow.

------------------------------------------------------------------------

## Phase 2 --- Research & Literature

Before selecting implementation approaches, the project studies the
relevant modelling and validation techniques.

Research areas include:

-   Hidden Markov Models
-   Bayesian inference
-   Bayesian Hidden Markov Models
-   regime-switching models
-   multivariate regime-switching VAR
-   Bayesian neural approaches
-   MC dropout and predictive uncertainty
-   time-series foundation models
-   sequential Monte Carlo / particle filtering
-   Bayesian Online Changepoint Detection
-   model ensembling
-   WAIC and LOO-based Bayesian comparison
-   calibration
-   conformal prediction
-   robust financial backtesting

**Output:** research-backed modelling choices and documented
assumptions.

------------------------------------------------------------------------

## Phase 3 --- Solution Architecture

The architecture separates concerns so that changes in one part of the
system do not require rewriting the entire project.

The high-level design is:

``` text
Data Sources
    ↓
Ingestion and Point-in-Time Storage
    ↓
Data Quality and Validation
    ↓
Feature Engineering
    ↓
Financial Analysis
    ↓
Regime Models
    ↓
Model Ensemble
    ↓
Uncertainty and Calibration
    ↓
Model Health
    ↓
Decision Support
    ↓
API / Reports / Deployment
```

Important architectural principles:

-   preserve point-in-time correctness,
-   keep research and production concerns separate,
-   make model families replaceable,
-   standardise model outputs,
-   retain model and data lineage,
-   support both batch processing and online inference where required.

------------------------------------------------------------------------

## Phase 4 --- Data Layer

This phase defines what data enters the system and how it is
represented.

Typical inputs include:

-   NIFTY 50
-   NIFTY Midcap 100
-   NIFTY Smallcap 100
-   India VIX
-   FII/DII flows
-   SIP data
-   USD/INR
-   government-security yields
-   credit-spread information
-   selected macroeconomic indicators
-   market breadth and related market-structure data

The data layer is responsible for more than loading files. It
establishes:

-   dataset definitions,
-   schemas,
-   source metadata,
-   timestamps,
-   provenance,
-   point-in-time handling,
-   validation expectations,
-   storage boundaries.

------------------------------------------------------------------------

## Phase 5 --- Data Engineering

The data-engineering layer turns source data into reliable inputs for
downstream analysis.

The pipeline follows the general flow:

``` text
Source
  ↓
Ingestion
  ↓
Raw Storage
  ↓
Loading / Parsing
  ↓
Standardisation
  ↓
Cleaning
  ↓
Quality Validation
  ↓
Transformation
  ↓
Preparation
  ↓
Integration
  ↓
Model-Ready Dataset
```

Important controls include:

-   missing-value handling,
-   duplicate detection,
-   timestamp validation,
-   chronological ordering,
-   unsupported-data detection,
-   range and consistency checks,
-   explicit failure behaviour.

The goal is to ensure that downstream models do not silently operate on
invalid data.

------------------------------------------------------------------------

## Phase 6 --- Feature Engineering

Raw prices alone do not provide enough context for regime detection.

The feature layer converts validated observations into economically
meaningful variables.

Feature groups include:

``` text
Returns
Price and trend
Momentum
Volatility
Volume
Calendar
Technical indicators
Cross-asset relationships
Macro indicators
Market breadth
Capital flows
```

The wider project design also considers specialised features such as:

-   relative large/mid/small-cap behaviour,
-   FII/DII z-scores,
-   SIP momentum,
-   flow-balance measures,
-   real-rate proxies,
-   currency-stress measures,
-   rolling correlation structure,
-   topological features,
-   sector-relationship embeddings.

Feature engineering follows the same discipline as the rest of the
project: a feature should have a documented definition and a defensible
reason for inclusion.

------------------------------------------------------------------------

## Phase 7 --- Financial Analysis

The purpose of this phase is to understand the behaviour represented by
the engineered features before treating them as model inputs.

Analysis includes relationships between:

-   returns and volatility,
-   market breadth and index direction,
-   large-, mid-, and small-cap performance,
-   institutional flows,
-   currency conditions,
-   yields and rates,
-   macroeconomic conditions,
-   correlation and market structure.

This phase provides the financial context needed to interpret later
model outputs.

------------------------------------------------------------------------

## Phase 8 --- Baseline Regime Engine

The project starts modelling with a simpler baseline before moving to
more computationally expensive approaches.

The baseline includes a frequentist Hidden Markov Model and supporting
regime analysis.

The typical workflow is:

``` text
Validated Features
      ↓
Chronological Training
      ↓
HMM Training
      ↓
State Probabilities
      ↓
State Interpretation
      ↓
Regime Labelling
      ↓
Transition and Duration Analysis
      ↓
Evaluation
```

The baseline examines:

-   regime probability paths,
-   state persistence,
-   transition matrices,
-   regime duration,
-   stickiness,
-   comparison across candidate state counts.

The five target states are interpreted post hoc in terms of the project
taxonomy rather than assuming that a latent-state model automatically
knows business labels.

------------------------------------------------------------------------

## Phase 9 --- Advanced Models

The advanced modelling phase evaluates complementary model families.

The intended model stack includes:

### Bayesian HMM

Extends latent-state modelling with posterior uncertainty and transition
uncertainty.

### Regime-Switching VAR

Captures multivariate relationships across variables such as:

-   returns,
-   volatility,
-   breadth,
-   capital flows,
-   INR,
-   yields.

### Bayesian Neural Approaches

Used to capture nonlinear relationships while estimating predictive
uncertainty through approaches such as:

-   MC dropout,
-   variational methods,
-   deep ensembles.

### Time-Series Foundation Models

Foundation models are evaluated as temporal representation or
forecasting components rather than assumed to be native regime
classifiers.

The project scope includes Chronos and comparison with at least one
additional model from the approved foundation-model family, such as
TimesFM, Lag-Llama, or Moirai.

A practical hybrid pattern is:

``` text
Market Feature Window
        ↓
Foundation Model Representation
        ↓
Bayesian / Probabilistic Classification Head
        ↓
Regime Probabilities + Uncertainty
```

### Sequential Inference

Particle filtering supports sequential state tracking, while Bayesian
Online Changepoint Detection helps identify discontinuities that smooth
regime models may react to slowly.

------------------------------------------------------------------------

## Phase 10 --- Ensemble & Uncertainty

No individual model is assumed to be universally correct.

Different models can capture different properties of the market, so
their outputs are combined through a documented ensemble layer.

``` text
HMM ───────────────┐
Bayesian HMM ──────┤
RS-VAR ────────────┤
Neural Model ──────┼──→ Ensemble
Foundation Model ──┤
Sequential Signals ─┘
                   ↓
          Combined Regime Distribution
```

The project evaluates approaches such as:

-   Bayesian Model Averaging,
-   constrained stacking,
-   Bayesian model comparison using WAIC / LOO,
-   out-of-sample probability performance.

The ensemble must preserve the evidence needed to understand:

-   which models contributed,
-   how strongly they contributed,
-   whether models agree,
-   whether disagreement is itself increasing uncertainty.

### Calibration and Conformal Prediction

A model that outputs `0.75` should be evaluated on whether predictions
made at approximately that confidence level are empirically reliable.

The uncertainty layer therefore includes:

-   probability calibration,
-   reliability analysis,
-   expected calibration error,
-   proper probabilistic scoring,
-   conformal prediction,
-   prediction-set coverage monitoring.

The objective is not to manufacture certainty. When the evidence is
ambiguous, the output should be allowed to remain ambiguous.

------------------------------------------------------------------------

## Phase 11 --- Validation

Financial time series cannot be validated like ordinary IID datasets.

Random splitting can leak information across time and produce misleading
performance.

The validation strategy therefore focuses on:

-   chronological splits,
-   walk-forward evaluation,
-   purging where required,
-   embargoing where required,
-   leakage prevention,
-   out-of-sample testing,
-   stress-period analysis,
-   sub-period analysis,
-   calibration testing,
-   proper scoring rules.

Important measures include, where applicable:

-   log loss,
-   Brier score,
-   Ranked Probability Score,
-   calibration diagnostics,
-   transition stability,
-   regime-duration behaviour,
-   stress-period performance,
-   model comparison,
-   ensemble performance.

The project also validates whether the ensemble genuinely improves on
its components rather than assuming that combining models automatically
creates a better result.

------------------------------------------------------------------------

## Phase 12 --- Decision Support Layer

A regime probability becomes useful only when it can inform a decision.

The decision-support workflow is:

``` text
Observe Market
      ↓
Generate Features
      ↓
Run Regime Models
      ↓
Combine Predictions
      ↓
Quantify Uncertainty
      ↓
Calibrate
      ↓
Check Model Health
      ↓
Explain
      ↓
Human Review
      ↓
Apply Portfolio Constraints
      ↓
Decision
      ↓
Monitor and Update
```

The intended use cases include:

-   tactical allocation tilts,
-   cash-buffer sizing,
-   sleeve weighting,
-   regime-aware risk assessment,
-   scenario analysis,
-   Investment Committee communication.

The final investment decision remains with the responsible human
process.

------------------------------------------------------------------------

## Phase 13 --- Deployment

Deployment packages the validated components for repeatable use.

The deployment boundary covers:

-   inference packaging,
-   configuration,
-   API exposure where implemented,
-   logging,
-   model metadata,
-   data/model version tracking,
-   operational checks,
-   online/batch reconciliation.

The objective is not merely to make the code executable. A
production-facing regime call should be traceable to the relevant model
version, input snapshot, and configuration.

------------------------------------------------------------------------

## Phase 14 --- Documentation

Documentation is treated as part of the engineering work.

The project documentation covers:

-   product requirements,
-   architecture,
-   phase definitions,
-   modelling assumptions,
-   data definitions,
-   feature definitions,
-   validation methodology,
-   model cards,
-   calibration evidence,
-   known limitations,
-   operational and governance context.

A complex quantitative model is difficult to trust if nobody can explain
how it was built, evaluated, or reproduced.

------------------------------------------------------------------------

## Phase 15 --- Finalization

The final phase focuses on integration and delivery.

Typical checks include:

``` text
End-to-end execution
Integration testing
Validation review
Documentation consistency
Code cleanup
Reproducibility checks
Results review
Demonstration readiness
Final deliverables
```

Completion means that implementation, tests, results, and documentation
tell the same story.

------------------------------------------------------------------------

# Repository Structure

The repository is organised by responsibility rather than by a single
monolithic script.

``` text
bayesian-regime-detection-engine/
│
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── prepared/
│
├── docs/
│   ├── research/
│   ├── decisions/
│   ├── models/
│   └── reports/
│
├── notebooks/
│   ├── exploration/
│   ├── analysis/
│   └── experiments/
│
├── reports/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── financial_analysis/
│   ├── models/
│   │   ├── baseline/
│   │   ├── hmm/
│   │   ├── bayesian_hmm/
│   │   ├── rs_var/
│   │   ├── neural/
│   │   ├── foundation/
│   │   ├── smc/
│   │   └── changepoint/
│   ├── ensemble/
│   ├── uncertainty/
│   ├── validation/
│   ├── decision_support/
│   ├── monitoring/
│   ├── api/
│   └── utils/
│
├── tests/
│
├── README.md
├── requirements.txt
└── pyproject.toml
```

The exact file layout can evolve, but the design principle should remain
the same: **data handling, feature generation, financial analysis,
modelling, validation, and deployment should remain independently
understandable and testable**.

------------------------------------------------------------------------

# Technology Stack

The project is primarily Python-based.

Core tools used across the project include:

``` text
Python
NumPy
Pandas
SciPy
scikit-learn
pytest
```

Modelling and statistical components may include:

``` text
hmmlearn
statsmodels
PyMC
NumPyro
ArviZ
```

Deep-learning and probabilistic components may include:

``` text
PyTorch
TensorFlow
TensorFlow Probability
```

Specialised components may include:

``` text
Chronos
TimesFM / Lag-Llama / Moirai
SHAP
MAPIE / CREPES
FilterPy
ruptures
TDA libraries
```

The project also includes cross-language validation requirements in its
broader deliverables, with selected regime and calibration checks
implemented or reconciled against R tooling.

------------------------------------------------------------------------

# Getting Started

## Prerequisites

Use the Python version specified by the repository configuration.

Before installing dependencies, check the project files:

``` bash
python --version
```

``` bash
pip --version
```

If `pyproject.toml` or other environment files specify a Python version,
use that version rather than assuming a generic runtime.

------------------------------------------------------------------------

## 1. Clone the Repository

``` bash
git clone <repository-url>
cd bayesian-regime-detection-engine
```

------------------------------------------------------------------------

## 2. Create a Virtual Environment

### Windows

``` powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## 3. Upgrade pip

``` bash
python -m pip install --upgrade pip
```

------------------------------------------------------------------------

## 4. Install Dependencies

If the repository uses `requirements.txt`:

``` bash
pip install -r requirements.txt
```

If the repository is configured as a Python package through
`pyproject.toml`, install it using the package workflow defined there.

For example:

``` bash
pip install -e .
```

Do not install dependencies from memory or from this README if the
repository configuration defines a different source of truth.

------------------------------------------------------------------------

## 5. Verify the Environment

``` bash
python --version
```

``` bash
python -m pytest --version
```

Then run the available test suite:

``` bash
python -m pytest -v
```

------------------------------------------------------------------------

# How to Run the Project

This repository should be run in layers.

Do not treat the project as a single opaque command unless a dedicated
orchestration entry point has been implemented.

The normal engineering workflow is:

``` text
1. Configure the environment
        ↓
2. Validate source data
        ↓
3. Run ingestion and preparation
        ↓
4. Generate features
        ↓
5. Run financial analysis
        ↓
6. Train / evaluate baseline models
        ↓
7. Run advanced models
        ↓
8. Combine model outputs
        ↓
9. Evaluate uncertainty and calibration
        ↓
10. Validate the complete pipeline
        ↓
11. Generate decision-support output
        ↓
12. Run the deployment interface
```

This sequence is deliberate. If something fails, the layer where it
failed should be identifiable.

------------------------------------------------------------------------

# Running Tests

The project uses `pytest`.

Run the complete suite:

``` bash
python -m pytest -v
```

Run all tests under the test directory:

``` bash
python -m pytest tests -v
```

Run a specific test file:

``` bash
python -m pytest tests/<test_file>.py -v
```

Run a phase-specific group:

``` bash
python -m pytest tests/test_phase6_*.py -v
```

For the implemented phase suites, use the exact test naming convention
present in the repository.

For example, Phase 5 validation is grouped by pipeline responsibility:

``` powershell
python -m pytest `
  tests/test_phase5_group_a_ingestion.py `
  tests/test_phase5_group_b_loading.py `
  tests/test_phase5_group_c_preparation.py `
  tests/test_phase5_group_d_integration.py `
  tests/test_phase5_group_e_final_validation.py `
  -v
```

On a shell that does not support PowerShell line continuation, place the
command on one line.

------------------------------------------------------------------------

# Recommended Development Workflow

For local development, the recommended approach is:

### Step 1 --- Start with tests

Before changing a module:

``` bash
python -m pytest tests/<relevant_test>.py -v
```

### Step 2 --- Make a focused change

Keep changes scoped to one concern:

``` text
data
features
financial analysis
model
ensemble
validation
deployment
```

### Step 3 --- Run the affected tests

``` bash
python -m pytest tests/<relevant_test>.py -v
```

### Step 4 --- Run the broader phase suite

``` bash
python -m pytest tests/test_phase<phase>_*.py -v
```

### Step 5 --- Commit the verified increment

``` bash
git status
git add .
git commit -m "phase-x: describe verified change"
git push
```

This is preferable to making a large number of unrelated changes and
debugging everything at the end.

------------------------------------------------------------------------

# Data Pipeline

The expected pipeline is:

``` text
Dataset Registry
      ↓
Ingestion
      ↓
Raw Data
      ↓
Load / Parse
      ↓
Clean / Standardise
      ↓
Validate
      ↓
Transform
      ↓
Prepare
      ↓
Integrate
      ↓
Feature-Ready Dataset
```

The data layer should preserve the distinction between:

``` text
Raw data
    ≠
Processed data
    ≠
Prepared / model-ready data
```

This separation makes debugging and lineage significantly easier.

A historical model evaluation must use the information that would
actually have been available at that point in time. That is a hard
requirement, not a formatting preference.

------------------------------------------------------------------------

# Feature Pipeline

Feature generation starts from validated and prepared data.

``` text
Prepared Data
      ↓
Feature Generation
      ↓
Feature Validation
      ↓
Feature Dataset
      ↓
Analysis / Modelling
```

Feature groups should be individually testable where practical.

Examples:

``` text
Returns
Price
Momentum
Volatility
Volume
Calendar
Technical
Cross-Asset
Macro
```

Regime-specific extensions can then be added without tightly coupling
them to ingestion or model code.

------------------------------------------------------------------------

# Modelling Workflow

The modelling workflow follows a progression from simple to complex.

``` text
Validated Features
        ↓
Baseline HMM
        ↓
Evaluate and Interpret
        ↓
Bayesian / Multivariate Models
        ↓
Neural Models
        ↓
Foundation Model Components
        ↓
Sequential Inference
        ↓
Out-of-Sample Comparison
```

Advanced models are not accepted merely because they are more
sophisticated.

Each model must justify its place through evidence such as:

-   probabilistic performance,
-   calibration,
-   robustness,
-   stability,
-   complementary behaviour,
-   usefulness to the ensemble.

------------------------------------------------------------------------

# Ensemble Workflow

Model outputs should be standardised before combination.

Conceptually:

``` text
Model Output
├── probabilities
├── uncertainty
├── diagnostics
├── model version
└── inference metadata
```

The ensemble receives these outputs and produces:

``` text
Combined probabilities
Model contribution / weights
Model disagreement
Ensemble diagnostics
```

The final combined output is then passed to uncertainty, calibration,
and decision-support components.

------------------------------------------------------------------------

# Validation Principles

The following rules are non-negotiable for the project.

## No Random Train/Test Splits for Time-Series Evaluation

Financial observations are temporally dependent.

Use chronological and walk-forward evaluation where appropriate.

## No Look-Ahead Bias

Historical predictions must not use future information, including data
revisions that were unavailable at the time.

## Calibration Matters

A probability forecast is not useful simply because it is numerically
high or low. It must be evaluated against realised outcomes.

## Accuracy Alone Is Not Enough

Regime models should be assessed using proper probabilistic metrics and
calibration diagnostics.

## Stress Periods Matter

The model should be inspected across materially different market
conditions rather than evaluated only on average performance.

## Ensemble Improvement Must Be Demonstrated

The ensemble should not be assumed to outperform its components. That
claim must be tested.

------------------------------------------------------------------------

# Decision-Support Boundary

The engine is a **decision-support system**.

Its intended path is:

``` text
Market Evidence
      ↓
Regime Assessment
      ↓
Probability Distribution
      ↓
Uncertainty
      ↓
Calibration
      ↓
Model Health
      ↓
Explanation
      ↓
Human Review
      ↓
Decision Within Constraints
```

It is explicitly not:

``` text
Prediction
   ↓
Automatic Trade
```

The output can support a portfolio or investment process, but it does
not remove the need for professional judgement, portfolio constraints,
risk controls, or governance.

------------------------------------------------------------------------

# Model Health and Traceability

A regime probability should not be considered sufficient on its own.

The broader system should be able to answer:

``` text
Which data snapshot was used?

Which model version generated the result?

Which configuration was active?

Were there data-quality issues?

Did the models agree?

Was a changepoint detected?

Is the current observation outside the expected training distribution?

How different is online inference from batch inference?

Can the historical call be reconstructed?
```

If these questions cannot be answered, the output may still be
mathematically generated, but it is not yet operationally defensible.

------------------------------------------------------------------------

# Expected Final Deliverables

The wider project brief includes deliverables beyond the core Python
implementation.

These include:

1.  **Main Report**
    -   theory,
    -   methodology,
    -   results,
    -   case studies,
    -   calibration evidence,
    -   ensemble design,
    -   regime comparison and validation.
2.  **Python Codebase**
    -   data and feature engineering,
    -   regime models,
    -   uncertainty and calibration,
    -   sequential inference,
    -   ensemble logic,
    -   simulation components.
3.  **R Codebase / Cross-Language Validation**
    -   selected HMM and Markov-switching implementations,
    -   Bayesian or changepoint components,
    -   calibration checks,
    -   reconciliation against Python outputs.
4.  **Backtesting and Simulation Engine**
    -   regime-aware allocation overlay,
    -   regime-conditioned simulation,
    -   risk measures,
    -   historical scenario replay,
    -   Investment Committee artefacts.
5.  **Model Card, Calibration and Validation Pack**
    -   model definitions,
    -   inputs and assumptions,
    -   diagnostics,
    -   calibration evidence,
    -   coverage monitoring,
    -   reconciliation checks.
6.  **Final Presentation and Demonstration**
    -   methodology,
    -   results,
    -   validation evidence,
    -   ensemble design,
    -   live or replayed end-to-end demonstration.

------------------------------------------------------------------------

# Engineering Standards

## Data

-   Keep raw inputs immutable where possible.
-   Preserve source and timestamp metadata.
-   Validate before downstream use.
-   Separate source handling from business logic.
-   Protect against look-ahead bias.

## Features

-   Define features explicitly.
-   Avoid undocumented transformations.
-   Keep training and inference definitions consistent.
-   Document economic rationale for material features.

## Models

-   Keep models modular.
-   Preserve diagnostics.
-   Record assumptions and versions.
-   Evaluate out of sample.
-   Do not hide model failures behind fallback outputs.

## Validation

-   Use time-aware methodology.
-   Evaluate probabilities, not only labels.
-   Test stress periods.
-   Keep final evaluation separate from repeated tuning.

## Deployment

-   Fail clearly when critical inputs are invalid.
-   Preserve logs and metadata.
-   Keep configuration explicit.
-   Make historical outputs reconstructable.

------------------------------------------------------------------------

# What This Project Does Not Claim

This repository does not claim to:

-   predict the exact future market price with certainty,
-   guarantee investment returns,
-   replace a portfolio manager or Investment Committee,
-   eliminate financial risk,
-   make a complex model trustworthy merely because it uses Bayesian or
    deep-learning terminology.

The value of the system comes from disciplined probabilistic modelling,
validation, uncertainty measurement, and clear integration into a human
decision process.

------------------------------------------------------------------------

# Contributing

Before making changes:

1.  Read the relevant project documentation.
2.  Understand the module boundary you are changing.
3.  Check the existing tests.
4.  Make a focused change.
5.  Run the affected tests.
6.  Run the broader phase suite.
7.  Update documentation when behaviour or assumptions change.

A clean quantitative codebase is not defined by the number of models it
contains. It is defined by whether another engineer can understand what
each component does, reproduce the results, and identify where an output
came from.

------------------------------------------------------------------------

# Disclaimer

This project is intended for research, educational, and decision-support
purposes.

It is not financial advice and does not guarantee future market
behaviour or investment performance.

All model outputs should be interpreted together with appropriate risk
controls, portfolio constraints, independent validation, and responsible
human judgement.
