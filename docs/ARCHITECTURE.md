PROJECT 1A
SOLUTION ARCHITECTURE
Bayesian Regime Detection Engine for Equity Direction Forecasting
Architecture Baseline, Version 1.0

## 1. Purpose of This Document
This document describes the proposed technical architecture for Project 1A. Its purpose is to translate the product requirements and decision workflow into a modular system that can be built, tested, changed and maintained without tightly coupling every part of the project.
This is an architecture baseline rather than a final implementation contract. Some technology choices, model implementations and infrastructure decisions should remain open until the research and validation phases provide enough evidence to make them confidently.
## 2. Architecture Goals
Keep the data, modelling, uncertainty, validation and decision-support layers clearly separated.
Make individual model families replaceable without rebuilding the entire system.
Preserve point-in-time data discipline throughout the pipeline.
Make every regime prediction traceable to its inputs and model versions.
Support both nightly batch processing and intraday online inference.
Keep research experiments separate from production-ready components.
Make model health and calibration visible to downstream decision-support components.
Allow the system to grow from a research prototype into a deployable decision-support platform.
## 3. High-Level Architecture
The proposed system can be viewed as a sequence of connected layers:
Data Sources
    ↓
Data Ingestion and Point-in-Time Store
    ↓
Data Quality and Validation
    ↓
Feature Engineering
    ↓
Financial Analysis and Regime Features
    ↓
Regime Model Layer
    ↓
Ensemble Layer
    ↓
Uncertainty and Calibration Layer
    ↓
Model Health and Monitoring
    ↓
Decision Support Layer
    ↓
API / Dashboard / Reports
## 4. Architecture Layers
### 4.1 Data Source Layer
Provides market and macroeconomic inputs required by the regime engine.
NIFTY indices
India VIX
FII/DII/SIP flows
Market breadth
USD/INR
Government-security yields
Credit spreads
Selected macroeconomic indicators
### 4.2 Data Ingestion and Storage
Collects, normalizes and stores source data while preserving timestamps and provenance.
Raw data storage
Normalized datasets
Point-in-time snapshots
Source metadata
Data versioning
### 4.3 Data Quality Layer
Checks whether incoming data is complete, valid and suitable for downstream use.
Missing-value checks
Duplicate checks
Timestamp validation
Range checks
Source consistency
Revision awareness
### 4.4 Feature Engineering Layer
Transforms market observations into economically meaningful features.
Returns
Trend and momentum
Volatility
Breadth
Capital flows
Macro indicators
Cross-asset relationships
### 4.5 Financial Analysis Layer
Provides financial context and derived measures used by the regime models.
Market state indicators
Volatility measures
Breadth measures
Liquidity indicators
Flow indicators
Macro transmission variables
### 4.6 Regime Model Layer
Hosts the individual models that estimate the hidden market state.
HMM
Bayesian HMM
Regime-Switching VAR
Bayesian Neural Network
Time-series foundation model components
Particle filtering
BOCPD
### 4.7 Ensemble Layer
Combines the individual model outputs into a unified regime probability distribution.
Bayesian Model Averaging
Time-aware constrained stacking
Model weights
Model agreement measures
### 4.8 Uncertainty and Calibration Layer
Adds uncertainty information and checks whether probabilities are reliable.
Epistemic uncertainty
Aleatoric uncertainty
Conformal calibration
Prediction sets
Coverage tracking
### 4.9 Model Health Layer
Determines whether the current model output is safe to use for downstream decision support.
Feature drift
Changepoint status
Online/batch reconciliation
Prediction stability
Out-of-distribution flags
Health status
### 4.10 Decision Support Layer
Turns the regime output into information that can support human investment decisions.
Regime summary
Conviction
Allocation implications
Constraints
Decision rationale
### 4.11 Presentation and Reporting Layer
Makes the output accessible to users and governance stakeholders.
Dashboard
API
Investment Committee artefact
Historical regime views
Model-health views
### 4.12 Governance and Lineage Layer
Preserves the information needed for reproducibility, audit and model governance.
Model versions
Feature versions
Data snapshots
Inference timestamps
Calibration metrics
Decision records
## 5. Core Application Flow
A normal daily decision-support flow should look like this:
Market / Macro Data
    ↓
Point-in-Time Snapshot
    ↓
Data Quality Checks
    ↓
Feature Generation
    ↓
Model Inference
    ↓
Model Ensemble
    ↓
Probability + Uncertainty
    ↓
Conformal Calibration
    ↓
Model Health Gate
    ↓
Explanation
    ↓
Human Review
    ↓
Decision Support
    ↓
Monitoring and Lineage
## 6. Batch and Online Architecture
### 6.1 Nightly Batch Path
The nightly process handles work that is relatively expensive and does not need to run after every market observation. It can refit selected models, update ensemble weights, recalibrate the conformal layer and create the reference posterior used by the online system.
Refresh point-in-time datasets.
Run data quality checks.
Generate the latest feature set.
Refit or update Bayesian HMM and RS-VAR components where required.
Update ensemble weights.
Recalibrate the selected conformal method.
Store the reference posterior and model versions.
Run validation and health checks.
### 6.2 Intraday Online Path
The online path is designed to react to new observations without performing a complete model refit after every update.
Receive the latest market observations.
Update the current feature state.
Run particle filtering where applicable.
Run BOCPD for potential changepoints.
Update the current regime assessment.
Compare the online posterior with the latest batch reference.
Update model-health flags.
Expose the latest decision-support output.
## 7. Model Layer Architecture
The model layer should use a common interface so that each model can be developed and evaluated independently. This is important because the project intentionally compares models with different assumptions and strengths.
Input Features
    ↓
Model Interface
    ├── HMM / Bayesian HMM
    ├── RS-VAR
    ├── Bayesian Neural Network
    └── Foundation Model + Classification Head
    ↓
Standardized Model Output
    ↓
Ensemble
## 8. Common Model Output
Each model should return a consistent structure so that the ensemble does not need to understand the internal implementation of every model.
| Field | Purpose |
| --- | --- |
| Regime probabilities | Probability assigned to each regime. |
| Model uncertainty | Uncertainty information available from the model. |
| Model version | Exact version used for inference. |
| Inference timestamp | Time of the prediction. |
| Health information | Model-specific health or validity indicators. |
| Diagnostics | Additional information required for evaluation and explanation. |

## 9. Ensemble Layer
The ensemble layer should consume standardized outputs from the model layer and produce one combined regime assessment. The individual models should remain independently inspectable so that model disagreement is not hidden.
Collect model probability outputs.
Apply the selected ensemble method.
Track model weights.
Measure model agreement and disagreement.
Produce the final regime probability vector.
Pass the result to uncertainty and calibration components.
Preserve the individual model outputs for audit and validation.
## 10. Uncertainty and Calibration Architecture
The uncertainty layer sits between the model ensemble and the decision-support layer. Its job is to make the confidence of the regime call visible and to prevent raw probabilities from being treated as perfectly calibrated.
Ensemble Probability
    ↓
Uncertainty Estimation
    ↓
Conformal Calibration
    ↓
Prediction Set + Coverage Information
    ↓
Decision Support
## 11. Model Health Architecture
Model health should be implemented as an explicit gate rather than as a dashboard metric that nobody uses.
Data quality status
Feature drift status
BOCPD changepoint status
Online versus batch reconciliation
Conformal coverage status
Prediction stability
Out-of-distribution indicators
Conceptually:
Prediction → Health Check → Fit for Decision Support? → Yes: Continue | No: Flag or Suppress
## 12. Decision Support Architecture
The decision-support layer should not turn a regime probability into an automatic trade. It should present the information needed for an investment professional to decide what, if anything, should be done.
Current regime and full probability distribution.
Uncertainty and conformal prediction set.
Main drivers and explanation.
Model agreement and health.
Change from the previous assessment.
Possible allocation implication.
Relevant portfolio constraints.
Decision and review status.
## 13. Data and Model Lineage
The architecture must preserve a clear chain from input data to the final decision-support output.
Data Snapshot → Feature Version → Model Version → Model Output → Ensemble → Calibration → Health → Explanation → Decision Record
This lineage is important both for reproducibility and for governance. A historical regime call should be explainable without depending on whatever the current version of the system happens to produce.
## 14. Proposed Repository Structure
The repository should keep business documentation, source code, experiments and tests separated.
project-1a-bayesian-regime-detection/
├── README.md
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── RULES.md
│   ├── PHASES.md
│   ├── DESIGN.md
│   ├── MEMORY.md
│   ├── research/
│   ├── decisions/
│   ├── model/
│   └── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── financial_analysis/
│   ├── models/
│   ├── ensemble/
│   ├── uncertainty/
│   ├── validation/
│   ├── decision_support/
│   ├── monitoring/
│   └── api/
├── configs/
├── notebooks/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── snapshots/
└── reports/
## 15. Technology Stack, Initial Baseline
The technology stack should be selected around the modelling and deployment requirements rather than around a fixed list of tools. The following is an initial baseline and can be revised during architecture research.
| Area | Initial direction | Purpose |
| --- | --- | --- |
| Language | Python | Primary research, modelling and backend language. |
| Data | Pandas / NumPy | Data processing and numerical operations. |
| Classical ML | scikit-learn | Baselines, metrics and supporting utilities. |
| Deep Learning | PyTorch | Bayesian neural network and deep learning components. |
| Time Series | Specialized Python libraries as validated | HMM, VAR, forecasting and temporal modelling. |
| API | FastAPI or equivalent | Expose production inference and decision-support outputs. |
| Testing | pytest | Unit and integration testing. |
| Experiment Tracking | To be selected | Track experiments, model versions and evaluation. |
| Storage | To be selected | Store raw data, processed features and model artifacts. |
| Deployment | To be selected | Serve the decision-support system reliably. |

## 16. Architecture Boundaries
The architecture should maintain clear boundaries between the following concerns:
Data collection and model inference.
Feature generation and financial interpretation.
Individual models and the ensemble.
Prediction and calibration.
Model output and investment decision.
Research experiments and production components.
Model monitoring and business reporting.
These boundaries are important because they make it easier to test a component, replace a model or change a data source without unexpectedly changing the rest of the system.
## 17. Security, Reliability and Governance Considerations
Credentials and data-source access should remain outside source code.
Sensitive configuration should be managed through environment or secret-management mechanisms.
Production outputs should be timestamped and versioned.
Model artifacts should be immutable once promoted.
Critical outputs should be logged with enough context for later investigation.
Failures should be explicit and visible rather than silently converted into empty or default predictions.
The system should distinguish research outputs from validated production outputs.
## 18. Architecture Decisions Still Open
The following decisions should be made only after the relevant research and technical evaluation:
Exact storage technology for point-in-time market data.
Exact time-series and Bayesian modelling libraries.
Final foundation model selection.
Final conformal calibration method.
Final ensemble implementation.
Experiment tracking platform.
Production deployment environment.
Dashboard technology.
API authentication and access-control approach.
Exact retraining and monitoring thresholds.
## 19. Architecture Principles
Modular by design. A model or data source should be replaceable without redesigning the entire system.
Time aware. All financial data processing and evaluation must respect the information available at the time.
Observable. Important system and model states should be visible and measurable.
Reproducible. Historical results should be tied to versioned data, models and configurations.
Fail safely. Unhealthy data or model states should prevent unreliable outputs from silently reaching decision support.
Human centred. The architecture supports human investment decisions rather than autonomous execution.
Research friendly. The structure should make it easy to test new ideas without destabilizing production components.
Governance ready. Lineage, versioning and validation should be part of the architecture from the beginning.
## 20. Architecture Definition of Done
The architecture baseline is considered complete when it clearly shows:
Where data enters the system.
Where point-in-time validation happens.
Where features are generated.
Where each model family lives.
How model outputs are standardized.
How the ensemble is connected.
Where uncertainty and calibration are applied.
Where model health is checked.
How outputs reach the decision-support layer.
How batch and online processing coexist.
How model and data lineage are preserved.
How the repository is organized.
Which technology decisions are fixed and which remain open.
Status: Architecture baseline complete. Implementation choices should be updated as research and Phase 3 architecture decisions are finalized.
## 21. Relationship With Other Project Documents
| Document | Relationship |
| --- | --- |
| PRD.md | Defines what the product must achieve and why. |
| ARCHITECTURE.md | Defines how the product is structured technically. |
| RULES.md | Defines engineering, research and AI usage boundaries. |
| PHASES.md | Defines how the project is executed over time. |
| DESIGN.md | Defines how the user-facing product should look and communicate. |
| MEMORY.md | Records the current project state, completed work and active decisions. |
