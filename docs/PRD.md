PROJECT 1A
PRODUCT REQUIREMENTS DOCUMENT
Bayesian Regime Detection Engine for Equity Direction Forecasting
Zetheta Algorithms Private Limited

## 1. Document Purpose
This document defines what Project 1A is intended to build, why it is being built, who it is meant to support, and what the finished product should be capable of doing. It acts as the main product reference for the project and will guide the research, architecture, development, validation and final delivery.
The requirements are based on the Zetheta project brief and the decisions already made while structuring the project. Technical implementation choices may evolve as research progresses, but the core product purpose and boundaries should remain clear.
## 2. Product Summary
Project 1A is a Bayesian Regime Detection Engine for the Indian equity market. Instead of trying to predict an exact future price, the system focuses on identifying the broader condition of the market and estimating how likely it is to move from one condition to another.
The engine combines several modelling approaches, including Hidden Markov Models, Bayesian deep learning, regime switching VAR models, time series foundation models and sequential Bayesian inference. Their outputs are brought together through a documented ensemble and calibrated so that the final probabilities can be interpreted and defended.
The intended result is a decision support product for a long only Indian mutual fund environment. The system can support tactical allocation tilts, cash buffer sizing and sleeve weighting, while leaving the final investment decision with the responsible human team.
## 3. Problem Statement
Traditional quantitative investment workflows often focus on predicting future prices. For a multi month horizon, this is difficult to defend because financial markets are noisy, relationships change over time, and the assumptions behind a model can stop holding when the market enters a new environment.
Project 1A takes a different approach. Rather than asking only, "What will the index price be?", the system asks, "What kind of market are we in, how likely is that assessment to be correct, and what evidence supports it?"
The product therefore needs to handle changing market conditions, model disagreement and uncertainty as first class parts of the decision process. It also needs enough documentation and lineage for the result to be reviewed by investment and governance stakeholders.
## 4. Product Objective
The main objective is to build an engine that classifies the Indian equity market into five defined regimes, Risk On, Late Cycle, Transitional, Post Shock and Risk Off, and produces calibrated, time varying probabilities for those regimes.
The product should help an investment team:
Understand the current market regime.
See how the probability of each regime changes over time.
Identify possible regime transitions early.
Understand how uncertain the model is.
Understand the main reasons behind a regime call.
Use the output as one input into tactical allocation decisions.
Monitor whether the model is behaving reliably.
Reconstruct an earlier call when the reasoning or evidence needs to be reviewed.
## 5. Target Users and Stakeholders
The primary users are expected to be professionals involved in research, portfolio construction and investment decisions.
| User / Stakeholder | Role in the workflow | What they need from the product |
| --- | --- | --- |
| Quant / Research Analyst | Studies the model output and market evidence. | Probabilities, model agreement, drivers, uncertainty and diagnostics. |
| Portfolio / Fund Manager | Uses the signal as an input to portfolio decisions. | Clear regime view, conviction, allocation implications and constraints. |
| Multi Asset Solutions Team | Uses regime information for broader allocation decisions. | Consistent market state information across the equity scheme line up. |
| Investment Committee | Reviews the investment case and supporting evidence. | Plain language rationale, probability, uncertainty, change from prior view and lineage. |
| Risk / Model Governance | Checks model reliability and controls. | Calibration, health indicators, validation results and reproducibility. |

## 6. User Problem and Decision Context
The product is useful when an investment professional needs to understand whether the market is behaving normally or moving into a different environment.
A typical interaction looks like this:
Market data → Regime engine → Probability and uncertainty → Analyst review → Investment decision → Monitoring
The engine should make the reasoning easier to inspect. A user should be able to see not only the dominant regime, but also the alternatives, the uncertainty around the call, the main drivers and whether the model is currently healthy.
## 7. Core Product Features
### 7.1 Market Data Integration
Bring together the market and macro inputs required for regime detection, with point in time handling and data quality checks.
NIFTY indices
India VIX
FII and DII flows
SIP flows
Market breadth
USD/INR
10Y Gilt yields
Credit spreads
Selected macro indicators
### 7.2 Regime Detection
Estimate the current state of the Indian equity market using multiple complementary models.
Risk On
Late Cycle
Transitional
Post Shock
Risk Off
### 7.3 Probabilistic Output
Return a probability distribution over all regimes instead of a single hard label.
Regime probabilities
Probability history
Regime transition probabilities where supported
### 7.4 Uncertainty Reporting
Show how reliable the prediction is and separate model related uncertainty from market noise.
Epistemic uncertainty
Aleatoric uncertainty
Model disagreement
### 7.5 Calibration
Check whether reported probabilities behave as expected when compared with observed outcomes.
Conformal prediction
Coverage monitoring
Time aware calibration
### 7.6 Model Health
Prevent unreliable model output from silently becoming an investment signal.
Feature drift
Changepoint status
Online and batch reconciliation
Prediction stability
Out of distribution indicators
### 7.7 Explainability
Give users a practical explanation for why the system reached its current view.
Important features
Model level diagnostics
Plain language rationale
Positive and negative contributors
### 7.8 Decision Support
Translate the regime view into information that can support investment decisions without executing them automatically.
Tactical allocation guidance
Cash buffer considerations
Sleeve weighting implications
Conviction based sizing
### 7.9 Audit and Lineage
Keep enough information to reproduce and explain historical regime calls.
Data snapshot
Feature version
Model version
Ensemble information
Inference timestamp
Health flags
Decision record
### 7.10 Monitoring and Governance
Track model performance after deployment and identify when retraining or review is needed.
Calibration monitoring
Drift monitoring
Retraining triggers
Model versioning
Champion challenger comparison
## 8. Functional Requirements
The system must classify the Indian equity market into the five defined regimes.
The system must produce a probability for each regime rather than only a single label.
The system must expose uncertainty alongside the regime probabilities.
The system must combine complementary model outputs through a documented ensemble.
The system must apply an appropriate calibration layer and report empirical coverage.
The system must monitor model health and be able to flag or suppress downstream action when health checks fail.
The system must provide an understandable explanation for the main drivers of a regime call.
The system must support historical and real time regime monitoring.
The system must provide outputs that can be used by downstream allocation and reporting components.
The system must preserve sufficient metadata to reconstruct an historical prediction.
The system must keep the final investment decision with the responsible human user.
## 9. Non Functional Requirements
Auditability: A historical call should be traceable from its input data through the model and ensemble to the final reported probability.
Reproducibility: The same versioned inputs and model configuration should reproduce the same result within the defined numerical tolerance.
Reliability: The system should identify data and model health problems rather than silently producing unreliable outputs.
Interpretability: A non technical investment stakeholder should be able to understand the main reason behind a regime call.
Time awareness: Historical evaluation must respect the information that was available at the time.
Modularity: Individual model families should be replaceable without redesigning the entire product.
Maintainability: Configuration, model versions, data definitions and project rules should be centrally documented.
Governance readiness: The product should preserve the information required for validation, review and audit.
## 10. Primary Product Output
The main output of the engine is a combined regime assessment containing:
| Output | Purpose |
| --- | --- |
| RegimeID | Identifies the market regime. |
| Probability | Shows the model's estimated probability for the regime. |
| Conformal bounds / prediction set | Shows calibrated uncertainty around the prediction. |
| Epistemic uncertainty | Shows uncertainty related to model knowledge and disagreement. |
| Aleatoric uncertainty | Shows uncertainty inherent in the market. |
| Dominant model | Identifies the model contributing most strongly to the combined view. |
| Model version | Identifies the exact model configuration used. |
| Inference timestamp | Records when the prediction was generated. |
| Health flags | Shows whether the model was operating within expected conditions. |
| Lineage information | Allows the prediction to be traced back to its inputs and configuration. |

## 11. Product Scope
### 11.1 In Scope
Indian equity market regime detection.
Five regime states, Risk On, Late Cycle, Transitional, Post Shock and Risk Off.
Market, breadth, flow, volatility and macro features relevant to the regime problem.
Bayesian and complementary regime modelling approaches specified in the project.
Model ensembling and probabilistic output.
Uncertainty quantification and conformal calibration.
Sequential state tracking and changepoint detection.
Explainability and model health monitoring.
Decision support for tactical allocation, cash buffer sizing and sleeve weighting.
Audit and model lineage.
### 11.2 Out of Scope
Autonomous trade execution.
Guaranteed return prediction.
A system that treats the dominant regime as an automatic buy or sell instruction.
Replacing portfolio managers, analysts or the Investment Committee.
Unrestricted investment recommendations that ignore scheme and portfolio constraints.
## 12. Product Success Criteria
Success should not be judged by one accuracy number. The product needs to demonstrate that its regime probabilities are useful, reasonably calibrated, stable enough for decision support and defensible to a professional reviewer.
| Area | What success looks like | Examples of measures |
| --- | --- | --- |
| Regime detection | The system can distinguish meaningful market states and transitions. | Regime accuracy, transition detection, confusion matrix |
| Probabilistic quality | Probabilities are useful rather than overconfident. | Log loss, Brier score, calibration error |
| Uncertainty | High uncertainty is visible when models disagree or the market becomes unusual. | Uncertainty trends, prediction set width |
| Calibration | Reported confidence aligns reasonably with observed outcomes. | Empirical coverage, calibration curves |
| Model health | The system identifies drift and abnormal behaviour. | PSI, changepoint flags, reconciliation gap |
| Decision usefulness | The output can support a better documented allocation decision. | Decision impact, drawdown/risk analysis, allocation stability |
| Auditability | Historical calls can be reconstructed and explained. | Lineage completeness, reproducibility checks |

## 13. Key Product Constraints
The project is designed around a short, intensive delivery window and therefore requires careful prioritisation.
Historical testing must respect time order and point in time information availability.
The system should be designed for long only mutual fund use cases and relevant scheme constraints.
Model outputs must be explainable and supported by evidence.
AI assisted development can accelerate implementation, but generated code and results must still be tested and validated.
Financial calculations and critical model outputs should be cross checked where the project requirements call for it.
## 14. Product Principles
Direction over price. The product focuses on market state and direction rather than pretending to know an exact future price.
Probability over labels. The product communicates a distribution of possible regimes.
Uncertainty is part of the answer. A prediction without its uncertainty is incomplete.
Multiple models over one black box. Different modelling assumptions should be compared and combined transparently.
Evidence before action. A regime call should have supporting evidence before it influences an investment decision.
Human accountability. The product supports investment professionals and does not replace them.
Auditability by design. Lineage and reproducibility are product requirements, not documentation added at the end.
Fail safely. When data or model health is poor, the system should flag or suppress action rather than hide the problem.
## 15. Core User Journey
A typical user journey is:
## 1. The system receives the latest point in time market and macro data.
## 2. The data is checked and converted into regime relevant features.
## 3. Multiple models assess the market state.
## 4. The ensemble produces a probability distribution.
## 5. Calibration and uncertainty information are added.
## 6. Model health checks determine whether the output is fit for decision support.
## 7. The system explains the main drivers of the call.
## 8. An analyst reviews the evidence.
## 9. The portfolio team considers the signal alongside portfolio constraints.
## 10. The resulting decision and supporting evidence are recorded.
## 11. The system continues monitoring the market and its own behaviour.
## 16. Example Product Interaction
Suppose the market has remained above its long term trend, but breadth is weakening, India VIX is rising and FII flows are turning negative. The engine may see the market as moving away from a comfortable Risk On state without yet having enough evidence to call a full Risk Off regime.
A possible output could be:
| Regime | Probability |
| --- | --- |
| Late Cycle | 52% |
| Transitional | 27% |
| Risk On | 14% |
| Risk Off | 5% |
| Post Shock | 2% |

The product would not simply tell the portfolio manager to sell. It would also show the uncertainty, model agreement, main drivers and model health. The investment team could then decide whether a small risk reduction is appropriate, subject to portfolio rules.
## 17. Product Acceptance Criteria
The five regime states are clearly defined and consistently represented.
The engine produces a complete probability distribution for every valid prediction.
Uncertainty information is included with the probability output.
The ensemble methodology is documented and reproducible.
Calibration performance is measured and reported.
Model health checks are implemented and visible to downstream users.
A user can understand the main drivers behind a regime call.
The output can be consumed by the decision support layer without requiring a specific model implementation.
Historical predictions retain sufficient lineage for reconstruction.
The product does not execute trades automatically.
The final system supports the intended long only mutual fund decision context.
## 18. Open Product Questions
The following questions are intentionally left open for the research and architecture phases. They are product decisions that should be resolved using evidence rather than assumptions.
What exact feature set provides the best balance between predictive value, economic meaning and auditability?
Which model families should receive the highest ensemble weight under different market conditions?
Which conformal method provides the most reliable coverage under Indian financial time series and distribution shift?
What probability and uncertainty thresholds should be required before an allocation tilt is considered?
How frequently should the decision support layer update for different market conditions?
What is the right level of allocation guidance without turning the product into an autonomous trading system?
Which dashboard views are most useful for the Investment Committee and portfolio team?
## 19. PRD Definition of Done
The PRD is considered complete when it clearly answers:
What problem are we solving?
Why is the problem important?
Who are we building for?
What decisions should the product support?
What are the core product capabilities?
What is inside and outside the scope?
What does the system need to produce?
How will product success be measured?
What constraints must the product respect?
What questions still need research before implementation?
Status: PRD baseline complete. The document should be updated if a later research or architecture decision materially changes the product requirements.
## 20. Source and Requirement Note
This PRD is based primarily on the Zetheta Project 1A brief and the Project 1A Decision Workflow documentation developed during the project. The brief defines the core objective, intended users and decision context, required modelling families, calibrated uncertainty, ensemble design and audit expectations. The Decision Workflow document expands those requirements into a practical product flow.
