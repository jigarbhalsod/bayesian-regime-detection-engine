Project 1A
1.6 Decision Workflow
Bayesian Regime Detection Engine for Equity Direction Forecasting

## 1.6.1 Abstract
The Bayesian Regime Detection Engine is designed as a decision-support system rather than an autonomous trading system. It takes market and macro-financial information and turns it into time-varying regime probabilities, along with a clear view of how uncertain those probabilities are. These outputs support human investment decisions without replacing them.
For the intended long-only Indian mutual-fund environment, the engine can support decisions such as tactical allocation tilts, cash-buffer sizing, and sleeve weighting. The aim is to give investment teams a probabilistic, explainable and auditable view of market conditions instead of a single deterministic market call.
Workflow principle: Observe → Infer → Quantify Uncertainty → Calibrate → Check Model Health → Explain → Human Review → Decide → Act Within Constraints → Monitor → Update.
## 1.6.2 Workflow Overview
The decision workflow is organized into sixteen connected stages:
Market Observation
Feature Engineering
Regime Models
Model Ensembling
Probability Output
Uncertainty Quantification
Conformal Calibration
Model Health Check
Explainability
Human Analyst Review
Investment Decision
Conviction-Based Sizing
Portfolio Constraints
Investment Committee Artefact
Continuous Monitoring
Retraining Governance


## 1.6.3 Stage 1 — Market Observation
The process starts with point-in-time market and macroeconomic information relevant to Indian equity-market conditions.
The main inputs include NIFTY indices, India VIX, FII and DII flows, SIP flows, market breadth, USD/INR, government-security yields, credit spreads and selected macro indicators.
The system must preserve the information that was actually available at the time of each prediction. This prevents later data revisions from leaking into historical decisions and helps make backtests representative of real decision-making conditions.
## 1.6.4 Stage 2 — Feature Engineering
Raw observations are converted into features that describe four broad parts of the market.
Market behaviour covers returns, trend, momentum and volatility. Market breadth covers measures such as advance/decline behaviour, the percentage of stocks above moving averages, and new highs or lows. Capital-flow features cover FII, DII and SIP activity. Macro features cover rates, INR, inflation, liquidity and credit conditions.
Each macro feature should have a clear economic rationale and a plausible transmission mechanism. A feature should not be included simply because it happens to improve model fit.
## 1.6.5 Stage 3 — Regime Models
The engine uses several complementary model families instead of depending on one model.
HMM and Bayesian HMM models are used to identify latent market states and estimate regime transitions. The Bayesian version also provides posterior uncertainty.
Regime-Switching VAR models capture joint dynamics across variables such as returns, volatility, breadth, flows, INR and yields.
Bayesian Neural Networks capture nonlinear relationships while providing predictive uncertainty.
Time-series foundation models such as Chronos and TimesFM can be used as temporal representation or forecasting components. Their continuous outputs are mapped into regime space through a downstream classification head rather than being treated as native regime probabilities.
Sequential inference methods such as particle filtering and Bayesian Online Changepoint Detection provide online state tracking and changepoint detection between batch refits.
## 1.6.6 Stage 4 — Model Ensembling
No individual model is treated as the final authority. The outputs from the different model families are combined through a documented ensemble layer.
The main approaches considered are Bayesian Model Averaging and time-aware constrained stacking.
Within the Bayesian model family, WAIC and PSIS-LOO can be used to compare predictive performance. Cross-family comparisons should instead use suitable out-of-sample probabilistic metrics such as Brier score and log loss.
The purpose of the ensemble is to use the different strengths of the models while reducing dependence on one modelling assumption.
## 1.6.7 Stage 5 — Probability Output
The engine produces a probability distribution across the defined market regimes rather than returning only one hard label.
For example, a daily output might look like this:
## 1.6.8 Stage 6 — Uncertainty Quantification
Every regime call is accompanied by uncertainty information.
Epistemic uncertainty represents uncertainty associated with model limitations, limited information, parameter uncertainty and disagreement across models. It can be estimated using Bayesian posterior variation and disagreement among ensemble members.
Aleatoric uncertainty represents the uncertainty that is inherent in the market and cannot be removed simply by choosing a different model.
The important principle is that a probability should never be presented without enough context to understand how reliable it is.
## 1.6.9 Stage 7 — Conformal Calibration
Raw model probabilities are not assumed to be perfectly reliable. A time-series-appropriate conformal calibration layer is therefore used to evaluate and improve the reliability of the predictive output.
Candidate approaches include adaptive or distribution-shift-aware conformal methods. The final method will be selected through the research and validation stages.
The output can be a prediction set rather than a forced single answer. For example, a 90% conformal prediction set could be {Late-Cycle, Transitional}. This makes uncertainty visible instead of creating false precision.
## 1.6.10 Stage 8 — Model Health Check
A strong-looking probability should not influence an allocation decision if the model itself is behaving abnormally.
The health layer checks BOCPD changepoint status, the gap between online and batch posteriors, feature drift such as PSI, conformal coverage, out-of-distribution indicators and prediction stability.
If the health checks fail, the downstream allocation action can be blocked or flagged even when the raw regime probability looks confident.
## 1.6.11 Stage 9 — Explainability
The system should be able to explain why a particular regime call was produced.
For feature-based models, methods such as SHAP can be used to identify important positive and negative contributors. For HMM and RS-VAR models, regime parameters, transition probabilities and emission characteristics provide a more natural explanation. Foundation-model components can be examined through temporal-window or embedding diagnostics.
The final explanation should be translated into plain language so that an Investment Committee can understand the main reasons behind the call without needing to inspect the model internals.
## 1.6.12 Stage 10 — Human Analyst Review
The quantitative output is reviewed by the relevant research or investment professional.
The reviewer considers the regime probabilities, model agreement, uncertainty, conformal prediction set, major drivers, changepoint status and model-health indicators, along with the wider market context.
The purpose is not to blindly accept the model or to override it without evidence. The purpose is to combine quantitative evidence with professional judgement.
## 1.6.13 Stage 11 — Investment Decision
The regime output is an input to an investment decision, not an automatic instruction.
Depending on the situation, it may inform a tactical allocation tilt, cash-buffer adjustment, sleeve reweighting, sector allocation change or broader risk reduction.
The engine provides evidence. The responsible investment team remains accountable for the actual decision.
## 1.6.14 Stage 12 — Conviction-Based Sizing
The size of an allocation response should reflect the strength and reliability of the regime signal.
A high regime probability combined with low uncertainty and a narrow conformal prediction set represents stronger conviction and can support a larger tilt.
Mixed probabilities, high uncertainty or a wider prediction set should result in a smaller action or no action.
The principle is simple: stronger evidence should allow more conviction, while ambiguous evidence should lead to restraint.
## 1.6.15 Stage 13 — Portfolio Constraints
Any allocation action must remain within the relevant portfolio and regulatory constraints.
These include long-only requirements, scheme-category restrictions, liquidity, turnover limits, transaction costs, market impact, no-trade bands and hysteresis.
Hysteresis helps prevent unnecessary switching by using different thresholds for entering and exiting a regime-driven action.
The practical sequence is therefore: model signal → conviction → constraints → permitted action.
## 1.6.16 Stage 14 — Investment Committee Artefact
The final decision-support output should be turned into a standardized Investment Committee artefact.
It should clearly communicate what the current regime is and how probable it is, why the model reached that view, how confident the system is, what changed from the previous assessment, what could happen next, and what allocation implication could reasonably be considered.
It should also preserve enough lineage to show exactly which data and model versions produced the call.
## 1.6.17 Stage 15 — Continuous Monitoring
The system operates at two speeds.
The nightly batch process can refit the Bayesian HMM and RS-VAR, update ensemble weights and recalibrate the conformal layer.
The intraday online process can use particle filtering and BOCPD to track the market state and detect potential changepoints. Online results are reconciled against the latest batch posterior.
This allows the system to respond to new information without treating every short-term movement as a completely new model-training problem.
## 1.6.18 Stage 16 — Retraining and Model Governance
Retraining should be driven by evidence that the model needs attention rather than by an arbitrary calendar schedule.
Possible triggers include feature drift, deteriorating conformal coverage, significant changepoints, excessive online-to-batch divergence and persistent prediction instability.
Every retraining event should be recorded. Governance should include model versioning, champion-challenger comparison, model-risk tiering, periodic validation, calibration monitoring, model cards and validation reports.
For promoted models, the training-data snapshot, priors, hyperparameters, calibration metrics and code version should be retained so that historical predictions can be reconstructed.
## 1.6.7A Example Probability Output
| Regime | Probability |
| --- | --- |
| Risk-On | 14% |
| Late-Cycle | 52% |
| Transitional | 27% |
| Risk-Off | 5% |
| Post-Shock | 2% |

## 1.6.19 Combined Output Contract
The combined regime output contract is the formal interface between the regime engine and the downstream decision-support components. It should carry enough information to understand the prediction, its uncertainty, its provenance and its current health.
| Field | Purpose |
| --- | --- |
| DateKey | Date of the regime assessment |
| RegimeID | Regime being reported |
| Probability | Ensembled regime probability |
| Conformal_Lower | Lower calibrated bound |
| Conformal_Upper | Upper calibrated bound |
| EpistemicUnc | Epistemic uncertainty |
| AleatoricUnc | Aleatoric uncertainty |
| DominantModel | Dominant or highest-weight model |
| ModelVersionID | Model version used for the call |
| FeatureVectorHash | Input-feature lineage and reproducibility reference |
| Source | Output or model source |
| InferenceTimestamp | Time at which the prediction was generated |
| HealthFlags | Current model-health status |

## 1.6.20 Auditability and Decision Lineage
Every regime call should be traceable through the following chain:
Data Snapshot → Feature Version → Model Version → Model Outputs → Ensemble Weights → Probability Distribution → Calibration → Uncertainty → Health Checks → Explanation → Human Review → Allocation Decision → Investment Committee Artefact
The system should retain the training-data snapshot, model version, priors, hyperparameters, calibration metrics, evaluation metrics and code version for promoted models. The final output should record which model versions generated the call. This makes it possible to reconstruct an historical prediction and explain why it was produced.
## 1.6.21 Decision Principles
Probability over labels. Report the full regime probability distribution rather than only the dominant label.
Uncertainty always accompanies probability. A probability should be interpreted together with its uncertainty.
Ensemble over a single black box. Use complementary model families through a documented ensemble.
Evidence before action. Allocation decisions should be supported by model, feature and health evidence.
Conviction controls action size. Stronger and better-calibrated signals can justify larger tilts.
Constraints always apply. Regulatory, portfolio, liquidity and execution constraints remain binding.
Humans remain accountable. The engine supports investment decisions but does not autonomously execute them.
Every call must be reconstructable. Data, features, models, probabilities, calibration and decisions must remain traceable.
## 1.6.22 End-to-End Example
Assume NIFTY remains above its 200-DMA while market breadth weakens, India VIX rises, FII flows turn modestly negative, DII flows remain positive and the INR stays relatively stable. The market now looks ambiguous between a Late-Cycle and Transitional state.
The ensemble could produce the following assessment:
| Regime | Probability |
| --- | --- |
| Late-Cycle | 52% |
| Transitional | 27% |
| Risk-On | 14% |
| Risk-Off | 5% |
| Post-Shock | 2% |

Suppose the 90% conformal prediction set is {Late-Cycle, Transitional} and epistemic uncertainty is elevated because the models disagree. If the health checks remain normal, the analyst can review the evidence and decide whether a small de-risking tilt is appropriate. The action still has to pass the portfolio constraints and hysteresis rules.
The important point is that the system does not recommend a large action simply because one regime has the highest probability. Probability, uncertainty, model health, evidence and portfolio constraints are considered together.
## 1.6.23 Definition of Done
The Decision Workflow documentation is complete when it can clearly answer the following questions:
What enters the system?
How is the regime inferred?
How are the models combined?
What exactly is produced?
How is uncertainty represented?
How is calibration performed and verified?
How is model health assessed?
Why was a particular call produced?
Who reviews the call?
How can the output influence allocation?
What constraints apply?
How is the decision recorded?
How is the system monitored after the decision?
How can an auditor reconstruct the complete decision chain?
Status: 1.6 Decision Workflow — COMPLETE.
## 1.6.24 Product Perspective
The important product distinction is that Project 1A is not simply a market-regime prediction model. It is a governed decision-support workflow in which the regime model is one component.
The product loop is:
Data → Intelligence → Confidence → Explanation → Decision → Outcome → Feedback
The value is created when the system helps an investment professional make a better-informed decision at the right level of conviction, while keeping the reasoning and evidence available for later review.
