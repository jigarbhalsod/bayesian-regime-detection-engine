PROJECT 1A
PROJECT RULES
Bayesian Regime Detection Engine for Equity Direction Forecasting
Engineering, Research, Data, AI and Documentation Guidelines

## 1. Purpose
This document defines the working rules for Project 1A. The goal is to keep the project consistent as the codebase, research, models and documentation grow. These rules are meant to make the work reproducible, understandable and safe to change.
The rules apply to both human work and AI assisted work. They are especially important for this project because financial modelling can produce convincing results even when the underlying data, validation or assumptions are wrong.
## 2. Core Working Principles
Understand the requirement before implementing it.
Prefer simple, testable solutions before adding complexity.
Keep research decisions and implementation decisions visible.
Never hide uncertainty when the model itself is uncertain.
Treat data quality and time awareness as first class requirements.
Do not accept a result simply because it looks statistically impressive.
Document important decisions when they are made, not weeks later.
Keep the project modular so that one component can be changed without breaking everything else.
## 3. Product Rules
### 3.1 Product Boundary
The system is a decision-support product, not an autonomous trading system.
A regime probability is an input to an investment decision, not an automatic buy or sell instruction.
Human investment professionals remain responsible for the final decision.
Any allocation guidance must respect portfolio and regulatory constraints.
### 3.2 Requirement Rules
Every major feature should have a clear reason for existing.
Features should be connected to a user problem, workflow requirement or model requirement.
Do not add functionality simply because it is technically interesting.
Separate must-have requirements from future enhancements.
When a requirement changes, update the relevant documentation and project memory.
## 4. Data Rules
### 4.1 Point-in-Time Discipline
Historical analysis must use only information that would have been available at the time of the prediction. Later data revisions, future observations or accidentally forward-filled information must not enter the feature set.
### 4.2 Data Provenance
Every important dataset should have a known source.
Record collection time and relevant observation dates.
Keep track of transformations applied to raw data.
Do not overwrite raw data with processed data.
Keep reproducible snapshots where historical reconstruction is required.
### 4.3 Data Quality
Check missing values before modelling.
Check duplicates and unexpected timestamps.
Check numerical ranges and obvious anomalies.
Investigate sudden data changes before treating them as market signals.
Do not silently replace bad data with assumptions.
If a data problem cannot be resolved, flag it and make the downstream impact explicit.
## 5. Financial Modelling Rules
Use time-aware validation for financial forecasting and regime detection.
Do not use random train/test splitting where it creates temporal leakage.
Check for look-ahead bias before trusting backtest results.
Keep model assumptions documented.
Use economically meaningful features where possible.
Do not add a macro feature only because it improves historical fit.
Treat regime definitions as explicit product assumptions, not labels discovered only after modelling.
Compare models using appropriate out-of-sample probabilistic metrics.
Report uncertainty together with the prediction.
Do not interpret a high probability as certainty.
## 6. Model Rules
### 6.1 Model Modularity
Each model family should be implemented behind a consistent interface. HMM, Bayesian HMM, RS-VAR, Bayesian Neural Network and foundation-model components should be independently testable and replaceable.
### 6.2 Model Outputs
Models should return standardized outputs wherever practical.
Model version information should accompany production predictions.
Model-specific diagnostics should remain available for validation.
Individual model outputs should not be hidden by the ensemble.
### 6.3 Ensemble Rules
Do not treat one model as automatically authoritative.
The ensemble method must be documented.
Model weights should be traceable.
Model disagreement should remain visible because it contributes to uncertainty.
Changes to ensemble methodology should be treated as model changes and recorded.
## 7. Uncertainty and Calibration Rules
Every regime prediction should be accompanied by uncertainty information.
Epistemic and aleatoric uncertainty should not be presented as the same thing.
Calibration must be evaluated out of sample.
Financial time series require methods that account for temporal dependence and distribution shift.
Coverage and calibration performance should be monitored after deployment.
A prediction set can be preferable to a forced single-regime answer when uncertainty is high.
## 8. Validation Rules
Validate the complete workflow, not only the final model.
Use walk-forward or other time-aware evaluation where appropriate.
Keep a clear separation between development data and evaluation data.
Do not tune repeatedly on the final test period.
Report probabilistic metrics such as Brier score and log loss where relevant.
Check calibration, regime stability and transition behaviour.
Evaluate performance during different market environments rather than relying only on aggregate results.
Record failed experiments and negative results when they affect a modelling decision.
## 9. Model Health Rules
Model health is part of the product, not just a monitoring dashboard.
Feature drift should be monitored.
Changepoints should be monitored.
Online and batch inference should be reconciled where both are used.
Conformal coverage should be tracked.
Unexpected prediction instability should be investigated.
If the model is unhealthy, downstream allocation action should be flagged or suppressed rather than silently continued.
## 10. Code Rules
### 10.1 Structure
Keep code modular and organized by responsibility.
Avoid putting business logic, model logic and data access in one large file.
Use clear names for variables, functions, classes and configuration values.
Keep configuration separate from core implementation where practical.
Avoid duplicated logic.
### 10.2 Error Handling
Errors should be explicit and meaningful.
Do not use broad exception handling to hide failures.
Log enough context to understand important failures.
Do not silently convert failed model inference into a valid-looking prediction.
Validate inputs at module boundaries.
Use safe defaults only when their meaning is clearly defined.
### 10.3 Testing
Critical data transformations should have tests.
Model interfaces should have tests.
Important output schemas should have tests.
Time-aware validation logic should be tested.
Integration tests should cover the path from data input to standardized model output.
Tests should be updated when behaviour changes.
## 11. AI Usage Rules
AI tools can be useful throughout the project, but they must accelerate engineering and research rather than replace judgement. Any AI generated material is treated as a draft until it has been checked.
### 11.1 AI May Be Used For
Generating code boilerplate.
Suggesting implementation approaches.
Explaining unfamiliar technical concepts.
Helping debug errors.
Creating test cases.
Improving documentation wording.
Summarizing research material.
Generating alternative designs for comparison.
### 11.2 AI Must Not Be Trusted Without Verification For
Financial performance claims.
Model evaluation results.
Dataset contents or statistics.
Research conclusions.
Regulatory requirements.
Library APIs that have not been checked.
Backtest results.
Claims about production behaviour.
Any result that is presented as measured evidence.
### 11.3 AI Boundaries
AI must not invent data, experiments or results.
AI must not fabricate citations or sources.
AI must not silently change a requirement.
AI generated code must pass the project's testing standards.
AI suggested financial logic must be reviewed before implementation.
When AI is uncertain, that uncertainty should be stated rather than hidden.
## 12. Research Rules
Use primary or high quality sources for important technical and financial claims.
Keep track of which ideas come from the project brief and which are proposed by the team.
Do not turn an untested idea into an architecture requirement without evidence.
Record important research findings in the relevant research documentation.
When two sources disagree, record the disagreement and investigate it.
Separate established knowledge, project requirements and our own design decisions.
## 13. Documentation Rules
Documentation should explain decisions, not only describe files.
Use simple and direct language.
Avoid unnecessary jargon where a simpler explanation works.
Keep important assumptions visible.
Do not claim something is implemented when it is only planned.
Use consistent names for regimes, models, datasets and system components.
Update documentation when a major decision changes.
Keep project memory current enough that work can continue without restarting from the beginning.
## 14. Decision Logging
Important project decisions should be recorded using a simple structure:
Decision
Context
Options considered
Chosen approach
Reason
Trade-offs
Date
Status
This is particularly useful when selecting models, data sources, calibration methods, ensemble strategies and deployment technologies.
## 15. Definition of Done Rules
A task should not be marked complete merely because the code runs. Depending on the task, completion should include:
Requirement understood.
Implementation completed.
Tests added or updated where appropriate.
Expected output checked.
Relevant documentation updated.
Known limitations recorded.
Result reproducible.
No unresolved critical error hidden from the project team.
## 16. File and Repository Rules
Keep documentation under docs/.
Keep source code under src/.
Keep tests under tests/.
Keep notebooks for exploration rather than core production logic.
Keep raw and processed data separated.
Do not commit credentials, private keys or secrets.
Use clear and consistent filenames.
Avoid committing large generated artifacts unless there is a clear reason.
## 17. Change Management
Project requirements and technical decisions will change as research progresses. Changes should be made deliberately rather than silently.
Identify what changed.
Explain why it changed.
Check which documents or modules are affected.
Update the relevant documentation.
Update tests when behaviour changes.
Record important architectural or product changes in the decision log.
Update MEMORY.md after meaningful project checkpoints.
## 18. Practical Rule for Working With Codex and ChatGPT
ChatGPT is primarily used for product thinking, research interpretation, documentation, planning and reasoning through decisions. Codex is primarily used for implementation, repository changes, code maintenance and keeping technical files aligned with the actual codebase.
Both tools should follow the same project rules and neither should be treated as an authority over the other.
## 19. Rule Priority
When two instructions or decisions conflict, use the following order:
Project and regulatory requirements.
Approved product requirements.
Approved architecture and design decisions.
Project rules in this document.
Implementation preferences.
Convenience.
## 20. Rules That Are Specific to Project 1A
Never allow look-ahead information into historical regime features.
Never treat the dominant regime as a guaranteed market outcome.
Always keep the full regime probability distribution.
Always report uncertainty with the regime assessment.
Do not hide model disagreement inside the ensemble.
Do not allow an unhealthy model state to silently drive allocation guidance.
Keep the investment decision with the human user.
Preserve enough lineage to reconstruct a historical regime call.
Evaluate the system as a decision-support workflow, not only as a prediction model.
## 21. Rules Definition of Done
This document is considered complete when it clearly defines:
How the project should be developed.
How data should be handled.
How financial models should be evaluated.
How code should be structured and tested.
How errors should be handled.
How AI tools may and may not be used.
How research and design decisions should be documented.
How model health and uncertainty should be treated.
How project changes should be recorded.
Which rules are specific to Project 1A.
Status: Project Rules baseline complete. The document should be updated when a new project-wide rule is formally adopted.
## 22. Relationship With Other Project Documents
| Document | Relationship |
| --- | --- |
| PRD.md | Defines what the product should achieve. |
| ARCHITECTURE.md | Defines how the system is structured. |
| RULES.md | Defines how the project should be built and managed. |
| PHASES.md | Defines the project roadmap and execution stages. |
| DESIGN.md | Defines the user-facing visual and interaction language. |
| MEMORY.md | Records the current project state and important continuity information. |
