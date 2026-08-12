PROJECT 1A
PROJECT MEMORY
Bayesian Regime Detection Engine for Equity Direction Forecasting
Continuity, current state, decisions and active work

## 1. Purpose
This document is the project's working memory. Its purpose is simple, to make it possible to continue the project without having to reconstruct the entire history from chat messages, notebooks or individual files.
Unlike the PRD, Architecture, Rules, Phases and Design documents, this file is expected to change frequently. It should reflect the current project position, the work that has been completed, the work currently in progress, important decisions and the next practical step.
## 2. Current Project Snapshot
| Item | Current state |
| --- | --- |
| Project | Project 1A, Bayesian Regime Detection Engine for Equity Direction Forecasting |
| Primary domain | Indian equity market and long only mutual fund decision support |
| Current major phase | Phase 2, Research & Literature |
| Phase 1 status | Completed |
| Documentation baseline | PRD, Architecture, Rules, Phases and Design completed |
| Current focus | Breaking Phase 2 into manageable research topics and completing the literature/research work |
| Next major technical phase | Phase 3, Solution Architecture refinement |

## 3. Project Objective
Build a Bayesian Regime Detection Engine that classifies the Indian equity market into defined market regimes and produces calibrated, time-varying regime probabilities with uncertainty. The output is intended to support human investment decisions such as tactical allocation tilts, cash-buffer sizing and sleeve weighting.
The system is a decision-support tool. It is not intended to execute trades autonomously or replace the judgement of investment professionals.
## 4. Defined Market Regimes
Risk On: Market conditions are generally supportive of risk taking.
Late Cycle: Conditions remain constructive, but signs of weakening or market maturity are increasing.
Transitional: Evidence suggests the market is moving between broader states.
Post Shock: The market is responding to or recovering from a significant shock.
Risk Off: Risk appetite has broadly deteriorated.
## 5. Completed Work
### 5.1 Business and BFSI Understanding
Business and BFSI context established.
Market regime concept defined.
Target users and stakeholders identified.
Decision workflow documented.
Core use-case direction established.
Regulatory and governance considerations identified.
Success metrics and scope direction established.
### 5.2 Decision Workflow
The documented workflow is:
Observe → Infer → Quantify Uncertainty → Calibrate → Check Model Health → Explain → Human Review → Decide → Act within Constraints → Monitor → Update
The workflow explicitly keeps the human investment decision separate from the model output. Probability, uncertainty, calibration, health and lineage are treated as part of the decision process rather than optional technical details.
### 5.3 Project Documentation Baseline
PRD, product requirements baseline completed.
ARCHITECTURE, solution architecture baseline completed.
RULES, project working rules completed.
PHASES, master project roadmap completed.
DESIGN, product and interface design baseline completed.
MEMORY, this continuity document.
## 6. Major Project Phases
| Phase | Name | Status |
| --- | --- | --- |
| 1 | Business & BFSI Understanding | Completed |
| 2 | Research & Literature | Current |
| 3 | Solution Architecture | Upcoming |
| 4 | Data Layer | Planned |
| 5 | Data Engineering | Planned |
| 6 | Feature Engineering | Planned |
| 7 | Financial Analysis | Planned |
| 8 | Baseline Regime Engine | Planned |
| 9 | Advanced Models | Planned |
| 10 | Ensemble & Uncertainty | Planned |
| 11 | Validation | Planned |
| 12 | Decision Support Layer | Planned |
| 13 | Deployment | Planned |
| 14 | Documentation | Planned |
| 15 | Finalization | Planned |

## 7. Current Phase, Research & Literature
The current objective is to understand the methods that may be used in the regime detection engine before committing to implementation choices. Research should be connected to the actual Project 1A problem rather than being treated as a general machine learning literature survey.
The main research areas are:
Market regime detection and regime-switching methods.
HMM and Bayesian HMM.
Regime-Switching VAR.
Bayesian Neural Networks and uncertainty.
Time-series foundation models and their use as feature extractors or auxiliary forecasters.
Sequential Monte Carlo and particle filtering.
Bayesian Online Changepoint Detection.
Model ensembling and Bayesian Model Averaging.
Time-aware stacking and probabilistic evaluation.
Conformal prediction for time-dependent financial data.
Probability calibration and uncertainty evaluation.
Financial and macroeconomic features relevant to Indian equity regimes.
## 8. Current Research Workflow
Research Topic
    ↓
Understand the Method
    ↓
Check Financial Relevance
    ↓
Study Evidence and Limitations
    ↓
Compare With Alternatives
    ↓
Decide: Use / Test / Reject / Keep Open
    ↓
Record the Decision
The goal is not to use every advanced method listed in the project brief. A method should earn its place through fit, evidence, practical value and the ability to validate it properly.
## 9. Important Decisions Already Made
| Decision | Current position |
| --- | --- |
| Decision-support boundary | The system supports human investment decisions and does not autonomously execute trades. |
| Probability over hard labels | The system should report the full regime probability distribution. |
| Uncertainty is mandatory | Regime probabilities should be accompanied by uncertainty information. |
| Multiple models | The architecture should compare and combine complementary model families rather than rely on one black box. |
| Calibration | Raw model probabilities should not automatically be treated as calibrated. |
| Model health gate | Unhealthy model states can block or suppress downstream decision support. |
| Human accountability | The final investment decision remains with the responsible human user. |
| Auditability | Historical regime calls should be reconstructable from data, features, models and configuration. |

## 10. Open Decisions
Which exact model implementations will be selected after research.
Which foundation model, if any, provides enough value to justify inclusion.
Which conformal method is most appropriate for the project's temporal setting.
Which ensemble method should be used.
Which features should make the final production feature set.
What probability and uncertainty thresholds should influence allocation support.
Which data sources are reliable enough for the final system.
Which deployment and dashboard technologies should be finalized.
## 11. Current Active Files
docs/
├── PRD.md
├── ARCHITECTURE.md
├── RULES.md
├── PHASES.md
├── DESIGN.md
└── MEMORY.md
The repository may contain additional research, reports, notebooks and implementation files. This section is only intended to identify the core project-management documents.
## 12. How This File Should Be Updated
Update MEMORY.md whenever one of the following happens:
A major phase is completed.
A research topic is accepted, rejected or materially changed.
A product or architecture decision is finalized.
A new important constraint is discovered.
The active file or workstream changes.
A major implementation milestone is reached.
A blocker appears or is resolved.
The next step becomes different from the previously planned step.
## 13. What Should Not Go Into MEMORY.md
Long technical explanations that already belong in another document.
Full research summaries.
Complete model specifications.
Temporary debugging details that have no future relevance.
Every small code change.
Duplicate copies of the PRD or architecture.
MEMORY.md should remain concise enough that someone can read it quickly and understand where the project stands.
## 14. Current Next Steps
Complete the remaining Phase 2 research topics.
Build the literature and method comparison matrix.
Record which approaches are recommended, optional or rejected.
Use the research findings to refine the architecture.
Move into Phase 3 with explicit technical decisions instead of assumptions.
## 15. Project Checkpoint Format
Future updates to this document should use a simple checkpoint structure:
Checkpoint Date
Current Phase
Completed Since Last Checkpoint
Current Work
Important Decisions
Open Questions
Blockers
Next Step
## 16. Product Management View
The project should always be evaluated from two perspectives. The first is whether the technical system works. The second is whether it solves the intended user problem in a way that can be trusted and used.
Ask what user decision the work enables.
Check whether a new feature adds meaningful value.
Prefer evidence over assumptions.
Watch for scope expansion caused by technically interesting ideas.
Record trade-offs when choosing between accuracy, complexity, explainability and maintainability.
Keep the final investment workflow visible while technical development progresses.
## 17. Memory Definition of Done
This document is doing its job when a person joining the project can answer these questions without reading the entire chat history:
What are we building?
Why are we building it?
What has already been completed?
Which phase are we currently in?
What are we working on right now?
What important decisions have already been made?
What remains undecided?
What should happen next?
Status: Memory baseline established. This document should be maintained throughout the project.
## 18. Relationship With Other Project Documents
| Document | Relationship |
| --- | --- |
| PRD.md | Defines the product requirements and objectives. |
| ARCHITECTURE.md | Defines the technical system structure. |
| RULES.md | Defines how the project should be developed and managed. |
| PHASES.md | Defines the full execution roadmap. |
| DESIGN.md | Defines the product and interface design direction. |
| MEMORY.md | Keeps the current project state and continuity information. |
