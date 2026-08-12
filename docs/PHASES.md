PROJECT 1A
PROJECT PHASES & ROADMAP
Bayesian Regime Detection Engine for Equity Direction Forecasting
Master execution roadmap and phase definitions

## 1. Purpose
This document provides the master roadmap for Project 1A. It breaks the project into clear phases so that the work can move from business understanding and research into architecture, implementation, validation and final delivery in a controlled way.
The roadmap is deliberately structured so that we do not start building advanced models before the business problem, data, regime definitions and evaluation approach are understood. Each phase has a clear purpose, expected outputs and a simple definition of done.
## 2. Project Roadmap at a Glance
| Phase | Name | Main purpose |
| --- | --- | --- |
| Phase 1 | Business & BFSI Understanding | Understand the business problem, users, market context, workflow, regulation and scope. |
| Phase 2 | Research & Literature | Study the relevant financial, statistical, Bayesian and machine learning approaches before implementation. |
| Phase 3 | Solution Architecture | Translate the requirements and research findings into a practical technical architecture. |
| Phase 4 | Data Layer | Define data sources, schemas, point-in-time storage and data contracts. |
| Phase 5 | Data Engineering | Build the pipelines that collect, clean, validate and transform the data. |
| Phase 6 | Feature Engineering | Create the market, breadth, flow, volatility and macro features used by the system. |
| Phase 7 | Financial Analysis | Study the financial behaviour and relationships represented by the features. |
| Phase 8 | Baseline Regime Engine | Build the first simple and explainable regime detection baseline. |
| Phase 9 | Advanced Models | Implement and evaluate the more advanced Bayesian, regime-switching, neural and foundation-model approaches. |
| Phase 10 | Ensemble & Uncertainty | Combine model outputs and add uncertainty and calibration. |
| Phase 11 | Validation | Test the complete system using time-aware and financially meaningful evaluation. |
| Phase 12 | Decision Support Layer | Turn the model output into useful information for investment decision-making. |
| Phase 13 | Deployment | Package and expose the validated system for reliable use. |
| Phase 14 | Documentation | Complete technical, product, model and governance documentation. |
| Phase 15 | Finalization | Perform the final review, cleanup, demonstration and delivery. |

## 3. How the Phases Fit Together
Understand the Problem
        ↓
Research the Possible Solutions
        ↓
Design the System
        ↓
Build the Data Foundation
        ↓
Create Financial Features
        ↓
Build a Baseline
        ↓
Add Advanced Models
        ↓
Combine Models + Uncertainty
        ↓
Validate Everything
        ↓
Turn Results Into Decision Support
        ↓
Deploy + Document + Finalize
## 4. Phase 1, Business & BFSI Understanding
Phase 1 establishes what the system is supposed to solve and how it fits into an Indian asset-management environment. This prevents the technical work from becoming disconnected from the actual investment workflow.
Understand the asset-management and mutual-fund context.
Understand the Indian equity-market environment relevant to the product.
Define the five market regimes.
Identify users and stakeholders.
Document the decision workflow.
Identify realistic use cases.
Understand relevant regulatory and governance considerations.
Define success metrics.
Set the product scope and boundaries.
Key deliverables: PRD, Decision Workflow, initial user and use-case documentation, scope and success criteria.
Definition of Done: A clear answer exists for what problem is being solved, for whom, in what business context, and what the product will and will not do.
## 5. Phase 2, Research & Literature
Phase 2 is about understanding the methods before choosing how to implement them. The goal is not to collect papers for the sake of having references. The goal is to understand which approaches are suitable for this specific regime-detection problem and why.
Study market regime detection and regime-switching methods.
Study HMM and Bayesian HMM approaches.
Study Bayesian neural networks and uncertainty decomposition.
Study regime-switching VAR models.
Study time-series foundation models and their possible use in regime classification.
Study sequential Monte Carlo and particle filtering.
Study BOCPD and changepoint detection.
Study model ensembling and Bayesian Model Averaging.
Study time-aware stacking and probabilistic evaluation.
Study conformal prediction under temporal dependence and distribution shift.
Study calibration and uncertainty evaluation.
Study financial regime features and economic transmission mechanisms.
Key deliverables: research notes, literature matrix, method comparison, identified gaps and recommended approaches.
Definition of Done: We can explain what each candidate method does, where it fits, its strengths and weaknesses, and why we may or may not use it.
## 6. Phase 3, Solution Architecture
Phase 3 converts the product requirements and research findings into a modular technical design. It should make clear how data, models, uncertainty, monitoring and decision support connect.
Finalize system components and boundaries.
Define data and model interfaces.
Define batch and online processing paths.
Define model output contracts.
Define lineage and versioning.
Define repository and module structure.
Select the initial technology stack.
Record important architecture decisions.
Key deliverable: Architecture baseline and architecture decision records.
Definition of Done: Another developer should be able to understand the major system components and how information moves between them.
## 7. Phase 4, Data Layer
Phase 4 defines the data foundation before large-scale engineering begins. The focus is on knowing exactly what data is required, what it means and how it will be stored.
Finalize required data sources.
Define source-level schemas.
Define point-in-time requirements.
Define timestamps and frequency rules.
Define raw, processed and snapshot storage.
Define data provenance and versioning.
Define data contracts.
Key deliverable: Data specification, schemas and source catalogue.
Definition of Done: Every required input has a defined source, meaning, frequency, timestamp rule and storage plan.
## 8. Phase 5, Data Engineering
Phase 5 turns the data design into working pipelines. The focus is reliable ingestion, cleaning, validation and reproducible transformation.
Build ingestion pipelines.
Normalize source formats.
Handle missing values and anomalies.
Apply point-in-time rules.
Create reproducible data snapshots.
Add data-quality checks.
Build processed datasets for feature engineering.
Key deliverable: Reproducible data pipeline and validated datasets.
Definition of Done: The required datasets can be regenerated consistently and pass defined quality checks.
## 9. Phase 6, Feature Engineering
Phase 6 converts the cleaned data into variables that describe market behaviour. The features should be useful statistically and understandable economically.
Market returns and trend features.
Momentum and volatility features.
Market breadth features.
Capital-flow features.
Macro and liquidity features.
Cross-market relationships.
Feature quality and stability checks.
Feature documentation and lineage.
Key deliverable: Versioned feature set and feature catalogue.
Definition of Done: Every production feature has a clear definition, source, transformation and rationale.
## 10. Phase 7, Financial Analysis
Phase 7 studies what the features actually tell us about the Indian market. This phase creates the bridge between feature engineering and model building.
Study relationships between market variables.
Examine behaviour across different market environments.
Study breadth, flows, volatility and macro relationships.
Check whether proposed features behave consistently over time.
Identify possible regime-specific patterns.
Record economically meaningful observations.
Key deliverable: Financial analysis report and feature interpretation notes.
Definition of Done: We understand the major financial relationships that the models are expected to learn.
## 11. Phase 8, Baseline Regime Engine
Phase 8 establishes a simple baseline before advanced modelling. The baseline gives us something understandable to compare against and helps identify whether additional complexity is actually useful.
Implement a simple regime classification baseline.
Define baseline training and evaluation procedure.
Measure baseline probabilistic performance.
Inspect regime stability.
Record baseline limitations.
Key deliverable: Reproducible baseline model and baseline evaluation report.
Definition of Done: We have a working reference point against which advanced models can be compared.
## 12. Phase 9, Advanced Models
Phase 9 implements the advanced modelling approaches that are justified by the research and architecture.
Bayesian HMM.
Regime-Switching VAR.
Bayesian Neural Network.
Time-series foundation model components.
Sequential Monte Carlo inference.
BOCPD for changepoint detection.
Each model should be evaluated independently before it is added to the ensemble.
Key deliverable: Independent model implementations and evaluation results.
Definition of Done: Each selected model has a reproducible implementation, evaluation result and documented limitations.
## 13. Phase 10, Ensemble & Uncertainty
Phase 10 combines the individual model outputs and adds the uncertainty and calibration layer that makes the system useful for decision support.
Implement the selected ensemble method.
Track model weights and disagreement.
Combine regime probabilities.
Estimate epistemic and aleatoric uncertainty.
Implement the selected conformal calibration method.
Monitor empirical coverage.
Define the final combined output contract.
Key deliverable: Combined regime engine with probability, uncertainty and calibration output.
Definition of Done: A single standardized regime assessment can be generated with probability, uncertainty, calibration and lineage information.
## 14. Phase 11, Validation
Phase 11 tests whether the complete system works as intended. Validation should cover statistical performance, calibration, stability, robustness and practical usefulness.
Walk-forward evaluation.
Probabilistic scoring.
Calibration evaluation.
Regime stability analysis.
Transition and changepoint evaluation.
Stress and sub-period testing.
Drift and robustness analysis.
Model-health evaluation.
Decision-support impact analysis.
Key deliverable: Full validation report.
Definition of Done: The team can explain where the system works, where it fails, how reliable its probabilities are and what limitations remain.
## 15. Phase 12, Decision Support Layer
Phase 12 turns the validated model output into something an investment professional can actually use. This is where the project becomes a decision-support product rather than remaining only a modelling experiment.
Create the regime summary.
Show probabilities and uncertainty.
Show model health.
Show the main drivers.
Show changes from previous assessments.
Translate conviction into potential allocation implications.
Apply portfolio constraints and no-trade logic.
Generate the Investment Committee artefact.
Key deliverable: Decision-support interface, reporting structure and allocation-support logic.
Definition of Done: A user can understand the current market view, its confidence, the reasons behind it and its possible portfolio implication.
## 16. Phase 13, Deployment
Phase 13 packages the validated system so that it can run reliably outside the research environment.
Package the inference pipeline.
Expose required APIs or services.
Set up configuration and secrets management.
Set up logging and monitoring.
Define model and data version handling.
Test deployment behaviour.
Document operational procedures.
Key deliverable: Deployable decision-support service.
Definition of Done: The validated system can be run reliably with documented operational steps.
## 17. Phase 14, Documentation
Phase 14 brings the technical, product and governance documentation together. Documentation should make the project understandable to someone who did not build it.
Finalize PRD.
Finalize architecture documentation.
Finalize project rules.
Finalize model documentation.
Finalize data documentation.
Finalize validation report.
Finalize decision workflow documentation.
Finalize deployment and operational documentation.
Finalize model and audit lineage documentation.
Key deliverable: Complete project documentation package.
Definition of Done: A reviewer can understand the product, architecture, data, models, validation, deployment and governance without relying on the development team's memory.
## 18. Phase 15, Finalization
Phase 15 is the final quality and delivery pass. It is not a new development phase. Its purpose is to make sure the work is coherent, reproducible and presentable.
Run final tests.
Check documentation against the implemented system.
Remove dead code and unnecessary files.
Confirm repository structure.
Confirm model and data versions.
Review known limitations.
Prepare final demonstration.
Prepare final report and handover material.
Key deliverable: Final project package.
Definition of Done: The repository, documentation, results and demonstration tell the same story and can be handed over cleanly.
## 19. Phase Dependencies
The phases are not completely isolated. Some phases can overlap, but the major dependency chain is:
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
                                      ↓
                             Phase 8 → Phase 9
                                      ↓
                              Phase 10 → Phase 11
                                      ↓
                              Phase 12 → Phase 13
                                      ↓
                              Phase 14 → Phase 15
For example, documentation should not wait until the end to exist. Working documentation is maintained throughout the project, while Phase 14 is the point where the complete documentation package is finalized.
## 20. Major Project Gates
| Gate | After | Question |
| --- | --- | --- |
| Business Gate | Phase 1 | Do we understand what we are building and why? |
| Research Gate | Phase 2 | Do we understand the methods well enough to choose an approach? |
| Architecture Gate | Phase 3 | Can the system be built as a coherent, modular product? |
| Data Gate | Phase 5 | Can we reliably produce valid point-in-time data? |
| Baseline Gate | Phase 8 | Do we have a meaningful reference model? |
| Model Gate | Phase 10 | Do the advanced models add value and provide usable uncertainty? |
| Validation Gate | Phase 11 | Is the system reliable enough for decision support? |
| Product Gate | Phase 12 | Can an intended user actually use the output? |
| Release Gate | Phase 15 | Is the complete project ready to present and hand over? |

## 21. Project Management Principles
Do not build ahead of understanding. Resolve important business and research questions before committing to complex implementation.
Baseline before complexity. Build something simple enough to understand before comparing it with advanced methods.
Measure before claiming. A model is not better because it is more complex. It needs evidence.
Document decisions. Important choices should be recorded with their reasoning.
Keep scope controlled. New ideas should be evaluated against the project objective before being added.
Work in checkpoints. Each major phase should produce something reviewable.
Keep the user visible. Technical progress should always connect back to the investment decision workflow.
## 22. Current Project Position
The project has completed the main Business and BFSI Understanding work for Phase 1, including the decision workflow documentation. The project documentation baseline is also being established through the PRD, Architecture and Rules documents.
The next major work area is Phase 2, Research and Literature. The purpose of this phase is to build enough technical and financial understanding to make informed architecture and modelling choices rather than selecting methods only because they appear sophisticated.
## 23. Roadmap Definition of Done
The roadmap is considered complete when:
All project phases are defined.
Each phase has a clear purpose.
Each phase has expected outputs.
Each phase has a practical Definition of Done.
Major dependencies are understood.
Major project gates are defined.
The current project position is recorded.
The roadmap can be updated without changing the overall product objective.
Status: Project Phases and Roadmap baseline complete.
## 24. Relationship With Other Project Documents
| Document | Relationship |
| --- | --- |
| PRD.md | Defines what the product should achieve. |
| ARCHITECTURE.md | Defines how the system is structured. |
| RULES.md | Defines how the project should be built and managed. |
| PHASES.md | Defines when and in what sequence the major work should happen. |
| DESIGN.md | Defines the user-facing visual and interaction language. |
| MEMORY.md | Records the current project state and continuity information. |
