PROJECT 1A
PRODUCT & INTERFACE DESIGN
Bayesian Regime Detection Engine for Equity Direction Forecasting
Visual language, dashboard structure and user interaction baseline

## 1. Purpose
This document defines the initial product and interface design direction for Project 1A. The design should help investment users understand the market regime, the confidence behind the assessment, the reasons supporting it and the possible decision implications without making the product look like an automated trading terminal.
The design is intentionally simple at this stage. Detailed screen layouts and visual specifications should be refined after the decision-support requirements and user workflows are fully tested.
## 2. Design Goals
Make the current market regime easy to understand within a few seconds.
Make probability and uncertainty visible together.
Show why the system reached its current view.
Make model health visible without overwhelming the main decision.
Keep investment implications separate from the model prediction itself.
Support quick review by portfolio managers and deeper review by analysts.
Make the interface suitable for an investment and governance environment rather than a retail trading app.
Keep the visual language consistent across dashboard, reports and Investment Committee artefacts.
## 3. Primary User Experience
The primary user should be able to answer five questions quickly:
What is the market regime right now?
How probable is that regime?
How confident should I be in the call?
Why does the system believe this?
Does this information suggest any decision implication?
## 4. Information Hierarchy
The most important information should appear first. Supporting evidence should become more detailed as the user moves down the page.
## 1. Current Regime + Probability
2. Uncertainty + Model Health
3. Probability Distribution Across Regimes
4. Main Drivers
5. Model Agreement
6. Recent Regime History
7. Possible Allocation Implication
8. Detailed Diagnostics and Lineage
## 5. Dashboard Structure
A first version of the main dashboard can follow this structure:
┌─────────────────────────────────────────────────────┐
│ Current Market Regime                               │
│ Late Cycle                         52%              │
│ Confidence: Moderate       Health: Healthy         │
├─────────────────────────────────────────────────────┤
│ Regime Probability Distribution                     │
│ Risk On       14%                                  │
│ Late Cycle   52%                                  │
│ Transitional 27%                                  │
│ Risk Off      5%                                  │
│ Post Shock    2%                                  │
├─────────────────────────────────────────────────────┤
│ Why this call?                                      │
│ Breadth ↓   VIX ↑   FII Flow ↓                     │
├─────────────────────────────────────────────────────┤
│ Model Agreement        │ Uncertainty                │
│ HMM      Late Cycle    │ Epistemic: Moderate       │
│ RS-VAR   Late Cycle    │ Aleatoric: High           │
│ BNN      Transitional  │ Set: LC, Transitional     │
├─────────────────────────────────────────────────────┤
│ Regime History                                      │
├─────────────────────────────────────────────────────┤
│ Decision Support                                   │
│ Possible small de-risking tilt, subject to rules   │
└─────────────────────────────────────────────────────┘
## 6. Main Dashboard Components
### 6.1 Current Regime Card
The primary card should show the dominant regime and its probability. It should also show a short confidence status and current model-health status.
Dominant regime
Probability
Confidence level
Health status
Assessment timestamp
### 6.2 Regime Probability View
A simple chart or horizontal bar view should show the probability assigned to all five regimes. The user should not need to open another screen to see the alternatives.
All five regimes
Probability values
Clear comparison
Historical comparison when useful
### 6.3 Uncertainty Panel
Uncertainty should be presented next to probability rather than hidden in a technical diagnostics section.
Epistemic uncertainty
Aleatoric uncertainty
Conformal prediction set
Coverage information where relevant
### 6.4 Drivers Panel
The drivers panel explains the main evidence behind the current call.
Top positive contributors
Top negative contributors
Feature names
Direction of effect
Plain language explanation
### 6.5 Model Agreement Panel
Users should be able to see whether different model families agree or disagree.
Model name
Dominant regime
Model probability
Model status
Agreement or disagreement
### 6.6 Regime History
A historical view should show how the regime probabilities and dominant regime have changed over time.
Regime timeline
Probability history
Transition points
Changepoint markers
### 6.7 Decision Support Panel
The decision support section should translate the model view into a possible portfolio implication without presenting it as an automatic instruction.
Conviction
Possible allocation implication
Relevant constraints
No-trade or hysteresis status
### 6.8 Health and Diagnostics
A deeper diagnostics section can be used by analysts and model governance users.
Drift
BOCPD status
Online/batch gap
Coverage
Out-of-distribution status
Model version
### 6.9 Lineage and Audit
Detailed lineage should be available when a user needs to reconstruct a historical call.
Data snapshot
Feature version
Model version
Inference timestamp
Ensemble information
Decision record
## 7. Visual Language
The visual language should feel analytical, calm and professional. The interface should help users assess risk and uncertainty rather than encourage rapid trading behaviour.
### 7.1 Colour Approach
Colour should communicate meaning consistently. It should not be used simply to make the dashboard visually busy.
| Use | Meaning | Design guidance |
| --- | --- | --- |
| Regime status | Current market state | Use one consistent visual treatment for each regime. |
| Risk warning | Elevated concern or unhealthy state | Use a restrained warning treatment. |
| Healthy status | Model operating normally | Use a subtle positive indicator. |
| Uncertainty | Confidence is limited | Use neutral visual emphasis rather than making uncertainty look like a failure. |
| Blocked action | Health or constraints prevent downstream action | Make the reason clear and visible. |

### 7.2 Suggested Regime Semantics
Risk On: Market conditions generally supportive of risk taking.
Late Cycle: Conditions remain constructive but signs of weakening or maturity are increasing.
Transitional: Evidence suggests the market is moving between broader states.
Post Shock: Market is responding to or recovering from a significant shock.
Risk Off: Conditions indicate a broad deterioration in risk appetite.
## 8. Typography
Typography should prioritize readability over visual novelty. The dashboard may contain a lot of numerical information, so users should be able to scan headings, values and explanations quickly.
Use a clean sans-serif typeface.
Use a clear hierarchy between page title, section heading, key metric and supporting text.
Use larger type for the dominant regime and probability.
Use smaller type for timestamps, model versions and technical metadata.
Avoid excessive capitalization.
Keep numerical formatting consistent throughout the product.
## 9. Data Visualization Rules
Use charts only when they help the user compare or understand a pattern.
Avoid decorative charts that do not support a decision.
Show exact values alongside important probability visuals.
Keep regime colours and labels consistent across every chart.
Make uncertainty visible in time-series views where practical.
Use annotations for important changepoints or major regime transitions.
Avoid presenting a probability chart in a way that suggests certainty.
## 10. User Interaction Principles
Scan first. A portfolio manager should understand the main state quickly.
Drill down when needed. Analysts should be able to inspect the evidence without forcing every user to see it.
Context before action. The possible allocation implication should appear after the model evidence.
No hidden uncertainty. Important uncertainty should never be buried behind a technical menu.
Explain changes. When the regime changes, the interface should make the change visible.
Keep decisions explicit. A model call and a human decision should be visually distinguishable.
## 11. Investment Committee View
The Investment Committee view should be more concise than the analyst dashboard. It should focus on the information needed to understand the current assessment and the decision context.
Current View
    ↓
Probability Distribution
    ↓
Confidence + Health
    ↓
Key Drivers
    ↓
What Changed
    ↓
Scenario / Transition Context
    ↓
Potential Allocation Implication
    ↓
Lineage / Review Information
## 12. Analyst View vs Portfolio Manager View
| Area | Analyst | Portfolio Manager |
| --- | --- | --- |
| Regime probability | Detailed | Summary |
| Model agreement | Detailed | High level |
| Feature drivers | Detailed | Top drivers |
| Uncertainty | Full detail | Decision relevant summary |
| Model health | Full diagnostics | Status and implications |
| Historical regime view | Detailed | Decision relevant trend |
| Allocation implication | Analytical | Primary decision context |
| Lineage | Full access | Available when needed |

## 13. Responsive and Practical Behaviour
The main regime and probability should remain visible on smaller screens.
Technical diagnostics can collapse into secondary sections.
Tables should remain readable without excessive horizontal scrolling.
Charts should retain their key information when resized.
Important alerts should not depend only on colour.
## 14. Accessibility and Usability
Do not rely on colour alone to communicate regime status.
Use text labels alongside visual indicators.
Maintain sufficient contrast.
Use readable font sizes.
Provide exact values alongside charts where precision matters.
Keep terminology consistent with the project documentation.
## 15. Error and Empty States
The product should clearly explain what happened when data or model output is unavailable.
Missing data: identify the affected source and time.
Model failure: show that the model failed instead of displaying an old value as current.
Health failure: explain why allocation support has been blocked or flagged.
Insufficient history: state that the requested analysis cannot yet be calculated reliably.
Delayed data: show the latest available timestamp.
## 16. Example User Scenario
A portfolio manager opens the dashboard before reviewing the day's allocation decisions. The first thing visible is the current regime, Late Cycle, with a probability of 52%. The next section shows that Transitional has 27% probability and the remaining regimes have lower probabilities.
The user then sees moderate confidence, a healthy model status and a conformal prediction set containing Late Cycle and Transitional. The drivers section highlights weakening breadth, higher VIX and negative FII flow as important contributors.
The manager can then inspect model agreement and the recent regime history. Only after reviewing this evidence does the interface show the possible allocation implication, such as a small de-risking tilt subject to portfolio constraints.
## 17. Design Boundaries
The dashboard should not look like a retail trading terminal.
The product should not encourage users to act on a single probability number.
Allocation implications should not visually overpower the evidence behind them.
Technical model diagnostics should not clutter the primary decision view.
Visual design should not imply that the system can predict the market with certainty.
## 18. Design System Components
The eventual design system should contain reusable components such as:
Regime status card.
Probability distribution chart.
Uncertainty indicator.
Model agreement table.
Driver explanation card.
Regime history chart.
Health status indicator.
Allocation implication card.
Decision record panel.
Lineage drawer or detail view.
## 19. Design Decisions Still Open
Final brand colour palette.
Exact typography family.
Dashboard technology.
Final chart library.
Exact visual treatment for each regime.
Best representation of epistemic and aleatoric uncertainty.
Best way to show conformal prediction sets.
Exact Investment Committee report layout.
User permissions and role-specific navigation.
## 20. Design Principles
Clarity over decoration. Every visual element should help the user understand something important.
Probability with context. A number should be shown with the information needed to interpret it.
Evidence before action. The interface should show the reasoning before the possible decision implication.
Calm risk communication. The design should communicate risk without creating unnecessary urgency.
Progressive detail. Simple information comes first, deeper diagnostics remain available.
Consistent language. Regimes, metrics and model states should use the same names everywhere.
Human decision visibility. The interface should clearly separate model output from human action.
## 21. Design Definition of Done
The design baseline is considered complete when it clearly defines:
The primary user experience.
The information hierarchy.
The main dashboard structure.
The core interface components.
The visual language and typography direction.
The main visualization rules.
The Investment Committee view.
The difference between analyst and portfolio manager views.
Error and health-state behaviour.
Accessibility expectations.
The major design boundaries.
Which visual decisions remain open for later prototyping.
Status: Product and Interface Design baseline complete. Detailed prototypes should be created after the user workflows and decision-support requirements are tested.
## 22. Relationship With Other Project Documents
| Document | Relationship |
| --- | --- |
| PRD.md | Defines what the product should achieve and who it supports. |
| ARCHITECTURE.md | Defines the system components that provide the information shown in the interface. |
| RULES.md | Defines the working and engineering rules the design must respect. |
| PHASES.md | Defines when detailed design and prototyping should happen. |
| DESIGN.md | Defines the visual and interaction language for the product. |
| MEMORY.md | Records the current design decisions and active design work. |
