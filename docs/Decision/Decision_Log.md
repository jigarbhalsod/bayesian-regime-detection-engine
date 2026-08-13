# 14. Decision Log

## Purpose

This document records the **important decisions made during Project 1A research** and the reasoning behind them.

The goal is to avoid losing track of:

- What we decided.
- Why we decided it.
- What is fixed.
- What still needs testing.
- What we intentionally left open.

> **Important: A decision marked USE still needs normal validation before becoming a permanent production choice.**

---

## Decision Status

| Status | Meaning |
|---|---|
| 🟢 **USE** | Current preferred direction; still subject to validation. |
| 🟡 **TEST** | Promising, but experiments must decide. |
| 🔴 **REJECT** | We do not plan to pursue it. |
| 🔵 **KEEP OPEN** | Interesting, but no need to decide now. |
| 🔒 **FIXED** | Locked project decision unless a strong reason appears to change it. |

---

# 1. Primary Market Target

**Decision:** 🔒 **NIFTY 50**

**Reason:** It is a strong and practical primary representation of the Indian equity market for Project 1A.

**Implication:** Regime detection and direction forecasting will initially focus on NIFTY 50.

---

# 2. Number of Regimes

**Decision:** 🔒 **Test K = 2–5**

**Reason:** We should not assume the market has one fixed number of regimes before testing the data.

**Possible structures:**

- K = 2 → Risk-On / Risk-Off
- K = 3 → Risk-On / Transition / Risk-Off
- K = 4 → Risk-On / Transition / Risk-Off / Recovery
- K = 5 → Risk-On / Transition / Risk-Off / Late-Cycle / Post-Shock

**Final K:** Will be selected using statistical quality, stability, financial meaning, and forecasting usefulness.

---

# 3. Regime Output

**Decision:** 🟢 **Use probabilistic regime outputs**

**Reason:** A market can sit between two states, so probabilities are more useful than only giving one hard label.

**Example:**

```text
Risk-On       15%
Transition    25%
Risk-Off      55%
Recovery       5%
```

---

# 4. Regime Definition

**Decision:** 🟢 **Data-discovered + economically interpreted**

**Reason:** The model should first discover market states from data, and then we interpret what those states mean financially.

**Principle:**

> **The model discovers the state; we interpret the state.**

---

# 5. HMM as the Main Regime Candidate

**Decision:** 🟢 **USE / Validate**

**Reason:** The market regime is hidden, while its effects can be observed through financial data, which matches the HMM problem structure well.

**Role:** Main candidate for hidden market regime detection.

---

# 6. Multivariate HMM

**Decision:** 🟢 **USE / Validate**

**Reason:** A market regime is not usually described by one signal, so using a compact set of financial variables should give the model a better picture of market conditions.

**Initial idea:** Use a small set of strong features rather than a huge feature set.

---

# 7. HMM Forward Filtering

**Decision:** 🟢 **USE**

**Reason:** It gives us a practical way to update regime probabilities as new market data arrives.

**Role:** Real-time regime updating.

---

# 8. HMM Historical Inference

**Decision:** 🟢 **Use Viterbi / Forward-Backward for research**

**Reason:** These methods can help us study historical regime sequences and validate the model, even though they use information differently from live detection.

**Important:** They must not be used as if future information were available in real-time prediction.

---

# 9. Bayesian HMM

**Decision:** 🟡 **TEST**

**Reason:** It can provide richer uncertainty information, but its additional complexity must be justified by better results.

**Role:** Advanced regime and uncertainty candidate.

---

# 10. Student-t HMM

**Decision:** 🟡 **TEST**

**Reason:** Financial returns can contain extreme movements that a simple Gaussian assumption may not handle well.

**Role:** Candidate for better handling of extreme market conditions.

---

# 11. RS-VAR

**Decision:** 🟡 **TEST**

**Reason:** RS-VAR can provide information HMM does not naturally provide, especially about how financial variables interact differently across regimes.

**Role:** Supporting model / financial relationship analysis.

---

# 12. Bayesian RS-VAR

**Decision:** 🔵 **KEEP OPEN**

**Reason:** It adds more uncertainty modelling but also significantly increases complexity, so we do not need it unless simpler RS-VAR proves useful and has an important limitation.

---

# 13. BNN

**Decision:** 🟡 **TEST**

**Reason:** BNN can be useful for nonlinear directional forecasting and uncertainty, but its extra complexity and potential latency must be justified.

**Role:** Directional forecasting component.

**Important:** BNN is not our primary hidden-regime detector.

---

# 14. Time-Series Foundation Models

**Decision:** 🟡 **TEST**

**Candidates:** Chronos, TimesFM

**Reason:** They may provide useful forecasting information, but general time-series performance does not guarantee useful performance on Indian equities.

**Role:** Forecasting benchmark / additional signal.

**Important:** They are not core regime detectors.

---

# 15. Particle Filtering

**Decision:** 🔵 **KEEP OPEN**

**Reason:** HMM Forward Filtering already gives us a practical real-time update mechanism, while Particle Filtering adds extra complexity and computation.

**Future use:** Consider only if experiments show that HMM-based online updating is not enough.

---

# 16. BOCPD

**Decision:** 🟡 **TEST**

**Reason:** BOCPD can provide an early warning that market behaviour may be changing, but it may also react to temporary market noise.

**Role:** Early-warning signal.

**Important:**

> BOCPD warns that something may be changing; it does not decide that the regime has definitely changed.

---

# 17. Early-Warning Feature

**Decision:** 🟢 **Include as a design direction**

**Reason:** An early-warning signal can make the system more cautious before a possible regime transition becomes obvious.

**Planned flow:**

```text
BOCPD
  ↓
Possible Change
  ↓
Situation becomes more cautious
  ↓
Model influence can change
```

This still needs experimental validation.

---

# 18. Ensemble Philosophy

**Decision:** 🔒 **Hybrid, situation-aware ensemble**

**Reason:** We do not want one model to permanently drive the whole system.

Different models should have more influence when they are better suited to the current situation, while the other models continue to provide supporting evidence.

---

# 19. Model Dominance

**Decision:** 🟡 **TEST situation-based dominance**

**Reason:** Different models have different strengths, so their influence should potentially change depending on the current market situation.

Examples:

| Situation | Possible stronger influence |
|---|---|
| Stable market | HMM |
| Possible transition | HMM + BOCPD |
| Extreme movement | Student-t HMM |
| Strong nonlinear behaviour | BNN |
| Strong variable interaction | RS-VAR |
| Forecast-heavy situation | BNN / TSFM |

These are hypotheses, not permanent rules.

---

# 20. Confidence Adjustment

**Decision:** 🟢 **Use confidence as an input to model influence**

**Reason:** A model's confidence should contribute to its influence, but confidence alone should not decide which model wins.

**Concept:**

> **Model confidence × situation-based influence = model contribution**

The exact formula will be decided during implementation and testing.

---

# 21. Weighted Averaging

**Decision:** 🟢 **USE as the first ensemble baseline**

**Reason:** It is simple, transparent, easy to test, and provides a strong baseline before adding more complicated combination methods.

---

# 22. Stacking

**Decision:** 🟡 **TEST**

**Reason:** Stacking can learn how much to rely on each model, but it adds another learning layer and can increase overfitting risk.

---

# 23. Bayesian Model Averaging

**Decision:** 🔵 **KEEP OPEN**

**Reason:** It is theoretically useful but currently appears more complex than necessary.

We will consider it only if simpler ensemble methods are not enough.

---

# 24. Model Disagreement

**Decision:** 🟢 **USE as an uncertainty signal**

**Reason:** Strong disagreement between models is useful information and can indicate that the final prediction should be treated more cautiously.

---

# 25. Calibration

**Decision:** 🟢 **USE**

**Reason:** Model confidence must be checked against actual outcomes before we rely on it for model weighting or decisions.

**Core question:**

> If the model says 70%, does the event actually happen roughly 70% of the time?

---

# 26. Adaptive Conformal Prediction

**Decision:** 🟡 **TEST**

**Reason:** It could provide additional prediction ranges or sets, but we first need to see whether the existing uncertainty methods are sufficient.

---

# 27. Initial Feature Strategy

**Decision:** 🔒 **Start with a compact set of 8–12 strong features**

**Reason:** Too many features can add noise, overfitting, redundancy, missing-data problems, and unnecessary complexity.

**Initial candidates:**

- NIFTY returns
- Momentum / trend
- Realized volatility
- India VIX
- Volume / turnover
- Market breadth
- USD/INR
- Crude oil
- Global equity signals

Additional features will be tested rather than automatically included.

---

# 28. Institutional Flows

**Decision:** 🟡 **TEST**

**Candidates:**

- FPI/FII flows
- DII flows

**Reason:** They may provide useful information about institutional activity, but their timing, quality, and incremental value must be tested.

---

# 29. Macro Features

**Decision:** 🟡 **TEST**

**Candidates:**

- Interest rates
- Inflation

**Reason:** They can describe broader market conditions, but their release timing and lower frequency make careful alignment necessary.

---

# 30. Slow-Moving / Optional Features

**Decision:** 🔵 **KEEP OPEN**

**Candidates:**

- GDP
- Sentiment
- Gold

**Reason:** They may provide useful information, but they are not necessary for the initial system and could add complexity without enough short-term value.

---

# 31. Data Timing Rule

**Decision:** 🔒 **No future information**

**Reason:** Financial research can easily suffer from look-ahead bias.

**Rule:**

> **At time `t`, the model can only use information that was actually available by time `t`.**

This applies especially to macroeconomic data that may be released after the period it describes.

---

# 32. Research-to-Decision Rule

**Decision:** 🔒 **Research does not automatically become implementation**

The process is:

```text
Research
   ↓
Candidate
   ↓
Experiment
   ↓
Compare
   ↓
Validate
   ↓
Keep / Remove
```

A simpler method that performs better should win.

---

# 33. Final Architecture Direction

**Decision:** 🟢 **Hybrid, situation-aware architecture**

Current conceptual structure:

```text
Market Data
     ↓
Feature Layer
     ↓
HMM / HMM Variants
     ↓
Regime + Uncertainty
     ↓
Situation Layer
     ↓
┌─────────┬─────────┬─────────┐
↓         ↓         ↓
BNN      RS-VAR     TSFM
↓         ↓         ↓
Direction Relationship Forecast
Signal      Signal     Signal
└─────────┬─────────┘
          ↓
     Hybrid Ensemble
          ↓
      Calibration
          ↓
Final Direction + Confidence
          ↓
    Decision Support
```

BOCPD provides an additional early-warning input to the situation layer.

---

# 34. What Is Still Open

The following are intentionally **not finalized**:

- Final K value.
- Final HMM variant.
- Whether Bayesian HMM is worth the complexity.
- Whether Student-t HMM is better than Gaussian HMM.
- Whether RS-VAR adds enough value.
- Whether BNN improves directional forecasting.
- Whether Chronos or TimesFM add useful information.
- Whether BOCPD improves the final system.
- Whether dynamic weighting beats simpler weighting.
- Whether stacking beats weighted averaging.
- Final feature list.
- Final mathematical weighting formula.
- Final production model stack.

These will be answered through experiments and validation.

---

# 35. Phase 2 Final Decision

**Decision:** 🟢 **Research complete**

We have:

- Defined the research strategy.
- Studied the major candidate methods.
- Compared their strengths and weaknesses.
- Defined our feature direction.
- Defined our uncertainty approach.
- Created the hybrid ensemble philosophy.
- Classified candidates as USE / TEST / KEEP OPEN.
- Identified what still needs experimental proof.

---

# 36. Next Step

The next phase is:

> **Phase 3, Solution Architecture**

The purpose of Phase 3 is to turn these research decisions into a clear technical system design.

---

## Decision Log Principle

> **Record the decision, record why we made it, and clearly separate what is fixed from what still needs proof.**

This document should be updated whenever a major Project 1A decision is made or an existing decision changes.
