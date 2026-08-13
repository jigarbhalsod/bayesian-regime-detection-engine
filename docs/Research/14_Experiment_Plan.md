# 15. Experiment Plan

## Purpose

This document defines **how we will test the research ideas from Phase 2** before deciding what becomes part of the final Project 1A system.

The main rule is:

> **Do not choose a model because it sounds good; compare it fairly and keep only what proves useful.**

---

# 1. What We Need to Prove

The experiments should answer five main questions:

1. **Can we detect meaningful market regimes?**
2. **Which model detects them best?**
3. **Can regime information improve NIFTY 50 direction forecasting?**
4. **Can early warnings and uncertainty improve the system?**
5. **Does our hybrid ensemble outperform simpler approaches?**

---

# 2. Experimental Philosophy

We will follow:

```text
Hypothesis
    ↓
Define experiment
    ↓
Use same data
    ↓
Use time-aware validation
    ↓
Run baseline
    ↓
Run candidate model
    ↓
Compare
    ↓
Analyze failures
    ↓
Decision
```

A model only becomes a preferred component if the evidence supports it.

---

# 3. Fair Comparison Rule

When comparing models, we should keep the following consistent wherever possible:

- Same historical period
- Same target
- Same train/test logic
- Same information availability rules
- Same evaluation periods
- Same core metrics
- Same leakage controls

This prevents one model from receiving an unfair advantage.

---

# 4. Experiment Groups

We will organize experiments into:

| Group | Purpose |
|---|---|
| **E1** | Baseline market behaviour |
| **E2** | Regime count |
| **E3** | HMM regime detection |
| **E4** | HMM variants |
| **E5** | RS-VAR |
| **E6** | Direction forecasting |
| **E7** | Online detection |
| **E8** | Early warning |
| **E9** | Uncertainty & calibration |
| **E10** | Ensemble |
| **E11** | Feature contribution |
| **E12** | Final system validation |

---

# 5. E1, Baseline Experiments

## Goal

Create simple reference models that advanced methods must beat.

Potential baselines:

- Buy-and-hold context
- Simple return-direction rule
- Rolling volatility threshold
- Rule-based regimes
- Simple clustering
- Simple directional forecasting model

## Question

> **Do our advanced models actually add value compared with simple approaches?**

## Decision

Every advanced model must be compared against at least one sensible baseline.

---

# 6. E2, Regime Count Experiment

## Goal

Test:

**K = 2, 3, 4, 5**

## Question

> **Which number of regimes gives the best balance between statistical quality, stability, financial meaning, and forecasting usefulness?**

## Compare

| K | Possible interpretation |
|---|---|
| 2 | Risk-On / Risk-Off |
| 3 | Risk-On / Transition / Risk-Off |
| 4 | Risk-On / Transition / Risk-Off / Recovery |
| 5 | Risk-On / Transition / Risk-Off / Late-Cycle / Post-Shock |

These labels are only examples.

The states will first be discovered from data and then interpreted.

## Decision

Final K is **not fixed yet**.

---

# 7. E3, HMM Regime Detection

## Goal

Test whether HMM can identify meaningful market states.

## Inputs

Start with a compact feature set such as:

- NIFTY returns
- Momentum
- Realized volatility
- India VIX
- Volume
- Market breadth

Additional features can be tested later.

## Outputs

For each time point:

- Regime probabilities
- Most likely regime
- Transition information

## Questions

- Are regimes stable?
- Do they have clear financial meaning?
- Are they persistent?
- Do they correspond to different market behaviour?
- Can they identify known stress periods?

---

# 8. E4, HMM Variant Comparison

We will compare:

1. Standard Gaussian HMM
2. Student-t HMM
3. Bayesian HMM

## Goal

Determine whether the more advanced versions actually improve the system.

## Compare

- Regime quality
- Stability
- Extreme-event behaviour
- Uncertainty
- Forecasting usefulness
- Computational cost
- Complexity

## Decision rule

> **Use the simplest HMM variant that provides meaningful improvement.**

---

# 9. E5, RS-VAR Experiment

## Goal

Test whether RS-VAR adds information that HMM does not.

## Questions

- Does it capture different relationships between financial variables?
- Does it provide useful regime-specific information?
- Does it improve forecasting?
- Is the added complexity justified?

## Comparison

```text
HMM only
   vs
HMM + RS-VAR
```

If RS-VAR does not add meaningful information, it should not be kept merely because it is more sophisticated.

---

# 10. E6, Direction Forecasting

## Goal

Test whether regime information improves NIFTY 50 direction forecasting.

## Candidate models

- Simple baseline
- BNN
- Chronos
- TimesFM
- Other simple forecasting baseline if needed

## Key comparison

```text
Forecasting without regime information
            vs
Forecasting with regime information
```

This is one of the most important experiments in the entire project.

Because the actual objective is not merely:

> "Detect regimes."

It is:

> **"Use regime information to improve equity-direction forecasting."**

---

# 11. E7, Online Regime Updating

## Goal

Test whether the system can update its regime belief as new data arrives.

## Main method

**HMM Forward Filtering**

## Question

> **Can the model update the current regime using only information available up to that moment?**

## Important rule

No future information is allowed.

---

# 12. E8, BOCPD Early-Warning Experiment

## Goal

Test whether BOCPD can provide useful early warning before or around genuine regime changes.

## Compare

```text
HMM alone
       vs
HMM + BOCPD
```

## Measure

- Detection speed
- False alarms
- Missed changes
- Stability
- Impact on forecasting
- Impact on model weighting

## Key question

> **Does the early warning actually improve the final system?**

If it only produces noise, we should remove it.

---

# 13. E9, Uncertainty & Calibration

## Goal

Check whether model probabilities and confidence scores can be trusted.

## Test

- Regime probabilities
- Direction probabilities
- Brier Score
- Log Loss
- Calibration curves
- ECE

## Compare

```text
Raw confidence
      vs
Calibrated confidence
```

## Key question

> **Does calibration make confidence more reliable and useful for model weighting?**

---

# 14. E10, Ensemble Experiments

We will compare ensemble strategies in increasing complexity.

### A. Equal Weighting

All models receive equal influence.

### B. Fixed Weighted Average

Models receive predefined weights.

### C. Situation-Based Weighting

Model influence changes depending on the current market situation.

### D. Stacking

A separate model learns how to combine the predictions.

## Comparison

```text
Single best model
      vs
Equal ensemble
      vs
Weighted ensemble
      vs
Situation-based ensemble
      vs
Stacking
```

## Key question

> **Does complexity actually improve performance?**

---

# 15. E11, Feature Contribution

## Goal

Find out which features actually help.

Potential groups:

- Price/returns
- Volatility
- Volume
- Breadth
- India VIX
- USD/INR
- Crude oil
- Global equity signals
- Institutional flows
- Interest rates
- Inflation

## Approach

Test:

```text
Core features
      vs
Core + additional feature group
```

We should not add features only because they improve one historical period.

They must show useful performance across unseen periods.

---

# 16. E12, Final System Validation

After individual experiments, test the complete candidate system.

Possible comparison:

```text
Baseline
   ↓
HMM
   ↓
HMM + Forecast Model
   ↓
HMM + Forecast Model + BOCPD
   ↓
Hybrid Ensemble
   ↓
Hybrid Ensemble + Calibration
```

The goal is to understand **which additions actually improve the complete system**.

---

# 17. Main Evaluation Metrics

## Regime Detection

Potential metrics:

- Regime stability
- Regime persistence
- State separation
- Historical event alignment
- Transition detection quality
- Economic interpretability

There may not be a single perfect "regime accuracy" metric because the true regime is not directly observable.

---

## Direction Forecasting

Potential metrics:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Directional hit rate
- Log Loss
- Brier Score

We should not rely on accuracy alone.

---

## Uncertainty

Use:

- Brier Score
- Log Loss
- Calibration curve
- ECE

---

## Early Warning

Use:

- Detection delay
- False alarm rate
- Missed-event rate
- Warning stability

---

## Practicality

Also record:

- Training time
- Inference latency
- Memory/computation
- Data requirements
- Implementation complexity

---

# 18. Financial Usefulness

Technical metrics are not enough.

We also need to ask:

> **Does the model provide useful information for the actual decision-support objective?**

A model can have a good statistical score but provide little practical value.

Therefore, final evaluation should consider:

- Forecast usefulness
- Stability
- Behaviour during stress periods
- Confidence quality
- Model disagreement
- Practical complexity

---

# 19. Validation Strategy

Financial time series must be validated chronologically.

We should avoid random train/test splitting when it would allow future information to influence past predictions.

Preferred structure:

```text
Past
 ↓
Train
 ↓
Validate
 ↓
Test
 ↓
Move forward
 ↓
Repeat
```

This should eventually lead to a **walk-forward / rolling evaluation setup**.

---

# 20. Stress-Period Testing

The system should be examined during important market conditions such as:

- Major crashes
- High-volatility periods
- Fast recoveries
- Long trends
- Sideways markets
- Sudden shocks

The purpose is to see whether the system behaves differently when market conditions change.

---

# 21. Leakage Checks

Before trusting any experiment, ask:

1. Was any future price information used?
2. Were future-calculated indicators accidentally included?
3. Were macro variables used before their release?
4. Were features normalized using future data?
5. Was the test period used while choosing model settings?
6. Did the ensemble weighting learn from future outcomes?

If yes, the experiment must be corrected.

---

# 22. Model Selection Rule

We should not select a model using one metric.

Use:

**Performance + Financial Meaning + Uncertainty + Robustness + Interpretability + Practicality**

A model that is slightly more accurate but dramatically more complex may not be the better choice.

---

# 23. Experiment Record Template

Every experiment should record:

```text
Experiment ID:
Date:
Hypothesis:

Model:
Baseline:

Dataset:
Time period:

Features:
Target:

Validation method:

Parameters:

Metrics:

Result:

Observed behaviour:

Failure cases:

Comparison:

Decision:
USE / TEST / REJECT / KEEP OPEN

Reason:

Next experiment:
```

This makes our research reproducible.

---

# 24. Experiment Naming

Use simple IDs:

```text
E01
E02
E03
...
```

For sub-experiments:

```text
E03.1
E03.2
E03.3
```

Example:

```text
E02.1 → K = 2
E02.2 → K = 3
E02.3 → K = 4
E02.4 → K = 5
```

---

# 25. Initial Experiment Order

We should not test everything randomly.

Recommended order:

```text
E01
Baselines
 ↓
E02
K = 2–5
 ↓
E03
Standard HMM
 ↓
E04
HMM Variants
 ↓
E05
RS-VAR
 ↓
E06
Direction Forecasting
 ↓
E07
Online Updating
 ↓
E08
BOCPD
 ↓
E09
Calibration
 ↓
E10
Ensemble
 ↓
E11
Feature Contribution
 ↓
E12
Final Validation
```

This prevents us from building a complicated ensemble before knowing whether the individual components are useful.

---

# 26. Important Experiment Principle

Do not ask:

> "Can we make this model work?"

Ask:

> **"Does this model add enough value to deserve a place in the system?"**

This prevents technology-driven decisions.

---

# 27. Expected Outcomes

Each experiment should produce one of:

### 🟢 USE

Evidence is strong enough to keep the method.

### 🟡 TEST / ITERATE

Promising, but more testing is needed.

### 🔴 REJECT

It does not provide enough value.

### 🔵 KEEP OPEN

Interesting but not currently necessary.

---

# 28. Final Experiment Decision Framework

```text
Experiment
    ↓
Performance
    +
Financial Meaning
    +
Robustness
    +
Uncertainty
    +
Practicality
    ↓
Compare Against Baseline
    ↓
Decision
```

---

# Final Principle

> **The final architecture will be earned through experiments, not assumed from research.**

Our research has given us the candidates.

The experiment phase will tell us:

**what works → what does not → what adds value → what belongs in the final Project 1A system.**

