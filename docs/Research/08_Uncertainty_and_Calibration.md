# 09. Uncertainty & Calibration

## Purpose

This document explains how Project 1A will handle **model uncertainty, confidence, and calibration**.

The main idea is simple:

> **The system should not only tell us what it predicts, but also tell us how sure it is, and that confidence should be trustworthy.**

---

## 1. Why Uncertainty Matters

Financial markets are uncertain.

A model may sometimes have a strong signal and sometimes have very little confidence.

For example:

```text
Risk-Off = 85%
```

is very different from:

```text
Risk-Off = 51%
Transition = 45%
```

The second situation is much more uncertain.

Project 1A should be able to recognize that difference.

---

## 2. What Is Uncertainty?

In simple terms:

> **Uncertainty tells us how unsure the model is about its prediction.**

It helps us understand when the system should be more confident and when it should be more cautious.

---

## 3. Two Main Types of Uncertainty

### Epistemic Uncertainty

This comes from **what the model does not know well**.

Examples:

- Limited training data.
- Limited information.
- Model uncertainty.
- A situation that looks very different from historical data.

This type of uncertainty can potentially be reduced with better data or better modelling.

---

### Aleatoric Uncertainty

This comes from **the market itself being unpredictable or noisy**.

Examples:

- Sudden news.
- Unexpected events.
- Random short-term price movements.

This type of uncertainty cannot simply be removed by collecting more data.

---

## 4. What Our System Should Output

Instead of only:

```text
Regime = Risk-Off
```

we want something closer to:

```text
Risk-Off       72%
Transition     20%
Risk-On         8%
```

For direction forecasting:

```text
Up      68%
Down    32%
```

The exact output format will be finalized during implementation.

---

## 5. Why Confidence Alone Is Not Enough

A model saying:

> "I am 90% confident"

does not automatically mean the prediction is trustworthy.

The model may be **overconfident**.

For example:

If a model gives 90% confidence many times, but the predicted event only happens 65% of the time, its confidence is poorly calibrated.

So we need to check whether confidence matches reality.

---

# 6. Calibration

## What Is Calibration?

Calibration asks:

> **When the model says 70%, does the event actually happen about 70% of the time?**

A well-calibrated model should have probabilities that roughly match observed outcomes.

---

## 7. Why Calibration Is Important for Project 1A

Our hybrid system will use model confidence to decide how much influence each model should have.

Therefore:

> **Bad confidence → Bad model weighting → Potentially bad final prediction.**

This makes calibration an important part of the system.

---

## 8. Calibration Example

Suppose we collect predictions where the model says:

```text
70% probability of Up
```

If roughly 70% of those predictions actually result in Up movement, the model is reasonably calibrated at that confidence level.

If only 40% result in Up movement, the model is overconfident.

---

## 9. Calibration Methods

We will consider simple calibration methods first.

### Platt / Logistic Calibration

Learns a simple mapping from the model's raw score to a better probability.

### Isotonic Regression

Learns a more flexible mapping without assuming a particular probability shape.

We do not need to choose between them in advance.

### Decision

🟡 **Test both where appropriate and keep the method that performs better on time-aware validation.**

---

## 10. Calibration Metrics

We will use metrics that help us measure probability quality.

### Brier Score

Measures how close predicted probabilities are to the actual outcomes.

Lower is better.

### Log Loss

Penalizes incorrect and overconfident probability predictions.

Lower is better.

### Calibration Curve

Compares predicted probabilities with actual outcome frequencies.

### Expected Calibration Error (ECE)

Measures the difference between predicted confidence and observed accuracy across probability groups.

These metrics should be evaluated using proper time-aware validation.

---

# 11. Uncertainty in HMM

HMM naturally provides probabilities for hidden states.

Example:

```text
Risk-On       15%
Transition    25%
Risk-Off      55%
Recovery       5%
```

This is more useful than only saying:

> Risk-Off

because it shows that the model is not completely certain.

---

# 12. Bayesian HMM and Uncertainty

A Bayesian HMM can go further by representing uncertainty around model parameters and hidden states.

This is one of the main reasons we are testing it.

Simple idea:

**Standard HMM**

> Gives regime probabilities.

**Bayesian HMM**

> Gives regime probabilities while also representing more uncertainty around the model itself.

This does not automatically mean Bayesian HMM will be better.

It must be tested.

---

# 13. BNN and Uncertainty

BNN is also relevant because it can provide uncertainty around its directional predictions.

For example:

```text
Up = 70%
Down = 30%
```

along with uncertainty related to the model's prediction.

This makes BNN useful for our hybrid forecasting layer.

---

# 14. Model Confidence in the Ensemble

We agreed that model confidence should not be used by itself.

Instead:

> **Model confidence × situation-based influence = model contribution**

For example:

```text
BNN confidence = 80%
Situation weight = 0.7

Contribution strength = based on both values
```

The exact mathematical formula will be determined during implementation and experimentation.

---

# 15. Model Disagreement

Disagreement between models is also useful.

Example:

```text
HMM      → Risk-Off
BNN      → Down
RS-VAR   → Neutral
TSFM     → Up
```

This means the models do not strongly agree.

The system should not hide this.

Possible response:

- Lower final confidence.
- Increase caution.
- Give more influence to models that are historically reliable in the current situation.
- Pass a disagreement flag to the decision-support layer.

---

# 16. Calibration and Model Dominance

Our hybrid architecture depends on knowing which model deserves more influence.

Therefore, calibration can help us avoid situations like:

```text
Model A:
Very confident but usually unreliable

Model B:
Moderately confident but historically reliable
```

The system should not automatically choose Model A simply because it reports a higher confidence.

Instead, it should consider:

- Current situation.
- Historical reliability.
- Calibration.
- Current confidence.
- Recent performance.
- Model disagreement.

---

# 17. Conformal Prediction

Conformal Prediction is another approach for expressing uncertainty.

In simple terms:

> **It can provide a range or set of possible outcomes rather than pretending that one exact prediction is certain.**

For example, instead of:

```text
Expected return = 1.2%
```

we might eventually provide:

```text
Expected range = -0.5% to +2.4%
```

The exact implementation depends on the forecasting target.

---

## 18. Adaptive Conformal Prediction

Financial time series are not perfectly stable.

Market behaviour can change over time.

Adaptive conformal methods are therefore interesting because they can adjust to changing conditions.

However, this adds another layer of complexity.

### Decision

🟡 **TEST**

It should not become mandatory until we see whether it provides useful uncertainty information beyond what our models already provide.

---

# 19. What We Will Use vs Test

| Component | Role | Decision |
|---|---|---|
| Regime probabilities | Show uncertainty about current regime | 🟢 **USE** |
| Model confidence | Input to model weighting | 🟢 **USE** |
| Calibration | Check whether confidence is trustworthy | 🟢 **USE** |
| Brier Score | Probability-quality metric | 🟢 **USE** |
| Log Loss | Probability-quality metric | 🟢 **USE** |
| Calibration Curve | Visual probability check | 🟢 **USE** |
| ECE | Calibration metric | 🟢 **USE** |
| Bayesian HMM | Better uncertainty modelling | 🟡 **TEST** |
| BNN uncertainty | Directional uncertainty | 🟡 **TEST with BNN** |
| Adaptive Conformal | Prediction ranges/sets | 🟡 **TEST** |

---

# 20. How This Fits the Hybrid Architecture

```text
                   Market Data
                       ↓
                 Model Predictions
                       ↓
              Model Confidence
                       ↓
                 Calibration
                       ↓
              Situation Understanding
                       ↓
           Confidence + Model Influence
                       ↓
                 Hybrid Ensemble
                       ↓
            Final Prediction + Confidence
                       ↓
                 Decision Support
```

The system should therefore produce not only:

> **What do we predict?**

but also:

> **How confident are we?**

and:

> **How trustworthy is that confidence?**

---

# 21. Main Risks

## Overconfidence

The model may report very high confidence when it should not.

## Poor Calibration

Predicted probabilities may not match real-world frequencies.

## Model Disagreement

Different models may give strongly different predictions.

## Regime Changes

A model may become less reliable when the market moves into a new condition.

## Overcomplicated Uncertainty System

Adding too many uncertainty methods can make the system unnecessarily difficult to understand and maintain.

---

# 22. Validation Rules

Uncertainty methods must be evaluated using:

- Time-aware validation.
- Out-of-sample data.
- Different market conditions.
- Stress periods.
- Model disagreement situations.
- Calibration quality.
- Forecasting usefulness.

We should never judge uncertainty quality only on training data.

---

# 23. Final Uncertainty Principle

> **The system should treat confidence as information that must itself be tested, not as a fact.**

A model saying 90% does not automatically make it trustworthy.

We need to verify whether its confidence matches reality.

---

# Final Decision

> **Project 1A will use probabilistic predictions and calibration as core requirements. Bayesian HMM and BNN will be tested for stronger uncertainty modelling, while Adaptive Conformal Prediction remains an optional experiment for later.**

The simplest mental model is:

**Predict → Measure confidence → Check confidence → Adjust model influence → Make final decision**
