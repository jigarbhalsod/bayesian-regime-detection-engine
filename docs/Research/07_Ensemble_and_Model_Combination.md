# 08. Ensemble & Model Combination

## Purpose

This document explains how we will make **multiple models work together** instead of allowing one model to control the whole system.

Our main idea is:

> **Different models should have more influence when they are better suited to the current market situation, while the other models provide supporting evidence.**

---

## 1. Why Do We Need an Ensemble?

Different models are good at different things.

For example:

- HMM → hidden market regime
- RS-VAR → financial relationships
- BNN → nonlinear direction forecasting
- TSFM → general time-series forecasting
- BOCPD → possible market change

Instead of asking one model to do everything, we can combine their useful information.

---

## 2. Our Main Principle

We do **not** want:

```text
One model
    ↓
Final prediction
```

We want:

```text
Multiple models
      ↓
Situation understanding
      ↓
Different model influence
      ↓
Combined prediction
```

### Core rule

> **No model permanently drives the whole system.**

---

## 3. Situation-Based Dominance

A model can become more influential when the current market situation matches its strengths.

Example:

| Situation | Possible dominant model | Supporting models |
|---|---|---|
| Stable market | HMM | BNN, RS-VAR, TSFM |
| Possible transition | HMM + BOCPD | BNN, RS-VAR |
| Extreme movement | Student-t HMM | BOCPD, BNN |
| Strong nonlinear behaviour | BNN | HMM, RS-VAR |
| Strong variable interaction | RS-VAR | HMM, BNN |
| Forecast-heavy situation | BNN / TSFM | HMM, RS-VAR |

These are **initial hypotheses**, not fixed rules.

Experiments will decide whether these dominance patterns actually work.

---

## 4. What Does Dominance Mean?

Dominance does **not** mean that one model completely controls the prediction.

It means:

> **That model gets more influence because it is currently more suitable for the situation.**

The other models still contribute supporting information.

Example:

```text
HMM          → 50% influence
BNN          → 25%
RS-VAR       → 15%
TSFM         → 10%
```

The exact numbers will be learned or selected during experimentation.

---

## 5. Confidence From Each Model

Each model may provide a confidence or probability.

For example:

```text
HMM:
Risk-Off = 80%

BNN:
Down = 70%

RS-VAR:
Bearish signal = Strong

TSFM:
Downward forecast = Moderate
```

We should not simply average these values.

A model's contribution should also depend on **how suitable that model is for the current situation**.

---

## 6. Situation-Adjusted Confidence

Our proposed idea is:

> **Model confidence × situation-based influence = model contribution**

For example:

```text
BNN confidence = 80%
BNN situation weight = 0.7

Contribution = 80% × 0.7
```

The exact mathematical formulation will be decided during implementation and testing.

The important idea is:

> **Confidence alone should not decide how much influence a model gets.**

---

## 7. Weighted Averaging

Our first ensemble method will be simple weighted averaging.

Each model receives a weight.

Example:

```text
HMM      → 0.40
BNN      → 0.30
RS-VAR   → 0.20
TSFM     → 0.10
```

Their predictions are combined using those weights.

### Why start here?

Because it is:

- Simple
- Easy to understand
- Easy to debug
- Easy to compare against
- Less likely to introduce unnecessary complexity

### Decision

🟢 **USE as the ensemble baseline**

---

## 8. Stacking

Stacking is a more advanced approach.

Instead of manually deciding the final combination, another model learns:

> **How much should we trust each model's prediction?**

Simple structure:

```text
HMM prediction
BNN prediction
RS-VAR prediction
TSFM prediction
        ↓
   Stacking Model
        ↓
Final Prediction
```

This can potentially learn better combinations than fixed weights.

However, it also introduces another model and another layer of overfitting risk.

### Decision

🟡 **TEST**

---

## 9. Bayesian Model Averaging

Bayesian Model Averaging (BMA) combines models using probabilistic model weights.

It is theoretically attractive because it can account for uncertainty about which model is best.

However, it can add unnecessary complexity for our first implementation.

### Decision

🔵 **KEEP OPEN**

We will consider BMA only if simpler ensemble methods are not sufficient.

---

## 10. Model Disagreement

Model disagreement is useful information.

Example:

```text
HMM      → Risk-Off
BNN      → Down
RS-VAR   → Neutral
TSFM     → Up
```

This tells us:

> **The system does not have a clear agreement.**

Instead of hiding this disagreement, we can use it as a signal of uncertainty.

### Possible response

- Reduce overall confidence.
- Increase the influence of the most reliable model for the current situation.
- Send the prediction to the decision-support layer with a caution flag.

---

## 11. Role of Calibration

Before using model confidence in the ensemble, we need to check whether that confidence is trustworthy.

For example:

> If a model says 80%, does the event actually happen roughly 80% of the time?

This is why **calibration is part of the overall system**.

```text
Model Prediction
      ↓
Calibration
      ↓
Trustworthy Probability
      ↓
Ensemble
```

The exact order may be refined during implementation depending on how each model produces its output.

---

## 12. Role of BOCPD

BOCPD does not directly make the final forecast.

It provides an **early-warning signal**.

Example:

```text
HMM:
Risk-On

BOCPD:
Possible change detected

↓

System becomes more cautious
↓

Model influence is adjusted
```

This can help the ensemble react differently during possible transitions.

---

## 13. Dynamic Model Influence

The long-term goal is for model influence to respond to the market situation.

Conceptually:

```text
Current Market Situation
          ↓
Model Suitability
          ↓
Model Weight
          ↓
Model Prediction
          ↓
Final Ensemble
```

Possible inputs to the weighting logic:

- Current regime.
- BOCPD warning.
- Model confidence.
- Recent model performance.
- Calibration quality.
- Current market volatility.
- Model reliability in similar historical situations.

---

## 14. Important Limitation

We should **not assume that dynamic weighting automatically improves performance**.

A complicated weighting system can overfit historical data.

Therefore, we need to compare:

1. Single best model.
2. Simple equal-weight ensemble.
3. Fixed weighted ensemble.
4. Situation-based weighted ensemble.
5. Stacking.

The simplest method that performs well should be preferred.

---

## 15. How We Will Validate the Ensemble

We will check:

### Predictive Performance

Does the ensemble improve directional forecasting?

### Robustness

Does it work across different market periods?

### Model Dependence

Does it still work if one model performs poorly?

### Calibration

Are the final probabilities trustworthy?

### Stability

Do the weights change too aggressively?

### Complexity

Is the improvement worth the additional system complexity?

### Practical Value

Does the ensemble improve the final decision-support system?

---

## 16. Avoiding Overfitting

Dynamic model weighting can easily become overfitted.

To reduce this risk:

- Use time-aware validation.
- Never use future information.
- Keep the number of weighting rules limited.
- Test on unseen periods.
- Compare against simple baselines.
- Do not change weights based only on historical performance without proper validation.

---

## 17. Our Proposed Ensemble Structure

```text
                    Market Data
                        ↓
                Situation Layer
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
       HMM            BOCPD           Features
        ↓               ↓               ↓
    Regime Info    Change Warning   Market Context
        │               │               │
        └───────────────┼───────────────┘
                        ↓
                Model Predictions
                        ↓
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
     BNN             RS-VAR            TSFM
       ↓                ↓                ↓
       └────────────────┼────────────────┘
                        ↓
             Situation-Based Weights
                        ↓
                 Ensemble Layer
                        ↓
                    Calibration
                        ↓
              Final Direction Signal
                        ↓
                Decision Support
```

This is the **research direction**, not a promise that every component will survive testing.

---

## 18. Current Decisions

| Component | Role | Decision |
|---|---|---|
| Weighted Averaging | Simple ensemble baseline | 🟢 **USE** |
| Situation-Based Weighting | Adjust model influence | 🟡 **TEST** |
| Stacking | Learn model combination | 🟡 **TEST** |
| Bayesian Model Averaging | Advanced model combination | 🔵 **KEEP OPEN** |
| Model Disagreement | Additional uncertainty signal | 🟢 **USE** |
| BOCPD | Early-warning input to weighting | 🟢 **USE as signal** |
| Calibration | Check confidence quality | 🟢 **USE** |

---

## 19. Final Ensemble Principle

The most important decision is:

> **No model gets permanent control. Model influence should depend on the current market situation, model reliability, and confidence.**

The final system should therefore behave like:

```text
Detect situation
      ↓
Understand which models are useful
      ↓
Give them appropriate influence
      ↓
Keep other models as supporting evidence
      ↓
Combine predictions
      ↓
Check confidence
      ↓
Produce final signal
```

---

## Final Decision

> **We will start with simple weighted averaging, then test situation-based weighting and stacking. We will keep Bayesian Model Averaging open unless simpler approaches fail to provide enough value.**

The guiding rule is:

> **Use the simplest ensemble that gives reliable improvement without making the system unnecessarily complex.**
